"""Aridity across Forsyth County: weekly precipitation GP with an ERA5-Land
covariate, combined with ERA5-Land FAO-56 reference evapotranspiration (ET0).

Precipitation model. For CoCoRaHS gauge s and week w (Monday to Sunday on the
day the rain fell), on the square-root scale and standardized (y and x share
the scale sd_y of the observed y):

    y[s,w] = b x[s,w] + v[s] + mu[w] + f_w(x_s) + e[s,w]
    v[s]   = m0 + g(x_s) + u[s]

    y        sqrt(gauge weekly total, mm)
    x        sqrt(ERA5-Land weekly total at the gauge, mm), b ~ N(1, 0.5)
    m0, g, u, mu, f_w, e and the week scale c[w] as in scripts/gp_weekly.py:
             persistent field g (Matern 3/2), site offset u, week level mu,
             weekly anomaly field f_w with amplitude a_f c[w], error
             e ~ N(0, (sigma c[w])^2 / n[s,w]).

A gauge-week enters with at least MIN_DAYS single-day reports. Its y and x are
both totals over the reported days scaled to 7 days, so a missed rainy day
lowers both and the GP models the gauge departure from ERA5-Land on the same
days; n[s,w] = reported days / 7. Weeks with no gauge reports stay in the
model (the covariate and prior carry them), so rolling windows have no holes.
Dry weeks are kept: y = 0 there, and c[w] lets those weeks have small spread.

Given the hyperparameters and b the model is linear-Gaussian, so mu, f_w and v
are integrated out (functions reused from gp_weekly.py) and NUTS samples only
the hyperparameters, b and eta. For each retained draw, (m0, g) and then the
weekly fields are drawn jointly over the prediction points (grid cells and
gauges) from their exact Gaussian conditionals. Weekly precipitation at a
point is max(0, latent)^2 on the mm scale, where latent excludes e.

Aridity. For a trailing window of L weeks (4, 13, 52; about 30, 90, 365 days)
ending at week w, P is the sum of weekly precipitation draws and ET0 the sum
of ERA5-Land ET0 at the point's native cell (treated as known). Reported:
AI = P / ET0 (UNEP aridity index) and P - ET0 (mm), posterior median and 90 %
interval. The county value uses area means over grid cells of P and of ET0.

Inputs (data/cache/, written by R/cache.R)
- cocorahs_stations.parquet, cocorahs_daily.parquet
- era5_land_points.parquet, era5_land_daily.parquet

Outputs (data/cache/)
- aridity_points.parquet   prediction points: grid cells and modelled gauges
- aridity_map.parquet      per window, week end (last MAP_WEEKS) and point:
                           AI, P - ET0, P (median, 90 % interval), ET0
- aridity_series.parquet   county area mean per window and week end (last
                           SHIP_WEEKS): AI, P - ET0, P, ET0
- aridity_summary.parquet  hyperparameters and checks: max r-hat, min bulk
                           ESS, divergences, leave-one-gauge-out 90 %
                           coverage within week, posterior predictive share
                           of zero gauge-weeks and 90 % interval coverage
- aridity_key.txt          fingerprint; the fit is skipped when unchanged

Assumptions and caveats
- Gaussian errors on the square-root scale; replicated values below zero
  count as zero weeks. Weeks independent given the hyperparameters;
  stationary isotropic kernels.
- ERA5-Land precipitation (about 9 km) is not downscaled, so all spatial
  detail below that scale comes from the gauges through g, u and f_w.
- ET0 is the ERA5-Land FAO-56 Penman-Monteith value with no uncertainty; the
  intervals cover precipitation only.
- CoCoRaHS reports are volunteer readings; a site offset mixes local climate
  with gauge exposure and observer practice.

Run: uv run --group analysis python scripts/gp_aridity.py
(skips in seconds when no new complete week has arrived)
"""

import hashlib
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gp_weekly as gw  # noqa: E402  (sets XLA host device count, x64)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import numpyro  # noqa: E402
import numpyro.distributions as dist  # noqa: E402
import pandas as pd  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
from numpyro.diagnostics import summary  # noqa: E402
from numpyro.infer import MCMC, NUTS, init_to_median  # noqa: E402

SEED = gw.SEED
CACHE_DIR = gw.CACHE_DIR
N_CHAINS = gw.N_CHAINS

MIN_DAYS = 4          # single-day reports per gauge-week
MIN_WEEKS = 26        # gauge-weeks per gauge
WINDOWS = (4, 13, 52)  # trailing weeks
SHIP_WEEKS = 104      # county series length (week ends)
MAP_WEEKS = 13        # week ends with maps
CHUNK = 16            # weeks per joint draw block
N_DRAWS = 200
WARMUP, SAMPLES = 500, 250
HYPER = gw.HYPER + ("b",)


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def load():
    st = pq.read_table(CACHE_DIR / "cocorahs_stations.parquet").to_pandas()
    d = pq.read_table(CACHE_DIR / "cocorahs_daily.parquet").to_pandas()
    pts = pq.read_table(CACHE_DIR / "era5_land_points.parquet").to_pandas()
    e = pq.read_table(CACHE_DIR / "era5_land_daily.parquet").to_pandas()
    e["date"] = pd.to_datetime(e.date)

    # Model weeks: complete weeks where every county cell point has all 7 days.
    cells_e = e[e.id.isin(pts.id[pts.kind == "cell"])]
    n_cells = (pts.kind == "cell").sum()
    full = cells_e.groupby("date").size()
    days = full.index[full == n_cells]
    wk = pd.Series(gw.week_start(days), index=days)
    wcount = wk.value_counts()
    weeks = np.sort(wcount.index[wcount == 7].to_numpy())
    last_rain = pd.to_datetime(d.date).max() - pd.Timedelta(days=1)
    weeks = weeks[gw.complete(weeks, last_rain)]
    # Contiguous run ending at the last week, so rolling windows have no holes.
    gaps = np.where(np.diff(weeks).astype("timedelta64[D]").astype(int) != 7)[0]
    if len(gaps):
        weeks = weeks[gaps[-1] + 1:]

    # Gauge-weeks: totals over reported days, gauge and ERA5-Land on the same days.
    d = d[d.pcpn.notna()].copy()
    d["date"] = pd.to_datetime(d.date) - pd.Timedelta(days=1)
    d = d.merge(e.rename(columns={"id": "sid"}), on=["sid", "date"], how="inner")
    d["week"] = gw.week_start(d.date)
    w = d.groupby(["sid", "week"]).agg(g=("pcpn", "sum"), x=("pcpn_mm", "sum"), n=("pcpn", "size")).reset_index()
    w = w[(w.n >= MIN_DAYS) & w.week.isin(weeks)].copy()
    w["y"] = np.sqrt(7.0 / w.n * 25.4 * w.g)
    w["x"] = np.sqrt(7.0 / w.n * w.x)
    w["n"] = w.n / 7.0

    nw = w.groupby("sid").size()
    st = st[st.sid.isin(nw.index[nw >= MIN_WEEKS])].dropna(subset=["lon", "lat"])
    st = st.sort_values("sid").reset_index(drop=True)
    w = w[w.sid.isin(st.sid)]
    W, S = len(weeks), len(st)
    si = pd.Index(st.sid).get_indexer(w.sid)
    wi = pd.Index(weeks).get_indexer(w.week)
    y, x, n, m = np.zeros((W, S)), np.zeros((W, S)), np.ones((W, S)), np.zeros((W, S), bool)
    y[wi, si], x[wi, si], n[wi, si], m[wi, si] = w.y, w.x, w.n, True

    # Prediction points: grid cells (nearest ERA5-Land cell point) then gauges.
    cells = gw.grid()
    cp = pts[pts.kind == "cell"].reset_index(drop=True)
    glon = cp.grid_lon.fillna(cp.lon).to_numpy()
    glat = cp.grid_lat.fillna(cp.lat).to_numpy()
    near = np.argmin((cells.lon.to_numpy()[:, None] - glon[None]) ** 2
                     + (cells.lat.to_numpy()[:, None] - glat[None]) ** 2, axis=1)
    cells["era5_id"] = cp.id.to_numpy()[near]
    gp_ = st[["sid", "name", "lon", "lat"]].copy()
    have = set(e.id.unique())
    gp_["era5_id"] = [s if s in have else None for s in gp_.sid]
    miss = gp_.era5_id.isna()
    if miss.any():  # gauge without its own pull: nearest cell point
        k = np.argmin((gp_.lon.to_numpy()[miss, None] - glon[None]) ** 2
                      + (gp_.lat.to_numpy()[miss, None] - glat[None]) ** 2, axis=1)
        gp_.loc[miss, "era5_id"] = cp.id.to_numpy()[k]
    pred = pd.concat([
        cells.assign(kind="cell", point=cells.cell.astype(str), name="")[["point", "kind", "name", "lon", "lat", "era5_id"]],
        gp_.assign(kind="gauge", point=gp_.sid)[["point", "kind", "name", "lon", "lat", "era5_id"]],
    ], ignore_index=True)

    # Weekly ERA5-Land P and ET0 at each prediction point's ERA5 point.
    e["week"] = gw.week_start(e.date)
    ew = e[e.week.isin(weeks)].groupby(["id", "week"]).agg(p=("pcpn_mm", "sum"), et0=("et0_mm", "sum"),
                                                             k=("pcpn_mm", "size")).reset_index()
    ew = ew[ew.k == 7]
    P_e, ET_e = (ew.pivot(index="week", columns="id", values=c).reindex(weeks) for c in ("p", "et0"))
    # Gauge points missing a week of their own pull fall back to the nearest cell point.
    for sid in gp_.era5_id:
        if sid in P_e.columns and P_e[sid].isna().any():
            lon, lat = gp_.loc[gp_.era5_id == sid, ["lon", "lat"]].iloc[0]
            c = cp.id[np.argmin((glon - lon) ** 2 + (glat - lat) ** 2)]
            for t in (P_e, ET_e):
                t[sid] = t[sid].fillna(t[c])
    xp = np.sqrt(P_e[pred.era5_id].to_numpy())
    et0 = ET_e[pred.era5_id].to_numpy()
    assert np.isfinite(xp).all() and np.isfinite(et0).all(), "ERA5-Land weeks incomplete"
    return dict(y=y, x=x, n=n, mask=m, weeks=weeks, sites=st, pred=pred, xp=xp, et0=et0)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def model(D, y, x, n, mask):
    W = y.shape[0]
    p = dict(
        tau=numpyro.sample("tau", dist.HalfNormal(1.5)),
        a_f=numpyro.sample("a_f", dist.HalfNormal(1.0)),
        ell_f=numpyro.sample("ell_f", dist.InverseGamma(4.0, 30.0)),
        s_eta=numpyro.sample("s_eta", dist.HalfNormal(0.5)),
        a_g=numpyro.sample("a_g", dist.HalfNormal(0.5)),
        ell_g=numpyro.sample("ell_g", dist.InverseGamma(4.0, 30.0)),
        s_u=numpyro.sample("s_u", dist.HalfNormal(0.5)),
        sigma=numpyro.sample("sigma", dist.HalfNormal(1.0)),
    )
    b = numpyro.sample("b", dist.Normal(1.0, 0.5))
    eta = numpyro.sample("eta", dist.Normal(0.0, 1.0).expand([W]))
    r = jnp.where(mask, y - b * x, 0.0)
    ll, _, _ = gw.marginal(D, r, n, mask, p, eta)
    numpyro.factor("y", ll)


def draw(D, Dps, Dpp, r, n, mask, p, eta, wsel, rng):
    """Joint draw over prediction points: m0 + g (all weeks) and mu[w] + f_w
    for weeks wsel, given site residuals r = y - b x (standardized)."""
    S, Pn = D.shape[0], Dps.shape[0]
    _, Pm, bv = gw.marginal(D, r, n, mask, p, eta)
    Pm, bv = np.asarray(Pm), np.asarray(bv)
    Lp = np.linalg.cholesky(Pm)
    v = np.linalg.solve(Pm, bv) + np.linalg.solve(Lp.T, rng.standard_normal(S))

    Sv = np.asarray(gw.site_cov(D, p["a_g"], p["ell_g"], p["s_u"]))
    C = 1.0 + p["a_g"] ** 2 * np.asarray(gw.matern32(Dps, p["ell_g"]))   # cov(m0 + g_p, v)
    A = np.linalg.solve(Sv, C.T).T
    prior = 1.0 + p["a_g"] ** 2 * np.asarray(gw.matern32(Dpp, p["ell_g"]))
    cov = prior - A @ C.T
    base = A @ v + np.linalg.cholesky(cov + 1e-8 * np.eye(Pn)) @ rng.standard_normal(Pn)

    s_all = np.exp(p["s_eta"] * np.asarray(eta))
    Rps = np.asarray(gw.matern32(Dps, p["ell_f"]))
    Rpp = np.asarray(gw.matern32(Dpp, p["ell_f"]))
    f = np.empty((len(wsel), Pn))
    for c0 in range(0, len(wsel), CHUNK):  # bounded memory: CHUNK x P x P per step
        ws = wsel[c0:c0 + CHUNK]
        s_w = s_all[ws]
        af_w = p["a_f"] * s_w
        mk, nk = mask[ws], n[ws]
        K = np.asarray(gw.week_cov(D, mk, nk, p["tau"], p["a_f"], s_w, p["ell_f"], p["sigma"]))
        res = np.where(mk, r[ws] - v[None], 0.0)
        Kps = p["tau"] ** 2 + (af_w ** 2)[:, None, None] * Rps[None]
        Kps = np.where(mk[:, None, :], Kps, 0.0)
        KiKsp = np.linalg.solve(K, np.transpose(Kps, (0, 2, 1)))            # w x S x P
        mean = np.einsum("wps,ws->wp", Kps, np.linalg.solve(K, res[..., None])[..., 0])
        cw = p["tau"] ** 2 + (af_w ** 2)[:, None, None] * Rpp[None] - Kps @ KiKsp
        cw = 0.5 * (cw + np.transpose(cw, (0, 2, 1))) + 1e-8 * np.eye(Pn)[None]
        L = np.linalg.cholesky(cw)
        f[c0:c0 + len(ws)] = mean + np.einsum("wij,wj->wi", L, rng.standard_normal(mean.shape))
    return v, base, f, s_all[wsel]


def q(x, axis=0):
    return np.quantile(x, [0.5, 0.05, 0.95], axis=axis)


def roll(a, L):
    """Trailing L-week sums along axis -2 (weeks); first L - 1 rows are NaN."""
    c = np.cumsum(a, axis=-2)
    out = np.full(a.shape, np.nan)
    out[..., L - 1:, :] = c[..., L - 1:, :] - np.concatenate(
        [np.zeros_like(c[..., :1, :]), c[..., :-L, :]], axis=-2)
    return out


def fit(a, key):
    t0 = time.time()
    y_raw, x_raw, n, mask, sites, pred = a["y"], a["x"], a["n"], a["mask"], a["sites"], a["pred"]
    mu_y, sd_y, mu_x = y_raw[mask].mean(), y_raw[mask].std(), x_raw[mask].mean()
    y = np.where(mask, (y_raw - mu_y) / sd_y, 0.0)
    x = np.where(mask, (x_raw - mu_x) / sd_y, 0.0)
    xp = (a["xp"] - mu_x) / sd_y
    Xs, Xp = gw.to_km(sites.lon, sites.lat), gw.to_km(pred.lon, pred.lat)
    D = np.asarray(gw.pdist(jnp.asarray(Xs), jnp.asarray(Xs)))
    Dps = np.asarray(gw.pdist(jnp.asarray(Xp), jnp.asarray(Xs)))
    Dpp = np.asarray(gw.pdist(jnp.asarray(Xp), jnp.asarray(Xp)))
    W, S = y.shape
    print(f"[aridity] {S} gauges, {W} weeks, {mask.sum()} gauge-weeks, {len(pred)} prediction points", flush=True)

    mcmc = MCMC(NUTS(model, target_accept_prob=0.9, max_tree_depth=8, init_strategy=init_to_median),
                num_warmup=WARMUP, num_samples=SAMPLES, num_chains=N_CHAINS, chain_method="parallel",
                progress_bar=False)
    mcmc.run(key, jnp.asarray(D), jnp.asarray(y), jnp.asarray(x), jnp.asarray(n), jnp.asarray(mask),
             extra_fields=("diverging",))
    div = int(mcmc.get_extra_fields()["diverging"].sum())
    sm = summary(mcmc.get_samples(group_by_chain=True))
    rhat = max(float(np.nanmax(v["r_hat"])) for v in sm.values())
    ess = min(float(np.nanmin(v["n_eff"])) for v in sm.values())
    post = {k: np.asarray(v) for k, v in mcmc.get_samples().items()}
    print(f"[aridity] NUTS {time.time() - t0:.0f} s, divergences {div}, max r-hat {rhat:.3f}, "
          f"min ESS {ess:.0f}", flush=True)

    # Weeks needed: series of SHIP_WEEKS week ends with the longest window.
    need = min(W, SHIP_WEEKS + max(WINDOWS) - 1)
    wsel = np.arange(W - need, W)
    is_cell = (pred.kind == "cell").to_numpy()
    gidx = np.where(~is_cell)[0]
    et0 = a["et0"][wsel]
    rng = np.random.default_rng(SEED)
    draws = rng.choice(len(post["tau"]), N_DRAWS, replace=False)
    Pw = np.empty((N_DRAWS, need, len(pred)), np.float32)
    zs, zero_rep, cover = [], [], []
    for k, i in enumerate(draws):
        p = {h: float(post[h][i]) for h in gw.HYPER}
        b = float(post["b"][i])
        r = np.where(mask, y - b * x, 0.0)
        v, base, f, s_w = draw(D, Dps, Dpp, r, n, mask, p, post["eta"][i], wsel, rng)
        lat = mu_y + sd_y * (b * xp[wsel] + base[None] + f)
        Pw[k] = np.clip(lat, 0, None) ** 2
        if k < 50:
            zs.append(gw.loo_z(D, r, n, mask, p, post["eta"][i], v))
            # Posterior predictive at the gauges (in sample), sqrt scale.
            mk = mask[wsel]
            sd_e = sd_y * p["sigma"] * s_w[:, None] / np.sqrt(n[wsel])
            yrep = lat[:, gidx] + sd_e * rng.standard_normal(sd_e.shape)
            zero_rep.append(np.mean(yrep[mk] <= 0))
            cover.append(yrep[mk])
    loo = float(np.mean(np.abs(np.concatenate(zs)) <= 1.645))
    mk = mask[wsel]
    yobs = y_raw[wsel][mk]
    reps = np.stack(cover)
    ppc_cover = float(np.mean((yobs >= np.quantile(reps, 0.05, 0)) & (yobs <= np.quantile(reps, 0.95, 0))))
    zero_obs = float(np.mean(yobs == 0))
    zr = q(np.array(zero_rep))
    print(f"[aridity] LOO-gauge 90% coverage {loo:.2f}; in-sample 90% PPC coverage {ppc_cover:.2f}; "
          f"zero gauge-weeks observed {zero_obs:.3f}, replicated {zr[0]:.3f} [{zr[1]:.3f}, {zr[2]:.3f}]", flush=True)

    weeks_end = a["weeks"][wsel] + np.timedelta64(6, "D")
    out_w = np.arange(need - min(SHIP_WEEKS, need), need)
    map_w = out_w[-MAP_WEEKS:]
    series, maps = [], []
    for L in WINDOWS:
        Pl = roll(Pw.astype(np.float64), L)                         # draws x weeks x points
        El = roll(et0[None], L)[0]                                  # weeks x points
        # County: area mean over grid cells of P and of ET0.
        Pc, Ec = Pl[:, :, is_cell].mean(-1), El[:, is_cell].mean(-1)
        rows = {"window": L, "week_end": weeks_end[out_w], "et0": Ec[out_w]}
        for nm, val in (("ai", Pc / Ec), ("def", Pc - Ec), ("p", Pc)):
            qq = q(val[:, out_w])
            rows.update({f"{nm}_est": qq[0], f"{nm}_lo": qq[1], f"{nm}_hi": qq[2]})
        series.append(pd.DataFrame(rows))
        Pm_, Em = Pl[:, map_w], El[map_w]
        nP = len(pred)
        rows = {"window": L, "week_end": np.repeat(weeks_end[map_w], nP),
                "point": np.tile(pred.point.to_numpy(), len(map_w)), "et0": Em.ravel()}
        for nm, val in (("ai", Pm_ / Em[None]), ("def", Pm_ - Em[None]), ("p", Pm_)):
            qq = q(val)
            rows.update({f"{nm}_est": qq[0].ravel(), f"{nm}_lo": qq[1].ravel(), f"{nm}_hi": qq[2].ravel()})
        maps.append(pd.DataFrame(rows))
    series = pd.concat(series, ignore_index=True)
    maps = pd.concat(maps, ignore_index=True)

    # Gauge reporting in each window ending at the last week (for tooltips).
    pts = pred.drop(columns="era5_id").copy()
    for L in WINDOWS:
        rep = np.zeros(len(pts), int)
        rep[gidx] = mask[W - L:].sum(0)
        pts[f"weeks_{L}"] = rep
    pts["n_weeks"] = 0
    pts.loc[gidx, "n_weeks"] = mask.sum(0)

    rows = []
    for h in HYPER:
        scale = sd_y if h in ("tau", "a_f", "a_g", "s_u", "sigma") else 1.0
        rows.append(("aridity", h, *q(post[h] * scale)))
    rows += [("aridity", k, float(v), np.nan, np.nan) for k, v in (
        ("divergences", div), ("max_rhat", rhat), ("min_ess", ess), ("loo_cover90", loo),
        ("ppc_cover90", ppc_cover), ("zero_obs", zero_obs), ("n_sites", S), ("n_weeks", W),
        ("n_site_weeks", int(mask.sum())), ("first_week", float(a["weeks"][0].astype("datetime64[D]").astype(int))))]
    rows.append(("aridity", "zero_rep", *zr))
    summ = pd.DataFrame(rows, columns=["var", "param", "est", "lo", "hi"])
    print(f"[aridity] {time.time() - t0:.0f} s total", flush=True)
    return pts, maps, series, summ


def fingerprint(a):
    h = hashlib.sha256(Path(__file__).read_bytes() + Path(gw.__file__).read_bytes()).hexdigest()[:16]
    return f"weeks:{str(a['weeks'][0])[:10]}..{str(a['weeks'][-1])[:10]} gauges:{len(a['sites'])} code:{h}"


def main():
    need = ["cocorahs_stations", "cocorahs_daily", "era5_land_points", "era5_land_daily"]
    if not all((CACHE_DIR / f"{f}.parquet").exists() for f in need):
        print("Aridity inputs missing (run scripts/update_cache.R); skipping.")
        return
    a = load()
    if len(a["weeks"]) < max(WINDOWS) + 8 or len(a["sites"]) < 3:
        print(f"Too little data for the aridity GP ({len(a['weeks'])} weeks, {len(a['sites'])} gauges); skipping.")
        return
    key_file = CACHE_DIR / "aridity_key.txt"
    fp = fingerprint(a)
    outs = [CACHE_DIR / f"aridity_{k}.parquet" for k in ("points", "map", "series", "summary")]
    if (os.environ.get("BW_GP_FORCE") != "1" and key_file.exists()
            and key_file.read_text().strip() == fp and all(o.exists() for o in outs)):
        print(f"Aridity surfaces current ({fp}); skipping fit. Set BW_GP_FORCE=1 to refit.")
        return
    pts, maps, series, summ = fit(a, jax.random.PRNGKey(SEED))
    f32 = [c for c in maps.columns if c.endswith(("_est", "_lo", "_hi")) or c == "et0"]
    maps[f32] = maps[f32].astype(np.float32)
    for df, k in ((pts, "points"), (maps, "map"), (series, "series"), (summ, "summary")):
        gw.write(df, CACHE_DIR / f"aridity_{k}.parquet")
    key_file.write_text(fp + "\n")
    print(f"Wrote aridity_*.parquet to {CACHE_DIR} ({fp})")


if __name__ == "__main__":
    main()
