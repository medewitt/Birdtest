"""Detection differences between PUC sites and site-level weather response.

Three NumPyro models, fit to the long-running PUC stations over the window
they all cover (WINDOW_START to the last cached day). A station-day enters
only if the station reported at least one detection that day (an "active"
day, as in R/prepare.R); silent-but-online days cannot be told apart from
offline days and are dropped.

1. Overall activity. Daily all-species total T[s,d]:

       T[s,d] ~ NegBin(mu, phi),  log mu = c[d] + g[s]

   c[d] is a shared day effect (season, weather, anything county-wide),
   g[s] has a sum-to-zero constraint. exp(g[s]) is the site's detection rate
   relative to the geometric mean of the sites on the same days.

2. Species composition. Weekly count y[j,s,w] for the N_TOP species:

       y ~ NegBin(mu, phi),  log mu = a[s,w] + m[j,month] + u[j,s]
       u[j,s] = tau[j] * z[j,s],  z ~ N(0, 1)

   a[s,w] absorbs effort and the site's overall rate, so u[j,s] is the log
   ratio of species j's share of detections at site s to its cross-site
   mean (soft-centred by the zero-mean prior).

3. Weather response. Daily total T[s,d] on active days:

       T ~ NegBin(mu, phi),  log mu = a[s,w] + b_t[s] * dT[d] + b_h[s] * dH[d]
       b_x[s] ~ N(beta_x, sigma_x)

   dT and dH are ERA5 county-centroid daily mean temperature (deg C) and
   relative humidity (%), as deviations from their mean within the ISO week,
   so a[s,w] carries the seasonal cycle and b_x[s] is the day-to-day
   response: proportional change in detections per 1 deg C (per 10 % RH)
   warmer (more humid) than the rest of that week. The raw Spearman
   correlation of T with the daily value, with a 14-day block bootstrap, is
   reported next to it for comparison.

Assumptions and caveats
- Detections are vocal activity weighted by detectability, not abundance.
  Site differences mix local abundance with microphone placement, ambient
  noise and BirdNET performance; they cannot be separated here.
- Counts are conditionally independent given the effects. Day-to-day
  autocorrelation is not modelled, so posterior intervals are somewhat too
  narrow (most for model 1 and 3, which are daily).
- All sites share the county-centroid weather series; site microclimate is
  not used because PUC and ERA5-Land station series cover only the sampled
  days (see the Microclimate page).

Checks: divergences and r-hat are printed; posterior predictive 90 %
interval coverage and zero fraction are written to checks.csv.

Run: uv run --group analysis python scripts/site_compare.py
Outputs go to output/site_compare/ (CSV tables and PNG figures).
"""

import os
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import numpyro
import numpyro.distributions as dist
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from numpyro.diagnostics import summary
from numpyro.infer import MCMC, NUTS, Predictive

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SEED = 1834
STATIONS = [9883, 17755, 20422, 21599, 21688]
WINDOW_START = np.datetime64("2026-01-03")
N_TOP = 25            # species in the composition model
BLOCK = 14            # days per bootstrap block for the raw correlations
B = 500               # bootstrap resamples
CHAINS, WARMUP, SAMPLES = 4, 1000, 1000
TARGET_ACCEPT = float(os.environ.get("BW_TARGET_ACCEPT", "0.95"))

CACHE_DIR = Path(os.environ.get("BW_CACHE_DIR", "data/cache"))
OUT = Path("output/site_compare")

numpyro.set_host_device_count(CHAINS)


# ---------------------------------------------------------------------- data

def read_counts():
    files = sorted((CACHE_DIR / "counts").glob("*.parquet"))
    tab = pa.concat_tables([pq.read_table(f) for f in files])
    keep = pc.and_(pc.is_in(tab["station_id"], pa.array(STATIONS, pa.int32())),
                   pc.greater_equal(tab["date"], pa.scalar(WINDOW_START.astype("datetime64[D]").item(), pa.date32())))
    tab = tab.filter(keep)
    return (tab["station_id"].to_numpy(),
            pc.cast(tab["date"], pa.int32()).to_numpy(),
            tab["species_id"].to_numpy(),
            tab["count"].to_numpy().astype(np.int64))


def read_species():
    t = pq.read_table(CACHE_DIR / "species.parquet")
    return dict(zip(t["species_id"].to_pylist(), t["common_name"].to_pylist()))


def read_weather():
    t = pq.read_table(CACHE_DIR / "weather_era5.parquet")
    day = pc.cast(t["date"], pa.int32()).to_numpy()
    return {int(d): (tm, rh) for d, tm, rh in
            zip(day, t["tmean"].to_numpy(), t["rhmean"].to_numpy().astype(float))}


def iso_week(day):
    """ISO year*100 + week for days since 1970-01-01 (Thursday rule)."""
    dt = day.astype("datetime64[D]")
    thu = dt + ((3 - (dt.astype(int) + 3) % 7)).astype("timedelta64[D]")  # Thursday of the ISO week
    yr = thu.astype("datetime64[Y]")
    wk = (thu - yr.astype("datetime64[D]")).astype(int) // 7 + 1
    return (yr.astype(int) + 1970) * 100 + wk


def month(day):
    dt = day.astype("datetime64[D]")
    return (dt.astype("datetime64[M]").astype(int) % 12)  # 0..11


# -------------------------------------------------------------------- models

def m_overall(day_idx, st_idx, n_day, n_st, y=None):
    c = numpyro.sample("c", dist.Normal(8, 3).expand([n_day]))
    g_raw = numpyro.sample("g_raw", dist.Normal(0, 1).expand([n_st - 1]))
    g = numpyro.deterministic("g", jnp.concatenate([g_raw, -g_raw.sum(keepdims=True)]))
    phi_inv = numpyro.sample("phi_inv", dist.Exponential(1))
    eta = c[day_idx] + g[st_idx]
    numpyro.sample("y", dist.NegativeBinomial2(jnp.exp(eta), 1 / phi_inv), obs=y)


def m_share(sp, st, sw, mo, J, S, K, y=None):
    a = numpyro.sample("a", dist.Normal(5, 5).expand([K]))
    m = numpyro.sample("m", dist.Normal(0, 3).expand([J, 12]))
    z = numpyro.sample("z", dist.Normal(0, 1).expand([J, S]))
    tau = numpyro.sample("tau", dist.HalfNormal(1).expand([J]))
    u = numpyro.deterministic("u", tau[:, None] * z)
    phi_inv = numpyro.sample("phi_inv", dist.Exponential(1))
    eta = a[sw] + m[sp, mo] + u[sp, st]
    numpyro.sample("y", dist.NegativeBinomial2(jnp.exp(eta), 1 / phi_inv), obs=y)


def m_weather(st, sw, dT, dH, S, K, y=None):
    a = numpyro.sample("a", dist.Normal(8, 3).expand([K]))
    beta_t = numpyro.sample("beta_t", dist.Normal(0, 0.1))
    beta_h = numpyro.sample("beta_h", dist.Normal(0, 0.1))
    sig_t = numpyro.sample("sigma_t", dist.HalfNormal(0.05))
    sig_h = numpyro.sample("sigma_h", dist.HalfNormal(0.05))
    zt = numpyro.sample("zt", dist.Normal(0, 1).expand([S]))
    zh = numpyro.sample("zh", dist.Normal(0, 1).expand([S]))
    b_t = numpyro.deterministic("b_t", beta_t + sig_t * zt)
    b_h = numpyro.deterministic("b_h", beta_h + sig_h * zh)
    phi_inv = numpyro.sample("phi_inv", dist.Exponential(1))
    eta = a[sw] + b_t[st] * dT + b_h[st] * dH
    numpyro.sample("y", dist.NegativeBinomial2(jnp.exp(eta), 1 / phi_inv), obs=y)


def fit(model, key, **data):
    mcmc = MCMC(NUTS(model, target_accept_prob=TARGET_ACCEPT),
                num_warmup=WARMUP, num_samples=SAMPLES, num_chains=CHAINS,
                progress_bar=False)
    mcmc.run(key, extra_fields=("diverging",), **data)
    post = mcmc.get_samples(group_by_chain=True)
    div = int(mcmc.get_extra_fields()["diverging"].sum())
    rhat = max(float(np.nanmax(v["r_hat"])) for v in summary(post).values())
    ess = min(float(np.nanmin(v["n_eff"])) for v in summary(post).values())
    print(f"  {model.__name__}: divergences={div}, max r-hat={rhat:.3f}, min ESS={ess:.0f}")
    flat = {k: np.asarray(v).reshape(-1, *v.shape[2:]) for k, v in post.items()}
    return flat, dict(model=model.__name__, divergences=div, max_rhat=rhat, min_ess=ess)


def ppc(model, key, post, y, **data):
    """90 % predictive interval coverage and zero fraction, observed vs replicated."""
    rep = np.asarray(Predictive(model, posterior_samples=post)(key, **data)["y"])
    lo, hi = np.quantile(rep, [0.05, 0.95], axis=0)
    return dict(ppc_cov90=float(np.mean((y >= lo) & (y <= hi))),
                zero_obs=float(np.mean(y == 0)), zero_rep=float(np.mean(rep == 0)))


def q(x, axis=0):
    return np.quantile(x, [0.5, 0.025, 0.975], axis=axis)


def write_csv(path, header, rows):
    with open(path, "w") as f:
        f.write(",".join(header) + "\n")
        for r in rows:
            f.write(",".join(f"{v:.4g}" if isinstance(v, float) else f'"{v}"' if isinstance(v, str) and "," in v else str(v)
                             for v in r) + "\n")


# ---------------------------------------------------------------------- main

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    key = jax.random.PRNGKey(SEED)
    rng = np.random.default_rng(SEED)
    sid, day, spid, cnt = read_counts()
    names = read_species()
    wx = read_weather()
    S = len(STATIONS)
    st_of = {s: i for i, s in enumerate(STATIONS)}
    print(f"Window {WINDOW_START} to {np.datetime64(int(day.max()), 'D')}, {len(cnt)} rows")

    # Daily totals per active station-day
    sd_keys, sd_inv = np.unique(np.stack([sid, day], 1), axis=0, return_inverse=True)
    T = np.bincount(sd_inv.ravel(), weights=cnt).astype(np.int64)
    d_st = np.array([st_of[s] for s in sd_keys[:, 0]])
    d_day = sd_keys[:, 1]
    days, d_dayidx = np.unique(d_day, return_inverse=True)
    n_active = np.bincount(d_st, minlength=S)

    # 1. Overall activity
    print("Fitting overall activity model")
    key, k1, k2 = jax.random.split(key, 3)
    post1, diag1 = fit(m_overall, k1, day_idx=d_dayidx, st_idx=d_st,
                       n_day=len(days), n_st=S, y=T)
    diag1.update(ppc(m_overall, k2, post1, T, day_idx=d_dayidx, st_idx=d_st,
                     n_day=len(days), n_st=S))
    g = q(np.exp(post1["g"]))
    write_csv(OUT / "overall_rate.csv",
              ["station_id", "active_days", "detections", "rate_ratio", "lo95", "hi95"],
              [(STATIONS[s], int(n_active[s]), int(T[d_st == s].sum()),
                float(g[0, s]), float(g[1, s]), float(g[2, s])) for s in range(S)])

    # 2. Species composition, weekly
    print("Fitting species composition model")
    tot = np.bincount(np.searchsorted(np.unique(spid), spid), weights=cnt)
    top = np.unique(spid)[np.argsort(-tot)[:N_TOP]]
    wk = iso_week(day)
    sw_keys = np.unique(np.stack([sid, wk], 1), axis=0)
    sw_month = {}
    for s_, w_, m_ in zip(sid, wk, month(day)):
        sw_month.setdefault((s_, w_), m_)
    in_top = np.isin(spid, top)
    J, K = len(top), len(sw_keys)
    sw_index = {tuple(k): i for i, k in enumerate(sw_keys)}
    Y = np.zeros((K, J), np.int64)
    sp_of = {int(x): j for j, x in enumerate(top)}
    np.add.at(Y, (np.array([sw_index[(a, b)] for a, b in zip(sid[in_top], wk[in_top])]),
                  np.array([sp_of[int(x)] for x in spid[in_top]])), cnt[in_top])
    sw_idx, sp_idx = np.meshgrid(np.arange(K), np.arange(J), indexing="ij")
    sw_idx, sp_idx, y2 = sw_idx.ravel(), sp_idx.ravel(), Y.ravel()
    st2 = np.array([st_of[s] for s in sw_keys[:, 0]])[sw_idx]
    mo2 = np.array([sw_month[tuple(k)] for k in sw_keys])[sw_idx]
    key, k1, k2 = jax.random.split(key, 3)
    data2 = dict(sp=sp_idx, st=st2, sw=sw_idx, mo=mo2, J=J, S=S, K=K)
    post2, diag2 = fit(m_share, k1, y=y2, **data2)
    diag2.update(ppc(m_share, k2, post2, y2, **data2))
    u = q(post2["u"] / np.log(2))  # log2 share ratio
    rows = []
    for j in range(J):
        for s in range(S):
            rows.append((names.get(int(top[j]), str(top[j])), STATIONS[s],
                         int(Y[st2.reshape(K, J)[:, 0] == s, j].sum()),
                         float(u[0, j, s]), float(u[1, j, s]), float(u[2, j, s])))
    write_csv(OUT / "species_share.csv",
              ["species", "station_id", "detections", "log2_share_ratio", "lo95", "hi95"], rows)

    # 3. Weather response
    print("Fitting weather response model")
    ok = np.array([int(d) in wx for d in d_day])
    tm = np.array([wx[int(d)][0] if o else np.nan for d, o in zip(d_day, ok)])
    rh = np.array([wx[int(d)][1] if o else np.nan for d, o in zip(d_day, ok)])
    wk_d = iso_week(d_day)
    # deviations from the ISO-week mean of the county series (same for all sites)
    dT, dH = np.full_like(tm, np.nan), np.full_like(rh, np.nan)
    for w in np.unique(wk_d):
        uday = np.unique(d_day[wk_d == w])
        mt = np.nanmean([wx[int(x)][0] for x in uday if int(x) in wx])
        mh = np.nanmean([wx[int(x)][1] for x in uday if int(x) in wx])
        dT[wk_d == w], dH[wk_d == w] = tm[wk_d == w] - mt, (rh[wk_d == w] - mh) / 10
    ok &= ~np.isnan(dT) & ~np.isnan(dH)
    swk3, sw3 = np.unique(np.stack([d_st[ok], wk_d[ok]], 1), axis=0, return_inverse=True)
    key, k1, k2 = jax.random.split(key, 3)
    data3 = dict(st=d_st[ok], sw=sw3.ravel(), dT=dT[ok], dH=dH[ok], S=S, K=len(swk3))
    post3, diag3 = fit(m_weather, k1, y=T[ok], **data3)
    diag3.update(ppc(m_weather, k2, post3, T[ok], **data3))
    bt = q(100 * np.expm1(post3["b_t"]))
    bh = q(100 * np.expm1(post3["b_h"]))
    pooled = {k: q(100 * np.expm1(post3[k])) for k in ("beta_t", "beta_h")}

    # Raw Spearman correlation with a moving-block bootstrap over days
    def spearman(x, y):
        rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
        return np.corrcoef(rx, ry)[0, 1]

    def boot_rho(x, y, dd):
        order = np.argsort(dd)
        x, y = x[order], y[order]
        n = len(x)
        starts = rng.integers(0, n - BLOCK + 1, size=(B, int(np.ceil(n / BLOCK))))
        idx = (starts[:, :, None] + np.arange(BLOCK)).reshape(B, -1)[:, :n]
        r = np.array([spearman(x[i], y[i]) for i in idx])
        return spearman(x, y), *np.quantile(r, [0.025, 0.975])

    rows = []
    for s in range(S):
        sel = ok & (d_st == s)
        rt = boot_rho(tm[sel], T[sel], d_day[sel])
        rr = boot_rho(rh[sel], T[sel], d_day[sel])
        rows.append((STATIONS[s], int(sel.sum()),
                     *map(float, rt), *map(float, rr),
                     float(bt[0, s]), float(bt[1, s]), float(bt[2, s]),
                     float(bh[0, s]), float(bh[1, s]), float(bh[2, s])))
    rows.append(("pooled", int(ok.sum()), *[float("nan")] * 6,
                 *map(float, pooled["beta_t"]), *map(float, pooled["beta_h"])))
    write_csv(OUT / "weather.csv",
              ["station_id", "days", "rho_tmean", "rho_lo95", "rho_hi95",
               "rho_rh", "rho_rh_lo95", "rho_rh_hi95",
               "pct_per_degC", "lo95", "hi95", "pct_per_10pctRH", "rh_lo95", "rh_hi95"], rows)

    write_csv(OUT / "checks.csv", list(diag1.keys()),
              [tuple(d.values()) for d in (diag1, diag2, diag3)])

    raw = {"t": np.array([r[2:5] for r in rows[:S]]), "h": np.array([r[5:8] for r in rows[:S]])}
    plot(STATIONS, g, n_active, u, [names.get(int(t), str(t)) for t in top],
         {"t": bt, "h": bh}, pooled, raw)
    print(f"Wrote {OUT}/")


# ---------------------------------------------------------------------- plots

def plot(stations, g, n_active, u, sp_names, bslope, pooled, raw):
    S = len(stations)
    lab = [str(s) for s in stations]

    fig, ax = plt.subplots(figsize=(6.5, 3))
    y = np.arange(S)
    ax.errorbar(g[0], y, xerr=[g[0] - g[1], g[2] - g[0]], fmt="o", color="#0072B2", capsize=3)
    ax.axvline(1, color="0.6", lw=0.8)
    ax.set_xscale("log", base=2)
    ticks = [0.5, 0.75, 1, 1.5, 2]
    ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
    ax.minorticks_off()
    ax.set_xlim(0.5, 2)
    ax.set_yticks(y, [f"PUC-{l} ({n} d)" for l, n in zip(lab, n_active)])
    ax.set_xlabel("Daily detections / geometric mean of sites (ratio, log scale)")
    ax.set_title("Overall detection rate by site, same days (95% CrI)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "overall_rate.png", dpi=150)
    plt.close(fig)

    J = len(sp_names)
    med = u[0]
    sig = (u[1] > 0) | (u[2] < 0)
    order = np.argsort(med.max(1) - med.min(1))[::-1]
    lim = np.ceil(np.nanmax(np.abs(med)))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("0.92")
    fig, ax = plt.subplots(figsize=(6, 0.28 * J + 1.5))
    im = ax.imshow(np.ma.masked_where(~sig, med)[order], cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
    for i, j in enumerate(order):
        for s in range(S):
            ax.text(s, i, f"{med[j, s]:+.1f}", ha="center", va="center", fontsize=6,
                    color="white" if sig[j, s] and med[j, s] < 0.2 * lim else "0.15")
    ax.set_xticks(range(S), [f"PUC-{l}" for l in lab], rotation=30, ha="right")
    ax.set_yticks(range(J), [sp_names[i] for i in order], fontsize=8)
    cb = fig.colorbar(im, ax=ax, shrink=0.6)
    cb.set_label("Share relative to cross-site mean (log2)")
    ax.set_title("Species share by site (gray: 95% CrI includes 0)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "species_share.png", dpi=150)
    plt.close(fig)

    y = np.arange(S)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True)
    for ax, v, pk, xl in ((axes[0], "t", "beta_t", "% change per 1 deg C above week mean"),
                          (axes[1], "h", "beta_h", "% change per 10 % RH above week mean")):
        b, p, r = bslope[v], pooled[pk], raw[v]
        ax.axvspan(p[1], p[2], color="#0072B2", alpha=0.12, label="Pooled 95% CrI")
        ax.errorbar(b[0], y + 0.12, xerr=[b[0] - b[1], b[2] - b[0]], fmt="o",
                    color="#0072B2", capsize=3, label="Site slope (week-adjusted model)")
        ax.axvline(0, color="0.6", lw=0.8)
        ax.set_xlabel(xl)
        ax2 = ax.twiny()
        ax2.errorbar(r[:, 0], y - 0.12, xerr=[r[:, 0] - r[:, 1], r[:, 2] - r[:, 0]],
                     fmt="s", color="#D55E00", capsize=3)
        ax2.set_xlim(-1, 1)
        ax2.set_xlabel("Spearman rho, daily total vs daily value", color="#D55E00", fontsize=8)
        ax2.tick_params(axis="x", colors="#D55E00", labelsize=8)
    axes[0].set_yticks(y, [f"PUC-{l}" for l in lab])
    h1, l1 = axes[0].get_legend_handles_labels()
    fig.legend(h1 + [plt.Line2D([], [], marker="s", color="#D55E00", ls="")],
               l1 + ["Raw Spearman rho, 14-day block bootstrap (top axis)"],
               loc="lower center", ncol=3, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(OUT / "weather.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
