"""Weekly Gaussian process surfaces for CoCoRaHS rainfall and PUC night temperature.

Two variables, one model. For site s and week w, with y standardized
(mean 0, SD 1 over all site-weeks of the variable):

    y[s,w] = v[s] + mu[w] + f_w(x_s) + e[s,w]
    v[s]   = m0 + g(x_s) + u[s]

    m0       overall level,                     m0 ~ N(0, 1)
    g(x)     persistent spatial field,          g ~ GP(0, a_g^2 k(. ; l_g))
    u[s]     site offset not shared with neighbours, u ~ N(0, s_u^2)
    mu[w]    county-wide level of week w,       mu ~ N(0, tau^2)
    f_w(x)   weekly anomaly field,              f_w ~ GP(0, (a_f c[w])^2 k(. ; l_f)),
             independent across weeks
    e[s,w]   reading error,                     e ~ N(0, (sigma c[w])^2 / n[s,w])
    c[w]     week scale, exp(s_eta eta[w]), eta ~ N(0, 1): weeks differ in
             how much sites differ (convective storms, clear calm nights)

k is the Matern 3/2 kernel on distance in km. Both lengthscales have an
InverseGamma(4, 30) prior (mean 10 km, P(l < 3 km) about 0.01): a field
shorter than the gauge spacing cannot be told apart from reading error, and
leaving that open makes the sampler crawl along the ridge between them. Given the hyperparameters the
model is linear-Gaussian, so mu[w] and f_w are integrated out within each
week and v is integrated out across weeks (Woodbury identity). NUTS samples
only the hyperparameters and eta. For each retained draw, v, then (m0, g),
then the weekly fields are drawn from their exact Gaussian conditionals.

Variables
- rain: log(1 + weekly CoCoRaHS total in mm). Weeks run Monday to Sunday on
  the day the rain fell (report date minus one, as in R/prepare.R). A
  gauge-week enters with at least MIN_DAYS single-day reports (trace = 0);
  its total is 7 x the mean daily amount. Multi-day accumulations are
  dropped. Only wet weeks (mean gauge total >= WET_MM) are modelled, because
  gauge differences are proportional to the rain that falls and carry no
  information in dry weeks. On the log scale a site offset d is a ratio:
  100 (exp(d) - 1) % more or less rain than the county-area mean, for totals
  well above 1 mm. n[s,w] = 1.
- tnight: PUC minus ERA5-Land air temperature (deg C), mean over local hours
  00-05 (the Microclimate page's "Night" hours) of hourly means with >= 30
  readings; a station-day needs >= 3 of the 6 hours, and the weekly value is
  the mean over sampled days, n[s,w] = number of sampled days. ERA5-Land is
  corrected to each station's elevation, so the delta is the departure from a
  reanalysis that does not resolve terrain below about 9 km.

Outputs (data/cache/)
- gp_sites.parquet       per site: offset d[s] = g(x_s) + u[s] minus the
                         county-area mean of g: posterior median, 90 %
                         interval, P(d > 0), terrain at the site, and
                         `distinct` (90 % interval excludes 0).
- gp_persistent.parquet  g on the prediction grid minus its county-area mean:
                         median, 90 % interval, P(> 0), terrain per cell.
- gp_weekly.parquet      weekly surface m0 + g(x) + mu[w] + f_w(x) on the grid,
                         back-transformed (rain in mm, tnight in deg C):
                         median and 90 % interval, for modelled weeks in the
                         last SHIP_WEEKS.
- gp_summary.parquet     hyperparameters on the variable's scale, Pearson
                         correlation over sites of d with TPI, HAND and
                         elevation (per posterior draw), and checks: max r-hat,
                         min bulk ESS, divergences, leave-one-site-out 90 %
                         predictive coverage within week, prior predictive SD
                         of site offsets.

Terrain (data/terrain.parquet, scripts/terrain.py) is not in the model, so
the correlation of site offsets with TPI and HAND is an independent check.
Cold-air drainage predicts negative night deltas at low TPI and low HAND.

Assumptions and caveats
- Gaussian errors on the transformed scale; weeks independent given the
  hyperparameters (no serial correlation in f_w); stationary isotropic
  kernels, so a ridge or river valley that steers storms is not represented.
- Hyperparameters and eta are sampled; everything else is drawn exactly
  given them.
- u[s] cannot separate a very local real effect from gauge exposure, observer
  practice, or (for PUC) sensor offset and enclosure heating. Distinct sites
  are hypotheses, not findings.
- tnight rests on the few PUC stations, clustered west of downtown with one
  in the southeast, sampled on the same few days per month; away from them
  the surface is mostly prior. One PUC station has location privacy on
  (fuzzed coordinates).

Run: uv run --group analysis python scripts/gp_weekly.py   (about 2 min; skips if current)
"""

import hashlib
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import numpyro
import numpyro.distributions as dist
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from jax.scipy.linalg import cho_solve
from numpyro.diagnostics import summary
from numpyro.infer import MCMC, NUTS

jax.config.update("jax_enable_x64", True)

SEED = 1834
N_CHAINS = 4
CACHE_DIR = Path(os.environ.get("BW_CACHE_DIR", "data/cache"))
TERRAIN = Path("data/terrain.parquet")
TZ_LOCAL = "America/New_York"

MIN_DAYS = 5          # single-day reports per gauge-week
WET_MM = 5.0          # rain weeks modelled: mean gauge total at least this
MIN_WEEKS = 8         # site-weeks per site
NIGHT = (0, 5)        # local hours, inclusive
SHIP_WEEKS = 104      # weekly surfaces written for the most recent weeks
GRID_STEP = 0.015     # deg, prediction grid
N_DRAWS = 200         # posterior draws used for surfaces and site offsets
WARMUP, SAMPLES = 500, 250
JITTER = 1e-6
HYPER = ("tau", "a_f", "ell_f", "s_eta", "a_g", "ell_g", "s_u", "sigma")

# Copy of forsyth_polygon() in R/config.R (maps::county, 29 vertices).
OUTLINE_LON = [-80.3917, -80.4089, -80.4146, -80.4089, -80.4318, -80.4433, -80.449, -80.4662,
               -80.4777, -80.4948, -80.4891, -80.4834, -80.4605, -80.4375, -80.4261, -80.4318,
               -80.4375, -80.4318, -80.4375, -80.4433, -80.4433, -80.0365, -80.0479, -80.1453,
               -80.1511, -80.2084, -80.2141, -80.3287, -80.3917]
OUTLINE_LAT = [35.9646, 35.976, 35.9875, 36.0161, 36.0276, 36.0448, 36.0734, 36.062, 36.0505,
               36.062, 36.0963, 36.1021, 36.1193, 36.1364, 36.1594, 36.1708, 36.1995, 36.2167,
               36.2281, 36.2338, 36.2568, 36.2453, 36.039, 36.039, 36.0219, 36.0219, 35.9875,
               35.9875, 35.9646]
LAT0 = 36.1
KM_LAT = 111.32
KM_LON = 111.32 * np.cos(np.deg2rad(LAT0))


def in_polygon(x, y, px, py):
    inside = np.zeros(len(x), bool)
    j = len(px) - 1
    with np.errstate(divide="ignore", invalid="ignore"):
        for i in range(len(px)):
            hit = ((py[i] > y) != (py[j] > y)) & (x < (px[j] - px[i]) * (y - py[i]) / (py[j] - py[i]) + px[i])
            inside ^= hit
            j = i
    return inside


def to_km(lon, lat):
    return np.column_stack([(np.asarray(lon) + 80.3) * KM_LON, (np.asarray(lat) - LAT0) * KM_LAT])


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def week_start(dates):
    d = pd.to_datetime(pd.Series(dates))
    return (d - pd.to_timedelta(d.dt.weekday, unit="D")).dt.normalize().to_numpy()


def complete(week, last):
    """Weeks whose Sunday is on or before the last day in the data."""
    return (pd.to_datetime(week) + pd.Timedelta(days=6)) <= last


def rain_weeks():
    st = pq.read_table(CACHE_DIR / "cocorahs_stations.parquet").to_pandas()
    d = pq.read_table(CACHE_DIR / "cocorahs_daily.parquet").to_pandas()
    d = d[d.pcpn.notna()].copy()
    d["week"] = week_start(pd.to_datetime(d.date) - pd.Timedelta(days=1))
    w = d.groupby(["sid", "week"]).agg(mm=("pcpn", "mean"), n=("pcpn", "size")).reset_index()
    w = w[(w.n >= MIN_DAYS) & complete(w.week, pd.to_datetime(d.date).max() - pd.Timedelta(days=1))].copy()
    w["total"] = 7 * 25.4 * w.mm
    wet = w.groupby("week").total.mean()
    w = w[w.week.isin(wet.index[wet >= WET_MM])].copy()
    w["y"] = np.log1p(w.total)
    w["n"] = 1.0
    st = st.rename(columns={"sid": "site"})[["site", "name", "lon", "lat"]].assign(privacy=False)
    return w.rename(columns={"sid": "site"})[["site", "week", "y", "n"]], st


def tnight_weeks():
    p = pq.read_table(CACHE_DIR / "puc_hourly.parquet").to_pandas()
    e = pq.read_table(CACHE_DIR / "era5_hourly.parquet").to_pandas()
    h = p[(p.n >= 30) & p.temp.between(-30, 50)].merge(e[["station_id", "time", "era5"]],
                                                        on=["station_id", "time"])
    h = h[h.era5.notna()]
    lt = h.time.dt.tz_convert(TZ_LOCAL)
    h = h[lt.dt.hour.between(*NIGHT)].assign(day=lt.dt.tz_localize(None).dt.normalize())
    h["delta"] = h.temp - h.era5
    sd = h.groupby(["station_id", "day"]).delta.agg(["mean", "size"]).reset_index()
    sd = sd[sd["size"] >= 3].copy()
    sd["week"] = week_start(sd.day)
    sd = sd[complete(sd.week, lt.max().tz_localize(None).normalize())]
    w = sd.groupby(["station_id", "week"])["mean"].agg(["mean", "size"]).reset_index()
    w = w.rename(columns={"station_id": "site", "mean": "y", "size": "n"})
    w["n"] = w.n.astype(float)
    st = pq.read_table(CACHE_DIR / "stations.parquet").to_pandas()
    st = st[st.station_type == "puc"].rename(columns={
        "station_id": "site", "station_name": "name", "station_lon": "lon", "station_lat": "lat",
        "location_privacy": "privacy"})[["site", "name", "lon", "lat", "privacy"]]
    return w[["site", "week", "y", "n"]], st


def arrays(w, st):
    """Week x site arrays with a mask; sites with < MIN_WEEKS site-weeks dropped."""
    nw = w.groupby("site").size()
    st = st[st.site.isin(nw.index[nw >= MIN_WEEKS])].dropna(subset=["lon", "lat"])
    st = st.sort_values("site").reset_index(drop=True)
    w = w[w.site.isin(st.site)]
    weeks = np.sort(w.week.unique())
    si = pd.Index(st.site).get_indexer(w.site)
    wi = pd.Index(weeks).get_indexer(w.week)
    W, S = len(weeks), len(st)
    y, n, m = np.zeros((W, S)), np.ones((W, S)), np.zeros((W, S), bool)
    y[wi, si], n[wi, si], m[wi, si] = w.y, w.n, True
    return dict(y=y, n=n, mask=m, weeks=weeks, sites=st)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def matern32(d, ell):
    r = jnp.sqrt(3.0) * d / ell
    return (1.0 + r) * jnp.exp(-r)


def pdist(a, b):
    return jnp.sqrt(jnp.sum((a[:, None, :] - b[None, :, :]) ** 2, -1) + 1e-12)


def week_cov(D, mask, n, tau, a_f, s_w, ell_f, sigma):
    """Covariance of y over sites, per week, for mu[w] + f_w + e, with the
    week's scale s_w on both f_w and e. Unobserved rows and columns are
    replaced by the identity, which leaves the observed block's likelihood
    unchanged when the matching residuals are zero."""
    S = D.shape[0]
    K = tau ** 2 + (s_w ** 2)[:, None, None] * (
        a_f ** 2 * matern32(D, ell_f)[None] + jnp.eye(S)[None] * (sigma ** 2 / n)[:, :, None])
    mm = mask[:, :, None] & mask[:, None, :]
    return jnp.where(mm, K, jnp.eye(S)[None])


def site_cov(D, a_g, ell_g, s_u):
    S = D.shape[0]
    return 1.0 + a_g ** 2 * matern32(D, ell_g) + (s_u ** 2 + JITTER) * jnp.eye(S)


def marginal(D, y, n, mask, p, eta):
    """log p(y | hyperparameters) with v integrated out, plus the pieces of
    v's Gaussian posterior: precision P and P mean = b."""
    S = D.shape[0]
    L = jnp.linalg.cholesky(week_cov(D, mask, n, p["tau"], p["a_f"], jnp.exp(p["s_eta"] * eta),
                                     p["ell_f"], p["sigma"]))
    Ki = jax.vmap(lambda l: cho_solve((l, True), jnp.eye(S)))(L)
    mm = mask[:, :, None] & mask[:, None, :]
    Ki = jnp.where(mm, Ki, 0.0)
    b = jnp.einsum("wij,wj->i", Ki, y)
    yKy = jnp.einsum("wi,wij,wj->", y, Ki, y)
    ldK = 2 * jnp.sum(jnp.log(jnp.diagonal(L, axis1=1, axis2=2)))
    Lv = jnp.linalg.cholesky(site_cov(D, p["a_g"], p["ell_g"], p["s_u"]))
    P = cho_solve((Lv, True), jnp.eye(S)) + Ki.sum(0)
    Lp = jnp.linalg.cholesky(P)
    c = jax.scipy.linalg.solve_triangular(Lp, b, lower=True)
    ldV = 2 * jnp.sum(jnp.log(jnp.diag(Lv)))
    ldP = 2 * jnp.sum(jnp.log(jnp.diag(Lp)))
    ll = -0.5 * (yKy - c @ c + ldK + ldV + ldP + mask.sum() * jnp.log(2 * jnp.pi))
    return ll, P, b


def model(D, y, n, mask):
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
    eta = numpyro.sample("eta", dist.Normal(0.0, 1.0).expand([W]))
    ll, _, _ = marginal(D, y, n, mask, p, eta)
    numpyro.factor("y", ll)


# ---------------------------------------------------------------------------
# Draws of the latent fields given the hyperparameters
# ---------------------------------------------------------------------------
def grid():
    lon = np.arange(min(OUTLINE_LON), max(OUTLINE_LON) + GRID_STEP, GRID_STEP)
    lat = np.arange(min(OUTLINE_LAT), max(OUTLINE_LAT) + GRID_STEP, GRID_STEP)
    gx, gy = (a.ravel() for a in np.meshgrid(lon, lat))
    ok = in_polygon(gx, gy, np.array(OUTLINE_LON), np.array(OUTLINE_LAT))
    return pd.DataFrame({"cell": np.arange(ok.sum()), "lon": np.round(gx[ok], 4), "lat": np.round(gy[ok], 4)})


def draw_fields(D, Dgs, Dgg, y, n, mask, p, eta, rng):
    """One joint draw: v at sites, m0, g on the grid (with its area mean gbar),
    and the weekly latent field on the grid, all on the standardized scale."""
    S, G = D.shape[0], Dgs.shape[0]
    _, P, b = marginal(D, y, n, mask, p, eta)
    P, b = np.asarray(P), np.asarray(b)
    Lp = np.linalg.cholesky(P)
    v = np.linalg.solve(P, b) + np.linalg.solve(Lp.T, rng.standard_normal(S))

    # (m0, g on grid) | v: joint Gaussian with cov(m0, v) = 1, cov(g, v) = a_g^2 R.
    Sv = np.asarray(site_cov(D, p["a_g"], p["ell_g"], p["s_u"]))
    Rgs = p["a_g"] ** 2 * np.asarray(matern32(Dgs, p["ell_g"]))
    C = np.vstack([np.ones((1, S)), Rgs])                      # (1 + G) x S
    A = np.linalg.solve(Sv, C.T).T
    mean = A @ v
    prior = np.zeros((1 + G, 1 + G))
    prior[0, 0] = 1.0
    prior[1:, 1:] = p["a_g"] ** 2 * np.asarray(matern32(Dgg, p["ell_g"]))
    cov = prior - A @ C.T
    Lc = np.linalg.cholesky(cov + 1e-8 * np.eye(1 + G))
    z = mean + Lc @ rng.standard_normal(1 + G)
    m0, gg = z[0], z[1:]

    # mu[w] + f_w on the grid | residuals y - v at the observed sites.
    s_w = np.exp(p["s_eta"] * eta)
    af_w = p["a_f"] * s_w
    K = np.asarray(week_cov(D, mask, n, p["tau"], p["a_f"], s_w, p["ell_f"], p["sigma"]))
    r = np.where(mask, y - v[None], 0.0)
    Kgs = p["tau"] ** 2 + (af_w ** 2)[:, None, None] * np.asarray(matern32(Dgs, p["ell_f"]))[None]
    Kgs = np.where(mask[:, None, :], Kgs, 0.0)
    wmean = np.einsum("wgs,ws->wg", Kgs, np.linalg.solve(K, r[..., None])[..., 0])
    V = np.linalg.solve(K, np.transpose(Kgs, (0, 2, 1)))
    wvar = np.clip(p["tau"] ** 2 + af_w[:, None] ** 2 - np.einsum("wgs,wsg->wg", Kgs, V), 0, None)
    weekly = m0 + gg[None] + wmean + np.sqrt(wvar) * rng.standard_normal(wmean.shape)
    return v, m0, gg, weekly


def loo_z(D, y, n, mask, p, eta, v):
    """Leave-one-site-out standardized residual within each week, given v."""
    K = np.asarray(week_cov(D, mask, n, p["tau"], p["a_f"], np.exp(p["s_eta"] * eta), p["ell_f"], p["sigma"]))
    Ki = np.linalg.inv(K)
    r = np.where(mask, y - v[None], 0.0)
    dg = np.diagonal(Ki, axis1=1, axis2=2)
    z = np.einsum("wij,wj->wi", Ki, r) / np.sqrt(dg)
    return z[mask & (mask.sum(1, keepdims=True) >= 2)]


def terrain_at(lon, lat, ter, half=0.0):
    """Terrain at points: nearest cell, or the mean over a square of half-width `half` deg."""
    tl, ta = ter.lon.to_numpy(), ter.lat.to_numpy()
    out = {}
    for k in ("elev_m", "tpi_m", "hand_m"):
        col = ter[k].to_numpy()
        vals = []
        for x, yv in zip(lon, lat):
            if half > 0:
                sel = (np.abs(tl - x) <= half) & (np.abs(ta - yv) <= half)
                vals.append(np.nanmean(col[sel]))
            else:
                vals.append(col[np.argmin((tl - x) ** 2 + (ta - yv) ** 2)])
        out[k] = np.round(np.array(vals, float), 2)
    return out


def q(x, axis=0):
    return np.quantile(x, [0.5, 0.05, 0.95], axis=axis)


def fit(name, w, st, ter, cells, key):
    t0 = time.time()
    a = arrays(w, st)
    y_raw, n, mask, sites = a["y"], a["n"], a["mask"], a["sites"]
    mu_y, sd_y = y_raw[mask].mean(), y_raw[mask].std()
    y = np.where(mask, (y_raw - mu_y) / sd_y, 0.0)
    Xs, Xg = to_km(sites.lon, sites.lat), to_km(cells.lon, cells.lat)
    D = np.asarray(pdist(jnp.asarray(Xs), jnp.asarray(Xs)))
    Dgs = np.asarray(pdist(jnp.asarray(Xg), jnp.asarray(Xs)))
    Dgg = np.asarray(pdist(jnp.asarray(Xg), jnp.asarray(Xg)))
    W, S = y.shape
    print(f"[{name}] {S} sites, {W} weeks, {mask.sum()} site-weeks", flush=True)

    # Prior predictive SD of site offsets v - m0 (variable's units).
    rng = np.random.default_rng(SEED)
    ag, lg, su = (np.abs(rng.normal(0, 0.5, 500)), 30.0 / rng.gamma(4.0, 1.0, 500),
                  np.abs(rng.normal(0, 0.5, 500)))
    pri = [rng.multivariate_normal(np.zeros(S), np.asarray(site_cov(D, a_, l_, s_)) - 1.0 + 1e-6 * np.eye(S))
           for a_, l_, s_ in zip(ag, lg, su)]
    pri_sd = float(np.std(pri) * sd_y)

    mcmc = MCMC(NUTS(model, target_accept_prob=0.9, max_tree_depth=8), num_warmup=WARMUP, num_samples=SAMPLES,
                num_chains=N_CHAINS, chain_method="sequential", progress_bar=False)
    mcmc.run(key, jnp.asarray(D), jnp.asarray(y), jnp.asarray(n), jnp.asarray(mask),
             extra_fields=("diverging", "num_steps"))
    ex = mcmc.get_extra_fields()
    div = int(ex["diverging"].sum())
    sm = summary(mcmc.get_samples(group_by_chain=True))
    rhat = max(float(np.nanmax(v["r_hat"])) for v in sm.values())
    ess = min(float(np.nanmin(v["n_eff"])) for v in sm.values())
    post = {k: np.asarray(v) for k, v in mcmc.get_samples().items()}
    print(f"[{name}] NUTS {time.time() - t0:.0f} s, mean steps {np.mean(ex['num_steps']):.0f}, "
          f"divergences {div}, max r-hat {rhat:.3f}, min ESS {ess:.0f}", flush=True)

    draws = rng.choice(len(post["tau"]), N_DRAWS, replace=False)
    G = len(cells)
    d, gc, weekly, zs = np.empty((N_DRAWS, S)), np.empty((N_DRAWS, G)), np.empty((N_DRAWS, W, G)), []
    for k, i in enumerate(draws):
        p = {h: float(post[h][i]) for h in HYPER}
        v, m0, gg, weekly[k] = draw_fields(D, Dgs, Dgg, y, n, mask, p, post["eta"][i], rng)
        gbar = gg.mean()
        d[k] = (v - m0 - gbar) * sd_y
        gc[k] = (gg - gbar) * sd_y
        if k < 50:
            zs.append(loo_z(D, y, n, mask, p, post["eta"][i], v))
    cover = float(np.mean(np.abs(np.concatenate(zs)) <= 1.645))

    ts = terrain_at(sites.lon, sites.lat, ter)
    tg = terrain_at(cells.lon, cells.lat, ter, half=GRID_STEP / 2)
    qd = q(d)
    site_tab = sites.assign(var=name, n_weeks=mask.sum(0), est=qd[0], lo=qd[1], hi=qd[2],
                            p_pos=(d > 0).mean(0), distinct=(qd[1] > 0) | (qd[2] < 0), **ts)
    qg = q(gc)
    pers = cells.assign(var=name, est=qg[0], lo=qg[1], hi=qg[2], p_pos=(gc > 0).mean(0), **tg)

    recent = a["weeks"] >= a["weeks"][-1] - np.timedelta64(7 * (SHIP_WEEKS - 1), "D")
    ys = weekly[:, recent] * sd_y + mu_y
    if name == "rain":
        ys = np.expm1(ys).clip(0, None)
    qw = q(ys)
    wk = a["weeks"][recent]
    weekly_tab = pd.DataFrame({
        "var": name, "week": np.repeat(wk, G), "cell": np.tile(cells.cell.to_numpy(), len(wk)),
        "est": qw[0].ravel(), "lo": qw[1].ravel(), "hi": qw[2].ravel(),
        "n_sites": np.repeat(mask[recent].sum(1), G)})

    rows = []
    for h in HYPER:
        rows.append((name, h, *q(post[h] * (sd_y if h in ("tau", "a_f", "a_g", "s_u", "sigma") else 1.0))))
    for k in ("tpi_m", "hand_m", "elev_m"):
        x = ts[k]
        ok = np.isfinite(x)
        r = np.array([np.corrcoef(di[ok], x[ok])[0, 1] for di in d])
        rows.append((name, f"cor_{k}", *q(r)))
    rows += [(name, k, float(v), np.nan, np.nan) for k, v in (
        ("divergences", div), ("max_rhat", rhat), ("min_ess", ess), ("loo_cover90", cover),
        ("prior_site_sd", pri_sd), ("post_site_sd", float(np.std(d))), ("n_sites", S), ("n_weeks", W),
        ("n_site_weeks", int(mask.sum())))]
    summ = pd.DataFrame(rows, columns=["var", "param", "est", "lo", "hi"])
    print(f"[{name}] LOO-site 90% coverage {cover:.2f}; distinct {int(site_tab.distinct.sum())} of {S}; "
          f"{time.time() - t0:.0f} s total", flush=True)
    return site_tab, pers, weekly_tab, summ


def write(df, path):
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path, compression="zstd")


def fingerprint(inputs):
    """Latest complete week per variable plus hashes of this script and the
    terrain layer. The fit reruns only when one of them changes (about once a
    week), or when BW_GP_FORCE=1; late CoCoRaHS corrections enter at the next
    weekly refit."""
    h = hashlib.sha256(Path(__file__).read_bytes() + TERRAIN.read_bytes()).hexdigest()[:16]
    weeks = " ".join(f"{k}:{str(np.max(w.week))[:10]}" for k, (w, _) in inputs.items())
    return f"{weeks} code:{h}"


def main():
    inputs = {"rain": rain_weeks(), "tnight": tnight_weeks()}
    key_file = CACHE_DIR / "gp_key.txt"
    fp = fingerprint(inputs)
    outs = [CACHE_DIR / f"gp_{k}.parquet" for k in ("sites", "persistent", "weekly", "summary")]
    if (os.environ.get("BW_GP_FORCE") != "1" and key_file.exists()
            and key_file.read_text().strip() == fp and all(o.exists() for o in outs)):
        print(f"GP surfaces current ({fp}); skipping fit. Set BW_GP_FORCE=1 to refit.")
        return
    ter = pq.read_table(TERRAIN).to_pandas()
    cells = grid()
    key = jax.random.PRNGKey(SEED)
    out = []
    for name, (w, st) in inputs.items():
        key, k = jax.random.split(key)
        out.append(fit(name, w, st, ter, cells, k))
    sites = pd.concat([o[0] for o in out], ignore_index=True)
    sites["site"] = sites.site.astype(str)
    write(sites, CACHE_DIR / "gp_sites.parquet")
    write(pd.concat([o[1] for o in out], ignore_index=True), CACHE_DIR / "gp_persistent.parquet")
    wk = pd.concat([o[2] for o in out], ignore_index=True)
    wk[["est", "lo", "hi"]] = wk[["est", "lo", "hi"]].astype(np.float32)
    write(wk, CACHE_DIR / "gp_weekly.parquet")
    write(pd.concat([o[3] for o in out], ignore_index=True), CACHE_DIR / "gp_summary.parquet")
    key_file.write_text(fp + "\n")
    print(f"Wrote gp_*.parquet to {CACHE_DIR} ({fp})")


if __name__ == "__main__":
    main()
