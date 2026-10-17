"""Annual seasonal component per species, written to data/cache/seasonal.parquet.

Model, per species and normalization (quasi-Poisson log-linear regression):

    log E[count_t] = log(offset_t) + b0 + b1 * t
                     + sum_{k=1..K} (a_k sin(2 pi k f_t) + c_k cos(2 pi k f_t))

count_t   county total of detections for the species on day t (all stations)
offset_t  all-species detections that day ("share") or active stations that
          day ("per_station"), as defined in R/prepare.R
t         years since the series midpoint (linear trend on the log scale)
f_t       fraction of the calendar year elapsed

The seasonal component is exp(harmonic sum), divided by its mean over the
calendar year: the expected rate on that day relative to the species' annual
average rate, with trend and effort removed (1 = average day, 0 = absent).
Scaling by the arithmetic rather than the geometric mean keeps the curve
bounded for migrants that are absent for months, where the log-scale
component runs off to minus infinity.

Intervals: moving-block bootstrap over days (28-day blocks, B = 200), refitting
the model on each resample; percentile 95% intervals. Blocks keep serial
correlation up to about four weeks.

Assumptions: the annual cycle is the same every year; the trend is linear on
the log scale; with fewer than two years of data the two are only partly
separable, and the fit leans on the months seen twice.

Run: uv run python scripts/seasonal.py
"""

import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

SEED = 1834
K = 3               # harmonics
B = 200             # bootstrap resamples
BLOCK = 28          # days per bootstrap block
N_TOP = int(os.environ.get("BW_SEASONAL_TOP", "40"))  # matches N_SHIP in R/prepare.R
MIN_DAYS = 60       # and detected on at least this many days
GRID_STEP = 5       # output resolution in days of year

CACHE_DIR = Path(os.environ.get("BW_CACHE_DIR", "data/cache"))
OUT = CACHE_DIR / "seasonal.parquet"


def read_counts():
    files = sorted((CACHE_DIR / "counts").glob("*.parquet"))
    if not files:
        raise SystemExit(f"No count files in {CACHE_DIR / 'counts'}")
    tab = pa.concat_tables([pq.read_table(f) for f in files])
    day = pc.cast(tab["date"], pa.int32()).to_numpy()  # days since 1970-01-01
    return (tab["station_id"].to_numpy(), day,
            tab["species_id"].to_numpy(), tab["count"].to_numpy().astype(float))


def design(day):
    """Intercept, linear trend in years, and K annual harmonic pairs."""
    dt = day.astype("datetime64[D]")
    year = dt.astype("datetime64[Y]")
    start = year.astype("datetime64[D]")
    length = ((year + np.timedelta64(1, "Y")).astype("datetime64[D]") - start).astype(float)
    frac = (dt - start).astype(float) / length
    t = (day - day.mean()) / 365.25
    cols = [np.ones_like(t), t]
    for k in range(1, K + 1):
        cols += [np.sin(2 * np.pi * k * frac), np.cos(2 * np.pi * k * frac)]
    return np.column_stack(cols)


def harmonics(frac):
    cols = []
    for k in range(1, K + 1):
        cols += [np.sin(2 * np.pi * k * frac), np.cos(2 * np.pi * k * frac)]
    return np.column_stack(cols)


def fit_poisson(X, y, off, ridge=1e-6, iters=50, tol=1e-8):
    """Poisson GLM by IRLS. A tiny ridge keeps fits finite when a species is
    absent for a long stretch; it does not move well-determined estimates."""
    beta = np.zeros(X.shape[1])
    beta[0] = np.log((y.sum() + 0.5) / np.exp(off).sum())
    pen = np.full(X.shape[1], ridge)
    pen[0] = 0.0
    for _ in range(iters):
        eta = np.clip(off + X @ beta, -30, 30)
        mu = np.exp(eta)
        z = eta - off + (y - mu) / mu
        XtW = X.T * mu
        new = np.linalg.solve(XtW @ X + np.diag(pen), XtW @ z)
        if np.max(np.abs(new - beta)) < tol:
            return new
        beta = new
    return beta


def block_index(n, rng):
    starts = rng.integers(0, n - BLOCK + 1, size=int(np.ceil(n / BLOCK)))
    return (starts[:, None] + np.arange(BLOCK)).ravel()[:n]


def main():
    station, day, species, count = read_counts()
    d0, d1 = day.min(), day.max()
    days = np.arange(d0, d1 + 1)
    n = days.size
    di = day - d0

    total = np.bincount(di, weights=count, minlength=n)
    st_day = np.unique(np.column_stack([di, station]), axis=0)
    n_st = np.bincount(st_day[:, 0], minlength=n).astype(float)

    X = design(days)
    G = harmonics(np.arange(365) / 365.0)    # daily, for the annual mean
    keep = np.arange(0, 365, GRID_STEP)       # rows written out
    grid_doy = keep + 1

    def curve(beta):
        r = np.exp(np.clip(G @ beta[2:], -50, 50))
        return (r / r.mean())[keep]
    offsets = {"share": total, "per_station": n_st}

    sp_ids, sp_inv = np.unique(species, return_inverse=True)
    sp_tot = np.bincount(sp_inv, weights=count)
    top = set(np.argsort(-sp_tot, kind="stable")[:N_TOP])
    rng = np.random.default_rng(SEED)
    boots = [block_index(n, rng) for _ in range(B)]  # shared by all fits

    out = {k: [] for k in ("species_id", "norm", "doy", "est", "lo", "hi")}
    for j, sid in enumerate(sp_ids):
        if j not in top:
            continue
        m = sp_inv == j
        y = np.bincount(di[m], weights=count[m], minlength=n)
        if (y > 0).sum() < MIN_DAYS:
            continue
        for norm, denom in offsets.items():
            ok = denom > 0
            off = np.log(np.where(ok, denom, 1.0))
            beta = fit_poisson(X[ok], y[ok], off[ok])
            est = curve(beta)
            draws = np.empty((B, grid_doy.size))
            for b, idx in enumerate(boots):
                idx = idx[ok[idx]]
                draws[b] = curve(fit_poisson(X[idx], y[idx], off[idx]))
            lo, hi = np.percentile(draws, [2.5, 97.5], axis=0)
            g = grid_doy.size
            out["species_id"].append(np.full(g, sid, dtype=np.int32))
            out["norm"] += [norm] * g
            out["doy"].append(grid_doy.astype(np.int16))
            for key, v in (("est", est), ("lo", lo), ("hi", hi)):
                out[key].append(v.astype(np.float32))

    tab = pa.table({
        "species_id": np.concatenate(out["species_id"]),
        "norm": pa.array(out["norm"]).dictionary_encode(),
        "doy": np.concatenate(out["doy"]),
        "est": np.concatenate(out["est"]),
        "lo": np.concatenate(out["lo"]),
        "hi": np.concatenate(out["hi"]),
    })
    pq.write_table(tab, OUT, compression="zstd")
    print(f"{OUT}: {tab.num_rows} rows, {len(set(tab['species_id'].to_pylist()))} species, "
          f"{OUT.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
