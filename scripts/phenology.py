"""Spring onset per species, and whether the rise looks like arrival or song.

Data: daily detection counts for the N_TOP species at the five long-running
PUC stations, on active station-days only (station reported any detection
that day). Each species is fitted on its own.

Model (hurdle negative binomial, per species j, site s, day d):

    P(y > 0)          = pi[s,d],   logit pi = g(d) + c[s]
    y - 1 | y > 0     ~ NegBin(mu, phi),  log mu = f(d) + b[s]

g and f are smooth curves over calendar time: second-order random walks on
weekly knots, linearly interpolated to days. c and b are site offsets with a
sum-to-zero constraint, so g and f describe the average site (pooled with
site offsets). The expected daily count is

    E[y](d) = pi(d) * (1 + mu(d))

The shifted (rather than zero-truncated) count part avoids a flat likelihood
as mu -> 0 when positive counts are mostly 1, which made chains disagree.

Season metrics, computed on every posterior draw for the average site:

    baseline B   mean of E[y] over the preceding winter (Dec 1 to Feb 28)
    onset        first day from Jan 15 to Jul 31 with E[y] >= max(2 B, 1) on
                 that day and the next K - 1 days (K = 7). The floor of one
                 detection per station-day keeps near-zero winter baselines
                 (summer migrants) from triggering on stray detections.
                 p_censored is the share of draws whose onset falls on Jan 15,
                 i.e. the rate already met the rule when the search began.
    departure    first day from Jan 15 to Jul 31 with E[y] <= B / 2 for K days
                 (for winter visitors, whose rate falls in spring)
    presence share
                 fraction of the log rise in E[y], from the winter mean to the
                 spring peak (Mar to Jul), carried by the presence part log pi
                 rather than the count part. Near 1: the species goes from
                 rarely detected to detected most days (arrival). Near 0: it
                 was already detected most winter days and sings more
                 (resident song cycle, or more birds of an already present
                 species).
    winter presence
                 mean pi over the winter window

The fraction of draws with an onset gives P(onset). The resident reference
is the mean log rise, E[y](d) / B, over the species labelled Resident in
data/species_status.csv; each species' excess over it is reported at its
spring peak.

Assumptions and caveats
- Detections are vocal activity weighted by detectability. A rise in E[y]
  can come from more birds, more song per bird, or both; the presence share
  separates them only partly (a resident pair that starts singing daily also
  raises presence).
- Winter 2024-25 and spring 2025 come from PUC-9883 alone; other sites start
  between October 2025 and January 2026.
- Off-season BirdNET false positives raise the winter baseline and can hide
  or delay an onset.
- Counts are conditionally independent given the curves; the random walk
  absorbs most serial correlation but not within-week bursts.

Run: uv run --group analysis python scripts/phenology.py
Outputs go to output/phenology/.
"""

import csv
import os
import time
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
from numpyro.infer import MCMC, NUTS

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SEED = 1834
STATIONS = [9883, 17755, 20422, 21599, 21688]
START = np.datetime64("2024-12-01")
TOP_FROM = np.datetime64("2026-01-03")  # species ranked as in site_compare.py
N_TOP = 25
K_PERSIST = 7
FLOOR = 1.0         # onset also needs E[y] >= 1 detection per station-day
KNOT = 7            # days between random-walk knots
CHAINS = 4
WARMUP = int(os.environ.get("BW_WARMUP", "1000"))
SAMPLES = int(os.environ.get("BW_SAMPLES", "1000"))
TARGET_ACCEPT = 0.95
KNOT_SD = 4.0       # prior SD of each knot value (logit or log scale)
F_CENTRED = os.environ.get("BW_F_CENTRED", "0") == "1"
DENSE = os.environ.get("BW_DENSE", "1") == "1"
THIN = 4            # draws kept for derived metrics (memory)
ONLY = os.environ.get("BW_ONLY")  # comma-separated common names, for testing

# Focused set. Residency labels are from data/species_status.csv where present;
# "(unlabelled)" marks species with no row there, classed by broad migratory
# strategy until a sourced row is added.
FOCUS = {
    "Carolina Wren": "Resident", "Northern Cardinal": "Resident",
    "Carolina Chickadee": "Resident", "Tufted Titmouse": "Resident",
    "Chimney Swift": "Summer (unlabelled)", "Chipping Sparrow": "Summer (unlabelled)",
    "Common Grackle": "Summer (unlabelled)",
    "White-throated Sparrow": "Migratory",
    "American Robin": "Both", "Eastern Phoebe": "Both",
    "Eastern Towhee": "Both", "American Goldfinch": "Both",
}
STATUS_COLOR = {"Resident": "#0072B2", "Both": "#009E73", "Migratory": "#D55E00",
                "Summer (unlabelled)": "#E69F00", "Not classified": "#999999"}

CACHE_DIR = Path(os.environ.get("BW_CACHE_DIR", "data/cache"))
OUT = Path("output/phenology")

numpyro.set_host_device_count(CHAINS)


# ---------------------------------------------------------------------- data

def read_data():
    files = sorted((CACHE_DIR / "counts").glob("*.parquet"))
    t = pa.concat_tables([pq.read_table(f) for f in files])
    t = t.filter(pc.and_(pc.is_in(t["station_id"], pa.array(STATIONS, pa.int32())),
                         pc.greater_equal(t["date"], pa.scalar(START.item(), pa.date32()))))
    sid = t["station_id"].to_numpy()
    day = pc.cast(t["date"], pa.int32()).to_numpy() - START.astype(int)
    spid = t["species_id"].to_numpy()
    cnt = t["count"].to_numpy().astype(np.int64)
    sp = pq.read_table(CACHE_DIR / "species.parquet").to_pylist()
    names = {r["species_id"]: (r["common_name"], r["scientific_name"]) for r in sp}
    with open("data/species_status.csv") as f:
        status = {r["scientific_name"]: r["status"] for r in csv.DictReader(f)}
    return sid, day, spid, cnt, names, status


def top_species(day, spid, cnt):
    sel = day >= (TOP_FROM - START).astype(int)
    u, inv = np.unique(spid[sel], return_inverse=True)
    tot = np.bincount(inv, weights=cnt[sel])
    return u[np.argsort(-tot)[:N_TOP]]


# --------------------------------------------------------------------- model

def rw2(name, n, scale, centred):
    """Second-order random walk on n knots. Centred (knot values sampled
    directly, Gaussian prior on second differences) for the count curve,
    which large daily counts inform well; non-centred (scaled increments) for
    the presence curve, which is weakly informed when P(detected) is near 0
    or 1 and otherwise forms a funnel with its scale."""
    sigma = numpyro.sample(f"{name}_sigma", dist.HalfNormal(scale))
    if centred:
        x = numpyro.sample(f"{name}_knots", dist.Normal(0, KNOT_SD).expand([n]))
        d2 = x[2:] - 2 * x[1:-1] + x[:-2]
        numpyro.factor(f"{name}_rw2", dist.Normal(0, sigma).log_prob(d2).sum())
        return x
    lvl = numpyro.sample(f"{name}_0", dist.Normal(0, 3))
    slope = numpyro.sample(f"{name}_slope", dist.Normal(0, 0.5))
    z = numpyro.sample(f"{name}_z", dist.Normal(0, 1).expand([n - 2]))
    acc = jnp.concatenate([jnp.zeros(2), jnp.cumsum(sigma * z)])
    x = lvl + jnp.cumsum(slope + acc) - slope
    # Weakly informative bound on each knot value. Without it the walk drifts
    # without limit through months with no detections (complete separation),
    # where only the random-walk prior constrains it, and chains disagree.
    numpyro.factor(f"{name}_ridge", dist.Normal(0, KNOT_SD).log_prob(x).sum())
    return x


def site_offsets(name, S, scale):
    raw = numpyro.sample(f"{name}_raw", dist.Normal(0, scale).expand([S]))
    return numpyro.deterministic(name, raw - raw.mean())


def hurdle(W, st, y, n_knots, S):
    g = numpyro.deterministic("g", rw2("g", n_knots, 0.5, centred=False))
    f = numpyro.deterministic("f", rw2("f", n_knots, 0.5, centred=F_CENTRED))
    c = site_offsets("c", S, 2.0)
    b = site_offsets("b", S, 2.0)
    phi = 1 / numpyro.sample("phi_inv", dist.Exponential(1))
    logit_pi = W @ g + c[st]
    log_mu = W @ f + b[st]
    nb = dist.NegativeBinomial2(jnp.exp(log_mu), phi).log_prob(jnp.maximum(y - 1, 0))
    ll = jnp.where(y == 0, jax.nn.log_sigmoid(-logit_pi), jax.nn.log_sigmoid(logit_pi) + nb)
    numpyro.factor("ll", ll.sum())


def interp_weights(day, n_days):
    """Sparse-as-dense linear interpolation matrix, days -> weekly knots."""
    n_knots = n_days // KNOT + 2
    pos = day / KNOT
    i0 = np.floor(pos).astype(int)
    w1 = pos - i0
    W = np.zeros((len(day), n_knots))
    W[np.arange(len(day)), i0] = 1 - w1
    W[np.arange(len(day)), i0 + 1] = w1
    return W, n_knots


# ------------------------------------------------------------------- metrics

def season_windows(n_days):
    out = []
    for yr in (2025, 2026):
        d = lambda s: int((np.datetime64(s) - START).astype(int))  # noqa: E731
        w0, w1 = max(0, d(f"{yr - 1}-12-01")), d(f"{yr}-03-01")
        s0, s1 = d(f"{yr}-01-15"), min(n_days, d(f"{yr}-08-01"))
        if s1 > s0:
            out.append((yr, w0, w1, s0, s1))
    return out


def first_run(mask, k):
    """Index of the first run of k True values along the last axis, or -1."""
    run = np.zeros(mask.shape[:-1], int)
    first = np.full(mask.shape[:-1], -1)
    for t in range(mask.shape[-1]):
        run = np.where(mask[..., t], run + 1, 0)
        hit = (run == k) & (first < 0)
        first = np.where(hit, t - k + 1, first)
    return first


def daily_curves(post, n_days):
    Wd, _ = interp_weights(np.arange(n_days), n_days)
    g = post["g"] @ Wd.T
    f = post["f"] @ Wd.T
    pi = 1 / (1 + np.exp(-g))
    ey = pi * (1 + np.exp(f))
    return pi, ey


def fmt_day(i):
    return str(START + np.timedelta64(int(i), "D")) if i >= 0 else ""


def q(x):
    return np.nanquantile(x, [0.5, 0.025, 0.975], axis=0)


# ---------------------------------------------------------------------- main

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sid, day, spid, cnt, names, status = read_data()
    n_days = int(day.max()) + 1
    S = len(STATIONS)
    st_of = {s: i for i, s in enumerate(STATIONS)}
    keys, inv = np.unique(np.stack([sid, day], 1), axis=0, return_inverse=True)
    inv = inv.ravel()
    st = np.array([st_of[s] for s in keys[:, 0]])
    W, n_knots = interp_weights(keys[:, 1], n_days)
    top = top_species(day, spid, cnt)
    if ONLY:
        want = set(ONLY.split(","))
        top = [j for j in top if names[int(j)][0] in want]
    seasons = season_windows(n_days)
    key = jax.random.PRNGKey(SEED)
    print(f"{len(keys)} active station-days, {n_days} days, {n_knots} knots")

    curves, rows, diag = {}, [], []
    for j in top:
        common, sci = names[int(j)]
        stat = FOCUS.get(common) or status.get(sci, "Not classified")
        y = np.zeros(len(keys), np.int64)
        np.add.at(y, inv[spid == j], cnt[spid == j])
        key, k1 = jax.random.split(key)
        t0 = time.time()
        mcmc = MCMC(NUTS(hurdle, target_accept_prob=TARGET_ACCEPT, dense_mass=DENSE), num_warmup=WARMUP,
                    num_samples=SAMPLES, num_chains=CHAINS, progress_bar=False)
        mcmc.run(k1, W=W, st=st, y=y, n_knots=n_knots, S=S, extra_fields=("diverging",))
        post_c = mcmc.get_samples(group_by_chain=True)
        div = int(mcmc.get_extra_fields()["diverging"].sum())
        summ = {k: v for k, v in summary(post_c).items() if not k.endswith(("_raw", "_z"))}
        rhat = max(float(np.nanmax(v["r_hat"])) for v in summ.values())
        ess = min(float(np.nanmin(v["n_eff"])) for v in summ.values())
        worst = sorted(((float(np.nanmax(v["r_hat"])), k) for k, v in summ.items()), reverse=True)[:3]
        print(f"  {common}: divergences={div}, max r-hat={rhat:.3f}, min ESS={ess:.0f}, "
              f"{time.time() - t0:.0f} s (worst: {', '.join(f'{k} {r:.3f}' for r, k in worst)})",
              flush=True)
        diag.append((common, div, rhat, ess))
        post = {k: np.asarray(v).reshape(-1, *v.shape[2:])[::THIN] for k, v in post_c.items()}
        pi, ey = daily_curves(post, n_days)

        for yr, w0, w1, s0, s1 in seasons:
            if not np.any((keys[:, 1] >= w0) & (keys[:, 1] < w1)):
                continue
            B = ey[:, w0:w1].mean(1, keepdims=True)
            on = first_run(ey[:, s0:s1] >= np.maximum(2 * B, FLOOR), K_PERSIST)
            off = first_run(ey[:, s0:s1] <= B / 2, K_PERSIST)
            ok_on, ok_off = on >= 0, off >= 0
            pk = s0 + np.argmax(ey[:, s0:s1], axis=1)
            idx = np.arange(len(pk))
            rise = np.log(ey[idx, pk]) - np.log(B[:, 0])
            pres = np.log(pi[idx, pk]) - np.log(pi[:, w0:w1].mean(1))
            share = np.where(rise > np.log(2), pres / rise, np.nan)
            on_q = q(np.where(ok_on, s0 + on, np.nan))
            off_q = q(np.where(ok_off, s0 + off, np.nan))
            rows.append(dict(
                species=common, status=stat, year=yr,
                p_onset=float(ok_on.mean()), p_censored=float(np.mean(on == 0)),
                onset=fmt_day(on_q[0]) if ok_on.mean() >= 0.5 else "",
                onset_lo=fmt_day(on_q[1]) if ok_on.mean() >= 0.5 else "",
                onset_hi=fmt_day(on_q[2]) if ok_on.mean() >= 0.5 else "",
                p_departure=float(ok_off.mean()),
                departure=fmt_day(off_q[0]) if ok_off.mean() >= 0.5 else "",
                departure_lo=fmt_day(off_q[1]) if ok_off.mean() >= 0.5 else "",
                departure_hi=fmt_day(off_q[2]) if ok_off.mean() >= 0.5 else "",
                winter_rate=q(B[:, 0]), peak_ratio=q(np.exp(rise)),
                winter_presence=q(pi[:, w0:w1].mean(1)),
                presence_share=q(share), p_share_gt_half=float(np.nanmean(share > 0.5)),
                _onset_draws=np.where(ok_on, s0 + on, -1), _rel=np.log(ey / B),
                _peak=pk))
        curves[common] = dict(status=stat, pi=q(pi), ey=q(ey), rel={})
        for r in rows:
            if r["species"] == common:
                curves[common]["rel"][r["year"]] = r["_rel"]

    # resident reference: mean log rise over labelled residents, per draw
    for yr, *_ in seasons:
        res = [r["_rel"] for r in rows if r["year"] == yr and r["status"] == "Resident"]
        if not res:
            continue
        n = min(len(x) for x in res)
        ref = np.mean([x[:n] for x in res], axis=0)
        for r in rows:
            if r["year"] == yr:
                idx = np.arange(n)
                ex = r["_rel"][:n][idx, r["_peak"][:n]] - ref[idx, r["_peak"][:n]]
                r["excess_over_residents"] = q(np.exp(ex))
                r["p_excess_gt_2"] = float(np.mean(ex > np.log(2)))
        curves["_ref"] = {yr: q(np.exp(ref))} | curves.get("_ref", {})

    write_tables(rows, diag)
    plot(curves, rows, seasons, n_days, keys)
    print(f"Wrote {OUT}/")


def write_tables(rows, diag):
    cols = ["species", "status", "year", "p_onset", "p_censored", "onset", "onset_lo", "onset_hi",
            "p_departure", "departure", "departure_lo", "departure_hi"]
    qcols = ["winter_rate", "peak_ratio", "winter_presence", "presence_share",
             "excess_over_residents"]
    with open(OUT / "onset.csv", "w") as f:
        f.write(",".join(cols + [f"{c}{s}" for c in qcols for s in ("", "_lo", "_hi")]
                         + ["p_share_gt_half", "p_excess_gt_2"]) + "\n")
        for r in rows:
            vals = [r[c] for c in cols]
            for c in qcols:
                vals += list(r.get(c, [np.nan] * 3))
            vals += [r["p_share_gt_half"], r.get("p_excess_gt_2", np.nan)]
            f.write(",".join(f"{v:.3g}" if isinstance(v, float) else str(v) for v in vals) + "\n")
    with open(OUT / "checks.csv", "w") as f:
        f.write("species,divergences,max_rhat,min_ess\n")
        for d in diag:
            f.write(f"{d[0]},{d[1]},{d[2]:.3f},{d[3]:.0f}\n")


# ---------------------------------------------------------------------- plots

def plot(curves, rows, seasons, n_days, keys):
    dates = START + np.arange(n_days).astype("timedelta64[D]")
    yr = max(s[0] for s in seasons)
    _, w0, w1, s0, s1 = [s for s in seasons if s[0] == yr][0]
    lo = w0 - 30
    x = dates[lo:s1 + 30]

    # 1. focused species: expected rate relative to winter baseline, and presence
    focus = [s for s in FOCUS if s in curves]
    nc = 4
    nr = int(np.ceil(len(focus) / nc))
    fig, axes = plt.subplots(2 * nr, nc, figsize=(12, 3.3 * nr + 1),
                             gridspec_kw={"height_ratios": [2.2, 1] * nr}, sharex=True)
    for i, sp in enumerate(focus):
        a1, a2 = axes[2 * (i // nc), i % nc], axes[2 * (i // nc) + 1, i % nc]
        cv = curves[sp]
        col = STATUS_COLOR.get(cv["status"], "#999999")
        rel = cv["rel"].get(yr)
        if rel is not None:
            rq = np.nanquantile(np.exp(rel[:, lo:s1 + 30]), [0.5, 0.025, 0.975], axis=0)
            a1.fill_between(x, rq[1], rq[2], color=col, alpha=0.25, lw=0)
            a1.plot(x, rq[0], color=col, lw=1.4)
        ref = curves.get("_ref", {}).get(yr)
        if ref is not None:
            a1.plot(x, ref[0][lo:s1 + 30], color="0.35", lw=0.9, ls=":")
        a1.axhline(1, color="0.6", lw=0.6)
        a1.axhline(2, color="0.3", lw=0.8, ls="--")
        a1.axvspan(dates[w0], dates[w1 - 1], color="0.9", lw=0)
        r = [r for r in rows if r["species"] == sp and r["year"] == yr]
        if r and r[0]["p_onset"] > 0:
            od = r[0]["_onset_draws"]
            od = od[od >= 0]
            on = np.quantile(od, [0.5, 0.025, 0.975]).astype(int)
            a1.axvspan(dates[on[1]], dates[on[2]], color=col, alpha=0.12, lw=0)
            a1.axvline(dates[on[0]], color=col, lw=1)
        a1.set_yscale("log")
        a1.set_title(f"{sp} ({cv['status']})", fontsize=9)
        pq_ = cv["pi"][:, lo:s1 + 30]
        a2.fill_between(x, pq_[1], pq_[2], color=col, alpha=0.25, lw=0)
        a2.plot(x, pq_[0], color=col, lw=1.2)
        a2.set_ylim(0, 1.02)
        a2.axvspan(dates[w0], dates[w1 - 1], color="0.9", lw=0)
        if i % nc == 0:
            a1.set_ylabel("Rate / winter mean")
            a2.set_ylabel("P(detected)")
    for ax in axes.ravel():
        ax.tick_params(labelsize=7)
    for ax in axes[-1]:
        for lab in ax.get_xticklabels():
            lab.set_rotation(45)
    fig.suptitle(f"{yr}: expected daily detections relative to winter mean (gray band), "
                 "95% CrI; dashed = doubling; dotted = resident reference;\n"
                 "vertical line and band = onset posterior median and 95% CrI; "
                 "lower panels = probability of any detection in a day", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(OUT / "focus_curves.png", dpi=150)
    plt.close(fig)

    # 2. onset dates, all species
    rr = sorted([r for r in rows if r["year"] == yr], key=lambda r: (r["onset"] or "9999"))
    fig, ax = plt.subplots(figsize=(7, 0.3 * len(rr) + 1.5))
    for i, r in enumerate(rr):
        col = STATUS_COLOR.get(r["status"], "#999999")
        if r["onset"]:
            m, a, b = (np.datetime64(r[k]) for k in ("onset", "onset_lo", "onset_hi"))
            ax.errorbar(m, i, xerr=[[m - a], [b - m]], fmt="o", color=col, capsize=2, ms=4)
        if r["departure"]:
            m, a, b = (np.datetime64(r[k]) for k in ("departure", "departure_lo", "departure_hi"))
            ax.errorbar(m, i, xerr=[[m - a], [b - m]], fmt="v", color=col, capsize=2, ms=5,
                        mfc="white")
    ax.set_yticks(range(len(rr)),
                  [f"{r['species']}  P(onset) {r['p_onset']:.2f}" for r in rr], fontsize=7)
    handles = [plt.Line2D([], [], marker="o", color=c, ls="") for c in STATUS_COLOR.values()]
    handles += [plt.Line2D([], [], marker="v", color="0.3", mfc="white", ls="")]
    ax.legend(handles, list(STATUS_COLOR) + ["Departure (rate halves)"], fontsize=7,
              loc="lower right", frameon=False)
    ax.set_xlabel(f"Date in {yr} (posterior median, 95% CrI)")
    ax.set_title(f"Onset: first of {K_PERSIST} days at 2x the winter mean, {yr}", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "onset_dates.png", dpi=150)
    plt.close(fig)

    # 3. arrival vs song: winter presence vs presence share of the rise
    fig, ax = plt.subplots(figsize=(6.5, 5))
    for r in [r for r in rows if r["year"] == yr]:
        col = STATUS_COLOR.get(r["status"], "#999999")
        wp, sh = r["winter_presence"], r["presence_share"]
        if np.isnan(sh[0]):
            continue
        ax.errorbar(wp[0], sh[0], xerr=[[wp[0] - wp[1]], [wp[2] - wp[0]]],
                    yerr=[[sh[0] - sh[1]], [sh[2] - sh[0]]], fmt="o", color=col,
                    capsize=2, ms=4, alpha=0.9)
        ax.annotate(r["species"], (wp[0], sh[0]), fontsize=6, xytext=(3, 3),
                    textcoords="offset points")
    ax.axhline(0.5, color="0.6", lw=0.6, ls="--")
    ax.set_xlabel("Winter P(detected on a day), average site")
    ax.set_ylabel("Share of spring rise carried by presence")
    ax.set_title(f"Arrival (upper left) vs more song from present birds (lower right), {yr}",
                 fontsize=9)
    handles = [plt.Line2D([], [], marker="o", color=c, ls="") for c in STATUS_COLOR.values()]
    ax.legend(handles, list(STATUS_COLOR), fontsize=7, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "arrival_vs_song.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
