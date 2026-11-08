# CPU port of experiments/v6/features_dd.py, source sha256 0eaa0f581d19628dab8b8cfd9e48a1ec4fd53a66fd912ffbedf50bf0be841be8. See auc_lab_20261107/claude/audit/cpu_port_notes.md.
import sys as _s, pathlib as _p; _s.path.insert(0, str(_p.Path(__file__).resolve().parents[2])); import ku_workspace as KU_ENV
"""Difference-domain block (IDEAS.md tier 1, ideas 4 and 5).
For every series the level/spread, change-point and trend families of clean/features.py are recomputed on the FIRST DIFFERENCES
(standardised by the before-segment increments), prefix `dd_`, plus trend-change-versus-random-walk statistics on the raw segments.
Usage: python features_dd.py train|test|aug [n_jobs] [limit]   -> tune_out/dd_<train|test>.parquet or tune_out/aug_dd_features.parquet
`aug` rebuilds the boundary-shift negatives with the same rng as experiments/v5/augment.py. 4 processes at below-normal priority."""
import sys, time, pathlib, ctypes, warnings
KU_ENV.low_priority()
sys.stdout.reconfigure(encoding="utf-8"); warnings.filterwarnings("ignore")
D = KU_ENV.root(); sys.path.insert(0, str(KU_ENV.code_root() / "clean"))
import numpy as np, pandas as pd
from scipy import stats
from joblib import Parallel, delayed
import features as CF
EPS = 1e-12


def ols(t, x):
    sl, ic = np.polyfit(t, x, 1); r = x - (sl * t + ic); return sl, ic, r


def kpss_stat(r):
    """KPSS-type statistic of residuals: sum of squared partial sums / (n^2 * long-run variance).  Small -> stationary around trend."""
    n = len(r); s = np.cumsum(r); lrv = max(r.var(), EPS)
    return float((s ** 2).sum() / (n ** 2 * lrv))


def trend_vs_rw(a, b):
    """Does the after segment look like a trend change (non-zero mean increment, small increment variance, stationary residuals
    around a line) or a random-walk excursion (zero-mean, large-variance increments, residuals that wander)?"""
    f = {}; da, db = np.diff(a), np.diff(b); na, nb = len(a), len(b)
    sda, sdb = max(da.std(), EPS), max(db.std(), EPS)
    f["dd_drift_t_pre"] = da.mean() / (sda / np.sqrt(len(da)))                      # t-stat of non-zero drift of increments
    f["dd_drift_t_post"] = db.mean() / (sdb / np.sqrt(len(db)))
    f["dd_drift_t_gap"] = f["dd_drift_t_post"] - f["dd_drift_t_pre"]
    f["dd_drift_t_agap"] = abs(f["dd_drift_t_gap"])
    f["dd_drift_vs_rw"] = abs(db.mean() - da.mean()) / (sda * np.sqrt(1 / len(da) + 1 / len(db)))   # drift change in units the before-RW allows
    ta, tb, tt = np.arange(na), np.arange(na, na + nb), np.arange(na + nb)
    sa_, ia_, ra = ols(ta, a); sb_, ib_, rb = ols(tb, b); sp_, ip_, rp = ols(tt, np.concatenate([a, b]))
    rss_p, rss_s = (rp ** 2).sum(), (ra ** 2).sum() + (rb ** 2).sum()
    f["dd_chow_f"] = float(((rss_p - rss_s) / 2) / max(rss_s / (na + nb - 4), EPS))      # Chow test for a line change at the boundary
    f["dd_chow_logf"] = float(np.log1p(max(f["dd_chow_f"], 0)))
    f["dd_resid_acf1_pre"], f["dd_resid_acf1_post"] = CF.acf1(ra), CF.acf1(rb)       # ~1 for a random walk around a line, low for trend+noise
    f["dd_resid_acf1_gap"] = f["dd_resid_acf1_post"] - f["dd_resid_acf1_pre"]
    f["dd_kpss_pre"], f["dd_kpss_post"] = kpss_stat(ra), kpss_stat(rb)
    f["dd_kpss_gap"] = f["dd_kpss_post"] - f["dd_kpss_pre"]; f["dd_kpss_logratio"] = CF.log_ratio(f["dd_kpss_post"], f["dd_kpss_pre"])
    # slope of the after segment in units of what a random walk with the before increments would produce (sd of OLS slope of a RW ~ sd*sqrt(6/5n))
    f["dd_slope_post_vs_rw"] = sb_ / (sda * np.sqrt(6 / (5 * nb)))
    f["dd_slope_pre_vs_rw"] = sa_ / (sda * np.sqrt(6 / (5 * na)))
    f["dd_slope_gap_vs_rw"] = abs(f["dd_slope_post_vs_rw"] - f["dd_slope_pre_vs_rw"])
    # increment-variance ratio after/before and trend-noise ratio (slope*n over increment sd)
    f["dd_inc_logvar_ratio"] = CF.log_ratio(db.var(), da.var())
    f["dd_trend_noise_post"] = abs(sb_) * nb / sdb; f["dd_trend_noise_pre"] = abs(sa_) * na / sda
    f["dd_trend_noise_gap"] = f["dd_trend_noise_post"] - f["dd_trend_noise_pre"]
    # the boundary increment itself relative to typical increments
    f["dd_boundary_inc"] = (b[0] - a[-1]) / sda
    f["dd_boundary_inc_abs"] = abs(f["dd_boundary_inc"])
    # variance ratio of 4-step vs 1-step increments (1 for RW, <1 mean-reverting, >1 trending) pre/post
    def vr(x, q=4):
        dq = x[q:] - x[:-q]; return float(dq.var() / max(q * np.diff(x).var(), EPS))
    f["dd_vr4_pre"], f["dd_vr4_post"] = vr(a), vr(b); f["dd_vr4_gap"] = f["dd_vr4_post"] - f["dd_vr4_pre"]
    return f


def one(ident, a, b):
    da, db = np.diff(a), np.diff(b); m, s = da.mean(), max(da.std(), EPS)
    zda, zdb = (da - m) / s, (db - m) / s
    f = {"id": ident}
    for name, fn in (("basic", CF.basic_gaps), ("cp", CF.change_point), ("trend", CF.trend_stability)):
        try: g = fn(zda, zdb)
        except Exception: g = {}
        for k, v in g.items():
            if k in ("n_pre", "n_post", "log_n", "boundary_frac"): continue
            f[f"dd_{k}"] = v
    try: f.update(trend_vs_rw(a.astype(float), b.astype(float)))
    except Exception: pass
    return f


if __name__ == "__main__":
    name = sys.argv[1]; nj = int(sys.argv[2]) if len(sys.argv) > 2 else 4; lim = int(sys.argv[3]) if len(sys.argv) > 3 else None; t0 = time.time()
    jobs = []
    if name == "aug":
        MIN_SIDE = 78; rng = np.random.default_rng(1)
        s = KU_ENV.read_series(D / "data" / "train_series.parquet"); y = pd.read_csv(D / "data" / "train_labels.csv", dtype={"id": str}).set_index("id").label
        for ident, g in s.groupby("id", sort=False, observed=True):                 # identical loop and rng draws to experiments/v5/augment.py
            v = g["value"].to_numpy(float); p = g["period"].to_numpy(); a, b = v[p == 0], v[p == 1]
            if y[ident] == 0:
                k = int(len(v) * rng.uniform(0.3, 0.7))
                if k >= MIN_SIDE and len(v) - k >= MIN_SIDE: jobs.append((f"{ident}s", v[:k], v[k:]))
            for seg in (a, b):
                if len(seg) >= 2 * MIN_SIDE: rng.uniform(0.4, 0.6)
        meta = pd.read_csv(D / "tune_out/aug_meta.csv", dtype=str); want = set(meta.id[meta.kind == "shift"])
        jobs = [j for j in jobs if j[0] in want]; assert len(jobs) == len(want), (len(jobs), len(want)); out = D / "tune_out" / "aug_dd_features.parquet"
    else:
        s = KU_ENV.read_series(D / "data" / f"{name}_series.parquet")
        for ident, g in s.groupby("id", sort=False, observed=True):
            v = g["value"].to_numpy(float); p = g["period"].to_numpy(); jobs.append((ident, v[p == 0], v[p == 1]))
        out = D / "tune_out" / f"dd_{name}.parquet"
    del s
    if lim: jobs = jobs[:lim]; out = out.with_name(out.stem + "_sample.parquet")
    rows = Parallel(n_jobs=nj, batch_size=32)(delayed(one)(*j) for j in jobs)
    F = pd.DataFrame(rows).set_index("id").replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    F.to_parquet(out); print(f"{name}: {F.shape} -> {out.name} in {time.time()-t0:.0f}s ({(time.time()-t0)/len(jobs)*1000:.0f} ms/series)", flush=True)
