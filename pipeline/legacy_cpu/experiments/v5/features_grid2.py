# CPU port of experiments/v5/features_grid2.py, source sha256 a624a018666927be469adf064c409d9d26ee9a505dc668fab30122909957c5f0. See auc_lab_20261107/claude/audit/cpu_port_notes.md.
import sys as _s, pathlib as _p; _s.path.insert(0, str(_p.Path(__file__).resolve().parents[2])); import ku_workspace as KU_ENV
"""Grid block 2: follows where grid block 1's gain came from (moving-std and diff views).
More volatility views: moving std at windows 5/20/40, moving range 10, EWMA std, |diff|, second diff, and the
same views on AR(4)-whitened residuals. Same per-period stats + two-sample tests + histogram distances.
Usage: python features_grid2.py train|test [n_jobs] [limit] -> tune_out/grid2_<name>.parquet"""
import sys, time, pathlib, ctypes, warnings
KU_ENV.low_priority()
sys.stdout.reconfigure(encoding="utf-8"); warnings.filterwarnings("ignore")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
sys.path.insert(0, str(KU_ENV.code_root() / "clean"))
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from features_grid import moving, period_stats, two_sample, hist_distances, EPS
import features as CF
D = KU_ENV.root()


def pad(v, n, fill):
    return np.concatenate([np.full(n - len(v), fill), v])


def views(x, tag):
    n = len(x); out = {}
    for w in (5, 20, 40):
        if n > w + 5: out[f"{tag}mstd{w}"] = pad(moving(x, w, np.std), n, x[:w].std())
    if n > 15:
        out[f"{tag}mrng10"] = pad(moving(x, 10, np.max) - moving(x, 10, np.min), n, np.ptp(x[:10]))
    out[f"{tag}ewstd"] = np.sqrt(pd.Series(x).ewm(span=10).var().bfill().to_numpy() + EPS)
    d = np.diff(x, prepend=x[0]); out[f"{tag}absdiff"] = np.abs(d); out[f"{tag}diff2"] = np.diff(d, prepend=d[0])
    return out


def one(ident, a, b):
    sa = max(a.std(), EPS); z = (np.concatenate([a, b]) - a.mean()) / sa; n_pre = len(a)
    coefs, c0, ra = CF.ar_fit(z[:n_pre], 4); rb = CF.ar_apply(z[n_pre:], coefs, c0)
    r = np.concatenate([ra, rb]); n_rpre = len(ra)
    f = {"id": ident}
    for t, v in views(z, "").items():
        pa, pb = v[:n_pre], v[n_pre:]
        sp, sq = period_stats(pa), period_stats(pb)
        for k in sp:
            f[f"{t}_{k}_pre"], f[f"{t}_{k}_post"] = sp[k], sq[k]; f[f"{t}_{k}_diff"] = sq[k] - sp[k]; f[f"{t}_{k}_ratio"] = sq[k] / (abs(sp[k]) + 1e-6)
        for k, val in two_sample(pa, pb).items(): f[f"{t}_{k}"] = float(val)
        for k, val in hist_distances(pa, pb).items(): f[f"{t}_{k}"] = float(val)
    for t, v in views(r, "r_").items():          # residual-based views: tests + a few stats only
        pa, pb = v[:n_rpre], v[n_rpre:]
        for k, val in two_sample(pa, pb).items(): f[f"{t}_{k}"] = float(val)
        for k, val in hist_distances(pa, pb).items(): f[f"{t}_{k}"] = float(val)
        f[f"{t}_mean_ratio"] = pb.mean() / (abs(pa.mean()) + 1e-6); f[f"{t}_std_ratio"] = pb.std() / (pa.std() + 1e-6)
    return f


if __name__ == "__main__":
    name = sys.argv[1]; nj = int(sys.argv[2]) if len(sys.argv) > 2 else 1; t0 = time.time()
    lim = int(sys.argv[3]) if len(sys.argv) > 3 else None
    s = KU_ENV.read_series(D / "data" / f"{name}_series.parquet")
    jobs = []
    for ident, g in s.groupby("id", sort=False, observed=True):
        v = g["value"].to_numpy(float); p = g["period"].to_numpy(); jobs.append((ident, v[p == 0], v[p == 1]))
    if lim: jobs = jobs[:lim]
    rows = Parallel(n_jobs=nj, batch_size=32)(delayed(one)(*j) for j in jobs)
    F = pd.DataFrame(rows).set_index("id").replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    out = D / "tune_out" / f"grid2_{name}{'_sample' if lim else ''}.parquet"; F.to_parquet(out)
    print(f"{name}: {F.shape} -> {out.name} in {time.time()-t0:.0f}s ({(time.time()-t0)/len(jobs)*1000:.0f} ms/series)")
