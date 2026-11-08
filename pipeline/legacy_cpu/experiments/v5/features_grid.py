# CPU port of experiments/v5/features_grid.py, source sha256 6bbd0cb8b683b42583aca4895278ccc330268c5f30719e9ea36f057a02f540b2. See auc_lab_20261107/claude/audit/cpu_port_notes.md.
import sys as _s, pathlib as _p; _s.path.insert(0, str(_p.Path(__file__).resolve().parents[2])); import ku_workspace as KU_ENV
"""Feature block B, built by a different recipe from the clean block (ADIA 2nd / 11th place style):
transform grid x per-period stats x two-sample tests x histogram distances x compression.
Transforms of the pre-standardised series: z, cumsum, dense rank, abs, moving mean, moving std, diff, square.
Usage: python features_grid.py train|test [n_jobs] [limit]   -> tune_out/grid_<name>.parquet"""
import sys, time, pathlib, ctypes, warnings, zlib, bz2
KU_ENV.low_priority()
sys.stdout.reconfigure(encoding="utf-8"); warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy import stats
from joblib import Parallel, delayed
D = KU_ENV.root()
EPS = 1e-12


def moving(x, w, fn):
    v = np.lib.stride_tricks.sliding_window_view(x, w)
    return fn(v, axis=1)


def transforms(z, n_pre):
    """Views of the whole series; each returns (pre, post) arrays."""
    out = {"z": z, "cumsum": np.cumsum(z - z[:n_pre].mean()), "rank": stats.rankdata(z, method="dense") / len(z),
           "abs": np.abs(z), "diff": np.diff(z, prepend=z[0]), "sq": z ** 2}
    w = 10
    out["mmean"] = np.concatenate([np.full(w - 1, z[:w].mean()), moving(z, w, np.mean)])
    out["mstd"] = np.concatenate([np.full(w - 1, z[:w].std()), moving(z, w, np.std)])
    return {k: (v[:n_pre], v[n_pre:]) for k, v in out.items()}


def period_stats(x):
    q = np.quantile(x, [.25, .75])
    return {"mean": x.mean(), "std": x.std(), "med": float(np.median(x)), "min": x.min(), "max": x.max(),
            "skew": stats.skew(x), "kurt": stats.kurtosis(x), "q25": q[0], "q75": q[1],
            "acf1": float(np.corrcoef(x[:-1], x[1:])[0, 1]) if x.std() > EPS and len(x) > 3 else 0.0}


def gini(x):
    a = np.sort(np.abs(x)) + EPS; n = len(a)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(a) / (n * a.sum()))


def theil(x):
    a = np.abs(x) + EPS; m = a.mean()
    return float(np.mean(a / m * np.log(a / m)))


def lz_complexity(bits):
    """Lempel-Ziv 76 phrase count of a binary string, normalised by n / log2 n."""
    s = "".join("1" if b else "0" for b in bits); n = len(s)
    # Kaspar-Schuster implementation
    i, k, l, c, k_max = 0, 1, 1, 1, 1
    while l + k <= n:
        if s[i + k - 1] == s[l + k - 1]:
            k += 1
        else:
            k_max = max(k, k_max); i += 1
            if i == l:
                c += 1; l += k_max; i, k, k_max = 0, 1, 1
            else:
                k = 1
    return float(c / (n / np.log2(max(n, 2))))


def compress_ratio(x):
    b = np.clip(np.round((x - x.mean()) / (x.std() + EPS) * 32) + 128, 0, 255).astype(np.uint8).tobytes()
    return len(zlib.compress(b, 6)) / len(b), len(bz2.compress(b, 5)) / len(b)


def hist_distances(a, b, bins=20):
    lo, hi = min(a.min(), b.min()), max(a.max(), b.max())
    ha, _ = np.histogram(a, bins, (lo, hi)); hb, _ = np.histogram(b, bins, (lo, hi))
    pa, pb = ha / ha.sum() + 1e-9, hb / hb.sum() + 1e-9
    bc = float(np.sqrt(pa * pb).sum())
    m = (pa + pb) / 2
    return {"bhatt": -np.log(bc), "hell": float(np.sqrt(max(1 - bc, 0))),
            "js": float((stats.entropy(pa, m) + stats.entropy(pb, m)) / 2)}


def two_sample(a, b):
    f = {"ks": stats.ks_2samp(a, b).statistic, "mwu": stats.mannwhitneyu(a, b).statistic / (len(a) * len(b)) - .5,
         "levene": stats.levene(a, b).statistic, "fligner": stats.fligner(a, b).statistic,
         "mood": stats.mood(a, b)[0], "ansari": stats.ansari(a, b).statistic / (len(a) * (len(a) + len(b) + 1) / 2) - .5 if len(a) > 2 else 0.0,
         "bartlett": stats.bartlett(a, b).statistic, "cvm": stats.cramervonmises_2samp(a, b).statistic,
         "energy": stats.energy_distance(a, b), "wass": stats.wasserstein_distance(a, b),
         "bm": stats.brunnermunzel(a, b).statistic, "ttest": stats.ttest_ind(a, b, equal_var=False).statistic}
    return f


def one(ident, a, b):
    sa = max(a.std(), EPS); z = (np.concatenate([a, b]) - a.mean()) / sa; n_pre = len(a)
    f = {"id": ident}
    for t, (pa, pb) in transforms(z, n_pre).items():
        sp, sq = period_stats(pa), period_stats(pb)
        for k in sp:
            f[f"{t}_{k}_pre"], f[f"{t}_{k}_post"] = sp[k], sq[k]
            f[f"{t}_{k}_diff"] = sq[k] - sp[k]; f[f"{t}_{k}_ratio"] = sq[k] / (abs(sp[k]) + 1e-6)
        for k, v in two_sample(pa, pb).items():
            f[f"{t}_{k}"] = float(v)
        for k, v in hist_distances(pa, pb).items():
            f[f"{t}_{k}"] = float(v)
        if t in ("z", "abs", "diff", "sq", "mstd"):
            f[f"{t}_gini_pre"], f[f"{t}_gini_post"] = gini(pa), gini(pb); f[f"{t}_gini_gap"] = f[f"{t}_gini_post"] - f[f"{t}_gini_pre"]
            f[f"{t}_theil_pre"], f[f"{t}_theil_post"] = theil(pa), theil(pb); f[f"{t}_theil_gap"] = f[f"{t}_theil_post"] - f[f"{t}_theil_pre"]
        if t in ("z", "diff"):
            (za, ba), (zb, bb) = compress_ratio(pa), compress_ratio(pb)
            f[f"{t}_gz_pre"], f[f"{t}_gz_post"], f[f"{t}_gz_gap"] = za, zb, zb - za
            f[f"{t}_bz_pre"], f[f"{t}_bz_post"], f[f"{t}_bz_gap"] = ba, bb, bb - ba
            la, lb = lz_complexity(pa > np.median(pa)), lz_complexity(pb > np.median(pb))
            f[f"{t}_lz_pre"], f[f"{t}_lz_post"], f[f"{t}_lz_gap"] = la, lb, lb - la
    # adaptive window pair (7th place): last 10% of pre vs first 60% of post, on z
    wa, wb = z[int(n_pre * 0.9):n_pre], z[n_pre:n_pre + int(len(b) * 0.6)]
    if len(wa) > 5 and len(wb) > 5:
        f["win_mean_gap"] = wb.mean() - wa.mean(); f["win_std_ratio"] = wb.std() / (wa.std() + EPS)
        f["win_ks"] = stats.ks_2samp(wa, wb).statistic; f["win_t"] = stats.ttest_ind(wa, wb, equal_var=False).statistic
    return f


if __name__ == "__main__":
    name = sys.argv[1]; nj = int(sys.argv[2]) if len(sys.argv) > 2 else 1; t0 = time.time()
    lim = int(sys.argv[3]) if len(sys.argv) > 3 else None
    s = KU_ENV.read_series(D / "data" / f"{name}_series.parquet")
    jobs = []
    for ident, g in s.groupby("id", sort=False, observed=True):
        v = g["value"].to_numpy(float); p = g["period"].to_numpy(); jobs.append((ident, v[p == 0], v[p == 1]))
    if lim:
        jobs = jobs[:lim]
    rows = Parallel(n_jobs=nj, batch_size=32)(delayed(one)(*j) for j in jobs)
    F = pd.DataFrame(rows).set_index("id").replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    out = D / "tune_out" / f"grid_{name}{'_sample' if lim else ''}.parquet"; F.to_parquet(out)
    print(f"{name}: {F.shape} -> {out.name} in {time.time()-t0:.0f}s ({(time.time()-t0)/len(jobs)*1000:.0f} ms/series)")
