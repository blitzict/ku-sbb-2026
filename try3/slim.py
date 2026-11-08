"""try3 fast path: compute only the feature columns the saved try3 models read.

The full recipe's grid / grid2 / dd / lrz6 / base-feature steps make 977-504 columns each; the 15 try3 models read
3-19 of them per table. These functions call the same original code (imported from the work folder) on the same
inputs, in the same order, and keep only the needed columns, so the values are identical (checked on all 4,000
test series against the full tables, see check_slim()).

python try3/slim.py <table> <work>        #table: base | grid | grid2 | dd | lrz6
"""
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
EPS = 1e-12

#columns each table must provide (the union over the 5 folds of model/columns_fold*.json)
NEED = {
    "base": ["abs_acf80_gap", "abs_ar1_kurt_gap", "abs_ar4_q90_gap", "abs_ar4_q99_gap", "abs_own4_q99_gap", "abs_raw_ent_gap",
             "absac10_gap", "band1_gap", "cz2_local_peak", "d1_2s_hist_l1", "gks_argmax_off", "gmwu_argmax_off",
             "own4_bm8_kurt_gap", "pacf5_post", "q_logstd_min", "r_emb3_mmd", "vroll48_2s_cvm", "z_emb3_energy", "z_sax_l1"],
    "grid": ["cumsum_kurt_ratio", "diff_gini_pre", "z_med_ratio"],
    "grid2": ["absdiff_acf1_ratio", "ewstd_kurt_ratio", "mrng10_cvm", "mstd20_cvm", "mstd20_skew_ratio", "r_mrng10_cvm"],
    "dd": ["dd_cabs_argmax_off", "dd_trend_noise_post", "dd_win_lv_last"],
    "lrz6": ["lrz_lr_ar_lr_excess"],
}
OUT = {"base": "clean/features_test.parquet", "grid": "tune_out/grid_test.parquet", "grid2": "tune_out/grid2_test.parquet",
       "dd": "tune_out/dd_test.parquet", "lrz6": "tune_out/lrz6_test.parquet"}


def load(work):
    #the work folder's own (frozen) feature modules
    for sub in ("", "clean", "experiments/v5", "experiments/v6"):
        p = str(Path(work) / sub)
        if p not in sys.path:
            sys.path.insert(0, p)
    import features as CF
    import features_grid as G
    return CF, G


def fast_stats(*modules):
    #exact statistic-only scipy paths (pipeline/fast_feature_stats.py); only with the pinned scipy it was checked on
    try:
        from pipeline.fast_feature_stats import install
        install(*modules)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------------------------- base (clean/features.py)
def base_row(CF, ident, a, b):
    sa = max(a.std(), EPS)
    za, zb = (a - a.mean()) / sa, (b - a.mean()) / sa
    rng = np.random.default_rng(int(ident[:8], 16))
    f = {"id": ident}
    ac = CF.autocorrelation(za, zb)
    f["pacf5_post"] = ac["pacf5_post"]
    ts = CF.trend_stability(za, zb)
    f["band1_gap"], f["q_logstd_min"] = ts["band1_gap"], ts["q_logstd_min"]
    f.update(change_point_part(CF, za, zb))
    f.update(innovation_part(CF, za, zb))
    f.update(volatility_part(CF, za, zb))
    jl = CF.joint_law(za, zb, rng)
    for k in ("r_emb3_mmd", "z_emb3_energy", "z_sax_l1"):
        if k in jl:
            f[k] = jl[k]
    return f


def change_point_part(CF, za, zb):
    #cz2_local_peak, gks_argmax_off, gmwu_argmax_off from CF.change_point
    stats = CF.stats
    f = {}
    z = np.concatenate([za, zb])
    n, b = len(z), len(za)
    s = z ** 2
    m = len(s)
    if not (b < 8 or b > m - 8):
        curve = CF.cusum_curve(s)
        w = max(int(0.05 * m), 4)
        lo, hi = max(b - 1 - w, 0), min(b - 1 + w, m - 2)
        f["cz2_local_peak"] = float(curve[b - 1] / max(curve[lo:hi + 1].max(), EPS))
    grid = np.unique(np.linspace(int(0.15 * n), int(0.85 * n), 40).astype(int))
    grid = grid[(grid >= 16) & (grid <= n - 16)]
    tests = {"ks": lambda l, r: stats.ks_2samp(l, r, method="asymp").statistic,
             "mwu": lambda l, r: abs(stats.mannwhitneyu(l, r).statistic / (len(l) * len(r)) - 0.5)}
    for name, fn in tests.items():
        curve = np.array([fn(z[:g], z[g:]) for g in grid])
        f[f"g{name}_argmax_off"] = (grid[int(curve.argmax())] - b) / n if len(curve) else 0.0
    return f


def innovation_part(CF, za, zb):
    f = {}
    w = {}
    for p in (1, 4):
        c, i, _ = CF.ar_fit(za, p)
        w[f"ar{p}"] = (CF.ar_apply(za, c, i), CF.ar_apply(zb, c, i))
    ca, ia, _ = CF.ar_fit(za, 4)
    cb, ib, _ = CF.ar_fit(zb, 4)
    w["own4"] = (CF.ar_apply(za, ca, ia), CF.ar_apply(zb, cb, ib))
    w["raw"] = (za, zb)
    want = {"ar1": ["kurt"], "ar4": ["q90", "q99"], "own4": ["q99"], "raw": ["ent"]}
    for key, ks in want.items():
        sa, sb = CF.shape_stats(w[key][0]), CF.shape_stats(w[key][1])
        for k in ks:
            f[f"abs_{key}_{k}_gap"] = abs(sb[k] - sa[k])
        if key == "own4":
            f["own4_bm8_kurt_gap"] = sb["bm8_kurt"] - sa["bm8_kurt"]
    #d1 whitening: two_sample_shape hist_l1 only
    ra, rb = np.diff(za), np.diff(zb)
    a = (ra - ra.mean()) / max(ra.std(), EPS)
    b = (rb - rb.mean()) / max(rb.std(), EPS)
    edges = np.linspace(-4, 4, 17)
    ha = np.histogram(np.clip(a, -4, 4), edges)[0] / len(a)
    hb = np.histogram(np.clip(b, -4, 4), edges)[0] / len(b)
    f["d1_2s_hist_l1"] = float(np.abs(ha - hb).sum())
    f["abs_acf80_gap"] = abs(CF.acf(zb, [80])[0] - CF.acf(za, [80])[0])
    return f


def volatility_part(CF, za, zb):
    f = {}
    c, i, ra = CF.ar_fit(za, 4)
    rb = CF.ar_apply(zb, c, i)

    def vol_std(r, w=48):
        v = pd.Series(r ** 2).rolling(w, min_periods=4).mean().bfill().to_numpy()
        return r / np.sqrt(np.maximum(v, EPS))
    va, vb = vol_std(ra), vol_std(rb)
    xa = (va - va.mean()) / max(va.std(), EPS)
    xb = (vb - vb.mean()) / max(vb.std(), EPS)
    f["vroll48_2s_cvm"] = CF.stats.cramervonmises_2samp(xa, xb).statistic
    f["absac10_gap"] = CF.acf(np.abs(rb), [10])[0] - CF.acf(np.abs(ra), [10])[0]
    return f


# ---------------------------------------------------------------------------------------------- grid / grid2 / dd
def grid_row(G, ident, a, b):
    sa = max(a.std(), EPS)
    z = (np.concatenate([a, b]) - a.mean()) / sa
    n_pre = len(a)
    f = {"id": ident}
    cs = np.cumsum(z - z[:n_pre].mean())
    kp, kq = G.stats.kurtosis(cs[:n_pre]), G.stats.kurtosis(cs[n_pre:])
    f["cumsum_kurt_ratio"] = kq / (abs(kp) + 1e-6)
    d = np.diff(z, prepend=z[0])
    f["diff_gini_pre"] = G.gini(d[:n_pre])
    mp, mq = float(np.median(z[:n_pre])), float(np.median(z[n_pre:]))
    f["z_med_ratio"] = mq / (abs(mp) + 1e-6)
    return f


def grid2_row(G, CF, ident, a, b):
    stats = G.stats
    sa = max(a.std(), EPS)
    z = (np.concatenate([a, b]) - a.mean()) / sa
    n_pre = len(a)
    f = {"id": ident}

    def pad(v, n, fill):
        return np.concatenate([np.full(n - len(v), fill), v])

    def acf1(x):
        return float(np.corrcoef(x[:-1], x[1:])[0, 1]) if x.std() > EPS and len(x) > 3 else 0.0
    n = len(z)
    d = np.diff(z, prepend=z[0])
    ad = np.abs(d)
    f["absdiff_acf1_ratio"] = acf1(ad[n_pre:]) / (abs(acf1(ad[:n_pre])) + 1e-6)
    ew = np.sqrt(pd.Series(z).ewm(span=10).var().bfill().to_numpy() + EPS)
    f["ewstd_kurt_ratio"] = stats.kurtosis(ew[n_pre:]) / (abs(stats.kurtosis(ew[:n_pre])) + 1e-6)
    if n > 15:
        v = pad(G.moving(z, 10, np.max) - G.moving(z, 10, np.min), n, np.ptp(z[:10]))
        f["mrng10_cvm"] = float(stats.cramervonmises_2samp(v[:n_pre], v[n_pre:]).statistic)
    if n > 25:
        v = pad(G.moving(z, 20, np.std), n, z[:20].std())
        f["mstd20_cvm"] = float(stats.cramervonmises_2samp(v[:n_pre], v[n_pre:]).statistic)
        f["mstd20_skew_ratio"] = stats.skew(v[n_pre:]) / (abs(stats.skew(v[:n_pre])) + 1e-6)
    coefs, c0, ra = CF.ar_fit(z[:n_pre], 4)
    rb = CF.ar_apply(z[n_pre:], coefs, c0)
    r = np.concatenate([ra, rb])
    n_rpre, nr = len(ra), len(r)
    if nr > 15:
        v = pad(G.moving(r, 10, np.max) - G.moving(r, 10, np.min), nr, np.ptp(r[:10]))
        f["r_mrng10_cvm"] = float(stats.cramervonmises_2samp(v[:n_rpre], v[n_rpre:]).statistic)
    return f


def dd_row(CF, ident, a, b):
    f = {"id": ident}
    da, db = np.diff(a), np.diff(b)
    m, s = da.mean(), max(da.std(), EPS)
    zda, zdb = (da - m) / s, (db - m) / s
    try:
        #CF.change_point(zda, zdb) pieces: cabs CUSUM argmax and the last after-window log variance
        z = np.concatenate([zda, zdb])
        n, bb = len(z), len(zda)
        part = {}
        if not (bb < 8 or bb > n - 8):
            curve = CF.cusum_curve(np.abs(z))
            part["dd_cabs_argmax_off"] = (np.arange(1, n)[int(curve.argmax())] - bb) / n if len(curve) else 0.0
        wl = max(min(64, len(zdb) // 2), 16)
        starts = np.arange(0, len(zdb) - wl + 1, max(wl // 2, 1))
        if len(starts) == 0:
            raise ValueError("no after windows")          #the full function fails here too (min of an empty array)
        part["dd_win_lv_last"] = np.log(max(zdb[starts[-1]:starts[-1] + wl].var(), EPS))
        f.update(part)
    except Exception:
        pass
    try:
        a, b = a.astype(float), b.astype(float)
        db = np.diff(b)
        nb = len(b)
        sdb = max(db.std(), EPS)
        tb = np.arange(len(a), len(a) + nb)
        sb_, _ = np.polyfit(tb, b, 1)
        f["dd_trend_noise_post"] = abs(sb_) * nb / sdb
    except Exception:
        pass
    return f


# ---------------------------------------------------------------------------------------------- lrz6 (features_lrz.py)
def lr_ar_block(FL, js):
    #features_lr.block() up to f["lr_ar_lr"], same tensors and order, the other statistics skipped
    import torch
    dev, L, PMAX = FL.dev, FL.L, FL.PMAX
    N = len(js); X = torch.zeros(N, L, device=dev); n = torch.tensor([min(len(v), L) for _, v, _ in js], device=dev); b = torch.tensor([k for _, _, k in js], device=dev)
    for i, (_, v, _) in enumerate(js): X[i, :min(len(v), L)] = torch.as_tensor(v[:L], device=dev, dtype=torch.float32)
    tt = torch.arange(L, device=dev)[None, :]; pre = (tt < b[:, None]).float(); post = ((tt >= b[:, None]) & (tt < n[:, None])).float(); valid = pre + post
    m = (X * pre).sum(1) / b; sd = torch.sqrt((((X - m[:, None]) * pre) ** 2).sum(1) / b).clamp(min=1e-6); Z = (X - m[:, None]) / sd[:, None] * valid

    def fit(p, mask):
        if p == 0:
            c = (Z * mask).sum(1) / mask.sum(1).clamp(min=1); r = (Z - c[:, None]) * valid; return c[:, None], r, mask.sum(1)
        U = Z.unfold(1, p + 1, 1); yv = U[:, :, -1]; A = torch.cat([U[:, :, :-1].flip(-1), torch.ones_like(yv)[..., None]], -1)
        mk = mask[:, p:][..., None]; Am = A * mk; G = Am.transpose(1, 2) @ A + 1e-4 * torch.eye(p + 1, device=dev); h = Am.transpose(1, 2) @ yv[..., None]
        beta = torch.linalg.solve(G, h).squeeze(-1); res = torch.zeros(N, L, device=dev); res[:, p:] = (yv - (A @ beta[..., None]).squeeze(-1)) * valid[:, p:]
        return beta, res, mask[:, p:].sum(1)
    fits = {}; bics = []
    for p in range(PMAX + 1):
        beta, r, ne = fit(p, pre); s2 = ((r * pre) ** 2).sum(1) / ne.clamp(min=1); bics.append(ne * torch.log(s2.clamp(min=1e-9)) + (p + 1) * torch.log(ne.clamp(min=2))); fits[p] = (beta, r, s2)
    pb = torch.stack(bics, 1).argmin(1)
    R = torch.stack([fits[p][1] for p in range(PMAX + 1)], 1)[torch.arange(N, device=dev), pb]
    vpost = post.clone()
    nb = vpost.sum(1)
    lr_ar = torch.zeros(N, device=dev); rss_keep = (R ** 2 * vpost).sum(1)
    for p in range(PMAX + 1):
        sel = pb == p
        if sel.any():                                 #orders no series selected are skipped (their result is never used)
            _, r2, ne2 = fit(p, post)
            rss_new = (r2 ** 2 * post).sum(1); lr_ar[sel] = (nb * torch.log(rss_keep.clamp(min=1e-9) / rss_new.clamp(min=1e-9)))[sel] - ((p + 1) * torch.log(nb.clamp(min=2)))[sel]
    return pd.DataFrame({"lr_ar_lr": lr_ar.float().cpu().numpy()}, index=[j[0] for j in js])


def lrz6_table(work):
    sys.argv = ["features_lrz.py", "test", "6"]
    import features_lr as FL
    jobs, _ = FL.build_jobs("test")
    K, MIN, CH = 6, 60, int(os.environ.get("CHUNK", "256"))

    def table(js):
        return pd.concat([lr_ar_block(FL, js[i:i + CH]) for i in range(0, len(js), CH)]).replace([np.inf, -np.inf], np.nan)
    real = table(jobs)
    nulls = []
    for side in ("pre", "post"):
        for fr in np.linspace(0.3, 0.7, K):
            js = []
            for ident, v, b in jobs:
                seg = v[:b] if side == "pre" else v[b:]; k = int(len(seg) * fr)
                js.append((ident, seg, k) if len(seg) >= 2 * MIN and k >= MIN and len(seg) - k >= MIN else (ident, v, b))
            T = table(js)
            T["_valid"] = [len(v[:b] if side == "pre" else v[b:]) >= 2 * MIN for _, v, b in jobs]
            nulls.append(T)
    N = pd.concat(nulls)
    N.loc[~N._valid, "lr_ar_lr"] = np.nan
    mu = N.groupby(level=0)["lr_ar_lr"].mean().loc[real.index]
    out = pd.DataFrame({"lrz_lr_ar_lr_excess": real["lr_ar_lr"] - mu}, index=real.index)
    return out


# ---------------------------------------------------------------------------------------------- driver
def series_jobs(work):
    import ku_workspace as KU
    s = KU.read_series(Path(work) / "data/test_series.parquet")
    jobs = []
    for ident, g in s.groupby("id", sort=False, observed=True):
        v = g["value"].to_numpy(float); p = g["period"].to_numpy(); jobs.append((ident, v[p == 0], v[p == 1]))
    return jobs


def _rows(args):
    table, work, chunk = args
    CF, G = load(work)
    fast_stats(CF, G)
    fn = {"base": lambda j: base_row(CF, *j), "grid": lambda j: grid_row(G, *j),
          "grid2": lambda j: grid2_row(G, CF, *j), "dd": lambda j: dd_row(CF, *j)}[table]
    return [fn(j) for j in chunk]


def finish(rows_or_frame, table):
    F = rows_or_frame if isinstance(rows_or_frame, pd.DataFrame) else pd.DataFrame(rows_or_frame).set_index("id")
    F = F.reindex(columns=NEED[table])
    F = F.replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    F.index.name = "id"
    return F


def build(table, work, workers=1, ids=None):
    work = str(Path(work).resolve())
    load(work)
    if table == "lrz6":
        return finish(lrz6_table(work), table)
    jobs = series_jobs(work)
    if ids is not None:
        keep = set(ids); jobs = [j for j in jobs if j[0] in keep]
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        size = max(len(jobs) // (workers * 8), 1)
        chunks = [jobs[i:i + size] for i in range(0, len(jobs), size)]
        with ProcessPoolExecutor(workers) as pool:
            rows = [r for part in pool.map(_rows, [(table, work, c) for c in chunks]) for r in part]
    else:
        rows = _rows((table, work, jobs))
    return finish(rows, table)


def main():
    table, work = sys.argv[1], Path(sys.argv[2])
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    t0 = time.time()
    F = build(table, work, workers)
    out = work / OUT[table]
    out.parent.mkdir(parents=True, exist_ok=True)
    F.to_parquet(out)
    print(f"slim {table}: {F.shape} -> {OUT[table]} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
