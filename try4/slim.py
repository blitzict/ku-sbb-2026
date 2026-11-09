"""try4 fast path: compute only the feature columns the saved try4 models read (model folder: KU_TRY4_MODEL).

The full recipe's grid / grid2 / dd / lrz6 / base-feature steps make 977-504 columns each; the 15 try4 models read
3-19 of them per table. These functions call the same original code (imported from the work folder) on the same
inputs, in the same order, and keep only the needed columns, so the values are identical (checked on all 4,000
test series against the full tables, see check_slim()).

python try4/slim.py <table> <work>        #table: base | grid | grid2 | dd | lrz6
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

#columns each table must provide: read from the model folder in use (slim_cols.needed, union over its 5 folds)
TABLE = {"base": "features", "grid": "grid", "grid2": "grid2", "dd": "dd", "lrz6": "lrz6"}
#columns the hand-written base / lrz6 paths produce; anything else falls back to the original code
BASE_HAND = ["abs_acf80_gap", "abs_ar1_kurt_gap", "abs_ar4_q90_gap", "abs_ar4_q99_gap", "abs_own4_q99_gap", "abs_raw_ent_gap",
             "absac10_gap", "band1_gap", "cz2_local_peak", "d1_2s_hist_l1", "gks_argmax_off", "gmwu_argmax_off",
             "own4_bm8_kurt_gap", "pacf5_post", "q_logstd_min", "r_emb3_mmd", "vroll48_2s_cvm", "z_emb3_energy", "z_sax_l1"]
LRZ6_HAND = ["lrz_lr_ar_lr_excess"]


def _need_for(table):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import slim_cols
    return slim_cols.needed(TABLE[table])


NEED = {}         #filled per table on first use (worker processes rebuild it from KU_TRY4_MODEL)


def cols(table):
    if table not in NEED:
        NEED[table] = _need_for(table)
    return NEED[table]


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
#every column is computed by name with the same expressions as the original one(); names the parser does not know
#are taken from the original one() for that series (exact, just slower)
STATS = ("mean", "std", "med", "min", "max", "skew", "kurt", "q25", "q75", "acf1")
TESTS = ("ks", "mwu", "levene", "fligner", "mood", "ansari", "bartlett", "cvm", "energy", "wass", "bm", "ttest")
HISTS = ("bhatt", "hell", "js")


def two_sample_one(stats, k, a, b):
    #one entry of features_grid.two_sample(a, b)
    if k == "ks":
        return stats.ks_2samp(a, b).statistic
    if k == "mwu":
        return stats.mannwhitneyu(a, b).statistic / (len(a) * len(b)) - .5
    if k == "levene":
        return stats.levene(a, b).statistic
    if k == "fligner":
        return stats.fligner(a, b).statistic
    if k == "mood":
        return stats.mood(a, b)[0]
    if k == "ansari":
        return stats.ansari(a, b).statistic / (len(a) * (len(a) + len(b) + 1) / 2) - .5 if len(a) > 2 else 0.0
    if k == "bartlett":
        return stats.bartlett(a, b).statistic
    if k == "cvm":
        return stats.cramervonmises_2samp(a, b).statistic
    if k == "energy":
        return stats.energy_distance(a, b)
    if k == "wass":
        return stats.wasserstein_distance(a, b)
    if k == "bm":
        return stats.brunnermunzel(a, b).statistic
    if k == "ttest":
        return stats.ttest_ind(a, b, equal_var=False).statistic
    raise KeyError(k)


def view_columns(G, f, t, rest_list, pa, pb, extra=None):
    #per-view columns of features_grid.one / features_grid2.one: period stats, two-sample tests, histogram distances
    extra = extra or {}
    sp = sq = hd = None
    for rest in rest_list:
        stat, _, form = rest.rpartition("_")
        if rest in extra:
            extra[rest](f, f"{t}_{rest}", pa, pb)
        elif stat in STATS and form in ("pre", "post", "diff", "ratio"):
            if sp is None:
                sp, sq = G.period_stats(pa), G.period_stats(pb)
            if form == "pre":
                f[f"{t}_{rest}"] = sp[stat]
            elif form == "post":
                f[f"{t}_{rest}"] = sq[stat]
            elif form == "diff":
                f[f"{t}_{rest}"] = sq[stat] - sp[stat]
            else:
                f[f"{t}_{rest}"] = sq[stat] / (abs(sp[stat]) + 1e-6)
        elif rest in TESTS:
            f[f"{t}_{rest}"] = float(two_sample_one(G.stats, rest, pa, pb))
        elif rest in HISTS:
            if hd is None:
                hd = G.hist_distances(pa, pb)
            f[f"{t}_{rest}"] = float(hd[rest])
        else:
            raise KeyError(rest)


GRID_VIEWS = ("z", "cumsum", "rank", "abs", "diff", "sq", "mmean", "mstd")


def grid_view(G, z, n_pre, t):
    #one entry of features_grid.transforms(z, n_pre)
    w = 10
    if t == "z":
        v = z
    elif t == "cumsum":
        v = np.cumsum(z - z[:n_pre].mean())
    elif t == "rank":
        v = G.stats.rankdata(z, method="dense") / len(z)
    elif t == "abs":
        v = np.abs(z)
    elif t == "diff":
        v = np.diff(z, prepend=z[0])
    elif t == "sq":
        v = z ** 2
    elif t == "mmean":
        v = np.concatenate([np.full(w - 1, z[:w].mean()), G.moving(z, w, np.mean)])
    else:
        v = np.concatenate([np.full(w - 1, z[:w].std()), G.moving(z, w, np.std)])
    return v[:n_pre], v[n_pre:]


def grid_extra(G, t):
    #gini/theil (z, abs, diff, sq, mstd) and compression/lz (z, diff) columns of features_grid.one
    ex = {}

    def put(fn):
        return lambda f, k, pa, pb: f.__setitem__(k, fn(pa, pb))
    if t in ("z", "abs", "diff", "sq", "mstd"):
        ex["gini_pre"] = put(lambda pa, pb: G.gini(pa))
        ex["gini_post"] = put(lambda pa, pb: G.gini(pb))
        ex["gini_gap"] = put(lambda pa, pb: G.gini(pb) - G.gini(pa))
        ex["theil_pre"] = put(lambda pa, pb: G.theil(pa))
        ex["theil_post"] = put(lambda pa, pb: G.theil(pb))
        ex["theil_gap"] = put(lambda pa, pb: G.theil(pb) - G.theil(pa))
    if t in ("z", "diff"):
        for j, nm in enumerate(("gz", "bz")):
            ex[f"{nm}_pre"] = put(lambda pa, pb, j=j: G.compress_ratio(pa)[j])
            ex[f"{nm}_post"] = put(lambda pa, pb, j=j: G.compress_ratio(pb)[j])
            ex[f"{nm}_gap"] = put(lambda pa, pb, j=j: G.compress_ratio(pb)[j] - G.compress_ratio(pa)[j])

        def lz(x):
            return G.lz_complexity(x > np.median(x))
        ex["lz_pre"] = put(lambda pa, pb: lz(pa))
        ex["lz_post"] = put(lambda pa, pb: lz(pb))
        ex["lz_gap"] = put(lambda pa, pb: lz(pb) - lz(pa))
    return ex


def split_view(name, views):
    for t in sorted(views, key=len, reverse=True):
        if name.startswith(t + "_"):
            return t, name[len(t) + 1:]
    return None, None


def grid_row(G, ident, a, b, need):
    sa = max(a.std(), EPS)
    z = (np.concatenate([a, b]) - a.mean()) / sa
    n_pre = len(a)
    f = {"id": ident}
    plan, other = {}, []
    for c in need:
        t, rest = split_view(c, GRID_VIEWS)
        if t:
            plan.setdefault(t, []).append(rest)
        else:
            other.append(c)
    for t, rest in plan.items():
        pa, pb = grid_view(G, z, n_pre, t)
        try:
            view_columns(G, f, t, rest, pa, pb, grid_extra(G, t))
        except KeyError:
            other += [f"{t}_{r}" for r in rest]
    wins = [c for c in other if c in ("win_mean_gap", "win_std_ratio", "win_ks", "win_t")]
    if wins:
        wa, wb = z[int(n_pre * 0.9):n_pre], z[n_pre:n_pre + int(len(b) * 0.6)]
        if len(wa) > 5 and len(wb) > 5:
            v = {"win_mean_gap": lambda: wb.mean() - wa.mean(), "win_std_ratio": lambda: wb.std() / (wa.std() + EPS),
                 "win_ks": lambda: G.stats.ks_2samp(wa, wb).statistic,
                 "win_t": lambda: G.stats.ttest_ind(wa, wb, equal_var=False).statistic}
            for c in wins:
                f[c] = v[c]()
    rest = [c for c in other if c not in wins]
    if rest:
        full = G.one(ident, a, b)
        f.update({c: full[c] for c in rest if c in full})
    return f


def pad(v, n, fill):
    return np.concatenate([np.full(n - len(v), fill), v])


GRID2_VIEWS = ("mstd5", "mstd20", "mstd40", "mrng10", "ewstd", "absdiff", "diff2")


def grid2_views(G, x, tag, names):
    #the entries of features_grid2.views(x, tag) listed in names (same conditions, same expressions)
    n = len(x)
    out = {}
    for w in (5, 20, 40):
        if f"{tag}mstd{w}" in names and n > w + 5:
            out[f"{tag}mstd{w}"] = pad(G.moving(x, w, np.std), n, x[:w].std())
    if f"{tag}mrng10" in names and n > 15:
        out[f"{tag}mrng10"] = pad(G.moving(x, 10, np.max) - G.moving(x, 10, np.min), n, np.ptp(x[:10]))
    if f"{tag}ewstd" in names:
        out[f"{tag}ewstd"] = np.sqrt(pd.Series(x).ewm(span=10).var().bfill().to_numpy() + EPS)
    if f"{tag}absdiff" in names or f"{tag}diff2" in names:
        d = np.diff(x, prepend=x[0])
        out[f"{tag}absdiff"] = np.abs(d)
        out[f"{tag}diff2"] = np.diff(d, prepend=d[0])
    return out


def grid2_row(G, CF, ident, a, b, need):
    sa = max(a.std(), EPS)
    z = (np.concatenate([a, b]) - a.mean()) / sa
    n_pre = len(a)
    f = {"id": ident}
    plan, rplan, other = {}, {}, []
    for c in need:
        if c.startswith("r_"):
            t, rest = split_view(c[2:], GRID2_VIEWS)
            if t:
                rplan.setdefault("r_" + t, []).append(rest)
            else:
                other.append(c)
        else:
            t, rest = split_view(c, GRID2_VIEWS)
            if t:
                plan.setdefault(t, []).append(rest)
            else:
                other.append(c)
    for t, v in grid2_views(G, z, "", plan).items():
        if t in plan:
            try:
                view_columns(G, f, t, plan[t], v[:n_pre], v[n_pre:])
            except KeyError:
                other += [f"{t}_{r}" for r in plan[t]]
    if rplan:
        coefs, c0, ra = CF.ar_fit(z[:n_pre], 4)
        rb = CF.ar_apply(z[n_pre:], coefs, c0)
        r = np.concatenate([ra, rb])
        n_rpre = len(ra)
        ex = {"mean_ratio": lambda f, k, pa, pb: f.__setitem__(k, pb.mean() / (abs(pa.mean()) + 1e-6)),
              "std_ratio": lambda f, k, pa, pb: f.__setitem__(k, pb.std() / (pa.std() + 1e-6))}
        for t, v in grid2_views(G, r, "r_", rplan).items():
            if t not in rplan:
                continue
            rest = rplan[t]
            if any(x not in ex and x not in TESTS and x not in HISTS for x in rest):
                other += [f"{t}_{x}" for x in rest]          #residual views only have tests, distances and two ratios
                continue
            view_columns(G, f, t, rest, v[:n_rpre], v[n_rpre:], ex)
    if other:
        import features_grid2 as G2
        full = G2.one(ident, a, b)
        f.update({c: full[c] for c in other if c in full})
    return f


CP_NAMES = ("cz", "cz2", "cabs", "cd", "cd2", "car1", "car1sq", "car4", "car4sq", "cprod")
CP_PARTS = ("at_b", "max", "ratio", "rank", "argmax_off", "local_peak")
WIN = ("win_ks_min", "win_ks_mean", "win_ks_last", "win_mz_min_abs", "win_mz_mean", "win_mz_last", "win_mz_consistency",
       "win_lv_min_abs", "win_lv_mean", "win_lv_last", "win_lv_consistency")


def cp_known(k):
    if k in WIN:
        return True
    nm, part = split_view(k, CP_NAMES)
    return nm is not None and part in CP_PARTS


def cp_part(CF, za, zb, keys):
    #the CUSUM-curve and after-window entries of CF.change_point(za, zb) named in keys (same expressions); raises
    #where the full function raises (no after windows), so the caller drops the whole family like the original
    f = {}
    z = np.concatenate([za, zb])
    n, b = len(z), len(za)
    want = {}
    for k in keys:
        if k not in WIN:
            nm, part = split_view(k, CP_NAMES)
            want.setdefault(nm, set()).add(part)
    r1 = CF.ar_fit(z, 1)[2] if {"car1", "car1sq"} & set(want) else None
    r4 = CF.ar_fit(z, 4)[2] if {"car4", "car4sq"} & set(want) else None
    dsa = max(np.diff(za).std(), EPS)
    for name in CP_NAMES:
        if name not in want:
            continue
        s = {"cz": lambda: z, "cz2": lambda: z ** 2, "cabs": lambda: np.abs(z), "cd": lambda: np.diff(z) / dsa,
             "cd2": lambda: (np.diff(z) / dsa) ** 2, "car1": lambda: r1, "car1sq": lambda: r1 ** 2, "car4": lambda: r4,
             "car4sq": lambda: r4 ** 2, "cprod": lambda: z[:-1] * z[1:]}[name]()
        m = len(s)
        bnd = b - (n - m)
        if bnd < 8 or bnd > m - 8:
            continue
        curve = CF.cusum_curve(s)
        at_b, positions = float(curve[bnd - 1]), np.arange(1, m)
        mx = float(curve.max()) if len(curve) else at_b
        for p in want[name]:
            if p == "at_b":
                f[f"{name}_at_b"] = at_b
            elif p == "max":
                f[f"{name}_max"] = mx
            elif p == "ratio":
                f[f"{name}_ratio"] = at_b / max(mx, EPS)
            elif p == "rank":
                f[f"{name}_rank"] = float((curve > at_b).mean()) if len(curve) else 0.0
            elif p == "argmax_off":
                f[f"{name}_argmax_off"] = (positions[int(curve.argmax())] - bnd) / m if len(curve) else 0.0
            else:
                w = max(int(0.05 * m), 4)
                lo, hi = max(bnd - 1 - w, 0), min(bnd - 1 + w, m - 2)
                f[f"{name}_local_peak"] = float(curve[bnd - 1] / max(curve[lo:hi + 1].max(), EPS))
    wl = max(min(64, len(zb) // 2), 16)
    starts = np.arange(0, len(zb) - wl + 1, max(wl // 2, 1))
    if len(starts) == 0:
        raise ValueError("no after windows")          #the full function fails here too (min of an empty array)
    wk = [k for k in keys if k in WIN]
    if any(k.startswith("win_ks") for k in wk):
        ks_w = np.array([CF.stats.ks_2samp(za, zb[s:s + wl], method="asymp").statistic for s in starts])
        f.update({"win_ks_min": ks_w.min(), "win_ks_mean": ks_w.mean(), "win_ks_last": ks_w[-1]})
    if any(k.startswith("win_mz") for k in wk):
        mz_w = np.array([zb[s:s + wl].mean() for s in starts])
        f.update({"win_mz_min_abs": np.abs(mz_w).min(), "win_mz_mean": mz_w.mean(), "win_mz_last": mz_w[-1],
                  "win_mz_consistency": abs(mz_w.mean()) / max(np.abs(mz_w).mean(), EPS)})
    if any(k.startswith("win_lv") for k in wk):
        lv_w = np.array([np.log(max(zb[s:s + wl].var(), EPS)) for s in starts])
        f.update({"win_lv_min_abs": np.abs(lv_w).min(), "win_lv_mean": lv_w.mean(), "win_lv_last": lv_w[-1],
                  "win_lv_consistency": abs(lv_w.mean()) / max(np.abs(lv_w).mean(), EPS)})
    return {k: f[k] for k in keys if k in f}


def dd_row(CF, ident, a, b, need):
    #features_dd.one(): change_point CUSUM / window pieces and dd_trend_noise_post by hand, everything else from one()
    f = {"id": ident}
    cp = [c[3:] for c in need if c.startswith("dd_") and cp_known(c[3:])]
    other = [c for c in need if c != "dd_trend_noise_post" and not (c.startswith("dd_") and cp_known(c[3:]))]
    if other:
        import features_dd as DD
        full = DD.one(ident, a, b)
        f.update({c: full[c] for c in other if c in full})
    if cp:
        da, db = np.diff(a), np.diff(b)
        m, s = da.mean(), max(da.std(), EPS)
        zda, zdb = (da - m) / s, (db - m) / s
        try:
            f.update({"dd_" + k: v for k, v in cp_part(CF, zda, zdb, cp).items()})
        except Exception:
            pass
    if "dd_trend_noise_post" in need:
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


def base_any(CF, ident, a, b, need):
    #hand-written base columns when they cover the request, else the original series_features() row; none -> id only
    if not need:
        return {"id": ident}
    if set(need) <= set(BASE_HAND):
        f = base_row(CF, ident, a, b)
    else:
        f = CF.series_features(ident, a, b)
    return {"id": ident, **{c: f[c] for c in need if c in f}}


def _rows(args):
    table, work, chunk, need = args
    CF, G = load(work)
    fast_stats(CF, G)
    fn = {"base": lambda j: base_any(CF, *j, need), "grid": lambda j: grid_row(G, *j, need),
          "grid2": lambda j: grid2_row(G, CF, *j, need), "dd": lambda j: dd_row(CF, *j, need)}[table]
    return [fn(j) for j in chunk]


def supported(table):
    #lrz6 has only the one hand-written column; every other table falls back to the original code per column
    return table != "lrz6" or set(cols(table)) <= set(LRZ6_HAND)


def finish(rows_or_frame, table):
    F = rows_or_frame if isinstance(rows_or_frame, pd.DataFrame) else pd.DataFrame(rows_or_frame).set_index("id")
    F = F.reindex(columns=cols(table))
    F = F.replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    F.index.name = "id"
    return F


def build(table, work, workers=1, ids=None):
    work = str(Path(work).resolve())
    load(work)
    if table == "lrz6":
        if not supported(table):
            raise SystemExit("lrz6: the model needs columns the slim path does not build; run the original step")
        return finish(lrz6_table(work), table)
    need = cols(table)
    jobs = series_jobs(work)
    if ids is not None:
        keep = set(ids); jobs = [j for j in jobs if j[0] in keep]
    if not need:
        #no column of this table is read (e2 base): the ids only, nothing computed
        return finish(pd.DataFrame(index=pd.Index([j[0] for j in jobs], name="id")), table)
    if workers > 1:
        from concurrent.futures import ProcessPoolExecutor
        size = max(len(jobs) // (workers * 8), 1)
        chunks = [jobs[i:i + size] for i in range(0, len(jobs), size)]
        with ProcessPoolExecutor(workers) as pool:
            rows = [r for part in pool.map(_rows, [(table, work, c, need) for c in chunks]) for r in part]
    else:
        rows = _rows((table, work, jobs, need))
    if not rows:
        return finish(pd.DataFrame(index=pd.Index([], name="id")), table)
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
