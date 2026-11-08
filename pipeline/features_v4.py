"""KU-SBB, clean version: one feature table per series, computed in a plain loop.

Every series is split at the given boundary into `a` (before, period 0) and `b` (after,
period 1).  Values are standardised by the mean and std of `a`, so every feature is unit-free.
Each family below is a function returning a dict of named numbers for one series.  The
families, in the order they were found to matter:

  basic_gaps          level / spread / distribution differences between before and after
  autocorrelation     how successive readings relate, at several lags, before vs after
  ar_models           AR(p) fits on `a` applied to `b`: do the old dynamics still explain the new data?
  trend_stability     slopes, boundary jump, is the change still there at the end of the series?
  change_point        is the boundary the best place to split the series, versus any other split?
  innovation_shape    shape of the noise after whitening (kurtosis, tails, two-sample shape tests)
  volatility_shape    the same after dividing by local volatility, plus the volatility process itself
  tail_index          Student-t degrees of freedom by likelihood, and likelihood-ratio statistics
  joint_law           energy distance / MMD between clouds of consecutive readings, SAX pattern histograms

Run:  python features.py          (writes features_train.parquet and features_test.parquet here)
Single-threaded on purpose; expect a few hours for the 28,000 series.
"""
import pathlib
import time
import warnings

import numpy as np
import pandas as pd
from scipy import signal, special, stats

warnings.filterwarnings("ignore")
HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE.parent / "data"
EPS = 1e-12


# ============================================================================ small helpers
def acf(x, lags):
    """Autocorrelation of x at each lag (0 when the lag is too long for the series)."""
    x = x - x.mean()
    denom = float((x * x).sum())
    if denom <= 0:
        return np.zeros(len(lags))
    return np.array([float((x[:-k] * x[k:]).sum() / denom) if k < len(x) else 0.0 for k in lags])


def acf1(x):
    return acf(x, [1])[0]


def pacf_from_acf(r, nlags):
    """Partial autocorrelations from r[0..nlags] (Durbin-Levinson recursion)."""
    pacf = np.zeros(nlags)
    phi = np.zeros((nlags + 1, nlags + 1))
    for k in range(1, nlags + 1):
        num = r[k] - sum(phi[k - 1, j] * r[k - j] for j in range(1, k))
        den = 1.0 - sum(phi[k - 1, j] * r[j] for j in range(1, k))
        phi[k, k] = num / den if abs(den) > EPS else 0.0
        for j in range(1, k):
            phi[k, j] = phi[k - 1, j] - phi[k, k] * phi[k - 1, k - j]
        pacf[k - 1] = phi[k, k]
    return pacf


def ar_fit(x, p):
    """Least-squares AR(p) with intercept: returns (coefficients, intercept, residuals)."""
    n = len(x)
    if n <= p + 2:
        return np.zeros(p), float(x.mean()), x - x.mean()
    X = np.column_stack([x[p - k - 1:n - k - 1] for k in range(p)] + [np.ones(n - p)])
    beta = np.linalg.lstsq(X, x[p:], rcond=None)[0]
    return beta[:p], float(beta[p]), x[p:] - X @ beta


def ar_apply(x, coefs, intercept):
    """Residuals of x under an AR model fitted elsewhere."""
    p = len(coefs)
    n = len(x)
    if n <= p:
        return x - x.mean()
    X = np.column_stack([x[p - k - 1:n - k - 1] for k in range(p)])
    return x[p:] - (X @ coefs + intercept)


def log_ratio(num, den):
    return float(np.log(max(num, EPS) / max(den, EPS)))


def cusum_curve(z):
    """Standardised mean-shift statistic for every split point of z (a change-point curve)."""
    n = len(z)
    cs = np.cumsum(z - z.mean())[:-1]
    k = np.arange(1, n)
    return np.abs(cs) / np.sqrt(k * (n - k) / n) / max(z.std(), EPS)


# ============================================================================ feature families
def basic_gaps(za, zb):
    """Level, spread and distribution differences between before (za) and after (zb)."""
    f = {}
    ma, mb, sa, sb = za.mean(), zb.mean(), za.std(), zb.std()
    pooled = max(float(np.sqrt((za.var() + zb.var()) / 2)), EPS)
    qa, qb = np.quantile(za, [.1, .25, .5, .75, .9]), np.quantile(zb, [.1, .25, .5, .75, .9])
    f["n_pre"], f["n_post"] = len(za), len(zb)
    f["log_n"], f["boundary_frac"] = np.log(len(za) + len(zb)), len(za) / (len(za) + len(zb))
    # the six features of the official baseline
    f["mean_gap"] = abs(mb - ma) / pooled
    f["median_gap"] = abs(qb[2] - qa[2]) / pooled
    f["ks"] = stats.ks_2samp(za, zb, method="asymp").statistic
    f["wasserstein"] = stats.wasserstein_distance(za, zb) / pooled
    f["gap_spread"] = abs(sb - sa) / pooled
    f["gap_iqr"] = abs((qb[3] - qb[1]) - (qa[3] - qa[1])) / pooled
    # signed and scale-aware versions, in units of the before-segment std (which is 1 after standardising)
    f["mean_gap_signed_pre"], f["median_gap_signed_pre"] = mb - ma, qb[2] - qa[2]
    f["log_std_ratio"] = log_ratio(sb, sa)
    f["log_iqr_ratio"] = log_ratio(qb[3] - qb[1], qa[3] - qa[1])
    f["q10_gap"], f["q90_gap"] = qb[0] - qa[0], qb[4] - qa[4]
    f["mad_ratio"] = log_ratio(np.median(np.abs(zb - qb[2])), np.median(np.abs(za - qa[2])))
    f["skew_pre"], f["skew_post"] = stats.skew(za), stats.skew(zb)
    f["skew_gap"] = f["skew_post"] - f["skew_pre"]
    f["kurt_pre"], f["kurt_post"] = stats.kurtosis(za), stats.kurtosis(zb)
    f["kurt_gap"] = f["kurt_post"] - f["kurt_pre"]
    f["welch_t"] = stats.ttest_ind(za, zb, equal_var=False).statistic
    f["mwu"] = stats.mannwhitneyu(za, zb).statistic / (len(za) * len(zb)) - 0.5
    f["levene"] = stats.levene(za, zb, center="median").statistic
    f["post_z_mean"], f["post_z_absmean"] = zb.mean(), np.abs(zb).mean()
    f["post_z_frac_out2"] = (np.abs(zb) > 2).mean() - (np.abs(za) > 2).mean()
    return f


def autocorrelation(za, zb):
    """How successive readings relate: ACF at several lags, PACF, ACF of differences, variance ratios."""
    f = {}
    lags = [1, 2, 3, 4, 5, 8, 10, 15, 20, 30]
    ra, rb = acf(za, lags), acf(zb, lags)
    for k, va, vb in zip(lags, ra, rb):
        f[f"acf{k}_pre"], f[f"acf{k}_post"], f[f"acf{k}_gap"] = va, vb, vb - va
    f["acf_gap_l2"] = float(np.sqrt(((rb - ra) ** 2).sum()))
    pa = pacf_from_acf(np.concatenate([[1.0], acf(za, [1, 2, 3, 4, 5])]), 5)
    pb = pacf_from_acf(np.concatenate([[1.0], acf(zb, [1, 2, 3, 4, 5])]), 5)
    for k in range(5):
        f[f"pacf{k + 1}_pre"], f[f"pacf{k + 1}_post"], f[f"pacf{k + 1}_gap"] = pa[k], pb[k], pb[k] - pa[k]
    da, db = np.diff(za), np.diff(zb)
    dra, drb = acf(da, [1, 2, 3]), acf(db, [1, 2, 3])
    for k in range(3):
        f[f"dacf{k + 1}_pre"], f[f"dacf{k + 1}_post"], f[f"dacf{k + 1}_gap"] = dra[k], drb[k], drb[k] - dra[k]
    dsa, dsb = max(da.std(), EPS), max(db.std(), EPS)
    f["log_diff_std_ratio"] = np.log(dsb / dsa)
    f["diff_mean_gap"] = (db.mean() - da.mean()) / dsa
    f["std_over_diffstd_pre"], f["std_over_diffstd_post"] = za.std() / dsa, zb.std() / dsb
    f["std_over_diffstd_gap"] = f["std_over_diffstd_post"] - f["std_over_diffstd_pre"]
    for k in (4, 16):   # variance ratio: 1 for white noise, >1 for persistent (random-walk-like) series
        def vr(x):
            d1, dk = np.diff(x), x[k:] - x[:-k]
            return float(dk.var() / (k * d1.var())) if d1.var() > EPS and len(dk) > 2 else 1.0
        f[f"vr{k}_pre"], f[f"vr{k}_post"] = vr(za), vr(zb)
        f[f"vr{k}_gap"] = f[f"vr{k}_post"] - f[f"vr{k}_pre"]
    return f


def ar_models(za, zb):
    """AR(p) fitted on the before part and applied to the after part, plus a Chow-style test."""
    f = {}
    n = len(za) + len(zb)
    for p in (1, 2, 4, 8):
        ca, ia, resa = ar_fit(za, p)
        cb, ib, resb = ar_fit(zb, p)
        sra, srb = max(resa.std(), EPS), max(resb.std(), EPS)
        f[f"ar{p}_persist_pre"], f[f"ar{p}_persist_post"] = ca.sum(), cb.sum()
        f[f"ar{p}_persist_gap"] = cb.sum() - ca.sum()
        f[f"ar{p}_coef_l2_gap"] = float(np.sqrt(((cb - ca) ** 2).sum()))
        f[f"ar{p}_log_resid_ratio"] = np.log(srb / sra)
        f[f"ar{p}_lrmean_gap"] = float(np.clip(ib / max(1 - cb.sum(), 0.05) - ia / max(1 - ca.sum(), 0.05), -50, 50))
        r = ar_apply(zb, ca, ia)                       # after-part residuals under the before-part model
        f[f"ar{p}_xres_mean"] = r.mean() / sra
        f[f"ar{p}_xres_logstd"] = np.log(max(r.std(), EPS) / sra)
        f[f"ar{p}_xres_cusum"] = float(np.abs(np.cumsum(r)).max() / (sra * np.sqrt(len(r))))
        f[f"ar{p}_xres_acf1"] = acf1(r)
        _, _, resp = ar_fit(np.concatenate([za, zb]), p)   # one model for the whole series vs two separate ones
        rss_p, rss_s = float((resp ** 2).sum()), float((resa ** 2).sum() + (resb ** 2).sum())
        f[f"ar{p}_chow"] = float(np.log(max(rss_p, EPS) / max(rss_s, EPS)) * n)
        if p <= 2:
            for j in range(p):
                f[f"ar{p}_c{j + 1}_pre"], f[f"ar{p}_c{j + 1}_post"] = ca[j], cb[j]
    return f


def trend_stability(za, zb):
    """Slopes and jumps at the boundary, and whether the after part stays changed to the end."""
    f = {}
    na, nb = len(za), len(zb)
    ta, tb = np.arange(na), np.arange(na, na + nb)
    sl_a, ic_a = np.polyfit(ta, za, 1)
    sl_b, ic_b = np.polyfit(tb, zb, 1)
    f["slope_pre"], f["slope_post"] = sl_a * na, sl_b * nb
    f["slope_gap"] = f["slope_post"] - f["slope_pre"]
    extrap = sl_a * tb + ic_a
    f["detrend_gap"] = (zb - extrap).mean()
    f["detrend_std_ratio"] = log_ratio((zb - extrap).std(), (za - (sl_a * ta + ic_a)).std())
    f["jump_at_boundary"] = zb[:5].mean() - za[-5:].mean()
    f["jump_at_boundary_vs_diff"] = (zb[0] - za[-1]) / max(np.diff(za).std(), EPS)
    # quarters of the after part relative to the before part: a lasting change is present in every quarter
    qmb = np.array([q.mean() for q in np.array_split(zb, 4)])
    qsb = np.array([q.std() for q in np.array_split(zb, 4)])
    qma = np.array([q.mean() for q in np.array_split(za, 4)])
    qsa = np.array([q.std() for q in np.array_split(za, 4)])
    f["q_z_min_abs"], f["q_z_max_abs"] = np.abs(qmb).min(), np.abs(qmb).max()
    f["q_z_first"], f["q_z_last"] = qmb[0], qmb[-1]
    f["q_z_same_sign"] = float(np.all(np.sign(qmb) == np.sign(qmb[0])))
    f["q_mean_wander_post"], f["q_mean_wander_pre"] = qmb.std(), qma.std()
    f["q_mean_wander_ratio"] = log_ratio(qmb.std(), qma.std())
    lq = np.log(np.maximum(qsb, EPS))
    f["q_logstd_min"], f["q_logstd_max"], f["q_logstd_last"] = lq.min(), lq.max(), lq[-1]
    f["q_logstd_wander_ratio"] = log_ratio(np.log(np.maximum(qsb, EPS)).std(), np.log(np.maximum(qsa, EPS)).std())
    h1, h2 = zb[: nb // 2], zb[nb // 2:]
    f["post_half_mean_diff"] = h2.mean() - h1.mean()
    f["post_half_logstd_diff"] = log_ratio(h2.std(), h1.std())
    f["post_last_quarter_gap"], f["post_last_quarter_logstd"] = qmb[-1], lq[-1]
    # spectral energy in four bands and spectral entropy, before vs after
    def band_power(x):
        fr, p = signal.welch(x - x.mean(), nperseg=min(64, len(x)))
        if p.sum() <= EPS:
            return np.zeros(4), 0.0
        p = p / p.sum()
        edges = [0, 0.03, 0.1, 0.25, 0.5 + 1e-9]
        bands = np.array([p[(fr >= edges[i]) & (fr < edges[i + 1])].sum() for i in range(4)])
        return bands, float(-(p[p > 0] * np.log(p[p > 0])).sum() / np.log(len(p)))
    bpa, enta = band_power(za)
    bpb, entb = band_power(zb)
    for i in range(4):
        f[f"band{i}_pre"], f[f"band{i}_post"], f[f"band{i}_gap"] = bpa[i], bpb[i], bpb[i] - bpa[i]
    f["spec_ent_pre"], f["spec_ent_post"], f["spec_ent_gap"] = enta, entb, entb - enta
    return f


def change_point(za, zb):
    """Is the given boundary the best place to split the series?  For several statistics we record
    the value at the boundary, the best value over other candidate splits, their ratio, the rank of
    the boundary, and the distance of the best split from the boundary."""
    f = {}
    z = np.concatenate([za, zb])
    n, b = len(z), len(za)
    r1 = ar_fit(z, 1)[2]
    r4 = ar_fit(z, 4)[2]
    dsa = max(np.diff(za).std(), EPS)

    def summarise(name, at_b, curve, positions, boundary, length):
        mx = float(curve.max()) if len(curve) else at_b
        f[f"{name}_at_b"], f[f"{name}_max"] = at_b, mx
        f[f"{name}_ratio"] = at_b / max(mx, EPS)
        f[f"{name}_rank"] = float((curve > at_b).mean()) if len(curve) else 0.0
        f[f"{name}_argmax_off"] = (positions[int(curve.argmax())] - boundary) / length if len(curve) else 0.0

    # CUSUM curves on several transforms of the series (each curve has one value per split point)
    transforms = {"cz": z, "cz2": z ** 2, "cabs": np.abs(z), "cd": np.diff(z) / dsa, "cd2": (np.diff(z) / dsa) ** 2,
                  "car1": r1, "car1sq": r1 ** 2, "car4": r4, "car4sq": r4 ** 2, "cprod": z[:-1] * z[1:]}
    for name, s in transforms.items():
        m = len(s)
        bnd = b - (n - m)                   # shorter transforms are aligned to the end of the series
        if bnd < 8 or bnd > m - 8:
            continue
        curve = cusum_curve(s)
        summarise(name, float(curve[bnd - 1]), curve, np.arange(1, m), bnd, m)
        w = max(int(0.05 * m), 4)
        lo, hi = max(bnd - 1 - w, 0), min(bnd - 1 + w, m - 2)
        f[f"{name}_local_peak"] = float(curve[bnd - 1] / max(curve[lo:hi + 1].max(), EPS))

    # distribution tests at the boundary and on a grid of 40 other splits
    grid = np.unique(np.linspace(int(0.15 * n), int(0.85 * n), 40).astype(int))
    grid = grid[(grid >= 16) & (grid <= n - 16)]
    tests = {"ks": lambda l, r: stats.ks_2samp(l, r, method="asymp").statistic,
             "mwu": lambda l, r: abs(stats.mannwhitneyu(l, r).statistic / (len(l) * len(r)) - 0.5),
             "lvr": lambda l, r: abs(log_ratio(r.var(), l.var()))}
    for name, fn in tests.items():
        summarise("g" + name, float(fn(z[:b], z[b:])), np.array([fn(z[:g], z[g:]) for g in grid]), grid, b, n)
    dz = np.diff(z)
    for name, fn in (("dks", tests["ks"]), ("dlvr", tests["lvr"])):
        summarise("g" + name, float(fn(dz[:b - 1], dz[b - 1:])), np.array([fn(dz[:g - 1], dz[g - 1:]) for g in grid]), grid - 1, b - 1, n)
    for p in (1, 4):   # Chow statistic at every grid split
        rss_pool = float((ar_fit(z, p)[2] ** 2).sum())
        chow = lambda l, r: np.log(max(rss_pool, EPS) / max(float((ar_fit(l, p)[2] ** 2).sum() + (ar_fit(r, p)[2] ** 2).sum()), EPS)) * n
        summarise(f"gchow{p}", float(chow(z[:b], z[b:])), np.array([chow(z[:g], z[g:]) for g in grid]), grid, b, n)

    # sliding windows after the boundary, each compared with the whole before part
    wl = max(min(64, len(zb) // 2), 16)
    starts = np.arange(0, len(zb) - wl + 1, max(wl // 2, 1))
    ks_w = np.array([tests["ks"](za, zb[s:s + wl]) for s in starts])
    mz_w = np.array([zb[s:s + wl].mean() for s in starts])
    lv_w = np.array([np.log(max(zb[s:s + wl].var(), EPS)) for s in starts])
    starts_a = np.arange(0, len(za) - wl + 1, max(wl // 2, 1))
    ks_a = np.array([tests["ks"](np.delete(za, np.s_[s:s + wl]), za[s:s + wl]) for s in starts_a]) if len(starts_a) else np.array([0.0])
    f["win_ks_min"], f["win_ks_mean"], f["win_ks_last"] = ks_w.min(), ks_w.mean(), ks_w[-1]
    f["win_ks_pre_mean"], f["win_ks_excess"] = ks_a.mean(), ks_w.mean() - ks_a.mean()
    f["win_ks_frac_above_pre"] = float((ks_w > np.quantile(ks_a, 0.9)).mean())
    f["win_mz_min_abs"], f["win_mz_mean"], f["win_mz_last"] = np.abs(mz_w).min(), mz_w.mean(), mz_w[-1]
    f["win_mz_consistency"] = abs(mz_w.mean()) / max(np.abs(mz_w).mean(), EPS)
    f["win_lv_min_abs"], f["win_lv_mean"], f["win_lv_last"] = np.abs(lv_w).min(), lv_w.mean(), lv_w[-1]
    f["win_lv_consistency"] = abs(lv_w.mean()) / max(np.abs(lv_w).mean(), EPS)
    return f


def shape_stats(r):
    """Shape of a residual sample after removing its own location and scale."""
    r = r - r.mean()
    z = r / max(r.std(), EPS)
    q = np.quantile(np.abs(z), [.9, .95, .99])
    f = {"kurt": stats.kurtosis(z), "skew": stats.skew(z), "mad_std": float(np.median(np.abs(z)) / 0.6745),
         "iqr_std": float(np.subtract(*np.quantile(z, [.75, .25])) / 1.349), "q90": q[0], "q95": q[1], "q99": q[2],
         "ex2": float((np.abs(z) > 2).mean()), "ex3": float((np.abs(z) > 3).mean()), "ex4": float((np.abs(z) > 4).mean()),
         "jb": float(len(z) / 6 * (stats.skew(z) ** 2 + stats.kurtosis(z) ** 2 / 4)), "ent": float(stats.differential_entropy(z)),
         "sq_kurt": stats.kurtosis(z ** 2), "sq_skew": stats.skew(z ** 2), "absmean": float(np.abs(z).mean())}
    a = np.sort(np.abs(z))[::-1]
    k = max(int(len(a) * 0.1), 5)
    f["hill"] = float(np.mean(np.log(a[:k] / a[k]))) if a[k] > EPS else 0.0     # Hill tail-index estimate
    f["bm8_kurt"] = stats.kurtosis(z[: len(z) // 8 * 8].reshape(-1, 8).mean(1)) if len(z) >= 32 else 0.0
    return f


def two_sample_shape(ra, rb):
    """Two-sample tests between two residual samples after each is standardised (shape only)."""
    za = (ra - ra.mean()) / max(ra.std(), EPS)
    zb = (rb - rb.mean()) / max(rb.std(), EPS)
    g = {"ks": stats.ks_2samp(za, zb).statistic, "ad": stats.anderson_ksamp([za, zb]).statistic,
         "cvm": stats.cramervonmises_2samp(za, zb).statistic, "energy": stats.energy_distance(za, zb),
         "wass": stats.wasserstein_distance(za, zb)}
    edges = np.linspace(-4, 4, 17)
    ha = np.histogram(np.clip(za, -4, 4), edges)[0] / len(za)
    hb = np.histogram(np.clip(zb, -4, 4), edges)[0] / len(zb)
    g["hist_l1"] = float(np.abs(ha - hb).sum())
    g["tail_l1"] = float(np.abs(ha[[0, 1, 14, 15]] - hb[[0, 1, 14, 15]]).sum())
    return g


def innovation_shape(za, zb):
    """The signal class that classic statistics miss: the SHAPE of the noise after whitening.
    Residuals under AR(1/4/8) fitted on the before part, first differences, AR(4) fitted on each part
    separately, and the raw values; shape statistics and two-sample shape tests on each."""
    f = {}
    whitenings = {}
    for p in (1, 4, 8):
        c, i, _ = ar_fit(za, p)
        whitenings[f"ar{p}"] = (ar_apply(za, c, i), ar_apply(zb, c, i))
    whitenings["d1"] = (np.diff(za), np.diff(zb))
    ca, ia, _ = ar_fit(za, 4)
    cb, ib, _ = ar_fit(zb, 4)
    whitenings["own4"] = (ar_apply(za, ca, ia), ar_apply(zb, cb, ib))
    whitenings["raw"] = (za, zb)
    for w, (ra, rb) in whitenings.items():
        sa, sb = shape_stats(ra), shape_stats(rb)
        for k in sa:
            f[f"{w}_{k}_pre"], f[f"{w}_{k}_post"], f[f"{w}_{k}_gap"] = sa[k], sb[k], sb[k] - sa[k]
        for k, v in two_sample_shape(ra, rb).items():
            f[f"{w}_2s_{k}"] = v
        f[f"{w}_logstd_ratio"] = log_ratio(rb.std(), ra.std())
    # nonlinear dependence, time-irreversibility and residual checks under the before-part AR(4)
    ra, rb = whitenings["ar4"]
    corr = lambda u, v: float(np.corrcoef(u, v)[0, 1]) if u.std() > 0 and v.std() > 0 else 0.0
    f["nl_x_x2"] = corr(zb[1:], zb[:-1] ** 2) - corr(za[1:], za[:-1] ** 2)
    f["nl_x2_x"] = corr(zb[1:] ** 2, zb[:-1]) - corr(za[1:] ** 2, za[:-1])
    f["nl_x2_x2"] = corr(zb[1:] ** 2, zb[:-1] ** 2) - corr(za[1:] ** 2, za[:-1] ** 2)
    f["nl_abs"] = corr(np.abs(zb[1:]), np.abs(zb[:-1])) - corr(np.abs(za[1:]), np.abs(za[:-1]))
    f["nl_x_x3"] = corr(zb[1:], zb[:-1] ** 3) - corr(za[1:], za[:-1] ** 3)

    def mutual_info_lag1(x, bins=8):
        d = np.digitize(x, np.quantile(x, np.linspace(0, 1, bins + 1)[1:-1]))
        h = np.histogram2d(d[:-1], d[1:], bins=bins)[0] + 1e-9
        p = h / h.sum()
        return float((p * np.log(p / (p.sum(1, keepdims=True) * p.sum(0, keepdims=True)))).sum())
    f["mi_lag1_pre"], f["mi_lag1_post"] = mutual_info_lag1(za), mutual_info_lag1(zb)
    f["mi_lag1_gap"] = f["mi_lag1_post"] - f["mi_lag1_pre"]
    f["nl_resid_x2"] = corr(rb[1:], rb[:-1] ** 2) - corr(ra[1:], ra[:-1] ** 2)
    f["nl_resid_abs"] = corr(np.abs(rb[1:]), np.abs(rb[:-1])) - corr(np.abs(ra[1:]), np.abs(ra[:-1]))
    f["nl_resid_sq_ac1"] = acf(rb ** 2, [1])[0] - acf(ra ** 2, [1])[0]
    f["nl_resid_sq_ac2"] = acf(rb ** 2, [2])[0] - acf(ra ** 2, [2])[0]
    da, db = np.diff(za), np.diff(zb)
    f["irrev_d3"] = float((db ** 3).mean() / max(db.std(), EPS) ** 3 - (da ** 3).mean() / max(da.std(), EPS) ** 3)
    f["irrev_asym"] = float((zb[1:] ** 2 * zb[:-1]).mean() - (zb[1:] * zb[:-1] ** 2).mean()) - float((za[1:] ** 2 * za[:-1]).mean() - (za[1:] * za[:-1] ** 2).mean())
    f["irrev_updown"] = float((db > 0).mean() - (da > 0).mean())
    sb_, sa_ = np.flatnonzero(np.diff(np.sign(db)) != 0), np.flatnonzero(np.diff(np.sign(da)) != 0)
    f["irrev_runlen"] = float(np.mean(np.diff(sb_)) - np.mean(np.diff(sa_))) if len(sb_) > 2 and len(sa_) > 2 else 0.0
    f["res_ks_shape"] = stats.ks_2samp(ra / max(ra.std(), EPS), rb / max(rb.std(), EPS)).statistic
    f["res_ks_raw"] = stats.ks_2samp(ra, rb).statistic
    f["res_q99_gap"] = np.quantile(np.abs(rb) / max(rb.std(), EPS), .99) - np.quantile(np.abs(ra) / max(ra.std(), EPS), .99)
    f["res_mean"] = rb.mean() / max(ra.std(), EPS)
    f["res_entropy_gap"] = stats.differential_entropy(rb / max(rb.std(), EPS)) - stats.differential_entropy(ra / max(ra.std(), EPS))
    f["res_zc_gap"] = float((np.diff(np.sign(rb)) != 0).mean() - (np.diff(np.sign(ra)) != 0).mean())
    f["res_acf1_post"], f["res_acf2_post"] = acf(rb, [1])[0], acf(rb, [2])[0]   # white if the old model still holds
    f["res_lbq_post"] = float(len(rb) * sum(acf(rb, [k])[0] ** 2 for k in range(1, 6)))
    for k in (40, 60, 80, 100):
        f[f"acf{k}_gap"] = acf(zb, [k])[0] - acf(za, [k])[0]
    fa, pa = signal.periodogram(za)
    fb, pb = signal.periodogram(zb)
    f["peak_freq_gap"] = float(fb[1:][pb[1:].argmax()] - fa[1:][pa[1:].argmax()])
    f["peak_share_gap"] = float(pb[1:].max() / pb[1:].sum() - pa[1:].max() / pa[1:].sum())
    f["third_order_gap"] = float((zb[2:] * zb[1:-1] * zb[:-2]).mean() - (za[2:] * za[1:-1] * za[:-2]).mean())
    f["smooth_gap"] = log_ratio(np.diff(zb, 2).var(), db.var()) - log_ratio(np.diff(za, 2).var(), da.var())
    return f


def volatility_shape(za, zb):
    """Shape of the residuals after dividing by a local volatility estimate (separates 'the noise got
    heavier-tailed' from 'the variance started clustering'), plus the volatility process itself."""
    f = {}
    c, i, ra = ar_fit(za, 4)
    rb = ar_apply(zb, c, i)
    for how, w in (("roll", 16), ("roll", 48), ("ewm", 16)):
        def vol_std(r):
            r2 = pd.Series(r ** 2)
            v = r2.rolling(w, min_periods=4).mean().bfill().to_numpy() if how == "roll" else r2.ewm(alpha=1 - 2 ** (-1 / w)).mean().to_numpy()
            return r / np.sqrt(np.maximum(v, EPS))
        va, vb = vol_std(ra), vol_std(rb)
        tag = f"v{how}{w}"
        sa, sb = shape_stats(va), shape_stats(vb)
        for k in ("kurt", "skew", "q99", "ex3", "mad_std", "ent"):
            f[f"{tag}_{k}_pre"], f[f"{tag}_{k}_post"] = sa[k], sb[k]
            f[f"{tag}_{k}_gap"], f[f"{tag}_{k}_agap"] = sb[k] - sa[k], abs(sb[k] - sa[k])
        for k, v in two_sample_shape(va, vb).items():
            if k in ("ks", "cvm", "energy"):
                f[f"{tag}_2s_{k}"] = v
    for k in (1, 2, 5, 10):
        f[f"sqac{k}_pre"], f[f"sqac{k}_post"] = acf(ra ** 2, [k])[0], acf(rb ** 2, [k])[0]
        f[f"sqac{k}_gap"] = f[f"sqac{k}_post"] - f[f"sqac{k}_pre"]
        f[f"absac{k}_gap"] = acf(np.abs(rb), [k])[0] - acf(np.abs(ra), [k])[0]

    def garch_persist(r):
        r2 = r ** 2
        if len(r2) < 12:
            return 0.0, 0.0
        beta = np.linalg.lstsq(np.column_stack([r2[:-1], np.ones(len(r2) - 1)]), r2[1:], rcond=None)[0]
        return float(beta[0]), log_ratio(r2.std(), r2.mean())
    (pa, vva), (pb, vvb) = garch_persist(ra), garch_persist(rb)
    f["garch_persist_pre"], f["garch_persist_post"], f["garch_persist_gap"] = pa, pb, pb - pa
    f["volofvol_pre"], f["volofvol_post"], f["volofvol_gap"] = vva, vvb, vvb - vva
    la = pd.Series(ra ** 2).rolling(16, min_periods=4).mean().bfill().to_numpy()
    lb = pd.Series(rb ** 2).rolling(16, min_periods=4).mean().bfill().to_numpy()
    f["lvol_ks"] = stats.ks_2samp(np.log(la + EPS), np.log(lb + EPS)).statistic
    f["lvol_logstd_gap"] = float(np.log(la + EPS).std() - np.log(lb + EPS).std())
    f["lvol_range_ratio"] = float(np.log((np.quantile(lb, .9) + EPS) / (np.quantile(lb, .1) + EPS)) - np.log((np.quantile(la, .9) + EPS) / (np.quantile(la, .1) + EPS)))
    return f


DF_GRID = np.array([1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 8, 10, 14, 20, 30, 50, 100, 300])


def t_loglik(z, df, scale):
    return float(len(z) * (special.gammaln((df + 1) / 2) - special.gammaln(df / 2) - 0.5 * np.log(df * np.pi) - np.log(scale))
                 - (df + 1) / 2 * np.log1p((z / scale) ** 2 / df).sum())


def t_fit(z):
    """Student-t (df, scale) by maximum likelihood: EM for the scale at each df on a grid, then a
    parabolic refinement of log-df.  Returns (log df, scale, log-likelihood)."""
    z = z - np.median(z)
    lls, scales = [], []
    for df in DF_GRID:
        s2 = max(z.var(), EPS)
        for _ in range(6):
            w = (df + 1) / (df + z ** 2 / s2)
            s2 = max((w * z ** 2).mean(), EPS)
        scales.append(np.sqrt(s2))
        lls.append(t_loglik(z, df, np.sqrt(s2)))
    lls = np.array(lls)
    i = int(lls.argmax())
    ldf = np.log(DF_GRID)
    best = float(ldf[i])
    if 0 < i < len(DF_GRID) - 1:
        x0, x1, x2 = ldf[i - 1:i + 2]
        y0, y1, y2 = lls[i - 1:i + 2]
        den = (x0 - x1) * (x0 - x2) * (x1 - x2)
        a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / den
        bq = (x2 ** 2 * (y0 - y1) + x1 ** 2 * (y2 - y0) + x0 ** 2 * (y1 - y2)) / den
        if a < 0:
            best = float(np.clip(-bq / (2 * a), x0, x2))
    return best, float(scales[i]), float(lls[i])


def gauss_loglik(z, scale):
    return float(-len(z) * (0.5 * np.log(2 * np.pi) + np.log(scale)) - 0.5 * (z ** 2).sum() / scale ** 2)


def tail_index(za, zb):
    """Tail index of the innovations by likelihood, and likelihood-ratio statistics: did the after-part
    residuals come from the same t distribution as the before-part ones?  'shape' is the part of the
    change that a pure scale change cannot explain."""
    f = {}
    whitenings = {}
    for p in (1, 4):
        c, i, _ = ar_fit(za, p)
        whitenings[f"ar{p}"] = (ar_apply(za, c, i), ar_apply(zb, c, i))
    whitenings["d1"] = (np.diff(za), np.diff(zb))
    ca, ia, _ = ar_fit(za, 4)
    cb, ib, _ = ar_fit(zb, 4)
    whitenings["own4"] = (ar_apply(za, ca, ia), ar_apply(zb, cb, ib))
    for w, (ra, rb) in whitenings.items():
        la, sa, lla = t_fit(ra)
        lb, sb, llb = t_fit(rb)
        f[f"{w}_ldf_pre"], f[f"{w}_ldf_post"] = la, lb
        f[f"{w}_ldf_gap"], f[f"{w}_ldf_agap"] = lb - la, abs(lb - la)
        f[f"{w}_tscale_lratio"] = log_ratio(sb, sa)
        n = len(rb)
        rbm = rb - np.median(rb)
        ll_pre_t = t_loglik(rbm, np.exp(la), sa)
        ll_pre_g, ll_own_g = gauss_loglik(rbm, max(ra.std(), EPS)), gauss_loglik(rbm, max(rb.std(), EPS))
        f[f"{w}_lr_t"] = (llb - ll_pre_t) / n            # any change, under the t likelihood
        f[f"{w}_lr_g"] = (ll_own_g - ll_pre_g) / n       # scale change only, under a Gaussian
        f[f"{w}_lr_shape"] = f[f"{w}_lr_t"] - f[f"{w}_lr_g"]
        f[f"{w}_t_vs_g_post"] = (llb - ll_own_g) / n
        f[f"{w}_t_vs_g_pre"] = (lla - gauss_loglik(ra - np.median(ra), max(ra.std(), EPS))) / len(ra)
        f[f"{w}_t_vs_g_gap"] = f[f"{w}_t_vs_g_post"] - f[f"{w}_t_vs_g_pre"]
        h = n // 2
        if h >= 24:
            l1, l2 = t_fit(rb[:h])[0], t_fit(rb[h:])[0]
            f[f"{w}_ldf_h1_gap"], f[f"{w}_ldf_h2_gap"] = l1 - la, l2 - la
            f[f"{w}_ldf_consist"] = float(np.sign(l1 - la) == np.sign(l2 - la)) * min(abs(l1 - la), abs(l2 - la))
        else:
            f[f"{w}_ldf_h1_gap"] = f[f"{w}_ldf_h2_gap"] = f[f"{w}_ldf_consist"] = 0.0
    return f


def joint_law(za, zb, rng):
    """Energy distance and kernel MMD between the clouds of consecutive-reading vectors
    (x_t, x_{t-1}, ...) before and after: a test on the whole joint law, not only marginals."""
    f = {}
    c, i, ra = ar_fit(za, 4)
    rb = ar_apply(zb, c, i)
    ra, rb = ra / max(ra.std(), EPS), rb / max(ra.std(), EPS)

    def embed(x, d):
        return np.column_stack([x[d - 1 - k: len(x) - k] for k in range(d)])

    def energy_mmd(A, B):
        if len(A) > 256:
            A = A[rng.choice(len(A), 256, replace=False)]
        if len(B) > 256:
            B = B[rng.choice(len(B), 256, replace=False)]
        sq = lambda P, Q: np.maximum((P ** 2).sum(1)[:, None] + (Q ** 2).sum(1)[None, :] - 2 * P @ Q.T, 0)
        dAA, dBB, dAB = sq(A, A), sq(B, B), sq(A, B)
        energy = 2 * np.sqrt(dAB).mean() - np.sqrt(dAA).mean() - np.sqrt(dBB).mean()
        med = np.median(np.concatenate([dAA[np.triu_indices(len(A), 1)], dBB[np.triu_indices(len(B), 1)]])) + EPS
        k = lambda Dm: np.exp(-Dm / med)
        return float(energy), float(k(dAA).mean() + k(dBB).mean() - 2 * k(dAB).mean())

    for src, (xa, xb) in {"z": (za, zb), "r": (ra, rb)}.items():
        for d in (2, 3, 5):
            f[f"{src}_emb{d}_energy"], f[f"{src}_emb{d}_mmd"] = energy_mmd(embed(xa, d), embed(xb, d))
        h = len(xa) // 2
        if h >= 16:   # the same statistic between the two halves of the before part, as a per-series reference
            e0, m0 = energy_mmd(embed(xa[:h], 3), embed(xa[h:], 3))
            f[f"{src}_emb3_energy_excess"], f[f"{src}_emb3_mmd_excess"] = f[f"{src}_emb3_energy"] - e0, f[f"{src}_emb3_mmd"] - m0
        else:
            f[f"{src}_emb3_energy_excess"] = f[f"{src}_emb3_mmd_excess"] = 0.0
        # SAX words: quantise with the before-part quartiles, count 3-letter patterns
        q = np.quantile(xa, [.25, .5, .75])
        sa_, sb_ = np.digitize(xa, q), np.digitize(xb, q)
        ha = np.bincount(sum(sa_[k: len(sa_) - 2 + k] * 4 ** k for k in range(3)), minlength=64) / max(len(sa_) - 2, 1)
        hb = np.bincount(sum(sb_[k: len(sb_) - 2 + k] * 4 ** k for k in range(3)), minlength=64) / max(len(sb_) - 2, 1)
        f[f"{src}_sax_l1"] = float(np.abs(ha - hb).sum())
        f[f"{src}_sax_kl"] = float((hb * np.log((hb + 1e-6) / (ha + 1e-6))).sum())
        f[f"{src}_sax_maxdiff"] = float(np.abs(ha - hb).max())
    return f


# ============================================================================ one series -> one row
def series_features(ident, a, b):
    """All families for one series.  `a` and `b` are the raw before/after values."""
    sa = max(a.std(), EPS)
    za, zb = (a - a.mean()) / sa, (b - a.mean()) / sa          # standardise by the before part
    rng = np.random.default_rng(int(ident[:8], 16))            # reproducible subsampling seed from the hex id
    f = {"id": ident}
    f.update(basic_gaps(za, zb))
    f.update(autocorrelation(za, zb))
    f.update(ar_models(za, zb))
    f.update(trend_stability(za, zb))
    f.update(change_point(za, zb))
    shape = innovation_shape(za, zb)
    f.update(shape)
    f.update(volatility_shape(za, zb))
    f.update(tail_index(za, zb))
    f.update(joint_law(za, zb, rng))
    # the magnitude of a shape change matters more than its direction: add |gap| for the shape family
    for k, v in shape.items():
        if k.endswith("_gap"):
            f["abs_" + k] = abs(v)
    return f


def extract(table):
    rows = []
    t0 = time.time()
    for i, (ident, g) in enumerate(table.groupby("id", sort=False)):
        v = g["value"].to_numpy(dtype=float)
        p = g["period"].to_numpy()
        rows.append(series_features(ident, v[p == 0], v[p == 1]))
        if (i + 1) % 500 == 0:
            print(f"  {i + 1} series, {time.time() - t0:.0f}s", flush=True)
    F = pd.DataFrame(rows).set_index("id")
    return F.replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)


if __name__ == "__main__":
    for name in ("train", "test"):
        t0 = time.time()
        series = pd.read_parquet(DATA / f"{name}_series.parquet")
        F = extract(series)
        F.to_parquet(HERE / f"features_{name}.parquet")
        print(f"{name}: {F.shape} -> features_{name}.parquet in {time.time() - t0:.0f}s")
