# CPU port of experiments/v6/features_ps.py, source sha256 0352f0c097acdc31d6d0f4a5daaeb39c2a6933b1d8d2fc4f50f7504670b1914d. See auc_lab_20261107/claude/audit/cpu_port_notes.md.
import sys as _s, pathlib as _p; _s.path.insert(0, str(_p.Path(__file__).resolve().parents[2])); import ku_workspace as KU_ENV
"""Persistence / unit-root / shape block (2026-11-03, literature research: LTK, Kim, Perron, GPH, GMM-LR, dCor). Prefix ps_. Intel Arc, vectorised.
  adf_{b,a,d}     ADF t-stat (4 lags, const) on the levels of each segment; d = after - before
  per_F, per_dt   Perron known-break regression on the whole series: dy_t ~ y_{t-1} + 1 + DU + DT + 4 lags; F of (DU, DT) and ADF-t change vs no dummies
  kim_lr          log Kim ratio: post vs pre n^-2 sum of squared partial sums of demeaned levels (unnormalised, both directions in one signed log)
  kimn_lr         same normalised by segment variance of first differences (persistence change net of scale)
  vr{2,8,32}_lr   Lo-MacKinlay variance ratio of first differences, log post/pre... computed on increments of the cumulated standardised series
  gph_{b,a,d}     GPH log-periodogram long-memory d on each segment (m = n^0.6 Fourier frequencies), on z
  gmm_lr          2-component Gaussian mixture log-likelihood: fit(pre)+fit(post)-fit(pooled), per point; plus gmm1 single-Gaussian version gauss_lr
  bc_d, l3_d, l4_d  bimodality coefficient, L-skew, L-kurtosis post - pre
  dc{1,2}_d       distance correlation at lag 1/2 (on z) post - pre; dca1_d on |z|
Usage: .venv/Scripts/python features_ps.py train|test|aug|var:<stem>"""
import sys, os, time, math, pathlib, warnings, ctypes; warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
try: ctypes.windll.kernel32.SetPriorityClass(ctypes.c_void_p(-1), 0x4000)
except Exception: pass
import numpy as np, pandas as pd, torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
name = sys.argv[1] if len(sys.argv) > 1 else "train"; sys.argv = sys.argv[:2]
from features_lr import build_jobs, O
dev = "cpu"; torch.set_num_threads(1); t0 = time.time()
EPS = 1e-9; LAGS = 4; DCMAX = 600


def pad(segs):
    n = torch.tensor([len(s) for s in segs], device=dev); L = int(n.max())
    S = torch.zeros(len(segs), L, device=dev)
    for i, s in enumerate(segs): S[i, :len(s)] = torch.as_tensor(s, device=dev, dtype=torch.float32)
    return S, n, torch.arange(L, device=dev)[None, :] < n[:, None]


def ols(Xd, y, m):
    """batched masked OLS: Xd (N,T,k), y (N,T), m (N,T) -> beta, resid ss, (X'X)^-1"""
    Xm = Xd * m[..., None]; G = Xm.transpose(1, 2) @ Xd + 1e-6 * torch.eye(Xd.shape[2], device=dev); Gi = torch.linalg.inv(G)
    b = (Gi @ (Xm.transpose(1, 2) @ y[..., None])).squeeze(-1); r = (y - (Xd @ b[..., None]).squeeze(-1)) * m
    return b, (r ** 2).sum(1), Gi


def adf_design(Y, M, extra=None):
    """dy_t on y_{t-1}, const, lags of dy; rows t >= LAGS+1 inside mask."""
    dy = Y[:, 1:] - Y[:, :-1]; T = dy.shape[1]; cols = [Y[:, :-1], torch.ones_like(dy)]
    for l in range(1, LAGS + 1): cols.append(torch.nn.functional.pad(dy, (l, 0))[:, :T])
    if extra is not None: cols += [e[:, 1:] for e in extra]
    Xd = torch.stack(cols, 2); m = (M[:, 1:] & (torch.arange(T, device=dev)[None, :] >= LAGS)).float()
    return Xd, dy, m


def adf_t(Y, M, extra=None):
    Xd, dy, m = adf_design(Y, M, extra); b, ss, Gi = ols(Xd, dy, m); df = (m.sum(1) - Xd.shape[2]).clamp(min=1)
    return b[:, 0] / torch.sqrt((ss / df) * Gi[:, 0, 0]).clamp(min=EPS), ss, df


def gph(S, n, M):
    out = []
    for i in range(S.shape[0]):                                                # per series FFT of exact length (cheap)
        x = S[i, :int(n[i])]; x = x - x.mean(); nn = x.shape[0]; mm = max(int(nn ** 0.6), 6)
        I = (torch.fft.rfft(x).abs() ** 2)[1:mm + 1] / nn; lam = 2 * math.pi * torch.arange(1, mm + 1, device=dev) / nn
        r = torch.log(4 * torch.sin(lam / 2) ** 2); yv = torch.log(I.clamp(min=EPS)); rc = r - r.mean()
        out.append(-(rc * (yv - yv.mean())).sum() / (rc ** 2).sum())
    return torch.stack(out)


def gmm_ll(S, M, iters=40):
    """mean log-lik per point of a 2-comp 1D GMM fitted by EM (and of a single Gaussian)."""
    Mf = M.float(); n = Mf.sum(1).clamp(min=1); mu0 = (S * Mf).sum(1) / n; v0 = (((S - mu0[:, None]) ** 2) * Mf).sum(1) / n + EPS
    sd0 = v0.sqrt(); g1 = (-0.5 * ((S - mu0[:, None]) ** 2) / v0[:, None] - 0.5 * torch.log(2 * math.pi * v0[:, None]))
    ll1 = (g1 * Mf).sum(1) / n
    mu = torch.stack([mu0 - 0.5 * sd0, mu0 + 0.5 * sd0], 1); var = torch.stack([v0, v0], 1) * 0.8; pi = torch.full_like(mu, 0.5)
    for _ in range(iters):
        lp = torch.log(pi[:, None, :]) - 0.5 * (S[..., None] - mu[:, None, :]) ** 2 / var[:, None, :] - 0.5 * torch.log(2 * math.pi * var[:, None, :])
        r = torch.softmax(lp, 2) * Mf[..., None]; nk = r.sum(1).clamp(min=1e-3)
        pi = nk / n[:, None]; mu = (r * S[..., None]).sum(1) / nk; var = ((r * (S[..., None] - mu[:, None, :]) ** 2).sum(1) / nk).clamp(min=0.01 * v0[:, None])
    lp = torch.log(pi[:, None, :]) - 0.5 * (S[..., None] - mu[:, None, :]) ** 2 / var[:, None, :] - 0.5 * torch.log(2 * math.pi * var[:, None, :])
    return (torch.logsumexp(lp, 2) * Mf).sum(1), ll1 * n, n


def lmom(S, M):
    big = torch.where(M, S, torch.full_like(S, float("inf"))); xs = torch.sort(big, 1).values; n = M.sum(1).float()
    i = torch.arange(S.shape[1], device=dev)[None, :].float(); ok = i < n[:, None]; xs = torch.where(ok, xs, torch.zeros_like(xs))
    b0 = xs.sum(1) / n; b1 = (xs * i / (n[:, None] - 1)).sum(1) / n; b2 = (xs * i * (i - 1) / ((n[:, None] - 1) * (n[:, None] - 2))).sum(1) / n
    b3 = (xs * i * (i - 1) * (i - 2) / ((n[:, None] - 1) * (n[:, None] - 2) * (n[:, None] - 3))).sum(1) / n
    l2 = 2 * b1 - b0; l3 = 6 * b2 - 6 * b1 + b0; l4 = 20 * b3 - 30 * b2 + 12 * b1 - b0
    return l3 / l2.clamp(min=EPS), l4 / l2.clamp(min=EPS)


def dcor(S, n, lag):
    """distance correlation of (x_t, x_{t-lag}) on the first DCMAX points of each segment."""
    T = min(int(n.max()), DCMAX) - lag; a = S[:, lag:lag + T]; b = S[:, :T]; m = (torch.arange(T, device=dev)[None, :] < (n.clamp(max=DCMAX) - lag)[:, None]).float()
    def cen(x):
        D = (x[:, :, None] - x[:, None, :]).abs(); mm = m[:, :, None] * m[:, None, :]; k = m.sum(1).clamp(min=2)
        r = (D * mm).sum(2) / k[:, None]; g = (D * mm).sum((1, 2)) / k ** 2
        return (D - r[:, :, None] - r[:, None, :] + g[:, None, None]) * mm, k
    A, k = cen(a); B, _ = cen(b)
    vab = (A * B).sum((1, 2)) / k ** 2; vaa = (A * A).sum((1, 2)) / k ** 2; vbb = (B * B).sum((1, 2)) / k ** 2
    return torch.sqrt(vab.clamp(min=0) / torch.sqrt(vaa * vbb).clamp(min=EPS))


def seg(S, n, M):
    f = {}; Mf = M.float(); nf = n.float()
    f["adf"], _, _ = adf_t(S, M)
    mu = (S * Mf).sum(1) / nf; e = (S - mu[:, None]) * Mf; c = torch.cumsum(e, 1) * Mf
    d = (S[:, 1:] - S[:, :-1]) * M[:, 1:]; vd = (d ** 2).sum(1) / (nf - 1).clamp(min=1)
    f["kim"] = (c ** 2).sum(1) / nf ** 2; f["kimn"] = f["kim"] / (nf * vd).clamp(min=EPS)
    z = torch.cumsum(d, 1)
    for q in (2, 8, 32):
        dq = (torch.nn.functional.pad(z, (q, 0))[:, q:] - torch.nn.functional.pad(z, (q, 0))[:, :-q]); mq = M[:, 1:] & (torch.arange(d.shape[1], device=dev)[None, :] >= q)
        f[f"vr{q}"] = ((dq ** 2) * mq).sum(1) / mq.sum(1).clamp(min=1) / (q * vd).clamp(min=EPS)
    f["gph"] = gph(S, n, M)
    sd = torch.sqrt((e ** 2).sum(1) / nf).clamp(min=EPS); Z = e / sd[:, None]
    m2 = ((Z ** 2) * Mf).sum(1) / nf; m3 = ((Z ** 3) * Mf).sum(1) / nf; m4 = ((Z ** 4) * Mf).sum(1) / nf
    f["bc"] = (m3 ** 2 + 1) / m4.clamp(min=EPS)
    f["l3"], f["l4"] = lmom(S, M)
    f["dc1"], f["dc2"] = dcor(Z, n, 1), dcor(Z, n, 2); f["dca1"] = dcor(Z.abs(), n, 1)
    return f


def block(js):
    pre, post, whole, ks = [], [], [], []
    for _, v, k in js:
        v = np.asarray(v, np.float64); m, s = v[:k].mean(), v[:k].std() + 1e-9; w = (v - m) / s
        pre.append(w[:k]); post.append(w[k:]); whole.append(w); ks.append(k)
    Sb, nb, Mb = pad(pre); Sa, na, Ma = pad(post); Sw, nw, Mw = pad(whole); kk = torch.tensor(ks, device=dev)
    fb, fa = seg(Sb, nb, Mb), seg(Sa, na, Ma); out = {}
    for key in fb:
        if key in ("adf", "gph", "bc", "l3", "l4", "dc1", "dc2", "dca1"): out[f"ps_{key}_d"] = fa[key] - fb[key]
        else: out[f"ps_{key}_lr"] = torch.log(fa[key].clamp(min=EPS) / fb[key].clamp(min=EPS))
    out["ps_adf_b"], out["ps_adf_a"], out["ps_gph_b"], out["ps_gph_a"] = fb["adf"], fa["adf"], fb["gph"], fa["gph"]
    # Perron known-break test on the whole series
    tt = torch.arange(Sw.shape[1], device=dev)[None, :]; DU = (tt >= kk[:, None]).float(); DT = (tt - kk[:, None]).clamp(min=0).float() / 100
    t0_, ss0, df0 = adf_t(Sw, Mw); t1_, ss1, df1 = adf_t(Sw, Mw, [DU, DT])
    out["ps_per_F"] = torch.log1p(((ss0 - ss1) / 2).clamp(min=0) / (ss1 / df1).clamp(min=EPS)); out["ps_per_dt"] = t1_ - t0_
    # Gaussian-mixture likelihood ratio (shape change), per point
    if True:
        Lm = max(Sb.shape[1], Sa.shape[1]); padf = lambda X: torch.nn.functional.pad(X, (0, Lm - X.shape[1]))
        S2 = torch.cat([padf(Sb), padf(Sa)], 0); M2 = torch.cat([padf(Mb.float()), padf(Ma.float())], 0) > 0
    g2, g1, _ = gmm_ll(S2, M2); W2, W1, nW = gmm_ll(Sw, Mw); N = Sb.shape[0]
    out["ps_gmm_lr"] = (g2[:N] + g2[N:] - W2) / nW; out["ps_gauss_lr"] = (g1[:N] + g1[N:] - W1) / nW; out["ps_gmm_x"] = out["ps_gmm_lr"] - out["ps_gauss_lr"]
    return pd.DataFrame({k: v.float().cpu().numpy() for k, v in out.items()}, index=[j[0] for j in js])


if __name__ == "__main__":
    jobs, _ = build_jobs(name)
    out = O / "aug_ps_features.parquet" if name == "aug" else (O / "var" / f"ps_{name[4:]}.parquet" if name.startswith("var:") else O / f"ps_{name}.parquet")
    CH = int(os.environ.get("CHUNK", "128")); parts = []
    for i in range(0, len(jobs), CH):
        parts.append(block(jobs[i:i + CH]))
        if i == 0: print(f"  first chunk {time.time()-t0:.1f}s", flush=True)
    F = pd.concat(parts).replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32); F.index.name = "id"; F.to_parquet(out)
    print(f"{name}: {F.shape} -> {out.name} in {time.time()-t0:.0f}s", flush=True)
