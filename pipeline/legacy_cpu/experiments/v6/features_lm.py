# CPU port of experiments/v6/features_lm.py, source sha256 199aee7831f6b11166837bdd18fffebf5dc23b0830fabaca8f2a3a406feb081c. See auc_lab_20261107/claude/audit/cpu_port_notes.md.
import sys as _s, pathlib as _p; _s.path.insert(0, str(_p.Path(__file__).resolve().parents[2])); import ku_workspace as KU_ENV
"""Efficient-score (LM / locally most powerful) block (2026-11-03, research: Andrews-Ploberger, LM tests are optimal as break size -> 0). Prefix lm_. Intel Arc.
H0 nuisances are fitted on the WHOLE series (AR(p), p by BIC over 0..8, on the full-series standardised values), unlike the lr_ blocks (before-only whitening).
Per observation score s_t for each break type; known-date efficient score S = sum_{after} s_t - (n2/n) sum_all s_t, information I = var(s) * n1 n2 / n.
  z_<type>   signed S / sqrt(I);  lm_<type> = z^2  (types: lev, sc, ar1, ar2, ar3, kurt, rlev (Student-t psi level, nu=5), lsc (log|e| scale))
  lbf_lev, lbf_sc  integrated-score log Bayes factor with a uniform magnitude prior (level a = 2.2 sd, log-scale a = 0.7)
  lbf_mix    log of the prior-weighted mean of exp(lbf_lev), exp(lbf_sc) and (shape, AR) LM-based BFs
  lmsum      sum over types of (lm - 1), the low-SNR-optimal linear combination (equal weights)
  phi_sum    sum of AR coefficients of the full-series fit (persistence; 1 - phi_sum scales the level score)
Usage: .venv/Scripts/python features_lm.py train|test|aug|var:<stem>"""
import sys, os, time, math, pathlib, warnings, ctypes; warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
try: ctypes.windll.kernel32.SetPriorityClass(ctypes.c_void_p(-1), 0x4000)
except Exception: pass
import numpy as np, pandas as pd, torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
name = sys.argv[1] if len(sys.argv) > 1 else "train"; sys.argv = sys.argv[:2]
from features_lr import build_jobs, O
dev = "cpu"; torch.set_num_threads(1); t0 = time.time()
P = 8; EPS = 1e-9; SQ2 = math.sqrt(2.0)
Phi = lambda x: 0.5 * (1 + torch.erf(x / SQ2))


def int_lbf(S, I, a):
    """log BF of a shift delta ~ U(-a, a) from score S and information I (Gaussian local approximation)."""
    I = I.clamp(min=EPS); m = S / I; sd = 1 / torch.sqrt(I)
    mass = (Phi((a - m) / sd) - Phi((-a - m) / sd)).clamp(min=1e-30)
    return 0.5 * torch.log(2 * math.pi / I) + S ** 2 / (2 * I) + torch.log(mass) - math.log(2 * a)


def block(js):
    N = len(js); n = torch.tensor([len(v) for _, v, _ in js], device=dev); k = torch.tensor([kk for _, _, kk in js], device=dev); L = int(n.max())
    X = torch.zeros(N, L, device=dev)
    for i, (_, v, _) in enumerate(js): X[i, :len(v)] = torch.as_tensor(v, device=dev, dtype=torch.float32)
    tt = torch.arange(L, device=dev)[None, :]; valid = tt < n[:, None]
    mu = (X * valid).sum(1) / n; sd = torch.sqrt((((X - mu[:, None]) * valid) ** 2).sum(1) / n).clamp(min=1e-6); Z = (X - mu[:, None]) / sd[:, None] * valid
    U = Z.unfold(1, P + 1, 1); yv = U[:, :, -1]; LAG = U[:, :, :-1].flip(-1)                       # rows t = P..L-1; LAG[..., j] = z_{t-1-j}
    T = yv.shape[1]; trow = torch.arange(P, P + T, device=dev)[None, :]; m = (trow < n[:, None]).float(); nr = m.sum(1)
    after = ((trow >= k[:, None]).float() * m); n2 = after.sum(1); n1 = nr - n2
    best_bic = torch.full((N,), float("inf"), device=dev); E = torch.zeros(N, T, device=dev); phis = torch.zeros(N, device=dev)
    for p in range(0, P + 1):
        A = torch.cat([torch.ones_like(yv)[..., None], LAG[..., :p]], -1); Am = A * m[..., None]
        G = Am.transpose(1, 2) @ A + 1e-4 * torch.eye(p + 1, device=dev); b = torch.linalg.solve(G, Am.transpose(1, 2) @ yv[..., None]).squeeze(-1)
        r = (yv - (A @ b[..., None]).squeeze(-1)) * m; s2 = (r ** 2).sum(1) / nr; bic = nr * torch.log(s2.clamp(min=EPS)) + (p + 1) * torch.log(nr)
        better = bic < best_bic; best_bic = torch.where(better, bic, best_bic); E = torch.where(better[:, None], r, E)
        phis = torch.where(better, b[:, 1:].sum(1) if p > 0 else torch.zeros(N, device=dev), phis)
    s2 = ((E ** 2).sum(1) / nr).clamp(min=EPS); z = E / torch.sqrt(s2)[:, None] * m
    w_frac = n2 / nr
    def eff(s, nm):
        s = s * m; mean = s.sum(1) / nr; var = (((s - mean[:, None]) * m) ** 2).sum(1) / nr
        S = (s * after).sum(1) - w_frac * s.sum(1); I = (var * n1 * n2 / nr).clamp(min=EPS)
        return S, I
    yl = yv * m; f = {}; SI = {}
    SI["lev"] = eff(z * (1 - phis)[:, None], "lev")
    SI["sc"] = eff(z ** 2 - 1, "sc")
    for j in (1, 2, 3): SI[f"ar{j}"] = eff(z * LAG[..., j - 1], f"ar{j}")
    SI["kurt"] = eff((z ** 4 - 2 * z ** 2 - 1) / 4, "kurt")
    SI["rlev"] = eff(6 * z / (5 + z ** 2), "rlev")
    SI["lsc"] = eff(torch.log(z.abs() + 1e-3), "lsc")
    for t_, (S, I) in SI.items():
        zz = S / torch.sqrt(I); f[f"lm_z_{t_}"] = zz; f[f"lm_lm_{t_}"] = zz ** 2
    f["lm_lmsum"] = sum(f[f"lm_lm_{t_}"] - 1 for t_ in SI)
    # integrated Bayes factors. level score above is per unit of the shift in z units scaled by (1 - phi): delta in sd units of z.
    S, I = SI["lev"]; f["lm_lbf_lev"] = int_lbf(S, I, 2.2)
    S, I = SI["sc"]; f["lm_lbf_sc"] = int_lbf(S, I, 1.4)                                         # score of z^2 - 1 is per unit of 2 * log-sd change
    lbf_other = lambda t_: torch.log1p(0.5 * 0.05 * (f[f"lm_lm_{t_}"] - 1)).nan_to_num(0.0).clamp(min=-5)   # local expansion 1 + 1/2 E[d^2] I (LM - 1)
    stk = torch.stack([f["lm_lbf_lev"] + math.log(0.33), f["lm_lbf_sc"] + math.log(0.33), lbf_other("kurt") + math.log(0.15),
                       torch.logsumexp(torch.stack([lbf_other(f"ar{j}") for j in (1, 2, 3)]), 0) - math.log(3) + math.log(0.09), torch.full_like(phis, math.log(0.10))], 0)
    f["lm_lbf_mix"] = torch.logsumexp(stk, 0); f["lm_phi_sum"] = phis
    return pd.DataFrame({kk: v.float().cpu().numpy() for kk, v in f.items()}, index=[j[0] for j in js])


if __name__ == "__main__":
    jobs, _ = build_jobs(name)
    out = O / "aug_lm_features.parquet" if name == "aug" else (O / "var" / f"lm_{name[4:]}.parquet" if name.startswith("var:") else O / f"lm_{name}.parquet")
    CH = int(os.environ.get("CHUNK", "1000")); parts = []
    for i in range(0, len(jobs), CH): parts.append(block(jobs[i:i + CH]))
    F = pd.concat(parts).replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32); F.index.name = "id"; F.to_parquet(out)
    print(f"{name}: {F.shape} -> {out.name} in {time.time()-t0:.0f}s", flush=True)
