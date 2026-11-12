# CPU port of experiments/v6/features_knn.py, source sha256 804bb46874c897a00159e3420ef64701ef806c509cf22a29a05f2baf158283f9. See auc_lab_20261107/claude/audit/cpu_port_notes.md.
import sys as _s, pathlib as _p; _s.path.insert(0, str(_p.Path(__file__).resolve().parents[2])); import ku_workspace as KU_ENV
"""Nonparametric before-segment surprise block (2026-11-03), Intel Arc. Prefix kn_.
The before segment is the model: every point t is described by its delay vector (x_{t-1}, ..., x_{t-m}); its forecast is the mean next value of the
k nearest delay vectors whose targets lie in the BEFORE segment (method of analogues). Before points are scored leave-out with a Theiler window
(|t - j| > m), after points against the full before pool, so each series supplies its own calibration. Catches dynamics an AR(p) cannot represent
(nonlinear / state-dependent changes) as well as ordinary level / scale shifts.
Per stream (z = before-standardised series, r = AR(12) innovations fitted on the before part) and embedding m:
  mse_lr, mae_lr      log ratio after / before of squared / absolute forecast error
  r2_d                change in predictive R^2 (1 - mse / variance about the before mean)
  nov_lr, novm_lr     log ratio of mean / median nearest-neighbour distance (state-space novelty)
  cp_nlp, cp_05, cp_01  conformal: rank of each after |error| among the before |errors| -> mean -log p, fraction p < .05 / .01 minus nominal
  ls_d                mean log predictive density (Gaussian kernel mixture over the k neighbours' next values) after minus before
  cs_max              max |CUSUM| of the after conformal scores (p - 0.5), scaled by sqrt(n_after) (a drifting / delayed change)
Usage: .venv/Scripts/python features_knn.py train|test|aug|var:<stem>"""
import sys, os, time, math, pathlib, warnings, ctypes; warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
try: ctypes.windll.kernel32.SetPriorityClass(ctypes.c_void_p(-1), 0x4000)
except Exception: pass
import numpy as np, pandas as pd, torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
name = sys.argv[1] if len(sys.argv) > 1 else "train"; sys.argv = sys.argv[:2]
from features_lr import build_jobs, O
dev = "cpu"; torch.set_num_threads(1); t0 = time.time()
L, P, K = 1024, 12, 10; BIG = 1e9


def surprise(S, cand, qb, qa, m, pre_):
    """S (N,L) stream; cand (N,L) bool = targets allowed as analogues (before, t >= m); qb / qa = before / after query masks."""
    N = S.shape[0]; Sp = torch.nn.functional.pad(S, (m, 0)); E = Sp.unfold(1, m, 1)[:, :L, :]          # E[:, t] = S[t-m : t]
    D = torch.cdist(E, E)                                                                                # (N, L, L)
    tt = torch.arange(L, device=dev); theiler = (tt[:, None] - tt[None, :]).abs() <= m
    D = torch.where(cand[:, None, :] & ~theiler[None], D, torch.full_like(D, BIG))
    d, idx = torch.topk(D, K, dim=2, largest=False); del D
    nxt = S.gather(1, idx.view(N, -1)).view(N, L, K)                                                    # analogues' next values
    pred = nxt.mean(2); e = S - pred; ok = d[..., 0] < BIG / 2
    qb, qa = qb & ok, qa & ok; nb, na = qb.sum(1).clamp(min=5).float(), qa.sum(1).clamp(min=5).float()
    mean = lambda x, q, n: (x * q).sum(1) / n
    f = {}; mse_b, mse_a = mean(e ** 2, qb, nb), mean(e ** 2, qa, na)
    f[f"{pre_}mse_lr"] = torch.log(mse_a.clamp(min=1e-9) / mse_b.clamp(min=1e-9))
    f[f"{pre_}mae_lr"] = torch.log(mean(e.abs(), qa, na).clamp(min=1e-9) / mean(e.abs(), qb, nb).clamp(min=1e-9))
    mu_b = mean(S, qb, nb); var_b = mean((S - mu_b[:, None]) ** 2, qb, nb).clamp(min=1e-9); var_a = mean((S - mu_b[:, None]) ** 2, qa, na).clamp(min=1e-9)
    f[f"{pre_}r2_d"] = (1 - mse_a / var_a) - (1 - mse_b / var_b)
    d1 = d[..., 0]                                                                                       # nearest-neighbour distance
    f[f"{pre_}nov_lr"] = torch.log(mean(d1, qa, na).clamp(min=1e-9) / mean(d1, qb, nb).clamp(min=1e-9))
    med = lambda x, q: torch.nanquantile(torch.where(q, x, torch.full_like(x, float("nan"))), 0.5, dim=1)
    f[f"{pre_}novm_lr"] = torch.log(med(d1, qa).clamp(min=1e-9) / med(d1, qb).clamp(min=1e-9))
    # conformal p-values of after |errors| against the before |errors|
    ab = torch.where(qb, e.abs(), torch.full_like(e, float("inf"))); srt = torch.sort(ab, 1).values
    ge = nb[:, None] - torch.searchsorted(srt, e.abs().contiguous(), right=False).float().clamp(max=nb[:, None])   # count of before >= after error
    p = (ge + 1) / (nb[:, None] + 1)
    f[f"{pre_}cp_nlp"] = mean(-torch.log(p), qa, na) - 1.0
    f[f"{pre_}cp_05"] = mean((p < 0.05).float(), qa, na) - 0.05; f[f"{pre_}cp_01"] = mean((p < 0.01).float(), qa, na) - 0.01
    sc = torch.cumsum((p - 0.5) * qa.float(), 1); f[f"{pre_}cs_max"] = (sc.abs() * qa.float()).max(1).values / na.sqrt()
    # nonparametric predictive log score: Gaussian kernel mixture over the neighbours' next values, bandwidth = before error sd
    h = mse_b.sqrt().clamp(min=1e-3)[:, None, None]
    ls = torch.logsumexp(-0.5 * ((S[..., None] - nxt) / h) ** 2, 2) - math.log(K) - torch.log(h[..., 0]) - 0.5 * math.log(2 * math.pi)
    f[f"{pre_}ls_d"] = mean(ls, qa, na) - mean(ls, qb, nb)
    return f


def block(js):
    N = len(js); X = torch.zeros(N, L, device=dev); n = torch.tensor([min(len(v), L) for _, v, _ in js], device=dev); b = torch.tensor([k for _, _, k in js], device=dev)
    for i, (_, v, _) in enumerate(js): X[i, :min(len(v), L)] = torch.as_tensor(v[:L], device=dev, dtype=torch.float32)
    tt = torch.arange(L, device=dev)[None, :]; pre = (tt < b[:, None]).float(); post = ((tt >= b[:, None]) & (tt < n[:, None])).float(); valid = pre + post
    m0 = (X * pre).sum(1) / b; sd = torch.sqrt((((X - m0[:, None]) * pre) ** 2).sum(1) / b).clamp(min=1e-6); Z = (X - m0[:, None]) / sd[:, None] * valid
    U = Z.unfold(1, P + 1, 1); yv = U[:, :, -1]; A = torch.cat([U[:, :, :-1].flip(-1), torch.ones_like(yv)[..., None]], -1)
    mk = pre[:, P:][..., None]; G = (A * mk).transpose(1, 2) @ A + 1e-3 * torch.eye(P + 1, device=dev); h = (A * mk).transpose(1, 2) @ yv[..., None]
    beta = torch.linalg.solve(G, h).squeeze(-1); R = torch.zeros(N, L, device=dev); R[:, P:] = (yv - (A @ beta[..., None]).squeeze(-1)) * valid[:, P:]
    rpre = pre * (tt >= P).float(); rs = torch.sqrt(((R * rpre) ** 2).sum(1) / rpre.sum(1).clamp(min=1)).clamp(min=1e-6); R = R / rs[:, None]
    f = {}
    for S, s0, nm in ((Z, 0, "z"), (R, P, "r")):
        for m in (3, 6):
            st = tt >= (s0 + m); cand = (pre > 0) & st; qb = cand.clone(); qa = (post > 0) & st
            f.update(surprise(S, cand, qb, qa, m, f"kn_{nm}{m}_"))
    return pd.DataFrame({k: v.float().cpu().numpy() for k, v in f.items()}, index=[j[0] for j in js])


if __name__ == "__main__":
    jobs, _ = build_jobs(name)
    out = O / "aug_kn_features.parquet" if name == "aug" else (O / "var" / f"kn_{name[4:]}.parquet" if name.startswith("var:") else O / f"kn_{name}.parquet")
    CH = int(os.environ.get("CHUNK", "48")); parts = []
    for i in range(0, len(jobs), CH):
        parts.append(block(jobs[i:i + CH]))
        if i == 0: print(f"  first chunk {time.time()-t0:.1f}s", flush=True)
    F = pd.concat(parts).replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32); F.index.name = "id"; F.to_parquet(out)
    print(f"{name}: {F.shape} -> {out.name} in {time.time()-t0:.0f}s", flush=True)
