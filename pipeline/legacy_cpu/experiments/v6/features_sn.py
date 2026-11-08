# CPU port of experiments/v6/features_sn.py, source sha256 a5b381b9b3c905bbca36acf863f834941d83926ab9b4f0d5df69a92faba68aae. See auc_lab_20261107/claude/audit/cpu_port_notes.md.
import sys as _s, pathlib as _p; _s.path.insert(0, str(_p.Path(__file__).resolve().parents[2])); import ku_workspace as KU_ENV
"""Self-normalised CUSUM block at the known boundary (Shao & Zhang 2010, JASA), Intel Arc. Prefix sn_.
For a stream x with boundary k in a series of n: D = k(n-k)/n^1.5 * (mean_before - mean_after);
V = n^-2 [ sum_{t<=k} (S_{1,t} - t/k S_{1,k})^2 + sum_{t>k} (S_{t,n} - (n-t+1)/(n-k) S_{k+1,n})^2 ]; features: D/sqrt(V) (signed) and D^2/V.
The normaliser is built from within-segment partial sums, so the statistic stays pivotal under serial dependence without a bandwidth or an AR model.
Streams (z = before-standardised series, r = AR(12) innovations fitted on the before part): x, |x - before median|, x^2, x_t*x_{t-1}; plus b/n.
Usage: .venv/Scripts/python features_sn.py train|test|aug|var:<stem>"""
import sys, os, time, pathlib, warnings, ctypes; warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
try: ctypes.windll.kernel32.SetPriorityClass(ctypes.c_void_p(-1), 0x4000)
except Exception: pass
import numpy as np, pandas as pd, torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
name = sys.argv[1] if len(sys.argv) > 1 else "train"; sys.argv = sys.argv[:2]
from features_lr import build_jobs, O
dev = "cpu"; torch.set_num_threads(1); t0 = time.time()
L, P = 1024, 12


def sn(x, ma, mb):
    """x (N,L) stream; ma/mb contiguous float masks (before / after). Returns signed D/sqrt(V)."""
    na, nb = ma.sum(1).clamp(min=2), mb.sum(1).clamp(min=2); n = na + nb
    xa, xb = x * ma, x * mb; Sa = torch.cumsum(xa, 1); SA = xa.sum(1, keepdim=True)
    ta = torch.cumsum(ma, 1); ra = ((Sa - ta / na[:, None] * SA) ** 2 * ma).sum(1)
    Sb = torch.flip(torch.cumsum(torch.flip(xb, [1]), 1), [1]); SB = xb.sum(1, keepdim=True)   # S_{t,n} over the after side
    tb = torch.flip(torch.cumsum(torch.flip(mb, [1]), 1), [1]); rb = ((Sb - tb / nb[:, None] * SB) ** 2 * mb).sum(1)
    D = na * nb / n ** 1.5 * (SA[:, 0] / na - SB[:, 0] / nb); V = ((ra + rb) / n ** 2).clamp(min=1e-9)
    return D / torch.sqrt(V)


def streams(S, ma, mb, pre_):
    med = (S * ma).sum(1) / ma.sum(1)  # mean as a cheap centre (the stream is before-standardised already)
    lagp = torch.zeros_like(S); lagp[:, 1:] = S[:, 1:] * S[:, :-1]
    mb_l = mb.clone(); mb_l[:, 1:] = mb[:, 1:] * mb[:, :-1]; ma_l = ma.clone(); ma_l[:, 0] = 0
    f = {}
    for nm, x, a_, b_ in (("lvl", S, ma, mb), ("abs", (S - med[:, None]).abs(), ma, mb), ("sq", S ** 2, ma, mb), ("lag", lagp, ma_l, mb_l)):
        t = sn(x, a_, b_); f[f"{pre_}{nm}"] = t; f[f"{pre_}{nm}_g"] = t ** 2
    return f


def block(js):
    N = len(js); X = torch.zeros(N, L, device=dev); n = torch.tensor([min(len(v), L) for _, v, _ in js], device=dev); b = torch.tensor([k for _, _, k in js], device=dev)
    for i, (_, v, _) in enumerate(js): X[i, :min(len(v), L)] = torch.as_tensor(v[:L], device=dev, dtype=torch.float32)
    tt = torch.arange(L, device=dev)[None, :]; pre = (tt < b[:, None]).float(); post = ((tt >= b[:, None]) & (tt < n[:, None])).float(); valid = pre + post
    m = (X * pre).sum(1) / b; sd = torch.sqrt((((X - m[:, None]) * pre) ** 2).sum(1) / b).clamp(min=1e-6); Z = (X - m[:, None]) / sd[:, None] * valid
    U = Z.unfold(1, P + 1, 1); yv = U[:, :, -1]; A = torch.cat([U[:, :, :-1].flip(-1), torch.ones_like(yv)[..., None]], -1)
    mk = pre[:, P:][..., None]; G = (A * mk).transpose(1, 2) @ A + 1e-3 * torch.eye(P + 1, device=dev); h = (A * mk).transpose(1, 2) @ yv[..., None]
    beta = torch.linalg.solve(G, h).squeeze(-1); R = torch.zeros(N, L, device=dev); R[:, P:] = (yv - (A @ beta[..., None]).squeeze(-1)) * valid[:, P:]
    rpre = pre * (tt >= P).float(); rs = torch.sqrt(((R * rpre) ** 2).sum(1) / rpre.sum(1).clamp(min=1)).clamp(min=1e-6); R = R / rs[:, None]
    f = streams(Z, pre, post, "sn_z_"); f.update(streams(R, rpre, post, "sn_r_")); f["sn_pos"] = b.float() / n.float()
    return pd.DataFrame({k: v.float().cpu().numpy() for k, v in f.items()}, index=[j[0] for j in js])


if __name__ == "__main__":
    jobs, _ = build_jobs(name)
    out = O / "aug_sn_features.parquet" if name == "aug" else (O / "var" / f"sn_{name[4:]}.parquet" if name.startswith("var:") else O / f"sn_{name}.parquet")
    CH = int(os.environ.get("CHUNK", "2000")); parts = [block(jobs[i:i + CH]) for i in range(0, len(jobs), CH)]
    F = pd.concat(parts).replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32); F.index.name = "id"; F.to_parquet(out)
    print(f"{name}: {F.shape} -> {out.name} in {time.time()-t0:.0f}s", flush=True)
