# CPU port of experiments/v6/features_sx.py, source sha256 6804325a8a9723c924858ce3a99008611202611d87cd7170f0d42b1fcff0f485. See auc_lab_20261107/claude/audit/cpu_port_notes.md.
import sys as _s, pathlib as _p; _s.path.insert(0, str(_p.Path(__file__).resolve().parents[2])); import ku_workspace as KU_ENV
"""Simple-extras block (2026-11-03), Intel Arc. Prefix sx_. Cheap classical statistics not in the table yet, each as post - pre (or a two-sample stat):
  h_{z,a}        aggregated-variance Hurst slope (block sums at scales 1..32) on z and |z|; d = post - pre
  vr{2,8,32}_{z,a}  log variance ratio var(q-block sums) / (q var) per segment; d = post - pre
  tp_d           turning-point rate (local extrema fraction, 2/3 under iid) post - pre
  up_d, upm_d    fraction of up-moves / log(mean up-move / mean down-move) post - pre (time irreversibility)
  op3_js, op4_js ordinal-pattern (order 3 / 4) Jensen-Shannon divergence pre vs post; op3_chi chi-square stat / n
  op3_ent_d      permutation entropy (order 3) post - pre
  rec_d          (max+min records per log n) post - pre
Segments are left-aligned in padded tensors with length masks, so everything is vectorised.
Usage: .venv/Scripts/python features_sx.py train|test|aug|var:<stem>"""
import sys, os, time, math, pathlib, warnings, ctypes; warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
try: ctypes.windll.kernel32.SetPriorityClass(ctypes.c_void_p(-1), 0x4000)
except Exception: pass
import numpy as np, pandas as pd, torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
name = sys.argv[1] if len(sys.argv) > 1 else "train"; sys.argv = sys.argv[:2]
from features_lr import build_jobs, O
dev = "cpu"; torch.set_num_threads(1); t0 = time.time()
EPS = 1e-9


def pad(segs):
    n = torch.tensor([len(s) for s in segs], device=dev); L = int(n.max())
    S = torch.zeros(len(segs), L, device=dev)
    for i, s in enumerate(segs): S[i, :len(s)] = torch.as_tensor(s, device=dev, dtype=torch.float32)
    return S, n, torch.arange(L, device=dev)[None, :] < n[:, None]


def seg_stats(S, n, M):
    """per-segment statistics of a left-aligned padded batch."""
    f = {}; nf = n.float()
    for nm, X in (("z", S), ("a", S.abs())):
        mu = (X * M).sum(1) / nf; Xc = (X - mu[:, None]) * M; v1 = (Xc ** 2).sum(1) / nf
        lv = []
        for q in (1, 2, 4, 8, 16, 32):
            nb = n // q; L = (X.shape[1] // q) * q
            B = Xc[:, :L].reshape(X.shape[0], -1, q).sum(2); bm = torch.arange(B.shape[1], device=dev)[None, :] < nb[:, None]
            vq = ((B ** 2) * bm).sum(1) / nb.clamp(min=1).float(); lr = torch.log((vq / (q * v1)).clamp(min=EPS))
            lr = torch.where(nb >= 8, lr, torch.zeros_like(lr)); lv.append(lr)
            if q in (2, 8, 32): f[f"vr{q}_{nm}"] = lr
        lq = torch.log(torch.tensor([1., 2, 4, 8, 16, 32], device=dev)); Y = torch.stack(lv, 1); lc = lq - lq.mean()
        f[f"h_{nm}"] = 0.5 * (1 + (Y * lc).sum(1) / (lc ** 2).sum())                                  # Hurst ~ 0.5 * (1 + slope)
    d = S[:, 1:] - S[:, :-1]; dm = M[:, 1:]; nd = dm.sum(1).clamp(min=1).float()
    up = (d > 0) & dm; dn = (d < 0) & dm
    f["up"] = up.sum(1) / nd
    f["upm"] = torch.log(((d * up).sum(1) / up.sum(1).clamp(min=1)).clamp(min=EPS) / ((-d * dn).sum(1) / dn.sum(1).clamp(min=1)).clamp(min=EPS))
    tp = ((d[:, 1:] * d[:, :-1]) < 0) & M[:, 2:]; f["tp"] = tp.sum(1) / (M[:, 2:].sum(1).clamp(min=1)).float()
    # records: running max / min within the segment
    big = torch.where(M, S, torch.full_like(S, -1e9)); rmax = torch.cummax(big, 1).values; nrec_max = ((big[:, 1:] > rmax[:, :-1]) & M[:, 1:]).sum(1)
    sml = torch.where(M, S, torch.full_like(S, 1e9)); rmin = torch.cummin(sml, 1).values; nrec_min = ((sml[:, 1:] < rmin[:, :-1]) & M[:, 1:]).sum(1)
    f["rec"] = (nrec_max + nrec_min).float() / torch.log(nf.clamp(min=3))
    # ordinal patterns
    H = {}
    for o in (3, 4):
        U = S.unfold(1, o, 1); R = U.argsort(2).argsort(2); code = (R * (o ** torch.arange(o, device=dev))).sum(2)
        uv = M.unfold(1, o, 1).all(2); K = o ** o
        C = torch.zeros(S.shape[0], K, device=dev).scatter_add_(1, code, uv.float())
        H[o] = C
    return f, H


def block(js):
    pre, post = [], []
    for _, v, k in js:
        a = np.asarray(v[:k], np.float64); m, s = a.mean(), a.std() + 1e-9
        pre.append((a - m) / s); post.append((np.asarray(v[k:], np.float64) - m) / s)
    Sb, nb, Mb = pad(pre); Sa, na, Ma = pad(post)
    fb, Hb = seg_stats(Sb, nb, Mb); fa, Ha = seg_stats(Sa, na, Ma)
    f = {f"sx_{k}_d": fa[k] - fb[k] for k in fb}
    for o in (3, 4):
        cb, ca = Hb[o], Ha[o]; pb = cb / cb.sum(1, keepdim=True).clamp(min=1); pa = ca / ca.sum(1, keepdim=True).clamp(min=1); m = 0.5 * (pa + pb)
        kl = lambda p, q: (p * torch.log((p + EPS) / (q + EPS))).sum(1)
        f[f"sx_op{o}_js"] = 0.5 * kl(pa, m) + 0.5 * kl(pb, m)
        if o == 3:
            tot = cb + ca; e_b = tot * (cb.sum(1, keepdim=True) / tot.sum(1, keepdim=True)); e_a = tot - e_b
            chi = (((cb - e_b) ** 2) / e_b.clamp(min=1e-3) + ((ca - e_a) ** 2) / e_a.clamp(min=1e-3)) * (tot > 0)
            f["sx_op3_chi"] = chi.sum(1) / tot.sum(1).clamp(min=1)
            ent = lambda p: -(p * torch.log(p + EPS)).sum(1); f["sx_op3_ent_d"] = ent(pa) - ent(pb)
    return pd.DataFrame({k: v.float().cpu().numpy() for k, v in f.items()}, index=[j[0] for j in js])


if __name__ == "__main__":
    jobs, _ = build_jobs(name)
    out = O / "aug_sx_features.parquet" if name == "aug" else (O / "var" / f"sx_{name[4:]}.parquet" if name.startswith("var:") else O / f"sx_{name}.parquet")
    CH = int(os.environ.get("CHUNK", "2000")); parts = []
    for i in range(0, len(jobs), CH): parts.append(block(jobs[i:i + CH]))
    F = pd.concat(parts).replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32); F.index.name = "id"; F.to_parquet(out)
    print(f"{name}: {F.shape} -> {out.name} in {time.time()-t0:.0f}s", flush=True)
