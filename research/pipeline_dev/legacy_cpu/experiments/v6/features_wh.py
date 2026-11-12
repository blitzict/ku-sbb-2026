# CPU port of experiments/v6/features_wh.py, source sha256 51e06a1dbfbdae17a2e8ccea58d242ed79649d8a194a7b127fda28daef331acb. See auc_lab_20261107/claude/audit/cpu_port_notes.md.
import sys as _s, pathlib as _p; _s.path.insert(0, str(_p.Path(__file__).resolve().parents[2])); import ku_workspace as KU_ENV
"""Whole-series / raw-scale block (2026-11-03, from ADIA structural-break write-ups: 7th place's top feature was whole-series dispersion of |x|). Prefix wh_.
Raw (unstandardised) scale statistics of the whole series and of each segment, plus segment/whole ratios and share-of-total encodings.
CPU numpy, seconds. Usage: python features_wh.py train|test|aug|var:<stem>"""
import sys, time, pathlib, warnings, ctypes; warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
try: ctypes.windll.kernel32.SetPriorityClass(ctypes.c_void_p(-1), 0x4000)
except Exception: pass
import numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
name = sys.argv[1] if len(sys.argv) > 1 else "train"; sys.argv = sys.argv[:2]
from features_lr import build_jobs, O
t0 = time.time(); E = 1e-12


def st(x):
    a = np.abs(x); m, s = x.mean(), x.std(); am, asd = a.mean(), a.std()
    c = np.cumsum(x - m); return {"std": s, "mean": m, "amean": am, "aidx": asd ** 2 / (am + E), "acv": asd / (am + E), "cv": s / (abs(m) + E),
                                  "kurt": ((x - m) ** 4).mean() / (s ** 4 + E), "q99": np.quantile(a, 0.99), "mad": np.median(np.abs(x - np.median(x))),
                                  "crng": (c.max() - c.min()) / (s * np.sqrt(len(x)) + E), "n": len(x)}


def feats(v, k):
    v = np.asarray(v, np.float64); W, B, A = st(v), st(v[:k]), st(v[k:]); f = {}
    for key in ("std", "amean", "aidx", "acv", "kurt", "q99", "mad", "crng"):
        f[f"wh_w_{key}"] = W[key]
        f[f"wh_b_{key}"], f[f"wh_a_{key}"] = B[key], A[key]
        f[f"wh_lr_{key}"] = np.log((A[key] + E) / (B[key] + E))
        f[f"wh_share_{key}"] = A[key] / (A[key] + B[key] + E)
        f[f"wh_aw_{key}"] = np.log((A[key] + E) / (W[key] + E)); f[f"wh_bw_{key}"] = np.log((B[key] + E) / (W[key] + E))
    f["wh_w_logcv"] = np.log(W["cv"] + E); f["wh_w_mean"] = W["mean"]; f["wh_d_mean_raw"] = A["mean"] - B["mean"]
    f["wh_frac_after"] = A["n"] / W["n"]; f["wh_n"] = W["n"]
    return f


if __name__ == "__main__":
    jobs, _ = build_jobs(name)
    out = O / "aug_wh_features.parquet" if name == "aug" else (O / "var" / f"wh_{name[4:]}.parquet" if name.startswith("var:") else O / f"wh_{name}.parquet")
    F = pd.DataFrame([feats(v, k) for _, v, k in jobs], index=[j[0] for j in jobs])
    F = F.replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32); F.index.name = "id"; F.to_parquet(out)
    print(f"{name}: {F.shape} -> {out.name} in {time.time()-t0:.0f}s", flush=True)
