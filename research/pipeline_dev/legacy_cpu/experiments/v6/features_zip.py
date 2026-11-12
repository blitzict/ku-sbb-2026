# CPU port of experiments/v6/features_zip.py, source sha256 3a56a2e734d65c47ffdc4edd3c701fb5aef8111617e0141be9fdf4251869ebb8. See auc_lab_20261107/claude/audit/cpu_port_notes.md.
import sys as _s, pathlib as _p; _s.path.insert(0, str(_p.Path(__file__).resolve().parents[2])); import ku_workspace as KU_ENV
"""Compression features (research 2026-11-03: AndrewFSee ADIA repo used gzip/bz2 compression ratios; missing family here). Prefix zp_. CPU, one process,
below-normal priority (zlib is cheap: ~30 s per 24k series). Each stream is quantised with the BEFORE side's quantile edges, so a distribution shift
moves after-side symbols into other bins. Per stream: per-symbol compressed size before / after (and their difference + log ratio), normalised
compression distance NCD(before, after), and the conditional cost of the after side given the before side relative to its own cost.
Streams: z (16 bins), z (4 bins), first difference (16 bins), |z| (8 bins), sign pattern of the difference (2 symbols).
Usage: .venv/Scripts/python features_zip.py train|test|aug|var:<stem>"""
import sys, time, zlib, pathlib, warnings, ctypes; warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
try: ctypes.windll.kernel32.SetPriorityClass(ctypes.c_void_p(-1), 0x4000)
except Exception: pass
import numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
name = sys.argv[1] if len(sys.argv) > 1 else "train"; sys.argv = sys.argv[:2]
from features_lr import build_jobs, O
t0 = time.time()
C = lambda b: len(zlib.compress(b, 9))


def sym(a, b, nb):
    e = np.quantile(a, np.linspace(0, 1, nb + 1)[1:-1]); return np.searchsorted(e, a).astype(np.uint8).tobytes(), np.searchsorted(e, b).astype(np.uint8).tobytes()


def feats(v, k):
    a, b = v[:k], v[k:]; m, s = a.mean(), a.std() + 1e-9; za, zb = (a - m) / s, (b - m) / s
    da, db = np.diff(za), np.diff(zb); f = {}
    streams = {"z16": sym(za, zb, 16), "z4": sym(za, zb, 4), "d16": sym(da, db, 16), "a8": sym(np.abs(za), np.abs(zb), 8),
               "sg": ((da > 0).astype(np.uint8).tobytes(), (db > 0).astype(np.uint8).tobytes())}
    for nm, (A, B) in streams.items():
        ca, cb, cab = C(A), C(B), C(A + B); pa, pb = ca / len(A), cb / len(B)
        f[f"zp_{nm}_pa"] = pa; f[f"zp_{nm}_pb"] = pb; f[f"zp_{nm}_d"] = pb - pa; f[f"zp_{nm}_lr"] = np.log(pb / pa)
        f[f"zp_{nm}_ncd"] = (cab - min(ca, cb)) / max(ca, cb); f[f"zp_{nm}_cond"] = (cab - ca) / cb
    return f


if __name__ == "__main__":
    jobs, _ = build_jobs(name)
    out = O / "aug_zp_features.parquet" if name == "aug" else (O / "var" / f"zp_{name[4:]}.parquet" if name.startswith("var:") else O / f"zp_{name}.parquet")
    F = pd.DataFrame([feats(v, k) for _, v, k in jobs], index=[j[0] for j in jobs]).replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    F.index.name = "id"; F.to_parquet(out); print(f"{name}: {F.shape} -> {out.name} in {time.time()-t0:.0f}s", flush=True)
