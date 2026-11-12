# CPU port of experiments/v5/augment.py, source sha256 66da72927fa6f8ed13d2d96d4a3c75d520282708fa666b4814cad91b69c009ac. See auc_lab_20261107/claude/audit/cpu_port_notes.md.
import sys as _s, pathlib as _p; _s.path.insert(0, str(_p.Path(__file__).resolve().parents[2])); import ku_workspace as KU_ENV
"""Augmented NEGATIVE examples built from the competition training series only.

kinds:
  shift : label-0 series, boundary moved to a random 30-70% position, full length kept   -> label 0
  pre   : any series, its BEFORE part alone, split at a random 40-60% point               -> label 0
  post  : any series, its AFTER part alone, split at a random 40-60% point                -> label 0
Features via clean/features.py (same 977 columns).  Output: aug_features.parquet + aug_meta.csv in tune_out.
"""
import sys, time, pathlib, ctypes
KU_ENV.low_priority()
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(KU_ENV.code_root() / "clean"))
import numpy as np, pandas as pd
from joblib import Parallel, delayed
import features as CF
D = KU_ENV.root()
MIN_SIDE = 78                      # the real data's smallest pre segment
rng = np.random.default_rng(1)
t0 = time.time()
s = KU_ENV.read_series(D / "data" / "train_series.parquet")
y = pd.read_csv(D / "data" / "train_labels.csv", dtype={"id": str}).set_index("id").label
jobs, meta = [], []
for ident, g in s.groupby("id", sort=False, observed=True):
    v = g["value"].to_numpy(float); p = g["period"].to_numpy(); a, b = v[p == 0], v[p == 1]
    if y[ident] == 0:                                   # full-length negative with a moved boundary
        k = int(len(v) * rng.uniform(0.3, 0.7))
        if k >= MIN_SIDE and len(v) - k >= MIN_SIDE:
            jobs.append((f"{ident}s", v[:k], v[k:])); meta.append((f"{ident}s", ident, "shift"))
    for seg, kind in ((a, "pre"), (b, "post")):         # within-segment negatives (same generator as the parent)
        if len(seg) >= 2 * MIN_SIDE:
            k = int(len(seg) * rng.uniform(0.4, 0.6)); k = min(max(k, MIN_SIDE), len(seg) - MIN_SIDE)
            jobs.append((f"{ident}{kind[1]}", seg[:k], seg[k:])); meta.append((f"{ident}{kind[1]}", ident, kind))
print(f"{len(jobs)} augmented series from {len(y)} originals ({time.time()-t0:.0f}s to build)", flush=True)
rows = Parallel(n_jobs=KU_ENV.jobs(), batch_size=64)(delayed(CF.series_features)(i, a, b) for i, a, b in jobs)
F = pd.DataFrame(rows).set_index("id").replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
F.to_parquet(D / "tune_out" / "aug_features.parquet")
pd.DataFrame(meta, columns=["id", "source", "kind"]).to_csv(D / "tune_out" / "aug_meta.csv", index=False)
print(f"wrote {F.shape} in {time.time()-t0:.0f}s")
