# CPU port of experiments/v5/aug_grid.py, source sha256 31c83267c606a07281cfe29c2471e64928b94a029eefbb095d375ea2daddf2fa. See auc_lab_20261107/claude/audit/cpu_port_notes.md.
import sys as _s, pathlib as _p; _s.path.insert(0, str(_p.Path(__file__).resolve().parents[2])); import ku_workspace as KU_ENV
"""Grid / grid2 features for the round-1 'shift' augmented series (rebuilt deterministically, same rng as augment.py),
so the augmentation can be combined with the grid blocks.  -> tune_out/aug_grid_features.parquet, aug_grid2_features.parquet"""
import sys, time, pathlib, ctypes, warnings
KU_ENV.low_priority()
sys.stdout.reconfigure(encoding="utf-8"); warnings.filterwarnings("ignore")
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from joblib import Parallel, delayed
import features_grid as G1, features_grid2 as G2
D = KU_ENV.root()
MIN_SIDE = 78
rng = np.random.default_rng(1); t0 = time.time()
s = KU_ENV.read_series(D / "data" / "train_series.parquet")
y = pd.read_csv(D / "data" / "train_labels.csv", dtype={"id": str}).set_index("id").label
jobs = []
for ident, g in s.groupby("id", sort=False, observed=True):                 # identical loop and rng draws to augment.py
    v = g["value"].to_numpy(float); p = g["period"].to_numpy(); a, b = v[p == 0], v[p == 1]
    if y[ident] == 0:
        k = int(len(v) * rng.uniform(0.3, 0.7))
        if k >= MIN_SIDE and len(v) - k >= MIN_SIDE:
            jobs.append((f"{ident}s", v[:k], v[k:]))
    for seg in (a, b):
        if len(seg) >= 2 * MIN_SIDE:
            rng.uniform(0.4, 0.6)                             # consume the same draws as augment.py (halves not needed)
meta = pd.read_csv(D / "tune_out/aug_meta.csv", dtype=str); want = set(meta.id[meta.kind == "shift"])
jobs = [j for j in jobs if j[0] in want]
assert len(jobs) == len(want), (len(jobs), len(want))
print(f"{len(jobs)} shift series rebuilt in {time.time()-t0:.0f}s", flush=True)
for mod, name in ((G1, "grid"), (G2, "grid2")):
    rows = Parallel(n_jobs=KU_ENV.jobs(), batch_size=64)(delayed(mod.one)(*j) for j in jobs)
    F = pd.DataFrame(rows).set_index("id").replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    F.to_parquet(D / "tune_out" / f"aug_{name}_features.parquet"); print(f"aug_{name}: {F.shape} ({time.time()-t0:.0f}s)", flush=True)
