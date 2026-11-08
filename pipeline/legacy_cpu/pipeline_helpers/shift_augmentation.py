"""Build only the shifted-boundary negatives consumed by the winning models.

The original augment.py also built pre-only/post-only rows that every retained
loader discarded. We consume their RNG draws, preserving the exact subsequent
shift boundaries, but avoid computing their unused 977-feature vectors.
"""
import sys
from time import monotonic
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
import ku_workspace as env

root = env.root()
sys.path.insert(0, str(root / "clean"))
import features

env.low_priority()
start = monotonic()
raw = env.read_series(root / "data/train_series.parquet")
labels = pd.read_csv(root / "data/train_labels.csv", dtype={"id": str}).set_index("id").label
rng = np.random.default_rng(1)
jobs, metadata = [], []
for ident, frame in raw.groupby("id", sort=False, observed=True):
    values, period = frame.value.to_numpy(float), frame.period.to_numpy()
    before, after = values[period == 0], values[period == 1]
    if labels[ident] == 0:
        cut = int(len(values) * rng.uniform(.3, .7))
        if cut >= 78 and len(values) - cut >= 78:
            jobs.append((f"{ident}s", values[:cut], values[cut:]))
            metadata.append((f"{ident}s", str(ident), "shift"))
    for segment in (before, after):
        if len(segment) >= 156:
            rng.uniform(.4, .6)
del raw
print(f"Computing {len(jobs)} used shifted-boundary negatives", flush=True)
rows = Parallel(n_jobs=env.jobs(), batch_size=64)(delayed(features.series_features)(*job) for job in jobs)
result = pd.DataFrame(rows).set_index("id").replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
result.to_parquet(root / "tune_out/aug_features.parquet")
pd.DataFrame(metadata, columns=["id", "source", "kind"]).to_csv(root / "tune_out/aug_meta.csv", index=False)
print(f"Wrote {result.shape}, {monotonic()-start:.1f}s", flush=True)
