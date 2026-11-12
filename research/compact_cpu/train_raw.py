"""Train a CPU break detector from the four original KU-SBB competition files.

From this project run: python cpu_0879/train_raw.py --workers 4

1. Split each series at its supplied 0-to-1 period transition.
2. Rebuild the detectors in raw_features.py using observations from that series.
3. Fit a LightGBM classifier in each of five fixed validation folds. Print ROC
   AUC over the pooled held-out predictions, then fit once on all labels.
4. Save a test submission in sample order, the full-data tree model (with
   feature names), and one validation prediction and fold number per source.

This is an independent raw-data training path. The exact ensemble output-layer
refit is train.py. This script needs the project's detector *code* under
experiments/weight_space, and opens no precomputed detector files or weights.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory
from time import monotonic

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

from raw_features import measure

HERE = Path(__file__).resolve().parent


def series_jobs(path, expected_ids):
    """Check ID coverage and chronology. Yield values in their original units."""
    frame = pq.read_table(path, columns=["id", "time", "value", "period"]).to_pandas(categories=["id"])
    if frame.isna().any().any() or set(frame.id.unique()) != set(expected_ids):
        raise ValueError(f"Missing observations or mismatched IDs in {path}")
    for ident, group in frame.groupby("id", sort=False, observed=True):
        if not group.time.is_monotonic_increasing:
            group = group.sort_values("time")
        t = group.time.to_numpy()
        period = group.period.to_numpy()
        x = group.value.to_numpy(dtype=float)
        cut = int(np.count_nonzero(period == 0))
        if (t[0] != 0 or not np.all(np.diff(t) == 1) or not np.isfinite(x).all()
                or not np.all(period[:cut] == 0) or not np.all(period[cut:] == 1)
                or min(cut, len(x)-cut) < 32 or max(cut, len(x)-cut) > 1024):
            raise ValueError(f"Unsupported time grid, values or segment lengths in {ident}")
        yield ident, x, cut


def extract(executor, path, ids):
    """All extraction is label-free. Preserve schema across conditional detectors."""
    started = monotonic()
    jobs = series_jobs(path, ids)
    results = executor.map(measure, jobs, chunksize=32) if executor else map(measure, jobs)
    by_id = {}
    for ident, values in results:
        by_id[ident] = values
        if len(by_id) % 2000 == 0:
            print(f"{path.stem}: {len(by_id):,}/{len(ids):,} series ({monotonic()-started:.0f}s)", flush=True)
    # A branch such as phase has zero evidence when its detector did not fire.
    return pd.DataFrame.from_dict(by_id, orient="index").reindex(ids).fillna(0).astype(np.float32)


def classifier(workers):
    """Fixed settings for every fold and the full-data model, with CPU histograms."""
    return LGBMClassifier(objective="binary", device_type="cpu", n_jobs=workers,
                          n_estimators=720, learning_rate=.025, num_leaves=27,
                          min_child_samples=58, reg_lambda=6, colsample_bytree=.62,
                          max_bin=127, random_state=7, deterministic=True,
                          force_col_wise=True, verbosity=-1)


def run(data, out_dir, workers):
    started = monotonic()
    labels = pd.read_csv(data / "train_labels.csv", dtype={"id": str}).set_index("id")
    sample = pd.read_csv(data / "sample_submission.csv", dtype={"id": str}).set_index("id")
    if (not labels.index.is_unique or not sample.index.is_unique or labels.index.hasnans
            or sample.index.hasnans or len(labels.index.intersection(sample.index))
            or list(labels.columns) != ["label"] or set(labels.label.unique()) != {0, 1}
            or list(sample.columns) != ["break_probability"]):
        raise ValueError("Expected distinct train/test IDs, binary labels and the KU-SBB sample schema.")
    print(f"Rebuilding detectors on CPU ({workers} workers).", flush=True)
    def read(executor):
        return (extract(executor, data / "train_series.parquet", labels.index),
                extract(executor, data / "test_series.parquet", sample.index))

    if workers == 1:
        with threadpool_limits(limits=1):
            train, test = read(None)
    else:
        # Each process handles independent series. Keep native linear algebra
        # single-threaded inside each worker to avoid nested oversubscription.
        with ProcessPoolExecutor(workers, initializer=worker_init) as executor:
            train, test = read(executor)
    train = train.sort_index(axis=1)
    test = test.reindex(columns=train.columns, fill_value=0)
    print(f"Rebuilt {train.shape[1]} detectors/features per series. Fitting trees.", flush=True)

    y = labels.label.to_numpy()
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
    oof = np.full(len(y), np.nan)
    fold_id = np.full(len(y), -1)
    for k, (fit, valid) in enumerate(folds.split(train, y)):
        model = classifier(workers).fit(train.iloc[fit], y[fit])
        oof[valid] = model.predict_proba(train.iloc[valid])[:, 1]
        fold_id[valid] = k
        print(f"Fold {k+1}/5  AUC {roc_auc_score(y[valid], oof[valid]):.8f}", flush=True)
    auc = roc_auc_score(y, oof)
    print(f"Raw-only pooled out-of-fold AUC  {auc:.8f}", flush=True)
    model = classifier(workers).fit(train, y)
    sample["break_probability"] = model.predict_proba(test)[:, 1]
    out_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=out_dir) as tmp:
        stage = Path(tmp)
        sample.to_csv(stage / "submission.csv")
        model.booster_.save_model(str(stage / "model.txt"))
        validation = pd.DataFrame({"label": y, "fold": fold_id, "break_probability": oof}, index=labels.index)
        validation.to_csv(stage / "validation.csv")
        for name in ("model.txt", "validation.csv", "submission.csv"):
            (stage / name).replace(out_dir / name)
    print(f"Wrote {len(sample):,} predictions to {out_dir / 'submission.csv'} ({monotonic()-started:.0f}s).")


def worker_init():
    threadpool_limits(limits=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=HERE.parent / "data", help="The four competition files")
    parser.add_argument("--out-dir", type=Path, default=HERE / "raw_run", help="Model and CSV output directory")
    parser.add_argument("--workers", type=int, default=4, help="CPU processes for detection and threads for training")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    run(args.data, args.out_dir, args.workers)
