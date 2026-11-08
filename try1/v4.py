"""try1: the original v4 model (0.83313 on kaggle). xgboost + lightgbm on the 977 base features, rank averaged.

python try1/v4.py predict --data data --out-dir try1_out     #load saved models, features from raw test series
python try1/v4.py train --data data --out-dir try1_out --validate
"""
import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
from time import monotonic

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MODEL = Path(__file__).resolve().parent / "model"


def feature_job(job):
    from pipeline.features_v4 import series_features
    return series_features(*job)


def worker_init():
    threadpool_limits(limits=1)


def raw_features(path, workers):
    """Rebuild all 977 original v4 features; no saved detector values are read."""
    table = pd.read_parquet(path)
    jobs = []
    for ident, group in table.groupby("id", sort=False, observed=True):
        group = group.sort_values("time")
        values, period = group.value.to_numpy(float), group.period.to_numpy()
        if not np.isfinite(values).all() or not np.all(np.diff(group.time) == 1):
            raise ValueError(f"Invalid raw observations for {ident}")
        jobs.append((ident, values[period == 0], values[period == 1]))
    start = monotonic()
    rows = []
    with ProcessPoolExecutor(workers, initializer=worker_init) as pool:
        for result in pool.map(feature_job, jobs, chunksize=8):
            rows.append(result)
            if len(rows) % 2000 == 0:
                print(f"{path.name}: {len(rows)}/{len(jobs)} series, {monotonic()-start:.0f}s", flush=True)
    return (pd.DataFrame(rows).set_index("id").replace([np.inf, -np.inf], np.nan)
            .fillna(0).astype(np.float32))


def write_csv(ids, probabilities, path, expected_ids):
    if len(ids) != 4000 or len(set(ids)) != 4000 or set(ids) != set(expected_ids):
        raise ValueError("Expected exactly the 4,000 official test IDs.")
    p = pd.Series(probabilities, index=ids).loc[expected_ids].to_numpy()
    if not np.isfinite(p).all() or not ((p >= 0)&(p <= 1)).all():
        raise ValueError("Invalid probabilities.")
    pd.DataFrame(dict(id=expected_ids, break_probability=p)).to_csv(path, index=False, lineterminator="\r\n")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def predict(folder, output, data, workers=2):
    import xgboost as xgb
    import lightgbm as lgb
    start = monotonic()
    sample = pd.read_csv(data/"sample_submission.csv", dtype={"id": str}).id
    #load the two saved models
    first = xgb.Booster(model_file=folder/"xgboost.ubj")
    first.set_param(dict(device="cpu", nthread=1))
    second = lgb.Booster(model_file=str(folder/"lightgbm.txt"))
    #977 features straight from the raw test series
    table = raw_features(data/"test_series.parquet", workers)
    if set(table.columns) != set(first.feature_names):
        raise ValueError("Extracted feature schema differs from the saved model.")
    table = table[first.feature_names]
    a = first.predict(xgb.DMatrix(table))
    b = second.predict(table, num_threads=1)
    #rank average of the two
    p = (rankdata(a)+rankdata(b))/(2*len(a))
    digest = write_csv(table.index, p, output/"submission.csv", sample)
    print(f"wrote {output/'submission.csv'} ({len(p)} rows, {monotonic()-start:.0f}s) sha256 {digest}")


def train_v4(folder, output, workers, validate, data):
    """Full raw CPU training using the original tuned tree hyperparameters."""
    import xgboost as xgb
    import lightgbm as lgb
    configuration = json.loads((folder/"hyperparameters.json").read_text())
    train = raw_features(data/"train_series.parquet", workers)
    test = raw_features(data/"test_series.parquet", workers)[train.columns]
    official = pd.read_csv(data/"train_labels.csv", dtype={"id": str}).set_index("id")
    labels = official.loc[train.index, "label"].to_numpy()
    if len(train) != 24000 or not train.index.is_unique:
        raise ValueError("Expected all 24,000 original training series.")
    train.to_parquet(output/"features_train.parquet")
    test.to_parquet(output/"features_test.parquet")
    params_x = dict(configuration["xgboost"], device="cpu")
    params_l = dict(configuration["lightgbm"], device_type="cpu")

    def fit(rows, query):
        first = xgb.train(params_x, xgb.DMatrix(train.iloc[rows], labels[rows]), configuration["xgboost_rounds"])
        second = lgb.train(params_l, lgb.Dataset(train.iloc[rows], labels[rows]), configuration["lightgbm_rounds"])
        a = first.predict(xgb.DMatrix(query))
        b = second.predict(query, num_threads=1)
        return first, second, a, b

    metrics = {}
    if validate:
        a, b = np.empty(len(train)), np.empty(len(train))
        for fold, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=configuration["seed"]).split(train, labels)):
            _, _, a[va], b[va] = fit(tr, train.iloc[va])
            print(f"Validation fold {fold+1}/5 complete", flush=True)
        metrics["local_auc"] = float(roc_auc_score(labels, rankdata(a)+rankdata(b)))
    first, second, a, b = fit(np.arange(len(train)), test)
    first.save_model(output/"xgboost.ubj")
    second.save_model(str(output/"lightgbm.txt"))
    sample = pd.read_csv(data/"sample_submission.csv", dtype={"id": str}).id
    metrics["submission_sha256"] = write_csv(test.index, (rankdata(a)+rankdata(b))/(2*len(a)), output/"submission.csv", sample)
    metrics.update(device="cpu", raw_training=True, hyperparameters=configuration)
    (output/"training_report.json").write_text(json.dumps(metrics, indent=2))


def main():
    parser = argparse.ArgumentParser(description="try1 v4 model")
    parser.add_argument("action", choices=("predict", "train"))
    parser.add_argument("--workers", type=int, default=2, choices=range(1, 9))
    parser.add_argument("--data", type=Path, default=ROOT/"data")
    parser.add_argument("--validate", action="store_true", help="five-fold validation when training")
    parser.add_argument("--out-dir", type=Path, default=ROOT/"try1_out")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.action == "predict":
        predict(MODEL, args.out_dir, args.data, args.workers)
    else:
        train_v4(MODEL, args.out_dir, args.workers, args.validate, args.data)


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        main()
