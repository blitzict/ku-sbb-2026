"""Run the five saved mixed-model neural checkpoints on RAW series, on CPU.

This rebuilds all 361 inputs, then applies the saved training-fold quantile
transforms and MLPs. It does not read cached test features or predictions.
Its output is the neural member's mean logit, not the complete 0.88078 ensemble.
The parent ensemble and its confidence-dependent correction remain separate.

Example from the ai folder:
  python try4/designated_088078/member_cpu.py --data data

Feature extraction uses at most two worker processes, one numerical thread
each. Neural inference uses two CPU threads after the workers have exited.
"""
import os

# Set before NumPy loads; Windows spawn workers inherit these limits.
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
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
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent


def measure(job):
    """Reuse the exact per-series numerical detectors used for this release."""
    sys.path.insert(0, str(ROOT / "cpu_0879"))
    from raw_features import measure as original_measure
    return original_measure(job)


def extract(raw_path, ids, columns, workers=2):
    """Compute features from observations; fail on schema/ID/detector errors."""
    if workers not in (1, 2):
        raise ValueError("Use one or two CPU workers.")
    table = pd.read_parquet(raw_path, columns=["id", "time", "period", "value"])
    if set(table.id.unique()) != set(ids) or len(set(ids)) != len(ids):
        raise ValueError("Raw series IDs and requested IDs differ or repeat.")
    jobs = []
    for ident, group in table.groupby("id", sort=False, observed=True):
        group = group.sort_values("time")
        values = group.value.to_numpy(float)
        period = group.period.to_numpy()
        cut = int((period == 0).sum())
        if (not np.isfinite(values).all() or not 0 < cut < len(group)
                or not np.all(period[:cut] == 0) or not np.all(period[cut:] == 1)
                or not np.all(np.diff(group.time) == 1)):
            raise ValueError(f"Invalid observations or boundary for {ident}")
        jobs.append((str(ident), values, cut))
    del table
    start, results = monotonic(), []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(measure, jobs, chunksize=8):
            results.append(result)
            if len(results) % 500 == 0:
                print(f"Raw features {len(results)}/{len(jobs)}, "
                      f"{monotonic()-start:.1f}s", flush=True)
    frame = pd.DataFrame([row for _, row in results],
                         index=pd.Index([ident for ident, _ in results], name="id"))
    # Conditional phase/pulse detectors intentionally omit their fields for
    # inapplicable series. The original training code filled these with zero.
    extra = set(frame.columns) - set(columns)
    if extra:
        raise ValueError(f"Unexpected detector columns: {sorted(extra)}")
    frame = frame.reindex(index=ids, columns=columns).fillna(0).astype(np.float32)
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Non-finite feature table")
    return frame


def predict(frame, model_dir):
    """Return the five-fold mean logit with CPU-only checkpoint loading."""
    import joblib
    import torch
    from torch import nn

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    votes = []
    with torch.inference_mode(), threadpool_limits(limits=2):
        for fold in range(5):
            config = json.loads((model_dir / f"fold{fold}.json").read_text())
            params = config["params"]
            width, hidden = len(frame.columns), params["hidden"]
            net = nn.Sequential(nn.Linear(width, hidden), nn.GELU(),
                nn.Dropout(params["dropout"]), nn.Linear(hidden, hidden // 2),
                nn.GELU(), nn.Dropout(params["dropout"]), nn.Linear(hidden // 2, 1))
            net.load_state_dict(torch.load(model_dir / f"fold{fold}.pt",
                                           map_location="cpu", weights_only=True))
            net.eval()
            transform = joblib.load(model_dir / f"fold{fold}_transform.joblib")
            values = transform.transform(frame.to_numpy()).astype(np.float32)
            data = torch.from_numpy(values)
            logit = torch.cat([net(chunk)[:, 0] for chunk in data.split(2048)]).numpy()
            if not np.isfinite(logit).all():
                raise ValueError(f"Fold {fold} returned invalid logits")
            votes.append(logit)
    return np.mean(votes, axis=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT.parents[1] / "data")
    parser.add_argument("--model", type=Path, default=ROOT / "model")
    parser.add_argument("--output", type=Path, default=ROOT / "out")
    parser.add_argument("--workers", type=int, choices=[1, 2], default=2)
    args = parser.parse_args()
    start = monotonic()
    ids = pd.read_csv(args.data / "sample_submission.csv", dtype={"id": str}).id
    if len(ids) != 4000 or not ids.is_unique:
        raise ValueError("Expected the 4,000 official test IDs")
    metadata = json.loads((args.model / "export_report.json").read_text())
    with threadpool_limits(limits=1):
        features = extract(args.data / "test_series.parquet", ids,
                           metadata["feature_names"], args.workers)
    logit = predict(features, args.model)
    args.output.mkdir(parents=True, exist_ok=True)
    output = args.output / "member_logits.npz"
    np.savez_compressed(output, test_ids=ids.to_numpy(dtype=str), test=logit)
    report = dict(scope="Raw-test inference for neural member ONLY; not full ensemble",
                  rows=len(ids), features=len(features.columns), device="cpu",
                  workers=args.workers, seconds=monotonic()-start,
                  input_sha256=hashlib.sha256((args.data / "test_series.parquet").read_bytes()).hexdigest(),
                  output_sha256=hashlib.sha256(output.read_bytes()).hexdigest())
    (args.output / "run.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
