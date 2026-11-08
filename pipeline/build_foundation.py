"""Start a fresh CPU rebuild using only raw data and fixed historical settings.

Creates an isolated workspace, copies CPU code and the four input files, then
recomputes the 977-feature base and raw sequence arrays. No old prediction or
feature archive is copied. Each completed split is fingerprinted for resuming.
This is the first stage of reconstruction, not a complete 0.88 model trainer.
"""
import os
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "1"

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
from time import monotonic

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def process(job):
    from pipeline.features_v4 import series_features
    ident, values, boundary = job
    return series_features(ident, values[:boundary], values[boundary:])


def initialize(work, data):
    work = work.resolve()
    if work == ROOT or work == data.resolve() or ROOT.is_relative_to(work):
        raise ValueError("Use a new child workspace, never the original project or raw data directory")
    marker = work/"CPU_REBUILD_WORKSPACE.json"
    if work.exists() and any(work.iterdir()) and not marker.is_file():
        raise ValueError("Refusing to mix a rebuild with an unmarked existing directory")
    work.mkdir(parents=True, exist_ok=True)
    for folder in ("data", "clean", "tune_out/weight_space", "weights", "new weights", "logs", "stages"):
        (work/folder).mkdir(parents=True, exist_ok=True)
    required = ("train_series.parquet", "test_series.parquet", "train_labels.csv", "sample_submission.csv")
    inputs = {}
    for filename in required:
        source = data/filename
        digest = sha(source)
        target = work/"data"/filename
        if target.exists() and sha(target) != digest:
            raise ValueError(f"Raw input changed in existing workspace: {filename}")
        if not target.exists():
            shutil.copy2(source, target)
        inputs[filename] = digest
    payload = dict(kind="fresh_cpu_rebuild", raw_input_sha256=inputs,
                   original_prediction_archives_used=False,
                   note="Only original raw files and fixed model/feature settings are imported")
    marker.write_text(json.dumps(payload, indent=2))
    # Copy source, never historical outputs. Later source ports can be refreshed
    # independently; stage fingerprints prevent an obsolete feature cache reuse.
    code = ROOT/"pipeline/legacy_cpu/experiments"
    for source in code.rglob("*.py"):
        relative = source.relative_to(code)
        target = work/"experiments"/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or sha(source) != sha(target):
            shutil.copy2(source, target)
    shutil.copy2(ROOT/"pipeline/features_v4.py", work/"clean/features.py")
    # These are hyperparameters and feature-name lists, not row-level data.
    settings = ["selected_ns3.csv", "selected_ns3_lm.csv", "var/all_cols.csv",
                "var/cols_cggd.csv", "redesign/cols_ns3_harm.csv", "best_xgb5.json"]
    settings += [str(p.relative_to(ROOT/"tune_out")) for p in (ROOT/"tune_out/loop").glob("fam_*.csv")]
    imported = []
    for relative in settings:
        source = ROOT/"tune_out"/relative
        if source.exists():
            target = work/"tune_out"/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            imported.append(dict(path=relative.replace("\\", "/"), sha256=sha(source)))
    (work/"stages/fixed_settings.json").write_text(json.dumps(imported, indent=2))
    return work


def build(work, split, workers, limit):
    raw = work/f"data/{split}_series.parquet"
    result = work/f"clean/features_{split}.parquet"
    sequences = work/f"tune_out/seq_cache_{split}.npz"
    record = work/f"stages/foundation_{split}.json"
    fingerprint = dict(raw_sha256=sha(raw), code_sha256=sha(ROOT/"pipeline/features_v4.py"),
                       limit=limit, builder_sha256=sha(Path(__file__)))
    if record.exists() and result.exists() and sequences.exists():
        previous = json.loads(record.read_text())
        if previous.get("input") == fingerprint and all(
                sha(work/path) == digest for path, digest in previous["outputs"].items()):
            print(f"{split}: verified completed stage; reusing its fresh outputs", flush=True)
            return
        raise ValueError(f"Changed foundation stage {split}; use a new workspace to avoid mixing runs")
    started = monotonic()
    frame = pq.read_table(raw, columns=["id", "time", "period", "value"]).to_pandas(categories=["id"])
    jobs = []
    for ident, group in frame.groupby("id", sort=False, observed=True):
        group = group.sort_values("time")
        values, period = group.value.to_numpy(float), group.period.to_numpy()
        boundary = int((period == 0).sum())
        if (not np.isfinite(values).all() or not 0 < boundary < len(values)
                or not np.all(period[:boundary] == 0) or not np.all(period[boundary:] == 1)
                or group.time.iloc[0] != 0 or not np.all(np.diff(group.time) == 1)):
            raise ValueError(f"Invalid raw series {ident}")
        jobs.append((str(ident), values, boundary))
        if limit and len(jobs) == limit:
            break
    del frame
    expected = 24000 if split == "train" else 4000
    if not limit and len(jobs) != expected:
        raise ValueError(f"Expected {expected} {split} series, got {len(jobs)}")
    ids = np.array([j[0] for j in jobs], dtype=str)
    length = np.array([min(len(j[1]), 1024) for j in jobs], dtype=np.int64)
    boundary = np.array([j[2] for j in jobs], dtype=np.int64)
    if (boundary >= length).any():
        raise ValueError("Sequence truncation would remove the complete after period")
    padded = np.zeros((len(ids), 1024), np.float32)
    for row, (_, values, _) in enumerate(jobs):
        padded[row, :length[row]] = values[:length[row]]
    np.savez(sequences, ids=ids, x=padded, n=length, b=boundary)
    del padded
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for row in pool.map(process, jobs, chunksize=8):
            rows.append(row)
            if len(rows) % 500 == 0:
                print(f"Foundation {split}: {len(rows)}/{len(jobs)}, {monotonic()-started:.0f}s", flush=True)
    features = (pd.DataFrame(rows).set_index("id").replace([np.inf, -np.inf], np.nan)
                .fillna(0).astype(np.float32))
    if not np.array_equal(ids, features.index) or len(features.columns) != 977:
        raise ValueError("Unexpected foundation feature shape or row order")
    features.to_parquet(result)
    report = dict(input=fingerprint, rows=len(features), columns=len(features.columns),
                  seconds=monotonic()-started, raw_data_only=True,
                  outputs={str(path.relative_to(work)).replace("\\", "/"): sha(path)
                           for path in (result, sequences)})
    record.write_text(json.dumps(report, indent=2))
    print(f"Foundation {split} complete: {report['seconds']:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, default=ROOT/"cpu_088_rebuild/work")
    parser.add_argument("--data", type=Path, default=ROOT/"data")
    parser.add_argument("--workers", type=int, choices=[1, 2], default=2)
    parser.add_argument("--limit", type=int, default=0, help="Smoke tests only, use a separate workspace")
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("limit must be non-negative")
    work = initialize(args.work, args.data)
    with threadpool_limits(limits=1):
        for split in ("train", "test"):
            build(work, split, args.workers, args.limit)


if __name__ == "__main__":
    main()
