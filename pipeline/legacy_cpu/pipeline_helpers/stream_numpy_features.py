"""Compute the original feature functions with a bounded output-row buffer.

The half-view training set has about 85,000 rows. Holding all their 977-field
Python dictionaries until the end exceeds the 8 GB target. Ordered joblib
results and 256-row parquet groups preserve IDs, values, and column ordering.
An unfinished parquet is never exposed as a completed feature file.
"""
import argparse
from pathlib import Path
import sys
from time import monotonic
import warnings

import ku_workspace as env


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stem")
    p.add_argument("families", nargs="?", default="clean,grid,grid2,dd")
    p.add_argument("--output-dir", default="tune_out/var", help="Relative workspace path; alternate folder is for verification")
    args = p.parse_args()
    root = env.root()
    target = (root / args.output_dir).resolve()
    if not target.is_relative_to(root) or target == root:
        raise ValueError("Output must stay inside the marked workspace")
    target.mkdir(parents=True, exist_ok=True)
    sys.path[:0] = [str(root / "clean"), str(root / "experiments/v5"), str(root / "experiments/v6")]
    import features
    import features_grid
    import features_grid2
    import features_dd
    import numpy as np
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq
    from joblib import Parallel, delayed

    functions = dict(clean=features.series_features, grid=features_grid.one,
                     grid2=features_grid2.one, dd=features_dd.one)
    families = args.families.split(",")
    if not families or set(families) - set(functions):
        raise ValueError(f"Unknown feature families: {families}")
    env.low_priority()
    warnings.filterwarnings("ignore")
    raw = env.read_series(root / f"tune_out/var/{args.stem}_series.parquet")
    jobs = []
    for ident, frame in raw.groupby("id", sort=False, observed=True):
        values = frame.value.to_numpy(float)
        period = frame.period.to_numpy()
        jobs.append((ident, values[period == 0], values[period == 1]))
    del frame, raw
    if not jobs:
        raise ValueError("No source series found")

    for family in families:
        out = target / f"{family}_{args.stem}.parquet"
        if out.exists():
            print(f"{family} {args.stem}: exists, skipped", flush=True)
            continue
        temporary = out.with_suffix(".partial.parquet")
        started = monotonic()
        writer, count, pending = None, 0, []
        try:
            rows = Parallel(n_jobs=env.jobs(), batch_size=32, return_as="generator")(
                delayed(functions[family])(*job) for job in jobs)
            for row in rows:
                pending.append(row)
                if len(pending) < 256 and count + len(pending) < len(jobs):
                    continue
                frame = (pd.DataFrame(pending).set_index("id").replace([np.inf, -np.inf], np.nan)
                         .fillna(0).astype(np.float32))
                expected = [job[0] for job in jobs[count:count + len(frame)]]
                if list(frame.index) != expected:
                    raise ValueError("Feature extraction changed source order")
                table = pa.Table.from_pandas(frame, preserve_index=True)
                if writer is None:
                    writer = pq.ParquetWriter(temporary, table.schema, compression="snappy")
                writer.write_table(table)
                count += len(frame)
                pending.clear()
                if count % 5120 == 0 or count == len(jobs):
                    print(f"{family} {args.stem}: {count}/{len(jobs)}, {monotonic()-started:.1f}s", flush=True)
            if count != len(jobs) or pending:
                raise ValueError("Incomplete feature extraction")
        finally:
            if writer is not None:
                writer.close()
        temporary.replace(out)
        print(f"Wrote {out.name}: {count} rows, {len(frame.columns)} columns", flush=True)


if __name__ == "__main__":
    main()
