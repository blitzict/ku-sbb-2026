"""Execute the complete CPU feature dependency graph on small raw subsets.

80 training series keep the ten-fold prior builder meaningful as a runtime
check. No AUC from these rows is a performance estimate. Historical feature
tables and row predictions are neither loaded nor copied.
"""
import json

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from pipeline.build_foundation import ROOT, initialize, build
from pipeline.sync_ports import sync
from pipeline.prepare_rebuild_stages import build as stages
from pipeline.run_stages import run


def main():
    data = ROOT / "cpu_088_rebuild/feature_graph_smoke_inputs"
    work = ROOT / "cpu_088_rebuild/feature_graph_smoke"
    data.mkdir(exist_ok=True)
    for split, count in (("train", 80), ("test", 16)):
        target = data / f"{split}_series.parquet"
        if not target.is_file():
            ids = pq.read_table(ROOT / f"data/{split}_series.parquet", columns=["id"]).column("id").unique().slice(0, count).to_pylist()
            table = pq.read_table(ROOT / f"data/{split}_series.parquet", filters=[("id", "in", ids)])
            pq.write_table(table, target)
            name = "train_labels.csv" if split == "train" else "sample_submission.csv"
            frame = pd.read_csv(ROOT / "data" / name, dtype={"id": str})
            frame[frame.id.isin(ids)].to_csv(data / name, index=False)
    if not (work / "CPU_REBUILD_WORKSPACE.json").is_file():
        initialize(work, data)
    sync(work)
    for part, count in (("train", 80), ("test", 16)):
        build(work, part, 2, count)
    manifest = stages()
    (work / "stages/requested_feature_graph.json").write_text(json.dumps(manifest, indent=2))
    for stage in manifest["stages"]:
        run(work, stage)
        for output in stage["outputs"]:
            if output.endswith(".parquet") and "_series" not in output:
                frame = pd.read_parquet(work / output)
                numeric = frame.select_dtypes(include="number")
                if not np.isfinite(numeric).all().all():
                    raise AssertionError(f"Non-finite numeric feature: {output}")
    report = dict(passed=len(manifest["stages"]), raw_train_rows=80, raw_test_rows=16,
                  score_validation=False, historical_row_predictions_used=False)
    (work / "stages/feature_graph_smoke.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
