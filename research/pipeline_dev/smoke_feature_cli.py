"""Exercise actual feature-builder CLIs using 16 raw series per split.

Creates a separate explicitly marked smoke workspace. It never launches these
commands against the full training inputs. The output is a runtime check, not
an AUC estimate.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from pipeline.build_foundation import ROOT, initialize, build
from pipeline.prepare_feature_stages import build as stages
from pipeline.run_stages import run
from pipeline.sync_ports import sync


def main():
    work = ROOT / "cpu_088_rebuild/cli_smoke"
    data = ROOT / "cpu_088_rebuild/cli_smoke_inputs"
    data.mkdir(exist_ok=True)
    for split in ("train", "test"):
        with np.load(ROOT / f"cpu_088_rebuild/smoke/tune_out/seq_cache_{split}.npz") as a:
            ids = a["ids"].tolist()
        table = pq.read_table(ROOT / f"data/{split}_series.parquet", filters=[("id", "in", ids)])
        pq.write_table(table, data / f"{split}_series.parquet")
        name = "train_labels.csv" if split == "train" else "sample_submission.csv"
        frame = pd.read_csv(ROOT / "data" / name, dtype={"id": str})
        frame[frame.id.isin(ids)].to_csv(data / name, index=False)
    initialize(work, data)
    sync(work)
    for split in ("train", "test"):
        build(work, split, 2, 16)
    # Exclude global generator diagnostics (too few rows) and augmentation
    # stages for this first CLI check. They need their own smoke below.
    chosen = [s for s in stages()["stages"] if s["group"] == "base"
              and len(s["argv"]) > 1 and s["argv"][1] in ("train", "test")]
    for stage in chosen:
        run(work, stage)
        for output in stage["outputs"]:
            frame = pd.read_parquet(work / output)
            if len(frame) != 16 or not np.isfinite(frame.to_numpy()).all():
                raise AssertionError(f"Invalid feature output: {output}")
    (work / "stages/cli_smoke.json").write_text(json.dumps(dict(
        passed=len(chosen), rows_per_split=16, score_validation=False), indent=2))
    print(f"PASS: {len(chosen)} actual CPU feature commands on fresh raw subsets")


if __name__ == "__main__":
    main()
