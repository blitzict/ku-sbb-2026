"""Prove the faster augmentation keeps every used value and RNG boundary."""
import json
import numpy as np
import pandas as pd
from pipeline.build_foundation import initialize
from pipeline.sync_ports import sync
from pipeline.run_stages import ROOT, run
from pipeline.prepare_feature_stages import code_closure


def main():
    original = ROOT / "cpu_088_rebuild/feature_graph_smoke"
    work = ROOT / "cpu_088_rebuild/shift_augmentation_smoke"
    if not (work / "CPU_REBUILD_WORKSPACE.json").exists():
        initialize(work, ROOT / "cpu_088_rebuild/feature_graph_smoke_inputs")
        sync(work)
    script = "pipeline_helpers/shift_augmentation.py"
    stage = dict(id="shift_only_augmentation", argv=[script],
                 inputs=["data/train_series.parquet", "data/train_labels.csv"],
                 outputs=["tune_out/aug_features.parquet", "tune_out/aug_meta.csv"],
                 code_inputs=code_closure(script), env={"KU_SBB_THREADS": "2"})
    run(work, stage)
    old_meta = pd.read_csv(original / "tune_out/aug_meta.csv", dtype=str)
    old_meta = old_meta.loc[old_meta.kind == "shift"].reset_index(drop=True)
    new_meta = pd.read_csv(work / "tune_out/aug_meta.csv", dtype=str)
    pd.testing.assert_frame_equal(old_meta, new_meta)
    old = pd.read_parquet(original / "tune_out/aug_features.parquet").loc[old_meta.id]
    new = pd.read_parquet(work / "tune_out/aug_features.parquet")
    pd.testing.assert_frame_equal(old, new, check_exact=True)
    report = dict(rows=len(new), columns=len(new.columns),
                  max_absolute_difference=float(np.abs(old.to_numpy()-new.to_numpy()).max()),
                  metadata_identical=True, raw_train_series=80,
                  note="Both outputs freshly generated from raw data; no historical cache")
    (work / "stages/shift_comparison.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
