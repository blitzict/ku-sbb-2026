"""Test late physical builders against the already fresh 80/16 raw smoke work."""
import json
import shutil
import numpy as np
import pandas as pd
from pipeline.run_stages import ROOT, run
from pipeline.prepare_late_feature_stages import build


def main():
    work = ROOT / "cpu_088_rebuild/feature_graph_smoke"
    if not (work / "CPU_REBUILD_WORKSPACE.json").exists():
        raise ValueError("Run the raw feature-graph smoke prerequisites first")
    # Add only new helpers; never replace code used by the running graph test.
    for name in ("late_features.py", "replay_saved_head.py"):
        target = work / "pipeline_helpers" / name
        source = ROOT / "pipeline/legacy_cpu/pipeline_helpers" / name
        if target.exists() and target.read_bytes() != source.read_bytes():
            raise ValueError(f"Helper already exists with different content: {name}")
        if not target.exists():
            shutil.copy2(source, target)
    stages = build()["stages"]
    for stage in stages:
        run(work, stage)
        for output in stage["outputs"]:
            if output.endswith(".parquet"):
                frame = pd.read_parquet(work / output)
                if not np.isfinite(frame.to_numpy()).all():
                    raise ValueError(f"Non-finite output {output}")
    report = dict(passed=len(stages), raw_train_rows=80, raw_test_rows=16, score_validation=False)
    (work / "stages/late_features_smoke.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
