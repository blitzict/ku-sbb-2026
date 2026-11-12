"""Compare streaming extraction to fresh full-buffer feature tables, bit exactly."""
import json
import shutil
import numpy as np
import pandas as pd
from pipeline.prepare_feature_stages import code_closure
from pipeline.run_stages import ROOT, run


def main():
    work = ROOT / "cpu_088_rebuild/feature_graph_smoke"
    helper = "pipeline_helpers/stream_numpy_features.py"
    source, target = ROOT / "pipeline/legacy_cpu" / helper, work / helper
    if target.exists() and target.read_bytes() != source.read_bytes():
        raise ValueError("Do not overwrite a previous smoke's source")
    if not target.exists():
        shutil.copy2(source, target)
    outputs = [f"tune_out/streaming_check/{f}_train_hv.parquet" for f in ("clean", "grid", "grid2", "dd")]
    stage = dict(id="streaming_feature_equivalence", argv=[helper, "train_hv", "--output-dir", "tune_out/streaming_check"],
                 inputs=["tune_out/var/train_hv_series.parquet"], outputs=outputs,
                 code_inputs=code_closure(helper), env={"KU_SBB_THREADS": "2", "KU_SBB_WORKERS": "6"})
    run(work, stage)
    comparisons = []
    for file in outputs:
        old = pd.read_parquet(work / "tune_out/var" / file.rsplit("/", 1)[1])
        new = pd.read_parquet(work / file)
        pd.testing.assert_frame_equal(old, new, check_exact=True)
        comparisons.append(dict(file=file, rows=len(new), columns=len(new.columns),
                                max_absolute_difference=float(np.abs(old.to_numpy()-new.to_numpy()).max())))
    report = dict(raw_train_series=80, original_and_new_fresh=True, exact=True, comparisons=comparisons)
    (work / "stages/streaming_feature_equivalence_check.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
