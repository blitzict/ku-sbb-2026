"""Exercise retained specialist feature builders on the fresh 80/16 subset."""
import argparse
import json
import shutil
import numpy as np
import pandas as pd
from pipeline.prepare_white_feature_stages import build
from pipeline.run_stages import ROOT, run


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--views", action="store_true")
    a = p.parse_args()
    work = ROOT / "cpu_088_rebuild/feature_graph_smoke"
    helper = "pipeline_helpers/white_model_features.py"
    src, dst = ROOT / "pipeline/legacy_cpu" / helper, work / helper
    if dst.exists() and dst.read_bytes() != src.read_bytes():
        raise ValueError("Never overwrite code in the active feature smoke")
    if not dst.exists():
        shutil.copy2(src, dst)
    selected = [s for s in build()["stages"] if ("white_feature_extra_" in s["id"]) == a.views]
    for s in selected:
        run(work, s)
        for file in s["outputs"]:
            if file.endswith(".parquet"):
                frame = pd.read_parquet(work / file)
                if not np.isfinite(frame.to_numpy()).all():
                    raise ValueError(f"Non-finite {file}")
    report = dict(passed=len(selected), views=a.views, raw_train_series=80,
                  raw_test_series=16, score_validation=False)
    name = "white_view_features_smoke" if a.views else "white_features_smoke"
    (work / "stages" / (name + ".json")).write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
