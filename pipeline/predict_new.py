"""run the saved 0.879 ensemble on new raw series, no training.

python -m pipeline.predict_new --models cpu_run --series new_series.parquet --out predict_run --csv submission.csv

--models is the finished training folder from train_087910_cpu.py (it holds every fitted model)
--out is a fresh folder, the trained folder is never written to
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "pipeline/full_cpu_stages.json"
FINAL = "new weights/wn_local_rank_shared_residual_response_candidate/submission.csv"


def stages():
    return json.loads(MANIFEST.read_text())["stages"]


def setup(models, out):
    #copy the frozen training source (not data, not models) into the prediction folder
    out.mkdir(parents=True, exist_ok=True)
    for p in models.rglob("*"):
        rel = p.relative_to(models)
        if rel.parts[0] in ("stages", "logs", "weights", "weights_experiments", "data"):
            continue
        if p.is_file() and (p.suffix == ".py" or (p.suffix == ".csv" and rel.parts[0] == "tune_out" and len(rel.parts) == 2)):
            t = out / rel
            t.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(p, t)
    (out / "CPU_REBUILD_WORKSPACE.json").write_text(json.dumps({"mode": "prediction", "models": str(models)}))


def one_stage(stage_id, models, out):
    #runs inside a fresh python so every stage gets clean imports, same as training
    from pipeline.inference import base_models, common, query_features, white_trees
    stage = {s["id"]: s for s in stages()}[stage_id]
    common.bind(out, models, stage["argv"][0])
    if query_features.mode(stage) is not None:
        query_features.predict_stage(stage, models, out)      #features for the new series
    elif white_trees.supported(stage):
        white_trees.predict_stage(stage, models, out)         #saved white/mech tree specialists
    elif base_models.supported(stage):
        base_models.predict_stage(stage, models, out)         #saved xgb/lgb/cat/mlp/seq models + fixed blends
    else:
        raise NotImplementedError(f"{stage_id}: no saved-model path for {stage['argv'][0]}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", type=Path, required=True)
    p.add_argument("--series", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--csv", type=Path, default=Path("submission.csv"))
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--stage")
    a = p.parse_args()
    models, out = a.models.resolve(), a.out.resolve()
    if a.stage:
        return one_stage(a.stage, models, out)

    from pipeline.inference import common, raw_features
    setup(models, out)
    common.bind(out, models)
    #1 raw series -> 977 base features + sequence cache
    raw_features.foundation(models, out, a.series, a.workers)
    #2 reversed / flipped / cropped views, same seeds as training
    crops = [(s["argv"][1], int(s["argv"][2]), 0.1, 0.4) for s in stages() if s["argv"][0].endswith("make_crops.py")]
    raw_features.variants(out, crops)
    #3 every recorded stage in training order: features, then saved models, then the fixed readouts
    env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT), str(ROOT / "pipeline/offline_hooks"), str(out)]),
           "KU_SBB_THREADS": "2", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
    for s in stages():
        print("stage", s["id"], flush=True)
        r = subprocess.run([sys.executable, "-B", "-m", "pipeline.predict_new", "--models", str(models), "--series", str(a.series),
                            "--out", str(out), "--stage", s["id"]], cwd=out, env=env)
        if r.returncode:
            raise SystemExit(f"stage {s['id']} failed")
    #4 map internal ids back and write id,break_probability
    sub = pd.read_csv(out / FINAL, dtype={"id": str})
    back = {m["internal_id"]: m["id"] for m in json.loads((out / "query_ids.json").read_text())}
    sub["id"] = sub["id"].map(back)
    sub[["id", "break_probability"]].to_csv(a.csv, index=False)
    print("wrote", a.csv, len(sub), "rows")


if __name__ == "__main__":
    main()
