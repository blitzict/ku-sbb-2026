"""predict_stage(): saved-model inference for one BASE/MECH stage of pipeline/full_cpu_stages.json.

    from pipeline.inference.base_models import predict_stage, supported
    written = predict_stage(stage, model_root, work)

model_root  complete trained workspace (read only: weights/, tune_out/redesign/boosters/, tune_out/var/*.npz column
            lists, the training ver_/contrib_ files whose OOF parts are kept)
work        prediction workspace holding the new test features under the training names (clean/features_test.parquet,
            tune_out/*_test.parquet, tune_out/var/*_test_<kind>.parquet, tune_out/weight_space/...)

Writes the stage's canonical test outputs into work (test_<tag>.npy; ver_<tag>.npz and contrib_<tag>.npz keep the
trained OOF arrays and replace only the test fields) and returns their paths. Train-only outputs (oof_, best_) are
copied unchanged from model_root when the stage declares them, so later readers find the files they expect.
model_root is never written. Raises for stages it does not cover (see supported()).
"""
import shutil
from pathlib import Path

import numpy as np

from pipeline.inference import combiners, neural, trees

ADAPTERS = {
    "experiments/v6/train_cv.py": trees.train_cv,
    "experiments/v6/train_var.py": trees.train_var,
    "experiments/v6/distill.py": trees.distill,
    "experiments/redesign/nested_tree.py": trees.nested_tree,
    "experiments/redesign/wnlaw_nested.py": trees.wnlaw_nested,
    "experiments/redesign/wnlaw_nested2.py": trees.wnlaw_nested,
    "experiments/redesign/nested_crop.py": trees.nested_views,
    "experiments/redesign/nested_rank.py": trees.nested_views,
    "experiments/v6/nn_var.py": neural.nn_var,
    "experiments/v6/nn_xpu.py": neural.nn_xpu,
    "experiments/v6/nn_seq.py": neural.nn_seq,
    "experiments/weight_edit_20261104/neural_head_edit.py": neural.neural_head_edit,
    "experiments/weight_edit_20261104/replay_head_edits.py": neural.replay_head_edits,
    "experiments/v6/wtta.py": combiners.wtta,
    "experiments/redesign/mech_blends.py": combiners.mech_blends,
    "pipeline_helpers/align_predictions.py": combiners.align_predictions,
    "experiments/weight_space/white_source_bag.py": combiners.white_source_bag,
    "experiments/weight_edit_20261104/stack_edit.py": combiners.stack_edit,
    "experiments/weight_edit_20261104/view_weight_edit.py": combiners.view_weight_edit,
}


def register(script, function):
    ADAPTERS[script] = function


def source_root(model_root):
    return Path(model_root)


def supported(stage):
    if stage["argv"][0] in combiners.TRAIN_ONLY:
        return True
    if stage["argv"][0] == "experiments/v6/train_var.py" and "--export" in stage["argv"]:
        return False
    return stage["argv"][0] in ADAPTERS


def _write(work, model_root, name, value):
    O, M = Path(work) / "tune_out", Path(model_root) / "tune_out"
    if isinstance(value, dict):
        path = O / f"{name}.npz"
        trained = M / f"{name}.npz"
        with np.load(trained, allow_pickle=True) as z:
            fields = {k: z[k] for k in z.files}
        for k, v in value.items():          # test arrays (and contrib tids) are replaced; labels must match training
            if k in ("kinds", "groups") and k in fields and not np.array_equal(fields[k].astype(str), np.asarray(v).astype(str)):
                raise ValueError(f"{name}: {k} differ from the trained file")
            fields[k] = v
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **fields)
    else:
        path = O / f"{name}.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, np.asarray(value))
    return path


def predict_stage(stage, model_root, work):
    model_root, work = Path(model_root).resolve(), Path(work).resolve()
    if work == model_root:
        raise ValueError("Prediction workspace must differ from the trained workspace")
    if stage["argv"][0] in combiners.TRAIN_ONLY:     # learned/train-side artifacts only: reuse them unchanged
        written = []
        for relative in stage["outputs"]:
            target = work / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_root(model_root) / relative, target)
            written.append(target)
        return written
    if not supported(stage):
        raise NotImplementedError(f"{stage['id']}: no saved-model adapter for {stage['argv'][0]}")
    result = ADAPTERS[stage["argv"][0]](stage, model_root, work)
    written = [_write(work, model_root, name, value) for name, value in result.items()]
    for relative in stage["outputs"]:
        name = Path(relative).name
        if name.startswith(("oof_", "best_")) and relative.startswith("tune_out/"):
            target = work / relative
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(model_root / relative, target)
    produced = {p.relative_to(work).as_posix() for p in written}
    missing = [o for o in stage["outputs"] if Path(o).name.startswith(("test_", "ver_", "contrib_")) and o not in produced]
    if missing:
        raise ValueError(f"{stage['id']}: adapter did not produce declared test outputs {missing}")
    return written


def main():
    """Subprocess entry: python -m pipeline.inference.base_models STAGE_ID [--manifest M].
    Reads KU_SBB_MODEL_ROOT (trained workspace) and KU_SBB_ROOT (prediction workspace) like the training stages."""
    import argparse
    import json
    import os
    from pipeline.run_stages import ROOT
    p = argparse.ArgumentParser(description=main.__doc__)
    p.add_argument("stage_id")
    p.add_argument("--manifest", type=Path, default=ROOT / "pipeline/full_cpu_stages.json")
    a = p.parse_args()
    stages = {s["id"]: s for s in json.loads(a.manifest.read_text())["stages"]}
    if a.stage_id not in stages:
        raise SystemExit(f"unknown stage {a.stage_id}")
    for path in predict_stage(stages[a.stage_id], os.environ["KU_SBB_MODEL_ROOT"], os.environ["KU_SBB_ROOT"]):
        print("wrote", path, flush=True)


if __name__ == "__main__":
    main()
