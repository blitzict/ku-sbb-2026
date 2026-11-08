"""Shared utilities for prediction-only adapters.

Adapters read learned parameters from model_root and write new predictions
to a separate work folder. Atomic replacement avoids modifying linked
reference files in the saved model directory.
"""
import json
import os
from pathlib import Path
import sys

import numpy as np


def bind(work, model_root, script=None):
    work, model_root = Path(work).resolve(), Path(model_root).resolve()
    if work == model_root or work.is_relative_to(model_root):
        raise ValueError("Prediction work must be separate from the saved model")
    if not (work / "CPU_REBUILD_WORKSPACE.json").is_file():
        raise ValueError("Missing prediction workspace marker")
    os.environ.update(KU_SBB_ROOT=str(work), KU_SBB_CODE_ROOT=str(work),
                      KU_SBB_MODEL_ROOT=str(model_root), KU_SBB_INFERENCE="1",
                      CUDA_VISIBLE_DEVICES="", ONEAPI_DEVICE_SELECTOR="*:cpu")
    paths = [work, work / "pipeline_helpers", work / "experiments/weight_space",
             work / "experiments/v6", work / "experiments/redesign",
             work / "experiments/weight_edit_20261104", work / "experiments/v5"]
    if script:
        paths.insert(0, (work / script).parent)
    sys.path[:0] = [str(path) for path in paths]
    return work, model_root


def destination(work, relative):
    work = Path(work).resolve()
    path = (work / relative).resolve()
    if not path.is_relative_to(work) or path == work:
        raise ValueError(f"Output escapes prediction workspace: {relative}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".prediction-partial")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def save_array(path, array):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".prediction-partial")
    with temporary.open("wb") as stream:
        np.save(stream, array)
    temporary.replace(path)


def save_table(path, table, **kwargs):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".prediction-partial.parquet")
    table.to_parquet(temporary, **kwargs)
    temporary.replace(path)


def replace_test_fields(model_root, work, relative, **updated):
    """Retain original OOF/reference arrays, replacing specified query fields."""
    with np.load(Path(model_root) / relative, allow_pickle=True) as archive:
        values = {key: archive[key] for key in archive.files}
    values.update(updated)
    save_npz(destination(work, relative), **values)


def require(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Saved model artifact is missing: {path}; prediction never refits it")
    return path


def option(stage, flag, default=None, converter=str):
    args = stage["argv"]
    if flag not in args:
        return default
    index = args.index(flag)
    if index + 1 >= len(args) or args[index + 1].startswith("--"):
        raise ValueError(f"Missing value for {flag}")
    return converter(args[index + 1])


def install_no_training_guard():
    """Fail loudly if a prediction adapter tries to train a model."""
    def forbidden(*args, **kwargs):
        raise RuntimeError("Model fitting is forbidden during saved-model prediction")
    import xgboost
    import lightgbm
    import catboost
    import torch
    xgboost.train = forbidden
    lightgbm.train = forbidden
    catboost.CatBoost.fit = forbidden
    catboost.CatBoostClassifier.fit = forbidden
    catboost.CatBoostRegressor.fit = forbidden
    torch.Tensor.backward = forbidden
    torch.autograd.backward = forbidden
    for name in ("SGD", "Adam", "AdamW", "RMSprop", "LBFGS", "Adagrad"):
        getattr(torch.optim, name).step = forbidden
    torch.set_num_threads(1)


def checked_probabilities(ids, values):
    ids, values = np.asarray(ids).astype(str), np.asarray(values)
    if values.shape != (len(ids),) or len(set(ids)) != len(ids):
        raise ValueError("Predictions must contain one value for each unique input identifier")
    if not np.isfinite(values).all() or not ((values >= 0) & (values <= 1)).all():
        raise ValueError("Predictions must be finite probabilities in [0, 1]")
    return ids, values
