"""columns the try4 models in use read, per feature table (union of the 5 folds' "top" lists).

the model folder comes from KU_TRY4_MODEL (predict.py sets it from --model); default try4/model.
"""
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent


def model_dir():
    return Path(os.environ.get("KU_TRY4_MODEL") or HERE / "model")


def needed_all(model=None):
    model = Path(model or model_dir())
    out = {}
    for k in range(5):
        for c in json.loads((model / f"columns_fold{k}.json").read_text())["top"]:
            t, col = c.split("__", 1)
            out.setdefault(t, set()).add(col)
    return {t: sorted(v) for t, v in out.items()}


def needed(table, model=None):
    return needed_all(model).get(table, [])
