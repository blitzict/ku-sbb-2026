import sys as _s, pathlib as _p; _s.path.insert(0, str(_p.Path(__file__).resolve().parents[2])); import ku_workspace as KU_ENV
"""Model saving for the CPU ports. Each save is reloaded and checked at once, so a failed save stops the run.

The historical nested runs saved no models, or only under --contribs. Saved models here belong to new CPU members, not to the 0.88 submissions.
"""
import json, pathlib
import numpy as np


def folder(O):
    d = pathlib.Path(O) / "redesign/boosters"; d.mkdir(parents=True, exist_ok=True); return d


def features(O, tag, names, **meta):
    (folder(O) / f"{tag}_features.json").write_text(json.dumps(dict(features=[str(n) for n in names], **meta), indent=1))


def save_xgb(O, tag, k, booster, X_check, margin=False):
    import xgboost as xgb
    p = folder(O) / f"{tag}_fold{k}.ubj"; booster.save_model(str(p)); b = xgb.Booster(); b.load_model(str(p))
    a = booster.predict(xgb.DMatrix(X_check), output_margin=margin); c = b.predict(xgb.DMatrix(X_check), output_margin=margin)
    if float(np.abs(a - c).max()) > 1e-6: raise RuntimeError(f"saved booster differs: {p}")
    return p


def save_cat(O, tag, k, model, X_check):
    from catboost import CatBoostClassifier
    p = folder(O) / f"{tag}_fold{k}.cbm"; model.save_model(str(p)); m = CatBoostClassifier(); m.load_model(str(p))
    if float(np.abs(model.predict_proba(X_check)[:, 1] - m.predict_proba(X_check)[:, 1]).max()) > 1e-6: raise RuntimeError(f"saved model differs: {p}")
    return p
