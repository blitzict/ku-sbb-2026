"""Saved-booster prediction for train_cv, train_var (+TTA versions), distill and the nested mechanism trees.

Each function returns {output name: array} using the names the training script writes (test_<tag>, ver_<tag> test
part, contrib_<tag> test part), for the inference folder's test IDs. Boosters are scored with a plain DMatrix of
the same float32 values; XGBoost split thresholds are training quantile cuts, so this equals the training script's
QuantileDMatrix(ref=train) scoring (checked by pipeline/check_inference_trees.py).
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit

from pipeline.inference import features as FE
from pipeline.inference import source


def _xgb(path, X, margin=False, contribs=False):
    import xgboost as xgb
    b = xgb.Booster()
    b.load_model(str(source.require(path)))
    X = np.ascontiguousarray(X, dtype=np.float32)
    if b.num_features() != X.shape[1]:
        raise ValueError(f"{path}: booster expects {b.num_features()} features, got {X.shape[1]}")
    if contribs:
        return b.predict(xgb.DMatrix(X), pred_contribs=True)
    return b.predict(xgb.DMatrix(X), output_margin=margin)


def _meta(model_root, tag):
    return json.loads(source.require(Path(model_root) / "weights" / f"{tag}.meta.json").read_text())


def _select(frame, cols, what):
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise KeyError(f"{what}: {len(missing)} trained columns missing from the new features, e.g. {missing[:3]}")
    return frame[cols]


def train_cv(stage, model_root, feature_root):
    a = source.parse_args(stage["argv"][0], stage["argv"][1:])
    if a.no_test:
        raise ValueError(f"{a.tag}: trained with --no-test, no refit model exists")
    meta = _meta(model_root, a.tag)
    Xt = _select(FE.train_cv_test(model_root, feature_root, a), meta["columns"], a.tag)
    base = Path(model_root) / "weights" / a.tag
    if a.model == "xgb":
        extra = json.loads(a.extra) if a.extra else {}
        if str(extra.get("objective", "")).startswith("rank:"):
            p = 1 / (1 + np.exp(-np.clip(_xgb(str(base) + ".ubj", Xt.to_numpy(np.float32), margin=True), -30, 30)))
        else:
            p = _xgb(str(base) + ".ubj", Xt.to_numpy(np.float32))
    elif a.model == "lgb":
        import lightgbm as lgb
        p = lgb.Booster(model_file=str(source.require(str(base) + ".txt"))).predict(Xt)
    elif a.model == "cat":
        from catboost import CatBoostClassifier
        m = CatBoostClassifier()
        m.load_model(str(source.require(str(base) + ".cbm")))
        p = m.predict_proba(Xt)[:, 1]
    else:
        raise ValueError(f"unknown model {a.model}")
    return {f"test_{a.tag}": p}


def train_var(stage, model_root, feature_root):
    a = source.parse_args(stage["argv"][0], stage["argv"][1:])
    if a.export:
        raise ValueError("export stages build feature matrices, not models")
    if a.sym:
        raise ValueError("--sym members are not part of the recipe")
    meta = _meta(model_root, a.tag)
    cols, ttak = meta["columns"], meta["tta_kinds"]
    outs = [FE.var_test(feature_root, cols).to_numpy()] + [FE.var_test(feature_root, cols, k).to_numpy() for k in ttak]
    base = Path(model_root) / "weights" / a.tag
    if a.model == "cat":
        from catboost import CatBoostClassifier
        m = CatBoostClassifier()
        m.load_model(str(source.require(str(base) + ".cbm")))
        tp = [m.predict_proba(o)[:, 1] for o in outs]
    else:
        tp = [_xgb(str(base) + ".ubj", o) for o in outs]
        if meta["params"]["objective"].startswith("rank"):
            tp = [1 / (1 + np.exp(-r)) for r in tp]
    return {f"test_{a.tag}": tp[0], f"test_{a.tag}_tta": np.mean(tp, 0),
            f"ver_{a.tag}": {"test": np.stack(tp).astype(np.float32), "kinds": np.array(["orig"] + ttak)}}


def distill(stage, model_root, feature_root):
    a = source.parse_args(stage["argv"][0], stage["argv"][1:])
    blocks = tuple(b for b in a.blocks.split(",") if b)
    Xt = FE.featset_test(model_root, feature_root, blocks)
    if a.cols:
        names = pd.read_csv(source.require(Path(model_root) / a.cols)).feature
        train_cols = set(FE.featset_columns(model_root, blocks))
        Xt = _select(Xt, [c for c in names if c in train_cols], a.tag)
    X = Xt.to_numpy(np.float32)
    test = np.zeros(len(X))
    for k in range(1, 6):
        test += _xgb(Path(model_root) / "weights" / a.tag / f"student{k}.ubj", X) / 5
    return {f"test_{a.tag}": test}


def _features_json(model_root, tag):
    return json.loads(source.require(Path(model_root) / f"tune_out/redesign/boosters/{tag}_features.json").read_text())


def _fold_boosters(model_root, tag, n):
    paths = [Path(model_root) / f"tune_out/redesign/boosters/{tag}_fold{k}.ubj" for k in range(n)]
    for p in paths:
        source.require(p)
    extra = Path(model_root) / f"tune_out/redesign/boosters/{tag}_fold{n}.ubj"
    if extra.exists():
        raise ValueError(f"{tag}: more fold boosters than the {n} splits in the stage settings")
    return paths


def _check_columns(frame, expected, what):
    got = [str(c) for c in frame.columns]
    if got != [str(c) for c in expected]:
        raise ValueError(f"{what}: assembled columns differ from the trained features.json")


def nested_tree(stage, model_root, feature_root, tag=None, tables=None, extra=None):
    """nested_tree.py, and wnlaw_nested(2).py via tag/tables (no --extra, law_outer0, 5 folds)."""
    if tag is None:
        a = source.parse_args(stage["argv"][0], stage["argv"][1:])
        tag, extra = a.tag, [t for t in a.extra.split(",") if t]
        tables = ["harmonic", "ar_law", "bounds", "curve", "noise_tests", "partition", "periodic", "phase_law", "trend",
                  "innov_v2", "mixture_law"] + extra
    O, WS = Path(feature_root) / "tune_out", Path(feature_root) / "tune_out/weight_space"
    tids = FE.test_ids(feature_root)
    cols = FE.matrix_columns(model_root, "mat_ns3_plain")
    src = lambda t, p: (O / f"redesign/{t[9:]}_{p}.parquet") if t.startswith("redesign:") else (WS / f"{t}_{p}.parquet")
    ns3t = pd.DataFrame(FE.var_test(feature_root, cols).to_numpy(), index=tids, columns=cols)
    At = pd.concat([ns3t] + [pd.read_parquet(source.require(src(t, "test"))).loc[tids].add_prefix(f"{t}__") for t in tables], axis=1)
    expected = _features_json(model_root, tag)["features"]
    test = np.zeros(len(tids))
    for k, booster in enumerate(_fold_boosters(model_root, tag, 5)):
        Lt = pd.read_parquet(source.require(WS / f"law_outer0/fold{k}_test.parquet")).loc[tids].add_prefix("wnlaw__")
        frame = pd.concat([At, Lt], axis=1)
        _check_columns(frame, expected, f"{tag} fold {k}")
        test += _xgb(booster, frame.to_numpy(np.float32)) / 5
    return {f"test_{tag}": test}


WNLAW = {"experiments/redesign/wnlaw_nested.py": ("MECHNEST", []),
         "experiments/redesign/wnlaw_nested2.py": ("MECHNEST2", ["jump_law_pulse", "jump_law_state"])}


def wnlaw_nested(stage, model_root, feature_root):
    tag, extra = WNLAW[stage["argv"][0]]
    tables = ["harmonic", "ar_law", "bounds", "curve", "noise_tests", "partition", "periodic", "phase_law", "trend",
              "innov_v2", "mixture_law"] + extra
    return nested_tree(stage, model_root, feature_root, tag=tag, tables=tables)


def nested_views(stage, model_root, feature_root):
    """nested_crop.py and nested_rank.py: original plus view rows, 5 or n fold boosters, weighted logit TTA."""
    script = stage["argv"][0]
    rank = script.endswith("nested_rank.py")
    a = source.parse_args(script, stage["argv"][1:])
    if rank and a.gate_phi is not None:
        raise ValueError("--gate-phi is not part of the recipe")
    feature_root = Path(feature_root)
    tids = FE.test_ids(feature_root)
    mask = a.mask.split(",") if rank else []
    LAWD = a.law_dir or f"law_outer{a.fold_seed}"
    ns = {"O": feature_root / "tune_out", "WS": feature_root / "tune_out/weight_space", "ids": None, "tids": tids,
          "pd": pd, "a": a, "LAWD": LAWD}
    source.definitions(script, ["table", "law"], ns)
    T = ["harmonic", "ar_law", "bounds", "curve", "noise_tests", "partition", "periodic", "phase_law", "trend", "innov_v2", "mixture_law"]
    VIEWS = [v for v in a.views.split(",") if v]
    EXTRA = [t for t in a.extra.split(",") if t]
    cols = FE.matrix_columns(model_root, "mat_ns3_c7")
    ALL = [None] + VIEWS

    def base(view):
        X = FE.var_test(feature_root, cols, view).to_numpy()
        parts = [] if "ns3" in mask else [pd.DataFrame(X, index=tids, columns=cols)]
        parts += [ns["table"](t, "test", view).add_prefix(f"{t}__") for t in T + EXTRA if t not in mask]
        return pd.concat(parts, axis=1)

    def law(view, k):
        if "wnlaw" in mask:
            return pd.DataFrame(index=tids)
        p = ns["WS"] / (LAWD if view is None else f"view_{view}/{LAWD}")
        return pd.read_parquet(source.require(p / f"fold{k}_test.parquet")).loc[tids].add_prefix("wnlaw__")

    B = {v: base(v) for v in ALL}
    for v in VIEWS:
        if list(B[v].columns) != list(B[None].columns):
            raise ValueError(f"{a.tag}: column mismatch in view {v}")
    info = _features_json(model_root, a.tag)
    WT = np.array([2.0] + [a.crop_w if v.startswith("c") else 1.0 for v in VIEWS])
    vert = np.zeros((len(ALL), len(tids)))
    contribs = bool(rank and a.contribs)
    if contribs:
        grp = np.array(info["groups"])
        GN = list(dict.fromkeys(grp))
        CT = np.zeros((len(tids), len(GN) + 1), np.float32)
    for k, booster in enumerate(_fold_boosters(model_root, a.tag, a.n_splits)):
        Fv = {v: pd.concat([B[v], law(v, k)], axis=1) for v in ALL}
        _check_columns(Fv[None], info["features"], f"{a.tag} fold {k}")
        for i, v in enumerate(ALL):
            X = Fv[v].to_numpy(np.float32)
            vert[i] += (expit(_xgb(booster, X, margin=True)) if rank else _xgb(booster, X)) / a.n_splits
        if contribs:
            C = _xgb(booster, Fv[None].to_numpy(np.float32), contribs=True)
            CT += np.column_stack([C[:, :-1][:, grp == g].sum(1) for g in GN] + [C[:, -1]]) / a.n_splits
    lg = lambda p: logit(np.clip(p, 1e-6, 1 - 1e-6))
    out = {f"test_{a.tag}": expit((WT[:, None] * lg(vert)).sum(0) / WT.sum()),
           f"ver_{a.tag}": {"test": vert.astype(np.float32), "kinds": np.array(["orig"] + VIEWS)}}
    if contribs:
        out[f"redesign/contrib_{a.tag}"] ={"test": CT, "groups": np.array(GN + ["bias"]), "tids": np.asarray(tids).astype(str)}
    return out
