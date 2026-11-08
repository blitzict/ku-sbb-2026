"""Test parts of fixed-weight combiners and saved-coefficient stacks.

wtta, mech_blends, align_predictions and white_source_bag use fixed weights: their test arrays are recomputed from
the members' new test arrays (written into the prediction workspace by the member adapters). stack_edit and
view_weight_edit apply the coefficients they saved at training time (weights_experiments/weight_edit_20261104/
{stack,view}_coefficients.json) to the new member test logits; nothing is refitted. write_config, finalize and simplify
produce only learned/train-side artifacts (member list, final coefficients, pruning scores, the "new weights" bundle):
base_models.predict_stage copies their declared outputs unchanged (TRAIN_ONLY).
"""
import json
from pathlib import Path

import numpy as np
from scipy.special import expit, logit

from pipeline.inference import features as FE
from pipeline.inference import source


def _test(work, name):
    p = np.load(source.require(Path(work) / "tune_out" / f"test_{name}.npy"))
    if p.ndim != 1 or not np.isfinite(p).all():
        raise ValueError(f"invalid test_{name}.npy")
    return p


def wtta(stage, model_root, work):
    script = stage["argv"][0]
    a = source.parse_args(script, stage["argv"][1:])
    WT = source.definitions(script, ["WT"], {"a": a})["WT"]
    acc = []
    for t in a.tags:
        with np.load(source.require(Path(work) / "tune_out" / f"ver_{t}.npz")) as z:
            W = np.array([WT[k] for k in z["kinds"]]); Lg = logit(np.clip(z["test"].astype(float), 1e-6, 1 - 1e-6))
        acc.append((W[:, None] * Lg).sum(0) / W.sum())
    return {f"test_{a.out}": expit(np.mean(acc, 0))}


def mech_blends(stage, model_root, work):
    script = stage["argv"][0]
    names = stage["argv"][1:]
    if not names:
        raise ValueError("mech_blends stage must name its blends")
    ns = source.definitions(script, ["SEQ", "LOGIT", "PROB", "VIEWS", "lg"], {"np": np, "logit": logit, "range": range})
    LOGIT, PROB, VIEWS, lg = ns["LOGIT"], ns["PROB"], ns["VIEWS"], ns["lg"]
    O = Path(work) / "tune_out"
    out = {}
    for name in names:
        if name in VIEWS:
            with np.load(source.require(O / f"ver_{VIEWS[name][0]}.npz")) as z:
                wt = VIEWS[name][1]; W = np.array([wt.get(k, 0) for k in z["kinds"]], float)
                out[f"test_{name}"] = expit((W[:, None] * lg(z["test"])).sum(0) / W.sum())
        else:
            A = np.stack([np.load(source.require(O / f"test_{m}.npy")) for m in {**LOGIT, **PROB}[name]])
            out[f"test_{name}"] = A.mean(0) if name in PROB else expit(lg(A).mean(0))
    return out


def align_predictions(stage, model_root, work):
    tag, name = stage["argv"][1], stage["argv"][2]
    score = _test(work, tag)
    tids = np.asarray(FE.test_ids(work)).astype(str)
    if score.shape != tids.shape or not ((score > 0) & (score < 1)).all():
        raise ValueError(f"Invalid test probability array for {tag}")
    return {f"weight_space/{name}": {"tids": tids, "test": score}}


def white_source_bag(stage, model_root, work):
    name, members = stage["argv"][1], stage["argv"][2:]
    WS = Path(work) / "tune_out/weight_space"
    total, tids = 0.0, None
    for member in members:
        with np.load(source.require(WS / f"{member}.npz")) as z:
            if tids is None:
                tids = z["tids"].copy()
            if not np.array_equal(tids, z["tids"]):
                raise ValueError(f"{member}: test IDs differ")
            p = z["test"]
            if not np.all((p > 0) & (p < 1)):
                raise ValueError(f"{member}: test probabilities outside (0, 1)")
            total = total + logit(p)
    return {f"weight_space/{name}": {"tids": tids, "test": expit(total / len(members))}}


def _member_logits(model_root, work, names=None):
    if names is None:
        names = json.loads(source.require(Path(model_root) / "submissions/sub_seqw5_stack.json").read_text())["members"]
    return names, np.column_stack([logit(np.clip(_test(work, n), 1e-6, 1 - 1e-6)) for n in names]).astype(np.float64)


COEF = "weights_experiments/weight_edit_20261104"


def stack_edit(stage, model_root, work):
    art = json.loads(source.require(Path(model_root) / COEF / "stack_coefficients.json").read_text())
    names, T = _member_logits(model_root, work, art["members"])
    tests = {}
    for key, theta in art["weights_and_intercepts"].items():
        th = np.array(theta)
        tests[key] = expit(T @ th[:-1] + th[-1])
    return {"weight_edit_20261104/stack_test": {**tests, "ids": np.asarray(FE.test_ids(work)).astype(str)}}


def view_test_matrix(model_root, work, names, T, archive=None):
    """view_weight_edit.expand, test columns only, from the members' new ver_ test arrays."""
    groups = source.definitions("experiments/weight_edit_20261104/view_weight_edit.py", ["GROUPS"], {})["GROUPS"]
    archive = archive or {}
    tcols, labels = [], []
    for i, name in enumerate(names):
        if name not in groups:
            tcols.append(T[:, i]); labels.append(name)
            continue
        tags, crop = groups[name]
        tests, kinds = [], None
        for tag in tags:
            with np.load(source.require(archive.get(tag, Path(work) / "tune_out" / f"ver_{tag}.npz"))) as z:
                if kinds is not None and not np.array_equal(kinds, z["kinds"]):
                    raise ValueError(f"{tag}: version kinds differ")
                kinds = z["kinds"]
                tests.append(logit(np.clip(z["test"].astype(float), 1e-6, 1 - 1e-6)))
        t = np.mean(tests, axis=0)
        wt = {"orig": 2., "flip": 2., "rev": 1., "revflip": 1.}
        weights = np.array([wt.get(str(k), crop) for k in kinds]); weights /= weights.sum()
        t += (T[:, i] - weights @ t)[None]
        if np.max(np.abs(weights @ t - T[:, i])) >= 1e-10:
            raise ValueError(f"{name}: view correction does not reproduce the member")
        tcols.extend(list(t)); labels.extend([f"{name}::{k}" for k in kinds])
    return np.column_stack(tcols), labels


def view_weight_edit(stage, model_root, work):
    art = json.loads(source.require(Path(model_root) / COEF / "view_coefficients.json").read_text())
    names, T = _member_logits(model_root, work, art["source_members"])
    VT, labels = view_test_matrix(model_root, work, names, T)
    if labels != art["labels"]:
        raise ValueError("expanded view labels differ from the trained coefficients")
    tests = {key: expit(VT @ np.array(th)[:-1] + np.array(th)[-1]) for key, th in art["weights_and_intercepts"].items()}
    return {"weight_edit_20261104/view_test": {**tests, "ids": np.asarray(FE.test_ids(work)).astype(str)}}


TRAIN_ONLY = {"pipeline_helpers/write_config.py", "experiments/weight_edit_20261104/finalize.py",
              "experiments/weight_edit_20261104/simplify.py"}
