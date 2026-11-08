"""Run only the retained final readouts, with their original numerical settings.

The historical experiment scripts also evaluated alternative routes and an
extra diagnostic partition. Those alternatives do not feed the selected CSV.
This entry point retains all six recorded output partitions, the same inner
folds, and the same final fit. It does not rerun discarded model selection.
All numerical functions below are imported from the reviewed CPU ports.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys

import ku_workspace as env
sys.path.insert(0, str(env.root() / "experiments/weight_space"))

import numpy as np
from scipy.special import expit
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
import white_loop as wl
import rebuild_policy
from white_checkpoint import SEEDS
from white_crop_replication import BASE, PREVIOUS
from white_local_rank_weights import weights
from white_nested import audit
from white_paired_branches import test_scores
from white_upstream_stack import collect, train_basis
from white_shared_residual import chart, MEMBERS
from white_opinion_spectrum import inputs, design
from white_joint import read_member
from partition_evidence import ROOT, OUT

FIRST = "local_rank_first_layer_overall"
FINAL = "local_rank_shared_residual_response"


def save(name, parent, ids, tids, logits, features, coefficients, scores,
         details, oof, dependencies, metadata):
    """Export the preselected recipe, while reporting its measured performance."""
    report = rebuild_policy.annotate(dict(scores=scores, audits=details, **metadata))
    (OUT / (name + ".json")).write_text(json.dumps(report, indent=2))
    np.savez_compressed(OUT / (name + "_oof.npz"), ids=ids.to_numpy(dtype=str), **oof)
    folder = ROOT / f"new weights/wn_{name}_candidate"
    origin = ROOT / f"new weights/wn_{parent}_candidate"
    folder.mkdir(exist_ok=True)
    csv, recipe = folder / "submission.csv", folder / "weights.json"
    submission = wl.ef.write_submission(expit(logits + features @ coefficients), tids, csv)
    deps = list(dict.fromkeys([Path(__file__).resolve(), origin / "weights.json", *dependencies]))
    recipe.write_text(json.dumps(dict(weights=coefficients.tolist(), **metadata,
        hashes={str(p.relative_to(ROOT)): wl.ef.sha(p) for p in deps}), indent=2))
    np.savez_compressed(folder / "replay_inputs.npz", ids=tids.to_numpy(dtype=str),
        order=tids.get_indexer(submission.id), parent_logits=logits, features=features)
    (folder / "validation.json").write_text(json.dumps(dict(**report,
        recipe_sha256=wl.ef.sha(recipe), submission_sha256=wl.ef.sha(csv)), indent=2))
    shutil.copyfile(origin / "replay.py", folder / "replay.py")
    print(name, "local AUC", scores[7]["new"]["auc"], flush=True)


def first_layer():
    """Exactly the conditional=False branch of white_regime_branches.main."""
    q, ids, tids, y, wt, wn, miss = wl.load()
    raw = collect(ids, tids)
    guide, n = raw[0], len(y)
    tr, te = np.arange(n), n + np.arange(len(tids))
    matrix = train_basis(raw, tr)
    coefficient = weights(matrix[tr], guide[tr], y)
    branch = guide[te] + matrix[te] @ coefficient
    oldpath = OUT / (BASE + "_oof.npz")
    uppath = OUT / "local_rank_upstream_joint_oof.npz"
    scores, details, predictions = {}, {}, {}
    with np.load(oldpath) as old, np.load(uppath) as upstream:
        assert np.array_equal(old["ids"], ids) and np.array_equal(upstream["ids"], ids)
        assert np.array_equal(upstream["tids"], tids)
        np.testing.assert_allclose(branch, upstream["branch_test"], atol=1e-7, rtol=0)
        for seed in SEEDS:
            previous, new = old[f"s{seed}"], upstream[f"s{seed}"]
            np.testing.assert_allclose(new, .75 * previous + .25 * upstream[f"branch_{seed}"], atol=1e-10, rtol=0)
            locked = np.empty(n)
            for fold in range(5):
                key = f"s{seed}_f{fold}_"
                locked[q[key + "va"]] = q[key + "zv"]
            scores[seed] = {k: wl.score(y, v, wt, wn, miss) for k, v in
                           (("locked", locked), ("previous", previous), ("new", new))}
            details[seed] = audit(y, miss, locked, new)
            predictions[f"s{seed}"] = new.copy()
    baseline = test_scores(BASE, tids)
    features = np.column_stack([guide[te] - baseline, matrix[te]])
    coeff = .25 * np.r_[1., coefficient]
    save(FIRST, BASE, ids, tids, baseline, features, coeff, scores, details, predictions,
         [oldpath, uppath, *raw[-1]], dict(conditional=False, blend=.25,
         selection="Fixed historical first_layer_overall recipe; alternate recovery route omitted"))


def upstream_inner():
    """The saved branch from white_control_ranks, without its unused calibration."""
    q, ids, tids, y, *_ = wl.load()
    raw = collect(ids, tids)
    guide = raw[0]
    path = OUT / "local_rank_upstream_joint_oof.npz"
    digest = wl.ef.sha(path)
    folder = OUT / "upstream_inner"
    folder.mkdir(exist_ok=True)
    for seed in SEEDS:
        for fold in range(5):
            target = folder / f"s{seed}_f{fold}.npz"
            train = q[f"s{seed}_f{fold}_tr"]
            if target.exists():
                with np.load(target) as a:
                    assert str(a["branch_sha"]) == digest and np.array_equal(a["ids"], ids[train])
                continue
            matrix = train_basis(raw, train)[train]
            branch = np.empty(len(train))
            splits = StratifiedKFold(3, shuffle=True, random_state=7811 + fold)
            for fit, hold in splits.split(train, y[train]):
                coeff = weights(matrix[fit], guide[train[fit]], y[train[fit]])
                branch[hold] = guide[train[hold]] + matrix[hold] @ coeff
            np.savez_compressed(target, ids=ids[train].to_numpy(dtype=str), branch=branch, branch_sha=digest)
        print("saved upstream inner folds", seed, flush=True)


def shared_response():
    """Same response chart and coefficient fit as white_shared_residual.run.

    The unused agreement candidate, separate ablation comparison archive, and
    extra partition 61817 do not affect any retained coefficient or score.
    Primary seed 7 and all five other recorded partitions remain measured.
    """
    q, ids, tids, y, wt, wn, miss = wl.load()
    banks, regime, files = inputs(ids, tids)
    members = []
    for name in MEMBERS:
        values, paths = read_member(name, ids, tids)
        members.append(values[:, 0])
        files.extend(paths)
    extra = np.column_stack(members)
    folder = OUT / ("frozen_" + FIRST)
    n = len(y)
    scores, details, predictions = {}, {}, {}
    for seed in SEEDS:
        previous, locked, new = (np.empty(n) for _ in range(3))
        for fold, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=seed).split(ids, y)):
            with np.load(folder / f"s{seed}_f{fold}.npz") as a:
                assert np.array_equal(a["train_ids"], ids[tr]) and np.array_equal(a["validation_ids"], ids[va])
                z, v = a["oof_train"], a["valid"]
            locked[va] = q[f"s{seed}_f{fold}_zv"]
            active = list(banks)
            active[2] = active[2][:, :0]
            baseline = design(active, regime, tr, va, z, v)[0]
            previous[va] = v + baseline[len(tr):] @ (.5 * weights(baseline[:len(tr)], z, y[tr]))
            matrix = chart(banks, regime, extra, tr, va, z, v)["response"]
            new[va] = v + matrix[len(tr):] @ (.5 * weights(matrix[:len(tr)], z, y[tr]))
        scores[seed] = {k: wl.score(y, value, wt, wn, miss) for k, value in
                       (("locked", locked), ("previous", previous), ("new", new))}
        details[seed] = audit(y, miss, previous, new)
        predictions[f"s{seed}"] = new
        print("response", seed, scores[seed]["new"], flush=True)
    with np.load(folder / "full.npz") as a:
        assert np.array_equal(a["train_ids"], ids) and np.array_equal(a["test_ids"], tids)
        assert str(a["parent_sha"]) == wl.ef.sha(ROOT / f"new weights/wn_{FIRST}_candidate/weights.json")
        z, v = a["oof_train"], a["test"]
    matrix = chart(banks, regime, extra, np.arange(n), n + np.arange(len(tids)), z, v)["response"]
    coeff = .5 * weights(matrix[:n], z, y)
    save(FINAL, FIRST, ids, tids, v, matrix[n:], coeff, scores, details, predictions, files,
         dict(mode="response", reference="Recomputed without-multiview baseline",
              evaluated_partitions=list(SEEDS), extra_diagnostic_partition_61817=False,
              target_local_auc=.87910))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("first-layer", "upstream-inner", "shared-response"))
    args = parser.parse_args()
    if not rebuild_policy.fixed():
        raise RuntimeError("This entry point requires the explicitly fixed reconstruction recipe")
    with threadpool_limits(limits=1):
        {"first-layer": first_layer, "upstream-inner": upstream_inner,
         "shared-response": shared_response}[args.stage]()
