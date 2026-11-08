# CPU port of experiments/weight_edit_20261104/finalize.py, source sha256 db7d4a84cce1bd26eb848db793b54b3f288c5ab6ae9f20b4c8847e9a3cc41ea6. See auc_lab_20261107/claude/audit/cpu_port_notes.md.
import sys as _s, pathlib as _p; _s.path.insert(0, str(_p.Path(__file__).resolve().parents[2])); import ku_workspace as KU_ENV
import rebuild_policy as KU_RB
"""Compare complete ensembles, save the best candidate, and replay its artifact.

Run without arguments after replay_head_edits.py. To reproduce its submission:
  python finalize.py --predict weights/auc_weight_edit_20261104/ensemble.json
"""
import argparse
import ctypes
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import logit, expit
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
from stack_edit import ROOT, OUT, load_data, logistic_fit, score, sha
from view_weight_edit import expand, fit_views


def inputs(neural):
    names, ids, tids, y, wt, X, T, base = load_data(ROOT / "submissions/sub_seqw5_stack.json")
    archives = {}
    if neural:
        j = names.index("MLPHV_bag_w")
        X[:, j] = logit(np.clip(np.load(OUT / "oof_head_v2_bag.npy"), 1e-6, 1-1e-6))
        T[:, j] = logit(np.clip(np.load(OUT / "test_head_v2_bag.npy"), 1e-6, 1-1e-6))
        archives = {t: OUT / f"ver_head_v2_{t}.npz" for t in ["MLPHV_big", "MLPHV_big_o1", "MLPHV_big_o2"]}
    V, VT, labels, groups, hashes = expand(names, X, T, archives)
    return names, ids, tids, y, wt, X, T, V, VT, labels, groups, hashes


def write_submission(p, tids, output):
    assert p.shape == (len(tids),) and np.isfinite(p).all()
    assert ((0 <= p) & (p <= 1)).all()
    sub = pd.read_csv(ROOT / "data/sample_submission.csv", dtype={"id": str})
    assert sub.id.is_unique and set(sub.id) == set(tids)
    sub["break_probability"] = pd.Series(p, index=tids).loc[sub.id].to_numpy()
    sub.to_csv(output, index=False)
    read = pd.read_csv(output, dtype={"id": str})
    assert read.columns.tolist() == ["id", "break_probability"] and len(read) == len(tids)
    assert np.array_equal(read.id, sub.id)
    assert np.max(np.abs(read.break_probability - sub.break_probability)) < 1e-14
    return sub


def predict(artifact, output):
    item = json.loads(artifact.read_text())
    for name, expected in item["input_hashes"].items():
        assert sha(ROOT / name) == expected, f"Changed input: {name}"
    names, ids, tids, y, wt, X, T, V, VT, labels, groups, hashes = inputs(item["uses_edited_neural_heads"])
    matrix, expected_labels = (VT, labels) if item["kind"] == "view_stack" else (T, names)
    assert len(item["feature_order"]) == len(set(item["feature_order"]))
    assert set(item["feature_order"]).issubset(expected_labels)
    matrix = matrix[:, [expected_labels.index(label) for label in item["feature_order"]]]
    theta = np.array(item["coefficients_and_intercept"])
    assert matrix.shape[1] + 1 == len(theta)
    return write_submission(expit(matrix @ theta[:-1] + theta[-1]), tids, output)


def bootstrap(y, base, new, wt, repeats=500):
    rng = np.random.default_rng(20261104)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    deltas = []
    for _ in range(repeats):
        ix = np.r_[rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]
        deltas.append([
            roc_auc_score(y[ix], new[ix]) - roc_auc_score(y[ix], base[ix]),
            roc_auc_score(y[ix], new[ix], sample_weight=wt[ix]) - roc_auc_score(y[ix], base[ix], sample_weight=wt[ix]),
        ])
    ci = np.quantile(deltas, [.025, .975], axis=0)
    return {"repeats": repeats, "auc_delta_95pct": ci[:, 0].tolist(), "weighted_auc_delta_95pct": ci[:, 1].tolist(),
            "scope": "Paired, class-stratified resampling of fixed OOF scores; does not include fitting/selection uncertainty"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predict")
    ap.add_argument("--output", default="submissions/sub_weight_edit_20261104_replay.csv")
    a = ap.parse_args()
    KU_ENV.low_priority()
    if a.predict:
        sub = predict(ROOT / a.predict, ROOT / a.output)
        print(f"Replayed {len(sub)} rows from saved coefficient artifact", flush=True)
        return
    names, ids, tids, y, wt, X, T, V, VT, labels, groups, hashes = inputs(True)
    results = {}
    oofs = {}
    old_views = json.loads((OUT / "view_results.json").read_text())
    old_oof = np.load(OUT / "view_oof.npz", allow_pickle=False)
    for split in [7, 29]:
        p_head, p_views = np.zeros(len(y)), np.zeros(len(y))
        for fold, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=split).split(X, y)):
            init = logistic_fit(X[tr], y[tr])
            p_head[va] = X[va] @ init[:-1] + init[-1]
            th = fit_views(V[tr], y[tr], groups, init, 100.)
            p_views[va] = V[va] @ th[:-1] + th[-1]
        results[str(split)] = {
            "baseline": old_views[str(split)]["baseline"],
            "view_weights": old_views[str(split)]["view_100"],
            "neural_head": score(y, p_head, wt),
            "neural_head_and_views": score(y, p_views, wt),
        }
        oofs[f"s{split}_baseline"] = old_oof[f"s{split}_baseline"]
        oofs[f"s{split}_view_weights"] = old_oof[f"s{split}_view_100"]
        oofs[f"s{split}_neural_head"] = p_head
        oofs[f"s{split}_neural_head_and_views"] = p_views
        print(json.dumps({"meta_split": split, "results": results[str(split)]}), flush=True)
    # Require both metrics to improve on both meta splits. Prefer the largest
    # mean AUC among eligible configurations. All comparisons are exploratory.
    eligible = [k for k in ["view_weights", "neural_head", "neural_head_and_views"]
                if all(results[str(s)][k][metric] > results[str(s)]["baseline"][metric]
                       for s in [7, 29] for metric in ["auc", "weighted_auc"])]
    chosen = max(eligible, key=lambda k: np.mean([results[str(s)][k]["auc"] for s in [7, 29]])) if eligible else "baseline"
    chosen = KU_RB.select("neural_head_and_views", chosen)      # Fixed recipe; measured `eligible` stays reported.
    use_neural = chosen in ["neural_head", "neural_head_and_views"]
    use_views = chosen in ["view_weights", "neural_head_and_views"]
    names, ids, tids, y, wt, X, T, V, VT, labels, groups, hashes = inputs(use_neural)
    init = logistic_fit(X, y)
    theta = fit_views(V, y, groups, init, 100.) if use_views else init
    matrix, feature_order = (VT, labels) if use_views else (T, names)
    p = expit(matrix @ theta[:-1] + theta[-1])
    json_audit = json.loads((OUT / "audit.json").read_text())
    input_hashes = dict(json_audit["prediction_hashes"])
    input_hashes.update(hashes)
    input_hashes["submissions/sub_seqw5_stack.json"] = sha(ROOT / "submissions/sub_seqw5_stack.json")
    for index_source in ["clean/features_test.parquet", "data/sample_submission.csv"]:
        input_hashes[index_source] = sha(ROOT / index_source)
    if use_neural:
        for file in ["oof_head_v2_bag.npy", "test_head_v2_bag.npy"]:
            path = OUT / file
            input_hashes[str(path.relative_to(ROOT))] = sha(path)
    delta = {s: {m: results[s][chosen][m] - results[s]["baseline"][m] for m in ["auc", "weighted_auc"]} for s in results}
    meets_bar = chosen != "baseline" and all(delta["7"][m] >= .0003 for m in delta["7"])
    ci = bootstrap(y, oofs["s7_baseline"], oofs[f"s7_{chosen}"], wt)
    artifact = {
        "selected_candidate": chosen, "kind": "view_stack" if use_views else "logistic_stack",
        "uses_edited_neural_heads": use_neural, "feature_order": feature_order,
        "coefficients_and_intercept": theta.tolist(), "view_penalty": 100 if use_views else None,
        "input_hashes": input_hashes, "cv": results["7"][chosen]["auc"],
        "cv_test_weighted": results["7"][chosen]["weighted_auc"],
        "scores": results, "improvement": delta, "bootstrap": ci,
        "meets_existing_promotion_bar": meets_bar, "target_auc": .87, "target_reached": False,
        "status": "candidate evaluated locally; no Kaggle submission performed",
        "validation_scope": "Exploratory meta CV of existing OOF predictions; not a new independent holdout or fully nested retraining",
        "reproduction": ".venv/Scripts/python experiments/weight_edit_20261104/finalize.py --predict weights/auc_weight_edit_20261104/ensemble.json",
        "neural_checkpoint_directory": "weights_experiments/weight_edit_20261104/neural_head" if use_neural else None,
    }
    artifact["target_reached"] = artifact["cv"] >= .87
    wd = ROOT / "weights/auc_weight_edit_20261104"
    wd.mkdir(exist_ok=True)
    (wd / "ensemble.json").write_text(json.dumps(artifact, indent=2))
    sub = write_submission(p, tids, ROOT / "submissions/sub_weight_edit_20261104.csv")
    (ROOT / "submissions/sub_weight_edit_20261104.json").write_text(json.dumps(artifact, indent=2))
    (OUT / "final_results.json").write_text(json.dumps(artifact, indent=2))
    np.savez(OUT / "final_oof.npz", **oofs, y=y, sample_weight=wt, ids=np.asarray(ids, dtype=str))
    print(json.dumps({"selected": chosen, "cv": artifact["cv"], "weighted_cv": artifact["cv_test_weighted"],
                      "delta": delta, "bootstrap": ci, "meets_promotion_bar": meets_bar,
                      "rows": len(sub), "target_reached": artifact["target_reached"]}), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
