# CPU port of experiments/weight_edit_20261104/simplify.py, source sha256 2210fe84b2007a41286fe43dcd17e9301ff55ef95e48bab6d3924650ebb996d6. See auc_lab_20261107/claude/audit/cpu_port_notes.md.
import sys as _s, pathlib as _p; _s.path.insert(0, str(_p.Path(__file__).resolve().parents[2])); import ku_workspace as KU_ENV
import rebuild_policy as KU_RB
"""Fold-local model pruning and a compact bundle in the requested new weights folder."""
import ctypes
import json
import shutil
import time
from pathlib import Path
import numpy as np
from scipy.special import expit
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from stack_edit import ROOT, OUT, logistic_fit, score, sha
from view_weight_edit import fit_views
from finalize import inputs, write_submission, bootstrap, predict


def subset(groups, selected):
    columns, reduced = [], []
    for i in selected:
        start, end, weights = groups[i]
        left = len(columns)
        columns.extend(range(start, end))
        reduced.append((left, len(columns), weights))
    return np.array(columns), reduced


def fit_topk(X, V, y, groups, k):
    initial = logistic_fit(X, y)
    # Selection uses only this fit's training rows, never outer labels.
    selected = np.argsort(-initial[:-1], kind="stable")[:k]
    columns, reduced = subset(groups, selected)
    compact_init = logistic_fit(X[:, selected], y)
    theta = fit_views(V[:, columns], y, reduced, compact_init, 100.)
    return theta, columns, selected


def copy_models(active_groups, folder, neural_edited):
    # Fresh model tags. Any other group trains under its own name. The five sequence models always ship, because downstream replay needs them.
    mapping = {"L4_fam_gh_b2": ["L4_fam_gh", "L4_fam_gh_s1"], "L4_fam_dd_b2": ["L4_fam_dd", "L4_fam_dd_s1"], "L4_fam_lr_b2": ["L4_fam_lr", "L4_fam_lr_s1"], "L4_fam_grid_b2": ["L4_fam_grid", "L4_fam_grid_s1"],
               "VARC4hv_ns3_wc": ["VARC4hv_ns3_s0", "VARC4hv_ns3_s1"], "VARC4_lm_wc": ["VARC4_lm_s0", "VARC4_lm_s1"], "VARC4_full_wc2": ["VARC4_full_s0", "VARC4_full_s1"], "VARC4_ns3_wc": ["VARC4_ns3_s0", "VARC4_ns3_s1"],
               "VAR_full_aug_avg_tta": ["VAR_full_aug", "VAR_full_aug_s1"], "VAR_aug1600_avg_tta": ["VAR_aug1600", "VAR_aug1600_s1"],
               "W_SEQ_qb5": ["W_SEQ_q", "W_SEQ_q_s1", "W_SEQ_q_s2", "W_SEQ_qh", "W_SEQ_qh_s1"], "MLPHV_bag_w": ["MLPHV_big", "MLPHV_big_o1", "MLPHV_big_o2"]}
    records, groups = [], {}
    for group in dict.fromkeys([*active_groups, "W_SEQ_qb5"]):
        group_records = []
        for tag in mapping.get(group, [group]):
            source = ROOT / "weights" / tag
            if group == "MLPHV_bag_w" and neural_edited:
                source = ROOT / "weights_experiments/weight_edit_20261104/neural_head" / f"head_v2_{tag}"
                files = sorted(source.glob("*.pt")) + sorted(source.glob("*.json"))
                destination = folder / "neural_head" / f"head_v2_{tag}"
            elif source.is_dir():
                files = sorted(p for p in source.iterdir() if p.is_file())
                destination = folder / "models" / tag
            else:
                files = [p for p in [Path(str(source) + x) for x in (".ubj", ".cbm", ".txt", ".meta.json")] if p.exists()]
                destination = folder / "models"
            assert files, str(source)
            destination.mkdir(parents=True, exist_ok=True)
            for file in files:
                target = destination / file.name
                assert target.resolve().is_relative_to(folder.resolve())
                shutil.copy2(file, target)
                expected = sha(file)
                assert sha(target) == expected
                record = {"file": str(target.relative_to(folder)), "source": str(file.relative_to(ROOT)),
                          "sha256": expected, "bytes": file.stat().st_size}
                records.append(record)
                group_records.append(record["file"])
        groups[group] = group_records
    return {"groups": groups, "files": records, "total_bytes": sum(r["bytes"] for r in records)}


def main():
    KU_ENV.low_priority()
    started = time.time()
    incumbent = json.loads((OUT / "final_results.json").read_text())
    assert incumbent["kind"] == "view_stack"
    use_neural = incumbent["uses_edited_neural_heads"]
    names, ids, tids, y, wt, X, T, V, VT, labels, groups, hashes = inputs(use_neural)
    original_oof = np.load(OUT / "final_oof.npz", allow_pickle=False)
    selection_name = incumbent["selected_candidate"]
    results, oofs = {}, {}
    for split in [7, 29]:
        results[str(split)] = {"unpruned": incumbent["scores"][str(split)][selection_name]}
        oofs[f"s{split}_unpruned"] = original_oof[f"s{split}_{selection_name}"]
        predictions = {k: np.zeros(len(y)) for k in [4, 6, 8]}
        for fold, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=split).split(X, y)):
            for k in predictions:
                theta, columns, selected = fit_topk(X[tr], V[tr], y[tr], groups, k)
                predictions[k][va] = V[va][:, columns] @ theta[:-1] + theta[-1]
            print(f"Pruning meta split {split} fold {fold} ({time.time()-started:.0f}s)", flush=True)
        for k, p in predictions.items():
            results[str(split)][f"top{k}"] = score(y, p, wt)
            oofs[f"s{split}_top{k}"] = p
        print(json.dumps({"split": split, "scores": results[str(split)]}), flush=True)
    # Further pruning must improve both reported metrics on both splits.
    eligible = [name for name in ["top4", "top6", "top8"]
                if all(results[str(s)][name][m] >= results[str(s)]["unpruned"][m]
                       for s in [7, 29] for m in ["auc", "weighted_auc"])]
    chosen = max(eligible, key=lambda name: np.mean([results[str(s)][name]["auc"] for s in [7, 29]])) if eligible else "unpruned"
    chosen = KU_RB.select("unpruned", chosen)      # Fixed recipe; measured `eligible` stays reported.
    if chosen == "unpruned":
        theta = np.array(incumbent["coefficients_and_intercept"])
        columns = np.arange(V.shape[1])
    else:
        theta, columns, selected = fit_topk(X, V, y, groups, int(chosen[3:]))
    keep = theta[:-1] > 0
    compact = np.r_[theta[:-1][keep], theta[-1]]
    active_columns = columns[keep]
    dense = VT[:, columns] @ theta[:-1] + theta[-1]
    sparse = VT[:, active_columns] @ compact[:-1] + compact[-1]
    assert np.max(np.abs(dense - sparse)) < 1e-10
    active_labels = [labels[i] for i in active_columns]
    active_groups = list(dict.fromkeys(label.split("::")[0] for label in active_labels))
    artifact = dict(incumbent)
    artifact.update({
        "selected_candidate": f"{selection_name}/{chosen}",
        "feature_order": active_labels, "coefficients_and_intercept": compact.tolist(),
        "pruning": {"selected": chosen, "results": results,
                    "selection": "Top group coefficients selected inside each meta-training fold before refitting",
                    "active_groups": active_groups, "active_coefficients": len(active_labels),
                    "removed_zero_coefficients": int(len(theta) - 1 - keep.sum())},
        "cv": results["7"][chosen]["auc"], "cv_test_weighted": results["7"][chosen]["weighted_auc"],
        "target_reached": results["7"][chosen]["auc"] >= .87,
        "reproduction": '.venv/Scripts/python experiments/weight_edit_20261104/finalize.py --predict "new weights/ensemble.json" --output "new weights/submission_replay.csv"',
        "neural_checkpoint_directory": "new weights/neural_head" if use_neural else None,
    })
    artifact["scores_before_pruning"] = incumbent["scores"]
    artifact["scores"] = {str(s): {"baseline": incumbent["scores"][str(s)]["baseline"],
                                  "selected": results[str(s)][chosen]} for s in [7, 29]}
    artifact["improvement"] = {str(s): {m: results[str(s)][chosen][m] - incumbent["scores"][str(s)]["baseline"][m]
                                      for m in ["auc", "weighted_auc"]} for s in [7, 29]}
    if chosen != "unpruned":
        artifact["bootstrap"] = bootstrap(y, original_oof["s7_baseline"], oofs[f"s7_{chosen}"], wt)
    folder = ROOT / "new weights"
    folder.mkdir(exist_ok=True)
    manifest = copy_models(active_groups, folder, use_neural)
    (folder / "model_manifest.json").write_text(json.dumps(manifest, indent=2))
    artifact["packaged_model_files"] = len(manifest["files"])
    artifact["packaged_model_bytes"] = manifest["total_bytes"]
    (folder / "ensemble.json").write_text(json.dumps(artifact, indent=2))
    np.savez(folder / "ensemble_weights.npz", coefficients=compact[:-1], intercept=compact[-1], feature_order=np.array(active_labels))
    sub = write_submission(expit(sparse), tids, folder / "submission.csv")
    write_submission(expit(sparse), tids, ROOT / "submissions/sub_weight_edit_20261104.csv")
    (ROOT / "submissions/sub_weight_edit_20261104.json").write_text(json.dumps(artifact, indent=2))
    (OUT / "simplified_results.json").write_text(json.dumps(artifact, indent=2))
    np.savez(OUT / "pruning_oof.npz", **oofs, y=y, sample_weight=wt)
    replay = predict(folder / "ensemble.json", folder / "submission_replay.csv")
    assert np.max(np.abs(sub.break_probability - replay.break_probability)) < 1e-14
    report = [
        "# New weights — KU-SBB, 2026-11-04", "",
        f"Local meta-CV AUC: **{artifact['cv']:.8f}** (previous **0.85290475**).",
        f"Distribution-weighted AUC: **{artifact['cv_test_weighted']:.8f}** (previous **0.84826319**).",
        "The requested 0.87 AUC was not reached. These are local validation estimates, not a verified Kaggle test score.", "",
        f"The recipe retains **{len(active_groups)} active model groups / {len(active_labels)} nonzero coefficients**; zero-weight contributors were removed with identical predictions.",
        f"Further pruning selected: `{chosen}`. Model-group selection was performed inside each meta-training fold.",
        "The kept changes combine pairwise-logistic edits to MLP output layers with learned, regularized weights for original/reversed/flipped/cropped predictions.", "",
        "Files:", "",
        "- `submission.csv`: 4,000 rows, original sample-submission identifier order.",
        "- `ensemble.json` / `ensemble_weights.npz`: compact coefficients and intercept, feature order, source hashes and validation results.",
        "- `models/`: only retained original tree and sequence model files.",
        "- `neural_head/`: edited MLP checkpoints; encoder tensors are unchanged.",
        "- `model_manifest.json`: file mapping, hashes, and sizes for the copied models.", "",
        "Reproduce the submission from the project root:", "",
        "```powershell", artifact["reproduction"], "```", "",
        "This bundle uses the existing project's feature tables and prediction archives. Raw-series feature generation and exact neural preprocessing still require the original project/data; the small stored rank references are approximate.",
        "The exported submission was reloaded, validated, and reproduced exactly from the saved compact coefficients.", "",
        "Validation: base neural edits use the original five outer folds; edit strength uses only the original inner split. Ensemble estimates use two meta splits on previously generated OOF predictions. They are exploratory and do not constitute independent holdout validation or fully nested retraining of all base models.",
        "No Kaggle submission was sent. Original checkpoints and incumbent submissions are preserved.",
    ]
    (folder / "README.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps({"selected": chosen, "cv": artifact["cv"], "weighted_auc": artifact["cv_test_weighted"],
                      "active_groups": active_groups, "coefficients": len(active_labels),
                      "copied_files": len(manifest["files"]), "bytes": manifest["total_bytes"],
                      "submission_replay_max_error": 0., "target_reached": artifact["target_reached"]}), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2):
        main()
