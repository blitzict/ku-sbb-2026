"""Measure the fresh CPU model against raw labels and validate its submission.

This reports the project's historical local output-fold protocol. It is not
a new unseen holdout: model and feature choices were selected historically.
The Kaggle public score cannot be measured from the local training labels.
"""
import argparse
import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from pipeline.run_stages import digest

TAG = "local_rank_shared_residual_response"


def validate(work, manifest, target=.87910):
    work = Path(work).resolve()
    marker = json.loads((work / "CPU_REBUILD_WORKSPACE.json").read_text())
    if marker.get("original_prediction_archives_used") is not False:
        raise ValueError("Validation requires the marked fresh raw-data rebuild")
    stages = json.loads(Path(manifest).read_text())["stages"]
    completed = {}
    checked, sources = {}, {}

    def check_hashes(expected):
        for relative, sha in expected.items():
            if relative not in checked:
                checked[relative] = digest(work / relative)
            if checked[relative] != sha:
                raise ValueError(f"Rebuild input, source, or output changed: {relative}")

    for split, rows in (("train", 24000), ("test", 4000)):
        foundation = json.loads((work / f"stages/foundation_{split}.json").read_text())
        if foundation["rows"] != rows or foundation.get("raw_data_only") is not True or foundation["input"]["limit"]:
            raise ValueError("A complete raw-data foundation is required")
        check_hashes(foundation["outputs"])
        check_hashes({f"data/{split}_series.parquet": foundation["input"]["raw_sha256"]})
        completed["foundation_" + split] = foundation["seconds"]
    check_hashes({"data/" + k: v for k, v in marker["raw_input_sha256"].items()})
    for stage in stages:
        ident = stage["id"]
        path = work / "stages" / (ident + ".json")
        if not path.is_file():
            raise ValueError(f"Cannot declare a complete rebuild: stage {ident} is missing")
        record = json.loads(path.read_text())
        if record.get("exit_code") != 0:
            raise ValueError(f"Stage did not complete successfully: {ident}")
        fingerprint = record["fingerprint"]
        actual_args, expected_args = fingerprint["argv"][:], stage["argv"][:]
        if Path(expected_args[0]).name in {"features_grid.py", "features_grid2.py", "features_dd.py", "harm_var.py"}:
            actual_args[2] = expected_args[2]  # Worker count affects execution speed only.
        if actual_args != expected_args:
            raise ValueError(f"Different training arguments in completed stage: {ident}")
        if set(fingerprint["inputs"]) != set(stage["inputs"]):
            raise ValueError(f"Different declared inputs in completed stage: {ident}")
        if set(fingerprint["code"]) != set([stage["argv"][0], *stage["code_inputs"]]):
            raise ValueError(f"Different source closure in completed stage: {ident}")
        for key, value in stage.get("env", {}).items():
            if key != "KU_SBB_WORKERS" and str(fingerprint["env"].get(key)) != str(value):
                raise ValueError(f"Different environment setting {key}: {ident}")
        if set(record["outputs"]) != set(stage["outputs"]):
            raise ValueError(f"Different declared outputs in completed stage: {ident}")
        check_hashes(fingerprint["inputs"])
        check_hashes(fingerprint["code"])
        check_hashes(record["outputs"])
        sources.update(fingerprint["code"])
        completed[ident] = record["seconds"]
    labels = pd.read_csv(work / "data/train_labels.csv", dtype={"id": str}).set_index("id")
    sample = pd.read_csv(work / "data/sample_submission.csv", dtype={"id": str})
    folder = work / f"new weights/wn_{TAG}_candidate"
    csv = folder / "submission.csv"
    prediction = pd.read_csv(csv, dtype={"id": str})
    if prediction.columns.tolist() != sample.columns.tolist():
        raise ValueError("Submission columns differ from the raw sample")
    if not prediction.id.is_unique or not prediction.id.equals(sample.id):
        raise ValueError("Submission identifiers or row order differ from the raw sample")
    values = prediction["break_probability"].to_numpy()
    if not np.isfinite(values).all() or not ((values >= 0) & (values <= 1)).all():
        raise ValueError("Invalid submission probabilities")
    scores = {}
    archive = work / f"tune_out/weight_space/{TAG}_oof.npz"
    with np.load(archive) as data:
        ids = pd.Index(data["ids"].astype(str))
        if not ids.is_unique or len(ids) != len(labels) or set(ids) != set(labels.index):
            raise ValueError("OOF identifiers do not cover the raw labels exactly once")
        y = labels.loc[ids, "label"].to_numpy()
        for key in data.files:
            if key == "ids":
                continue
            score = data[key]
            if score.shape != y.shape or not np.isfinite(score).all():
                raise ValueError(f"Invalid OOF scores: {key}")
            scores[key] = float(roc_auc_score(y, score))
    measured = scores["s7"]
    report = dict(
        status="measured", device="cpu", training_origin="raw competition files",
        target_local_auc=target, local_auc=measured, local_auc_rounded_5=round(measured, 5),
        target_met_at_reported_precision=round(measured, 5) >= target,
        protocol="Historical output-fold protocol, primary StratifiedKFold seed 7; fixed historical selections",
        unseen_holdout=False, public_leaderboard_auc=None, output_partition_aucs=scores,
        train_series=len(labels), test_series=len(prediction), completed_stages=len(completed),
        summed_stage_seconds=sum(completed.values()),
        submission=str(csv), submission_sha256=digest(csv), oof_sha256=digest(archive),
        manifest_sha256=digest(Path(manifest)), source_sha256=sources,
        raw_input_sha256=marker["raw_input_sha256"],
        checked_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    path = work / "cpu_validation.json"
    path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--target", type=float, default=.87910)
    args = parser.parse_args()
    validate(args.work, args.manifest, args.target)
