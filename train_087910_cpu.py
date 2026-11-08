"""Train the fixed competition ensemble from raw files, entirely on CPU.

Usage (Python 3.12 or later):
    python train_087910_cpu.py --data data --work cpu_run

Four raw files belong in --data: train_series.parquet, test_series.parquet,
train_labels.csv, and sample_submission.csv. The fixed feature lists and
hyperparameters ship with the code. Historical features, predictions and
model checkpoints are not inputs to training.

The run is resumable: repeat the same command after interruption. Completed
stages are reused only when their input, source and output hashes match.
Changing a training setting requires a new work directory.

The notebook calls this one entry point. Its supporting modules preserve the
full recorded ensemble rather than substituting a smaller, weaker model.
"""
import os
import sys
if sys.version_info < (3, 12):
    raise SystemExit("This fixed CPU recipe requires Python 3.12 or newer.")
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["ONEAPI_DEVICE_SELECTOR"] = "*:cpu"
os.environ["KU_SBB_REBUILD_FIXED"] = "1"

import argparse
import json
from pathlib import Path
from pipeline.run_stages import ROOT, run, digest
from pipeline.build_foundation import initialize, build
from pipeline.sync_ports import sync
from pipeline.validate_rebuild import validate
from pipeline.training_config import prepare, plan as configuration_plan


def feature_boundary(stages):
    """The recorded preprocessing prefix ends at the first detector fit."""
    return next(i for i, stage in enumerate(stages) if stage["id"].startswith("models_"))


def save_model_record(work, manifest, report):
    """The fitted ensemble is a directory; its checkpoints are saved during fit.

    Record its recipe and configuration beside the actual weights. Keep this
    whole directory: model.json alone is not a serialized estimator.
    """
    recipe = work / "training_recipe.json"
    recipe.write_bytes(Path(manifest).read_bytes())
    record = dict(format="cpu-competition-ensemble-directory", device="cpu",
                  local_auc=report["local_auc"], recipe="training_recipe.json",
                  configuration="TRAINING_CONFIG.json", validation="cpu_validation.json",
                  submission=Path(report["submission"]).relative_to(work).as_posix(),
                  checkpoint_directories=["weights", "tune_out", "new weights"],
                  note="Keep the whole model directory, including fitted preprocessing and source. "
                       "The separate new-series prediction interface is not included in this source release.")
    path = work / "model.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(record, indent=2), encoding="utf-8")
    temporary.replace(path)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Directory containing the four raw competition files")
    parser.add_argument("--work", type=Path, default=Path("cpu_run"), help="New or previously marked training directory")
    parser.add_argument("--workers", type=int, choices=range(1, 7), default=2, help="Feature workers; affects speed, not model hyperparameters")
    parser.add_argument("--manifest", type=Path, default=ROOT / "pipeline/full_cpu_stages.json")
    parser.add_argument("--config", type=Path, default=ROOT / "pipeline/hyperparameters.json",
                        help="Feature selections and training hyperparameters (JSON)")
    parser.add_argument("--phase", choices=("all", "features", "models"), default="all",
                        help="Notebook split: apply recorded transformations, then train/combine; default runs both")
    args = parser.parse_args()
    if not args.manifest.is_file():
        raise RuntimeError("The full verified stage manifest has not been packaged yet")
    specification = json.loads(args.manifest.read_text())
    if specification.get("complete_dependency_graph") is not True:
        raise RuntimeError("Refusing to train with an incomplete dependency graph")
    proposed = configuration_plan(args.config, args.manifest)
    work, data = args.work.resolve(), args.data.resolve()
    marker = work / "CPU_REBUILD_WORKSPACE.json"
    boundary = feature_boundary(proposed.manifest["stages"])
    if args.phase == "models":
        required = [work / f"stages/foundation_{part}.json" for part in ("train", "test")]
        required += [work / "stages" / (stage["id"] + ".json")
                     for stage in proposed.manifest["stages"][:boundary]]
        if not marker.is_file() or any(not path.is_file() for path in required):
            raise ValueError("Run the transformations cell (--phase features) first")
    if marker.exists():
        configuration_record = work / "TRAINING_CONFIG.json"
        if configuration_record.exists():
            configured = json.loads(configuration_record.read_text())
            if (configured["config_sha256"] != proposed.config_sha256
                    or configured["base_manifest_sha256"] != digest(args.manifest)):
                raise ValueError("Training settings changed; use a new work folder")
        elif not proposed.default and any(
                isinstance(value := json.loads(path.read_text()), dict) and "fingerprint" in value
                for path in (work / "stages").glob("*.json")):
            raise ValueError("Custom training settings need a fresh work folder")
        # Do not call initialize on resume: it copies source before checking
        # completed-stage fingerprints. sync() below protects frozen source.
        old = json.loads(marker.read_text())
        for name, expected in old["raw_input_sha256"].items():
            if digest(data / name) != expected or digest(work / "data" / name) != expected:
                raise ValueError(f"Raw input changed: {name}; use a new work directory")
    else:
        initialize(work, data)
    sync(work)
    configuration = prepare(work, args.config, args.manifest)
    workers = configuration.workers or args.workers
    specification = configuration.manifest
    if args.phase != "models":
        for part in ("train", "test"):
            build(work, part, workers, limit=0)
    stages = specification["stages"]
    if args.phase == "features":
        stages = stages[:boundary]
    elif args.phase == "models":
        stages = stages[boundary:]
    for stage in stages:
        stage = {**stage, "env": {**stage.get("env", {}), "KU_SBB_WORKERS": str(workers)}}
        # Reviewed worker argv is an execution setting, never a fitted setting.
        name = Path(stage["argv"][0]).name
        if name in {"features_grid.py", "features_grid2.py", "features_dd.py", "harm_var.py"}:
            stage["argv"] = [*stage["argv"]]
            stage["argv"][2] = str(workers)
        # Completed feature stages may have used fewer workers. Retain those
        # execution settings so changing speed does not discard valid work.
        record = work / "stages" / (stage["id"] + ".json")
        if record.exists():
            fingerprint = json.loads(record.read_text())["fingerprint"]
            if name in {"features_grid.py", "features_grid2.py", "features_dd.py", "harm_var.py"}:
                old_args = fingerprint["argv"]
                if old_args[:2] == stage["argv"][:2] and old_args[3:] == stage["argv"][3:]:
                    stage["argv"][2] = old_args[2]
            old_workers = fingerprint["env"].get("KU_SBB_WORKERS")
            if old_workers is not None:
                stage["env"]["KU_SBB_WORKERS"] = old_workers
        run(work, stage)
    if args.phase == "features":
        print("Recorded transformations saved. Continue with --phase models.")
        return
    report = validate(work, configuration.manifest_path)
    saved = save_model_record(work, configuration.manifest_path, report)
    print(f"CPU local AUC: {report['local_auc']:.8f}")
    print(f"Submission: {report['submission']}")
    print(f"Saved ensemble: {saved.parent} (keep the whole directory)")
    if not report["target_met_at_reported_precision"]:
        print("The measured run did not reach 0.87910. The report preserves the actual score.")


if __name__ == "__main__":
    main()
