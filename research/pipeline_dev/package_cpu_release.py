"""Create a code-only Colab release, with measured or explicitly unverified status.

An explicit allowlist excludes competition data, historical predictions,
model checkpoints, credentials, and research logs. The preserved releases in
the original project are never moved or deleted.
"""
import argparse
import json
from pathlib import Path
import shutil
from pipeline.run_stages import ROOT, digest


def files():
    direct = ["train_087910_cpu.py", "requirements-cpu-full.txt", "CPU_087910.ipynb",
              "pipeline/features_v4.py", "pipeline/build_foundation.py", "pipeline/run_stages.py",
              "pipeline/sync_ports.py", "pipeline/validate_rebuild.py", "pipeline/full_cpu_stages.json",
              "pipeline/offline_hooks/sitecustomize.py", "pipeline/retained_branch_equivalence.json",
              "pipeline/training_config.py", "pipeline/hyperparameters.json",
              "pipeline/hyperparameters.template.json", "pipeline/write_cpu_notebook.py"]
    manifest = json.loads((ROOT / "pipeline/full_cpu_stages.json").read_text())
    # Include the measured recipe's complete source dependencies, including
    # explicitly tracked importlib dispatchers, without unused experiments.
    source = {p for stage in manifest["stages"]
              for p in [stage["argv"][0], *stage["code_inputs"]]}
    direct += ["pipeline/legacy_cpu/" + p for p in source if p != "clean/features.py"]
    direct += [p for p in manifest["prerequisite_inputs"] if p.startswith("tune_out/")
               and p.endswith((".csv", ".json"))]
    return sorted(set(direct))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work", type=Path, default=ROOT / "cpu_088_rebuild/work")
    p.add_argument("--out", type=Path, default=ROOT / "cpu_088_rebuild/colab_release")
    p.add_argument("--preview", action="store_true", help="List package counts without copying or claiming a score")
    p.add_argument("--unverified-source", action="store_true",
                   help="Publish code first, without a measured CPU score or a target-achievement claim")
    args = p.parse_args()
    selected = files()
    if args.preview:
        print(json.dumps(dict(files=len(selected), bytes=sum((ROOT / f).stat().st_size for f in selected),
                              excludes="Raw data, historic features/predictions/checkpoints, credentials, logs"), indent=2))
        return
    report_path = args.work / "cpu_validation.json"
    if args.unverified_source:
        report = dict(status="unverified_source", local_auc=None, target_local_auc=.87910,
                      target_met_at_reported_precision=None, colab_execution_verified=False,
                      note="Code supplied before the full CPU verification run, at the user's request.")
    else:
        report = json.loads(report_path.read_text())
        if report.get("status") != "measured" or not report.get("target_met_at_reported_precision"):
            raise ValueError("For code before verification use --unverified-source; no measured score will be claimed")
        if digest(ROOT / "pipeline/full_cpu_stages.json") != report["manifest_sha256"]:
            raise ValueError("The stage manifest changed after validation")
        for relative, sha in report["source_sha256"].items():
            original = ROOT / "pipeline/features_v4.py" if relative == "clean/features.py" else ROOT / "pipeline/legacy_cpu" / relative
            if digest(original) != sha:
                raise ValueError(f"Source differs from the measured run: {relative}")
    notebook = json.loads((ROOT / "CPU_087910.ipynb").read_text())
    if notebook["metadata"]["cpu_rebuild"]["measured_local_auc"] != report["local_auc"]:
        raise ValueError("Regenerate the notebook from the measured validation report first")
    target = args.out.resolve()
    if not target.is_relative_to(ROOT / "cpu_088_rebuild"):
        raise ValueError("Use a new release directory under cpu_088_rebuild")
    if target.exists() and any(target.iterdir()):
        raise ValueError("Use an empty release directory; existing files are preserved")
    target.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for relative in selected:
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, destination)
        hashes[relative] = digest(destination)
    shutil.copy2(ROOT / "CPU_REBUILD_README.md", target / "README.md")
    status_name = "SOURCE_STATUS.json" if args.unverified_source else "CPU_RUN_RESULT.json"
    (target / status_name).write_text(json.dumps(report, indent=2))
    (target / ".gitignore").write_text("data/\ncpu_run/\nnotebook_settings.json\n.venv/\n__pycache__/\n*.pyc\n.env\n")
    # Keep measured source bytes intact across Windows and Linux checkouts.
    (target / ".gitattributes").write_text("* -text\n")
    for name in ("README.md", status_name, ".gitignore", ".gitattributes"):
        hashes[name] = digest(target / name)
    (target / "PACKAGE_MANIFEST.json").write_text(json.dumps(dict(
        scope="CPU training source and fixed settings only", local_auc=report["local_auc"],
        historical_numerical_inputs=False, files=hashes), indent=2))
    print(f"Prepared {len(selected)} files in {target}; no upload or deletion performed")


if __name__ == "__main__":
    main()
