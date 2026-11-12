"""after the full-recipe training (try4 in the repo) finishes: push the trained model + results to the private repo.

python -m pipeline.publish_try3 --work B:/AI_cpu_087910_20261108/work

#code: already in the repo (the exact recipe the run used)
#model: every fitted model file from the work folder, zipped into <=1.9 GB parts and attached to a private
#       github release "try4-models" (too big for git); try4/MODEL_FILES.json lists every file + sha256
#results: cpu_validation.json, model.json, submission.csv, inference notebook -> try4/, README updated
#feature tables, raw data, logs and step records stay local (they are competition-data derived / huge)
"""
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "blitzict/ku-sbb-2026"
PART = 1_900_000_000
SKIP_DIRS = {"data", "logs", "stages", "clean", "__pycache__", "pipeline", "experiments", "pipeline_helpers"}
SKIP_SUFFIX = {".parquet", ".py", ".pyc", ".log", ".txt.partial"}


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def model_files(work):
    #fitted models and the small learned state they need; no feature tables, no raw data
    for p in sorted(work.rglob("*")):
        rel = p.relative_to(work)
        if not p.is_file() or rel.parts[0] in SKIP_DIRS or p.suffix in SKIP_SUFFIX:
            continue
        if rel.parts[0] == "tune_out" and (len(rel.parts) > 1 and rel.parts[1] == "var"):
            continue                                   #per-view feature tables
        if "streams" in p.name or p.name.startswith("seq_cache") or p.name.startswith("aug_"):
            continue                                   #feature caches
        if p.suffix == ".npz" and p.stem.endswith(("_train", "_test")):
            continue                                   #per-series feature arrays, rebuilt from raw data
        yield rel


def git(*args, cwd):
    subprocess.run(["git", "-c", "user.name=blitzict", "-c", "user.email=blitzict@users.noreply.github.com", *args],
                   cwd=cwd, check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--clone", type=Path, default=None)
    a = p.parse_args()
    work = a.work.resolve()
    report = json.loads((work / "cpu_validation.json").read_text())
    clone = (a.clone or work.parent / "publish_repo").resolve()
    if not clone.exists():
        subprocess.run(["gh", "repo", "clone", REPO, str(clone)], check=True)
    git("pull", "-q", "--rebase", "origin", "main", cwd=clone)

    #1 model files -> zip parts -> release assets
    files = list(model_files(work))
    listing = [{"path": r.as_posix(), "bytes": (work / r).stat().st_size, "sha256": sha(work / r)} for r in files]
    stage = work.parent / "publish_parts"
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir()
    parts, size, z = [], 0, None
    for item in listing:
        if z is None or size + item["bytes"] > PART:
            if z:
                z.close()
            parts.append(stage / f"try3_models_part{len(parts) + 1:02d}.zip")
            z, size = zipfile.ZipFile(parts[-1], "w", zipfile.ZIP_DEFLATED, allowZip64=True), 0
        z.write(work / item["path"], item["path"])
        item["part"] = parts[-1].name
        size += item["bytes"]
    if z:
        z.close()
    auc = report["local_auc"]
    notes = (f"try4: full 0.879 recipe retrained from the raw competition files (RTX 5070 + Intel Arc + CPU).\n"
             f"Local out-of-fold AUC {auc:.5f}. Unzip all parts into the training work folder, then run\n"
             f"python -m pipeline.predict_new --models <work> --series test_series.parquet --out predict_run --csv submission.csv")
    subprocess.run(["gh", "release", "delete", "try4-models", "-R", REPO, "--yes"], capture_output=True)
    subprocess.run(["gh", "release", "create", "try4-models", "-R", REPO, "--title", f"try4 models (AUC {auc:.5f})",
                    "--notes", notes, *map(str, parts)], check=True)

    #2 results + notebook into try3/
    t3 = clone / "try4"
    t3.mkdir(exist_ok=True)
    (t3 / "MODEL_FILES.json").write_text(json.dumps(listing, indent=1))
    for name in ("cpu_validation.json", "model.json"):
        if (work / name).is_file():
            shutil.copy2(work / name, t3 / name)
    sub = work / report["submission"] if not Path(report["submission"]).is_absolute() else Path(report["submission"])
    shutil.copy2(sub, t3 / "submission.csv")
    shutil.copy2(ROOT / "inference.ipynb", t3 / "inference.ipynb")
    for f in ("pipeline/train_gpu.py", "pipeline/gpu_launch.py", "pipeline/predict_new.py", "pipeline/publish_try3.py"):
        shutil.copy2(ROOT / f, clone / f)
    readme = t3 / "README.md"
    text = readme.read_text(encoding="utf8")
    status = (f"**Status:** finished. Local out-of-fold AUC **{auc:.5f}**. {len(listing)} saved model files "
              f"({sum(i['bytes'] for i in listing) / 1e9:.2f} GB) are attached to the private release `try4-models` "
              f"({len(parts)} zip parts), listed with their hashes in `MODEL_FILES.json`. `submission.csv` holds this run's test "
              f"predictions, and `inference.ipynb` loads the models and scores new series.")
    text = text.split("**Status:**")[0].rstrip() + "\n\n" + status + "\n"
    readme.write_text(text, encoding="utf8")
    git("add", "-A", cwd=clone)
    git("commit", "-q", "-m", f"try4 finished: local AUC {auc:.5f}, saved models in release try4-models\n\n"
        "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>", cwd=clone)
    git("push", "-q", "origin", "main", cwd=clone)
    #3 the qualification tag follows the finished try3 (user request)
    git("tag", "-f", "-a", "qualification", "-m", "qualification submission: designated 58102347, try4 trained", "HEAD", cwd=clone)
    git("push", "-q", "-f", "origin", "qualification", cwd=clone)
    print("published", auc, len(listing), "files,", len(parts), "parts", flush=True)


if __name__ == "__main__":
    sys.exit(main())
