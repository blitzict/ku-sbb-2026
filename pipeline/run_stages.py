"""Resume explicitly reviewed CPU rebuild stages in a fresh workspace.

The manifest contains an ordered `stages` list. Each stage has `id`, `argv`
(Python script relative to the workspace, followed by arguments), `inputs`,
`outputs`, and optional `code_inputs` and `env`. No shell commands are evaluated.
Files and parameters are hashed, and an exit code alone never marks a stage
complete. The foundation builder is separate because it creates the workspace.

Example:
  python -m pipeline.run_stages --work cpu_088_rebuild/work --manifest pipeline/stages.json --list

For a measured Windows run, wrap this command with pipeline.benchmark_cpu in
training mode. The reviewed manifest is still being assembled from the audit;
the historical candidate-command document is not an executable manifest.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from time import monotonic

ALLOWED_ENV = {"KU_SBB_THREADS", "KU_SBB_WORKERS", "CHUNK", "LRZ_TAG", "SAVE_WEIGHTS", "KU_SBB_REBUILD_FIXED"}
ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            result.update(block)
    return result.hexdigest()


def inside(work, relative):
    path = (work/relative).resolve()
    if not path.is_relative_to(work) or path == work:
        raise ValueError(f"Path escapes the rebuild workspace: {relative}")
    return path


def hashed_files(work, paths):
    result = {}
    for relative in paths:
        path = inside(work, relative)
        if not path.is_file():
            raise FileNotFoundError(f"Missing stage dependency: {path}")
        result[relative] = digest(path)
    return result


def run(work, stage):
    test_overrides = {k for k in ("KU_SBB_TREE_ROUNDS", "KU_SBB_CAT_ITERS", "KU_SBB_NN_EPOCHS")
                      if k in os.environ}
    if test_overrides:
        raise ValueError(f"Remove test-only hyperparameter overrides before the fixed rebuild: {sorted(test_overrides)}")
    ident = stage["id"]
    if not ident.replace("_", "").replace("-", "").isalnum():
        raise ValueError(f"Invalid stage ID: {ident}")
    args = stage["argv"]
    if not args or not all(isinstance(value, str) for value in args):
        raise ValueError("argv must contain a Python script and string arguments")
    script = inside(work, args[0])
    if script.suffix != ".py" or not script.is_file():
        raise ValueError(f"Stage must name a copied Python source file: {script}")
    custom_env = stage.get("env", {})
    if set(custom_env)-ALLOWED_ENV:
        raise ValueError(f"Unreviewed environment settings in stage {ident}")
    inputs = hashed_files(work, stage.get("inputs", []))
    code = hashed_files(work, list(dict.fromkeys([args[0], *stage.get("code_inputs", [])])))
    expected = stage["outputs"]
    if not expected:
        raise ValueError(f"Stage {ident} declares no outputs")
    for relative in expected:
        inside(work, relative).parent.mkdir(parents=True, exist_ok=True)
    fingerprint = dict(argv=args, inputs=inputs, code=code, env=custom_env)
    marker = work/"stages"/(ident+".json")
    if marker.exists():
        old = json.loads(marker.read_text())
        if old.get("fingerprint") != fingerprint:
            raise ValueError(f"Completed stage {ident} has different inputs/code; use a new workspace")
        if old.get("outputs") != hashed_files(work, expected):
            raise ValueError(f"Completed stage {ident} has changed outputs")
        print(f"SKIP {ident}: verified inputs, source and outputs", flush=True)
        return
    # A failed stage can leave resumable fold checkpoints. Bind those partial
    # files to the same inputs and source before allowing the next attempt.
    attempt = work / "stages" / (ident + ".attempt.json")
    if attempt.exists():
        partial = json.loads(attempt.read_text())
        if partial.get("fingerprint") != fingerprint:
            raise ValueError(f"Interrupted stage {ident} has different inputs/code; use a new workspace")
    attempt.write_text(json.dumps(dict(stage=ident, fingerprint=fingerprint, status="running"), indent=2))
    env = os.environ.copy()
    for key in ("CHUNK", "LRZ_TAG"):
        env.pop(key, None)  # Only the stage's reviewed setting may override the script default.
    env.update(KU_SBB_ROOT=str(work), KU_SBB_CODE_ROOT=str(work), KU_SBB_THREADS="2",
               KU_SBB_RAW_ONLY="1", KU_SBB_ORIGINAL_ROOT=str(ROOT), KU_SBB_OFFLINE="1",
               CUDA_VISIBLE_DEVICES="", ONEAPI_DEVICE_SELECTOR="*:cpu", SAVE_WEIGHTS="1",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               PYTHONUNBUFFERED="1")
    env.update({key: str(value) for key, value in custom_env.items()})
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT/"pipeline/offline_hooks"), str(work),
                                         env.get("PYTHONPATH", "")])
    start = monotonic()
    log = work/"logs"/(ident+".log")
    print(f"START {ident}: {script.name}", flush=True)
    with log.open("a", encoding="utf-8") as stream:
        result = subprocess.run([sys.executable, "-B", str(script), *args[1:]],
                                cwd=work, env=env, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Stage {ident} exited {result.returncode}; see {log}")
    outputs = hashed_files(work, expected)
    report = dict(stage=ident, fingerprint=fingerprint, outputs=outputs,
                  seconds=monotonic()-start, exit_code=0)
    marker.write_text(json.dumps(report, indent=2))
    attempt.write_text(json.dumps(dict(stage=ident, fingerprint=fingerprint, status="complete"), indent=2))
    print(f"DONE {ident}: {report['seconds']:.1f}s; outputs hashed", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--through", help="Stop after this stage")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--workers", type=int, choices=range(1, 7),
                        help="Optional CPU feature worker count for training; recorded in each stage")
    args = parser.parse_args()
    work = args.work.resolve()
    if not (work/"CPU_REBUILD_WORKSPACE.json").is_file():
        raise ValueError("A marked fresh CPU workspace is required")
    manifest = json.loads(args.manifest.read_text())
    stages = manifest["stages"]
    ids = [stage["id"] for stage in stages]
    if len(ids) != len(set(ids)):
        raise ValueError("Stage IDs must be unique")
    if args.through is not None and args.through not in ids:
        raise ValueError("Unknown --through stage")
    (work/"stages").mkdir(exist_ok=True)
    (work/"logs").mkdir(exist_ok=True)
    for stage in stages:
        if args.workers is not None:
            stage = {**stage, "env": {**stage.get("env", {}), "KU_SBB_WORKERS": str(args.workers)}}
        if args.list:
            print(stage["id"], stage["argv"])
        else:
            run(work, stage)
        if stage["id"] == args.through:
            break


if __name__ == "__main__":
    main()
