"""try3 step 1: raw competition files -> the feature tables and law priors try3 trains on.

python try3/build_features.py --data data --work try3_run --workers 4

Runs only the recorded recipe steps (pipeline/full_cpu_stages.json) that produce try3's 35 feature tables
(train and test side) and the seed-7 fold-wise noise-law priors, using the same runner, code and settings as the
full recipe (resumable; completed steps are verified by hash). Then train with:

python try3/train.py --work try3_run --out try3/model
"""
import os
import sys
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[key] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["ONEAPI_DEVICE_SELECTOR"] = "*:cpu"
os.environ["KU_SBB_REBUILD_FIXED"] = "1"
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from pipeline.run_stages import ROOT, run          # noqa: E402
from pipeline.build_foundation import initialize, build   # noqa: E402
from pipeline.sync_ports import sync               # noqa: E402
from pipeline.training_config import prepare       # noqa: E402
from train import TABLES                           # noqa: E402

WS = "tune_out/weight_space"


def table_path(stem):
    if stem in ("bf", "dd", "gh", "grid", "grid2", "harm", "lm", "lr", "lr5", "lrz", "lrz6", "ps", "sn"):
        return f"tune_out/{stem}_{{}}.parquet"
    return f"{WS}/{stem}_{{}}.parquet"


def recipe(stages):
    #the steps whose outputs try3 needs, plus everything those steps read, in recorded order
    targets = [table_path(s).format(p) for s in TABLES for p in ("train", "test")]
    targets += [f"{WS}/law_outer7/fold{k}_{p}" for k in range(5) for p in ("train.parquet", "test.parquet", "rows.npz", "prior.json")]
    prod = {}
    for i, s in enumerate(stages):
        for o in s["outputs"]:
            prod.setdefault(o, i)
    missing = [t for t in targets if t not in prod]
    if missing:
        raise ValueError(f"no recipe step makes {missing[:3]}")
    todo, got = list(targets), set()
    while todo:
        o = todo.pop()
        if o in prod and prod[o] not in got:
            got.add(prod[o])
            todo += stages[prod[o]]["inputs"]
    return [stages[i] for i in sorted(got)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True, help="folder with the four raw competition files")
    p.add_argument("--work", type=Path, default=Path("try3_run"))
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--list", action="store_true", help="only print the steps that would run")
    a = p.parse_args()
    manifest = ROOT / "pipeline/full_cpu_stages.json"
    steps = recipe(json.loads(manifest.read_text())["stages"])
    print(len(steps), "recipe steps", flush=True)
    if a.list:
        print("\n".join(s["id"] for s in steps))
        return
    work = a.work.resolve()
    if not (work / "CPU_REBUILD_WORKSPACE.json").exists():
        initialize(work, a.data.resolve())
    sync(work)
    prepare(work, ROOT / "pipeline/hyperparameters.json", manifest)
    #raw series -> base features + sequence cache (train and test)
    for part in ("train", "test"):
        build(work, part, a.workers, limit=0)
    for stage in steps:
        stage = {**stage, "env": {**stage.get("env", {}), "KU_SBB_WORKERS": str(a.workers)}}
        if Path(stage["argv"][0]).name in {"features_grid.py", "features_grid2.py", "features_dd.py"}:
            stage["argv"] = [*stage["argv"]]
            stage["argv"][2] = str(a.workers)
        run(work, stage)
    print("try3 feature tables ready in", work, flush=True)


if __name__ == "__main__":
    main()
