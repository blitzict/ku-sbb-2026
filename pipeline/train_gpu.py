"""finish the 0.879 training faster: independent steps run in parallel, model fits use the gpu.

python -m pipeline.train_gpu --data data --work cpu_run --workers 6 --parallel-late

same recipe (pipeline/full_cpu_stages.json), same step records as train_087910_cpu.py, so either runner
can resume the other. every fitted model is saved in the work folder as it finishes.
"""
import argparse
import json
import os
import queue
import re
import subprocess as real_subprocess
import sys
import time
import types
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline import run_stages  # noqa: E402

#model scripts that go through the gpu launcher: xgboost ones on the rtx 5070, torch nets on the intel arc
NV_SCRIPTS = {"train_cv.py", "train_var.py", "distill.py", "nested_tree.py", "nested_crop.py", "nested_rank.py",
              "wnlaw_nested.py", "wnlaw_nested2.py"}
ARC_SCRIPTS = {"nn_var.py", "nn_xpu.py", "nn_seq.py"}   #torch feature scripts stay on cpu: tested ~20x slower on the arc (many tiny ops)
GPU_SCRIPTS = NV_SCRIPTS | ARC_SCRIPTS
XPU_PYTHON = ROOT / ".venv-xpu/Scripts/python.exe"


#late torch heads (weight_space/, weight_edit) hard-code device='cpu'; gpu_launch swaps it, results come back via .cpu()
LATE_DIRS = ("weight_space/", "weight_edit_20261104/")
WORK = None
DEVICES = queue.Queue()   #one token per gpu: two torch jobs at once, one on each card
for _d in ("cuda", "xpu"):
    DEVICES.put(_d)


#wrappers whose torch code sits in imported weight_space modules; their replay checks compare against the
#gpu-trained sequence models, so they run on the gpu too
LATE_WRAPPERS = {"replay_trend_sequence.py", "white_channel_replay.py"}


def late_torch(script):
    script = Path(script)
    if script.name in LATE_WRAPPERS:
        return True
    if not any(d in script.as_posix() for d in LATE_DIRS) or not script.is_file():
        return False
    t = script.read_text(encoding="utf8")
    return re.search(r"^(import|from) torch", t, re.M) is not None and re.search(r"""device=(['"])cpu\1""", t) is not None


def kind(stage):
    name = Path(stage["argv"][0]).name
    if name in NV_SCRIPTS:
        return "nv"
    if name in ARC_SCRIPTS or late_torch(WORK / stage["argv"][0]):
        return "arc"
    return "cpu"
#late steps read files they don't declare, so they wait for everything before them
BARRIER = ("weight_space/", "weight_edit_20261104/", "fixed_final_readouts", "write_config", "align_predictions",
           "mech_blends", "wtta.py", "late_features", "white_model_features", "component_moves")
WORKER_ARGV = {"features_grid.py", "features_grid2.py", "features_dd.py", "harm_var.py"}


def gpu_run(cmd, cwd=None, env=None, **kw):
    #swap "python -B script args" for "python -B -m pipeline.gpu_launch script args" on model scripts
    script = Path(cmd[2])
    late = late_torch(script)
    if script.name not in GPU_SCRIPTS and not late:
        return real_subprocess.run(cmd, cwd=cwd, env=env, **kw)
    env = dict(env)
    env.update(KU_SBB_GPU="1")
    if late:
        env["KU_SBB_DEVICE_SWAP"] = "1"
    env["PYTHONPATH"] = os.pathsep.join([env.get("PYTHONPATH", ""), str(ROOT)])
    launch = ["-B", "-m", "pipeline.gpu_launch", str(script), *cmd[3:]]
    if script.name in NV_SCRIPTS:
        cuda = {**env, "CUDA_VISIBLE_DEVICES": "0", "KU_SBB_TORCH_DEV": "cuda", "ONEAPI_DEVICE_SELECTOR": ""}
        return real_subprocess.run([cmd[0], *launch], cwd=cwd, env=cuda, **kw)
    dev = DEVICES.get()
    try:
        if dev == "xpu" and XPU_PYTHON.is_file():
            arc = {**env, "KU_SBB_TORCH_DEV": "xpu", "CUDA_VISIBLE_DEVICES": ""}
            arc.pop("ONEAPI_DEVICE_SELECTOR", None)
            r = real_subprocess.run([str(XPU_PYTHON), *launch], cwd=cwd, env=arc, **kw)
        else:
            cuda = {**env, "CUDA_VISIBLE_DEVICES": "0", "KU_SBB_TORCH_DEV": "cuda", "ONEAPI_DEVICE_SELECTOR": ""}
            r = real_subprocess.run([cmd[0], *launch], cwd=cwd, env=cuda, **kw)
    finally:
        DEVICES.put(dev)
    if r.returncode == 0:
        return r
    #e.g. the big MLPs keep their rank-gauss data on the cpu: run the unmodified cpu recipe for that step
    print(f"{dev} run of {script.name} failed, running it on the cpu", flush=True)
    return real_subprocess.run(cmd, cwd=cwd, env={k: v for k, v in env.items() if k not in ("KU_SBB_GPU", "KU_SBB_DEVICE_SWAP")}, **kw)


def prepared(work, stages, workers):
    #same per-step execution settings as train_087910_cpu.py so the step records match
    out = []
    for stage in stages:
        stage = {**stage, "env": {**stage.get("env", {}), "KU_SBB_WORKERS": str(workers)}}
        name = Path(stage["argv"][0]).name
        if name in WORKER_ARGV:
            stage["argv"] = [*stage["argv"]]
            stage["argv"][2] = str(workers)
        record = work / "stages" / (stage["id"] + ".json")
        if record.exists():
            fp = json.loads(record.read_text())["fingerprint"]
            if name in WORKER_ARGV and fp["argv"][:2] == stage["argv"][:2] and fp["argv"][3:] == stage["argv"][3:]:
                stage["argv"][2] = fp["argv"][2]
            if fp["env"].get("KU_SBB_WORKERS") is not None:
                stage["env"]["KU_SBB_WORKERS"] = fp["env"]["KU_SBB_WORKERS"]
        out.append(stage)
    return out


#late steps that scan whole folders always wait for everything before them
SERIAL = ("simplify.py",)


def dependencies(stages, relaxed=False):
    #step i waits for earlier steps that write its inputs or the same outputs; late steps wait for all earlier.
    #relaxed: late steps start once their declared inputs and every earlier early step are done; their only
    #optional reads are their own resume checkpoints, so a missing undeclared file fails the step and it is
    #rerun after everything before it (see main)
    writer, deps = {}, []
    for i, s in enumerate(stages):
        late = any(b in s["argv"][0] for b in BARRIER)
        if late and (not relaxed or any(x in s["argv"][0] for x in SERIAL)):
            d = set(range(i))
        elif late:
            d = {j for j in range(i) if not any(b in stages[j]["argv"][0] for b in BARRIER)}
            d |= {writer[x] for x in s.get("inputs", []) + s["outputs"] if x in writer}
        else:
            d = {writer[x] for x in s.get("inputs", []) + s["outputs"] if x in writer}
        deps.append(d)
        for o in s["outputs"]:
            writer[o] = i
    return deps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--cpu-jobs", type=int, default=3)
    p.add_argument("--gpu-jobs", type=int, default=2, help="parallel xgboost fits on the rtx 5070")
    p.add_argument("--arc-jobs", type=int, default=2, help="parallel torch nets, one per gpu (rtx 5070 + intel arc)")
    p.add_argument("--parallel-late", action="store_true", help="run independent late steps at the same time")
    a = p.parse_args()
    work = a.work.resolve()
    global WORK
    WORK = work
    if not (work / "CPU_REBUILD_WORKSPACE.json").is_file():
        raise SystemExit("start with train_087910_cpu.py first; this runner only continues a marked work folder")
    from pipeline.training_config import prepare
    from pipeline.sync_ports import sync
    sync(work)
    config = prepare(work, ROOT / "pipeline/hyperparameters.json", ROOT / "pipeline/full_cpu_stages.json")
    stages = prepared(work, config.manifest["stages"], a.workers)
    deps = dependencies(stages, a.parallel_late)
    full = dependencies(stages)
    retried = set()
    run_stages.subprocess = types.SimpleNamespace(run=gpu_run, STDOUT=real_subprocess.STDOUT)

    done = {i for i, s in enumerate(stages) if (work / "stages" / (s["id"] + ".json")).exists()}
    for i in sorted(done):
        run_stages.run(work, stages[i])   #verifies the finished step, prints SKIP
    limit = {"cpu": a.cpu_jobs, "nv": a.gpu_jobs, "arc": a.arc_jobs}
    running, failed = {}, []

    with ThreadPoolExecutor(sum(limit.values())) as pool:
        while len(done) < len(stages) and not failed:
            busy = {k: 0 for k in limit}
            for i in running.values():
                busy[kind(stages[i])] += 1
            for i in range(len(stages)):
                if i in done or i in running.values() or not deps[i] <= done:
                    continue
                k = kind(stages[i])
                if busy[k] >= limit[k]:
                    continue
                running[pool.submit(run_stages.run, work, stages[i])] = i
                busy[k] += 1
            if not running:
                raise SystemExit("nothing runnable; dependency problem")
            finished, _ = wait(list(running), return_when=FIRST_COMPLETED)
            for f in finished:
                i = running.pop(f)
                if f.exception() and i not in retried and deps[i] != full[i]:
                    #started early: rerun it once everything before it is done
                    print(f"{stages[i]['id']} failed when started early, rerunning it after all earlier steps", flush=True)
                    retried.add(i)
                    deps[i] = full[i]
                elif f.exception():
                    failed.append((stages[i]["id"], f.exception()))
                else:
                    done.add(i)
                    print(f"[{len(done)}/{len(stages)}] {time.strftime('%H:%M')} done {stages[i]['id']}", flush=True)
        wait(list(running))
    if failed:
        raise SystemExit(f"failed: {failed}")
    from pipeline.validate_rebuild import validate
    import train_087910_cpu as T
    report = validate(work, config.manifest_path)
    saved = T.save_model_record(work, config.manifest_path, report)
    print(f"local AUC: {report['local_auc']:.8f}\nsubmission: {report['submission']}\nsaved ensemble: {saved.parent}")


if __name__ == "__main__":
    main()
