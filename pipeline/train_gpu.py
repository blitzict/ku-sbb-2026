"""finish the 0.879 training faster: independent steps run in parallel, model fits use the gpu.

python -m pipeline.train_gpu --data data --work B:/AI_cpu_087910_20261108/work --workers 6

same recipe (pipeline/full_cpu_stages.json), same step records as train_087910_cpu.py, so either runner
can resume the other. every fitted model is saved in the work folder as it finishes.
"""
import argparse
import json
import os
import subprocess as real_subprocess
import sys
import threading
import time
import types
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline import run_stages  # noqa: E402

#model scripts that go through the gpu launcher
GPU_SCRIPTS = {"train_cv.py", "train_var.py", "distill.py", "nested_tree.py", "nested_crop.py", "nested_rank.py",
               "wnlaw_nested.py", "wnlaw_nested2.py", "nn_var.py", "nn_xpu.py", "nn_seq.py"}
#late steps read files they don't declare, so they wait for everything before them
BARRIER = ("weight_space/", "weight_edit_20261104/", "fixed_final_readouts", "write_config", "align_predictions",
           "mech_blends", "wtta.py", "late_features", "white_model_features", "component_moves")
WORKER_ARGV = {"features_grid.py", "features_grid2.py", "features_dd.py", "harm_var.py"}


def gpu_run(cmd, cwd=None, env=None, **kw):
    #swap "python -B script args" for "python -B -m pipeline.gpu_launch script args" on model scripts
    script = Path(cmd[2])
    if script.name in GPU_SCRIPTS:
        env = dict(env)
        env.update(CUDA_VISIBLE_DEVICES="0", KU_SBB_GPU="1", ONEAPI_DEVICE_SELECTOR="")
        env["PYTHONPATH"] = os.pathsep.join([env.get("PYTHONPATH", ""), str(ROOT)])
        cmd = [cmd[0], "-B", "-m", "pipeline.gpu_launch", str(script), *cmd[3:]]
    return real_subprocess.run(cmd, cwd=cwd, env=env, **kw)


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


def dependencies(stages):
    #step i waits for earlier steps that write its inputs or the same outputs; late steps wait for all earlier
    writer, deps = {}, []
    for i, s in enumerate(stages):
        if any(b in s["argv"][0] for b in BARRIER):
            d = set(range(i))
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
    p.add_argument("--gpu-jobs", type=int, default=2)
    a = p.parse_args()
    work = a.work.resolve()
    if not (work / "CPU_REBUILD_WORKSPACE.json").is_file():
        raise SystemExit("start with train_087910_cpu.py first; this runner only continues a marked work folder")
    from pipeline.training_config import prepare
    from pipeline.sync_ports import sync
    sync(work)
    config = prepare(work, ROOT / "pipeline/hyperparameters.json", ROOT / "pipeline/full_cpu_stages.json")
    stages = prepared(work, config.manifest["stages"], a.workers)
    deps = dependencies(stages)
    run_stages.subprocess = types.SimpleNamespace(run=gpu_run, STDOUT=real_subprocess.STDOUT)

    done = {i for i, s in enumerate(stages) if (work / "stages" / (s["id"] + ".json")).exists()}
    for i in sorted(done):
        run_stages.run(work, stages[i])   #verifies the finished step, prints SKIP
    gpu_slots = threading.Semaphore(a.gpu_jobs)
    running, failed = {}, []

    def job(i):
        s = stages[i]
        if Path(s["argv"][0]).name in GPU_SCRIPTS:
            with gpu_slots:
                run_stages.run(work, s)
        else:
            run_stages.run(work, s)

    with ThreadPoolExecutor(a.cpu_jobs + a.gpu_jobs) as pool:
        while len(done) < len(stages) and not failed:
            ready = [i for i in range(len(stages)) if i not in done and i not in running.values() and deps[i] <= done]
            cpu_busy = sum(1 for i in running.values() if Path(stages[i]["argv"][0]).name not in GPU_SCRIPTS)
            gpu_busy = len(running) - cpu_busy
            for i in ready:
                is_gpu = Path(stages[i]["argv"][0]).name in GPU_SCRIPTS
                if (is_gpu and gpu_busy >= a.gpu_jobs) or (not is_gpu and cpu_busy >= a.cpu_jobs):
                    continue
                running[pool.submit(job, i)] = i
                gpu_busy += is_gpu
                cpu_busy += not is_gpu
            if not running:
                raise SystemExit("nothing runnable; dependency problem")
            finished, _ = wait(list(running), return_when=FIRST_COMPLETED)
            for f in finished:
                i = running.pop(f)
                if f.exception():
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
