"""try3 (0.87659 public): load the saved models and score new raw series. nothing is trained here.

python try3/predict.py --series new_series.parquet --ref <training work folder> --out predict_try3 --csv submission.csv

--ref is the training work folder (frozen feature code + training-side reference tables some formulas need)
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
WS = "tune_out/weight_space"
SEED = 7                                   #the law-prior folds the models were trained on
MODELS = ["lgb_law", "lgb_law_s2", "xgb_law"]
#steps replaced by slim.py (same code, only the columns the models read; identical values)
SLIM = {"features_004_features_grid": "grid", "features_006_features_grid2": "grid2", "features_009_features_dd": "dd",
        "features_021_features_lrz": "lrz6"}
SKIP_DIRS = {"stages", "logs", "weights", "weights_experiments", "data", "tune_out", "new weights", "submissions", "outputs"}


def stages():
    return json.loads((ROOT / "pipeline/full_cpu_stages.json").read_text())["stages"]


def table_path(stem):
    #where each feature table lives in the work folder
    if stem == "features":
        return "clean/features_{}.parquet"
    if stem in ("bf", "dd", "gh", "grid", "grid2", "harm", "lm", "lr", "lr5", "lrz", "lrz6", "ps", "sn"):
        return f"tune_out/{stem}_{{}}.parquet"
    return f"{WS}/{stem}_{{}}.parquet"


def needed(model):
    #feature tables the saved models use, and the recipe steps that make their test side
    cols = [json.loads((model / f"columns_fold{k}.json").read_text()) for k in range(5)]
    stems = sorted({c.split("__")[0] for d in cols for c in d["top"]})
    targets = [table_path(s).format("test") for s in stems] + [f"{WS}/wnlaw_physical_test.npz"]
    st = stages()
    prod = {}
    for i, s in enumerate(st):
        for o in s["outputs"]:
            prod.setdefault(o, i)
    todo, got = list(targets), set()
    while todo:
        o = todo.pop()
        if o in prod and prod[o] not in got:
            got.add(prod[o])
            todo += st[prod[o]]["inputs"]
    return cols, stems, [st[i] for i in sorted(got)]


def is_reference(stage):
    a = stage["argv"]
    return ("--part" in a and a[a.index("--part") + 1] == "train") or (len(a) > 2 and a[2] == "train" and "white_model_features" in a[0])


def setup(ref, out, recipe):
    #frozen feature code + training-side reference tables, copied (the training folder is never written to)
    def copy(rel):
        (out / rel).parent.mkdir(parents=True, exist_ok=True)
        if not (out / rel).exists():
            shutil.copy2(ref / rel, out / rel)
    for p in ref.rglob("*.py"):
        if p.relative_to(ref).parts[0] not in SKIP_DIRS:
            copy(p.relative_to(ref))
    for p in (ref / "tune_out").glob("*.csv"):
        copy(p.relative_to(ref))
    files = ["data/train_series.parquet", "data/train_labels.csv", "clean/features_train.parquet", "tune_out/seq_cache_train.npz"]
    for s in recipe:
        if is_reference(s):
            files += s["outputs"]
        files += [i for i in s["inputs"] if "train" in i]
    for rel in dict.fromkeys(files):
        if (ref / rel).is_file():
            copy(rel)
    (out / "CPU_REBUILD_WORKSPACE.json").write_text(json.dumps({"mode": "prediction"}))


def env_for(out):
    return {**os.environ, "KU_SBB_ROOT": str(out), "KU_SBB_CODE_ROOT": str(out), "KU_SBB_THREADS": "1",
            "KU_SBB_RAW_ONLY": "1", "KU_SBB_ORIGINAL_ROOT": str(ROOT), "KU_SBB_OFFLINE": "1", "CUDA_VISIBLE_DEVICES": "",
            "ONEAPI_DEVICE_SELECTOR": "*:cpu", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
            "PYTHONUNBUFFERED": "1", "PYTHONPATH": os.pathsep.join([str(ROOT / "pipeline/offline_hooks"), str(out), str(ROOT)])}


def run_stage(out, ref, stage):
    a = stage["argv"]
    (out / "logs").mkdir(exist_ok=True)
    env = {**env_for(out), **{k: str(v) for k, v in stage.get("env", {}).items()}, "KU_SBB_THREADS": "1"}
    if stage["id"] in SLIM:
        cmd = [sys.executable, "-B", str(HERE / "slim.py"), SLIM[stage["id"]], str(out)]
    elif Path(a[0]).name == "mechanism.py" and a[1].startswith("jump_law_"):
        cmd = [sys.executable, "-B", str(HERE / "slim_mech.py"), *a[1:]]           #same code, only the used kernels/columns
    elif Path(a[0]).name == "white_model_features.py" and a[1].startswith("boundary_"):
        cmd = [sys.executable, "-B", str(HERE / "slim_mech.py"), *a[1:]]
    elif Path(a[0]).name == "white_filtered_steps.py":
        cmd = [sys.executable, "-B", str(HERE / "slim_mech.py"), "filtered_step"]
    elif Path(a[0]).name == "component_moves_cpu.py":
        cmd = [sys.executable, "-B", str(HERE / "slim_moves.py"), *a[1:]]          #same script, only the used columns
    elif "test" in a[1:] or "--part" in a or Path(a[0]).name == "copy_table.py":
        cmd = [sys.executable, "-B", str(out / a[0]), *a[1:]]           #the original script, test part only
    else:
        #steps that do train+test in one go: run only their test side
        cmd = [sys.executable, "-B", str(HERE / "predict.py"), "--one-stage", stage["id"], "--ref", str(ref), "--out", str(out),
               "--series", "-"]
    with open(out / "logs" / f"{stage['id']}.log", "w") as log:
        r = subprocess.run(cmd, cwd=out, env=env, stdout=log, stderr=subprocess.STDOUT)
    if r.returncode:
        raise SystemExit(f"{stage['id']} failed, see {out / 'logs' / (stage['id'] + '.log')}")


def run_all(out, ref, recipe, jobs):
    #dependency-ordered, up to `jobs` steps at once (each step single-threaded)
    todo = [s for s in recipe if not is_reference(s)]
    made = {o: s["id"] for s in todo for o in s["outputs"]}
    deps = {s["id"]: {made[i] for i in s["inputs"] if i in made and made[i] != s["id"]} for s in todo}
    done, timing, running = set(), {}, {}
    with ThreadPoolExecutor(jobs) as pool:
        while todo or running:
            for s in [s for s in todo if deps[s["id"]] <= done][:max(jobs - len(running), 0)]:
                todo.remove(s)
                print("step", s["id"], flush=True)
                running[pool.submit(lambda s=s: (time.time(), run_stage(out, ref, s), time.time()))] = s["id"]
            if not running:
                raise SystemExit(f"unresolvable steps: {[s['id'] for s in todo]}")
            finished, _ = wait(running, return_when=FIRST_COMPLETED)
            for f in finished:
                sid = running.pop(f)
                t0, _, t1 = f.result()
                timing[sid] = round(t1 - t0, 1)
                done.add(sid)
    return timing


def foundation(ref, out, series, workers):
    #raw_features.foundation (id mapping, raw test file, sequence cache) with the slim base-feature table
    import hashlib, re
    import ku_workspace
    from pipeline.inference.common import save_npz
    from pipeline.inference.raw_features import rows
    import slim
    original = rows(series)
    training = set(pd.read_csv(ref / "data/train_labels.csv", usecols=["id"], dtype=str).id)
    mapping, mapped, seen = [], [], set()
    for ident, values, boundary in original:
        internal = ident if re.fullmatch(r"[0-9a-fA-F]{8}.*", ident) and ident not in training else hashlib.sha256(("prediction:" + ident).encode()).hexdigest()
        if internal in seen or internal in training:
            raise ValueError("internal id collision")
        seen.add(internal)
        mapping.append(dict(id=ident, internal_id=internal))
        mapped.append((internal, values, boundary))
    (out / "data").mkdir(parents=True, exist_ok=True)
    w = ku_workspace.SeriesWriter(out / "data/test_series.parquet")
    for ident, values, boundary in mapped:
        w.add(ident, values, boundary)
    w.close()
    pd.DataFrame({"id": [x[0] for x in mapped], "break_probability": 0.0}).to_csv(out / "data/sample_submission.csv", index=False)
    (out / "query_ids.json").write_text(json.dumps(mapping, indent=2))
    ids = np.asarray([x[0] for x in mapped], dtype=str)
    padded = np.zeros((len(ids), 1024), dtype=np.float32)
    for i, (_, values, _) in enumerate(mapped):
        padded[i, :len(values)] = values
    save_npz(out / "tune_out/seq_cache_test.npz", ids=ids, x=padded, n=np.asarray([len(x[1]) for x in mapped], dtype=np.int64),
             b=np.asarray([x[2] for x in mapped], dtype=np.int64))
    F = slim.build("base", out, workers)
    if not np.array_equal(F.index.astype(str), ids):
        raise ValueError("base feature order differs from the series order")
    F.to_parquet(out / "clean/features_test.parquet")


def one_stage(stage_id, ref, out):
    from pipeline.inference import common, query_features
    stage = next(s for s in stages() if s["id"] == stage_id)
    common.bind(out, ref, stage["argv"][0])
    query_features.predict_stage(stage, ref, out)


def law_prior(model, out):
    #apply the saved fold priors (fitted on each fold's training labels) to the new series' law values
    sys.path.insert(0, str(out / "experiments/weight_space"))
    from scipy.special import logsumexp
    from white_law_head import unpack
    with np.load(out / WS / "wnlaw_physical_test.npz") as z:
        ids, values = z["ids"].astype(str), z["values"].astype(float)
    folds = []
    for k in range(5):
        prm = json.loads((model / f"law_outer{SEED}/fold{k}_prior.json").read_text())
        _, lik, feats = unpack(values, np.asarray(prm["law"]))
        odds = logsumexp(lik + np.log(prm["change"]), axis=1) - logsumexp(lik + np.log(prm["null"]), axis=1)
        arr = np.column_stack([odds, feats]).clip(-80, 80)
        folds.append(pd.DataFrame(arr, index=ids, columns=[f"law__v{j}" for j in range(arr.shape[1])]).astype(np.float32))
    return folds


def scores(model, out, cols, stems, ids):
    import lightgbm as lgb
    import xgboost as xgb
    frames = []
    for s in stems:
        f = pd.read_parquet(out / table_path(s).format("test"))
        f.index = f.index.astype(str)
        frames.append(f.loc[ids].astype(np.float32).add_prefix(f"{s}__"))
    X = pd.concat(frames, axis=1).replace([np.inf, -np.inf], np.nan)
    law = law_prior(model, out)
    total = {n: np.zeros(len(ids)) for n in MODELS}
    for k in range(5):
        #same column order as training: that fold's top features, then its law-prior columns
        A = pd.concat([X[cols[k]["top"]], law[k].loc[ids][cols[k]["law"]]], axis=1)
        total["lgb_law"] += lgb.Booster(model_file=str(model / f"lgb_law_fold{k}.txt")).predict(A) / 5
        total["lgb_law_s2"] += lgb.Booster(model_file=str(model / f"lgb_law_s2_fold{k}.txt")).predict(A) / 5
        bx = xgb.Booster()
        bx.load_model(str(model / f"xgb_law_fold{k}.json"))
        bx.set_param({"device": "cpu"})
        total["xgb_law"] += bx.predict(xgb.DMatrix(A)) / 5
    return pd.DataFrame(total, index=ids)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--series", required=True)
    p.add_argument("--model", type=Path, default=HERE / "model")
    p.add_argument("--ref", type=Path, default=Path("B:/AI_cpu_087910_20261108/work"))
    p.add_argument("--out", type=Path, default=Path("predict_try3"))
    p.add_argument("--csv", type=Path, default=Path("submission.csv"))
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--one-stage")
    a = p.parse_args()
    ref, out = a.ref.resolve(), a.out.resolve()
    if a.one_stage:
        return one_stage(a.one_stage, ref, out)
    if out.exists() and any(out.iterdir()):
        raise SystemExit("use an empty --out folder")
    t_start = time.time()
    cols, stems, recipe = needed(a.model)
    setup(ref, out, recipe)
    from pipeline.inference import common
    common.bind(out, ref)
    for k, v in env_for(out).items():
        if k.startswith("KU_SBB") or k.endswith("_THREADS"):
            os.environ[k] = v
    sys.path.insert(0, str(HERE))
    #1 raw series -> base features (the 19 used columns) + sequence cache
    print("base features", flush=True)
    t0 = time.time()
    foundation(ref, out, Path(a.series), a.workers)
    timing = {"setup": round(t0 - t_start, 1), "base": round(time.time() - t0, 1)}
    #2 the other feature tables, same scripts as training (slim where possible), test side only, in parallel
    timing.update(run_all(out, ref, recipe, a.workers))
    #3 saved models: 2 lightgbm + 1 xgboost per fold, fold-averaged
    ids = pd.read_parquet(out / "clean/features_test.parquet", columns=[]).index.astype(str)
    sc = scores(a.model, out, cols, stems, ids)
    sc.to_parquet(out / "model_scores.parquet")
    #4 rank-average blend of the 3 models -> id,break_probability
    blend = sc.rank().mean(axis=1) / len(sc)
    back = {m["internal_id"]: m["id"] for m in json.loads((out / "query_ids.json").read_text())}
    pd.DataFrame({"id": [back[i] for i in ids], "break_probability": blend.to_numpy()}).to_csv(a.csv, index=False)
    timing["total"] = round(time.time() - t_start, 1)
    (out / "timing.json").write_text(json.dumps(timing, indent=1))
    print("wrote", a.csv, len(blend), "rows in", timing["total"], "s", flush=True)


if __name__ == "__main__":
    main()
