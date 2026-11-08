"""Parity checks for pipeline.inference tree adapters (train_cv xgb/lgb/cat, train_var+TTA, distill, nested_tree,
wnlaw_nested2, nested_crop, nested_rank with contributions).

1. Builds a tiny synthetic workspace (scratch folder), runs the ORIGINAL training scripts on it with small round counts
   (saving models as they do), so each script writes its own test predictions.
2. Replay parity: adapters read the saved models plus a separate prediction folder holding copies of the test-side
   feature files; outputs must equal the scripts' own test arrays. This is replay of the same test rows, not new raw data.
3. New IDs: the prediction folder is rewritten with a shuffled subset of test rows under new names; outputs must equal
   the replay rows in that order.
No real data, no full fits. Usage: python -m pipeline.check_inference_trees [scratch_dir]
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from pipeline.inference import base_models, source
from pipeline.run_stages import ROOT

LEG = source.LEGACY
S = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp())
W, P1, P2 = S / "trained", S / "predict_same", S / "predict_new"
for d in (W, P1, P2):
    if d.exists():
        shutil.rmtree(d)
rng = np.random.default_rng(7)
N, NT, NA = 300, 60, 90
ids = np.array([f"{i:05d}" for i in range(N)])
tids = np.array([f"t{i:04d}" for i in range(NT)])
aids = np.array([f"a{i:04d}" for i in range(NA)])
y = (rng.random(N) < 0.45).astype(int)
SUF = {"rev": "r", "flip": "f", "revflip": "rf", "c1": "c1", "c2": "c2"}
KINDS = ["rev", "c1", "c2"]
T = ["harmonic", "ar_law", "bounds", "curve", "noise_tests", "partition", "periodic", "phase_law", "trend", "innov_v2", "mixture_law"]
EXTRA = ["jump_law_pulse", "jump_law_state"]


def frame(index, cols, signal=True):
    x = rng.standard_normal((len(index), len(cols))).astype(np.float32)
    if signal and len(index) == N:
        x[:, 0] += y * 0.8
    x[rng.random(x.shape) < 0.03] = np.nan
    return pd.DataFrame(x, index=pd.Index(index, name="id"), columns=cols)


def put(path, df):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


def build():
    (W / "CPU_REBUILD_WORKSPACE.json").parent.mkdir(parents=True)
    (W / "CPU_REBUILD_WORKSPACE.json").write_text("{}")
    O, V, WS = W / "tune_out", W / "tune_out/var", W / "tune_out/weight_space"
    (W / "data").mkdir()
    pd.DataFrame({"id": ids, "label": y}).to_csv(W / "data/train_labels.csv", index=False)
    clean = [f"f{i}" for i in range(10)] + ["sd_pre", "sd_post"]
    put(W / "clean/features_train.parquet", frame(ids, clean)); put(W / "clean/features_test.parquet", frame(tids, clean, False))
    put(O / "aug_features.parquet", frame(aids, clean, False))
    pd.DataFrame({"id": aids, "kind": rng.choice(["shift", "other"], NA, p=[.8, .2]), "source": rng.choice(ids, NA)}).to_csv(O / "aug_meta.csv", index=False)
    blocks = {"grid": "g", "grid2": "h", "dd": "d", "lr": "l"}
    for b, p in blocks.items():
        cols = [f"{p}{i}" for i in range(5)]
        put(O / f"{b}_train.parquet", frame(ids, cols)); put(O / f"{b}_test.parquet", frame(tids, cols, False))
        put(O / f"aug_{b}_features.parquet", frame(aids, cols, False))
    shutil.copy2(ROOT / "tune_out/best_xgb5.json", O / "best_xgb5.json")
    sel = ["f0", "f1", "f2", "f3", "sd_pre", "grid__g0", "grid__g1", "dd__d0", "dd__d2", "f5"]
    pd.DataFrame({"feature": sel}).to_csv(O / "selected_ns3.csv", index=False)
    pd.DataFrame({"feature": sel[:7] + ["grid2__h1", "lr__l0", "not_a_column"]}).to_csv(O / "selected_ns1.csv", index=False)
    for k in KINDS:   # transformed versions read by train_var (ids + suffix)
        for fp, cols in (("clean", clean), ("grid", [f"g{i}" for i in range(5)]), ("dd", [f"d{i}" for i in range(5)])):
            put(V / f"{fp}_train_{k}.parquet", frame([i + SUF[k] for i in ids], cols))
            put(V / f"{fp}_test_{k}.parquet", frame([i + SUF[k] for i in tids], cols, False))
    for t in T + EXTRA:
        cols = [f"{t[:3]}{i}" for i in range(3)]
        put(WS / f"{t}_train.parquet", frame(ids, cols)); put(WS / f"{t}_test.parquet", frame(tids, cols, False))
        for v in ("c1", "c2"):
            if t == "harmonic":
                put(V / f"harm_train_{v}.parquet", frame([i + SUF[v] for i in ids], cols)); put(V / f"harm_test_{v}.parquet", frame([i + SUF[v] for i in tids], cols, False))
            else:
                put(WS / f"view_{v}/{t}_train.parquet", frame(ids, cols)); put(WS / f"view_{v}/{t}_test.parquet", frame(tids, cols, False))
    lcols = [f"w{i}" for i in range(4)]
    for k, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(ids, y)):
        for base in (WS / "law_outer0", WS / "view_c1/law_outer0", WS / "view_c2/law_outer0"):
            put(base / f"fold{k}_train.parquet", frame(ids, lcols)); put(base / f"fold{k}_test.parquet", frame(tids, lcols, False))
        np.savez(WS / f"law_outer0/fold{k}_rows.npz", train_ids=ids[tr], validation_ids=ids[va])
    pd.DataFrame({"id": ids, "part": "pre", "phi1_ar1": rng.random(N)}).to_parquet(O / "generator_segments.parquet")


def run(script, *args, rounds=None):
    env = {**os.environ, "KU_SBB_ROOT": str(W), "KU_SBB_CODE_ROOT": str(W), "KU_SBB_THREADS": "1", "SAVE_WEIGHTS": "1",
           "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": ""}
    if rounds:
        env["KU_SBB_TREE_ROUNDS"] = str(rounds)
    r = subprocess.run([sys.executable, "-B", str(LEG / script), *args], cwd=W, env=env, capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"{script} {args} failed:\n{r.stdout[-2000:]}\n{r.stderr[-3000:]}")


STAGES = [
    ("experiments/v6/train_cv.py", ["--tag", "cvx", "--model", "xgb", "--blocks", "grid,grid2,dd,lr", "--xgb-policy", "lossguide", "--cols", "tune_out/selected_ns1.csv"]),
    ("experiments/v6/train_cv.py", ["--tag", "cvd", "--model", "xgb", "--blocks", "grid,grid2,dd", "--seed", "1", "--refit-mult", "0.2"]),
    ("experiments/v6/train_cv.py", ["--tag", "cvr", "--model", "xgb", "--blocks", "grid,dd", "--xgb-policy", "lossguide", "--extra", '{"objective":"rank:pairwise","rounds":40,"learning_rate":0.03}']),
    ("experiments/v6/train_cv.py", ["--tag", "cvl", "--model", "lgb", "--blocks", "grid,grid2,dd"]),
    ("experiments/v6/train_cv.py", ["--tag", "cvc", "--model", "cat", "--blocks", "grid,grid2,dd", "--cat-iters", "30"]),
    ("experiments/v6/train_var.py", ["--tag", "var1", "--aug", "--kinds", "rev,c1", "--tta-kinds", "rev,c1,c2", "--rounds", "40", "--params", '{"max_bin":64,"rounds":40}']),
    ("experiments/v6/distill.py", ["--tag", "dst", "--cols", "tune_out/selected_ns1.csv", "--rounds", "30"]),
    ("experiments/redesign/nested_tree.py", ["--tag", "NT_s0", "--seed", "0"]),
    ("experiments/redesign/wnlaw_nested2.py", []),
    ("experiments/redesign/nested_crop.py", ["--tag", "NC_s0", "--views", "c1,c2", "--extra", "jump_law_pulse,jump_law_state"]),
    ("experiments/redesign/nested_rank.py", ["--tag", "NR_s0", "--views", "c1", "--extra", "jump_law_pulse,jump_law_state", "--contribs"]),
]


def test_side(path):
    import re   # "noise_tests_train" is a training table: match the word "test" only
    return path.suffix == ".parquet" and re.search(r"(^|_)test(_|\.)", path.name) is not None


def copy_tests(dst):
    for p in W.rglob("*.parquet"):
        if test_side(p):
            t = dst / p.relative_to(W); t.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, t)
    (dst / "CPU_REBUILD_WORKSPACE.json").write_text("{}")


def rewrite_new(dst, order, names):
    """Shuffled subset of test rows under new IDs (suffix variants follow their base ID)."""
    old = tids[order]
    for p in W.rglob("*.parquet"):
        if not test_side(p):
            continue
        df = pd.read_parquet(p)
        sample = str(df.index[0])
        suf = next((s for s in sorted(set(SUF.values()), key=len, reverse=True) if sample.endswith(s) and sample[:-len(s)] in set(tids)), "")
        sel = df.loc[[o + suf for o in old]]
        sel.index = pd.Index([n + suf for n in names], name=df.index.name)
        t = dst / p.relative_to(W); t.parent.mkdir(parents=True, exist_ok=True); sel.to_parquet(t)
    (dst / "CPU_REBUILD_WORKSPACE.json").write_text("{}")


def same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and np.array_equal(a, b)


build()
run("experiments/v6/train_var.py", "--tag", "EXP", "--kinds", "", "--tta-kinds", "", "--colfile", "tune_out/selected_ns3.csv", "--export", "tune_out/var/mat_ns3_plain.npz")
run("experiments/v6/train_var.py", "--tag", "exp_c7", "--aug", "--kinds", "rev,c1,c2", "--tta-kinds", "rev,c1,c2", "--export", "tune_out/var/mat_ns3_c7.npz")
for script, args in STAGES:
    run(script, *args, rounds=40)
copy_tests(P1)
order = rng.permutation(NT)[:37]
names = [f"new{j:03d}" for j in range(len(order))]
rewrite_new(P2, order, names)

results = {}
for script, args in STAGES:
    stage = {"id": Path(script).stem + "_" + (args[1] if args else "w"), "argv": [script, *args], "outputs": []}
    for label, pred, sel in (("replay", P1, slice(None)), ("new_ids", P2, order)):
        out = base_models.ADAPTERS[script](stage, W, pred)
        ok = True
        for name, value in out.items():
            if isinstance(value, dict):
                with np.load(W / f"tune_out/{name}.npz", allow_pickle=True) as z:
                    ref = z["test"]
                    if "groups" in value:
                        ok &= same(z["groups"].astype(str), value["groups"].astype(str))
                ok &= same(ref[..., sel] if ref.shape[-1] == NT and ref.ndim == 2 and name.startswith("ver") else ref[sel], value["test"])
            else:
                ok &= same(np.load(W / f"tune_out/{name}.npy")[sel], value)
        results[f"{stage['id']} {label} {sorted(out)}"] = ok

# Writer: keeps trained OOF fields, replaces test fields, refuses the trained folder.
stage = {"id": "nested_rank_NR_s0", "argv": ["experiments/redesign/nested_rank.py", *STAGES[-1][1]],
         "outputs": ["tune_out/oof_NR_s0.npy", "tune_out/test_NR_s0.npy", "tune_out/ver_NR_s0.npz", "tune_out/redesign/contrib_NR_s0.npz"]}
try:
    base_models.predict_stage(stage, W, W); results["refuses_trained_folder"] = False
except ValueError:
    results["refuses_trained_folder"] = True
stage["outputs"] = ["tune_out/oof_NR_s0.npy", "tune_out/test_NR_s0.npy", "tune_out/ver_NR_s0.npz"]
base_models.predict_stage(stage, W, P2)
with np.load(W / "tune_out/ver_NR_s0.npz") as a, np.load(P2 / "tune_out/ver_NR_s0.npz") as b:
    results["writer_keeps_oof_replaces_test"] = same(a["oof"], b["oof"]) and same(a["test"][:, order], b["test"])
results["writer_copies_oof"] = same(np.load(W / "tune_out/oof_NR_s0.npy"), np.load(P2 / "tune_out/oof_NR_s0.npy"))
# Missing checkpoint fails instead of fitting.
moved = W / "weights/cvx.ubj"; moved.rename(moved.with_suffix(".bak"))
try:
    base_models.ADAPTERS["experiments/v6/train_cv.py"]({"argv": ["experiments/v6/train_cv.py", *STAGES[0][1]]}, W, P1); results["missing_model_raises"] = False
except FileNotFoundError:
    results["missing_model_raises"] = True
moved.with_suffix(".bak").rename(moved)

for k, v in results.items():
    print(("PASS " if v else "FAIL ") + k)
print("scratch:", S)
raise SystemExit(0 if all(results.values()) else 1)
