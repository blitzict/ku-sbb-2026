"""Parity checks for pipeline.inference neural adapters: nn_var (x3 sources), neural_head_edit (x3), replay_head_edits,
nn_xpu, nn_seq. Same method as check_inference_trees: the ORIGINAL scripts train on a tiny synthetic workspace (1-2
epochs), then adapters given only saved states/references plus copies of the test-side inputs must reproduce the
scripts' test arrays (replay, bit-exact), and a shuffled subset under new IDs must give the same rows (networks see a
different batch size there, so equality is to float32 rounding; the max difference is printed and bounded by 1e-6).
Usage: python -m pipeline.check_inference_neural [scratch_dir]
"""
import os
for _k in ("KU_SBB_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_k] = "1"   # the training runs below use one thread; bit parity needs the same thread count
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.inference import base_models, source

LEG = source.LEGACY
S = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp())
W, P1, P2 = S / "trained", S / "predict_same", S / "predict_new"
for d in (W, P1, P2):
    if d.exists():
        shutil.rmtree(d)
rng = np.random.default_rng(11)
N, NT, NA, NE = 260, 50, 80, 120
ids = np.array([f"{i:05d}" for i in range(N)]); tids = np.array([f"t{i:04d}" for i in range(NT)])
aids = np.array([f"a{i:04d}" for i in range(NA)]); eids = np.array([f"e{i:04d}" for i in range(NE)])
y = (rng.random(N) < 0.45).astype(int)
SUF = {"rev": "r", "flip": "f", "revflip": "rf", "c1": "c1", "c2": "c2"}
KINDS = list(SUF)


def frame(index, cols, lab=None):
    x = rng.standard_normal((len(index), len(cols))).astype(np.float32)
    if lab is not None:
        x[:, 0] += lab * 0.9
    x[:, 1] = np.round(x[:, 1])
    x[rng.random(x.shape) < 0.04] = np.nan
    return pd.DataFrame(x, index=pd.Index(index, name="id"), columns=cols)


def put(path, df):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


def build():
    W.mkdir(parents=True)
    (W / "CPU_REBUILD_WORKSPACE.json").write_text("{}")
    O, V = W / "tune_out", W / "tune_out/var"
    (W / "data").mkdir()
    pd.DataFrame({"id": ids, "label": y}).to_csv(W / "data/train_labels.csv", index=False)
    clean = [f"f{i}" for i in range(8)]
    put(W / "clean/features_train.parquet", frame(ids, clean, y)); put(W / "clean/features_test.parquet", frame(tids, clean))
    put(O / "aug_features.parquet", frame(aids, clean))
    pd.DataFrame({"id": aids, "kind": "shift", "source": rng.choice(ids, NA)}).to_csv(O / "aug_meta.csv", index=False)
    for b, p in {"grid": "g", "grid2": "h", "dd": "d"}.items():
        cols = [f"{p}{i}" for i in range(4)]
        put(O / f"{b}_train.parquet", frame(ids, cols, y)); put(O / f"{b}_test.parquet", frame(tids, cols)); put(O / f"aug_{b}_features.parquet", frame(aids, cols))
    sel = clean[:6] + ["grid__g0", "grid__g1", "dd__d0", "dd__d3"]
    pd.DataFrame({"feature": sel}).to_csv(V.parent / "cggd.csv", index=False)
    for k in KINDS:
        for fp, cols in (("clean", clean), ("grid", [f"g{i}" for i in range(4)]), ("dd", [f"d{i}" for i in range(4)])):
            put(V / f"{fp}_train_{k}.parquet", frame([i + SUF[k] for i in ids], cols, y)); put(V / f"{fp}_test_{k}.parquet", frame([i + SUF[k] for i in tids], cols))
    for fp, cols in (("clean", clean), ("grid", [f"g{i}" for i in range(4)]), ("dd", [f"d{i}" for i in range(4)])):
        put(V / f"{fp}_train_hv.parquet", frame(eids, cols))
    pd.DataFrame({"id": eids, "kind": rng.choice(["h0", "h1", "p"], NE), "label": rng.integers(0, 2, NE), "source": rng.choice(ids, NE)}).to_csv(V / "hv_meta.csv", index=False)
    L = 1024
    for split, idx in (("train", ids), ("test", tids)):
        n = rng.integers(120, 400, len(idx)); b = (n * rng.uniform(0.3, 0.7, len(idx))).astype(np.int64)
        x = np.zeros((len(idx), L), np.float32)
        for r in range(len(idx)):
            x[r, :n[r]] = rng.standard_normal(n[r]) + (y[r] * 0.8 * (np.arange(n[r]) >= b[r]) if split == "train" else 0)
        np.savez(O / f"seq_cache_{split}.npz", ids=np.array(list(idx), dtype=object), x=x, n=n.astype(np.int64), b=b)


def run(script, *args):
    env = {**os.environ, "KU_SBB_ROOT": str(W), "KU_SBB_CODE_ROOT": str(W), "KU_SBB_THREADS": "1", "SAVE_WEIGHTS": "1",
           "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "CUDA_VISIBLE_DEVICES": ""}
    r = subprocess.run([sys.executable, "-B", str(LEG / script), *args], cwd=W, env=env, capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"{script} {args} failed:\n{r.stdout[-2000:]}\n{r.stderr[-3000:]}")


NNV = ["--mat", "tune_out/var/mat_cggd_hv.npz", "--aug", "--extra", "--kinds", "rev,flip,revflip,c1,c2", "--seeds", "4", "--hidden", "1024", "--layers", "3", "--drop", "0.4", "--epochs", "2", "--patience", "1"]
STAGES = [("experiments/v6/nn_var.py", ["--tag", "MLPHV_big", *NNV]),
          ("experiments/v6/nn_var.py", ["--tag", "MLPHV_big_o1", *NNV, "--seed-offset", "1"]),
          ("experiments/v6/nn_var.py", ["--tag", "MLPHV_big_o2", *NNV, "--seed-offset", "2"]),
          ("experiments/v6/wtta.py", ["--out", "MLPHV_bag_w", "MLPHV_big", "MLPHV_big_o1", "MLPHV_big_o2"]),
          ("experiments/weight_edit_20261104/neural_head_edit.py", ["--source", "MLPHV_big", "--seed-offset", "0", "--run-name", "head_v2_MLPHV_big"]),
          ("experiments/weight_edit_20261104/neural_head_edit.py", ["--source", "MLPHV_big_o1", "--seed-offset", "1", "--run-name", "head_v2_MLPHV_big_o1"]),
          ("experiments/weight_edit_20261104/neural_head_edit.py", ["--source", "MLPHV_big_o2", "--seed-offset", "2", "--run-name", "head_v2_MLPHV_big_o2"]),
          ("experiments/weight_edit_20261104/replay_head_edits.py", []),
          ("experiments/v6/nn_xpu.py", ["--tag", "v6_dd_mlp", "--blocks", "grid,grid2,dd", "--seeds", "2", "--epochs", "2", "--patience", "1"]),
          ("experiments/v6/nn_seq.py", ["--tag", "W_SEQ_t", "--epochs", "1", "--patience", "1", "--qin", "4", "--hq", "2", "--ch", "16", "--seeds", "0", "1", "--threads", "1"])]


def test_side(p):
    return re.search(r"(^|_)test(_|\.)", p.name) is not None and p.suffix in (".parquet", ".npz") and "tune_out/weight_edit" not in p.as_posix()


def copy_tests(dst):
    for p in W.rglob("*"):
        if p.is_file() and test_side(p) and not p.name.startswith(("test_", "ver_")):
            t = dst / p.relative_to(W); t.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, t)
    (dst / "CPU_REBUILD_WORKSPACE.json").write_text("{}")


def rewrite_new(dst, order, names):
    old = tids[order]
    for p in W.rglob("*"):
        if not (p.is_file() and test_side(p)) or p.name.startswith(("test_", "ver_")):
            continue
        t = dst / p.relative_to(W); t.parent.mkdir(parents=True, exist_ok=True)
        if p.suffix == ".npz":
            z = np.load(p, allow_pickle=True)
            np.savez(t, ids=np.array(names, dtype=object), x=z["x"][order], n=z["n"][order], b=z["b"][order]); continue
        df = pd.read_parquet(p); s0 = str(df.index[0])
        suf = next((s for s in sorted(SUF.values(), key=len, reverse=True) if s0.endswith(s) and s0[:-len(s)] in set(tids)), "")
        sel = df.loc[[o + suf for o in old]]; sel.index = pd.Index([n + suf for n in names], name=df.index.name); sel.to_parquet(t)
    (dst / "CPU_REBUILD_WORKSPACE.json").write_text("{}")


def reference_value(name, sel):
    """The training script's own test array for an adapter output name, restricted to sel."""
    if name.endswith("/head_oof") or "/head_fold" in name:
        with np.load(W / f"tune_out/{name}.npz") as z:
            return {k: z[k][..., sel] for k in ("original_test", "edited_test")}
    if name.split("/")[-1].startswith("ver_"):
        with np.load(W / f"tune_out/{name}.npz") as z:
            return {"test": z["test"][:, sel]}
    return np.load(W / f"tune_out/{name}.npy")[sel]


build()
run("experiments/v6/train_var.py", "--tag", "EXP_cggd_hv", "--aug", "--kinds", "rev,flip,revflip,c1,c2", "--tta-kinds", "rev,flip,revflip,c1,c2",
    "--extra", "hv", "--colfile", "tune_out/cggd.csv", "--export", "tune_out/var/mat_cggd_hv.npz")
for script, args in STAGES:
    run(script, *args)
copy_tests(P1)
order = rng.permutation(NT)[:31]
names = [f"new{j:03d}" for j in range(len(order))]
rewrite_new(P2, order, names)

results, worst = {}, {}
for script, args in STAGES:
    if script not in base_models.ADAPTERS:
        continue
    stage = {"id": Path(script).stem + "_" + (args[1] if args else "replay"), "argv": [script, *args], "outputs": []}
    for label, pred, sel in (("replay", P1, slice(None)), ("new_ids", P2, order)):
        out = base_models.ADAPTERS[script](stage, W, pred)
        exact, diff = True, 0.0
        for name, value in out.items():
            ref = reference_value(name, sel)
            pairs = [(ref[k], value[k]) for k in ref] if isinstance(ref, dict) else [(ref, value)]
            for r, v in pairs:
                r, v = np.asarray(r, float), np.asarray(v, float)
                exact &= r.shape == v.shape and np.array_equal(r, v)
                diff = max(diff, float(np.abs(r - v).max()) if r.shape == v.shape else np.inf)
        key = f"{stage['id']} {label} {sorted(out)}"
        results[key] = exact if label == "replay" else diff <= 1e-6
        worst[key] = diff
miss = W / "weights/v6_dd_mlp/seed100_rankgauss_ref.npz"; miss.rename(miss.with_suffix(".bak"))
try:
    base_models.ADAPTERS["experiments/v6/nn_xpu.py"]({"argv": ["experiments/v6/nn_xpu.py", *STAGES[8][1]]}, W, P1); results["missing_reference_raises"] = False
except FileNotFoundError:
    results["missing_reference_raises"] = True
miss.with_suffix(".bak").rename(miss)
for k, v in results.items():
    print(("PASS " if v else "FAIL ") + k + (f"  max|diff|={worst[k]:.3g}" if k in worst else ""))
print("scratch:", S)
raise SystemExit(0 if all(results.values()) else 1)
