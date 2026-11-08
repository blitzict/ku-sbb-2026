"""Parity checks for combiner adapters: wtta, mech_blends, align_predictions, white_source_bag, write_config (train-only),
stack_edit and view_weight_edit (saved coefficients). Synthetic member predictions; the ORIGINAL scripts run first and
fit/write their outputs, then base_models.predict_stage recomputes every test output in a separate prediction folder,
chained in stage order. Replay must be bit-exact; a shuffled subset under new IDs must give the replay rows (the stack
matrix products see a different row count, so equality is to float rounding, bound 1e-12).
Usage: python -m pipeline.check_inference_combiners [scratch_dir]
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

from pipeline.inference import base_models, source

S = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp())
W, P1, P2 = S / "trained", S / "predict_same", S / "predict_new"
for d in (W, P1, P2):
    if d.exists():
        shutil.rmtree(d)
rng = np.random.default_rng(5)
N, NT = 400, 70
ids = np.array([f"{i:05d}" for i in range(N)]); tids = np.array([f"t{i:04d}" for i in range(NT)])
y = (rng.random(N) < 0.45).astype(int)
O = W / "tune_out"
C8 = ["orig", "rev", "flip", "revflip", "c1", "c2", "c3", "c4"]; C6 = C8[:6]


def prob(n, signal=None):
    z = rng.standard_normal(n) + (0 if signal is None else 1.2 * signal)
    return (1 / (1 + np.exp(-z))).astype(np.float64)


def member(tag):
    np.save(O / f"oof_{tag}.npy", prob(N, y)); np.save(O / f"test_{tag}.npy", prob(NT))


def ver(tag, kinds):
    np.savez(O / f"ver_{tag}.npz", oof=np.stack([prob(N, y) for _ in kinds]).astype(np.float32),
             test=np.stack([prob(NT) for _ in kinds]).astype(np.float32), kinds=np.array(kinds))


def build():
    (W / "data").mkdir(parents=True); O.mkdir(); (O / "weight_space").mkdir(); (W / "clean").mkdir()
    (W / "CPU_REBUILD_WORKSPACE.json").write_text("{}")
    pd.DataFrame({"id": ids, "label": y}).to_csv(W / "data/train_labels.csv", index=False)
    pd.DataFrame(index=pd.Index(ids, name="id")).assign(f=0.0).to_parquet(W / "clean/features_train.parquet")
    pd.DataFrame(index=pd.Index(tids, name="id")).assign(f=0.0).to_parquet(W / "clean/features_test.parquet")
    np.savez(O / "var/mat_ns3_plain.npz" if (O / "var").mkdir() or True else None, ids=ids, tid=tids, cols=np.array(["f"]))
    for t in ["VARC4hv_ns3_s0", "VARC4hv_ns3_s1"]:
        ver(t, C8)
    for t in ["MLPHV_big", "MLPHV_big_o1", "MLPHV_big_o2"]:
        ver(t, C6)
    ver("MECH3C12RV_s0", ["orig", "c1", "c2", "rev"])
    for t in ["m_a", "m_b", "MECH3C12V_s0", "MECH3C12V_s1", "MECH3C12V_s2", "VAR_aug1600_tta", "VAR_aug1600_s1_tta", "MECHNEST", "MECHNEST2"]:
        member(t)


def run(script, *args):
    env = {**os.environ, "KU_SBB_ROOT": str(W), "KU_SBB_CODE_ROOT": str(W), "KU_SBB_THREADS": "1", "OMP_NUM_THREADS": "1",
           "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "PYTHONPATH": str(source.LEGACY)}
    r = subprocess.run([sys.executable, "-B", str(source.LEGACY / script), *args], cwd=W, env=env, capture_output=True, text=True)
    if r.returncode:
        sys.exit(f"{script} {args} failed:\n{r.stdout[-2000:]}\n{r.stderr[-3000:]}")


MEMBERS = ["m_a", "m_b", "VARC4hv_ns3_wc", "MLPHV_bag_w"]
STAGES = [
    ("experiments/v6/wtta.py", ["--crop-w", "0.125", "--out", "VARC4hv_ns3_wc", "VARC4hv_ns3_s0", "VARC4hv_ns3_s1"]),
    ("experiments/v6/wtta.py", ["--out", "MLPHV_bag_w", "MLPHV_big", "MLPHV_big_o1", "MLPHV_big_o2"]),
    ("experiments/redesign/mech_blends.py", ["MECH3C12V_bag"]),
    ("experiments/redesign/mech_blends.py", ["VAR_aug1600_avg_tta"]),
    ("experiments/redesign/mech_blends.py", ["MECH3C12RV_revlow"]),
    ("pipeline_helpers/align_predictions.py", ["MECHNEST", "claude_law"]),
    ("pipeline_helpers/align_predictions.py", ["MECHNEST2", "claude_jump"]),
    ("experiments/weight_space/white_source_bag.py", ["claude_pair", "claude_law", "claude_jump"]),
    ("pipeline_helpers/write_config.py", ["submissions/sub_seqw5_stack.json", json.dumps({"members": MEMBERS})]),
    ("experiments/weight_edit_20261104/stack_edit.py", ["--pairs", "4096", "--splits", "7"]),
    ("experiments/weight_edit_20261104/view_weight_edit.py", []),
]
OUTPUTS = {  # declared test-side outputs to compare, per stage index
    0: ["tune_out/test_VARC4hv_ns3_wc.npy"], 1: ["tune_out/test_MLPHV_bag_w.npy"], 2: ["tune_out/test_MECH3C12V_bag.npy"],
    3: ["tune_out/test_VAR_aug1600_avg_tta.npy"], 4: ["tune_out/test_MECH3C12RV_revlow.npy"],
    5: ["tune_out/weight_space/claude_law.npz"], 6: ["tune_out/weight_space/claude_jump.npz"], 7: ["tune_out/weight_space/claude_pair.npz"],
    8: ["submissions/sub_seqw5_stack.json"], 9: ["tune_out/weight_edit_20261104/stack_test.npz"], 10: ["tune_out/weight_edit_20261104/view_test.npz"]}

build()
for script, args in STAGES:
    run(script, *args)

INPUTS = [p for p in O.rglob("*") if p.is_file() and not any(p.as_posix().endswith(o) for v in OUTPUTS.values() for o in v)]


def prepare(dst, order=None, names=None):
    for p in INPUTS + [W / "clean/features_test.parquet", W / "CPU_REBUILD_WORKSPACE.json"]:
        t = dst / p.relative_to(W); t.parent.mkdir(parents=True, exist_ok=True)
        if order is None or not (p.name.startswith(("test_", "ver_")) or p.name == "features_test.parquet"):
            shutil.copy2(p, t); continue
        if p.name == "features_test.parquet":
            pd.read_parquet(p).iloc[order].set_axis(pd.Index(names, name="id")).to_parquet(t)
        elif p.suffix == ".npy":
            np.save(t, np.load(p)[order])
        else:
            with np.load(p) as z:
                np.savez(t, **{k: (z[k][:, order] if k == "test" else z[k]) for k in z.files})


prepare(P1)
order = rng.permutation(NT)[:41]
names = [f"new{j:03d}" for j in range(len(order))]
prepare(P2, order, names)


def load(path):
    if path.suffix == ".npy":
        return {"": np.load(path)}
    if path.suffix == ".json":
        return {"": np.frombuffer(path.read_bytes(), np.uint8)}
    with np.load(path, allow_pickle=True) as z:
        return {k: z[k] for k in z.files}


results, worst = {}, {}
for label, pred, sel in (("replay", P1, slice(None)), ("new_ids", P2, order)):
    for i, (script, args) in enumerate(STAGES):
        stage = {"id": f"s{i}", "argv": [script, *args], "outputs": OUTPUTS[i]}
        base_models.predict_stage(stage, W, pred)
        ok, diff = True, 0.0
        for rel in OUTPUTS[i]:
            ref, got = load(W / rel), load(pred / rel)
            if set(ref) != set(got):
                ok = False; continue
            for k in ref:
                r, g = ref[k], got[k]
                if k in ("ids", "tids"):
                    ok &= np.array_equal(np.asarray(g).astype(str), (tids[sel] if label == "replay" else np.array(names)).astype(str)) if len(g) != N else np.array_equal(r, g)
                    continue
                if r.ndim and r.shape[-1] == NT and rel != OUTPUTS[8][0]:
                    r = r[..., sel]
                if r.dtype.kind in "fc":
                    exact = r.shape == g.shape and np.array_equal(r, g)
                    d = float(np.abs(r - g).max()) if r.shape == g.shape else np.inf
                    diff = max(diff, d)
                    ok &= exact if label == "replay" else d <= 1e-12
                else:
                    ok &= np.array_equal(r, g)
        key = f"{script.split('/')[-1]} {' '.join(args[:3])[:40]} {label}"
        results[key], worst[key] = ok, diff
for k, v in results.items():
    print(("PASS " if v else "FAIL ") + k + f"  max|diff|={worst[k]:.3g}")
print("scratch:", S)
raise SystemExit(0 if all(results.values()) else 1)
