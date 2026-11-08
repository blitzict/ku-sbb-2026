# CPU port of experiments/redesign/nested_tree.py, source sha256 e1b0bbcf59ed4a3fe1190c147b48bf6d3818d526df473ad1695df1fe9441f023, with verified model saving. See auc_lab_20261107/claude/audit/cpu_port_notes.md.
import sys as _s, pathlib as _p; _s.path.insert(0, str(_p.Path(__file__).resolve().parents[2])); import ku_workspace as KU_ENV
"""Nested version of the ns3 + mechanism tables + white-law tree (2026-11-04). For canonical outer fold k, the white-law columns come from Codex's
tune_out/weight_space/law_outer0/fold{k}_{train,test}.parquet, whose priors used only fold k's training labels. Tree k trains on fold k's training
rows and scores its validation rows and the test set; test = mean of the five fold trees. No aug, P1, 5070.
Writes tune_out/oof_MECHNEST2.npy, tune_out/test_MECHNEST2.npy."""
import sys, time, gc, ctypes, warnings, pathlib; warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
try: ctypes.windll.kernel32.SetPriorityClass(ctypes.c_void_p(-1), 0x4000)
except Exception: pass
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
import argparse
ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--extra", default="jump_law_pulse,jump_law_state"); a = ap.parse_args()
D = KU_ENV.root(); O = D / "tune_out"; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent)); import save_models as SM; WS = O / "weight_space"; t0 = time.time()
M = np.load(O / "var/mat_ns3_plain.npz", allow_pickle=True); ids, tids = M["ids"], M["tid"]; y, w = M["y"], M["w"]
ns3, ns3t = pd.DataFrame(M["X"], index=ids, columns=list(M["cols"])), pd.DataFrame(M["Xt"], index=tids, columns=list(M["cols"]))
T = ["harmonic", "ar_law", "bounds", "curve", "noise_tests", "partition", "periodic", "phase_law", "trend", "innov_v2", "mixture_law"] + [t for t in a.extra.split(",") if t]
src = lambda t, p: (O / f"redesign/{t[9:]}_{p}.parquet") if t.startswith("redesign:") else (WS / f"{t}_{p}.parquet")
A = pd.concat([ns3] + [pd.read_parquet(src(t, "train")).loc[ids].add_prefix(f"{t}__") for t in T], axis=1)
At = pd.concat([ns3t] + [pd.read_parquet(src(t, "test")).loc[tids].add_prefix(f"{t}__") for t in T], axis=1)
P = {"objective": "binary:logistic", "device": "cpu", "tree_method": "hist", "nthread": KU_ENV.threads(), "seed": a.seed, "grow_policy": "lossguide", "max_depth": 0, "verbosity": 0,
     "max_bin": 64, "learning_rate": 0.009549022390375738, "subsample": 0.8624554592415571, "colsample_bytree": 0.3282007743260973, "colsample_bynode": 0.7222561808470165,
     "reg_lambda": 0.2992087055688533, "reg_alpha": 0.01697153710754649, "gamma": 2.349446308922097, "min_child_weight": 0.4623536055943267, "max_leaves": 73}
oof = np.full(len(y), np.nan); test = np.zeros(len(tids))
for k, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(A, y)):
    rows = np.load(WS / f"law_outer0/fold{k}_rows.npz", allow_pickle=True)
    assert set(rows["validation_ids"].astype(str)) == set(ids[va].astype(str)) and set(rows["train_ids"].astype(str)) == set(ids[tr].astype(str)), f"fold {k} mismatch"
    L = pd.read_parquet(WS / f"law_outer0/fold{k}_train.parquet").loc[ids].add_prefix("wnlaw__"); Lt = pd.read_parquet(WS / f"law_outer0/fold{k}_test.parquet").loc[tids].add_prefix("wnlaw__")
    X = pd.concat([A, L], axis=1).to_numpy(np.float32); Xt = pd.concat([At, Lt], axis=1).to_numpy(np.float32)
    d = xgb.QuantileDMatrix(X[tr], y[tr], max_bin=64); b = xgb.train(P, d, int(__import__("os").environ.get("KU_SBB_TREE_ROUNDS", "1600")))
    SM.features(O, a.tag, pd.concat([A.iloc[:1], L.iloc[:1]], axis=1).columns); SM.save_xgb(O, a.tag, k, b, X[va])
    oof[va] = b.predict(xgb.QuantileDMatrix(X[va], ref=d, max_bin=64)); test += b.predict(xgb.QuantileDMatrix(Xt, ref=d, max_bin=64)) / 5
    print(f"fold {k}: {roc_auc_score(y[va], oof[va]):.4f}  ({time.time()-t0:.0f}s)", flush=True); del d, b; gc.collect()
print(f"{a.tag}: {roc_auc_score(y, oof):.5f} / {roc_auc_score(y, oof, sample_weight=w):.5f}", flush=True)
np.save(O / f"oof_{a.tag}.npy", oof); np.save(O / f"test_{a.tag}.npy", test)
