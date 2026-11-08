# CPU port of experiments/v6/distill.py, source sha256 4b0be3675a53843f10e8773e170a9d82a68004a5620ec5390acbdc37c541b3b6. See auc_lab_20261107/claude/audit/cpu_port_notes.md.
import sys as _s, pathlib as _p; _s.path.insert(0, str(_p.Path(__file__).resolve().parents[2])); import ku_workspace as KU_ENV
"""Nested soft-target distillation (tree-side idea): inside each outer fold, an inner 5-fold leaf-wise XGBoost bag produces out-of-fold teacher
probabilities for the outer-training rows only; the student is trained on t = alpha * p_teacher + (1 - alpha) * y (shift negatives get alpha * p + 0)
and evaluated on the untouched outer validation fold.  CPU port.
Usage: python distill.py --tag v6_distill --cols tune_out/selected_ns1.csv --alpha 0.7 [--blocks grid,grid2,dd,lr] [--seed 0]"""
import sys, json, time, pathlib, argparse, warnings, ctypes, gc; warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
try: ctypes.windll.kernel32.SetPriorityClass(ctypes.c_void_p(-1), 0x4000)
except Exception: pass
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import numpy as np, pandas as pd, xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from featset import load, O
ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); ap.add_argument("--blocks", default="grid,grid2,dd,lr"); ap.add_argument("--cols", default=None)
ap.add_argument("--alpha", type=float, default=0.7); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--colsample", type=float, default=0.7); ap.add_argument("--rounds", type=int, default=499)
a = ap.parse_args(); t0 = time.time()
X, Xt, A, src, y, w, folds = load(blocks=tuple(b for b in a.blocks.split(",") if b))
if a.cols: keep = [c for c in pd.read_csv(a.cols).feature if c in X.columns]; X, Xt, A = X[keep], Xt[keep], A[keep]
Xn, Xtn, An = X.to_numpy(np.float32), Xt.to_numpy(np.float32), A.to_numpy(np.float32); ids = X.index.to_numpy(); del X, Xt, A; gc.collect()
P = {"objective": "binary:logistic", "device": "cpu", "tree_method": "hist", "nthread": KU_ENV.threads(), "seed": a.seed, "grow_policy": "lossguide", "max_leaves": 76, "max_depth": 0, "learning_rate": 0.0128,
     "min_child_weight": 3.75, "subsample": 0.8938, "colsample_bytree": a.colsample, "colsample_bynode": 1.0, "reg_alpha": 0.3403, "reg_lambda": 1.1182, "gamma": 0.4257, "verbosity": 0, "max_bin": 256}
print(f"{a.tag}: {Xn.shape[1]} columns, alpha {a.alpha}", flush=True)


STUDENT = []


def fit_predict(Xtr, ytr, Xva_list, rounds=a.rounds):
    d = xgb.QuantileDMatrix(Xtr, ytr, max_bin=256); b = xgb.train(P, d, rounds)
    outs = [b.predict(xgb.QuantileDMatrix(Xv, ref=d, max_bin=256)) for Xv in Xva_list]
    if len(Xva_list) >= 3:                                 # Student models only; inner teachers score two sets.
        STUDENT.append(0); (O.parent / "weights" / a.tag).mkdir(parents=True, exist_ok=True); b.save_model(str(O.parent / "weights" / a.tag / f"student{len(STUDENT)}.ubj")); KU_ENV.check_xgb(O.parent / "weights" / a.tag / f"student{len(STUDENT)}.ubj", b, xgb.DMatrix(Xva_list[0]))
    del d, b; gc.collect(); return outs


oof = np.zeros(len(y)); test = np.zeros(len(Xtn)); hard = np.zeros(len(y))
for k, (tr, va) in enumerate(folds):
    tk = time.time(); sel = np.isin(src, ids[tr]); Xa = An[sel]
    # inner teachers on the outer-training rows (real rows only for the inner split; shift rows always in the inner training set)
    p_in = np.zeros(len(tr)); p_aug = np.zeros(sel.sum())
    for itr, iva in StratifiedKFold(5, shuffle=True, random_state=100 + k).split(tr, y[tr]):
        Xtr_ = np.vstack([Xn[tr[itr]], Xa]); ytr_ = np.concatenate([y[tr[itr]], np.zeros(len(Xa))])
        pv, pa = fit_predict(Xtr_, ytr_, [Xn[tr[iva]], Xa]); p_in[iva] = pv; p_aug += pa / 5
    t_real = a.alpha * p_in + (1 - a.alpha) * y[tr]; t_aug = a.alpha * p_aug
    Xtr_ = np.vstack([Xn[tr], Xa]); ttr = np.concatenate([t_real, t_aug])
    pv, pte, ph = fit_predict(Xtr_, ttr, [Xn[va], Xtn, Xn[tr]]); oof[va] = pv; test += pte / 5
    print(f"  fold {k}: student val AUC {roc_auc_score(y[va], pv):.4f}  (teacher inner OOF AUC {roc_auc_score(y[tr], p_in):.4f})  ({time.time()-tk:.0f}s)", flush=True)
cv, cvw = roc_auc_score(y, oof), roc_auc_score(y, oof, sample_weight=w)
print(f"{a.tag}: CV AUC {cv:.5f}  test-weighted {cvw:.5f}  ({time.time()-t0:.0f}s)", flush=True)
np.save(O / f"oof_{a.tag}.npy", oof); np.save(O / f"test_{a.tag}.npy", test)
json.dump({"note": f"distill alpha={a.alpha} cols={a.cols} blocks={a.blocks} seed={a.seed}", "cv": float(cv), "cv_test_weighted": float(cvw)}, open(O / f"best_{a.tag}.json", "w"), indent=1)
