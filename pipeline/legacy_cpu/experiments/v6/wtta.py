# CPU port of experiments/v6/wtta.py, source sha256 e8084c06dd220f02b5beb613124283a9319de27cac156d7964e9c568c3ab425d. See auc_lab_20261107/claude/audit/cpu_port_notes.md.
import sys as _s, pathlib as _p; _s.path.insert(0, str(_p.Path(__file__).resolve().parents[2])); import ku_workspace as KU_ENV
"""Weighted test-time averaging from the per-version predictions saved by train_var.py (ver_<tag>.npz). Logit mean with fixed version weights
(orig/flip 2, rev/revflip 1, crops 0.5; chosen on VARC_ns3_s0 and checked on other seeds). Averages several tags (seeds) into one member.
Usage: python wtta.py --out <member> <tag> [<tag> ...]  -> tune_out/oof_<member>.npy, test_<member>.npy, best_<member>.json"""
import sys, json, argparse, pathlib, numpy as np, pandas as pd
from scipy.special import logit, expit
from sklearn.metrics import roc_auc_score
D = KU_ENV.root(); O = D / "tune_out"
ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("tags", nargs="+"); ap.add_argument("--crop-w", type=float, default=0.5); ap.add_argument("--dec-w", type=float, default=0.5); a = ap.parse_args()
WT = {"orig": 2.0, "flip": 2.0, "rev": 1.0, "revflip": 1.0, **{c: a.crop_w for c in ("c1", "c2", "c3", "c4", "c5", "c6")}, **{c: a.dec_w for c in ("d2a", "d2b", "a2")}}
ids = pd.read_parquet(D / "clean/features_train.parquet", columns=[]).index
y = pd.read_csv(D / "data/train_labels.csv", dtype={"id": str}).set_index("id").label.loc[ids].to_numpy()
w = KU_ENV.diag_weights(O)
res = {}
for part in ("oof", "test"):
    acc = []
    for t in a.tags:
        z = np.load(O / f"ver_{t}.npz"); W = np.array([WT[k] for k in z["kinds"]]); Lg = logit(np.clip(z[part].astype(float), 1e-6, 1 - 1e-6))
        acc.append((W[:, None] * Lg).sum(0) / W.sum())
        if part == "oof": print(f"  {t}: {roc_auc_score(y, acc[-1]):.5f} / {roc_auc_score(y, acc[-1], sample_weight=w):.5f}")
    res[part] = expit(np.mean(acc, 0)); np.save(O / f"{part}_{a.out}.npy", res[part])
cv, cvw = roc_auc_score(y, res["oof"]), roc_auc_score(y, res["oof"], sample_weight=w)
json.dump({"note": f"weighted TTA {WT} over {a.tags}", "cv": float(cv), "cv_test_weighted": float(cvw)}, open(O / f"best_{a.out}.json", "w"), indent=1)
print(f"{a.out}: {cv:.5f} / {cvw:.5f}")
