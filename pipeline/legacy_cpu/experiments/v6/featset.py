# CPU port of experiments/v6/featset.py, source sha256 686ef444a0f64152595d6b0192185f2cd9f987dfc57374ce39c2607563912311. See auc_lab_20261107/claude/audit/cpu_port_notes.md.
import sys as _s, pathlib as _p; _s.path.insert(0, str(_p.Path(__file__).resolve().parents[2])); import ku_workspace as KU_ENV
"""Shared loader for the v6 feature table: clean + grid blocks + optional ratio/difference block, float32, plus the
boundary-shift negatives (with their source ids), labels, test-weights and the canonical 5 folds."""
import json, pathlib
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
D = KU_ENV.root(); O = D / "tune_out"


def rp(p, cols=None):
    return pd.read_parquet(p, columns=cols).astype(np.float32)


def rd_block(df, stems):
    out = {}; eps = np.float32(1e-6)
    for s in stems:
        p, q = df[s + "_pre"].to_numpy(), df[s + "_post"].to_numpy()
        out[f"rd_{s}_lr"] = np.log((np.abs(q) + eps) / (np.abs(p) + eps))
        out[f"rd_{s}_ad"] = np.abs(q - p)
        out[f"rd_{s}_sr"] = (q - p) / (np.abs(q) + np.abs(p) + eps)
    return pd.DataFrame(out, index=df.index).astype(np.float32)


def load(blocks=("grid", "grid2"), rd=False, with_aug=True):
    X = rp(D / "clean/features_train.parquet"); cols = list(X.columns); Xt = rp(D / "clean/features_test.parquet", cols)
    if with_aug:
        A = rp(O / "aug_features.parquet", cols)
        M = pd.read_csv(O / "aug_meta.csv", dtype=str).set_index("id").loc[A.index]; keep = (M.kind == "shift").to_numpy()
        A = A[keep]; src = M.source[keep].to_numpy()
    else:
        A, src = X.iloc[:0], np.array([], dtype=object)
    for b in blocks:
        B = rp(O / f"{b}_train.parquet").loc[X.index]; bc = list(B.columns)
        Bt = rp(O / f"{b}_test.parquet", bc).loc[Xt.index]
        Ba = rp(O / f"aug_{b}_features.parquet", bc).loc[A.index] if with_aug else B.iloc[:0]
        B.columns = Bt.columns = Ba.columns = [f"{b}__{c}" for c in bc]
        X = pd.concat([X, B], axis=1); Xt = pd.concat([Xt, Bt], axis=1); A = pd.concat([A, Ba], axis=1)
    if rd:
        stems = [c[:-4] for c in cols if c.endswith("_pre") and (c[:-4] + "_post") in cols]
        X = pd.concat([X, rd_block(X, stems)], axis=1); Xt = pd.concat([Xt, rd_block(Xt, stems)], axis=1); A = pd.concat([A, rd_block(A, stems)], axis=1)
    for df in (X, Xt, A): df.replace([np.inf, -np.inf], np.nan, inplace=True)
    y = pd.read_csv(D / "data/train_labels.csv", dtype={"id": str}).set_index("id").label.loc[X.index].to_numpy()
    w = KU_ENV.diag_weights(O)
    folds = list(StratifiedKFold(5, shuffle=True, random_state=0).split(X, y))
    return X, Xt, A, src, y, w, folds
