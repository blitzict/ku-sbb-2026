# CPU port of experiments/v6/features_ix.py, source sha256 9d3783e10ee6e3f5289560f1baae07ea754e3fccc66f5c835a00495fc8086d1d. See auc_lab_20261107/claude/audit/cpu_port_notes.md.
import sys as _s, pathlib as _p; _s.path.insert(0, str(_p.Path(__file__).resolve().parents[2])); import ku_workspace as KU_ENV
"""Crossed-interaction block (ADIA write-ups: pairwise combinations of the top features, then selection).
Top-K ns3 features by real importance (null_importance_ns3.csv order); each z-scored with training-set median / IQR (labels unused);
for every pair: product, difference, sum of z-scores -> tune_out/ix_{train,test}.parquet, aug_ix_features.parquet."""
import pathlib, itertools, sys
import numpy as np, pandas as pd
D = KU_ENV.root(); O = D / "tune_out"; K = int(sys.argv[1]) if len(sys.argv) > 1 else 30
top = list(pd.read_csv(O / "selected_ns3.csv").feature[:K])
def table(kind):
    base = {"train": D / "clean/features_train.parquet", "test": D / "clean/features_test.parquet", "aug": O / "aug_features.parquet"}[kind]
    idx = pd.read_parquet(base, columns=[]).index; out = pd.DataFrame(index=idx)
    for c in top:
        if "__" in c:
            b, col = c.split("__", 1); p = {"train": O / f"{b}_train.parquet", "test": O / f"{b}_test.parquet", "aug": O / f"aug_{b}_features.parquet"}[kind]
            out[c] = pd.read_parquet(p, columns=[col])[col].reindex(idx).to_numpy()
        else: out[c] = pd.read_parquet(base, columns=[c])[c].to_numpy()
    return out.astype(np.float64).replace([np.inf, -np.inf], np.nan)
T = {k: table(k) for k in ("train", "test", "aug")}
med = T["train"].median(); iqr = (T["train"].quantile(0.75) - T["train"].quantile(0.25)).replace(0, 1)
for k, df in T.items():
    Z = ((df - med) / iqr).clip(-20, 20); F = {}
    for a, b in itertools.combinations(range(K), 2):
        x, y = Z.iloc[:, a].to_numpy(), Z.iloc[:, b].to_numpy(); F[f"ix{a}_{b}_mul"] = x * y; F[f"ix{a}_{b}_dif"] = x - y; F[f"ix{a}_{b}_sum"] = x + y
    F = pd.DataFrame(F, index=df.index).astype(np.float32); F.index.name = "id"
    F.to_parquet(O / (f"ix_{k}.parquet" if k != "aug" else "aug_ix_features.parquet")); print(k, F.shape)
pd.DataFrame({"i": range(K), "feature": top}).to_csv(O / "loop/ix_top.csv", index=False)
