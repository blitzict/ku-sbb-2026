"""Real numbers for the slides, from try4's saved fold models (no retraining).

run from C:/Users/Blitz/Downloads/Code/claude/ai with the project venv.
"""
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path.cwd()
REPO = ROOT / "ku-sbb-2026"
sys.path.insert(0, str(REPO / "try4"))
from select_features import matrix, tables  # noqa: E402
from train import TABLES, law  # noqa: E402

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
M = REPO / "try4/model"
WS = "tune_out/weight_space"

tabs = [t for t in tables(ROOT) if t.split("/")[-1].replace("_{}.parquet", "") in TABLES]
X = matrix(ROOT, "train", tabs)
lab = pd.read_csv(ROOT / "data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
y = lab.loc[X.index].to_numpy()
pos = pd.Series(np.arange(len(X)), index=X.index)
print("X", X.shape, "breaks", y.mean())

names = ["lgb_law", "lgb_law_s2", "xgb_law"]
oof = {n: np.zeros(len(y)) for n in names}
fold_of = np.full(len(y), -1)
gain = {}
first_split = None
for f in range(5):
    rows = np.load(ROOT / WS / f"law_outer7/fold{f}_rows.npz")
    val = pos.loc[rows["validation_ids"].astype(str)].to_numpy()
    fold_of[val] = f
    cols = json.loads((M / f"columns_fold{f}.json").read_text())
    A = pd.concat([X[cols["top"]], law(ROOT, 7, f, "train", X.index)], axis=1)
    for n in ["lgb_law", "lgb_law_s2"]:
        b = lgb.Booster(model_file=str(M / f"{n}_fold{f}.txt"))
        oof[n][val] = b.predict(A.iloc[val][b.feature_name()])
        if n == "lgb_law":
            g = pd.Series(b.feature_importance("gain"), index=b.feature_name())
            for k, v in (g / g.sum()).items():
                gain.setdefault(k, []).append(v)
    bx = xgb.Booster()
    bx.load_model(str(M / f"xgb_law_fold{f}.json"))
    oof["xgb_law"][val] = bx.predict(xgb.DMatrix(A.iloc[val][bx.feature_names]))

res = {n: roc_auc_score(y, v) for n, v in oof.items()}
blend = np.mean([rankdata(oof[n]) for n in names], axis=0) / len(y)
res["blend"] = roc_auc_score(y, blend)
res["per_fold"] = [roc_auc_score(y[fold_of == f], blend[fold_of == f]) for f in range(5)]
print(json.dumps(res, indent=1))

fpr, tpr, _ = roc_curve(y, blend)
idx = np.unique(np.linspace(0, len(fpr) - 1, 200).astype(int))
res["roc"] = {"fpr": fpr[idx].round(4).tolist(), "tpr": tpr[idx].round(4).tolist()}

# decile hit rate: share of real breaks in each tenth of the ranking
order = np.argsort(-blend)
dec = np.array_split(order, 10)
res["decile_break_rate"] = [float(y[d].mean()) for d in dec]
# top 20% precision, and recall at 50% of the list
res["share_breaks_in_top_half"] = float(y[order[: len(y) // 2]].sum() / y.sum())

# score distributions
bins = np.linspace(0, 1, 26)
res["hist_break"] = np.histogram(blend[y == 1], bins)[0].tolist()
res["hist_nobreak"] = np.histogram(blend[y == 0], bins)[0].tolist()

# feature gain share (mean over folds; 0 where a fold did not keep it)
g = pd.Series({k: sum(v) / 5 for k, v in gain.items()}).sort_values(ascending=False)
top = g.head(12)
single = {}
for k in top.index:
    col = X[k] if k in X.columns else None
    if col is None:  # law column: fold-specific, evaluate on each fold's validation rows
        s = np.zeros(len(y))
        for f in range(5):
            L = law(ROOT, 7, f, "train", X.index)
            s[fold_of == f] = L[k].to_numpy()[fold_of == f]
        col = pd.Series(s, index=X.index)
    v = col.fillna(col.median()).to_numpy()
    a = roc_auc_score(y, v)
    single[k] = max(a, 1 - a)
res["top_features"] = [{"name": k, "gain_share": float(top[k]), "auc_alone": float(single[k])} for k in top.index]
res["law_gain_share"] = float(g[[k for k in g.index if k.startswith("law__")]].sum())
res["n_features_total_candidates"] = int(X.shape[1])
print(json.dumps(res["top_features"], indent=1), res["law_gain_share"])
(OUT / "try4_oof_stats.json").write_text(json.dumps(res, indent=1))
np.save(OUT / "try4_blend_oof.npy", blend)
