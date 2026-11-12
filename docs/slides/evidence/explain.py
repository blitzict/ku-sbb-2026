"""Specialist stats, the exact picked features, and one worked prediction (try4, held-out)."""
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path.cwd()
REPO = ROOT / "ku-sbb-2026"
sys.path.insert(0, str(REPO / "try4"))
from select_features import matrix, tables  # noqa: E402
from train import TABLES, law  # noqa: E402

OUT = Path(sys.argv[1])
M = REPO / "try4/model"
WS = "tune_out/weight_space"
res = {}

# ---- try3's specialist ---------------------------------------------------------------
lab0 = pd.read_csv(ROOT / "data/train_labels.csv", dtype={"id": str})
orders = {"labels_file": lab0["label"].to_numpy(),
          "sorted_ids": lab0.sort_values("id")["label"].to_numpy()}
spec = {}
for tag in ["MECH3C12V_s0", "MECH3C12V_s1", "MECH3C12V_s2", "MECH3C12V_bag", "MECH3C1_s0", "MECH3C12_s0"]:
    f = ROOT / f"tune_out/oof_{tag}.npy"
    if f.exists():
        v = np.load(f)
        v = v.reshape(24000, -1)[:, 0] if v.size >= 24000 else v
        spec[tag] = {k: float(roc_auc_score(o, v)) for k, o in orders.items()}
ncols = -1
res["specialist_auc"] = spec
res["specialist_ns3_cols"] = int(ncols)
print("specialist", spec, "ns3 cols", ncols, flush=True)

# ---- try4 data -------------------------------------------------------------------------
tabs = [t for t in tables(ROOT) if t.split("/")[-1].replace("_{}.parquet", "") in TABLES]
X = matrix(ROOT, "train", tabs)
lab = pd.read_csv(ROOT / "data/train_labels.csv", dtype={"id": str}).set_index("id")["label"]
y = lab.loc[X.index].to_numpy()
for tag in list(spec):
    v = np.load(ROOT / f"tune_out/oof_{tag}.npy"); v = v.reshape(24000, -1)[:, 0] if v.size >= 24000 else v
    spec[tag]["X_index"] = float(roc_auc_score(y, v))
res["specialist_auc"] = spec
print("specialist", spec, flush=True)
pos = pd.Series(np.arange(len(X)), index=X.index)
blend = np.load(OUT / "try4_blend_oof.npy")

# exact picked features
per_fold = [json.loads((M / f"columns_fold{f}.json").read_text()) for f in range(5)]
tops = [set(p["top"]) for p in per_fold]
inall = set.intersection(*tops)
union = set.union(*tops)
fam = lambda c: c.split("__")[0]
res["picked"] = {
    "candidates": int(X.shape[1]), "per_fold": 100, "law_cols": len(per_fold[0]["law"]),
    "in_all_5": len(inall), "union": len(union),
    "fold0_by_family": pd.Series([fam(c) for c in per_fold[0]["top"]]).value_counts().to_dict(),
    "in_all_5_list": sorted(inall),
    "law_list": per_fold[0]["law"],
}
print("picked", {k: v for k, v in res["picked"].items() if not k.endswith("list")}, flush=True)

# ---- naive stats per series (to find a break that is invisible by eye) ---------------
s = pd.read_parquet(ROOT / "data/train_series.parquet")
st = s.groupby(["id", "period"])["value"].agg(["mean", "std", "count"]).unstack("period")
st = st.loc[X.index]
shift = (np.abs(st["mean"][1] - st["mean"][0]) / st["std"][0]).to_numpy()
sdr = np.abs(np.log(st["std"][1] / st["std"][0])).to_numpy()

rank = pd.Series(blend).rank(pct=True).to_numpy()
cand = np.flatnonzero((y == 1) & (rank > 0.97) & (shift < 0.08) & (sdr < 0.12))
print("candidates", len(cand), flush=True)

fold_of = np.full(len(y), -1)
for f in range(5):
    rows = np.load(ROOT / WS / f"law_outer7/fold{f}_rows.npz")
    fold_of[pos.loc[rows["validation_ids"].astype(str)].to_numpy()] = f

def explain(i):
    f = int(fold_of[i])
    cols = per_fold[f]
    L = law(ROOT, 7, f, "train", X.index)
    A = pd.concat([X[cols["top"]], L], axis=1)
    b = lgb.Booster(model_file=str(M / f"lgb_law_fold{f}.txt"))
    row = A.iloc[[i]][b.feature_name()]
    contrib = b.predict(row, pred_contrib=True)[0]
    names = b.feature_name()
    base = contrib[-1]
    c = pd.Series(contrib[:-1], index=names)
    top = c.reindex(c.abs().sort_values(ascending=False).index).head(8)
    # percentile of each top feature among no-break series in the same fold's validation rows
    val = fold_of == f
    out = []
    for k, v in top.items():
        allv = A.loc[:, k].to_numpy()
        ref = allv[val & (y == 0)]
        x = float(A.iloc[i][k])
        out.append({"name": k, "value": x, "contrib": float(v), "pct_vs_nobreak": float((ref < x).mean())})
    p = float(b.predict(row)[0])
    return {"fold": f, "label": int(y[i]), "lgb_law_prob": p, "base_logodds": float(base), "sum_logodds": float(contrib.sum()),
            "blend_rank_pct": float(rank[i]), "mean_shift_sd": float(shift[i]), "abs_log_sd_ratio": float(sdr[i]),
            "length": int(st["count"].iloc[i].sum()), "pre_len": int(st["count"][0].iloc[i]), "rest_contrib": float(c.drop(top.index).sum()),
            "n_cols": len(names), "top": out}

# choose the candidate whose explanation is most concentrated in readable features
best = None
for i in cand[:60]:
    e = explain(i)
    score = e["lgb_law_prob"]
    if best is None or score > best[0]:
        best = (score, i, e)
res["example_break"] = best[2]
# a contrasting no-break that looks dramatic by eye but is correctly scored low
cand0 = np.flatnonzero((y == 0) & (rank < 0.05) & (sdr > 0.35))
print("no-break candidates", len(cand0), flush=True)
if len(cand0):
    res["example_nobreak"] = explain(cand0[np.argmax(sdr[cand0])])
print(json.dumps(res["example_break"], indent=1), flush=True)
print(json.dumps(res.get("example_nobreak"), indent=1), flush=True)
(OUT / "explain.json").write_text(json.dumps(res, indent=1))
