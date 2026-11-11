"""Where try4 fails: held-out AUC and miss rates by series length, point position and how visible the break is."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
sys.path.insert(0, str(Path.cwd() / "ku-sbb-2026/try4"))
from select_features import matrix, tables
from train import TABLES
OUT = Path(sys.argv[1]); ROOT = Path.cwd()
tabs = [t for t in tables(ROOT) if t.split("/")[-1].replace("_{}.parquet", "") in TABLES]
idx = matrix(ROOT, "train", tabs[:1]).index
y = pd.read_csv(ROOT / "data/train_labels.csv", dtype={"id": str}).set_index("id")["label"].loc[idx].to_numpy()
p = np.load(OUT / "try4_blend_oof.npy")
s = pd.read_parquet(ROOT / "data/train_series.parquet")
st = s.groupby(["id", "period"])["value"].agg(["mean", "std", "count"]).unstack("period").loc[idx]
n = st["count"].sum(axis=1).to_numpy(); pos = (st["count"][0] / st["count"].sum(axis=1)).to_numpy()
shift = (np.abs(st["mean"][1] - st["mean"][0]) / st["std"][0]).to_numpy()
sdr = np.abs(np.log(st["std"][1] / st["std"][0])).to_numpy()
vis = np.maximum(shift, sdr)
thr = np.median(p)          # "flagged" = top half of the ranking
res = {"n": int(len(y)), "breaks": int(y.sum()), "threshold": "top half"}
def grp(name, v, edges, labels):
    out = []
    for lo, hi, lab in zip(edges[:-1], edges[1:], labels):
        m = (v >= lo) & (v < hi)
        a = roc_auc_score(y[m], p[m]) if len(set(y[m])) == 2 else None
        out.append({"label": lab, "n": int(m.sum()), "breaks": int(y[m].sum()), "auc": a,
                    "break_caught": float((p[m & (y == 1)] > thr).mean()) if (m & (y == 1)).any() else None,
                    "nobreak_flagged": float((p[m & (y == 0)] > thr).mean()) if (m & (y == 0)).any() else None})
    res[name] = out
q = np.quantile(n, [0, 1/3, 2/3, 1]); q[-1] += 1
grp("by_length", n, q, [f"{int(q[0])}–{int(q[1])} pts", f"{int(q[1])}–{int(q[2])} pts", f"{int(q[2])}–{int(q[3]-1)} pts"])
grp("by_position", pos, [0, .4333, .5667, 1.01], ["point at 30–43%", "point at 43–57%", "point at 57–70%"])
grp("by_visibility", vis, [0, .05, .15, .5, 1e9], ["< 0.05 SD", "0.05–0.15", "0.15–0.5", "> 0.5 SD"])
res["length_range"] = [int(n.min()), int(n.max())]
res["pos_range"] = [float(pos.min()), float(pos.max())]
# errors: missed breaks vs caught breaks
miss = (y == 1) & (p <= thr); caught = (y == 1) & (p > thr); fa = (y == 0) & (p > thr)
res["missed_breaks"] = int(miss.sum()); res["false_alarms"] = int(fa.sum())
res["missed_median_vis"] = float(np.median(vis[miss])); res["caught_median_vis"] = float(np.median(vis[caught]))
res["missed_share_vis_lt_0.05"] = float((vis[miss] < .05).mean())
res["fa_median_vis"] = float(np.median(vis[fa])); res["nobreak_median_vis"] = float(np.median(vis[y == 0]))
print(json.dumps(res, indent=1))
(OUT / "errors.json").write_text(json.dumps(res, indent=1))
