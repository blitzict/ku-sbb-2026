"""Single-measure baselines: AUC of the mean shift and the spread ratio, straight from the raw practice series (no model).

python extra/slides/evidence/naive.py   (from the folder holding data/)
"""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

s = pd.read_parquet("data/train_series.parquet")
st = s.groupby(["id", "period"])["value"].agg(["mean", "std", "count"]).unstack("period")
y = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id")["label"].loc[st.index].to_numpy()
auc = lambda v: max(roc_auc_score(y, np.nan_to_num(v)), 1 - roc_auc_score(y, np.nan_to_num(v)))
shift = np.abs(st["mean"][1] - st["mean"][0]) / st["std"][0]
ratio = np.abs(np.log(st["std"][1] / st["std"][0]))
print(json.dumps({"mean_shift_auc": auc(shift), "std_ratio_auc": auc(ratio),
                  "breaks_mean_shift_below_0.1sd": float((shift[y == 1] < 0.1).mean())}, indent=1))
