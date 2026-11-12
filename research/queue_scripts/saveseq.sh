#!/bin/bash
# retrain the raw-sequence member runs with weight saving -> weights/W_SEQ_*/ (Arc), then rebuild the 3-run mean and compare
cd /c/Users/Blitz/Downloads/Code/claude/ai; XPY=.venv-xpu/Scripts/python; PY=.venv/Scripts/python
A="--epochs 45 --patience 12 --qin 16"
$XPY experiments/v6/nn_seq.py --tag W_SEQ_q $A --seeds 0 2>&1 | grep -v "^  \|seed" | tail -2
$XPY experiments/v6/nn_seq.py --tag W_SEQ_q_s1 $A --seeds 1 2>&1 | grep -v "^  \|seed" | tail -2
$XPY experiments/v6/nn_seq.py --tag W_SEQ_qh $A --hq 8 --seeds 0 2>&1 | grep -v "^  \|seed" | tail -2
$PY - <<'PY'
import numpy as np, json
from scipy.special import logit, expit
from sklearn.metrics import roc_auc_score
import pandas as pd
ids = pd.read_parquet("clean/features_train.parquet", columns=[]).index
y = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id").label.loc[ids].to_numpy()
pt = np.load("tune_out/adv_ptest_train.npy"); w = pt / np.clip(1 - pt, 1e-3, None)
for part in ("oof", "test"):
    m = expit(np.mean([logit(np.clip(np.load(f"tune_out/{part}_W_{t}.npy"), 1e-6, 1 - 1e-6)) for t in ("SEQ_q", "SEQ_q_s1", "SEQ_qh")], 0)); np.save(f"tune_out/{part}_W_SEQ_qb3.npy", m)
o = np.load("tune_out/oof_W_SEQ_qb3.npy"); print("W_SEQ_qb3", round(roc_auc_score(y, o), 5), round(roc_auc_score(y, o, sample_weight=w), 5), "(orig 0.83627 / 0.82963)")
print("test max|diff| vs orig", float(np.abs(np.load("tune_out/test_W_SEQ_qb3.npy") - np.load("tune_out/test_SEQ_qb3.npy")).max()))
PY
echo QUEUE_DONE
