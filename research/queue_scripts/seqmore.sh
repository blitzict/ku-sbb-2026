#!/bin/bash
cd /c/Users/Blitz/Downloads/Code/claude/ai; XPY=.venv-xpu/Scripts/python; PY=.venv/Scripts/python; A="--epochs 45 --patience 12 --qin 16"
$XPY experiments/v6/nn_seq.py --tag W_SEQ_q_s2 $A --seeds 2 2>&1 | grep "CV tta" | tail -1
$XPY experiments/v6/nn_seq.py --tag W_SEQ_qh_s1 $A --hq 8 --seeds 1 2>&1 | grep "CV tta" | tail -1
$PY - <<'PY'
import numpy as np, pandas as pd
from scipy.special import logit, expit
from sklearn.metrics import roc_auc_score
ids = pd.read_parquet("clean/features_train.parquet", columns=[]).index
y = pd.read_csv("data/train_labels.csv", dtype={"id": str}).set_index("id").label.loc[ids].to_numpy()
pt = np.load("tune_out/adv_ptest_train.npy"); w = pt / np.clip(1 - pt, 1e-3, None)
T = ("SEQ_q", "SEQ_q_s1", "SEQ_qh", "SEQ_q_s2", "SEQ_qh_s1")
for part in ("oof", "test"): np.save(f"tune_out/{part}_W_SEQ_qb5.npy", expit(np.mean([logit(np.clip(np.load(f"tune_out/{part}_W_{t}.npy"), 1e-6, 1 - 1e-6)) for t in T], 0)))
o = np.load("tune_out/oof_W_SEQ_qb5.npy"); print("W_SEQ_qb5", round(roc_auc_score(y, o), 5), round(roc_auc_score(y, o, sample_weight=w), 5))
PY
$PY experiments/v6/nnstack.py --base submissions/sub_seqw_stack.json --drop W_SEQ_qb3 --add W_SEQ_qb5 2>&1 | tail -2
echo QUEUE_DONE
