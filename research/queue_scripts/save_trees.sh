#!/bin/bash
# retrain the stack's tree members with weight saving -> weights/ (tags W_<member>); 5070, one job at a time
cd /c/Users/Blitz/Downloads/Code/claude/ai; PY=.venv/Scripts/python
until grep -q QUEUE_DONE tune_out/q/hv2.log; do sleep 30; done
wg() { while tasklist | grep -qi slay; do sleep 60; done; }
P1='{"max_bin":64,"learning_rate":0.009549022390375738,"subsample":0.8624554592415571,"colsample_bytree":0.3282007743260973,"colsample_bynode":0.7222561808470165,"reg_lambda":0.2992087055688533,"reg_alpha":0.01697153710754649,"gamma":2.349446308922097,"min_child_weight":0.4623536055943267,"max_leaves":73,"rounds":1600}'
K=rev,flip,revflip,c1,c2,c3,c4
for s in 0 1; do wg; t=W_VARC4_ns3_s$s; $PY experiments/v6/train_var.py --tag $t --aug --seed $s --kinds $K --tta-kinds $K --params "$P1" > tune_out/log_$t.txt 2>&1; tail -2 tune_out/log_$t.txt | head -1; done
$PY experiments/v6/wtta.py --crop-w 0.125 --out W_VARC4_ns3_wc W_VARC4_ns3_s0 W_VARC4_ns3_s1 | tail -1
for s in 0 1; do wg; t=W_VAR_full_aug_s$s; $PY experiments/v6/train_var.py --tag $t --aug --seed $s --colfile tune_out/var/all_cols.csv --params '{"colsample_bytree":0.36,"rounds":800}' > tune_out/log_$t.txt 2>&1; tail -2 tune_out/log_$t.txt | head -1; done
RX='{"objective":"rank:pairwise","lambdarank_pair_method":"mean","lambdarank_num_pair_per_sample":8,"rounds":1500,"learning_rate":0.03}'
BL=grid,grid2,dd,lr5,lrz,gh
for s in 0 1; do wg; t=W_L3_rank_lr03_s$s; $PY experiments/v6/train_cv.py --tag $t --seed $s --xgb-policy lossguide --blocks $BL --cols tune_out/selected_ns3.csv --colsample 0.7 --extra "$RX" > tune_out/log_$t.txt 2>&1; grep "CV AUC" tune_out/log_$t.txt; done
for s in 0 1; do wg; t=W_L4_fam_gh_s$s; $PY experiments/v6/train_cv.py --tag $t --seed $s --xgb-policy lossguide --blocks $BL --cols tune_out/loop/fam_gh.csv > tune_out/log_$t.txt 2>&1; grep "CV AUC" tune_out/log_$t.txt; done
wg; t=W_v6_ns1sel1_xlg; $PY experiments/v6/train_cv.py --tag $t --xgb-policy lossguide --blocks grid,grid2,dd,lr --cols tune_out/selected_ns1_sel1.csv --colsample 0.7 > tune_out/log_$t.txt 2>&1; grep "CV AUC" tune_out/log_$t.txt
wg; t=W_v6_dd_cat; $PY experiments/v6/train_cv.py --tag $t --model cat --cat-gpu --blocks grid,grid2,dd > tune_out/log_$t.txt 2>&1; grep "CV AUC" tune_out/log_$t.txt
$PY - <<'PY'
import numpy as np
from scipy.stats import spearmanr
pairs = [("VARC4_ns3_s0","W_VARC4_ns3_s0"),("VARC4_ns3_s1","W_VARC4_ns3_s1"),("VAR_full_aug","W_VAR_full_aug_s0"),("VAR_full_aug_s1","W_VAR_full_aug_s1"),
         ("L3_rank_lr03_s0","W_L3_rank_lr03_s0"),("L3_rank_lr03_s1","W_L3_rank_lr03_s1"),("L4_fam_gh","W_L4_fam_gh_s0"),("L4_fam_gh_s1","W_L4_fam_gh_s1"),
         ("v6_ns1sel1_xlg","W_v6_ns1sel1_xlg"),("v6_dd_cat","W_v6_dd_cat")]
for o, n in pairs:
    try: a, b = np.load(f"tune_out/test_{o}.npy"), np.load(f"tune_out/test_{n}.npy"); print(f"{o:20s} test spearman old vs saved {spearmanr(a, b)[0]:.5f}  max|diff| {np.abs(a-b).max():.2e}")
    except Exception as e: print(o, "compare failed", e)
PY
echo QUEUE_DONE
