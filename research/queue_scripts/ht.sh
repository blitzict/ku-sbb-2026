#!/bin/bash
# test-series segment rows (ht; user-approved): features, ns3 VARC4 + hv + ht 2 seeds -> stack swap for VARC4hv; then MLP + hv + ht 4 seeds
cd /c/Users/Blitz/Downloads/Code/claude/ai; PY=.venv/Scripts/python; XPY=../../.venv-xpu/Scripts/python; XP=.venv-xpu/Scripts/python
(cd experiments/v6 && for f in "features_lr.py var:train_ht lr5" "features_garch.py var:train_ht" "features_sn.py var:train_ht" "features_lm.py var:train_ht"; do $XPY $f 2>&1 | tail -1; done; LRZ_TAG=lrz6 $XPY features_lrz.py var:train_ht 2>&1 | tail -1)
$PY experiments/v6/var_numpy_features.py train_ht > tune_out/logs/log_var_numpy_train_ht.txt 2>&1; tail -1 tune_out/logs/log_var_numpy_train_ht.txt
echo HT_FEATURES_DONE
P1='{"max_bin":64,"learning_rate":0.009549022390375738,"subsample":0.8624554592415571,"colsample_bytree":0.3282007743260973,"colsample_bynode":0.7222561808470165,"reg_lambda":0.2992087055688533,"reg_alpha":0.01697153710754649,"gamma":2.349446308922097,"min_child_weight":0.4623536055943267,"max_leaves":73,"rounds":1600}'
K=rev,flip,revflip,c1,c2,c3,c4
for s in 0 1; do while tasklist | grep -qi slay; do sleep 60; done; t=VARC4ht_ns3_s$s; $PY experiments/v6/train_var.py --tag $t --aug --seed $s --kinds $K --tta-kinds $K --extra hv,ht --params "$P1" > tune_out/log_$t.txt 2>&1; tail -2 tune_out/log_$t.txt | head -1; done
$PY experiments/v6/wtta.py --crop-w 0.125 --out VARC4ht_ns3_wc VARC4ht_ns3_s0 VARC4ht_ns3_s1 | tail -1
$PY experiments/v6/nnstack.py --base submissions/sub_seqw5_stack.json --add VARC4ht_ns3_wc --drop VARC4hv_ns3_wc 2>&1 | tail -2
K5=rev,flip,revflip,c1,c2
$PY experiments/v6/train_var.py --tag EXP_cggd_ht --aug --kinds $K5 --tta-kinds $K5 --extra hv,ht --colfile tune_out/var/cols_cggd.csv --export tune_out/var/mat_cggd_ht.npz 2>&1 | tail -1
$XP experiments/v6/nn_var.py --tag MLPHT_big --mat tune_out/var/mat_cggd_ht.npz --aug --extra --kinds $K5 --seeds 4 --hidden 1024 --layers 3 --drop 0.4 2>&1 | grep -v "^  fold" | tail -2
$PY experiments/v6/wtta.py --out MLPHT_big_w MLPHT_big | tail -1
echo QUEUE_DONE
