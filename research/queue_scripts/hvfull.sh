#!/bin/bash
# full numpy blocks for the segment rows (train only), then ns3 VARC4 + hv (2 seeds) -> wtta -> stack; optionally with a2
cd /c/Users/Blitz/Downloads/Code/claude/ai; PY=.venv/Scripts/python
until grep -q A2_FEATURES_DONE tune_out/q/a2full.log; do sleep 60; done
$PY experiments/v6/var_numpy_features.py train_hv > tune_out/logs/log_var_numpy_train_hv.txt 2>&1; tail -1 tune_out/logs/log_var_numpy_train_hv.txt
echo HV_FULL_FEATURES_DONE
until grep -q QUEUE_DONE tune_out/q/a2full.log; do sleep 60; done
P1='{"max_bin":64,"learning_rate":0.009549022390375738,"subsample":0.8624554592415571,"colsample_bytree":0.3282007743260973,"colsample_bynode":0.7222561808470165,"reg_lambda":0.2992087055688533,"reg_alpha":0.01697153710754649,"gamma":2.349446308922097,"min_child_weight":0.4623536055943267,"max_leaves":73,"rounds":1600}'
K=rev,flip,revflip,c1,c2,c3,c4
for s in 0 1; do while tasklist | grep -qi slay; do sleep 60; done; t=VARC4hv_ns3_s$s; $PY experiments/v6/train_var.py --tag $t --aug --seed $s --kinds $K --tta-kinds $K --extra hv --params "$P1" > tune_out/log_$t.txt 2>&1; tail -2 tune_out/log_$t.txt | head -1; done
$PY experiments/v6/wtta.py --crop-w 0.125 --out VARC4hv_ns3_wc VARC4hv_ns3_s0 VARC4hv_ns3_s1 | tail -1
$PY experiments/v6/nnstack.py --base submissions/sub_c4_stack.json --add VARC4hv_ns3_wc --drop VARC4_ns3_wc 2>&1 | tail -2
$PY experiments/v6/nnstack.py --base submissions/sub_c4_stack.json --add VARC4hv_ns3_wc 2>&1 | tail -2
echo QUEUE_DONE
