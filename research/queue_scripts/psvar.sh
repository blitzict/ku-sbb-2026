#!/bin/bash
# ps_ block for the 7 copy kinds (Arc), then VARC4 (7 kinds, P1) + ps, 2 seeds -> wtta crop-w 0.125 -> stack swap for VARC4_ns3_wc
cd /c/Users/Blitz/Downloads/Code/claude/ai; PY=.venv/Scripts/python
for k in rev flip revflip c1 c2 c3 c4; do for sp in train test; do
  [ -f tune_out/var/ps_${sp}_$k.parquet ] || (cd experiments/v6 && ../../.venv-xpu/Scripts/python features_ps.py var:${sp}_$k | tail -1); done; done
echo FEATURES_DONE
until grep -q QUEUE_DONE tune_out/q/ps.log 2>/dev/null; do sleep 30; done
CF=$(cat tune_out/q/ps_choice.txt 2>/dev/null || echo ps)
P1='{"max_bin":64,"learning_rate":0.009549022390375738,"subsample":0.8624554592415571,"colsample_bytree":0.3282007743260973,"colsample_bynode":0.7222561808470165,"reg_lambda":0.2992087055688533,"reg_alpha":0.01697153710754649,"gamma":2.349446308922097,"min_child_weight":0.4623536055943267,"max_leaves":73,"rounds":1600}'
K="--kinds rev,flip,revflip,c1,c2,c3,c4 --tta-kinds rev,flip,revflip,c1,c2,c3,c4"
for s in 0 1; do while tasklist | grep -qi slay; do sleep 60; done; t=VARC4_${CF}_s$s; $PY experiments/v6/train_var.py --tag $t --aug --seed $s $K --colfile tune_out/selected_ns3_$CF.csv --params "$P1" > tune_out/log_$t.txt 2>&1; tail -2 tune_out/log_$t.txt | head -1; done
$PY experiments/v6/wtta.py --crop-w 0.125 --out VARC4_${CF}_wc VARC4_${CF}_s0 VARC4_${CF}_s1 | tail -1
$PY experiments/v6/nnstack.py --base submissions/sub_c4_stack.json --add VARC4_${CF}_wc --drop VARC4_ns3_wc 2>&1 | tail -2
$PY experiments/v6/nnstack.py --base submissions/sub_c4_stack.json --add VARC4_${CF}_wc 2>&1 | tail -2
echo QUEUE_DONE
