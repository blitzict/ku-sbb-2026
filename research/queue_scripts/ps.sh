#!/bin/bash
# persistence/shape block ps_: compute test+aug on the Arc, then ns3+ps (22) and ns3+ps8, no aug, 2 seeds; baseline 0.84619 / 0.84594
cd /c/Users/Blitz/Downloads/Code/claude/ai; PY=.venv/Scripts/python
(cd experiments/v6 && ../../.venv-xpu/Scripts/python features_ps.py test | tail -1 && ../../.venv-xpu/Scripts/python features_ps.py aug | tail -1)
P1='{"max_bin":64,"learning_rate":0.009549022390375738,"subsample":0.8624554592415571,"colsample_bytree":0.3282007743260973,"colsample_bynode":0.7222561808470165,"reg_lambda":0.2992087055688533,"reg_alpha":0.01697153710754649,"gamma":2.349446308922097,"min_child_weight":0.4623536055943267,"max_leaves":73,"rounds":1600}'
for cf in ps ps8; do for s in 0 1; do while tasklist | grep -qi slay; do sleep 60; done; t=PS_ns3_${cf}_s$s
$PY experiments/v6/train_var.py --tag $t --seed $s --kinds "" --tta-kinds "" --colfile tune_out/selected_ns3_$cf.csv --params "$P1" > tune_out/log_$t.txt 2>&1; tail -2 tune_out/log_$t.txt | head -1; done; done
echo QUEUE_DONE
