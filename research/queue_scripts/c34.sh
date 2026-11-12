#!/bin/bash
# Crop sets 3 (10-40% trims, seed 37) and 4 (milder 5-20% trims, seed 41): Arc features, CPU numpy features, then ns3 on the 5070 with all 7 copy kinds, 2 seeds.
cd /c/Users/Blitz/Downloads/Code/claude/ai
PY=.venv/Scripts/python; XPY=../../.venv-xpu/Scripts/python
wait_game() { while tasklist | grep -qi slay; do echo "game running, waiting"; sleep 60; done; }
$PY experiments/v6/make_crops.py c3 37 0.1 0.4; $PY experiments/v6/make_crops.py c4 41 0.05 0.2
for st in test_c3 train_c3 test_c4 train_c4; do
  (cd experiments/v6 && $XPY features_lr.py var:$st lr5 2>&1 | tail -1; $XPY features_garch.py var:$st 2>&1 | tail -1; LRZ_TAG=lrz6 $XPY features_lrz.py var:$st 2>&1 | tail -1; $XPY features_sn.py var:$st 2>&1 | tail -1)
  $PY experiments/v6/var_numpy_features.py $st > tune_out/logs/log_var_numpy_$st.txt 2>&1; tail -1 tune_out/logs/log_var_numpy_$st.txt
done
echo FEATURES_DONE
P1='{"max_bin":64,"learning_rate":0.009549022390375738,"subsample":0.8624554592415571,"colsample_bytree":0.3282007743260973,"colsample_bynode":0.7222561808470165,"reg_lambda":0.2992087055688533,"reg_alpha":0.01697153710754649,"gamma":2.349446308922097,"min_child_weight":0.4623536055943267,"max_leaves":73,"rounds":1600}'
K="--kinds rev,flip,revflip,c1,c2,c3,c4 --tta-kinds rev,flip,revflip,c1,c2,c3,c4"
for s in 0 1; do wait_game; t=VARC4_ns3_s$s; $PY experiments/v6/train_var.py --tag $t --aug --seed $s $K --params "$P1" > tune_out/log_$t.txt 2>&1; tail -2 tune_out/log_$t.txt | head -1; done
echo QUEUE_DONE
