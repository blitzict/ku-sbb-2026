#!/bin/bash
# Mild crop sets 5 (5-20%, seed 53) and 6 (5-30%, seed 59); ns3 with all 9 copy kinds, 2 seeds; training waits for the CatBoost queue.
cd /c/Users/Blitz/Downloads/Code/claude/ai
PY=.venv/Scripts/python; XPY=../../.venv-xpu/Scripts/python
wait_game() { while tasklist | grep -qi slay; do echo "game running, waiting"; sleep 60; done; }
$PY experiments/v6/make_crops.py c5 53 0.05 0.2; $PY experiments/v6/make_crops.py c6 59 0.05 0.3
for st in test_c5 train_c5 test_c6 train_c6; do
  (cd experiments/v6 && $XPY features_lr.py var:$st lr5 2>&1 | tail -1; $XPY features_garch.py var:$st 2>&1 | tail -1; LRZ_TAG=lrz6 $XPY features_lrz.py var:$st 2>&1 | tail -1; $XPY features_sn.py var:$st 2>&1 | tail -1)
  $PY experiments/v6/var_numpy_features.py $st > tune_out/logs/log_var_numpy_$st.txt 2>&1; tail -1 tune_out/logs/log_var_numpy_$st.txt
done
echo FEATURES_DONE
until grep -q QUEUE_DONE tune_out/q/catc.log 2>/dev/null; do sleep 30; done
P1='{"max_bin":64,"learning_rate":0.009549022390375738,"subsample":0.8624554592415571,"colsample_bytree":0.3282007743260973,"colsample_bynode":0.7222561808470165,"reg_lambda":0.2992087055688533,"reg_alpha":0.01697153710754649,"gamma":2.349446308922097,"min_child_weight":0.4623536055943267,"max_leaves":73,"rounds":1600}'
K="--kinds rev,flip,revflip,c1,c2,c3,c4,c5,c6 --tta-kinds rev,flip,revflip,c1,c2,c3,c4,c5,c6"
for s in 0 1; do wait_game; t=VARC6_ns3_s$s; $PY experiments/v6/train_var.py --tag $t --aug --seed $s $K --params "$P1" > tune_out/log_$t.txt 2>&1; tail -2 tune_out/log_$t.txt | head -1; done
echo QUEUE_DONE
