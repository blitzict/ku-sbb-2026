#!/bin/bash
# Time-scale augmentation screen (d2a/d2b decimation, a2 aggregation) on the Arc-family proxy (lr5+gh+lrz+sn+lm, 164 cols), then XGB random-forest member.
cd /c/Users/Blitz/Downloads/Code/claude/ai; PY=.venv/Scripts/python; XPY=../../.venv-xpu/Scripts/python
wait_game() { while tasklist | grep -qi slay; do sleep 60; done; }
for st in test_d2a train_d2a test_d2b train_d2b test_a2 train_a2; do
  (cd experiments/v6 && $XPY features_lr.py var:$st lr5 2>&1 | tail -1; $XPY features_garch.py var:$st 2>&1 | tail -1; LRZ_TAG=lrz6 $XPY features_lrz.py var:$st 2>&1 | tail -1; $XPY features_sn.py var:$st 2>&1 | tail -1; $XPY features_lm.py var:$st 2>&1 | tail -1)
done
echo ARC_FEATURES_DONE
P1='{"max_bin":64,"learning_rate":0.009549022390375738,"subsample":0.8624554592415571,"colsample_bytree":0.3282007743260973,"colsample_bynode":0.7222561808470165,"reg_lambda":0.2992087055688533,"reg_alpha":0.01697153710754649,"gamma":2.349446308922097,"min_child_weight":0.4623536055943267,"max_leaves":73,"rounds":1600}'
B=rev,flip,revflip,c1,c2,c3,c4
run() { wait_game; t=$1; shift; $PY experiments/v6/train_var.py --tag $t --aug --colfile tune_out/q/arcfam.csv --params "$P1" "$@" > tune_out/log_$t.txt 2>&1; tail -2 tune_out/log_$t.txt | head -1; }
run PX_base_s0 --kinds $B --tta-kinds $B
run PX_dec_s0 --kinds $B,d2a,d2b --tta-kinds $B,d2a,d2b
run PX_agg_s0 --kinds $B,a2 --tta-kinds $B,a2
run PX_dec_tta0_s0 --kinds $B,d2a,d2b --tta-kinds $B
for t in PX_base_s0 PX_dec_s0 PX_agg_s0 PX_dec_tta0_s0; do $PY experiments/v6/wtta.py --crop-w 0.125 --out ${t}_wc $t | tail -1; done
echo PROXY_DONE
RF='{"learning_rate":1.0,"num_parallel_tree":400,"subsample":0.632,"colsample_bytree":1.0,"colsample_bynode":0.33,"grow_policy":"depthwise","max_depth":14,"max_leaves":0,"min_child_weight":4,"reg_lambda":1.0,"reg_alpha":0,"gamma":0,"max_bin":128,"rounds":1}'
wait_game; t=VARC4_rf_s0; $PY experiments/v6/train_var.py --tag $t --aug --kinds $B --tta-kinds $B --params "$RF" > tune_out/log_$t.txt 2>&1; tail -2 tune_out/log_$t.txt | head -1
$PY experiments/v6/wtta.py --crop-w 0.125 --out VARC4_rf_wc VARC4_rf_s0 | tail -1
$PY experiments/v6/nnstack.py --base submissions/sub_c4_stack.json --add VARC4_rf_wc 2>&1 | tail -2
echo QUEUE_DONE
