#!/bin/bash
# length-matched real segment rows (make_halves.py) on the Arc-family proxy; ref PX_base_s0_wc 0.84547 / 0.83923
cd /c/Users/Blitz/Downloads/Code/claude/ai; PY=.venv/Scripts/python; XPY=../../.venv-xpu/Scripts/python
(cd experiments/v6 && for f in "features_lr.py var:train_hv lr5" "features_garch.py var:train_hv" "features_sn.py var:train_hv" "features_lm.py var:train_hv"; do $XPY $f 2>&1 | tail -1; done; LRZ_TAG=lrz6 $XPY features_lrz.py var:train_hv 2>&1 | tail -1)
echo HV_FEATURES_DONE
until grep -q QUEUE_DONE tune_out/q/kw2.log; do sleep 60; done
P1='{"max_bin":64,"learning_rate":0.009549022390375738,"subsample":0.8624554592415571,"colsample_bytree":0.3282007743260973,"colsample_bynode":0.7222561808470165,"reg_lambda":0.2992087055688533,"reg_alpha":0.01697153710754649,"gamma":2.349446308922097,"min_child_weight":0.4623536055943267,"max_leaves":73,"rounds":1600}'
B=rev,flip,revflip,c1,c2,c3,c4
run() { while tasklist | grep -qi slay; do sleep 60; done; t=$1; shift; $PY experiments/v6/train_var.py --tag $t --aug --kinds $B --tta-kinds $B --colfile tune_out/q/arcfam.csv --params "$P1" "$@" > tune_out/log_$t.txt 2>&1; tail -2 tune_out/log_$t.txt | head -1; $PY experiments/v6/wtta.py --crop-w 0.125 --out ${t}_wc $t | tail -1; }
run PX_hv_s0 --extra hv
run PX_hv05_s0 --extra hv --extra-w 0.5
run PX_hvneg_s0 --extra hv --extra-kinds h0,h1 --extra-w 0.25
echo QUEUE_DONE
