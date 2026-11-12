#!/bin/bash
# per-kind training weights on VARC4 ns3 (seed 0 reference wtta 0.85040 / 0.84596). Waits for dec.sh.
cd /c/Users/Blitz/Downloads/Code/claude/ai; PY=.venv/Scripts/python
until grep -q QUEUE_DONE tune_out/q/dec.log; do sleep 60; done
P1='{"max_bin":64,"learning_rate":0.009549022390375738,"subsample":0.8624554592415571,"colsample_bytree":0.3282007743260973,"colsample_bynode":0.7222561808470165,"reg_lambda":0.2992087055688533,"reg_alpha":0.01697153710754649,"gamma":2.349446308922097,"min_child_weight":0.4623536055943267,"max_leaves":73,"rounds":1600}'
K=rev,flip,revflip,c1,c2,c3,c4
run() { while tasklist | grep -qi slay; do sleep 60; done; t=$1; shift; $PY experiments/v6/train_var.py --tag $t --aug --kinds $K --tta-kinds $K --params "$P1" "$@" > tune_out/log_$t.txt 2>&1; tail -2 tune_out/log_$t.txt | head -1; $PY experiments/v6/wtta.py --crop-w 0.125 --out ${t}_wc $t | tail -1; }
run KW_c05_s0 --kind-w c1=0.5,c2=0.5,c3=0.5,c4=0.5
run KW_c025_s0 --kind-w c1=0.25,c2=0.25,c3=0.25,c4=0.25
run KW_r05_s0 --kind-w rev=0.5,revflip=0.5
echo QUEUE_DONE
