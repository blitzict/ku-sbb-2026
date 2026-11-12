#!/bin/bash
# Crop augmentation tests on the 5070 (starts when the CPU crop features exist). Same params as the kept aug members, kinds + c1, TTA + c1.
cd /c/Users/Blitz/Downloads/Code/claude/ai
until [ -f tune_out/var/dd_train_c1.parquet ]; do sleep 30; done
PY=.venv/Scripts/python
wait_game() { while tasklist | grep -qi slay; do echo "game running, waiting"; sleep 60; done; }
run() { wait_game; tag=$1; shift; $PY experiments/v6/train_var.py --tag $tag --aug "$@" > tune_out/log_$tag.txt 2>&1; echo "$(tail -2 tune_out/log_$tag.txt | head -1)"; }
P1='{"max_bin":64,"learning_rate":0.009549022390375738,"subsample":0.8624554592415571,"colsample_bytree":0.3282007743260973,"colsample_bynode":0.7222561808470165,"reg_lambda":0.2992087055688533,"reg_alpha":0.01697153710754649,"gamma":2.349446308922097,"min_child_weight":0.4623536055943267,"max_leaves":73,"rounds":1600}'
K="--kinds rev,flip,revflip,c1 --tta-kinds rev,flip,revflip,c1"
for s in 0 1; do run VARC_ns3_s$s --seed $s $K --params "$P1"; done
for s in 0 1; do run VARC_full_s$s --seed $s $K --colfile tune_out/var/all_cols.csv --params '{"colsample_bytree":0.36,"rounds":800}'; done
echo QUEUE_DONE
