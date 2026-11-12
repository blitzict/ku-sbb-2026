#!/bin/bash
# CatBoost (5070) on cggd with all 7 copy kinds + weighted TTA; waits for the c34 queue (one 5070 job at a time).
cd /c/Users/Blitz/Downloads/Code/claude/ai; PY=.venv/Scripts/python
until grep -q QUEUE_DONE tune_out/q/c34.log 2>/dev/null && grep -q QUEUE_DONE tune_out/q/ptune.log 2>/dev/null; do sleep 30; done
while tasklist | grep -qi slay; do echo "game running, waiting"; sleep 60; done
K=rev,flip,revflip,c1,c2,c3,c4; t=VARC_cat_s0
$PY experiments/v6/train_var.py --tag $t --aug --model cat --seed 0 --kinds $K --tta-kinds $K --colfile tune_out/var/cols_cggd.csv > tune_out/log_$t.txt 2>&1; tail -2 tune_out/log_$t.txt | head -1
$PY experiments/v6/wtta.py --out ${t}_w $t | tail -1
echo QUEUE_DONE
