#!/bin/bash
# XGB leaf-weight refresh screen (seed 0; base = same model with plain wtta, VARC4 s0 ref 0.85040)
cd /c/Users/Blitz/Downloads/Code/claude/ai; PY=.venv/Scripts/python; K=rev,flip,revflip,c1,c2,c3,c4
until grep -q QUEUE_DONE tune_out/q/lc.log; do sleep 60; done
[ -f tune_out/var/mat_ns3_c7.npz ] || $PY experiments/v6/train_var.py --tag exp_c7 --aug --kinds $K --tta-kinds $K --export tune_out/var/mat_ns3_c7.npz | tail -1
while tasklist | grep -qi slay; do sleep 60; done
$PY experiments/v6/leaf_refresh.py --npz tune_out/var/mat_ns3_c7.npz --tag LREF --copy-w 0 0.25 2>&1 | grep -v "^  fold" 
echo QUEUE_DONE
