#!/bin/bash
# XGB pairwise continuation (tree analogue of the GPT head refit); waits for the 5070 part of ht.sh
cd /c/Users/Blitz/Downloads/Code/claude/ai; PY=.venv/Scripts/python; K=rev,flip,revflip,c1,c2,c3,c4
until grep -q "VARC4ht_ns3_wc\|nnstack" tune_out/q/ht.log && ! tasklist | grep -qi slay; do sleep 60; done
until grep -q "nnstack" tune_out/q/ht.log; do sleep 30; done
[ -f tune_out/var/mat_ns3_c7hv.npz ] || $PY experiments/v6/train_var.py --tag exp_c7hv --aug --kinds $K --tta-kinds $K --extra hv --export tune_out/var/mat_ns3_c7hv.npz | tail -1
$PY experiments/follow_gpt/xgb_pair_finetune.py --npz tune_out/var/mat_ns3_c7hv.npz --tag PF_ns3_s0 2>&1
echo QUEUE_DONE
