#!/bin/bash
# crop MLP + both segment sets (Arc), 4 seeds; ref MLPHV_big_w 0.84143 / 0.83459 (one set)
cd /c/Users/Blitz/Downloads/Code/claude/ai; PY=.venv/Scripts/python; XPY=.venv-xpu/Scripts/python; K=rev,flip,revflip,c1,c2
$PY experiments/v6/train_var.py --tag EXP_cggd_hv2 --aug --kinds $K --tta-kinds $K --extra hv,hvb --colfile tune_out/var/cols_cggd.csv --export tune_out/var/mat_cggd_hv2.npz 2>&1 | tail -1
until grep -q QUEUE_DONE tune_out/q/seqhv.log; do sleep 60; done
$XPY experiments/v6/nn_var.py --tag MLPHV2_big --mat tune_out/var/mat_cggd_hv2.npz --aug --extra --kinds $K --seeds 4 --hidden 1024 --layers 3 --drop 0.4 2>&1 | grep -v "^  fold" | tail -2
$PY experiments/v6/wtta.py --out MLPHV2_big_w MLPHV2_big | tail -1
echo QUEUE_DONE
