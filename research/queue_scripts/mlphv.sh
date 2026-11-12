#!/bin/bash
# crop MLP + segment rows (Arc). ref W_VARNC_mlp_big: plain 0.83720, tta 0.83892; bag wtta 0.84189
cd /c/Users/Blitz/Downloads/Code/claude/ai; PY=.venv/Scripts/python; XPY=.venv-xpu/Scripts/python; K=rev,flip,revflip,c1,c2
$PY experiments/v6/train_var.py --tag EXP_cggd_hv --aug --kinds $K --tta-kinds $K --extra hv --colfile tune_out/var/cols_cggd.csv --export tune_out/var/mat_cggd_hv.npz 2>&1 | tail -1
$XPY experiments/v6/nn_var.py --tag MLPHV_big --mat tune_out/var/mat_cggd_hv.npz --aug --extra --kinds $K --seeds 4 --hidden 1024 --layers 3 --drop 0.4 2>&1 | grep -v "^  fold" | tail -2
$PY experiments/v6/wtta.py --out MLPHV_big_w MLPHV_big | tail -1
$PY experiments/v6/wtta.py --out tmp_mlpref_w W_VARNC_mlp_big | tail -1
echo QUEUE_DONE
