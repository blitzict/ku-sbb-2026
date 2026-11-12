#!/bin/bash
# weight-space averaging (EMA of weights) for the crop MLP, Arc. Same settings as VARNC_mlp_big.
cd /c/Users/Blitz/Downloads/Code/claude/ai; XPY=.venv-xpu/Scripts/python; PY=.venv/Scripts/python; K=rev,flip,revflip,c1,c2
$PY experiments/v6/wtta.py --crop-w 0.125 --out tmp_mlpbig_ref VARNC_mlp_big | tail -1
for d in 0.999 0.9995; do t=EMA_mlp_big_$d
$XPY experiments/v6/nn_var.py --tag $t --mat tune_out/var/mat_cggd_c.npz --aug --kinds $K --seeds 4 --hidden 1024 --layers 3 --drop 0.4 --ema $d 2>&1 | grep -v "^  fold" | tail -2
$PY experiments/v6/wtta.py --crop-w 0.125 --out ${t}_w $t | tail -1; done
echo QUEUE_DONE
