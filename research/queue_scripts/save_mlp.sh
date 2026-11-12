#!/bin/bash
# retrain the MLP bag (3 runs x 4 seeds) with weight saving -> weights/W_VARNC_mlp_big*/ ; Arc
cd /c/Users/Blitz/Downloads/Code/claude/ai; XPY=.venv-xpu/Scripts/python; PY=.venv/Scripts/python; K=rev,flip,revflip,c1,c2
for o in 0 1 2; do t=W_VARNC_mlp_big$([ $o -gt 0 ] && echo _o$o)
$XPY experiments/v6/nn_var.py --tag $t --mat tune_out/var/mat_cggd_c.npz --aug --kinds $K --seeds 4 --hidden 1024 --layers 3 --drop 0.4 --seed-offset $o 2>&1 | grep -v "^  fold" | tail -2; done
$PY experiments/v6/wtta.py --out W_VARNC_mlp_bag_w W_VARNC_mlp_big W_VARNC_mlp_big_o1 W_VARNC_mlp_big_o2 | tail -1
echo QUEUE_DONE
