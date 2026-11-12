#!/bin/bash
# MLP (crops + segment rows) trained with BCE + in-batch pairwise loss; ref MLPHV_big_w 0.84143 / 0.83459 (4 seeds)
cd /c/Users/Blitz/Downloads/Code/claude/ai; PY=.venv/Scripts/python; XPY=.venv-xpu/Scripts/python; K=rev,flip,revflip,c1,c2
for pw in 0.5 2; do t=MLPPAIR_$pw
$XPY experiments/v6/nn_var.py --tag $t --mat tune_out/var/mat_cggd_hv.npz --aug --extra --kinds $K --seeds 4 --hidden 1024 --layers 3 --drop 0.4 --pair-w $pw 2>&1 | grep -v "^  fold" | tail -2
$PY experiments/v6/wtta.py --out ${t}_w $t | tail -1; done
echo QUEUE_DONE
