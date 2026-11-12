#!/bin/bash
# 12-seed bag of the crop MLP + segment rows (Arc), then stack swap for VARNC_mlp_bag_w on sub_c4hv_stack
cd /c/Users/Blitz/Downloads/Code/claude/ai; PY=.venv/Scripts/python; XPY=.venv-xpu/Scripts/python; K=rev,flip,revflip,c1,c2
for o in 1 2; do $XPY experiments/v6/nn_var.py --tag MLPHV_big_o$o --mat tune_out/var/mat_cggd_hv.npz --aug --extra --kinds $K --seeds 4 --hidden 1024 --layers 3 --drop 0.4 --seed-offset $o 2>&1 | grep -v "^  fold" | tail -2; done
$PY experiments/v6/wtta.py --out MLPHV_bag_w MLPHV_big MLPHV_big_o1 MLPHV_big_o2 | tail -1
$PY experiments/v6/nnstack.py --base submissions/sub_c4hv_stack.json --add MLPHV_bag_w --drop VARNC_mlp_bag_w 2>&1 | tail -2
$PY experiments/v6/nnstack.py --base submissions/sub_c4hv_stack.json --add MLPHV_bag_w 2>&1 | tail -2
echo QUEUE_DONE
