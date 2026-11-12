#!/bin/bash
# cggd MLP with all 7 copy kinds (Arc), 3 seed sets; waits for c3/c4 features.
cd /c/Users/Blitz/Downloads/Code/claude/ai
until grep -q FEATURES_DONE tune_out/q/c34.log 2>/dev/null; do sleep 30; done
K=rev,flip,revflip,c1,c2,c3,c4
.venv/Scripts/python experiments/v6/train_var.py --tag EXP_cggd_c7 --aug --kinds $K --tta-kinds $K --colfile tune_out/var/cols_cggd.csv --export tune_out/var/mat_cggd_c7.npz 2>&1 | tail -1
for o in 0 1 2; do .venv-xpu/Scripts/python experiments/v6/nn_var.py --tag VARNC7_mlp_o$o --mat tune_out/var/mat_cggd_c7.npz --aug --kinds $K --seeds 4 --hidden 1024 --layers 3 --drop 0.4 --seed-offset $o 2>&1 | grep -E "plain|Error|Trace"; done
.venv/Scripts/python experiments/v6/wtta.py --out VARNC7_mlp_bag_w VARNC7_mlp_o0 VARNC7_mlp_o1 VARNC7_mlp_o2 | tail -1
echo QUEUE_DONE
