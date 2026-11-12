#!/bin/bash
# Arc MLP with crop copies (c1,c2) on the cggd table: export (short CPU), then big MLP 4 seeds; compare to VARN_mlp_big (0.8381 tta).
cd /c/Users/Blitz/Downloads/Code/claude/ai
K=rev,flip,revflip,c1,c2
.venv/Scripts/python experiments/v6/train_var.py --tag EXP_cggd_c --aug --kinds $K --tta-kinds $K --colfile tune_out/var/cols_cggd.csv --export tune_out/var/mat_cggd_c.npz 2>&1 | tail -1
.venv-xpu/Scripts/python experiments/v6/nn_var.py --tag VARNC_mlp_big --mat tune_out/var/mat_cggd_c.npz --aug --kinds $K --seeds 4 --hidden 1024 --layers 3 --drop 0.4 2>&1 | grep -v "^  fold" 
.venv/Scripts/python experiments/v6/wtta.py --out VARNC_mlp_big_w VARNC_mlp_big | tail -1
echo QUEUE_DONE
