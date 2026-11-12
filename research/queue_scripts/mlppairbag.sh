#!/bin/bash
cd /c/Users/Blitz/Downloads/Code/claude/ai; PY=.venv/Scripts/python; XPY=.venv-xpu/Scripts/python; K=rev,flip,revflip,c1,c2
for o in 1 2; do $XPY experiments/v6/nn_var.py --tag MLPPAIR_0.5_o$o --mat tune_out/var/mat_cggd_hv.npz --aug --extra --kinds $K --seeds 4 --hidden 1024 --layers 3 --drop 0.4 --pair-w 0.5 --seed-offset $o 2>&1 | grep -v "^  fold" | tail -1; done
$PY experiments/v6/wtta.py --out MLPPAIR_bag_w MLPPAIR_0.5 MLPPAIR_0.5_o1 MLPPAIR_0.5_o2 | tail -1
$PY - <<'PY'
import sys; sys.path.insert(0, "experiments/follow_gpt")
from common import *
P3 = (["MLPPAIR_0.5", "MLPPAIR_0.5_o1", "MLPPAIR_0.5_o2"], O)
ref = gpt_design(seq="views"); report("ref: GPT recipe + SEQ views (edited MLP heads)", ref)
report("pairwise-trained MLP bag instead", gpt_design(seq="views", mlp=P3))
report("both MLP bags", np.column_stack([ref] + ver(P3[0], V4 + ["c2"], base=O)))
PY
echo QUEUE_DONE
