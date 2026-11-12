#!/bin/bash
cd /c/Users/Blitz/Downloads/Code/claude/ai; PY=.venv/Scripts/python; XPY=.venv-xpu/Scripts/python; K=rev,flip,revflip,c1,c2
for o in 1 2; do $XPY experiments/v6/nn_var.py --tag MLPHT_big_o$o --mat tune_out/var/mat_cggd_ht.npz --aug --extra --kinds $K --seeds 4 --hidden 1024 --layers 3 --drop 0.4 --seed-offset $o 2>&1 | grep -v "^  fold" | tail -1; done
$PY experiments/v6/wtta.py --out MLPHT_bag_w MLPHT_big MLPHT_big_o1 MLPHT_big_o2 | tail -1
$PY - <<'PY'
import sys; sys.path.insert(0, "experiments/follow_gpt")
from common import *
report("view recipe, MLPHV bag (unedited heads)", gpt_design(seq="views", mlp=(["MLPHV_big", "MLPHV_big_o1", "MLPHV_big_o2"], O)))
report("view recipe, MLPHT bag (unedited heads)", gpt_design(seq="views", mlp=(["MLPHT_big", "MLPHT_big_o1", "MLPHT_big_o2"], O)))
report("view recipe, MLPHV bag (GPT-edited heads)", gpt_design(seq="views"))
PY
echo QUEUE_DONE
