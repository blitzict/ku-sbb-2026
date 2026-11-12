#!/bin/bash
# weights for the two older members of sub_c4hv_stack (VARC4_lm, VARC4_full); 5070
cd /c/Users/Blitz/Downloads/Code/claude/ai; PY=.venv/Scripts/python; K=rev,flip,revflip,c1,c2,c3,c4
P1='{"max_bin":64,"learning_rate":0.009549022390375738,"subsample":0.8624554592415571,"colsample_bytree":0.3282007743260973,"colsample_bynode":0.7222561808470165,"reg_lambda":0.2992087055688533,"reg_alpha":0.01697153710754649,"gamma":2.349446308922097,"min_child_weight":0.4623536055943267,"max_leaves":73,"rounds":1600}'
for s in 0 1; do while tasklist | grep -qi slay; do sleep 60; done; t=W_VARC4_lm_s$s; $PY experiments/v6/train_var.py --tag $t --aug --seed $s --kinds $K --tta-kinds $K --colfile tune_out/selected_ns3_lm.csv --params "$P1" > tune_out/log_$t.txt 2>&1; tail -2 tune_out/log_$t.txt | head -1; done
for s in 0 1; do while tasklist | grep -qi slay; do sleep 60; done; t=W_VARC4_full_s$s; $PY experiments/v6/train_var.py --tag $t --aug --seed $s --kinds $K --tta-kinds $K --colfile tune_out/var/all_cols.csv --params '{"colsample_bytree":0.36,"rounds":800}' > tune_out/log_$t.txt 2>&1; tail -2 tune_out/log_$t.txt | head -1; done
$PY - <<'PY'
import numpy as np
for o in ("VARC4_lm_s0","VARC4_lm_s1","VARC4_full_s0","VARC4_full_s1"):
    a, b = np.load(f"tune_out/test_{o}.npy"), np.load(f"tune_out/test_W_{o}.npy"); print(o, "max|diff| old vs saved", float(np.abs(a - b).max()))
PY
echo QUEUE_DONE
