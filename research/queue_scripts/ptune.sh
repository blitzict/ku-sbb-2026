#!/bin/bash
# Re-tune ns3 XGB params under augmentation (c1,c2 copies; the old params were tuned on originals only). Seed 0, compare to VARC2_ns3_s0.
# Yields the 5070 as soon as the c34 queue finishes its features (one 5070 job at a time).
cd /c/Users/Blitz/Downloads/Code/claude/ai; PY=.venv/Scripts/python
wait_game() { while tasklist | grep -qi slay; do echo "game running, waiting"; sleep 60; done; }
B='"max_bin":64,"learning_rate":0.009549022390375738,"subsample":0.8624554592415571,"colsample_bynode":0.7222561808470165,"reg_lambda":0.2992087055688533,"reg_alpha":0.01697153710754649,"gamma":2.349446308922097'
K="--kinds rev,flip,revflip,c1,c2 --tta-kinds rev,flip,revflip,c1,c2"
declare -A V
V[r2400]="{$B,\"colsample_bytree\":0.3282,\"min_child_weight\":0.4624,\"max_leaves\":73,\"rounds\":2400}"
V[l127]="{$B,\"colsample_bytree\":0.3282,\"min_child_weight\":1.0,\"max_leaves\":127,\"rounds\":1600}"
V[cs22r2400]="{$B,\"colsample_bytree\":0.22,\"min_child_weight\":0.4624,\"max_leaves\":73,\"rounds\":2400}"
V[mcw3]="{$B,\"colsample_bytree\":0.3282,\"min_child_weight\":3.0,\"max_leaves\":73,\"rounds\":1600}"
for v in r2400 l127 cs22r2400 mcw3; do
  grep -q FEATURES_DONE tune_out/q/c34.log && { echo "yielding to c34"; break; }
  wait_game; t=PT_${v}_s0; $PY experiments/v6/train_var.py --tag $t --aug --seed 0 $K --params "${V[$v]}" > tune_out/log_$t.txt 2>&1
  echo "$(tail -2 tune_out/log_$t.txt | head -1)"; $PY experiments/v6/wtta.py --out ${t}_w $t 2>&1 | tail -1
done
echo QUEUE_DONE
