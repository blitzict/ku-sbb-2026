#!/bin/bash
# Second crop set: ns3 with rev/flip/revflip/c1/c2 copies + weighted TTA over all, 2 seeds; starts after c2 features and the cluster queue.
cd /c/Users/Blitz/Downloads/Code/claude/ai
until [ -f tune_out/var/dd_train_c2.parquet ] && grep -q QUEUE_DONE tune_out/q/cs.log 2>/dev/null; do sleep 30; done
PY=.venv/Scripts/python
wait_game() { while tasklist | grep -qi slay; do echo "game running, waiting"; sleep 60; done; }
P1='{"max_bin":64,"learning_rate":0.009549022390375738,"subsample":0.8624554592415571,"colsample_bytree":0.3282007743260973,"colsample_bynode":0.7222561808470165,"reg_lambda":0.2992087055688533,"reg_alpha":0.01697153710754649,"gamma":2.349446308922097,"min_child_weight":0.4623536055943267,"max_leaves":73,"rounds":1600}'
K="--kinds rev,flip,revflip,c1,c2 --tta-kinds rev,flip,revflip,c1,c2"
for s in 0 1; do wait_game; t=VARC2_ns3_s$s; $PY experiments/v6/train_var.py --tag $t --aug --seed $s $K --params "$P1" > tune_out/log_$t.txt 2>&1; tail -2 tune_out/log_$t.txt | head -1; done
echo QUEUE_DONE
