#!/bin/bash
# 3-class lookalike target, ns3, no copies; q 0.6 and 0.75, seed 0 (binary baseline ZP_ns3_s0 0.84619 / 0.84214). Waits for the lm queue (one 5070 job).
cd /c/Users/Blitz/Downloads/Code/claude/ai; PY=.venv/Scripts/python
[ -f tune_out/var/mat_ns3_plain.npz ] || $PY experiments/v6/train_var.py --tag EXP --kinds "" --tta-kinds "" --export tune_out/var/mat_ns3_plain.npz | tail -1
until grep -q QUEUE_DONE tune_out/q/lm.log 2>/dev/null; do sleep 30; done
for q in 0.6 0.75; do while tasklist | grep -qi slay; do sleep 60; done; $PY experiments/v6/three_class.py --npz tune_out/var/mat_ns3_plain.npz --tag TC_q${q}_s0 --q $q 2>&1 | grep -v "^  fold" ; done
echo QUEUE_DONE
