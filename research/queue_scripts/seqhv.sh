#!/bin/bash
# sequence model + segment rows (Arc). ref W_SEQ_q (seed 0, 45 ep) 0.83431 / 0.82751
cd /c/Users/Blitz/Downloads/Code/claude/ai; XPY=.venv-xpu/Scripts/python; PY=.venv/Scripts/python
$XPY experiments/v6/nn_seq.py --tag W_SEQHV_q --epochs 15 --patience 5 --qin 16 --seeds 0 --extra hv 2>&1 | grep "CV tta" | tail -1
$PY experiments/v6/nnstack.py --base submissions/sub_seqw5_stack.json --add W_SEQHV_q 2>&1 | tail -2
echo QUEUE_DONE
