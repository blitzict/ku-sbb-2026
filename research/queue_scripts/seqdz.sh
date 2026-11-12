#!/bin/bash
cd /c/Users/Blitz/Downloads/Code/claude/ai; XPY=.venv-xpu/Scripts/python; PY=.venv/Scripts/python
$XPY experiments/v6/nn_seq.py --tag W_SEQdz_q --epochs 45 --patience 12 --qin 16 --seeds 0 --chans z,dz,e,ae,e2,ind,pos 2>&1 | grep "CV tta" | tail -1
$PY experiments/v6/nnstack.py --base submissions/sub_seqw5_stack.json --add W_SEQdz_q 2>&1 | tail -2
echo QUEUE_DONE
