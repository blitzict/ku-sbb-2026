#!/bin/bash
# Per-family null-importance refinement (gh, lr families), then aug+TTA models with 4 seeds each on the 5070; seeds 2,3 of the unrefined
# families for a like-for-like 4-seed comparison. Waits while Slay the Spire runs.
cd /c/Users/Blitz/Downloads/Code/claude/ai
PY=.venv/Scripts/python
wait_game() { while tasklist | grep -qi slay; do echo "game running, waiting"; sleep 60; done; }
wait_game; $PY experiments/v6/null_select.py --tag fam_gh_sel --blocks gh --cols tune_out/loop/fam_gh.csv --nulls 16 2>&1 | grep kept
wait_game; $PY experiments/v6/null_select.py --tag fam_lr_sel --blocks lr5,lrz --cols tune_out/loop/fam_lr.csv --nulls 16 2>&1 | grep kept
run() { wait_game; tag=$1; shift; $PY experiments/v6/train_var.py --tag $tag --aug "$@" > tune_out/log_$tag.txt 2>&1; echo "$(tail -2 tune_out/log_$tag.txt | head -1)"; }
for f in gh lr; do
  for s in 0 1 2 3; do run VARF_${f}sel_s$s --seed $s --colfile tune_out/selected_fam_${f}_sel.csv; done
  for s in 2 3; do run VARF_${f}_s$s --seed $s --colfile tune_out/loop/fam_$f.csv; done
done
echo QUEUE_DONE
