#!/bin/bash
# 5070 queue: aug+TTA versions of the remaining stack member types, one job at a time; waits while Slay the Spire runs.
cd /c/Users/Blitz/Downloads/Code/claude/ai
PY=.venv/Scripts/python; TV=experiments/v6/train_var.py
run() { while tasklist | grep -qi slay; do echo "game running, waiting"; sleep 60; done; tag=$1; shift; $PY $TV --tag $tag --aug "$@" > tune_out/log_$tag.txt 2>&1; echo "$(tail -2 tune_out/log_$tag.txt | head -1)"; }
for s in 0 1; do
  for f in gh lr dd grid; do run VARF_${f}_s$s --seed $s --colfile tune_out/loop/fam_$f.csv; done
  run VAR_ns1sel1_s$s --seed $s --colfile tune_out/selected_ns1_sel1.csv --params '{"colsample_bytree":0.7}'
done
run VAR_cat_cggd_s0 --seed 0 --model cat --colfile tune_out/var/cols_cggd.csv
echo QUEUE_DONE
