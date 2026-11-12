#!/bin/bash
# Crop versions (7 kinds, crop-w 0.125) of the family / ns1sel1 members; yields the 5070 when c56 features finish.
cd /c/Users/Blitz/Downloads/Code/claude/ai; PY=.venv/Scripts/python; TV=experiments/v6/train_var.py; K=rev,flip,revflip,c1,c2,c3,c4
run() { grep -q FEATURES_DONE tune_out/q/c56.log && { echo "yield"; exit 0; }; while tasklist | grep -qi slay; do sleep 60; done; tag=$1; shift; $PY $TV --tag $tag --aug --kinds $K --tta-kinds $K "$@" > tune_out/log_$tag.txt 2>&1; echo "$(tail -2 tune_out/log_$tag.txt | head -1)"; }
for m in gh lr; do for s in 0 1; do run VARFC_${m}_s$s --seed $s --colfile tune_out/loop/fam_$m.csv; done
  $PY experiments/v6/wtta.py --crop-w 0.125 --out VARFC_${m}_wc VARFC_${m}_s0 VARFC_${m}_s1 | tail -1; done
for s in 0 1; do run VARC_ns1sel1_s$s --seed $s --colfile tune_out/selected_ns1_sel1.csv --params '{"colsample_bytree":0.7}'; done
$PY experiments/v6/wtta.py --crop-w 0.125 --out VARC_ns1sel1_wc VARC_ns1sel1_s0 VARC_ns1sel1_s1 | tail -1
echo QUEUE_DONE
