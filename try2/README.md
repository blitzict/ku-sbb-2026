# try2: 0.875 model (load and predict)

This is a single LightGBM model, saved as 5 fold models, trained on 3,625 label-free features: the base, statistical and mechanism tables listed in `../extra/FEATURES.md`. Out-of-fold AUC is **0.86987**; Kaggle public score **0.875**, using 5 stratified folds with seed 7.

- `model/`: `fold0.txt` to `fold4.txt`, `columns.json` (feature order) and `model.json` (the settings).
- `train_compact.py`: trains the model from the CPU feature tables made by `train_087910_cpu.py`.
- `predict_compact.py`: loads the saved models and scores new raw series. It rebuilds every feature with the same frozen scripts used in training and writes `id,break_probability`. Nothing is refit.
- `inference.ipynb`: notebook wrapper around `predict_compact.py`. Run on the 4,000 raw test series, it reproduced the submitted predictions exactly (max difference 0).

```console
python try2/predict_compact.py --series data/test_series.parquet --ref <training work folder> --out predict_0870 --csv submission.csv
```

`--ref` is the training work folder. A few feature formulas compare against training-side reference tables, so the script needs that folder; the competition data is not stored in this repo.

**Check:** on 200 real test series run from raw data, all 3,625 features and every prediction matched the training-time tables exactly (max difference 0).

This model is the stand-in until the full 0.879 run finishes and its saved models can be loaded the same way.

## Rebuild from scratch

All the code is in this repo. Starting from the four competition files in `data/`:

```console
#1 features: computes every table this model uses (stages before the model phase)
python train_087910_cpu.py --data data --work cpu_run --workers 6 --phase features
#2 train the 5 fold models and save them, with column order and out-of-fold AUC
python try2/train_compact.py --work cpu_run --out try2/model
#3 score new series with the saved models
python try2/predict_compact.py --series data/test_series.parquet --ref cpu_run --out predict_try2 --csv submission.csv
```

The feature code is `pipeline/features_v4.py` plus the scripts under `pipeline/legacy_cpu/`. They are listed in `../extra/FEATURES.md`, and the exact steps are in `pipeline/full_cpu_stages.json`.

## Original probe

`pipeline/probe_compact_cpu.py` is the one-off development script that first measured this model (0.869869 out-of-fold AUC):

```console
python pipeline/probe_compact_cpu.py --features cpu_run --output probe_out
```

`try2/train_compact.py` uses the same tables, settings and folds, gives identical fold scores, and also saves the column order and test predictions.
