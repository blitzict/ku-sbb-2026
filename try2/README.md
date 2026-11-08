# try2: 0.875 model (load and predict)

This is a single LightGBM model, saved as 5 fold models, trained on 3,625 label-free features: the base, statistical and mechanism tables listed in `../FEATURES.md`. Out-of-fold AUC is **0.86987**; Kaggle public score **0.875**, using 5 stratified folds with seed 7.

- `model/`: `fold0.txt` to `fold4.txt`, `columns.json` (feature order) and `model.json` (the settings).
- `train_compact.py`: trains the model from the CPU feature tables made by `train_087910_cpu.py`.
- `predict_compact.py`: loads the saved models and scores new raw series. It rebuilds every feature with the same frozen scripts used in training and writes `id,break_probability`. Nothing is refit.
- `inference.ipynb`: notebook wrapper around `predict_compact.py`.

```console
python try2/predict_compact.py --series data/test_series.parquet --ref <training work folder> --out predict_0870 --csv submission.csv
```

`--ref` is the training work folder. A few feature formulas compare against training-side reference tables, so the script needs that folder; the competition data is not stored in this repo.

**Check:** on 200 real test series run from raw data, all 3,625 features and every prediction matched the training-time tables exactly (max difference 0).

This model is the stand-in until the full 0.879 run finishes and its saved models can be loaded the same way.
