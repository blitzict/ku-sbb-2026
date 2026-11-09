# try4: slimmed try3 ⭐ selected (0.87583 public, 0.8724 local)

The "fewest features" version of the full 0.879 recipe ([try3](../try3/README.md)), built to meet the inference limit (30 minutes on 2 CPU cores, 8 GB RAM). Each fold keeps its **top 100 features** from 35 label-free tables and adds **36 noise-law prior columns**. Three tree models per fold (2 LightGBM, 1 XGBoost) are rank-blended. Kaggle submission **58142084** scored **0.87583** public.

Inference on the 4,000 test series takes **about 22 minutes on 2 CPU cores** (2.6 GB peak RAM), measured on Kaggle pinned to 2 cores.

| Version | Local OOF AUC | Kaggle public |
|---|---:|---:|
| top 100 features from 39 tables (earlier version, submission 58130092) | 0.8727 | 0.87659 |
| **this model**: the 4 most expensive tables dropped (lrz6, garch, bounds, base features) | **0.8724** | **0.87583** |

Dropping those 4 tables saves about 7 minutes of inference for -0.0004 local AUC.

**How the validation stays honest:** the folds are the seed-7 folds the law priors were fitted on. In each fold, features are ranked on that fold's training rows only, and the prior is fitted on that fold's training labels only. Tables that are priors fitted on all training labels (`transition_odds`, `transport_state`) are excluded.

## Run

```console
#1 raw competition files -> the 35 feature tables + law priors (only the 77 recipe steps try4 needs)
python try4/build_features.py --data data --work try4_run --workers 4
#2 train + save the 15 models (reproduces model/ exactly)
python try4/train.py --work try4_run --out try4/model
#3 score new raw series with the saved models -> id,break_probability
python try4/predict.py --series data/test_series.parquet --ref try4_run --out predict_try4 --csv submission.csv --workers 2
```

Or open `inference.ipynb`, which wraps step 3. `--ref` is the folder from step 1: a few feature formulas compare a series against training-side reference tables, and the competition data can't be stored in this repo. The final blend averages ranks, so score all test series in one run.

## Files

- `build_features.py`: raw files to feature tables, using the full recipe's runner and code.
- `train.py`: trains and saves the model. `TABLES` lists the 35 tables.
- `predict.py`: loads the saved models and scores new raw series, running independent steps in parallel.
- `slim.py`, `slim_mech.py`, `slim_moves.py`, `slim_white.py`, `slim_cols.py`: faster versions of the slow feature steps. They run the same code but compute only the columns the saved models read, so the values are identical (checked against the full steps; `KU_TRY4_NO_SLIM=1` switches them off).
- `select_features.py`: the feature-count sweep used to pick k = 100.
- `model/`: 15 fold models, each fold's chosen columns (`columns_fold*.json`), the fitted fold priors (`law_outer7/`) and `results.json`.
- `submission.csv`: this model's test predictions (submission 58142084).
