# try3: compact model (0.87659 public, 0.8727 local)

This is the "fewest features" version of the 0.879 recipe. Each fold keeps only its **top 100 features**, out of the ~5,100 label-free columns, and adds **36 noise-law prior columns**. Three tree models are trained per fold and rank-blended. Kaggle submission **58130092** scored **0.87659** public.

| Step | Local OOF AUC |
|---|---:|
| one LightGBM, top 100 features | 0.8708 |
| + fold-wise noise-law prior (36 columns) | 0.8723 |
| + rank blend of 2 LightGBM + 1 XGBoost | **0.8727** |

For comparison, try2 scores 0.8699 local with 3,625 features, and all 5,097 columns in one model give 0.8712. Across the 5 folds the model uses 187 distinct features in total. The biggest families are native law (state), latent level (pulse), the base statistics, grid2 and boundary law. The full ranking is in `model/features.json`.

**How the validation stays honest:** the folds are the seed-7 folds the law priors were fitted on. In each fold, the features are ranked using that fold's training rows only, and the prior is fitted on that fold's training labels only. The held-out rows never influence anything they are scored with. Two tables, `transition_odds` and `transport_state`, are priors fitted on all training labels, so they are excluded to avoid leakage.

## Files

- `select_features.py`: the feature-count sweep (AUC with 25 up to all 5,097 features).
- `train.py`: trains and saves the model.
- `predict.py`: loads the saved models and scores new raw series.
- `inference.ipynb`: notebook wrapper around `predict.py`.
- `model/`:
  - `lgb_law_fold*.txt`, `lgb_law_s2_fold*.txt`, `xgb_law_fold*.json`: 15 fitted models.
  - `columns_fold*.json`: each fold's chosen features.
  - `law_outer7/fold*_prior.json`: the fitted priors.
  - `results.json`: the scores above.

## Run

```console
#features from raw data (only the feature steps are needed)
python train_087910_cpu.py --data data --work cpu_run --workers 6 --phase features
#train + save (the training run that made model/ reproduced the submitted CSV exactly)
python try3/train.py --work cpu_run --out try3/model --k 100
#score new raw series with the saved models
python try3/predict.py --series data/test_series.parquet --ref cpu_run --out predict_try3 --csv submission.csv
```

`--ref` is the training work folder. A few feature formulas compare against training-side tables, so the predictor needs that folder; it can't be in this repo because it holds competition data. The final blend averages ranks, so each model's score is relative to the batch being scored. Score all test series in one run.

**Check:** `predict.py` was run from raw data on 200 real test series. The two LightGBM models matched the training-time scores exactly, and XGBoost differed by at most 3e-8. It took about 8 minutes with 4 workers on a busy machine.
