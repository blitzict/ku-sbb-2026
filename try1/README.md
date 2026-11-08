# try1: v4 model (0.83313 on Kaggle)

This was the first submission. It uses two gradient-boosted tree models on the 977 base features (`pipeline/features_v4.py`), and its local 5-fold AUC is 0.8384.

- `model/xgboost.ubj` and `model/lightgbm.txt`: the saved models.
- `model/hyperparameters.json`: the tuned settings.
- `model/submission.csv`: the submitted predictions.
- `v4.py`: loads the saved models, builds the 977 features from the raw test series, and writes `id,break_probability`. The final score is the average of the XGBoost and LightGBM rank percentiles. `train` refits both models from the raw training series.

```console
python try1/v4.py predict --data data --out-dir try1_out
python try1/v4.py train --data data --out-dir try1_out --validate
```

**Check:** `predict`, run on the raw test series with 6 workers, reproduced the submitted `model/submission.csv` exactly in 122 seconds.

## Rebuild from scratch

All the code is here: the features are in `pipeline/features_v4.py`, and the training and prediction are in `try1/v4.py`. `python try1/v4.py train --data data --validate` builds the 977 features from the raw series, runs 5-fold validation, refits both models on all 24,000 series and writes the submission.
