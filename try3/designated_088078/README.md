# Designated submission: Kaggle 58102347 (0.88078 public)

This is the mixed ensemble: the full 0.879 detector ensemble plus a correction from a 5-fold MLP trained with synthetic data. Its local out-of-fold AUC is 0.87910, and it scored **0.88078** on the public leaderboard.

```console
python try3/designated_088078/predict.py --data data --out submission.csv
```

This command writes the designated CSV and checks it against `model/submission.csv`. The SHA-256 of the output is `1f1a333b00113ca02f501cd6f90eef7c6afa13263eaf58284a638808a003237e`, identical to the file submitted to Kaggle. It runs in seconds on CPU.

How the prediction is made:

- The final layer is `p = sigmoid(parent_logit + features @ weights)`, with 5 weights. Its inputs are stored in `model/mixed_replay.npz`.
- The parent logit is the output of the 0.879 detector ensemble. Its code is the full recipe in `pipeline/`, documented in `docs/FEATURES.md` and `docs/qualification/report.pdf`.
- **Limitation:** many of the original upstream models were saved only as predictions, so `predict.py` replays the saved detector outputs. Retraining the recipe from raw data (`python -m pipeline.train_gpu ...`, see `try3/`) saves every model, but fresh fits give slightly different predictions, so it will not match this CSV within 1e-6.
- The synthetic-data MLP part can be recomputed from the raw test series:

  ```console
  python try3/designated_088078/member_cpu.py --data data --workers 2
  ```

  Checked: on the 4,000 test series this took 87 seconds with 2 workers, and its logits match the saved GPU member within 4e-6. It rebuilds the 361 inputs and applies the saved fold models: `model/fold0-4.pt`, `fold*_transform.joblib` and `fold*.json`. The feature code is in `cpu_0879/raw_features.py` and `experiments/weight_space/`; these are the exact original versions used for this submission.
