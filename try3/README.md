# try3: full 0.879 recipe (highest score: submission 58102347, 0.88078 public)

This is our best model, but scoring new series with it takes hours on a large machine, far beyond the inference limit (30 minutes on 2 CPU cores, 8 GB RAM). The selected solution is its slimmed version, [try4](../try4/README.md).

The complete detector ensemble: 51 base models (XGBoost / LightGBM / CatBoost / MLP / sequence CNN) on about 5,100 features and 8 data views, then weight heads and a local-rank correction. Local out-of-fold AUC **0.87910**. With the synthetic-data MLP correction it scored **0.88078** public (Kaggle submission **58102347**).

- [designated_088078/](designated_088078/README.md): rewrites the designated CSV exactly (SHA-256 identical to the submitted file) in seconds:

  ```console
  python try3/designated_088078/predict.py --data data --out submission.csv
  ```

- `inference.ipynb`: notebook for the designated submission. It writes the 0.88078 CSV, checks it against the submitted file, and can recompute the MLP part from the raw test series.

- Training code, from the raw competition files (at the repo root, shared with try2/try4):
  - `pipeline/full_cpu_stages.json`: all 473 steps with their arguments, inputs and outputs.
  - `pipeline/legacy_cpu/` and `pipeline/features_v4.py`: every feature, model and combiner script.
  - `train_087910_cpu.py` (CPU, one step at a time) or `pipeline/train_gpu.py` (parallel, XGBoost and nets on GPU). Both write the same step records.

  ```console
  python -m pipeline.train_gpu --data data --work cpu_run --workers 6
  ```

- Scoring new series with retrained models: `pipeline/predict_new.py` / `inference.ipynb` at the repo root.

**Status:** the retrain from raw data with `pipeline/train_gpu.py --parallel-late` finished with local AUC **0.87886** (original 0.87910); its test predictions have Spearman 0.9994 with the 0.88078 file. The saved-model inference path (`pipeline/predict_new.py` with `pipeline/inference/`) has not been verified end to end: a full run on the test series stopped at a feature-export step. Because try3 cannot meet the inference limit anyway, we did not pursue it further.
