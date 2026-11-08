# try4: full 0.879 recipe (training in progress)

The code is at the repo root. It is the same recipe that scored 0.88077 / 0.88078 on Kaggle (0.879 local AUC), ported to run from the raw competition files:

- `pipeline/full_cpu_stages.json`: all 473 steps, with their arguments, inputs and outputs.
- `pipeline/legacy_cpu/`: every feature, model and combiner script (172 files), plus `pipeline/features_v4.py` for the 977 base features.
- `train_087910_cpu.py`: CPU runner, one step at a time.
- `pipeline/train_gpu.py` with `pipeline/gpu_launch.py`: the same steps, run in parallel, with XGBoost and the neural nets on CUDA. The two runners write the same step records, so either can continue the other.
- `pipeline/inference/`, `pipeline/predict_new.py` and `inference.ipynb`: load the saved models and score new series.
- `../FEATURES.md`: what every feature family is.

```console
python -m pipeline.train_gpu --data data --work cpu_run --workers 6      #or: python train_087910_cpu.py --data data --work cpu_run
python -m pipeline.predict_new --models cpu_run --series data/test_series.parquet --out predict_run --csv submission.csv
```

**Status:** the training run is in progress. Its models and score will be added here when it finishes.
