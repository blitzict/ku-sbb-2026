# Blitzict: structural break detector

Team alias **Blitzict**. The full model, **[try3](try3/README.md)**, has our highest public score (**58102347**, 0.88078), but its inference takes hours and does not fit the limit of 30 minutes on 2 CPU cores with 8 GB RAM. The selected solution is therefore **[try4](try4/README.md)**, a slimmed version of try3 with the full code from raw series to features, training and inference (~22 min on 2 CPU cores, 2.6 GB RAM).

| | Model | Kaggle public | Local AUC |
|---|---|---:|---:|
| [try1](try1/README.md) | XGBoost + LightGBM on the 977 base features | 0.83313 | 0.83835 |
| [try2](try2/README.md) | LightGBM on 3,625 features | 0.87539 | 0.86987 |
| [try3](try3/README.md) (highest score) | Full 0.879 recipe (51 detectors, weight heads, local-rank correction) + MLP correction; inference too slow for the 30 min / 2 core limit | **0.88078** | 0.87910 |
| [try4](try4/README.md) ⭐ **selected** | Slimmed try3: top-100 features + 36 noise-law columns, 2 LightGBM + 1 XGBoost; scores 4,000 raw series in ~22 min on 2 CPU cores | 0.87583 | 0.87235 |

Each folder has its training code, saved model, an inference script and an `inference.ipynb` notebook that scores raw series.

Run try4 from the raw competition files:

```console
python try4/build_features.py --data data --work try4_run --workers 4
python try4/train.py --work try4_run --out try4/model
python try4/predict.py --series data/test_series.parquet --ref try4_run --out predict_try4 --csv submission.csv --workers 2
```

Write try3's 0.88078 submission CSV: `python try3/designated_088078/predict.py --data data --out submission.csv`.

`data/` must hold the competition files (`train_series.parquet`, `test_series.parquet`, `train_labels.csv`, `sample_submission.csv`); they are not in this repo. Setup: `pip install -r requirements-cpu-full.txt` (Python 3.12+).

## Extra resources

- [extra/qualification/report.pdf](extra/qualification/report.pdf): the 3-page report (pipeline, feature families, results).
- [extra/FEATURES.md](extra/FEATURES.md): every feature family.
- `pipeline/`: shared code for all tries (feature scripts, the 473-step recipe, GPU/CPU runners, inference).
- `CPU_087910.ipynb` / `train_087910_cpu.py`: training notebook and script for the full recipe; `inference.ipynb`: scoring notebook.
- `extra/PACKAGE_MANIFEST.json`, `extra/SOURCE_STATUS.json`: file hashes and source provenance.

## AI use

Claude Opus 5.5 was used to implement my ideas, optimize the training for my dual-GPU setup, and later port the extracted features to a CPU-only run.
