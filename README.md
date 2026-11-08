# Blitzict: structural break detector

Team alias **Blitzict**. Designated Kaggle submission: **58102347** (0.88078 public), in [try4](try4/README.md).

| | Model | Kaggle public | Local AUC |
|---|---|---:|---:|
| [try1](try1/README.md) | XGBoost + LightGBM on the 977 base features | 0.83313 | 0.83835 |
| [try2](try2/README.md) | LightGBM on 3,625 features | 0.87539 | 0.86987 |
| [try3](try3/README.md) | Compact: top-100 features + 36 noise-law columns, 2 LightGBM + 1 XGBoost | 0.87659 | 0.87271 |
| [try4](try4/README.md) ⭐ **best** | Full 0.879 recipe (51 detectors, weight heads, local-rank correction) + MLP correction | **0.88078** | 0.87910 |

Each folder has its training code, saved model, and a script or notebook that scores raw series.

Write the designated submission CSV:

```console
python try4/designated_088078/predict.py --data data --out submission.csv
```

`data/` must hold the competition files (`train_series.parquet`, `test_series.parquet`, `train_labels.csv`, `sample_submission.csv`); they are not in this repo. Setup: `pip install -r requirements-cpu-full.txt` (Python 3.12+).

## Extra resources

- [extra/qualification/report.pdf](extra/qualification/report.pdf): the 3-page report (pipeline, feature families, results).
- [extra/FEATURES.md](extra/FEATURES.md): every feature family.
- `pipeline/`: shared code for all tries (feature scripts, the 473-step recipe, GPU/CPU runners, inference).
- `CPU_087910.ipynb` / `train_087910_cpu.py`: training notebook and script for the full recipe; `inference.ipynb`: scoring notebook.
- `extra/PACKAGE_MANIFEST.json`, `extra/SOURCE_STATUS.json`: file hashes and source provenance.

## AI use

Claude Opus 5.5 was used to implement my ideas, optimize the training for my dual-GPU setup, and later port the extracted features to a CPU-only run.
