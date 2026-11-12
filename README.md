# 🥇 1st place: KU Applied AI Challenge 2026 (KU-SBB)

**Team Blitzict.** Structural-break detection on synthetic time series: each series comes with a marked point, and the model gives the probability that the process generating the data really changed there. Scored by ROC-AUC.

This repository is the complete winning solution: code from raw series to submission, the saved models, the report, the finalist slides, and the research archive behind them.

| | |
|---|---|
| **Result** | 1st place |
| **Selected model** | [try4](try4/README.md): 0.87583 public, 0.8724 held-out, scores 4,000 series in ~22 min on 2 CPU cores with 2.6 GB RAM |
| **Highest score** | [try3](try3/README.md): 0.88078 public, 0.87910 held-out (Kaggle submission 58102347) |
| **Data** | 24,000 labelled series (12,000 breaks), 4,000 test series, 256 to 1,024 points each |
| **Rules** | No outside data, no pretrained models; inference within 30 min on 2 CPU cores and 8 GB RAM |

## The four models

| | Model | Kaggle public | Held-out AUC |
|---|---|---:|---:|
| [try1](try1/README.md) | XGBoost + LightGBM on the 977 base features | 0.83313 | 0.83835 |
| [try2](try2/README.md) | One LightGBM on 3,625 statistical + mechanism features | 0.87539 | 0.86987 |
| [try3](try3/README.md) | Full recipe: 51 detectors on 8 views of each series, weight heads, local-rank correction, MLP correction. Highest score, but inference takes hours | **0.88078** | 0.87910 |
| [try4](try4/README.md) ⭐ **selected** | try3 cut down to fit the inference limit: top-100 features + 36 noise-law columns per fold, 2 LightGBM + 1 XGBoost, rank blend | 0.87583 | 0.87235 |

Each folder has its training code, saved model, an inference script and an `inference.ipynb` notebook that scores raw series.

**The idea that won.** Simple before/after comparisons barely beat a coin flip (AUC 0.62), because most breaks change how the noise behaves rather than the average. The gain came from *mechanism features*: Bayes-factor and likelihood tests of how the series was produced on each side of the point (noise scale, noise shape, restarts, memory, rhythm), plus a noise-law prior fitted inside each fold. One specialist model built on those reaches 0.8748 alone, 99.5% of the 51-model ensemble. The full story is in [docs/JOURNEY.md](docs/JOURNEY.md).

## Quick start

`data/` must hold the four competition files (`train_series.parquet`, `test_series.parquet`, `train_labels.csv`, `sample_submission.csv`). They are not in this repo.

```console
pip install -r requirements-cpu-full.txt     # Python 3.12+
```

Run the selected model, try4, from the raw competition files:

```console
python try4/build_features.py --data data --work try4_run --workers 4
python try4/train.py --work try4_run --out try4/model
python try4/predict.py --series data/test_series.parquet --ref try4_run --out predict_try4 --csv submission.csv --workers 2
```

Write try3's 0.88078 submission CSV (byte-identical to the submitted file):

```console
python try3/designated_088078/predict.py --data data --out submission.csv
```

## Repository layout

| Path | What it holds |
|---|---|
| `try1/` … `try4/` | The four models: code, saved models, inference notebooks, READMEs |
| `pipeline/` | Shared code for all tries: feature scripts, the 473-step recipe (`full_cpu_stages.json`), CPU/GPU runners, inference |
| `train_087910_cpu.py`, `CPU_087910.ipynb` | Script and notebook that train the full recipe from raw data |
| `inference.ipynb` | Scores new series with a retrained full recipe |
| `tune_out/` | Selected feature lists and fixed settings the recipe reads |
| [`docs/`](docs/README.md) | Report, slides, explainer, feature reference, the full write-up, research notes |
| [`research/`](research/README.md) | Archive: every experiment script, 620 scored runs, weight-edit candidates, queue scripts |

## Documentation

- [docs/JOURNEY.md](docs/JOURNEY.md): the whole competition from first baseline to final model. What worked, what did not, and why.
- [docs/qualification/report.pdf](docs/qualification/report.pdf): the 3-page qualification report.
- [docs/slides/slides.pdf](docs/slides/slides.pdf): the 33-slide finalist presentation, also as [PowerPoint](docs/slides/slides.pptx).
- [docs/FEATURES.md](docs/FEATURES.md): every feature family.
- [docs/explainer/](docs/explainer/README.md): glossary, Q&A prep sheet and a runnable feature demo.
- [Release `v1.0-first-place`](https://github.com/blitzict/ku-sbb-2026/releases/tag/v1.0-first-place): the two technical explainer videos.

Tags: `qualification` is the commit archived by the organisers; `v1.0-first-place` is the final state.

## AI use

Claude Opus 5.5 was used to implement my ideas, optimize the training for my dual-GPU setup, and later port the extracted features to a CPU-only run.
