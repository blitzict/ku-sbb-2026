# Blitzict: structural break detector (0.879 recipe)

- Team alias: `Blitzict`
- Designated Kaggle submission ID: _to be filled in_
- Report: [qualification/report.pdf](qualification/report.pdf) (source: [qualification/report.md](qualification/report.md))
- Training notebook: [CPU_087910.ipynb](CPU_087910.ipynb); training script: `train_087910_cpu.py`
- Inference notebook: [inference.ipynb](inference.ipynb)
- Features used: [FEATURES.md](FEATURES.md)

Command that trains and writes the CSV:

```console
python train_087910_cpu.py --data data --work cpu_run --workers 2
```

Command that scores new series with the saved models and writes `id,break_probability`:

```console
python -m pipeline.predict_new --models cpu_run --series data/test_series.parquet --out predict_run --csv submission.csv
```

**Status:** This is the code for the 0.879 recipe. The full CPU training run has not been completed, so `cpu_run` and its saved models do not exist yet, and the prediction command has not been run end to end. Some late `white_*` head steps still have no saved-model adapter in `pipeline/inference`, so `predict_new` stops with `NotImplementedError` when it reaches one. The repository contains no competition data, labels, or organizer files.

---

# Raw-data CPU rebuild

**Status: the full CPU verification run is paused. The code is provided first, as requested. The target is local AUC 0.87910; it has not yet been measured for this rebuild.**

`CPU_087910.ipynb` is the Colab front end. `train_087910_cpu.py` is the same training entry point for a local machine. Both start with the four raw competition files and fit the recorded ensemble. No historical feature tables, predictions, or trained checkpoints are imported into training.

The notebook follows five steps: **load the same competition data, visualize examples, apply the existing formulas and transformations, train and combine the specified models, and save the fitted ensemble**. It runs the recorded recipe; it does not discover new features or search for a new model. New-series prediction is in `inference.ipynb` / `pipeline/predict_new.py` (see status above).

The settings cell exposes feature-list edits, shared tree settings, model hyperparameters, and member overrides. Leaving the dictionaries empty retains the historical defaults. The exact feature names ship in `tune_out/*.csv`; `pipeline/hyperparameters.template.json` documents the lists and available controls. Change the WORK folder when changing the recipe.

## Run locally

Use Python 3.12 or newer. Install the CPU build of PyTorch, then the remaining pinned packages:

```console
python -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-cpu-full.txt
python train_087910_cpu.py --data data --work cpu_run --workers 2
```

The data directory must contain `train_series.parquet`, `test_series.parquet`, `train_labels.csv`, and `sample_submission.csv`. There are 24,000 training series and 4,000 test series. Each prediction is the probability of a break in one series.

Use a new work folder for the first run. To resume an interrupted run, repeat the command with the same data, work folder, and settings. Completed stages verify input, source, and output hashes. Interrupted stages preserve checkpoints and reject changes to the inputs or code that produced them.

The notebook separates that same command into `--phase features` and `--phase models`. Together they execute the original ordered recipe exactly once; the default `--phase all` runs both. A few model-specific transformations stay with their model fits because the recorded dependencies require that order. To use your own settings, add `--config path/to/settings.json`.

The final outputs are:

- `cpu_run/cpu_validation.json`: the measured local AUC, all six output-fold scores, CSV hash, and raw-data hashes.
- `cpu_run/model.json`: description of the saved ensemble, with its recorded configuration and recipe. The fitted model consists of the whole `cpu_run` folder, including checkpoints and preprocessing; this JSON alone is not the model.
- `cpu_run/new weights/wn_local_rank_shared_residual_response_candidate/submission.csv`: the newly trained model's submission.
- `cpu_run/stages/` and `cpu_run/logs/`: resumable completion records and stage logs.

The score checker treats 0.87910 or better at five decimal places as reaching the requested target. It records the full precision AUC as well. A failed target check is reported as the actual result, never replaced with an archived score.

## What is trained

The complete executable recipe is in `pipeline/full_cpu_stages.json`. It contains script arguments, fixed hyperparameters, required inputs, declared outputs, and source dependencies. Supporting Python functions retain numerical settings that were originally constants in those functions.

| Part | Included information or method |
|---|---|
| Statistical features | The 977 original features, grid and difference features, distribution and likelihood comparisons, ranks, moments, quantiles, dependence, and spectral summaries. |
| Physical features | Trend and curvature, oscillation and phase, autoregression, latent state, volatility, jumps, heavy tails, mixtures, boundaries, memory, and conditional distribution changes. |
| Views | Recorded reversal, sign-flip, crop, and real-segment variants; boundary-shift negatives. Augmented rows follow their original source's training fold. |
| Detectors | Fixed XGBoost, LightGBM, CatBoost, MLP, sequence, and physical specialist recipes. Required checkpoints are saved by the new CPU fits. |
| Weight-space work | Sequence weight edits and neural-head adaptation, regularized ensemble readouts, regime-dependent corrections, and local pairwise ranking losses. |
| Final combination | The retained first layer is 75% of the three-source-vote readout plus 25% of the upstream branch. The final response correction uses half of the fitted local-ranking coefficient vector. |

The dense MLP block has 2,171 columns: 977 original features, 492 grid features, 504 second-grid features, and 198 difference features. Tree members use their recorded lists of 170, 191, or 206 selected columns, or the 2,247-column full list, depending on the member. The exact column names ship in the CSV files under `tune_out`; the stage manifest selects the appropriate list for each fit.

The local-ranking fit preserves its recorded settings: 32 sampled nearby negative comparisons per eligible positive, sampling seed 1729, score neighborhood 1.35, pair weighting width 0.62, logistic temperature 0.55, binary-loss multiplier 0.47, and quadratic coefficient penalty 84. L-BFGS-B uses at most 300 iterations with `ftol=1e-11`.

The earlier readout settings include view-stack penalty 100, expert-fusion regularization 700, mechanism penalty 40, trend penalty 60, extended-readout penalty 35, and hard-state-tree penalty 100. Other detector settings are recorded per member in the manifest and source. This notebook runs those settings; it does not launch Optuna or a new architecture search.

## How the score is checked

Base models use their recorded folds and inner stopping splits. Later readouts retain five outer folds for each of seeds 7, 29, 0, 123, 2024, and 11041, with the recorded inner-fold offsets where required. The headline local AUC is calculated directly from the seed-7 out-of-fold scores and the original labels.

This reproduces the project's historical local validation protocol. Feature lists and model choices were selected during earlier development using the same training collection, so the score is not an untouched holdout estimate. It is also separate from the historical public leaderboard scores of 0.88077 and 0.88078.

The reconstruction exports the preselected recipe even if an old experiment's promotion rule would reject its newly measured result. Reports distinguish fixed selection from measured improvement. Discarded comparison routes and diagnostic partition 61817 do not feed the final coefficients or the primary score and are omitted. The retained first-layer formula matches all six historical OOF arrays exactly and the historical full-fit test logits within 1.78e-15; that check is calculation evidence, not a fresh training result.

## CPU and memory

All training and prediction operators use CPU. The paused local reconstruction used at most six logical processors during training and a 6 GiB Windows process-tree memory cap. The notebook defaults to two feature workers. A complete peak-memory and duration result is still pending, and the notebook has not been executed in Colab yet.

The local Windows environment has PyTorch 2.11.0+cu128 installed, with all model operators explicitly placed on CPU. The notebook installs the CPU-only 2.11.0 wheel on Linux. These builds and their numerical libraries can differ in floating-point results; a measured Windows score is not a claim of an independently measured Colab score.

The ports select CPU explicitly, including modules that retain historical GPU-related filenames. Accelerator autocasting is disabled. The Student-t normalization constant is evaluated in float64 and then cast to the feature dtype; this avoids cancellation in the old float32 calculation at large degrees of freedom. These numerical changes and fresh stochastic fits can change predictions, so the CPU score must be measured rather than copied from the historical GPU result.

The feature extractor writes bounded batches to Parquet instead of retaining every feature dictionary. Its clean/grid/grid2/difference outputs matched the original functions bit for bit on 280 freshly generated real-segment rows. The full pipeline needs substantial disk space for raw views, features, fold checkpoints, and resumable records; these are generated outputs, not GitHub inputs.

Mixture-component features retain their original 45-row batches and overlap at most three batches on CPU. Their state and pulse outputs matched the original serial extractor bit for bit on fresh 80-row inputs. A separate 135-row check loading the full input arrays matched one-worker and three-worker results exactly; its process tree used 2.89 GiB at peak. The dictionary is shared read-only, and results retain their original row order.

Neural rank-Gaussian preprocessing sorts 64 columns at a time and transforms 4,096 rows at a time. It retains the full rank reference rather than a subsample. The bounded implementation matched the former implementation in numerical edge-case checks and all 76 outputs of the small neural training comparison. A synthetic full-size preprocessing check used 3.69 GiB; the complete training peak remains pending.

Allow at least 60 GiB of free work storage. The historical required outputs alone total about 25 GiB, before the newly saved full neural rank references. Intermediate files stay in WORK so completed stages can be verified and resumed. The repository contains only the required source and fixed settings. It does not contain raw competition data or historical prediction arrays.

## Preserved releases

The original v4 / 0.83313 release and both historical 0.88 releases remain preserved separately. This rebuild does not overwrite them. Matching a local AUC does not establish that a new CSV is identical to an old Kaggle submission or that separate qualification runtime requirements have been met.

## AI-use account

The user will supply the combined account across all sessions here.
