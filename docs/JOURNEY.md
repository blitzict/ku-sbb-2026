# The journey: from 0.833 to 1st place

KU Applied AI Challenge 2026, KU-SBB track. Team Blitzict. Work began on 31 October 2026.

This is the full account of the competition: what was tried, in order, what each idea was worth, what failed, and how the final model was chosen. Scores are ROC-AUC. "Held-out" means pooled 5-fold out-of-fold AUC on the 24,000 training series; "public" means the Kaggle public leaderboard.

## Contents

1. [The task](#1-the-task)
2. [Timeline](#2-timeline)
3. [What each idea was worth](#3-what-each-idea-was-worth)
4. [The ideas that worked](#4-the-ideas-that-worked)
5. [What did not work](#5-what-did-not-work)
6. [What the data turned out to be](#6-what-the-data-turned-out-to-be)
7. [Validation](#7-validation)
8. [From try3 to try4: fitting the inference limit](#8-from-try3-to-try4-fitting-the-inference-limit)
9. [Limits](#9-limits)
10. [Hardware and tools](#10-hardware-and-tools)
11. [Lessons](#11-lessons)

## 1. The task

- **Input:** a time series of 256 to 1,024 points (median 637) with a marked point 30–70% of the way in. `period` is 0 before the point and 1 after it.
- **Output:** one `break_probability` per series: did the process generating the data really change at the marked point?
- **Data:** 24,000 labelled training series (exactly 12,000 breaks), 4,000 test series. All synthetic.
- **Metric:** ROC-AUC, so only the order of the predictions matters.
- **Rules:** no outside data and no pretrained models. Inference must rerun on 2 CPU cores, 8 GB RAM, within 30 minutes, matching the submitted file within 0.000001.

Why it is hard: most breaks change how the noise behaves (scale, shape, memory, spikes, rhythm), not the average. 39% of breaks move the average by under 0.1 standard deviations. Comparing the mean or the spread before and after gives AUC 0.62.

## 2. Timeline

| Date | What happened | Held-out | Public |
|---|---|---:|---:|
| 31 Oct | First feature bank. Found that the signal classic features miss is the *shape* of the innovation distribution after AR whitening (kurtosis, tails, Student-t degrees of freedom by likelihood). XGBoost 0.8225 → 0.8345 → 0.838. Chose a clean single-thread model over a 14-model stack (0.8401). This is **try1** | 0.8384 | 0.83313 |
| 1 Nov | Transform-grid blocks (moving-std and difference views with two-sample tests) +0.002; boundary-shift negatives +0.001; 198 difference-domain features +0.002. Pinned down the generator ([section 6](#6-what-the-data-turned-out-to-be)) | 0.8442 | |
| 2 Nov | Null-importance feature selection (~150 columns), oscillator whitening fix, contrast and GARCH blocks. The breakthrough of the day: time-reversed and sign-flipped copies of every series for training, with test-time averaging | 0.8501 | |
| 3 Nov | Crop views (trim 10–40% off the outer ends) with weighted test-time averaging. Learning curve measured: +0.005 AUC per doubling of training rows, so more *real* rows became the main lever. Length-matched real segments added as negatives | 0.8510 | |
| 4 Nov | Weight edits: pairwise refit of MLP output layers, learned weights per view (0.85290 → 0.85339). Then the mechanism work: periodic and phase detectors (→ 0.867), trend and pulse handling, noise-law evidence, and the mechanism specialist with fold-fitted noise priors (0.8748 alone). Local-rank correction on top | 0.87911 | **0.88077** |
| 7 Nov | Synthetic-data track: a recursive simulator lifts single MLPs (0.8538 → 0.8562) but adds about nothing to the ensemble. The mixed ensemble with a simulator-trained MLP scores 0.88078, a difference of 0.00001. Qualification cutoff at 23:59 | 0.87910 | **0.88078** |
| 8 Nov | Qualification package: report, private repo, `qualification` tag. The 0.88 model could not rerun from raw data (many upstream models had been saved as predictions only), so a full CPU rebuild of the 473-step recipe started. **try2**, one LightGBM on 3,625 features, served as the reproducible model | 0.86987 | 0.87539 |
| 9–11 Nov | Full retrain from raw data finished at 0.87886 (original 0.87910; test predictions Spearman 0.9994 with the 0.88078 file). This is **try3**. Its inference takes hours, so **try4** was cut from it to meet the 30-minute limit | 0.87235 | 0.87583 |
| After | Finalist evaluation (presentation and Q&A) with try4 as the selected model | | |
| | **Result: 1st place** | | |

## 3. What each idea was worth

Local score as each part was added to the full model:

| Step | Held-out AUC |
|---|---:|
| Eyeballing: change in mean or spread | 0.62 |
| Trees on before/after statistics (977 features) | 0.838–0.840 |
| + best extra measurement groups (grids, difference domain, likelihood ratios) | 0.845 |
| + 7 extra views of each series (reversed, flipped, both, 4 crops) | 0.850 |
| + periodic and phase detectors | 0.859–0.867 |
| + mechanism specialist with noise-law priors | 0.874–0.875 |
| + blending, weight heads and local-rank correction | 0.879 |
| + MLP trained partly on simulated series | 0.879 (public 0.88077 → 0.88078) |

The largest single jump, about +0.015, came from the specialist. One idea beat a week of tuning.

## 4. The ideas that worked

**Innovation shape (31 Oct).** Fit an AR model, look at the residuals, and compare their distribution before and after: kurtosis, tails, L-moments, Student-t degrees of freedom fitted by likelihood, likelihood-ratio statistics. Every one of these has a univariate AUC near 0.54; the signal is the *size* of the shape change in either direction, which trees pick up.

**More views of the same series (2–3 Nov).** A break is still a break if the series is played backwards or flipped in sign, and still there if the outer ends are trimmed. Training on those copies and averaging over them at test time (original and flipped weight 2, reversed 1, crops 0.5) was worth +0.006. Views of one source series always stay in the same fold.

**Real rows, not invented ones (3–4 Nov).** A single before or after segment of any series contains no break by construction, so it can serve as a negative with a pseudo-boundary. These were paired with matched crops of real breaks around the true boundary.

**Mechanism features (4 Nov).** Instead of describing how a series looks, test how it was produced. For each side of the point, Bayes factors and likelihoods compare hypotheses: one noise scale or two, one noise shape or two (39 candidate shapes), one hidden drifting process or a restart at the point, changed autoregressive memory, changed phase, frequency or amplitude of an oscillation, a short pulse versus a permanent shift. The best single column, the noise-scale test `native_law_state__native_free_scale`, reaches 0.82 alone.

**Noise-law priors.** For each kind of noise, how often did noise like this mean a break? Fitted on each fold's training labels only and given to the model as 36 columns.

**The specialist (MECH3C12V).** One XGBoost model on 11 mechanism tables, 5 jump-law tables, the volatility-law table, a selected statistical block and the 36 prior columns, trained with two crop views. Built up as: base 0.8713 → + jump laws 0.8716 → + crop c1 0.8730 → + crop c2 0.8733 → + volatility law 0.8743 → 3-seed bag 0.8748. That is 99.5% of the 51-model ensemble.

**Weight heads and local-rank correction.** Small regularised corrections to the blended score: quantile-aligned detector outputs, family components, paired detector differences, a local positive-versus-negative ranking loss, with the fitted update halved. Each step was kept only if it improved the score on data it had not learned from.

## 5. What did not work

Kept here so nobody repeats it. Gains quoted are on the held-out AUC of the full model.

| Tried | Result |
|---|---|
| Raw-scale features, ordinal patterns, BDS tests | No gain |
| Donor positives, half-series negatives, score flooring | No gain |
| Ratio and difference feature blocks, `colsample` 1.0, generalised cross-entropy loss | No gain |
| Importance-weighted training for train/test drift | No gain, although test-weighted CV matched the leaderboard |
| A specialist model for the random-walk pocket (AUC 0.70 there) | Worse than the general model |
| Hard thresholds or flagging rules on top of the scores | Always lowered AUC |
| Raw-series networks (3k-input MLP, CNN, transfer CNN) | 0.794 alone, +0.0004 in a stack |
| Exact likelihood-ratio block | 0.757 as a single statistic, +0.000 in the model: the feature bank already contained it |
| Exact generalised-normal and bimodal shape tests, Bayes-factor member, process-aware stacker | Within ±0.0005 |
| Monotone constraints, depth regimes, `max_bin` 1024, distillation, heavy regularisation | No gain |
| Boundary-transient and multi-window features | No gain: the interventions are clean steps |
| Segment swapping between series | Junction artefact the model learned instead |
| Per-family and cluster-level feature selection, manual per-cell stack weights | All lost |
| EMA weight averaging, XGBoost leaf refresh | No gain |
| A simulator matched to the pinned generator | Still separable from real data (adversarial AUC 0.79); synthetic rows help single models, not the ensemble |
| Looking harder at the missed breaks | Every raw view scores 0.50–0.52 on missed-versus-clean. They are small draws, not a missing feature |

About 620 scored runs are listed in [`research/results/experiment_ledger.csv`](../research/results/experiment_ledger.csv).

## 6. What the data turned out to be

Worked out from the training series on 1 November and used to decide which tests to build:

- **Processes:** about 50% white noise, 25% AR(1) with the coefficient spread over (−0.95, 0.8), 10% damped AR(2) oscillators, 7.5% random walks.
- **Innovations:** 47% Gaussian, 13% heavy-tailed, 7% light-tailed, 7.5% skewed, 3% GARCH, plus a bimodal-mixture family in about 23% of series.
- **Breaks:** one type per series. Level 33%, scale 33%, shape 15%, AR 8%. Magnitudes go all the way down to zero.

That last point sets the ceiling. About 12% of breaks are draws so small that nothing separates them from non-breaks: 1,464 of the 12,000 breaks were scored below the median by all ten models of the time. On the part of the data that can be learned, the models sit at 0.93, level with the optimal tests for the white-noise cell.

Train and test differ slightly (adversarial AUC 0.57; the test set has more wandering before-segments), which explains why early leaderboard scores ran about 0.005 below local ones.

## 7. Validation

- Five stratified folds of whole series, seed 7: 19,200 series to fit, 4,800 held out, every series held out exactly once.
- All views of a source series (reversed, flipped, cropped, segments) stay in that series' fold.
- Feature ranking, noise-law priors and quantile transforms are fitted on each fold's training rows only. Fitting the priors on all labels inflates the score from 0.873 to 0.8774; tables fitted that way are excluded from try4.
- try4 per fold: 0.868 to 0.882; pooled 0.8723; public 0.8758.
- A 4,000-series leaderboard score has a standard error of about 0.005, so models were picked by local validation, not by the public board.

Known limits of the research validation, as stated in the report: some upstream predictions in try3 are historical out-of-fold archives rather than a complete nested retrain; one fixed 170-column selection and some early tuned settings saw labels outside later folds; the simulator's break-size settings came from research on all training labels.

## 8. From try3 to try4: fitting the inference limit

try3 scores highest but needs hours to score new series. The rules allow 30 minutes on 2 cores.

| | try3 | try4 |
|---|---:|---:|
| Models | 51 + correction heads | 3 per fold |
| Columns | about 5,100 | 136 per fold |
| Recipe steps | 473 | 77 |
| Inference, 4,000 series, 2 cores | hours | 21.8 min |
| Peak memory | | 2.6 GB |
| Held-out AUC | 0.87910 | 0.87235 |
| Public | 0.88078 | 0.87583 |

How the cut was made:

1. **Models.** Keep what the specialist showed mattered: mechanism tables plus priors. Two LightGBM and one XGBoost per fold, rank-blended.
2. **Columns.** In each fold, a LightGBM ranks 3,589 candidate columns by gain on that fold's training rows; keep the top 100 and add the 36 prior columns. `k = 100` came from a sweep over 25 to 1,600. 39 columns are picked in all five folds.
3. **Compute.** The `slim_*.py` scripts run the same feature code but compute only the columns the 15 saved models read. Values are identical to the full steps. Dropping the four most expensive tables saved another 7 minutes for −0.0004.

The limit cost about 0.005 AUC. try4 keeps 99.4% of try3's score.

## 9. Limits

- **Quiet breaks slip through.** 43% of breaks that move the average or spread by under 0.15 SD land in the bottom half of the scores.
- **Short series are harder.** 0.82 on the shortest third (256–512 points) against 0.91 on the longest.
- **A ranking, not a probability.** The final blend averages ranks, so the output orders series but is not calibrated, and all test series must be scored in one run.
- **try3 is not byte-reproducible from raw data.** Its designated CSV is rewritten exactly from saved detector outputs; a fresh retrain gives 0.87886 with near-identical ranking.
- **Built for this data.** Designed around these synthetic series and not tested elsewhere.

## 10. Hardware and tools

- One PC: 24 cores, 64 GB RAM, an RTX 5070 (XGBoost and CatBoost on CUDA) and an Intel Arc Pro B70 (PyTorch `xpu` for the MLP and sequence models). The final models train and run on CPU.
- Python 3.12, LightGBM 4.7.0, XGBoost 3.4.1, scikit-learn 1.9.1, NumPy 2.5.4, pandas 3.1.0, SciPy 1.18.1, PyTorch, Optuna for development-time tuning.
- AI use: Claude Opus 5.5 (Anthropic) turned my ideas into code, sped up training on the two GPUs, ported the features to a CPU-only run, and helped build the slides and the scripts behind their numbers. A second coding-agent session (Codex) built the physical mechanism tables under `experiments/weight_space`; every candidate from it was replayed and verified before use.

## 11. Lessons

1. **Describe how the data was made, not how it looks.** The best mechanism column alone scores 0.82; eyeballing scores 0.62.
2. **One good idea beats a week of tuning.** try2 passed a week of polishing try1 in a day; the specialist was worth more than everything stacked on top of it.
3. **Know the ceiling.** Once the generator was pinned down and the missed breaks shown to be unrecoverable, effort moved to things that could still pay.
4. **Real rows beat invented rows.** Views, crops and segments of real series helped; simulated series did not survive ensembling.
5. **Save the models, not only the predictions.** The 0.88 ensemble could not be rerun from raw data at qualification time, and rebuilding it cost two days.
6. **Trust local validation over the public board.** With ±0.005 of noise on 4,000 series, chasing the leaderboard is chasing noise.
7. **Cut on purpose.** 51 models became 3 and hours became 22 minutes, for 0.005.
