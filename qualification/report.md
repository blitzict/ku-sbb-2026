# Blitzict | KU Applied AI Challenge

Qualification report draft - 7 November 2026

Team alias is provisional. The designated Kaggle submission ID is still needed. This draft documents the historical 0.88078 candidate. Some upstream model checkpoints were never saved; the current files cannot reproduce that entire model from raw test data. A fresh CPU rebuild is in progress and has no validated score yet.

## 1. Method and model inputs

The task is binary structural-break detection with a supplied boundary. Each example is a complete series, not a single time point. The competition supplies 24,000 labelled series (12,000 breaks and 12,000 controls) and 4,000 unlabelled test series. Observations are sorted by time and split at the period-0 to period-1 transition. No test labels were used.

The original v4 baseline computes 977 features: level and spread gaps, empirical distribution differences, autocorrelations, autoregressive fit and prediction errors, trend stability, boundary evidence, innovation and volatility shape, tail likelihoods, and joint distributions of successive observations. Its main features use the before-period mean and standard deviation for normalization. XGBoost (487 rounds, depth 10, learning rate 0.0101) and LightGBM (499 rounds, 76 leaves, learning rate 0.0128) are combined by averaging test-score percentiles. All exact settings are in models/v4_083313/hyperparameters.json.

The stronger ensemble adds mechanism-specific detectors. These compare shared versus changed oscillation phase, frequency and amplitude; remove coherent periodic components; model autoregressive memory and random walks; distinguish trends and short pulses from permanent shifts; and compare volatility clustering, shocks, two-peaked noise, rank dependence and temporal ordering. Trees, multilayer perceptrons and sequence networks provide complementary predictions. Reversal, sign-flip and cropped views supply additional evidence about the same source series.

Weight-space programming here means editing or refitting existing model and ensemble weights. It includes neural output-layer edits, learned weights on augmented views, and small regularized corrections to an established ensemble. The final original correction has 152 coefficients. Training-fold quantile transforms and conditional centering align detector scales; family components and ridge projections reduce redundant votes; paired detector differences and regime interactions preserve complementary information. Its objective combines local positive-negative ranking loss, binary cross-entropy and L2 regularization, optimized with L-BFGS-B. The fitted update is multiplied by 0.5. The implementation is experiments/weight_space/white_shared_residual.py and white_local_rank_weights.py.

The mixed candidate adds a five-model MLP on 361 label-free physical measurements. Each fold uses a 257-quantile normal transform fitted on real training rows. Architecture: 361 inputs, a hidden layer, a half-width hidden layer, GELU activations and dropout, then one logit. AdamW uses batches of 1,024 and a cosine schedule. Synthetic examples receive weight 0.05; epochs are adjusted to approximately match the real-only number of optimizer steps. A five-coefficient, confidence- and regime-dependent residual correction combines its mean logit with the original ensemble.

The project implements the feature functions, simulator, view construction and ensemble corrections. Standard tree learners, neural layers, preprocessing and optimizers come from the libraries listed on page 3. Historical scripts and parameters are retained while their full CPU rebuild is recovered.

<!-- PAGE BREAK -->

# Validation, results and limitations

## 2. Validation protocol

The principal ensemble checks use five stratified folds with seed 7: 19,200 original series for fitting and 4,800 held out per fold. Pooled validation covers all 24,000 series once. Views of a source stay together within each member's folds. Base members and v4 use seed 0; later ensemble folds differ. Additional seeds reuse the same labelled dataset and are not independent test sets.

For the new MLP, quantile normalization is fitted on real training rows only. Synthetic donors must belong to that fit split; shape simulations are eligible only when every candidate donor is also in the fit split. Inner cross-fitted predictions supply training inputs to the new ensemble correction, excluding outer-validation labels. Final inference averages five saved fold checkpoints. The retained architecture and stopping settings were selected through 60 Optuna trial attempts (53 complete and seven pruned); qualification inference does not rerun the search.

These checks have limits. Upstream predictions are historical out-of-fold archives, not a complete nested retrain at every layer. The fixed 170-column selection in selected_ns3.csv and earlier tuned parameters used labels outside later validation folds. Simulator magnitude priors also originated in research using all training labels. Step matching was chosen after inspecting initial seed-7 results, and seed-29 comparisons share tuning decisions. These are research validation scores subject to selection optimism, not an untouched final holdout estimate.

## 3. Results and error analysis

| Model | Local pooled ROC-AUC | Public leaderboard |
| --- | --- | --- |
| Constant prediction baseline (analytical) | 0.50000000 | Not submitted |
| Original v4 trees | 0.83835372 | 0.83313 |
| Compact raw CPU detector, separate baseline | 0.86523822 | Not recorded |
| Original preserved ensemble | 0.87911421 | 0.88077 |
| Mixed synthetic MLP ensemble | 0.87909706 | 0.88078 |

Leaderboard values were reported by the user; the submission IDs have not yet been verified. The mixed model is lower locally by 0.00001715 and higher on the reported public leaderboard by 0.00001. That small difference is not evidence of a reliable generalization advantage. The compact baseline is a distinct model, not a reproduction of either 0.88 submission.

A major earlier improvement came from recognizing coherent oscillations: phase or amplitude can change while mean and variance remain stable. Adding these detectors and scoring their residuals took one local ensemble from 0.853385 to 0.867148. Later gains combined trend/pulse handling, noise-law evidence, source-aware crops and constrained residual corrections. Those milestones combine several changes and are not all isolated ablations.

An investigated cohort contained 821 historically missed positive series with white-noise-like appearance. The original final ensemble recovered 88 of them at a 20% false-positive rate, compared with 42 for the earlier 0.870943 model. Its AUC on those positives against all 12,000 negatives was 0.498719; on the remaining 23,179 series it was 0.907051. This cohort was identified during development, so it is an error-analysis view, not an independent benchmark. Weakly expressed changes and wandering or volatile controls remain difficult. More synthetic data did not consistently improve the full ensemble, and a classifier could distinguish real from simulated feature distributions.

<!-- PAGE BREAK -->

# Reproduction and sources

## 4. CPU execution and release status

Use Python 3.12.1. From the project root, install the recorded core dependencies with `python -m pip install -r requirements-cpu.txt`. Neural checkpoint inference additionally needs the tested PyTorch and joblib versions recorded in qualification/environment.json. Install dependencies before disconnecting the network. No command below submits to Kaggle or publishes a repository.

In the original local workspace, `python run_cpu.py predict --model mixed --out-dir outputs/mixed` writes the preserved mixed CSV using models/mixed_088078/mixed_replay.npz. This exact archived-input replay does not reconstruct the upstream detectors. Several crop and mechanism trees saved predictions but no models. It therefore does not meet the raw-input rerun requirement. Its SHA-256 is 1f1a333b00113ca02f501cd6f90eef7c6afa13263eaf58284a638808a003237e. Test-derived replay inputs are not approved repository contents.

The new neural component has a raw-test CPU path: `python -m pipeline.mixed_member_cpu --data data --workers 2`. It rebuilds 361 inputs and loads fold0.pt through fold4.pt and saved transforms in models/mixed_088078. A two-logical-processor, 6 GiB-capped run took 98.906 seconds, peaking at 1.48 GB job committed memory. The earlier CPU comparison differed from the archived Arc probabilities by at most 0.000000179. This verifies only that component.

V4 raw inference: `python run_cpu.py predict --model v4 --raw --workers 2 --out-dir outputs/v4_raw`. Checkpoints: models/v4_083313/xgboost.ubj and lightgbm.txt. Under enforced two-processor affinity and a 6 GiB cap it took 666.359 seconds, peaking at 1.59 GB job committed memory; all 4,000 probabilities matched exactly. Python socket access was blocked. Full v4 training: `python run_cpu.py train --model v4 --workers 2 --validate`. Fresh 0.88 training remains unfinished and cannot be claimed to reproduce the old submitted CSV.

The complete designated model must process supplied raw test data offline on two CPU cores, within 8 GB RAM and 30 minutes, matching its submitted probabilities within absolute 0.000001. The historical 0.88 model does not currently satisfy this requirement. A newly trained CPU model would require its own validation and submission. No competition data, test labels or organizer files may be uploaded. Submission designation, human review, reviewer access and the qualification tag remain unresolved.

## 5. Sources, data and fixed settings

Numerical and learning libraries: NumPy, SciPy, pandas, PyArrow, scikit-learn, threadpoolctl, XGBoost, LightGBM, PyTorch and joblib; Optuna was used for development-time tuning. Installed versions are recorded separately. Project method evidence is in cpu_0879/method.html, the preserved release validation/export JSON files, auc_lab_20261107/claude/RESULTS.md, and the model training scripts. Public research references already collected by the project remain in docs/RESEARCH_IDEAS.md and docs/RESEARCH_TREES.md; a complete reused-code provenance review is still pending.

The v3 simulator generated 25,977 examples from training-negative donors with recorded lineage. Earlier augmentation included boundary-shift negatives and 85,376 length-matched real-segment rows. A separate test-derived augmentation experiment exists; its use in the complete historical chain is not fully audited. No test labels or outside pretrained model were used. Historical training used RTX 5070 and Intel Arc GPUs; the reconstruction uses CPU. MLP seeds are 112 plus fold, transform seed 71, and pair-sampler seed 1729. Exact MLP settings are in models/mixed_088078/fold0.json through fold4.json.

## AI-use and human verification - complete last

Reserved for the team's consolidated account across sessions, as requested. Record the AI tools used, the work they assisted with, the sources checked, and what the team independently reviewed or ran. This section is intentionally unfinished. Automated smoke tests and numerical comparisons described above do not constitute a claim that a human teammate has checked them.
