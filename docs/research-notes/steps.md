# Exact steps for the saved-weight experiment

Date: 2026-11-04, Asia/Dubai. Project root: `C:\Users\Blitz\Downloads\Code\claude\ai`.

This records the work performed in this chat: inspection, experiments, unsuccessful attempts, code revisions, selection, simplification, packaging, and verification. Paths below are relative to that project root unless explicitly absolute. Some independent jobs overlapped; the numbered steps describe their logical dependency order. The appendices contain exact recorded numeric results, all 60 final neural edit decisions, the final coefficient values, the copied-file manifest, and complete snapshots of the six experiment scripts.

**Result: local meta-CV AUC increased from 0.85290475 to 0.8533852152777779. Distribution-weighted AUC increased from 0.8482631885976268 to 0.8486642821177613. The requested 0.87 AUC was not reached.** These are local validation results; no hidden Kaggle test score was obtained and no submission was sent to Kaggle.

## 1. Read the existing project before changing weights

1. Read `.agents/skills/kaggle-local/SKILL.md` and its `references/ensembling-strategies.md` for the local workflow.
2. Inventoried the workspace, `weights/`, `weights_experiments/`, `data/`, `tune_out/`, `submissions/`, `clean/`, `experiments/`, and `docs/`. Searched for applicable `AGENTS.md` files; none were found in the searched workspace/parent locations.
3. Read `README.md`, `clean`/data context, `data/README_competition.md`, `data/START_HERE.md`, and `data/manifest.json`.
4. Read the existing experiment history, especially `docs/EXPLORATION_NOTES.md`, `docs/LOOP_LEDGER.md`, and `docs/LOOP_BRIEF.md`.
5. Inspected existing implementation files: `experiments/v6/nnstack.py`, `stack2.py`, `blend.py`, `inspect_edit.py`, `leaf_refresh.py`, `nn_var.py`, `nn_seq.py`, `train_var.py`, and `wtta.py`.
6. Read saved model metadata, the latest submission recipes, and the queue scripts documenting how MLPHV, augmentation views and the sequence bag were built, including `tune_out/q/mlphv.sh`, `mlphvbag.sh`, `save_mlp.sh`, and matches in the other queue scripts.
7. Identified `submissions/sub_seqw5_stack.json` as the baseline: 51 listed members, eight positive coefficients in its full-data meta fit, reported AUC 0.85290475 and weighted AUC 0.84826320404797. The project README's older best-score references were not used as the actual baseline.

The repository already contained prior attempts at family-specific calibration, XGBoost leaf refresh, EMA weight averaging, ranking objectives for tree training, and many feature/model searches. I **read their historical results**; I did not rerun those earlier experiments or claim they were new work in this chat. In particular, I did not train new tree structures or edit the saved tree leaf values in this run.

## 2. Verify environment and inspect data/model integrity

Checked running Python processes and GPU availability. The NVIDIA status command was:

```powershell
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
```

Observed an NVIDIA GeForce RTX 5070 with 12,227 MiB total memory. The initial query reported 1,511 MiB used and 0% GPU utilization. Neural and ranking-gradient experiments ran on the Intel Arc Pro B70, using `.venv-xpu/Scripts/python`.

Environment probes reported:

| Item | Observed value |
|---|---|
| `.venv` PyTorch | `2.11.0+cu128`, CUDA available |
| `.venv` XGBoost | `3.4.1` |
| `.venv-xpu` PyTorch | `2.15.0+xpu`, XPU available |
| XPU device | `Intel(R) Arc(TM) Pro B70 Graphics` |
| Python checked while writing this record | `3.12.1`, 64-bit Windows |

The first sandboxed `.venv` import failed with `PermissionError` while reading `.venv/Lib/site-packages/typing_extensions.py`. The same local environment was then run outside that sandbox through the execution tool's approval review and succeeded. No packages were installed or upgraded. Experiment scripts request below-normal Windows process priority and limit CPU/BLAS threads to two; the small logistic optimizers and audit/file operations use the CPU, while neural inference and ranking gradients use the Intel GPU.

Inspected `weights/MLPHV_big/fold0_seed0.pt` with `torch.load(..., map_location='cpu', weights_only=True)`. It had 23 tensors, 4,336,644 stored tensor elements including buffers/counters, finite values, and linear weight shapes `(1024,2171)`, `(1024,1024)`, `(1024,1024)`, `(1,1024)`. This element count is not a count of trainable parameters alone.

Audited the original data against `data/manifest.json`; all four data/submission-file hashes matched. Confirmed:

- 24,000 labelled series, exactly 12,000 positives and 12,000 negatives.
- 4,000 unlabelled test series.
- Training-series parquet: 15,350,281 rows, 157,446,535 bytes.
- Test-series parquet: 2,584,536 rows, 26,439,678 bytes.
- Raw schema: `id`, `time`, `value`, `period`.
- `n_pre`: minimum 78, median 303, maximum 715.
- `n_post`: minimum 77, median 302, maximum 711.
- Unique canonical training/test identifiers; training labels matched the training index; training and test IDs did not overlap.
- All 51 baseline OOF/test prediction pairs had expected shapes, finite values, and probabilities in `[0,1]`.
- Constant predictions scored 0.5 AUC.

Inspected all 15 saved `.ubj` tree models: 17,297 trees and 1,243,279 finite leaf values. The initial `weights/` inventory contained 207 files totaling 3,421,914,294 bytes before this experiment added its own artifact directory. The CatBoost file was inventoried; the detailed leaf-value audit covered the 15 XGBoost `.ubj` files.

The data/tree audit was a one-off Python command that hashed the four input files, loaded each `.ubj` into `xgboost.Booster`, serialized its tree representation to JSON, collected `split_conditions` for nodes whose `left_children == -1`, asserted finite values, and wrote `tune_out/weight_edit_20261104/data_and_tree_audit.json`. It did not mutate the models. Source prediction hashes and ID/shape checks were written to `audit.json` in that directory. Exact audit records appear in the appendices.

Also consulted official documentation for the ROC-AUC scorer and PyTorch state-dictionary loading:

- https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html
- https://docs.pytorch.org/tutorials/beginner/saving_loading_models

These were documentation lookups, not sources of extra training data or pretrained models.

## 3. Write the experiment plan and preserve originals

Created `experiments/weight_edit_20261104/COMPETITION_PLAN.md` before implementing the experiments. The initial plan was to reproduce the baseline, test ensemble coefficient edits, then test neural output-layer edits with validation confined to the original folds. It later included learned augmentation-view weights and, following the user's instruction, creation of `new weights/` and fold-local top-4/6/8 member pruning.

New experiments were written under `experiments/weight_edit_20261104/`, outputs under `tune_out/weight_edit_20261104/`, and experimental checkpoints under `weights_experiments/weight_edit_20261104/`. Existing data, feature matrices, trained source checkpoints, and previous submission files were preserved. The final candidate was initially exported under `weights/auc_weight_edit_20261104/` and subsequently packaged in the requested `new weights/` folder.

## 4. Reproduce the original ensemble

Created `stack_edit.py`. It obtains the exact member list from `submissions/sub_seqw5_stack.json`, uses `clean/features_train.parquet` and `features_test.parquet` index order, aligns labels by ID, and loads `tune_out/oof_<member>.npy` and `test_<member>.npy`.

Each probability is clipped to `[1e-6,1-1e-6]` and converted to a logit. The original nonnegative logistic stack minimizes:

```text
0.1 * sum(log(1 + exp(z_i)) - y_i*z_i) + 0.5 * sum_j(coef_j^2)
z = X @ coef + intercept
coef_j >= 0; intercept unrestricted
```

Optimizer: SciPy L-BFGS-B, analytic gradient, zero initialization, maximum 5,000 iterations. Five-fold meta CV uses `StratifiedKFold(5, shuffle=True, random_state=7)`. A second meta split uses seed 29. Scores are pooled over each split's out-of-fold predictions.

Distribution weights are derived from the existing adversarial train/test probabilities:

```text
w = adv_ptest_train / clip(1 - adv_ptest_train, 1e-3, infinity)
w = w / mean(w)
```

The reproduced primary baseline was exactly 0.85290475 plain AUC and 0.8482631885976268 weighted AUC. The latter differs from the old JSON's 0.84826320404797 only in the last numerical digits. All before/after comparisons use the freshly reproduced same-split baseline.

## 5. Test direct ensemble ranking-loss weight edits

Ran:

```powershell
.venv-xpu/Scripts/python -u experiments/weight_edit_20261104/stack_edit.py
```

For each meta-training fold, initialized coefficients from its baseline logistic fit and sampled 524,288 positive/negative pairs with NumPy's default RNG. The RNG seed was `20261104 + 100*meta_split_seed + fold`; the full-data fit used seed 20261104. Evaluation rows were excluded from pair sampling and optimization.

For `d = (positive_score - negative_score) / temperature`, tested:

| Name | Pair loss | Temperature | Penalty on squared distance from baseline coefficients |
|---|---|---:|---:|
| `pairlog` | mean `softplus(-d)` | 1.0 | 0.001 |
| `auc_t075` | mean `sigmoid(-d)` | 0.75 | 0.002 |
| `auc_t025` | mean `sigmoid(-d)` | 0.25 | 0.002 |
| `auc_t010` | mean `sigmoid(-d)` | 0.10 | 0.01 |

Coefficients remained nonnegative. Float32 loss/gradient calculations ran on XPU; L-BFGS-B used `maxiter=250`, `ftol=2e-9`, `gtol=2e-6`, `maxls=30`. Since pair losses do not determine the intercept, a training-only scalar update subsequently matched the training positive rate. Its loop used at most 20 steps, each clipped to `[-2,2]`, with convergence threshold `1e-9`.

None of the four edits consistently improved both metrics across both meta splits. These candidates were retained as experiment records, not selected for the final model.

{{STACK_RESULTS}}

Saved `stack_results.json`, `stack_oof.npz`, `stack_test.npz`, and `stack_fold_coefficients.npz` under the experiment output directory, and `stack_coefficients.json` under the experimental weight directory.

## 6. Reconstruct exact neural preprocessing

Read `tune_out/var/mat_cggd_hv.npz`. Relevant shapes were `X=(24000,2171)`, `Xt=(4000,2171)`, `src=(12000,)`, and `Esrc/Ey=(85376,)`. The archive also contains the original and transformed feature matrices, labels, IDs, feature names, sample weights, and shift/segment augmentation data.

The existing network is a three-hidden-layer MLP: each hidden layer has Linear(1024), BatchNorm, SiLU and Dropout(0.4), followed by `net.12`, a linear layer from 1,024 features to one logit. In evaluation mode dropout is disabled. The original saved checkpoints and BatchNorm statistics were loaded strictly; all encoder tensors were frozen.

Canonical outer folds are `StratifiedKFold(5, shuffle=True, random_state=0)`. Each has 19,200 training rows and 4,800 outer validation rows. For fold `f` and source seed offset `o`, the original split is reconstructed as:

```python
perm = np.random.RandomState(f + 100 * o).permutation(outer_train_indices)
inner = perm[:1920]
core = perm[1920:]
```

Thus the core has 17,280 original rows. Only their shift negatives, transformed copies and matching real-segment rows fit the rank transform. Original, inner and outer IDs are checked for the required separation. Source IDs are mapped back to canonical training rows.

The saved `fold*_rankgauss_ref.npy` references were downsampled every 32 rows. They were therefore insufficient for exact replay. Rebuilt the full training reference for each fold/offset from core originals, core-source shift rows, five transformed core views and core-source segment rows. For each column:

1. Replace NaNs in the reference by positive infinity and sort it on the GPU.
2. Count finite reference values, with minimum count one.
3. For each input value, compute the mean of its left/right search positions.
4. Calculate `u = (mid_position + 0.5)/(finite_count + 1)`.
5. Clip `u` to `[1e-4,1-1e-4]` and convert through `sqrt(2)*erfinv(2*u-1)`.
6. Set transformed missing inputs to zero.

Inference batches were at most 8,192 rows. The six view order and original relative weights were:

```text
views:   orig, rev, flip, revflip, c1, c2
weights: 2,    1,   2,    1,       .5, .5    (divide by 7)
```

Loaded all four source seeds per fold to check the original archived mean predictions, even during the initial two-seed edit trial. Maximum original replay probability error was 5.960464477539063e-8. The script's hard guard was mean error below `2e-4` and maximum below `.01`; observed errors were much smaller.

A performance-only revision replaced slow object-array `np.isin` checks with `pd.Index(ids).get_indexer(...)` and boolean core membership masks. Membership and fold choices were unchanged.

## 7. Initial two-seed neural-head trial, and why it was revised

Created the first revision of `neural_head_edit.py`. The historical commands were:

```powershell
.venv-xpu/Scripts/python -u experiments/weight_edit_20261104/neural_head_edit.py --folds 0
.venv-xpu/Scripts/python -u experiments/weight_edit_20261104/neural_head_edit.py --resume
```

At that time the script edited seeds 0 and 1 of `MLPHV_big`, and wrote results directly into `tune_out/weight_edit_20261104/head_*` and checkpoints directly into `weights_experiments/weight_edit_20261104/neural_head/`. The first command ran fold 0; the second reused it and completed folds 1–4. The initial source revision was subsequently edited in place, so **these old commands do not reproduce that historical revision when run against today's defaults**. Its results and checkpoints remain; the source snapshots at the end of this file are the final revisions.

Fold 0 alone was mixed: seed 0 went from approximately 0.836196 to 0.835953, while seed 1 went from 0.834407 to 0.835449. Completed the five folds before judging it.

{{PILOT_RESULTS}}

The initial head optimizer fitted a training positive-rate intercept only on edited folds, while unchanged heads kept their old intercepts. Pairwise loss does not identify an intercept, and the source models had been trained with augmentation-induced class proportions. This treatment could introduce inconsistent score offsets across edited and unchanged folds. Revised the intercept rule to preserve the **original head's mean probability on the core rows**, rather than match the core label frequency. The old trial was not promoted as the final method.

Also added `--source`, `--seed-offset`, `--run-name`, and `--edit-seeds`, changed the default edit seed set to all four seeds, and isolated each run's files in its own directory.

## 8. Exact final neural-head optimization

For each source checkpoint, extracted its last hidden representation for all six views. Formed the weighted mean hidden representation `H` with the view weights in Step 6. Only the two tensors `net.12.weight` and `net.12.bias` were eligible to change.

With original head `w0,b0`:

```text
scale = max(sample_standard_deviation(H_core, axis=0), 0.03)
X = H_core / scale
baseline_core_score = H_core @ w0 + b0
score(delta) = baseline_core_score + X @ delta
objective = mean softplus(-(score_pos - score_neg)) + lambda * sum(delta^2)
new_weight = w0 + alpha * delta / scale
```

Sampled 262,144 positive/negative pairs from original core rows only, with RNG seed `20261104 + 100*fold + seed`. Each regularization fit starts at zero delta. L-BFGS-B settings were `maxiter=160`, `gtol=2e-6`, `ftol=2e-9`, `maxls=30`; the loss and its gradients ran in float32 on XPU.

For each checkpoint, scanned lambdas in the exact order `[0.03,0.3,3.0]` and interpolation strengths in the order `[0.25,0.5,1.0]`. Started with the original head as the selected option (`alpha=0`). A candidate replaced the current selected option only if the optimizer succeeded and its inner AUC exceeded the **current best** inner AUC by more than 0.0003. This is a sequential margin rule, not simply taking the maximum over all nine scores. Outer-fold labels were not used to select the edit.

For an accepted change, updated the bias to match the original mean core probability. The Newton-style intercept loop used at most 20 iterations, denominator floor `1e-8`, steps clipped to `[-2,2]`, and stopping threshold `1e-8`. If no edit was accepted, both original head tensors were kept.

All encoder weights and BatchNorm buffers were asserted equal to their originals before saving each checkpoint. Saved source hashes, selected lambda/alpha, inner scores, all nine trial scores, outer metrics and weight-change norm to per-fold JSON metadata.

Ran these three source ensembles, sequentially on the Intel GPU:

```powershell
.venv-xpu/Scripts/python -u experiments/weight_edit_20261104/neural_head_edit.py --source MLPHV_big --seed-offset 0 --run-name head_v2_MLPHV_big
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
.venv-xpu/Scripts/python -u experiments/weight_edit_20261104/neural_head_edit.py --source MLPHV_big_o1 --seed-offset 1 --run-name head_v2_MLPHV_big_o1
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
.venv-xpu/Scripts/python -u experiments/weight_edit_20261104/neural_head_edit.py --source MLPHV_big_o2 --seed-offset 2 --run-name head_v2_MLPHV_big_o2
```

That evaluated 3 source ensembles × 4 seeds × 5 outer folds = 60 checkpoints. Forty accepted a head change; twenty retained their original head. Each source retained its own original core/inner partition via seed offset 0, 1 or 2. Individual-model improvements did not always survive averaging; this is why the complete ensemble was evaluated subsequently.

## 9. Learn relative weights of the saved augmentation views

Created and ran:

```powershell
.venv-xpu/Scripts/python -u experiments/weight_edit_20261104/view_weight_edit.py
```

Expanded four baseline member groups into their saved per-view logit predictions:

| Member group | Source archives `ver_<tag>.npz` | Relative crop weight |
|---|---|---:|
| `VARC4hv_ns3_wc` | `VARC4hv_ns3_s0`, `VARC4hv_ns3_s1` | 0.125 |
| `VARC4_lm_wc` | `VARC4_lm_s0`, `VARC4_lm_s1` | 0.125 |
| `VARC4_full_wc2` | `VARC4_full_s0`, `VARC4_full_s1` | 0.125 |
| `MLPHV_bag_w` | `MLPHV_big`, `MLPHV_big_o1`, `MLPHV_big_o2` | 0.5 |

Other initial view weights were original=2, flip=2, reverse=1, reverse+flip=1; normalized within each group. Averaged clipped logits across the archive tags. Added the same deterministic clipping/rounding correction to every view of a group so that the old relative weights reproduce the original member's logit exactly. This correction used predictions, not labels. Checked its reconstruction to better than `1e-10` on both train and test arrays.

The expanded design had 77 columns. For a group's coefficient vector `c` and its original normalized view weights `v`, penalized deviations from `v*sum(c)`. The objective was:

```text
0.1 * sum(logistic_loss)
+ 0.5 * sum_groups [ sum(c)^2 + rho * ||c - v*sum(c)||^2 ]
```

All coefficients were constrained nonnegative, with an unrestricted intercept. Initialized from the baseline logistic fit, distributed across each member's old view proportions. L-BFGS-B settings: `maxiter=2000`, `ftol=1e-11`, `gtol=1e-6`.

Tested `rho=10,100,1000` on meta split seeds 7 and 29. Penalty 100 was carried forward as the more regularized setting within 0.00005 plain AUC of the best view-only setting. This was a choice informed by these validation results, not a separate independent confirmation.

{{VIEW_RESULTS}}

Saved the candidate coefficients to `weights_experiments/weight_edit_20261104/view_coefficients.json`, plus `view_results.json`, `view_oof.npz` and `view_test.npz` in the experiment output directory.

## 10. Reload edited checkpoints and reconstruct prediction archives

Created `replay_head_edits.py` and ran it before the complete ensemble comparison:

```powershell
.venv-xpu/Scripts/python -u experiments/weight_edit_20261104/replay_head_edits.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
.venv-xpu/Scripts/python -u experiments/weight_edit_20261104/finalize.py
```

The replay script reconstructed the exact fold-specific preprocessing again, loaded all 60 serialized checkpoint files, checked original checkpoint hashes, asserted finite edited tensors, and checked that every tensor except `net.12.weight` and `net.12.bias` was identical to its original.

The frozen encoder retained BF16 inference. The edited final linear layer used float32, matching the optimization. Computed each view's prediction from the reloaded network, checked its weighted logit against the corresponding optimization output with a `1e-4` maximum-error guard, and observed maximum error `4.5912606374542975e-5`.

For each source ensemble, averaged the four seeds' logits per view. Test logits were additionally averaged over the five fold models. Applied sigmoid and stored float32 probability archives, matching the former archive format. The TTA calculation then clips probabilities to `[1e-6,1-1e-6]`, takes logits, applies the original six view weights, averages the three source ensembles, and applies sigmoid.

{{HEAD_REPLAY_RESULTS}}

Wrote:

- `ver_head_v2_MLPHV_big.npz`, `ver_head_v2_MLPHV_big_o1.npz`, `ver_head_v2_MLPHV_big_o2.npz`.
- `oof_head_v2_bag.npy`, `test_head_v2_bag.npy`.
- `head_replay_results.json`.

Later added `--checkpoint-root` to replay from the copied bundle and checked its `--help` handling. The full replay above was run against the experimental checkpoint directory; the copied files were subsequently verified byte-for-byte by SHA-256. A second complete GPU replay from the copied directory was not necessary and was not claimed to have been run.

## 11. Evaluate complete ensembles and choose the retained method

`finalize.py` compared three configurations against the same original baseline on meta split seeds 7 and 29:

1. Learned view weights with the original neural bag.
2. Original stack design with the edited neural bag substituted for `MLPHV_bag_w`.
3. Edited neural bag plus learned view weights, penalty 100.

Selection required **both plain and weighted AUC to improve on both splits**. Among eligible candidates, selected the highest mean plain AUC across the two splits. The combined method won.

{{COMBINED_RESULTS}}

Fitted the chosen meta-model on all saved OOF training predictions, then applied it to test predictions. Exported the provisional `weights/auc_weight_edit_20261104/ensemble.json`, `submissions/sub_weight_edit_20261104.csv` and its JSON metadata. Recorded full results and OOF arrays in `final_results.json` and `final_oof.npz`.

The gain exceeded the experiment plan's +0.0003 bar for both primary-split metrics. The older incumbent files were nevertheless preserved; the new candidate was saved under its own name.

Computed 500 paired, class-stratified bootstrap resamples of the fixed primary-split OOF scores, using RNG seed 20261104. Positive and negative rows were resampled separately with replacement, preserving class counts. On each resample, computed new-minus-old plain and weighted AUC. The 2.5th/97.5th percentiles are:

{{BOOTSTRAP}}

These intervals do not include model fitting, hyperparameter selection or model-selection uncertainty.

## 12. Create `new weights` and test simplification

After the user requested that folder, created it explicitly:

```powershell
New-Item -ItemType Directory -Path 'C:\Users\Blitz\Downloads\Code\claude\ai\new weights' -Force | Out-Null
```

Created and ran:

```powershell
.venv-xpu/Scripts/python -u experiments/weight_edit_20261104/simplify.py
```

For each meta-training fold, fitted the nonnegative logistic stack over the 51 member groups, sorted its group coefficients descending with stable ordering, retained the top `k` groups, refitted the reduced logistic initialization, and then fitted the reduced view stack with penalty 100. Tested `k=4,6,8`. Group selection used only that meta-training fold, never the outer meta-validation labels.

Further pruning was eligible only if it did not reduce either metric on either split versus the complete combined edit. No pruning choice passed consistently. The saved metadata therefore says `pruning.selected = unpruned`: no additional top-k truncation was kept. This does **not** mean zero coefficients were retained.

{{PRUNING_RESULTS}}

Removed every exactly zero final coefficient (`coefficient > 0` is the retention test). This changed the exported design from 77 columns to 20 nonzero coefficients across seven active model groups. Checked dense-versus-sparse test logits with tolerance `1e-10`. This was exact-zero removal, not threshold pruning, rounding, weight quantization, or deletion of neural layers.

The original baseline's 51-member list already had only eight positive full-data coefficients. Therefore the simplification should not be described as eliminating 44 formerly active models. The retained seven groups also contain multiple seed/fold checkpoints; they are not seven single neural models.

## 13. Package the retained files and validate the submission

Copied only files belonging to the retained model groups into `new weights/`. Tree and sequence checkpoints were copied from existing `weights/`; MLP checkpoints were copied from the verified edited experiment outputs. Checked every destination path stayed under the requested folder, copied with `shutil.copy2`, and verified source and destination SHA-256 hashes matched.

The model bundle has 118 copied model/metadata files totaling 1,176,262,348 bytes: 9 XGBoost `.ubj` files, 85 PyTorch `.pt` files (60 MLP plus 25 sequence checkpoints), and 24 model/fold metadata JSON files. These counts exclude the top-level ensemble files, CSVs, README and this steps file.

Saved:

- `new weights/ensemble.json`: compact coefficients/intercept, exact feature order, source hashes, scores, selection details, bootstrap and replay command.
- `new weights/ensemble_weights.npz`: the same 20 coefficients, intercept and feature order as numeric arrays.
- `new weights/model_manifest.json`: source-to-copy mapping, SHA-256 and size for every retained file.
- `new weights/models/`: retained existing tree/sequence files.
- `new weights/neural_head/`: edited MLP files and per-fold edit metadata.
- `new weights/submission.csv` and `new weights/submission_replay.csv`.
- Matching `submissions/sub_weight_edit_20261104.csv` and its JSON.
- `new weights/README.md`, plus numeric packaging/pruning evidence in `tune_out/weight_edit_20261104/simplified_results.json` and `pruning_oof.npz`.

Updated the artifact replay code to support sparse feature selection by the saved names, preserving their exact order. Before writing the CSV, checked 4,000 finite probabilities in `[0,1]`, unique identifiers matching the sample submission, and exact sample-submission row order. After CSV writing, read it back and checked column names, IDs, row count and numeric round-trip error below `1e-14`.

Inside `simplify.py`, reloaded the saved compact artifact through `finalize.predict(...)`, validated its input hashes, recomputed the CSV, and asserted a maximum difference below `1e-14`. Later compared the three CSV SHA-256 hashes with `Get-FileHash`; they were identical.

{{SUBMISSION_VERIFICATION}}

## 14. Record the work and check the scripts

Created or updated these documentation files during the experiment:

- `experiments/weight_edit_20261104/COMPETITION_PLAN.md` — the initial plan plus subsequent view-weight/simplification additions.
- `docs/WEIGHT_EDIT_20261104.md` — measured results, implementation, replay, limitations and commands.
- `README.md` — appended the dated candidate summary.
- `docs/EXPLORATION_NOTES.md` — appended the dated experiment outcomes.
- `docs/LOOP_LEDGER.md` — appended commands/results/verdicts for the new experiments.
- `new weights/README.md` — packaged-model and submission instructions.
- This `new weights/steps.md` — added at the user's follow-up request, after verifying existing source/results. Writing this document did not retrain models or change weights.

The substantive code was kept in six new scripts under `experiments/weight_edit_20261104/`. Existing training scripts in `experiments/v6/` were read but not modified by this run.

Ran a syntax check on the first five scripts after their implementation:

```powershell
.venv-xpu/Scripts/python -m py_compile experiments/weight_edit_20261104/stack_edit.py experiments/weight_edit_20261104/view_weight_edit.py experiments/weight_edit_20261104/neural_head_edit.py experiments/weight_edit_20261104/replay_head_edits.py experiments/weight_edit_20261104/finalize.py
```

`simplify.py` was added afterward and verified by executing its complete pruning, packaging, hash-checking and submission-replay workflow. The later sparse-artifact replay change was exercised by that workflow. The final packaged-checkpoint argument addition was checked with:

```powershell
.venv-xpu/Scripts/python experiments/weight_edit_20261104/replay_head_edits.py --help
```

No separate unit-test suite was added; verification was the actual fold evaluations, checkpoint replay, tensor/hash comparisons and complete submission replay.

## 15. Reproduce the final result with the final script revisions

These commands describe the final reproducible workflow, not a claim that every earlier exploratory launch used the final CLI defaults. Run them from the project root. They reuse the original project feature matrices and base prediction archives; they are not a new raw-data training pipeline.

To reproduce only the packaged submission:

```powershell
.venv-xpu/Scripts/python experiments/weight_edit_20261104/finalize.py --predict "new weights/ensemble.json" --output "new weights/submission_replay.csv"
```

To reconstruct the edited neural prediction archives from the copied checkpoints:

```powershell
.venv-xpu/Scripts/python experiments/weight_edit_20261104/replay_head_edits.py --checkpoint-root "new weights/neural_head"
```

To rerun the final scientific workflow, including the rejected ensemble-weight screen:

```powershell
Set-Location -LiteralPath 'C:\Users\Blitz\Downloads\Code\claude\ai'
.venv-xpu/Scripts/python -u experiments/weight_edit_20261104/stack_edit.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
.venv-xpu/Scripts/python -u experiments/weight_edit_20261104/neural_head_edit.py --source MLPHV_big --seed-offset 0 --run-name head_v2_MLPHV_big
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
.venv-xpu/Scripts/python -u experiments/weight_edit_20261104/neural_head_edit.py --source MLPHV_big_o1 --seed-offset 1 --run-name head_v2_MLPHV_big_o1
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
.venv-xpu/Scripts/python -u experiments/weight_edit_20261104/neural_head_edit.py --source MLPHV_big_o2 --seed-offset 2 --run-name head_v2_MLPHV_big_o2
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
.venv-xpu/Scripts/python -u experiments/weight_edit_20261104/view_weight_edit.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
.venv-xpu/Scripts/python -u experiments/weight_edit_20261104/replay_head_edits.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
.venv-xpu/Scripts/python -u experiments/weight_edit_20261104/finalize.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
.venv-xpu/Scripts/python -u experiments/weight_edit_20261104/simplify.py
```

Those training/export commands overwrite their experiment outputs and the newly named candidate, so use a copy of the project if retaining this exact snapshot is important. The commands do not target original training data or source weight files. Seeded procedures and hashes make the inputs/method auditable; GPU/library changes can still produce small numerical differences. Submission-only replay was verified exactly in the recorded environment.

## 16. Interpretation and remaining limits

- The final AUC is **0.8533852152777779**, not 0.87. The shortfall is approximately 0.016615.
- The neural head edits exclude each canonical outer fold's labels from fitting and inner selection. Earlier project-level model selection still used these data.
- Ensemble scores are meta-level CV over existing OOF base predictions, not fully nested retraining of every base learner.
- The two meta splits reuse the same 24,000 labelled rows; the second split is not an independent dataset.
- Multiple settings were compared and chosen using these validation data. Bootstrap intervals condition on the already fitted OOF scores and omit that selection/fitting uncertainty.
- Distribution-weighted AUC uses the existing adversarial density weights; it is not an observed test-label AUC.
- The copied checkpoints are genuine model files, but this bundle relies on the original project's feature tables, preprocessing data and prediction archives. The approximate rank references alone cannot reproduce exact neural inference.
- Source tree/sequence models were retained and copied, not retrained or internally edited. The only internal model parameters changed here were selected MLP output-layer weights/biases.
- No outside training data, pretrained downloads, pseudo-labels, new synthetic series, threshold tricks, hidden test labels, Kaggle uploads, Git commits or messages to other people were used in this run.

## Appendix A. Exact final coefficients and group totals

The prediction is `sigmoid(intercept + sum(coefficient_j * expanded_logit_feature_j))`. View features include the deterministic reconstruction correction in `view_weight_edit.expand`; they should not be replaced with arbitrary raw probabilities. The coefficients are nonnegative logistic weights and do not have to sum to one.

{{FINAL_COEFFICIENTS}}

## Appendix B. All 60 final neural decisions

The table is taken directly from the copied per-fold JSON files. `alpha=0` and no lambda means the original head was retained. Outer AUC columns are per-fold diagnostic results, not the pooled ensemble score. Full nine-trial inner optimization records and original-source hashes remain in the indicated JSON files.

{{NEURAL_DECISIONS}}

## Appendix C. Audits and artifact inventory

{{AUDIT_RECORDS}}

{{ARTIFACT_INVENTORY}}

## Appendix D. Exact packaged-file manifest

This snapshot lists every copied model/metadata file, source, size and SHA-256, matching `new weights/model_manifest.json`. Top-level documentation, ensemble coefficient files and CSVs are separate artifacts.

{{MODEL_MANIFEST}}

## Appendix E. Final experiment plan and exact source snapshots

These are the final source revisions as read when this steps document was created. They include the calibration fix, efficient source-ID matching, source/offset CLI, sparse artifact replay, and packaged checkpoint-root option. The earlier two-seed revision was overwritten in place; its differing behavior and retained outputs are explicitly documented in Step 7.

Each source snapshot is accompanied by the SHA-256 of the actual file bytes. Text displayed below uses normalized Markdown line endings; compare hashes against the on-disk source files, not a manually copied code block. To restore a script, keep its original project-relative location because `ROOT` is calculated from `__file__`.

{{SOURCE_SNAPSHOTS}}
