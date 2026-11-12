# New weights — KU-SBB, 2026-11-04

The [audit of the other missed breaks](other_missed_breaks.md) counts how many positive–negative pairs must change order to reach 0.880 and records the full-cohort cost of the attempted recoveries.

The [highest verified overall weights](../../research/weight_candidates/wn_local_rank_shared_residual_response_candidate/README.md) reach **0.87911421 local ROC-AUC**, with **0.498719** on the 821 historical misses against all negatives and **0.907051** on the remaining series. They recover **24** at 10% false positives and **88** at 20%. The [previous weights](../../research/weight_candidates/wn_local_rank_without_multiview_trees_candidate/README.md) score 0.879053 overall and 0.499236 on the 821. The [recovery-focused weights](../../research/weight_candidates/wn_local_rank_concordance_tree_channels_shared_candidate/README.md) recover **90** at 20% false positives.

The [verified recovery weights](../../research/weight_candidates/wn_local_rank_linked_views_candidate/README.md) score **0.878545 local ROC-AUC**. They recover 85 of the 821 historical misses at 20% false positives, up from 42 with the locked baseline. AUC on the remaining 23,179 series rose from 0.901074 to 0.906303. These weights combine the [two-peak signal](../../research/weight_candidates/wn_local_rank_early_mode_peaks_candidate/README.md) with trees trained on different source partitions and a tree that learns jointly from three views of each source. The [0.878079 weights](../../research/weight_candidates/wn_local_rank_family_agreement_candidate/README.md) recover 87 at the same operating point. The [0.876827 weights](../../research/weight_candidates/wn_stage_reversible_rank_candidate/README.md) have higher AUC on the 821 against all negatives, 0.503539 versus 0.500583. The [signal findings](white_noise_signal_findings.md) show measured gains and examples from real series.

The [recovery-focused weights](../../research/weight_candidates/wn_local_rank_concordance_tree_channels_shared_candidate/README.md) recover **90 at 20% false positives**, with 0.510282 AUC on the 821, 0.905349 on the remainder and 0.878320 overall. The [0.870943 baseline](../../research/weight_candidates/hard_state_tree_candidate/README.md) and [earlier 0.870503 candidate](../../research/weight_candidates/extended_curve_net_bank_both_pulse_joint_head_family_state_net_candidate/README.md) record the preceding weight edits and validation.

## Original ensemble in this directory

Local meta-CV AUC: **0.85338522** (previous **0.85290475**).
Distribution-weighted AUC: **0.84866428** (previous **0.84826319**).
These figures describe the original ensemble's local validation.

The recipe retains **7 active model groups / 20 nonzero coefficients**; zero-weight contributors were removed with identical predictions.
Further pruning selected: `unpruned`. Model-group selection was performed inside each meta-training fold.
The kept changes combine pairwise-logistic edits to MLP output layers with learned, regularized weights for original/reversed/flipped/cropped predictions.

Files:

- `submission.csv`: 4,000 rows, original sample-submission identifier order.
- `ensemble.json` / `ensemble_weights.npz`: compact coefficients and intercept, feature order, source hashes and validation results.
- `models/`: only retained original tree and sequence model files.
- `neural_head/`: edited MLP checkpoints; encoder tensors are unchanged.
- `model_manifest.json`: file mapping, hashes, and sizes for the copied models.

Reproduce the submission from the project root:

```powershell
.venv-xpu/Scripts/python experiments/weight_edit_20261104/finalize.py --predict "new weights/ensemble.json" --output "new weights/submission_replay.csv"
```

This bundle uses the existing project's feature tables and prediction archives. Raw-series feature generation and exact neural preprocessing still require the original project/data; the small stored rank references are approximate.
The exported submission was reloaded, validated, and reproduced exactly from the saved compact coefficients.

Validation: base neural edits use the original five outer folds; edit strength uses only the original inner split. Ensemble estimates use two meta splits on previously generated OOF predictions. They are exploratory and do not constitute independent holdout validation or fully nested retraining of all base models.
No Kaggle submission was sent. Original checkpoints and incumbent submissions are preserved.
