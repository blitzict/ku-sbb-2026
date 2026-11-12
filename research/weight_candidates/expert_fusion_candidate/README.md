# Regime-dependent ensemble candidate

`submission.csv` contains a break probability for each of the 4,000 competition test series in sample-submission order. The parent is the top-level `new weights/ensemble.json`, SHA-256 `f769bdde48cd8151f0d15cfb428a179b383abc748c82a922df84627b72014921`.

The parent gives each expert group a constant set of coefficients. The new readout adds a logit correction using the seven groups' disagreements, the incumbent score, and eleven process descriptors including segment lengths, autocorrelation, tail shape, and volatility clustering. The feature transformations were programmed explicitly. The correction coefficients were fitted by regularized logistic regression on each meta-training fold. The seven groups reuse the parent's base models.

| Five-fold split seed | Parent AUC | Candidate AUC | Parent weighted AUC | Candidate weighted AUC |
|---|---:|---:|---:|---:|
| 7 | 0.853385 | 0.853766 | 0.848664 | 0.849023 |
| 29 | 0.853500 | 0.853997 | 0.848730 | 0.849236 |
| 0 | 0.853449 | 0.853891 | 0.848680 | 0.849116 |

The primary paired AUC difference is +0.000381. Its approximate 95% paired DeLong interval is −0.000096 to +0.000857 conditional on the stored predictions. Four of five folds improved on split 7, five of five on 29 and four of five on 0. All three partitions reuse the same 24,000 labelled series and precomputed base predictions. The intervals omit model-selection and refitting uncertainty. Hidden-test AUC is unknown.

Four readout designs were compared. Only the regime-dependent readout cleared +0.0003 for both metrics on all three splits. A further comparison using two-sided dynamics or spline gates did not clear that threshold relative to this candidate. The full results and aligned OOF scores are under `tune_out/weight_space/expert_fusion_*` and `tune_out/weight_space/paired_regime_*`.

From the project root, reproduce the CSV with the existing feature tables, prediction archives and Python environment.

```powershell
.venv-xpu/Scripts/python experiments/weight_space/expert_fusion.py --replay "new weights/expert_fusion_candidate/weights.json" --output "new weights/expert_fusion_candidate/submission_replay.csv"
```

The recipe verifies source hashes. The expected CSV SHA-256 is `627e75687a83e52282ffdeeb12738f2f22da237ead6dc29a912ec44c107b4171`.
