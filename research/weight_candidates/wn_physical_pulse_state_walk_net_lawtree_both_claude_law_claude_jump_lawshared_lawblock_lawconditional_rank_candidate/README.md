# White-noise break recovery

These weights combine the relative likelihoods of a level, scale or noise-shape change with a walk-specific jump measurement, two separately trained tree models, and neural estimates. Two additions compare short blocks of the original observations and estimate how plausible each physical explanation is given the series’ measured properties.

| Primary split 7 | Frozen baseline | Candidate |
| --- | ---: | ---: |
| Ordinary ROC-AUC | 0.870943 | 0.875312 |
| 821 historical misses versus all 12,000 non-breaks | 0.460677 | 0.490865 |
| ROC-AUC on the other 23,179 series | 0.901074 | 0.903546 |
| Distribution-weighted ROC-AUC | 0.865530 | 0.870002 |
| Historical misses recovered at 5% false positives | 8 | 9 |
| At 10% false positives | 11 | 19 |
| At 20% false positives | 42 | 72 |
| At 30% false positives | 98 | 149 |

540 historical misses moved ahead of more non-breaks and 281 moved behind more, a net gain of 297,420 pair orderings. All six splits improved overall and in both comparison groups. Ordinary AUC ranged from 0.874564 to 0.875475. Historical miss membership was used only for evaluation. The splits reuse the same 24,000 labeled series and archived base-model out-of-fold predictions. The blind test AUC is unknown.

Run `python replay.py` from this directory to reproduce the 4,000 probabilities and check their SHA-256. The verified submission checksum is `553ad1f0208cd37ea62a154ad5a4270fc57d9b5ac924b22a3ff33e7b2ff18331`. The full-precision weights, per-split results and ordered predictions are in `weights.json`, `validation.json` and `submission.csv`.
