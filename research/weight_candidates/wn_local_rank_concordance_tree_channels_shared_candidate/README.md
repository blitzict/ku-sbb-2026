# Weights for recovering more of the 821 historical misses

These weights recover **90 of 821** at a 20% false-positive rate, compared with **42** for the locked baseline and **85** for the 0.878545 ensemble. AUC over all 24,000 series is **0.878320**. All 4,000 test predictions were reproduced from the saved weights and inputs.

| Primary local measure | Locked baseline | Recovery weights |
| --- | ---: | ---: |
| Overall ROC-AUC | 0.870943 | 0.878320 |
| 821 historical positives against all 12,000 negatives | 0.460677 | 0.510282 |
| Remaining 23,179 series | 0.901074 | 0.905349 |
| Recovered at 10% false positives | 11 | 23 |
| Recovered at 20% false positives | 42 | 90 |
| Recovered at 30% false positives | 98 | 171 |

The output fit uses how much each physical feature family contributed to two separate forests’ scores. It gives extra weight to ranking ambiguous white-looking training positives, defined by observed pre-boundary statistics and an out-of-fold score. The historical IDs are reserved for evaluation. The six output-weight partitions reuse the same labelled population and base predictions. The 4,000 test labels are unavailable.

[Predictions](submission.csv) · [Replay](replay.py) · [Paired uncertainty and threshold counts](paired_audit.json) · [Six-split results](validation.json) · [Training script](../../experiments/weight_space/white_concordance.py)
