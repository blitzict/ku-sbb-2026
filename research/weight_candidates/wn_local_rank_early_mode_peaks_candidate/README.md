# Two-peak correction on the original and current detector families

The original project sequence models score two views of the observed series. One emphasizes which of two common levels each observation visits. The other emphasizes motion around those levels. A fitted mixture must show evidence for two peaks within at least one segment for the correction to carry weight. The output weights use the labels on training rows. Membership in the 821 historical misses is used only for evaluation.

| Primary local split | Locked baseline | Before two-peak correction | Final |
| --- | ---: | ---: | ---: |
| All 24,000 series | 0.870943 | 0.878182 | 0.878242 |
| 821 historical positives against all 12,000 negatives | 0.460677 | 0.499605 | 0.500454 |
| Remaining 23,179 series | 0.901074 | 0.905985 | 0.905987 |
| Historical misses recovered at 20% false positives | 42 | 86 | 85 |

Overall and historical-subset AUC increased on all six output-weight partitions compared with the immediate parent. Remaining-series AUC rose on two and fell on four by at most 0.000022. It exceeded the locked baseline on all six. Overall AUC ranged from 0.877405 to 0.878242. The partitions reuse the labelled series and base out-of-fold predictions.

Run `.venv-xpu/Scripts/python "new weights/wn_local_rank_early_mode_peaks_candidate/replay.py"` from the project root to reproduce the [4,000 predictions](submission.csv) from stored scores, features and weights. The predictions are in sample-submission order. [Validation](validation.json) · [Paired recovery counts](paired_audit.json) · [Signal examples](../white_noise_signal_findings.md)
