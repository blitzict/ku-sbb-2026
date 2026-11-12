# White-noise recovery weights

The model combines Claude's trees trained on real cropped series with estimates of how likely one segment's observations are under the other segment's distribution, a volatility-aware tree and the previous ensemble. Every crop stays in the training fold of its source series. Historical miss flags enter the evaluation only.

| Primary local evaluation | Locked parent | Candidate |
| --- | ---: | ---: |
| All 24,000 series, ROC-AUC | 0.870943 | 0.876101 |
| Remaining 23,179 series, ROC-AUC | 0.901074 | 0.903771 |
| 821 historical misses against 12,000 negatives, ROC-AUC | 0.460677 | 0.499329 |
| Misses found at 10% false positives | 11 | 21 |
| Misses found at 20% false positives | 42 | 81 |
| Misses found at 30% false positives | 98 | 163 |

Across six partitions of the output-weight fit, overall AUC ranged from 0.875493 to 0.876101. Those partitions reuse the same series and base out-of-fold predictions. The distribution-weighted primary AUC was 0.870861. The 4,000 test series have no local labels.

Run `.venv-xpu/Scripts/python "new weights/wn_stage_family_density_candidate/replay.py"` from the project root to check the [submission](submission.csv) against the saved test inputs and weights. The [validation record](validation.json) includes all six splits and the hashes.
