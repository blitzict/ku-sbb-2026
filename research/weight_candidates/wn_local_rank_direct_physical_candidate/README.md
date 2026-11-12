# Direct vote from trees trained on different sets of real series

Two physical tree ensembles contribute up to 20% of the final score after their logits are scaled to comparable training-set quartiles. The vote is smaller on series that look white before the boundary and for which the existing model is uncertain. The first ensemble averages Claude’s original two-crop bag and a new source split. The other averages three source splits of the rank tree. All copies of each real series belong to the same base-training fold.

| Primary split | Locked baseline | Before the vote | Final |
| --- | ---: | ---: | ---: |
| Overall ROC-AUC | 0.870943 | 0.878242 | 0.878537 |
| 821 historical positives against all 12,000 negatives | 0.460677 | 0.500454 | 0.500487 |
| Remaining 23,179 series | 0.901074 | 0.905987 | 0.906301 |
| Recovered at 10% false positives | 11 | 21 | 21 |
| Recovered at 20% false positives | 42 | 85 | 85 |

Overall AUC rose on all six output-weight partitions relative to the immediate parent and ranged from 0.877824 to 0.878542. These partitions reuse the same labelled series. The weighted primary AUC was 0.872948. An independent replay by Claude matched all 4,000 rows byte for byte.

Run `.venv-xpu/Scripts/python "new weights/wn_local_rank_direct_physical_candidate/replay.py"` from the project root. [Predictions](submission.csv) · [Six paired evaluations](validation.json) · [Fixed false-positive rates and intervals](paired_audit.json) · [Training script](../../experiments/weight_space/white_fold_vote.py)
