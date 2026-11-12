# Joint ranking of three views of the same series

The primary local ROC-AUC is **0.878545**. One rank tree learns a single score from the full series and two crops that keep the marked boundary. Its loss also evaluates each view individually. This tree contributes alongside the separately trained rank trees, Claude's three-partition crop ensemble, and the previous physical ensemble.

| Primary evaluation | Locked reference | These weights |
| --- | ---: | ---: |
| Overall ROC-AUC | 0.870943 | 0.878545 |
| 821 historical misses against all 12,000 negatives | 0.460677 | 0.500583 |
| Remaining 23,179 series | 0.901074 | 0.906303 |
| Recovered at 10% false positives | 11 | 21 |
| Recovered at 20% false positives | 42 | 85 |

Across six output-weight partitions, overall AUC ranged from 0.877842 to 0.878548. These evaluations reuse the same 24,000 labelled series. Primary distribution-weighted AUC was 0.872972. Claude independently replayed all 4,000 test predictions byte-for-byte. The test labels are unavailable.

Run `.venv-xpu/Scripts/python "new weights/wn_local_rank_linked_views_candidate/replay.py"` from the project root. [Predictions](submission.csv) · [Recipe](weights.json) · [Paired audit](paired_audit.json) · [Six partitions](validation.json)
