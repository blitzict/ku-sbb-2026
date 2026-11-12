# White-noise recovery with crop and reversal training

The primary ROC-AUC reached **0.876827** across the 24,000 labelled series. The weights combine Claude’s volatility-aware trees trained on actual crops and reversed series with two trees optimized for positive–negative ranking and earlier estimates of changes in timing, level, scale and noise shape. Every transformed view of a source series remains in its source fold.

| Primary local measurement | Locked baseline | Candidate |
| --- | ---: | ---: |
| All series | 0.870943 | 0.876827 |
| Other 23,179 series | 0.901074 | 0.904242 |
| 821 historical misses against all negatives | 0.460677 | 0.503539 |
| Historical misses found at 10% false positives | 11 | 21 |
| Historical misses found at 20% false positives | 42 | 87 |
| Historical misses found at 30% false positives | 98 | 172 |

Of the 821 historical misses, 559 moved above more negatives and 262 moved below more negatives. The net change was 422,286 correctly ordered positive–negative pairs involving those misses. The historical indicator is used only for evaluation.

The six output-weight partitions scored 0.876115 to 0.876827. They share the same series and base out-of-fold predictions. The primary test-distribution-weighted AUC was 0.871382. See the [six paired evaluations](validation.json) and [fixed-prediction uncertainty](paired_audit.json).

Run `.venv-xpu/Scripts/python "new weights/wn_stage_reversible_rank_candidate/replay.py"` from the project root to reproduce the [4,000 test predictions](submission.csv). Test labels are unavailable.
