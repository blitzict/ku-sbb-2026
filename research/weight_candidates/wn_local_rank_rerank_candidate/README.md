# Ranking series with similar scores

These weights adjust the frozen three-seed volatility ensemble by comparing disagreements among experts on series with similar expected break probabilities. Three inner partitions of each outer training fold supplied held-out inputs for the adjustment. The adjustment used real positive and negative training examples. The list of 821 historical misses was reserved for evaluating the resulting weights.

| Primary local measurement | Locked 0.870943 baseline | Updated weights |
| --- | ---: | ---: |
| All 24,000 series, ROC-AUC | 0.870943 | 0.877828 |
| Other 23,179 series, ROC-AUC | 0.901074 | 0.905739 |
| 821 misses against the 12,000 negatives, ROC-AUC | 0.460677 | 0.497787 |
| Misses found at 10% false positives | 11 | 20 |
| Misses found at 20% false positives | 42 | 83 |
| Misses found at 30% false positives | 98 | 163 |

Across six output-weight partitions, ordinary AUC ranged from 0.877142 to 0.877828. These partitions reuse the same series and base out-of-fold predictions. The primary distribution-weighted AUC was 0.872351. The [preceding 0.876827 candidate](../wn_stage_reversible_rank_candidate/README.md) recovered 87 historical misses at 20% false positives.

Run `.venv-xpu/Scripts/python "new weights/wn_local_rank_rerank_candidate/replay.py"` from the project root to reproduce the [4,000 test predictions](submission.csv). The [validation record](validation.json) gives each split and the submission hash. The [paired audit](paired_audit.json) compares the frozen scores on the same rows. Test labels are unavailable.
