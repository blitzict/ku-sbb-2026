# Weights from the first joint fit of original and physical detectors

The primary local ROC-AUC is **0.87868428** on 24,000 labelled series. The model recovers **85 of 821** historically missed breaks at a 20% false-positive rate, up from 42 with the locked baseline. The dedicated recovery weights recover 90 at that operating point. [Recovery weights](../wn_local_rank_concordance_tree_channels_shared_candidate/README.md)

| Primary local comparison | Locked baseline | These weights |
| --- | ---: | ---: |
| Overall ROC-AUC | 0.870943 | 0.878684 |
| 821 historical positives versus all negatives | 0.460677 | 0.496122 |
| Remaining 23,179 series | 0.901074 | 0.906780 |
| Recovered at 10% false positives | 11 | 21 |
| Recovered at 20% false positives | 42 | 85 |
| Recovered at 30% false positives | 98 | 154 |

These weights give 25% of the final log odds to a model that learns the relative contributions of the original estimators, crop views and physical detectors together in its first fit. Compared with the immediate 0.878540 ensemble, its overall AUC improved in all six output partitions. On the primary partition, AUC on the 821 fell from 0.500508 to 0.496122. Both cohorts improved against the locked 0.870943 baseline in all six partitions. [Training](../../experiments/weight_space/white_upstream_stack.py) · [First-layer reconstruction](../../experiments/weight_space/white_regime_branches.py) · [Six-partition measurements](validation.json)

The output partitions share the same subjects and base out-of-fold predictions. All 4,000 unlabelled test predictions replayed byte for byte. [Predictions](submission.csv) · [Replay](replay.py) · [Paired local comparison](paired_audit.json)
