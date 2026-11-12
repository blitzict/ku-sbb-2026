# Weights that combine agreement and disagreement across models

The primary local ROC-AUC is **0.87888256**. Compared with the 0.87868428 parent, AUC rose on all six output-weight partitions. On the primary partition both the historical misses and the remaining series improved. The weights recovered 24 of the 821 at 10% false positives, up from 21 with the parent. At 20% false positives they recovered 83, down from 85.

| Primary measurement | Locked baseline | Immediate parent | These weights |
| --- | ---: | ---: | ---: |
| Overall ROC-AUC | 0.870943 | 0.878684 | 0.878883 |
| 821 historical positives against all negatives | 0.460677 | 0.496122 | 0.498841 |
| Remaining 23,179 series | 0.901074 | 0.906780 | 0.906793 |
| Recovered at 10% false positives | 11 | 21 | 24 |
| Recovered at 20% false positives | 42 | 85 | 83 |

The output fit aligns the score ranges of the original detectors, the crop-based trees and the newer sequence and likelihood models. Within each family it estimates the shared directions of disagreement on the training folds, then learns how those directions shift the parent score. Adding pairwise products of the directions did not improve overall AUC consistently. [Training and comparison](../../experiments/weight_space/white_opinion_spectrum.py) · [Six partitions](validation.json)

All 4,000 test predictions replayed byte for byte. The six partitions reuse the same labelled sources and fixed base-model out-of-fold scores. The dedicated recovery weights retain the best observed recovery at 20% false positives, with 90 of 821. [Replay](replay.py) · [Predictions](submission.csv) · [Paired audit](paired_audit.json) · [Recovery weights](../wn_local_rank_concordance_tree_channels_shared_candidate/README.md)

An independent reconstruction matched the primary logits to within 0.000000000063. On the additional output-weight split declared before fitting, the same recipe scored 0.878317 overall, 0.498251 on the 821 and 0.906229 on the remainder. It recovered 85 at 20% false positives. This check used the same labelled observations. Claude also independently replayed the 4,000 test predictions. [Additional partition](additional_partition_61817.json) · [Coordination record](../../docs/COORDINATION.md)
