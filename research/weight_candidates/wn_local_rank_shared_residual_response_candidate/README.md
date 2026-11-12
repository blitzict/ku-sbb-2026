# Weights comparing how detectors respond to a change

These weights reach **0.87911421 primary local ROC-AUC**. They distinguish the vote common to several detector families from the differences between paired detectors. The pairs compare the order of signed and large movements, the response after a shock, rank dependence across source folds, and a tree with some physical inputs withheld. The change improved overall AUC on all seven output-weight partitions. It does not isolate the contribution of any single pair. [Training](../../experiments/weight_space/white_shared_residual.py) · [Validation](validation.json)

| Primary local metric | Locked baseline | Previous weights | These weights |
| --- | ---: | ---: | ---: |
| Overall ROC-AUC | 0.870943 | 0.879053 | **0.879114** |
| 821 historical positives against 12,000 negatives | 0.460677 | 0.499236 | 0.498719 |
| Remaining 23,179 series | 0.901074 | 0.906947 | **0.907051** |
| Recovered at 10% false positives | 11 | 21 | **24** |
| Recovered at 20% false positives | 42 | 86 | **88** |

The additional partition scored **0.878467**, up from 0.878405. The partitions reuse the same labelled series and base-model out-of-fold predictions. At the primary 50% false-positive point, recovery fell from 411 to 409. On the primary partition another **127,554** correctly ordered positive-negative pairs would reach 0.880. The [dedicated recovery weights](../wn_local_rank_concordance_tree_channels_shared_candidate/README.md) recover 90 at 20% false positives.

All 4,000 test predictions replayed exactly. Test labels are unavailable. [Predictions](submission.csv) · [Weights](weights.json) · [Replay](replay.py) · [Pair accounting for the other misses](../other_missed_breaks.md)
