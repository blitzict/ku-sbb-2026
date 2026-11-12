# Weights trained on closely scored positive and negative series

These weights use disagreements among the same detector families as the 0.878883 ensemble. The change is in how those disagreements receive weight. The fit compares each eligible positive to 32 negatives with similar starting scores and also fits the labels of every training series. Overall AUC improved on all six output-weight partitions. [Fitting code](../../experiments/weight_space/white_family_objectives.py) · [Six partitions](validation.json)

| Primary split | Locked baseline | Previous best | These weights |
| --- | ---: | ---: | ---: |
| Overall ROC-AUC | 0.870943 | 0.878883 | **0.878971** |
| Historical 821 against all 12,000 negatives | 0.460677 | 0.498841 | 0.498771 |
| Remaining 23,179 series | 0.901074 | 0.906793 | 0.906893 |
| Recovered at 10% false positives | 11 | 24 | 24 |
| Recovered at 20% false positives | 42 | 83 | **86** |

Both cohort AUCs exceed the locked baseline. Against the previous best, AUC on the 821 fell by 0.000070 despite recovering three more at the 20% cutoff. The specialist still recovers 90 there. Historical miss membership is used only for evaluation. [Paired audit](paired_audit.json) · [Recovery specialist](../wn_local_rank_concordance_tree_channels_shared_candidate/README.md)

The primary score of **0.87897076** is 148,210 correctly ordered pairs short of 0.880000 on these 24,000 labelled series. The partitions refit the output weights on the same observations and fixed base-model out-of-fold scores. The test set has no public labels. All 4,000 test predictions replayed byte for byte. [Replay](replay.py) · [Predictions](submission.csv)

On the additional output partition the same pairwise readout scored 0.878408, versus 0.878317 for the previous probability-based readout. Claude independently reproduced all 4,000 test predictions. [Additional partition](additional_partition_61817.json) · [Claude's replay](../../docs/COORDINATION.md)
