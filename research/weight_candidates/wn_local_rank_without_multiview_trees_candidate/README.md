# Weights with one fewer source of duplicate votes

The primary local ROC-AUC is **0.87905263**. The underlying ensemble already uses trees trained on several windows of each series. Refitting the final layer without a second set of coefficients for those same models improved overall AUC on all six output-weight partitions. Omitting the sequence and likelihood networks instead lowered primary AUC from 0.878971 to 0.878843. [Paired family comparison](../../experiments/weight_space/white_family_ablation.py) · [Six-partition results](validation.json)

| Primary split | Locked baseline | Previous best | These weights |
| --- | ---: | ---: | ---: |
| Overall ROC-AUC | 0.870943 | 0.878971 | **0.879053** |
| 821 historical positives against all 12,000 negatives | 0.460677 | 0.498771 | **0.499236** |
| Remaining 23,179 series | 0.901074 | 0.906893 | **0.906947** |
| Recovered at 10% false positives | 11 | 24 | 21 |
| Recovered at 20% false positives | 42 | 86 | 86 |

The 0.880 target is 136,421 correctly ordered pairs away on the primary partition. The dedicated recovery weights recover 90 at 20% false positives. All 4,000 predictions from these weights replayed exactly. The six partitions reuse the same 24,000 labelled series and frozen base-model out-of-fold scores. [Replay](replay.py) · [Predictions](submission.csv) · [Paired audit](paired_audit.json) · [Recovery specialist](../wn_local_rank_concordance_tree_channels_shared_candidate/README.md)

On the additional output-weight partition the score was **0.878405**, versus **0.878408** with all families in the final fit. The primary gain did not reproduce on that partition. Historical AUC there was 0.497677 and remaining-series AUC was 0.906366. Claude independently reproduced the test predictions. [Additional partition](additional_partition_61817.json) · [Claude's replay](../../docs/COORDINATION.md)
