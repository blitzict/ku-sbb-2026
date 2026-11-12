# Break scores from agreement among detector families

These weights compare detectors on series with similar starting scores. They allow the effect of a disagreement to depend on whether it comes from the crop trees, rank trees, noise-law trees, likelihood models or sequence models. The group definitions use the models' construction. The 821 historical misses enter only the evaluation.

| Measurement on primary split 7 | Locked baseline | These weights |
| --- | ---: | ---: |
| ROC-AUC across all series | 0.870943 | 0.878079 |
| AUC on the 821 against all negatives | 0.460677 | 0.498826 |
| AUC on the other 23,179 series | 0.901074 | 0.905932 |
| Historical misses recovered at 10% false positives | 11 | 22 |
| Historical misses recovered at 20% false positives | 42 | 87 |

The six local AUCs ranged from 0.877276 to 0.878079. The weighted primary AUC was 0.872514. The output folds reuse the same labelled series and the same underlying out-of-fold predictions. See the [validation record](validation.json), [paired audit](paired_audit.json) and [original signal findings](../white_noise_signal_findings.md).

Run `.venv-xpu/Scripts/python "new weights/wn_local_rank_family_agreement_candidate/replay.py"` from the project root to reproduce the [4,000 predictions](submission.csv) and verify their hash. The [training script](../../experiments/weight_space/white_expert_agreement.py) rebuilds the family interactions with parent `reversible_rank` and tag `family_agreement`.
