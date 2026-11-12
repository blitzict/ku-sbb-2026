# White-noise break recovery

The output weights compare observed before-and-after changes with 12 plausible noise laws under different assumptions about which physical parameter changed. They use the original corrected level, the innovation left after accounting for recent levels and volatility, and a separate calculation for a permanent jump in a random walk. The final edit combines that evidence with an adapted sequence network, a tree trained on the full likelihoods and Claude’s separately trained tree. Priors used by each tree come only from its outer training rows.

| Primary split 7 | Frozen baseline | Candidate |
| --- | ---: | ---: |
| Ordinary ROC-AUC | 0.870943 | 0.874050 |
| 821 historical white-noise-like breaks against all 12,000 non-breaks | 0.460677 | 0.486010 |
| ROC-AUC on the other 23,179 series | 0.901074 | 0.902548 |
| Distribution-weighted ROC-AUC | 0.865530 | 0.868699 |
| Historical misses found at 5% false-positive rate | 8 | 8 |
| At 10% false-positive rate | 11 | 17 |
| At 20% false-positive rate | 42 | 74 |
| At 30% false-positive rate | 98 | 150 |

Relative to the baseline, 522 of the 821 moved above more non-breaks, 297 moved below more, and two kept the same rank. The resulting net gain was 249,581 positive-negative pair orderings involving these 821. Historical membership was used only to audit predictions. All six repeated splits improved overall, on the 821 and on the remaining series. Their overall AUCs ranged from 0.873350 to 0.874315.

`submission.csv` contains 4,000 test probabilities in sample-submission order. Run `python replay.py` here to reproduce the CSV and check its SHA-256. `validation.json` contains the scores and fixed false-positive-rate counts on each split. `weights.json` records the edit and its input hashes. The output-layer splits reuse the same 24,000 labeled series and the archived base-model out-of-fold predictions. The competition test AUC is unknown.
