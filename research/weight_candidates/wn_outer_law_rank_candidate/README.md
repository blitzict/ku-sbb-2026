# White-noise break recovery

This candidate ranks each complete series by the evidence for a change in its noise distribution. The evidence compares 31 possible noise laws before and after the supplied boundary, after accounting for drift, periodic signals and short disturbances. Each fold estimates the prior probabilities of the possible changes using only its training labels. A small edit to the previous output weights gives more training weight to series with closely overlapping scores. The historical miss list is used only for evaluation.

| Primary held-out metric, split 7 | Frozen state tree | New weights |
| --- | ---: | ---: |
| Overall ROC-AUC | 0.870943 | 0.871460 |
| 821 historical white-noise-like misses against all 12,000 non-breaks | 0.460677 | 0.465515 |
| ROC-AUC on the other 23,179 series | 0.901074 | 0.901274 |
| Distribution-weighted ROC-AUC | 0.865530 | 0.866032 |

489 of the 821 moved above more non-breaks, and 332 moved below more non-breaks. The net gain was 47,666 positive-negative pair orderings. At a fixed 20% false-positive rate, the candidate identified 47 of the misses, up from 42. At 10% it identified 11 with either set of weights.

All three primary metrics improved on each of six repeated splits of the same labeled series. The archived upstream models contribute out-of-fold predictions from their original training folds. Repeated splits of the output layer reuse those predictions. Competition test labels are unavailable.

`validation.json` records the paired scores and operating points. Run `python replay.py` inside this directory to reproduce the 4,000 probabilities in sample-submission order and compare their file hash to the archived submission. The training and independent replay used the repository's Intel Python environment.
