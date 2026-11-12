# KU-SBB weights for the 0.867 target

`submission.csv` is the candidate for the 4,000 official test series. The primary local ROC-AUC is **0.867148**, up from **0.853385** for the original packaged ensemble. The test-distribution-weighted AUC is **0.860589**. The test labels are unavailable.

| Meta split seed | ROC-AUC | Role |
| --- | ---: | --- |
| 7 | 0.867148 | Primary comparison |
| 29 | 0.867504 | Stability check |
| 0 | 0.867387 | Canonical base-model folds |
| 123 | 0.867152 | Locked-method check |
| 2024 | 0.867154 | Locked-method check |
| 61543 | 0.867318 | Locked-method check |

All six estimates reuse the same 24,000 labeled series. Each refit of the ensemble output weights uses only the training portion of that split. Base members use their archived out-of-fold predictions. The additional seeds check sensitivity to the split. Hidden-test performance is unknown.

## What changed

The model estimates whether the signal contains a stable repeating period. If so, it estimates how the level, amplitude, phase and noise change across the supplied boundary. It also runs the original sequence checkpoints on the observations left after removing the fitted oscillation. A separate correction uses the same checkpoints on estimated innovations of persistent series. The final weights combine where these predictions agree and disagree with the earlier ensemble.

## Replay

From the project root on Windows, run the following command. The candidate folder contains the output weights and inference inputs required by this replay for the official test set.

```powershell
& .venv-xpu/Scripts/python "new weights/mechanism_phase_ar_candidate/replay.py"
```

The project-level replay reconstructs the output features from the saved signal measurements and neural predictions.

```powershell
& .venv-xpu/Scripts/python experiments/weight_space/mechanism_edit.py --replay "new weights/mechanism_phase_ar_candidate/weights.json" --output "new weights/mechanism_phase_ar_candidate/submission_replay.csv"
```

Both paths reproduce SHA-256 `5fa933f305252be2e082de1bcd08ea7790aea52544f62f4c8cd9b78e3914a5c5` in the project environment. The replay checks that every test identifier occurs once in the sample submission order.

## Validation and source

`validation.json` records ordinary and weighted AUC, split roles and artifact hashes. The additional-split evaluation can be reproduced with `experiments/weight_space/verify_target.py`. The feature-generation steps are in `harmonic_change.py`, `periodic_components.py`, `phase_likelihood.py` and `innovation_transform.py` under `experiments/weight_space`. The two saved-checkpoint passes use `periodic_replay.py` with the default prefix and with `--prefix innov_v2`. The preceding weights are in `expert_fusion_candidate`, `harmonic_candidate` and `seasonal_seq_candidate`.
