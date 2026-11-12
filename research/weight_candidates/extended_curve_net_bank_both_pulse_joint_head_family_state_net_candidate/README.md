# KU-SBB weights at 0.87050 local ROC-AUC

`submission.csv` holds the 4,000 test predictions in the competition sample order. `weights.json` fixes the 183 output coefficients, the input hashes and the regularization strength of 35.

| Five-fold meta split | Ordinary AUC | Test-distribution-weighted AUC |
| --- | ---: | ---: |
| 7, primary | 0.870503 | 0.864961 |
| 29 | 0.871070 | 0.865514 |
| 0, same split as the base models | 0.870623 | 0.865032 |
| 123, after freezing | 0.870283 | 0.864687 |
| 2024, after freezing | 0.870445 | 0.864711 |
| 11041, after freezing | 0.871061 | 0.865451 |
| 777, independent Claude check | 0.87091 | 0.86527 |
| 31337, independent Claude check | 0.87056 | 0.86499 |

All splits reuse the competition's 24,000 labeled series. The base predictions are out of fold on split 0. Each meta split refits the readout on its training rows. The other seeds measure sensitivity to the meta split. Claude independently reran the locked method on seeds 777 and 31337 and recorded the results in the [coordination log](../../docs/COORDINATION.md). The leaderboard AUC is unknown.

## How the weights changed

The earlier 0.86715 package detected a recurring wave, measured changes in its phase and amplitude, and ran existing neural weights on the remaining noise. The next edit distinguished gradual drift from a boundary jump and let the same networks score the detrended series. The final readout combines those scores with statistical comparisons, scores after removing short pulses, and scores on innovations after accounting for serial dependence and changing volatility. The last addition raised primary AUC from 0.869764 to 0.870503.

## Replay

Run from this folder with NumPy, pandas, SciPy and threadpoolctl installed. This rebuilds `submission_replay.csv` from `replay_inputs.npz` and checks its SHA-256 against the frozen result.

```powershell
python replay.py
```

To recompute the output from the parent package and the project feature tables, run from the project root.

```powershell
$winner = 'new weights/extended_curve_net_bank_both_pulse_joint_head_family_state_net_candidate'
& .venv-xpu/Scripts/python experiments/weight_space/extend_readout.py --replay "$winner/weights.json" --output "$winner/submission_replay.csv"
```

`validation.json` records the additional split checks and file hashes. The SHA-256 of the submission is `882d40679fee53c697a72dafee58b7e90290f4531d0e0ced63149e347abf1320`.
