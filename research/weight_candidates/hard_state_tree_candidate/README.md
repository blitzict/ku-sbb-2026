# KU-SBB hard-case weights

This candidate reached **0.870943** local out-of-fold ROC-AUC on the primary split, up from **0.870503**. The distribution-weighted score rose from **0.864961** to **0.865530**. A small correction compares two saved tree scores. One tree sees evidence from detrended signals. The other also sees the signal remaining after accounting for recent observations and changing volatility. Their disagreement helps decide how much to adjust the existing score.

| Meta split | Previous AUC | New AUC | Old missed breaks against all negatives, before → after |
| --- | ---: | ---: | ---: |
| 7 | 0.870503 | 0.870943 | 0.489965 → 0.491923 |
| 29 | 0.871070 | 0.871504 | 0.492159 → 0.494023 |
| 0 | 0.870623 | 0.871091 | 0.489483 → 0.491481 |
| 123 | 0.870283 | 0.870684 | 0.490039 → 0.491884 |
| 2024 | 0.870445 | 0.870860 | 0.489984 → 0.491796 |
| 11041 | 0.871061 | 0.871423 | 0.490827 → 0.492678 |

The 1,464 historically missed breaks and 928 historical false alarms came from ten older models. Their labels and IDs were used only to evaluate the predictions. On split 7 the edit corrected a net 63,348 positive-negative orderings, including 47,529 involving at least one member of these two groups. All six splits reuse the 24,000 labeled series. The competition test AUC is unknown.

## Predictions and replay

`submission.csv` contains the 4,000 test probabilities in sample-submission order. `weights.json` records the eight output weights, source hashes and scores. Run `python replay.py` in this directory to regenerate the CSV from the saved inputs and verify its SHA-256. The project replay reads the original parent probabilities and tree inputs.

```powershell
$winner = 'new weights/hard_state_tree_candidate'
& .venv-xpu/Scripts/python experiments/weight_space/hard_expert_trial.py --replay "$winner/weights.json" --output "$winner/submission_replay.csv"
```

The [pair budget](../../tune_out/weight_space/hard_pair_budget.json) measures the gap to 0.90. The [paired experiment](../../tune_out/weight_space/hard_state_tree_verified.json) gives results by family and split. The [coordination log](../../docs/COORDINATION.md) records Claude's analysis and the follow-up comparisons.
