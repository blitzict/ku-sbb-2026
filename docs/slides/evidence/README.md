# Numbers behind the slides

Every figure on the results, validation, error-analysis and limitations slides comes from these files. Nothing was retrained: the held-out predictions are rebuilt from try4's saved fold models.

| File | What it holds |
|---|---|
| `analyze.py` → `try4_oof_stats.json` | try4's out-of-fold predictions rebuilt from `try4/model` (AUC 0.87234, recorded 0.87235): per-round AUC, decile break rates, feature gain shares, single-feature AUCs |
| `explain.py` → `explain.json` | try3's specialist MECH3C12V alone (bag 0.8748), the exact columns try4 picked, and one held-out break explained with LightGBM `pred_contrib` |
| `errors.py` → `errors.json` | error analysis: held-out AUC by series length and point position, breaks caught by how visible they are, missed breaks and false alarms (run after `analyze.py`) |
| `naive.py` → `naive.json` | simple baselines on the raw series: mean shift and spread ratio (AUC 0.62 each) |
| `kaggle_timing.json`, `timing_summary.json` | per-step timing of the 2-core Kaggle run (1,306.7 s) |

Run from the folder that holds `data/` and `tune_out/` (`python ku-sbb-2026/docs/slides/evidence/analyze.py <out_dir>`). The competition data is not included.
