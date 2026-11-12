# Research archive

Everything from the research phase that is not part of the released pipeline. It is kept as a record: nothing here is needed to run try1 to try4, and the scripts expect the original working folder layout (`experiments/`, `tune_out/`, `weights/`, `data/`), so most will not run from this repository as they are.

The scripts the released models do use are in [`pipeline/legacy_cpu/experiments/`](../pipeline/legacy_cpu/experiments/). The story that ties it together is [`docs/JOURNEY.md`](../docs/JOURNEY.md).

| Path | What it holds |
|---|---|
| [results/experiment_ledger.csv](results/experiment_ledger.csv) | One row per scored run (about 620): date, record name, cross-validation AUC, test-weighted AUC and the run's own note |
| `results/result_records.zip` | The raw result records the ledger was built from (1,290 JSON files) |
| `pipeline_dev/` | Pipeline files left out of the release: experiment scripts that were tried and dropped (`legacy_cpu/experiments/`), stage builders (`prepare_*.py`), smoke tests and checks (`smoke_*.py`, `check_*.py`), intermediate stage lists (`*_stages.json`) |
| `weight_candidates/` | The record of each weight-edit candidate from 4 November: `README.md`, `validation.json`, `weights.json` and its replay script. Index in [`docs/research-notes/weight_candidates.md`](../docs/research-notes/weight_candidates.md) |
| `models/ensemble_088077/` | The preserved 0.88077 ensemble (the parent of the 0.88078 submission): weights, validation record, replay inputs and its submission file |
| `submissions/` | Stack recipes for the 0.8529 and 0.8534 ensembles |
| `compact_cpu/` | The compact raw-CPU detector (0.8652), the first CPU-only baseline |
| `queue_scripts/` | The shell queues that launched the GPU training runs |
| `dev_scripts/` | One-off helper scripts |

Not archived: the intermediate feature tables, out-of-fold arrays and model checkpoints of the research phase (about 100 GB). The models needed to reproduce the submitted results are in `try1/` to `try4/`; the rest can be regenerated with `pipeline/train_gpu.py` or `train_087910_cpu.py`.
