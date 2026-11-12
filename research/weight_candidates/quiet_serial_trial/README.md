# Trial weights for other missed serial breaks

The trial scores **0.87917396** on the primary local partition, up from **0.87911421**. It wins **8,604** additional positive–negative rankings out of the 127,554 needed and leaves **118,950** to reach 0.880. It improves six of seven output-weight partitions. On seed 29 it loses 0.000040. The partitions reuse the same 24,000 labelled series and the underlying models’ out-of-fold predictions. The [confirmed weights](../wn_local_rank_shared_residual_response_candidate/README.md) remain the reference.

| Primary local group | Change in correctly ordered pairs |
| --- | ---: |
| 821 historical white-noise-like breaks | +665 |
| 1,857 other breaks the reference misses at 20% false positives | +7,993 |
| 9,322 other breaks the reference already catches there | −54 |

Within those 1,857 other misses, the 1,419 outside the original historical list gained 17,375 rankings. The other 438 lost 9,382. At the same false-positive rate, the trial catches one additional break and keeps 88 of the 821. [Pair accounting](../../tune_out/weight_space/other_recovery_pair_accounting.csv)

The rule edits low scores only when the before segment shows serial dependence and all six recent detector families give low scores. It combines short-transition neighborhoods, block repeatability, localization of the change in time and an existing memory score. Training-fold controls set the correction scale. Historical miss IDs enter only the audit. [Training](../../experiments/weight_space/other_quiet_reordering.py) · [Fitted parameters](weights.json) · [Validation](validation.json)

The rule changes 724 of the 4,000 unlabelled test scores. The exported predictions replayed byte for byte. [Test predictions](submission.csv) · [Replay](replay.py)
