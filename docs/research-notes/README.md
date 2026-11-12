# Research notes

Notes written while the work was happening. They are kept as written, so paths inside them point at the original working folder (`experiments/`, `tune_out/`, `new weights/`), not at this repository. The matching scripts are in [`pipeline/legacy_cpu/experiments/`](../../pipeline/legacy_cpu/experiments/) and [`research/`](../../research/README.md).

| File | What it records |
|---|---|
| [steps.md](steps.md) | Exact log of the saved-weight experiment of 4 November 2026: output-layer edits and learned view weights, 0.85290 → 0.85339 |
| [weight_candidates.md](weight_candidates.md) | Index of the weight candidates that followed, up to the 0.87911 ensemble. Each candidate's own record is in [`research/weight_candidates/`](../../research/weight_candidates/) |
| [other_missed_breaks.md](other_missed_breaks.md) | Audit of the missed breaks: how many positive–negative pairs must change order to reach 0.880 |
| [white_noise_signal_findings.md](white_noise_signal_findings.md) | What signal exists in the 821 white-noise-like missed breaks, with [examples](white_noise_signal_examples.png) |
| [compact_cpu_method.html](compact_cpu_method.html) | Method page for the compact raw-CPU detector (0.8652), the first CPU-only baseline. Its scripts are in [`research/compact_cpu/`](../../research/compact_cpu/) |
