# Features used by the 0.879 model

This file lists every feature family the 0.879 ensemble needs. The source of truth is the recipe in `pipeline/full_cpu_stages.json`, which has 473 steps; this list was taken from it. Each family is computed from the raw `train_series.parquet` and `test_series.parquet` files only. Column counts come from the CPU rebuild tables and exclude the `id` column.

Every series is split at its given boundary into a "before" part and an "after" part. Almost every feature compares the two parts.

## 1. Statistical blocks (tree and MLP inputs)

| Block | Script | Columns | What it measures |
|---|---|---:|---|
| clean | `pipeline/features_v4.py` (foundation) | 977 | Moments, quantiles, ranks, KS/AD/MMD distribution tests, autocorrelation, spectral and change statistics on before vs after |
| grid | `experiments/v5/features_grid.py` | 492 | The same comparisons on fixed-size grid windows around the boundary |
| grid2 | `experiments/v5/features_grid2.py` | 504 | A second window grid (other window sizes) |
| dd | `experiments/v6/features_dd.py` | 198 | Statistics of the differenced series (increments) |
| lr / lr5 | `experiments/v6/features_lr.py` | 46 / 46 | Likelihood-ratio tests for a mean/variance change at the boundary |
| lrz / lrz6 | `experiments/v6/features_lrz.py` | 50 / 50 | Standardised (z-scored) likelihood ratios |
| gh | `experiments/v6/features_garch.py` | 30 | GARCH-style volatility fit before vs after |
| lm | `experiments/v6/features_lm.py` | 21 | Linear-model (trend) fits before vs after |
| bf | `experiments/v6/features_bf.py` | 26 | Bayes-factor change tests |
| ps | `experiments/v6/features_ps.py` | 22 | Power-spectrum comparisons |
| sn | `experiments/v6/features_sn.py` | 17 | Self-normalised change statistics |
| harm | `pipeline_helpers/mechanism.py harmonic` | 36 | Harmonic (periodic) fit changes |
| generator | `pipeline_helpers/generator_features.py` | segment table | Generator-style measurements of each segment, then before/after deltas |

The dense MLP block is clean + grid + grid2 + dd (977 + 492 + 504 + 198 = 2,171 columns). The tree members use the selected lists shipped in `tune_out/selected_*.csv`, which have 170, 191 or 206 columns, or the full 2,247-column list.

## 2. Transformed views (augmentation and test-time averaging)

`experiments/v6/make_variants.py` and `make_crops.py` build 7 extra views of every series:

- rev (time reversed)
- flip (sign flipped)
- revflip
- c1 to c4: random crops around the boundary, with seeds 11, 23, 37 and 41, keeping 60–90% of each side and at least 80 points

Halves (`make_halves.py`) and boundary-shift negatives (`shift_augmentation.py`) are extra training rows only.

`stream_numpy_features.py`, `harm_var.py` and the `var:` modes of lr/lrz/gh/lm recompute the clean/grid/grid2/dd/harm/lr/lrz/gh/lm blocks on each view. The view models (VARC4*, MLPHV*) are trained on all views and average their predictions over views at test time.

## 3. Mechanism / physical features (the "white" and law tables)

These are what lift the model from about 0.85 to 0.875 and above. Built by `pipeline_helpers/mechanism.py`, `late_features.py`, `component_moves_cpu.py` and `white_model_features.py`.

| Table | Columns | Idea |
|---|---:|---|
| periodic | 37 | Periodic components and phase before vs after |
| innov_v2 / alt_innov | 60 | Innovation (one-step prediction error) statistics |
| trend | 21 | Trend and level change |
| curve | 20 | Curvature likelihoods |
| ar_law | 21 | Autoregressive law likelihood change |
| phase_law | 18 | Phase likelihood change |
| mixture_law | 15 | Mixture-model law change |
| partition | 48 | Partition evidence for the split point |
| pulse / state | 12 / 21 | Pulse filter and latent-state filter outputs |
| volatility_law | 50 | Volatility regime change |
| jump_law_{pulse,state,nullstate,prestate,smoothvol} | 149 each | Jump-law likelihoods under 5 different nulls |
| wnlaw_raw / wnlaw_physical | npz | White-noise law values; turned into fold-wise priors (`law_outer*`) |
| noise_tests | 84 | Noise/whiteness tests |
| bounds | 26 | Bounded-range evidence |
| mode_residual | 13 | Residual after removing the dominant mode |
| walk_jump | 44 | Random walk vs jump evidence |
| quantile_alignment | 42 | Quantile alignment before vs after |
| latent_level_{pulse,raw} | 33 each | Latent level shifts |
| native_law_{state,pulse} | 278 each | Native law tables |
| tail_edges / tail_channels | 44 / 12 | Heavy-tail changes |
| component_moves_{state,pulse} | 20 each | Mixture component moves |
| boundary_law_{pulse,state} | 102 each | Laws near the boundary |
| hetero_jump | 114 | Heteroscedastic jumps |
| markov_copula | 72 | Markov copula dependence change |
| memory_law1 / memory_law2 | 42 / 39 | Long-memory changes |
| filtered_step | 32 | Filtered step detector |
| transition_odds / transport_state | 8 / 18 | Learned population transition priors and transport |
| vol_smoothing | 48 | Smoothed volatility |
| early tables (trend_clock, state_tests, shape_bank, shape_counts, deconv_pulse, conditional_scores, trend_population) | various | Inputs to the early heads |

The same mechanism tables are rebuilt on the c1/c2 crop views (`white_view_pipeline.py`, `white_view_states.py`, `white_view_extra.py`) for the crop specialists.

## 4. Where the features go

1. **Base members (51):**
   - XGB/LGB/Cat on the statistical blocks (`train_cv.py`, 44 runs).
   - View models (`train_var.py`).
   - MLPs (`nn_var.py`, `nn_xpu.py`).
   - Sequence CNN on the raw padded series (`nn_seq.py`).
   - Distilled trees.
2. **Mechanism specialists:**
   - `nested_tree.py`, `nested_crop.py`, `nested_rank.py`, `wnlaw_nested*.py`.
   - XGB on mechanism + law-prior features, fit fold by fold.
   - MECH3C12V reaches about 0.874 on its own.
3. **Weight-space heads:**
   - expert_fusion, residual/seasonal/mechanism/trend heads, the extended readout and the hard-state tree.
   - Then the `white_*` scripts: opinion panels, rank boosts, crop laws, pair trees and the final local-rank shared residual response.
   - The last step is `white_head_shared_response`, which writes `new weights/wn_local_rank_shared_residual_response_candidate/submission.csv`.

The historical out-of-fold AUC goes up as more of these are added:

| What is added | AUC |
|---|---:|
| grid XGB | 0.840 |
| + ns3 selected blocks | 0.845 |
| + 7 views | 0.850 |
| + harmonic | 0.859 |
| + mechanism nested crop | 0.874–0.875 |
| full recipe | 0.879 |
