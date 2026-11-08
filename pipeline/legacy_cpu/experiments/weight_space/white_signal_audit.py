# CPU port; original experiments/weight_space/white_signal_audit.py, SHA256 e843bcbc823415822ec480871ec3f06f4ff61447639a8424c256cd93f2e622e1
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Run a matched six-split ablation for each proposed source of hidden structure."""
import contextlib as _cpu_context
import os as _cpu_os
from pathlib import Path as _CpuPath

def _cpu_workspace():
    value = _cpu_os.environ.get('KU_SBB_ROOT')
    if not value:
        raise RuntimeError('Set KU_SBB_ROOT to the fresh CPU rebuild workspace')
    root = _CpuPath(value).resolve()
    if not (root / 'CPU_REBUILD_WORKSPACE.json').is_file():
        raise RuntimeError('CPU ports require a marked rebuild workspace')
    return root
import argparse
import json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from white_local_rank_weights import centered
from white_nested import fit, audit
from white_checkpoint import SEEDS
from partition_evidence import OUT
import white_loop as wl

def terms(name):
    if name == 'forecast_break_directions':
        return (name, None)
    if name == 'shock_ringing':
        return (name, None)
    if name == 'forecast_weight_change':
        return (name, None)
    if name == 'ordered_path_vote':
        return (name, None)
    if name == 'phase_coupling':
        return (name, None)
    if name == 'time_arrow':
        return (name, None)
    if name == 'context_changes':
        return (name, [f'{ch}_{law}_{col}' for ch in ('pulse', 'state') for law in ('sign', 'size', 'tail', 'size_to_sign', 'sign_to_size') for col in ('ordering', 'replicate')])
    if name == 'memory_events':
        return (name, ['prob_on', 'prob_off', 'prob_duration', 'prob_strength', 'prob_both', 'stronger', 'longer_while_active', 'changed_while_active'])
    if name == 'shock_quantile_vote':
        return (name, None)
    if name == 'source_mask':
        return ('paired_source_votes', ['mask', 'mask_phase', 'mask_cluster'])
    if name == 'rank_replication':
        return ('paired_source_votes', ['ranks', 'ranks_phase', 'ranks_cluster'])
    if name == 'paired_source_votes':
        return (name, None)
    if name == 'run_age':
        return ('run_age', None)
    if name == 'seasonal_recurrence':
        return ('repeat_compact', None)
    if name == 'conditional_phase':
        return ('phase_distribution', [f'{s}_{what}_64' for s in ('innovation', 'amplitude', 'memory') for what in ('rotate', 'reshape', 'on', 'pre_repeats', 'post_repeats', 'second_half_same_clock')])
    if name == 'noisy_hidden_level':
        return ('latent_level_pulse', ['observed_step1', 'kick1', 'driver1', 'slope1', 'pulse1', 'restart_state', 'restart_new_scale', 'restart_new_process', 'slow_hidden_pre', 'hidden_post', 'walk_pre'])
    if name == 'noise_modes':
        return ('component_moves_pulse', None)
    if name == 'state_persistence':
        return ('markov_copula', [f'state_f{k}_{q}' for k in range(6) for q in ('memory', 'on', 'off')])
    if name == 'coupled_scale':
        return ('transport_state', ['v1', 'v4', 'v7', 'v10', 'v13', 'v15', 'v16', 'v17'])
    if name == 'leverage':
        return ('asymmetric_response', None)
    if name == 'burst_spacing':
        return ('shock_spacing', None)
    if name == 'noise_polarity':
        return ('noise_polarity', None)
    if name == 'waveform_shape':
        return ('waveform_shape', None)
    if name == 'variance_ramp':
        return ('variance_ramp', None)
    if name == 'segment_population':
        return ('segment_population', None)
    if name == 'two_mode_motion':
        return ('mode_residual', None)
    if name == 'heavy_tail_motion':
        return ('tail_channels', None)
    if name == 'mode_sequence':
        return ('mode_residual_channel_effect', None)
    if name == 'tail_sequence':
        return ('tail_channels_channel_effect', None)
    if name == 'support_edges':
        return ('tail_edges', None)
    if name == 'cold_start':
        return ('cold_start_effect', None)
    if name == 'mode_switching':
        return ('mode_switching', None)
    if name == 'mode_encoders':
        return ('mode_encoders', None)
    if name == 'volatility_channels':
        return ('volatility_channel_effect', None)
    if name == 'ordinal_motifs':
        return ('ordinal_motifs', None)
    if name == 'intrinsic_time':
        return ('intrinsic_time', None)
    if name == 'wandering_frequency':
        return ('wandering_frequency', None)
    if name == 'quantile_alignment':
        return ('quantile_alignment', None)
    if name == 'smooth_path':
        return ('smooth_path', None)
    if name == 'wavelet_modulation':
        return ('wavelet_modulation', None)
    if name in ('volatility_restart', 'latent_duration'):
        return (name, None)
    if name == 'stable_factors':
        return (name, ['stable_increment', 'combined'])
    if name == 'stable_control':
        return ('stable_factors', ['physical'])
    if name == 'shock_echo':
        return (name, None)
    if name == 'cube_components':
        return (name, ['mean_times_variance', 'central_third', 'max_post_shock', 'after_removing_max', 'abs_after_removing_max', 'max_shock_near_cut', 'clipped_central_third', 'repeated_central_third', 'up_tail_frequency', 'down_tail_frequency'])
    if name == 'wandering_slope':
        return (name, None)
    if name == 'value_bands':
        return (name, [f'{v}_{p}_replicated_{term}' for v in ('raw', 'shape') for p in ('quantile', 'valley') for term in ('any', 'narrow', 'shoulders', 'direction')])
    if name == 'conditional_noise_law':
        return (name, [f'{ch}_{term}' for ch in ('raw', 'shape', 'whitened') for term in ('shared', 'extra', 'by_state', 'opposed', 'tail_separate')])
    if name == 'motif_dictionary':
        return (name, [f'{ch}_d{d}_{kind}_specific_repeat' for ch in ('pulse', 'innovation', 'size', 'two_mode') for d in (1, 2, 4, 8) for kind in ('counts', 'response')])
    if name == 'named_signals':
        return (name, ['clock_given_wave', 'volatility_without_wave', 'decay_vote', 'cluster_scale_vote', 'noise_density_vote', 'joint'])
    if name == 'weak_phase':
        return (name, [f'{ch}_{s}' for ch in ('raw', 'pulse') for s in ('continuation', 'pre', 'post', 'replicated_free_change', 'replicated_restart', 'replicated_inversion', 'replicated_phase_change')])
    if name == 'local_transport_score':
        return (name, [f'{ch}_{m}_{s}' for ch in ('pulse', 'state') for m in ('information', 'evidence') for s in ('shift', 'scale', 'two_modes', 'lower_mode', 'upper_mode', 'lower_tail', 'upper_tail')])
    if name.startswith('tree_channels_'):
        return (name, None)
    if name == 'second_clock':
        return ('second_clock', ['pre_repeat', 'post_repeat', 'carry', 'reverse_carry', 'two_clocks_same', 'two_clocks_split', 'two_clocks_new', 'two_clocks_lost'])
    if name == 'conditional_drift':
        return ('conditional_drift', [f'{ch}_{key}' for ch in ('pulse', 'state') for key in ('pre_repeat', 'post_repeat', 'pre_to_post', 'pre_specific', 'post_specific', 'change', 'change_repeat')])
    if name in ('two_peaks', 'fine_steps'):
        return ('value_spacing', [name + '_' + s for s in ('repeats', 'magnitude', 'abs_magnitude', 'change', 'strength')])
    if name.startswith('reliable_'):
        columns = {'reliable_modes': ['mode_within_mode', 'mode_occupancy', 'mode_agree', 'mode_within_given_motion', 'mode_visits_given_counts'], 'reliable_tails': ['tail_core', 'tail_bursts', 'tail_core_given_shift'], 'reliable_edges': ['edge_up_move', 'edge_up_run', 'edge_down_move', 'edge_down_run']}
        return ('reliable_signals', columns[name])
    if name == 'view_consistency':
        return ('paired_view_v1', None)
    raise KeyError(name)

def run(tag, names):
    q, ids, tids, y, wt, wn, miss = wl.load()
    folder = OUT / (('frozen_' if tag.startswith('local_rank_') else 'frozen_stage_') + tag)

    def tab(t, cols):
        return pd.read_parquet(OUT / f'{t}_train.parquet', columns=cols).loc[ids]
    white = tab('innov_v2', ['pre_white']).to_numpy()[:, 0]
    walk = tab('innov_v2', ['pre_walk']).to_numpy()[:, 0]
    allresults = {}
    for name in names:
        h = np.arcsinh(tab(*terms(name)).to_numpy().clip(-24, 55))
        h = np.column_stack([h, 0.6 * h * white[:, None], 0.6 * h * walk[:, None]])
        scores = {}
        details = {}
        for s in SEEDS:
            old = np.empty(len(y))
            new = np.empty(len(y))
            for f in range(5):
                k = f's{s}_f{f}_'
                tr, va = [q[k + c] for c in ('tr', 'va')]
                with np.load(folder / f's{s}_f{f}.npz') as a:
                    assert np.array_equal(a['train_ids'], ids[tr]) and np.array_equal(a['validation_ids'], ids[va])
                    zr, zv = (a['oof_train'], a['valid'])
                xx, means = centered(h[tr], zr)
                vv, _ = centered(h[va], zv, means)
                w = fit(xx, zr, y[tr], True)
                old[va] = zv
                new[va] = zv + vv @ w
            a = wl.score(y, old, wt, wn, miss)
            b = wl.score(y, new, wt, wn, miss)
            scores[s] = {k: b[k] - a[k] for k in ('auc', 'remaining', 'weighted', 'target_vs_neg')}
            details[s] = audit(y, miss, old, new)
        allresults[name] = dict(delta=scores, threshold=details[7], positive_splits=sum((v['auc'] > 0 for v in scores.values())))
        print(name, allresults[name]['positive_splits'], {k: round(v, 7) for k, v in scores[7].items()}, details[7]['candidate'], flush=True)
    (OUT / ('signal_ablation_' + tag + '_' + '_'.join(names) + '.json')).write_text(json.dumps(allresults, indent=2))
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('parent')
    p.add_argument('names', nargs='+')
    a = p.parse_args()
    with threadpool_limits(limits=1):
        run(a.parent, a.names)
