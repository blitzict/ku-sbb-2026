# CPU port; original experiments/weight_space/white_reliable_signals.py, SHA256 7ff53c265a26846e940dfa269a664d7f7350b5dc80a12bf8077fa58650b1bded
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Give a structural test weight only where its underlying noise regime is measurable."""
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
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from partition_evidence import OUT
import white_loop as wl

def run():
    _, ids, tids, y, wt, wn, miss = wl.load()
    for part, ix in (('train', ids), ('test', tids)):
        get = lambda name: pd.read_parquet(OUT / f'{name}_{part}.parquet').loc[ix]
        s = get('mode_residual')
        h = get('mode_residual_channel_effect')
        edge = get('tail_edges')
        tail = get('tail_channels')
        a = get('tail_channels_channel_effect')
        gate = ((s.gate.to_numpy() - 0.5) / 0.45).clip(0, 1)
        gtail = ((tail.gate.to_numpy() - 0.65) / 0.32).clip(0, 1)
        output = {}
        d, e = (h.v0.to_numpy(), h.v1.to_numpy())
        for name, v in (('within_mode', d), ('occupancy', e), ('agree', np.sign(d + e) * np.minimum(abs(d), abs(e))), ('within_given_motion', d * np.tanh(abs(s.residual_change) / 2)), ('visits_given_counts', e * np.tanh(abs(s.occupancy_change) / 2))):
            output['mode_' + name] = gate * v
        for name, v in (('core', a.v0), ('bursts', a.v1), ('core_given_shift', a.v0 * np.tanh(abs(tail.core_location) / 2))):
            output['tail_' + name] = gtail * v
        for side in ('up', 'down'):
            band = edge.filter(like='normalized_' + side + '_').copy()
            band.columns = band.columns.str.removeprefix('normalized_' + side + '_')
            g = ((np.maximum(band.pre_bounded, band.post_bounded) - 0.55) / 0.4).clip(0, 1)
            output['edge_' + side + '_move'] = g * np.arcsinh(band.bounded_move.clip(-6, 15))
            output['edge_' + side + '_run'] = g * np.arcsinh(band.tail_run.clip(-6, 15))
        df = pd.DataFrame(output, index=ix).rename_axis('id')
        assert np.isfinite(df).all().all()
        df.to_parquet(OUT / f'reliable_signals_{part}.parquet')
        if part == 'train':
            p = get('pulse_seq').filter(regex='^separate_').mean(1).to_numpy()
            signal = s.gate.to_numpy() > 0.8
            for name, rows in (('all strong', signal), ('historical and matched negatives', signal & (miss | (y == 0)))):
                print(name, rows.sum(), sum(miss & rows), [round(roc_auc_score(y[rows], v[rows]), 6) for v in (p, p + d, p + e)], flush=True)
if __name__ == '__main__':
    run()
