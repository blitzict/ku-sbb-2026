# CPU port; original experiments/weight_space/white_channel_replay.py, SHA256 6e7c421b8215e6cdcad8da6a33cf3ae2669ea6c2cf80dccb0e8515660ec15d11
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Apply the original verified sequence weights to the newly separated noise channels."""
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
import numpy as np
import pandas as pd
from scipy.special import expit
from threadpoolctl import threadpool_limits
from partition_evidence import OUT
import periodic_replay

def run(tag):
    periodic_replay.run(tag)
    scores = []
    indices = []
    for part in ('train', 'test'):
        seq = pd.read_parquet(OUT / f'{tag}_seq_{part}.parquet')
        pulse = pd.read_parquet(OUT / f'pulse_seq_{part}.parquet').loc[seq.index]
        gate = pd.read_parquet(OUT / f'{tag}_{part}.parquet').loc[seq.index, 'gate'].to_numpy()
        old = pulse.filter(regex='^separate_').to_numpy()
        a = seq.filter(regex='^common_').to_numpy()
        b = seq.filter(regex='^separate_').to_numpy()
        assert old.shape == a.shape == b.shape and a.shape[1] > 0
        d, e = ((a - old).clip(-8, 8), (b - old).clip(-8, 8))
        means = [d.mean(1), e.mean(1), abs(d).mean(1), abs(e).mean(1), (a - b).mean(1)]
        features = np.column_stack([*means, *(gate * v for v in means)])
        tab = pd.DataFrame(features, index=seq.index, columns=[f'v{i}' for i in range(features.shape[1])])
        assert np.isfinite(features).all()
        tab.to_parquet(OUT / f'{tag}_channel_effect_{part}.parquet')
        scores.append(np.column_stack([expit(old.mean(1) + gate * delta) for delta in means[:2]]))
        indices.append(seq.index.to_numpy(dtype=str))
    for k, label in enumerate(('core', 'other')):
        np.savez_compressed(OUT / f'law_{tag}_{label}.npz', ids=indices[0], tids=indices[1], oof=scores[0][:, k], test=scores[1][:, k])
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('tag', choices=('mode_residual', 'tail_channels'))
    with threadpool_limits(limits=2):
        run(p.parse_args().tag)
