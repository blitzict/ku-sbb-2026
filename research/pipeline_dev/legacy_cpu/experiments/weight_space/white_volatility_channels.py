# CPU port; original experiments/weight_space/white_volatility_channels.py, SHA256 0859591483262551b832604ef21a3dd87c3e84a6f3729200d9589bb49f56e17d
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Rescore noise after removing predictable changes in its local amplitude."""
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
from scipy.signal import lfilter
from threadpoolctl import threadpool_limits
from partition_evidence import OUT
import periodic_replay

def make_streams(part):
    path = OUT / f'volatility_channels_streams_{part}.npz'
    if path.exists():
        return
    with np.load(OUT / f'pulse_streams_{part}.npz') as a:
        ids, values, lengths, boundaries = (a[k] for k in ('ids', 'separate', 'n', 'b'))
    ar = pd.read_parquet(OUT / f'state_{part}.parquet').loc[ids, ['ar1', 'ar2']].to_numpy()
    vol = pd.read_parquet(OUT / f'volatility_law_{part}.parquet').loc[ids]
    scales = np.zeros_like(values)
    corrected = np.zeros_like(values)
    for i, (n, b) in enumerate(zip(lengths, boundaries)):
        x = values[i, :n].astype(float)
        predictor = ar[i, 0] * np.r_[x[0], x[:-1]] + ar[i, 1] * np.r_[x[0], x[0], x[:-2]]
        for side, sl in (('pre', slice(0, b)), ('post', slice(b, n))):
            e = x[sl] - predictor[sl]
            e -= np.median(e)
            variance = max(np.median(abs(e)) ** 2 * 2.198, np.mean(e ** 2) * 0.1, 1e-05)
            history = np.r_[variance, np.minimum(e[:-1] ** 2, 30 * variance)]
            posterior = vol.iloc[i][[side + '_decay' + str(k) for k in range(5)]].to_numpy()
            h = np.full(len(e), variance * max(0.0, 1.0 - posterior.sum()))
            for prob, beta in zip(posterior, (0.0, 0.35, 0.68, 0.86, 0.95)):
                avg = lfilter([1 - beta], [1, -beta], history, zi=[beta * variance])[0]
                h += prob * (0.38 * variance + 0.62 * avg)
            corrected[i, sl] = x[sl] + e * (np.sqrt(variance / h) - 1)
            scales[i, sl] = 0.5 * np.log(h)
    assert np.isfinite(corrected).all() and np.isfinite(scales).all()
    np.savez_compressed(path, ids=ids, n=lengths, b=boundaries, common=corrected, separate=scales)

def run():
    for part in ('train', 'test'):
        make_streams(part)
    periodic_replay.run('volatility_channels')
    for part in ('train', 'test'):
        seq = pd.read_parquet(OUT / f'volatility_channels_seq_{part}.parquet')
        pulse = pd.read_parquet(OUT / f'pulse_seq_{part}.parquet').loc[seq.index]
        v = pd.read_parquet(OUT / f'volatility_law_{part}.parquet').loc[seq.index]
        reference = pulse.filter(regex='^separate_').mean(1)
        standardized = seq.filter(regex='^common_').mean(1) - reference
        envelope = seq.filter(regex='^separate_').mean(1) - reference
        g = ((np.maximum(v.pre_cluster, v.post_cluster) - 0.5) / 0.45).clip(0, 1)
        table = pd.DataFrame(dict(standardized=g * standardized, envelope=g * envelope, onset=(v.post_cluster - v.pre_cluster) * g * envelope, persistence=np.minimum(v.pre_cluster, v.post_cluster) * g * standardized), index=seq.index)
        assert np.isfinite(table).all().all()
        table.to_parquet(OUT / f'volatility_channel_effect_{part}.parquet')
        print(part, 'strong on either side', int((g > 0.8).sum()), flush=True)
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        run()
