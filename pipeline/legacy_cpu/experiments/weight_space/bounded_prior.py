# CPU port; original experiments/weight_space/bounded_prior.py, SHA256 e7650da795b12ca7096f8d0ed59cb7665c78cc8fda8c634889df0e110b98e370
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Finite-support prior corrections in physical units; use label-free segment fields only."""
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
from scipy.special import log_ndtr
import expert_fusion as ef

def log_mass(x, se, lo, hi):
    """Log probability of an interval under a Gaussian with stable tails."""
    se = np.maximum(se, 1e-06)
    a, b = ((lo - x) / se, (hi - x) / se)
    left = a + b <= 0
    big = np.where(left, log_ndtr(b), log_ndtr(-a))
    small = np.where(left, log_ndtr(a), log_ndtr(-b))
    return np.maximum(-80, big + np.log(-np.expm1(np.minimum(small - big, -1e-15))))

def contrasts(x, z, s, t, lo, hi):
    precision = s ** (-2) + t ** (-2)
    shared = (x / s ** 2 + z / t ** 2) / precision
    m0 = log_mass(shared, precision ** (-0.5), lo, hi)
    ma = log_mass(x, s, lo, hi)
    mb = log_mass(z, t, lo, hi)
    return np.column_stack([ma, mb, m0, ma - m0, ma + mb - m0, np.maximum(ma - mb, 0), abs(x), abs(z), abs(z) - abs(x)])

def main():
    fields = ['id', 'part', 'n', 'phi1_ar1', 'persist4', 'innov_sd4', 'sd', 'mean', 'kurt', 'sqacf1']
    data = pd.read_parquet(ef.O / 'generator_segments.parquet', columns=fields)
    for part in ['train', 'test']:
        ids = pd.read_parquet(ef.ROOT / f'clean/features_{part}.parquet', columns=[]).index
        pre, post = [data.loc[data.part == side].set_index('id').loc[ids] for side in ['pre', 'post']]
        assert pre.index.equals(post.index) and len(pre) == len(ids)

        def measurement(frame):
            n = frame.n.to_numpy()
            sigma = frame.innov_sd4.to_numpy().clip(0.02)
            ar = np.clip(frame.persist4, -1.5, 0.99)
            mean_se = sigma / (1 - ar) / np.sqrt(n)
            mean_se = np.sqrt(mean_se ** 2 + 0.07 ** 2)
            sq = np.clip(frame.sqacf1, 0, 0.8)
            tail = np.clip(frame['kurt'] + 2, 0.4, 36)
            scale_se = np.sqrt(tail / n * (1 + sq) / (1 - sq) / 4 + 0.035 ** 2)
            return (frame['mean'].to_numpy(), np.log(sigma), mean_se, np.asarray(scale_se))
        x, l, s, u = measurement(pre)
        z, r, t, v = measurement(post)
        data_blocks = {}

        def add(prefix, values):
            for j, col in enumerate(values.T):
                data_blocks[f'{prefix}_{j}'] = col
        add('location', contrasts(x, z, s, t, -2, 2))
        add('logscale', contrasts(l, r, u, v, -np.log(2), np.log(2)))
        for col in ['phi1_ar1', 'persist4', 'sqacf1', 'kurt']:
            data_blocks[f'{col}_pre'] = pre[col].values
            data_blocks[f'{col}_post'] = post[col].values
        tab = pd.DataFrame(data_blocks, index=ids)
        assert np.isfinite(tab).all().all()
        tab.to_parquet(ef.OUT / f'bounds_{part}.parquet')
if __name__ == '__main__':
    main()
