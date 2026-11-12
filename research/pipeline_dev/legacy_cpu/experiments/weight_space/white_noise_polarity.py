# CPU port; original experiments/weight_space/white_noise_polarity.py, SHA256 e6b3b894d009e82f634e12f3aff71ececafa9ab41808b98053623e4cde09bffc
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Test if the direction of a noise excursion changes while its size stays similar."""
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
from scipy.special import betaln, logsumexp
from threadpoolctl import threadpool_limits
from partition_evidence import OUT
BIN = (3, 6, 12)
CONCENTRATION = np.array([3.0, 9.0, 30.0])[:, None]

def score(counts):

    def marginal(x):
        down, up = x.T
        return (betaln(up + CONCENTRATION, down + CONCENTRATION) - betaln(CONCENTRATION, CONCENTRATION)).sum(1)

    def mix(x):
        return logsumexp(x) - np.log(len(x))
    a, b, c, d = counts
    left = a + b
    right = c + d
    leftbay = marginal(left)
    rightbay = marginal(right)
    joint = marginal(left + right)
    null = mix(joint)
    symleft = -left.sum() * np.log(2)
    symright = -right.sum() * np.log(2)
    return [mix(marginal(left + right[:, ::-1])) - null, mix(leftbay + rightbay) - null, mix(rightbay) + symleft - null, mix(leftbay) + symright - null, mix(leftbay) - symleft, mix(rightbay) - symright, mix(leftbay) - mix(marginal(a)) + b.sum() * np.log(2)]

def contrasts(x, j):
    a, b = (x[:j], x[j:])
    z = []
    for v in (a, b):
        m = np.median(v)
        d = v - m
        scale = max(np.median(abs(d)) * 1.4826, 0.18 * d.std(), 1e-05)
        z.append(d / scale)
    z = np.concatenate(z)
    mag = abs(z)
    quarter = np.repeat([0, 1, 2, 3], [j // 2, j - j // 2, len(b) // 2, len(b) - len(b) // 2])
    out = []
    for nb in BIN:
        cut = np.quantile(mag, np.arange(1, nb) / nb)
        idx = np.searchsorted(cut, mag)
        cnt = np.bincount(2 * (quarter * nb + idx) + (z > 0), minlength=4 * nb * 2).reshape(4, nb, 2)
        out.extend(score(cnt))
    return out

def extract(part):
    dst = OUT / f'noise_polarity_{part}.parquet'
    if dst.exists():
        return
    with np.load(OUT / f'hetero_path_{part}.npz') as a:
        ids, u, n, b, h = (a[k] for k in ('ids', 'u', 'n', 'b', 'common'))
    records = []
    for row, scale, end, j in zip(u, h, n, b):
        e = row[:end]
        v = e / scale[:end].clip(0.01)
        dx = np.r_[np.diff(e[:j]), np.diff(e[j:])]
        records.append([*contrasts(e, j), *contrasts(v, j), *contrasts(dx, j - 1)])
    cols = [f'{kind}_{k}_{metric}' for kind in ('innovation', 'vol_adjusted', 'successive_difference') for k in BIN for metric in ('flip', 'reshape', 'on', 'off', 'pre', 'post', 'replicate')]
    tab = pd.DataFrame(np.asarray(records, dtype=np.float32), index=ids, columns=cols)
    tab.index.name = 'id'
    assert np.isfinite(tab).all().all()
    tab.to_parquet(dst)
    print('noise polarity', part, tab.shape, flush=True)
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        for part in ('train', 'test'):
            extract(part)
