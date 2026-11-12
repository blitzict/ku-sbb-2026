# CPU port; original experiments/weight_space/white_local_transport_score.py, SHA256 b9e1c8beb74c76f9092ba40c4dd09bc05cf969606f05571000ce55794516fed8
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Cross-fit the score of a small physical displacement from the observed noise density.

For a transport x -> x + h*T(x), the first-order log-density score is
-(T' + T * (log p)'). Fit the pooled density slope directly by score matching
(Hyvärinen, JMLR 2005) on the other blocks of each observed source.
"""
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
from scipy.special import expit, logsumexp
from threadpoolctl import threadpool_limits
from partition_evidence import OUT
KNOT = np.array([-2.0, -1.35, -0.85, -0.42, 0.0, 0.42, 0.85, 1.35, 2.0])
SCALES = np.array([0.4] * len(KNOT))
NAMES = ('shift', 'scale', 'two_modes', 'lower_mode', 'upper_mode', 'lower_tail', 'upper_tail')

def design(z):
    v = np.tanh(z / 3)
    dist = (z[:, None] - KNOT) / SCALES
    basis = np.exp(-0.5 * dist ** 2)
    B = np.column_stack([v, basis])
    derivative = np.column_stack([(1 - v * v) / 3, -dist / SCALES * basis])
    return (B, derivative)

def field(z):
    u = np.tanh(1.6 * z)
    low = np.exp(-0.5 * ((z + 1) / 0.55) ** 2)
    high = np.exp(-0.5 * ((z - 1) / 0.55) ** 2)
    left = expit(-3 * (z + 1.15))
    right = expit(3 * (z - 1.15))
    val = np.column_stack([np.ones(len(z)), z, u, low, high, left, right])
    grad = np.column_stack([np.zeros(len(z)), np.ones(len(z)), 1.6 * (1 - u * u), -(z + 1) / 0.55 ** 2 * low, -(z - 1) / 0.55 ** 2 * high, -3 * left * (1 - left), 3 * right * (1 - right)])
    return (val, grad)

def one(x, cut):
    block = np.r_[np.arange(cut) * 8 // cut, 8 + np.arange(len(x) - cut) * 8 // (len(x) - cut)]
    out = np.empty((len(x), len(NAMES)))
    quality = []
    for half in (0, 1):
        tr = block % 2 == half
        te = ~tr
        m = np.median(x[tr])
        scale = max(np.median(abs(x[tr] - m)) * 1.4826, 0.19 * x[tr].std(), 0.0001)
        z = (x - m) / scale
        b, d = design(z)
        s0 = -4 * z / (3 + z * z)
        penalty = np.diag([10.0, *[4.5] * len(KNOT)])
        coef = np.linalg.solve(b[tr].T @ b[tr] + penalty, -(d[tr].sum(0) + b[tr].T @ s0[tr]))
        s = s0[te] + b[te] @ coef
        a, g = field(z[te])
        out[te] = (-g - a * s[:, None]).clip(-18, 18)
        improve = 0.5 * (s * s - s0[te] ** 2) + d[te] @ coef
        quality.append(-improve.mean())
    moments = np.stack([out[block == i].mean(0) for i in range(16)])
    a, b = (moments[:8], moments[8:])
    eff = cut * (len(x) - cut) / len(x)
    diff = b.mean(0) - a.mean(0)
    point = out.var(0)
    var = 0.5 * (a.var(0, ddof=1) + b.var(0, ddof=1)) / 8 + 0.5 * point / eff + 0.02 / eff
    statistic = diff / np.sqrt(var)
    A = diff ** 2 / var
    detect = point ** 2 / var
    ratios = []
    for size in (0.1, 0.28, 0.68):
        prior = detect * size ** 2
        ratios.append(0.5 * (A * prior / (1 + prior) - np.log1p(prior)))
    rep = (b[::2] - a[::2]).mean(0) * (b[1::2] - a[1::2]).mean(0) / var
    return np.r_[np.log1p(detect), statistic, np.arcsinh(rep), logsumexp(ratios, 0) - np.log(len(ratios)), quality]

def extract(part):
    path = OUT / f'local_transport_score_{part}.parquet'
    if path.exists():
        return
    parts = []
    for ch in ('pulse', 'state'):
        with np.load(OUT / f'{ch}_streams_{part}.npz') as a:
            ids, x, n, b = (a[k] for k in ('ids', 'separate', 'n', 'b'))
        res = np.stack([one(v[:end].astype(float), j) for v, end, j in zip(x, n, b)])
        names = [ch + '_' + m + '_' + s for m in ('information', 'signed', 'replicate', 'evidence') for s in NAMES]
        parts.append(pd.DataFrame(res, index=pd.Index(ids, name='id'), columns=[*names, ch + '_fit0', ch + '_fit1']))
    assert parts[0].index.equals(parts[1].index)
    table = pd.concat(parts, axis=1)
    assert np.isfinite(table).all().all()
    table.astype(np.float32).to_parquet(path)
    print(part, table.shape, flush=True)
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        for part in ('train', 'test'):
            extract(part)
