# CPU port; original experiments/weight_space/other_memory_horizons.py, SHA256 e6cb6625a5e011bb987a82ea8ecd2eeb8a10f9e777be06b12d3410927bed8b5c
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Separate how strongly adjacent values agree from how long their memory lasts."""
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
import json
from functools import lru_cache
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits
from partition_evidence import OUT
ADJ = np.r_[-0.97, -0.92, -0.82, -0.7, -0.56, -0.42, -0.3, -0.2, -0.115, -0.045, 0.0, 0.045, 0.115, 0.2, 0.3, 0.42, 0.56, 0.7, 0.82, 0.92, 0.97]
EXT = np.array([0.0, 0.2, 0.45, 0.7, 0.9])
DAMP = ADJ[:, None] + np.sign(ADJ[:, None]) * (0.995 - abs(ADJ[:, None])) * EXT
SHARE = np.divide(ADJ[:, None], DAMP, out=np.zeros_like(DAMP), where=DAMP != 0)
P = np.gradient(ADJ) * np.exp(-0.35 * ADJ ** 2)
P = P / P.sum() * 0.71
P[ADJ == 0] += 0.29
Q = np.array([0.56, 0.16, 0.12, 0.1, 0.06])
LOG_P, LOG_Q = (np.log(P), np.log(Q))

@lru_cache(None)
def spectrum(length):
    h = np.arange(length)
    corr = SHARE[:, :, None] * DAMP[:, :, None] ** h
    corr[:, :, 0] = 1.0
    f = 2 * np.fft.rfft(corr * (1 - h / length), axis=-1).real - 1
    freq = np.ones(f.shape[-1] - 1)
    if length % 2 == 0:
        freq[-1] = 0.5
    f = f[:, :, 1:]
    assert np.min(f) > 0
    return ((freq / f).reshape(-1, len(freq)).T.copy(), (np.log(f) * freq).sum(2).ravel(), freq.sum())

def periods(v):
    c = np.median(v)
    sd = max(1.4826 * np.median(abs(v - c)), 0.25 * v.std(), 0.002)
    x = ((v - c) / sd).clip(-5, 5)
    return abs(np.fft.rfft(x - x.mean())[1:]) ** 2 / len(v)

def collect(rows, lengths, cuts):
    groups = {}
    for i, (row, n, b) in enumerate(zip(rows, lengths, cuts)):
        for side, x in enumerate((row[:b], row[b:n])):
            for parity, block in enumerate(np.array_split(x, 2)):
                groups.setdefault(len(block), []).append((i, 2 * side + parity, periods(block)))
    likelihood = np.zeros((len(rows), 4, len(ADJ), len(EXT)), np.float32)
    for length, items in groups.items():
        w, offset, mass = spectrum(length)
        for start in range(0, len(items), 1024):
            batch = items[start:start + 1024]
            ii, jj, periodogram = zip(*batch)
            s = (np.stack(periodogram) @ w / mass).clip(1e-11)
            ll = -mass * np.log(s) - offset
            ll -= ll.max(1, keepdims=True)
            likelihood[np.array(ii), np.array(jj)] = ll.reshape(-1, len(ADJ), len(EXT))
    return likelihood

def hypotheses(a, b):
    joint = a + b + LOG_P[:, None] + LOG_Q
    null = logsumexp(joint, axis=(1, 2))

    def marginal(v):
        return logsumexp(v + LOG_P[:, None] + LOG_Q, axis=(1, 2))
    same_adj = logsumexp(logsumexp(a + LOG_Q, axis=2) + logsumexp(b + LOG_Q, axis=2) + LOG_P, axis=1)
    same_fraction = logsumexp(logsumexp(a + LOG_P[:, None], axis=1) + logsumexp(b + LOG_P[:, None], axis=1) + LOG_Q, axis=1)
    both = marginal(a) + marginal(b)
    return np.stack([same_adj - null, same_fraction - null, both - null, both - same_adj, both - same_fraction], 1)

def main(part):
    with np.load(OUT / f'pulse_streams_{part}.npz') as a:
        ids, x, n, b = [a[k] for k in ('ids', 'separate', 'n', 'b')]
    ll = collect(x, n, b)
    joint = hypotheses(ll[:, 0] + ll[:, 1], ll[:, 2] + ll[:, 3])
    first = hypotheses(ll[:, 0], ll[:, 2])
    second = hypotheses(ll[:, 1], ll[:, 3])
    cols = ('memory_at_fixed_lag1', 'lag1_at_fixed_extension', 'both', 'lag1_extra', 'extension_extra')
    df = pd.DataFrame({f'{name}_{k}': v[:, j] for name, v in (('all', joint), ('replicate', np.minimum(first, second)), ('held', joint - first - second)) for j, k in enumerate(cols)}, index=ids)
    for side, r in (('before', ll[:, :2].sum(1)), ('after', ll[:, 2:].sum(1))):
        w = r + LOG_P[:, None] + LOG_Q
        w = np.exp(w - logsumexp(w, axis=(1, 2), keepdims=True))
        df[side + '_adjacent'] = (w * ADJ[:, None]).sum((1, 2))
        df[side + '_persistence'] = (w * abs(DAMP)).sum((1, 2))
        df[side + '_memory_at_lag5'] = (w * SHARE * DAMP ** 5).sum((1, 2))
    assert np.isfinite(df).all().all()
    df.to_parquet(OUT / f'other_memory_horizons_{part}.parquet')
    print(part, df.shape, flush=True)
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        for part in ('train', 'test'):
            main(part)
