# CPU port; original experiments/weight_space/white_waveform_shape.py, SHA256 983184f1716e4652fe9aad284463103f69784a108d62cc87c548535266b84c58
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Test whether the shape of a repeating wave changes while its basic period persists."""
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
from functools import lru_cache
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from partition_evidence import ROOT, OUT

@lru_cache(maxsize=128)
def basis(period):
    t = np.arange(1025)
    angle = t[:, None] * 2 * np.pi / period * np.array([1.0, 2.0, 3.0, 0.5])[None, :]
    return np.column_stack([np.ones(len(t)), np.cos(angle), np.sin(angle)])

def fit(x, p):
    A = basis(p)[:len(x)]
    beta = np.linalg.lstsq(A, x, rcond=1e-06)[0]
    z = beta[1:5] - 1j * beta[5:]
    angle = np.angle(z[0])
    resid = x - A @ beta
    sigma = max(np.median(abs(resid - np.median(resid))) * 1.4826, 0.3 * resid.std(), 0.0001)
    error = sigma * np.sqrt(2 / len(x))
    amp = max(abs(z[0]), 1.8 * error)
    u = z[1:3] * np.exp(-1j * np.array([2.0, 3.0]) * angle)
    return (u / amp, error / amp, abs(z[3]) / amp, abs(z[0]) / error)

def measure(a, b, p, g):
    idx = [np.array_split(x, 2) for x in (a, b)]
    left, right, (l0, l1), (r0, r1) = (fit(a, p), fit(b, p), [fit(x, p) for x in idx[0]], [fit(x, p) for x in idx[1]])

    def strength(u, v):
        return (u[0] * v[0].conj()).real / (u[1] * v[1])

    def replicated(first, last):
        return np.clip(strength(first, last), -40, 120)

    def norm(u, v):
        return np.hypot(u[1], v[1]).clip(0.001)
    d0, d1 = (l0[0] - r0[0], l1[0] - r1[0])
    change = (d0 * d1.conj()).real / (norm(l0, r0) * norm(l1, r1))
    gap = (left[0] - right[0]) / norm(left, right)
    pre = replicated(l0, l1)
    post = replicated(r0, r1)
    sub = (left[2] - right[2]) / norm(left, right)
    f = np.r_[pre, post, change, abs(gap), abs(left[0]) / left[1], abs(right[0]) / right[1], (left[0] * right[0].conj()).real / (left[1] * right[1]), sub, (l0[2] - r0[2]) * (l1[2] - r1[2]) / (norm(l0, r0) * norm(l1, r1))]
    return (f.clip(-50, 120) * g).astype(np.float32)

def extract(part):
    dst = OUT / f'waveform_shape_{part}.parquet'
    if dst.exists():
        return
    with np.load(ROOT / f'tune_out/seq_cache_{part}.npz', allow_pickle=True) as v:
        ids = v['ids'].astype(str)
        x, n, b = (v[k] for k in ('x', 'n', 'b'))
    osc = pd.read_parquet(OUT / f'periodic_{part}.parquet').loc[ids]
    cols = [f'{kind}_{n}' for kind in ('pre_repeats', 'post_repeats', 'change_repeats', 'change', 'pre_amplitude', 'post_amplitude', 'continues') for n in (2, 3)] + ['period_doubling', 'doubling_repeats']
    result = np.zeros((len(ids), len(cols)), dtype=np.float32)
    for i in np.flatnonzero((osc.gate > 0.025) & (osc.period <= 48)):
        j = b[i]
        raw = x[i, :n[i]].astype(float)
        result[i] = measure(raw[:j], raw[j:], int(osc.period.iat[i]), float(osc.gate.iat[i]))
    tab = pd.DataFrame(result, index=pd.Index(ids, name='id'), columns=cols)
    assert np.isfinite(tab).all().all()
    tab.to_parquet(dst)
    print('waveform', part, tab.shape, 'tested', int((result != 0).any(1).sum()), flush=True)
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        for part in ('train', 'test'):
            extract(part)
