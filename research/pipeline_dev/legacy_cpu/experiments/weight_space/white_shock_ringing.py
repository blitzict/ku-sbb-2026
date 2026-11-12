# CPU port; original experiments/weight_space/white_shock_ringing.py, SHA256 4e3ca2958d04b76919f40c065f01897ffa788f0e3e84f6dfc8f853092c82b099
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Test whether unusually large innovations excite a damped oscillation."""
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
from threadpoolctl import threadpool_limits
from partition_evidence import OUT
from white_signature_screen import pairs

@lru_cache(None)
def atoms(lags):
    t = np.arange(1, lags + 1, dtype=float)
    nuisance = np.column_stack([np.ones(lags), np.exp(-t / 3.0), np.exp(-t / 12.0)])
    q = np.linalg.qr(nuisance)[0]
    result = []
    for period in (8, 12, 16, 20, 25, 32, 40):
        phase = 2 * np.pi * t / period
        ring = np.exp(-t[:, None] / (0.85 * period)) * np.column_stack([np.cos(phase), np.sin(phase)])
        ring -= q @ (q.T @ ring)
        result.append(np.linalg.qr(ring)[0])
    return np.stack(result)

def chart(v):
    center = np.median(v)
    scale = max(1.4826 * np.median(abs(v - center)), 0.2 * v.std(), 1e-05)
    z = ((v - center) / scale).clip(-6, 6)
    size = abs(z)
    tail = np.maximum(size - 1.35, 0.0)
    raw = np.column_stack([np.sign(z) * tail, tail])
    linear = np.column_stack([np.ones(len(z)), z, size])
    trigger = raw - linear @ np.linalg.lstsq(linear, raw, rcond=1e-07)[0]
    trigger /= np.sqrt(np.mean(trigger ** 2, axis=0) + 0.035)
    response = np.column_stack([np.tanh(z), np.log1p(size ** 2)])
    response -= response.mean(0)
    response /= np.sqrt(np.mean(response ** 2, axis=0) + 0.015)
    w = min(32, len(z) // 2)
    width = 1 << (2 * len(v) - 1).bit_length()
    a = np.fft.rfft(trigger, n=width, axis=0)
    b = np.fft.rfft(response, n=width, axis=0)
    future = np.fft.irfft(a.conj()[:, :, None] * b[:, None, :], n=width, axis=0)[1:w + 1]
    future /= np.sqrt(len(v) - np.arange(1, w + 1))[:, None, None]
    return np.einsum('lij,plk->ijpk', future, atoms(w))

def measure(pre, post):
    sizes = np.array([[len(v) for v in np.array_split(x, 2)] for x in (pre, post)])
    groups = np.array([[chart(v) for v in np.array_split(x, 2)] for x in (pre, post)])
    a, b = sizes
    wl = np.sqrt(b / (a + b))[:, None, None, None, None]
    wr = np.sqrt(a / (a + b))[:, None, None, None, None]
    d = groups[1] * wr - groups[0] * wl
    out = {}
    for i, shock in enumerate(('signed', 'unsigned')):
        for j, response in enumerate(('direction', 'amplitude')):
            label = shock + '_' + response
            u, v = d[:, i, j]
            dot = np.sum(u * v, axis=-1)
            out[label + '_repeat'] = dot.mean()
            pick = np.argmax(np.stack([(u * u).sum(-1), (v * v).sum(-1)]), axis=1)
            out[label + '_selected'] = dot[pick].mean()
            prior = groups[0, :, i, j]
            after = groups[1, :, i, j]
            out[label + '_before'] = np.mean(np.sum(prior[0] * prior[1], axis=-1))
            out[label + '_after'] = np.mean(np.sum(after[0] * after[1], axis=-1))
    return out

def run(part):
    with np.load(OUT / f'pulse_streams_{part}.npz') as a:
        ids, x, n, b = [a[k] for k in ('ids', 'separate', 'n', 'b')]
    ar = pd.read_parquet(OUT / f'state_{part}.parquet').loc[ids, ['ar1', 'ar2']].to_numpy()
    rows = []
    for row, end, cut, p in zip(x, n, b, ar):

        def innovations(v):
            return (v[2:] - v[1:-1] * p[0] - v[:-2] * p[1]).astype(float)
        rows.append(measure(innovations(row[:cut]), innovations(row[cut:end])))
    frame = pd.DataFrame(rows, index=ids)
    assert np.isfinite(frame).all().all()
    frame.to_parquet(OUT / f'shock_ringing_{part}.parquet')
    print('shock ringing', part, frame.shape, flush=True)

def matched():
    ix, h, control, _ = pairs()
    df = pd.read_parquet(OUT / 'shock_ringing_train.parquet').loc[ix]
    s = df.filter(like='_repeat').mean(axis=1).to_numpy()
    d = s[h] - s[control]
    rng = np.random.default_rng(32981)
    boots = np.concatenate([rng.choice(d, (256, len(h))).mean(1) for _ in range(64)])
    result = dict(n=len(h), case=float(s[h].mean()), control=float(s[control].mean()), difference=float(d.mean()), pair_win=float(np.mean(d > 0)), paired_95_interval=np.quantile(boots, [0.025, 0.975]).tolist())
    (OUT / 'shock_ringing_matched.json').write_text(json.dumps(result, indent=2))
    print(result, flush=True)
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        for part in ('train', 'test'):
            run(part)
        matched()
