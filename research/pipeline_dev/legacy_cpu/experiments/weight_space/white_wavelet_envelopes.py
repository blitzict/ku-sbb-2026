# CPU port; original experiments/weight_space/white_wavelet_envelopes.py, SHA256 8b2b219633c32c396e99b02f1cfe079fe06bd56351216eb9d3dd74bf7d000d8b
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Compare the strength of modulations of a wave across a known break, at fixed wave energy.

The fixed filter cascade follows Andén & Mallat, Deep Scattering Spectrum,
and Bruna et al., Intermittent process analysis with scattering moments.
Only the competition's real observations are transformed.
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
from scipy import fft
from threadpoolctl import threadpool_limits
from partition_evidence import OUT
PERIODS = (3, 5, 8, 13, 21, 34)
MOD = (8, 16, 32, 64)
NFFT = 1024
PAD = 128

def atoms(periods):
    f = fft.fftfreq(NFFT)
    chart = []
    for p in periods:
        p0 = 1 / p
        a = np.exp(-0.5 * ((f - p0) / (0.32 * p0)) ** 2) * (f > 0)
        chart.append(a / np.sqrt(np.mean(a * a)))
    return np.stack(chart).astype(np.float32)

def side(padded, start, length):
    t = np.arange(NFFT)[None, :] - PAD
    back = t % (2 * length[:, None] - 2)
    index = np.minimum(back, 2 * length[:, None] - 2 - back) + start[:, None]
    a = np.take_along_axis(padded, index, axis=1)
    keep = (t >= 0) & (t < length[:, None])
    blocks = [(keep & (4 * t // length[:, None] == i)).astype(np.float32) for i in range(4)]
    blocks = np.stack([m / m.sum(1, keepdims=True) for m in blocks], axis=2)

    def means(x):
        return np.einsum('bct,btj->bcj', x, blocks, optimize=True)
    h = fft.fft(a, axis=1)[:, None, :] * atoms(PERIODS)
    envelope = abs(fft.ifft(h, axis=2))
    first = means(envelope)
    second = means(envelope ** 2)
    energy = np.log(first + 1e-05)
    sparse = np.log((second + 1e-06) / (first ** 2 + 1e-06))
    spectrum = fft.fft(envelope, axis=2)
    mod = []
    names = []
    for period, g in zip(MOD, atoms(MOD)):
        inds = [i for i, k in enumerate(PERIODS) if period >= 2 * k]
        wave = fft.ifft(spectrum[:, inds] * g, axis=2)
        val = np.log((means(abs(wave)) + 1e-05) / (first[:, inds] + 1e-05))
        mod.append(val)
        names.extend((f'p{PERIODS[i]}_m{period}' for i in inds))
    return (np.concatenate([energy, sparse, *mod], axis=1), [f'energy_{k}' for k in PERIODS] + [f'burst_{k}' for k in PERIODS] + names)

def extract(part):
    path = OUT / f'wavelet_envelopes_{part}.parquet'
    if path.exists():
        return
    blocks = []
    for name in ('pulse', 'state'):
        with np.load(OUT / f'{name}_streams_{part}.npz') as a:
            ids, x, n, b = (a[k] for k in ('ids', 'separate', 'n', 'b'))
        assert np.max(np.maximum(b, n - b)) + PAD < NFFT
        output = []
        for ix in np.array_split(np.arange(len(n)), (len(n) + 95) // 96):
            v = x[ix].astype(np.float32)
            pair = [side(v, s, k) for s, k in ((np.zeros(len(ix), int), b[ix]), (b[ix], n[ix] - b[ix]))]
            A, B = [p[0] for p in pair]
            names = pair[0][1]
            d = B.mean(2) - A.mean(2)
            se = np.sqrt((A.var(2, ddof=1) + B.var(2, ddof=1)) * 0.25 + 0.035 ** 2)
            changes = np.arcsinh(d / se)
            rep = (B[:, :, ::2].mean(2) - A[:, :, ::2].mean(2)) * (B[:, :, 1::2].mean(2) - A[:, :, 1::2].mean(2))
            vals = [A.mean(2), B.mean(2), changes, np.arcsinh(rep / se ** 2)]
            output.append(np.column_stack(vals))
        cols = [f'{name}_{op}_{c}' for op in ('pre', 'post', 'signed', 'repeat') for c in names]
        blocks.append(pd.DataFrame(np.vstack(output), index=pd.Index(ids, name='id'), columns=cols))
    assert blocks[0].index.equals(blocks[1].index)
    tab = pd.concat(blocks, axis=1)
    assert np.isfinite(tab).all().all()
    tab.astype(np.float32).to_parquet(path)
    print(part, tab.shape, flush=True)

def compact(part):
    tab = pd.read_parquet(OUT / f'wavelet_envelopes_{part}.parquet')
    result = {}
    for channel in ('pulse', 'state'):
        for mod in MOD:
            cols = [c for c in tab if c.startswith(channel + '_repeat_p') and c.endswith('_m' + str(mod))]
            signed = [c.replace('_repeat_', '_signed_') for c in cols]
            rep = tab[cols].to_numpy()
            d = tab[signed].to_numpy()
            result[f'{channel}_strength_{mod}'] = rep.max(1)
            result[f'{channel}_typical_{mod}'] = rep.mean(1)
            result[f'{channel}_direction_{mod}'] = d.mean(1)
    pd.DataFrame(result, index=tab.index).to_parquet(OUT / f'wavelet_modulation_{part}.parquet')
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        for p in ('train', 'test'):
            extract(p)
            compact(p)
