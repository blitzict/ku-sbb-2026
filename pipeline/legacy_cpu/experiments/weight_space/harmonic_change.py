# CPU port; original experiments/weight_space/harmonic_change.py, SHA256 af16e403dda79b6fa545560f7046ce76fec6462c5c4cad46a7b8dbe8290ef3b5
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Compare a continuous oscillator with a phase or frequency reset at the break.

For each trial frequency, integrate out separate level offsets and solve the
two linear harmonic coefficients on each side and on the whole record. Period
evidence uses a grid mixture. Also test for changes in the demodulated series.
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
from functools import lru_cache
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits
from partition_evidence import ROOT, OUT
NFFT = 2048
K = np.arange(6, 513)
F = K / NFFT
OMEGA = 2 * np.pi * F

@lru_cache(maxsize=1600)
def moments(length, side):
    offset = -length if side == 'left' else 0
    midpoint = offset + (length - 1) / 2
    rot = np.exp(1j * OMEGA * offset)
    s = np.exp(1j * OMEGA * midpoint) * np.sin(length * OMEGA / 2) / np.sin(OMEGA / 2)
    s2 = np.exp(2j * OMEGA * midpoint) * np.sin(length * OMEGA) / np.sin(OMEGA)
    cc = (length + s2.real) * 0.5 - s.real ** 2 / length
    ss = (length - s2.real) * 0.5 - s.imag ** 2 / length
    cs = s2.imag * 0.5 - s.real * s.imag / length
    return (rot, s, np.stack([cc, ss, cs]))

def equations(x, side):
    rot, s, G = moments(len(x), side)
    target = x - x.mean()
    xy = np.fft.rfft(target, n=NFFT)[K].conj() * rot
    return (G, np.stack([xy.real, xy.imag]), float(target @ target))

def fit(G, xy):
    cc, ss, cs = G
    x, y = xy
    det = (cc * ss - cs ** 2).clip(1e-05)
    a, b = ((ss * x - cs * y) / det, (cc * y - cs * x) / det)
    return (np.maximum(a * x + b * y, 0), a, b)

def score(a, b):
    n, m = (len(a), len(b))
    A, u, va = equations(a, 'left')
    B, v, vb = equations(b, 'right')
    h0, c0, s0 = fit(A + B, u + v)
    h1, c1, s1 = fit(A, u)
    h2, c2, s2 = fit(B, v)
    w = max(va + vb, 1e-12)

    def ll(h):
        return -0.5 * (n + m) * np.log1p(-np.clip(h / w, 0, 0.99999))

    def integrate(x):
        return logsumexp(x) - np.log(len(x))
    shared = ll(h0)
    reset = ll(h1 + h2)
    sep = ll(h1.max() + h2.max())
    f0, f1, f2, fr = map(np.argmax, [shared, h1, h2, reset])
    peak = integrate(shared)
    result = [peak, integrate(reset) - peak, reset.max() - shared.max(), sep - reset.max(), sep - shared.max(), F[f0], F[fr], abs(F[f1] - F[f2]), h0[f0] / w, h1[f1] / max(va, 1e-08), h2[f2] / max(vb, 1e-08)]
    noise = np.sqrt(max((w - h1[fr] - h2[fr]) / (n + m - 6), 1e-10))
    vpre, vpost = (c1[fr] + 1j * s1[fr], c2[fr] + 1j * s2[fr])
    result.extend([abs(abs(vpre) - abs(vpost)) / noise, abs(np.angle(vpre * vpost.conjugate())), abs(vpre - vpost) * np.sqrt(n * m / (n + m)) / noise])
    phase = np.arange(-n, m) * OMEGA[f0]
    harmonic = c0[f0] * np.cos(phase) + s0[f0] * np.sin(phase)
    r, q = (a - harmonic[:n], b - harmonic[n:])
    variance = (r.var() * n + q.var() * m) / (n + m)
    level = abs(r.mean() - q.mean()) / np.sqrt(max(variance * (1 / n + 1 / m), 1e-12))
    scale = abs(np.log(max(r.std(), 1e-08) / max(q.std(), 1e-08)))
    result.extend([level, scale, min(10.0, max(peak / 10, 0)) * level, min(10.0, max(peak / 10, 0)) * scale])
    return result
NAMES = ['shared', 'phase_bf', 'phase_lr', 'freq_lr', 'separate_lr', 'frequency', 'phase_frequency', 'frequency_gap', 'fraction', 'pre_fraction', 'post_fraction', 'amplitude_gap', 'angle', 'complex_gap', 'demod_level', 'demod_scale', 'osc_level', 'osc_scale']

def run(part):
    with np.load(ROOT / f'tune_out/seq_cache_{part}.npz', allow_pickle=True) as z:
        ids, x, n, b = (z[k] for k in ('ids', 'x', 'n', 'b'))
    index = pd.read_parquet(ROOT / f'clean/features_{part}.parquet', columns=[]).index
    assert np.array_equal(ids, index)
    values = []
    for i in range(len(ids)):
        s = x[i, :n[i]].astype(float)
        l, r = (s[:b[i]], s[b[i]:])
        pre, post = (l - l.mean(), r - r.mean())
        den = pre[:-1] @ pre[:-1] + post[:-1] @ post[:-1]
        phi = np.clip((pre[:-1] @ pre[1:] + post[:-1] @ post[1:]) / max(den, 1e-10), -0.99, 0.99)
        ea = l[1:] - phi * l[:-1]
        eb = r - phi * np.r_[l[-1], r[:-1]]
        values.append(score(l, r) + score(ea, eb))
    tab = pd.DataFrame(values, index=index, columns=[s + '_' + k for s in ['raw', 'ar1'] for k in NAMES])
    assert np.isfinite(tab).all().all()
    tab.to_parquet(OUT / f'harmonic_{part}.parquet')
    print(part, 'median spectral fraction', tab.raw_fraction.median(), 'over 30%', (tab.raw_fraction > 0.3).mean(), flush=True)
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        run('train')
        run('test')
