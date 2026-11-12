# CPU port; original experiments/weight_space/white_oscillation.py, SHA256 ca0804110271283f8a7d67dc69928c6adeaaf6cbe6068a00bece7b36f24c0db0
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Test held-out phase and amplitude of narrow lines in levels and volatility."""
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
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.ndimage import gaussian_filter1d
from threadpoolctl import threadpool_limits
import white_loop as wl
from partition_evidence import OUT
NFFT = 2048
FREQ = np.fft.rfftfreq(NFFT)
BANDS = [(2, 10), (10, 40), (40, 110)]

def spectra(row, b):
    signals = []
    for v in (row[:b], row[b:]):
        z = v - np.median(v)
        s = max(np.median(abs(z)) * 1.4826, 0.18 * z.std(), 1e-06)
        z /= s
        mag = np.log1p(z * z).clip(0, 5)
        signals.append(np.stack([z.clip(-3, 3), mag, np.tanh(2.5 * (abs(z) - 1.5)), np.sign(z)]))
    a, c = [np.fft.rfft(r - r.mean(1, keepdims=True), n=NFFT, axis=1) / np.sqrt(r.shape[1]) for r in signals]
    c *= np.exp(-2j * np.pi * b * FREQ)
    pa, pc = (abs(a) ** 2, abs(c) ** 2)

    def background(p):
        smooth = gaussian_filter1d(p, 18, axis=1)
        return gaussian_filter1d(np.minimum(p, 3 * smooth), 21, axis=1).clip(0.02)
    ba, bc = (background(pa), background(pc))
    a, c = (a / np.sqrt(ba), c / np.sqrt(bc))
    n = len(row)
    pre = b / n
    shared = abs(np.sqrt(pre) * a + np.sqrt(1 - pre) * c) ** 2
    diff = abs(np.sqrt(1 - pre) * a - np.sqrt(pre) * c) ** 2
    out = []
    rows = np.arange(a.shape[0])
    for lo, hi in BANDS:
        sl = (FREQ >= 1 / hi) & (FREQ <= 1 / lo)
        x, y = (abs(a[:, sl]) ** 2, abs(c[:, sl]) ** 2)
        u, v = (shared[:, sl], diff[:, sl])
        dot = (a[:, sl] * c[:, sl].conj()).real
        ja, jb, jc = (t.argmax(1) for t in (x, y, u))
        out.extend([x[rows, ja], y[rows, ja], dot[rows, ja], y[rows, jb], x[rows, jb], dot[rows, jb], u[rows, jc], v[rows, jc], (u + v).max(1) - u.max(1), logsumexp(0.55 * v, axis=1) - np.log(v.shape[1]), np.abs(FREQ[sl][ja] - FREQ[sl][jb]) * min(b, n - b)])
    return np.stack(out, 1)

def load(part):
    path = OUT / f'vol_line_{part}.npz'
    if not path.exists():
        with np.load(OUT / f'pulse_streams_{part}.npz') as q:
            ids, x, n, b = (q[k] for k in ('ids', 'separate', 'n', 'b'))
        features = np.stack([spectra(v[:e].astype(float), j) for v, e, j in zip(x, n, b)]).astype(np.float32)
        np.savez_compressed(path, ids=ids, features=features)
        print(part, 'volatility spectra', flush=True)
    with np.load(path) as q:
        return (q['ids'], q['features'])

def main():
    _, ids, tids, y, _, wn, miss = wl.load()
    train, a = load('train')
    test, b = load('test')
    assert np.array_equal(train, ids) and np.array_equal(test, tids)
    for title, mask in (('stable white', wn & (y == 0)), ('broken white', wn & (y == 1)), ('miss', miss)):
        for j in (0, 1):
            ans = []
            for k in (0, 11, 22):
                high = mask & (a[:, j, k] > 8)
                ans.append([int(high.sum()), *[round(float(r), 2) for r in np.mean(a[high, j, k + 1:k + 3], axis=0)]])
            print(title, j, ans, flush=True)
    raw = np.concatenate([a, b])
    f = np.arcsinh(raw / 3).reshape(len(raw), -1)
    vol = np.r_[*[pd.read_parquet(OUT / f'state_{p}.parquet').loc[ix, 'clustering'].to_numpy() for p, ix in (('train', ids), ('test', tids))],]

    def design(ix, z):
        v = f[ix]
        return np.column_stack([v, v * vol[ix, None]])
    wl.run('vol_lines', design, 300.0, dependencies=[Path(__file__).resolve(), *[OUT / f'vol_line_{p}.npz' for p in ('train', 'test')]])
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
