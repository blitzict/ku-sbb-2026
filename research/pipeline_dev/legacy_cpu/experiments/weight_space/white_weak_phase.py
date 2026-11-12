# CPU port; original experiments/weight_space/white_weak_phase.py, SHA256 12f21896c4d5d258c8ca58c07db1eaaa0df76b2820e817c1271eb70884555fcd
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Search below the usual seasonal gate, integrating over weak clocks and phase resets."""
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
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits
from partition_evidence import ROOT, OUT
FFT = 2048
K = np.arange(int(np.ceil(FFT / 49)), int(FFT / 8) + 1)
OMEGA = 2 * np.pi * K / FFT
ANGLES = np.array([np.pi / 6, -np.pi / 6, np.pi / 3, -np.pi / 3, np.pi / 2, -np.pi / 2, 2 * np.pi / 3, -2 * np.pi / 3, np.pi])

def fft_stats(z, m):
    n = m.sum(1, keepdims=True)
    F = fft.rfft(m, n=FFT, axis=1)
    H = fft.rfft(z * m, n=FFT, axis=1)
    trig = F[:, K]
    xx, xy, yy = (0.5 * (n + F[:, 2 * K].real), -0.5 * F[:, 2 * K].imag, 0.5 * (n - F[:, 2 * K].real))
    c, s = (trig.real, -trig.imag)
    xx = xx - c * c / n
    xy = xy - c * s / n
    yy = yy - s * s / n
    mu = H[:, :1].real / n
    xz = H[:, K].real - c * mu
    yz = -H[:, K].imag - s * mu
    return (xx, xy, yy, xz, yz)

def rotate(stats, angle):
    xx, xy, yy, dx, dy = stats
    c, s = (np.cos(angle), np.sin(angle))
    return (c * c * xx + 2 * c * s * xy + s * s * yy, (c * c - s * s) * xy + c * s * (yy - xx), s * s * xx - 2 * c * s * xy + c * c * yy, c * dx + s * dy, -s * dx + c * dy)

def score(stats):
    xx, xy, yy, dx, dy = stats
    acc = []
    for amp in (0.1, 0.24, 0.5):
        prior = 1 / amp ** 2
        X, Y = (xx + prior, yy + prior)
        det = (X * Y - xy * xy).clip(0.5 * prior ** 2)
        explained = (Y * dx * dx - 2 * xy * dx * dy + X * dy * dy) / det
        acc.append(0.5 * (explained - np.log(det / prior ** 2)))
    return logsumexp(acc, axis=0) - np.log(len(acc))

def joined(a, b):
    return tuple((x + y for x, y in zip(a, b)))

def evidence(z, a, b, cut):
    L, R = (fft_stats(z, a), fft_stats(z, b))
    joint = score(joined(L, R))
    independent = score(L) + score(R)
    rotation = np.stack([score(joined(L, rotate(R, k))) for k in ANGLES])
    local = score(joined(L, rotate(R, -cut[:, None] * OMEGA)))
    inverse = rotation[-1]

    def integrate(v):
        return logsumexp(v, axis=-1) - np.log(len(K))
    ll = [integrate(v) for v in (joint, independent, local, inverse)]
    ll.append(logsumexp(integrate(rotation), axis=0) - np.log(len(ANGLES)))
    ll.extend([integrate(score(L)), integrate(score(R))])
    return np.stack(ll, axis=1)

def extract(part):
    path = OUT / f'weak_phase_{part}.parquet'
    if path.exists():
        return
    with np.load(ROOT / f'tune_out/seq_cache_{part}.npz', allow_pickle=True) as a:
        ids, n, b, raw = (a['ids'].astype(str), a['n'], a['b'], a['x'])
    with np.load(OUT / f'pulse_streams_{part}.npz') as a:
        assert np.array_equal(a['ids'], ids) and np.array_equal(a['b'], b) and np.array_equal(a['n'], n)
        pulse = a['separate']
    blocks = []
    names = ['continuation', 'free_change', 'restart', 'inversion', 'phase_change', 'pre', 'post']
    for ch, x in (('raw', raw), ('pulse', pulse)):
        allscores = []
        for ix in np.array_split(np.arange(len(ids)), (len(ids) + 39) // 40):
            z = np.zeros((len(ix), x.shape[1]))
            ts = np.arange(x.shape[1])[None, :]
            cut = b[ix]
            left = ts < cut[:, None]
            valid = ts < n[ix, None]
            right = ~left & valid
            band = np.where(left, ts * 8 // cut[:, None], (ts - cut[:, None]) * 8 // (n[ix] - cut)[:, None])
            for j, i in enumerate(ix):
                for sl in (slice(0, b[i]), slice(b[i], n[i])):
                    v = x[i, sl].astype(float)
                    d = v - np.median(v)
                    scale = max(1.4826 * np.median(abs(d)), 0.2 * v.std(), 1e-05)
                    z[j, sl] = (d / scale).clip(-3.0, 3.0)
            full = evidence(z, left, right, cut)
            half = np.stack([evidence(z, left & (band % 2 == k), right & (band % 2 == k), cut) for k in (0, 1)])
            ratios = full - full[:, :1]
            train = half - half[:, :, :1]
            held = ratios[None, :, :] - train
            allscores.append(np.column_stack([full, ratios[:, 1:5], held[:, :, 1:5].min(0)]))
        columns = [ch + '_' + k for k in names] + [ch + '_' + s + '_' + k for s in ('vs_same', 'replicated') for k in names[1:5]]
        blocks.append(pd.DataFrame(np.vstack(allscores), index=pd.Index(ids, name='id'), columns=columns))
    df = pd.concat(blocks, axis=1).clip(-100, 200)
    assert np.isfinite(df).all().all()
    df.astype(np.float32).to_parquet(path)
    print(part, df.shape, flush=True)

def diagnose():
    import white_loop as wl
    _, ids, _, y, _, wn, miss = wl.load()
    table = pd.read_parquet(OUT / 'weak_phase_train.parquet').loc[ids]
    active = pd.read_parquet(OUT / 'phase_law_train.parquet').loc[ids, 'df'].to_numpy() > 0
    for ch in ('raw', 'pulse'):
        v = table[ch + '_replicated_phase_change'].to_numpy()
        absolute = table[ch + '_phase_change'].to_numpy()
        strong = (v > 5) & (absolute > 5)
        for tag, g in (('old untested historical', miss & ~active), ('old tested historical', miss & active), ('old untested white negatives', (y == 0) & wn & ~active), ('all negatives', y == 0)):
            print(ch, tag, int(g.sum()), int((g & strong).sum()), round(np.quantile(v[g], 0.95), 3), flush=True)
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        for p in ('train', 'test'):
            extract(p)
        diagnose()
