# CPU port; original experiments/weight_space/white_second_clock.py, SHA256 9bcb73d43bf87dc984c81ae95da06fcd909a082887fd439ebb22ec954b4ab32f
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Test whether a second non-harmonic clock predicts held-out blocks of the de-waved series."""
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
from threadpoolctl import threadpool_limits
from partition_evidence import OUT
F = np.fft.rfftfreq(2048)
NAMES = ['pre', 'post', 'pre_repeat', 'post_repeat', 'carry', 'reverse_carry', 'same', 'split', 'new', 'lost']

def one(x, cut, period):
    sides = [x[:cut].copy(), x[cut:].copy()]
    for v in sides:
        v -= np.median(v)
        s = max(1.4826 * np.median(abs(v)), 0.2 * v.std(), 0.0001)
        v[:] = (v / s).clip(-3.5, 3.5)
    z = np.concatenate(sides)
    t = np.arange(len(z))
    half = [t < cut, t >= cut]
    blocks = np.r_[np.arange(cut) * 8 // cut, np.arange(len(x) - cut) * 8 // (len(x) - cut)]
    usable = (F >= 1 / 72) & (F <= 1 / 5)
    if period > 0:
        for ratio in (0.5, 1.0, 2.0, 3.0):
            usable &= abs(F - ratio / period) > max(1.8 / min(map(len, sides)), 0.025 * ratio / period)

    def select(mask):
        score = abs(np.fft.rfft((z - np.mean(z[mask])) * mask, n=2048)) ** 2
        return F[np.argmax(np.where(usable, score, -np.inf))]

    def clock(freq, mask):
        phase = 2 * np.pi * freq * t
        h = np.column_stack([np.cos(phase), np.sin(phase)])
        h -= h[mask].mean(0)
        a = h[mask]
        w = np.linalg.solve(a.T @ a + np.eye(2) * 9.0, a.T @ (z[mask] - z[mask].mean()))
        return h @ w

    def score(mask, v):
        r = z[mask] - z[mask].mean()
        return (r @ v[mask] - 0.5 * v[mask] @ v[mask]) / (np.mean(r * r) + 0.05)
    rows = []
    for k in (0, 1):
        tr = blocks % 2 == k
        te = ~tr
        pre, post = [tr & i for i in half]
        a, b = [te & i for i in half]
        l, r, c = [clock(select(mask), mask) for mask in (pre, post, tr)]
        own = [score(a, l), score(b, r)]
        carry = [score(b, l), score(a, r)]
        same = score(a, c) + score(b, c)
        rows.append([*own, *carry, same, sum(own) - same, own[1] - carry[0], own[0] - carry[1]])
    vals = np.array(rows)
    return np.r_[vals.sum(0)[:2], vals.min(0)[:2], vals.sum(0)[2:]]

def main():
    import white_loop as wl
    _, ids, tids, y, _, wn, miss = wl.load()
    for p, ix in (('train', ids), ('test', tids)):
        dst = OUT / f'second_clock_{p}.parquet'
        if not dst.exists():
            tab = pd.read_parquet(OUT / f'periodic_{p}.parquet').loc[ix]
            with np.load(OUT / f'periodic_streams_{p}.npz') as a:
                assert np.array_equal(ix, a['ids'])
                x, b, n = (a[k] for k in ('separate', 'b', 'n'))
            rows = [one(v[:end].astype(float), j, pd0 if g > 0.5 else 0) for v, end, j, pd0, g in zip(x, n, b, tab.period, tab.gate)]
            frame = pd.DataFrame(rows, index=ix, columns=NAMES).clip(-65, 140)
            assert np.isfinite(frame).all().all()
            frame['first_clock'] = tab.gate
            for col in ('same', 'split', 'new', 'lost'):
                frame['two_clocks_' + col] = frame[col] * tab.gate
            frame.astype(np.float32).to_parquet(dst)
        if p == 'train':
            t = pd.read_parquet(dst).loc[ix]
            pre = (t.pre > 5) & (t.pre_repeat > 0)
            for name, mask in (('negative', y == 0), ('negative known clock', (y == 0) & (t.first_clock > 0.5)), ('white negative', (y == 0) & wn), ('historical', miss)):
                print(name, int(mask.sum()), int((mask & pre).sum()), int((mask & pre & (t.carry > 0)).sum()), flush=True)
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
