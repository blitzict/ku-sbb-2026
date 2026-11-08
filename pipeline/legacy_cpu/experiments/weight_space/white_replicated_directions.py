# CPU port; original experiments/weight_space/white_replicated_directions.py, SHA256 8b2d3b9754a04d92cb7d4411d728895614f7df8e34e39acbb3650303774bcaae
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Learn changes that repeat in disjoint observed windows of other labelled sources."""
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
from scipy.linalg import eigh
from threadpoolctl import threadpool_limits
from partition_evidence import OUT
NAMES = 'center sign magnitude variance shoulder skew center_mass lower_tail upper_tail lower_outer upper_outer'.split()
LAGS = (1, 2, 5, 12, 20)

def blocks(x, cut):
    med = np.median(x)
    d = x - med
    scale = max(1.4826 * np.median(abs(d)), 0.18 * d.std(), 1e-05)
    z = (d / scale).clip(-4.0, 4.0)
    B = np.r_[np.arange(cut) * 4 // cut, 4 + np.arange(len(x) - cut) * 4 // (len(x) - cut)]
    weights = np.bincount(B)

    def grouped(v, where):
        nu = np.bincount(B[where], minlength=8)
        return np.bincount(B[where], weights=v[where], minlength=8) / nu.clip(1)
    take = np.ones(len(z), bool)
    stats = [grouped(v, take) for v in (z, np.tanh(2.5 * z), abs(z), z * z / 2, np.cos(2 * z), np.tanh(z) * abs(z) ** 2, abs(z) < 0.45, z < -0.85, z > 0.85, z < -1.65, z > 1.65)]
    u = np.tanh(z)
    mag = np.tanh(abs(z) - 0.7)
    for k in LAGS:
        yn = np.roll(u, k)
        an = np.roll(mag, k)
        ok = np.r_[np.zeros(k, bool), B[k:] == B[:-k]]
        for v in (u * yn, mag * an, mag * yn, u * an):
            stats.append(grouped(v, ok))
    mat = np.stack(stats, axis=1)
    sides = []
    for k in (0, 1):
        a, b = (np.array([k, k + 2]), np.array([k + 4, k + 6]))
        n, m = (weights[a].sum(), weights[b].sum())
        side = np.average(mat[b], axis=0, weights=weights[b]) - np.average(mat[a], axis=0, weights=weights[a])
        sides.append(side * np.sqrt(n * m / (n + m)))
    return sides

def data(ids, tids):
    paths = []
    values = []
    for part, ix in (('train', ids), ('test', tids)):
        p = OUT / f'replicated_direction_{part}.npz'
        paths.append(p)
        if not p.exists():
            fields = []
            for channel in ('pulse', 'state'):
                src = OUT / f'{channel}_streams_{part}.npz'
                with np.load(src) as a:
                    assert np.array_equal(a['ids'], ix)
                    x, n, c = (a[k] for k in ('separate', 'n', 'b'))
                fields.append(np.stack([blocks(v[:end].astype(float), j) for v, end, j in zip(x, n, c)]))
            m = np.concatenate(fields, axis=2).astype(np.float32)
            np.savez_compressed(p, ids=ix.to_numpy(dtype=str), blocks=m)
        with np.load(p) as a:
            assert np.array_equal(a['ids'], ix)
            values.append(a['blocks'])
    x = np.vstack(values)
    assert np.isfinite(x).all()
    return (x, paths)

def measure(x, g, y, tr, va):
    results = []
    limit = np.quantile(abs(x[tr]).reshape(-1, x.shape[2]), 0.97, axis=0).clip(0.16)
    a, b = (np.tanh(x[:, 0] / limit), np.tanh(x[:, 1] / limit))

    def moments(w, i, j):
        w = w / w.sum()
        return i.T @ (w[:, None] * j)
    for gate in g.T:
        w0, w1 = (gate[tr] * (y[tr] == 0), gate[tr] * (y[tr] == 1))
        a0, b0 = (a[tr], b[tr])
        m0 = np.average((a0 + b0) * 0.5, weights=w0, axis=0)
        a0, b0 = (a0 - m0, b0 - m0)
        left, right = (a[va] - m0, b[va] - m0)
        within = 0.5 * (moments(w0, a0, a0) + moments(w0, b0, b0))
        within += np.diag(np.diag(within) * 0.33 + 0.018)
        diff = moments(w1, a0, b0) - moments(w0, a0, b0)
        eig, V = eigh((diff + diff.T) * 0.5, within, check_finite=False)
        top = np.argsort(eig)[::-1][:8]
        v = V[:, top]
        wl, wr = (left @ v, right @ v)
        cross = (wl * wr).clip(-35, 50)
        for sl in (slice(0, 1), slice(1, 3), slice(3, 8)):
            support = np.arcsinh(cross[:, sl].mean(1))
            amp = np.log1p(0.5 * (wl[:, sl] ** 2 + wr[:, sl] ** 2).mean(1))
            results.extend([gate[va] * support, gate[va] * amp])
    return np.column_stack(results)
if __name__ == '__main__':
    import white_loop as wl
    with threadpool_limits(limits=1):
        _, ids, tids, *_ = wl.load()
        x, _ = data(ids, tids)
        print('disjoint signed changes', x.shape, flush=True)
