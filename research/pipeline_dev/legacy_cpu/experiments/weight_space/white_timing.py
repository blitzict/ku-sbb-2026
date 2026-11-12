# CPU port; original experiments/weight_space/white_timing.py, SHA256 36aaa7b6bc4a8cb212f63112a1b5494fc26282631644db233e204fbb8b432fc3
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Learn temporal change shapes from the two halves of real training records."""
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
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.linalg import eigh
from scipy.special import expit
from threadpoolctl import threadpool_limits
from partition_evidence import ROOT, OUT
from physical_870 import fit
import white_loop as wl

@lru_cache(maxsize=1024)
def time_basis(n):
    t = (np.arange(n) + 0.5) / n
    rows = [np.ones(n)] + [np.cos(np.pi * k * t) for k in range(1, 6)]
    for span in (1, 4, 12, 32):
        rows.append(np.exp(-np.arange(n) / span))
    v = np.stack(rows)
    return v / np.sqrt((v * v).sum(1, keepdims=True))

def signal(x, center, scale):
    z = (x - center) / scale
    s = z.clip(-2.7, 2.7)
    lag = np.r_[s[0], s[:-1]]
    mag = np.tanh(abs(z) - 0.75)
    return np.column_stack([s, np.tanh(2 * z) - 0.5 * s, np.log1p(z * z).clip(0, 5), np.tanh(3 * (abs(z) - 1.9)), np.cos(3 * z), np.sin(2 * z), s * lag, mag * np.r_[mag[0], mag[:-1]]])

def chart(raw, b):
    center = np.median(raw)
    scale = max(1.4826 * np.median(abs(raw - center)), 0.2 * raw.std(), 1e-07)
    a, c = (signal(x, center, scale) for x in (raw[:b], raw[b:]))
    avg = (a.sum(0) + c.sum(0)) / len(raw)
    sd = np.sqrt(((a - avg) ** 2).sum(0) / len(raw) + ((c - avg) ** 2).sum(0) / len(raw)).clip(0.09)
    a, c = (x - avg for x in (a, c))
    va = sd ** 2
    for k in (1, 2, 3, 4, 5):
        cov = ((a[:-k] * a[k:]).sum(0) + (c[:-k] * c[k:]).sum(0)) / len(raw)
        va = va + 2 * (1 - k / 6) * cov
    std = np.sqrt(np.maximum(va, 0.18 * sd ** 2))
    p = time_basis(len(a)) @ a[::-1] / std
    q = time_basis(len(c)) @ c / std
    return np.vstack([(q[0] - p[0]) / np.sqrt(2), p[1:], q[1:]]).T.clip(-28, 28)

def extract(part):
    path = OUT / f'timing_charts_{part}.npz'
    if path.exists():
        with np.load(path) as q:
            return (q['ids'], q['X'])
    frames = []
    for name in ('raw', 'pulse', 'state'):
        if name == 'raw':
            src, key = (ROOT / f'tune_out/seq_cache_{part}.npz', 'x')
        else:
            src, key = (OUT / f'{name}_streams_{part}.npz', 'separate')
        with np.load(src, allow_pickle=True) as q:
            ids, x, n, b = (q[k] for k in ('ids', key, 'n', 'b'))
        frames.append(np.stack([chart(v[:n1].astype(float), b1) for v, n1, b1 in zip(x, n, b)]).astype(np.float32))
        print(part, name, 'temporal charts', flush=True)
    f = np.concatenate(frames, axis=1)
    np.savez(path, ids=ids.astype(str), X=f)
    return (ids, f)

def contrasts(x, tr, zr, y, group):
    coords = []
    coeff = []
    pr = expit(zr)
    for g in (group, 1 - group):
        weights = [g[tr] * (1 - y) * pr, g[tr] * y * (1 - pr)]
        weights = [v / max(v.sum(), 1) for v in weights]
        for j in range(x.shape[1]):
            xx = x[:, j].clip(-10, 10)
            xt = xx[tr]
            cov = [xt.T @ (xt * v[:, None]) for v in weights]
            ridge = np.eye(xx.shape[1]) * 0.35
            _, V = eigh(cov[1] + ridge, cov[0] + ridge, check_finite=False)
            modes = V[:, [0, 1, -2, -1]]
            f = np.arcsinh(np.abs(xx @ modes))
            coords.append(f * g[:, None])
            coeff.append(modes)
    return (np.column_stack(coords), np.asarray(coeff))

def run():
    q, ids, tids, y, wt, wn, miss = wl.load()
    tri, xt = extract('train')
    tei, xv = extract('test')
    assert np.array_equal(tri, ids) and np.array_equal(tei, tids)
    x = np.concatenate([xt, xv])
    del xt, xv
    group = np.concatenate([pd.read_parquet(OUT / f'innov_v2_{p}.parquet').loc[ix, 'pre_white'].to_numpy() for p, ix in (('train', ids), ('test', tids))])
    te = len(y) + np.arange(len(tids))
    tr_res = np.empty((len(y), 2))
    ts_res = np.zeros((len(tids), 2))
    parameters = {}
    for i in range(5):
        key = f's0_f{i}_'
        tr, va, zr = (q[key + k] for k in ('tr', 'va', 'zr'))
        f, V = contrasts(x, tr, zr, y[tr], group)
        mean, sd = (f[tr].mean(0), f[tr].std(0).clip(0.12))
        f = ((f - mean) / sd).clip(-7, 7)
        for j, penalty in enumerate((360.0, 1800.0)):
            w = fit(f[tr], zr, y[tr], penalty)
            tr_res[va, j] = f[va] @ w
            ts_res[:, j] += f[te] @ w / 5
            parameters[f'fold{i}_w{j}'] = w
        parameters.update({f'fold{i}_V': V, f'fold{i}_mean': mean, f'fold{i}_sd': sd})
        print('temporal readout', i, flush=True)
    path = OUT / 'timing_crossfit.npz'
    np.savez_compressed(path, ids=tri.astype(str), tids=tei.astype(str), train=tr_res, test=ts_res, **parameters)
    f = np.vstack([tr_res, ts_res])

    def design(ix, z):
        return np.column_stack([f[ix], f[ix] * group[ix, None], f[ix] * np.tanh(z[:, None] / 2)])
    wl.run('timing', design, 60.0, dependencies=[Path(__file__).resolve(), path])
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        run()
