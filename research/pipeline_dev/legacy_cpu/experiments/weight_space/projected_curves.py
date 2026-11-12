# CPU port; original experiments/weight_space/projected_curves.py, SHA256 92f70e8e87fd842ac18d00a101ca72aff9df3e4ad50601df4cae2c5db52a5bd2
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Real-series quantile curves of joint lag and innovation measurements."""
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
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from partition_evidence import ROOT, OUT, fit_ar, P
Q = np.array([0.005, 0.015, 0.035, 0.065, 0.1, 0.16, 0.24, 0.35, 0.5, 0.65, 0.76, 0.84, 0.9, 0.935, 0.965, 0.985, 0.995])

def directions(n, seed):
    w = np.random.default_rng(seed).normal(size=(14, n))
    w /= np.linalg.norm(w, axis=1, keepdims=True)
    return np.vstack([np.eye(n), w]).T.astype(np.float32)
ODD = directions(9, 145)
EVEN = directions(8, 146)

def projected(z, e):
    zw = np.lib.stride_tricks.sliding_window_view(z, 9)
    ew = np.lib.stride_tricks.sliding_window_view(e, 9)
    u, v = (zw[:, -1], ew[:, -1])
    dz = u - zw[:, -2]
    odd = np.column_stack([u, v, zw[:, -2], zw[:, -3], zw[:, -5], dz, ew[:, -2], ew[:, -3], zw[:, :-1].mean(1)])
    even = np.column_stack([abs(u), abs(v), abs(dz), ew[:, :-1].std(1), zw[:, :-1].std(1), np.clip(u * zw[:, -2], -9, 9), np.clip(v * ew[:, -2], -9, 9), np.clip(abs(v) * abs(ew[:, -2]), 0, 9)])
    h = np.column_stack([odd.astype(np.float32) @ ODD, even.astype(np.float32) @ EVEN])
    return np.quantile(h, Q, axis=0).T

def standardize(a, b):
    joined = np.r_[a, b]
    m = np.median(joined)
    scale = max(np.median(abs(joined - m)) * 1.4826, 0.1 * joined.std(), 1e-07)
    return (np.clip((a - m) / scale, -8, 8), np.clip((b - m) / scale, -8, 8), scale)

def make(x, b):
    a, c = (x[:b], x[b:])
    u, v = fit_ar(a, c)
    za, zc, raw_scale = standardize(a[P:], c[P:])
    ea, ec, res_scale = standardize(u, v)
    curve = np.stack([projected(za, ea), projected(zc, ec)])
    ctx = [np.log(len(a)), np.log(len(c)), np.log(raw_scale), np.log(res_scale), abs(np.mean(x)), abs(a.mean() - c.mean()) / raw_scale, np.corrcoef(a[:-1], a[1:])[0, 1], np.corrcoef(c[:-1], c[1:])[0, 1]]
    return (curve, np.asarray(ctx, np.float32))

def run(part, extra):
    with np.load(ROOT / f'tune_out/seq_cache_{part}.npz', allow_pickle=True) as f:
        ids, x, n, b = (f[k] for k in ('ids', 'x', 'n', 'b'))
    index = pd.read_parquet(ROOT / f'clean/features_{part}.parquet', columns=[]).index
    assert np.array_equal(index, ids)
    if extra:
        labels = pd.read_csv(ROOT / 'data/train_labels.csv', dtype={'id': str}).set_index('id').loc[index, 'label'].to_numpy()
    curves = np.empty((len(ids), 2, ODD.shape[1] + EVEN.shape[1], len(Q)), np.float16)
    context = np.empty((len(ids), 8), np.float32)
    for i in range(len(ids)):
        s = x[i, :n[i]].astype(float)
        boundary = int(b[i])
        if extra:
            rng = np.random.default_rng(91003 + i)
            if labels[i] == 0:
                choices = np.arange(int(0.28 * n[i]), int(0.72 * n[i]))
                choices = choices[abs(choices - boundary) > 0.1 * n[i]]
                boundary = int(rng.choice(choices))
            else:
                lo = int(rng.uniform(0, 0.35) * boundary)
                hi = len(s) - int(rng.uniform(0, 0.35) * (len(s) - boundary))
                s = s[lo:hi]
                boundary -= lo
        curves[i], context[i] = make(s, boundary)
        if (i + 1) % 4000 == 0:
            print(part, extra, i + 1, flush=True)
    assert np.isfinite(curves).all() and np.isfinite(context).all()
    suffix = 'extra' if extra else part
    np.savez(OUT / f'projected_{suffix}.npz', ids=index.to_numpy(dtype=str), q=Q, odd=ODD, even=EVEN, curve=curves, context=context)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', choices=['train', 'test', 'extra', 'all'], default='all')
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        for item in ['train', 'test', 'extra'] if args.part == 'all' else [args.part]:
            run('train' if item == 'extra' else item, item == 'extra')
