# CPU port; original experiments/weight_space/white_cube_components.py, SHA256 1849dfd89f0909e2d435ce917c4be21b14362e8fa9a484ebe56d798aa2834bd9
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Decompose the original cubic innovation test into drift, skew and isolated shocks."""
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

def residuals(x, b, p):
    pre = x[:b]
    z = (x - pre.mean()) / max(pre.std(), 1e-06)
    if p:
        v = np.lib.stride_tricks.sliding_window_view(z, p + 1)
        D = np.column_stack([v[:, :-1][:, ::-1], np.ones(len(v))])
        X = D[:b - p]
        beta = np.linalg.solve(X.T @ X + 0.0001 * np.eye(p + 1), X.T @ v[:b - p, -1])
        r = v[:, -1] - D @ beta
    else:
        r = z - z[:b].mean()
    k = b - p
    return (r / max(np.sqrt(np.mean(r[:k] ** 2)), 1e-06), k)

def pieces(v):
    mu = v.mean()
    e = v - mu
    return np.array([mu ** 3, 3 * mu * np.mean(e ** 2), np.mean(e ** 3)])

def measure(x, cut, p):
    r, b = residuals(x, cut, p)
    a, c = (r[:b], r[b:])
    scale = max((a ** 3).std(), 1e-06) * np.sqrt(1 / len(a) + 1 / len(c))
    orig = (np.mean(c ** 3) - np.mean(a ** 3)) / scale
    d = (pieces(c) - pieces(a)) / scale
    assert abs(orig - d.sum()) < 1e-05 * (1 + abs(orig))
    cubed = c ** 3
    j = np.argmax(abs(cubed))
    largest = cubed[j] / (len(c) * scale)
    clipped = np.clip(r, -3.2, 3.2)
    ac, cc = (clipped[:b], clipped[b:])
    D = (pieces(cc) - pieces(ac)) / scale
    halves = []
    for fold in (0, 1):
        subsets = [v[np.arange(len(v)) * 8 // len(v) % 2 == fold] for v in (ac, cc)]
        u, w = subsets
        dd = pieces(w) - pieces(u)
        zsd = max((u ** 3).std(), 0.25) * np.sqrt(1 / len(u) + 1 / len(w))
        halves.append(dd / zsd)
    corroboration = np.prod(halves, axis=0)
    raw_local = abs(c[:min(8, len(c))]) ** 3
    return np.r_[orig, d, largest, orig - largest, abs(orig - largest), float(j < 8) * largest, raw_local.sum() / (len(c) * scale), D, corroboration, np.mean(c > 2.5) - np.mean(a > 2.5), np.mean(c < -2.5) - np.mean(a < -2.5)]

def extract(part):
    target = OUT / f'cube_components_{part}.parquet'
    if target.exists():
        return
    with np.load(OUT / f'innov_v2_streams_{part}.npz') as q:
        ids, x, n, b = (q[k] for k in ('ids', 'separate', 'n', 'b'))
    original = pd.read_parquet(OUT / f'noise_tests_{part}.parquet', columns=['lr_p_bic', 'lr_cube_t']).loc[ids]
    D = np.stack([measure(v[:end].astype(float), j, int(p)) for v, end, j, p in zip(x, n, b, original.lr_p_bic)])
    relative = abs(D[:, 0] - original.lr_cube_t.to_numpy()) / (1 + abs(D[:, 0]))
    print(part, 'archived cubic test relative discrepancy', np.quantile(relative, [0.5, 0.99, 1]), flush=True)
    assert np.quantile(relative, 0.99) < 0.001 and np.max(relative) < 0.08
    names = ['original', 'mean_cubed', 'mean_times_variance', 'central_third', 'max_post_shock', 'after_removing_max', 'abs_after_removing_max', 'max_shock_near_cut', 'near_abs_cube', 'clipped_mean_cubed', 'clipped_mean_times_variance', 'clipped_central_third', 'repeated_mean_cubed', 'repeated_mean_times_variance', 'repeated_central_third', 'up_tail_frequency', 'down_tail_frequency']
    df = pd.DataFrame(D, index=pd.Index(ids, name='id'), columns=names)
    assert df.index.is_unique and np.isfinite(df).all().all()
    df.astype(np.float32).to_parquet(target)
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        for part in ('train', 'test'):
            extract(part)
