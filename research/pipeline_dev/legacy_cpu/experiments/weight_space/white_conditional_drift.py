# CPU port; original experiments/weight_space/white_conditional_drift.py, SHA256 ee1a8020bc0ccbb08c8de618af9335b95f5a3fd717fd07fcea737ab3ec820925
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Look for changes in how a shock predicts the next step at different amplitudes."""
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
from scipy.special import expit
from threadpoolctl import threadpool_limits
from partition_evidence import OUT

def history(x):
    d = x - np.median(x)
    scale = max(1.4826 * np.median(abs(d)), 0.22 * d.std(), 1e-05)
    z = (d / scale).clip(-4.0, 4.0)
    first, second = (z[1:-1], z[:-2])
    linear = np.column_stack([np.ones(len(first)), first, second])
    extra = np.column_stack([abs(first), np.tanh(first) - 0.6 * first, np.maximum(abs(first) - 1.2, 0) * np.sign(first), np.tanh(first) * np.tanh(second), np.tanh(second) * abs(np.tanh(first)), np.tanh(first - second) * np.tanh(abs(first) - 0.8)])
    X = np.column_stack([linear, extra])
    block = np.arange(len(x)) * 8 // len(x)
    ok = block[2:] == block[:-2]
    return [(X[ok & (block[2:] % 2 == i)], z[2:][ok & (block[2:] % 2 == i)]) for i in (0, 1)]

def stats(x, y):
    a = x[:, :3]
    wa = np.linalg.solve(a.T @ a + np.diag([1.0, 4.0, 4.0]), a.T @ np.column_stack([y, x[:, 3:]]))
    resid = y - a @ wa[:, 0]
    h = x[:, 3:] - a @ wa[:, 1:]
    noise = max(np.mean(resid ** 2), 0.04)
    return (wa, h.T @ h / noise, h.T @ resid / noise, noise)

def measure(x, cut):
    sides = [history(v) for v in (x[:cut], x[cut:])]
    records = []
    for a, b in ((0, 1), (1, 0)):
        models = [stats(*side[a]) for side in sides]
        gram = [m[1] for m in models]
        cross = [m[2] for m in models]
        prior = np.eye(gram[0].shape[0]) * 12.0
        learned = [np.linalg.solve(g + prior, c) for g, c in zip(gram, cross)]
        shared = np.linalg.solve(sum(gram) + prior, sum(cross))
        result = []
        for i, side in enumerate(sides):
            X, y = side[b]
            nuisance, G, C, noise = models[i]
            h = X[:, 3:] - X[:, :3] @ nuisance[:, 1:]
            r = y - X[:, :3] @ nuisance[:, 0]

            def gain(w):
                d = h @ w
                return (r @ d - 0.5 * (d @ d)) / noise
            result.append([gain(learned[i]), gain(learned[1 - i]), gain(shared)])
        records.append(result)
    vals = np.array(records)
    own, other, pool = vals.sum(0).T
    each = vals[:, :, 0]
    return np.r_[own, each.min(0), other, pool, own - other, own.sum() - pool.sum(), (vals[:, :, 0] - vals[:, :, 2]).sum(1).min()]
NAMES = ['pre', 'post', 'pre_repeat', 'post_repeat', 'post_to_pre', 'pre_to_post', 'common_pre', 'common_post', 'pre_specific', 'post_specific', 'change', 'change_repeat']

def main():
    import white_loop as wl
    q, idx, tids, y, wt, wn, miss = wl.load()
    for name, ix in (('train', idx), ('test', tids)):
        path = OUT / f'conditional_drift_{name}.parquet'
        if not path.exists():
            frames = []
            for channel in ('pulse', 'state'):
                with np.load(OUT / f'{channel}_streams_{name}.npz') as a:
                    assert np.array_equal(a['ids'], ix)
                    x, n, c = (a[k] for k in ('separate', 'n', 'b'))
                v = [measure(z[:end].astype(float), cut) for z, end, cut in zip(x, n, c)]
                frames.append(pd.DataFrame(v, index=ix, columns=NAMES).add_prefix(channel + '_'))
            frame = pd.concat(frames, axis=1)
            assert np.isfinite(frame).all().all()
            frame.astype(np.float32).to_parquet(path)
        if name == 'train':
            tab = pd.read_parquet(path).loc[ix]
            for channel in ('pulse', 'state'):
                block = tab.filter(like=channel + '_').copy()
                block.columns = block.columns.str.removeprefix(channel + '_')
                flag = (block.pre > 5) & (block.pre_repeat > 0)
                for name2, group in (('negative', y == 0), ('white negative', (y == 0) & wn), ('historical', miss)):
                    g = group & flag
                    print(channel, name2, int(g.sum()), 'also predicts post', int((g & (block.pre_to_post > 0)).sum()), flush=True)
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
