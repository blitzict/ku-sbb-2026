# CPU port; original experiments/weight_space/hard_case_signals.py, SHA256 b1e99b87b55eab82d52b1e638a7ec97ad6f54b957e719306eb3723a84e58003c
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Two label-free checks for weak AR transitions and repeated noise-shape changes."""
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
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits
from partition_evidence import OUT
RATIOS = np.exp(np.array([-0.65, -0.35, -0.15, 0.0, 0.15, 0.35, 0.65]))
RATIO_P = np.array([0.07, 0.15, 0.16, 0.24, 0.16, 0.15, 0.07])

def ar_evidence(raw, cut, order):
    scale = max(np.median(abs(raw - np.median(raw))) * 1.4826, 0.15 * raw.std(), 1e-05)
    v = (raw - np.median(raw)) / scale
    y = v[order:]
    X = np.column_stack([np.ones(len(y))] + [v[order - j:len(v) - j] for j in range(1, order + 1)])
    a, b = (X[:cut - order], X[cut - order:])
    left, right = (y[:len(a)], y[len(a):])
    k = order + 1
    G = np.zeros((2 * k, 2 * k))
    G[:k, :k] = a.T @ a
    H = np.tile(b.T @ b, (2, 2))
    c = np.r_[a.T @ left, np.zeros(k)]
    d = np.tile(b.T @ right, 2)
    W = RATIOS ** (-2)
    gram = G[None, :, :] + W[:, None, None] * H
    cross = c[None, :] + W[:, None] * d
    yy = left @ left + W * (right @ right)

    def evidence(active, sds):
        g = gram[:, active][:, :, active]
        u = cross[:, active]
        prior = np.asarray(sds)
        mat = g + np.diag(prior ** (-2))
        proj = np.linalg.solve(mat, u[:, :, None])[:, :, 0]
        det = np.linalg.slogdet(mat)[1] + 2 * np.log(prior).sum()
        err = (yy - (proj * u).sum(1)).clip(0)
        return -0.5 * det - (2.0 + len(y) / 2) * np.log(1.0 + 0.5 * err) - len(right) * np.log(RATIOS)
    base = list(range(k))
    std = [2.0, 0.55] + ([0.35] if order == 2 else [])
    ref = evidence(base, std)
    null = ref[3]
    scale_only = logsumexp(ref + np.log(RATIO_P)) - null
    changes = {}
    for name, which, sigma in (('level', [k], [0.6]), ('memory', list(range(k + 1, 2 * k)), [0.22] * order), ('joint', list(range(k, 2 * k)), [0.6] + [0.22] * order)):
        vals = evidence(base + which, std + sigma)
        changes[name] = vals[3] - null
        changes[name + '_scale'] = logsumexp(vals + np.log(RATIO_P)) - null
    changes['scale'] = scale_only
    one = logsumexp([changes['level'], changes['memory'], scale_only]) - np.log(3)
    changes['joint_excess'] = changes['joint_scale'] - one
    return changes

def noise_shape(raw, cut):
    a, b = (raw[2:cut], raw[cut:])

    def basis(x):
        med = np.median(x)
        scale = max(np.median(abs(x - med)) * 1.4826, 0.12 * x.std(), 1e-05)
        z = (x - med) / scale
        return np.column_stack([np.cos(z), np.cos(2 * z), np.cos(3 * z), np.sin(z), np.sin(2 * z), abs(z) > 1.5, abs(z) > 2.5, abs(z) > 3.5, np.sign(z) * (abs(z) > 1.8)])
    u, v = (basis(a), basis(b))
    scale = np.maximum(np.vstack([u, v]).std(0), 0.045)
    out = {}
    for length in (13, 32):
        groups = []
        for phase in (0, 1):
            s = u[np.arange(len(u)) // length % 2 == phase]
            t = v[np.arange(len(v)) // length % 2 == phase]
            ne = np.sqrt(len(s) * len(t) / (len(s) + len(t)))
            groups.append((s.mean(0) - t.mean(0)) * ne / scale)
        h, j = groups
        prod = np.sign(h * j) * np.minimum(abs(h), abs(j))
        for name, ix in [('core', [0, 1, 2]), ('skew', [3, 4, 8]), ('tail', [5, 6, 7])]:
            p = prod[ix]
            out[f'{name}_{length}_repeat'] = p.mean()
            out[f'{name}_{length}_strong'] = p.max()
            out[f'{name}_{length}_mismatch'] = np.mean(abs(h[ix] - j[ix]))
    return out

def run(part):
    with np.load(OUT / f'pulse_streams_{part}.npz') as a:
        ids, x, n, b = (a[k] for k in ('ids', 'separate', 'n', 'b'))
    with np.load(OUT / f'state_streams_{part}.npz') as a:
        assert np.array_equal(ids, a['ids']) and np.array_equal(b, a['b']) and np.array_equal(n, a['n'])
        v = a['separate']
    rows = []
    for i in range(len(x)):
        row = x[i, :n[i]].astype(float)
        f = {f'ar{k}_{key}': val for k in (1, 2) for key, val in ar_evidence(row, b[i], k).items()}
        f.update(noise_shape(v[i, :n[i]].astype(float), b[i]))
        rows.append(f)
    frame = pd.DataFrame(rows, index=pd.Index(ids, name='id'))
    assert np.isfinite(frame).all().all()
    frame.to_parquet(OUT / f'hard_signals_{part}.parquet')
    print(part, frame.shape, flush=True)
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        run('train')
        run('test')
