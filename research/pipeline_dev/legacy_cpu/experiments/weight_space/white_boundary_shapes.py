# CPU port; original experiments/weight_space/white_boundary_shapes.py, SHA256 d514c534fb4051f26164d91a59eb6208f96bfa8e58a2f0054ae82dc40f15205e
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Learn local boundary shapes from genuine breaks and controls, accounting for interior noise."""
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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from partition_evidence import OUT, ROOT

def windows(part):
    path = OUT / f'boundary_shapes_{part}.npz'
    if not path.exists():
        with np.load(ROOT / f'tune_out/seq_cache_{part}.npz', allow_pickle=True) as a:
            ids, raw, n, b = (a[k] for k in ('ids', 'x', 'n', 'b'))
            ids = ids.astype(str)
        with np.load(OUT / f'pulse_streams_{part}.npz') as a:
            assert np.array_equal(a['ids'], ids)
            clean = a['separate']
        state = pd.read_parquet(OUT / f'state_{part}.parquet').loc[ids]
        ar = state[['ar1', 'ar2']].to_numpy()
        result = np.empty((len(ids), 4, 3, 24), np.float32)
        for i, (length, cut) in enumerate(zip(n, b)):
            centers = np.array([cut, cut // 2, cut + (length - cut) // 2])
            locs = centers[:, None] + np.arange(-36, 36)
            assert locs.min() >= 0 and locs.max() < length
            for k, row in enumerate((raw[i, :length], clean[i, :length])):
                predictor = ar[i, 0] * np.r_[row[0], row[:-1]] + ar[i, 1] * np.r_[row[0], row[0], row[:-2]]
                innovation = row - predictor
                med = np.median(innovation)
                sigma = max(1.4826 * np.median(abs(innovation - med)), 0.15 * innovation.std(), 0.001)
                for j, (v, width) in enumerate(((innovation, 1), (row, 3))):
                    ids_time = centers[:, None] + np.arange(-12 * width, 12 * width)
                    x = v[ids_time].reshape(3, 24, width).mean(2)
                    x -= np.median(x[:, [0, 1, 2, 3, -4, -3, -2, -1]], axis=1)[:, None]
                    result[i, 2 * k + j] = np.arcsinh(x / sigma).clip(-5, 5)
        assert np.isfinite(result).all()
        np.savez_compressed(path, ids=ids, x=result)
    with np.load(path) as a:
        ids, x = (a['ids'], a['x'])
    gate = pd.read_parquet(OUT / f'innov_v2_{part}.parquet').loc[ids, ['pre_white', 'pre_walk']].to_numpy()
    return (ids, x, np.column_stack((np.ones(len(x)), gate)))

def project(training, scored, data, gate, y):
    out = []
    for channel in range(data.shape[1]):
        v = data[:, channel]
        dim = v.shape[-1]

        def covariance(ix, w, where):
            h = v[ix][:, where].reshape(-1, dim)
            weights = np.repeat(w, len(where))
            return h.T * weights @ h / np.maximum(weights.sum(), 1.0)
        for k in range(gate.shape[1]):
            cls = [training[y[training] == c] for c in (0, 1)]
            mats = []
            for ix in cls:
                w = gate[ix, k] ** 2
                center = covariance(ix, w, [0])
                inside = covariance(ix, w, [1, 2])
                mats.append((center, inside))
            reference = 0.5 * (mats[0][1] + mats[1][1])
            comparisons = [reference + 0.55 * (c - i) for c, i in mats]
            target = 0.5 * (comparisons[0] + comparisons[1])
            tau = max(0.12 * np.trace(reference) / dim, 0.025)
            target = (target + target.T) * 0.5 + tau * np.eye(dim)
            d = comparisons[1] - comparisons[0]
            vals, dirs = eigh((d + d.T) * 0.5, target)
            z = v[scored] @ dirs
            energy = np.log1p(z * z)
            for select in (slice(-4, None), slice(0, 4)):
                strength = vals[select]
                gain = (energy[:, :, select] * strength).sum(2)
                local = gain[:, 0]
                relative = local - gain[:, 1:].mean(1)
                out.extend([gate[scored, k] * local, gate[scored, k] * relative])
    return np.column_stack(out)

def run():
    import json
    from white_checkpoint import SEEDS
    from white_local_rank_weights import centered
    from white_nested import audit, fit
    import white_loop as wl
    q, ids, tids, y, wt, wn, miss = wl.load()
    a, x, g = windows('train')
    assert np.array_equal(a, ids)
    cache = OUT / 'frozen_local_rank_early_mode_peaks'
    reports = {}
    for s in SEEDS:
        old, new = (np.empty(len(y)), np.empty(len(y)))
        for f in range(5):
            k = f's{s}_f{f}_'
            tr, va = (q[k + 'tr'], q[k + 'va'])
            with np.load(cache / f's{s}_f{f}.npz') as p:
                assert np.array_equal(p['train_ids'], ids[tr]) and np.array_equal(p['validation_ids'], ids[va])
                u, v = (p['oof_train'], p['valid'])
            h = project(tr, va, x, g, y)
            inner = np.empty((len(tr), h.shape[1]))
            for aa, bb in StratifiedKFold(3, shuffle=True, random_state=803 + f).split(tr, y[tr]):
                inner[bb] = project(tr[aa], tr[bb], x, g, y)
            tx, r = centered(inner, u)
            val, _ = centered(h, v, r)
            new[va] = v + val @ fit(tx, u, y[tr], True)
            old[va] = v
        reports[s] = dict(parent=wl.score(y, old, wt, wn, miss), new=wl.score(y, new, wt, wn, miss), recovery=audit(y, miss, old, new))
        print('boundary covariance', s, reports[s]['new'], flush=True)
    (OUT / 'boundary_shape_trial.json').write_text(json.dumps(reports, indent=2))
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        run()
