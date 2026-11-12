# CPU port; original experiments/weight_space/other_transition_prior.py, SHA256 7382772031ad34fa65bd8e23e1f55a87f70aadc005b4cd6e66559b236944cc00
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Learn how frequently each proposed memory change appears in the two classes."""
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
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from other_response_gate import run
from partition_evidence import OUT
from white_local_rank_weights import weights

def mixture(likelihood, rows, initial):
    vals = likelihood[rows]
    prior = initial.copy()
    for _ in range(20):
        density = (vals @ prior).clip(1e-85)
        count = prior * (vals.T @ (1 / density))
        revised = (count + 220 * initial) / (len(rows) + 220)
        if max(abs(revised - prior)) < 2e-05:
            prior = revised
            break
        prior = revised
    return prior / prior.sum()

def factory(ids, tids, y):
    frames = []
    files = []
    for part, ix in (('train', ids), ('test', tids)):
        p = OUT / f'transition_catalog_{part}.parquet'
        frames.append(pd.read_parquet(p).loc[ix])
        files.append(p)
    tab = pd.concat(frames)
    likelihood = np.exp(tab.to_numpy(dtype=float))
    same = np.array([k.endswith('_same') for k in tab.columns])
    unit = np.array([k.startswith('unit_') for k in tab.columns])

    def prior(k):
        mass = 0.3 if k.startswith('zero_') else 0.12 if k.startswith('unit_') else 0.145
        if k.endswith('_same'):
            return mass * 0.68
        if k.endswith('_local'):
            return mass * 0.13
        return mass * 0.04
    start = np.array([prior(k) for k in tab.columns])
    start /= start.sum()

    def measure(tr, va):
        ps = [mixture(likelihood, tr[y[tr] == i], start) for i in (0, 1)]

        def marginal(w, mask):
            return np.log((likelihood[va][:, mask] @ w[mask]).clip(1e-85))
        allcol = np.ones(len(start), bool)
        full = marginal(ps[1], allcol) - marginal(ps[0], allcol)
        stay = marginal(ps[1], same) - marginal(ps[0], same)
        changed = marginal(ps[1], ~same) - marginal(ps[0], ~same)
        unit_start = marginal(ps[1], unit) - marginal(ps[0], unit)
        return np.column_stack([full, stay, changed, full - stay, unit_start]).clip(-8, 8)

    def fit(tr, va, z, v, unused_train, unused_valid):
        inn = np.empty((len(tr), 5))
        for a, b in StratifiedKFold(3, shuffle=True, random_state=682).split(tr, y[tr]):
            inn[b] = measure(tr[a], tr[b])

        def shape(h, score):
            return np.column_stack([h, 0.6 * h * np.tanh(score[:, None] / 1.6)])
        train = shape(inn, z)
        valid = shape(measure(tr, va), v)
        w = 0.65 * weights(train, z, y[tr])
        return (v + valid @ w, valid, w)
    return (fit, [*files, Path(__file__).resolve(), Path(__file__).with_name('other_transition_catalog.py')])
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        run(factory, 'local_rank_transition_prior')
