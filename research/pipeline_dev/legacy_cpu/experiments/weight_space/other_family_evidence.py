# CPU port; original experiments/weight_space/other_family_evidence.py, SHA256 69d13195ac25aa7ac97c2e49b94e2da287fd730073c0c38c4c216b26d8b44c7c
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Estimate when another detector adds evidence at the incumbent's risk level."""
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
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.special import expit
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from other_miss_witnesses import make
from other_response_gate import run
from partition_evidence import ROOT
from white_local_rank_weights import weights

def percentile(reference, query):
    sorted_scores = np.sort(reference)
    return (np.searchsorted(sorted_scores, query) + 0.5) / (len(reference) + 1)

def lookup(train, query, y, g, h):
    nz, nd = (16, 21)

    def position(x):
        a = (x[:, 0] * nz - 0.5).clip(0, nz - 1)
        b = ((np.arcsinh(x[:, 1] / 0.12) / np.arcsinh(1 / 0.12) + 1) * nd / 2 - 0.5).clip(0, nd - 1)
        return np.stack([a, b])
    locations = position(train)
    cell = np.rint(locations).astype(int)
    query_loc = position(query)
    dist = []
    for cls in (0, 1):

        def histogram(mask):
            grid = np.zeros((nz, nd))
            np.add.at(grid, (cell[0, mask], cell[1, mask]), 1.0)
            return gaussian_filter(grid, (1.15, 1.35), mode='reflect')
        full = histogram(y == cls) + 0.55
        full /= full.sum(1, keepdims=True)
        log_density = np.empty(len(query))
        for family in np.unique(h):
            sel = h == family
            num = histogram((g == family) & (y == cls)) + 45 * full
            num /= num.sum(1, keepdims=True)
            log_density[sel] = map_coordinates(np.log(num), query_loc[:, sel], order=1, mode='nearest')
        dist.append(log_density)
    return (dist[1] - dist[0]).clip(-2.5, 2.5)

def factory(ids, tids, y):
    groups, files = make(ids, tids)
    E = np.column_stack([b.mean(1) for b in groups])
    path = ROOT / 'tune_out/generator_segments.parquet'
    gen = pd.read_parquet(path, columns=['id', 'part', 'phi1_ar1', 'p_bic'])
    gen = gen[gen.part == 'pre'].set_index('id').loc[ids.append(tids)]
    phi = gen.phi1_ar1.to_numpy()
    family = np.select([phi > 0.95, gen.p_bic.to_numpy() == 0, gen.p_bic.to_numpy() >= 5, phi < -0.2, phi > 0.2], [0, 1, 2, 3, 4], 5)

    def estimate(tr, va, z, v):
        ix = np.r_[tr, va]
        n = len(tr)
        neg = y[tr] == 0
        parent = percentile(z[neg], np.r_[z, v])
        scores = np.column_stack([percentile(E[tr[neg], j], E[ix, j]) for j in range(E.shape[1])])
        raw = np.column_stack([scores, scores.mean(1), np.sort(scores, axis=1)[:, -2]])
        features = []
        for vote in raw.T:
            both = np.column_stack([parent, vote - parent])
            features.append(lookup(both[:n], both[n:], y[tr], family[tr], family[va]))
        return np.column_stack(features)

    def fit(tr, va, z, v, unused_train, unused_valid):
        inner = np.empty((len(tr), E.shape[1] + 2))
        for a, b in StratifiedKFold(3, shuffle=True, random_state=6517).split(tr, y[tr]):
            inner[b] = estimate(tr[a], tr[b], z[a], z[b])
        validation = estimate(tr, va, z, v)
        w = 0.7 * weights(inner, z, y[tr])
        return (v + validation @ w, validation, w)
    return (fit, [*files, path, Path(__file__).resolve()])
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        run(factory, 'local_rank_conditional_detector_evidence')
