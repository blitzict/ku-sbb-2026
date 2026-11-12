# CPU port; original experiments/weight_space/innovation_library.py, SHA256 d40892e8c0c20bf4de2507a31ab11469ce4fdf80c136f7880680bedaee9c17dc
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Fit an empirical library of innovation laws to real training before segments.

Use the same family library to score the shared, changed-location, changed-scale
and independent-shape hypotheses. A Student-t reference supplies smooth tails.
The library sees neither validation cases nor class labels.
"""
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
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.special import gammaln, logsumexp
from sklearn.cluster import MiniBatchKMeans
from threadpoolctl import threadpool_limits
import expert_fusion as ef
from partition_evidence import fit_ar
import residual_trial as rt
from scipy.special import expit
GRID = np.linspace(-5, 5, 49)
STEP = GRID[1] - GRID[0]
COUNT = len(GRID)
MODES = ('mixture', 'mechanisms', 'context')

def t_logpdf(z):
    return gammaln(3) - gammaln(2.5) - 0.5 * np.log(3 * np.pi) - 3 * np.log1p(z * z / 3)

def interpolate_counts(z, weight=None):
    u = np.clip((z - GRID[0]) / STEP, 0, COUNT - 1)
    j = u.astype(int)
    f = u - j
    if weight is None:
        weight = 1.0
    return np.bincount(j, weight * (1 - f), minlength=COUNT) + np.bincount(np.minimum(j + 1, COUNT - 1), weight * f, minlength=COUNT)

def representation(x, loc, sd):
    sd = max(sd, 1e-06)
    z = (x - loc) / sd
    return np.r_[interpolate_counts(z), t_logpdf(z).sum() - len(x) * np.log(sd)]

def hypotheses(a, b):
    n, m = (len(a), len(b))
    ma, mb = (a.mean(), b.mean())
    va, vb = (max(a.var(), 1e-10), max(b.var(), 1e-10))
    mu = (n * ma + m * mb) / (n + m)
    within = (n * va + m * vb) / (n + m)
    global_var = within + n * m / (n + m) ** 2 * (ma - mb) ** 2
    common = mu
    for _ in range(4):
        p, q = (n / (va + (ma - common) ** 2), m / (vb + (mb - common) ** 2))
        common = (p * ma + q * mb) / (p + q)
    fits = [(mu, mu, global_var, global_var), (ma, mb, within, within), (common, common, va + (ma - common) ** 2, vb + (mb - common) ** 2), (ma, mb, va, vb)]
    return [[representation(a, l, np.sqrt(s)), representation(b, r, np.sqrt(t))] for l, r, s, t in fits]

def prepare(part):
    with np.load(ef.O / f'seq_cache_{part}.npz', allow_pickle=True) as a:
        ids, x, n, b = (a[k] for k in ('ids', 'x', 'n', 'b'))
    index = pd.read_parquet(ef.ROOT / f'clean/features_{part}.parquet', columns=[]).index
    assert np.array_equal(index, ids)
    H = np.empty((len(ids), 2, 4, 2, COUNT + 1), np.float32)
    for i in range(len(ids)):
        raw = x[i, :n[i]].astype(float)
        a, c = (raw[:b[i]], raw[b[i]:])
        H[i, 0] = hypotheses(a, c)
        H[i, 1] = hypotheses(*fit_ar(a, c))
        if (i + 1) % 6000 == 0:
            print(part, i + 1, flush=True)
    assert np.isfinite(H).all()
    np.savez(ef.OUT / f'innovation_library_{part}.npz', ids=index.to_numpy(dtype=str), H=H)

def fit_law(H, indices):
    before = H[indices, 3, 0, :COUNT].astype(float)
    freq = before / before.sum(1, keepdims=True)
    X = np.sqrt(gaussian_filter1d(freq, 0.85, axis=1).clip(0))
    km = MiniBatchKMeans(n_clusters=16, batch_size=1024, n_init=2, random_state=417, max_iter=100)
    labels = km.fit_predict(X)
    z = np.linspace(-100, 100, 40001)
    measure = np.exp(t_logpdf(z))
    measure /= measure.sum()
    base = interpolate_counts(z, measure)
    coeffs = []
    priors = np.bincount(labels, minlength=km.n_clusters).astype(float) + 1
    for k in range(km.n_clusters):
        count = before[labels == k].sum(0)
        raw = np.log((count + 800 * base) / ((count.sum() + 800) * base))
        c = gaussian_filter1d(raw, 0.8).clip(-4, 4)
        c -= np.log(np.sum(measure * np.exp(np.interp(z, GRID, c))))
        coeffs.append(c)
    return (np.array(coeffs), np.log(priors / priors.sum()))

def evidence(H, theta, prior):
    L = H[..., :COUNT] @ theta.T + H[..., COUNT, None]
    pooled = logsumexp(L.sum(axis=2) + prior, axis=-1)
    separate = logsumexp(L + prior, axis=-1).sum(axis=2)
    n = H[:, 0, :, :COUNT].sum(axis=-1)
    penalty = 0.5 * np.log(n.prod(axis=1) / n.sum(axis=1))
    level = pooled[:, 1] - pooled[:, 0] - penalty
    scale = pooled[:, 2] - pooled[:, 0] - penalty
    loc_scale = pooled[:, 3] - pooled[:, 0] - 2 * penalty
    shape = separate[:, 0] - pooled[:, 0]
    shape_standard = separate[:, 3] - pooled[:, 3]
    joint = separate[:, 3] - pooled[:, 0] - 2 * penalty
    mix = logsumexp(np.column_stack([level, scale, shape, loc_scale]) + np.log([0.35, 0.35, 0.2, 0.1]), axis=1)
    return np.column_stack([mix, level, scale, shape, shape_standard, loc_scale, joint])

def library_scores(H, tr, indices):
    fits = [fit_law(H[:, j], tr) for j in range(2)]
    return (np.column_stack([evidence(H[indices, j], *fits[j]) for j in range(2)]), fits)

def design(mode, X, z, context):
    cols = X[:, [0, 7]] if mode == 'mixture' else X
    parts = [cols, cols * np.tanh(z[:, None] / 2)]
    if mode == 'context':
        parts += [cols * context[:, j, None] for j in (2, 3, 6)]
    return np.column_stack(parts)

def main():
    data = ef.read_data()
    _, ids, tids, y, wt, _, _, _, _, _, _, D, _ = data
    banks = []
    for part, index in [('train', ids), ('test', tids)]:
        with np.load(ef.OUT / f'innovation_library_{part}.npz') as s:
            assert np.array_equal(s['ids'], index)
            banks.append(s['H'])
    H = np.concatenate(banks)
    del banks
    folds, full_z = rt.baseline(data)
    scores, saved = ({}, {})
    for split, parts in folds.items():
        pred = {k: np.empty(len(y)) for k in ['anchor', *MODES]}
        for fi, (tr, va, zr, zv) in enumerate(parts):
            measurements, _ = library_scores(H, tr, np.r_[tr, va])
            q = ef.distribution_maps(measurements[:len(tr)])
            X = ef.ranks(measurements, q)
            Cr = ef.distribution_maps(D[tr, :11])
            cr, cv = (ef.ranks(D[tr, :11], Cr), ef.ranks(D[va, :11], Cr))
            pred['anchor'][va] = zv
            for mode in MODES:
                th = ef.fit_head(design(mode, X[:len(tr)], zr, cr), zr, y[tr])
                pred[mode][va] = zv + design(mode, X[len(tr):], zv, cv) @ th[:-1] + th[-1]
            print(f'split {split} fold {fi}', flush=True)
        scores[split] = {k: ef.score(y, v, wt) for k, v in pred.items()}
        saved.update({f's{split}_{k}': v for k, v in pred.items()})
        print(split, scores[split], flush=True)
    eligible = [m for m in MODES if all((row[m][k] - row['anchor'][k] >= 0.0003 for row in scores.values() for k in ('auc', 'weighted_auc')))]
    chosen = max(eligible, key=lambda m: np.mean([row[m]['auc'] for row in scores.values()])) if eligible else None
    report = {'scores': scores, 'selected': chosen}
    (ef.OUT / 'innovation_library_results.json').write_text(json.dumps(report, indent=2))
    np.savez(ef.OUT / 'innovation_library_oof.npz', ids=ids.to_numpy(dtype=str), **saved)
    if chosen:
        dest = ef.PACKAGE / 'innovation_library_candidate'
        dest.mkdir(exist_ok=True)
        values, fits = library_scores(H, np.arange(len(y)), np.arange(len(H)))
        q = ef.distribution_maps(values[:len(y)])
        cq = ef.distribution_maps(D[:len(y), :11])
        context = ef.ranks(D[:, :11], cq)
        X = design(chosen, ef.ranks(values, q), full_z, context)
        th = ef.fit_head(X[:len(y)], full_z[:len(y)], y)
        path = dest / 'weights.npz'
        np.savez(path, mode=chosen, quantiles=q, context_ref=cq, theta=th, template=[v[0] for v in fits], prior=[v[1] for v in fits])
        ef.write_submission(expit(full_z[len(y):] + X[len(y):] @ th[:-1] + th[-1]), tids, dest / 'submission.csv')
        with np.load(path) as s:
            values = np.column_stack([evidence(H[len(y):, j], s['template'][j], s['prior'][j]) for j in range(2)])
            X = design(str(s['mode']), ef.ranks(values, s['quantiles']), full_z[len(y):], ef.ranks(D[len(y):, :11], s['context_ref']))
            th = s['theta']
        ef.write_submission(expit(full_z[len(y):] + X @ th[:-1] + th[-1]), tids, dest / 'submission_replay.csv')
        assert ef.sha(dest / 'submission.csv') == ef.sha(dest / 'submission_replay.csv')
    print('Selected', chosen, flush=True)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    with threadpool_limits(limits=2):
        if args.prepare:
            for part in ('train', 'test'):
                prepare(part)
        main()
