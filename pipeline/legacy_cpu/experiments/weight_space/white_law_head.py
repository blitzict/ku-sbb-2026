# CPU port; original experiments/weight_space/white_law_head.py, SHA256 12b1cda3e2be921304dd4f4e61c6d6665ec9a0d5684a2b5dd46f1c3b1236aa11
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Cross-fit the latent intervention frequencies from real examples, then route their evidence."""
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
from scipy.special import logsumexp
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
import white_loop as wl
from partition_evidence import OUT, ROOT
from white_laws import LAWS, FAMILIES

def responsibilities(M, p):
    q = M + np.log(p + 1e-24)
    return np.exp(q - logsumexp(q, axis=1, keepdims=True))

def weights(M, prior, mass):
    p = prior
    for _ in range(40):
        resp = responsibilities(M, p)
        new = (resp.sum(0) + mass * prior) / (len(M) + mass)
        if np.max(abs(p - new)) < 1e-05:
            break
        p = new
    return p

def unpack(M, law):
    k = len(LAWS)
    nf = len(FAMILIES)
    index = [np.array([j for j, (tag, _, _) in enumerate(LAWS) if tag == g]) for g in FAMILIES]
    same, loc, scale, joint = np.split(M[:, :4 * k], 4, axis=1)
    cross = M[:, 4 * k:4 * k + nf * nf]
    a, b = (M[:, -2 * k:-k], M[:, -k:])
    fam = np.array([law[ix].sum() for ix in index])
    partial = [logsumexp(v + np.log(law), 1) for v in (same, loc, scale, joint)]
    switch = logsumexp(cross + np.log(np.outer(fam, fam).ravel()), 1)
    null = partial[0]
    tests = np.column_stack([v - null for v in (*partial[1:], switch)])
    probs = np.r_[np.repeat([0.58, 0.13, 0.13, 0.04], k) * np.tile(law, 4), 0.12 * np.outer(fam, fam).ravel()]
    all_terms = np.column_stack([same, loc, scale, joint, cross])
    all_terms -= all_terms.max(1, keepdims=True)
    ra, rb = (responsibilities(a, law), responsibilities(b, law))
    pa = np.column_stack([ra[:, ix].sum(1) for ix in index])
    pb = np.column_stack([rb[:, ix].sum(1) for ix in index])
    f = [tests, logsumexp(tests, 1) - np.log(4), pa, pb, pa * pb]
    for j, ix in enumerate(index):
        base = logsumexp(same[:, ix] + np.log(law[ix]), 1)
        f.append(np.column_stack([(logsumexp(v[:, ix] + np.log(law[ix]), 1) - base).clip(-15, 40) for v in (loc, scale, joint)]) * pa[:, [j]])
    return (probs, all_terms, np.column_stack(f))

def prepare(tag):
    ids, vs = ([], [])
    for part in ('train', 'test'):
        a = np.load(OUT / f'wnlaw_{tag}_{part}.npz')
        ids.append(pd.Index(a['ids']))
        vs.append(a['values'].astype(float))
    M = np.vstack(vs)
    n = len(ids[0])
    k = len(LAWS)
    g = pd.read_parquet(ROOT / 'tune_out/generator_segments.parquet', columns=['id', 'part', 'p_bic'])
    wn = g[g.part == 'pre'].set_index('id').loc[ids[0], 'p_bic'].to_numpy() == 0
    y = pd.read_csv(ROOT / 'data/train_labels.csv', dtype={'id': str}).set_index('id').loc[ids[0], 'label'].to_numpy()
    before = M[:, -2 * k:-k]
    ref = np.array([{'normal': 0.43, 'student': 0.24, 'contam': 0.13, 'bimodal': 0.16, 'expower': 0.04}[key] / sum((v[0] == key for v in LAWS)) for key, _, _ in LAWS])
    O = None
    preds = []
    params = []
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=0).split(np.arange(n), y):
        ix = tr[wn[tr]]
        law = weights(before[ix], ref, 150.0)
        p, L, F = unpack(M, law)
        P = [weights(L[ix[y[ix] == label]], p, 180.0) for label in (0, 1)]
        lr = logsumexp(L + np.log(P[1]), 1) - logsumexp(L + np.log(P[0]), 1)
        f = np.column_stack([lr, F]).clip(-80, 80)
        if O is None:
            O = np.full((n, f.shape[1]), np.nan)
        O[va] = f[va]
        preds.append(f[n:])
        params.append({'law': law.tolist(), 'null': P[0].tolist(), 'change': P[1].tolist()})
    assert np.isfinite(O).all()
    (OUT / f'wnlaw_{tag}_priors.json').write_text(json.dumps(params))
    for part, idset, f in (('train', ids[0], O), ('test', ids[1], np.mean(preds, 0))):
        pd.DataFrame(f, index=idset, columns=[f'v{j}' for j in range(f.shape[1])]).to_parquet(OUT / f'wnlaw_{tag}_scores_{part}.parquet')

def trial(tag, extra):
    q, ids, tids, *_ = wl.load()
    f = np.vstack([pd.read_parquet(OUT / f'wnlaw_{tag}_scores_{part}.parquet').loc[index].to_numpy() for part, index in (('train', ids), ('test', tids))])

    def field(name, col):
        return np.r_[*[pd.read_parquet(OUT / f'{name}_{part}.parquet').loc[index, col].to_numpy() for part, index in (('train', ids), ('test', tids))],]
    pre = field('innov_v2', 'pre_white')
    joint = field('innov_v2', 'white')
    walk = field('innov_v2', 'pre_walk')
    per = field('periodic', 'gate')
    tr = field('curve', 'gate')
    pre2 = field('state', 'pre2')
    gate = (1 - walk) * (1 - per) * (0.15 + 0.85 * np.maximum(pre, joint)) * (1 - 0.7 * pre2)

    def design(ix, z):
        H = f[ix]
        d = (H[:, 0] - z).clip(-8, 8)
        vals = [np.ones(len(z)), np.tanh(z / 2), d, d * np.tanh(z / 2), np.arcsinh(H[:, 1:6].clip(-15, 45)), H[:, 6:21], np.arcsinh(H[:, 21:].clip(-15, 40)), tr[ix] * d]
        return gate[ix, None] * np.column_stack(vals)
    paths = [Path(__file__).resolve(), Path(__file__).with_name('white_laws.py').resolve()]
    paths.extend((OUT / f'wnlaw_{tag}_scores_{part}.parquet' for part in ('train', 'test')))
    wl.run('law_' + tag, design, 100.0, (7, 29, 0, 123, 2024, 11041) if extra else (7, 29, 0), paths)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--tag', default='physical')
    p.add_argument('--prepare', action='store_true')
    p.add_argument('--verify', action='store_true')
    a = p.parse_args()
    with threadpool_limits(limits=2):
        if a.prepare:
            prepare(a.tag)
        trial(a.tag, a.verify)
