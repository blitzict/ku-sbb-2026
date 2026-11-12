# CPU port; original experiments/weight_space/population_null.py, SHA256 cf9fb7d449e61d0b287eb2c2a1be3e01c3ffe470a72503d9996ca72def6a0d01
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Conditional empirical likelihoods estimated from other real labelled series."""
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
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.special import expit
from threadpoolctl import threadpool_limits
import expert_fusion as ef
import residual_trial as rt
K = 160
MODES = ['null', 'density', 'combined']

def summaries(ids, tids):
    columns = ['part', 'id', 'n', 'mean', 'innov_sd4', 'phi1_ar1', 'persist4', 'kurt', 'skew', 'sqacf1', 'phi1', 'phi2', 'phi3', 'phi4', 'tail2', 'tail3', 'spec_peak_share']
    raw = pd.read_parquet(ef.O / 'generator_segments.parquet', columns=columns)
    ix = ids.append(tids)
    a, b = [raw.loc[raw.part == side].set_index('id').loc[ix] for side in ['pre', 'post']]
    assert a.index.equals(ix) and b.index.equals(ix) and ix.is_unique
    n, m = (a.n.values, b.n.values)
    size = np.sqrt(n * m / (n + m))
    sigma = a.innov_sd4.values.clip(0.02)
    mu, nu = (a['mean'].values, b['mean'].values)
    scale = np.log(b.innov_sd4.values.clip(0.02) / sigma)
    signed = (nu - mu) * size * (1 - a.persist4.values.clip(-1.5, 0.99)) / sigma
    desc = np.column_stack([np.log(size), a.phi1_ar1, a.phi2, a.sqacf1, a['kurt'], a.spec_peak_share, sigma, abs(mu), abs(a['skew'])])
    stats = [abs(signed), signed * np.sign(mu), abs(nu), abs(nu) - abs(mu), scale, np.abs(scale), scale * np.sign(np.log(sigma)), b.innov_sd4, abs(b['skew'] - a['skew']), b['kurt'] - a['kurt'], abs(b.tail2 - a.tail2), abs(b.tail3 - a.tail3)]
    stats += [abs(b[c] - a[c]) * size for c in ['phi1', 'phi2', 'phi3', 'phi4', 'sqacf1', 'spec_peak_share']]
    return (desc, np.column_stack(stats))

def local_evidence(C, S, y, train_indices, query_indices, k=K):
    """Remove a training query from its own class bank before using any labels."""
    arrays = []
    for label in (0, 1):
        ref = train_indices[y[train_indices] == label]
        tree = cKDTree(C[ref])
        blocks = []
        for start in range(0, len(query_indices), 192):
            query = query_indices[start:start + 192]
            _, neighbors = tree.query(C[query], k=k + 1, workers=1)
            rows = ref[neighbors]
            valid = rows != query[:, None]
            selected = valid & (np.cumsum(valid, axis=1) <= k)
            rows = rows[selected].reshape(len(query), k)
            assert not np.any(rows == query[:, None])
            delta = S[rows] - S[query, None, :]
            rank = ((delta < 0).sum(1) + 0.5 * (delta == 0).sum(1) + 1) / (k + 2)
            kernel = (1 / (1 + (delta / 0.13) ** 2)).sum(axis=1) + 2.0
            blocks.append(np.column_stack([rank, kernel]))
        arrays.append(np.vstack(blocks))
    rank0, kernel0 = np.split(arrays[0], 2, axis=1)
    _, kernel1 = np.split(arrays[1], 2, axis=1)
    surprise = -np.log(2 * np.minimum(rank0, 1 - rank0))
    lr = np.log(kernel1 / kernel0)
    return (surprise, lr)

def design(mode, parts, z):
    s, lr = parts
    cols = []
    if mode in ('null', 'combined'):
        cols.extend([s, s * np.tanh(z[:, None] / 2)])
    if mode in ('density', 'combined'):
        cols.extend([lr, lr * np.tanh(z[:, None] / 2), np.max(lr, axis=1, keepdims=True)])
    return np.column_stack(cols)

def compute(C, S, y, tr, query):
    qc, qs = (ef.distribution_maps(C[tr]), ef.distribution_maps(S[tr]))
    cn, sn = (ef.ranks(C, qc), ef.ranks(S, qs))
    cn *= np.array([0.6, 1.5, 0.8, 0.7, 1.1, 0.6, 1.0, 0.7, 0.7])
    return local_evidence(cn, sn, y, tr, query)

def main():
    data = ef.read_data()
    _, ids, tids, y, wt, *_ = data
    C, S = summaries(ids, tids)
    folds, full = rt.baseline(data)
    scores, saved = ({}, {})
    for split, batches in folds.items():
        pred = {k: np.empty(len(y)) for k in ['anchor', *MODES]}
        for fi, (tr, va, zr, zv) in enumerate(batches):
            result = compute(C, S, y, tr, np.r_[tr, va])
            rr, rv = [tuple((x[p] for x in result)) for p in (slice(0, len(tr)), slice(len(tr), None))]
            pred['anchor'][va] = zv
            for mode in MODES:
                th = ef.fit_head(design(mode, rr, zr), zr, y[tr])
                pred[mode][va] = zv + design(mode, rv, zv) @ th[:-1] + th[-1]
            print(f'split {split} fold {fi} complete', flush=True)
        scores[split] = {k: ef.score(y, v, wt) for k, v in pred.items()}
        saved.update({f's{split}_{k}': v for k, v in pred.items()})
        print(split, scores[split], flush=True)
    eligible = [m for m in MODES if all((row[m][k] - row['anchor'][k] >= 0.0003 for row in scores.values() for k in ('auc', 'weighted_auc')))]
    chosen = max(eligible, key=lambda m: np.mean([row[m]['auc'] for row in scores.values()])) if eligible else None
    report = {'scores': scores, 'selected': chosen}
    (ef.OUT / 'population_null_results.json').write_text(json.dumps(report, indent=2))
    np.savez(ef.OUT / 'population_null_oof.npz', ids=ids.to_numpy(dtype=str), **saved)
    if chosen:
        dest = ef.PACKAGE / 'population_null_candidate'
        dest.mkdir(exist_ok=True)
        tr = np.arange(len(y))
        parts = compute(C, S, y, tr, np.arange(len(full)))
        F = design(chosen, parts, full)
        th = ef.fit_head(F[tr], full[tr], y)
        path = dest / 'weights.json'
        inputs = [rt.ANCHOR, ef.O / 'generator_segments.parquet', ef.ROOT / 'data/train_labels.csv', Path(__file__).resolve()]
        recipe = {'mode': chosen, 'weights': th.tolist(), 'inputs': {str(p.relative_to(ef.ROOT)): ef.sha(p) for p in inputs}, **report}
        path.write_text(json.dumps(recipe, indent=2))
        ef.write_submission(expit(full[len(y):] + F[len(y):] @ th[:-1] + th[-1]), tids, dest / 'submission.csv')
        recipe = json.loads(path.read_text())
        for name, h in recipe['inputs'].items():
            assert ef.sha(ef.ROOT / name) == h
        th = np.array(recipe['weights'])
        parts = compute(C, S, y, tr, np.arange(len(y), len(full)))
        replay = full[len(y):] + design(recipe['mode'], parts, full[len(y):]) @ th[:-1] + th[-1]
        ef.write_submission(expit(replay), tids, dest / 'submission_replay.csv')
        assert ef.sha(dest / 'submission.csv') == ef.sha(dest / 'submission_replay.csv')
    print('Selected', chosen, flush=True)
if __name__ == '__main__':
    with threadpool_limits(limits=2):
        main()
