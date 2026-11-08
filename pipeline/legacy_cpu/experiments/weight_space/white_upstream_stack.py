# CPU port; original experiments/weight_space/white_upstream_stack.py, SHA256 b577ccb4afa70d5a0424f9c3d1122129e2df24a17ffe8e74a5dc33585e1158d4
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Fit the original and mechanism-aware experts jointly from the first set of weights."""
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
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from threadpoolctl import threadpool_limits
from white_joint import read_member
from white_crop_replication import BASE, crop_scores
from white_local_rank_weights import weights
from white_checkpoint import SEEDS
from white_nested import audit
from partition_evidence import ROOT, OUT
import white_loop as wl

def collect(ids, tids):
    ef = wl.ef
    state = ef.read_data()
    assert state[1].equals(ids) and state[2].equals(tids)
    spec = state[0]
    keep = [state[8].index(c) for c in spec['feature_order']]
    old = np.vstack([state[6], state[7]])[:, keep]
    groups = []
    for a, b, _ in state[9]:
        cols = [j for j, k in enumerate(keep) if a <= k < b]
        if cols:
            groups.append(old[:, cols])
    panels, harmonic, paths = crop_scores(ids, tids)
    sources = ('claude_crop_three_partitions', 'law_independent_rank_bag_29_7', 'law_linked_rank_s29', 'claude_revorig', 'claude_bag3', 'lawforecast', 'lawblock', 'lawshared', 'lawretrieval', 'lawvoltree', 'lawpredictive', 'net', 'law_rank_dependence_s29')
    experts = []
    for name in sources:
        arr, pp = read_member(name, ids, tids)
        experts.append(arr[:, 0])
        paths.extend(pp)

    def table(tag, cols):
        frames = []
        for n, index in (('train', ids), ('test', tids)):
            p = OUT / f'{tag}_{n}.parquet'
            paths.append(p)
            frames.append(pd.read_parquet(p, columns=cols).loc[index].to_numpy())
        return np.vstack(frames)
    g = np.column_stack([table('innov_v2', ['pre_white', 'pre_walk']), table('volatility_law', ['pre_cluster']), table('curve', ['gate']), table('periodic', ['gate'])])
    direct = np.column_stack(experts)
    guide = direct[:, :2].mean(1)
    original = np.column_stack([b.mean(1) for b in groups])
    panel = np.column_stack([direct, original, harmonic[:, 0], harmonic[:, 1:4].mean(1), harmonic[:, 4:].mean(1), panels.mean(0)])
    spread = np.column_stack([b.std(1) for b in groups if b.shape[1] > 1] + [panels[:, :, 0].std(0), panels.mean(0)[:, 1:].std(1), harmonic[:, 1:4].std(1)])
    paths.extend([ef.BASE, Path(ef.__file__), Path(__file__).resolve(), Path(__file__).with_name('white_crop_replication.py')])
    return (guide, panel, spread, g, state[11], state[12], paths)

def train_basis(raw, tr):
    guide, panel, spread, reg, D, nc, _ = raw
    center = np.median(panel[tr], axis=0)
    lo, hi = np.quantile(panel[tr], [0.25, 0.75], axis=0)
    scale = (np.quantile(guide[tr], 0.75) - np.quantile(guide[tr], 0.25)) / (hi - lo).clip(0.2)
    calibrated = (panel - center) * scale + np.median(guide[tr])
    delta = (calibrated - guide[:, None]).clip(-6, 6)
    small = np.tanh(delta / 1.6)
    d = wl.ef.ranks(D, wl.ef.distribution_maps(D[tr]))
    physical = d[:, nc:]
    context = d[:, :nc]
    opinions = np.column_stack([delta[:, :2].mean(1), delta[:, 3:5].mean(1), delta[:, 5:9].mean(1), delta[:, 9:13].mean(1), delta[:, 13:-6].mean(1), delta[:, -6:-3].mean(1)])
    effects = np.column_stack([np.ones(len(guide)), guide, delta, small, 0.5 * np.abs(small), 0.6 * spread, reg, context, physical, 0.5 * physical * np.tanh(guide[:, None] / 2), 0.65 * (small[:, :, None] * reg[:, None, :]).reshape(len(guide), -1), 0.65 * (opinions[:, :, None] * context[:, None, :]).reshape(len(guide), -1)])
    return effects

def main(blend):
    q, ids, tids, y, wt, wn, miss = wl.load()
    raw = collect(ids, tids)
    guide = raw[0]
    n = len(y)
    folder = ROOT / f'new weights/wn_{BASE}_candidate'
    tag = 'local_rank_upstream_joint'
    reference = np.load(OUT / (BASE + '_oof.npz'))
    assert np.array_equal(reference['ids'], ids)
    scores, pred, details = ({}, {}, {})
    for s in SEEDS:
        ref = reference[f's{s}']
        upstream = np.empty(n)
        for f in range(5):
            k = f's{s}_f{f}_'
            tr, va = (q[k + 'tr'], q[k + 'va'])
            h = train_basis(raw, tr)
            w = weights(h[tr], guide[tr], y[tr])
            upstream[va] = guide[va] + h[va] @ w
        p = (1 - blend) * ref + blend * upstream
        pred.update({f's{s}': p, f'branch_{s}': upstream})
        scores[s] = {k: wl.score(y, v, wt, wn, miss) for k, v in (('previous', ref), ('branch', upstream), ('new', p))}
        details[s] = audit(y, miss, ref, p)
        print(s, [(k, round(v['auc'], 7), round(v['target_vs_neg'], 7), round(v['remaining'], 7)) for k, v in scores[s].items()], flush=True)
    passed = all((v['new']['auc'] > v['previous']['auc'] for v in scores.values())) and all((scores[7]['new'][k] >= scores[7]['previous'][k] for k in ('remaining', 'target_vs_neg')))
    result = dict(blend=blend, scores=scores, audits=details, accepted=bool(passed))
    (OUT / (tag + '.json')).write_text(json.dumps(result, indent=2))
    rows = np.arange(n)
    test = n + np.arange(len(tids))
    h = train_basis(raw, rows)
    w = weights(h[rows], guide[rows], y)
    new = guide[test] + h[test] @ w
    np.savez_compressed(OUT / (tag + '_oof.npz'), ids=ids.to_numpy(dtype=str), tids=tids.to_numpy(dtype=str), branch_test=new, **pred)
    if passed:
        p = json.loads((folder / 'weights.json').read_text())
        with np.load(folder / 'replay_inputs.npz') as a:
            assert np.array_equal(a['ids'], tids)
            base = a['parent_logits'] + a['features'] @ p['weights']
        dst = ROOT / f'new weights/wn_{tag}_candidate'
        dst.mkdir(exist_ok=True)
        rec, csv = (dst / 'weights.json', dst / 'submission.csv')
        F = np.column_stack([guide[test] - base, h[test]])
        coeff = np.r_[blend, blend * w]
        sub = wl.ef.write_submission(expit(base + F @ coeff), tids, csv)
        paths = list(dict.fromkeys([folder / 'weights.json', *raw[-1]]))
        rec.write_text(json.dumps(dict(weights=coeff.tolist(), blend=blend, hashes={str(p.relative_to(ROOT)): wl.ef.sha(p) for p in paths}), indent=2))
        np.savez_compressed(dst / 'replay_inputs.npz', ids=tids.to_numpy(dtype=str), order=tids.get_indexer(sub.id), parent_logits=base, features=F)
        (dst / 'validation.json').write_text(json.dumps(dict(**result, recipe_sha256=wl.ef.sha(rec), submission_sha256=wl.ef.sha(csv)), indent=2))
        shutil.copyfile(folder / 'replay.py', dst / 'replay.py')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--blend', type=float, default=0.25)
    with threadpool_limits(limits=1):
        main(parser.parse_args().blend)
