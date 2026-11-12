# CPU port; original experiments/weight_space/white_group_contrasts.py, SHA256 b7e9d9535bb48655156385ac047f8a6adaacf1a62408fb7ee4769c63bbeb5419
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Compare the physical groups in two saved source-rank forests on matched real controls."""
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
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from threadpoolctl import threadpool_limits
from partition_evidence import ROOT, OUT
import white_loop as wl

def main():
    q, ids, tids, y, wt, wn, miss = wl.load()
    groups = None
    arrays = []
    for split in (29, 7):
        tag = f'MECH3C12V_srank_k{split}_s0'
        with np.load(ROOT / f'tune_out/redesign/contrib_{tag}.npz') as a, np.load(ROOT / f'tune_out/ver_{tag}.npz') as expected:
            assert np.array_equal(a['ids'], ids) and np.array_equal(a['tids'], tids)
            assert expected['kinds'][0] == 'orig'
            cols = list(a['groups'].astype(str))
            assert groups is None or groups == cols
            groups = cols
            err = abs(expit(a['oof'].sum(1)) - expected['oof'][0]).max()
            print(tag, 'OOF additive probability error', err, flush=True)
            assert err < 2e-05
            arrays.append(np.vstack([a['oof'], a['test']]).astype(float))
    arr = np.mean(arrays, axis=0)
    old = np.load(OUT / 'local_rank_three_source_votes_oof.npz')
    assert np.array_equal(old['ids'], ids)
    z = old['s7']
    regime = pd.read_parquet(OUT / 'innov_v2_train.parquet').loc[ids, 'pre_white'].to_numpy()
    n = pd.read_parquet(ROOT / 'clean/features_train.parquet', columns=['n_pre', 'n_post']).loc[ids].sum(1).to_numpy()
    sm = np.digitize(z, np.quantile(z, np.linspace(0, 1, 13)[1:-1]))
    wp = np.digitize(regime, [0.65, 0.85, 0.94])
    length = np.digitize(n, np.quantile(n, [0.33, 0.67]))
    cell = 3 * (4 * sm + wp) + length
    neg = y == 0
    nc = np.bincount(cell[neg], minlength=144)
    pc = np.bincount(cell[miss], minlength=144)
    ok = (nc >= 8) & (pc > 0)
    use = miss & ok[cell]
    control = neg & ok[cell]
    w = pc[cell[control]] / nc[cell[control]]
    result = {'covered_historical': int(use.sum()), 'matched_negatives': int(control.sum()), 'groups': {}}
    for j, k in enumerate(groups[:-1]):
        v = arr[:len(ids), j]
        gap = v[use].mean() - np.average(v[control], weights=w)
        result['groups'][k] = dict(raw_gap=float(v[miss].mean() - v[neg].mean()), matched_gap=float(gap), raw_auc=float(roc_auc_score(y[miss | neg], v[miss | neg])))
    for k, r in sorted(result['groups'].items(), key=lambda t: t[1]['matched_gap'], reverse=True):
        print(k, round(r['raw_gap'], 4), round(r['matched_gap'], 4), flush=True)
    print('Matched support', use.sum(), control.sum(), flush=True)
    (OUT / 'group_contrasts_matched.json').write_text(json.dumps(result, indent=2))
    for part, ix, x in (('train', ids, arr[:len(ids)]), ('test', tids, arr[len(ids):])):
        tab = pd.DataFrame(x[:, :-1], index=ix, columns=['full_' + k for k in groups[:-1]]).rename_axis('id')
        tab.to_parquet(OUT / f'tree_channels_claude_{part}.parquet')
        other = pd.read_parquet(OUT / f'tree_channels_independent_29_independent_7_{part}.parquet').loc[ix]
        combined = pd.concat([tab, other.add_prefix('ours_').rename(columns=lambda n: n.replace('ours_full_', 'full_ours_').replace('ours_late_', 'late_ours_'))], axis=1)
        combined.to_parquet(OUT / f'tree_channels_shared_{part}.parquet')
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
