# CPU port; original experiments/weight_space/white_transition_priors.py, SHA256 493c202802b6b4e33b6d14c0e94213e73d9c27f18618d51964dc80066904826b
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Infer class-conditional frequencies of single-family noise changes from real folds."""
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
import torch
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from white_native_law import MASS
from white_jump_priors import tensor
from partition_evidence import OUT, ROOT

def chart(tab, coordinate):
    m = tab.filter(regex='^' + coordinate + '_').copy()
    m.columns = m.columns.str.removeprefix(coordinate + '_')
    terms = []
    weights = []
    for kind, prob in (('null', 0.62), ('pair', 0.16), ('free_scale_from', 0.11), ('free_mean_from', 0.11)):
        if kind == 'pair':
            p = np.outer(MASS, MASS).ravel()
            col = [f'pair{i}_{j}' for i in range(8) for j in range(8)]
        else:
            p = MASS
            col = [kind + str(i) for i in range(8)]
        terms.append(m[col].to_numpy() - np.log(p))
        weights.extend(prob * p)
    return (tensor(np.column_stack(terms)), tensor(weights))

def run():
    out = OUT / 'transition_prior_models'
    out.mkdir(exist_ok=True)
    inp = {}
    vectors = []
    for stream in ('state', 'pulse'):
        a, b = [pd.read_parquet(OUT / f'native_law_{stream}_{part}.parquet') for part in ('train', 'test')]
        if inp:
            assert ids.equals(a.index) and tids.equals(b.index)
        ids, tids = (a.index, b.index)
        for reference in ('native', 'standard'):
            t, p = chart(pd.concat([a, b]), reference)
            inp[stream + '_' + reference] = t
    n = len(ids)
    y = pd.read_csv(ROOT / 'data/train_labels.csv', dtype={'id': str}).set_index('id').loc[ids, 'label'].to_numpy()
    g = np.concatenate([pd.read_parquet(OUT / f'innov_v2_{part}.parquet').loc[ix, 'pre_white'] for part, ix in (('train', ids), ('test', tids))])
    gates = tensor(np.stack([g, 1 - g], 1))
    for f, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(ids, y)):
        fields = {}
        params = {}
        for key, L in inp.items():
            for h, temp in enumerate((0.4, 0.94)):
                E = torch.exp((L - L.amax(1, keepdim=True)) * temp).clamp_min(1e-30)
                coeff = []
                for k in (0, 1):
                    ix = tr[y[tr] == k]
                    v = E[ix]
                    gate = gates[ix]
                    w = p.repeat(2, 1)
                    for _ in range(60):
                        pred = (v @ w.T).clamp_min(1e-28)
                        count = w * ((gate / pred).T @ v)
                        w = (count + 165 * p) / (count.sum(1, keepdim=True) + 165)
                    coeff.append(w)
                null, change = [(gates * (E @ w.T)).sum(1).clamp_min(1e-29).log() for w in coeff]
                fields[key + str(h)] = (change - null).cpu().numpy()
                params[key + str(h)] = torch.stack(coeff).cpu().numpy()
        np.savez_compressed(out / f'fold{f}.npz', training_ids=ids[tr].to_numpy(dtype=str), validation_ids=ids[va].to_numpy(dtype=str), **params)
        frame = pd.DataFrame(fields, index=ids.append(tids))
        assert np.isfinite(frame).all().all()
        if f == 0:
            val = pd.DataFrame(index=ids, columns=frame.columns, dtype=float)
        val.loc[ids[va]] = frame.loc[ids[va]]
        vectors.append(frame.loc[tids].to_numpy())
    assert np.isfinite(val).all().all()
    val.to_parquet(OUT / 'transition_odds_train.parquet')
    pd.DataFrame(np.mean(vectors, 0), index=tids, columns=val.columns).to_parquet(OUT / 'transition_odds_test.parquet')
    print('transition priors', val.shape, len(tids), flush=True)
if __name__ == '__main__':
    with threadpool_limits(limits=1), torch.inference_mode():
        torch.set_num_threads(1)
        run()
