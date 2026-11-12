# CPU port; original experiments/weight_space/white_new_partition.py, SHA256 2fb14bdc2f0c9a1ef6518f78ff357ef6f17d33cb370604ffee4073ba13923610
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Check the frozen stage and readout recipes on a separately specified output split."""
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
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from white_baseline_opinions import inputs
from white_local_rank_weights import centered, weights
from white_nested import fit, audit
from white_outer_priors import generate
from white_stage import new_inputs
from white_consult import compile_inputs
from partition_evidence import ROOT, OUT
import white_loop as wl

def main(seed, visit=None):
    name = 'local_rank_early_opinions'
    folder = ROOT / f'new weights/wn_{name}_candidate'
    recipe = folder / 'weights.json'
    spec = json.loads(recipe.read_text())
    stage = ROOT / f'new weights/wn_stage_{spec['parent']}_candidate/weights.json'
    source = json.loads(stage.read_text())
    q, ids, tids, y, wt, wn, miss = wl.load()
    data = wl.ef.read_data()
    assert data[1].equals(ids) and data[2].equals(tids)
    xx, _ = wl.he.sources('state_tree', 'train', ids)
    context = pd.read_parquet(OUT / 'state_train.parquet').loc[ids]
    earlier = json.loads(wl.he.PARENT.read_text())
    mods = wl.er.modules(earlier['spec'])
    _, tables = wl.er.tables(data, mods)
    with threadpool_limits(limits=2):
        folds, _ = wl.er.parent(data, (seed,))
    stage_inputs, _ = compile_inputs(ids, tids, source['models'], ['volatility'])
    side, _ = new_inputs(source['tables'], ids, tids)
    last, _ = inputs(ids, tids, spec['models'], spec['physics'])
    prior = OUT / f'law_outer{seed}'
    if not (prior / 'fold4_rows.npz').exists():
        generate(seed, 'physical')
    results = {}
    for s in (7, seed):
        fixed, mid, final = [np.empty(len(y)) for _ in range(3)]
        for f, (tr, va, zr, zv) in enumerate(folds[s]):
            with threadpool_limits(limits=2):
                tx, vx = [wl.er.features(mods, tables, z, i) for z, i in ((zr, tr), (zv, va))]
                w = wl.fit(tx, zr, y[tr], earlier['selected'])
                zr, zv = (zr + tx @ w, zv + vx @ w)
                tx, vx = [wl.he.features(xx, z, i, context) for z, i in ((zr, tr), (zv, va))]
                w = wl.fit(tx, zr, y[tr], 100.0)
                zr, zv = (zr + tx @ w, zv + vx @ w)
                fixed[va] = zv
            if s == 7:
                assert max(abs(zv - q[f's7_f{f}_zv'])) < 2e-06
            d = OUT / f'law_outer{s}'
            with np.load(d / f'fold{f}_rows.npz') as a:
                assert np.array_equal(a['train_ids'], ids[tr]) and np.array_equal(a['validation_ids'], ids[va])
            h = pd.read_parquet(d / f'fold{f}_train.parquet').loc[ids].to_numpy()
            train, valid = [np.column_stack([stage_inputs(h[i], i, z, 'full'), side[i]]) for i, z in ((tr, zr), (va, zv))]
            w = fit(train, zr, y[tr], True)
            zv = zv + valid @ w
            mid[va] = zv
            inner = np.empty(len(tr))
            for a, b in StratifiedKFold(3, shuffle=True, random_state=906 + f).split(tr, y[tr]):
                w = fit(train[a], zr[a], y[tr[a]], True)
                inner[b] = zr[b] + train[b] @ w
            tx, mean = centered(last(tr, inner), inner)
            vx, _ = centered(last(va, zv), zv, mean)
            final[va] = zv + vx @ weights(tx, inner, y[tr])
            if visit is not None:
                visit(s, f, tr, va, inner, last(tr, inner), final[va], fixed[va])
        if s == 7:
            with np.load(OUT / (name + '_oof.npz')) as a:
                assert max(abs(final - a['s7'])) < 2e-06
        results[s] = {k: wl.score(y, v, wt, wn, miss) for k, v in (('baseline', fixed), ('stage', mid), ('candidate', final))}
        results[s]['recovery'] = audit(y, miss, fixed, final)
        print('verified partition', s, results[s]['candidate'], results[s]['recovery']['candidate'], flush=True)
    if visit is not None:
        return
    path = folder / f'additional_partition_{seed}.json'
    path.write_text(json.dumps(dict(recipe_sha256=wl.ef.sha(recipe), scores=results, scope='New output-weight partition on the same series and existing base OOF predictions. Primary split 7 was independently reconstructed first.'), indent=2))
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('seed', type=int)
    with threadpool_limits(limits=1):
        main(p.parse_args().seed)
