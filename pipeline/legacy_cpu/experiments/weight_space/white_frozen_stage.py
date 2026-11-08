# CPU port; original experiments/weight_space/white_frozen_stage.py, SHA256 4efe78fe340479de1bb3cfc3d8f630794c4a6f7cf78a9f242cdcfbc7b47d0d50
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Rebuild a saved joint readout and cross-fit its training-side offsets."""
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
from white_stage import new_inputs
from white_consult import compile_inputs
from white_checkpoint import SEEDS
from white_nested import fit
from partition_evidence import OUT, ROOT
import white_loop as wl

def prepare(tag):
    folder = ROOT / f'new weights/wn_stage_{tag}_candidate'
    recipe = folder / 'weights.json'
    source = json.loads(recipe.read_text())
    digest = wl.ef.sha(recipe)
    dest = OUT / ('frozen_stage_' + tag)
    dest.mkdir(exist_ok=True)
    q, ids, tids, y, *_ = wl.load()
    base, _ = compile_inputs(ids, tids, source['models'], ['volatility'])
    side, _ = new_inputs(source['tables'], ids, tids)
    with np.load(OUT / f'wn_stage_{tag}_oof.npz') as a:
        assert np.array_equal(a['ids'], ids)
        ref = {s: a[f's{s}'] for s in SEEDS}
    for s in SEEDS:
        for f in range(5):
            dst = dest / f's{s}_f{f}.npz'
            if dst.exists():
                with np.load(dst) as a:
                    assert str(a['parent_sha']) == digest
                continue
            k = f's{s}_f{f}_'
            tr, va, zr, zv = [q[k + c] for c in ('tr', 'va', 'zr', 'zv')]
            d = OUT / f'law_outer{s}'
            with np.load(d / f'fold{f}_rows.npz') as r:
                assert np.array_equal(ids[tr], r['train_ids']) and np.array_equal(ids[va], r['validation_ids'])
            h = pd.read_parquet(d / f'fold{f}_train.parquet').loc[ids].to_numpy()
            tx = np.column_stack([base(h[tr], tr, zr, 'full'), side[tr]])
            vx = np.column_stack([base(h[va], va, zv, 'full'), side[va]])
            w = fit(tx, zr, y[tr], True)
            v = zv + vx @ w
            assert np.max(abs(v - ref[s][va])) < 2e-06
            cv = np.empty(len(tr))
            for a, b in StratifiedKFold(3, shuffle=True, random_state=906 + f).split(tr, y[tr]):
                ww = fit(tx[a], zr[a], y[tr][a], True)
                cv[b] = zr[b] + tx[b] @ ww
            np.savez_compressed(dst, parent_sha=digest, train_ids=ids[tr].to_numpy(dtype=str), validation_ids=ids[va].to_numpy(dtype=str), oof_train=cv, valid=v)
        print('verified joint weights and inner scores', tag, s, flush=True)
    with np.load(folder / 'replay_inputs.npz') as r:
        assert np.array_equal(r['ids'], tids)
        test = r['parent_logits'] + r['features'] @ np.array(source['weights'])
    np.savez_compressed(dest / 'full.npz', train_ids=ids.to_numpy(dtype=str), test_ids=tids.to_numpy(dtype=str), parent_sha=digest, oof_train=ref[7], test=test)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('tag')
    with threadpool_limits(limits=1):
        prepare(p.parse_args().tag)
