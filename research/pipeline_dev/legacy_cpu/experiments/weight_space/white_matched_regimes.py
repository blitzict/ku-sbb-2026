# CPU port; original experiments/weight_space/white_matched_regimes.py, SHA256 47c2cf333878892611d2cf73e6634cc0cb3eabdcb38677aed89041c04f45c2f8
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Compare a gated physical response with controls at similar starting odds."""
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
from scipy.ndimage import gaussian_filter1d
from white_local_rank_weights import GRID
from white_nested import fit, audit
from white_checkpoint import SEEDS
from partition_evidence import OUT
import white_loop as wl
from threadpoolctl import threadpool_limits

def panel(kind, part, ix):

    def table(name):
        return pd.read_parquet(OUT / f'{name}_{part}.parquet').loc[ix]
    if kind == 'two_modes':
        source = table('mode_encoders').to_numpy()
        gate = ((table('mode_residual').gate.to_numpy() - 0.5) / 0.45).clip(0, 1)
    else:
        source = table('volatility_channel_effect').to_numpy()
        vol = table('volatility_law')
        gate = ((np.maximum(vol.pre_cluster, vol.post_cluster).to_numpy() - 0.5) / 0.45).clip(0, 1)
    source = np.divide(source, gate[:, None], out=np.zeros_like(source), where=gate[:, None] > 0)
    return (gate, np.arcsinh(source.clip(-24, 55)))

def reference(x, z, g):
    spacing = GRID[1] - GRID[0]
    idx = np.rint((z - GRID[0]) / spacing).astype(int).clip(0, len(GRID) - 1)
    total = np.zeros((len(GRID), x.shape[1]))
    np.add.at(total, idx, g[:, None] * x)
    count = np.bincount(idx, weights=g, minlength=len(GRID))
    prior = g @ x / max(g.sum(), 1)
    return (gaussian_filter1d(total, 4.5, axis=0) + 3 * prior) / (gaussian_filter1d(count, 4.5)[:, None] + 3)

def compare(x, z, g, ref):
    mu = np.column_stack([np.interp(z, GRID, v) for v in ref.T])
    return g[:, None] * (x - mu)

def run(kind):
    q, ids, _, y, wt, wn, miss = wl.load()
    cache = OUT / 'frozen_local_rank_early_mode_peaks'
    names = ('two_modes', 'volatility') if kind == 'both' else (kind,)
    sources = [panel(n, 'train', ids) for n in names]
    scores = {}
    for s in SEEDS:
        z0, z1 = (np.empty(len(y)), np.empty(len(y)))
        for f in range(5):
            k = f's{s}_f{f}_'
            tr, va = (q[k + 'tr'], q[k + 'va'])
            with np.load(cache / f's{s}_f{f}.npz') as a:
                assert np.array_equal(ids[tr], a['train_ids']) and np.array_equal(ids[va], a['validation_ids'])
                u, v = (a['oof_train'], a['valid'])
            tx, vx = ([], [])
            controls = y[tr] == 0
            for gate, features in sources:
                ref = reference(features[tr][controls], u[controls], gate[tr][controls])
                tx.append(compare(features[tr], u, gate[tr], ref))
                vx.append(compare(features[va], v, gate[va], ref))
            w = fit(np.column_stack(tx), u, y[tr], True)
            z0[va], z1[va] = (v, v + np.column_stack(vx) @ w)
        old, new = (wl.score(y, a, wt, wn, miss) for a in (z0, z1))
        scores[s] = dict(parent=old, new=new, audit=audit(y, miss, z0, z1))
    dest = OUT / f'matched_regimes_{kind}.json'
    dest.write_text(json.dumps(scores, indent=2))
    print(kind, [(s, round(v['new']['auc'] - v['parent']['auc'], 7)) for s, v in scores.items()], scores[7]['new'], scores[7]['audit']['candidate'], flush=True)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('experiments', nargs='+', choices=('two_modes', 'volatility', 'both'))
    with threadpool_limits(limits=1):
        for kind in p.parse_args().experiments:
            run(kind)
