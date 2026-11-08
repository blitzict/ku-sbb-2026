# CPU port; original experiments/weight_space/white_checkpoint.py, SHA256 636951f7ca56e664452f1015a5e6d81bb666b2fa22dc53b939081b5550abe784
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Reconstruct the exact train/validation offsets of the jointly verified 0.87405 weights."""
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
import json
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from threadpoolctl import threadpool_limits
import white_loop as wl
from white_nested import gate_fields, law_design, fit
from white_outer_priors import load, fit_on
from white_joint import read_member
from partition_evidence import OUT, ROOT
NAME = 'physical_pulse_state_walk_net_lawtree_both_claude_law_rank'
FOLDER = ROOT / f'new weights/wn_{NAME}_candidate'
CACHE = OUT / 'frozen874_offsets.npz'
SEEDS = (7, 29, 0, 123, 2024, 11041)

def prepare():
    if CACHE.exists():
        return
    q, ids, tids, y, *_ = wl.load()
    n = len(y)

    def frame(tag):
        return pd.concat([pd.read_parquet(OUT / f'{tag}_{p}.parquet').loc[ix] for p, ix in (('train', ids), ('test', tids))])
    g, t, wh = gate_fields(ids, tids)
    walk = frame('innov_v2').pre_walk.to_numpy()
    blocks = []
    members = []
    for name in ('pulse', 'state'):
        a = frame('jump_law_' + name)
        gate = 0.2 + 0.8 * wh if name == 'pulse' else 1 - 0.35 * wh
        blocks.append(gate[:, None] * np.arcsinh(a.clip(-24, 60).to_numpy()))
        h = a[['level2', 'scale2', 'shape0', 'shape1']].clip(-14, 35).to_numpy()
        members.append(((np.logaddexp.reduce(h, axis=1) - np.log(h.shape[1]))[:, None], gate))
    a = frame('walk_jump')
    blocks.append(walk[:, None] * np.arcsinh(a.clip(-25, 60).to_numpy()))
    members.append((a.total.to_numpy()[:, None], walk))
    for model in ('net', 'lawtree_both', 'claude_law'):
        f, _ = read_member(model, ids, tids)
        members.append((f, np.ones(len(wh))))
    const = np.column_stack(blocks)

    def design(h, i, z):
        rr = [law_design(h, z, g[i], t[i]), const[i]]
        for a, v in members:
            d = v[i, None] * (a[i] - z[:, None]).clip(-7, 8)
            rr.extend([d, d * np.tanh(z[:, None] / 2), d * wh[i, None], d * t[i, None], d * walk[i, None]])
        return np.column_stack(rr)
    output = {'ids': ids.to_numpy(dtype=str), 'tids': tids.to_numpy(dtype=str), 'parent_hash': np.array(wl.ef.sha(FOLDER / 'weights.json'))}
    ref = np.load(OUT / f'wn_{NAME}_oof.npz')
    for seed in SEEDS:
        for fold in range(5):
            key = f's{seed}_f{fold}_'
            tr, va, zr, zv = [q[key + k] for k in ('tr', 'va', 'zr', 'zv')]
            h = pd.read_parquet(OUT / f'law_outer{seed}/fold{fold}_train.parquet').loc[ids].to_numpy()
            f = design(h[tr], tr, zr)
            v = design(h[va], va, zv)
            w = fit(f, zr, y[tr], True)
            output[key + 'zr'] = zr + f @ w
            output[key + 'zv'] = zv + v @ w
            assert np.max(abs(output[key + 'zv'] - ref[f's{seed}'][va])) < 2e-07
    fi, ft, M, mask, yy = load('physical')
    assert ids.equals(fi) and tids.equals(ft) and np.array_equal(y, yy)
    tr = np.arange(n)
    te = n + np.arange(len(tids))
    h, _ = fit_on(M, mask, y, tr)
    w = np.array(json.loads((FOLDER / 'weights.json').read_text())['weights'])
    output['zr_all'] = q['zr_all'] + design(h[tr], tr, q['zr_all']) @ w
    prediction = q['zt'] + design(h[te], te, q['zt']) @ w
    sub = pd.read_csv(FOLDER / 'submission.csv', dtype={'id': str}).set_index('id').loc[tids, 'break_probability']
    assert np.max(abs(expit(prediction) - sub.to_numpy())) < 2e-13
    output['zt'] = logit(sub.to_numpy())
    np.savez_compressed(CACHE, **output)
    print('0.87405 replayed on 30 outer folds and 4,000 test records', flush=True)
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        prepare()
