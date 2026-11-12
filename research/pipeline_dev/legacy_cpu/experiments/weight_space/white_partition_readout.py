# CPU port; original experiments/weight_space/white_partition_readout.py, SHA256 92a6c62375a1e88fa275afb81f83239d80c9db696d2caef7322e78672dc8e110
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Independently rebuild the later readouts on a previously unused output partition."""
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
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from white_crop_replication import BASE
from white_local_rank_weights import centered, weights
from white_nested import fit, audit
from white_upstream_stack import collect, train_basis
from white_joint import read_member
from white_freeze_opinion import TAG, step
from white_opinion_spectrum import inputs
from white_signal_audit import terms
from white_new_partition import main as early
from partition_evidence import ROOT, OUT
import white_loop as wl

def table(ids, tag, cols):
    return pd.read_parquet(OUT / f'{tag}_train.parquet', columns=cols).loc[ids]

def settings(ids, tids):
    reg = table(ids, 'innov_v2', ['pre_white', 'pre_walk']).to_numpy()
    h = np.arcsinh(table(ids, *terms('reliable_modes')).to_numpy().clip(-24, 55))
    signal = np.column_stack([h, 0.6 * h * reg[:, :1], 0.6 * h * reg[:, 1:2]])
    spec = json.loads((ROOT / f'new weights/wn_{BASE}_candidate/weights.json').read_text())
    experts = np.column_stack([read_member(m, ids, tids)[0][:len(ids), 0] for m in spec['members']])
    return (reg, signal, experts, spec)

def mode(signal, tr, va, z, v, y):
    h, loc = centered(signal[tr], z)
    test, _ = centered(signal[va], v, loc)
    w = fit(h, z, y[tr], True)
    return (z + h @ w, v + test @ w)

def run(seed):
    _, ids, tids, y, wt, wn, miss = wl.load()
    reg, signal, experts, spec = settings(ids, tids)
    raw = collect(ids, tids)
    guide = raw[0]
    banks, gates, _ = inputs(ids, tids)
    scores = {s: {k: np.full(len(ids), np.nan) for k in ('locked', 'parent', 'candidate')} for s in (7, seed)}
    with np.load(OUT / (TAG + '_oof.npz')) as a:
        assert np.array_equal(ids, a['ids'])
        archived = a['s7']

    def visit(s, f, tr, va, early_z, features, v, locked):

        def inner_split(rs):
            return StratifiedKFold(3, shuffle=True, random_state=rs + f).split(tr, y[tr])
        z = np.empty(len(tr))
        for a, b in inner_split(1103):
            h, ref = centered(features[a], early_z[a])
            hh, _ = centered(features[b], early_z[b], ref)
            z[b] = early_z[b] + hh @ weights(h, early_z[a], y[tr[a]])
        _, v = mode(signal, tr, va, z, v, y)
        zmode = np.empty(len(tr))
        for a, b in inner_split(4343):
            _, zmode[b] = mode(signal, tr[a], tr[b], z[a], z[b], y)
        lo, med, hi = np.quantile(experts[tr], [0.25, 0.5, 0.75], axis=0)
        l, c, r = np.quantile(zmode, [0.25, 0.5, 0.75])

        def vote(ix, p):
            e = (experts[ix] - med) * (r - l) / (hi - lo) + c
            fade = 1 - 0.85 * reg[ix, 0] * np.exp(-0.5 * ((p + 1.1) / 1.25) ** 2)
            return p + (e - p[:, None]) * fade[:, None] @ spec['weights']
        h = train_basis(raw, tr)
        v = 0.75 * vote(va, v) + 0.25 * (guide[va] + h[va] @ weights(h[tr], guide[tr], y[tr]))
        upstream = np.empty(len(tr))
        for a, b in inner_split(7811):
            upstream[b] = guide[tr[b]] + h[tr[b]] @ weights(h[tr[a]], guide[tr[a]], y[tr[a]])
        z = 0.75 * vote(tr, zmode) + 0.25 * upstream
        vv = step(banks, gates, tr, va, z, v, y)
        if s == seed:
            cache = OUT / 'frozen_local_rank_first_layer_overall'
            pack = ROOT / 'new weights/wn_local_rank_first_layer_overall_candidate/weights.json'
            np.savez_compressed(cache / f's{s}_f{f}.npz', train_ids=ids[tr].to_numpy(dtype=str), validation_ids=ids[va].to_numpy(dtype=str), oof_train=z, valid=v, parent_sha=wl.ef.sha(pack), locked=locked)
        if s == 7:
            assert np.max(abs(vv - archived[va])) < 5e-05, (f, np.max(abs(vv - archived[va])))
        for k, vals in (('locked', locked), ('parent', v), ('candidate', vv)):
            scores[s][k][va] = vals
        print('verified', s, f, flush=True)
    early(seed, visit)
    out = {}
    for s, bank in scores.items():
        assert all((np.isfinite(a).all() for a in bank.values()))
        out[s] = {k: wl.score(y, v, wt, wn, miss) for k, v in bank.items()}
        out[s]['recovery'] = audit(y, miss, bank['locked'], bank['candidate'])
        print(s, out[s]['candidate'], out[s]['recovery']['candidate'], flush=True)
    folder = ROOT / f'new weights/wn_{TAG}_candidate'
    (folder / f'additional_partition_{seed}.json').write_text(json.dumps(dict(recipe_sha256=wl.ef.sha(folder / 'weights.json'), scores=out, primary_max_logit_error=float(np.max(abs(scores[7]['candidate'] - archived))), scope='Predeclared output split on the same 24,000 sources and existing base OOF scores. The primary partition was reconstructed and checked first.'), indent=2))
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        run(61817)
