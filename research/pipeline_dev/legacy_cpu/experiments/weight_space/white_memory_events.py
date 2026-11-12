# CPU port; original experiments/weight_space/white_memory_events.py, SHA256 d52508c880e420af692860d93d9f7e596b3c2bb98a104e0454ff504634ac7001
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Decompose the existing volatility-memory evidence into mutually exclusive physical changes."""
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
import torch
from scipy.special import logit
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from threadpoolctl import threadpool_limits
from white_volatility_law import prepare, table, tensor, PRIOR, MASS, K, E, H, LOGS, MIX, DECAY
from partition_evidence import ROOT, OUT
import white_loop as wl

def extract(part):
    dst = OUT / f'memory_events_{part}.parquet'
    if dst.exists():
        return
    with np.load(OUT / f'pulse_streams_{part}.npz') as a:
        ids, raw, n, b = [a[k] for k in ('ids', 'separate', 'n', 'b')]
    phi = pd.read_parquet(OUT / f'state_{part}.parquet').loc[ids, ['ar1', 'ar2']].to_numpy()
    original = pd.read_parquet(OUT / f'volatility_law_{part}.parquet').loc[ids]
    bank = table()
    pdyn = tensor(PRIOR).log()
    pl = tensor(MASS[K] / MASS[K].sum()).log()
    rows = []
    decay = np.r_[0.0, np.repeat(DECAY, len(MIX) - 1)]
    amp = np.r_[0.0, np.tile(MIX[1:], len(DECAY))]
    dd = decay[None, :] - decay[:, None]
    da = amp[None, :] - amp[:, None]
    active = amp > 0
    both = active[:, None] & active[None, :]
    sets = dict(quiet=~active[:, None] & ~active[None, :], on=~active[:, None] & active[None, :], off=active[:, None] & ~active[None, :], unchanged=both & (dd == 0) & (da == 0), duration=both & (dd != 0) & (da == 0), strength=both & (dd == 0) & (da != 0), both=both & (dd != 0) & (da != 0))
    assert np.all(np.sum(list(sets.values()), axis=0) == 1)
    for start in range(0, len(ids), 64):
        sl = slice(start, start + 64)
        obs, scale = map(np.asarray, zip(*(prepare(r[:end].astype(float), j, ar) for r, end, j, ar in zip(raw[sl], n[sl], b[sl], phi[sl]))))
        size = len(scale)
        ll = (tensor(obs).reshape(-1, len(E) * len(H)) @ bank).reshape(size, len(DECAY), 2, len(MIX), len(K), len(LOGS)).transpose(1, 2)
        ll = torch.cat([ll[:, :, 0, 0:1], ll[:, :, :, 1:].flatten(2, 3)], 2)
        ll -= ll.amax((2, 3, 4), keepdim=True)
        a, c = ll.unbind(1)
        logs = np.log(scale[:, None]) + LOGS
        ps = tensor(np.log(0.88 / 1.4 / (1 + np.exp(((abs(logs) - 0.7) / 0.06).clip(-30, 30))) + 0.12 * np.exp(-abs(logs)) * 0.5) + np.log(np.diff(LOGS)[0]))
        ctx = pl[None, :, None] + ps[:, None, :]
        prior = pdyn[None, :, None, None]
        ad, cd = [torch.logsumexp(v + prior, 1) for v in (a, c)]
        logmass = ad + cd + ctx
        alt = torch.logsumexp(logmass, (1, 2))
        null = torch.logsumexp(a + c + prior + ctx[:, None, :, :], (1, 2, 3))
        memory = (alt - null).cpu().numpy()
        support = (logmass - alt[:, None, None]).exp()
        left = (a + prior).softmax(1)
        right = (c + prior).softmax(1)
        prob = ((left * support[:, None, :, :]).flatten(2) @ right.flatten(2).transpose(1, 2)).cpu().numpy().astype(float)
        assert np.allclose(prob.sum((1, 2)), 1.0, atol=0.0002)
        check = original.iloc[sl]
        assert np.max(abs(memory.clip(-100, 160) - check.new_memory)) < 0.003
        for ix, col in ((prob[:, 0].sum(1), 'memory_on'), (prob[:, :, 0].sum(1), 'memory_off')):
            ok = (ix > 1e-25) & (abs(check[col].to_numpy()) < 95)
            assert np.allclose((memory + np.log(ix.clip(1e-100)) - np.log(PRIOR[0]))[ok], check[col].to_numpy()[ok], atol=0.003)
        rec = {}
        for name, mask in sets.items():
            p = prob[:, mask].sum(1)
            rec['prob_' + name] = p
            rec['log_evidence_' + name] = (memory + np.log(p.clip(1e-40))).clip(-80, 100)
        rec['net_on'] = rec['prob_on'] - rec['prob_off']
        rec['stronger'] = (prob * da).sum((1, 2))
        rec['longer_while_active'] = (prob * both * dd).sum((1, 2))
        rec['changed_while_active'] = rec['prob_duration'] + rec['prob_strength'] + rec['prob_both']
        rows.append(pd.DataFrame(rec, index=ids[sl]))
    tab = pd.concat(rows)
    assert np.isfinite(tab).all().all()
    tab.to_parquet(dst)
    print('verified the joint memory likelihood', part, tab.shape, flush=True)

def diagnose():
    _, ids, _, y, _, _, miss = wl.load()
    raw = ROOT / 'tune_out'
    p, q = [np.load(raw / f'oof_{tag}.npy') for tag in ('MECH3C12_s0', 'MECH3C12V_s0')]
    p, q = [logit(v.clip(1e-06, 1 - 1e-06)) for v in (p, q)]
    i = p >= np.quantile(p[y == 0], 0.8)
    j = q >= np.quantile(q[y == 0], 0.8)
    vol = pd.read_parquet(OUT / 'volatility_law_train.parquet').loc[ids]
    stats = pd.concat([vol[['new_memory', 'pre_cluster', 'post_cluster']], pd.read_parquet(OUT / 'memory_events_train.parquet').loc[ids]], axis=1)
    neg = np.flatnonzero(y == 0)
    records = []
    matched = []
    for name, mask in (('recovered', (y == 1) & ~i & j), ('lost', (y == 1) & i & ~j), ('recovered_821', miss & ~i & j), ('all_821', miss)):
        rows = np.flatnonzero(mask)
        for tag, pred in (('before', p), ('after', q)):
            cost = cdist(pred[rows, None], pred[neg, None], metric='cityblock')
            a, b = linear_sum_assignment(cost)
            rr, cc = (rows[a], neg[b])
            assert len(set(cc)) == len(rr)
            order = np.random.default_rng(6103).permutation(len(rr))
            contrast = stats.to_numpy()[rr] - stats.to_numpy()[cc]
            sd = stats.to_numpy()[neg].std(0).clip(0.0001)
            for k, col in enumerate(stats.columns):
                d = contrast[:, k]
                records.append(dict(panel=name, matching_score=tag, n=len(rr), quantity=col, case=stats[col].to_numpy()[rr].mean(), control=stats[col].to_numpy()[cc].mean(), gap_sd=d.mean() / sd[k], half1=d[order[::2]].mean() / sd[k], half2=d[order[1::2]].mean() / sd[k]))
            matched.append(dict(panel=name, matching_score=tag, n=len(rr), median_abs_score_gap=float(np.median(cost[a, b])), max_abs_score_gap=float(np.max(cost[a, b]))))
    report = pd.DataFrame(records)
    report.to_csv(OUT / 'memory_events_recovery.csv', index=False)
    (OUT / 'memory_events_matches.json').write_text(json.dumps(matched, indent=2))
    print(report[report.panel.str.startswith('recovered') & (report.matching_score == 'after') & report.quantity.isin(['new_memory', 'prob_on', 'prob_off', 'changed_while_active', 'net_on', 'stronger', 'longer_while_active'])].round(4).to_string(index=False), flush=True)
if __name__ == '__main__':
    with threadpool_limits(limits=1), torch.inference_mode():
        torch.set_num_threads(1)
        for part in ('train', 'test'):
            extract(part)
        diagnose()
