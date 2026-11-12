# CPU port; original experiments/weight_space/other_strict_memory.py, SHA256 1838b758de3f52b0b47acf40a1ecf0469209633c1a1cdb040b35f167cdd42f10
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Separate exact unit persistence from the tiny off-grid mass of the archived prior."""
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
from scipy.special import expit
from threadpoolctl import threadpool_limits
from partition_evidence import OUT
from white_memory_law import bank, row, point, D, S, P, IDS
from white_jump_priors import tensor

def compute(a, b, ref, scale):
    lse = torch.logsumexp
    free = expit((0.96 - abs(ref[:, None] + D)) / 0.025) * np.gradient(D)
    free /= free.sum(1, keepdims=True)
    zero = point(-ref)
    soft = point(1 - ref)
    prior = 0.54 * zero + 0.38 * free + 0.08 * soft
    strict = np.where(soft > 1e-11, soft, 0)

    def lp(m):
        with np.errstate(divide='ignore'):
            return tensor(np.log(m))[:, None, :, None]
    s = scale[:, None]
    logsd = np.log(s) + S
    v = 0.9 * expit((0.71 - abs(logsd)) / 0.05) + 0.1 * np.exp(-abs(logsd))
    nuisance = tensor(np.log(P))[None, :, None] + tensor(np.log(v / v.sum(1, keepdims=True)))[:, None, :]

    def integrate(t):
        return lse(t + nuisance, (1, 2))
    null = integrate(lse(a + b + lp(prior), 2))
    left, right = (lse(a + lp(prior), 2), lse(b + lp(prior), 2))

    def fixed(m):
        p, q = (lse(a + lp(m), 2), lse(b + lp(m), 2))
        return {'start': integrate(p + right) - null, 'both': integrate(p + q) - null, 'before': integrate(p) - integrate(left), 'after': integrate(q) - integrate(right)}
    u, padded = (fixed(strict), fixed(soft))
    out = dict(strict=u['start'], old=padded['start'], unit_share=(u['start'] - padded['start']).clamp_max(0).exp(), strict_both=u['both'], strict_release=u['start'] - u['both'], pre_only=u['before'], post_only=u['after'], after_minus_before=u['after'] - u['before'], old_before=padded['before'], flexible_change=integrate(left + right) - null)
    uniform = np.full_like(prior, 1 / len(D))
    for label, m in (('grid', uniform), ('physical', free)):
        pre = lse(a + lp(m), 2)
        post = lse(b + lp(m), 2)
        same = integrate(lse(a + b + lp(m), 2))
        out[label + '_start'] = integrate(pre + right) - null
        out[label + '_end'] = integrate(left + post) - null
        out[label + '_common'] = same - null
        out[label + '_pre_only'] = integrate(pre) - integrate(left)
        out[label + '_post_only'] = integrate(post) - integrate(right)
        out[label + '_release'] = integrate(pre + right) - same
    reconstructed = torch.logaddexp(out['strict'], out['grid_start'] + np.log(len(D) * 1e-12))
    assert (reconstructed - out['old']).abs().max() < 0.002
    return out

def run(part, grid):
    with np.load(OUT / f'pulse_streams_{part}.npz') as a:
        ids, x, n, b = [a[k] for k in ('ids', 'separate', 'n', 'b')]
    phi = pd.read_parquet(OUT / f'state_{part}.parquet').loc[ids, ['ar1', 'ar2']].to_numpy()
    blocks = []
    for start in range(0, len(n), 64):
        h, scale, r = ([], [], [])
        for v, j, m, p in zip(x[start:start + 64], b[start:start + 64], n[start:start + 64], phi[start:start + 64]):
            mat, sd, ref = row(v[:m].astype(float), j, p, 1)
            h.append(mat)
            scale.append(sd)
            r.append(ref)
        vals = (tensor(np.asarray(h)).reshape(2 * len(h), -1) @ grid).reshape(len(h), 2, len(IDS), len(D), len(S))
        vals -= vals.amax((2, 3, 4), keepdim=True)
        out = compute(*vals.unbind(1), np.array(r), np.array(scale))
        blocks.append(pd.DataFrame({k: v.cpu().numpy() for k, v in out.items()}, index=ids[start:start + 64]))
    frame = pd.concat(blocks)
    orig = pd.read_parquet(OUT / f'memory_law1_{part}.parquet').loc[ids, 'unit_start'].to_numpy()
    err = np.max(abs(orig - frame.old.clip(-80, 100).to_numpy()))
    assert err < 0.004, err
    assert np.isfinite(frame).all().all()
    frame.to_parquet(OUT / f'strict_unit_memory_{part}.parquet')
    print('unit memory', part, 'reproduction error', err, flush=True)

def matched():
    f = pd.read_parquet(OUT / 'strict_unit_memory_train.parquet')
    pair = pd.read_csv(OUT / 'other_disagreement_pairs.csv')
    pair = pair[pair.replication]
    a, b = (f.loc[pair.case_id], f.loc[pair.control_id])
    rows = []
    for k in f:
        u, v = (a[k].to_numpy(), b[k].to_numpy())
        rows.append(dict(term=k, case=float(np.median(u)), control=float(np.median(v)), win=float(np.mean((u > v) + 0.5 * (u == v)))))
    report = dict(replication_pairs=len(pair), true_unit_dominates_cases=int(sum(a.unit_share > 0.5)), true_unit_dominates_controls=int(sum(b.unit_share > 0.5)), components=rows)
    (OUT / 'strict_unit_memory_disagreements.json').write_text(json.dumps(report, indent=2))
    print(report, flush=True)
if __name__ == '__main__':
    with threadpool_limits(limits=1), torch.inference_mode():
        torch.set_num_threads(1)
        grid = bank()
        for part in ('train', 'test'):
            run(part, grid)
        matched()
