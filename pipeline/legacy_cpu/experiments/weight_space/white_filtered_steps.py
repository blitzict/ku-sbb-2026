# CPU port; original experiments/weight_space/white_filtered_steps.py, SHA256 681ee17aa02ddd707ab67ba641a35d60d3a47db2a5a7551db46371cf31b8c068
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Fit boundary intervention shapes through an AR filter and local volatility."""
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
from threadpoolctl import threadpool_limits
from white_jump_priors import tensor
from partition_evidence import OUT
NAMES = ('null', 'observed_step', 'innovation_step', 'reset', 'short_ramp', 'long_ramp', 'transient', 'smooth_pulse', 'early_step', 'late_step', 'ar1', 'ar2')
LAW_MASS = np.array([0.35, 0.22, 0.12, 0.16, 0.15])

def emissions():
    prob = np.array([[1.0, 0.0, 0.0], [0.56, 0.36, 0.08], [0.93, 0.07, 0.0], [0.5, 0.5, 0.0], [0.5, 0.5, 0.0]])
    mean = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0], [-0.74, 0.74, 0], [-0.9, 0.9, 0]])
    sd = np.array([[1.0, 1.0, 1.0], [0.64, 1.04, 2.2], [0.81, 2.42, 1.0], [0.673, 0.673, 1.0], [0.436, 0.436, 1.0]])
    return [tensor(v)[None, None, :, None, :] for v in (np.log(prob.clip(1e-35)), mean, sd)]

def designs(raw, end, b, ar, scale):
    x = raw[:end].astype(float)
    t = np.arange(end) - b
    step = (t >= 0).astype(float)

    def filtered(d):
        return d[2:] - ar[0] * d[1:-1] - ar[1] * d[:-2]
    center = np.median(x[:b])
    return np.stack([np.zeros(end - 2), filtered(step), step[2:], (t[2:] == 0).astype(float), filtered(np.clip((t + 1) / 5, 0, 1)), filtered(np.clip((t + 1) / 24, 0, 1)), filtered(step * np.exp(-np.maximum(t, 0) / 8)), filtered(np.exp(-0.5 * (t / 3.5) ** 2)), filtered((t >= -8).astype(float)), filtered((t >= 8).astype(float)), step[2:] * (x[1:-1] - center) / scale, step[2:] * (x[1:-1] - x[:-2]) / scale]).clip(-25, 25)

def regression(obs, h, D, valid, bank):
    v = valid[:, None, None, :]
    D = D[:, :, None, :]
    y = obs[:, None, None, :]
    h = h[:, None, None, :]
    invh2 = 1 / h.square()
    n = v.sum(3)

    def total(t):
        return (t * v).sum(3, keepdim=True)
    inv = total(invh2)
    dm = total(invh2 * D) / inv
    ym = total(invh2 * y) / inv
    d = D - dm
    G = total(invh2 * d * d)
    A = total(invh2 * d * (y - ym)) / (G + 1.0)
    B = ym
    lp, m, s = bank
    S = torch.ones_like(A)
    prec = 1 / s.square()
    for _ in range(6):
        r = (y - B - A * d) / h
        z = (r / S)[..., None]
        comp = -0.5 * (z - m).square() * prec - s.log() + lp
        u = comp.softmax(4)
        w = (u * prec).sum(4)
        g = (u * prec * m).sum(4)
        target = y - h * S * g / w.clamp_min(1e-08)
        iw = invh2 * w / S.square()
        den = total(iw).clamp_min(1e-08)
        xm = total(iw * d) / den
        bm = total(iw * target) / den
        A = total(iw * (d - xm) * (target - bm)) / (total(iw * (d - xm).square()) + 1.0)
        B = bm - A * xm
        r = (y - B - A * d) / h
        v2 = total(w * r.square()) / n[..., None]
        v1 = total(r * g) / n[..., None]
        S = ((torch.sqrt((v1.square() + 4 * v2).clamp_min(1e-09)) - v1) / 2).clamp(0.28, 3.5)
    z = ((y - B - A * d) / h / S)[..., None]
    comp = -0.5 * (z - m).square() * prec - s.log() + lp
    resp = comp.softmax(4)
    grad = (z - m) * prec
    information = (resp * (prec - grad.square())).sum(4) + (resp * grad).sum(4).square()
    information = information * invh2 / S.square()
    s0 = total(information).clamp_min(0.1)
    I = (total(information * d.square()) - total(information * d).square() / s0).clamp_min(0)
    ll = total(torch.logsumexp(comp, 4) - S.log()) - 0.5 * A.square() - 0.5 * torch.log1p(I) - 0.5 * s0.log()
    return (ll[:, :, :, 0], A[:, :, :, 0], I[:, :, :, 0])

def extract(part):
    dest = OUT / f'filtered_step_{part}.parquet'
    if dest.exists():
        return
    with np.load(OUT / f'pulse_streams_{part}.npz') as q:
        ids, x, n, b = [q[k] for k in ('ids', 'separate', 'n', 'b')]
    with np.load(OUT / f'hetero_path_{part}.npz') as q:
        assert np.array_equal(ids, q['ids']) and np.array_equal(n - 2, q['n'])
        hh = q['common']
    ar = pd.read_parquet(OUT / f'state_{part}.parquet').loc[ids, ['ar1', 'ar2']].to_numpy()
    bank = emissions()
    mass = tensor(np.log(LAW_MASS))
    rows = []
    for start in range(0, len(n), 30):
        sl = slice(start, start + 30)
        B = len(x[sl])
        T = int(n[sl].max()) - 2
        obs = np.zeros((B, T))
        h = np.ones_like(obs)
        ok = np.zeros_like(obs)
        D = np.zeros((B, len(NAMES), T))
        for i, (row, end, j, phi, height) in enumerate(zip(x[sl], n[sl], b[sl], ar[sl], hh[sl])):
            L = end - 2
            u = row[2:end] - phi[0] * row[1:end - 1] - phi[1] * row[:end - 2]
            s = max(1.4826 * np.median(abs(u - np.median(u))), 0.18 * np.std(u), 0.015)
            obs[i, :L] = (u - np.median(u)) / s
            h[i, :L] = height[:L]
            ok[i, :L] = 1
            D[i, :, :L] = designs(row, end, j, phi, s)
        ll, amp, info = regression(tensor(obs), tensor(h), tensor(D), tensor(ok), bank)
        assert torch.isfinite(ll).all()
        rel = ll - ll[:, 0:1]
        lp = ll[:, 0] + mass
        w = lp.softmax(1)
        zero = torch.logsumexp(lp, 1)
        evidence = torch.logsumexp(ll + mass, 2) - zero[:, None]
        rec = {name: evidence[:, k] for k, name in enumerate(NAMES) if k}
        for k in (1, 2, 3, 5, 10, 11):
            rec[NAMES[k] + '_amplitude'] = (w * amp[:, k].abs()).sum(1)
            rec[NAMES[k] + '_signal'] = (w * info[:, k].sqrt() * amp[:, k].abs()).sum(1)
            rec[NAMES[k] + '_modes'] = (w[:, 3:] * rel[:, k, 3:].clamp(-20, 60)).sum(1)
        rec['placement'] = evidence[:, 1] - torch.logsumexp(evidence[:, [8, 9]], 1) + np.log(2)
        rec['permanent'] = torch.logsumexp(evidence[:, [1, 2, 4, 5]], 1) - torch.logsumexp(evidence[:, [3, 6, 7]], 1) + np.log(0.75)
        rec['output_or_driver'] = evidence[:, 1] - evidence[:, 2]
        arr = torch.stack(list(rec.values()), 1).clamp(-80, 110).cpu().numpy()
        rows.append(pd.DataFrame(arr, columns=list(rec), index=ids[sl]))
    frame = pd.concat(rows)
    frame.index.name = 'id'
    assert np.isfinite(frame).all().all()
    frame.to_parquet(dest)
    print(part, 'filtered steps through AR and volatility', frame.shape, flush=True)
if __name__ == '__main__':
    with torch.inference_mode(), threadpool_limits(limits=1):
        torch.set_num_threads(1)
        for part in ('train', 'test'):
            extract(part)
