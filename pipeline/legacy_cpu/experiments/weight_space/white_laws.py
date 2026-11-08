# CPU port; original experiments/weight_space/white_laws.py, SHA256 6be511481b04a6d7c100e80753760bfb9d7370569bf6bf8f6e5ef392e8769d43
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Integrate location, scale and noise-shape interventions over a shared density bank."""
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
import torch
from scipy.special import gammaln, logsumexp
from threadpoolctl import threadpool_limits
from partition_evidence import ROOT, OUT
AXIS = np.r_[-np.geomspace(9.15, 90, 28)[::-1], np.linspace(-9, 9, 241), np.geomspace(9.15, 90, 28)]
MU = np.linspace(-1.8, 1.8, 53)
LOG_SD = np.linspace(-1.0, 1.0, 37)
LAWS = [('normal', 0.0, 0.0)] + [('student', d, 0.0) for d in (3.0, 4.0, 5.0, 7.0, 10.0, 16.0, 28.0)]
LAWS += [('contam', p, r) for r in (2.25, 4.5, 8.0) for p in (0.006, 0.025, 0.08, 0.18)]
LAWS += [('bimodal', d, 0.0) for d in (0.45, 0.65, 0.78, 0.86, 0.92, 0.96)]
LAWS += [('expower', beta, 0.0) for beta in (1.0, 1.4, 2.7, 4.0, 8.0)]
FAMILIES = list(dict.fromkeys((k for k, _, _ in LAWS)))

def dictionary():
    x = (AXIS[None, None, :] - MU[:, None, None]) / np.exp(LOG_SD)[None, :, None]
    normal = lambda u: -0.5 * u * u - 0.5 * np.log(2 * np.pi)
    out = []
    for kind, k, r in LAWS:
        if kind == 'normal':
            ll = normal(x)
        elif kind == 'student':
            ll = gammaln((k + 1) / 2) - gammaln(k / 2) - 0.5 * np.log(np.pi * (k - 2)) - 0.5 * (k + 1) * np.log1p(x * x / (k - 2))
        elif kind == 'contam':
            sd = 1 / np.sqrt(1 - k + k * r * r)
            ll = np.logaddexp(np.log1p(-k) + normal(x / sd) - np.log(sd), np.log(k) + normal(x / (sd * r)) - np.log(sd * r))
        elif kind == 'bimodal':
            sd = np.sqrt(1 - k * k)
            ll = np.logaddexp(normal((x - k) / sd), normal((x + k) / sd)) - np.log(2 * sd)
        else:
            scale = np.exp(0.5 * (gammaln(1 / k) - gammaln(3 / k)))
            ll = np.log(k / (2 * scale)) - gammaln(1 / k) - np.abs(x / scale) ** k
        out.append(np.maximum(-20000.0, ll - LOG_SD[None, :, None]))
    return torch.tensor(np.asarray(out, np.float32), device='cpu').flatten(0, 2).T.contiguous()

def hist(v):
    left = np.searchsorted(AXIS, v, side='right').clip(1, len(AXIS) - 1) - 1
    r = ((v - AXIS[left]) / (AXIS[left + 1] - AXIS[left])).clip(0, 1)
    return np.bincount(left, 1 - r, minlength=len(AXIS)) + np.bincount(left + 1, r, minlength=len(AXIS))

def prep_raw(part, tag):
    if tag == 'raw':
        with np.load(ROOT / f'tune_out/seq_cache_{part}.npz', allow_pickle=True) as q:
            ids, x, n, b = [q[k] for k in ('ids', 'x', 'n', 'b')]
    else:
        with np.load(OUT / f'pulse_streams_{part}.npz') as q:
            ids, x, n, b = [q[k] for k in ('ids', 'separate', 'n', 'b')]
    H = []
    center = []
    ref = []
    for r, j, end in zip(x, b, n):
        row = r[:end].astype(float)
        c = np.median(row)
        s = max(np.median(abs(row - c)) * 1.4826, 0.18 * row.std(), 0.01)
        v = (row - c) / s
        H.append([hist(v[:j]), hist(v[j:])])
        center.append(c)
        ref.append(s)
    return (np.asarray(ids, dtype=str), np.array(H, np.float32), np.array(center), np.array(ref))

def priors(center, scale):
    mu = center[:, None] + scale[:, None] * MU
    pmu = np.log(0.89 / 4 / (1 + np.exp((abs(mu) - 2).clip(-20, 20) / 0.075)) + 0.11 * np.exp(-abs(mu) / 2) / 4)
    pmu += np.log(scale[:, None] * (MU[1] - MU[0]))
    logs = np.log(scale)[:, None] + LOG_SD
    ps = np.log(0.88 / 1.4 / (1 + np.exp((abs(logs) - 0.7).clip(-20, 20) / 0.045)) + 0.12 * np.exp(-abs(logs)) * 0.5)
    ps += np.log(LOG_SD[1] - LOG_SD[0])
    return [torch.tensor(q, dtype=torch.float32, device='cpu') for q in (pmu, ps)]

def extract(part, tag, bank):
    ids, H, center, scale = prep_raw(part, tag)
    rows = []
    for start in range(0, len(H), 48):
        end = min(start + 48, len(H))
        size = end - start
        obs = torch.tensor(H[start:end], device='cpu')
        ll = (obs.reshape(-1, len(AXIS)) @ bank).reshape(size, 2, len(LAWS), len(MU), len(LOG_SD))
        ll -= ll.amax((2, 3, 4), keepdim=True)
        a, b = ll.unbind(1)
        mu, sd = priors(center[start:end], scale[start:end])
        pm = mu[:, None, :, None]
        ps = sd[:, None, None, :]
        lse = torch.logsumexp
        la, lb = (lse(v + pm + ps, (2, 3)) for v in (a, b))
        factors = [lse(a + b + pm + ps, (2, 3)), lse(lse(a + pm, 2) + lse(b + pm, 2) + sd[:, None, :], 2), lse(lse(a + ps, 3) + lse(b + ps, 3) + mu[:, None, :], 2), la + lb]
        A = []
        B = []
        for family in FAMILIES:
            idx = [i for i, (f, _, _) in enumerate(LAWS) if f == family]
            A.append(lse(a[:, idx], 1) - np.log(len(idx)))
            B.append(lse(b[:, idx], 1) - np.log(len(idx)))
        A = torch.stack(A, 1)
        B = torch.stack(B, 1)
        transitions = []
        for i in range(len(FAMILIES)):
            transitions.append(lse(A[:, i:i + 1] + B + pm + ps, (2, 3)))
        factors.extend([torch.cat(transitions, 1), la, lb])
        rows.append(torch.cat(factors, 1).cpu().numpy())
    M = np.concatenate(rows)
    assert np.isfinite(M).all()
    np.savez_compressed(OUT / f'wnlaw_{tag}_{part}.npz', ids=ids, values=M, center=center, scale=scale)
    print(tag, part, M.shape, flush=True)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--tag', choices=('physical', 'raw'), default='physical')
    args = p.parse_args()
    with threadpool_limits(limits=1), torch.inference_mode():
        torch.set_num_threads(1)
        bank = dictionary()
        for part in ('train', 'test'):
            extract(part, args.tag, bank)
