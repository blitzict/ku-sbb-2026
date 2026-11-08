# CPU port; original experiments/weight_space/white_markov_copula.py, SHA256 ed59532f98ce8aa4d8f340fbc6db2c61e24b4c377f3b10ed99cc3f75fa17c14a
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Test whether the order of a segment's ranks changes at the marked boundary."""
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
from scipy.optimize import brentq
from scipy.special import expit, logsumexp
from scipy.stats import rankdata
from threadpoolctl import threadpool_limits
from partition_evidence import OUT

def emission_bank():
    u = (np.arange(256) + 0.5) / 256
    forms = [(u - 0.5, 0.5), (2 * abs(u - 0.5) - 0.5, 0.5), (2 * abs(u - 0.5) - 0.8, 0.18), (u - 0.8, 0.18), (0.2 - u, 0.18), (np.sin(2 * np.pi * u), 0.5)]
    cond = []
    mass = []
    persistence = []
    prior = []
    for form, q in forms:
        for slope in (5.0, 18.0):
            offset = brentq(lambda b: expit(slope * form + b).mean() - q, -50, 50)
            for rho in (0.0, -0.8 * min(q / (1 - q), (1 - q) / q), 0.2, 0.55, 0.82, 0.96):
                cond.append(expit(slope * form + offset))
                mass.append(q)
                persistence.append(rho)
            prior.extend([0.62, 0.07, 0.1, 0.09, 0.08, 0.04])
    return (np.stack(cond, 1), np.asarray(mass), np.asarray(persistence), np.log(prior))

def tensor(x):
    return torch.as_tensor(x, device='cpu', dtype=torch.float32)

def likelihood(codes, length, bank):
    post, q, rho = bank
    x = post[codes]
    e0 = (1 - x) / (1 - q)
    e1 = x / q
    t01 = q * (1 - rho)
    t11 = q + rho * (1 - q)
    a = (1 - t01) * e0
    b = t01 * e1
    c = (1 - t11) * e0
    d = t11 * e1
    valid = torch.arange(codes.shape[1], device='cpu')[None, :] < length[:, None]
    valid = valid[:, :, None]
    a = torch.where(valid, a, 1.0)
    d = torch.where(valid, d, 1.0)
    b = b * valid
    c = c * valid
    scale = torch.zeros_like(q)[None, :]
    while a.shape[1] > 1:
        al, bl, cl, dl = [v[:, ::2] for v in (a, b, c, d)]
        ar, br, cr, dr = [v[:, 1::2] for v in (a, b, c, d)]
        a, b, c, d = (al * ar + bl * cr, al * br + bl * dr, cl * ar + dl * cr, cl * br + dl * dr)
        norm = (a + b + c + d) * 0.5
        scale = scale + norm.log().sum(1)
        a, b, c, d = [v / norm for v in (a, b, c, d)]
    return (scale + ((1 - q) * (a[:, 0] + b[:, 0]) + q * (c[:, 0] + d[:, 0])).log()).cpu().numpy()

def features(A, C, prior):
    records = {}
    lp = prior.reshape(6, 2, 6)[0, 0]
    shape = (len(A), 6, 2, 6)
    A = A.reshape(shape)
    C = C.reshape(shape)
    for family in range(6):
        a, c = (A[:, family], C[:, family])
        aa = logsumexp(a + lp, 2)
        cc = logsumexp(c + lp, 2)
        null = logsumexp(a + c + lp, (1, 2)) - np.log(2)
        vals = dict(null=null, on=logsumexp(cc, 1) - np.log(2) - null, off=logsumexp(aa, 1) - np.log(2) - null, memory=logsumexp(aa + cc, 1) - np.log(2) - null, pre=logsumexp(aa, 1) - np.log(2), post=logsumexp(cc, 1) - np.log(2))
        for tag, v in vals.items():
            records[f'f{family}_{tag}'] = v
    return records

def extract(part):
    path = OUT / f'markov_copula_{part}.parquet'
    if path.exists():
        return
    cond, q, rho, prior = emission_bank()
    bank = tuple((tensor(v) for v in (cond, q, rho)))
    blocks = []
    for stream in ('pulse', 'state'):
        with np.load(OUT / f'{stream}_streams_{part}.npz') as data:
            ids, x, n, b = [data[k] for k in ('ids', 'separate', 'n', 'b')]
        sub = []
        for begin in range(0, len(ids), 72):
            sl = slice(begin, begin + 72)
            ll = []
            for side in (0, 1):
                pieces = [v[:j] if side == 0 else v[j:end] for v, j, end in zip(x[sl], b[sl], n[sl])]
                lens = np.array([len(v) for v in pieces])
                T = 1 << (int(lens.max()) - 1).bit_length()
                code = np.zeros((len(lens), T), np.int32)
                for i, v in enumerate(pieces):
                    code[i, :len(v)] = np.minimum((rankdata(v) - 0.5) * 256 / len(v), 255).astype(np.int32)
                ll.append(likelihood(torch.as_tensor(code, device='cpu', dtype=torch.long), torch.as_tensor(lens, device='cpu'), bank))
            rec = features(*ll, prior)
            sub.append(pd.DataFrame(rec, index=ids[sl]))
        blocks.append(pd.concat(sub).add_prefix(stream + '_'))
    frame = pd.concat(blocks, axis=1).clip(-80, 120).astype(np.float32)
    frame.index.name = 'id'
    assert np.isfinite(frame).all().all() and frame.index.is_unique
    frame.to_parquet(path)
    print(part, 'hidden-state ordering', frame.shape, flush=True)

def self_check():
    cond, q, rho, _ = emission_bank()
    x = np.array([[37, 215, 84, 91], [0, 43, 85, 255]])
    pp = cond[x]
    s = np.broadcast_to(q, (2, len(q))).copy()
    v = np.zeros_like(s)
    for t in range(4):
        s = q + rho * (s - q)
        a = s * pp[:, t] / q
        b = (1 - s) * (1 - pp[:, t]) / (1 - q)
        v += np.log(a + b)
        s = a / (a + b)
    actual = likelihood(torch.tensor(x, device='cpu'), torch.tensor([4, 4], device='cpu'), tuple((tensor(k) for k in (cond, q, rho))))
    assert np.max(abs(actual - v)) < 3e-05 and np.max(abs(actual[:, rho == 0])) < 2e-06
if __name__ == '__main__':
    with threadpool_limits(limits=1), torch.inference_mode():
        torch.set_num_threads(1)
        self_check()
        for part in ('train', 'test'):
            extract(part)
