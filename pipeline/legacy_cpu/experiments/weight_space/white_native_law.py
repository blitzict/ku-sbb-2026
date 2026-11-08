# CPU port; original experiments/weight_space/white_native_law.py, SHA256 a232f32b0781ca131b3f721db40ddc5bb4f0f6b9d6794a4cb6239a72356d214a
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Compare asymmetric noise transitions at fixed physical or marginal parameters."""
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
import numpy as np
import pandas as pd
import torch
from scipy.special import gammaln
from threadpoolctl import threadpool_limits
from white_laws import AXIS, hist
from white_jump_priors import tensor
from partition_evidence import OUT
import white_loop as wl
M = np.linspace(-1.85, 1.85, 43)
S = np.linspace(-1.8, 1.25, 41)
FAM = ['normal', 't', 'outliers', 'modes', 'occupancy', 'skew_normal', 'skew_t', 'spikes']
MASS = np.array([0.31, 0.3, 0.125, 0.14, 0.035, 0.027, 0.033, 0.03])

def normal(x):
    return -0.5 * x * x - 0.5 * np.log(2 * np.pi)

def tpdf(x, df):
    return gammaln((df + 1) / 2) - gammaln(df / 2) - 0.5 * np.log(np.pi * df) - (df + 1) / 2 * np.log1p(x * x / df)

def tpl(x, df, a):
    l, r = (np.exp(-a), np.exp(a))
    unit = lambda u: tpdf(u, df) if df else normal(u)
    val = unit(np.where(x < 0, x / l, x / r)) + np.log(2 / (l + r))
    return val

def laws():
    yield (0, 0.0, 1.0, normal)
    for df in (3.5, 5.0, 8.0, 14.0):
        yield (1, 0.0, np.sqrt(df / (df - 2)), lambda x, d=df: tpdf(x, d))
    for p, k in ((0.008, 5.0), (0.025, 4.5), (0.07, 3.0), (0.12, 2.25)):
        yield (2, 0.0, np.sqrt(1 - p + p * k * k), lambda x, p=p, k=k: np.logaddexp(np.log1p(-p) + normal(x), np.log(p) + normal(x / k) - np.log(k)))
    for d in (0.72, 1.2, 1.8, 2.65):
        yield (3, 0.0, np.sqrt(1 + d * d), lambda x, d=d: np.logaddexp(normal(x - d), normal(x + d)) - np.log(2))
    for d in (1.25, 2.2):
        for p in (0.25, 0.37, 0.63, 0.75):
            yield (4, d * (2 * p - 1), np.sqrt(1 + 4 * p * (1 - p) * d * d), lambda x, d=d, p=p: np.logaddexp(np.log(p) + normal(x - d), np.log1p(-p) + normal(x + d)))
    for df in (0.0, 6.0):
        for a in (-0.43, -0.2, 0.2, 0.43):
            l, r = (np.exp(-a), np.exp(a))
            moment = np.sqrt(2 / np.pi) if not df else np.exp(np.log(2 * np.sqrt(df) / (df - 1)) + gammaln((df + 1) / 2) - gammaln(df / 2) - 0.5 * np.log(np.pi))
            mean = moment * (r - l)
            var = (1.0 if not df else df / (df - 2)) * (l ** 3 + r ** 3) / (l + r) - mean ** 2
            yield (5 if not df else 6, mean, np.sqrt(var), lambda x, d=df, a=a: tpl(x, d, a))
    for p, d, k in ((0.025, 4.0, 1.7), (0.075, 2.8, 1.4), (0.15, 1.9, 1.0)):
        for sign in (-1, 1):
            mean = p * d * sign
            v = 1 - p + p * (d * d + k * k) - mean * mean
            yield (7, mean, np.sqrt(v), lambda x, p=p, d=sign * d, k=k: np.logaddexp(np.log1p(-p) + normal(x), np.log(p) + normal((x - d) / k) - np.log(k)))

def bank(native):
    xx = (AXIS[:, None, None] - M[None, :, None]) / np.exp(S)
    group = []
    values = []
    for family, mean, sd, fn in laws():
        x = xx if native else mean + sd * xx
        v = fn(x) - S + (0 if native else np.log(sd))
        values.append(v.astype(np.float32))
        group.append(family)
    return (tensor(np.stack(values, 1)).clip(-24000, 10).flatten(1).contiguous(), np.array(group))

def extract(part, stream):
    dst = OUT / f'native_law_{stream}_{part}.parquet'
    if dst.exists():
        return
    with np.load(OUT / f'{stream}_streams_{part}.npz') as f:
        ids, u, n, b = (f[k] for k in ('ids', 'separate', 'n', 'b'))
    counts = []
    m = []
    s = []
    for row, end, j in zip(u, n, b):
        x = row[:end].astype(float)
        center = np.median(x)
        sd = max(np.median(abs(x - center)) * 1.4826, 0.18 * x.std(), 0.01)
        z = (x - center) / sd
        counts.append([hist(z[:j]), hist(z[j:])])
        m.append(center)
        s.append(sd)
    counts = np.asarray(counts, np.float32)
    m, s = (np.array(m), np.array(s))
    blocks = []
    for native in (True, False):
        B, groups = bank(native)
        idx = [np.flatnonzero(groups == k) for k in range(len(FAM))]
        pi = tensor([MASS[k] / len(idx[k]) for k in groups]).log()
        pf = tensor(MASS).log()
        rows = []
        lse = torch.logsumexp
        for start in range(0, len(m), 32):
            sl = slice(start, min(start + 32, len(m)))
            N = len(m[sl])
            ll = (tensor(counts[sl]).reshape(-1, len(AXIS)) @ B).reshape(N, 2, len(groups), len(M), len(S))
            ll -= ll.amax((2, 3, 4), keepdim=True)
            a, c = ll.unbind(1)
            mu = m[sl, None] + s[sl, None] * M
            logs = np.log(s[sl, None]) + S
            pm = tensor(np.log(0.86 / 4 / (1 + np.exp(((abs(mu) - 2) / 0.085).clip(-35, 35))) + 0.14 * np.exp(-abs(mu) / 2) / 4) + np.log(s[sl, None] * np.diff(M)[0]))
            lo = -1.2 if native else -0.7
            prior = 0.82 / (0.7 - lo) / (1 + np.exp(((lo - logs) / 0.08).clip(-35, 35))) / (1 + np.exp(((logs - 0.7) / 0.08).clip(-35, 35))) + 0.18 * np.exp(-abs(logs)) * 0.5
            ps = tensor(np.log(prior) + np.log(np.diff(S)[0]))
            pm0 = pm[:, None, :, None]
            ps0 = ps[:, None, None, :]
            null = lse(a + c + pm0 + ps0, (2, 3))
            total = lse(null + pi, 1)
            A, C = [torch.stack([lse(v[:, j], 1) - np.log(len(j)) for j in idx], 1) for v in (a, c)]

            def collapse(t, p, ax):
                return lse(t + p, ax)
            records = {}
            shared = []
            frees = []
            freem = []
            for i in range(len(FAM)):
                j = A[:, i:i + 1]
                shared.append(lse(j + C + pm0 + ps0, (2, 3)))
                frees.append(lse(collapse(j, ps0, 3) + collapse(C, ps0, 3) + pm[:, None, :], 2))
                freem.append(lse(collapse(j, pm0, 2) + collapse(C, pm0, 2) + ps[:, None, :], 2))
            for typ, result in (('fixed', shared), ('free_scale', frees), ('free_mean', freem)):
                joint = torch.stack(result, 1) + pf[None, :, None] + pf[None, None, :]
                records[typ] = lse(joint, (1, 2)) - total
                for i in range(len(FAM)):
                    records[f'{typ}_from{i}'] = lse(joint[:, i], 1) - total
                    records[f'{typ}_to{i}'] = lse(joint[:, :, i], 1) - total
                    if typ == 'fixed':
                        for j in range(len(FAM)):
                            records[f'pair{i}_{j}'] = joint[:, i, j] - total
            for side, v in (('pre', A), ('post', C)):
                probs = (lse(v + pm0 + ps0, (2, 3)) + pf).softmax(1)
                for j in range(len(FAM)):
                    records[f'{side}{j}'] = probs[:, j]
            for k, ii in enumerate(idx):
                ref = lse(null[:, ii] + pi[ii], 1) - total
                records[f'null{k}'] = ref
            arr = torch.stack(list(records.values()), 1).clip(-55, 110).cpu().numpy()
            rows.append(pd.DataFrame(arr, index=ids[sl], columns=list(records)))
        blocks.append(pd.concat(rows).add_prefix('native_' if native else 'standard_'))
        del B
    out = pd.concat(blocks, axis=1)
    out.index.name = 'id'
    assert np.isfinite(out).all().all()
    out.to_parquet(dst)
    print(part, stream, 'asymmetry and physical scale', out.shape, flush=True)

def run(stream):
    _, ids, tids, *_ = wl.load()
    a = pd.concat([pd.read_parquet(OUT / f'native_law_{stream}_{p}.parquet').loc[ix] for p, ix in (('train', ids), ('test', tids))])
    gate = np.concatenate([pd.read_parquet(OUT / f'innov_v2_{p}.parquet').loc[ix, 'pre_white'] for p, ix in (('train', ids), ('test', tids))])
    v = np.arcsinh(a.to_numpy().clip(-22, 55))

    def design(ix, z):
        return np.column_stack([v[ix], v[ix] * gate[ix, None]])
    wl.run('native_law_' + stream, design, 270.0, save=False)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('streams', nargs='+', choices=('pulse', 'state'))
    with threadpool_limits(limits=1), torch.inference_mode():
        torch.set_num_threads(1)
        for stream in p.parse_args().streams:
            for part in ('train', 'test'):
                extract(part, stream)
            run(stream)
