# CPU port; original experiments/weight_space/other_sign_symmetries.py, SHA256 b5d6921046a3b320cff001ff30a6f506065d6f09956f3cf2046e2496d3714c95
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Test whether an observed noise or lag pattern changes its orientation at the boundary."""
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
from scipy.special import gammaln, logsumexp
from threadpoolctl import threadpool_limits
from partition_evidence import ROOT, OUT
PERM = np.arange(8)
REV = ((PERM & 1) << 2) + (PERM & 2) + ((PERM & 4) >> 2)

def mass(count):
    return (gammaln(count + 0.6) - gammaln(0.6)).sum(-1) - gammaln(count.sum(-1) + 4.8) + gammaln(4.8)

def evidence(pre, post):
    a = pre.sum(0)
    r = post.sum(0)
    null = mass(a + r)
    alts = [mass(a + r[:, PERM ^ 7]), mass(a + r[:, REV]), mass(a + post[0][:, PERM ^ 2] + post[1][:, PERM ^ 5]), mass(a + post[0][:, PERM ^ 5] + post[1][:, PERM ^ 2])]
    mirror, reverse, alt1, alt2 = alts
    alt = np.logaddexp(alt1, alt2) - np.log(2.0)
    free = mass(a) + mass(r)
    return np.array([*(c.sum() - null.sum() for c in (mirror, reverse, alt, free)), logsumexp([mirror.sum(), reverse.sum(), alt.sum()]) - np.log(3.0) - free.sum()])

def one(row, cut):
    s = row - np.median(row)
    t = np.arange(len(s))
    block = np.where(t < cut, t * 6 // cut, 6 + (t - cut) * 6 // (len(t) - cut))
    out = []
    for lag in (1, 3):
        j = t[2 * lag:][block[2 * lag:] == block[:-2 * lag]]
        a, b, c = [s[j - i * lag] for i in (2, 1, 0)]
        group = np.digitize(abs(b), np.quantile(abs(b), [0.4, 0.8]))
        for x in (s, np.r_[s[:cut] - np.median(s[:cut]), s[cut:] - np.median(s[cut:])]):
            code = sum(((x[j - i * lag] > 0).astype(int) * (1 << i) for i in (0, 1, 2)))
            k = ((((j >= cut).astype(int) * 2 + block[j] % 2) * 2 + j % 2) * 3 + group) * 8 + code
            h = np.bincount(k, minlength=2 * 2 * 2 * 3 * 8).reshape(2, 2, 2, 3, 8)
            joint = evidence(h[0].sum(0), h[1].sum(0))
            half = [evidence(h[0, p], h[1, p]) for p in (0, 1)]
            out.extend(np.r_[joint, np.minimum(*half)])
    return np.arcsinh(np.array(out)).astype(np.float32)

def build(part):
    cols = []
    tables = []
    for tag in ('raw', 'pulse', 'state'):
        p = ROOT / f'tune_out/seq_cache_{part}.npz' if tag == 'raw' else OUT / f'{tag}_streams_{part}.npz'
        with np.load(p, allow_pickle=tag == 'raw') as a:
            ids, r, n, b = [a[k] for k in ('ids', 'x' if tag == 'raw' else 'separate', 'n', 'b')]
        names = [f'{tag}_lag{l}_{m}_{rep}_{term}' for l in (1, 3) for m in ('joint', 'side_center') for rep in ('full', 'replicate') for term in ('mirror', 'backwards', 'alternating', 'free', 'structured_advantage')]
        data = np.stack([one(x[:end].astype(float), cut) for x, end, cut in zip(r, n, b)])
        tables.append(pd.DataFrame(data, index=ids.astype(str), columns=names))
        assert np.isfinite(data).all() and tables[-1].index.equals(tables[0].index)
    table = pd.concat(tables, axis=1)
    table.to_parquet(OUT / f'sign_symmetries_{part}.parquet')
    print(part, table.shape, flush=True)
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        for part in ('train', 'test'):
            build(part)
