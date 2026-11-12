# CPU port; original experiments/weight_space/white_asymmetric_response.py, SHA256 3aea74fb34dea2cb3aa9f4488836311e5b9978b6e391c3422fb4613cafc6d58c
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Test sign-dependent volatility and magnitude-dependent drift in real innovations."""
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
from scipy.signal import lfilter
from threadpoolctl import threadpool_limits
from white_walk import logpdf
from white_jump_priors import SELECT, tensor
from white_volatility_law import interp
from partition_evidence import OUT
X = np.r_[-np.geomspace(5.3, 70, 14)[::-1], np.linspace(-5, 5, 97), np.geomspace(5.3, 70, 14)]
V = np.linspace(-3.0, 3.0, 29)
D = np.linspace(-0.9, 0.9, 25)
S = np.linspace(-0.96, 0.96, 23)
KF = [0, 1, 2, 5, 6, 8, 10, 11]
PI = np.array([0.36, 0.09, 0.19, 0.09, 0.065, 0.065, 0.08, 0.06])
PI /= PI.sum()
P = np.exp(-abs(D) / 0.28)
P /= P.sum()
P *= 0.31
P[len(D) // 2] += 0.69
PR = np.exp(-S ** 2 / (2 * 0.46 ** 2))
PR /= PR.sum()
PROGRAMS = ('prior_sign', 'long_signed_shock', 'prior_magnitude', 'long_magnitude')

def bank(location):
    v = V[None, :, None, None]
    d = D[None, None, :, None]
    s = S[None, None, None, :]
    sig = np.exp(s + (0 if location else v * d))
    z = (X[:, None, None, None] - (v * d if location else 0)) / sig
    return tensor(np.stack([logpdf(z, SELECT[k]) - np.log(sig) for k in KF], 2)).clamp_min(-30000.0).reshape(len(X) * len(V), -1).contiguous()

def chart(u, h, j):
    mid = np.r_[np.full(j, np.median(u[:j])), np.full(len(u) - j, np.median(u[j:]))]
    e = (u - mid) / h
    spread = max(1.4826 * np.median(abs(e)), 0.18 * e.std(), 0.01)
    e = e / spread
    prev = np.r_[0.0, e[:-1]]
    mag = np.log1p(abs(prev)) - 0.55
    v = [np.tanh(1.2 * prev), lfilter([0.25], [1.0, -0.75], np.tanh(prev) * np.minimum(abs(prev), 4)), mag, lfilter([0.25], [1.0, -0.75], mag)]
    pos = np.arange(len(u))
    quarter = (pos >= j).astype(int) * 2 + np.where(pos < j, 2 * pos // j, 2 * (pos - j) // (len(u) - j))
    i, alpha = interp(e, X)
    cells = len(X) * len(V)
    records = []
    for w in v:
        k, beta = interp(w, V)
        axis = (quarter * cells + i * len(V) + k).astype(int)
        out = np.zeros(4 * cells)
        for x, p in ((0, 1 - alpha), (1, alpha)):
            for y, q in ((0, 1 - beta), (1, beta)):
                out += np.bincount(axis + x * len(V) + y, weights=p * q, minlength=4 * cells)
        records.append(out.reshape(4, -1))
    return np.stack(records)

def evaluate(L):
    lse = torch.logsumexp
    pk = tensor(np.log(PI))
    pd = tensor(np.log(P))
    ps = tensor(np.log(PR))
    zero = len(D) // 2
    reduce = lambda u: lse(u + ps, 3)
    a, b, c, d = L.unbind(1)
    l, r = (reduce(a + b), reduce(c + d))
    null = lse(l + r + pd + pk[None, :, None], (1, 2))
    pre, post = [lse(u + pd, 2) for u in (l, r)]
    same0 = lse(l[:, :, zero] + r[:, :, zero] + pk, 1)
    onset = lse(l[:, :, zero] + post + pk, 1) - null
    off = lse(pre + r[:, :, zero] + pk, 1) - null

    def expected(t):
        w = (t + pd + pk[None, :, None]).flatten(1).softmax(1).reshape(t.shape)
        return (w * tensor(D)).sum((1, 2))
    ua = reduce(a)
    ub = reduce(b)
    transfer = lse(ua + ub + pd + pk[None, :, None], (1, 2)) - lse(ua + pd + pk[None, :, None], (1, 2))
    no = lse(ua[:, :, zero] + ub[:, :, zero] + pk, 1) - lse(ua[:, :, zero] + pk, 1)
    return dict(stable=null - same0, change=lse(pre + post + pk, 1) - null, on=onset, off=off, signed_pre=expected(l), signed_post=expected(r), heldout_pre=transfer - no)

def extract(part):
    dst = OUT / f'asymmetric_response_{part}.parquet'
    if dst.exists():
        return
    with np.load(OUT / f'hetero_path_{part}.npz') as a:
        ids, u, n, b, h = (a[k] for k in ('ids', 'u', 'n', 'b', 'common'))
    banks = [bank(False), bank(True)]
    rows = []
    for start in range(0, len(ids), 56):
        sl = slice(start, start + 56)
        hs = [chart(r[:end], v[:end], j) for r, v, end, j in zip(u[sl], h[sl], n[sl], b[sl])]
        H = tensor(np.stack(hs))
        cols = {}
        for p, name in enumerate(PROGRAMS):
            ll = (H[:, p].flatten(0, 1) @ banks[int(p >= 2)]).reshape(len(hs), 4, len(KF), len(D), len(S))
            ll -= ll.amax((2, 3, 4), keepdim=True)
            cols.update({name + '_' + k: v for k, v in evaluate(ll).items()})
        arr = torch.stack(list(cols.values()), 1).clamp(-60, 100).cpu().numpy()
        rows.append(pd.DataFrame(arr, index=ids[sl], columns=list(cols)))
    frame = pd.concat(rows)
    frame.index.name = 'id'
    assert np.isfinite(frame).all().all()
    frame.to_parquet(dst)
    print(part, 'sign-dependent volatility and asymmetric drift', frame.shape, flush=True)
if __name__ == '__main__':
    with threadpool_limits(limits=1), torch.inference_mode():
        torch.set_num_threads(1)
        for part in ('train', 'test'):
            extract(part)
