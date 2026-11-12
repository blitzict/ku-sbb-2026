# CPU port; original experiments/weight_space/white_mode_switching.py, SHA256 74abbb77f2c2d48780e6464487b4d6f857d498629ee38ceceb1c23416df32343
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Test for changes in the order of visits to two independently supported noise peaks."""
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
from scipy.special import expit, logit, logsumexp
from threadpoolctl import threadpool_limits
from white_remove_noise_modes import mixture
from partition_evidence import OUT
R = np.array([-0.88, -0.52, -0.23, 0.0, 0.13, 0.34, 0.57, 0.78, 0.94])
PR = np.log([0.03, 0.04, 0.07, 0.62, 0.07, 0.065, 0.055, 0.035, 0.015])
ODDS = np.array([-1.15, -0.65, -0.3, 0, 0.3, 0.65, 1.15])
PP = -abs(ODDS) / 0.48
PP -= logsumexp(PP)

def likelihood(w, p, q, rho):
    n = 2 ** int(np.ceil(np.log2(len(w))))
    low = np.ones((n, 1))
    high = low.copy()
    low[:len(w), 0] = (1 - w) / (1 - p)
    high[:len(w), 0] = w / p
    a = q * (1 - rho)
    b = q + rho * (1 - q)
    aa = (1 - a) * low
    ab = a * high
    ba = (1 - b) * low
    bb = b * high
    logscale = np.zeros(np.size(q))
    while len(aa) > 1:
        A, B, C, D = (v[::2] for v in (aa, ab, ba, bb))
        E, F, G, H = (v[1::2] for v in (aa, ab, ba, bb))
        aa, ab, ba, bb = (A * E + B * G, A * F + B * H, C * E + D * G, C * F + D * H)
        norm = 0.5 * (aa + ab + ba + bb)
        logscale += np.log(norm).sum(0)
        aa, ab, ba, bb = (v / norm for v in (aa, ab, ba, bb))
    return (logscale + np.log((1 - q) * (aa[0] + ab[0]) + q * (ba[0] + bb[0]))).reshape(len(ODDS), len(R))

def measure(v, j, gate):
    a, b, s, p, _, _ = mixture(v)
    w = expit(logit(p) + (b - a) * (v - (a + b) / 2) / (s * s)).clip(1e-08, 1 - 1e-08)
    q = expit(logit(p) + ODDS)[:, None] * np.ones_like(R)
    rho = R * np.where(R < 0, np.minimum(q / (1 - q), (1 - q) / q), 1.0)
    l = likelihood(w[:j], p, q.ravel(), rho.ravel())
    r = likelihood(w[j:], p, q.ravel(), rho.ravel())

    def mix(x):
        return logsumexp(x + PP[:, None] + PR)
    same = mix(l + r)
    lr, rr = (logsumexp(v + PR, axis=1) for v in (l, r))
    lq, rq = (logsumexp(v + PP[:, None], axis=0) for v in (l, r))
    zero = np.flatnonzero(R == 0)[0]
    keep = lambda x: logsumexp(x + PP)

    def expected(x):
        return (rho * np.exp(x + PP[:, None] + PR - mix(x))).sum()
    values = [mix(l) - keep(l[:, zero]), mix(r) - keep(r[:, zero]), same - keep(l[:, zero] + r[:, zero]), keep(lr + rr) - same, logsumexp(lq + rq + PR) - same, mix(l) + mix(r) - same, keep(l[:, zero] + rr) - same, keep(r[:, zero] + lr) - same, expected(l), expected(r), expected(r) - expected(l)]
    return gate * np.asarray(values).clip(-60, 100)

def extract(part):
    dst = OUT / f'mode_switching_{part}.parquet'
    if dst.exists():
        return
    with np.load(OUT / f'pulse_streams_{part}.npz') as a:
        ids, x, b, n = (a[k] for k in ('ids', 'separate', 'b', 'n'))
    strength = pd.read_parquet(OUT / f'mode_residual_{part}.parquet').loc[ids, 'gate'].to_numpy()
    gate = ((strength - 0.5) / 0.45).clip(0, 1)
    cols = 'pre_memory post_memory shared_memory altered_memory altered_occupancy altered_both onset offset pre_rho post_rho delta_rho'.split()
    rows = np.zeros((len(ids), len(cols)), np.float32)
    for i in np.flatnonzero(gate):
        rows[i] = measure(x[i, :n[i]].astype(float), b[i], gate[i])
    assert np.isfinite(rows).all()
    pd.DataFrame(rows, index=pd.Index(ids, name='id'), columns=cols).to_parquet(dst)
    print('two-mode switching', part, int((gate > 0).sum()), flush=True)
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        for part in ('train', 'test'):
            extract(part)
