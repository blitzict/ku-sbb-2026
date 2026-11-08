# CPU port; original experiments/weight_space/white_remove_noise_modes.py, SHA256 84cdfc1d6441749528d9a29537f18b428716ee430444f0855697dd88672bf196
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Separate visits to two real noise levels from motion within each noise level."""
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
from scipy.special import expit, logit
from threadpoolctl import threadpool_limits
from partition_evidence import OUT

def mixture(v):
    q = np.quantile(v, [0.23, 0.77])
    span = max(v.std(), 0.01)
    a, b = (q[0], q[1])
    sd = max(0.58 * span, 0.01)
    p = 0.5
    for _ in range(24):
        w = expit(logit(p) + (b - a) * (v - (a + b) / 2) / (sd * sd))
        p = np.clip(w.mean(), 0.06, 0.94)
        a = v @ (1 - w) / max(1.0, (1 - w).sum())
        b = v @ w / max(1.0, w.sum())
        sd = np.sqrt(np.mean((1 - w) * (v - a) ** 2 + w * (v - b) ** 2)).clip(0.12 * span, None)
    log_l = np.logaddexp(np.log(1 - p) - 0.5 * ((v - a) / sd) ** 2, np.log(p) - 0.5 * ((v - b) / sd) ** 2).sum() - len(v) * np.log(sd)
    log_single = -0.5 * len(v) - len(v) * np.log(span)
    return (a, b, sd, p, log_l - log_single - np.log(len(v)), (b - a) / sd)

def one(v, j):
    left, right = (v[:j], v[j:])
    a, b, s, p, pooled, sep = mixture(v)
    l = mixture(left)
    r = mixture(right)

    def reliable(param):
        return expit((param[4] - 2.0) / 1.8) * expit((param[5] - 2.5) / 0.38)
    gl, gr = (reliable(l), reliable(r))
    gate = max(gl, gr) * expit((sep - 2.15) / 0.35)
    w = expit(logit(p) + (b - a) * (v - (a + b) / 2) / (s * s))
    center = a + w * (b - a)
    residue = (1 - gate) * v + gate * (v - center + v.mean())
    occupancy = (1 - gate) * v + gate * (center + 0.18 * (v - center))
    dv = lambda x: x[j:].mean() - x[:j].mean()
    nd = np.sqrt(j * (len(v) - j) / len(v))
    wm = [w[:j], w[j:]]
    res = v - center

    def gap(weight):
        g = []
        for u, e in zip(weight, (res[:j], res[j:])):
            g.append(u @ e / max(u.sum(), 1.0))
        return (g[1] - g[0]) * nd / s
    tests = [gate, gl, gr, sep, l[5], r[5], pooled, l[4], r[4], dv(w) * nd, dv(res) * nd / s, gap(wm), gap([1 - v for v in wm])]
    return (residue, occupancy, np.asarray(tests).clip(-80, 150))

def extract(part):
    stream = OUT / f'mode_residual_streams_{part}.npz'
    path = OUT / f'mode_residual_{part}.parquet'
    if stream.exists() and path.exists():
        return
    with np.load(OUT / f'pulse_streams_{part}.npz') as a:
        ids, x, n, b = (a[k] for k in ('ids', 'separate', 'n', 'b'))
    e = np.zeros_like(x)
    u = np.zeros_like(x)
    chart = []
    for i, (row, end, j) in enumerate(zip(x, n, b)):
        v = row[:end].astype(float)
        e[i, :end], u[i, :end], s = one(v, j)
        chart.append(s)
    cols = 'gate pre_gate post_gate separation separation_pre separation_post pooled_bf pre_bf post_bf occupancy_change residual_change upper_motion lower_motion'.split()
    tab = pd.DataFrame(np.asarray(chart, dtype=np.float32), columns=cols, index=pd.Index(ids, name='id'))
    assert np.isfinite(tab).all().all() and np.isfinite(e).all() and np.isfinite(u).all()
    tab.to_parquet(path)
    np.savez_compressed(stream, ids=ids, n=n, b=b, common=e, separate=u)
    print('within-mode residual and mode occupancy', part, tab.shape, 'strongly gated', int((tab.gate > 0.8).sum()), flush=True)
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        for part in ('train', 'test'):
            extract(part)
