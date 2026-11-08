# CPU port; original experiments/weight_space/state_tests.py, SHA256 9a5d976f52bd9ee502fe223b8f22a55d1713a47f13d63c7869d36a5ad00aa6d4
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Measure distribution and serial-law changes on the reconstructed innovations."""
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
from threadpoolctl import threadpool_limits
from partition_evidence import OUT
PROBS = np.array([0.01, 0.025, 0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.95, 0.975, 0.99])

def descriptors(a, b):
    n, m = (len(a), len(b))
    ne = np.sqrt(n * m / (n + m))
    q1, q2 = (np.quantile(a, PROBS), np.quantile(b, PROBS))
    w1, w2 = (max(q1[8] - q1[4], 1e-07), max(q2[8] - q2[4], 1e-07))
    sd = np.sqrt((a.var() * n + b.var() * m) / (n + m)).clip(1e-07)
    lev = (q2 - q1) * ne / sd
    shape = ((q2 - q2[6]) / w2 - (q1 - q1[6]) / w1) * ne
    ratio = np.log(w2 / w1) * ne
    f = {}
    for name, h in (('raw', lev), ('shape', shape)):
        f[name + '_absmean'] = abs(h.mean())
        f[name + '_spread'] = h.std()
        f[name + '_asym'] = np.mean(abs(h[:6] + h[-1:-7:-1]))
        f[name + '_scale'] = np.mean(abs(h[:6] - h[-1:-7:-1]))
        for k in (1, 3, 5):
            f[f'{name}_span{k}'] = abs(h[-1 - k] - h[k])
            f[f'{name}_skew{k}'] = abs(h[-1 - k] + h[k] - 2 * h[6])
    f.update(median=abs(lev[6]), ratio=abs(ratio), mean=abs(a.mean() - b.mean()) * ne / sd, mean_sd=sd, pre_sd=a.std(), post_sd=b.std(), raw_center=abs(q1[6]), size=ne)
    z1, z2 = ((a - q1[6]) / w1, (b - q2[6]) / w2)
    for name, u, v in (('full', (a - q1[6]) / w1, (b - q1[6]) / w1), ('unit', z1, z2)):
        freq = np.array([0.6, 1.3, 2.2, 3.5, 5.5, 8.0])
        u, v = (u[:, None] * freq, v[:, None] * freq)
        ea = np.exp(1j * u)
        eb = np.exp(1j * v)
        dist = (ea.mean(0) - eb.mean(0)) * ne
        for k, val in enumerate(abs(dist)):
            f[f'{name}_freq{k}'] = val
    for name, u, v in (('innov', z1, z2), ('abs', abs(z1), abs(z2))):
        u = (u - np.median(u)).clip(-3, 3)
        v = (v - np.median(v)).clip(-3, 3)
        u -= u.mean()
        v -= v.mean()
        da, db = (max(u @ u, 1e-08), max(v @ v, 1e-08))
        for k in (1, 2, 4):
            pa, pb = (u[k:] @ u[:-k] / da, v[k:] @ v[:-k] / db)
            f[f'{name}_pre{k}'] = pa
            f[f'{name}_lag{k}'] = abs(pa - pb) * ne
    return f

def run(part):
    frames = []
    for tag, key in [('pulse', 'separate'), ('state', 'common'), ('state', 'separate')]:
        with np.load(OUT / f'{tag}_streams_{part}.npz') as q:
            ids, x, n, b = (q[k] for k in ('ids', key, 'n', 'b'))
        frame = pd.DataFrame([descriptors(r[2:j].astype(float), r[j:e].astype(float)) for r, j, e in zip(x, b, n)], index=pd.Index(ids, name='id'))
        assert np.isfinite(frame).all().all()
        frames.append(frame.add_prefix(tag + '_' + key + '_').astype(np.float32))
    data = pd.concat(frames, axis=1)
    data.to_parquet(OUT / f'state_tests_{part}.parquet')
    print(part, data.shape, flush=True)
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        run('train')
        run('test')
