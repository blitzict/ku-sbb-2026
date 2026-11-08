# CPU port; original experiments/weight_space/white_alt_innov.py, SHA256 aa84e3549b80b2f95af20d32d1f702a8d7ac19e01064202105a3637beff4f7ee
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Alternative innovations from common or pre-boundary dynamics and two-sided volatility."""
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
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import lfilter
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits
import torch
from partition_evidence import OUT
from white_jump_priors import extract, run

def sv(v):
    total = max(np.var(v), 0.1)
    likelihood = [0.0]
    paths = [np.zeros(len(v))]
    for rho in (0.5, 0.8, 0.94, 0.983):
        for frac in (0.12, 0.32, 0.58):
            var = total * frac
            r = total - var
            q = var * (1 - rho * rho)
            coeff = r * (1 - rho * rho) - q
            p = (-coeff + np.sqrt(coeff ** 2 + 4 * q * r)) * 0.5
            k = p / (p + r)
            decay = rho * (1 - k)
            f = lfilter([0.0, rho * k], [1.0, -decay], v)
            back = lfilter([0.0, rho * k], [1.0, -decay], v[::-1])[::-1]
            ll = -0.5 * (len(v) * np.log((p + r) / total) + np.sum((v - f) ** 2) / (p + r) - v @ v / total)
            rev = -0.5 * (len(v) * np.log((p + r) / total) + np.sum((v - back) ** 2) / (p + r) - v @ v / total)
            likelihood.append(0.5 * (ll + rev))
            paths.append(((f + back) / (2 - p / var)).clip(-1.8, 1.8))
    lp = np.log(np.r_[0.89, np.full(12, 0.11 / 12)]) + likelihood
    z = logsumexp(lp)
    w = np.exp(lp - z)
    return (w @ np.stack(paths), np.array(likelihood), z, 1 - w[0])

def transform(raw, b, ar, pre, walk):
    first = np.r_[raw[:1], raw[:-1]]
    second = np.r_[raw[:1], raw[:1], raw[:-2]]
    mid = (1 - walk) * np.median(raw[:b]) + walk * raw[0]
    local = raw - pre[0] * (first - mid) - pre[1] * (second - mid)
    common = raw - ar[0] * (first - mid) - ar[1] * (second - mid)
    ma, mb = (np.median(common[2:b]), np.median(common[b:]))
    e = common - np.r_[np.full(b, ma), np.full(len(raw) - b, mb)]
    sizes = [max(np.median(abs(p)) * 1.4826, 0.18 * p.std(), 1e-05) for p in (e[2:b], e[b:])]
    sd = np.r_[np.full(b, sizes[0]), np.full(len(raw) - b, sizes[1])]
    v = np.log(0.12 + abs(e) / sd)
    v -= np.r_[np.full(b, v[:b].mean()), np.full(len(raw) - b, v[b:].mean())]
    h0, l0, z0, w0 = sv(v)
    ha, la, za, wa = sv(v[:b])
    hb, lb, zb, wb = sv(v[b:])
    h = np.r_[ha, hb]
    corrected = common + e * (np.exp(-1.12 * h) - 1)
    return (local, corrected, [z0, za, zb, za + zb - z0, w0, wa, wb, np.mean(abs(h)), np.mean(abs(h - h0)), *la, *lb, *la + lb - l0])

def prepare(part):
    paths = {k: OUT / f'{k}_streams_{part}.npz' for k in ('nullstate', 'prestate', 'smoothvol')}
    if all((p.exists() for p in paths.values())):
        return
    with np.load(OUT / f'pulse_streams_{part}.npz') as a:
        ids, x, n, b = (a[k] for k in ('ids', 'separate', 'n', 'b'))
    st = pd.read_parquet(OUT / f'state_{part}.parquet').loc[ids]
    pre = st[['pre_ar1', 'pre_ar2']].to_numpy()
    common = st[['ar1', 'ar2']].to_numpy()
    walk = st.walk.to_numpy()
    with np.load(OUT / f'state_streams_{part}.npz') as a:
        assert np.array_equal(ids, a['ids'])
        null = a['common']
    local = np.zeros_like(x)
    smooth = np.zeros_like(x)
    fields = []
    for i, (v, j, end) in enumerate(zip(x, b, n)):
        raw = v[:end].astype(float)
        null[i, :end] += walk[i] * common[i].sum() * (raw[0] - raw.mean())
        local[i, :end], smooth[i, :end], f = transform(raw, j, common[i], pre[i], walk[i])
        fields.append(f)
    for k, arr in (('nullstate', null), ('prestate', local), ('smoothvol', smooth)):
        assert np.isfinite(arr).all()
        np.savez(paths[k], ids=ids, n=n, b=b, separate=arr)
    frame = pd.DataFrame(fields, index=pd.Index(ids, name='id')).astype(np.float32)
    frame.columns = [f'sv{c}' for c in frame.columns]
    frame.to_parquet(OUT / f'vol_smoothing_{part}.parquet')
    print(part, 'two-sided innovations', len(ids), flush=True)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('streams', nargs='+', choices=('nullstate', 'prestate', 'smoothvol'))
    args = p.parse_args()
    with threadpool_limits(limits=1), torch.inference_mode():
        torch.set_num_threads(1)
        for part in ('train', 'test'):
            prepare(part)
        for stream in args.streams:
            for part in ('train', 'test'):
                extract(part, stream)
            run(stream)
