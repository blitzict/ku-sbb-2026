# CPU port; original experiments/weight_space/white_boundary_law.py, SHA256 cae0b521aca9472332175e5d9496875d232b75bc5ae2c7be7ac1db06e49f4f4e
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Measure how tightly each physical break model locates the supplied boundary."""
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
M = np.linspace(-1.65, 1.65, 41)
S = np.linspace(-1.15, 1.15, 29)
KINDS = ('level', 'scale', 'shape', 'any', 'scale_shape', 'level_shape')
W = np.array([0.32, 0.12, 0.18, 0.09, 0.06, 0.045, 0.045, 0.035, 0.025, 0.02, 0.02, 0.02, 0.02])
W /= W.sum()

def dictionary():
    x = (AXIS[:, None, None] - M[None, :, None]) / np.exp(S)
    gaussian = lambda z: -0.5 * z * z - 0.5 * np.log(2 * np.pi)
    laws = [gaussian(x)]
    for nu in (4.0, 8.0, 16.0):
        laws.append(gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(np.pi * (nu - 2)) - (nu + 1) / 2 * np.log1p(x * x / (nu - 2)))
    for fraction, ratio in ((0.03, 4.5), (0.1, 2.5)):
        sd = 1 / np.sqrt(1 - fraction + fraction * ratio ** 2)
        laws.append(np.logaddexp(np.log1p(-fraction) + gaussian(x / sd) - np.log(sd), np.log(fraction) + gaussian(x / (sd * ratio)) - np.log(sd * ratio)))
    for separation, p in ((0.71, 0.5), (0.86, 0.5), (0.94, 0.5), (0.81, 0.28), (0.81, 0.72), (0.92, 0.32), (0.92, 0.68)):
        sd = np.sqrt(1 - separation ** 2)
        d = separation / np.sqrt(4 * p * (1 - p))
        laws.append(np.logaddexp(np.log1p(-p) + gaussian((x + 2 * p * d) / sd), np.log(p) + gaussian((x - 2 * (1 - p) * d) / sd)) - np.log(sd))
    assert len(laws) == len(W)
    return tensor(np.stack(laws, 1) - S).clip(-20000.0, 10).flatten(1).contiguous()

def prepare(raw, b):
    m = np.median(raw)
    s = max(1.4826 * np.median(abs(raw - m)), 0.18 * raw.std(), 0.01)
    x = (raw - m) / s
    a = np.rint(np.array([0.74, 0.48, 0.27, 0.13, 0.055]) * b).astype(int)
    c = np.rint(np.array([0.055, 0.13, 0.27, 0.48, 0.74]) * (len(x) - b)).astype(int)
    cuts = np.r_[b - a, b - 3, b - 1, b, b + 1, b + 3, b + c]
    assert np.all(np.diff(cuts) >= 0) and cuts[0] > 0 and (cuts[-1] < len(x))
    edges = np.r_[0, cuts, len(x)]
    chunks = np.stack([hist(x[i:j]) for i, j in zip(edges[:-1], edges[1:])])
    counts = np.cumsum(chunks, axis=0, dtype=np.float32)
    return (counts, m, s)

def evidence(left, right, mu, sd, mass):
    lse = torch.logsumexp
    pm = mu[:, None, None, :, None]
    ps = sd[:, None, None, None, :]
    w = mass[None, None, :, None, None]
    pooled = lse(left + right + pm + ps + w, (2, 3, 4))
    lm, rm = (lse(t + pm, 3) for t in (left, right))
    ls, rs = (lse(t + ps, 4) for t in (left, right))
    lf, rf = (lse(t + w, 2) for t in (left, right))
    x = lse(lm + rm + mass[None, None, :, None] + sd[:, None, None, :], (2, 3))
    y = lse(ls + rs + mass[None, None, :, None] + mu[:, None, None, :], (2, 3))
    shape = lse(lf + rf + mu[:, None, :, None] + sd[:, None, None, :], (2, 3))
    a, b = (lse(t + pm + ps + w, (2, 3, 4)) for t in (left, right))
    sv = lse(lse(lf + sd[:, None, None, :], 3) + lse(rf + sd[:, None, None, :], 3) + mu[:, None, :], 2)
    mv = lse(lse(lf + mu[:, None, :, None], 2) + lse(rf + mu[:, None, :, None], 2) + sd[:, None, :], 2)
    return torch.stack([x, y, shape, a + b, sv, mv], -1) - pooled[:, :, None]

def extract(part, stream):
    path = OUT / f'boundary_law_{stream}_{part}.parquet'
    if path.exists():
        return
    with np.load(OUT / f'{stream}_streams_{part}.npz') as d:
        ids, n, b, x = (d[k] for k in ('ids', 'n', 'b', 'separate'))
    bank = dictionary()
    mass = tensor(W).log()
    blocks = []
    for start in range(0, len(x), 24):
        end = min(len(x), start + 24)
        group = [prepare(r[:l].astype(float), j) for r, l, j in zip(x[start:end], n[start:end], b[start:end])]
        counts, m, s = map(np.asarray, zip(*group))
        size = len(m)
        ll = (tensor(counts).reshape(-1, len(AXIS)) @ bank).reshape(size, -1, len(W), len(M), len(S))
        left = ll[:, :-1]
        right = ll[:, -1:] - left
        left = left - left.amax((2, 3, 4), keepdim=True)
        right = right - right.amax((2, 3, 4), keepdim=True)
        absolute = m[:, None] + s[:, None] * M
        scale = np.log(s[:, None]) + S
        mu = tensor(np.log(0.89 / 4 / (1 + np.exp(((abs(absolute) - 2) / 0.075).clip(-35, 35))) + 0.11 * np.exp(-abs(absolute) / 2) / 4) + np.log(s[:, None] * np.diff(M)[0]))
        sd = tensor(np.log(0.88 / 1.4 / (1 + np.exp(((abs(scale) - 0.7) / 0.045).clip(-35, 35))) + 0.12 * np.exp(-abs(scale)) * 0.5) + np.log(np.diff(S)[0]))
        v = evidence(left, right, mu, sd, mass).cpu().numpy()
        center = v[:, 7, :]
        features = {}
        for j, name in enumerate(KINDS):
            h = v[:, :, j]
            c = center[:, j]
            features[name + '_at'] = c
            for offset, tag in ((1, 'one'), (2, 'three'), (3, 'near'), (4, 'short'), (5, 'mid'), (6, 'long'), (7, 'far')):
                dl = c - h[:, 7 - offset]
                dr = c - h[:, 7 + offset]
                features[name + '_' + tag + '_weak'] = np.minimum(dl, dr)
                features[name + '_' + tag + '_strong'] = np.maximum(dl, dr)
            for t in (0.35, 1.0):
                rel = (h - c[:, None]) * t
                features[name + f'_odds{t}'] = -np.logaddexp.reduce(rel[:, [0, 1, 2, 3, 4, 10, 11, 12, 13, 14]], axis=1) + np.log(10)
        blocks.append(pd.DataFrame(features, index=ids[start:end]).clip(-140, 240).astype(np.float32))
    df = pd.concat(blocks)
    df.index.name = 'id'
    assert np.isfinite(df).all().all()
    df.to_parquet(path)
    print(part, stream, df.shape, flush=True)

def run(stream):
    _, ids, tids, *_ = wl.load()
    a = pd.concat([pd.read_parquet(OUT / f'boundary_law_{stream}_{p}.parquet').loc[ix] for p, ix in (('train', ids), ('test', tids))])
    w = np.concatenate([pd.read_parquet(OUT / f'innov_v2_{p}.parquet').loc[ix, 'pre_white'] for p, ix in (('train', ids), ('test', tids))])
    v = np.arcsinh(a.to_numpy().clip(-26, 65))

    def design(ix, z):
        return np.column_stack([v[ix], v[ix] * w[ix, None]])
    wl.run('boundary_law_' + stream, design, 360.0, save=False)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('streams', nargs='+', choices=('state', 'pulse', 'nullstate'))
    with torch.inference_mode(), threadpool_limits(limits=1):
        torch.set_num_threads(1)
        for stream in p.parse_args().streams:
            for part in ('train', 'test'):
                extract(part, stream)
            run(stream)
