# CPU port; original experiments/weight_space/white_population_transport.py, SHA256 967bd7b6a426cfcbd35534e44cfca76c5740f69f1f0bef1bc17785634802be26
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Learn the physical distribution of clean regimes and transport it through breaks."""
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
from scipy.ndimage import gaussian_filter
from scipy.special import expit
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits
from white_jump_priors import SELECT, MASS, tensor
from white_walk import logpdf
from partition_evidence import OUT, ROOT
MU = np.linspace(-4.2, 4.2, 93)
S = np.linspace(-1.65, 1.32, 39)
AX = np.r_[-np.geomspace(8.2, 100, 26)[::-1], np.linspace(-8, 8, 401), np.geomspace(8.2, 100, 26)]
KK = [0, 1, 2, 4, 5, 6, 8, 9, 10, 11]
K = len(KK)
KEEP = 640

def chart(v):
    j = (np.searchsorted(AX, v) - 1).clip(0, len(AX) - 2)
    a = ((v - AX[j]) / (AX[j + 1] - AX[j])).clip(0, 1)
    return np.bincount(j, 1 - a, minlength=len(AX)) + np.bincount(j + 1, a, minlength=len(AX))

def base():
    m = 0.82 * expit((2 - abs(MU)) / 0.12) / 4 + 0.18 * np.exp(-abs(MU) / 1.5) / 3
    s = 0.78 * expit((0.72 - abs(S)) / 0.1) + 0.22 * np.exp(-abs(S))
    p = MASS[KK, None, None] * m[None, :, None] * s[None, None, :]
    return p / p.sum()

def proposals():
    specs = [('null', 0.0)]
    for name, amounts in (('location', (0.09, 0.23, 0.55, 1.1, 1.9)), ('absolute', (0.2, 0.65, 1.5)), ('width', (0.06, 0.16, 0.34, 0.6, 0.84)), ('zero_gain', (0.06, 0.16, 0.34, 0.6, 0.84)), ('part_gain', (0.13, 0.35, 0.62))):
        for d in amounts:
            specs.extend([(name, d), (name, -d)])
    specs.extend([('family', j) for j in range(K)])
    return specs

def target_coords(k, m, s, name, d):
    loc = MU[0] + m * (MU[1] - MU[0])
    log = S[0] + s * (S[1] - S[0])
    if name == 'location':
        loc = loc + d * log.exp()
    if name == 'absolute':
        loc = loc + d
    if name in ('width', 'zero_gain', 'part_gain'):
        log = log + d
        if name != 'width':
            loc = loc * np.exp(d * (1.0 if name == 'zero_gain' else 0.5))
    if name == 'family':
        k = torch.full_like(k, int(d))
    return (k, (loc - MU[0]) / (MU[1] - MU[0]), (log - S[0]) / (S[1] - S[0]))

def lookup(post, k, m, s):
    valid = (m >= -0.05) & (m <= len(MU) - 0.95) & (s >= -0.05) & (s <= len(S) - 0.95)
    m = m.clamp(0, len(MU) - 1)
    s = s.clamp(0, len(S) - 1)
    i = m.long().clamp_max(len(MU) - 2)
    j = s.long().clamp_max(len(S) - 2)
    u = m - i
    v = s - j

    def get(a, b):
        return post.gather(1, (k * len(MU) + i + a) * len(S) + j + b)
    o = (1 - u) * ((1 - v) * get(0, 0) + v * get(0, 1)) + u * ((1 - v) * get(1, 0) + v * get(1, 1))
    return o.masked_fill(~valid, -450)

def prepare(stream, part):
    dst = OUT / f'population_transport_{stream}_{part}.npz'
    if dst.exists():
        return dst
    with np.load(OUT / f'{stream}_streams_{part}.npz') as a:
        ids, x, n, b = [a[k] for k in ('ids', 'separate', 'n', 'b')]
    xx = (AX[:, None, None] - MU[None, :, None]) / np.exp(S)
    bank = tensor(np.stack([logpdf(xx, SELECT[k]) - S for k in KK], 1)).clamp_min(-30000).flatten(1).contiguous()
    prior = tensor(np.log(base())).flatten()
    specs = proposals()
    ii = np.empty((len(ids), KEEP), np.uint16)
    left = np.empty_like(ii, dtype=np.float16)
    right = np.empty((len(ids), KEEP, len(specs)), np.float16)
    retained = []
    assert len(prior) < 65536
    for start in range(0, len(ids), 70):
        sl = slice(start, start + 70)
        hs = []
        for row, end, j in zip(x[sl], n[sl], b[sl]):
            hs.extend([chart(row[:j]), chart(row[j:end])])
        L = (tensor(np.stack(hs)) @ bank).reshape(-1, 2, len(prior))
        L -= L.amax(2, keepdim=True)
        a, c = L.unbind(1)
        sc, idx = (a + prior).topk(KEEP, dim=1)
        retained.extend(torch.exp(sc.logsumexp(1) - (a + prior).logsumexp(1)).clamp_max(1.0).cpu().tolist())
        ii[sl] = idx.cpu().numpy().astype(np.uint16)
        left[sl] = a.gather(1, idx).clamp_min(-450).cpu().numpy()
        k = idx // (len(MU) * len(S))
        m = (idx // len(S) % len(MU)).float()
        s = (idx % len(S)).float()
        out = [lookup(c, *target_coords(k, m, s, name, d)) for name, d in specs]
        right[sl] = torch.stack(out, 2).clamp_min(-450).cpu().numpy()
    np.savez(dst, ids=ids, i=ii, a=left, b=right, retained=np.array(retained, np.float32))
    print(stream, part, 'retained pre likelihood mass', np.quantile(retained, [0, 0.01, 0.5]), flush=True)
    return dst

def prior_fit(indices, likelihood, white, tr):
    shape = base().shape
    old = base().flatten()
    pop = []
    for gate in (white, 1 - white):
        g = gate[tr]
        ix = indices[tr].astype(np.int64)
        a = likelihood[tr].astype(np.float32)
        r = np.exp(a - a.max(1, keepdims=True))
        p = old.copy()
        for _ in range(12):
            v = r * p[ix]
            v *= g[:, None] / v.sum(1, keepdims=True).clip(1e-30)
            cnt = np.bincount(ix.ravel(), weights=v.ravel(), minlength=len(old)).reshape(shape)
            cnt = gaussian_filter(cnt, (0, 0.95, 1.0), mode='nearest').ravel()
            p = (cnt + 185 * old) / (cnt.sum() + 185)
        pop.append(p)
    return np.stack(pop)

def odds(idx, a, b, white, pop):
    result = []
    spec = proposals()
    types = np.array([s for s, d in spec])
    for start in range(0, len(a), 96):
        sl = slice(start, start + 96)
        ix = idx[sl]
        w = white[sl, None]
        pr = tensor(w * pop[0, ix] + (1 - w) * pop[1, ix]).log()
        z = torch.logsumexp((tensor(a[sl]) + pr)[:, :, None] + tensor(b[sl]), 1)
        z = z - z[:, 0:1]
        items = []
        all_latent = z.clamp(-60, 90).cpu().numpy()
        for name in ('location', 'absolute', 'width', 'zero_gain', 'part_gain', 'family'):
            cols = np.flatnonzero(types == name)
            q = z[:, cols]
            items.append(torch.logsumexp(q, 1) - np.log(len(cols)))
            if name != 'family':
                items.extend([torch.logsumexp(q[:, :2], 1) - np.log(2), torch.logsumexp(q[:, 0::2], 1) - torch.logsumexp(q[:, 1::2], 1)])
        result.append((all_latent, torch.stack(items, 1).clamp(-45, 80).cpu().numpy()))
    return [np.concatenate(v) for v in zip(*result)]

def fit(stream):
    A = [np.load(prepare(stream, part)) for part in ('train', 'test')]
    ids, tids = [p['ids'] for p in A]
    idx, a, b = [np.concatenate([p[k] for p in A]) for k in ('i', 'a', 'b')]
    n = len(ids)
    y = pd.read_csv(ROOT / 'data/train_labels.csv', dtype={'id': str}).set_index('id').loc[ids, 'label'].to_numpy()
    white = np.concatenate([pd.read_parquet(OUT / f'innov_v2_{part}.parquet').loc[ix, 'pre_white'] for part, ix in (('train', ids), ('test', tids))])
    folders = OUT / f'population_transport_{stream}_models'
    folders.mkdir(exist_ok=True)
    from white_law_head import weights
    from scipy.special import logsumexp
    prior = np.r_[0.64, np.repeat(0.36 / (len(proposals()) - 1), len(proposals()) - 1)]
    test = []
    for f, (tr, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=0).split(ids, y)):
        pop = prior_fit(idx, a, white, tr)
        latent, small = odds(idx, a, b, white, pop)
        extra = []
        rates = []
        for temp in (0.45, 1.0):
            z = temp * latent
            p = [weights(z[tr[y[tr] == k]], prior, 130.0) for k in (0, 1)]
            extra.append(logsumexp(z + np.log(p[1]), 1) - logsumexp(z + np.log(p[0]), 1))
            rates.append(p)
        h = np.column_stack([small, *extra])
        assert np.isfinite(h).all()
        if not test:
            out = np.empty((n, h.shape[1]))
        out[va] = h[va]
        test.append(h[n:])
        np.savez_compressed(folders / f'fold{f}.npz', training_ids=ids[tr], validation_ids=ids[va], pop=pop, rates=rates)
        print(stream, 'population fold', f, flush=True)
    for part, ix, rows in (('train', ids, out), ('test', tids, np.mean(test, 0))):
        t = pd.DataFrame(rows, index=pd.Index(ix, name='id'), columns=[f'v{k}' for k in range(rows.shape[1])])
        t.to_parquet(OUT / f'transport_{stream}_{part}.parquet')
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('streams', nargs='+', choices=('state', 'pulse'))
    with threadpool_limits(limits=1), torch.inference_mode():
        torch.set_num_threads(1)
        for stream in p.parse_args().streams:
            fit(stream)
