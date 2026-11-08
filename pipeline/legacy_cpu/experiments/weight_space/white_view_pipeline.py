# CPU port; original experiments/weight_space/white_view_pipeline.py, SHA256 3ce022e011dc67b6aeed7bdfb15f836b614486c283a694629d17c8c25a7d1bfd
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Run original mechanism builders on each actual cropped/reversed series."""
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
import importlib
import json
import sys
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from scipy.special import logsumexp
from threadpoolctl import threadpool_limits
from partition_evidence import ROOT, OUT
from white_law_head import unpack
import white_laws
import bounded_prior

def materialize(root, view, part):
    rawdir = root / 'tune_out'
    clean = root / 'clean'
    rawdir.mkdir(exist_ok=True)
    clean.mkdir(exist_ok=True)
    dest = rawdir / f'seq_cache_{part}.npz'
    if dest.exists():
        return
    with np.load(ROOT / f'tune_out/seq_cache_{part}.npz', allow_pickle=True) as q:
        ids, x, n, b = (q[k] for k in ('ids', 'x', 'n', 'b'))
    ids = ids.astype(str)
    idx = pd.Index(ids, name='id')
    dims = pd.read_parquet(ROOT / f'tune_out/var/clean_{part}_{view}.parquet', columns=['n_pre', 'n_post'])
    suffix = 'r' if view == 'rev' else view
    keys = [k + suffix for k in ids]
    v = dims.loc[keys].to_numpy(dtype=int)
    nb = v[:, 0]
    nn = v.sum(1)
    if view == 'rev':
        assert np.array_equal(n, nn) and np.array_equal(n - b, nb)
    else:
        assert (nb <= b).all() and (nn - nb <= n - b).all()
    out = np.zeros_like(x)
    for i in range(len(ids)):
        start = b[i] - nb[i]
        out[i, :nn[i]] = x[i, :n[i]][::-1] if view == 'rev' else x[i, start:start + nn[i]]
    pd.DataFrame(v, index=idx, columns=['n_pre', 'n_post']).to_parquet(clean / f'features_{part}.parquet')
    mask = ['lr_bim_sep_pre', 'lr_bim_sep_post']
    lr = pd.read_parquet(ROOT / f'tune_out/var/lr5_{part}_{view}.parquet', columns=mask).loc[keys]
    lr.index = idx
    lr.to_parquet(rawdir / f'lr5_{part}.parquet')
    np.savez(dest, ids=ids, x=out, n=nn, b=nb)
    print('verified raw lengths and order', view, part, flush=True)

def measure(rows):
    n = torch.tensor([len(a) for a in rows], device='cpu')
    L = int(n.max())
    N = len(rows)
    padded = np.zeros((N, L), np.float32)
    for i, r in enumerate(rows):
        padded[i, :len(r)] = r
    x = torch.tensor(padded, device='cpu')
    t = torch.arange(L, device='cpu')[None, :]
    mask = t < n[:, None]
    mu = (x * mask).sum(1) / n
    sd = (((x - mu[:, None]) * mask).square().sum(1) / n).sqrt()
    z = (x - mu[:, None]) / sd[:, None] * mask
    criteria = []
    resid = []
    fields = {}
    for p in range(17):
        if p == 0:
            r = z
        else:
            w = z.unfold(1, p + 1, 1)
            response = w[:, :, -1]
            d = torch.cat([w[:, :, :-1].flip(-1), torch.ones_like(response[:, :, None])], 2)
            m = mask[:, p:]
            dm = d * m[:, :, None]
            b = torch.linalg.solve(dm.transpose(1, 2) @ d + 0.0001 * torch.eye(p + 1, device='cpu'), dm.transpose(1, 2) @ response[:, :, None])
            r = torch.nn.functional.pad((response - (d @ b)[:, :, 0]) * m, (p, 0))
            if p == 1:
                fields['phi1_ar1'] = b[:, 0, 0]
            if p == 4:
                fields['persist4'] = b[:, :p, 0].sum(1)
        ne = n - p
        v = r.square().sum(1) / ne
        if p == 4:
            fields['innov_sd4'] = v.sqrt() * sd
        criteria.append(ne * v.log() + (p + 1) * ne.log())
        resid.append(r)
    p = torch.stack(criteria, 1).argmin(1)
    r = torch.stack(resid, 1)[torch.arange(N, device='cpu'), p]
    mr = mask & (t >= p[:, None])
    rr = r / (r.square().sum(1) / (n - p)).sqrt()[:, None] * mr
    r2 = (rr.square() - 1) * mr
    fields.update(n=n, mean=mu, sd=sd, kurt=rr.pow(4).sum(1) / (n - p) - 3, sqacf1=(r2[:, 1:] * r2[:, :-1]).sum(1) / r2.square().sum(1).clamp_min(1e-12))
    return pd.DataFrame({k: v.cpu().numpy() for k, v in fields.items()})

def bounds(root):
    rows = []
    for part in ('train', 'test'):
        with np.load(root / f'tune_out/seq_cache_{part}.npz') as q:
            ids, n, b, x = (q[k] for k in ('ids', 'n', 'b', 'x'))
        for start in range(0, len(ids), 192):
            sl = slice(start, min(start + 192, len(ids)))
            for side in ('pre', 'post'):
                v = [r[:j] if side == 'pre' else r[j:end] for r, end, j in zip(x[sl], n[sl], b[sl])]
                h = measure(v)
                h['id'] = ids[sl]
                h['part'] = side
                rows.append(h)
    pd.concat(rows).to_parquet(root / 'tune_out/generator_segments.parquet', index=False)
    with patch.multiple(bounded_prior.ef, ROOT=root, O=root / 'tune_out', OUT=root):
        bounded_prior.main()

def outer(root):
    with patch.multiple(white_laws, ROOT=root, OUT=root):
        bank = white_laws.dictionary()
        for part in ('train', 'test'):
            if not (root / f'wnlaw_physical_{part}.npz').exists():
                white_laws.extract(part, 'physical', bank)
    dest = root / 'law_outer0'
    dest.mkdir(exist_ok=True)
    for f in range(5):
        p = OUT / 'law_outer0' / f'fold{f}_prior.json'
        w = json.loads(p.read_text())
        for part in ('train', 'test'):
            with np.load(root / f'wnlaw_physical_{part}.npz') as q:
                ids, M = (q['ids'], q['values'].astype(float))
            _, L, F = unpack(M, np.array(w['law']))
            score = logsumexp(L + np.log(w['change']), 1) - logsumexp(L + np.log(w['null']), 1)
            v = np.column_stack([score, F]).clip(-80, 80)
            cols = [f'v{j}' for j in range(v.shape[1])]
            pd.DataFrame(v, index=pd.Index(ids, name='id'), columns=cols).to_parquet(dest / f'fold{f}_{part}.parquet')
        rows = np.load(OUT / f'law_outer0/fold{f}_rows.npz')
        np.savez_compressed(dest / f'fold{f}_rows.npz', **{k: rows[k] for k in rows.files})
    print('all crop priors verified', root.name, flush=True)

def build(view):
    root = OUT / f'view_{view}'
    root.mkdir(exist_ok=True)
    parts = ('train', 'test')
    for p in parts:
        materialize(root, view, p)
    modules = [('periodic', 'periodic_components'), ('innov_v2', 'innovation_transform'), ('trend', 'trend_components'), ('curve', 'curve_likelihood'), ('ar_law', 'ar_likelihood'), ('phase_law', 'phase_likelihood'), ('mixture_law', 'mixture_law'), ('partition', 'partition_evidence'), ('pulse', 'pulse_filter')]
    for tag, modname in modules:
        mod = importlib.import_module(modname)
        with patch.multiple(mod, ROOT=root, OUT=root):
            for part in parts:
                if not (root / f'{tag}_{part}.parquet').exists():
                    mod.run(part)
    if not (root / 'bounds_test.parquet').exists():
        bounds(root)
    if not (root / 'noise_tests_test.parquet').exists():
        sys.path.insert(0, str(ROOT / 'experiments/v6'))
        mod = importlib.import_module('transfer_member')
        with patch.multiple(mod, ROOT=root, OUT=root):
            mod.prepare()
    outer(root)
    for p in parts:
        ix = pd.read_parquet(root / f'clean/features_{p}.parquet', columns=[]).index
        for tag in ['bounds', 'noise_tests', *[t for t, _ in modules]]:
            df = pd.read_parquet(root / f'{tag}_{p}.parquet')
            ref = pd.read_parquet(OUT / f'{tag}_{p}.parquet')
            assert ix.equals(df.index) and list(ref.columns) == list(df.columns) and np.isfinite(df.to_numpy()).all(), (tag, p)
    (root / 'READY').write_text('Verified original identifiers, columns, finiteness, and the five original training priors.')
    print('READY', view, flush=True)
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('views', nargs='+', choices=('c1', 'c2', 'c3', 'c4', 'rev'))
    with torch.inference_mode(), threadpool_limits(limits=1):
        torch.set_num_threads(1)
        for v in p.parse_args().views:
            build(v)
