# CPU port; original experiments/weight_space/white_phase_coupling.py, SHA256 f2cd3e43b96938f8f9f5434793d3d6f471ebdc32bceb495188e0dda932939598
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Look for reproducible changes in how the phases of three frequencies line up."""
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
import json
from functools import lru_cache
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from white_signature_screen import pairs
from partition_evidence import OUT

@lru_cache(None)
def triads(w):
    i, j = np.triu_indices(w // 2 - 1)
    i, j = (i + 1, j + 1)
    take = i + j < w // 2
    return (i[take], j[take], np.hanning(w))

def spectrum(x, w):
    i, j, taper = triads(w)
    step = w // 2
    starts = np.rint(np.linspace(0, len(x) - w, 1 + (len(x) - w) // step)).astype(int)
    windows = np.lib.stride_tricks.sliding_window_view(x, w)[starts] * taper
    f = np.fft.rfft(windows, axis=1)
    lhs = f[:, i] * f[:, j]
    rhs = f[:, i + j]
    product = lhs * rhs.conjugate()
    diagonal = np.sum(windows ** 3, axis=1)
    scale = np.sqrt(np.mean(abs(lhs) ** 2, axis=0) * np.mean(abs(rhs) ** 2, axis=0)).clip(1e-06)
    return (np.mean(product - diagonal[:, None], axis=0) / scale, len(starts))

def side(x, w):
    scale = max(1.4826 * np.median(abs(x - np.median(x))), 0.15 * x.std(), 0.001)
    x = ((x - np.median(x)) / scale).clip(-5, 5)
    x -= x.mean()
    return [spectrum(a, w) for a in np.array_split(x, 2)]

def measure(left, right, w):
    a, b = (side(left, w), side(right, w))
    deltas = []
    pre = []
    post = []
    for (l, n), (r, m) in zip(a, b):
        deltas.append((r - l) * np.sqrt(n * m / (n + m)))
        pre.append(l * np.sqrt(n))
        post.append(r * np.sqrt(m))
    i, j, _ = triads(w)
    res = {}
    for name, pick in (('harmonic', i == j), ('mixed', i != j)):
        if not np.any(pick):
            continue
        u, v = [h[pick] for h in deltas]
        repeat = u * v.conj()
        res[name + '_replicated'] = repeat.real.mean()
        res[name + '_rotation'] = (u.imag * v.imag).mean()
        res[name + '_magnitude'] = np.mean(abs(0.5 * (u + v)) ** 2)
        res[name + '_pooled_order'] = np.mean((post[0][pick] * post[1][pick].conj() - pre[0][pick] * pre[1][pick].conj()).real)
    return res

def main(part):
    with np.load(OUT / f'pulse_streams_{part}.npz') as f:
        ids, raw, n, b = [f[k] for k in ('ids', 'separate', 'n', 'b')]
    phi = pd.read_parquet(OUT / f'state_{part}.parquet').loc[ids, ['ar1', 'ar2']].to_numpy()
    records = []
    for row, end, cut, p in zip(raw, n, b, phi):

        def err(v):
            return (v[2:] - p[0] * v[1:-1] - p[1] * v[:-2]).astype(float)
        pre, post = (err(row[:cut]), err(row[cut:end]))
        rec = {}
        for w in (16, 32):
            rec.update({f'w{w}_{k}': v for k, v in measure(pre, post, w).items()})
        records.append(rec)
    df = pd.DataFrame(records, index=ids)
    assert np.isfinite(df).all().all()
    df.to_parquet(OUT / f'phase_coupling_{part}.parquet')
    print('biphase', part, df.shape, flush=True)

def matched():
    ids, a, b, meta = pairs()
    tab = pd.read_parquet(OUT / 'phase_coupling_train.parquet').loc[ids]
    v = tab.filter(like='_replicated').mean(1).to_numpy()
    d = v[a] - v[b]
    rng = np.random.default_rng(745)
    perm = np.concatenate([(rng.integers(2, size=(512, len(d)), dtype=np.int8) * 2 - 1) @ d / len(d) for _ in range(48)])
    out = dict(n=len(a), positive=float(v[a].mean()), controls=float(v[b].mean()), gap=float(d.mean()), pair_win=float(np.mean((d > 0) + 0.5 * (d == 0))), one_sided_sign_flip_p=float((1 + np.sum(perm >= d.mean())) / (1 + len(perm))), scope='Historical white-like, low-vote, phase-inactive pairs matched before this test. Reused observational cohort.')
    (OUT / 'phase_coupling_matched.json').write_text(json.dumps(out, indent=2))
    print(out, flush=True)
if __name__ == '__main__':
    t = np.arange(80)
    series = np.sin(0.41 * t) + 0.5 * np.cos(0.82 * t) + 0.1 * np.sin(0.7 * t)
    u, _ = spectrum(series, 16)
    v, _ = spectrum(series[::-1], 16)
    np.testing.assert_allclose(u, v.conj(), atol=2e-07)
    with threadpool_limits(limits=1):
        for part in ('train', 'test'):
            main(part)
        matched()
