# CPU port; original experiments/weight_space/periodic_components.py, SHA256 29b702ad39a7ce816e80853dacc82448a9325257563ea67383097f3d6fced291
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Detect narrow spectral lines and isolate the underlying noise in real series."""
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
from functools import lru_cache
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import ks_2samp
from threadpoolctl import threadpool_limits
from harmonic_change import fit
from partition_evidence import ROOT, OUT
PERIOD = np.arange(8, 65)
OMEGA = 2 * np.pi / PERIOD
TIME = np.arange(-1024, 1025)
COS = np.cos(TIME[:, None] * OMEGA)
SIN = np.sin(TIME[:, None] * OMEGA)

@lru_cache(maxsize=1600)
def gram(n, left):
    t = slice(1024 - n, 1024) if left else slice(1024, 1024 + n)
    c, s = (COS[t], SIN[t])
    cc, ss = (c.sum(0), s.sum(0))
    G = np.stack([(c * c).sum(0) - cc * cc / n, (s * s).sum(0) - ss * ss / n, (c * s).sum(0) - cc * ss / n])
    return (t, G)

def eq(x, left):
    t, G = gram(len(x), left)
    z = x - x.mean()
    return (G, np.stack([z @ COS[t], z @ SIN[t]]), z @ z)

def residuals(a, b):
    n, m = (len(a), len(b))
    A, u, va = eq(a, True)
    B, v, vb = eq(b, False)
    h, ca, sa = fit(A + B, u + v)
    h1, c1, s1 = fit(A, u)
    h2, c2, s2 = fit(B, v)
    i = np.argmax(h1 + h2)
    p = PERIOD[i]
    t = np.arange(-n, m)
    c = np.cos(t * OMEGA[i])
    s = np.sin(t * OMEGA[i])
    fa, fb = (h1[i] / max(va, 1e-09), h2[i] / max(vb, 1e-09))
    frac = (h1[i] + h2[i]) / max(va + vb, 1e-09)
    alpha = expit((frac - 0.16) / 0.035) * expit((max(fa, fb) - 0.17) / 0.045)
    common = ca[i] * c + sa[i] * s
    separate = np.r_[c1[i] * c[:n] + s1[i] * s[:n], c2[i] * c[n:] + s2[i] * s[n:]]
    raw = np.r_[a, b]
    ycommon, ysep = (raw - alpha * common, raw - alpha * separate)
    r, q = (a - separate[:n], b - separate[n:])
    nv = max((n * r.var() + m * q.var()) / (n + m), 1e-09)
    sig = np.sqrt(nv)
    j, k = (np.argmax(h1), np.argmax(h2))
    gain = (h1[i] + h2[i] - h[i]) / nv
    gain_f = (h1[j] + h2[k] - h1[i] - h2[i]) / nv
    q1, q2 = (complex(c1[i], s1[i]), complex(c2[i], s2[i]))
    length = np.sqrt(n * m / (n + m))
    feats = {'gate': alpha, 'period': p, 'fraction': frac, 'pre_fraction': fa, 'post_fraction': fb, 'common_fraction': h[i] / max(va + vb, 1e-09), 'phase_gain': np.log1p(max(0, gain)), 'freq_gain': np.log1p(max(0, gain_f)), 'period_diff': abs(PERIOD[j] - PERIOD[k]), 'amplitude_change': abs(abs(q1) - abs(q2)) * length / sig, 'phase_change': abs(np.angle(q1 * q2.conjugate())) * min(abs(q1), abs(q2)) * length / sig}
    for name, values in (('common', ycommon), ('separate', ysep)):
        u, v = (values[:n], values[n:])
        mu, mv = (u.mean(), v.mean())
        sd1, sd2 = (max(u.std(), 1e-07), max(v.std(), 1e-07))
        s1, s2 = ((u - mu) / sd1, (v - mv) / sd2)
        rr = lambda w, d: w[d:] @ w[:-d] / max(w @ w, 1e-08)
        ma, mb = (np.median(u), np.median(v))
        abs1, abs2 = (abs(u - ma), abs(v - mb))
        med1, med2 = (max(np.median(abs1), 1e-07), max(np.median(abs2), 1e-07))
        local = dict(level=abs(mu - mv) * length / sig, median=abs(ma - mb) * length / sig, scale=abs(np.log(sd2 / sd1)) * length, mad=abs(np.log(med2 / med1)) * length, shape=ks_2samp(s1, s2, method='asymp').statistic * length, absmean=abs(np.mean(abs(s1)) - np.mean(abs(s2))) * length, phi1=abs(rr(s1, 1) - rr(s2, 1)) * length, phi2=abs(rr(s1, 2) - rr(s2, 2)) * length, seasonal=abs(rr(s1, p) - rr(s2, p)) * length, sqphi=abs(rr(s1 ** 2 - 1, 1) - rr(s2 ** 2 - 1, 1)) * length, logscale=np.log(sig), absmean_pre=abs(mu), absmean_post=abs(mv))
        for key, val in local.items():
            feats[name + '_' + key] = val
    return (ycommon, ysep, feats)

def run(part):
    with np.load(ROOT / f'tune_out/seq_cache_{part}.npz', allow_pickle=True) as z:
        ids, x, n, b = (z[k] for k in ('ids', 'x', 'n', 'b'))
    index = pd.read_parquet(ROOT / f'clean/features_{part}.parquet', columns=[]).index
    assert np.array_equal(ids, index)
    common, separate = (np.zeros_like(x), np.zeros_like(x))
    rows = []
    for i in range(len(x)):
        s = x[i, :n[i]].astype(float)
        u, v, f = residuals(s[:b[i]], s[b[i]:])
        common[i, :n[i]], separate[i, :n[i]] = (u, v)
        rows.append(f)
    tab = pd.DataFrame(rows, index=index)
    assert np.isfinite(tab).all().all() and np.isfinite(common).all() and np.isfinite(separate).all()
    tab.to_parquet(OUT / f'periodic_{part}.parquet')
    np.savez(OUT / f'periodic_streams_{part}.npz', ids=index.to_numpy(dtype=str), common=common, separate=separate, n=n, b=b)
    print(part, 'gate > .5', (tab.gate > 0.5).mean(), flush=True)
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        run('train')
        run('test')
