# CPU port; original experiments/weight_space/phase_likelihood.py, SHA256 86a5f1545ee3957670af4872bbde6a85bd8fd372b319dcd6c2ec10a12b85eac1
# Float32 CPU operations replace accelerator autocast. Fresh workspace required.
"""Estimate the period, phase, amplitude and observation-law interventions separately."""
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
from scipy.special import gammaln, logsumexp
from threadpoolctl import threadpool_limits
from partition_evidence import ROOT, OUT
DFS = np.array([3.0, 4.0, 6.0, 10.0, 20.0, 100.0])

def student_beta(M, x, beta, df):
    scale = max(1e-05, np.std(x - M @ beta))
    for _ in range(5):
        e = x - M @ beta
        w = (df + 1) / (df + (e / scale) ** 2)
        G = M.T @ (w[:, None] * M)
        beta = np.linalg.solve(G + 1e-08 * np.eye(len(beta)), M.T @ (w * x))
        scale = np.sqrt(max(1e-10, np.mean(w * (x - M @ beta) ** 2)))
    e = (x - M @ beta) / scale
    w = (df + 1) / (df + e ** 2)
    G = M.T @ (w[:, None] * M)
    bread = np.linalg.inv(G + 1e-08 * np.eye(len(beta)))
    score = M * (e * w)[:, None]
    meat = score.T @ score
    for lag in (1, 2):
        cov = score[lag:].T @ score[:-lag]
        meat += (1 - lag / 3) * (cov + cov.T)
    robust = scale ** 2 * bread @ meat @ bread
    return (beta, scale, robust, e)

def ll_grid(res):
    s = max(res.std(), 1e-08)
    scaled = res[:, None] / s
    a = np.sqrt((DFS - 2) / DFS)
    for _ in range(5):
        u = scaled / a
        a = np.sqrt(np.maximum(1e-10, np.mean((DFS + 1) / (DFS + u * u) * scaled ** 2, axis=0)))
    u = scaled / a
    norm = gammaln((DFS + 1) / 2) - gammaln(DFS / 2) - 0.5 * np.log(DFS * np.pi) - np.log(a * s)
    return len(res) * norm - 0.5 * (DFS + 1) * np.log1p(u * u / DFS).sum(0)

def one(s, b, p):
    n, m = (b, len(s) - b)
    t = np.arange(len(s))
    theta = 2 * np.pi * t / p
    M = np.column_stack([np.ones(len(s)), np.cos(theta), np.sin(theta)])
    A, B = (M[:n], M[n:])
    a, c = (s[:n], s[n:])
    ba, bc = (np.linalg.lstsq(A, a, rcond=None)[0], np.linalg.lstsq(B, c, rcond=None)[0])
    la, lc = (ll_grid(a - A @ ba), ll_grid(c - B @ bc))
    d = DFS[np.argmax(la + lc)]
    ba, sa, va, ea = student_beta(A, a, ba, d)
    bc, sc, vc, ec = student_beta(B, c, bc, d)
    sig = np.sqrt((n * sa * sa + m * sc * sc) / (n + m))
    v = va + vc + 1e-09 * np.eye(3)
    delta = bc - ba
    amp1, amp2 = (np.linalg.norm(ba[1:]), np.linalg.norm(bc[1:]))
    wa, wb = (ba[1:] / max(amp1, 1e-07), bc[1:] / max(amp2, 1e-07))
    normal = wa + wb
    normal /= max(np.linalg.norm(normal), 1e-07)
    tangent = np.array([-normal[1], normal[0]])
    vec = np.stack([normal, tangent])
    vrot = vec @ v[1:, 1:] @ vec.T
    change = vec @ delta[1:]
    axes = np.r_[delta[0], change]
    variance = np.r_[v[0, 0], np.diag(vrot)].clip(1e-09)
    stat = axes ** 2 / variance
    prior = np.array([0.8 * sig, 0.6 * sig, amp1 * 0.9 + 0.2 * sig]) ** 2
    bf = 0.5 * (stat * prior / (variance + prior) - np.log1p(prior / variance))
    reset = np.array([[np.cos(theta[b]), -np.sin(theta[b])], [np.sin(theta[b]), np.cos(theta[b])]])
    wh = ba[1:]

    def wave_loss(dif):
        return 0.5 * dif @ np.linalg.solve(v[1:, 1:], dif)
    reference = wave_loss(delta[1:])
    resets = [reference - wave_loss(bc[1:] - r @ wh) for r in (reset, reset.T, -np.eye(2))]
    logratio = np.log(sc / sa)
    iscale = n * m / (n + m) * 2 * d / (d + 3)
    bscale = 0.5 * (iscale * logratio ** 2 * (0.3 * iscale / (1 + 0.3 * iscale)) - np.log1p(0.3 * iscale))
    shape = logsumexp(la) + logsumexp(lc) - logsumexp(la + lc) - np.log(len(DFS))
    residual_shape = abs(np.median(abs(ea)) - np.median(abs(ec))) * np.sqrt(n * m / (n + m))

    def corr(e, lag):
        ec = e - np.mean(e)
        return ec[lag:] @ ec[:-lag] / max(ec @ ec, 1e-10)
    temporal = max(abs(corr(ea, 1) - corr(ec, 1)), abs(corr(ea, 2) - corr(ec, 2))) * np.sqrt(n * m / (n + m))
    values = dict(df=d, df_l=DFS[la.argmax()], df_r=DFS[lc.argmax()], shape_bf=shape, scale_bf=bscale, level_bf=bf[0], amplitude_bf=bf[1], phase_bf=bf[2], common_phase=reference, reset_bf=max(resets), amp_l=amp1 / sig, amp_r=amp2 / sig, scale_l=sa, scale_r=sc, shape_abs=residual_shape, temporal=temporal)
    comp = np.array([bf[0], bf[1], bf[2], bscale, shape])
    values['max_evidence'] = comp.max()
    values['mixture_evidence'] = logsumexp(np.r_[comp + np.log([0.22, 0.12, 0.18, 0.27, 0.11]), np.log(0.1)])
    return values

def run(part):
    with np.load(ROOT / f'tune_out/seq_cache_{part}.npz', allow_pickle=True) as z:
        ids, x, n, b = (z[k] for k in ('ids', 'x', 'n', 'b'))
    p = pd.read_parquet(OUT / f'periodic_{part}.parquet')
    assert np.array_equal(p.index, ids)
    rows = []
    for i in range(len(ids)):
        if p.gate.iat[i] > 0.025 and 8 <= p.period.iat[i] <= 49:
            rows.append(one(x[i, :n[i]].astype(float), b[i], p.period.iat[i]))
        else:
            rows.append({})
    frame = pd.DataFrame(rows, index=p.index).fillna(0)
    assert np.isfinite(frame).all().all()
    frame.to_parquet(OUT / f'phase_law_{part}.parquet')
    print(part, frame.shape, flush=True)
if __name__ == '__main__':
    with threadpool_limits(limits=1):
        run('train')
        run('test')
