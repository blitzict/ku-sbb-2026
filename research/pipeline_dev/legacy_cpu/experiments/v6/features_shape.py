# CPU port of experiments/v6/features_shape.py, source sha256 fe68f50021ae44c2c8b3ddb0c3c4eae5db01aeb17a1b930649063477eb4b0300. See auc_lab_20261107/claude/audit/cpu_port_notes.md.
import sys as _s, pathlib as _p; _s.path.insert(0, str(_p.Path(__file__).resolve().parents[2])); import ku_workspace as KU_ENV
"""Shape / frequency / ordinal block (research survey 2026-11-02), batched on the Intel Arc (torch xpu). Prefix sh_.
Two streams per series: z = series standardised by the before mean/sd; r = AR(12) innovations fitted on the before part (whitened).
Per stream, before vs after (each side self-standardised where the feature is about shape):
  Hermite moments He3..He8 (diff and z-stat, chi2 of He3..He6), L-moment ratios tau3/tau4 and log L-scale ratio,
  empirical characteristic-function distance at t = 0.5..3 (Epps-Pulley style), undecimated Haar energy per scale (log ratio and share diff),
  ACF lags 1..4 diff and Blackman-Tukey cepstrum c1..c6 diff, permutation entropy order 3/4 (before, after, diff, Jensen-Shannon of patterns).
Usage: .venv/Scripts/python features_shape.py train|test|aug|var:<stem>  -> tune_out/sh_<name>.parquet, aug_sh_features.parquet, var/sh_<stem>.parquet"""
import sys, os, time, math, pathlib, warnings, ctypes; warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
try: ctypes.windll.kernel32.SetPriorityClass(ctypes.c_void_p(-1), 0x4000)
except Exception: pass
import numpy as np, pandas as pd, torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
name = sys.argv[1] if len(sys.argv) > 1 else "train"; sys.argv = sys.argv[:2]
from features_lr import build_jobs, O
dev = "cpu"; torch.set_num_threads(1); t0 = time.time()
L, P = 1024, 12; TS = [0.5, 1.0, 1.5, 2.0, 3.0]; HS = [1, 2, 4, 8, 16, 32]; NL = 16; NF = 16


def hermite(u):
    u2 = u * u
    return [u * (u2 - 3), u2 * u2 - 6 * u2 + 3, u * (u2 * u2 - 10 * u2 + 15), u2 ** 3 - 15 * u2 * u2 + 45 * u2 - 15,
            u * (u2 ** 3 - 21 * u2 * u2 + 105 * u2 - 105), u2 ** 4 - 28 * u2 ** 3 + 210 * u2 * u2 - 420 * u2 + 105]


def stream_feats(S, ma, mb, pre_):
    """S (N,L) stream; ma/mb float masks of the before/after side (contiguous). Returns dict of (N,) tensors."""
    f = {}; na, nb = ma.sum(1).clamp(min=2), mb.sum(1).clamp(min=2)

    def side_std(m, n):
        mu = (S * m).sum(1) / n; sd = torch.sqrt((((S - mu[:, None]) * m) ** 2).sum(1) / n).clamp(min=1e-6)
        return ((S - mu[:, None]) / sd[:, None]).clamp(-6, 6) * m
    ua, ub = side_std(ma, na), side_std(mb, nb)
    # Hermite moments
    ha, hb = hermite(ua), hermite(ub); chi = 0
    for k, (a, b) in enumerate(zip(ha, hb), start=3):
        d = (b * mb).sum(1) / nb - (a * ma).sum(1) / na; zst = d / math.sqrt(math.factorial(k)) / torch.sqrt(1 / na + 1 / nb)
        f[f"{pre_}he{k}_d"] = d; f[f"{pre_}he{k}_z"] = zst
        if k <= 6: chi = chi + zst ** 2
    f[f"{pre_}he_chi"] = chi

    # L-moments via sorted order statistics
    def lmom(u, m, n):
        v = torch.where(m > 0, u, torch.full_like(u, 1e9)); xs, _ = torch.sort(v, 1); i = torch.arange(1, L + 1, device=dev, dtype=torch.float32)[None, :]
        ok = (i <= n[:, None]).float(); xs = xs * ok; nn = n[:, None]
        b0 = xs.sum(1) / n; b1 = (xs * (i - 1) / (nn - 1)).sum(1) / n; b2 = (xs * (i - 1) * (i - 2) / ((nn - 1) * (nn - 2))).sum(1) / n
        b3 = (xs * (i - 1) * (i - 2) * (i - 3) / ((nn - 1) * (nn - 2) * (nn - 3))).sum(1) / n
        l2 = (2 * b1 - b0).clamp(min=1e-6); l3 = 6 * b2 - 6 * b1 + b0; l4 = 20 * b3 - 30 * b2 + 12 * b1 - b0; return l2, l3 / l2, l4 / l2
    _, t3a, t4a = lmom(ua, ma, na); _, t3b, t4b = lmom(ub, mb, nb)
    f[f"{pre_}lt3_d"] = t3b - t3a; f[f"{pre_}lt4_d"] = t4b - t4a; f[f"{pre_}lt4_a"] = t4b
    la2, _, _ = lmom(S * ma, ma, na); lb2, _, _ = lmom(S * mb, mb, nb); f[f"{pre_}l2_lr"] = torch.log(lb2 / la2)
    # empirical characteristic function distance
    tot = 0
    for t in TS:
        ca, sa = (torch.cos(t * ua) * ma).sum(1) / na, (torch.sin(t * ua) * ma).sum(1) / na
        cb, sb = (torch.cos(t * ub) * mb).sum(1) / nb, (torch.sin(t * ub) * mb).sum(1) / nb
        d2 = ((ca - cb) ** 2 + (sa - sb) ** 2) / (1 / na + 1 / nb); f[f"{pre_}ecf{t}"] = d2; tot = tot + d2 * math.exp(-t * t / 2)
    f[f"{pre_}ecf_w"] = tot
    # undecimated Haar energies (stream in before-sd units, so scale changes show)
    C = torch.cat([torch.zeros(S.shape[0], 1, device=dev), torch.cumsum(S * (ma + mb), 1)], 1); Ea, Eb = [], []
    for h in HS:
        t = torch.arange(2 * h - 1, L, device=dev); first = C[:, t - h + 1] - C[:, t - 2 * h + 1]; second = C[:, t + 1] - C[:, t - h + 1]
        d2 = ((first - second) / (h * math.sqrt(2))) ** 2
        wa = ma[:, t] * ma[:, t - 2 * h + 1]; wb = mb[:, t] * mb[:, t - 2 * h + 1]
        Ea.append((d2 * wa).sum(1) / wa.sum(1).clamp(min=1)); Eb.append((d2 * wb).sum(1) / wb.sum(1).clamp(min=1))
    Ea, Eb = torch.stack(Ea, 1).clamp(min=1e-8), torch.stack(Eb, 1).clamp(min=1e-8); sa_, sb_ = Ea / Ea.sum(1, keepdim=True), Eb / Eb.sum(1, keepdim=True)
    for j, h in enumerate(HS): f[f"{pre_}haar{h}_lr"] = torch.log(Eb[:, j] / Ea[:, j]); f[f"{pre_}haar{h}_sd"] = sb_[:, j] - sa_[:, j]
    f[f"{pre_}haar_tv"] = (sb_ - sa_).abs().sum(1)

    # ACF + Blackman-Tukey cepstrum on self-standardised sides
    def acf(u, m):
        r = [torch.ones(u.shape[0], device=dev)]
        for k in range(1, NL + 1):
            w = m[:, k:] * m[:, :-k]; r.append((u[:, k:] * u[:, :-k] * w).sum(1) / w.sum(1).clamp(min=1))
        return torch.stack(r, 1)
    Ra, Rb = acf(ua, ma), acf(ub, mb)
    for k in range(1, 5): f[f"{pre_}acf{k}_d"] = Rb[:, k] - Ra[:, k]
    lag = torch.arange(NL + 1, device=dev, dtype=torch.float32); win = 1 - lag / (NL + 1); fr = torch.linspace(0, math.pi, NF, device=dev)
    cosm = torch.cos(fr[:, None] * lag[None, :]); cosm[:, 1:] *= 2
    cosc = torch.cos(fr[:, None] * torch.arange(1, 7, device=dev, dtype=torch.float32)[None, :])
    ceps = lambda R: torch.log(((R * win) @ cosm.T).clamp(min=1e-3)) @ cosc / NF
    ca_, cb_ = ceps(Ra), ceps(Rb)
    for k in range(6): f[f"{pre_}cep{k+1}_d"] = cb_[:, k] - ca_[:, k]
    # permutation entropy
    for d in (3, 4):
        W = S.unfold(1, d, 1); rk = W.argsort(-1).argsort(-1); code = (rk * (d ** torch.arange(d, device=dev))).sum(-1); T = W.shape[1]; hs = []
        for m in (ma, mb):
            w = m[:, :T] * m[:, d - 1:d - 1 + T]; H = torch.zeros(S.shape[0], d ** d, device=dev).scatter_add_(1, code, w); hs.append(H / H.sum(1, keepdim=True).clamp(min=1))
        lf = math.log(math.factorial(d)); ent = lambda p: -(p * torch.log(p.clamp(min=1e-12))).sum(1) / lf
        pa, pb = hs; ea, eb = ent(pa), ent(pb)
        f[f"{pre_}pe{d}_a"] = ea; f[f"{pre_}pe{d}_b"] = eb; f[f"{pre_}pe{d}_d"] = eb - ea
        f[f"{pre_}pe{d}_js"] = (ent((pa + pb) / 2) - (ea + eb) / 2) * lf * (na * nb / (na + nb))
    return f


def block(js):
    N = len(js); X = torch.zeros(N, L, device=dev); n = torch.tensor([min(len(v), L) for _, v, _ in js], device=dev); b = torch.tensor([k for _, _, k in js], device=dev)
    for i, (_, v, _) in enumerate(js): X[i, :min(len(v), L)] = torch.as_tensor(v[:L], device=dev, dtype=torch.float32)
    tt = torch.arange(L, device=dev)[None, :]; pre = (tt < b[:, None]).float(); post = ((tt >= b[:, None]) & (tt < n[:, None])).float(); valid = pre + post
    m = (X * pre).sum(1) / b; sd = torch.sqrt((((X - m[:, None]) * pre) ** 2).sum(1) / b).clamp(min=1e-6); Z = (X - m[:, None]) / sd[:, None] * valid
    U = Z.unfold(1, P + 1, 1); yv = U[:, :, -1]; A = torch.cat([U[:, :, :-1].flip(-1), torch.ones_like(yv)[..., None]], -1)
    mk = pre[:, P:][..., None]; G = (A * mk).transpose(1, 2) @ A + 1e-3 * torch.eye(P + 1, device=dev); h = (A * mk).transpose(1, 2) @ yv[..., None]
    beta = torch.linalg.solve(G, h).squeeze(-1); R = torch.zeros(N, L, device=dev); R[:, P:] = (yv - (A @ beta[..., None]).squeeze(-1)) * valid[:, P:]
    rpre = pre * (tt >= P).float(); rs = torch.sqrt(((R * rpre) ** 2).sum(1) / rpre.sum(1).clamp(min=1)).clamp(min=1e-6); R = R / rs[:, None]
    f = stream_feats(Z, pre, post, "sh_z_"); f.update(stream_feats(R, rpre, post, "sh_r_"))
    return pd.DataFrame({k: v.float().cpu().numpy() for k, v in f.items()}, index=[j[0] for j in js])


if __name__ == "__main__":
    jobs, _ = build_jobs(name)
    out = O / "aug_sh_features.parquet" if name == "aug" else (O / "var" / f"sh_{name[4:]}.parquet" if name.startswith("var:") else O / f"sh_{name}.parquet")
    CH = int(os.environ.get("CHUNK", "2000")); parts = []; done = 0
    for i in range(0, len(jobs), CH):
        parts.append(block(jobs[i:i + CH])); done += len(parts[-1]); print(f"  {done} done ({time.time()-t0:.0f}s)", flush=True)
    F = pd.concat(parts).replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32); F.index.name = "id"; F.to_parquet(out)
    print(f"{name}: {F.shape} -> {out.name} in {time.time()-t0:.0f}s", flush=True)
