# CPU port of experiments/v6/features_clip.py, source sha256 9c2dcf4d4bb5887a98b7b41d844ef20bc1b27c747f256e2d2661b8b3d91aca8d. See auc_lab_20261107/claude/audit/cpu_port_notes.md.
import sys as _s, pathlib as _p; _s.path.insert(0, str(_p.Path(__file__).resolve().parents[2])); import ku_workspace as KU_ENV
"""Before-quantile cleaning block (survey 2026-11-02: gsoisson ADIA repo winsorises both sides at the BEFORE 0.2/99.8% quantiles), Intel Arc. Prefix cl_.
Streams: z (before mean/sd standardised) and r (AR(12) innovations fitted on the before part). Per stream, with thresholds = before-side quantiles:
  clipped-sd log ratio and clipped mean shift for clip bands 0.2/99.8, 1/99, 5/95 %;
  after exceedance rates above / below each before quantile (Jeffreys logit minus nominal, binomial z), two-sided tail rates of |x|;
  crossing rates of the before q25/q50/q75 (logit diff), Fano factor of |x| > before q95(|x|) in windows of 50 (log ratio);
  robust scale: log ratios of MAD, IQR, IDR (q90-q10); median shift in before-MAD units; quantile shifts at 9 levels in before-IQR units.
Usage: .venv/Scripts/python features_clip.py train|test|aug|var:<stem>"""
import sys, os, time, math, pathlib, warnings, ctypes; warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
try: ctypes.windll.kernel32.SetPriorityClass(ctypes.c_void_p(-1), 0x4000)
except Exception: pass
import numpy as np, pandas as pd, torch
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
name = sys.argv[1] if len(sys.argv) > 1 else "train"; sys.argv = sys.argv[:2]
from features_lr import build_jobs, O
dev = "cpu"; torch.set_num_threads(1); t0 = time.time()
L, P = 1024, 12; BIG = 1e9
CLIPS = [(0.002, 0.998), (0.01, 0.99), (0.05, 0.95)]
LEVELS = [0.002, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.998]
FINE = [0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95]


def mquant(S, m, n, qs):
    """Masked per-row quantiles (linear interpolation). S,m (N,L); n (N,) counts; qs list -> (N, len(qs))."""
    xs, _ = torch.sort(torch.where(m > 0, S, torch.full_like(S, BIG)), 1); out = []
    for q in qs:
        pos = q * (n - 1); lo = pos.floor().long().clamp(min=0); hi = (lo + 1).clamp(max=L - 1); fr = pos - lo.float()
        hi = torch.minimum(hi, (n - 1).long()); a = xs.gather(1, lo[:, None])[:, 0]; b = xs.gather(1, hi[:, None])[:, 0]; out.append(a + fr * (b - a))
    return torch.stack(out, 1)


def jl(k, m): return torch.log((k + 0.5) / (m - k + 0.5))


def stream_feats(S, ma, mb, pre_):
    f = {}; na, nb = ma.sum(1).clamp(min=2), mb.sum(1).clamp(min=2)
    mean = lambda x, m, n: (x * m).sum(1) / n
    sdv = lambda x, m, n: torch.sqrt((((x - mean(x, m, n)[:, None]) * m) ** 2).sum(1) / n).clamp(min=1e-6)
    # clipped sd / mean with before-quantile bands
    for lo_q, hi_q in CLIPS:
        q = mquant(S, ma, na, [lo_q, hi_q]); C = torch.minimum(torch.maximum(S, q[:, :1]), q[:, 1:])
        tag = f"{int(round(lo_q * 1000))}"
        f[f"{pre_}csd{tag}_lr"] = torch.log(sdv(C, mb, nb) / sdv(C, ma, na)); f[f"{pre_}cmu{tag}_d"] = (mean(C, mb, nb) - mean(C, ma, na)) / sdv(C, ma, na)
    # exceedance rates against before quantiles
    Q = mquant(S, ma, na, LEVELS)
    for j, lv in enumerate(LEVELS):
        above = lv >= 0.5; k = ((S > Q[:, j:j + 1]).float() * mb).sum(1) if above else ((S < Q[:, j:j + 1]).float() * mb).sum(1)
        nom = 1 - lv if above else lv; tag = f"{'hi' if above else 'lo'}{int(round(lv * 1000))}"
        f[f"{pre_}ex_{tag}_jl"] = jl(k, nb) - math.log(nom / (1 - nom)); f[f"{pre_}ex_{tag}_z"] = (k - nb * nom) / torch.sqrt(nb * nom * (1 - nom))
    A = S.abs(); QA = mquant(A, ma, na, [0.5, 0.9, 0.95, 0.99])
    for j, lv in enumerate([0.5, 0.9, 0.95, 0.99]):
        k = ((A > QA[:, j:j + 1]).float() * mb).sum(1); nom = 1 - lv; f[f"{pre_}abs{int(lv * 100)}_z"] = (k - nb * nom) / torch.sqrt(nb * nom * (1 - nom))
    # crossing rates of before q25/q50/q75
    for j, lv in ((4, 25), (5, 50), (6, 75)):
        s = (S > Q[:, j:j + 1]).float(); cr = (s[:, 1:] != s[:, :-1]).float()
        wa = ma[:, 1:] * ma[:, :-1]; wb = mb[:, 1:] * mb[:, :-1]
        f[f"{pre_}cross{lv}_d"] = jl((cr * wb).sum(1), wb.sum(1).clamp(min=1)) - jl((cr * wa).sum(1), wa.sum(1).clamp(min=1))
    # Fano factor of tail exceedances (volatility clustering), windows of 50 aligned to each side
    e = (A > QA[:, 2:3]).float(); W = 50; fano = []
    for m in (ma, mb):
        c = torch.cat([torch.zeros(S.shape[0], 1, device=dev), torch.cumsum(e * m, 1)], 1); cm = torch.cat([torch.zeros(S.shape[0], 1, device=dev), torch.cumsum(m, 1)], 1)
        t = torch.arange(W, L + 1, device=dev); cnt = c[:, t] - c[:, t - W]; full = ((cm[:, t] - cm[:, t - W]) == W).float()
        nw = full.sum(1).clamp(min=1); mu = (cnt * full).sum(1) / nw; var = (((cnt - mu[:, None]) * full) ** 2).sum(1) / nw
        fano.append((var + 0.05) / (mu + 0.05))
    f[f"{pre_}fano_lr"] = torch.log(fano[1] / fano[0])
    # robust scale and location
    qa = mquant(S, ma, na, FINE); qb = mquant(S, mb, nb, FINE)
    mada = mquant((S - qa[:, 4:5]).abs(), ma, na, [0.5])[:, 0].clamp(min=1e-6); madb = mquant((S - qb[:, 4:5]).abs(), mb, nb, [0.5])[:, 0].clamp(min=1e-6)
    iqa, iqb = (qa[:, 6] - qa[:, 2]).clamp(min=1e-6), (qb[:, 6] - qb[:, 2]).clamp(min=1e-6); ida, idb = (qa[:, 7] - qa[:, 1]).clamp(min=1e-6), (qb[:, 7] - qb[:, 1]).clamp(min=1e-6)
    f[f"{pre_}mad_lr"] = torch.log(madb / mada); f[f"{pre_}iqr_lr"] = torch.log(iqb / iqa); f[f"{pre_}idr_lr"] = torch.log(idb / ida)
    f[f"{pre_}tailw_d"] = torch.log(idb / iqb) - torch.log(ida / iqa); f[f"{pre_}med_d"] = (qb[:, 4] - qa[:, 4]) / mada
    for j, lv in enumerate(FINE): f[f"{pre_}q{int(lv * 100)}_d"] = (qb[:, j] - qa[:, j]) / iqa
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
    f = stream_feats(Z, pre, post, "cl_z_"); f.update(stream_feats(R, rpre, post, "cl_r_"))
    return pd.DataFrame({k: v.float().cpu().numpy() for k, v in f.items()}, index=[j[0] for j in js])


if __name__ == "__main__":
    jobs, _ = build_jobs(name)
    out = O / "aug_cl_features.parquet" if name == "aug" else (O / "var" / f"cl_{name[4:]}.parquet" if name.startswith("var:") else O / f"cl_{name}.parquet")
    CH = int(os.environ.get("CHUNK", "2000")); parts = []
    for i in range(0, len(jobs), CH): parts.append(block(jobs[i:i + CH]))
    F = pd.concat(parts).replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32); F.index.name = "id"; F.to_parquet(out)
    print(f"{name}: {F.shape} -> {out.name} in {time.time()-t0:.0f}s", flush=True)
