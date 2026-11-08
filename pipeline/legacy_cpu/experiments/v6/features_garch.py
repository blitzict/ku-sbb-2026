# CPU port of experiments/v6/features_garch.py, source sha256 4cff681dd99d453e105d1b0bb5d7ff2cf5077f75b068dbbba00ab9d744d0e1ac. See auc_lab_20261107/claude/audit/cpu_port_notes.md.
import sys as _s, pathlib as _p; _s.path.insert(0, str(_p.Path(__file__).resolve().parents[2])); import ku_workspace as KU_ENV
"""GARCH block (tree-side idea 3), Intel Arc.  On the AR(p, BIC<=16)-whitened residuals of each series:
  GARCH(1,1) fitted on the BEFORE part by grid MLE (omega = 1 - a - b so the unconditional variance is 1; a in {0,.05,.1,.2,.3}, b in {0,.5,.7,.8,.9,.95});
  the filter is run forward through the whole series with the before parameters -> filtered stream e_t = r_t / sigma_t with an iid null;
  on e_t: level t, scale log-ratio z, two-sample KS (quantile-grid CDF distance), kurtosis gap, t-vs-Gaussian shape LR;
  exact LRs: unconditional-variance change with (a, b) shared (grid over log f), (a, b) change with variance shared; persistence before/after;
  filtered-vs-raw log variance ratios.  ~20 columns, prefix gh_.
Usage: .venv/Scripts/python features_garch.py train|test|aug -> tune_out/gh_<name>.parquet / aug_gh_features.parquet"""
import sys, time, pathlib
sys.stdout.reconfigure(encoding="utf-8"); HERE = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd, torch
import features_lr as FL
dev, L, PMAX, O = FL.dev, FL.L, FL.PMAX, FL.O
name = sys.argv[1]; t0 = time.time()
jobs, out = FL.build_jobs(name); out = out.with_name(out.name.replace("lr_", "gh_").replace("aug_lr_", "aug_gh_"))
AG = torch.tensor([0.0, 0.05, 0.1, 0.2, 0.3], device=dev); BG = torch.tensor([0.0, 0.5, 0.7, 0.8, 0.9, 0.95], device=dev)
grid = torch.tensor([(a_, b_) for a_ in AG.tolist() for b_ in BG.tolist() if a_ + b_ < 0.995], device=dev); G = len(grid)
LOGF = torch.linspace(-0.9, 0.9, 19, device=dev); DFS = FL.DFS


# asymmetric generalised normal: unnormalised f(u) = exp(-(|u|/s)^beta) with s = 1 left of 0 and s = e^q right of 0; standardised numerically to mean 0, var 1
_u = torch.linspace(-14, 14, 28001, device=dev); _du = float(_u[1] - _u[0])
AGN = torch.tensor([(q_, b_) for q_ in torch.linspace(-1.5, 1.5, 21).tolist() for b_ in FL.BETAS.tolist()], device=dev)
_AGN_PAR = []
for q_, b_ in AGN.tolist():
    sR = float(np.exp(q_)); fu = torch.exp(-(torch.where(_u < 0, _u.abs(), _u.abs() / sR)) ** b_); Zc = float((fu * _du).sum()); mu = float((_u * fu * _du).sum() / Zc); var = float(((_u - mu) ** 2 * fu * _du).sum() / Zc)
    _AGN_PAR.append((mu, float(np.sqrt(var)), float(np.log(Zc)), sR, b_))
_AGN_PAR = torch.tensor(_AGN_PAR, device=dev)


def agn_logpdf(z, q, beta):
    i = int(((AGN[:, 0] - q).abs() + (AGN[:, 1] - beta).abs()).argmin()); mu, sig, logZ, sR, b_ = _AGN_PAR[i].tolist()
    u = sig * z + mu; return float(np.log(sig)) - logZ - (torch.where(u < 0, u.abs(), u.abs() / sR)) ** b_


def whiten(js):
    """Before-fit AR(p by BIC) residuals, standardised by the before residual sd; masks for valid pre / post residual positions."""
    N = len(js); X = torch.zeros(N, L, device=dev); n = torch.tensor([min(len(v), L) for _, v, _ in js], device=dev); b = torch.tensor([k for _, _, k in js], device=dev)
    for i, (_, v, _) in enumerate(js): X[i, :min(len(v), L)] = torch.as_tensor(v[:L], device=dev, dtype=torch.float32)
    tt = torch.arange(L, device=dev)[None, :]; pre = (tt < b[:, None]).float(); post = ((tt >= b[:, None]) & (tt < n[:, None])).float(); valid = pre + post
    m = (X * pre).sum(1) / b; sd = torch.sqrt((((X - m[:, None]) * pre) ** 2).sum(1) / b).clamp(min=1e-6); Z = (X - m[:, None]) / sd[:, None] * valid
    best = None
    for p in range(PMAX + 1):
        if p == 0: c = (Z * pre).sum(1) / b; r = (Z - c[:, None]) * valid; ne = b.float()
        else:
            U = Z.unfold(1, p + 1, 1); yv = U[:, :, -1]; A = torch.cat([U[:, :, :-1].flip(-1), torch.ones_like(yv)[..., None]], -1); mk = pre[:, p:][..., None]; Am = A * mk
            beta = torch.linalg.solve(Am.transpose(1, 2) @ A + 1e-4 * torch.eye(p + 1, device=dev), Am.transpose(1, 2) @ yv[..., None]).squeeze(-1)
            r = torch.zeros(N, L, device=dev); r[:, p:] = (yv - (A @ beta[..., None]).squeeze(-1)) * valid[:, p:]; ne = (b - p).float()
        s2 = ((r * pre) ** 2).sum(1) / ne.clamp(min=1); bic = ne * torch.log(s2.clamp(min=1e-9)) + (p + 1) * torch.log(ne.clamp(min=2))
        if best is None: best = (bic, r, torch.full((N,), p, device=dev))
        else:
            better = bic < best[0]; best = (torch.where(better, bic, best[0]), torch.where(better[:, None], r, best[1]), torch.where(better, torch.full_like(best[2], p), best[2]))
    r, pb = best[1], best[2]; vpre = pre.clone(); vpost = post.clone()
    for i in range(N): vpre[i, :int(pb[i])] = 0
    na, nb = vpre.sum(1), vpost.sum(1); sa = torch.sqrt((r ** 2 * vpre).sum(1) / na).clamp(min=1e-6); return r / sa[:, None], vpre, vpost, na, nb, b


def garch_h(R2, a_, b_, start_h=1.0):
    """h_t = (1-a-b) + a r_{t-1}^2 + b h_{t-1}, batched over series (R2: (N, L)); a_, b_ broadcastable (N,) or scalars."""
    N = R2.shape[0]; h = torch.empty_like(R2); h[:, 0] = start_h; om = 1 - a_ - b_
    for t in range(1, R2.shape[1]): h[:, t] = om + a_ * R2[:, t - 1] + b_ * h[:, t - 1]
    return h.clamp(min=1e-4)


def ll_seg(R2, h, mask): return (-0.5 * (torch.log(h) + R2 / h) * mask).sum(1)


def block(js):
    Rs, vpre, vpost, na, nb, b = whiten(js); N = len(js); R2 = Rs ** 2; f = {}
    # grid MLE of (a, b) on the before part; store the before log-lik of every grid point for the shared-parameter tests
    LLa = torch.zeros(N, G, device=dev); LLb = torch.zeros(N, G, device=dev)
    vb = ((R2 * vpost).sum(1) / nb).clamp(min=1e-6); R2n = torch.where(vpost > 0, R2 / vb[:, None], R2)     # after part rescaled to unit variance so (a, b) is not confounded with a scale change
    for g in range(G):
        h = garch_h(R2, grid[g, 0], grid[g, 1]); LLa[:, g] = ll_seg(R2, h, vpre); hn = garch_h(R2n, grid[g, 0], grid[g, 1]); LLb[:, g] = ll_seg(R2n, hn, vpost)
    ga = LLa.argmax(1); gb = LLb.argmax(1); f["gh_a_pre"] = grid[ga, 0]; f["gh_b_pre"] = grid[ga, 1]; f["gh_persist_pre"] = grid[ga].sum(1); f["gh_persist_post"] = grid[gb].sum(1)
    f["gh_persist_gap"] = f["gh_persist_post"] - f["gh_persist_pre"]; f["gh_garch_vs_const_pre"] = (LLa.max(1).values - LLa[:, 0]) / na
    # (a, b) change with variance shared: after under its own best vs under the before best
    f["gh_ab_change_lr"] = LLb.max(1).values - LLb[torch.arange(N, device=dev), ga]
    # filtered stream with the before parameters run through the whole series
    hb = garch_h(R2, grid[ga, 0], grid[ga, 1]); E = Rs / torch.sqrt(hb); E2 = E ** 2
    # unconditional-variance change on the after part with (a, b) shared: scale r by exp(-logf) on the after part (variance f^2) -> ll over grid of f
    base = ll_seg(R2, hb, vpost); best_f = torch.full((N,), -1e9, device=dev); arg_f = torch.zeros(N, device=dev)
    for lf in LOGF:
        R2s = torch.where(vpost > 0, R2 * torch.exp(-2 * lf), R2); hs = garch_h(R2s, grid[ga, 0], grid[ga, 1]); ll = ll_seg(R2s, hs, vpost) - nb * lf
        better = ll > best_f; best_f = torch.where(better, ll, best_f); arg_f = torch.where(better, lf.expand(N), arg_f)
    f["gh_var_change_lr"] = best_f - base; f["gh_var_change_logf"] = arg_f
    # tests on the filtered stream
    ma, mb = (E * vpre).sum(1) / na, (E * vpost).sum(1) / nb; sa_ = torch.sqrt(((E - ma[:, None]) ** 2 * vpre).sum(1) / na).clamp(min=1e-6); sb_ = torch.sqrt(((E - mb[:, None]) ** 2 * vpost).sum(1) / nb).clamp(min=1e-6)
    f["gh_e_level_t"] = (mb - ma) / (sa_ * torch.sqrt(1 / na + 1 / nb)); f["gh_e_level_abs_t"] = f["gh_e_level_t"].abs()
    f["gh_e_logsd_ratio"] = torch.log(sb_ / sa_); f["gh_e_scale_z"] = f["gh_e_logsd_ratio"].abs() * torch.sqrt(nb / 2)
    ka = (((E - ma[:, None]) / sa_[:, None]) ** 4 * vpre).sum(1) / na - 3; kb = (((E - mb[:, None]) / sb_[:, None]) ** 4 * vpost).sum(1) / nb - 3; f["gh_e_kurt_gap"] = kb - ka; f["gh_e_kurt_agap"] = (kb - ka).abs()
    # two-sample KS on E via the before quantile grid (33 points)
    qs = torch.linspace(0.02, 0.98, 33, device=dev); Ea = E.masked_fill(vpre == 0, float("nan")); Q = torch.nanquantile(Ea, qs, dim=1).T          # (N, 33)
    Fa = torch.stack([((E <= Q[:, j:j+1]).float() * vpre).sum(1) / na for j in range(33)], 1); Fb = torch.stack([((E <= Q[:, j:j+1]).float() * vpost).sum(1) / nb for j in range(33)], 1)
    f["gh_e_ks"] = (Fa - Fb).abs().max(1).values * torch.sqrt(na * nb / (na + nb)); f["gh_e_cvm"] = ((Fa - Fb) ** 2).mean(1) * (na * nb / (na + nb))
    # shape LR on E: after under own best t df vs before best df (both rescaled)
    Eb = (E - mb[:, None]) / sb_[:, None]; Ea2 = (E - ma[:, None]) / sa_[:, None]
    lla = torch.stack([(FL.t_logpdf(Ea2, df) * vpre).sum(1) for df in DFS], 1); llb = torch.stack([(FL.t_logpdf(Eb, df) * vpost).sum(1) for df in DFS], 1)
    f["gh_e_shape_lr"] = llb.max(1).values - llb[torch.arange(N, device=dev), lla.argmax(1)]
    # the other shape families on the filtered stream (GN, bimodal, two-piece), symmetric LR (after under own best vs before best, plus reverse)
    ar_ = torch.arange(N, device=dev)
    for nm, fn, gridv in (("gn", FL.gn_logpdf, FL.BETAS), ("bim", FL.bimodal_logpdf, FL.DSEP), ("skew", FL.twopiece_logpdf, FL.QS)):
        la = torch.stack([(fn(Ea2, g_) * vpre).sum(1) for g_ in gridv], 1); lb = torch.stack([(fn(Eb, g_) * vpost).sum(1) for g_ in gridv], 1); ia, ib = la.argmax(1), lb.argmax(1)
        f[f"gh_e_{nm}_shape_lr_sym"] = (lb.max(1).values - lb[ar_, ia]) + (la.max(1).values - la[ar_, ib]); f[f"gh_e_{nm}_gap"] = gridv[ib] - gridv[ia]
    # asymmetric generalised normal: two-piece scales with a GN tail exponent; before fit over (q, beta); after LR for a beta change with q held, and the reverse
    la = torch.stack([(agn_logpdf(Ea2, qb_[0], qb_[1]) * vpre).sum(1) for qb_ in AGN], 1); lb = torch.stack([(agn_logpdf(Eb, qb_[0], qb_[1]) * vpost).sum(1) for qb_ in AGN], 1)
    ia = la.argmax(1); qa_ = AGN[ia, 0]; same_q = (AGN[:, 0][None, :] == qa_[:, None])                 # (N, 210) mask of grid points sharing the before q
    lb_q = lb.masked_fill(~same_q, -1e9); f["gh_agn_beta_change_lr"] = lb_q.max(1).values - lb[ar_, ia]
    f["gh_agn_shape_lr_sym"] = (lb.max(1).values - lb[ar_, ia]) + (la.max(1).values - la[ar_, lb.argmax(1)]); f["gh_agn_q_pre"] = qa_; f["gh_agn_beta_pre"] = AGN[ia, 1]
    # filtered vs raw variance ratio (how much of the raw scale change the filter absorbed)
    raw_ratio = torch.log(torch.sqrt((R2 * vpost).sum(1) / nb) / torch.sqrt((R2 * vpre).sum(1) / na).clamp(min=1e-6)); f["gh_raw_logsd_ratio"] = raw_ratio; f["gh_absorbed"] = raw_ratio - f["gh_e_logsd_ratio"]
    return pd.DataFrame({k: v.float().cpu().numpy() for k, v in f.items()}, index=[j[0] for j in js])


parts = []
CH = int(__import__('os').environ.get('CHUNK', '2000'))
for i in range(0, len(jobs), CH):
    parts.append(block(jobs[i:i + CH]))
    if (i // 2000) % 4 == 3: print(f"  {i + 2000} done ({time.time()-t0:.0f}s)", flush=True)
F = pd.concat(parts).replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32); F.index.name = "id"; F.to_parquet(out)
print(f"{name}: {F.shape} -> {out.name} in {time.time()-t0:.0f}s", flush=True)
