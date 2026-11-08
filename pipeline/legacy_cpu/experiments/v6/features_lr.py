# CPU port of experiments/v6/features_lr.py, source sha256 75da77857a41cf2d9db87bf784bf4b2d96af90585620e5fae1b45e8014d8b245. See auc_lab_20261107/claude/audit/cpu_port_notes.md.
import sys as _s, pathlib as _p; _s.path.insert(0, str(_p.Path(__file__).resolve().parents[2])); import ku_workspace as KU_ENV
"""Exact likelihood-ratio block (docs/GENERATOR.md section 4.2), batched on the Intel Arc.
For every series: AR(p) fitted on the before part (p by BIC, 0..8) gives whitened residuals r_t over the whole series and the before
innovation sd.  Under the pinned generator each break is ONE of: level shift (residual mean shifts by delta*(1-sum phi)), scale change
(residual sd times f), AR change, innovation-shape change.  We compute, under Gaussian and under the before-fitted Student-t law:
  lr_level_t      t-statistic of the after residual mean;  lr_level_lbf  log Bayes factor with delta ~ U(-2.5, 2.5) before-sd
  lr_scale_lr     log-likelihood ratio of the MLE scale factor; lr_scale_lbf log Bayes factor with log f ~ U(-0.9, 0.9)
  lr_ar_lr        LR of refitting AR(p) on the after part vs keeping the before coefficients (Chow-type), BIC-penalised
  lr_shape_lr     LR of the after residuals under their own best t-df vs the before-fitted df (Gaussian vs t family switch)
  lr_mix_lbf      log of the prior-weighted mixture of the four Bayes factors (0.33 level, 0.33 scale, 0.15 shape, 0.08 AR, 0.11 none)
plus the same four with the sides swapped (after-fit, before scored) and the BIC order, persistence, df.  ~30 columns, prefix lr_.
Usage: .venv/Scripts/python features_lr.py train|test|aug  -> tune_out/lr_<name>.parquet / aug_lr_features.parquet"""
import sys, os, time, pathlib, warnings, ctypes; warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
try: ctypes.windll.kernel32.SetPriorityClass(ctypes.c_void_p(-1), 0x4000)
except Exception: pass
import numpy as np, pandas as pd, torch
D = KU_ENV.root(); O = D / "tune_out"
dev = "cpu"; torch.set_num_threads(1); t0 = time.time()
L, PMAX = 1024, 16; name = sys.argv[1] if len(sys.argv) > 1 else "train"; PREFIX = sys.argv[2] if len(sys.argv) > 2 else "lr"
DFS = torch.tensor([2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 25.0, 50.0, 1e4], device=dev)
DELTA = torch.linspace(-2.5, 2.5, 51, device=dev); LOGF = torch.linspace(-0.9, 0.9, 37, device=dev)

# ---------------------------------------------------------------- functions
BETAS = torch.tensor([0.8, 1.0, 1.3, 1.6, 2.0, 2.5, 3.0, 4.0, 6.0, 10.0], device=dev)


def gn_logpdf(z, beta):
    """log density of a unit-variance generalised normal with shape beta (2 = Gaussian, <2 heavy, >2 light/uniform-like)."""
    alpha = torch.sqrt(torch.exp(torch.lgamma(1 / beta) - torch.lgamma(3 / beta)))
    return torch.log(beta) - torch.log(2 * alpha) - torch.lgamma(1 / beta) - (z.abs() / alpha) ** beta


DSEP = torch.linspace(0.0, 0.97, 25, device=dev)


def bimodal_logpdf(z, d):
    """log density of a unit-variance symmetric two-Gaussian mixture: components at +-d with sd sqrt(1-d^2) (d=0 -> Gaussian)."""
    s2 = (1 - d ** 2).clamp(min=1e-3); c = -0.5 * torch.log(2 * np.pi * s2) + np.log(0.5)
    return c + torch.logaddexp(-(z - d) ** 2 / (2 * s2), -(z + d) ** 2 / (2 * s2))


QS = torch.linspace(-1.5, 1.5, 21, device=dev)


def twopiece_logpdf(z, q):
    """log density of a zero-mean, unit-variance two-piece normal with asymmetry q = log(s_R / s_L): N(m, s_L) left of m, N(m, s_R) right."""
    r = torch.exp(q); sl = 1.0; sr = r                                    # shape first, then standardise
    mu_raw = np.sqrt(2 / np.pi) * (sr - sl); var_raw = (sr ** 2 + sl ** 2) / 2 + (sr - sl) ** 2 * (1 - 2 / np.pi) / 1.0 - 0.0
    var_raw = (1 - 2 / np.pi) * (sr - sl) ** 2 + sr * sl                 # exact variance of the two-piece normal with sd_L = sl, sd_R = sr
    k = torch.sqrt(var_raw); m = -mu_raw / k; sL, sR = sl / k, sr / k      # standardised: mode at m, mean 0, variance 1
    c = np.log(2.0) - 0.5 * np.log(2 * np.pi) - torch.log(sL + sR)
    u = z - m; return c - 0.5 * torch.where(u < 0, (u / sL) ** 2, (u / sR) ** 2)


def t_logpdf(z, df):
    """log density of a unit-variance Student-t (df>2) at z; df=1e4 ~ Gaussian."""
    sc = torch.sqrt((df - 2) / df); u = z / sc
    return KU_ENV.t_constant(df, z) - torch.log(sc) - (df + 1) / 2 * torch.log1p(u ** 2 / df)


def block(js):
    N = len(js); X = torch.zeros(N, L, device=dev); n = torch.tensor([min(len(v), L) for _, v, _ in js], device=dev); b = torch.tensor([k for _, _, k in js], device=dev)
    for i, (_, v, _) in enumerate(js): X[i, :min(len(v), L)] = torch.as_tensor(v[:L], device=dev, dtype=torch.float32)
    tt = torch.arange(L, device=dev)[None, :]; pre = (tt < b[:, None]).float(); post = ((tt >= b[:, None]) & (tt < n[:, None])).float(); valid = pre + post
    m = (X * pre).sum(1) / b; sd = torch.sqrt((((X - m[:, None]) * pre) ** 2).sum(1) / b).clamp(min=1e-6); Z = (X - m[:, None]) / sd[:, None] * valid
    f = {}

    def fit(p, mask):
        """AR(p)+intercept by masked least squares; returns beta (N,p+1), residuals over the whole series (N,L; zero for t<p), n_eff of mask."""
        if p == 0:
            c = (Z * mask).sum(1) / mask.sum(1).clamp(min=1); r = (Z - c[:, None]) * valid; return c[:, None], r, mask.sum(1)
        U = Z.unfold(1, p + 1, 1); yv = U[:, :, -1]; A = torch.cat([U[:, :, :-1].flip(-1), torch.ones_like(yv)[..., None]], -1)
        mk = mask[:, p:][..., None]; Am = A * mk; G = Am.transpose(1, 2) @ A + 1e-4 * torch.eye(p + 1, device=dev); h = Am.transpose(1, 2) @ yv[..., None]
        beta = torch.linalg.solve(G, h).squeeze(-1); res = torch.zeros(N, L, device=dev); res[:, p:] = (yv - (A @ beta[..., None]).squeeze(-1)) * valid[:, p:]
        return beta, res, mask[:, p:].sum(1)

    # BIC order on the before part
    fits = {}; bics = []
    for p in range(PMAX + 1):
        beta, r, ne = fit(p, pre); s2 = ((r * pre) ** 2).sum(1) / ne.clamp(min=1); bics.append(ne * torch.log(s2.clamp(min=1e-9)) + (p + 1) * torch.log(ne.clamp(min=2))); fits[p] = (beta, r, s2)
    pb = torch.stack(bics, 1).argmin(1); f["lr_p_bic"] = pb.float()
    R = torch.stack([fits[p][1] for p in range(PMAX + 1)], 1)[torch.arange(N, device=dev), pb]              # residuals under the before fit
    persist = torch.stack([fits[p][0][:, :p].sum(1) if p > 0 else torch.zeros(N, device=dev) for p in range(PMAX + 1)], 1)[torch.arange(N, device=dev), pb]
    f["lr_persist"] = persist
    vpre = pre.clone(); vpost = post.clone()
    for i in range(N): vpre[i, :int(pb[i])] = 0                                                           # first p residuals undefined
    na, nb = vpre.sum(1), vpost.sum(1); sa = torch.sqrt((R ** 2 * vpre).sum(1) / na).clamp(min=1e-6); Rs = R / sa[:, None]
    # innovation law on the before part: best t df
    lls = torch.stack([(t_logpdf(Rs, df) * vpre).sum(1) / na for df in DFS], 1); dfa = DFS[lls.argmax(1)]; f["lr_df_pre"] = dfa
    gauss_pre = (t_logpdf(Rs, DFS[-1]) * vpre).sum(1) / na; f["lr_tvsg_pre"] = lls.max(1).values - gauss_pre
    for law, df_use in (("g", torch.full((N,), 1e4, device=dev)), ("t", dfa)):
        lp = lambda z: t_logpdf(z, df_use[:, None])
        base = (lp(Rs) * vpost).sum(1)                                                                   # after residuals under the before law
        # level: residual shift c = delta*(1-persist) in before-sd units
        c = DELTA[None, :] * (1 - persist)[:, None] / sa[:, None]                                       # (N, G) in residual-sd units
        ll = torch.stack([(lp(Rs - c[:, g:g+1]) * vpost).sum(1) for g in range(len(DELTA))], 1) - base[:, None]
        f[f"lr_level_lbf_{law}"] = torch.logsumexp(ll, 1) - np.log(len(DELTA)); f[f"lr_level_maxlr_{law}"] = ll.max(1).values
        # scale: residual sd times exp(logf)
        ll = torch.stack([(lp(Rs / torch.exp(LOGF[g])) * vpost).sum(1) - nb * LOGF[g] for g in range(len(LOGF))], 1) - base[:, None]
        f[f"lr_scale_lbf_{law}"] = torch.logsumexp(ll, 1) - np.log(len(LOGF)); f[f"lr_scale_maxlr_{law}"] = ll.max(1).values; f[f"lr_scale_logf_{law}"] = LOGF[ll.argmax(1)]
    mb = (Rs * vpost).sum(1) / nb; f["lr_level_t"] = mb * torch.sqrt(nb); f["lr_level_abs_t"] = f["lr_level_t"].abs()
    sb = torch.sqrt((Rs ** 2 * vpost).sum(1) / nb); f["lr_scale_logsd"] = torch.log(sb.clamp(min=1e-6))
    # AR change: refit at order pb on the after part (compute all orders, gather), Chow-type LR with BIC penalty
    lr_ar = torch.zeros(N, device=dev); rss_keep = (R ** 2 * vpost).sum(1)
    for p in range(PMAX + 1):
        _, r2, ne2 = fit(p, post); sel = pb == p
        if sel.any():
            rss_new = (r2 ** 2 * post).sum(1); lr_ar[sel] = (nb * torch.log(rss_keep.clamp(min=1e-9) / rss_new.clamp(min=1e-9)))[sel] - ((p + 1) * torch.log(nb.clamp(min=2)))[sel]
    f["lr_ar_lr"] = lr_ar
    # shape: after residuals under their own best df (rescaled) vs under the before law (rescaled the same way) -> family switch evidence
    Rb = Rs / sb[:, None]; llb = torch.stack([(t_logpdf(Rb, df) * vpost).sum(1) for df in DFS], 1); best = llb.max(1).values; dfb = DFS[llb.argmax(1)]
    under_pre = torch.stack([(t_logpdf(Rb, df) * vpost).sum(1) for df in DFS], 1)[torch.arange(N, device=dev), lls.argmax(1)]
    f["lr_shape_lr"] = best - under_pre; f["lr_df_post"] = dfb; f["lr_logdf_gap"] = torch.log(dfb) - torch.log(dfa)
    # generalised-normal shape family: covers light tails (uniform-like) that the t family cannot represent
    Ra = Rs; lga = torch.stack([(gn_logpdf(Ra, bt) * vpre).sum(1) / na for bt in BETAS], 1); ba = lga.argmax(1); f["lr_gn_beta_pre"] = BETAS[ba]
    lgb_ = torch.stack([(gn_logpdf(Rb, bt) * vpost).sum(1) for bt in BETAS], 1); bb = lgb_.argmax(1); f["lr_gn_beta_post"] = BETAS[bb]
    f["lr_gn_shape_lr"] = lgb_.max(1).values - lgb_[torch.arange(N, device=dev), ba]                 # after residuals: own best beta vs the before beta
    f["lr_gn_logbeta_gap"] = torch.log(BETAS[bb]) - torch.log(BETAS[ba])
    # symmetric version: before residuals under the after beta
    Ra_s = Rs; lga_full = torch.stack([(gn_logpdf(Ra_s, bt) * vpre).sum(1) for bt in BETAS], 1)
    f["lr_gn_shape_lr_sym"] = f["lr_gn_shape_lr"] + lga_full.max(1).values - lga_full[torch.arange(N, device=dev), bb]
    lbf_gn = (f["lr_gn_shape_lr_sym"] - np.log(len(BETAS))).clamp(min=-20, max=50)
    # symmetric two-Gaussian mixture family (bimodal innovations): separation d on a grid, unit variance
    lma = torch.stack([(bimodal_logpdf(Rs, d) * vpre).sum(1) for d in DSEP], 1); da_ = lma.argmax(1); f["lr_bim_sep_pre"] = DSEP[da_]
    lmb = torch.stack([(bimodal_logpdf(Rb, d) * vpost).sum(1) for d in DSEP], 1); db_ = lmb.argmax(1); f["lr_bim_sep_post"] = DSEP[db_]
    f["lr_bim_shape_lr_sym"] = (lmb.max(1).values - lmb[torch.arange(N, device=dev), da_]) + (lma.max(1).values - lma[torch.arange(N, device=dev), db_])
    f["lr_bim_sep_gap"] = DSEP[db_] - DSEP[da_]; f["lr_bim_vs_gauss_pre"] = (lma.max(1).values - lma[:, 0]) / na
    lbf_bim = (f["lr_bim_shape_lr_sym"] - np.log(len(DSEP))).clamp(min=-20, max=50)
    # two-piece normal (skewed) family: asymmetry q fitted before and after; shape LR both ways; level/scale LBFs under the before-fitted skewed law
    lqa = torch.stack([(twopiece_logpdf(Rs, q_) * vpre).sum(1) for q_ in QS], 1); qa = lqa.argmax(1); f["lr_skew_q_pre"] = QS[qa]
    lqb = torch.stack([(twopiece_logpdf(Rb, q_) * vpost).sum(1) for q_ in QS], 1); qb = lqb.argmax(1); f["lr_skew_q_post"] = QS[qb]
    f["lr_skew_shape_lr_sym"] = (lqb.max(1).values - lqb[torch.arange(N, device=dev), qa]) + (lqa.max(1).values - lqa[torch.arange(N, device=dev), qb])
    f["lr_skew_q_gap"] = QS[qb] - QS[qa]; f["lr_skew_vs_gauss_pre"] = (lqa.max(1).values - lqa[:, 10]) / na
    qsel = QS[qa][:, None]; lp_s = lambda z: twopiece_logpdf(z, qsel); base_s = (lp_s(Rs) * vpost).sum(1)
    c = DELTA[None, :] * (1 - persist)[:, None] / sa[:, None]
    ll = torch.stack([(lp_s(Rs - c[:, g:g+1]) * vpost).sum(1) for g in range(len(DELTA))], 1) - base_s[:, None]; f["lr_level_lbf_skew"] = torch.logsumexp(ll, 1) - np.log(len(DELTA))
    ll = torch.stack([(lp_s(Rs / torch.exp(LOGF[g])) * vpost).sum(1) - nb * LOGF[g] for g in range(len(LOGF))], 1) - base_s[:, None]; f["lr_scale_lbf_skew"] = torch.logsumexp(ll, 1) - np.log(len(LOGF))
    # direct odd-moment statistics: t-test of r^3 after vs before, and the quantile-skewness gap
    cu_a = (Rs ** 3 * vpre).sum(1) / na; cu_b = (Rs ** 3 * vpost).sum(1) / nb; cu_sd = torch.sqrt(((Rs ** 3 - cu_a[:, None]) ** 2 * vpre).sum(1) / na).clamp(min=1e-6)
    f["lr_cube_t"] = (cu_b - cu_a) / (cu_sd * torch.sqrt(1 / na + 1 / nb)); f["lr_cube_abs_t"] = f["lr_cube_t"].abs()
    lbf_skew = (f["lr_skew_shape_lr_sym"] - np.log(len(QS))).clamp(min=-20, max=50)
    # Bayes mixture with the pinned prior shares (BIC-penalised LR -> approximate log BF for ar and shape)
    lbf_ar = (lr_ar / 2).clamp(min=-20, max=50); lbf_shape = ((f["lr_shape_lr"] - np.log(len(DFS))) / 1).clamp(min=-20, max=50)
    for law in ("g", "t"):
        parts = torch.stack([np.log(0.33) + f[f"lr_level_lbf_{law}"], np.log(0.33) + f[f"lr_scale_lbf_{law}"], np.log(0.08) + lbf_shape, np.log(0.05) + lbf_gn, np.log(0.04) + lbf_bim, np.log(0.03) + lbf_skew, np.log(0.08) + lbf_ar, torch.full((N,), np.log(0.06), device=dev)], 1)
        f[f"lr_mix_lbf_{law}"] = torch.logsumexp(parts, 1); f[f"lr_mix_argmax_{law}"] = parts.argmax(1).float()
    f["lr_n_post"] = nb; f["lr_sa"] = sa
    return pd.DataFrame({k: v.float().cpu().numpy() for k, v in f.items()}, index=[j[0] for j in js])




def build_jobs(name):
    jobs = []
    if name == "aug":
        MIN_SIDE = 78; rng = np.random.default_rng(1)
        s = KU_ENV.read_series(D / "data/train_series.parquet"); y = pd.read_csv(D / "data/train_labels.csv", dtype={"id": str}).set_index("id").label
        for ident, g in s.groupby("id", sort=False, observed=True):
            v = g["value"].to_numpy(np.float64); p = g["period"].to_numpy(); a, b = v[p == 0], v[p == 1]
            if y[ident] == 0:
                k = int(len(v) * rng.uniform(0.3, 0.7))
                if k >= MIN_SIDE and len(v) - k >= MIN_SIDE: jobs.append((f"{ident}s", v, k))
            for seg in (a, b):
                if len(seg) >= 2 * MIN_SIDE: rng.uniform(0.4, 0.6)
        want = set(pd.read_csv(O / "aug_meta.csv", dtype=str).query("kind == 'shift'").id); jobs = [j for j in jobs if j[0] in want]; out = O / f"aug_{PREFIX}_features.parquet"
    elif name.startswith("var:"):
        stem = name[4:]
        for ident, g in KU_ENV.read_series(O / "var" / f"{stem}_series.parquet").groupby("id", sort=False, observed=True):
            v = g["value"].to_numpy(np.float64); jobs.append((ident, v, int(g["period"].to_numpy().argmax())))
        out = O / "var" / f"{PREFIX}_{stem}.parquet"
    else:
        for ident, g in KU_ENV.read_series(D / "data" / f"{name}_series.parquet").groupby("id", sort=False, observed=True):
            v = g["value"].to_numpy(np.float64); jobs.append((ident, v, int(g["period"].to_numpy().argmax())))
        out = O / f"{PREFIX}_{name}.parquet"
    print(f"{name}: {len(jobs)} series ({time.time()-t0:.0f}s)", flush=True)


    return jobs, out


def run(jobs, out, label):
    CH = int(os.environ.get('CHUNK', '2000')); chunks = [jobs[i:i + CH] for i in range(0, len(jobs), CH)]; parts = []
    for ci, js in enumerate(chunks):
        parts.append(block(js))
        if ci % 5 == 4: print(f"  {sum(len(p) for p in parts)} done ({time.time()-t0:.0f}s)", flush=True)
    F = pd.concat(parts).replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32); F.index.name = "id"; F.to_parquet(out)
    print(f"{label}: {F.shape} -> {out.name} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    jobs, out = build_jobs(name); run(jobs, out, name)
