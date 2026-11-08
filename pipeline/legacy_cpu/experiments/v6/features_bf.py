# CPU port of experiments/v6/features_bf.py, source sha256 5afea83ce8968af25ab61aac2372db8f16c90379274d0f76079a7ed56991c3b3. See auc_lab_20261107/claude/audit/cpu_port_notes.md.
import sys as _s, pathlib as _p; _s.path.insert(0, str(_p.Path(__file__).resolve().parents[2])); import ku_workspace as KU_ENV
"""Conjugate Bayes-factor block, batched on the Arc.  Gaussian AR(p) regression y_t = c + sum phi_j y_{t-j} + e with a Normal-Inverse-Gamma
prior (beta | s2 ~ N(0, s2 V0), s2 ~ IG(a0, b0)); the marginal likelihood of a segment is closed-form, so the evidence for a change at the
boundary with ALL nuisance parameters integrated out is:
  bf_full_p{p}   log ML(before) + log ML(after) - log ML(whole), p in {0, 1, 2, 4} and the BIC order (bf_full_bic)
  bf_level_p{p}  log ML(whole with an after-indicator regressor) - log ML(whole): level shift only, coefficients and variance shared
  bf_scale_p{p}  variance-only change: residuals of the pooled fit, IG evidence for two variances vs one
  bf_ar_p{p}     coefficient change with shared variance ~ full minus scale (approximation reported as is)
Series standardised by the before part; V0 = 10 I, a0 = b0 = 1 (weak, scale-free on standardised data).  ~22 columns, prefix bf_.
Usage: .venv/Scripts/python features_bf.py train|test|aug -> tune_out/bf_<name>.parquet / aug_bf_features.parquet"""
import sys, time, pathlib, warnings, ctypes; warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
try: ctypes.windll.kernel32.SetPriorityClass(ctypes.c_void_p(-1), 0x4000)
except Exception: pass
import numpy as np, pandas as pd, torch
D = KU_ENV.root(); O = D / "tune_out"
dev = "cpu"; torch.set_num_threads(1); t0 = time.time()
L = 1024; name = sys.argv[1]; V0, A0, B0 = 10.0, 1.0, 1.0; ORDERS = [0, 1, 2, 4, 8]

jobs = []
if name == "aug":
    MIN_SIDE = 78; rng = np.random.default_rng(1)
    s = pd.read_parquet(D / "data/train_series.parquet"); y = pd.read_csv(D / "data/train_labels.csv", dtype={"id": str}).set_index("id").label
    for ident, g in s.groupby("id", sort=False):
        v = g["value"].to_numpy(np.float64); p = g["period"].to_numpy(); a, b = v[p == 0], v[p == 1]
        if y[ident] == 0:
            k = int(len(v) * rng.uniform(0.3, 0.7))
            if k >= MIN_SIDE and len(v) - k >= MIN_SIDE: jobs.append((f"{ident}s", v, k))
        for seg in (a, b):
            if len(seg) >= 2 * MIN_SIDE: rng.uniform(0.4, 0.6)
    want = set(pd.read_csv(O / "aug_meta.csv", dtype=str).query("kind == 'shift'").id); jobs = [j for j in jobs if j[0] in want]; out = O / "aug_bf_features.parquet"
else:
    for ident, g in pd.read_parquet(D / "data" / f"{name}_series.parquet").groupby("id", sort=False):
        jobs.append((ident, g["value"].to_numpy(np.float64), int(g["period"].to_numpy().argmax())))
    out = O / f"lr_{name}.parquet".replace("lr_", "bf_")
print(f"{name}: {len(jobs)} series ({time.time()-t0:.0f}s)", flush=True)


def log_ml(G, h, yy, n, k):
    """log marginal likelihood of y ~ N(X beta, s2 I) under beta|s2 ~ N(0, s2 V0 I), s2 ~ IG(A0, B0); G = X'X (N,k,k), h = X'y (N,k), yy = y'y (N,), n = rows (N,)."""
    I = torch.eye(k, device=dev); Vn_inv = G + I / V0; mu = torch.linalg.solve(Vn_inv, h[..., None]).squeeze(-1)
    bn = B0 + 0.5 * (yy - (mu * h).sum(-1)); an = A0 + n / 2
    logdet = -torch.logdet(Vn_inv) - k * np.log(V0)                                  # log det(Vn) - log det(V0)
    return 0.5 * logdet + A0 * np.log(B0) - an * torch.log(bn.clamp(min=1e-12)) + torch.lgamma(an) - np.log(np.pi) * n / 2 - torch.lgamma(torch.tensor(A0, device=dev))


def block(js):
    N = len(js); X = torch.zeros(N, L, device=dev); n = torch.tensor([min(len(v), L) for _, v, _ in js], device=dev); b = torch.tensor([k for _, _, k in js], device=dev)
    for i, (_, v, _) in enumerate(js): X[i, :min(len(v), L)] = torch.as_tensor(v[:L], device=dev, dtype=torch.float32)
    tt = torch.arange(L, device=dev)[None, :]; pre = (tt < b[:, None]).float(); post = ((tt >= b[:, None]) & (tt < n[:, None])).float(); valid = pre + post
    m = (X * pre).sum(1) / b; sd = torch.sqrt((((X - m[:, None]) * pre) ** 2).sum(1) / b).clamp(min=1e-6); Z = (X - m[:, None]) / sd[:, None] * valid
    f = {}; bic_scores = []; store = {}
    for p in ORDERS:
        if p == 0: yv = Z; A = torch.ones(N, L, 1, device=dev); masks = {"pre": pre, "post": post, "all": valid}
        else:
            U = Z.unfold(1, p + 1, 1); yv = U[:, :, -1]; A = torch.cat([U[:, :, :-1].flip(-1), torch.ones_like(yv)[..., None]], -1)
            masks = {"pre": pre[:, p:] * (tt[:, p:] >= p).float(), "post": post[:, p:], "all": valid[:, p:]}
        # the AR-lag rows for the after part use before values as lags at the boundary (correct: the process continues)
        ml = {}; res_pool = None
        for key, mk in masks.items():
            mk_ = mk[..., None]; G = (A * mk_).transpose(1, 2) @ A; h = ((A * mk_).transpose(1, 2) @ yv[..., None]).squeeze(-1); yy = (yv ** 2 * mk).sum(1); nn = mk.sum(1)
            ml[key] = log_ml(G, h, yy, nn, A.shape[-1])
            if key == "all":
                I = torch.eye(A.shape[-1], device=dev); mu = torch.linalg.solve(G + I / V0, h[..., None]).squeeze(-1); res_pool = (yv - (A @ mu[..., None]).squeeze(-1)) * mk
        f[f"bf_full_p{p}"] = ml["pre"] + ml["post"] - ml["all"]
        # level-only: add the after indicator as a regressor on the whole series
        Ad = torch.cat([A, masks["post"][..., None]], -1); mk_ = masks["all"][..., None]; G = (Ad * mk_).transpose(1, 2) @ Ad; h = ((Ad * mk_).transpose(1, 2) @ yv[..., None]).squeeze(-1)
        f[f"bf_level_p{p}"] = log_ml(G, h, (yv ** 2 * masks["all"]).sum(1), masks["all"].sum(1), Ad.shape[-1]) - ml["all"]
        # scale-only: pooled-fit residuals, two IG variances vs one
        ra2 = (res_pool ** 2 * masks["pre"]).sum(1); rb2 = (res_pool ** 2 * masks["post"]).sum(1); na, nb = masks["pre"].sum(1), masks["post"].sum(1)
        ig = lambda ss, nn: A0 * np.log(B0) - (A0 + nn / 2) * torch.log(B0 + ss / 2) + torch.lgamma(A0 + nn / 2) - torch.lgamma(torch.tensor(A0, device=dev)) - np.log(np.pi) * nn / 2
        f[f"bf_scale_p{p}"] = ig(ra2, na) + ig(rb2, nb) - ig(ra2 + rb2, na + nb)
        f[f"bf_ar_p{p}"] = f[f"bf_full_p{p}"] - f[f"bf_scale_p{p}"]
        bic_scores.append(ml["pre"]); store[p] = (f[f"bf_full_p{p}"], f[f"bf_level_p{p}"], f[f"bf_scale_p{p}"])
    best = torch.stack(bic_scores, 1).argmax(1); f["bf_order"] = best.float()
    for j, nm in enumerate(("full", "level", "scale")): f[f"bf_{nm}_best"] = torch.stack([store[p][j] for p in ORDERS], 1)[torch.arange(N, device=dev), best]
    f["bf_max_full"] = torch.stack([store[p][0] for p in ORDERS], 1).max(1).values
    f["bf_mix_best"] = torch.logsumexp(torch.stack([np.log(0.4) + f["bf_level_best"], np.log(0.4) + f["bf_scale_best"], np.log(0.2) + f["bf_full_best"]], 1), 1)
    return pd.DataFrame({k: v.float().cpu().numpy() for k, v in f.items()}, index=[j[0] for j in js])


parts = [block(jobs[i:i + 2000]) for i in range(0, len(jobs), 2000)]
F = pd.concat(parts).replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32); F.index.name = "id"; F.to_parquet(out)
print(f"{name}: {F.shape} -> {out.name} in {time.time()-t0:.0f}s", flush=True)
