# CPU port of experiments/v6/generator_probe.py, source sha256 8f719778a8efe4b1a195cc56efffac3f9c2e0e80a8b3eff74e86855411520ec4. See auc_lab_20261107/claude/audit/cpu_port_notes.md.
import sys as _s, pathlib as _p; _s.path.insert(0, str(_p.Path(__file__).resolve().parents[2])); import ku_workspace as KU_ENV
"""Pin down the generator from the data.  All segment fits are batched in torch on the Intel Arc (xpu).
Part A (no-break process): every before segment (24k train + 4k test) and every after segment of a label-0 series (12k) is break-free.
  For each: AR(p) by least squares for p = 0..8, BIC order, coefficients, innovation sd, innovation skew / excess kurtosis / tail share,
  Student-t df by grid MLE, ACF of residuals (MA check) and of squared residuals (GARCH check), spectral peak share (seasonality check).
Part B (interventions): the same fits on the after segment of every train series; parameter differences after-before for label 1 vs label 0
  (label 0 differences = pure estimation noise).  Deconvolving the two gives the intervention types and magnitudes.
Outputs: tune_out/generator_segments.parquet (one row per segment), docs/GENERATOR.md numbers printed to stdout, figures graphs/48-51.
Usage: .venv-xpu/Scripts/python generator_probe.py"""
import sys, time, pathlib, warnings, ctypes, json; warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
try: ctypes.windll.kernel32.SetPriorityClass(ctypes.c_void_p(-1), 0x4000)
except Exception: pass
import numpy as np, pandas as pd, torch
D = KU_ENV.root(); O = D / "tune_out"
dev = "cpu"; torch.set_num_threads(1); t0 = time.time()
L = 1024; PMAX = 16


def load(path):
    segs, meta = [], []
    for ident, g in pd.read_parquet(path).groupby("id", sort=False):
        v = g["value"].to_numpy(np.float32); b = int(g["period"].to_numpy().argmax())
        segs.append(v[:b]); meta.append((ident, "pre", b, len(v) - b)); segs.append(v[b:]); meta.append((ident, "post", b, len(v) - b))
    return segs, meta


segs, meta = load(D / "data/train_series.parquet"); nt = len(segs)
st, mt = load(D / "data/test_series.parquet"); segs += st; meta += mt
N = len(segs); n = torch.tensor([len(s) for s in segs], device=dev); X = torch.zeros(N, L, device=dev)
for i, s in enumerate(segs): X[i, :len(s)] = torch.as_tensor(s, device=dev)
M = pd.DataFrame(meta, columns=["id", "part", "n_pre", "n_post"]); M["train"] = np.arange(N) < nt
y = pd.read_csv(D / "data/train_labels.csv", dtype={"id": str}).set_index("id").label
M["label"] = M.id.map(y).fillna(-1).astype(int)
print(f"{N} segments on {dev} ({time.time()-t0:.0f}s)", flush=True)
tt = torch.arange(L, device=dev)[None, :]; valid = (tt < n[:, None]).float()
mean = (X * valid).sum(1) / n; sd = torch.sqrt((((X - mean[:, None]) * valid) ** 2).sum(1) / n)
Z = ((X - mean[:, None]) / sd[:, None]) * valid


def ar_fit(Z, p):
    """Batched AR(p) with intercept on the standardised segment.  Returns beta (N, p+1), residuals (N, L) (zeros where undefined), n_eff."""
    if p == 0:
        r = Z * valid; return torch.zeros(N, 1, device=dev), r, n.float()
    U = Z.unfold(1, p + 1, 1)                                   # (N, L-p, p+1): x_{t-p} .. x_t
    yv = U[:, :, -1]; lag = U[:, :, :-1].flip(-1)              # lags x_{t-1} .. x_{t-p}
    A = torch.cat([lag, torch.ones_like(yv)[..., None]], -1)   # (N, L-p, p+1)
    m = (tt[:, p:] < n[:, None]).float()[..., None]             # valid rows t < n
    Am = A * m; G = Am.transpose(1, 2) @ A + 1e-4 * torch.eye(p + 1, device=dev); h = (Am.transpose(1, 2) @ yv[..., None])
    beta = torch.linalg.solve(G, h).squeeze(-1)                 # (N, p+1)
    res = (yv - (A @ beta[..., None]).squeeze(-1)) * m.squeeze(-1)
    r = torch.zeros(N, L, device=dev); r[:, p:] = res; return beta, r, (n - p).float()


out = {}
bic = []; fits = {}
for p in range(PMAX + 1):
    beta, r, ne = ar_fit(Z, p); s2 = (r ** 2).sum(1) / ne; bic.append(ne * torch.log(s2) + (p + 1) * torch.log(ne)); fits[p] = (beta, r, ne, s2)
bic = torch.stack(bic, 1); p_bic = bic.argmin(1); out["p_bic"] = p_bic
aic = torch.stack([ne * torch.log(fits[p][3]) + 2 * (p + 1) for p in range(PMAX + 1)], 1); out["p_aic"] = aic.argmin(1)
# coefficients at p=4 (common basis) and at p=1
b4 = fits[4][0]; out.update({f"phi{j+1}": b4[:, j] for j in range(4)}); out["phi1_ar1"] = fits[1][0][:, 0]
out["persist4"] = b4[:, :4].sum(1); out["innov_sd4"] = torch.sqrt(fits[4][3]) * sd            # innovation sd in raw units
out["mean"] = mean; out["sd"] = sd; out["n"] = n.float()
# residual diagnostics at the BIC order (gather)
R = torch.stack([fits[p][1] for p in range(PMAX + 1)], 1)[torch.arange(N, device=dev), p_bic]      # (N, L)
ne = n.float() - p_bic.float(); rm = (tt >= p_bic[:, None]).float() * valid
rs = torch.sqrt((R ** 2).sum(1) / ne); Rs = R / rs[:, None] * rm
out["skew"] = (Rs ** 3).sum(1) / ne; out["kurt"] = (Rs ** 4).sum(1) / ne - 3
out["tail3"] = ((Rs.abs() > 3).float() * rm).sum(1) / ne; out["tail2"] = ((Rs.abs() > 2).float() * rm).sum(1) / ne
q = lambda v, qq: torch.quantile(v, qq, dim=1)
# robust shape: ratio of quantile ranges (uniform-like < Gaussian < heavy)
out["q_ratio"] = (q(Rs.masked_fill(rm == 0, float("nan")).nan_to_num(0), 0.99) - q(Rs.masked_fill(rm == 0, float("nan")).nan_to_num(0), 0.01)) / \
                 (q(Rs.masked_fill(rm == 0, float("nan")).nan_to_num(0), 0.75) - q(Rs.masked_fill(rm == 0, float("nan")).nan_to_num(0), 0.25) + 1e-6)
# residual ACF (MA / misfit check) and squared-residual ACF (GARCH check)
def acf(V, m, k):
    a = (V[:, k:] * V[:, :-k] * m[:, k:] * m[:, :-k]).sum(1); d = (V ** 2 * m).sum(1); return a / d
for k in (1, 2, 3, 5, 10): out[f"racf{k}"] = acf(Rs, rm, k)
R2 = (Rs ** 2 - 1) * rm
for k in (1, 2, 5): out[f"sqacf{k}"] = acf(R2, rm, k)
# Student-t df by grid MLE on standardised residuals (scale free, scale re-fitted by moment ratio)
dfs = torch.tensor([2.1, 2.3, 2.6, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 25.0, 50.0, 200.0], device=dev)
ll = []
for df in dfs:
    scale = torch.sqrt((df - 2) / df) if df > 2 else torch.tensor(0.5, device=dev)   # unit-variance t has scale sqrt((df-2)/df)
    z = Rs / scale; l = KU_ENV.t_constant(df, z) - torch.log(scale) - (df + 1) / 2 * torch.log1p(z ** 2 / df)
    ll.append((l * rm).sum(1) / ne)
ll = torch.stack(ll, 1); out["t_df"] = dfs[ll.argmax(1)]; gauss = (-0.5 * np.log(2 * np.pi) - 0.5 * Rs ** 2) * rm
out["ll_t_minus_gauss"] = ll.max(1).values - gauss.sum(1) / ne
# seasonality / spectral peak share on the standardised segment
F = torch.fft.rfft(Z * valid, n=L, dim=1).abs() ** 2; F[:, 0] = 0; out["spec_peak_share"] = F.max(1).values / F.sum(1); out["spec_peak_freq"] = F.argmax(1).float() / L
# level drift inside the segment (first half vs second half mean, in innovation-sd units) and linear slope
half = (tt < (n[:, None] // 2)).float() * valid; m1 = (Z * half).sum(1) / half.sum(1); m2 = (Z * (valid - half)).sum(1) / (valid - half).sum(1)
out["half_mean_diff"] = (m2 - m1) * sd / out["innov_sd4"]
S = pd.DataFrame({k: v.float().cpu().numpy() for k, v in out.items()}); S = pd.concat([M.reset_index(drop=True), S], axis=1)
S.to_parquet(O / "generator_segments.parquet"); print(f"segment table {S.shape} written ({time.time()-t0:.0f}s)", flush=True)

# ------------------------------------------------------------------ Part A summary: the no-break process
free = S[(S.part == "pre") | ((S.part == "post") & (S.label == 0))]
print("\n== A. break-free segments:", len(free))
print("BIC order share:", (free.p_bic.value_counts(normalize=True).sort_index() * 100).round(1).to_dict())
print("AIC order share:", (free.p_aic.value_counts(normalize=True).sort_index() * 100).round(1).to_dict())
print("phi1 (AR1 fit) quantiles 1/5/25/50/75/95/99%:", np.round(np.percentile(free.phi1_ar1, [1, 5, 25, 50, 75, 95, 99]), 3).tolist())
print("persistence (sum phi, AR4) quantiles:", np.round(np.percentile(free.persist4, [1, 5, 25, 50, 75, 95, 99]), 3).tolist())
print("share |phi2..4| > 0.15 at AR4:", np.round([(free[f"phi{j}"].abs() > 0.15).mean() for j in (2, 3, 4)], 3).tolist())
print("residual acf1/acf2/acf5 at BIC order, mean |.|:", np.round([free[f"racf{k}"].abs().mean() for k in (1, 2, 5)], 4).tolist(), " (2/sqrt(n) ~", round(float((2 / np.sqrt(free.n)).mean()), 3), ")")
print("squared-residual acf1 quantiles 50/90/95/99%:", np.round(np.percentile(free.sqacf1, [50, 90, 95, 99]), 3).tolist())
print("innovation skew quantiles 5/25/50/75/95%:", np.round(np.percentile(free["skew"], [5, 25, 50, 75, 95]), 3).tolist())
print("innovation excess kurtosis quantiles 5/25/50/75/95/99%:", np.round(np.percentile(free["kurt"], [5, 25, 50, 75, 95, 99]), 3).tolist())
print("t df (MLE grid) share:", (free.t_df.value_counts(normalize=True).sort_index() * 100).round(1).to_dict())
print("ll(t)-ll(gauss) per obs quantiles 50/75/90/95/99%:", np.round(np.percentile(free.ll_t_minus_gauss, [50, 75, 90, 95, 99]), 4).tolist())
print("spectral peak share quantiles 50/90/99%:", np.round(np.percentile(free.spec_peak_share, [50, 90, 99]), 3).tolist())
print("raw mean quantiles 1/50/99%:", np.round(np.percentile(free["mean"], [1, 50, 99]), 2).tolist(), " raw sd quantiles 1/50/99%:", np.round(np.percentile(free.sd, [1, 50, 99]), 2).tolist())
print("innovation sd (raw units) quantiles 1/50/99%:", np.round(np.percentile(free.innov_sd4, [1, 50, 99]), 3).tolist())

# ------------------------------------------------------------------ Part B: interventions = after - before, label 1 vs label 0
pre = S[(S.part == "pre") & S.train].set_index("id"); post = S[(S.part == "post") & S.train].set_index("id").loc[pre.index]
lab = pre.label.to_numpy()
dd = pd.DataFrame(index=pre.index)
dd["d_mean"] = (post["mean"] - pre["mean"]) / pre.innov_sd4                       # level shift in innovation-sd units
dd["d_logsd_innov"] = np.log(post.innov_sd4 / pre.innov_sd4)                          # innovation scale change
dd["d_logsd_level"] = np.log(post.sd / pre.sd)                                        # marginal sd change
dd["d_phi1"] = post.phi1 - pre.phi1; dd["d_persist"] = post.persist4 - pre.persist4
dd["d_phi_l2"] = np.sqrt(sum((post[f"phi{j}"] - pre[f"phi{j}"]) ** 2 for j in (1, 2, 3, 4)))
dd["d_kurt"] = post["kurt"] - pre["kurt"]; dd["d_skew"] = post["skew"] - pre["skew"]; dd["d_logdf"] = np.log(post.t_df) - np.log(pre.t_df)
dd["d_tail3"] = post.tail3 - pre.tail3; dd["d_qratio"] = np.log(post.q_ratio / pre.q_ratio); dd["d_sqacf1"] = post.sqacf1 - pre.sqacf1
dd["post_half_drift"] = post.half_mean_diff; dd["pre_half_drift"] = pre.half_mean_diff
dd["label"] = lab; dd.to_parquet(O / "generator_deltas.parquet")
print("\n== B. after-before parameter differences (label 0 = estimation noise, label 1 = noise + intervention)")
rows = []
for c in [c for c in dd.columns if c.startswith("d_") or c.endswith("drift")]:
    a0, a1 = dd[c][lab == 0].abs(), dd[c][lab == 1].abs()
    q0 = np.percentile(a0, 95)                                                        # 95th percentile of noise
    rows.append((c, np.median(a0), np.median(a1), q0, (a1 > q0).mean(), (a0 > q0).mean()))
T = pd.DataFrame(rows, columns=["delta", "median|d| label0", "median|d| label1", "noise p95", "share label1 > noise p95", "share label0 > noise p95"])
print(T.round(3).to_string(index=False))
# break type attribution: which deltas exceed noise p95, per break (co-occurrence)
keys = ["d_mean", "d_logsd_innov", "d_phi_l2", "d_kurt"]; names = ["level", "scale", "ar", "shape"]
E = np.column_stack([(dd[k][lab == 1].abs() > np.percentile(dd[k][lab == 0].abs(), 95)).to_numpy() for k in keys])
cnt = E.sum(1); print("\nbreaks with 0/1/2/3+ exceedances (p95 of noise):", {int(k): round(float((cnt == k).mean()), 3) for k in range(4)})
print("single-type shares among single-exceedance breaks:", {nm: round(float(E[cnt == 1][:, j].mean()), 3) for j, nm in enumerate(names)})
P = E.T.astype(float) @ E.astype(float) / len(E); print("co-exceedance matrix (share of all breaks):\n", pd.DataFrame(P, index=names, columns=names).round(3).to_string())
exp_ind = np.outer(E.mean(0), E.mean(0)); print("ratio observed/independent co-exceedance:\n", pd.DataFrame(P / exp_ind, index=names, columns=names).round(2).to_string())
# magnitude distribution of level shifts beyond noise: signed d_mean for label 1 vs 0
for c in ("d_mean", "d_logsd_innov", "d_phi1", "d_logdf"):
    print(f"{c}: label1 signed quantiles 1/5/10/25/50/75/90/95/99%:", np.round(np.percentile(dd[c][lab == 1], [1, 5, 10, 25, 50, 75, 90, 95, 99]), 3).tolist())
    print(f"{' '*len(c)}  label0 signed quantiles:", np.round(np.percentile(dd[c][lab == 0], [1, 5, 10, 25, 50, 75, 90, 95, 99]), 3).tolist())
# step or ramp?  among breaks with clear level shift: drift inside the after segment vs inside before
big = (lab == 1) & (dd.d_mean.abs() > np.percentile(dd.d_mean[lab == 0].abs(), 99))
print(f"\nclear level breaks (n={int(big.sum())}): median |within-after half-drift| {dd.post_half_drift[big].abs().median():.3f} vs within-before {dd.pre_half_drift[big].abs().median():.3f} (ramp would make after >> before)")
print(f"done {time.time()-t0:.0f}s")
