# CPU port of experiments/v6/features_ssa.py, source sha256 eb3be4b649c8522027e2ff5aca34bdf3ae13b82556915e6ae06bf03bf5965454. See auc_lab_20261107/claude/audit/cpu_port_notes.md.
import sys as _s, pathlib as _p; _s.path.insert(0, str(_p.Path(__file__).resolve().parents[2])); import ku_workspace as KU_ENV
"""SSA / lag-covariance and AR(2)-oscillator block on the Arc (template: features_lr.py).
Per series (z-scored by the before part): lag-vector covariances of the before and after parts for window M in (8, 24);
Box-M log-det LR (equality of lag covariances), eigen-share gaps, SSA subspace residual (after windows projected on the before top-r subspace,
r = 1, 2, 4, relative to the before windows' own residual), and AR(2) root modulus / angle (damping, period) before vs after.
Usage: .venv/Scripts/python features_ssa.py train|test|aug  -> tune_out/ssa_<name>.parquet / aug_ssa_features.parquet"""
import sys, time, pathlib, ctypes
sys.stdout.reconfigure(encoding="utf-8")
try: ctypes.windll.kernel32.SetPriorityClass(ctypes.c_void_p(-1), 0x4000)
except Exception: pass
import numpy as np, pandas as pd, torch
D = KU_ENV.root(); O = D / "tune_out"; name = sys.argv[1]; PREFIX = "ssa"; L = 1024
dev = "cpu"; torch.set_num_threads(1); t0 = time.time()


def block(js):
    N = len(js); X = torch.zeros(N, L, device=dev); n = torch.tensor([min(len(v), L) for _, v, _ in js], device=dev); b = torch.tensor([k for _, _, k in js], device=dev)
    for i, (_, v, _) in enumerate(js): X[i, :min(len(v), L)] = torch.as_tensor(v[:L], device=dev, dtype=torch.float32)
    tt = torch.arange(L, device=dev)[None, :]; pre = (tt < b[:, None]).float(); post = ((tt >= b[:, None]) & (tt < n[:, None])).float(); valid = pre + post
    m = (X * pre).sum(1) / b; sd = torch.sqrt((((X - m[:, None]) * pre) ** 2).sum(1) / b).clamp(min=1e-6); Z = (X - m[:, None]) / sd[:, None] * valid
    f = {}
    for M in (8, 24):
        U = Z.unfold(1, M, 1); st = torch.arange(U.shape[1], device=dev)[None, :]
        wa = ((st + M) <= b[:, None]).float(); wb = ((st >= b[:, None]) & ((st + M) <= n[:, None])).float(); na, nb = wa.sum(1).clamp(min=2), wb.sum(1).clamp(min=2)
        Ca = torch.einsum("nlm,nlk->nmk", U * wa[..., None], U) / na[:, None, None]; Cb = torch.einsum("nlm,nlk->nmk", U * wb[..., None], U) / nb[:, None, None]
        Cp = (Ca * na[:, None, None] + Cb * nb[:, None, None]) / (na + nb)[:, None, None]; I = 1e-4 * torch.eye(M, device=dev)
        ld = lambda C: torch.linalg.slogdet(C + I)[1]
        f[f"ssa{M}_boxm"] = ((na + nb) * ld(Cp) - na * ld(Ca) - nb * ld(Cb)) / M
        ea, Va = torch.linalg.eigh(Ca + I); eb = torch.linalg.eigvalsh(Cb + I); ea, eb = ea.flip(1), eb.flip(1); Va = Va.flip(2)
        sa, sb = ea / ea.sum(1, keepdim=True), eb / eb.sum(1, keepdim=True)
        f[f"ssa{M}_top1_gap"] = sb[:, 0] - sa[:, 0]; f[f"ssa{M}_spec_l1"] = (sa - sb).abs().sum(1); f[f"ssa{M}_logtr_gap"] = torch.log(eb.sum(1) / ea.sum(1))
        if M == 24:
            nrm = (U ** 2).sum(-1)
            for r in (1, 2, 4):
                P = (U @ Va[:, :, :r]) ** 2; res = nrm - P.sum(-1)
                ra = (res * wa).sum(1) / na; rb = (res * wb).sum(1) / nb; f[f"ssa{M}_subres{r}"] = torch.log(rb.clamp(min=1e-6) / ra.clamp(min=1e-6))
                f[f"ssa{M}_subres{r}_rel"] = torch.log((rb / (eb.sum(1) + 1e-6)).clamp(min=1e-6) / (ra / (ea.sum(1) + 1e-6)).clamp(min=1e-6))
    # AR(2) on each side: phi from masked least squares, root modulus (damping) and angle (2*pi/period)
    U3 = Z.unfold(1, 3, 1); y_ = U3[:, :, 2]; A = torch.stack([U3[:, :, 1], U3[:, :, 0], torch.ones_like(y_)], -1); st = torch.arange(U3.shape[1], device=dev)[None, :]
    ph = {}
    for side, mk in (("pre", ((st + 3) <= b[:, None]).float()), ("post", ((st >= b[:, None]) & ((st + 3) <= n[:, None])).float())):
        Am = A * mk[..., None]; beta = torch.linalg.solve(Am.transpose(1, 2) @ A + 1e-4 * torch.eye(3, device=dev), (Am.transpose(1, 2) @ y_[..., None])).squeeze(-1)
        p1, p2 = beta[:, 0], beta[:, 1]; disc = p1 ** 2 + 4 * p2; cplx = disc < 0
        mod = torch.where(cplx, torch.sqrt((-p2).clamp(min=0)), torch.maximum((p1 + torch.sqrt(disc.clamp(min=0))).abs(), (p1 - torch.sqrt(disc.clamp(min=0))).abs()) / 2)
        ang = torch.where(cplx, torch.acos((p1 / (2 * torch.sqrt((-p2).clamp(min=1e-9)))).clamp(-1, 1)), torch.zeros_like(p1))
        ph[side] = (p1, p2, mod, ang); f[f"osc_phi1_{side}"] = p1; f[f"osc_phi2_{side}"] = p2; f[f"osc_mod_{side}"] = mod; f[f"osc_ang_{side}"] = ang
    f["osc_mod_gap"] = ph["post"][2] - ph["pre"][2]; f["osc_ang_gap"] = ph["post"][3] - ph["pre"][3]; f["osc_ang_agap"] = f["osc_ang_gap"].abs()
    f["osc_phi_l2"] = torch.sqrt((ph["post"][0] - ph["pre"][0]) ** 2 + (ph["post"][1] - ph["pre"][1]) ** 2)
    return pd.DataFrame({k: v.float().cpu().numpy() for k, v in f.items()}, index=[j[0] for j in js])


def build_jobs(name):
    jobs = []
    if name == "aug":
        MIN_SIDE = 78; rng = np.random.default_rng(1)
        s = pd.read_parquet(D / "data/train_series.parquet"); y = pd.read_csv(D / "data/train_labels.csv", dtype={"id": str}).set_index("id").label
        for ident, g in s.groupby("id", sort=False):
            v = g["value"].to_numpy(np.float64); p = g["period"].to_numpy(); a, bb = v[p == 0], v[p == 1]
            if y[ident] == 0:
                k = int(len(v) * rng.uniform(0.3, 0.7))
                if k >= MIN_SIDE and len(v) - k >= MIN_SIDE: jobs.append((f"{ident}s", v, k))
            for seg in (a, bb):
                if len(seg) >= 2 * MIN_SIDE: rng.uniform(0.4, 0.6)
        want = set(pd.read_csv(O / "aug_meta.csv", dtype=str).query("kind == 'shift'").id); jobs = [j for j in jobs if j[0] in want]; out = O / f"aug_{PREFIX}_features.parquet"
    else:
        for ident, g in pd.read_parquet(D / "data" / f"{name}_series.parquet").groupby("id", sort=False):
            v = g["value"].to_numpy(np.float64); jobs.append((ident, v, int(g["period"].to_numpy().argmax())))
        out = O / f"{PREFIX}_{name}.parquet"
    print(f"{name}: {len(jobs)} series ({time.time()-t0:.0f}s)", flush=True); return jobs, out


if __name__ == "__main__":
    jobs, out = build_jobs(name); parts = [block(jobs[i:i + 1000]) for i in range(0, len(jobs), 1000)]
    F = pd.concat(parts).replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32); F.index.name = "id"; F.to_parquet(out)
    print(f"{name}: {F.shape} -> {out.name} in {time.time()-t0:.0f}s", flush=True)
