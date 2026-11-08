# CPU port of experiments/v6/features_lrz.py, source sha256 edacfc2c11a7cd1cf46933f66609c06449b615f19102befcfb785257897ea176. See auc_lab_20261107/claude/audit/cpu_port_notes.md.
import sys as _s, pathlib as _p; _s.path.insert(0, str(_p.Path(__file__).resolve().parents[2])); import ku_workspace as KU_ENV
"""Per-series null calibration of the exact statistics at pseudo-boundaries (idea 1 of the tree-side list), Intel Arc.
For every series the full LR block is recomputed on K pseudo-boundaries placed inside the BEFORE part (series = v[:b], boundary at
fraction f of b) and K inside the AFTER part (series = v[b:], boundary at fraction f of n-b).  Both sub-series are homogeneous under
either label, so the 2K values of each statistic form that series' own null.  Emitted per statistic: z = (value at the real boundary
- null mean) / null sd, the rank of the real value among the 2K+1, the null mean on the before part and on the after part, and their
difference (a within-after change is a misspecification flag).  Real data only; nothing is simulated.
Usage: .venv/Scripts/python features_lrz.py train|test|aug [K=6]  -> tune_out/lrz_<name>.parquet / aug_lrz_features.parquet"""
import sys, os, time, pathlib
sys.stdout.reconfigure(encoding="utf-8"); HERE = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import numpy as np, pandas as pd
import features_lr as FL
name = sys.argv[1]; K = int(sys.argv[2]) if len(sys.argv) > 2 else 6; t0 = time.time()
jobs, _ = FL.build_jobs(name); TAG = os.environ.get("LRZ_TAG", "lrz")
out = (FL.O / "var" / f"{TAG}_{name[4:]}.parquet") if name.startswith("var:") else (FL.O / f"aug_{TAG}_features.parquet" if name == "aug" else FL.O / f"{TAG}_{name}.parquet")
STATS = ["lr_level_abs_t", "lr_scale_logsd", "lr_scale_maxlr_t", "lr_ar_lr", "lr_shape_lr", "lr_gn_shape_lr_sym", "lr_bim_shape_lr_sym", "lr_level_lbf_t", "lr_scale_lbf_t", "lr_mix_lbf_t"]
MIN = 60


def table(js):
    CH = int(os.environ.get('CHUNK', '2000')); parts = [FL.block(js[i:i + CH]) for i in range(0, len(js), CH)]; return pd.concat(parts)[STATS].replace([np.inf, -np.inf], np.nan)


real = table(jobs); real["lr_scale_logsd"] = real["lr_scale_logsd"].abs(); print(f"real boundary stats ({time.time()-t0:.0f}s)", flush=True)
fracs = np.linspace(0.3, 0.7, K); nulls = []
for side in ("pre", "post"):
    for f in fracs:
        js = []
        for ident, v, b in jobs:
            seg = v[:b] if side == "pre" else v[b:]; k = int(len(seg) * f)
            if len(seg) >= 2 * MIN and k >= MIN and len(seg) - k >= MIN: js.append((ident, seg, k))
            else: js.append((ident, v, b))          # too short: fall back to the real split (will be masked out below)
        T = table(js); T["lr_scale_logsd"] = T["lr_scale_logsd"].abs(); T["_valid"] = [len((v[:b] if side == "pre" else v[b:])) >= 2 * MIN for _, v, b in jobs]
        T["_side"] = side; nulls.append(T); print(f"  null {side} f={f:.2f} ({time.time()-t0:.0f}s)", flush=True)
N = pd.concat(nulls); N.loc[~N._valid, STATS] = np.nan
F = {}
for c in STATS:
    g = N.groupby(level=0)[c]; mu = g.mean().loc[real.index]; sd = g.std().loc[real.index]; mpre = N[N._side == "pre"].groupby(level=0)[c].mean().loc[real.index]; mpost = N[N._side == "post"].groupby(level=0)[c].mean().loc[real.index]
    F[f"lrz_{c}_z"] = (real[c] - mu) / (sd + 1e-6); F[f"lrz_{c}_nullmean"] = mu; F[f"lrz_{c}_postpre"] = mpost - mpre; F[f"lrz_{c}_excess"] = real[c] - mu
# rank of the real value among its nulls (vectorised via sorting)
Nv = N[STATS].copy(); Nv["_id"] = Nv.index
for c in STATS:
    joined = Nv[["_id", c]].merge(real[[c]].rename(columns={c: "_real"}), left_on="_id", right_index=True)
    F[f"lrz_{c}_rank"] = (joined[c] < joined["_real"]).groupby(joined["_id"]).mean().loc[real.index]
out_df = pd.DataFrame(F, index=real.index).replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32); out_df.index.name = "id"; out_df.to_parquet(out)
print(f"{name}: {out_df.shape} -> {out.name} in {time.time()-t0:.0f}s", flush=True)
