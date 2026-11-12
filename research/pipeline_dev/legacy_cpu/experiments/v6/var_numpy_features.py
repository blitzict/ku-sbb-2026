# CPU port of experiments/v6/var_numpy_features.py, source sha256 f0c244f1de9d12b2e6da6bd79545f2eaca8429a097d17a94ede19503151e0520. See auc_lab_20261107/claude/audit/cpu_port_notes.md.
import sys as _s, pathlib as _p; _s.path.insert(0, str(_p.Path(__file__).resolve().parents[2])); import ku_workspace as KU_ENV
"""Numpy feature families (clean 977, grid, grid2, dd) for the transformed series in tune_out/var. 4 processes, below-normal priority.
Usage: python var_numpy_features.py <stem> [fam,...]   e.g. train_rev   -> tune_out/var/<fam>_<stem>.parquet"""
import sys, time, pathlib, ctypes, warnings
KU_ENV.low_priority(); warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
HERE = pathlib.Path(__file__).resolve().parent; D = KU_ENV.root(); V = D / "tune_out" / "var"
sys.path[:0] = [str(KU_ENV.code_root() / "clean"), str(HERE.parent / "v5"), str(HERE)]
import numpy as np, pandas as pd
from joblib import Parallel, delayed
import features as CF, features_grid as G1, features_grid2 as G2, features_dd as DD
FN = {"clean": CF.series_features, "grid": G1.one, "grid2": G2.one, "dd": DD.one}
stem = sys.argv[1]; fams = sys.argv[2].split(",") if len(sys.argv) > 2 else list(FN); t0 = time.time()
jobs = [(i, g["value"].to_numpy(float)[g["period"].to_numpy() == 0], g["value"].to_numpy(float)[g["period"].to_numpy() == 1]) for i, g in KU_ENV.read_series(V / f"{stem}_series.parquet").groupby("id", sort=False, observed=True)]
for fam in fams:
    out = V / f"{fam}_{stem}.parquet"
    if out.exists(): print(f"{fam} {stem}: exists, skipped", flush=True); continue
    rows = Parallel(n_jobs=KU_ENV.jobs(), batch_size=32)(delayed(FN[fam])(*j) for j in jobs)
    F = pd.DataFrame(rows).set_index("id").replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32); F.to_parquet(out)
    print(f"{fam} {stem}: {F.shape} ({time.time()-t0:.0f}s)", flush=True)
