# CPU port of experiments/redesign/harm_aug.py, source sha256 3e9dd4fb2d52c7802267cbf9afa56d06463e61d99b261a5f89fd1c17549c6562. See auc_lab_20261107/claude/audit/cpu_port_notes.md.
import sys as _s, pathlib as _p; _s.path.insert(0, str(_p.Path(__file__).resolve().parents[2])); import ku_workspace as KU_ENV
"""Harmonic block for the boundary-shift negatives (same jobs as features_lr build_jobs('aug')) -> tune_out/aug_harm_features.parquet."""
import sys, time, pathlib, ctypes, warnings; warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
try: ctypes.windll.kernel32.SetPriorityClass(ctypes.c_void_p(-1), 0x4000)
except Exception: pass
D = KU_ENV.root(); sys.path.insert(0, str(KU_ENV.experiments() / "v6")); sys.path.insert(0, str(KU_ENV.experiments() / "redesign")); sys.path.insert(0, str(KU_ENV.experiments() / "weight_space"))
import numpy as np, pandas as pd
from joblib import Parallel, delayed
if __name__ == "__main__":
    sys.argv = ["features_lr.py", "aug"]; from features_lr import build_jobs
    import harm_var as hv, harmonic_change as hc
    jobs, _ = build_jobs("aug"); t0 = time.time()
    rows = Parallel(n_jobs=4, batch_size=64)(delayed(hv.one)(np.asarray(v, np.float32).astype(float), k) for _, v, k in jobs)
    tab = pd.DataFrame(rows, index=[j[0] for j in jobs], columns=[s + "_" + k for s in ["raw", "ar1"] for k in hc.NAMES])
    tab.index.name = "id"; tab.to_parquet(D / "tune_out/aug_harm_features.parquet"); print("aug", tab.shape, f"{time.time()-t0:.0f}s", flush=True)
