# CPU port of experiments/redesign/harm_var.py, source sha256 3eab9aaeea45a72eaefecc829ee5c45daa101fa213273ad0ba36657feb3fa6cc. See auc_lab_20261107/claude/audit/cpu_port_notes.md.
import sys as _s, pathlib as _p; _s.path.insert(0, str(_p.Path(__file__).resolve().parents[2])); import ku_workspace as KU_ENV
"""Harmonic block (Codex's experiments/weight_space/harmonic_change.score, imported unchanged) for the transformed views in tune_out/var
(rev, flip, revflip, c1..c4, hv rows, ...). Output tune_out/var/harm_<split>_<kind>.parquet, same 36 columns as weight_space/harmonic_*.parquet.
CPU, below-normal priority, joblib processes. Usage: python harm_var.py train_rev [n_jobs]"""
import sys, time, pathlib, ctypes, warnings; warnings.filterwarnings("ignore"); sys.stdout.reconfigure(encoding="utf-8")
try: ctypes.windll.kernel32.SetPriorityClass(ctypes.c_void_p(-1), 0x4000)
except Exception: pass
D = KU_ENV.root(); sys.path.insert(0, str(KU_ENV.experiments() / "weight_space"))
import numpy as np, pandas as pd
from joblib import Parallel, delayed
stem = sys.argv[1] if len(sys.argv) > 1 else ""; NJ = int(sys.argv[2]) if len(sys.argv) > 2 else 4; t0 = time.time(); V = D / "tune_out" / "var"


def one(v, b):
    import harmonic_change as hc
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        l, r = v[:b], v[b:]; pre, post = l - l.mean(), r - r.mean()
        den = pre[:-1] @ pre[:-1] + post[:-1] @ post[:-1]
        phi = np.clip((pre[:-1] @ pre[1:] + post[:-1] @ post[1:]) / max(den, 1e-10), -.99, .99)
        ea = l[1:] - phi * l[:-1]; eb = r - phi * np.r_[l[-1], r[:-1]]
        return hc.score(l, r) + hc.score(ea, eb)


def init_priority():
    try: ctypes.windll.kernel32.SetPriorityClass(ctypes.c_void_p(-1), 0x4000)
    except Exception: pass


if __name__ == "__main__":
    import harmonic_change as hc
    lim = int(sys.argv[3]) if len(sys.argv) > 3 else None
    jobs = [(i, g["value"].to_numpy(np.float32).astype(float), int(g["period"].to_numpy().argmax())) for i, g in KU_ENV.read_series(V / f"{stem}_series.parquet").groupby("id", sort=False, observed=True)][:lim]
    rows = Parallel(n_jobs=NJ, batch_size=64)(delayed(one)(v, b) for _, v, b in jobs)
    cols = [s + "_" + k for s in ["raw", "ar1"] for k in hc.NAMES]
    tab = pd.DataFrame(rows, index=[j[0] for j in jobs], columns=cols).astype(np.float64)
    if lim is None: tab.to_parquet(V / f"harm_{stem}.parquet")
    print(f"{stem}: {tab.shape} in {time.time()-t0:.0f}s", flush=True)
