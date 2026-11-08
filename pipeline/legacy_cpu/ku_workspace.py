"""Workspace rules for the CPU ports.

Every port writes under KU_SBB_ROOT. It must be a fresh rebuild workspace holding CPU_REBUILD_WORKSPACE.json, so a stray run cannot overwrite the
historical tune_out or weights folders.
"""
import ctypes, os, pathlib


def root():
    value = os.environ.get("KU_SBB_ROOT")
    if not value: raise RuntimeError("Set KU_SBB_ROOT to the fresh CPU rebuild workspace")
    path = pathlib.Path(value).resolve()
    if not (path / "CPU_REBUILD_WORKSPACE.json").is_file(): raise RuntimeError(f"{path} has no CPU_REBUILD_WORKSPACE.json marker")
    return path


def code_root():
    """Folder holding clean/features.py. KU_SBB_CODE_ROOT overrides it, for example in a portable snapshot."""
    return pathlib.Path(os.environ.get("KU_SBB_CODE_ROOT") or pathlib.Path(__file__).resolve().parents[2])


def experiments():
    """Folder holding the ported v5, v6, redesign and weight_space code."""
    base = code_root()
    return base / "experiments" if (base / "experiments/v6/train_cv.py").is_file() and "legacy_cpu" in str(base) or os.environ.get("KU_SBB_CODE_ROOT") else base / "pipeline/legacy_cpu/experiments"


def threads(default=4):
    return int(os.environ.get("KU_SBB_THREADS", default))


def low_priority():
    try: ctypes.windll.kernel32.SetPriorityClass(ctypes.c_void_p(-1), 0x4000)
    except (AttributeError, OSError): pass


def jobs():
    """Worker count. Default min(2, threads) for the 8 GB inference target.

    KU_SBB_WORKERS opts in to more training workers, bounded 1..6. It changes speed only, not features, hyperparameters or random streams.
    """
    w = os.environ.get("KU_SBB_WORKERS")
    return max(1, min(6, int(w))) if w else min(2, threads())


def diag_weights(tune_out):
    """Optional weights for a printed diagnostic AUC. None when absent; fitting and the main AUC never use them."""
    import numpy as np
    p = pathlib.Path(tune_out) / "adv_ptest_train.npy"
    if not p.is_file(): return None
    t = np.load(p); w = t / np.clip(1 - t, 1e-3, None); return w / w.mean()


def read_series(path):
    """Competition-format series with dictionary-encoded ids, far smaller than object strings. Row order is unchanged."""
    import pyarrow.parquet as pq
    return pq.read_table(path, read_dictionary=["id"]).to_pandas()


class SeriesWriter:
    """Stream competition-format rows to parquet in row groups, keeping row order."""

    def __init__(self, path, every=2000):
        import pyarrow as pa, pyarrow.parquet as pq
        self.pa, self.schema = pa, pa.schema([("id", pa.string()), ("time", pa.int64()), ("value", pa.float64()), ("period", pa.int8())])
        self.w = pq.ParquetWriter(path, self.schema); self.every, self.buf, self.ids, self.rows = every, [], 0, 0

    def add(self, ident, value, boundary):
        import numpy as np
        t = np.arange(len(value), dtype=np.int64)
        self.buf.append((ident, t, np.asarray(value, np.float64), (t >= boundary).astype(np.int8))); self.ids += 1; self.rows += len(t)
        if len(self.buf) >= self.every: self.flush()

    def flush(self):
        import numpy as np
        if self.buf:
            a = self.pa
            self.w.write_table(a.table([a.array(np.repeat([b[0] for b in self.buf], [len(b[1]) for b in self.buf]).tolist(), a.string()), a.array(np.concatenate([b[1] for b in self.buf])), a.array(np.concatenate([b[2] for b in self.buf])), a.array(np.concatenate([b[3] for b in self.buf]))], schema=self.schema))
        self.buf = []

    def close(self):
        self.flush(); self.w.close(); return self.ids, self.rows


def t_constant(df, like):
    """lgamma((df+1)/2) - lgamma(df/2) - log(df*pi)/2 in float64, cast to the type of `like`.

    In float32 at df=1e4 the two lgamma values are near 37,000 and their difference is wrong by about 0.003 per observation, which is enough to pick the wrong law.
    """
    import math, torch
    d = torch.as_tensor(df, dtype=torch.float64, device=like.device)
    return (torch.lgamma((d + 1) / 2) - torch.lgamma(d / 2) - 0.5 * torch.log(d * math.pi)).to(like.dtype)


def check_xgb(path, booster, dmatrix):
    """Reload a saved booster and stop if it does not reproduce the live one."""
    import numpy as np, xgboost as xgb
    b = xgb.Booster(); b.load_model(str(path))
    if float(np.abs(b.predict(dmatrix, output_margin=True) - booster.predict(dmatrix, output_margin=True)).max()) > 1e-6: raise RuntimeError(f"saved booster differs: {path}")


def rank_reference(path, rows):
    """Full sorted training columns, compressed. Exact inference needs all of them; NaN sorts last, as in the live transform."""
    import numpy as np
    np.savez_compressed(path, sorted=np.sort(np.asarray(rows, np.float32), 0))
