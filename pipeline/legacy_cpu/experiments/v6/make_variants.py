# CPU port of experiments/v6/make_variants.py, source sha256 7a527ba2a9adf9fa18e1c228dbd260287b0594144216cc73128f746caff18ff1. See auc_lab_20261107/claude/audit/cpu_port_notes.md.
import sys as _s, pathlib as _p; _s.path.insert(0, str(_p.Path(__file__).resolve().parents[2])); import ku_workspace as KU_ENV
"""Label-preserving transforms of the real series (user-approved 2026-11-02): rev = time-reversed (boundary b -> n-b), flip = negated values,
revflip = both. Writes tune_out/var/<split>_<kind>_series.parquet in the competition format (id, time, value, period); ids get suffix r / f / rf
(the hex prefix is kept so clean/features.py seeds stay valid)."""
import pathlib, numpy as np, pandas as pd
D = KU_ENV.root(); V = D / "tune_out" / "var"
for split in ("train", "test"):
    s = KU_ENV.read_series(D / "data" / f"{split}_series.parquet"); w = {k: KU_ENV.SeriesWriter(V / f"{split}_{k}_series.parquet") for k in ("rev", "flip", "revflip")}
    for ident, g in s.groupby("id", sort=False, observed=True):
        v = g["value"].to_numpy(np.float64); b = int(g["period"].to_numpy().argmax()); n = len(v)
        for kind, vv, bb, suf in (("rev", v[::-1], n - b, "r"), ("flip", -v, b, "f"), ("revflip", -v[::-1], n - b, "rf")): w[kind].add(ident + suf, vv, bb)
    for kind, wr in w.items(): print(split, kind, wr.close()[0], "series")
