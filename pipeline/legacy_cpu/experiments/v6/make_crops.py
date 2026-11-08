# CPU port of experiments/v6/make_crops.py, source sha256 7e0c9f23058cee521818557da4910ca1f18820eb39a71175dd3e86fcb3822cf8. See auc_lab_20261107/claude/audit/cpu_port_notes.md.
import sys as _s, pathlib as _p; _s.path.insert(0, str(_p.Path(__file__).resolve().parents[2])); import ku_workspace as KU_ENV
"""Crop transforms of the real series (user-approved 2026-11-02): trim a random 10-40% off the START of the before segment and, independently,
off the END of the after segment, keeping at least 80 points per side (a shorter side is kept whole). The boundary stays exactly where it was
relative to the kept data, so the label is unchanged; nothing is generated. Writes tune_out/var/<split>_<stem>_series.parquet in the competition
format; ids get suffix 'c<k>' (hex prefix kept). Usage: python make_crops.py c1 [seed]"""
import sys, pathlib, numpy as np, pandas as pd
D = KU_ENV.root(); V = D / "tune_out" / "var"
stem = sys.argv[1] if len(sys.argv) > 1 else "c1"; seed = int(sys.argv[2]) if len(sys.argv) > 2 else 11; MINKEEP = 80
LO = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1; HI = float(sys.argv[4]) if len(sys.argv) > 4 else 0.4
for k, split in enumerate(("train", "test")):
    rng = np.random.default_rng(seed + 1000 * k); s = KU_ENV.read_series(D / "data" / f"{split}_series.parquet"); wr = KU_ENV.SeriesWriter(V / f"{split}_{stem}_series.parquet"); kept = []
    for ident, g in s.groupby("id", sort=False, observed=True):
        v = g["value"].to_numpy(np.float64); p = g["period"].to_numpy(); a, b = v[p == 0], v[p == 1]
        na = max(min(len(a), MINKEEP), int(round(len(a) * (1 - rng.uniform(LO, HI)))))
        nb = max(min(len(b), MINKEEP), int(round(len(b) * (1 - rng.uniform(LO, HI)))))
        vv = np.concatenate([a[len(a) - na:], b[:nb]]); t = np.arange(len(vv)); kept.append(len(vv) / len(v))
        wr.add(ident + stem, vv, na)
    print(split, stem, wr.close()[0], "series, mean kept fraction", round(float(np.mean(kept)), 3))
