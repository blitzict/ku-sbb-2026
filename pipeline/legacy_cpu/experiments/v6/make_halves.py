# CPU port of experiments/v6/make_halves.py, source sha256 7171eb466cd45a9b4dba02145a1cef5d5ab404fbdad42f720997599cb6f84e39. See auc_lab_20261107/claude/audit/cpu_port_notes.md.
import sys as _s, pathlib as _p; _s.path.insert(0, str(_p.Path(__file__).resolve().parents[2])); import ku_workspace as KU_ENV
"""Length-matched segment rows (2026-11-03). Real data only, nothing generated:
  negatives  h0 / h1: the before (h0) or after (h1) segment of ANY training series on its own, with a pseudo-boundary at U(0.3, 0.7)
             of its length (both sides >= MINS). A single segment of the generator is break-free (one intervention, at the real boundary).
  positives  pK: a crop of a label-1 series around its REAL boundary to the same length as a negative (before share U(0.3, 0.7)),
             one per negative, so short rows are 50/50 and length carries no label information.
Every row carries its source id (fold isolation). Writes tune_out/var/train_hv_series.parquet + tune_out/var/hv_meta.csv.
Usage: python make_halves.py [seed]"""
import sys, pathlib, numpy as np, pandas as pd
D = KU_ENV.root(); V = D / "tune_out" / "var"
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 5; STEM = sys.argv[2] if len(sys.argv) > 2 else "hv"; SRC = sys.argv[3] if len(sys.argv) > 3 else "train"; rng = np.random.default_rng(seed); MINS = 48; MINLEN = 160
s = KU_ENV.read_series(D / "data" / "train_series.parquet"); y = pd.read_csv(D / "data/train_labels.csv", dtype={"id": str}).set_index("id").label
ser = {i: (g["value"].to_numpy(np.float64), int(g["period"].to_numpy().argmax())) for i, g in s.groupby("id", sort=False, observed=True)}
pos = [i for i in ser if y[i] == 1]; WR = KU_ENV.SeriesWriter(V / f"train_{STEM}_series.parquet"); meta = []; LEN = []
segsrc = ser if SRC == "train" else {i: (g["value"].to_numpy(np.float64), int(g["period"].to_numpy().argmax())) for i, g in KU_ENV.read_series(D / "data" / "test_series.parquet").groupby("id", sort=False, observed=True)}
# SRC=test: the negatives are single segments of TEST series (break-free by construction, no labels used); positives stay crops of TRAIN breaks


def add(rid, vv, k, src, lab, kind):
    WR.add(rid, vv, k); LEN.append(len(vv)); meta.append((rid, src, lab, kind))


npc = {}
for i, (v, b) in segsrc.items():
    for h, seg in (("h0", v[:b]), ("h1", v[b:])):
        L = len(seg)
        if L < MINLEN: continue
        k = int(round(L * rng.uniform(0.3, 0.7))); k = min(max(k, MINS), L - MINS); add(i + h, seg, k, i, 0, h)
        for _ in range(50):                                          # matched positive crop
            j = pos[rng.integers(len(pos))]; vj, bj = ser[j]; kb = int(round(L * rng.uniform(0.3, 0.7))); kb = min(max(kb, MINS), L - MINS)
            if kb <= bj and L - kb <= len(vj) - bj:
                c = npc.get(j, 0); npc[j] = c + 1; add(f"{j}p{c}", vj[bj - kb:bj + L - kb], kb, j, 1, "p"); break
WR.close()
M = pd.DataFrame(meta, columns=["id", "source", "label", "kind"]); M.to_csv(V / f"{STEM}_meta.csv", index=False)
print(M.kind.value_counts().to_dict(), "label rate", round(M.label.mean(), 3), "lengths", pd.Series(LEN).describe()[["min", "50%", "max"]].to_dict())
