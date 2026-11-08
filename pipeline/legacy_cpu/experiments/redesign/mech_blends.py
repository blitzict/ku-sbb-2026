"""Fixed blends that originally existed only as inline commands. Every recipe was proven against archived arrays.

Runs after the member models. Reads tune_out/{oof,test}_<member>.npy and ver_<tag>.npz, writes tune_out/{oof,test}_<blend>.npy.
Usage: python mech_blends.py [name ...]    no names: every blend whose inputs exist
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import numpy as np
from scipy.special import expit, logit
import ku_workspace as KU_ENV

O = KU_ENV.root() / "tune_out"
SEQ = ("W_SEQ_q", "W_SEQ_q_s1", "W_SEQ_qh", "W_SEQ_q_s2", "W_SEQ_qh_s1")
LOGIT = {"MECH3C12V_bag": [f"MECH3C12V_s{s}" for s in range(3)], "MECH3C1_bag": [f"MECH3C1_s{s}" for s in range(3)], "MECHNEST2_bag": [f"MECHNEST2_s{s}" for s in range(3)],
         "MECHNEST3_bag": [f"MECHNEST3_s{s}" for s in range(3)], "W_SEQ_qb5": list(SEQ)}
PROB = {f"L4_fam_{f}_b2": [f"L4_fam_{f}", f"L4_fam_{f}_s1"] for f in ("gh", "dd", "lr", "grid")}
PROB.update({"VAR_aug1600_avg_tta": ["VAR_aug1600_tta", "VAR_aug1600_s1_tta"], "VAR_full_aug_avg_tta": ["VAR_full_aug_tta", "VAR_full_aug_s1_tta"]})
VIEWS = {"MECH3C12RV_origonly": ("MECH3C12RV_s0", {"orig": 1}), "MECH3C12RV_revlow": ("MECH3C12RV_s0", {"orig": 2, "c1": .125, "c2": .125, "rev": .25})}
lg = lambda a: logit(np.clip(a.astype(float), 1e-6, 1 - 1e-6))


def inputs(name):
    return [O / f"ver_{VIEWS[name][0]}.npz"] if name in VIEWS else [O / f"{p}_{m}.npy" for m in {**LOGIT, **PROB}[name] for p in ("oof", "test")]


def make(name):
    for part in ("oof", "test"):
        if name in VIEWS:
            z = np.load(O / f"ver_{VIEWS[name][0]}.npz"); wt = VIEWS[name][1]; W = np.array([wt.get(k, 0) for k in z["kinds"]], float)
            out = expit((W[:, None] * lg(z[part])).sum(0) / W.sum())
        else:
            A = np.stack([np.load(O / f"{part}_{m}.npy") for m in {**LOGIT, **PROB}[name]])
            out = A.mean(0) if name in PROB else expit(lg(A).mean(0))
        np.save(O / f"{part}_{name}.npy", out)


if __name__ == "__main__":
    names = sys.argv[1:] or [n for n in (*LOGIT, *PROB, *VIEWS) if all(p.exists() for p in inputs(n))]
    for n in names: make(n); print("wrote", n, flush=True)
