"""write the designated kaggle submission 58102347 (0.88078 public).

python try4/designated_088078/predict.py --data data --out submission.csv

#the final layer of the 0.88078 model: p = sigmoid(parent_logit + features @ weights)
#parent_logit and the 5 correction features come from model/mixed_replay.npz (saved detector outputs)
#the neural part of those features can be recomputed from raw series with member_cpu.py
"""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=HERE.parents[1] / "data")
    p.add_argument("--out", type=Path, default=Path("submission.csv"))
    a = p.parse_args()
    #load the saved final-layer inputs and weights
    with np.load(HERE / "model/mixed_replay.npz", allow_pickle=False) as z:
        ids, prob = z["test_ids"].astype(str), expit(z["parent_logits"] + z["features"] @ z["weights"])
    #order like the sample submission and write id,break_probability
    order = pd.read_csv(a.data / "sample_submission.csv", dtype={"id": str}).id
    out = pd.Series(prob, index=ids).loc[order]
    pd.DataFrame({"id": order, "break_probability": out.to_numpy()}).to_csv(a.out, index=False, lineterminator="\r\n")
    #check against the csv that was submitted to kaggle
    ref = pd.read_csv(HERE / "model/submission.csv", dtype={"id": str}).set_index("id").loc[order, "break_probability"]
    diff = float(np.abs(ref.to_numpy() - out.to_numpy()).max())
    print(f"wrote {a.out}, {len(out)} rows, max diff vs submitted csv {diff:.2e}, sha256 {hashlib.sha256(a.out.read_bytes()).hexdigest()}")
    if diff > 1e-6:
        raise SystemExit("does not match the submitted csv within 1e-6")


if __name__ == "__main__":
    main()
