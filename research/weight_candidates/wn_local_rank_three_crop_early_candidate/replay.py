"""Rebuild the official test predictions from the stored output weights and inputs."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from threadpoolctl import threadpool_limits


def main(output):
    folder = Path(__file__).resolve().parent
    check = json.loads((folder/"validation.json").read_text())
    recipe = folder/"weights.json"
    assert hashlib.sha256(recipe.read_bytes()).hexdigest()==check["recipe_sha256"]
    theta = np.array(json.loads(recipe.read_text())["weights"])
    with np.load(folder/"replay_inputs.npz") as inputs:
        p = expit(inputs["parent_logits"]+inputs["features"]@theta)
        order = inputs["order"]
        assert np.array_equal(np.sort(order),np.arange(4000))
        sub = pd.DataFrame({"id":inputs["ids"][order],"break_probability":p[order]})
    assert sub.id.is_unique and np.isfinite(p).all() and (p>=0).all() and (p<=1).all()
    sub.to_csv(output,index=False,lineterminator="\r\n")
    assert hashlib.sha256(output.read_bytes()).hexdigest()==check["submission_sha256"]
    print("Verified 4,000 predictions",flush=True)


if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output",type=Path,default=Path(__file__).with_name("submission_replay.csv"))
    with threadpool_limits(limits=2):
        main(p.parse_args().output)
