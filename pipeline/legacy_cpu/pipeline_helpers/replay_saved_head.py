"""Replay a refitted output layer from its fresh, saved detector inputs.

This checks the output layer only. Raw-data feature generation and model
inference must run first; this file is not the full raw-data inference path.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit


def replay(folder, output):
    recipe = folder / "weights.json"
    validation = json.loads((folder / "validation.json").read_text())
    if hashlib.sha256(recipe.read_bytes()).hexdigest() != validation["recipe_sha256"]:
        raise ValueError("Output-layer weights changed")
    weights = np.asarray(json.loads(recipe.read_text())["weights"])
    with np.load(folder / "replay_inputs.npz") as data:
        scores = expit(data["parent_logits"] + data["features"] @ weights)
        ids, order = data["ids"], data["order"]
    if not np.array_equal(np.sort(order), np.arange(len(ids))) or not np.isfinite(scores).all():
        raise ValueError("Invalid replay inputs or order")
    result = pd.DataFrame({"id": ids[order], "break_probability": scores[order]})
    if not result.id.is_unique:
        raise ValueError("Duplicate identifiers")
    # Match finalize.write_submission's native-platform newline convention.
    result.to_csv(output, index=False)
    if hashlib.sha256(output.read_bytes()).hexdigest() != validation["submission_sha256"]:
        raise ValueError("Replay CSV differs from the trained output")
    print(f"Verified {len(result)} output-layer predictions", flush=True)


def package(folder):
    """Package this check beside a newly fitted head with existing replay inputs."""
    import shutil
    validation = folder / "validation.json"
    report = json.loads(validation.read_text()) if validation.exists() else {}
    report.update(recipe_sha256=hashlib.sha256((folder / "weights.json").read_bytes()).hexdigest(),
                  submission_sha256=hashlib.sha256((folder / "submission.csv").read_bytes()).hexdigest(),
                  scope="Saved output-layer inputs; not standalone raw-data inference")
    validation.write_text(json.dumps(report, indent=2))
    shutil.copy2(Path(__file__), folder / "replay.py")
    replay(folder, folder / "submission_replay.csv")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path(__file__).with_name("submission_replay.csv"))
    replay(Path(__file__).resolve().parent, p.parse_args().output)
