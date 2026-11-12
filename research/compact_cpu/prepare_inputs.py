"""Rebuild training_inputs.npz from the original project's detector archives.

The portable trainer only needs train.py and this NPZ. Run this optional builder
in the original project environment if you need to trace or regenerate the data.
It calls the confirmed feature code on one CPU thread. It never reads the final
model's weights, predictions, replay inputs, historical miss list or test labels.
"""
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "experiments" / "weight_space"))
from white_opinion_spectrum import NAME as PARENT, inputs
from white_shared_residual import MEMBERS, chart
from white_joint import read_member
from expert_fusion import sha


def build(destination):
    cache = ROOT / "tune_out" / "weight_space" / ("frozen_" + PARENT)
    parent_hash = sha(ROOT / "new weights" / f"wn_{PARENT}_candidate" / "weights.json")
    with np.load(cache / "full.npz") as full:
        ids = pd.Index(full["train_ids"])
        test_ids = pd.Index(full["test_ids"])
    labels = pd.read_csv(ROOT / "data/train_labels.csv", dtype={"id": str}).set_index("id")
    order = pd.read_csv(ROOT / "data/sample_submission.csv", dtype={"id": str}).id
    if not (ids.is_unique and test_ids.is_unique and labels.index.is_unique and order.is_unique
            and set(ids) == set(labels.index) and set(test_ids) == set(order)):
        raise ValueError("Source IDs do not match the official label and submission files.")
    y = labels.loc[ids, "label"].to_numpy()
    banks, regimes, _ = inputs(ids, test_ids)
    paired = np.column_stack([read_member(name, ids, test_ids)[0][:, 0] for name in MEMBERS])
    data = dict(ids=ids.to_numpy(dtype=str), test_ids=test_ids.to_numpy(dtype=str),
                submission_order=test_ids.get_indexer(order), labels=y)
    folds = StratifiedKFold(5, shuffle=True, random_state=7).split(ids, y)
    # Each fold has its own basis. Combining rows from different bases before
    # training would silently change the model and contaminate the validation.
    splits = [(f"fold{k}", cache / f"s7_f{k}.npz", tr, va, "validation_ids", "valid")
              for k, (tr, va) in enumerate(folds)]
    splits.append(("full", cache / "full.npz", np.arange(len(y)),
                   len(y) + np.arange(len(test_ids)), "test_ids", "test"))
    all_ids = ids.append(test_ids)
    for name, path, tr, query, id_key, score_key in splits:
        with np.load(path) as saved:
            if (str(saved["parent_sha"]) != parent_hash or
                not np.array_equal(saved["train_ids"], all_ids[tr]) or
                not np.array_equal(saved[id_key], all_ids[query])):
                raise ValueError(f"Unexpected parent weights or row order in {path.name}")
            train_base, query_base = saved["oof_train"], saved[score_key]
        design = chart(banks, regimes, paired, tr, query, train_base, query_base)["response"]
        data.update({f"{name}_features": design, f"{name}_base": np.r_[train_base, query_base],
                     f"{name}_train": tr, f"{name}_query": query})
        print(f"{name}: {len(tr):,} training rows, {len(query):,} prediction rows, "
              f"{design.shape[1]} features", flush=True)
    with TemporaryDirectory(dir=destination.parent) as tmp:
        new = Path(tmp) / destination.name
        np.savez_compressed(new, **data)
        new.replace(destination)
    print(f"Saved {destination} ({destination.stat().st_size / 1024**2:.1f} MiB)")


if __name__ == "__main__":
    with threadpool_limits(limits=1):
        build(HERE / "training_inputs.npz")
