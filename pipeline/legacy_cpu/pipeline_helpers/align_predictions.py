"""Package newly trained predictions with their verified source identifiers.

The historical white_import_tree read IDs from a later head's cache. Only the
IDs were used. Read the same order directly from the fresh feature matrix so
specialist training does not depend on an unrelated later fitted head.
"""
import argparse
import numpy as np
import pandas as pd
import ku_workspace as env


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("tag")
    p.add_argument("name")
    a = p.parse_args()
    root = env.root()
    with np.load(root / "tune_out/var/mat_ns3_plain.npz", allow_pickle=True) as mat:
        ids, tids = mat["ids"].astype(str), mat["tid"].astype(str)
    for part, expected in (("train", ids), ("test", tids)):
        actual = pd.read_parquet(root / f"clean/features_{part}.parquet", columns=[]).index.to_numpy(dtype=str)
        if not np.array_equal(actual, expected):
            raise ValueError(f"Fresh matrix {part} ID order differs from raw feature order")
    values = dict(ids=ids, tids=tids)
    for part, expected in (("oof", ids), ("test", tids)):
        score = np.load(root / f"tune_out/{part}_{a.tag}.npy")
        if score.shape != expected.shape or not np.isfinite(score).all() or not ((score > 0) & (score < 1)).all():
            raise ValueError(f"Invalid {part} probability array for {a.tag}")
        values[part] = score
    out = root / "tune_out/weight_space" / (a.name + ".npz")
    if out.exists():
        with np.load(out) as previous:
            if not all(np.array_equal(previous[k], v) for k, v in values.items()):
                raise ValueError(f"Existing aligned predictions differ: {out}")
    else:
        np.savez_compressed(out, **values)
    print(f"Aligned {a.tag} as {a.name}: {len(ids)} train, {len(tids)} test")


if __name__ == "__main__":
    main()
