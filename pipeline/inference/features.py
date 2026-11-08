"""Test-side feature matrices, assembled exactly as the training scripts assemble their test rows.

All readers take the inference folder (feature_root) and return rows in the order of
feature_root/clean/features_test.parquet. Column lists come from the trained model (saved meta, features.json or
the training-side matrix), never from the new data, and a missing column raises.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from pipeline.inference import source


def rp(path, cols=None):
    return pd.read_parquet(source.require(path), columns=cols).astype(np.float32)


def test_ids(feature_root):
    return pd.read_parquet(source.require(Path(feature_root) / "clean/features_test.parquet"), columns=[]).index


def columns(path):
    """Column names of a parquet file without reading data (pandas index columns excluded)."""
    f = pq.ParquetFile(source.require(path))
    meta = f.schema_arrow.pandas_metadata or {}
    index = {c for c in meta.get("index_columns", []) if isinstance(c, str)}
    return [n for n in f.schema_arrow.names if n not in index]


def finite(df):
    return df.replace([np.inf, -np.inf], np.nan)


def featset_columns(model_root, blocks, rd=False):
    """Training column names of v6/featset.load, from the trained parquet schemas (no data read)."""
    model_root = Path(model_root)
    clean = columns(model_root / "clean/features_train.parquet")
    names = list(clean)
    for b in blocks:
        names += [f"{b}__{c}" for c in columns(model_root / f"tune_out/{b}_train.parquet")]
    if rd:
        stems = [c[:-4] for c in clean if c.endswith("_pre") and (c[:-4] + "_post") in clean]
        names += [f"rd_{s}_{k}" for s in stems for k in ("lr", "ad", "sr")]
    return names


def featset_test(model_root, feature_root, blocks, rd=False):
    """Test frame of v6/featset.load (clean + prefixed blocks + optional rd block), train column names and order."""
    model_root, feature_root = Path(model_root), Path(feature_root)
    clean = columns(model_root / "clean/features_train.parquet")
    Xt = rp(feature_root / "clean/features_test.parquet", clean)
    for b in blocks:
        bc = columns(model_root / f"tune_out/{b}_train.parquet")
        Bt = rp(feature_root / f"tune_out/{b}_test.parquet", bc).loc[Xt.index]
        Bt.columns = [f"{b}__{c}" for c in bc]
        Xt = pd.concat([Xt, Bt], axis=1)
    if rd:
        ns = source.definitions("experiments/v6/featset.py", ["rd_block"], {"np": np, "pd": pd})
        stems = [c[:-4] for c in clean if c.endswith("_pre") and (c[:-4] + "_post") in clean]
        Xt = pd.concat([Xt, ns["rd_block"](Xt, stems)], axis=1)
    return finite(Xt)


def train_cv_test(model_root, feature_root, a):
    """Test frame of v6/train_cv.py before column selection (clean + blocks + optional rd)."""
    if a.contrast:
        raise ValueError("--contrast standardizes with train+test statistics (transductive); no saved form exists")
    if a.cat_desc:
        raise ValueError("--cat-desc members are not part of the recipe; refusing an unverified path")
    blocks = tuple(b for b in a.blocks.split(",") if b)
    return featset_test(model_root, feature_root, blocks, rd=a.rd)


def var_test(feature_root, cols, kind=None):
    """train_var.assemble for test rows: the listed columns, original rows or one transformed version (rows in
    test-ID order; a version's rows are read under id + suffix, as train_var does)."""
    feature_root = Path(feature_root)
    O, V = feature_root / "tune_out", feature_root / "tune_out/var"
    ns = {"COLS": list(cols), "CLEAN": [c for c in cols if "__" not in c], "_C": {"cols": None}, "pd": pd, "np": np}
    source.definitions("experiments/v6/train_var.py", ["BLOCKS", "need", "assemble"], ns)
    suffix = source.definitions("experiments/v6/train_var.py", ["SUF"], {})["SUF"] if kind else None
    rd = lambda p: pd.read_parquet(source.require(p), columns=ns["_C"]["cols"])
    tids = test_ids(feature_root)
    if kind is None:
        X = ns["assemble"](lambda fp: rd(feature_root / "clean/features_test.parquet") if fp == "clean" else rd(O / f"{fp}_test.parquet"))
        return X.loc[tids]
    X = ns["assemble"](lambda fp: rd(V / f"{fp}_test_{kind}.parquet"))
    return X.loc[[i + suffix[kind] for i in tids]]


def matrix_columns(model_root, name):
    """Column list stored in a training-side export matrix (tune_out/var/<name>.npz), read without the arrays."""
    with np.load(source.require(Path(model_root) / f"tune_out/var/{name}.npz"), allow_pickle=True) as z:
        return [str(c) for c in z["cols"]]
