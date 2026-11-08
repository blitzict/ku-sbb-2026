"""Build query-only raw inputs for the saved competition ensemble.

No labels or fitted models are used here. Non-hexadecimal caller identifiers
are mapped to stable hexadecimal internal identifiers for the original
feature extractor's deterministic per-series random seed.
"""
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.util
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .common import save_npz, save_table

_feature_function = None


def _initialize_features(source):
    global _feature_function
    spec = importlib.util.spec_from_file_location("prediction_features", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _feature_function = module.series_features


def _feature_row(item):
    ident, values, boundary = item
    return _feature_function(ident, values[:boundary], values[boundary:])


def rows(path):
    table = pq.read_table(path, columns=["id", "time", "value", "period"], read_dictionary=["id"]).to_pandas()
    result = []
    for ident, frame in table.groupby("id", sort=False, observed=True):
        frame = frame.sort_values("time")
        values = frame.value.to_numpy(dtype=float)
        period = frame.period.to_numpy()
        boundary = int(np.count_nonzero(period == 0))
        if (not np.isfinite(values).all() or len(values) > 1024
                or boundary < 17 or len(values) - boundary < 17
                or not np.all(period[:boundary] == 0) or not np.all(period[boundary:] == 1)
                or not np.array_equal(frame.time.to_numpy(), np.arange(len(values)))
                or values[:boundary].std() == 0 or values[boundary:].std() == 0):
            raise ValueError(f"Series {ident!r} must follow the competition format: at most 1024 finite points, "
                             "times 0..n-1, and nonconstant before/after segments of at least 17 points")
        result.append((str(ident), values, boundary))
    if not result:
        raise ValueError("No input series")
    return result


def foundation(model_root, work, series, workers=2):
    """Generate new-ID mapping, the raw test file, sequence cache and 977 features."""
    import ku_workspace
    model_root, work = Path(model_root), Path(work)
    original = rows(series)
    training_ids = set(pd.read_csv(model_root / "data/train_labels.csv", usecols=["id"], dtype=str).id)
    mapping, mapped, seen = [], [], set()
    for ident, values, boundary in original:
        internal = ident if re.fullmatch(r"[0-9a-fA-F]{8}.*", ident) and ident not in training_ids else hashlib.sha256(("prediction:" + ident).encode()).hexdigest()
        if internal in seen or internal in training_ids:
            raise ValueError("Internal query identifier collision")
        seen.add(internal)
        mapping.append(dict(id=ident, internal_id=internal))
        mapped.append((internal, values, boundary))
    data = work / "data"
    data.mkdir(parents=True, exist_ok=True)
    writer = ku_workspace.SeriesWriter(data / "test_series.parquet")
    for ident, values, boundary in mapped:
        writer.add(ident, values, boundary)
    writer.close()
    pd.DataFrame({"id": [x[0] for x in mapped], "break_probability": 0.0}).to_csv(data / "sample_submission.csv", index=False)
    (work / "query_ids.json").write_text(json.dumps(mapping, indent=2))
    ids = np.asarray([x[0] for x in mapped], dtype=str)
    n = np.asarray([len(x[1]) for x in mapped], dtype=np.int64)
    b = np.asarray([x[2] for x in mapped], dtype=np.int64)
    padded = np.zeros((len(ids), 1024), dtype=np.float32)
    for i, (_, values, _) in enumerate(mapped):
        padded[i, :len(values)] = values
    save_npz(work / "tune_out/seq_cache_test.npz", ids=ids, x=padded, n=n, b=b)
    with ProcessPoolExecutor(max_workers=workers, initializer=_initialize_features,
                             initargs=(str(work / "clean/features.py"),)) as pool:
        features = list(pool.map(_feature_row, mapped, chunksize=8))
    frame = pd.DataFrame(features).set_index("id").replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32)
    if not np.array_equal(frame.index, ids) or len(frame.columns) != 977:
        raise ValueError("Unexpected query feature schema or identifier order")
    save_table(work / "clean/features_test.parquet", frame)
    return ids


def variants(work, crop_specs):
    """Use the same transformations and test RNG streams as training's recipe."""
    import ku_workspace
    work = Path(work)
    source = rows(work / "data/test_series.parquet")
    folder = work / "tune_out/var"
    folder.mkdir(parents=True, exist_ok=True)
    writers = {kind: ku_workspace.SeriesWriter(folder / f"test_{kind}_series.parquet")
               for kind in ("rev", "flip", "revflip")}
    for ident, values, boundary in source:
        for kind, result, split, suffix in (
            ("rev", values[::-1], len(values) - boundary, "r"),
            ("flip", -values, boundary, "f"),
            ("revflip", -values[::-1], len(values) - boundary, "rf"),
        ):
            writers[kind].add(ident + suffix, result, split)
    for writer in writers.values():
        writer.close()
    for stem, seed, low, high in crop_specs:
        rng = np.random.default_rng(seed + 1000)  # The historical test split uses offset 1000.
        writer = ku_workspace.SeriesWriter(folder / f"test_{stem}_series.parquet")
        for ident, values, boundary in source:
            before, after = values[:boundary], values[boundary:]
            left = max(min(len(before), 80), int(round(len(before) * (1 - rng.uniform(low, high)))))
            right = max(min(len(after), 80), int(round(len(after) * (1 - rng.uniform(low, high)))))
            writer.add(ident + stem, np.concatenate([before[-left:], after[:right]]), left)
        writer.close()


def generator(model_root, work, batch=128):
    """Keep the saved training reference and append only newly measured query rows."""
    import generator_features as module
    import torch
    from threadpoolctl import threadpool_limits
    model_root, work = Path(model_root), Path(work)
    original = pd.read_parquet(model_root / "tune_out/generator_segments.parquet")
    original = original.loc[original["train"].astype(bool)].copy()
    frames, segments, metadata = [], [], []

    def flush():
        if not segments:
            return
        info = pd.DataFrame(metadata, columns=["id", "part", "n_pre", "n_post"])
        info["train"], info["label"] = False, -1
        frames.append(pd.concat([info, module.measure(segments)], axis=1))
        segments.clear()
        metadata.clear()

    with torch.inference_mode(), threadpool_limits(1):
        torch.set_num_threads(1)
        for ident, values, boundary in rows(work / "data/test_series.parquet"):
            values = values.astype(np.float32)
            for side, sample in (("pre", values[:boundary]), ("post", values[boundary:])):
                segments.append(sample)
                metadata.append((ident, side, boundary, len(values) - boundary))
                if len(segments) == batch:
                    flush()
        flush()
    combined = pd.concat([original, *frames], ignore_index=True)
    save_table(work / "tune_out/generator_segments.parquet", combined)
    save_table(work / "tune_out/generator_deltas.parquet", module.deltas(combined))
