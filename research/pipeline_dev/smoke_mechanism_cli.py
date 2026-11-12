"""Run complete physical-feature commands on a fresh 16-series raw subset."""
import json

import numpy as np
import pandas as pd

from pipeline.build_foundation import ROOT, initialize, build
from pipeline.sync_ports import sync
from pipeline.prepare_feature_stages import build as feature_stages
from pipeline.prepare_mechanism_stages import build as mechanism_stages
from pipeline.run_stages import run


def main():
    work = ROOT / "cpu_088_rebuild/mechanism_cli_smoke"
    data = ROOT / "cpu_088_rebuild/cli_smoke_inputs"
    initialize(work, data)
    sync(work)
    for part in ("train", "test"):
        build(work, part, 2, 16)
    for stage in feature_stages()["stages"]:
        if stage["argv"][-1:] == ["lr5"] and stage["argv"][1] in ("train", "test"):
            run(work, stage)
    chosen = mechanism_stages()["stages"]
    for stage in chosen:
        run(work, stage)
        for output in stage["outputs"]:
            path = work / output
            if path.suffix == ".parquet":
                frame = pd.read_parquet(path)
                if len(frame) != 16 or not np.isfinite(frame.to_numpy()).all():
                    raise AssertionError(f"Invalid physical feature output: {path}")
            else:
                with np.load(path) as values:
                    if len(values["ids"]) != 16:
                        raise AssertionError(f"Wrong physical feature row count: {path}")
                    for key in values:
                        array = values[key]
                        if np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all():
                            raise AssertionError(f"Non-finite array {key}: {path}")
    (work / "stages/mechanism_cli_smoke.json").write_text(json.dumps(dict(
        passed=len(chosen), rows_per_split=16, score_validation=False), indent=2))
    print(f"PASS: {len(chosen)} physical-feature commands on fresh raw subsets")


if __name__ == "__main__":
    main()
