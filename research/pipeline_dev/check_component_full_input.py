"""Check identical batch results with one and three workers on real inputs.

Only 135 rows are calculated. Loading the full stream arrays exercises the
same input-memory requirement as the production stage; no model is fitted.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
from time import monotonic

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "cpu_088_rebuild/work"


def main():
    env = os.environ.copy()
    env.update(KU_SBB_ROOT=str(WORK), KU_SBB_CODE_ROOT=str(WORK), KU_SBB_RAW_ONLY="1",
               KU_SBB_ORIGINAL_ROOT=str(ROOT), KU_SBB_OFFLINE="1", CUDA_VISIBLE_DEVICES="",
               ONEAPI_DEVICE_SELECTOR="*:cpu", KU_SBB_THREADS="1")
    env["PYTHONPATH"] = os.pathsep.join((str(ROOT / "pipeline/offline_hooks"), str(WORK)))
    times = {}
    for workers in (1, 3):
        started = monotonic()
        subprocess.run([sys.executable, "-B", str(ROOT / "pipeline/legacy_cpu/pipeline_helpers/component_moves_cpu.py"),
                        "state", "train", "--workers", str(workers), "--limit", "135",
                        "--output", f"tune_out/component_thread_checks/full_input_{workers}.parquet"],
                       cwd=WORK, env=env, check=True)
        times[str(workers)] = monotonic() - started
    frames = [pd.read_parquet(WORK / f"tune_out/component_thread_checks/full_input_{w}.parquet") for w in (1, 3)]
    if not frames[0].index.equals(frames[1].index) or not frames[0].columns.equals(frames[1].columns):
        raise AssertionError("Row order or feature names changed")
    a, b = [np.ascontiguousarray(frame.to_numpy()) for frame in frames]
    if a.dtype != b.dtype or not np.array_equal(a.view(np.uint8), b.view(np.uint8)):
        raise AssertionError(f"Values changed by {np.max(abs(a - b))}")
    report = dict(rows=len(a), columns=a.shape[1], bit_exact=True, maximum_difference=0.0, seconds=times)
    (ROOT / "cpu_088_rebuild/component_full_input_equivalence.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
