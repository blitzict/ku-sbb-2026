"""Compare concurrent batches with the existing fresh serial feature check."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / "cpu_088_rebuild/feature_graph_smoke"
HELPER = "pipeline_helpers/component_moves_cpu.py"


def main():
    shutil.copy2(ROOT / "pipeline/legacy_cpu" / HELPER, WORK / HELPER)
    env = os.environ.copy()
    env.update(KU_SBB_ROOT=str(WORK), KU_SBB_CODE_ROOT=str(WORK), KU_SBB_RAW_ONLY="1",
               KU_SBB_ORIGINAL_ROOT=str(ROOT), KU_SBB_OFFLINE="1", CUDA_VISIBLE_DEVICES="",
               ONEAPI_DEVICE_SELECTOR="*:cpu", KU_SBB_THREADS="1")
    env["PYTHONPATH"] = os.pathsep.join((str(ROOT / "pipeline/offline_hooks"), str(WORK)))
    reports = []
    for stream in ("state", "pulse"):
        output = f"tune_out/component_thread_checks/{stream}.parquet"
        subprocess.run([sys.executable, "-B", str(WORK / HELPER), stream, "train",
                        "--workers", "2", "--output", output], cwd=WORK, env=env, check=True)
        original = pd.read_parquet(WORK / f"tune_out/weight_space/component_moves_{stream}_train.parquet")
        threaded = pd.read_parquet(WORK / output)
        if not original.index.equals(threaded.index) or not original.columns.equals(threaded.columns):
            raise AssertionError("Feature names or row order changed")
        a, b = (np.ascontiguousarray(frame.to_numpy()) for frame in (original, threaded))
        if a.dtype != b.dtype or not np.array_equal(a.view(np.uint8), b.view(np.uint8)):
            raise AssertionError(f"Concurrent batches changed values: {np.max(abs(a - b))}")
        reports.append(dict(stream=stream, rows=len(original), columns=len(original.columns),
                            bit_exact=True, maximum_difference=0.0))
    (ROOT / "cpu_088_rebuild/component_thread_equivalence.json").write_text(json.dumps(reports, indent=2))
    print(json.dumps(reports, indent=2), flush=True)


if __name__ == "__main__":
    main()
