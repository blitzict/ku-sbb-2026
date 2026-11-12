"""Compare the memory-bounded generator port with its unbatched CPU source.

Uses only the existing 16-series raw smoke inputs. No historical features feed
either implementation. This test is needed because the 8 GB target requires
changing allocation strategy in a numerically sensitive feature builder.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from pipeline.build_foundation import initialize
from pipeline.sync_ports import sync
from pipeline.run_stages import ROOT


def main():
    work = initialize(ROOT / "cpu_088_rebuild/generator_numeric_smoke", ROOT / "cpu_088_rebuild/cli_smoke_inputs")
    sync(work)
    environment = os.environ.copy()
    environment.update(KU_SBB_ROOT=str(work), KU_SBB_CODE_ROOT=str(work), KU_SBB_THREADS="1",
                       KU_SBB_RAW_ONLY="1", KU_SBB_ORIGINAL_ROOT=str(ROOT), KU_SBB_OFFLINE="1",
                       CUDA_VISIBLE_DEVICES="", ONEAPI_DEVICE_SELECTOR="*:cpu", OMP_NUM_THREADS="1",
                       OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    environment["PYTHONPATH"] = os.pathsep.join([str(ROOT / "pipeline/offline_hooks"), str(work)])
    def execute(script, *args):
        subprocess.run([sys.executable, "-B", str(work / script), *args], cwd=work, env=environment, check=True)
    execute("experiments/v6/generator_probe.py")
    import numpy as np
    import pandas as pd
    reference = pd.read_parquet(work / "tune_out/generator_segments.parquet")
    reference_delta = pd.read_parquet(work / "tune_out/generator_deltas.parquet")
    shutil.copy2(work / "tune_out/generator_segments.parquet", work / "stages/unbatched_reference.parquet")
    results = []
    for batch in (7, 128):
        execute("pipeline_helpers/generator_features.py", "--batch", str(batch))
        actual = pd.read_parquet(work / "tune_out/generator_segments.parquet")
        actual_delta = pd.read_parquet(work / "tune_out/generator_deltas.parquet")
        pd.testing.assert_frame_equal(actual, reference, rtol=2e-6, atol=2e-6)
        pd.testing.assert_frame_equal(actual_delta, reference_delta, rtol=2e-6, atol=2e-6)
        cols = reference.select_dtypes(include="number").columns
        error = float(np.max(np.abs(actual[cols].to_numpy() - reference[cols].to_numpy())))
        results.append(dict(batch=batch, rows=len(actual), max_abs_difference=error, status="pass"))
    out = work / "stages/generator_numeric_check.json"
    out.write_text(json.dumps(dict(reference="fresh unbatched CPU port with stable Student-t constant", results=results), indent=2))
    print(out.read_text())


if __name__ == "__main__":
    main()
