"""Exercise CPU mechanism extraction on a small fresh-data workspace."""
import os
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"

import argparse
import importlib
import json
from pathlib import Path
import sys
from time import monotonic

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("work", type=Path)
    args = parser.parse_args()
    work = args.work.resolve()
    if not (work/"CPU_REBUILD_WORKSPACE.json").exists():
        raise ValueError("A marked foundation workspace is required")
    os.environ["KU_SBB_ROOT"] = str(work)
    sys.path.insert(0, str(work/"experiments/weight_space"))
    modules = [("periodic", "periodic_components"), ("innov_v2", "innovation_transform"),
               ("trend", "trend_components"), ("curve", "curve_likelihood"),
               ("ar_law", "ar_likelihood"), ("phase_law", "phase_likelihood"),
               ("mixture_law", "mixture_law"), ("partition", "partition_evidence"),
               ("pulse", "pulse_filter"), ("state", "state_filter")]
    reports = []
    torch.set_num_threads(1)
    with threadpool_limits(limits=1), torch.inference_mode():
        # mixture_law consumes the likelihood block; build that prerequisite
        # from the freshly reconstructed observations, not a historical table.
        sys.path.insert(0, str(work/"experiments/v6"))
        likelihood = importlib.import_module("features_lr")
        for split in ("train", "test"):
            with np.load(work/f"tune_out/seq_cache_{split}.npz") as cache:
                jobs = [(str(i), values[:n].astype(float), int(b)) for i, values, n, b in
                        zip(cache["ids"], cache["x"], cache["n"], cache["b"])]
            frame = likelihood.block(jobs)
            frame.index.name = "id"
            frame.replace([np.inf, -np.inf], np.nan).fillna(0).astype(np.float32).to_parquet(
                work/f"tune_out/lr5_{split}.parquet")
        for tag, module_name in modules:
            start = monotonic()
            module = importlib.import_module(module_name)
            for split in ("train", "test"):
                module.run(split)
                path = work/f"tune_out/weight_space/{tag}_{split}.parquet"
                frame = pd.read_parquet(path)
                ids = pd.read_parquet(work/f"clean/features_{split}.parquet", columns=[]).index
                if not frame.index.equals(ids) or not np.isfinite(frame.to_numpy()).all():
                    raise ValueError(f"Invalid output from {module_name}/{split}")
            reports.append(dict(module=module_name, passed=True, seconds=monotonic()-start))
            print(f"PASS {module_name}", flush=True)
        module = importlib.import_module("white_laws")
        start = monotonic()
        bank = module.dictionary()
        if bank.device.type != "cpu" or not torch.isfinite(bank).all():
            raise ValueError("Noise-law dictionary is invalid or not on CPU")
        for tag in ("raw", "physical"):
            for split in ("train", "test"):
                module.extract(split, tag, bank)
        reports.append(dict(module="white_laws", passed=True, seconds=monotonic()-start,
                            device=str(bank.device), dictionary_shape=list(bank.shape)))
        print("PASS white_laws", flush=True)
    (work/"stages/mechanism_smoke.json").write_text(json.dumps(reports, indent=2))
    print(f"{len(reports)} mechanism smoke tests passed; full-scale validation is separate.")


if __name__ == "__main__":
    main()
