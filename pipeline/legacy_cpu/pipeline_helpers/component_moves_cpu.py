"""Compute the same 45-row mixture batches concurrently on CPU.

The dictionary, prior, reductions, batch boundaries and row order are those
of white_component_moves.extract. Only independent batches overlap. Each
batch uses one Torch/BLAS thread; workers share the read-only dictionary.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import monotonic
import sys

import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits
import ku_workspace


def calculate(module, part, stream, workers, limit=0):
    with np.load(module.OUT / f"{stream}_streams_{part}.npz") as data:
        ids, x, n, b = [data[k] for k in ("ids", "separate", "n", "b")]
    if limit:
        ids, x, n, b = [v[:limit] for v in (ids, x, n, b)]
    groups = {}
    for key, values in (("lower_fixed", module.L), ("upper_fixed", module.U),
                        ("midpoint_fixed", module.L + module.U),
                        ("separation_fixed", module.U - module.L)):
        indices = [np.flatnonzero(values == i) for i in np.unique(values)]
        ix = np.full((len(indices), max(map(len, indices))), -1)
        for j, items in enumerate(indices):
            ix[j, :len(items)] = items
        groups[key] = tuple(torch.as_tensor(v, device="cpu") for v in (ix.clip(0), ix >= 0))
    bank = module.dictionary()

    @torch.inference_mode()
    def batch(start):
        sl = slice(start, start + 45)
        hs, centers, scales = [], [], []
        for raw, end, boundary in zip(x[sl], n[sl], b[sl]):
            y = raw[:end].astype(float)
            center = np.median(y)
            scale = max(np.median(abs(y - center)) * 1.4826, 0.18 * y.std(), 0.01)
            z = (y - center) / scale
            hs.extend((module.hist(z[:boundary]), module.hist(z[boundary:])))
            centers.append(center)
            scales.append(scale)
        values = (module.tensor(np.asarray(hs)) @ bank).reshape(
            len(centers), 2, len(module.L), len(module.LOGS), len(module.PROB))
        values -= values.amax((2, 3, 4), keepdim=True)
        before, after = values.unbind(1)
        prior = module.priors(np.array(centers), np.array(scales))
        record = module.summarize(before, after, prior, groups)
        array = torch.stack(list(record.values()), 1).clamp(-60, 100).cpu().numpy()
        return pd.DataFrame(array, columns=list(record), index=ids[sl])

    started, rows = monotonic(), []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, frame in enumerate(pool.map(batch, range(0, len(ids), 45)), 1):
            rows.append(frame)
            if i % 20 == 0:
                print(f"{stream} {part}: {min(i * 45, len(ids))}/{len(ids)} rows, "
                      f"{monotonic() - started:.0f}s", flush=True)
    result = pd.concat(rows)
    result.index.name = "id"
    if not result.index.is_unique or not np.isfinite(result).all().all():
        raise ValueError("Invalid component-move feature output")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stream", choices=("state", "pulse"))
    parser.add_argument("part", choices=("train", "test"))
    parser.add_argument("--workers", type=int, choices=(1, 2, 3))
    parser.add_argument("--limit", type=int, default=0, help="Numerical checks only; full stages leave this at zero")
    parser.add_argument("--output", type=Path, help="Alternate in-workspace output for numerical checks")
    args = parser.parse_args()
    root = ku_workspace.root()
    sys.path[:0] = [str(root / "experiments/weight_space"), str(root / "experiments/v6")]
    sys.argv = ["component_moves_cpu"]
    import white_component_moves as module
    output = (root / args.output).resolve() if args.output else module.OUT / f"component_moves_{args.stream}_{args.part}.parquet"
    if not output.is_relative_to(root) or args.limit < 0:
        raise ValueError("Use an in-workspace output and a non-negative row limit")
    if args.limit and not args.output:
        raise ValueError("A limited numerical check needs a separate output path")
    torch.set_num_threads(1)
    with threadpool_limits(1), torch.inference_mode():
        result = calculate(module, args.part, args.stream,
                           args.workers or min(3, ku_workspace.jobs()), args.limit)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output)
    print(f"Wrote {result.shape} CPU features to {output}", flush=True)


if __name__ == "__main__":
    main()
