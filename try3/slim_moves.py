"""try3 fast path for pipeline_helpers/component_moves_cpu.py: same script, but summarize() only builds the columns the
try3 models read (state: all, dispersion, midpoint_fixed; pulse: centers_dispersion). Each kept column goes through
the same tensor operations as the original summarize(), so the values are identical.

python try3/slim_moves.py state|pulse test        (run inside the prediction work folder, like the original step)
"""
import sys
from pathlib import Path

import numpy as np
import torch

KEEP = {"state": ("all", "dispersion", "midpoint_fixed"), "pulse": ("centers_dispersion",)}
DIMS = {"dispersion": 2, "occupancy": 3, "dispersion_occupancy": (2, 3), "centers": 1, "centers_occupancy": (1, 3),
        "centers_dispersion": (1, 2), "all": (1, 2, 3)}


def slim_summarize(module, keep):
    def summarize(a, b, prior, groups):
        lse = torch.logsumexp
        z = lse(a + b + prior, (1, 2, 3))
        out = {}
        for tag in keep:
            if tag in DIMS:
                out[tag] = module.collapse(a, b, prior, DIMS[tag]) - z
            else:
                index, valid = groups[tag]
                pp = prior[:, index].masked_fill(~valid[None, :, :, None, None], -100000.0)
                terms = lse(pp + a[:, index], 2) + lse(pp + b[:, index], 2) - lse(pp, 2)
                out[tag] = lse(terms, (1, 2, 3)) - z
        return out
    return summarize


def main():
    stream = sys.argv[1]
    import ku_workspace
    root = ku_workspace.root()
    sys.path.insert(0, str(root / "pipeline_helpers"))
    import component_moves_cpu as cm
    sys.path[:0] = [str(root / "experiments/weight_space"), str(root / "experiments/v6")]
    saved, sys.argv = sys.argv, ["component_moves_cpu"]
    import white_component_moves as module
    sys.argv = saved
    module.summarize = slim_summarize(module, KEEP[stream])
    cm.main()


if __name__ == "__main__":
    sys.argv = [sys.argv[0], *sys.argv[1:]]
    main()
