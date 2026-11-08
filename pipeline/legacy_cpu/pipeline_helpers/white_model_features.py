"""Prepare inputs used by retained noise-law models without fitting trial heads."""
import argparse
import sys
from unittest.mock import patch
import ku_workspace as env


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("kind", choices=("hetero", "memory1", "memory2", "boundary_state", "boundary_pulse", "view_extra"))
    p.add_argument("part", choices=("train", "test"))
    p.add_argument("--view", choices=("c1", "c2"))
    a = p.parse_args()
    root = env.root()
    sys.path.insert(0, str(root / "experiments/weight_space"))
    sys.argv = ["white_model_features"]
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(1)
    with torch.inference_mode(), threadpool_limits(1):
        if a.kind == "hetero":
            import white_hetero_shift as m
            m.profiles(a.part)
            m.extract(a.part)
        elif a.kind.startswith("memory"):
            import white_memory_law as m
            m.extract(a.part, int(a.kind[-1]), m.bank())
        elif a.kind.startswith("boundary_"):
            import white_boundary_law as m
            m.extract(a.part, a.kind.removeprefix("boundary_"))
        else:
            if a.view is None:
                raise ValueError("view_extra requires --view")
            folder = root / "tune_out/weight_space" / f"view_{a.view}"
            if not (folder / "STATES_READY").exists():
                raise ValueError("View state features must be built first")
            import white_walk
            import white_native_law
            with patch.multiple(white_walk, ROOT=folder, OUT=folder):
                white_walk.extract(a.part)
            with patch.object(white_native_law, "OUT", folder):
                white_native_law.extract(a.part, "state")


if __name__ == "__main__":
    main()
