"""Run a physical feature builder without its historical experiment/search.

Some old files mixed feature generation with a follow-up model experiment.
The reviewed dispatch below runs only the required generation function.
"""
import argparse
import importlib
import sys
from unittest.mock import patch

import ku_workspace

BUILDERS = {
    "periodic": ("periodic_components", "run"),
    "innov_v2": ("innovation_transform", "run"),
    "trend": ("trend_components", "run"),
    "curve": ("curve_likelihood", "run"),
    "ar_law": ("ar_likelihood", "run"),
    "phase_law": ("phase_likelihood", "run"),
    "mixture_law": ("mixture_law", "run"),
    "partition": ("partition_evidence", "run"),
    "harmonic": ("harmonic_change", "run"),
    "pulse": ("pulse_filter", "run"),
    "state": ("state_filter", "run"),
    "volatility_law": ("white_volatility_law", "extract"),
    "jump_law_pulse": ("white_jump_priors", "extract"),
    "jump_law_state": ("white_jump_priors", "extract"),
    "wnlaw_raw": ("white_laws", "extract"),
    "wnlaw_physical": ("white_laws", "extract"),
    "noise_tests": ("transfer_member", "prepare"),
    "bounds": ("bounded_prior", "main"),
    "alt_innov": ("white_alt_innov", "prepare"),
    "jump_law_nullstate": ("white_jump_priors", "extract"),
    "jump_law_prestate": ("white_jump_priors", "extract"),
    "jump_law_smoothvol": ("white_jump_priors", "extract"),
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("kind", choices=sorted(BUILDERS))
    p.add_argument("--part", choices=("train", "test"))
    p.add_argument("--view", choices=("c1", "c2", "c3", "c4", "rev"))
    a = p.parse_args()
    root = ku_workspace.root()
    sys.path[:0] = [str(root / "experiments/weight_space"), str(root / "experiments/v6")]
    # Imported v6 modules historically inspect argv at import time.
    sys.argv = ["mechanism"]
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(1)
    name, function = BUILDERS[a.kind]
    module = importlib.import_module(name)
    with threadpool_limits(limits=1), torch.inference_mode():
        if a.view:
            if a.kind != "volatility_law" or not a.part:
                raise ValueError("Only the per-split volatility builder uses --view")
            folder = root / "tune_out/weight_space" / f"view_{a.view}"
            if not (folder / "STATES_READY").is_file():
                raise ValueError("Build and verify the view state tables first")
            with patch.object(module, "OUT", folder):
                getattr(module, function)(a.part)
            return
        if a.kind in {"noise_tests", "bounds"}:
            if a.part:
                raise ValueError("This builder writes both splits in one call")
            getattr(module, function)()
            return
        if not a.part:
            raise ValueError("Specify --part for this builder")
        args = [a.part]
        if a.kind.startswith("jump_law_"):
            args.append(a.kind.removeprefix("jump_law_"))
        elif a.kind.startswith("wnlaw_"):
            args.extend([a.kind.removeprefix("wnlaw_"), module.dictionary()])
        getattr(module, function)(*args)


if __name__ == "__main__":
    main()
