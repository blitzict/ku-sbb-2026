"""Required late physical features, without their historical follow-up searches."""
import argparse
import importlib
import sys
import ku_workspace

BUILDERS = {
    "mode_residual": ("white_remove_noise_modes", None),
    "walk_jump": ("white_walk", None),
    "quantile_alignment": ("white_quantile_alignment", None),
    "component_moves_state": ("white_component_moves", "state"),
    "component_moves_pulse": ("white_component_moves", "pulse"),
    "latent_level_pulse": ("white_latent_level", "pulse"),
    "latent_level_raw": ("white_latent_level", "raw"),
    "native_law_state": ("white_native_law", "state"),
    "native_law_pulse": ("white_native_law", "pulse"),
    "tail_edges": ("white_tail_edges", None),
    "tail_channels": ("white_tail_channels", None),
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("kind", choices=sorted(BUILDERS))
    p.add_argument("part", choices=("train", "test"))
    a = p.parse_args()
    root = ku_workspace.root()
    sys.path[:0] = [str(root / "experiments/weight_space"), str(root / "experiments/v6")]
    sys.argv = ["late_features"]
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(1)
    module, stream = BUILDERS[a.kind]
    builder = importlib.import_module(module)
    with threadpool_limits(1), torch.inference_mode():
        if module == "white_latent_level":
            builder.extract(stream, a.part)
        elif stream is None:
            builder.extract(a.part)
        else:
            builder.extract(a.part, stream)


if __name__ == "__main__":
    main()
