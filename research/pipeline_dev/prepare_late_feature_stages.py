"""Required physical inputs for the late recovery heads; no discarded searches."""
import argparse
import json
from pathlib import Path
from pipeline.prepare_feature_stages import ROOT, code_closure

WS = "tune_out/weight_space/"
SPECS = [
    ("mode_residual", "white_remove_noise_modes", ["pulse_streams.npz"], True),
    ("walk_jump", "white_walk", ["@seq_cache.npz", "pulse_streams.npz", "innov_v2.parquet"], False),
    ("quantile_alignment", "white_quantile_alignment", ["pulse_streams.npz", "state_streams.npz"], False),
    ("component_moves_state", "white_component_moves", ["state_streams.npz"], False),
    ("component_moves_pulse", "white_component_moves", ["pulse_streams.npz"], False),
    ("latent_level_pulse", "white_latent_level", ["pulse_streams.npz"], False),
    ("latent_level_raw", "white_latent_level", ["@seq_cache.npz"], False),
    ("native_law_state", "white_native_law", ["state_streams.npz"], False),
    ("native_law_pulse", "white_native_law", ["pulse_streams.npz"], False),
    ("tail_edges", "white_tail_edges", ["pulse_streams.npz"], False),
    ("tail_channels", "white_tail_channels", ["pulse_streams.npz"], True),
]


def build():
    stages = []
    helper = "pipeline_helpers/late_features.py"
    for name, module, files, stream in SPECS:
        for part in ("train", "test"):
            deps = []
            for file in files:
                base, ext = file.split(".")
                deps.append(("tune_out/" + base[1:] if base.startswith("@") else WS + base) + f"_{part}.{ext}")
            selected_helper = "pipeline_helpers/component_moves_cpu.py" if name.startswith("component_moves_") else helper
            command_name = name.removeprefix("component_moves_") if selected_helper != helper else name
            stages.append(dict(id=f"late_feature_{name}_{part}", argv=[selected_helper, command_name, part], inputs=deps,
                               outputs=[WS + f"{name}_{part}.parquet"] + ([WS + f"{name}_streams_{part}.npz"] if stream else []),
                               code_inputs=[selected_helper, *code_closure(f"experiments/weight_space/{module}.py")],
                               env={"KU_SBB_THREADS": "2", "KU_SBB_REBUILD_FIXED": "1"}))
    for name, argv, deps, models in [
        ("transport_state", ["experiments/weight_space/white_population_transport.py", "state"],
         ["state_streams.npz", "innov_v2.parquet"], "population_transport_state_models"),
        ("transition_odds", ["experiments/weight_space/white_transition_priors.py"],
         ["native_law_state.parquet", "native_law_pulse.parquet", "innov_v2.parquet"], "transition_prior_models"),
    ]:
        required = ["data/train_labels.csv"]
        for part in ("train", "test"):
            required += [WS + f"{Path(dep).stem}_{part}{Path(dep).suffix}" for dep in deps]
        stages.append(dict(id=f"late_feature_{name}", argv=argv, inputs=required,
                           outputs=[WS + f"{name}_{p}.parquet" for p in ("train", "test")]
                           + [WS + f"{models}/fold{f}.npz" for f in range(5)],
                           code_inputs=code_closure(argv[0]),
                           env={"KU_SBB_THREADS": "2", "KU_SBB_REBUILD_FIXED": "1"}))
    return dict(status="Reviewed late physical inputs; full-size and CLI checks pending", stages=stages)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=ROOT / "pipeline/late_feature_stages.json")
    a = p.parse_args()
    m = build()
    a.out.write_text(json.dumps(m, indent=2))
    print(f"Wrote {len(m['stages'])} late physical input stages")
