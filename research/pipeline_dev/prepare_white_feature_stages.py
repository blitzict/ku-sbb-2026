"""Additional deterministic inputs needed by retained white-noise specialists."""
from pipeline.prepare_feature_stages import code_closure

WS = "tune_out/weight_space/"


def build():
    stages = []
    helper = "pipeline_helpers/white_model_features.py"
    for part in ("train", "test"):
        for kind, module, outputs, deps in (
            ("hetero", "white_hetero_shift", [f"hetero_path_{part}.npz", f"hetero_jump_{part}.parquet"],
             [f"pulse_streams_{part}.npz", f"state_{part}.parquet"]),
            *[(f"memory{lag}", "white_memory_law", [f"memory_law{lag}_{part}.parquet"],
               [f"pulse_streams_{part}.npz", f"state_{part}.parquet"]) for lag in (1, 2)],
            *[(f"boundary_{s}", "white_boundary_law", [f"boundary_law_{s}_{part}.parquet"],
               [f"{s}_streams_{part}.npz"]) for s in ("state", "pulse")],
        ):
            stages.append(dict(id=f"white_feature_{kind}_{part}", argv=[helper, kind, part],
                               inputs=[WS + x for x in deps], outputs=[WS + x for x in outputs],
                               code_inputs=[helper, *code_closure(f"experiments/weight_space/{module}.py")]))
    for view in ("c1", "c2", "rev"):
        folder = WS + f"view_{view}/"
        script = "experiments/weight_space/white_view_extra.py"
        stages.append(dict(id=f"white_feature_extra_{view}", argv=[script, view],
                           inputs=[folder + "STATES_READY"] + [path for p in ("train", "test") for path in
                                   [folder + f"tune_out/seq_cache_{p}.npz", folder + f"pulse_streams_{p}.npz",
                                    folder + f"state_streams_{p}.npz", folder + f"innov_v2_{p}.parquet",
                                    folder + f"volatility_law_{p}.parquet",
                                    *[WS + f"{tag}_{p}.parquet" for tag in ("walk_jump", "native_law_state", "volatility_law")]]],
                           outputs=[folder + f"{tag}_{p}.parquet" for p in ("train", "test")
                                    for tag in ("walk_jump", "native_law_state")] + [folder + "EXTRA_READY"],
                           code_inputs=code_closure(script)))
    for name, module, tables in (
        ("filtered_step", "white_filtered_steps", ["pulse_streams.npz", "hetero_path.npz", "state.parquet"]),
        ("markov_copula", "white_markov_copula", ["pulse_streams.npz", "state_streams.npz"]),
    ):
        script = f"experiments/weight_space/{module}.py"
        stages.append(dict(id=f"white_feature_{name}", argv=[script],
                           inputs=[WS + f"{x.split('.')[0]}_{p}.{x.split('.')[1]}"
                                   for p in ("train", "test") for x in tables],
                           outputs=[WS + f"{name}_{p}.parquet" for p in ("train", "test")],
                           code_inputs=code_closure(script)))
    return dict(status="Required specialist inputs; runtime checks pending", stages=stages)
