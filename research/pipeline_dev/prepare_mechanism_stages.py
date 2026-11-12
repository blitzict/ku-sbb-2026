"""Write the ordered label-free physical feature stages needed by the ensemble."""
import argparse
import json
from pathlib import Path

from pipeline.prepare_feature_stages import ROOT, code_closure

WS = "tune_out/weight_space/"
VIEW_BUILDERS = ("periodic_components", "innovation_transform", "trend_components",
                 "curve_likelihood", "ar_likelihood", "phase_likelihood",
                 "mixture_law", "partition_evidence", "pulse_filter", "transfer_member")


def view_code_inputs():
    """Include the fixed modules loaded by white_view_pipeline's importlib loop."""
    scripts = ("white_view_pipeline", *VIEW_BUILDERS)
    return sorted({p for name in scripts
                   for p in code_closure(f"experiments/weight_space/{name}.py")})


# Every item names a feature table; ':' marks an npz stream dependency.
SPECS = [
    ("periodic", "periodic_components", [], True),
    ("innov_v2", "innovation_transform", ["periodic", ":periodic"], True),
    ("trend", "trend_components", ["periodic", "innov_v2"], True),
    ("curve", "curve_likelihood", ["trend", "periodic", "innov_v2"], True),
    ("ar_law", "ar_likelihood", ["periodic", ":periodic"], False),
    ("phase_law", "phase_likelihood", ["periodic"], False),
    ("mixture_law", "mixture_law", [":innov_v2"], False),
    ("partition", "partition_evidence", [], False),
    ("harmonic", "harmonic_change", [], False),
    ("pulse", "pulse_filter", ["periodic", ":periodic", ":curve"], True),
    ("state", "state_filter", [":pulse", "trend"], True),
    ("volatility_law", "white_volatility_law", [":pulse", "state"], False),
    ("jump_law_pulse", "white_jump_priors", [":pulse"], False),
    ("jump_law_state", "white_jump_priors", [":state"], False),
    ("wnlaw_raw", "white_laws", [], False),
    ("wnlaw_physical", "white_laws", [":pulse"], False),
]


def build():
    result = []
    helper = "pipeline_helpers/mechanism.py"
    for tag, module, dependencies, streams in SPECS:
        code = list(dict.fromkeys([helper, *code_closure(f"experiments/weight_space/{module}.py")]))
        for part in ("train", "test"):
            inputs = [f"tune_out/seq_cache_{part}.npz", f"clean/features_{part}.parquet"]
            for dep in dependencies:
                inputs.append(WS + (f"{dep[1:]}_streams_{part}.npz" if dep.startswith(":") else f"{dep}_{part}.parquet"))
            if tag == "mixture_law":
                inputs.append(f"tune_out/lr5_{part}.parquet")
            output = WS + f"{tag}_{part}" + (".npz" if tag.startswith("wnlaw_") else ".parquet")
            outputs = [output]
            if streams:
                outputs.append(WS + f"{tag}_streams_{part}.npz")
            result.append(dict(id=f"mechanism_{tag}_{part}", argv=[helper, tag, "--part", part],
                               inputs=inputs, outputs=outputs, code_inputs=code,
                               env={"KU_SBB_THREADS": "2", "CHUNK": "128"}))
    # Noise tests precede the transfer model and therefore run independently.
    result.append(dict(id="mechanism_noise_tests", argv=[helper, "noise_tests"],
                       inputs=[path for part in ("train", "test") for path in
                               [WS + f"innov_v2_streams_{part}.npz", f"clean/features_{part}.parquet"]],
                       outputs=[WS + f"noise_tests_{part}.parquet" for part in ("train", "test")],
                       code_inputs=[helper, *code_closure("experiments/weight_space/transfer_member.py")],
                       env={"KU_SBB_THREADS": "2", "CHUNK": "128"}))
    result.append(dict(id="mechanism_bounds", argv=[helper, "bounds"],
                       inputs=["tune_out/generator_segments.parquet", *[f"clean/features_{p}.parquet" for p in ("train", "test")]],
                       outputs=[WS + f"bounds_{p}.parquet" for p in ("train", "test")],
                       code_inputs=[helper, *code_closure("experiments/weight_space/bounded_prior.py")]))
    for part in ("train", "test"):
        result.append(dict(id=f"mechanism_alt_innov_{part}", argv=[helper, "alt_innov", "--part", part],
                           inputs=[WS + f"{name}_{part}.{ext}" for name, ext in
                                   (("pulse_streams", "npz"), ("state_streams", "npz"), ("state", "parquet"))],
                           outputs=[WS + f"{name}_streams_{part}.npz" for name in ("nullstate", "prestate", "smoothvol")]
                           + [WS + f"vol_smoothing_{part}.parquet"],
                           code_inputs=[helper, *code_closure("experiments/weight_space/white_alt_innov.py")]))
        for stream in ("nullstate", "prestate", "smoothvol"):
            result.append(dict(id=f"mechanism_jump_law_{stream}_{part}",
                               argv=[helper, f"jump_law_{stream}", "--part", part],
                               inputs=[WS + f"{stream}_streams_{part}.npz"],
                               outputs=[WS + f"jump_law_{stream}_{part}.parquet"],
                               code_inputs=[helper, *code_closure("experiments/weight_space/white_jump_priors.py")]))
    for seed in (0, 7, 29, 123, 2024, 11041):
        script = "experiments/weight_space/white_outer_priors.py"
        result.append(dict(id=f"mechanism_outer_priors_{seed}", argv=[script, "--seeds", str(seed)],
                           inputs=[WS + f"wnlaw_physical_{p}.npz" for p in ("train", "test")]
                           + ["tune_out/generator_segments.parquet", "data/train_labels.csv"],
                           outputs=prior_outputs(WS + f"law_outer{seed}", priors=True),
                           code_inputs=code_closure(script)))
    return {"status": "Required mechanism preparation; validation is recorded per completed stage", "stages": result}


def prior_outputs(folder, folds=5, priors=False):
    return [f"{folder}/fold{k}_{name}" for k in range(folds)
            for name in ["train.parquet", "test.parquet", "rows.npz", *(["prior.json"] if priors else [])]]


def view_stages(views=("c1", "c2", "c3", "c4", "rev")):
    """Physical features on each actual crop/reversal, after basic view features."""
    result = []
    modules = ["periodic", "innov_v2", "trend", "curve", "ar_law", "phase_law", "mixture_law", "partition", "pulse"]
    streams = ["periodic", "innov_v2", "curve", "pulse"]
    extra = ["bounds", "noise_tests", *modules]
    for view in views:
        folder = WS + f"view_{view}/"
        script = "experiments/weight_space/white_view_pipeline.py"
        result.append(dict(id=f"mechanism_view_{view}", argv=[script, view],
                           inputs=[path for p in ("train", "test") for path in
                                   [f"tune_out/seq_cache_{p}.npz", f"tune_out/var/clean_{p}_{view}.parquet",
                                    f"tune_out/var/lr5_{p}_{view}.parquet", *[WS + f"{tag}_{p}.parquet" for tag in extra]]]
                           + prior_outputs(WS + "law_outer0", priors=True),
                           outputs=[folder + f"{tag}_{p}.parquet" for p in ("train", "test") for tag in extra]
                           + [folder + f"{tag}_streams_{p}.npz" for p in ("train", "test") for tag in streams]
                           + [folder + f"wnlaw_physical_{p}.npz" for p in ("train", "test")]
                           + [folder + f"tune_out/seq_cache_{p}.npz" for p in ("train", "test")]
                           + prior_outputs(folder + "law_outer0") + [folder + "READY"],
                           code_inputs=view_code_inputs()))
        script = "experiments/weight_space/white_view_states.py"
        jump = [f"jump_law_{s}" for s in ("pulse", "state", "nullstate", "prestate", "smoothvol")]
        result.append(dict(id=f"mechanism_view_states_{view}", argv=[script, view],
                           inputs=[folder + "READY"] + result[-1]["outputs"]
                           + [WS + f"{tag}_{p}.parquet" for p in ("train", "test") for tag in jump],
                           outputs=[folder + f"{tag}_{p}.parquet" for p in ("train", "test") for tag in ["state", "vol_smoothing", *jump]]
                           + [folder + f"{tag}_streams_{p}.npz" for p in ("train", "test") for tag in ("state", "nullstate", "prestate", "smoothvol")]
                           + [folder + "STATES_READY"], code_inputs=code_closure(script)))
        helper = "pipeline_helpers/mechanism.py"
        for part in ("train", "test"):
            result.append(dict(id=f"mechanism_view_volatility_{view}_{part}",
                               argv=[helper, "volatility_law", "--part", part, "--view", view],
                               inputs=[folder + "STATES_READY", folder + f"pulse_streams_{part}.npz", folder + f"state_{part}.parquet"],
                               outputs=[folder + f"volatility_law_{part}.parquet"],
                               code_inputs=[helper, *code_closure("experiments/weight_space/white_volatility_law.py")]))
        if view in ("c1", "c2"):
            for seed in (7, 29):
                script = "experiments/weight_space/white_crop_partition.py"
                result.append(dict(id=f"mechanism_view_prior_{view}_{seed}", argv=[script, str(seed), view],
                                   inputs=[folder + f"wnlaw_physical_{p}.npz" for p in ("train", "test")]
                                   + ["data/train_labels.csv"] + prior_outputs(WS + f"law_outer{seed}", priors=True)
                                   + prior_outputs(folder + "law_outer0"),
                                   outputs=prior_outputs(folder + f"law_outer{seed}"), code_inputs=code_closure(script)))
    if {"c1", "c2"}.issubset(views):
        script = "experiments/weight_space/white_tenfold_priors.py"
        result.append(dict(id="mechanism_tenfold_prior43", argv=[script, "43"],
                           inputs=[WS + prefix + f"wnlaw_physical_{p}.npz" for prefix in ("", "view_c1/", "view_c2/") for p in ("train", "test")]
                           + ["data/train_labels.csv", "tune_out/generator_segments.parquet"],
                           outputs=[path for prefix in ("", "view_c1/", "view_c2/") for path in
                                    prior_outputs(WS + prefix + "law_outer10x43", folds=10, priors=prefix == "")],
                           code_inputs=code_closure(script)))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=ROOT / "pipeline/mechanism_stages.json")
    a = p.parse_args()
    d = build()
    a.out.write_text(json.dumps(d, indent=2))
    print(f"Wrote {len(d['stages'])} physical-feature stages")
