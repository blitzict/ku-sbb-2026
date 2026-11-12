"""Assemble the fixed CPU recipe; reject missing or cyclic dependencies.

This is a packaging step. The notebook executes the resulting manifest and
does not need the historical research folders used to recover the recipe.
"""
import json
from pathlib import Path
from pipeline.prepare_feature_stages import ROOT, code_closure
from pipeline.prepare_rebuild_stages import order
from pipeline.prepare_white_head_stages import build as late_heads
from pipeline.prepare_mechanism_stages import view_code_inputs


def build():
    folder = ROOT / "pipeline"
    features = json.loads((folder / "rebuild_feature_stages_cpu6_stream.json").read_text())
    stages = features["stages"][:]
    for name in ("base_model_stages", "mech_model_stages", "early_head_stages", "white_model_stages"):
        stages.extend(json.loads((folder / (name + ".json")).read_text())["stages"])
    heads = late_heads()
    (folder / "white_head_stages.json").write_text(json.dumps(heads, indent=2))
    stages.extend(heads["stages"])
    # extend_readout loads these five fixed recipe modules with importlib;
    # ordinary static import tracing cannot see the constructed module names.
    # They are executed in model/head stages, not in the feature extraction
    # functions that merely import the same module's unused definitions.
    feature_ids = {s["id"] for s in features["stages"]}
    for stage in stages:
        if Path(stage["argv"][0]).stem == "white_view_pipeline":
            # These crop stages have not started in the current original-view
            # phase. Track every dynamically dispatched builder on resume.
            stage["code_inputs"] = view_code_inputs()
        if stage["id"] not in feature_ids:
            stage["code_inputs"] = code_closure(stage["argv"][0])
            if "experiments/weight_space/extend_readout.py" in stage["code_inputs"]:
                extra = [p for name in ("curve", "bank", "pulse", "head", "state")
                         for p in code_closure(f"experiments/weight_space/{name}_channels.py")]
                stage["code_inputs"] = sorted(set(stage["code_inputs"] + extra))
    names = [stage["id"] for stage in stages]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate stage identifiers")
    stages = order(stages, features["prerequisite_inputs"])
    return dict(
        complete_dependency_graph=True,
        status="Graph checked; complete fresh CPU score still pending",
        training_origin="raw competition files and fixed historical settings",
        historical_prediction_inputs=False, device="cpu", target_local_auc=.87910,
        primary_output_fold_seed=7, other_output_fold_seeds=[29, 0, 123, 2024, 11041],
        omitted_unselected_branches=heads["omitted_unselected_branches"],
        prerequisite_inputs=features["prerequisite_inputs"],
        training_profile=features["training_profile"], stages=stages,
    )


if __name__ == "__main__":
    result = build()
    dest = ROOT / "pipeline/full_cpu_stages.json"
    dest.write_text(json.dumps(result, indent=2))
    print(f"Validated {len(result['stages'])} stages; wrote {dest}")
