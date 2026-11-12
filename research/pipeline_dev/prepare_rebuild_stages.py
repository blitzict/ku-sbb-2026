"""Combine the reviewed raw-feature stages in dependency order.

No experiment search is launched. Fixed feature-name lists and model choices
are retained. This manifest currently covers physical inputs and all views;
base-model/readout stages will be added only after their recovered commands
have been checked. An incomplete training manifest is explicitly labelled.
"""
import argparse
import json

from pipeline.prepare_feature_stages import ROOT, build as feature_stages
from pipeline.prepare_mechanism_stages import build as mechanism_stages, view_stages
from pipeline.prepare_late_feature_stages import build as late_stages
from pipeline.prepare_white_feature_stages import build as white_stages


def order(stages, available):
    produced = {}
    for stage in stages:
        for output in stage["outputs"]:
            if output in produced:
                raise ValueError(f"Two producers for {output}: {produced[output]}, {stage['id']}")
            produced[output] = stage["id"]
    unknown = {name for stage in stages for name in stage["inputs"] if name not in produced and name not in available}
    if unknown:
        raise ValueError(f"Undeclared required inputs: {sorted(unknown)}")
    pending, result = stages[:], []
    available = set(available)
    while pending:
        ready = next((s for s in pending if set(s["inputs"]).issubset(available)), None)
        if ready is None:
            unresolved = {s["id"]: sorted(set(s["inputs"]) - available) for s in pending}
            raise ValueError(f"Cyclic dependencies: {unresolved}")
        result.append(ready)
        available.update(ready["outputs"])
        pending.remove(ready)
    return result


def build():
    feature = feature_stages()["stages"]
    physical = mechanism_stages()["stages"]
    views = view_stages()
    settings = json.loads((ROOT / "pipeline/legacy_cpu/feature_manifest.json").read_text())["meta"]["fixed_config_copy_into_workspace"]
    available = [*settings, "data/train_labels.csv", "data/sample_submission.csv"]
    for part in ("train", "test"):
        available.extend([f"data/{part}_series.parquet", f"clean/features_{part}.parquet", f"tune_out/seq_cache_{part}.npz"])
    # Put original-series inputs before augmentation where dependencies allow.
    base = [s for s in feature if s["group"] == "base" and "aug" not in s["argv"][0]]
    remaining = [s for s in feature if s not in base]
    stages = order([*base, *physical, *late_stages()["stages"], *white_stages()["stages"], *remaining, *views], available)
    return dict(status="Reviewed feature preparation; model training not yet included",
                historical_row_predictions_used=False, prerequisite_inputs=available, stages=stages)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default=str(ROOT / "pipeline/rebuild_feature_stages.json"))
    a = p.parse_args()
    manifest = build()
    from pathlib import Path
    Path(a.out).write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {len(manifest['stages'])} ordered stages to {a.out}")
