"""Convert the recovered feature recipes into explicit, resumable CPU stages.

This is not the full model-training manifest. It covers feature preparation;
mechanism tables are a separate prerequisite before harmonic/view matrices.
All argv lists are reviewed data, never shell snippets. Heavy stages are not
launched by this command.
"""
import argparse
import ast
import json
from pathlib import Path

from pipeline.run_stages import ROOT

RAW = ["data/train_series.parquet", "data/test_series.parquet"]
LABELS = "data/train_labels.csv"


def code_closure(script):
    """Fingerprint local imports; imports from installed libraries are pinned separately."""
    base = ROOT / "pipeline/legacy_cpu"
    candidates = {}
    for path in base.rglob("*.py"):
        candidates.setdefault(path.stem, []).append(path)
    candidates.setdefault("features", []).append(ROOT / "pipeline/features_v4.py")
    pending, seen = [base / script, base / "ku_workspace.py"], set()
    while pending:
        path = pending.pop()
        if path in seen:
            continue
        seen.add(path)
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.update(alias.name.rsplit(".", 1)[-1] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.add(node.module.rsplit(".", 1)[-1])
            elif isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.endswith(".py"):
                names.add(Path(node.value).stem)
        for name in names:
            pending.extend(candidates.get(name, []))
    return sorted("clean/features.py" if path == ROOT / "pipeline/features_v4.py"
                  else path.relative_to(base).as_posix() for path in seen)


def build():
    recovered = json.loads((ROOT / "pipeline/legacy_cpu/feature_manifest.json").read_text())
    result = []
    outputs = []
    for step in recovered["steps"]:
        if step["group"] == "foundation":
            outputs.extend(step["outputs"])
            continue
        args = step["argv"][1:]
        script = args[0].replace("pipeline/legacy_cpu/", "")
        args = [script, *args[1:]]
        filename = Path(script).name
        dependency = []
        if script == "COPY":
            dependency = [args[1]]
            args = ["pipeline_helpers/copy_table.py", *args[1:]]
        elif filename in ("augment.py", "aug_grid.py", "harm_aug.py"):
            dependency = [RAW[0], LABELS]
            if filename == "augment.py":
                # Every retained loader selects kind == shift. Consume all
                # RNG draws, but skip feature extraction for discarded rows.
                # Exact fresh-data comparison: check_shift_augmentation.py.
                args = ["pipeline_helpers/shift_augmentation.py"]
            else:
                dependency += ["tune_out/aug_meta.csv"]
        elif filename in ("make_variants.py", "make_crops.py"):
            dependency = RAW[:]
        elif filename == "make_halves.py":
            dependency = [RAW[0], LABELS]
        elif filename in ("var_numpy_features.py", "harm_var.py"):
            dependency = [f"tune_out/var/{args[1]}_series.parquet"]
            if filename == "var_numpy_features.py":
                # Same numerical functions and input order; write bounded
                # batches instead of retaining every feature dictionary.
                # Bit-exact comparison: check_stream_features.py.
                args[0] = "pipeline_helpers/stream_numpy_features.py"
            elif len(args) > 2:
                args[2] = "2"
        elif filename.startswith("features_"):
            split = args[1]
            if split.startswith("var:"):
                dependency = [f"tune_out/var/{split[4:]}_series.parquet"]
            elif split == "aug":
                dependency = [RAW[0], LABELS, "tune_out/aug_meta.csv"]
            else:
                dependency = [f"data/{split}_series.parquet"]
            if filename in ("features_grid.py", "features_grid2.py", "features_dd.py"):
                args[2] = "2"
        elif filename == "generator_probe.py":
            dependency = [*RAW, LABELS]
            args = ["pipeline_helpers/generator_features.py", "--batch", "128"]
        elif filename == "train_var.py":
            # Matrix exports consume many blocks and views. Hash all prior
            # feature outputs here, once per matrix, to include that closure.
            dependency = [*outputs, LABELS]
            dependency += [x for x in recovered["meta"]["fixed_config_copy_into_workspace"]
                           if not x.endswith("adv_ptest_train.npy")]
        else:
            raise ValueError(f"Unreviewed feature producer: {script}")
        env = {"KU_SBB_THREADS": "2", "CHUNK": "256", **step.get("env", {})}
        row = dict(id=f"features_{step['id']:03d}_{Path(args[0]).stem}",
                   group=step["group"], argv=args,
                   inputs=list(dict.fromkeys(dependency)), outputs=step["outputs"],
                   code_inputs=code_closure(args[0]), env=env)
        result.append(row)
        outputs.extend(step["outputs"])
    # These feed later physical heads; the historical base-member-only audit
    # did not include them. Both original writers and CPU ports are present.
    extra = []
    for family in ("ps", "sn"):
        script = f"experiments/v6/features_{family}.py"
        for split in ("train", "test"):
            extra.append(dict(id=f"features_extra_{family}_{split}", group="base",
                              argv=[script, split], inputs=[f"data/{split}_series.parquet"],
                              outputs=[f"tune_out/{family}_{split}.parquet"],
                              code_inputs=code_closure(script),
                              env={"KU_SBB_THREADS": "2", "CHUNK": "128"}))
    position = next(i for i, stage in enumerate(result) if stage["group"] != "base")
    result[position:position] = extra
    return {"status": "Feature preparation only; full-size runtime validation pending",
            "stages": result}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "pipeline/feature_stages.json")
    args = parser.parse_args()
    value = build()
    args.out.write_text(json.dumps(value, indent=2))
    print(f"Wrote {len(value['stages'])} feature stages to {args.out}")
