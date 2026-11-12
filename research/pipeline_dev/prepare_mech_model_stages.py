"""Connect recovered physical specialist training and fixed prediction blends."""
import argparse
import json
from pathlib import Path
from pipeline.prepare_feature_stages import ROOT, code_closure
from pipeline.prepare_rebuild_stages import build as features, order


def build():
    raw = json.loads((ROOT / "pipeline/legacy_cpu/mech_manifest.json").read_text())
    feat = features()
    available = set(feat["prerequisite_inputs"])
    available.update(o for s in feat["stages"] for o in s["outputs"])
    common = ["data/train_labels.csv", "clean/features_train.parquet", "clean/features_test.parquet"]
    stages = []
    for item in raw["steps"]:
        argv = item["argv"][1:]
        argv[0] = argv[0].removeprefix("pipeline/legacy_cpu/")
        # Historical view loaders accept a modern view folder or an old flat
        # fallback. The fresh feature graph always writes the first path.
        required = [path.split("|", 1)[0] for path in item.get("inputs", [])]
        name = Path(argv[0]).stem
        if name == "white_import_tree":
            argv[0] = "pipeline_helpers/align_predictions.py"
            required += ["tune_out/var/mat_ns3_plain.npz"]
            required += [f"tune_out/{p}_{argv[1]}.npy" for p in ("oof", "test")]
        elif name == "white_source_bag":
            required += [f"tune_out/weight_space/{tag}.npz" for tag in argv[2:]]
        elif name == "mech_blends":
            tag = argv[1]
            if tag in ("MECH3C12RV_origonly", "MECH3C12RV_revlow"):
                required += ["tune_out/ver_MECH3C12RV_s0.npz"]
            elif tag in ("MECH3C12V_bag", "MECH3C1_bag", "MECHNEST2_bag", "MECHNEST3_bag"):
                required += [f"tune_out/{p}_{tag.removesuffix('_bag')}_s{s}.npy"
                             for s in range(3) for p in ("oof", "test")]
            else:
                raise ValueError(f"Unreviewed blend {tag}")
        elif item["kind"] != "tree":
            raise ValueError(f"Unreviewed producer {argv}")
        stages.append(dict(id=f"mech_models_{item['id']:03d}_{item['tag']}", argv=argv,
                           inputs=list(dict.fromkeys([*common, *required])),
                           outputs=[*item["outputs"], *item.get("checkpoints", [])],
                           code_inputs=code_closure(argv[0]),
                           env={"KU_SBB_THREADS": "2", "KU_SBB_REBUILD_FIXED": "1", "SAVE_WEIGHTS": "1"},
                           evidence=item["evidence"]))
    return dict(status="Reviewed physical specialist recipes; full-size validation pending",
                stages=order(stages, available), prerequisite_inputs=sorted(available))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=ROOT / "pipeline/mech_model_stages.json")
    a = p.parse_args()
    result = build()
    a.out.write_text(json.dumps(result, indent=2))
    print(f"Wrote {len(result['stages'])} physical specialist model stages")
