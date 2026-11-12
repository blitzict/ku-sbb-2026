"""Turn recovered fixed member settings into fingerprinted CPU training stages.

This covers the base 51-member ensemble and its neural/view head edits. The
later physical specialists and output layers are separate stages. This file
does not launch training or change any recorded hyperparameters.
"""
import argparse
import ast
import csv
import json
from pathlib import Path

from pipeline.prepare_feature_stages import ROOT, code_closure
from pipeline.prepare_rebuild_stages import build as feature_manifest, order


def options(argv):
    result = {}
    i = 1
    while i < len(argv):
        if argv[i].startswith("--"):
            key = argv[i][2:]
            if i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                result[key] = argv[i + 1]
                i += 2
            else:
                result[key] = True
                i += 1
        else:
            i += 1
    return result


def columns(path):
    with (ROOT / path).open(newline="", encoding="utf-8-sig") as f:
        return [row["feature"] for row in csv.DictReader(f)]


def inputs(argv):
    name, opts = Path(argv[0]).stem, options(argv)
    common = ["data/train_labels.csv", "clean/features_train.parquet", "clean/features_test.parquet"]
    if name in {"train_cv", "nn_xpu", "distill"}:
        result = common[:]
        aug = not opts.get("no-aug", False)
        if aug:
            result += ["tune_out/aug_features.parquet", "tune_out/aug_meta.csv"]
        for block in str(opts.get("blocks", "grid,grid2,dd,lr" if name == "distill" else "grid,grid2")).split(","):
            if block:
                result += [f"tune_out/{block}_{part}.parquet" for part in ("train", "test")]
                if aug:
                    result.append(f"tune_out/aug_{block}_features.parquet")
        if name == "train_cv":
            result.append("tune_out/best_xgb5.json")
        if opts.get("cols"):
            result.append(opts["cols"])
        if opts.get("drop-train"):
            raise ValueError("A source-ID exclusion file needs a separately reviewed producer")
        return result
    if name == "train_var":
        colfile = opts.get("colfile", "tune_out/selected_ns3.csv")
        mapping = {"lr": "lr5", "lrz": "lrz6"}
        prefixes = {mapping.get(c.split("__")[0], c.split("__")[0]) if "__" in c else "clean" for c in columns(colfile)}
        kinds = [v for v in str(opts.get("kinds", "rev,flip,revflip")).split(",") if v]
        tta = [v for v in str(opts.get("tta-kinds", "rev,flip,revflip")).split(",") if v]
        result = common + [colfile, "tune_out/aug_meta.csv"]
        for prefix in sorted(prefixes):
            result += ([f"clean/features_{part}.parquet" for part in ("train", "test")] if prefix == "clean"
                       else [f"tune_out/{prefix}_{part}.parquet" for part in ("train", "test")])
            result.append("tune_out/aug_features.parquet" if prefix == "clean" else f"tune_out/aug_{prefix}_features.parquet")
            result += [f"tune_out/var/{prefix}_train_{view}.parquet" for view in dict.fromkeys(kinds + tta)]
            result += [f"tune_out/var/{prefix}_test_{view}.parquet" for view in tta]
        for extra in str(opts.get("extra", "")).split(","):
            if extra:
                result += [f"tune_out/var/{extra}_meta.csv", *[f"tune_out/var/{prefix}_train_{extra}.parquet" for prefix in sorted(prefixes)]]
        return result
    if name == "nn_var":
        return [opts["mat"]]
    if name == "nn_seq":
        if opts.get("extra") or opts.get("aux"):
            raise ValueError("Sequence extra/aux settings need explicit dependencies")
        return common + [f"tune_out/seq_cache_{part}.npz" for part in ("train", "test")]
    if name == "wtta":
        # Known manifest commands put --out <name> first and other options last.
        tags = []
        i = 1
        while i < len(argv):
            if argv[i].startswith("--"):
                i += 2
            else:
                tags.append(argv[i])
                i += 1
        return common + [f"tune_out/ver_{tag}.npz" for tag in tags]
    if name == "mech_blends":
        tag = argv[1]
        if tag == "W_SEQ_qb5":
            members = ["W_SEQ_q", "W_SEQ_q_s1", "W_SEQ_qh", "W_SEQ_q_s2", "W_SEQ_qh_s1"]
        elif tag.startswith("L4_fam_") and tag.endswith("_b2"):
            members = [tag[:-3], tag[:-3] + "_s1"]
        elif tag in {"VAR_aug1600_avg_tta", "VAR_full_aug_avg_tta"}:
            base = tag.removesuffix("_avg_tta")
            members = [base + "_tta", base + "_s1_tta"]
        else:
            raise ValueError(f"Unreviewed base blend: {tag}")
        return [f"tune_out/{part}_{member}.npy" for member in members for part in ("oof", "test")]
    raise ValueError(f"No explicit inputs for {name}")


def saved_outputs(argv):
    name, opts = Path(argv[0]).stem, options(argv)
    tag = opts.get("tag")
    if name in {"train_cv", "train_var"}:
        ext = {"xgb": "ubj", "lgb": "txt", "cat": "cbm"}[opts.get("model", "xgb")]
        return [f"weights/{tag}.{ext}", f"weights/{tag}.meta.json"]
    if name == "nn_var":
        return [f"weights/{tag}/fold{k}_seed{s}.pt" for k in range(5) for s in range(int(opts.get("seeds", 3)))]
    if name == "nn_seq":
        seed = int(opts.get("seeds", 0))
        return [f"weights/{tag}/fold{k}_seed{seed}.pt" for k in range(5)] + [f"tune_out/seqpart/{tag}_f{k}.npz" for k in range(5)]
    if name == "nn_xpu":
        return [f"weights/{tag}/seed{100*k+s}.pt" for k in range(5) for s in range(int(opts.get("seeds", 3)))]
    # The distillation and exact-transform filenames are being finalized by
    # their port owner; they must be added before qualification packaging.
    return []


def build():
    raw = json.loads((ROOT / "pipeline/legacy_cpu/model_manifest.json").read_text())
    feature = feature_manifest()
    available = set(feature["prerequisite_inputs"])
    available.update(o for step in feature["stages"] for o in step["outputs"])
    stages, previous = [], []
    for item in raw["steps"]:
        argv = item["argv"][1:]
        argv[0] = argv[0].removeprefix("pipeline/legacy_cpu/")
        if argv[0] == "WRITE_JSON":
            argv[0] = "pipeline_helpers/write_config.py"
            required = []
        elif item["stage"] == "stage7":
            required = [*previous, "tune_out/var/mat_cggd_hv.npz", "data/train_labels.csv", "data/sample_submission.csv",
                        "clean/features_train.parquet", "clean/features_test.parquet"]
        else:
            required = inputs(argv)
        outputs = list(dict.fromkeys([*item["outputs"], *item.get("checkpoints", []), *saved_outputs(argv)]))
        if item["tag"] == "simplify":
            outputs.append("new weights/model_manifest.json")
            for tag, seed in (("W_SEQ_q", 0), ("W_SEQ_q_s1", 1), ("W_SEQ_qh", 0),
                              ("W_SEQ_q_s2", 2), ("W_SEQ_qh_s1", 1)):
                outputs.extend(f"new weights/models/{tag}/fold{k}_seed{seed}.pt" for k in range(5))
        stages.append(dict(id=f"models_{item['id']:03d}_{item['tag']}", argv=argv,
                           inputs=list(dict.fromkeys(required)), outputs=outputs,
                           code_inputs=code_closure(argv[0]),
                           env={"KU_SBB_REBUILD_FIXED": "1", "KU_SBB_THREADS": "2", "SAVE_WEIGHTS": "1"},
                           evidence=item["evidence"]))
        previous.extend(outputs)
    ordered = order(stages, available)
    return dict(status="Base-member and head-edit training; full runtime validation pending. Later specialists not included.",
                stages=ordered, prerequisite_inputs=sorted(available),
                remaining_checkpoint_declarations=["Additional active stage7 copies must be read from the generated model_manifest"])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=ROOT / "pipeline/base_model_stages.json")
    a = p.parse_args()
    value = build()
    a.out.write_text(json.dumps(value, indent=2))
    print(f"Wrote {len(value['stages'])} base-model and head-edit stages")
