"""Focused tests for pipeline.training_config. No model fits; temporary workspaces only.

Run: python -m pipeline.check_training_config
"""
import json
import shutil
import tempfile
from pathlib import Path

from pipeline import training_config as TC
from pipeline.run_stages import ROOT

base = TC.load_base()
results = {}


def expect_error(name, config, text):
    try:
        TC.plan(config)
    except ValueError as error:
        results[name] = text in str(error)
        if not results[name]:
            print(name, "wrong error:", error)
        return
    results[name] = False
    print(name, "was accepted")


# Defaults reproduce the manifest object and the packaged bytes; template as written is a no-op.
p = TC.plan(None)
results["default_is_base"] = p.default and p.manifest == base and not p.settings and not p.changed_stages
results["template_is_noop"] = TC.plan(TC.template()).default
results["hyperparameters_json_is_noop"] = TC.plan(ROOT / "pipeline/hyperparameters.json").default

# Real controls.
nn = next(s for s in base["stages"] if s["argv"][0].endswith("v6/nn_var.py"))
cv = next(s for s in base["stages"] if s["id"] == "models_026_v6_ns3_xlg_s0")
custom = {"workers": 3,
          "feature_selections": {"tune_out/selected_ns3.csv": {"remove": ["car1_rank"], "add": []}},
          "shared_settings": {"tune_out/best_xgb5.json": {"iters": 400, "params": {"max_depth": 8}}},
          "script_options": {"experiments/v6/nn_var.py": {"--epochs": 5, "--lr": 0.0005}},
          "stage_options": {nn["id"]: {"--hidden": 256, "--drop": 0.2}}}
q = TC.plan(custom)
stage = next(s for s in q.manifest["stages"] if s["id"] == nn["id"])
results["options_applied"] = (TC._value(stage["argv"], "--epochs") == ["5"] and TC._value(stage["argv"], "--hidden") == ["256"]
                              and TC._value(stage["argv"], "--lr") == ["0.0005"] and stage["outputs"] == nn["outputs"])
names = q.settings["tune_out/selected_ns3.csv"].decode().split("\r\n")
results["selection_written"] = names[0] == "feature" and "car1_rank" not in names and len(names) - 2 == len(TC.read_names(ROOT / "tune_out/selected_ns3.csv")) - 1
xgb = json.loads(q.settings["tune_out/best_xgb5.json"])
results["xgb_written"] = xgb["iters"] == 400 and xgb["params"]["max_depth"] == 8 and xgb["params"]["learning_rate"] == json.loads((ROOT / "tune_out/best_xgb5.json").read_text())["params"]["learning_rate"]
users = {s["id"] for s in q.manifest["stages"] if "tune_out/best_xgb5.json" in s["inputs"]}
results["undeclared_readers_declared"] = "models_001_v6_lr3_xlg" in users and len(users) > 47
var = next(s for s in q.manifest["stages"] if s["id"] == "models_047_VAR_full_aug")
results["implicit_colfile_declared"] = "tune_out/selected_ns3.csv" in var["inputs"]
results["workers"] = q.workers == 3

# Guards.
expect_error("gpu_param", {"shared_settings": {"tune_out/best_xgb5.json": {"params": {"device": "cuda"}}}}, "cannot be changed")
expect_error("tree_method", {"shared_settings": {"tune_out/best_xgb5.json": {"params": {"tree_method": "gpu_hist"}}}}, "cannot be changed")
expect_error("structural_option", {"stage_options": {cv["id"]: {"--tag": "x"}}}, "not a reviewed")
expect_error("cols_option", {"stage_options": {cv["id"]: {"--blocks": "grid"}}}, "not a reviewed")
expect_error("unaccepted_option", {"stage_options": {cv["id"]: {"--epochs": 3}}}, "does not accept")
expect_error("seed_in_tag", {"stage_options": {"models_002_v6_lr3_xlg_s1": {"--seed": 5}}}, "names an output")
expect_error("seed_in_folder", {"stage_options": {"mechanism_outer_priors_7": {"--seeds": [8]}}}, "names an output")
expect_error("bad_range", {"stage_options": {nn["id"]: {"--drop": 1.5}}}, "outside")
expect_error("bad_type", {"stage_options": {nn["id"]: {"--hidden": 2.5}}}, "integer")
expect_error("unknown_stage", {"stage_options": {"models_999": {"--seed": 1}}}, "Unknown stage")
expect_error("unknown_key", {"device": "cuda"}, "Unknown config keys")
expect_error("workers_range", {"workers": 9}, "workers")
expect_error("remove_missing", {"feature_selections": {"tune_out/selected_ns3.csv": {"remove": ["not_a_feature"]}}}, "cannot remove")
expect_error("add_unknown", {"feature_selections": {"tune_out/selected_ns3.csv": {"add": ["made_up__x"]}}}, "unknown feature")
foreign = next(n for n in TC.feature_catalog() if n.startswith("lr5__"))
expect_error("add_foreign_block", {"feature_selections": {"tune_out/selected_ns1.csv": {"add": [foreign]}}}, "does not load blocks")
expect_error("not_a_list", {"feature_selections": {"tune_out/best_xgb5.json": {"add": ["car1_rank"]}}}, "not a packaged")
results["default_sha_stable"] = TC.plan(TC.template()).config_sha256 == TC.plan(None).config_sha256 == TC.plan(ROOT / "pipeline/hyperparameters.json").config_sha256
expect_error("empty_list", {"feature_selections": {"tune_out/loop/fam_gh.csv": {"remove": TC.read_names(ROOT / "tune_out/loop/fam_gh.csv")}}}, "cannot be empty")
broken = json.loads(json.dumps(base)); broken["stages"][5]["outputs"] = ["tune_out/x.npy"]
try:
    TC.verify_effective(base, broken); results["output_identity_guard"] = False
except ValueError:
    results["output_identity_guard"] = True
dropped = json.loads(json.dumps(base)); dropped["stages"].pop(3)
try:
    TC.verify_effective(base, dropped); results["no_stage_deletion"] = False
except ValueError:
    results["no_stage_deletion"] = True

# Workspace behaviour.
tmp = Path(tempfile.mkdtemp())
try:
    w = tmp / "fresh"; (w / "stages").mkdir(parents=True); (w / "CPU_REBUILD_WORKSPACE.json").write_text("{}")
    (w / "tune_out").mkdir(); shutil.copy2(ROOT / "tune_out/selected_ns3.csv", w / "tune_out/selected_ns3.csv")
    r = TC.prepare(w, custom)
    results["prepare_writes"] = (w / "tune_out/selected_ns3.csv").read_bytes() == q.settings["tune_out/selected_ns3.csv"] and r.manifest_path == w / TC.EFFECTIVE
    results["prepare_idempotent"] = TC.prepare(w, custom).config_sha256 == r.config_sha256
    try:
        TC.prepare(w, None); results["resume_mismatch_refused"] = False
    except ValueError:
        results["resume_mismatch_refused"] = True
    paused = tmp / "paused"; (paused / "stages").mkdir(parents=True); (paused / "CPU_REBUILD_WORKSPACE.json").write_text("{}")
    (paused / "stages/models_001.json").write_text(json.dumps({"fingerprint": {}}))
    before = sorted(p.name for p in paused.rglob("*"))
    d = TC.prepare(paused, None)
    results["paused_default_reused"] = d.default and d.manifest_path == TC.BASE_MANIFEST and sorted(p.name for p in paused.rglob("*")) == sorted(before + [TC.RECORD])
    try:
        TC.prepare(paused, custom); results["custom_on_started_refused"] = False
    except ValueError:
        results["custom_on_started_refused"] = True
finally:
    shutil.rmtree(tmp)

for k, v in results.items():
    print(("PASS " if v else "FAIL ") + k)
raise SystemExit(0 if all(results.values()) else 1)
