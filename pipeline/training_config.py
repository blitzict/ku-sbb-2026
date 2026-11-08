"""Training hyperparameters for the fixed CPU recipe: validated config -> effective stage manifest.

The default config (no overrides) gives back the reviewed 473-stage manifest unchanged and leaves every packaged
settings file byte for byte as it is, so a paused default workspace resumes with all completed stages reused.

A config file (JSON, see pipeline/hyperparameters.json and pipeline/hyperparameters.template.json) may set:

  workers              CPU feature/fit worker count 1..6, or null to keep the caller's value. Speed only.
  feature_selections   {"tune_out/<packaged list>.csv": {"remove": [names], "add": [names]}}
                       Edits a packaged feature-name list. Removed names must be in that list. Added names must be
                       real feature names from the packaged catalog (feature_catalog()), and a prefixed name such as
                       "lrz__x" must come from a block that every train_cv/distill/nn_xpu stage using the list loads.
  shared_settings      {"tune_out/best_xgb5.json": {"iters": int, "params": {name: value}}}
                       XGBoost tree rounds and booster parameters shared by the train_cv xgb members. Device, tree
                       method, objective, metric, seed and thread settings cannot be changed.
  script_options       {"experiments/v6/nn_var.py": {"--epochs": 20, ...}}   applies to every stage of that script
  stage_options        {"models_085_...": {"--lr": 0.0005, ...}}             one stage; wins over script_options
                       Only reviewed training options (TUNABLE below) that the script's own argparse accepts.

Guards: no GPU options or environment changes; no stage is added, removed or reordered; scripts, outputs, output
tags and code inputs never change; an option whose current value names an output (for example --seeds 7 writing
law_outer7/, or --seed 1 in a *_s1 tag) cannot be changed for that stage; every stage that reads an overridden
settings file (in argv or anywhere in its code closure) gets that file declared as an input, so its fingerprint
covers the new bytes. A work folder records its config; resuming it with a different config is refused.

API for train_087910_cpu.py (after initialize() and sync(), before any stage runs):

    from pipeline.training_config import prepare
    plan = prepare(work, config_path_or_None, manifest_path)
    workers = plan.workers or args.workers
    for stage in plan.manifest["stages"]: ...run(work, stage)...
    validate(work, plan.manifest_path)

prepare() validates, refuses a mismatched resume, writes overridden settings files into the workspace and returns
the effective manifest. Command line: python -m pipeline.training_config --check CONFIG | --template OUT |
--catalog [LIST] | --effective CONFIG --out FILE.
"""
import argparse
import ast
import copy
import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

from pipeline.run_stages import ROOT, digest, inside

BASE_MANIFEST = ROOT / "pipeline/full_cpu_stages.json"
LEGACY = ROOT / "pipeline/legacy_cpu"
RECORD = "TRAINING_CONFIG.json"
EFFECTIVE = "effective_stages.json"
TOP_KEYS = {"workers", "feature_selections", "shared_settings", "script_options", "stage_options"}

# Reviewed training options: (kind, minimum, maximum). Structural options (--tag, --cols, --blocks, --model,
# --extra, --aug, --kinds, --views, --part, --out, ...) are deliberately absent.
TUNABLE = {
    "--seed": ("int", 0, 2**31 - 1), "--seeds": ("int", 0, 2**31 - 1), "--seed-offset": ("int", 0, 10**6),
    "--fold-seed": ("int", 0, 2**31 - 1),
    "--epochs": ("int", 1, 10**4), "--patience": ("int", 1, 10**4), "--lr": ("float", 1e-7, 1.0),
    "--wd": ("float", 0.0, 10.0), "--bs": ("int", 1, 10**6), "--hidden": ("int", 1, 65536),
    "--layers": ("int", 1, 64), "--drop": ("float", 0.0, 0.95), "--ch": ("int", 1, 4096), "--k": ("int", 1, 255),
    "--ema": ("float", 0.0, 0.99999), "--pair-w": ("float", 0.0, 100.0),
    "--rounds": ("int", 1, 10**6), "--cat-iters": ("int", 1, 10**6), "--colsample": ("float", 1e-3, 1.0),
    "--refit-mult": ("float", 0.05, 20.0), "--alpha": ("float", 0.0, 1.0), "--aug-weight": ("float", 0.0, 100.0),
    "--extra-w": ("float", 0.0, 100.0), "--crop-p": ("float", 0.0, 1.0), "--crop-max": ("float", 0.0, 1.0),
}
# Seed-count options mean "how many seeds" in these scripts, but "which seeds" (a list) where argparse says nargs.
XGB_PARAMS = {"learning_rate": (1e-5, 1.0), "max_depth": (1, 32), "min_child_weight": (0.0, 1e4),
              "subsample": (1e-3, 1.0), "colsample_bytree": (1e-3, 1.0), "colsample_bynode": (1e-3, 1.0),
              "gamma": (0.0, 1e4), "reg_lambda": (0.0, 1e5), "reg_alpha": (0.0, 1e5), "max_bin": (2, 4096)}
SHARED = {"tune_out/best_xgb5.json"}
# Consumers that drop unknown names silently, so added names must come from blocks they load.
BLOCK_CONSUMERS = {"train_cv.py", "distill.py", "nn_xpu.py"}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def load_base(manifest=BASE_MANIFEST):
    return json.loads(Path(manifest).read_text())


def selection_files(base):
    return sorted(p for p in base["prerequisite_inputs"] if p.startswith("tune_out/") and p.endswith(".csv"))


def read_names(path):
    lines = Path(path).read_bytes().decode("utf8").splitlines()
    if not lines or lines[0].strip() != "feature":
        raise ValueError(f"{path}: expected a one-column CSV with header 'feature'")
    return [line.strip() for line in lines[1:] if line.strip()]


def feature_catalog(base=None):
    """Every feature name in the packaged lists, with the lists it appears in."""
    base = base or load_base()
    catalog = {}
    for relative in selection_files(base):
        for name in read_names(ROOT / relative):
            catalog.setdefault(name, []).append(relative)
    return catalog


def script_options(script):
    """Options the script's argparse accepts: {option: {"type": int|float|str|None, "nargs": ..., "action": ...}}."""
    tree = ast.parse((LEGACY / script).read_text(encoding="utf8"))
    found = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "add_argument":
            names = [a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str)]
            info = {k.arg: (k.value.id if isinstance(k.value, ast.Name) else
                            k.value.value if isinstance(k.value, ast.Constant) else ast.unparse(k.value))
                    for k in node.keywords if k.arg in ("type", "nargs", "action")}
            for name in names:
                if name.startswith("--"):
                    found[name] = info
    return found


def _option_span(argv, option):
    """(start, end) of option and its values in argv, or None. Values run until the next --option."""
    if option not in argv:
        return None
    if argv.count(option) != 1:
        raise ValueError(f"Option {option} appears more than once")
    start = argv.index(option)
    end = start + 1
    while end < len(argv) and not argv[end].startswith("--"):
        end += 1
    return start, end


def _format(option, value, info):
    kind, low, high = TUNABLE[option]
    many = info.get("nargs") in ("*", "+")
    values = value if isinstance(value, list) else [value]
    if isinstance(value, list) and not many:
        raise ValueError(f"{option} takes one value here")
    if not values:
        raise ValueError(f"{option} needs at least one value")
    out = []
    for v in values:
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(f"{option} must be numeric, got {v!r}")
        if kind == "int" and (not isinstance(v, int)):
            raise ValueError(f"{option} must be an integer, got {v!r}")
        if not math.isfinite(v) or not low <= v <= high:
            raise ValueError(f"{option}={v} is outside [{low}, {high}]")
        out.append(str(int(v)) if kind == "int" else repr(float(v)))
    if info.get("type") not in (("int",) if kind == "int" else ("float", "int")) and info.get("type") is not None:
        raise ValueError(f"{option}: script type {info.get('type')} does not match {kind}")
    return out


def _names_output(stage, values):
    for v in values:
        pattern = re.compile(rf"(?<![0-9.]){re.escape(v)}(?![0-9])")
        if any(pattern.search(Path(o).name) or pattern.search(Path(o).parent.name) for o in stage["outputs"]):
            return True
    return False


def apply_options(stage, options):
    """Return a new argv with the reviewed options set; raise on anything unreviewed or identity-changing."""
    argv = list(stage["argv"])
    accepted = script_options(argv[0])
    for option, value in options.items():
        if option not in TUNABLE:
            raise ValueError(f"{stage['id']}: {option} is not a reviewed training option")
        if option not in accepted:
            raise ValueError(f"{stage['id']}: {argv[0]} does not accept {option}")
        info = accepted[option]
        if info.get("action") in ("store_true", "store_false"):
            raise ValueError(f"{stage['id']}: {option} is a switch, not a hyperparameter")
        new = _format(option, value, info)
        span = _option_span(argv, option)
        if span:
            current = argv[span[0] + 1:span[1]]
            if current != new and _names_output(stage, current):
                raise ValueError(f"{stage['id']}: {option} {' '.join(current)} names an output; it cannot change")
            argv[span[0]:span[1]] = [option, *new]
        else:
            argv += [option, *new]
    return argv


def _blocks(stage):
    span = _option_span(stage["argv"], "--blocks")
    if span:
        return set(stage["argv"][span[0] + 1].split(","))
    accepted = script_options(stage["argv"][0]).get("--blocks")
    default = {"train_cv.py": "grid,grid2", "nn_xpu.py": "grid,grid2", "distill.py": "grid,grid2,dd,lr"}
    return set(default[Path(stage["argv"][0]).name].split(",")) if accepted else set()


def _users(base, relative):
    """Stages that read a settings file: named in argv, declared, or mentioned anywhere in their code closure."""
    name = Path(relative).name
    texts = {}
    users = []
    for stage in base["stages"]:
        hit = relative in stage["inputs"] or any(relative.split("/", 1)[1] in a or a == relative for a in stage["argv"][1:])
        if not hit:
            for code in [stage["argv"][0], *stage["code_inputs"]]:
                path = ROOT / "pipeline/features_v4.py" if code == "clean/features.py" else LEGACY / code
                if path.suffix == ".py":
                    if path not in texts:
                        texts[path] = path.read_text(encoding="utf8")
                    if name in texts[path]:
                        hit = True
                        break
        if hit:
            users.append(stage)
    return users


def _check_selection(base, catalog, relative, edit):
    if relative not in selection_files(base):
        raise ValueError(f"{relative} is not a packaged feature-name list")
    if set(edit) - {"remove", "add"}:
        raise ValueError(f"{relative}: only 'remove' and 'add' are allowed")
    remove, add = list(edit.get("remove", [])), list(edit.get("add", []))
    if not all(isinstance(n, str) for n in remove + add):
        raise ValueError(f"{relative}: feature names must be strings")
    if len(set(remove)) != len(remove) or len(set(add)) != len(add) or set(remove) & set(add):
        raise ValueError(f"{relative}: duplicate or contradictory names")
    names = read_names(ROOT / relative)
    missing = [n for n in remove if n not in names]
    if missing:
        raise ValueError(f"{relative}: cannot remove names not in the list: {missing[:5]}")
    present = [n for n in add if n in names]
    if present:
        raise ValueError(f"{relative}: already selected: {present[:5]}")
    unknown = [n for n in add if n not in catalog]
    if unknown:
        raise ValueError(f"{relative}: unknown feature names (not in the packaged catalog): {unknown[:5]}")
    for stage in _users(base, relative):
        if Path(stage["argv"][0]).name in BLOCK_CONSUMERS:
            blocks = _blocks(stage)
            foreign = [n for n in add if "__" in n and n.split("__")[0] not in blocks]
            if foreign:
                raise ValueError(f"{relative}: {stage['id']} does not load blocks for {foreign[:5]}")
    result = [n for n in names if n not in set(remove)] + add
    if not result:
        raise ValueError(f"{relative}: a feature list cannot be empty")
    return result


def _check_shared(relative, edit):
    if relative not in SHARED:
        raise ValueError(f"{relative} is not an editable shared settings file")
    if set(edit) - {"iters", "params"}:
        raise ValueError(f"{relative}: only 'iters' and 'params' are editable")
    data = json.loads((ROOT / relative).read_text())
    if "iters" in edit:
        v = edit["iters"]
        if isinstance(v, bool) or not isinstance(v, int) or not 1 <= v <= 10**6:
            raise ValueError(f"{relative}: iters must be an integer in [1, 1000000]")
        data["iters"] = v
    for key, v in edit.get("params", {}).items():
        if key not in XGB_PARAMS:
            raise ValueError(f"{relative}: parameter {key} cannot be changed (device, method, objective, seed and threads are fixed)")
        low, high = XGB_PARAMS[key]
        integer = key in ("max_depth", "max_bin")
        if isinstance(v, bool) or not isinstance(v, int if integer else (int, float)) or not math.isfinite(v) or not low <= v <= high:
            raise ValueError(f"{relative}: {key}={v!r} must be {'an integer' if integer else 'a number'} in [{low}, {high}]")
        data["params"][key] = v
    return data


@dataclass
class Plan:
    config: dict
    config_sha256: str
    default: bool
    workers: int | None
    manifest: dict
    manifest_path: Path
    settings: dict          # relative path -> bytes to write in the workspace
    changed_stages: list


def normalize(config):
    config = {k: v for k, v in (config or {}).items() if not k.startswith("_")}
    unknown = set(config) - TOP_KEYS
    if unknown:
        raise ValueError(f"Unknown config keys: {sorted(unknown)}")
    workers = config.get("workers")
    if workers is not None and (isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 6):
        raise ValueError("workers must be an integer 1..6 or null")
    clean = {"workers": workers}
    for key in ("feature_selections", "shared_settings", "script_options", "stage_options"):
        value = config.get(key) or {}
        if not isinstance(value, dict):
            raise ValueError(f"{key} must be an object")
        # Empty edits ({"remove": [], "add": []}, {}) are the default and change nothing.
        clean[key] = {k: v for k, v in value.items() if v and (not isinstance(v, dict) or any(v.values()))}
    return clean


def plan(config=None, manifest=BASE_MANIFEST):
    """Validate a config (dict, path or None) and build the effective manifest. Writes nothing."""
    if isinstance(config, (str, Path)):
        config = json.loads(Path(config).read_text())
    config = normalize(config)
    base = load_base(manifest)
    stages = base["stages"]
    by_id = {s["id"]: s for s in stages}
    effective = copy.deepcopy(base)
    settings, extra_inputs = {}, {}

    catalog = feature_catalog(base) if config["feature_selections"] else {}
    for relative, edit in config["feature_selections"].items():
        names = _check_selection(base, catalog, relative, edit)
        settings[relative] = ("feature\r\n" + "".join(n + "\r\n" for n in names)).encode()
    for relative, edit in config["shared_settings"].items():
        settings[relative] = json.dumps(_check_shared(relative, edit), indent=1).encode()
    for relative in settings:
        for stage in _users(base, relative):
            extra_inputs.setdefault(stage["id"], []).append(relative)

    scripts = {s["argv"][0] for s in stages}
    for script in config["script_options"]:
        if script not in scripts:
            raise ValueError(f"No stage runs {script}")
    for ident in config["stage_options"]:
        if ident not in by_id:
            raise ValueError(f"Unknown stage {ident}")
    changed = []
    for stage, new in zip(stages, effective["stages"]):
        options = {**config["script_options"].get(stage["argv"][0], {}), **config["stage_options"].get(stage["id"], {})}
        if options:
            new["argv"] = apply_options(stage, options)
        for relative in extra_inputs.get(stage["id"], []):
            if relative not in new["inputs"]:
                new["inputs"].append(relative)
        if new != stage:
            changed.append(stage["id"])
    verify_effective(base, effective)
    default = not settings and not changed
    sha = _sha(_canonical(config))
    if not default:
        effective["training_config"] = dict(sha256=sha, changed_stages=len(changed),
                                            overridden_settings=sorted(settings), status="unmeasured custom settings")
    return Plan(config, sha, default, config["workers"], base if default else effective,
                Path(manifest), settings, changed)


def verify_effective(base, effective):
    """Structural identity: same stages in the same order, same scripts, outputs, code and env; inputs only grow."""
    a, b = base["stages"], effective["stages"]
    if [s["id"] for s in a] != [s["id"] for s in b]:
        raise ValueError("Stages cannot be added, removed or reordered")
    for old, new in zip(a, b):
        if set(new) != set(old):
            raise ValueError(f"{old['id']}: stage fields changed")
        for key in ("outputs", "code_inputs", "env"):
            if new.get(key) != old.get(key):
                raise ValueError(f"{old['id']}: {key} cannot change")
        if new["argv"][0] != old["argv"][0]:
            raise ValueError(f"{old['id']}: script cannot change")
        if not set(old["inputs"]) <= set(new["inputs"]):
            raise ValueError(f"{old['id']}: declared inputs cannot be removed")
        if not set(new["inputs"]) - set(old["inputs"]) <= set(base["prerequisite_inputs"]):
            raise ValueError(f"{old['id']}: only packaged settings files can be added as inputs")
        for token in new["argv"][1:]:   # the script name itself (e.g. nn_xpu.py, a CPU port) is fixed above
            if token not in old["argv"] and re.search(r"cuda|xpu|gpu", token, re.I):
                raise ValueError(f"{old['id']}: GPU options are not allowed")
        for option in ("--tag", "--out", "--run-name", "--prefix", "--source"):
            if _value(old["argv"], option) != _value(new["argv"], option):
                raise ValueError(f"{old['id']}: {option} cannot change")


def _value(argv, option):
    span = _option_span(argv, option)
    return argv[span[0] + 1:span[1]] if span else None


def prepare(work, config=None, manifest=BASE_MANIFEST):
    """Validate, guard the workspace, write overridden settings into it and return the Plan.

    Call after initialize() and sync() (sync re-copies the packaged lists) and before any stage runs. The default
    config returns the base manifest object and path unchanged and writes no settings file."""
    work = Path(work).resolve()
    if not (work / "CPU_REBUILD_WORKSPACE.json").is_file():
        raise ValueError("A marked CPU rebuild workspace is required")
    result = plan(config, manifest)
    record_path = work / RECORD
    record = dict(config_sha256=result.config_sha256, default=result.default, config=result.config,
                  base_manifest_sha256=digest(Path(manifest)), changed_stages=result.changed_stages,
                  settings_sha256={k: hashlib.sha256(v).hexdigest() for k, v in result.settings.items()})
    if record_path.exists():
        old = json.loads(record_path.read_text())
        if (old["config_sha256"], old["base_manifest_sha256"]) != (record["config_sha256"], record["base_manifest_sha256"]):
            raise ValueError("This work folder was prepared with different training settings; use a new work folder")
    else:
        started = [p for p in (work / "stages").glob("*.json")
                   if isinstance(json.loads(p.read_text()), dict) and "fingerprint" in json.loads(p.read_text())]
        if started and not result.default:
            raise ValueError("Custom training settings need a fresh work folder; this one already has completed stages")
    for relative, data in result.settings.items():
        target = inside(work, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or target.read_bytes() != data:
            temporary = target.with_name(target.name + ".tmp")
            temporary.write_bytes(data)
            temporary.replace(target)
    if not result.default:
        path = work / EFFECTIVE
        text = json.dumps(result.manifest, indent=2)
        if path.exists() and path.read_text() != text:
            raise ValueError("The recorded effective manifest differs; use a new work folder")
        path.write_text(text)
        result.manifest_path = path
    if not record_path.exists():
        record_path.write_text(json.dumps(record, indent=2))
    return result


def template(base=None):
    """A complete, documented config whose values are the defaults (so it changes nothing as written)."""
    base = base or load_base()
    reference = {}
    for stage in base["stages"]:
        accepted = script_options(stage["argv"][0])
        current = {}
        for option in TUNABLE:
            if option in accepted and accepted[option].get("action") not in ("store_true", "store_false"):
                value = _value(stage["argv"], option)
                if value is not None:
                    locked = _names_output(stage, value)
                    current[option] = " ".join(value) + (" (names an output; locked)" if locked else "")
        if current:
            reference[stage["id"]] = current
    xgb = json.loads((ROOT / "tune_out/best_xgb5.json").read_text())
    return {
        "_about": "Defaults reproduce the reviewed recipe exactly. Every edit gives an unmeasured model; a work "
                  "folder keeps the settings it started with. Keys starting with '_' are ignored.",
        "workers": None,
        "feature_selections": {relative: {"remove": [], "add": []} for relative in selection_files(base)},
        "shared_settings": {"tune_out/best_xgb5.json": {}},
        "script_options": {},
        "stage_options": {},
        "_feature_list_sizes": {relative: len(read_names(ROOT / relative)) for relative in selection_files(base)},
        "_best_xgb5_defaults": {"iters": xgb["iters"], "params": {k: xgb["params"][k] for k in XGB_PARAMS}},
        "_tunable_options": {k: f"{v[0]} in [{v[1]}, {v[2]}]" for k, v in TUNABLE.items()},
        "_current_stage_options": reference,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", type=Path, help="Validate a config and summarize its effect")
    group.add_argument("--template", type=Path, help="Write a documented default config")
    group.add_argument("--catalog", nargs="?", const="", help="List feature names (all, or one packaged list)")
    group.add_argument("--effective", type=Path, help="Write the effective manifest for a config to --out")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.template:
        args.template.write_text(json.dumps(template(), indent=1) + "\n")
        print(f"Wrote {args.template}")
    elif args.catalog is not None:
        names = read_names(ROOT / args.catalog) if args.catalog else sorted(feature_catalog())
        print("\n".join(names))
    else:
        result = plan(args.check or args.effective)
        if args.effective:
            if not args.out:
                parser.error("--effective needs --out")
            args.out.write_text(json.dumps(result.manifest, indent=2))
        print(json.dumps(dict(default=result.default, config_sha256=result.config_sha256, workers=result.workers,
                              changed_stages=result.changed_stages, overridden_settings=sorted(result.settings)), indent=1))


if __name__ == "__main__":
    main()
