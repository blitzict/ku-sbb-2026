"""Copy reviewed CPU source and fixed settings into a marked rebuild workspace.

This copies code and feature-name lists, never old features or predictions.
Completed stage records protect code that has already been used.
"""
import argparse
import json
from pathlib import Path
import shutil

from pipeline.run_stages import ROOT, digest, inside


def sync(work):
    work = work.resolve()
    if work == ROOT or not (work / "CPU_REBUILD_WORKSPACE.json").is_file():
        raise ValueError("Use a marked fresh rebuild workspace")
    frozen = {}
    for record in (work / "stages").glob("*.json"):
        value = json.loads(record.read_text())
        if isinstance(value, dict):
            frozen.update(value.get("fingerprint", {}).get("code", {}))
    pairs = []
    base = ROOT / "pipeline/legacy_cpu"
    for folder in ("experiments", "pipeline_helpers"):
        for source in (base / folder).rglob("*.py"):
            pairs.append((source, source.relative_to(base).as_posix(), "source"))
    # Shared modules live at the snapshot root (including bounded-memory
    # rank preprocessing). Copy Python source only, never caches or weights.
    pairs.extend((source, source.name, "source") for source in base.glob("*.py"))
    # Fixed selections are disclosed as historical selections, not newly
    # fold-nested selection. Never import adv_ptest_train.npy (old row weights).
    for name in ("selected_ns1.csv", "selected_ns1_sel1.csv", "selected_ns1_sel2.csv", "selected_ns2.csv", "selected_ns_max.csv"):
        pairs.append((ROOT / "tune_out" / name, "tune_out/" + name, "feature_names"))
    records = []
    for source, relative, kind in pairs:
        sha = digest(source)
        if relative in frozen and frozen[relative] != sha:
            raise ValueError(f"Source changed after a completed stage: {relative}")
        target = inside(work, relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists() or digest(target) != sha:
            shutil.copy2(source, target)
        records.append(dict(path=relative, sha256=sha, kind=kind))
    (work / "stages/source_snapshot.json").write_text(json.dumps(records, indent=2))
    print(f"Synchronized {len(records)} source/settings files; no historical predictions imported")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    sync(parser.parse_args().work)
