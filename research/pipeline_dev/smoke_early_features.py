"""Exercise newly connected feature producers on the existing fresh 80/16 data."""
import json
import shutil
from pipeline.run_stages import ROOT, run, digest


def main():
    work = ROOT / "cpu_088_rebuild/feature_graph_smoke"
    for filename in ("early_feature_extract.py", "replay_trend_sequence.py"):
        source = ROOT / "pipeline/legacy_cpu/pipeline_helpers" / filename
        target = work / "pipeline_helpers" / filename
        if target.exists() and digest(target) != digest(source):
            raise ValueError(f"Existing helper changed: {filename}")
        if not target.exists():
            shutil.copy2(source, target)
    stages = json.loads((ROOT / "pipeline/early_head_stages.json").read_text())["stages"]
    for stage in stages:
        if stage["id"].startswith("early_feat_"):
            run(work, stage)
    print("Early feature wiring passed on freshly regenerated 80 train / 16 test series", flush=True)


if __name__ == "__main__":
    main()
