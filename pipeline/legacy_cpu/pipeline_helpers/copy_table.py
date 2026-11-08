"""Give a freshly generated table its second historical filename."""
import argparse
from pathlib import Path
import shutil
import ku_workspace

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("source")
    p.add_argument("target")
    a = p.parse_args()
    root = ku_workspace.root()
    source, target = [(root / value).resolve() for value in (a.source, a.target)]
    if any(not value.is_relative_to(root) or value == root for value in (source, target)):
        raise ValueError("Table must be inside the fresh workspace")
    if source == target:
        raise ValueError("Source and target must differ")
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
