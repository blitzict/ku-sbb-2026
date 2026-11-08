"""Write a reviewed names-only ensemble configuration inside the workspace."""
import argparse
import json
import ku_workspace

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("path")
p.add_argument("json")
a = p.parse_args()
root = ku_workspace.root()
path = (root / a.path).resolve()
if not path.is_relative_to(root) or path.suffix != ".json":
    raise ValueError("Configuration must be a JSON file inside the workspace")
value = json.loads(a.json)
if set(value) != {"members"} or not all(isinstance(x, str) for x in value["members"]):
    raise ValueError("Only a list of member names is imported")
if len(value["members"]) != len(set(value["members"])):
    raise ValueError("Duplicate member names")
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(value, indent=2))
