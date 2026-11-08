"""Read definitions out of the frozen training scripts without running them.

parse_args() rebuilds a script's own argparse parser (only the parser statements run) and parses a stage argv, so
defaults are the trained ones. definitions() executes selected top-level functions, classes and assignments in a
namespace the caller provides (paths pointing at the inference folder, the new test IDs, ...).
"""
import argparse
import ast
import hashlib
import sys
from pathlib import Path

from pipeline.run_stages import ROOT

LEGACY = ROOT / "pipeline/legacy_cpu"
if str(LEGACY) not in sys.path:
    sys.path.insert(0, str(LEGACY))
import ku_workspace  # noqa: E402  (only threads()/jobs() style helpers are used, no workspace access)

_TREES = {}


def tree(script):
    path = LEGACY / script
    if path not in _TREES:
        _TREES[path] = ast.parse(path.read_text(encoding="utf8"), str(path))
    return _TREES[path]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _parser_name(node):
    if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
        f = node.value.func
        if isinstance(f, ast.Attribute) and f.attr == "ArgumentParser" and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            return node.targets[0].id
    return None


def parse_args(script, argv):
    """argparse.Namespace for argv using the script's own top-level parser statements."""
    body, name = [], None
    for node in tree(script).body:
        found = _parser_name(node)
        if found:
            name = found
            body.append(node)
        elif (name and isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
              and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == "add_argument"
              and isinstance(node.value.func.value, ast.Name) and node.value.func.value.id == name):
            body.append(node)
    if not name:
        raise ValueError(f"{script}: no top-level argparse parser")
    namespace = {"argparse": argparse, "KU_ENV": ku_workspace, "env": ku_workspace}
    exec(compile(ast.Module(body=body, type_ignores=[]), str(LEGACY / script), "exec"), namespace)
    return namespace[name].parse_args(list(argv))


def definitions(script, names, namespace):
    """Execute the named top-level defs/classes/assignments of script into namespace (in source order)."""
    wanted, body = set(names), []
    for node in tree(script).body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in wanted:
            body.append(node)
        elif isinstance(node, ast.Assign) and all(isinstance(t, ast.Name) and t.id in wanted for t in node.targets):
            body.append(node)
        elif (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute)
              and isinstance(node.value.func.value, ast.Name) and node.value.func.value.id in wanted
              and node.value.func.attr in ("update", "append", "extend")):
            body.append(node)          # e.g. PROB.update({...}) right after PROB = {...}
    got = {n.name if isinstance(n, (ast.FunctionDef, ast.ClassDef)) else n.targets[0].id for n in body if not isinstance(n, ast.Expr)}
    missing = wanted - got
    if missing:
        raise ValueError(f"{script}: definitions not found: {sorted(missing)}")
    exec(compile(ast.Module(body=body, type_ignores=[]), str(LEGACY / script), "exec"), namespace)
    return namespace


def require(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Required saved artifact is missing (no fitting fallback): {path}")
    return path
