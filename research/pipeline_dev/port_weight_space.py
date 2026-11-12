"""Create CPU source copies of the historical mechanism/weight-space code.

Original files are never edited. The AST transformation changes device
selection and disables accelerator autocast, preserving numerical expressions,
model settings and feature definitions. This is a port, not an assertion that
every historical experiment is part of the final rebuilt model.

All ported data roots require an explicitly marked fresh rebuild workspace.
Numerical and integration smoke tests are recorded separately.
"""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/"experiments/weight_space"
DESTINATION = ROOT/"pipeline/legacy_cpu/experiments/weight_space"

HELPER = '''
import contextlib as _cpu_context
import os as _cpu_os
from pathlib import Path as _CpuPath

def _cpu_workspace():
    value = _cpu_os.environ.get("KU_SBB_ROOT")
    if not value:
        raise RuntimeError("Set KU_SBB_ROOT to the fresh CPU rebuild workspace")
    root = _CpuPath(value).resolve()
    if not (root / "CPU_REBUILD_WORKSPACE.json").is_file():
        raise RuntimeError("CPU ports require a marked rebuild workspace")
    return root
'''


def name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return name(node.value)+"."+node.attr
    return ""


class CPU(ast.NodeTransformer):
    def __init__(self):
        self.changes = []

    def visit_Constant(self, node):
        if isinstance(node.value, str) and node.value in {"xpu", "cuda", "xpu:0", "cuda:0"}:
            self.changes.append("device:"+node.value)
            return ast.copy_location(ast.Constant("cpu"), node)
        return node

    def visit_Assign(self, node):
        if (any(isinstance(t, ast.Name) and t.id == "ROOT" for t in node.targets)
                and "__file__" in ast.unparse(node.value)):
            self.changes.append("workspace_root")
            node.value = ast.Call(func=ast.Name(id="_cpu_workspace", ctx=ast.Load()), args=[], keywords=[])
        return self.generic_visit(node)

    def visit_Assert(self, node):
        if isinstance(node.test, ast.Call) and name(node.test.func) == "torch.xpu.is_available":
            self.changes.append("remove_accelerator_requirement")
            return None
        return self.generic_visit(node)

    def visit_Call(self, node):
        callee = name(node.func)
        if callee in {"torch.autocast", "torch.amp.autocast", "torch.xpu.amp.autocast"}:
            self.changes.append("disable_accelerator_autocast")
            return ast.copy_location(ast.Call(
                func=ast.Attribute(ast.Name("_cpu_context", ast.Load()), "nullcontext", ast.Load()),
                args=[], keywords=[]), node)
        if callee in {"torch.xpu.empty_cache", "torch.xpu.synchronize"}:
            self.changes.append("remove_accelerator_cache_call")
            return ast.copy_location(ast.Constant(None), node)
        if callee == "torch.xpu.manual_seed_all":
            self.changes.append("cpu_random_seed")
            node.func = ast.Attribute(ast.Name("torch", ast.Load()), "manual_seed", ast.Load())
        return self.generic_visit(node)


def main():
    DESTINATION.mkdir(parents=True, exist_ok=True)
    manifest = []
    for path in sorted(SOURCE.glob("*.py")):
        original = path.read_bytes()
        tree = ast.parse(original.decode("utf-8-sig"))
        edit = CPU()
        tree = edit.visit(tree)
        offset = 0
        if tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant):
            offset = 1
        while offset < len(tree.body) and isinstance(tree.body[offset], ast.ImportFrom) and tree.body[offset].module == "__future__":
            offset += 1
        tree.body[offset:offset] = ast.parse(HELPER).body
        ast.fix_missing_locations(tree)
        source_hash = hashlib.sha256(original).hexdigest()
        content = (f"# CPU port; original experiments/weight_space/{path.name}, SHA256 {source_hash}\n"
                   "# Float32 CPU operations replace accelerator autocast. Fresh workspace required.\n"
                   + ast.unparse(tree)+"\n")
        compile(content, str(path), "exec")
        forbidden = [name(n) for n in ast.walk(tree) if isinstance(n, ast.Attribute)
                     and name(n).startswith(("torch.xpu", "torch.cuda"))]
        if forbidden:
            raise RuntimeError(f"Unported GPU APIs in {path.name}: {forbidden}")
        target = DESTINATION/path.name
        target.write_text(content, encoding="utf-8")
        manifest.append(dict(file=str(target.relative_to(ROOT)).replace("\\", "/"),
                             source_sha256=source_hash,
                             port_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                             changes=edit.changes, compiled=True, fully_tested=False))
    output = ROOT/"pipeline/weight_space_port_manifest.json"
    output.write_text(json.dumps(manifest, indent=2))
    print(f"Created {len(manifest)} CPU source copies; original files unchanged.")
    print("Compilation and GPU-API scans passed; runtime tests are separate.")


if __name__ == "__main__":
    main()
