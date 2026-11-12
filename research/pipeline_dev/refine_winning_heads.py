"""Adapt only the winning readout path to fixed-recipe CPU retraining.

The original experimental sources remain untouched. Explicit settings below
come from the saved winning recipes (winning_head_chain.json). Statistical
checks and accepted flags are retained. With KU_SBB_REBUILD_FIXED=1, selection
uses the historical setting and saves newly fitted weights regardless of
whether the old search promotion threshold is crossed. This is NOT a claim
that the rebuilt model has reached the old score.
"""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "pipeline/legacy_cpu/experiments/weight_space"
FIXED = {
    "expert_fusion": {"chosen": "regime", "MODES": ("regime",)},
    "residual_trial": {"chosen": "tails", "MODES": ("tails",)},
    "seasonal_head": {"chosen": "local", "MODES": ("local",)},
    "mechanism_edit": {"chosen": 40., "penalties": [40.]},
    "trend_870": {"chosen": 60., "penalties": [60.]},
    "extend_readout": {"chosen": 35., "penalties": [35.]},
}
# These functions already have fixed parameter values. Only allow saving the
# fixed trained result without turning a measured failed criterion into True.
EXPORT_ONLY = {
    "hard_expert_trial", "white_loop", "white_nested", "white_joint",
    "white_extended", "white_consult", "white_stage", "white_local_rank_weights",
    "white_signal_package", "white_fold_vote", "white_concordance",
    "white_crop_replication", "white_regime_branches", "white_shared_residual",
}
MARKER = "# Fixed winning-recipe CPU adaptation v1"


class Adapt(ast.NodeTransformer):
    def __init__(self, name):
        self.settings = FIXED.get(name, {})
        self.changes = []

    def visit_Assign(self, node):
        self.generic_visit(node)
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            key = node.targets[0].id
            if key in self.settings:
                value = ast.parse(repr(self.settings[key]), mode="eval").body
                node.value = ast.Call(func=ast.Attribute(ast.Name("_rebuild", ast.Load()), "select", ast.Load()),
                                      args=[value, node.value], keywords=[])
                self.changes.append(f"fixed historical {key}={self.settings[key]!r}")
            elif key in {"report", "rep", "result"} and (
                    isinstance(node.value, ast.Dict) and any(isinstance(k, ast.Constant) and k.value in
                        {"scores", "accepted", "selected", "results"} for k in node.value.keys) or
                    isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name) and node.value.func.id == "dict"
                    and any(k.arg in {"scores", "accepted", "selected", "results"} for k in node.value.keywords)):
                node.value = ast.Call(func=ast.Attribute(ast.Name("_rebuild", ast.Load()), "annotate", ast.Load()),
                                      args=[node.value], keywords=[])
                self.changes.append("label recipe selection independently of validation")
        return node

    def visit_If(self, node):
        self.generic_visit(node)
        # Only the condition is changed. Assignments computing accepted,
        # passed, specialist and eligible are NEVER changed.
        allowed = {"accepted", "eligible", "passed", "ok", "specialist"}
        if not any(isinstance(x, ast.Name) and x.id in allowed for x in ast.walk(node.test)):
            return node
        # Export guards contain a file/submission write, or an early exit.
        body = ast.Module(body=node.body, type_ignores=[])
        text = ast.unparse(body)
        export_body = any(word in text for word in ("weights.json", "submission.csv", "write_text", "savez"))
        early_exit = len(node.body) == 1 and isinstance(node.body[0], (ast.Return, ast.Continue))
        if not export_body and not early_exit:
            return node
        class Condition(ast.NodeTransformer):
            def visit_Name(self, value):
                if value.id in allowed:
                    return ast.Call(func=ast.Attribute(ast.Name("_rebuild", ast.Load()), "export", ast.Load()),
                                    args=[value], keywords=[])
                return value
        node.test = Condition().visit(node.test)
        self.changes.append("export fixed recipe independently of measured promotion criterion")
        return node


def main():
    report = []
    for name in sorted(set(FIXED) | EXPORT_ONLY):
        path = FOLDER / (name + ".py")
        if not path.exists():
            # Never infer a similarly named producer. The reviewed inventory
            # records absent optional modules explicitly.
            report.append(dict(module=name, status="not present; no mutation"))
            continue
        source = path.read_text(encoding="utf-8")
        if MARKER in source:
            report.append(dict(module=name, status="already adapted"))
            continue
        tree = ast.parse(source)
        edit = Adapt(name)
        tree = edit.visit(tree)
        if not edit.changes:
            report.append(dict(module=name, status="no relevant export guard"))
            continue
        position = 1 if tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(tree.body[0].value, ast.Constant) else 0
        tree.body.insert(position, ast.Import(names=[ast.alias(name="rebuild_policy", asname="_rebuild")]))
        ast.fix_missing_locations(tree)
        result = MARKER + "\n" + ast.unparse(tree) + "\n"
        compile(result, str(path), "exec")
        path.write_text(result, encoding="utf-8", newline="\n")
        report.append(dict(module=name, status="adapted; full runtime check pending",
                           before_sha256=hashlib.sha256(source.encode()).hexdigest(),
                           after_sha256=hashlib.sha256(result.encode()).hexdigest(), changes=edit.changes))
    (ROOT / "pipeline/winning_head_port_changes.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
