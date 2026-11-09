"""run one frozen training script with xgboost and the torch nets on the gpu.

python -m pipeline.gpu_launch experiments/v6/train_cv.py --tag ...

the script file on disk is not changed (its hash stays the same), only at load time:
#xgboost.train / XGBModel get device=cuda
#nn_var / nn_xpu / nn_seq and the lr / lrz / garch / lm feature scripts get dev = "xpu" (arc) or "cuda" instead of dev = "cpu"
"""
import os
import re
import sys
from pathlib import Path

TORCH_GPU = {"nn_var.py", "nn_xpu.py", "nn_seq.py", "features_lr.py", "features_lrz.py", "features_garch.py", "features_lm.py"}
SHARED_DEV = {"features_lrz.py", "features_garch.py"}


def patch_xgboost():
    import xgboost as xgb
    original = xgb.train

    def train(params, *args, **kwargs):
        params = dict(params)
        params["device"] = "cuda"
        params["tree_method"] = "hist"
        return original(params, *args, **kwargs)

    xgb.train = train
    xgb.training.train = train
    get = xgb.XGBModel.get_xgb_params

    def get_xgb_params(self):
        p = get(self)
        p["device"] = "cuda"
        return p

    xgb.XGBModel.get_xgb_params = get_xgb_params
    #reloaded boosters predict on the gpu too, so the scripts' save/reload checks compare like with like
    load = xgb.Booster.load_model

    def load_model(self, fname):
        out = load(self, fname)
        self.set_param({"device": "cuda"})
        return out

    xgb.Booster.load_model = load_model


DEVICE_ARGS = (r'\bdev = "cpu"', r"""device=(['"])cpu\1""", r"""\.to\((['"])cpu\1\)""", r"""map_location=(['"])cpu\1""")


def swap_device(source, target):
    n = 0
    for pattern, repl in zip(DEVICE_ARGS, (f'dev = "{target}"', f"device='{target}'", f".to('{target}')", f"map_location='{target}'")):
        source, k = re.subn(pattern, repl, source)
        n += k
    return source, n


def swap_imports(root, target):
    #modules imported from the work folder load with the same device swap (the files on disk stay unchanged)
    import importlib.abc
    import importlib.machinery
    import importlib.util

    class Loader(importlib.machinery.SourceFileLoader):
        def get_code(self, fullname):
            src = Path(self.path).read_text(encoding="utf8")
            return compile(swap_device(src, target)[0], self.path, "exec")

    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path, target_=None):
            spec = importlib.machinery.PathFinder.find_spec(name, path)
            if spec is None or not spec.origin or not spec.origin.endswith(".py"):
                return spec
            origin = Path(spec.origin).resolve()
            if root not in origin.parents:
                return spec
            return importlib.util.spec_from_file_location(name, origin, loader=Loader(name, str(origin)),
                                                          submodule_search_locations=spec.submodule_search_locations)

    sys.meta_path.insert(0, Finder())


def main():
    script = Path(sys.argv[1]).resolve()
    sys.argv = [str(script), *sys.argv[2:]]
    sys.path.insert(0, str(script.parent))
    source = script.read_text(encoding="utf8")
    if script.name in TORCH_GPU and os.environ.get("KU_SBB_GPU") == "1":
        target = os.environ.get("KU_SBB_TORCH_DEV", "cuda")   #cuda = rtx 5070, xpu = intel arc
        source, n = re.subn(r'\bdev = "cpu"', f'dev = "{target}"', source)
        helper = script.parent / "features_lr.py"
        if script.name in SHARED_DEV and helper.is_file():
            #lrz / garch take their device from features_lr, so load that module with the same switch
            import importlib.util
            spec = importlib.util.spec_from_file_location("features_lr", helper)
            module = importlib.util.module_from_spec(spec)
            sys.modules["features_lr"] = module
            saved_argv = sys.argv
            code = re.sub(r'\bdev = "cpu"', f'dev = "{target}"', helper.read_text(encoding="utf8"))
            exec(compile(code, str(helper), "exec"), module.__dict__)
            sys.argv = saved_argv
            n += 1
        if n == 0:
            raise SystemExit(f"{script.name}: device line not found")
    elif os.environ.get("KU_SBB_DEVICE_SWAP") == "1":
        #late torch heads hard-code the cpu (device=, .to(), map_location=) in the script and in the helper
        #modules it imports from the work folder; swap all of them, results already come back with .cpu()
        target = os.environ.get("KU_SBB_TORCH_DEV", "cuda")
        source, n = swap_device(source, target)   #wrappers (n == 0) only import the torch code
        swap_imports(Path(os.environ.get("KU_SBB_ROOT", script.parent)).resolve(), target)
    try:
        patch_xgboost()
    except ImportError:
        pass
    code = compile(source, str(script), "exec")
    module = {"__name__": "__main__", "__file__": str(script), "__builtins__": __builtins__}
    exec(code, module)


if __name__ == "__main__":
    main()
