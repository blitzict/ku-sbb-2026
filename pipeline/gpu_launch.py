"""run one frozen training script with xgboost and the torch nets on the gpu.

python -m pipeline.gpu_launch experiments/v6/train_cv.py --tag ...

the script file on disk is not changed (its hash stays the same), only at load time:
#xgboost.train / XGBModel get device=cuda
#nn_var / nn_xpu / nn_seq get dev = "xpu" (arc) or "cuda" instead of dev = "cpu"
"""
import os
import re
import sys
from pathlib import Path

TORCH_GPU = {"nn_var.py", "nn_xpu.py", "nn_seq.py"}


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


def main():
    script = Path(sys.argv[1]).resolve()
    sys.argv = [str(script), *sys.argv[2:]]
    sys.path.insert(0, str(script.parent))
    source = script.read_text(encoding="utf8")
    if script.name in TORCH_GPU and os.environ.get("KU_SBB_GPU") == "1":
        target = os.environ.get("KU_SBB_TORCH_DEV", "cuda")   #cuda = rtx 5070, xpu = intel arc
        source, n = re.subn(r'\bdev = "cpu"', f'dev = "{target}"', source)
        if n == 0:
            raise SystemExit(f"{script.name}: device line not found")
    try:
        patch_xgboost()
    except ImportError:
        pass
    code = compile(source, str(script), "exec")
    module = {"__name__": "__main__", "__file__": str(script), "__builtins__": __builtins__}
    exec(code, module)


if __name__ == "__main__":
    main()
