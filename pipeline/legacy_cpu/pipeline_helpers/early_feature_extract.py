"""Run only the feature-extraction step of four weight-space scripts.

Their own main() goes on to train a late white_loop head, which needs the full ensemble chain. These tables are inputs to that chain, so the
early stage must extract them alone. The extraction functions are called unchanged.
usage: early_feature_extract.py shape_counts|deconv_pulse_measure|conditional_scores|trend_population
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/weight_space"))
from threadpoolctl import threadpool_limits

if __name__ == "__main__":
    which = sys.argv[1]
    with threadpool_limits(limits=1):
        for part in ("train", "test"):
            if which == "shape_counts":
                import white_shapes; white_shapes.data(part)
            elif which == "deconv_pulse_measure":
                import white_deconv; white_deconv.measurements(part, "pulse")
            elif which == "conditional_scores":
                import white_influence; white_influence.extract(part)
            elif which == "trend_population":
                import white_trend_prior; white_trend_prior.extract(part)
            else:
                raise SystemExit(f"unknown table {which}")
            print(which, part, "done", flush=True)
