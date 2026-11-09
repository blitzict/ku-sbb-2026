"""Fit the trend870 net head, then replay and check it, as the historical package did.

trend_870.py --mode net writes weights.json, submission.csv and replay_inputs.npz; validation.json only comes from a second
run with --check after a replay to submission_replay.csv. The recipe lists validation.json as an output of this step, so
the three calls run here in order with the same fixed settings.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/weight_space"))
from threadpoolctl import threadpool_limits
import trend_870

if __name__ == "__main__":
    with threadpool_limits(limits=2):
        trend_870.run("net")
        rec = trend_870.ef.PACKAGE / "trend870_net_candidate/weights.json"
        trend_870.replay(rec, rec.parent / "submission_replay.csv")
        trend_870.run("net", rec)
