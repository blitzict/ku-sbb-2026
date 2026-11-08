"""Run periodic_replay for the trend streams. Its command line lists only some prefixes; run() handles trend the same way.

Writes tune_out/weight_space/trend_seq_{train,test}.parquet and trend_replay/fold*.npz.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments/weight_space"))
from threadpoolctl import threadpool_limits
import periodic_replay

if __name__ == "__main__":
    with threadpool_limits(limits=2):
        periodic_replay.run("trend")
