"""Optional offline guard, activated only by pipeline.benchmark_cpu.

CPython audit hooks reject Python socket connections and name resolution in
the benchmark and its spawned Python workers. This is an application-level
guard, not an operating-system firewall for arbitrary native programs.
"""
import os
import sys
from pathlib import Path

if os.environ.get("KU_SBB_OFFLINE") == "1":
    def reject_network(event, args):
        if event in {"socket.connect", "socket.getaddrinfo", "socket.gethostbyname",
                     "socket.gethostbyaddr", "socket.sendto"}:
            raise PermissionError("Network access is disabled for this CPU benchmark")
    sys.addaudithook(reject_network)

if os.environ.get("KU_SBB_RAW_ONLY") == "1":
    original = Path(os.environ["KU_SBB_ORIGINAL_ROOT"]).resolve()
    workspace = Path(os.environ["KU_SBB_ROOT"]).resolve()
    forbidden = [original/name for name in
                 ("tune_out", "new weights", "weights", "weights_experiments",
                  "features_old", "models", "outputs", "submissions", "auc_lab_20261107",
                  "cpu_0879/run", "cpu_0879/raw_run")]
    forbidden_files = {original/"clean/features_train.parquet",
                       original/"clean/features_test.parquet",
                       original/"clean/submission.csv",
                       original/"cpu_0879/training_inputs.npz"}

    def reject_archives(event, args):
        if event != "open" or not args or not isinstance(args[0], (str, bytes)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if path.is_relative_to(workspace):
            return
        if path in forbidden_files or any(path.is_relative_to(parent) for parent in forbidden):
            raise PermissionError(f"Raw rebuild cannot read or write original archives: {path}")

    sys.addaudithook(reject_archives)
