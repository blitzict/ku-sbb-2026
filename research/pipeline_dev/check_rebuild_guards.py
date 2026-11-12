"""Verify that fresh-stage subprocesses cannot consume historical data caches."""
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
env = os.environ.copy()
env.update(KU_SBB_ROOT=str(ROOT/"cpu_088_rebuild/smoke"), KU_SBB_ORIGINAL_ROOT=str(ROOT),
           KU_SBB_RAW_ONLY="1", KU_SBB_OFFLINE="1", PYTHONPATH=str(ROOT/"pipeline/offline_hooks"))
script = '''
from pathlib import Path
import socket
tests = []
for filename in ["models/mixed_088078/mixed_replay.npz", "tune_out/adv_ptest_train.npy",
                 "clean/features_train.parquet", "cpu_0879/training_inputs.npz"]:
    try:
        with open(filename, "rb") as f: f.read(1)
    except PermissionError:
        tests.append(filename)
    else:
        raise AssertionError("Historical input was not blocked: "+filename)
try:
    socket.getaddrinfo("example.com", 443)
except PermissionError:
    tests.append("network")
else:
    raise AssertionError("Network name resolution was not blocked")
with open("cpu_088_rebuild/smoke/CPU_REBUILD_WORKSPACE.json") as f:
    assert "fresh_cpu_rebuild" in f.read()
print("PASS: four original caches and network access blocked; fresh workspace readable")
'''
subprocess.run([sys.executable, "-B", "-c", script], cwd=ROOT, env=env, check=True)
