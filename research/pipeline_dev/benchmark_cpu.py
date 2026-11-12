"""Run an owned Python process tree with Windows CPU/memory/time limits.

Examples, from the project root (use the Python environment with model deps):
  python -m pipeline.benchmark_cpu --name mixed_member -- -m pipeline.mixed_member_cpu
  python -m pipeline.benchmark_cpu --name v4 -- run_cpu.py predict --model v4 --raw --workers 2 --out-dir outputs/v4_limited

Two logical processors are enforced for inference, including descendants.
Training can explicitly request up to six processors without changing the cap.
A Windows Job Object caps total committed memory at 6 GiB, leaving headroom on
an 8 GB machine. Its peak committed memory is recorded, not mislabeled as RSS.
The timeout covers all model work from process launch. Python socket access is
blocked by an audit hook; no downloads or dependency installs occur here.

This measures resources only. A separate numerical comparison must establish
the designated CSV's <=1e-6 tolerance and whether the command uses raw inputs.
"""
import argparse
import ctypes as C
from ctypes import wintypes as W
import json
import os
from pathlib import Path
import subprocess
import sys
from time import monotonic, sleep

ROOT = Path(__file__).resolve().parents[1]


class BasicLimits(C.Structure):
    _fields_ = [("process_time", C.c_longlong), ("job_time", C.c_longlong),
                ("flags", W.DWORD), ("min_ws", C.c_size_t), ("max_ws", C.c_size_t),
                ("active_process_limit", W.DWORD), ("affinity", C.c_size_t),
                ("priority", W.DWORD), ("scheduling_class", W.DWORD)]


class IOCounters(C.Structure):
    _fields_ = [(name, C.c_ulonglong) for name in
                ("read_ops", "write_ops", "other_ops", "read_bytes", "write_bytes", "other_bytes")]


class ExtendedLimits(C.Structure):
    _fields_ = [("basic", BasicLimits), ("io", IOCounters),
                ("process_memory_limit", C.c_size_t), ("job_memory_limit", C.c_size_t),
                ("peak_process_memory", C.c_size_t), ("peak_job_memory", C.c_size_t)]


def check(ok):
    if not ok:
        raise C.WinError(C.get_last_error())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--seconds", type=int, default=1800)
    parser.add_argument("--memory-gib", type=float, default=6)
    parser.add_argument("--training", action="store_true", help="Allow up to seven days; this is not an inference qualification check")
    parser.add_argument("--processors", type=int, default=2, choices=range(1, 7),
                        help="Training CPU allowance; inference always requires two")
    parser.add_argument("--core-placement", choices=("spread", "consecutive"), default="spread",
                        help="Training only: use distinct physical cores, or the former consecutive-logical policy")
    parser.add_argument("python_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if os.name != "nt":
        parser.error("This resource harness uses Windows Job Objects.")
    if not args.training and args.processors != 2:
        parser.error("Inference measurements require exactly two logical processors")
    maximum_seconds = 7 * 86400 if args.training else 1800
    if (not args.name.replace("_", "").replace("-", "").isalnum()
            or not 0 < args.seconds <= maximum_seconds or not 0 < args.memory_gib <= 6):
        parser.error(f"Use a simple run name, <={maximum_seconds} seconds and <=6 GiB.")
    arguments = args.python_args
    if arguments and arguments[0] == "--":
        arguments = arguments[1:]
    if not arguments:
        parser.error("Pass Python arguments after --")
    kernel = C.WinDLL("kernel32", use_last_error=True)
    kernel.GetCurrentProcess.restype = W.HANDLE
    kernel.GetProcessAffinityMask.argtypes = [W.HANDLE, C.POINTER(C.c_size_t), C.POINTER(C.c_size_t)]
    kernel.SetProcessAffinityMask.argtypes = [W.HANDLE, C.c_size_t]
    kernel.CreateJobObjectW.argtypes = [C.c_void_p, W.LPCWSTR]
    kernel.CreateJobObjectW.restype = W.HANDLE
    kernel.SetInformationJobObject.argtypes = [W.HANDLE, C.c_int, C.c_void_p, W.DWORD]
    kernel.QueryInformationJobObject.argtypes = [W.HANDLE, C.c_int, C.c_void_p, W.DWORD, C.c_void_p]
    kernel.AssignProcessToJobObject.argtypes = [W.HANDLE, W.HANDLE]
    kernel.TerminateJobObject.argtypes = [W.HANDLE, W.UINT]
    kernel.CloseHandle.argtypes = [W.HANDLE]
    process_mask, system_mask = C.c_size_t(), C.c_size_t()
    check(kernel.GetProcessAffinityMask(kernel.GetCurrentProcess(), C.byref(process_mask), C.byref(system_mask)))
    processors = [i for i in range(C.sizeof(C.c_size_t)*8) if process_mask.value & (1 << i)][:args.processors]
    if args.training and args.core_placement == "spread":
        from pipeline.cpu_topology import core_masks, select_processors
        processors = select_processors(process_mask.value, args.processors, core_masks())
    if len(processors) != args.processors:
        raise RuntimeError(f"{args.processors} available logical processors are required")
    affinity = sum(1 << i for i in processors)
    check(kernel.SetProcessAffinityMask(kernel.GetCurrentProcess(), affinity))
    job = kernel.CreateJobObjectW(None, None)
    check(job)
    limits = ExtendedLimits()
    limits.basic.flags = 0x10 | 0x200 | 0x2000  # affinity, job memory, kill on job close
    limits.basic.affinity = affinity
    limits.job_memory_limit = int(args.memory_gib*1024**3)
    child = None
    assigned = False
    output = ROOT/"outputs/resource_checks"/args.name
    output.mkdir(parents=True, exist_ok=True)
    report = {"command": [sys.executable, "-B", *arguments],
              "mode": "training" if args.training else "inference",
              "logical_processors": processors, "affinity_enforced": True,
              "core_placement": args.core_placement if args.training else "consecutive (inference protocol)",
              "memory_limit_bytes": limits.job_memory_limit,
              "timeout_seconds": args.seconds,
              "network_guard": "CPython socket audit hook; not an OS firewall",
              "scope": "Resource measurement only; check raw-input provenance and numerical equality separately"}
    started = monotonic()
    try:
        check(kernel.SetInformationJobObject(job, 9, C.byref(limits), C.sizeof(limits)))
        env = os.environ.copy()
        env.update(KU_SBB_OFFLINE="1", CUDA_VISIBLE_DEVICES="", ONEAPI_DEVICE_SELECTOR="*:cpu",
                   OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                   NUMEXPR_NUM_THREADS="1", PYTHONUNBUFFERED="1")
        env["PYTHONPATH"] = os.pathsep.join([str(ROOT/"pipeline/offline_hooks"), str(ROOT),
                                            env.get("PYTHONPATH", "")])
        with (output/"stdout.log").open("w", encoding="utf-8") as stream:
            child = subprocess.Popen(report["command"], cwd=ROOT, env=env,
                                     stdout=stream, stderr=subprocess.STDOUT)
            check(kernel.AssignProcessToJobObject(job, int(child._handle)))
            assigned = True
            while child.poll() is None:
                if monotonic()-started > args.seconds:
                    kernel.TerminateJobObject(job, 124)
                    child.wait()
                    report["timed_out"] = True
                    break
                sleep(.2)
        observed = ExtendedLimits()
        check(kernel.QueryInformationJobObject(job, 9, C.byref(observed), C.sizeof(observed), None))
        report.update(exit_code=child.returncode, seconds=monotonic()-started,
                      peak_job_committed_bytes=observed.peak_job_memory,
                      peak_process_committed_bytes=observed.peak_process_memory,
                      passed_resource_limits=(child.returncode == 0 and
                          monotonic()-started <= args.seconds and observed.peak_job_memory <= limits.job_memory_limit))
    finally:
        if child is not None and child.poll() is None:
            if assigned:
                kernel.TerminateJobObject(job, 125)
            else:
                child.terminate()
            child.wait()
        kernel.CloseHandle(job)
    (output/"resources.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    raise SystemExit(0 if report["passed_resource_limits"] else 1)


if __name__ == "__main__":
    main()
