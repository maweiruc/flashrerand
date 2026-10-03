#!/usr/bin/env python3
"""Reproduce the FP64 NumPy/CUDA benchmark without importing another checkout.

  python benchmark.py run --smoke --output smoke.json
  python benchmark.py run --output assets/benchmark.json
  python benchmark.py plot assets/benchmark.json --output assets/performance.svg

Each measurement runs in a fresh child process. Matplotlib is only needed by
plot. --smoke --cpu-only checks the harness on a machine without CUDA; its
diagnostic results cannot produce a CPU/CUDA performance plot.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parent
THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS")
CASES = ((1024, 16, 1024, 32), (10000, 128, 2048, 64), (10000, 128, 8192, 64))
SMOKE_CASES = ((33, 3, 17, 3), (65, 4, 33, 5), (65, 4, 65, 5))
SEEDS = (42, 43, 44)
ATOL, RTOL = 1e-10, 1e-11


def digest(data):
    return hashlib.sha256(data).hexdigest()


def sources(package=None):
    if package is None:
        spec = importlib.util.find_spec("flashrerand")
        if spec is None or spec.origin is None:
            raise ImportError("Install flashrerand before running the benchmark")
        package = Path(spec.origin).resolve().parent
    paths = sorted(package.glob("*.py")) + sorted(package.glob("*.cu"))
    return {"benchmark.py": digest(Path(__file__).read_bytes()),
            **{"flashrerand/" + path.name: digest(path.read_bytes()) for path in paths}}


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def physical_cores():
    """Count available physical cores when possible; expose conservative fallback."""
    affinity = set(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    if sys.platform.startswith("linux"):
        try:
            cores = set()
            for block in Path("/proc/cpuinfo").read_text().split("\n\n"):
                item = dict(line.split(":", 1) for line in block.splitlines() if ":" in line)
                item = {key.strip(): value.strip() for key, value in item.items()}
                if {"processor", "physical id", "core id"} <= item.keys():
                    if affinity is None or int(item["processor"]) in affinity:
                        cores.add((item["physical id"], item["core id"]))
            if cores:
                return {"available_cores": len(cores), "detection": "physical cores within process affinity"}
        except (OSError, ValueError):
            pass
    if sys.platform == "darwin":
        try:
            count = int(subprocess.check_output(["sysctl", "-n", "hw.physicalcpu"], text=True, stderr=subprocess.DEVNULL).strip())
            if count > 0:
                return {"available_cores": count, "detection": "physical cores reported by sysctl"}
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    return {"available_cores": max(1, len(affinity) if affinity else os.cpu_count() or 1),
            "detection": "logical CPU fallback; physical core count unavailable"}


def cpu_model():
    if sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    if sys.platform == "darwin":
        try:
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.SubprocessError):
            pass
    return platform.machine()


def environment(np, device):
    """Allowlisted public metadata: deliberately excludes paths and identifiers."""
    result = {"python": platform.python_version(), "os": platform.system(),
              "architecture": platform.machine(), "cpu_model": cpu_model(),
              "numpy": np.__version__, "device": device,
              "thread_environment": {name: os.getenv(name) for name in THREAD_ENV}}
    try:
        import threadpoolctl
        keys = ("user_api", "internal_api", "num_threads", "version", "threading_layer", "architecture")
        result["threadpools"] = [{key: row.get(key) for key in keys} for row in threadpoolctl.threadpool_info()]
        result["threadpoolctl"] = threadpoolctl.__version__
    except ImportError:
        result["threadpools"] = None
        result["threadpool_note"] = "threadpoolctl unavailable; environment limits are recorded, actual thread counts unverified"
    try:
        build = np.__config__.show(mode="dicts").get("Build Dependencies", {})
        result["blas"] = {kind: {key: build.get(kind, {}).get(key) for key in ("name", "version")}
                          for kind in ("blas", "lapack")}
    except (AttributeError, TypeError):
        result["blas"] = None
    if device == "cuda":
        import cupy as cp
        props = cp.cuda.runtime.getDeviceProperties(0)
        name = props["name"]
        result["gpu"] = {"name": name.decode() if isinstance(name, bytes) else str(name),
            "total_memory_bytes": int(props["totalGlobalMem"]),
            "compute_capability": "%s.%s" % (props["major"], props["minor"]),
            "visible_device_index": 0, "cupy": cp.__version__,
            "cuda_runtime": int(cp.cuda.runtime.runtimeGetVersion()),
            "cuda_driver": int(cp.cuda.runtime.driverGetVersion())}
        try:
            result["gpu"]["cublas_math_mode"] = int(cp.cuda.cublas.getMathMode(cp.cuda.device.get_cublas_handle()))
        except (AttributeError, RuntimeError):
            result["gpu"]["cublas_math_mode"] = None
    return result


def array_hash(np, array):
    value = np.ascontiguousarray(array)
    return digest(str(value.shape).encode() + value.dtype.str.encode() + value.tobytes())


def validate(np, x, result, config):
    from flashrerand.core import reference_scores
    w, scores, ids = result.assignments, result.scores, result.candidate_ids
    if not all(isinstance(a, np.ndarray) for a in (w, scores, ids)):
        raise AssertionError("all returned arrays must reside on the host")
    if w.shape != (config["k"], config["n"]) or scores.shape != (config["k"],) or ids.shape != scores.shape:
        raise AssertionError("incorrect output shape")
    if scores.dtype != np.float64 or not np.isfinite(scores).all():
        raise AssertionError("scores must be finite FP64")
    if not np.isin(w, [0, 1]).all() or not np.all(w.sum(axis=1) == config["n1"]):
        raise AssertionError("assignments violate the fixed treatment count")
    if ids.dtype.kind not in "iu" or np.any(ids < 0) or np.any(ids >= config["candidates"]) or len(np.unique(ids)) != len(ids):
        raise AssertionError("invalid candidate IDs")
    if not np.array_equal(np.lexsort((ids, scores)), np.arange(config["k"])):
        raise AssertionError("selected scores/IDs are not in deterministic ascending order")
    expected = reference_scores(x, w, config["n1"], criterion="full")
    error = np.abs(scores - expected)
    if not np.allclose(scores, expected, atol=ATOL, rtol=RTOL):
        raise AssertionError("independent FP64 selected-score oracle failed")
    return {"status": "passed", "count": config["k"], "atol": ATOL, "rtol": RTOL,
            "max_abs_error": float(error.max()), "binary_fixed_count": True, "host_output": True,
            "own_score_order_exact": True, "assignments_sha256": array_hash(np, w),
            "scores_sha256": array_hash(np, scores), "candidate_ids_sha256": array_hash(np, ids),
            "scope": "selected assignments only; not certification of complete candidate-pool Top-K ranking"}


def worker(config):
    record = {"status": "running", "config": config, "seconds": [], "phase": "imports"}
    try:
        # The parent has set BLAS/OpenMP limits before this first NumPy import.
        import numpy as np
        import flashrerand
        from flashrerand import rerandomize
        from flashrerand.arrays import synchronize
        package = Path(flashrerand.__file__).resolve().parent
        record["source_sha256"] = sources(package)
        record["package_version"] = flashrerand.__version__
        record["phase"] = "input_generation"
        x = np.random.default_rng(2026).standard_normal((config["n"], config["p"])).astype(np.float64)
        record["input_sha256"] = array_hash(np, x)
        kwargs = {key: config[key] for key in ("n1", "candidates", "k", "seed", "device", "batch_size")}

        def call():
            synchronize(config["device"])
            start = time.perf_counter()
            result = rerandomize(x, **kwargs)
            synchronize(config["device"])
            elapsed = time.perf_counter() - start
            if not math.isfinite(elapsed) or elapsed <= 0:
                raise AssertionError("invalid elapsed time")
            return result, elapsed

        record["phase"] = "first_call"
        result, record["first_call_seconds"] = call()
        del result
        record["phase"] = "warmup"
        for _ in range(config["warmup"]):
            result, _ = call()
            del result
        record["phase"] = "measurement"
        for repeat in range(config["repeats"]):
            result, elapsed = call()
            record["seconds"].append(elapsed)
            if repeat + 1 < config["repeats"]:
                del result
        record["median_seconds"] = statistics.median(record["seconds"])
        record["phase"] = "untimed_oracle"
        record["validation"] = validate(np, x, result, config)
        del result
        record["phase"] = "environment"
        record["environment"] = environment(np, config["device"])
        record["status"], record["phase"] = "completed", "done"
    except Exception as exc:
        # Private terminal diagnostics may contain paths. Public JSON stores only
        # an exception class and stage, never arbitrary exception/traceback text.
        traceback.print_exc(file=sys.stderr)
        record.update(status="failed", error={"type": type(exc).__name__, "code": "worker_exception",
                      "message": "See the private terminal traceback; it is excluded from the public result."})
    return record


def config_for(case, device, seed, threads, batch, warmup, repeats):
    n, p, candidates, k = case
    return {"n": n, "p": p, "n1": n // 2, "candidates": candidates, "k": k,
            "case_id": "n%d-p%d-c%d-k%d" % case, "device": device, "seed": seed,
            "threads": threads, "batch_size": batch, "warmup": warmup, "repeats": repeats,
            "precision": "fp64", "criterion": "full_covariance_ddof1", "whitening": "pre",
            "sampler": "keys" if device == "cpu" else "repair", "balance": "gemm", "selection": "all",
            "prepare_backend": "cpu" if device == "cpu" else "cuda-qr",
            "validation": "checked" if device == "cpu" else "deferred"}


def launch(config, expected_source, private_log=None):
    env = os.environ.copy()
    env.update({name: str(config["threads"]) for name in THREAD_ENV})
    env.update(PYTHONNOUSERSITE="1", PYTHONUTF8="1", PYTHONUNBUFFERED="1",
               CUDA_VISIBLE_DEVICES="0" if config["device"] == "cuda" else "",
               CUPY_TF32="0", NVIDIA_TF32_OVERRIDE="0")
    for name in ("PYTHONPATH", "PYTHONHOME"):
        env.pop(name, None)
    try:
        process = subprocess.run([sys.executable, str(Path(__file__).resolve()), "_worker"],
            input=json.dumps(config), text=True, capture_output=True, env=env, cwd=ROOT, timeout=1800)
        if private_log is not None and (process.stderr or process.returncode != 0):
            with private_log.open("a", encoding="utf-8") as stream:
                stream.write("\nCONFIG " + json.dumps(config) + "\nSTDOUT\n" + process.stdout + "\nSTDERR\n" + process.stderr)
        if process.stderr:
            sys.stderr.write(process.stderr)
        record = json.loads(process.stdout)
        if not isinstance(record, dict) or record.get("config") != config:
            raise ValueError("unexpected worker output")
        if process.returncode != 0 or record.get("status") != "completed":
            record["status"] = "failed"
        elif record.get("source_sha256") != expected_source:
            record.update(status="failed", error={"code": "source_changed_during_run"})
        return record
    except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
        if private_log is not None:
            with private_log.open("a", encoding="utf-8") as stream:
                stream.write("\nCONFIG " + json.dumps(config) + "\n" + repr(exc) + "\n")
                if "process" in locals():
                    stream.write("STDOUT\n" + process.stdout + "\nSTDERR\n" + process.stderr)
        return {"status": "failed", "config": config, "seconds": [],
                "error": {"type": type(exc).__name__, "code": "child_process_or_protocol_failure"}}


def checkpoint(path, report):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def summarize(report):
    groups = {}
    for row in report["jobs"]:
        if row["status"] != "completed":
            raise ValueError("cannot summarize a failed or incomplete measurement")
        c = row["config"]
        key = (c["case_id"], c["device"])
        groups.setdefault(key, []).append(row)
    output = []
    for case in report["protocol"]["cases"]:
        name = "n%d-p%d-c%d-k%d" % tuple(case)
        item = {"case_id": name, "n": case[0], "p": case[1], "candidates": case[2], "k": case[3], "devices": {}}
        for device in report["protocol"]["devices"]:
            rows = sorted(groups[(name, device)], key=lambda row: row["config"]["seed"])
            if [row["config"]["seed"] for row in rows] != list(SEEDS):
                raise ValueError("each device/case requires the same three distinct seeds")
            medians = [statistics.median(row["seconds"]) for row in rows]
            item["devices"][device] = {"seed_medians_seconds": dict(zip(map(str, SEEDS), medians)),
                "median_seconds": statistics.median(medians), "min_seed_median_seconds": min(medians),
                "max_seed_median_seconds": max(medians)}
        if set(item["devices"]) == {"cpu", "cuda"}:
            cpu, gpu = groups[(name, "cpu")], groups[(name, "cuda")]
            if len({row["input_sha256"] for row in cpu + gpu}) != 1:
                raise ValueError("input hashes differ between paired jobs")
            ratios = {str(seed): item["devices"]["cpu"]["seed_medians_seconds"][str(seed)] /
                      item["devices"]["cuda"]["seed_medians_seconds"][str(seed)] for seed in SEEDS}
            item.update(paired_seed_cpu_over_cuda=ratios, median_paired_ratio=statistics.median(ratios.values()))
        output.append(item)
    return output


def run(args):
    partial = args.output.with_suffix(".partial.json")
    private_log = args.output.with_suffix(".private.log")
    if args.output.suffix != ".json" or any(path.exists() or path.is_symlink() for path in
            (args.output, partial, partial.with_suffix(partial.suffix + ".tmp"), private_log)):
        raise ValueError("output must be a new .json file, with no existing sibling checkpoint")
    if args.cpu_only and not args.smoke:
        raise ValueError("--cpu-only is restricted to --smoke diagnostics")
    cores = physical_cores()
    threads = sorted({1, min(8, cores["available_cores"])})
    cases, batches = (SMOKE_CASES, (4, 16)) if args.smoke else (CASES, (512, 2048))
    devices = ("cpu",) if args.cpu_only else ("cpu", "cuda")
    warmup, repeats = (1, 2) if args.smoke else (2, 5)
    report = {"schema_version": 1, "benchmark": "flashrerand-v0-cpu-cuda", "status": "running",
        "created_utc": now(), "diagnostic_only": bool(args.smoke), "source_sha256": sources(),
        "cpu_core_detection": cores, "cpu_tuning": {"jobs": [], "selected": None}, "jobs": [],
        "protocol": {"cases": [list(case) for case in cases], "devices": list(devices), "seeds": list(SEEDS),
            "data_seed": 2026, "input_generator": "numpy.default_rng(2026).standard_normal((n,p)), FP64",
            "warmup": warmup, "repeats": repeats, "cpu_threads_candidates": threads, "cpu_batch_candidates": list(batches),
            "gpu_batch_size": 16 if args.smoke else 2048, "gpu_host_threads": 1,
            "gpu_selection": "CUDA_VISIBLE_DEVICES=0; no other benchmark process should run concurrently",
            "timing_scope": "host X through validation, preparation, sampling, scoring, selection and complete host W/scores/IDs",
            "excluded_from_timing": ["process/import startup", "input generation", "untimed oracle", "file I/O"],
            "first_call_scope": "process-first API call after imports and backend synchronization; module/NVRTC disk caches may exist, so this is not guaranteed cold compilation; recorded separately before additional warmups",
            "cpu_method": "NumPy vectorized random keys + GEMM; checked full API",
            "cuda_method": "CUDA repair sampling + cuBLAS GEMM; deferred full API",
            "comparability": "same statistical task and output contract; different samplers and candidate pools even at the same seed",
            "aggregation": "within-seed median, then median of three seed medians; ratio is median of paired seed ratios",
            "oracle_scope": "selected outputs only; complete-pool correctness is tested separately",
            "order": "serial fresh child processes; alternate CPU/CUDA order by case and seed",
            "tuning_rule": "one representative case; minimum CPU median, ties prefer fewer threads then smaller batch; freeze globally"}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with private_log.open("x", encoding="utf-8") as stream:
        stream.write("Private child-process diagnostics. Do not publish this file; it may contain local paths.\n")
    checkpoint(partial, report)

    def execute(config, destination):
        row = launch(config, report["source_sha256"], private_log)
        destination.append(row)
        checkpoint(partial, report)
        print("%s %s seed=%s threads=%s batch=%s %s" % (config["case_id"], config["device"], config["seed"],
              config["threads"], config["batch_size"], row["status"]), flush=True)
        if row["status"] != "completed":
            raise RuntimeError("a measurement failed; its checkpoint is preserved")

    try:
        for thread_count in threads:
            for batch in batches:
                execute(config_for(cases[1], "cpu", 42, thread_count, batch, 1, 1 if args.smoke else 3), report["cpu_tuning"]["jobs"])
        winner = min(report["cpu_tuning"]["jobs"], key=lambda row: (statistics.median(row["seconds"]),
                         row["config"]["threads"], row["config"]["batch_size"]))
        report["cpu_tuning"]["selected"] = {"threads": winner["config"]["threads"],
            "batch_size": winner["config"]["batch_size"], "median_seconds": statistics.median(winner["seconds"]),
            "case_id": winner["config"]["case_id"], "seed": 42}
        checkpoint(partial, report)
        for case_index, case in enumerate(cases):
            for seed_index, seed in enumerate(SEEDS):
                order = devices if (case_index + seed_index) % 2 == 0 else tuple(reversed(devices))
                for device in order:
                    execute(config_for(case, device, seed, winner["config"]["threads"] if device == "cpu" else 1,
                        winner["config"]["batch_size"] if device == "cpu" else report["protocol"]["gpu_batch_size"],
                        warmup, repeats), report["jobs"])
        report["summary"] = summarize(report)
        report["status"] = "completed"
    except (Exception, KeyboardInterrupt) as exc:
        report.update(status="failed", error={"type": type(exc).__name__, "code": "run_incomplete"})
        print(str(exc), file=sys.stderr)
    report["finished_utc"] = now()
    checkpoint(partial, report)
    if report["status"] == "completed":
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
    return int(report["status"] != "completed")


def plot(args):
    if args.output.exists() or args.output.is_symlink():
        raise ValueError("plot output must be a new file")
    report = json.loads(args.input.read_text(encoding="utf-8"))
    if report.get("schema_version") != 1 or report.get("status") != "completed":
        raise ValueError("plot requires a completed version-1 benchmark result")
    if report.get("diagnostic_only"):
        raise ValueError("smoke diagnostics cannot produce a publication performance plot")
    if set(report["protocol"]["devices"]) != {"cpu", "cuda"}:
        raise ValueError("plot requires paired CPU and CUDA measurements")
    summary = summarize(report)
    if summary != report.get("summary"):
        raise ValueError("saved summary differs from raw timing samples")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    fig, ax = plt.subplots(figsize=(9.4, 5.3))
    colors, offsets = {"cpu": "#6b7280", "cuda": "#087f8c"}, {"cpu": -.19, "cuda": .19}
    for device in ("cpu", "cuda"):
        x = [i + offsets[device] for i in range(len(summary))]
        values = [item["devices"][device]["median_seconds"] * 1000 for item in summary]
        label = ("NumPy CPU: keys + GEMM (threads: %d)" % report["cpu_tuning"]["selected"]["threads"]
                 if device == "cpu" else "CUDA: repair + GEMM")
        ax.bar(x, values, width=.33, color=colors[device], label=label, alpha=.86)
        for center, value, item in zip(x, values, summary):
            seeds = item["devices"][device]["seed_medians_seconds"]
            ax.scatter([center + shift for shift in (-.05, 0, .05)], [seeds[str(seed)] * 1000 for seed in SEEDS],
                       color="#172126", s=16, zorder=4)
            ax.annotate("%.2f" % value if value < 100 else "%.0f" % value, (center, value),
                        xytext=(0, 8), textcoords="offset points", ha="center", fontsize=9)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: "%g" % value))
    ax.set_ylabel("Complete API latency (ms, log scale)")
    ax.set_xticks(range(len(summary)), ["n=%s, p=%s\nC=%s, K=%s\n%.2f× median paired ratio" %
                    (item["n"], item["p"], item["candidates"], item["k"], item["median_paired_ratio"]) for item in summary])
    ax.set_title("FP64 rerandomization: NumPy baseline and CUDA", loc="left", pad=17, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", which="major", alpha=.2)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False, bbox_to_anchor=(0, 1.02))
    low, high = ax.get_ylim()
    ax.set_ylim(low, high * 4)
    gpu_row = next(row for row in report["jobs"] if row["config"]["device"] == "cuda")
    cpu_row = next(row for row in report["jobs"] if row["config"]["device"] == "cpu")
    hardware = cpu_row["environment"]["cpu_model"] + " · " + gpu_row["environment"]["gpu"]["name"]
    fig.text(.08, .045, hardware, fontsize=8)
    fig.text(.08, .015, "Bars: median of 3 seed medians; dots: seed medians (5 repetitions each). Different samplers; same search budget and host output.", fontsize=7.5)
    fig.tight_layout(rect=(0, .10, 1, 1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    runner = commands.add_parser("run", help="tune CPU once, then run serial paired measurements")
    runner.add_argument("--output", type=Path, required=True)
    runner.add_argument("--smoke", action="store_true", help="small diagnostic version of the entire pipeline")
    runner.add_argument("--cpu-only", action="store_true", help="CPU-only diagnostics; requires --smoke")
    drawing = commands.add_parser("plot", help="render an existing completed result; requires Matplotlib")
    drawing.add_argument("input", type=Path)
    drawing.add_argument("--output", type=Path, required=True, help="new .svg, .png, or other Matplotlib-supported file")
    commands.add_parser("_worker", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.command == "_worker":
        result = worker(json.load(sys.stdin))
        print(json.dumps(result, allow_nan=False))
        return int(result["status"] != "completed")
    try:
        return run(args) if args.command == "run" else plot(args)
    except (OSError, ValueError, KeyError, ImportError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
