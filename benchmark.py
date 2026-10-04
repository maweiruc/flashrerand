#!/usr/bin/env python3
"""Reproduce the FP64 NumPy/CUDA benchmark without importing another checkout.

  python benchmark.py run --smoke --output smoke.json
  python benchmark.py run --output assets/benchmark.json
  python benchmark.py plot assets/benchmark.json --output assets/performance.svg

Each CPU or GPU block runs in a fresh child process. With --baseline-source,
both GPU versions run interleaved in the same child. Matplotlib is only needed by
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
CASES = ((1024, 16, 1024, 32), (10000, 128, 2048, 64),
         (10000, 128, 8192, 64), (1024, 16, 8192, 64))
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
    record = {"status": "running", "config": config, "seconds": [],
              "warmup_seconds": [], "phase": "imports"}
    try:
        # The parent has set BLAS/OpenMP limits before this first NumPy import.
        import numpy as np
        import flashrerand
        from flashrerand import rerandomize
        from flashrerand.arrays import synchronize
        package = Path(flashrerand.__file__).resolve().parent
        record["source_sha256"] = sources(package)
        record["package_version"] = flashrerand.__version__
        expected_version = "0.1.0" if config["implementation"] == "v0" else "0.2.0"
        if flashrerand.__version__ != expected_version:
            raise ValueError("benchmark implementation label does not match the installed package version")
        record["phase"] = "input_generation"
        x = np.random.default_rng(2026).standard_normal((config["n"], config["p"])).astype(np.float64)
        record["input_sha256"] = array_hash(np, x)
        kwargs = {key: config[key] for key in ("n1", "candidates", "k", "seed", "device", "batch_size")}
        if config["implementation"] != "v0":
            kwargs.update(sampler="auto", backend="auto")

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
            result, elapsed = call()
            record["warmup_seconds"].append(elapsed)
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
        record["execution"] = result.metadata.get("execution", {
            "actual_sampler": "keys" if config["device"] == "cpu" else "repair",
            "actual_backend": "gemm", "policy_status": "frozen_v0"})
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


def paired_order(case_index, seed, repeat):
    """Rotate the first version across seeds/cases; mirror successive rounds."""
    rotation = case_index + SEEDS.index(seed) + repeat
    return ("cuda", "v0") if rotation % 2 == 0 else ("v0", "cuda")


def module_origins_match(package, expected):
    """Check canonical/aliased package origins without publishing local paths."""
    expected = Path(expected).resolve()
    if Path(package.__file__).resolve() != expected / "__init__.py":
        raise AssertionError("benchmark package origin differs from the declared source")
    for name, module in list(sys.modules.items()):
        if name == package.__name__ or name.startswith(package.__name__ + "."):
            origin = getattr(module, "__file__", None)
            if origin is not None:
                try:
                    Path(origin).resolve().relative_to(expected)
                except ValueError as exc:
                    raise AssertionError("benchmark submodule escaped its package origin") from exc


def load_baseline_alias(source):
    """Load frozen V0 without changing canonical flashrerand or sys.path."""
    package = Path(source).resolve() / "flashrerand"
    name = "_flashrerand_benchmark_v0"
    if name in sys.modules:
        raise RuntimeError("baseline alias already exists in this child process")
    spec = importlib.util.spec_from_file_location(name, package / "__init__.py",
                                                submodule_search_locations=[str(package)])
    if spec is None or spec.loader is None:
        raise ImportError("cannot load frozen V0 package")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    module_origins_match(module, package)
    if module.__version__ != "0.1.0":
        raise ValueError("baseline alias must contain FlashRerand 0.1.0")
    return module


def gpu_pair_worker(payload):
    """Both GPU versions share one process/context; no timed-pair file I/O."""
    configs = payload["configs"]
    rows = {name: dict(status="running", config=config, seconds=[], warmup_seconds=[],
                       measured_calls=[], phase="imports") for name, config in configs.items()}
    phase = "imports"
    try:
        import numpy as np
        import flashrerand
        from flashrerand.arrays import synchronize
        if flashrerand.__version__ != "0.2.0":
            raise ValueError("canonical benchmark package must be FlashRerand 0.2.0")
        current_origin = Path(payload["current_package"]).resolve()
        module_origins_match(flashrerand, current_origin)
        baseline = load_baseline_alias(payload["baseline_source"])
        modules = {"cuda": flashrerand, "v0": baseline}
        baseline_origin = Path(payload["baseline_source"]).resolve() / "flashrerand"
        if baseline_origin == current_origin:
            raise ValueError("baseline and current package origins must differ")
        config = configs["cuda"]
        if (config["implementation"], config["backend"], configs["v0"]["implementation"],
                configs["v0"]["backend"], config["threads"]) != ("cuda", "auto", "v0", "gemm", 1):
            raise ValueError("paired GPU routes must be current AUTO and V0 GEMM with one host thread")
        paired_keys = ("n", "p", "n1", "candidates", "k", "seed", "device", "batch_size",
                       "threads", "warmup", "repeats", "case_id")
        if any(config[key] != configs["v0"][key] for key in paired_keys) or config["device"] != "cuda":
            raise ValueError("GPU pair must have identical CUDA task and measurement settings")
        for name, module in modules.items():
            row = rows[name]
            row["source_sha256"] = sources(Path(module.__file__).resolve().parent)
            row["package_version"] = module.__version__
            if row["source_sha256"] != payload["expected_sources"][name]:
                raise AssertionError("GPU pair source changed before measurement")
        phase = "input_generation"
        x = np.random.default_rng(2026).standard_normal((config["n"], config["p"])).astype(np.float64)
        input_hash = array_hash(np, x)
        for row in rows.values():
            row["input_sha256"] = input_hash
        kwargs = {key: config[key] for key in ("n1", "candidates", "k", "seed", "device", "batch_size")}

        def execution_info(result, name):
            return result.metadata.get("execution", {
                "actual_sampler": "repair", "actual_backend": "gemm", "policy_status": "frozen_v0"})

        def call(name):
            options = kwargs if name == "v0" else dict(kwargs, sampler="auto", backend="auto")
            synchronize("cuda")
            start = time.perf_counter()
            result = modules[name].rerandomize(x, **options)
            synchronize("cuda")
            elapsed = time.perf_counter() - start
            if not math.isfinite(elapsed) or elapsed <= 0:
                raise AssertionError("invalid paired elapsed time")
            return result, elapsed

        schedule = []
        phase = "first_call"
        order = paired_order(payload["case_index"], config["seed"], 0)
        for position, name in enumerate(order):
            result, elapsed = call(name)
            rows[name]["first_call_seconds"] = elapsed
            rows[name]["first_call_metadata"] = result.metadata
            rows[name]["first_call_execution"] = execution_info(result, name)
            rows[name]["first_call_timings"] = dict(result.timings)
            schedule.append(dict(phase=phase, round=0, position=position, implementation=name))
            del result
        phase = "warmup"
        for repeat in range(config["warmup"]):
            order = paired_order(payload["case_index"], config["seed"], repeat)
            for position, name in enumerate(order):
                result, elapsed = call(name)
                rows[name]["warmup_seconds"].append(elapsed)
                schedule.append(dict(phase=phase, round=repeat, position=position, implementation=name))
                del result
        phase = "measurement"
        last_results = {}
        for repeat in range(config["repeats"]):
            order = paired_order(payload["case_index"], config["seed"], repeat)
            for position, name in enumerate(order):
                result, elapsed = call(name)
                rows[name]["seconds"].append(elapsed)
                rows[name]["measured_calls"].append(dict(round=repeat, position=position,
                    order=list(order), seconds=elapsed))
                schedule.append(dict(phase=phase, round=repeat, position=position, implementation=name))
                last_results[name] = result
        # Oracles, environment queries, hashing and serialization occur only after
        # both versions have finished the entire timed block.
        phase = "untimed_oracle"
        module_origins_match(flashrerand, current_origin)
        module_origins_match(baseline, baseline_origin)
        for name, module in list(sys.modules.items()):
            if name == baseline.__name__ or name.startswith(baseline.__name__ + "."):
                canonical = "flashrerand" + name[len(baseline.__name__):]
                if sys.modules.get(canonical) is module:
                    raise AssertionError("baseline alias shares a FlashRerand module with current package")
        for name, row in rows.items():
            after = sources(Path(modules[name].__file__).resolve().parent)
            if after != row["source_sha256"]:
                raise AssertionError("GPU pair source changed during measurement")
            row["source_sha256_after"] = after
            row["validation"] = validate(np, x, last_results[name], configs[name])
            row["execution"] = execution_info(last_results[name], name)
            row["median_seconds"] = statistics.median(row["seconds"])
            row["package_origin_verified"] = True
            row["import_mode"] = "canonical" if name == "cuda" else "independent_package_alias"
        phase = "environment"
        info = environment(np, "cuda")
        pair_id = config["case_id"] + "-s" + str(config["seed"])
        for row in rows.values():
            row["environment"] = info
            row["pairing"] = dict(id=pair_id, design="same_process_interleaved_gpu_versions_v1",
                case_index=payload["case_index"], schedule=schedule,
                no_io_or_oracle_inside_timed_block=True)
            row["status"], row["phase"] = "completed", "done"
        return dict(status="completed", records=[rows[name] for name in ("cuda", "v0")])
    except Exception as exc:
        traceback.print_exc(file=sys.stderr)
        for row in rows.values():
            row.update(status="failed", phase=phase, error=dict(type=type(exc).__name__,
                       code="paired_gpu_worker_exception", message="See the private stderr log."))
        return dict(status="failed", records=[rows[name] for name in ("cuda", "v0")])


def config_for(case, device, seed, threads, batch, warmup, repeats, implementation=None):
    n, p, candidates, k = case
    return {"n": n, "p": p, "n1": n // 2, "candidates": candidates, "k": k,
            "case_id": "n%d-p%d-c%d-k%d" % case, "device": device, "seed": seed,
            "implementation": implementation or device,
            "threads": threads, "batch_size": batch, "warmup": warmup, "repeats": repeats,
            "precision": "fp64", "criterion": "full_covariance_ddof1", "whitening": "pre",
            "sampler": "keys" if device == "cpu" else "repair",
            "backend": "gemm" if implementation == "v0" else "auto", "selection": "all",
            "prepare_backend": "cpu" if device == "cpu" else "cuda-qr",
            "validation": "checked" if device == "cpu" else "deferred"}


def launch(config, expected_source, private_log=None, baseline_source=None):
    env = os.environ.copy()
    env.update({name: str(config["threads"]) for name in THREAD_ENV})
    env.update(PYTHONNOUSERSITE="1", PYTHONUTF8="1", PYTHONUNBUFFERED="1",
               CUDA_VISIBLE_DEVICES="0" if config["device"] == "cuda" else "",
               CUPY_TF32="0", NVIDIA_TF32_OVERRIDE="0")
    for name in ("PYTHONPATH", "PYTHONHOME"):
        env.pop(name, None)
    try:
        command = [sys.executable, str(Path(__file__).resolve()), "_worker"]
        if baseline_source is not None:
            command = [sys.executable, "-c",
                "import runpy,sys; script,source=sys.argv[1:]; sys.path.insert(0,source); "
                "sys.argv=[script,'_worker']; runpy.run_path(script,run_name='__main__')",
                str(Path(__file__).resolve()), str(baseline_source)]
        process = subprocess.run(command,
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


def launch_gpu_pair(config, expected_source, baseline_source, baseline_hashes, case_index, private_log):
    old_config = dict(config, implementation="v0", backend="gemm")
    spec = importlib.util.find_spec("flashrerand")
    payload = dict(configs={"cuda": config, "v0": old_config}, case_index=case_index,
                   current_package=str(Path(spec.origin).resolve().parent),
                   baseline_source=str(baseline_source.resolve()),
                   expected_sources={"cuda": expected_source, "v0": baseline_hashes})
    env = os.environ.copy()
    env.update({name: "1" for name in THREAD_ENV})
    env.update(PYTHONNOUSERSITE="1", PYTHONUTF8="1", PYTHONUNBUFFERED="1",
               CUDA_VISIBLE_DEVICES="0", CUPY_TF32="0", NVIDIA_TF32_OVERRIDE="0")
    for name in ("PYTHONPATH", "PYTHONHOME"):
        env.pop(name, None)
    try:
        process = subprocess.run([sys.executable, str(Path(__file__).resolve()), "_gpu_pair_worker"],
            input=json.dumps(payload), text=True, capture_output=True, env=env, cwd=ROOT, timeout=1800)
        if process.stderr or process.returncode:
            with private_log.open("a", encoding="utf-8") as stream:
                stream.write("\nGPU PAIR " + json.dumps(config) + "\nSTDOUT\n" + process.stdout + "\nSTDERR\n" + process.stderr)
            if process.stderr:
                sys.stderr.write(process.stderr)
        record = json.loads(process.stdout)
        rows = record["records"]
        if len(rows) != 2 or [row.get("config") for row in rows] != [config, old_config]:
            raise ValueError("unexpected paired GPU worker output")
        for name, row in zip(("cuda", "v0"), rows):
            if process.returncode or record.get("status") != "completed" or row.get("status") != "completed":
                row["status"] = "failed"
            elif row.get("source_sha256") != payload["expected_sources"][name]:
                row.update(status="failed", error=dict(code="source_changed_during_run"))
        return rows
    except (OSError, subprocess.TimeoutExpired, ValueError, KeyError, TypeError) as exc:
        with private_log.open("a", encoding="utf-8") as stream:
            stream.write("\nGPU PAIR " + json.dumps(config) + "\n" + repr(exc) + "\n")
        return [dict(status="failed", config=c, seconds=[], error=dict(type=type(exc).__name__,
                     code="paired_child_process_or_protocol_failure")) for c in (config, old_config)]


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
        if report.get("schema_version", 2) >= 3:
            if (len(row["seconds"]) != c["repeats"] or len(row["warmup_seconds"]) != c["warmup"] or
                    not all(math.isfinite(value) and value > 0 for value in
                            row["seconds"] + row["warmup_seconds"] + [row["first_call_seconds"]])):
                raise ValueError("measurement counts or durations differ from the recorded protocol")
        key = (c["case_id"], c["implementation"])
        groups.setdefault(key, []).append(row)
    output = []
    for case in report["protocol"]["cases"]:
        name = "n%d-p%d-c%d-k%d" % tuple(case)
        item = {"case_id": name, "n": case[0], "p": case[1], "candidates": case[2], "k": case[3], "devices": {}}
        for device in report["protocol"]["implementations"]:
            rows = sorted(groups[(name, device)], key=lambda row: row["config"]["seed"])
            if [row["config"]["seed"] for row in rows] != list(SEEDS):
                raise ValueError("each device/case requires the same three distinct seeds")
            medians = [statistics.median(row["seconds"]) for row in rows]
            item["devices"][device] = {"seed_medians_seconds": dict(zip(map(str, SEEDS), medians)),
                "median_seconds": statistics.median(medians), "min_seed_median_seconds": min(medians),
                "max_seed_median_seconds": max(medians)}
        if {"cpu", "cuda"} <= set(item["devices"]):
            cpu, gpu = groups[(name, "cpu")], groups[(name, "cuda")]
            if len({row["input_sha256"] for row in cpu + gpu}) != 1:
                raise ValueError("input hashes differ between paired jobs")
            ratios = {str(seed): item["devices"]["cpu"]["seed_medians_seconds"][str(seed)] /
                      item["devices"]["cuda"]["seed_medians_seconds"][str(seed)] for seed in SEEDS}
            item.update(paired_seed_cpu_over_cuda=ratios, median_paired_ratio=statistics.median(ratios.values()))
        if "v0" in item["devices"]:
            paired = groups[(name, "v0")] + groups[(name, "cuda")]
            if len({row["input_sha256"] for row in paired}) != 1:
                raise ValueError("input hashes differ between version comparisons")
            if report.get("schema_version", 2) >= 3:
                for seed in SEEDS:
                    current = next(row for row in groups[(name, "cuda")] if row["config"]["seed"] == seed)
                    old = next(row for row in groups[(name, "v0")] if row["config"]["seed"] == seed)
                    pair = current.get("pairing")
                    if (not pair or pair != old.get("pairing") or
                            pair.get("design") != "same_process_interleaved_gpu_versions_v1" or
                            pair.get("id") != name + "-s" + str(seed)):
                        raise ValueError("GPU version comparison requires a matching same-process pair")
                    if any(not row.get("package_origin_verified") or row.get("source_sha256_after") !=
                           row.get("source_sha256") for row in (current, old)):
                        raise ValueError("paired package origin/source verification is missing")
                    schedule = []
                    for phase, rounds in (("first_call", 1), ("warmup", current["config"]["warmup"]),
                                          ("measurement", current["config"]["repeats"])):
                        for repeat in range(rounds):
                            order = paired_order(pair["case_index"], seed, repeat)
                            schedule.extend(dict(phase=phase, round=repeat, position=position,
                                                 implementation=route) for position, route in enumerate(order))
                    if pair.get("schedule") != schedule:
                        raise ValueError("paired call order differs from the frozen rotation")
                    for row, route in ((current, "cuda"), (old, "v0")):
                        expected_calls = []
                        for repeat, value in enumerate(row["seconds"]):
                            order = paired_order(pair["case_index"], seed, repeat)
                            expected_calls.append(dict(round=repeat, position=order.index(route),
                                                       order=list(order), seconds=value))
                        if row.get("measured_calls") != expected_calls:
                            raise ValueError("paired raw samples are inconsistent with call order")
            ratios = {str(seed): item["devices"]["v0"]["seed_medians_seconds"][str(seed)] /
                      item["devices"]["cuda"]["seed_medians_seconds"][str(seed)] for seed in SEEDS}
            item.update(paired_seed_v0_over_cuda=ratios,
                        median_version_ratio=statistics.median(ratios.values()))
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
    cases = SMOKE_CASES if args.smoke else CASES
    cpu_threads, cpu_batch = (1, 16) if args.smoke else (8, 2048)
    if not args.smoke and cores["available_cores"] < 8:
        raise ValueError("the frozen benchmark requires at least eight available physical cores")
    if args.baseline_source is not None and (args.cpu_only or args.smoke):
        raise ValueError("--baseline-source requires the full CPU/CUDA benchmark")
    if args.baseline_source is not None and not (args.baseline_source / "flashrerand" / "__init__.py").is_file():
        raise ValueError("--baseline-source must contain the frozen V0 flashrerand package")
    devices = ("cpu",) if args.cpu_only else ("cpu", "cuda")
    implementations = devices + (("v0",) if args.baseline_source is not None else ())
    baseline_hashes = sources(args.baseline_source.resolve() / "flashrerand") if args.baseline_source else None
    warmup, repeats = (1, 2) if args.smoke else (2, 5)
    report = {"schema_version": 3, "benchmark": "flashrerand-v02-cpu-cuda", "status": "running",
        "created_utc": now(), "diagnostic_only": bool(args.smoke), "source_sha256": sources(),
        "baseline_source_sha256": baseline_hashes,
        "cpu_core_detection": cores, "cpu_tuning": {"jobs": [], "selected": {
            "threads": cpu_threads, "batch_size": cpu_batch,
            "reason": "fixed pre-registered configuration; no new CPU tuning"}}, "jobs": [],
        "protocol": {"cases": [list(case) for case in cases], "devices": list(devices), "seeds": list(SEEDS),
            "implementations": list(implementations),
            "data_seed": 2026, "input_generator": "numpy.default_rng(2026).standard_normal((n,p)), FP64",
            "warmup": warmup, "repeats": repeats, "cpu_threads": cpu_threads, "cpu_batch_size": cpu_batch,
            "gpu_batch_size": 16 if args.smoke else 2048, "gpu_host_threads": 1,
            "gpu_selection": "CUDA_VISIBLE_DEVICES=0; no other benchmark process should run concurrently",
            "timing_scope": "host X through validation, preparation, sampling, scoring, selection and complete host W/scores/IDs",
            "excluded_from_timing": ["process/import startup", "input generation", "untimed oracle", "file I/O"],
            "first_call_scope": ("first call per GPU version in a shared process; the earlier route can warm context/library/module caches for the later route; disk caches may also exist; neither guaranteed cold compilation nor an independent process-first call" if args.baseline_source else "process-first API call after imports and backend synchronization; module/NVRTC disk caches may exist, so this is not guaranteed cold compilation; recorded separately before additional warmups"),
            "cpu_method": "NumPy vectorized random keys + GEMM; checked full API",
            "cuda_method": "CUDA repair sampling + actual public AUTO; actual route recorded per job",
            "comparability": "same statistical task and output contract; different samplers and candidate pools even at the same seed",
            "aggregation": "within-seed median, then median of three seed medians; ratio is median of paired seed ratios",
            "oracle_scope": "selected outputs only; complete-pool correctness is tested separately",
            "order": "alternate CPU block and GPU block by case/seed; one fresh child per block",
            "gpu_version_pairing": "same_process_interleaved_gpu_versions_v1" if args.baseline_source else None,
            "gpu_version_order": "rotate first route by case/seed; mirror successive rounds; one first call, two warmups and five timed calls per route" if args.baseline_source else None,
            "gpu_version_imports": "canonical current package plus independently imported frozen V0 alias; origins and source bytes verified" if args.baseline_source else None,
            "amendment": "Release benchmark GPU versions paired within each child to reduce fresh-process confounding; earlier formal panels remain separate evidence",
            "tuning_rule": "CPU8/B2048 frozen before this release batch; GPU B2048; no runtime tuning"}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with private_log.open("x", encoding="utf-8") as stream:
        stream.write("Private child-process diagnostics. Do not publish this file; it may contain local paths.\n")
    checkpoint(partial, report)

    def execute(config, destination):
        is_baseline = config["implementation"] == "v0"
        row = launch(config, baseline_hashes if is_baseline else report["source_sha256"], private_log,
                     args.baseline_source.resolve() if is_baseline else None)
        destination.append(row)
        checkpoint(partial, report)
        print("%s %s seed=%s threads=%s batch=%s %s" % (config["case_id"], config["implementation"], config["seed"],
              config["threads"], config["batch_size"], row["status"]), flush=True)
        if row["status"] != "completed":
            raise RuntimeError("a measurement failed; its checkpoint is preserved")

    def execute_gpu_block(case, case_index, seed):
        config = config_for(case, "cuda", seed, 1, report["protocol"]["gpu_batch_size"],
                            warmup, repeats, "cuda")
        rows = launch_gpu_pair(config, report["source_sha256"], args.baseline_source,
                               baseline_hashes, case_index, private_log)
        report["jobs"].extend(rows)
        checkpoint(partial, report)
        print("%s GPU pair seed=%s %s" % (config["case_id"], seed,
              "completed" if all(row["status"] == "completed" for row in rows) else "failed"), flush=True)
        if any(row["status"] != "completed" for row in rows):
            raise RuntimeError("a paired measurement failed; its checkpoint is preserved")

    try:
        for case_index, case in enumerate(cases):
            for seed_index, seed in enumerate(SEEDS):
                order = devices if (case_index + seed_index) % 2 == 0 else tuple(reversed(devices))
                for device in order:
                    if device == "cuda" and args.baseline_source is not None:
                        execute_gpu_block(case, case_index, seed)
                    else:
                        execute(config_for(case, device, seed, cpu_threads if device == "cpu" else 1,
                            cpu_batch if device == "cpu" else report["protocol"]["gpu_batch_size"],
                            warmup, repeats, device), report["jobs"])
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
    if report.get("schema_version") not in (2, 3) or report.get("status") != "completed":
        raise ValueError("plot requires a completed version-2 or version-3 benchmark result")
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

    has_baseline = "v0" in report["protocol"]["implementations"]
    if has_baseline:
        fig, (ax, version_ax) = plt.subplots(2, 1, figsize=(10.6, 8.2),
            gridspec_kw={"height_ratios": [3, 1.15]}, sharex=True)
    else:
        fig, ax = plt.subplots(figsize=(10.6, 5.6))
    colors, offsets = {"cpu": "#6b7280", "cuda": "#087f8c"}, {"cpu": -.19, "cuda": .19}
    for device in ("cpu", "cuda"):
        x = [i + offsets[device] for i in range(len(summary))]
        values = [item["devices"][device]["median_seconds"] * 1000 for item in summary]
        label = ("NumPy CPU: keys + GEMM (%d-thread limit)" % report["cpu_tuning"]["selected"]["threads"]
                 if device == "cpu" else "CUDA: repair + AUTO (actual routes below)")
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
    labels = []
    for item in summary:
        rows = [row for row in report["jobs"] if row["config"]["case_id"] == item["case_id"]
                and row["config"]["implementation"] == "cuda"]
        route = "/".join(sorted({row["execution"]["actual_backend"] for row in rows}))
        labels.append("n=%s, p=%s\nC=%s, K=%s\n%s; CPU/GPU %.2f×" %
                      (item["n"], item["p"], item["candidates"], item["k"], route, item["median_paired_ratio"]))
    ax.set_xticks(range(len(summary)), labels)
    if has_baseline:
        version_ax.axhline(1, color="#6b7280", linestyle="--", linewidth=1)
        version_ax.bar(range(len(summary)), [item["median_version_ratio"] for item in summary],
                       color="#5172a1", width=.5, alpha=.85)
        for index, item in enumerate(summary):
            ratios = item["paired_seed_v0_over_cuda"]
            version_ax.scatter([index + shift for shift in (-.05, 0, .05)],
                               [ratios[str(seed)] for seed in SEEDS], s=16, color="#172126", zorder=4)
            label_height = max(item["median_version_ratio"], max(ratios.values()))
            version_ax.annotate("%.2f×" % item["median_version_ratio"],
                (index, label_height), xytext=(0, 7),
                textcoords="offset points", ha="center", fontsize=9)
        version_ax.set_ylabel("V0 / 0.2.0 latency\n(higher is faster)")
        version_ax.set_ylim(0, max(1, max(max(item["paired_seed_v0_over_cuda"].values()) for item in summary)) * 1.35)
        version_ax.spines[["top", "right"]].set_visible(False)
        version_ax.grid(axis="y", alpha=.2)
        version_ax.set_axisbelow(True)
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
    note = "Bars: median of 3 seed medians; dots: seed medians (5 repetitions each). Different CPU/GPU samplers; same budget and host output."
    if report["protocol"].get("gpu_version_pairing"):
        note += "\nGPU versions share one process and alternate call order; ratios use paired seed medians."
    fig.text(.08, .015, note, fontsize=7.5)
    fig.tight_layout(rect=(0, .10, 1, 1))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    runner = commands.add_parser("run", help="run serial paired measurements with frozen CPU8/B2048")
    runner.add_argument("--output", type=Path, required=True)
    runner.add_argument("--smoke", action="store_true", help="small diagnostic version of the entire pipeline")
    runner.add_argument("--cpu-only", action="store_true", help="CPU-only diagnostics; requires --smoke")
    runner.add_argument("--baseline-source", type=Path,
                        help="optional frozen V0 source directory for newly measured version comparisons")
    drawing = commands.add_parser("plot", help="render an existing completed result; requires Matplotlib")
    drawing.add_argument("input", type=Path)
    drawing.add_argument("--output", type=Path, required=True, help="new .svg, .png, or other Matplotlib-supported file")
    commands.add_parser("_worker", help=argparse.SUPPRESS)
    commands.add_parser("_gpu_pair_worker", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.command == "_worker":
        result = worker(json.load(sys.stdin))
        print(json.dumps(result, allow_nan=False))
        return int(result["status"] != "completed")
    if args.command == "_gpu_pair_worker":
        result = gpu_pair_worker(json.load(sys.stdin))
        print(json.dumps(result, allow_nan=False))
        return int(result["status"] != "completed")
    try:
        return run(args) if args.command == "run" else plot(args)
    except (OSError, ValueError, KeyError, ImportError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
