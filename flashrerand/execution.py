"""Internal execution policy shared by the public API and research benchmarks.

No online tuning, sampler substitution, or retry with another backend occurs.
Static device descriptors are cached by process and CUDA ordinal; prepared
array ownership and the current device are checked on every CUDA call.
"""

from copy import deepcopy
from functools import lru_cache
import os
import time

import numpy as np

from .arrays import get_xp
from .core import PreparedProblem
from .engine import _integer, sample_topk

VERSION = "0.2.0"
RULE_REVISION = "round3-dispatch-v1"
# The preregistered round-three gate failed; explicit Packed remains available.
AUTO_PACKED_ENABLED = False
RULE_STATUS = "disabled_after_regression_gate"


def resolve_options(device, sampler, backend):
    """Validate requested options before preparing data; resolve only sampler."""
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be 'cpu' or 'cuda'")
    if sampler not in ("auto", "repair", "sequential", "keys"):
        raise ValueError("sampler must be 'auto', 'repair', 'sequential' or 'keys'")
    if backend not in ("auto", "gemm", "packed", "fused"):
        raise ValueError("backend must be 'auto', 'gemm', 'packed' or 'fused'")
    actual_sampler = ("repair" if device == "cuda" else "keys") if sampler == "auto" else sampler
    if device == "cpu":
        if actual_sampler != "keys" or backend not in ("auto", "gemm"):
            raise ValueError("CPU requires sampler='auto'/'keys' and backend='auto'/'gemm'")
    elif actual_sampler == "keys" or (actual_sampler == "sequential" and backend == "packed"):
        raise ValueError("CUDA supports repair with auto/gemm/packed/fused and sequential with auto/gemm/fused")
    return actual_sampler


@lru_cache(maxsize=32)
def _device_properties(process_id, index):
    """Return an immutable descriptor; the PID separates inherited cache data."""
    cp = get_xp("cuda")
    properties = cp.cuda.runtime.getDeviceProperties(index)
    name = properties["name"]
    if isinstance(name, bytes):
        name = name.decode("utf-8")
    return (index, str(name), int(properties["major"]), int(properties["minor"]),
            int(properties.get("totalGlobalMem", 0)), str(cp.__version__))


def _hardware(problem):
    cp = get_xp("cuda")
    index = int(cp.cuda.runtime.getDevice())
    for key in ("x", "xc", "z", "factor", "correction"):
        value = getattr(problem, key)
        if value is not None and (not isinstance(value, cp.ndarray) or int(value.device.id) != index):
            raise ValueError("prepared arrays must belong to the current CUDA device")
    ordinal, name, major, minor, memory, cupy_version = _device_properties(os.getpid(), index)
    # Mutable caller metadata never aliases the immutable cache entry.
    return dict(index=ordinal, name=name, compute_capability=[major, minor],
                memory_bytes=memory, cupy_version=cupy_version)


def _choose_backend(problem, *, sampler, backend, n1, candidates, k, batch_size, hardware):
    if backend != "auto":
        return backend, "explicit_backend"
    if problem.device == "cpu":
        return "gemm", "cpu_default"
    if sampler != "repair":
        return "gemm", "sampler_default"
    if not AUTO_PACKED_ENABLED:
        return "gemm", "packed_auto_disabled"
    if ((problem.criterion, problem.precision, problem.whitening, problem.method,
            problem.metadata.get("prepare_backend")) !=
            ("full", "fp64", "pre", "cholesky", "cuda-qr") or
            any(problem.metadata.get("experimental_prepare_flags", {}).values())):
        return "gemm", "prepared_settings_outside_frozen_rule"
    if (hardware.get("name") not in ("NVIDIA A100-PCIE-40GB", "A100-PCIE-40GB") or
            hardware.get("compute_capability") != [8, 0]):
        return "gemm", "hardware_outside_frozen_rule"
    if not (512 <= problem.n <= 1024 and 8 <= problem.p <= 32):
        return "gemm", "shape_outside_frozen_rule"
    if abs(2 * n1 - problem.n) > 1:
        return "gemm", "allocation_outside_frozen_rule"
    if not (2051 <= candidates <= 65536 and 1 <= k <= 64 and batch_size == 2048):
        return "gemm", "budget_or_batch_outside_frozen_rule"
    return "packed", "inside_frozen_packed_region"


def _validate_problem(problem):
    if not isinstance(problem, PreparedProblem):
        raise TypeError("problem must be a PreparedProblem")
    if (problem.criterion, problem.precision, problem.whitening) != ("full", "fp64", "pre"):
        raise ValueError("execution requires full covariance, FP64 and pre-whitening")
    if problem.z is None or problem.correction is None:
        raise ValueError("execution requires prepared data and correction")


def _execution_record(problem, result, *, requested_sampler, requested_backend,
                      sampler, backend, hardware, reason, start_id, candidates):
    if backend == "packed":
        kernel = "score_tiled_variant<double,true,8,64>"
        sample_path = "philox_repair_direct_packed"
        representation = "global_packed_uint8"
        tile = dict(candidates=8, subjects=64)
    elif backend == "fused":
        max_p = next(size for size in (32, 128, 512, 1024) if problem.p <= size)
        kernel = ("score_fused_repair" if sampler == "repair" else
                  f"score_fused_sequential<double,{max_p}>")
        sample_path = "philox_" + sampler + "_fused"
        representation = "shared_packed_uint32_little_endian" if sampler == "repair" else "on_the_fly"
        tile = dict(candidates=8, subjects=64) if sampler == "repair" else None
    else:
        kernel = "cupy_fp64_gemm" if problem.device == "cuda" else "numpy_fp64_gemm"
        sample_path = "philox_" + sampler
        representation, tile = "global_dense_uint8", None
    return dict(
        requested_device=problem.device, actual_device=problem.device,
        requested_sampler=requested_sampler, actual_sampler=sampler,
        requested_backend=requested_backend, actual_backend=backend,
        reason=reason, rule_revision=RULE_REVISION, rule_status=RULE_STATUS,
        auto_packed_enabled=bool(AUTO_PACKED_ENABLED), experimental=backend == "fused",
        kernel=kernel, sampling_path=sample_path, assignment_representation=representation,
        selected_replay=result.metadata.get("selected_replay", "checked_dense_to_host"),
        tile=tile, workspace="per_batch_allocations", workspace_reuse=False,
        hardware=hardware, hardware_cache="process_and_device" if hardware is not None else None,
        rng="philox4x32-10", replay_version=result.metadata["replay_version"],
        seed=result.metadata["seed"],
        candidate_id_range=dict(start=start_id, stop_exclusive=start_id + candidates),
        batch_size=result.metadata["batch_size"],
        preprocessing=dict(criterion=problem.criterion, precision=problem.precision,
            whitening=problem.whitening, method=problem.method,
            prepare_backend=problem.metadata.get("prepare_backend"),
            experimental_flags=deepcopy(problem.metadata.get("experimental_prepare_flags", {}))),
        software=dict(flashrerand=VERSION, numpy=np.__version__,
                      cupy=hardware["cupy_version"] if hardware else None),
    )


def sample_prepared(problem, *, n1, candidates, k=1, seed=0, batch_size=2048,
                    sampler="auto", backend="auto", _start_id=0):
    """Internal prepared search, timed with identical public dispatch overhead.

    There is no stable public prepared API. ``_start_id`` is an internal replay
    test hook. Every result contains independent host arrays and metadata.
    """
    begin = time.perf_counter()
    _validate_problem(problem)
    actual_sampler = resolve_options(problem.device, sampler, backend)
    n1 = _integer("n1", n1, 1, problem.n - 1)
    candidates = _integer("candidates", candidates, 1, 2**63 - 1)
    k = _integer("k", k, 1, candidates)
    batch_size = _integer("batch_size", batch_size, 1, 2**31 - 1)
    seed = _integer("seed", seed, 0, 2**64 - 1)
    start_id = _integer("start_id", _start_id, 0, 2**64 - 1)
    if start_id + candidates > 2**64:
        raise ValueError("candidate ID range overflows uint64")
    if backend == "fused":
        if actual_sampler == "repair" and (problem.n > 1024 or problem.p > 32):
            raise ValueError("repair + fused requires n <= 1024 and p <= 32")
        if actual_sampler == "sequential" and problem.p > 1024:
            raise ValueError("sequential + fused requires p <= 1024")
    hardware = _hardware(problem) if problem.device == "cuda" else None
    actual_backend, reason = _choose_backend(problem, sampler=actual_sampler, backend=backend,
        n1=n1, candidates=candidates, k=k, batch_size=batch_size, hardware=hardware)
    options = dict(n1=n1, candidates=candidates, k=k, seed=seed, batch_size=batch_size,
                   sampler=actual_sampler, balance="gemm", selection="all",
                   validation="deferred" if problem.device == "cuda" else "checked",
                   output="assignments", start_id=start_id)
    if actual_backend == "packed":
        options["_packed_repair"] = True
    if problem.device == "cuda" and actual_sampler == "sequential":
        options["_deferred_sequential"] = True
        if actual_backend == "fused":
            options["balance"] = "fused"
    elif actual_backend == "fused":
        options["_repair_fused"] = True
    result = sample_topk(problem, **options)
    # The engine already owns a fresh metadata snapshot for this result.
    result.metadata["execution"] = _execution_record(problem, result,
        requested_sampler=sampler, requested_backend=backend, sampler=actual_sampler,
        backend=actual_backend, hardware=hardware, reason=reason,
        start_id=start_id, candidates=candidates)
    result.metadata["version"] = VERSION
    result.timings["engine"] = result.timings["total"]
    result.timings["total"] = time.perf_counter() - begin
    return result
