"""Opt-in CUDA dispatch experiment; not part of the public V0 API.

The rule's validation evidence belongs to the accompanying experiment report. It changes
only repair storage/scoring: candidate addresses, sampler and replay stay fixed.
There is no tuning, prepared-data cache, or fallback after a failed engine call.
"""
import time

from .arrays import get_xp
from .core import PreparedProblem, prepare
from .engine import _integer, sample_topk

RULE_REVISION = "round2-dispatch-v1"
AUTO_PACKED_ROUTE = "C1"
BACKENDS = {"gemm": "A", "packed": "C1", "workspace": "W1"}
PREPARE_OPTIONS = dict(criterion="full", precision="fp64", whitening="pre",
                       method="cholesky", device="cuda", prepare_backend="cuda-qr")


def _choose_route(*, n, p, n1, candidates, k, batch_size, device,
                  criterion="full", precision="fp64", whitening="pre", method="cholesky",
                  prepare_backend="cuda-qr", prepare_flags=None):
    """Pure conservative rule. Callers validate arguments before selection."""
    if ((criterion, precision, whitening, method, prepare_backend) !=
            ("full", "fp64", "pre", "cholesky", "cuda-qr") or any((prepare_flags or {}).values())):
        return "A", "prepared_settings_outside_frozen_rule"
    name = device.get("name", "")
    if name not in ("NVIDIA A100-PCIE-40GB", "A100-PCIE-40GB") or device.get("compute_capability") != [8, 0]:
        return "A", "hardware_outside_frozen_rule"
    if not (512 <= n <= 1024 and 8 <= p <= 32):
        return "A", "shape_outside_frozen_rule"
    if abs(2 * n1 - n) > 1:
        return "A", "allocation_outside_frozen_rule"
    if not (2051 <= candidates <= 65536 and 1 <= k <= 64 and batch_size == 2048):
        return "A", "budget_or_batch_outside_frozen_rule"
    return AUTO_PACKED_ROUTE, "inside_frozen_packed_region"


def _hardware(problem):
    """Read the actual current device; do not cache or use a supplied label."""
    cp = get_xp("cuda")
    index = int(cp.cuda.runtime.getDevice())
    # The engine uses the current device/stream. Reject stale prepared arrays on
    # another device instead of dispatching with an unrelated hardware label.
    for key in ("x", "xc", "z", "factor", "correction"):
        value = getattr(problem, key)
        if value is not None and (not isinstance(value, cp.ndarray) or int(value.device.id) != index):
            raise ValueError("prepared arrays must belong to the current CUDA device")
    properties = cp.cuda.runtime.getDeviceProperties(index)
    name = properties["name"]
    if isinstance(name, bytes):
        name = name.decode("utf-8")
    return dict(index=index, name=str(name),
                compute_capability=[int(properties["major"]), int(properties["minor"])])


def _validate_problem(problem):
    if not isinstance(problem, PreparedProblem):
        raise TypeError("problem must be a PreparedProblem")
    if problem.device != "cuda":
        raise ValueError("experimental dispatch requires a CUDA prepared problem")
    if (problem.z if problem.whitening == "pre" else problem.xc) is None or problem.correction is None:
        raise ValueError("experimental dispatch requires prepared data and correction")


def _route_for_problem(problem, *, n1, candidates, k, batch_size, device):
    return _choose_route(n=problem.n, p=problem.p, n1=n1, candidates=candidates,
        k=k, batch_size=batch_size, device=device, criterion=problem.criterion,
        precision=problem.precision, whitening=problem.whitening, method=problem.method,
        prepare_backend=problem.metadata.get("prepare_backend"),
        prepare_flags=problem.metadata.get("experimental_prepare_flags"))


def sample_prepared(problem, *, n1, candidates, k=1, seed=0, batch_size=2048,
                    backend="auto", _start_id=0):
    """Search a CUDA problem, including per-call dispatch overhead.

    ``gemm``, ``packed`` and ``workspace`` force A, C1 and W1 respectively.
    Uncovered prepared settings use the existing GEMM path without changing the
    statistical criterion. ``_start_id`` is an internal full-pool/replay test hook, not a new public
    sampler contract. Host output arrays are independently owned by each call.
    """
    begin = time.perf_counter()
    _validate_problem(problem)
    if backend != "auto" and backend not in BACKENDS:
        raise ValueError("backend must be auto, gemm, packed or workspace")
    # Reuse the engine's input rules; invalid input must not choose a method.
    n1 = _integer("n1", n1, 1, problem.n - 1)
    candidates = _integer("candidates", candidates, 1, 2**63 - 1)
    k = _integer("k", k, 1, candidates)
    batch_size = _integer("batch_size", batch_size, 1, 2**31 - 1)
    seed = _integer("seed", seed, 0, 2**64 - 1)
    start_id = _integer("start_id", _start_id, 0, 2**64 - 1)
    if start_id + candidates > 2**64:
        raise ValueError("candidate ID range overflows uint64")
    device = _hardware(problem)
    if backend == "auto":
        route, reason = _route_for_problem(problem, n1=n1, candidates=candidates,
            k=k, batch_size=batch_size, device=device)
    else:
        route, reason = BACKENDS[backend], "explicit_backend"
    options = dict(n1=n1, candidates=candidates, k=k, seed=seed, batch_size=batch_size,
                   sampler="repair", balance="gemm", selection="all", validation="deferred",
                   output="assignments", start_id=start_id)
    if route in ("C1", "W1"):
        options["_packed_repair"] = True
    if route == "W1":
        options["_packed_workspace"] = True
    result = sample_topk(problem, **options)
    result.metadata["dispatch"] = dict(
        experimental=True, acceptance="opt_in_research", requested_backend=backend,
        actual_route=route, sampler="repair", rng=result.metadata["replay_version"],
        scorer=problem.precision + "_gemm" if route == "A" else "fp64_packed_tiled_c8_s64",
        tile=None if route == "A" else dict(candidates=8, subjects=64),
        batch_size=batch_size, device=device, reason=reason, revision=RULE_REVISION,
        cache="none", criterion=problem.criterion, precision=problem.precision,
        whitening=problem.whitening, method=problem.method,
        prepare_backend=problem.metadata.get("prepare_backend"),
        prepare_flags=dict(problem.metadata.get("experimental_prepare_flags", {})))
    result.timings["engine"] = result.timings["total"]
    result.timings["total"] = time.perf_counter() - begin
    return result


def rerandomize(X, *, n1, candidates, k=1, seed=0, batch_size=2048, backend="auto"):
    """Prepare fresh host X and return complete host results using CUDA only.

    This is an explicit experimental API. The published public entry point and
    its defaults are unchanged. All timing includes dispatch and host outputs.
    """
    begin = time.perf_counter()
    problem = prepare(X, **PREPARE_OPTIONS)
    result = sample_prepared(problem, n1=n1, candidates=candidates, k=k, seed=seed,
                             batch_size=batch_size, backend=backend)
    result.timings["search"] = result.timings["total"]
    result.timings["total"] = time.perf_counter() - begin
    return result
