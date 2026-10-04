"""FP64 fixed-budget rerandomization on NumPy or CUDA."""

import time as _time

from .core import prepare as _prepare
from .engine import SampleResult
from .execution import VERSION as __version__, resolve_options as _resolve_options
from .execution import sample_prepared as _sample_prepared

__all__ = ["rerandomize", "SampleResult"]


def rerandomize(X, *, n1, candidates, k=1, seed=0, batch_size=2048,
                device="cuda", sampler="auto", backend="auto") -> SampleResult:
    """Return the global best k assignments from a fixed candidate budget.

    X is a host array of shape (n, p). Scores use full sample covariance
    (ddof=1) and FP64 arithmetic. Each assignment treats exactly n1 units.
    Returned assignments, scores and candidate IDs are NumPy arrays.

    The default sampler is repair on CUDA and keys on CPU. Automatic backend
    selection never changes that sampler. GEMM and Packed share the repair
    candidate stream, but floating-point scoring can affect close rankings.
    Fused routes are experimental and require an explicit backend override.
    CPU accepts only keys with auto/gemm. CUDA accepts repair with
    auto/gemm/packed/fused and sequential with auto/gemm/fused. Unsupported
    combinations raise errors; there is no automatic CPU or sampler fallback.

    Execution details are recorded in metadata['execution']. Internal modules,
    including prepared-data search, are not stable public APIs.
    """
    begin = _time.perf_counter()
    _resolve_options(device, sampler, backend)
    problem = _prepare(X, criterion="full", precision="fp64",
                       whitening="pre", method="cholesky", device=device,
                       prepare_backend="cuda-qr" if device == "cuda" else "cpu")
    result = _sample_prepared(
        problem, n1=n1, candidates=candidates, k=k, seed=seed,
        batch_size=batch_size, sampler=sampler, backend=backend,
    )
    result.timings["search"] = result.timings["total"]
    result.timings["total"] = _time.perf_counter() - begin
    return result
