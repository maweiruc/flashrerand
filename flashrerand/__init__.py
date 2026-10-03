"""FP64 fixed-budget rerandomization on NumPy or CUDA."""

import time as _time

from .core import prepare as _prepare
from .engine import SampleResult, sample_topk as _sample_topk

__version__ = "0.1.0"
__all__ = ["rerandomize", "SampleResult"]


def rerandomize(X, *, n1, candidates, k=1, seed=0, device="cuda",
                batch_size=2048) -> SampleResult:
    """Return the global best k assignments from a fixed candidate budget.

    X is a host array of shape (n, p). Scores use full sample covariance
    (ddof=1) and FP64 arithmetic. Each assignment treats exactly n1 units.
    Returned assignments, scores and candidate IDs are NumPy arrays.

    CUDA uses repair sampling; CPU uses vectorized random keys. The same
    seed therefore does not produce the same pool across devices. There is
    no automatic CPU fallback. Internal modules are not stable public APIs.
    """
    if device not in ("cpu", "cuda"):
        raise ValueError("device must be 'cpu' or 'cuda'")
    begin = _time.perf_counter()
    problem = _prepare(X, criterion="full", precision="fp64",
                       whitening="pre", method="cholesky", device=device,
                       prepare_backend="cuda-qr" if device == "cuda" else "cpu")
    result = _sample_topk(
        problem, n1=n1, candidates=candidates, k=k, seed=seed,
        sampler="repair" if device == "cuda" else "keys",
        balance="gemm", selection="all", batch_size=batch_size,
        output="assignments", validation="deferred" if device == "cuda" else "checked",
    )
    result.timings["search"] = result.timings["total"]
    result.timings["total"] = _time.perf_counter() - begin
    result.metadata["version"] = __version__
    return result
