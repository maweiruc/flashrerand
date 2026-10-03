"""Private GEMM path for engine-owned fixed-count proposals.

Public score() continues validating arbitrary external W. The engine uses this
only for its own repair sampler, accumulates RNG errors, and invokes public
global topk() to reject every nonfinite score before returning any result.
"""
from .arrays import get_xp


def score_generated(problem, w, n1):
    xp = get_xp("cuda")
    data = problem.z if problem.whitening == "pre" else problem.xc
    dtype = data.dtype
    if w.ndim != 2 or w.shape[1] != problem.n or w.dtype != xp.uint8:
        raise ValueError("internal sampler produced an invalid assignment buffer")
    imbalance = xp.matmul(w.astype(dtype, copy=False), data) - dtype.type(n1 / problem.n) * problem.correction
    if problem.whitening == "post":
        if problem.criterion == "diagonal":
            imbalance = imbalance / xp.diag(problem.factor)
        else:
            from cupyx.scipy.linalg import solve_triangular
            imbalance = solve_triangular(problem.factor, imbalance.T, lower=True, check_finite=False).T
    return dtype.type(problem.n / (n1 * (problem.n - n1))) * xp.sum(
        imbalance * imbalance, axis=1, dtype=dtype)
