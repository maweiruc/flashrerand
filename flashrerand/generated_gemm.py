"""Private GEMM path for engine-owned fixed-count proposals.

Public score() continues validating arbitrary external W. The engine uses this
for its own repair sampler or private deferred sequential path, accumulates RNG
errors, and invokes public global topk() to reject every nonfinite score before
returning any result.
"""
from .arrays import get_xp


def score_generated(problem, w, n1, *, _profile=None):
    """Score trusted proposals, optionally exposing diagnostic sub-stages.

    _profile(name, fn) must execute fn and return its result. The callback owns
    any timing synchronization. Post-whitening solves belong to epilogue.
    The ordinary path retains its original expression and allocation lifetime.
    """
    if _profile is not None and not callable(_profile):
        raise TypeError("_profile must be callable or None")
    xp = get_xp("cuda")
    data = problem.z if problem.whitening == "pre" else problem.xc
    dtype = data.dtype
    if w.ndim != 2 or w.shape[1] != problem.n or w.dtype != xp.uint8:
        raise ValueError("internal sampler produced an invalid assignment buffer")
    if _profile is not None:
        converted = _profile("w_convert", lambda: w.astype(dtype, copy=False))
        product = _profile("gemm", lambda: xp.matmul(converted, data))
        del converted

        def epilogue():
            imbalance = product - dtype.type(n1 / problem.n) * problem.correction
            if problem.whitening == "post":
                if problem.criterion == "diagonal":
                    imbalance = imbalance / xp.diag(problem.factor)
                else:
                    from cupyx.scipy.linalg import solve_triangular
                    imbalance = solve_triangular(problem.factor, imbalance.T,
                                                 lower=True, check_finite=False).T
            return dtype.type(problem.n / (n1 * (problem.n - n1))) * xp.sum(
                imbalance * imbalance, axis=1, dtype=dtype)

        return _profile("epilogue", epilogue)
    imbalance = xp.matmul(w.astype(dtype, copy=False), data) - dtype.type(n1 / problem.n) * problem.correction
    if problem.whitening == "post":
        if problem.criterion == "diagonal":
            imbalance = imbalance / xp.diag(problem.factor)
        else:
            from cupyx.scipy.linalg import solve_triangular
            imbalance = solve_triangular(problem.factor, imbalance.T, lower=True, check_finite=False).T
    return dtype.type(problem.n / (n1 * (problem.n - n1))) * xp.sum(
        imbalance * imbalance, axis=1, dtype=dtype)
