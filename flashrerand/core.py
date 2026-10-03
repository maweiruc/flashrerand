"""Covariate preparation and auditable CPU/CuPy balance-score paths.

Both criteria return (n1*n0/n) times the squared standardized mean difference.
The diagonal criterion uses population variances (ddof=0); the full criterion
uses the sample covariance (ddof=1). These are deliberately different criteria.
"""

from dataclasses import dataclass
from typing import Any, Dict
from contextlib import contextmanager, nullcontext
import hashlib
import time

import numpy as np

from .arrays import _integer, cuda_math_metadata, get_xp, to_numpy


@dataclass
class PreparedProblem:
    n: int
    p: int
    device: str
    precision: str
    criterion: str
    whitening: str
    method: str
    x: Any
    xc: Any
    z: Any
    factor: Any
    correction: Any
    metadata: Dict[str, Any]


def _matrix(X):
    X = to_numpy(X)
    if X.ndim != 2 or X.shape[0] < 2 or X.shape[1] < 1:
        raise ValueError("X must have shape (n,p), with n >= 2 and p >= 1")
    if X.dtype.kind not in "fiu":
        raise TypeError("X must contain real numeric values")
    X = np.asarray(X, dtype=np.float64)
    if not np.all(np.isfinite(X)):
        raise ValueError("X must contain only finite values")
    return X


def _input_fingerprint(X64):
    # Preserve the existing byte format without creating two full-size bytes
    # copies (tobytes followed by concatenation with the prefix).
    digest = hashlib.sha256(("float64:%d:%d:" % X64.shape).encode("ascii"))
    digest.update(memoryview(np.ascontiguousarray(X64)).cast("B"))
    return digest.hexdigest()


class _PrepareTimer:
    """Explicit synchronization is only enabled for a diagnostic call."""

    def __init__(self, enabled, xp):
        self.enabled, self.xp, self.stages = enabled, xp, {}
        self.begin = None

    def sync(self):
        if self.xp is not np:
            self.xp.cuda.get_current_stream().synchronize()

    @contextmanager
    def stage(self, name):
        if not self.enabled:
            yield
            return
        self.sync()
        begin = time.perf_counter()
        if self.begin is None:
            self.begin = begin
        try:
            yield
        finally:
            self.sync()
            self.stages[name] = self.stages.get(name, 0.0) + time.perf_counter() - begin

    def result(self):
        if not self.enabled:
            return None
        self.sync()
        return {
            "scope": "this diagnostic prepare call; synchronized wall time, including host validation and transfers",
            "total_seconds": time.perf_counter() - self.begin,
            "stages_seconds": self.stages,
        }


def _prepare_spectrum(singular, n, p, condition_limit):
    """Apply the unchanged CPU rank/condition thresholds to a spectrum."""
    singular = np.asarray(singular, dtype=np.float64)
    if singular.shape != (p,) or not np.all(np.isfinite(singular)):
        raise ValueError("covariate singular values are nonfinite")
    tolerance = np.finfo(np.float64).eps * max(n, p) * singular[0]
    if singular[-1] <= tolerance:
        raise ValueError("centered covariates must have full column rank")
    condition_xc = float(singular[0] / singular[-1])
    condition_covariance = condition_xc ** 2
    if condition_covariance > condition_limit:
        raise ValueError("sample covariance exceeds condition_limit")
    return condition_xc, condition_covariance, float(tolerance)


def prepare(X, criterion="diagonal", precision="fp32", whitening="pre",
            method="cholesky", device="cpu", condition_limit=1e10,
            prepare_backend="cpu", profile=False):
    """Prepare covariates with explicit CPU, hybrid CUDA, or CUDA QR backends.

``cpu`` retains CPU arithmetic and optionally transfers its result. ``cuda``
retains the original CPU FP64 diagnostics, but prepares the working arrays on
the GPU. ``cuda-qr`` replaces the tall CPU diagnostic SVD with GPU QR + SVD(R);
it retains CPU FP64 centering so the diagnostic matrix is unchanged. Near a
rank/condition boundary it falls back to the original CPU SVD. No columns are
dropped or regularized. ``profile=True`` instruments this call; benchmarks
must invoke it separately from their uninstrumented timing passes.
"""
    if criterion not in ("diagonal", "full"):
        raise ValueError("criterion must be 'diagonal' or 'full'")
    if precision not in ("fp32", "fp64"):
        raise ValueError("precision must be 'fp32' or 'fp64'")
    if whitening not in ("pre", "post"):
        raise ValueError("whitening must be 'pre' or 'post'")
    if method not in ("cholesky", "qr"):
        raise ValueError("method must be 'cholesky' or 'qr'")
    if prepare_backend not in ("cpu", "cuda", "cuda-qr"):
        raise ValueError("prepare_backend must be 'cpu', 'cuda', or 'cuda-qr'")
    if prepare_backend != "cpu" and device != "cuda":
        raise ValueError("CUDA prepare backends require device='cuda'")
    if not isinstance(profile, (bool, np.bool_)):
        raise TypeError("profile must be boolean")
    if isinstance(condition_limit, (bool, np.bool_)):
        raise TypeError("condition_limit must be a finite number >= 1")
    try:
        condition_limit = float(condition_limit)
    except (TypeError, ValueError) as exc:
        raise TypeError("condition_limit must be a finite number >= 1") from exc
    if not np.isfinite(condition_limit) or condition_limit < 1:
        raise ValueError("condition_limit must be a finite number >= 1")
    xp = get_xp(device)
    work_xp = np if prepare_backend == "cpu" else xp
    timer = _PrepareTimer(bool(profile), xp)
    with timer.stage("host_validation_and_centering"):
        X64 = _matrix(X)
        n, p = X64.shape
        with np.errstate(over="ignore", invalid="ignore"):
            centered64 = X64 - X64.mean(axis=0)
            variances64 = np.mean(centered64 * centered64, axis=0)
        if not np.all(np.isfinite(centered64)) or not np.all(np.isfinite(variances64)):
            raise ValueError("centering or variance overflowed; rescale covariates explicitly")
        if np.any(variances64 <= 0) or np.any(np.all(X64 == X64[0], axis=0)):
            raise ValueError("constant covariate columns are not supported")
        if criterion == "full" and p >= n - 1:
            raise ValueError("full Mahalanobis requires p < n-1")
    with timer.stage("math_mode"):
        cuda_math = cuda_math_metadata() if device == "cuda" else None
    gpu_centered64 = diagnostic_q = diagnostic_r = None
    singular = tolerance = None
    diagnostic_backend = "not_required"
    fallback_reasons = []
    gpu_diagnostic_values = None
    diagnostic_qr_mode = None
    condition_xc = condition_covariance = None
    if criterion == "full":
        if prepare_backend == "cuda-qr":
            from cupyx import errstate
            with timer.stage("diagnostic_transfer"):
                gpu_centered64 = xp.asarray(np.ascontiguousarray(centered64))
            with timer.stage("diagnostic_gpu_qr_svd"):
                try:
                    with errstate(linalg="raise"):
                        # Only an FP64 QR whitening path can reuse Q. Other
                        # paths need R for the spectrum, so skip forming Q.
                        diagnostic_qr_mode = "reduced" if precision == "fp64" and method == "qr" else "r"
                        if diagnostic_qr_mode == "reduced":
                            diagnostic_q, diagnostic_r = xp.linalg.qr(gpu_centered64, mode="reduced")
                        else:
                            diagnostic_r = xp.linalg.qr(gpu_centered64, mode="r")
                        singular = to_numpy(xp.linalg.svd(diagnostic_r, compute_uv=False))
                except np.linalg.LinAlgError:
                    fallback_reasons.append("gpu_qr_or_svd_failure")
                if singular is not None:
                    if singular.shape != (p,) or not np.all(np.isfinite(singular)):
                        fallback_reasons.append("nonfinite_gpu_spectrum")
                    else:
                        rank_tol = np.finfo(np.float64).eps * max(n, p) * singular[0]
                        # These guards trigger CPU re-evaluation, never relax a
                        # threshold. They are deliberately conservative rather
                        # than a claim of a rigorous QR error bound.
                        ratio = float(singular[-1] / singular[0]) if singular[0] > 0 else 0.0
                        with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                            condition = float(np.square(np.divide(np.float64(1.0), ratio)))
                        gpu_diagnostic_values = {
                            "singular_min": float(singular[-1]), "singular_max": float(singular[0]),
                            "rank_tolerance": float(rank_tol),
                            "condition_covariance": condition if np.isfinite(condition) else None,
                        }
                        if singular[-1] <= 10.0 * rank_tol:
                            fallback_reasons.append("rank_boundary_guard")
                        if condition >= condition_limit / 10.0:
                            fallback_reasons.append("condition_boundary_guard")
                if not fallback_reasons:
                    diagnostic_backend = "cuda_qr_svd_r"
            if fallback_reasons:
                with timer.stage("diagnostic_cpu_fallback"):
                    singular = np.linalg.svd(centered64, compute_uv=False)
                diagnostic_backend = "cpu_svd_fallback"
                # A failed GPU factorization must not supply the working QR.
                if "gpu_qr_or_svd_failure" in fallback_reasons or "nonfinite_gpu_spectrum" in fallback_reasons:
                    diagnostic_q = diagnostic_r = None
        else:
            with timer.stage("diagnostic_cpu_svd"):
                singular = np.linalg.svd(centered64, compute_uv=False)
            diagnostic_backend = "cpu_svd"
        condition_xc, condition_covariance, tolerance = _prepare_spectrum(
            singular, n, p, condition_limit)
    dtype = np.dtype("float32" if precision == "fp32" else "float64")
    q = None
    qr_reused = False
    centering_device = "cpu" if work_xp is np else "cuda"
    centering_fallback = None
    centering_guard = float(1.0 / np.sqrt(np.finfo(dtype).eps))
    # When offsets dominate variation, different reduction trees can change
    # the working mean appreciably. Preserve the CPU path for those inputs.
    large_offset = bool(np.any(np.max(np.abs(X64), axis=0) / np.sqrt(variances64) > centering_guard))
    typed_rank_backend = "not_required"
    typed_rank_tolerance = None
    typed_rank_fallback = []
    typed_rank_gpu_spectrum = None
    if work_xp is np:
        linalg_errors = nullcontext
    else:
        from cupyx import errstate
        linalg_errors = lambda: errstate(linalg="raise")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        with timer.stage("working_input_transfer"):
            x = X64.astype(dtype) if work_xp is np else work_xp.asarray(X64, dtype=dtype)
        with timer.stage("working_centering"):
            if precision == "fp64" and work_xp is np:
                xc = centered64
            elif precision == "fp64" and gpu_centered64 is not None:
                xc = gpu_centered64
                centering_device = "cpu"
            elif work_xp is not np and large_offset:
                host_x = X64.astype(dtype)
                host_xc = centered64 if precision == "fp64" else host_x - host_x.mean(axis=0)
                xc = work_xp.asarray(np.ascontiguousarray(host_xc))
                centering_device = "cpu"
                centering_fallback = "large_offset_reduction_guard"
            else:
                xc = x - x.mean(axis=0)
            if not bool(work_xp.all(work_xp.isfinite(xc))):
                raise ValueError("covariates cannot be centered in the requested precision")
            if precision == "fp32" and bool(work_xp.any(work_xp.all(x == x[0], axis=0))):
                if criterion == "full":
                    raise ValueError("requested precision loses the covariate matrix rank")
                raise ValueError("variance is zero in the requested precision")
        # FP64 already passed the original centered64 diagnostic. FP32 must
        # independently check rank *after* casting and working-precision mean.
        if criterion == "full" and precision == "fp32":
            with timer.stage("working_precision_rank"):
                if prepare_backend == "cuda-qr":
                    typed_singular = None
                    try:
                        with linalg_errors():
                            typed_r = work_xp.linalg.qr(xc.astype(work_xp.float64), mode="r")
                            typed_singular = to_numpy(work_xp.linalg.svd(typed_r, compute_uv=False))
                    except np.linalg.LinAlgError:
                        typed_rank_fallback.append("gpu_qr_or_svd_failure")
                    if typed_singular is not None:
                        if typed_singular.shape != (p,) or not np.all(np.isfinite(typed_singular)):
                            typed_rank_fallback.append("nonfinite_gpu_spectrum")
                        else:
                            typed_rank_tolerance = float(np.finfo(np.float64).eps * max(n, p) * typed_singular[0])
                            typed_rank_gpu_spectrum = {
                                "singular_min": float(typed_singular[-1]),
                                "singular_max": float(typed_singular[0]),
                                "rank_tolerance": typed_rank_tolerance,
                                "qr_mode": "r",
                            }
                            # This is only a fallback guard. The CPU fallback
                            # uses the original matrix_rank threshold, without
                            # accepting a matrix merely because it is close.
                            if typed_singular[-1] <= 10.0 * typed_rank_tolerance:
                                typed_rank_fallback.append("rank_boundary_guard")
                    if typed_rank_fallback:
                        if np.linalg.matrix_rank(to_numpy(xc).astype(np.float64)) < p:
                            raise ValueError("requested precision loses the covariate matrix rank")
                        typed_rank_backend = "cpu_svd_fallback_of_working_centered_matrix"
                    else:
                        typed_rank_backend = "cuda_qr_svd_r_of_working_centered_matrix"
                else:
                    if np.linalg.matrix_rank(to_numpy(xc).astype(np.float64)) < p:
                        raise ValueError("requested precision loses the covariate matrix rank")
                    typed_rank_backend = "cpu_svd_of_working_centered_matrix"
        with timer.stage("factorization"):
            try:
                if criterion == "diagonal":
                    variance = work_xp.mean(xc * xc, axis=0)
                    if not bool(work_xp.all(work_xp.isfinite(variance))) or bool(work_xp.any(variance <= 0)):
                        raise ValueError("variance is zero or nonfinite in the requested precision")
                    factor = work_xp.diag(work_xp.sqrt(variance)).astype(dtype, copy=False)
                else:
                    with linalg_errors():
                        if method == "cholesky":
                            covariance = (xc.T @ xc) / dtype.type(n - 1)
                            if not bool(work_xp.all(work_xp.isfinite(covariance))):
                                raise ValueError("covariance overflowed in the requested precision")
                            factor = work_xp.linalg.cholesky(covariance)
                        else:
                            if precision == "fp64" and diagnostic_q is not None:
                                q, r = diagnostic_q, diagnostic_r
                                qr_reused = True
                            else:
                                q, r = work_xp.linalg.qr(xc, mode="reduced")
                            factor = r.T / dtype.type(np.sqrt(n - 1))
            except np.linalg.LinAlgError as exc:
                raise ValueError("covariance factorization failed in the requested precision") from exc
            if not bool(work_xp.all(work_xp.isfinite(factor))) or bool(work_xp.any(work_xp.diag(factor) == 0)):
                raise ValueError("nonfinite or singular whitening factor")
        z = None
        with timer.stage("whitening"):
            if whitening == "pre":
                if criterion == "diagonal":
                    z = xc / work_xp.diag(factor)
                elif q is not None:
                    z = q * dtype.type(np.sqrt(n - 1))
                elif work_xp is np:
                    z = np.linalg.solve(factor, xc.T).T
                else:
                    from cupyx.scipy.linalg import solve_triangular
                    z = solve_triangular(factor, xc.T, lower=True, check_finite=False).T
                z = work_xp.ascontiguousarray(z, dtype=dtype)
                if not bool(work_xp.all(work_xp.isfinite(z))):
                    raise ValueError("whitening produced nonfinite values")
        with timer.stage("correction"):
            correction = work_xp.sum(z if z is not None else xc, axis=0, dtype=dtype)
            if not bool(work_xp.all(work_xp.isfinite(correction))):
                raise ValueError("centering correction overflowed")
    with timer.stage("metadata_and_fingerprint"):
        metadata = {
        "ddof": 0 if criterion == "diagonal" else 1,
        "score_normalization": "(n1*n0/n) * squared_standardized_mean_difference",
        "preprocessing_dtype": dtype.name,
        "preprocessing_device": "cpu" if work_xp is np else "cuda",
        "prepare_backend": prepare_backend,
        "centering_device": centering_device,
        "centering_fallback": centering_fallback,
        "centering_dynamic_range_guard": centering_guard,
        "diagnostic_dtype": "float64",
        "diagnostic_backend": diagnostic_backend,
        "diagnostic_fallback": fallback_reasons,
        "diagnostic_fallback_guards": {"rank_tolerance_multiplier": 10.0, "condition_limit_divisor": 10.0},
        "diagnostic_gpu_spectrum": gpu_diagnostic_values,
        "diagnostic_qr_mode": diagnostic_qr_mode,
        "diagnostic_rank_tolerance": tolerance,
        "diagnostic_singular_min": None if singular is None else float(singular[-1]),
        "diagnostic_singular_max": None if singular is None else float(singular[0]),
        "working_precision_rank_backend": typed_rank_backend,
        "working_precision_rank_gpu_tolerance": typed_rank_tolerance,
        "working_precision_rank_gpu_spectrum": typed_rank_gpu_spectrum,
        "working_precision_rank_fallback": typed_rank_fallback,
        "qr_diagnostic_reused": qr_reused,
        "condition_xc": condition_xc,
        "condition_covariance": condition_covariance,
        "condition_limit": condition_limit,
        "correction_space": "whitened" if whitening == "pre" else "centered",
        "max_abs_centering_residual": float(work_xp.max(work_xp.abs(xc.sum(axis=0)))),
        "factorization": method if criterion == "full" else "diagonal",
        "input_sha256": _input_fingerprint(X64),
        }
        if cuda_math is not None:
            metadata["cuda_math"] = cuda_math
    with timer.stage("output_transfer_and_layout"):
        problem = PreparedProblem(
            n, p, device, precision, criterion, whitening, method,
            xp.asarray(work_xp.ascontiguousarray(x)), xp.asarray(work_xp.ascontiguousarray(xc)),
            None if z is None else xp.asarray(z),
            xp.asarray(work_xp.ascontiguousarray(factor)), xp.asarray(correction), metadata,
        )
    metadata["prepare_profile"] = timer.result()
    return problem


def score_scale(problem, n1):
    n1 = _integer(n1, "n1", 1, problem.n - 1)
    return problem.n / (n1 * (problem.n - n1))


def _assignments(W, n, n1, xp):
    W = xp.asarray(W)
    if W.ndim != 2 or W.shape[1] != n:
        raise ValueError("W must have shape (B,n)")
    if W.dtype.kind not in "biuf":
        raise TypeError("W must contain real binary values")
    if not bool(xp.all((W == 0) | (W == 1))):
        raise ValueError("W must contain only zero and one")
    if not bool(xp.all(xp.sum(W, axis=1, dtype=xp.int64) == n1)):
        raise ValueError("every assignment must contain exactly n1 treated subjects")
    return W


def score(problem, W, n1, backend="gemm", representation="dense"):
    """Score fixed-size candidates; custom CUDA paths are loaded only on demand."""
    if not isinstance(problem, PreparedProblem):
        raise TypeError("problem must be returned by prepare()")
    n1 = _integer(n1, "n1", 1, problem.n - 1)
    if representation not in ("dense", "packed"):
        raise ValueError("representation must be 'dense' or 'packed'")
    if not isinstance(backend, str):
        raise TypeError("backend must be a string")
    xp = get_xp(problem.device)
    if problem.device == "cuda" and backend != "gemm":
        from .cuda_backend import score_cuda
        return score_cuda(problem, W, n1, backend=backend, representation=representation)
    if backend not in ("gemm", "gather", "reduce"):
        raise ValueError("CPU backend must be 'gemm', 'gather', or 'reduce'")
    if representation == "packed":
        from .samplers import unpack
        W = unpack(W, problem.n, device=problem.device)
    W = _assignments(W, problem.n, n1, xp)
    data = problem.z if problem.whitening == "pre" else problem.xc
    dtype = data.dtype
    pi = dtype.type(n1 / problem.n)
    if backend == "gemm":
        if problem.device == "cuda":
            problem.metadata["cuda_math"] = cuda_math_metadata()
        imbalance = xp.matmul(W.astype(dtype, copy=False), data) - pi * problem.correction
    elif backend == "reduce":
        # Intentionally simple CPU reference; no (B,n,p) temporary is required.
        imbalance = xp.empty((W.shape[0], problem.p), dtype=dtype)
        for i in range(W.shape[0]):
            imbalance[i] = xp.sum(W[i, :, None] * data, axis=0) - pi * problem.correction
    else:
        imbalance = xp.empty((W.shape[0], problem.p), dtype=dtype)
        use_treated = n1 <= problem.n - n1
        for i in range(W.shape[0]):
            if use_treated:
                imbalance[i] = xp.sum(data[W[i] == 1], axis=0) - pi * problem.correction
            else:
                imbalance[i] = (dtype.type(1) - pi) * problem.correction - xp.sum(data[W[i] == 0], axis=0)
    if problem.whitening == "post":
        if problem.criterion == "diagonal":
            imbalance = imbalance / xp.diag(problem.factor)
        elif problem.device == "cuda":
            from cupyx.scipy.linalg import solve_triangular
            imbalance = solve_triangular(problem.factor, imbalance.T,
                                         lower=True, check_finite=False).T
        else:
            imbalance = np.linalg.solve(problem.factor, imbalance.T).T
    result = dtype.type(score_scale(problem, n1)) * xp.sum(imbalance * imbalance, axis=1, dtype=dtype)
    if not bool(xp.all(xp.isfinite(result))):
        raise FloatingPointError("balance evaluation produced nonfinite scores")
    return result


def reference_scores(X, W, n1, criterion="full"):
    """Independent FP64 group-mean + covariance-solve implementation."""
    if criterion not in ("diagonal", "full"):
        raise ValueError("criterion must be 'diagonal' or 'full'")
    X = _matrix(X)
    n, p = X.shape
    n1 = _integer(n1, "n1", 1, n - 1)
    W = _assignments(to_numpy(W), n, n1, np)
    if criterion == "full":
        if p >= n - 1:
            raise ValueError("full Mahalanobis requires p < n-1")
        covariance = np.atleast_2d(np.cov(X, rowvar=False, ddof=1))
        if np.linalg.matrix_rank(X - X.mean(axis=0)) < p:
            raise ValueError("centered covariates must have full column rank")
    else:
        variance = np.var(X, axis=0, ddof=0)
        if np.any(variance <= 0) or not np.all(np.isfinite(variance)):
            raise ValueError("diagonal variances must be finite and positive")
    result = np.empty(W.shape[0], dtype=np.float64)
    for i, assignment in enumerate(W):
        difference = X[assignment == 1].mean(axis=0) - X[assignment == 0].mean(axis=0)
        if criterion == "full":
            quadratic = difference @ np.linalg.solve(covariance, difference)
        else:
            quadratic = np.sum(difference * difference / variance)
        result[i] = (n1 * (n - n1) / n) * quadratic
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("reference evaluation produced nonfinite scores")
    return result
