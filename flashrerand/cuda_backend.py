"""Internal CuPy/CUDA implementations, imported lazily.

These kernels require an NVIDIA CUDA device and NVRTC. Additional internal
kernel variants remain bundled because RawModule resolves their symbols.
Only the fixed public rerandomize configuration is supported in V0.
"""

from functools import lru_cache
from pathlib import Path
import operator

import numpy as np


def _cupy():
    try:
        import cupy as cp
    except ImportError as exc:
        raise RuntimeError("CUDA requires CuPy and a compatible NVIDIA CUDA installation") from exc
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            raise RuntimeError("No NVIDIA CUDA device is available")
    except Exception as exc:
        raise RuntimeError("CUDA is unavailable; a working NVIDIA driver and device are required") from exc
    return cp


@lru_cache(maxsize=None)
def _module(device_id):
    cp = _cupy()
    source = Path(__file__).with_name("kernels.cu").read_text()
    names = []
    for dtype in ("float", "double"):
        names += [f"score_reduce<{dtype}>", f"score_tiled<{dtype},false>",
                  f"score_tiled<{dtype},true>"]
        names += [f"score_fused_sequential<{dtype},{p}>" for p in (32, 128, 512, 1024)]
    with cp.cuda.Device(device_id):
        return cp.RawModule(code=source, options=("--std=c++11", "--fmad=false"),
                            name_expressions=names)


def _kernel(name):
    cp = _cupy()
    return _module(cp.cuda.runtime.getDevice()).get_function(name)


def _integer(value, name, low, high):
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if isinstance(value, (bool, np.bool_)) or not low <= result <= high:
        raise ValueError(f"{name} must be in [{low}, {high}]")
    return result


def _shape(n, n1):
    # Rounded subject tiles and signed CUDA indices must remain representable.
    n = _integer(n, "n", 2, 2**31 - 32)
    n1 = _integer(n1, "n1", 1, n - 1)
    return n, n1


def _ids(candidate_ids, cp):
    ids = cp.asarray(candidate_ids)
    if ids.ndim != 1 or ids.dtype.kind not in "iu":
        raise ValueError("candidate_ids must be a one-dimensional integer array")
    if ids.dtype.kind == "i" and bool(cp.any(ids < 0).item()):
        raise ValueError("candidate_ids must be nonnegative uint64 values")
    if ids.size > 2**31 - 1:
        raise ValueError("CUDA supports fewer than 2**31 candidates per batch")
    return cp.ascontiguousarray(ids, dtype=cp.uint64)


def _launch_1d(kernel, size, args):
    if size:
        kernel((min((int(size) + 255) // 256, 65535),), (256,), args)


def _check_rng(error):
    code = int(error.item())
    if code:
        raise RuntimeError(f"RNG replay address space exhausted (CUDA error {code})")


def generate_cuda(n, n1, candidate_ids, seed, sampler="keys", *, _error_out=None):
    """Generate fixed-count proposals as a C-contiguous uint8 (B, n) array.

    RNG: Philox4x32-10; counter=(ID.lo, ID.hi, draw, stage<<24|attempt).
    Keys join word0 as the low 32 bits and word1 as the high 32 bits.
    Repair starts from fair bits; it is primarily intended for near-1:1 ratios.
    """
    n, n1 = _shape(n, n1)
    seed = _integer(seed, "seed", 0, 2**64 - 1)
    if sampler not in ("keys", "sequential", "repair"):
        raise ValueError("sampler must be keys, sequential, or repair")
    cp = _cupy()
    if _error_out is not None:
        if sampler != "repair" or _error_out.shape != (1,) or _error_out.dtype != cp.uint32:
            raise ValueError("deferred RNG errors require repair and a uint32[1] device flag")
    ids = _ids(candidate_ids, cp)
    count = int(ids.size)
    w = cp.zeros((count, n), dtype=cp.uint8)
    if not count:
        return w
    scalar_args = (ids, np.uint64(seed), np.int32(n))
    if sampler == "keys":
        keys = cp.empty((count, n), dtype=cp.uint64)
        attempts = cp.zeros(count, dtype=cp.uint32)
        active = cp.ones(count, dtype=cp.uint8)
        while True:
            _launch_1d(_kernel("generate_keys"), count * n,
                       scalar_args + (np.int32(count), attempts, active, keys))
            order = cp.argsort(keys, axis=1)
            ordered = cp.take_along_axis(keys, order, axis=1)
            collisions = cp.any(ordered[:, 1:] == ordered[:, :-1], axis=1)
            if not bool(cp.any(collisions).item()):
                break
            if bool(cp.any(attempts[collisions] == (2**24 - 1)).item()):
                raise RuntimeError("keys sampler exhausted the retry address space")
            attempts[collisions] += cp.uint32(1)
            active = collisions.astype(cp.uint8)
        rows = cp.arange(count)[:, None]
        w[rows, order[:, :n1]] = 1
    elif sampler == "sequential":
        error = cp.zeros(1, dtype=cp.uint32)
        _kernel("generate_sequential")(((count + 127) // 128,), (128,),
            scalar_args + (np.int32(n1), np.int32(count), w, error))
        _check_rng(error)
    else:
        _launch_1d(_kernel("generate_fair_bits"), count * ((n + 31) // 32) * 32,
                   scalar_args + (np.int32(count), w))
        counts = w.sum(axis=1, dtype=cp.int64)
        error = cp.zeros(1, dtype=cp.uint32) if _error_out is None else _error_out
        _kernel("repair_counts")(((count + 127) // 128,), (128,),
            scalar_args + (np.int32(n1), np.int32(count), counts, w, error))
        if _error_out is None:
            _check_rng(error)
    return w


def _dense(w, cp):
    w = cp.asarray(w)
    if w.ndim != 2 or w.dtype != cp.uint8:
        raise ValueError("dense assignments must be a two-dimensional uint8 array")
    if w.shape[1] < 1 or w.shape[1] > 2**31 - 32 or w.shape[0] > 2**31 - 1:
        raise ValueError("assignment shape is outside the supported CUDA range")
    if bool(cp.any(w > 1).item()):
        raise ValueError("assignments must contain only 0 and 1")
    return cp.ascontiguousarray(w)


def pack_cuda(w):
    """Pack each row, low subject index in the least-significant bit."""
    cp = _cupy()
    w = _dense(w, cp)
    count, n = w.shape
    packed = cp.empty((count, (n + 7) // 8), dtype=cp.uint8)
    _launch_1d(_kernel("pack_rows"), packed.size,
               (w, packed, np.int32(n), np.int32(count)))
    return packed


def _packed(packed, n, cp):
    packed = cp.asarray(packed)
    if (packed.ndim != 2 or packed.dtype != cp.uint8
            or packed.shape[1] != (n + 7) // 8 or packed.shape[0] > 2**31 - 1):
        raise ValueError("packed assignments must be uint8 with shape (B, ceil(n/8))")
    if n % 8 and bool(cp.any(packed[:, -1] >> (n % 8)).item()):
        raise ValueError("unused packed tail bits must be zero")
    return cp.ascontiguousarray(packed)


def unpack_cuda(packed, n):
    cp = _cupy()
    n = _integer(n, "n", 1, 2**31 - 32)
    packed = _packed(packed, n, cp)
    count = packed.shape[0]
    w = cp.empty((count, n), dtype=cp.uint8)
    _launch_1d(_kernel("unpack_rows"), w.size,
               (packed, w, np.int32(n), np.int32(count)))
    return w


def _problem(problem, n1, cp):
    if problem.device != "cuda":
        raise ValueError("CUDA scoring requires a problem prepared on device='cuda'")
    if problem.whitening != "pre":
        raise ValueError("custom CUDA scoring supports only whitening='pre'")
    n, n1 = _shape(problem.n, n1)
    p = _integer(problem.p, "p", 1, 2**31 - 32)
    if problem.precision not in ("fp32", "fp64"):
        raise ValueError("precision must be fp32 or fp64")
    dtype = cp.dtype("float32" if problem.precision == "fp32" else "float64")
    z, correction = cp.asarray(problem.z), cp.asarray(problem.correction)
    if z.shape != (n, p) or correction.shape != (p,):
        raise ValueError("prepared Z/correction has the wrong shape")
    if z.dtype != dtype or correction.dtype != dtype:
        raise ValueError("prepared arrays do not match the declared precision")
    # Finiteness/rank are prepare() invariants; rescanning the whole Z at every
    # candidate batch would obscure the kernels' data movement.
    return n, p, n1, dtype, cp.ascontiguousarray(z), cp.ascontiguousarray(correction)


def score_cuda(problem, w, n1, backend="reduce", representation="dense"):
    """Score fixed assignments; full and diagonal normalization is in Z.

    All custom paths accumulate an entire coordinate before squaring it.
    'packed' uses the SAME tiled algorithm as 'tiled', changing W decoding only.
    """
    if backend not in ("reduce", "tiled", "packed"):
        raise ValueError("backend must be reduce, tiled, or packed")
    if representation not in ("dense", "packed"):
        raise ValueError("representation must be dense or packed")
    if (backend == "packed") != (representation == "packed"):
        raise ValueError("backend='packed' requires representation='packed'; other backends require dense")
    cp = _cupy()
    n, p, n1, dtype, z, correction = _problem(problem, n1, cp)
    w = _packed(w, n, cp) if representation == "packed" else _dense(w, cp)
    if representation == "dense" and w.shape[1] != n:
        raise ValueError("assignments do not match problem.n")
    # Include validation cost in public end-to-end timings. For packed inputs
    # count with a byte lookup, without materializing dense W.
    if representation == "packed":
        table = cp.asarray([bin(i).count("1") for i in range(256)], dtype=cp.uint8)
        counts = table[w].sum(axis=1, dtype=cp.int64)
    else:
        counts = w.sum(axis=1, dtype=cp.int64)
    if bool(cp.any(counts != n1).item()):
        raise ValueError("each assignment must have exactly n1 treated subjects")
    count = int(w.shape[0])
    output = cp.empty(count, dtype=dtype)
    if count:
        cpp = "float" if dtype == cp.dtype("float32") else "double"
        name = (f"score_reduce<{cpp}>" if backend == "reduce"
                else f"score_tiled<{cpp},{'true' if backend == 'packed' else 'false'}>")
        scale = dtype.type(n / (n1 * (n - n1)))
        _kernel(name)(((count + 3) // 4,), (128,),
            (w, z, correction, np.int32(n), np.int32(p), np.int32(n1),
             np.int32(count), scale, output))
        if not bool(cp.all(cp.isfinite(output)).item()):
            raise FloatingPointError("balance evaluation produced nonfinite scores")
    return output


def fused_sequential_cuda(problem, n1, candidate_ids, seed):
    """Generate the sequential sampler's proposals and score without storing W.

    This initial fused kernel supports p <= 1024; it is an experimental path,
    not an automatically selected or performance-validated default.
    """
    seed = _integer(seed, "seed", 0, 2**64 - 1)
    cp = _cupy()
    n, p, n1, dtype, z, correction = _problem(problem, n1, cp)
    if p > 1024:
        raise ValueError("fused sequential scoring currently supports p <= 1024")
    ids = _ids(candidate_ids, cp)
    count = int(ids.size)
    output = cp.empty(count, dtype=dtype)
    if count:
        cpp = "float" if dtype == cp.dtype("float32") else "double"
        max_p = next(limit for limit in (32, 128, 512, 1024) if p <= limit)
        scale = dtype.type(n / (n1 * (n - n1)))
        error = cp.zeros(1, dtype=cp.uint32)
        _kernel(f"score_fused_sequential<{cpp},{max_p}>")(
            ((count + 3) // 4,), (128,),
            (ids, np.uint64(seed), z, correction, np.int32(n), np.int32(p),
             np.int32(n1), np.int32(count), scale, output, error))
        _check_rng(error)
        if not bool(cp.all(cp.isfinite(output)).item()):
            raise FloatingPointError("balance evaluation produced nonfinite scores")
    return output
