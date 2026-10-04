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


@lru_cache(maxsize=None)
def _tiled_module(device_id):
    """Keep PK0 compilation separate from the unchanged default module."""
    cp = _cupy()
    names = [f"score_tiled_variant<double,{packed},{candidates},{subjects}>"
             for packed in ("false", "true") for candidates in (4, 8)
             for subjects in (32, 64)]
    with cp.cuda.Device(device_id):
        return cp.RawModule(code=Path(__file__).with_name("kernels.cu").read_text(),
                            options=("--std=c++11", "--fmad=false"),
                            name_expressions=names)


@lru_cache(maxsize=None)
def _packed_repair_module(device_id):
    """Compile the opt-in PK1 sampler without warming the default module."""
    cp = _cupy()
    with cp.cuda.Device(device_id):
        return cp.RawModule(code=Path(__file__).with_name("kernels.cu").read_text(),
                            options=("--std=c++11", "--fmad=false", "-DFLASHRERAND_PK1"))


@lru_cache(maxsize=None)
def _repair_fused_module(device_id, limits=None):
    """Isolate the production prototype from test-only exhaustion builds."""
    cp = _cupy()
    options = ["--std=c++11", "--fmad=false", "-DFLASHRERAND_REPAIR_FUSED"]
    if limits is not None:
        attempt_max, trial_limit, force_reject = limits
        options += [f"-DFLASHRERAND_FUSED_ATTEMPT_MAX={attempt_max}u",
                    f"-DFLASHRERAND_FUSED_TRIAL_LIMIT={trial_limit}ull",
                    f"-DFLASHRERAND_FUSED_FORCE_REJECT={int(force_reject)}"]
    with cp.cuda.Device(device_id):
        return cp.RawModule(code=Path(__file__).with_name("kernels.cu").read_text(),
                            options=tuple(options))


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


def generate_cuda(n, n1, candidate_ids, seed, sampler="keys", *, _error_out=None,
                  _empty_output=False, _profile=None):
    """Generate fixed-count proposals as a C-contiguous uint8 (B, n) array.

    RNG: Philox4x32-10; counter=(ID.lo, ID.hi, draw, stage<<24|attempt).
    Keys join word0 as the low 32 bits and word1 as the high 32 bits.
    Repair starts from fair bits; it is primarily intended for near-1:1 ratios.
    Internal G1 opts into uninitialized output only for kernels that overwrite
    every valid entry. Keys always needs a zero-filled assignment buffer.
    A diagnostic callback _profile(name, fn) must execute fn and return its
    result. No additional synchronization is performed when it is None.
    Repair and sequential may supply a private sticky _error_out flag; the
    caller then owns the final RNG check. Keys does not support deferral.
    """
    n, n1 = _shape(n, n1)
    seed = _integer(seed, "seed", 0, 2**64 - 1)
    if sampler not in ("keys", "sequential", "repair"):
        raise ValueError("sampler must be keys, sequential, or repair")
    if not isinstance(_empty_output, (bool, np.bool_)):
        raise ValueError("_empty_output must be a boolean")
    if _profile is not None and not callable(_profile):
        raise TypeError("_profile must be callable or None")
    cp = _cupy()
    if _error_out is not None:
        if (sampler not in ("repair", "sequential") or not isinstance(_error_out, cp.ndarray)
                or _error_out.shape != (1,) or _error_out.dtype != cp.uint32):
            raise ValueError("deferred RNG errors require repair/sequential and a uint32[1] device flag")
    ids = _ids(candidate_ids, cp)
    count = int(ids.size)
    allocate = cp.empty if _empty_output and sampler != "keys" else cp.zeros
    w = allocate((count, n), dtype=cp.uint8)
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
        error = cp.zeros(1, dtype=cp.uint32) if _error_out is None else _error_out
        args = scalar_args + (np.int32(n1), np.int32(count), w, error)
        if _profile is None:
            _kernel("generate_sequential")(((count + 127) // 128,), (128,), args)
        else:
            _profile("sequential", lambda: _kernel("generate_sequential")(
                ((count + 127) // 128,), (128,), args))
        if _error_out is None:
            _check_rng(error)
    else:
        bits_args = scalar_args + (np.int32(count), w)
        bits_size = count * ((n + 31) // 32) * 32
        if _profile is None:
            _launch_1d(_kernel("generate_fair_bits"), bits_size, bits_args)
            counts = w.sum(axis=1, dtype=cp.int64)
        else:
            _profile("fair_bits", lambda: _launch_1d(
                _kernel("generate_fair_bits"), bits_size, bits_args))
            counts = _profile("count", lambda: w.sum(axis=1, dtype=cp.int64))
        error = cp.zeros(1, dtype=cp.uint32) if _error_out is None else _error_out
        args = scalar_args + (np.int32(n1), np.int32(count), counts, w, error)
        if _profile is None:
            _kernel("repair_counts")(((count + 127) // 128,), (128,), args)
        else:
            _profile("repair", lambda: _kernel("repair_counts")(
                ((count + 127) // 128,), (128,), args))
        if _error_out is None:
            _check_rng(error)
    return w


def _workspace_buffer(value, shape, dtype, cp, name):
    """Validate internal output storage without reading device values."""
    if (not isinstance(value, cp.ndarray) or value.shape != shape
            or value.dtype != dtype or not value.flags.c_contiguous
            or value.device.id != cp.cuda.runtime.getDevice()):
        raise ValueError(f"{name} must be a contiguous current-device {dtype} buffer of shape {shape}")
    return value


def _workspace_no_overlap(output, inputs):
    """Reject writable aliasing using pointer metadata, never a device read."""
    if not output.nbytes:
        return
    begin, end = output.data.ptr, output.data.ptr + output.nbytes
    for value in inputs:
        if value is not None and value.nbytes and (
                begin < value.data.ptr + value.nbytes and value.data.ptr < end):
            raise ValueError("workspace output must not overlap another input or scratch buffer")


def generate_packed_repair_cuda(n, n1, candidate_ids, seed, *, _error_out=None,
                                _profile=None, _out=None, _counts_out=None):
    """Private PK1 repair sampler, with no dense assignment intermediate.

    The byte layout, Philox addresses and serial flip order are identical to
    packing generate_cuda(..., sampler='repair'). Every output byte, including
    zero padding, is written before use. A supplied sticky error is never reset.
    Private workspace callers may provide both output and count scratch. They
    own their lifetime and must finish consuming a batch before reusing it.
    """
    n, n1 = _shape(n, n1)
    seed = _integer(seed, "seed", 0, 2**64 - 1)
    if _profile is not None and not callable(_profile):
        raise TypeError("_profile must be callable or None")
    cp = _cupy()
    if _error_out is not None and (not isinstance(_error_out, cp.ndarray) or
            _error_out.shape != (1,) or _error_out.dtype != cp.uint32):
        raise ValueError("deferred RNG errors require a uint32[1] device flag")
    ids = _ids(candidate_ids, cp)
    count = int(ids.size)
    if (_out is None) != (_counts_out is None):
        raise ValueError("packed workspace requires both _out and _counts_out")
    if _out is None:
        packed = cp.empty((count, (n + 7) // 8), dtype=cp.uint8)
        counts = None
    else:
        packed = _workspace_buffer(_out, (count, (n + 7) // 8), cp.uint8, cp, "_out")
        counts = _workspace_buffer(_counts_out, (count,), cp.int64, cp, "_counts_out")
        _workspace_no_overlap(packed, (counts, ids, _error_out))
        _workspace_no_overlap(counts, (ids, _error_out))
    if not count:
        return packed
    module = _packed_repair_module(cp.cuda.runtime.getDevice())
    if counts is None:
        counts = cp.empty(count, dtype=cp.int64)
    error = cp.zeros(1, dtype=cp.uint32) if _error_out is None else _error_out
    scalar = (ids, np.uint64(seed), np.int32(n))

    def measured(name, call):
        return call() if _profile is None else _profile(name, call)

    measured("fair_bits", lambda: _launch_1d(
        module.get_function("generate_packed_fair_bits"), count * ((n + 31) // 32),
        scalar + (np.int32(count), packed)))
    measured("count", lambda: _launch_1d(module.get_function("count_packed_rows"), count,
        (packed, np.int32(n), np.int32(count), counts)))
    measured("repair", lambda: _launch_1d(module.get_function("repair_packed_counts"), count,
        scalar + (np.int32(n1), np.int32(count), counts, packed, error)))
    if _error_out is None:
        _check_rng(error)
    return packed


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


def score_fixed_w_cuda(problem, w, n1, *, representation="dense",
                       candidate_tile=4, subject_tile=32, _out=None):
    """Trusted FP64/pre fixed-W PK0 experiment, not a public scoring API.

    The caller must validate binary entries, exact treatment counts, packed
    tail bits, and input/output finiteness OUTSIDE the timed region. Only array
    metadata is checked here; no device scalar is read and no error flag is
    accumulated. Deferred E2E callers must own sticky RNG validation and the
    full-pool finite-score check, as engine.sample_topk does.
    Dense and packed variants use identical tiles and accumulation order.
    _out is private search-owned storage; its metadata is checked, and the
    caller must keep score regions disjoint across batches until selection.
    """
    candidate_tile = _integer(candidate_tile, "candidate_tile", 4, 8)
    subject_tile = _integer(subject_tile, "subject_tile", 32, 64)
    if candidate_tile not in (4, 8) or subject_tile not in (32, 64):
        raise ValueError("PK0 tiles must use 4/8 candidates and 32/64 subjects")
    if representation not in ("dense", "packed"):
        raise ValueError("representation must be dense or packed")
    if problem.precision != "fp64":
        raise ValueError("PK0 fixed-W experiments require precision='fp64'")
    cp = _cupy()
    n, p, n1, dtype, z, correction = _problem(problem, n1, cp)
    width = n if representation == "dense" else (n + 7) // 8
    if (not isinstance(w, cp.ndarray) or w.ndim != 2 or w.dtype != cp.uint8
            or w.shape[1] != width or w.shape[0] > 2**31 - 1
            or not w.flags.c_contiguous):
        raise ValueError("trusted W must be a C-contiguous CUDA uint8 array with the expected shape")
    count = int(w.shape[0])
    output = (cp.empty(count, dtype=dtype) if _out is None else
              _workspace_buffer(_out, (count,), dtype, cp, "_out"))
    if _out is not None:
        _workspace_no_overlap(output, (w, z, correction))
    if count:
        packed = "true" if representation == "packed" else "false"
        name = f"score_tiled_variant<double,{packed},{candidate_tile},{subject_tile}>"
        kernel = _tiled_module(cp.cuda.runtime.getDevice()).get_function(name)
        kernel(((count + candidate_tile - 1) // candidate_tile,),
               (candidate_tile * 32,),
               (w, z, correction, np.int32(n), np.int32(p), np.int32(n1),
                np.int32(count), dtype.type(n / (n1 * (n - n1))), output))
    return output


def fused_repair_cuda(problem, n1, candidate_ids, seed, *, _error_out=None,
                      _debug_packed_out=None, _test_limits=None):
    """Experimental repair-first scoring with shared packed membership only.

    Production stores no batch assignment matrix in global memory. An optional
    private packed output supports exact full-pool candidate tests; test limits
    compile a separate module to exercise finite RNG exhaustion and barriers.
    A supplied error flag is sticky. Its caller must check both the flag and
    ALL output scores before selection, as sample_topk's deferred path does.
    """
    if (problem.device != "cuda" or problem.criterion != "full" or
            problem.precision != "fp64" or problem.whitening != "pre"):
        raise ValueError("repair fused requires CUDA/full/FP64/pre")
    if problem.n > 1024 or problem.p > 32:
        raise ValueError("repair fused currently supports n <= 1024 and p <= 32")
    seed = _integer(seed, "seed", 0, 2**64 - 1)
    if _test_limits is not None:
        if not isinstance(_test_limits, tuple) or len(_test_limits) != 3:
            raise ValueError("_test_limits must be (attempt_max, trial_limit, force_reject)")
        attempt = _integer(_test_limits[0], "attempt_max", 0, 2**24 - 1)
        trial = _integer(_test_limits[1], "trial_limit", 0, 2**32)
        if not isinstance(_test_limits[2], (bool, np.bool_)):
            raise ValueError("force_reject must be boolean")
        _test_limits = (attempt, trial, bool(_test_limits[2]))
    cp = _cupy()
    # A cross-device input must not be read through an invalid device pointer.
    device_id = cp.cuda.runtime.getDevice()
    for value in (problem.z, problem.correction, candidate_ids):
        if isinstance(value, cp.ndarray) and value.device.id != device_id:
            raise ValueError("repair fused inputs must belong to the current CUDA device")
    if _error_out is not None:
        _workspace_buffer(_error_out, (1,), cp.uint32, cp, "_error_out")
    n, p, n1, dtype, z, correction = _problem(problem, n1, cp)
    ids = _ids(candidate_ids, cp)
    count = int(ids.size)
    if _debug_packed_out is not None:
        _workspace_buffer(_debug_packed_out, (count, (n + 7) // 8), cp.uint8,
                          cp, "_debug_packed_out")
        _workspace_no_overlap(_debug_packed_out, (z, correction, ids, _error_out))
    if _error_out is not None:
        _workspace_no_overlap(_error_out, (z, correction, ids, _debug_packed_out))
    output = cp.empty(count, dtype=dtype)
    error = cp.zeros(1, dtype=cp.uint32) if _error_out is None else _error_out
    if count:
        kernel = _repair_fused_module(device_id, _test_limits).get_function("score_fused_repair")
        kernel(((count + 7) // 8,), (256,),
               (ids, np.uint64(seed), z, correction, np.int32(n), np.int32(p),
                np.int32(n1), np.int32(count), dtype.type(n / (n1 * (n - n1))),
                output, error, np.uint64(0) if _debug_packed_out is None else _debug_packed_out))
    if _error_out is None:
        _check_rng(error)
        if not bool(cp.all(cp.isfinite(output)).item()):
            raise FloatingPointError("balance evaluation produced nonfinite scores")
    return output


def fused_sequential_cuda(problem, n1, candidate_ids, seed, *, _error_out=None):
    """Generate the sequential sampler's proposals and score without storing W.

    This initial fused kernel supports p <= 1024; it is an experimental path,
    not an automatically selected or performance-validated default.
    A supplied private uint32[1] error flag is sticky and suppresses host RNG
    and score-finiteness reads. Its engine caller must check the flag and ALL
    candidate scores before selection/output. Ordinary calls keep both checks.
    """
    seed = _integer(seed, "seed", 0, 2**64 - 1)
    cp = _cupy()
    if _error_out is not None and (not isinstance(_error_out, cp.ndarray)
            or _error_out.shape != (1,) or _error_out.dtype != cp.uint32):
        raise ValueError("deferred RNG errors require a uint32[1] device flag")
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
        error = cp.zeros(1, dtype=cp.uint32) if _error_out is None else _error_out
        _kernel(f"score_fused_sequential<{cpp},{max_p}>")(
            ((count + 3) // 4,), (128,),
            (ids, np.uint64(seed), z, correction, np.int32(n), np.int32(p),
             np.int32(n1), np.int32(count), scale, output, error))
        if _error_out is None:
            _check_rng(error)
            if not bool(cp.all(cp.isfinite(output)).item()):
                raise FloatingPointError("balance evaluation produced nonfinite scores")
    return output
