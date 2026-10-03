"""Small, lazy NumPy/CuPy interoperability helpers."""

import importlib
import numbers
import os

import numpy as np


def get_xp(device="cpu"):
    """Return NumPy or lazily import CuPy for an explicitly requested device."""
    if device == "cpu":
        return np
    if device != "cuda":
        raise ValueError("device must be 'cpu' or 'cuda'")
    if os.environ.get("CUPY_TF32", "0") != "0":
        raise ValueError("FlashRerand requires CUPY_TF32=0 before importing CuPy")
    try:
        return importlib.import_module("cupy")
    except ImportError as exc:
        raise ImportError("CUDA execution requires CuPy; install a build matching CUDA") from exc


def cuda_math_metadata():
    """Request pedantic cuBLAS arithmetic when exposed, and report actual state.

No claim about exact real arithmetic or cross-kernel bitwise equality is made.
CuPy's documented TF32 opt-in environment variable is rejected by get_xp().
"""
    xp = get_xp("cuda")
    info = {"cupy_tf32": os.environ.get("CUPY_TF32", "0"),
            "cublas_math_mode": None, "pedantic_requested": False}
    try:
        routines = importlib.import_module("cupy._core._routines_linalg")
        for dtype in (np.float32, np.float64):
            compute_type = routines.get_compute_type(dtype)
            name = str(routines.compute_type_to_str(compute_type))
            info["cupy_compute_type_" + np.dtype(dtype).name] = name
            if "TF32" in name or "BF16" in name or "FP16" in name:
                raise ValueError("CuPy has a cached reduced-precision compute mode: " + name)
    except (ImportError, AttributeError) as exc:
        info["compute_type_note"] = "CuPy compute-type inspection unavailable: " + type(exc).__name__
    try:
        cublas = importlib.import_module("cupy_backends.cuda.libs.cublas")
        handle = xp.cuda.device.get_cublas_handle()
        pedantic = getattr(cublas, "CUBLAS_PEDANTIC_MATH", None)
        if pedantic is not None:
            cublas.setMathMode(handle, pedantic)
            info["pedantic_requested"] = True
        info["cublas_math_mode"] = int(cublas.getMathMode(handle))
    except (ImportError, AttributeError) as exc:
        info["math_mode_note"] = "cuBLAS math-mode inspection unavailable: " + type(exc).__name__
    return info


def array_device(array):
    """Identify a NumPy/CuPy array without importing CuPy for CPU inputs."""
    return "cuda" if type(array).__module__.split(".")[0] == "cupy" else "cpu"


def synchronize(device="cpu"):
    xp = get_xp(device)
    if device == "cuda":
        xp.cuda.get_current_stream().synchronize()


def to_numpy(array):
    if array_device(array) == "cuda":
        return get_xp("cuda").asnumpy(array)
    return np.asarray(array)


def _integer(value, name, minimum=0, maximum=None):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, numbers.Integral):
        raise TypeError("%s must be an integer" % name)
    value = int(value)
    if value < minimum or (maximum is not None and value > maximum):
        raise ValueError("%s is out of its supported range" % name)
    return value


def _unsigned_array(value, name, maximum):
    result = to_numpy(value)
    if result.dtype.kind not in "iu":
        raise TypeError("%s must contain integers" % name)
    if result.size and (np.any(result < 0) or np.any(result > maximum)):
        raise ValueError("%s is out of its supported range" % name)
    return result.astype(np.uint64, copy=False)


def _candidate_ids(value):
    result = _unsigned_array(value, "candidate_ids", (1 << 64) - 1)
    if result.ndim != 1:
        raise ValueError("candidate_ids must be one-dimensional")
    return result
