"""Fixed-size uniform proposal samplers with addressable Philox streams."""

import numpy as np

from .arrays import (
    _candidate_ids, _integer, array_device, get_xp, to_numpy,
)
from .rng import philox_words

_MAX_ATTEMPT = (1 << 24) - 1
_MAX_DRAW = (1 << 32) - 1


def _randbelow(seed, ids, bound, draw, stage):
    """Reject the incomplete initial interval of 32-bit words, then reduce."""
    ids = np.asarray(ids, dtype=np.uint64)
    output = np.empty(ids.shape, dtype=np.uint64)
    pending = np.ones(ids.shape, dtype=bool)
    attempt = 0
    threshold = (1 << 32) % bound
    while np.any(pending):
        raw = philox_words(seed, ids[pending], draw, stage, attempt)[..., 0].astype(np.uint64)
        good = raw >= threshold
        positions = np.flatnonzero(pending)
        output.flat[positions[good]] = raw[good] % np.uint64(bound)
        pending.flat[positions[good]] = False
        if np.any(pending):
            if attempt == _MAX_ATTEMPT:
                raise RuntimeError("bounded-integer RNG exhausted its rejection-attempt address")
            attempt += 1
    return output


def _keys(n, n1, ids, seed):
    out = np.zeros((len(ids), n), dtype=np.uint8)
    remaining = np.arange(len(ids))
    subjects = np.arange(n, dtype=np.uint64)[None, :]
    attempt = 0
    while remaining.size:
        words = philox_words(seed, ids[remaining, None], subjects, 1, attempt)
        keys = words[..., 0].astype(np.uint64) | (words[..., 1].astype(np.uint64) << np.uint64(32))
        order = np.argsort(keys, axis=1, kind="stable")
        sorted_keys = np.take_along_axis(keys, order, axis=1)
        tied = np.any(sorted_keys[:, 1:] == sorted_keys[:, :-1], axis=1)
        good = ~tied
        out[remaining[good, None], order[good, :n1]] = 1
        remaining = remaining[tied]
        if remaining.size:
            if attempt == _MAX_ATTEMPT:
                raise RuntimeError("random-key sampler exhausted its collision-attempt address")
            attempt += 1
    return out


def _sequential(n, n1, ids, seed):
    out = np.zeros((len(ids), n), dtype=np.uint8)
    r = np.full(len(ids), n1, dtype=np.int64)
    for i in range(n):
        m = n - i
        forced = r == m
        out[forced, i] = 1
        active = (r > 0) & ~forced
        if np.any(active):
            chosen = _randbelow(seed, ids[active], m, i, 2) < r[active]
            out[active, i] = chosen.astype(np.uint8)
        r -= out[:, i]
    return out


def _repair(n, n1, ids, seed):
    groups = np.arange((n + 31) // 32, dtype=np.uint64)
    words = philox_words(seed, ids[:, None], groups[None, :], 3)[..., 0]
    positions = np.arange(n)
    out = ((words[:, positions // 32] >> (positions % 32).astype(np.uint32)) & np.uint32(1)).astype(np.uint8)
    for row, candidate_id in enumerate(ids):
        count = int(out[row].sum())
        trial = 0
        while count != n1:
            index = int(_randbelow(seed, np.array([candidate_id], dtype=np.uint64), n, trial, 4)[0])
            if count > n1 and out[row, index] == 1:
                out[row, index] = 0
                count -= 1
            elif count < n1 and out[row, index] == 0:
                out[row, index] = 1
                count += 1
            if count != n1 and trial == _MAX_DRAW:
                raise RuntimeError("repair sampler exhausted its trial address")
            trial += 1
    return out


def generate(n, n1, candidate_ids, seed, sampler="keys", device="cpu"):
    """Generate a (B,n) uint8 proposal matrix; repeated IDs replay identically."""
    n = _integer(n, "n", 2, _MAX_DRAW)
    n1 = _integer(n1, "n1", 1, n - 1)
    seed = _integer(seed, "seed", 0, (1 << 64) - 1)
    ids = _candidate_ids(candidate_ids)
    if sampler not in ("keys", "sequential", "repair"):
        raise ValueError("sampler must be 'keys', 'sequential', or 'repair'")
    get_xp(device)
    if device == "cuda":
        from .cuda_backend import generate_cuda
        return generate_cuda(n, n1, ids, seed, sampler)
    if ids.size == 0:
        return np.empty((0, n), dtype=np.uint8)
    return {"keys": _keys, "sequential": _sequential, "repair": _repair}[sampler](n, n1, ids, seed)


def pack(W, device=None):
    """Pack subjects into uint8 bytes, least-significant bit first; zero padding."""
    if device is None:
        device = array_device(W)
    xp = get_xp(device)
    W = xp.asarray(W)
    if W.ndim != 2 or W.shape[1] < 1:
        raise ValueError("W must be a two-dimensional matrix with at least one column")
    if W.dtype.kind not in "biuf" or not bool(xp.all((W == 0) | (W == 1))):
        raise ValueError("W must contain only binary assignments")
    W = W.astype(xp.uint8, copy=False)
    if device == "cuda":
        from .cuda_backend import pack_cuda
        return pack_cuda(W)
    return np.packbits(W, axis=1, bitorder="little")


def unpack(packed, n, device=None):
    """Unpack canonical little-endian bytes to a (B,n) uint8 matrix."""
    n = _integer(n, "n", 1, _MAX_DRAW)
    if device is None:
        device = array_device(packed)
    xp = get_xp(device)
    packed = xp.asarray(packed)
    if packed.ndim != 2 or packed.shape[1] != (n + 7) // 8:
        raise ValueError("packed shape must be (B, ceil(n/8))")
    if packed.dtype != xp.dtype("uint8"):
        raise TypeError("packed assignments must have dtype uint8")
    if n % 8 and bool(xp.any(packed[:, -1] >> (n % 8))):
        raise ValueError("unused padding bits must be zero")
    if device == "cuda":
        from .cuda_backend import unpack_cuda
        return unpack_cuda(packed, n)
    return np.unpackbits(packed, axis=1, count=n, bitorder="little")
