"""Addressable Philox4x32-10 random words, shared with the CUDA kernels.

Counter: (candidate low32, candidate high32, draw, stage<<24 | attempt).
Key: (seed low32, seed high32). No address field silently wraps.
"""

import numpy as np

from .arrays import _integer, _unsigned_array

MASK32 = (1 << 32) - 1
M0 = 0xD2511F53
M1 = 0xCD9E8D57
W0 = 0x9E3779B9
W1 = 0xBB67AE85
PHILOX_ZERO_VECTOR = (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8)


def philox_words(seed, candidate_ids, draw, stage, attempt=0):
    """Return uint32 words of shape broadcast(ids, draw, stage, attempt)+(4,)."""
    seed = _integer(seed, "seed", 0, (1 << 64) - 1)
    ids = _unsigned_array(candidate_ids, "candidate_ids", (1 << 64) - 1)
    draw = _unsigned_array(draw, "draw", MASK32)
    stage = _unsigned_array(stage, "stage", 255)
    attempt = _unsigned_array(attempt, "attempt", (1 << 24) - 1)
    ids, draw, stage, attempt = np.broadcast_arrays(ids, draw, stage, attempt)
    c0 = ids & np.uint64(MASK32)
    c1 = ids >> np.uint64(32)
    c2 = draw.copy()
    c3 = (stage << np.uint64(24)) | attempt
    k0, k1 = np.uint64(seed & MASK32), np.uint64(seed >> 32)
    mask = np.uint64(MASK32)
    for _ in range(10):
        p0 = np.uint64(M0) * c0
        p1 = np.uint64(M1) * c2
        c0, c1, c2, c3 = (
            (p1 >> np.uint64(32)) ^ c1 ^ k0,
            p1 & mask,
            (p0 >> np.uint64(32)) ^ c3 ^ k1,
            p0 & mask,
        )
        k0 = (k0 + np.uint64(W0)) & mask
        k1 = (k1 + np.uint64(W1)) & mask
    return np.stack((c0, c1, c2, c3), axis=-1).astype(np.uint32)


def philox_scalar(seed, candidate_id, draw, stage, attempt=0):
    return tuple(int(word) for word in philox_words(seed, candidate_id, draw, stage, attempt))


def known_zero_vector():
    """Return the implementation's output for the standard all-zero test vector."""
    return philox_scalar(0, 0, 0, 0)
