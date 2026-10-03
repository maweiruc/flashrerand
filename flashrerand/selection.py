"""Exact global Top-K selection ordered by (score, candidate ID)."""

import numpy as np

from .arrays import _integer, array_device, get_xp


def _inputs(scores, ids, device=None):
    if device is None:
        device = array_device(scores)
    xp = get_xp(device)
    scores, ids = xp.asarray(scores), xp.asarray(ids)
    if scores.ndim != 1 or ids.ndim != 1 or scores.shape != ids.shape:
        raise ValueError("scores and ids must be one-dimensional arrays of equal length")
    if scores.dtype.kind not in "fiu" or scores.dtype.kind == "b":
        raise TypeError("scores must be real numeric values")
    if ids.dtype.kind not in "iu":
        raise TypeError("ids must contain integers")
    if bool(xp.any(ids < 0)):
        raise ValueError("ids must be nonnegative")
    if not bool(xp.all(xp.isfinite(scores))):
        raise ValueError("scores must be finite; NaN and Inf are not valid candidates")
    return xp, scores, ids.astype(xp.uint64, copy=False)


def topk(scores, ids, k):
    """Return (scores, IDs) for the smallest k, with exact ties ordered by ID."""
    xp, scores, ids = _inputs(scores, ids)
    k = _integer(k, "k", 0, scores.size)
    order = _order(xp, scores, ids)[:k]
    return scores[order], ids[order]


def _order(xp, scores, ids):
    if xp is np:
        return np.lexsort((ids, scores))
    # CuPy lexsort uses a single homogeneous array. Stacking floating scores
    # with uint64 IDs would lose ID bits. Stable passes preserve both dtypes.
    by_id = xp.argsort(ids, kind="stable")
    return by_id[xp.argsort(scores[by_id], kind="stable")]


class StreamingTopK:
    """Keep the exact global Top-K of all candidates supplied to update()."""

    def __init__(self, k, device="cpu"):
        self.k = _integer(k, "k")
        self.device = device
        self._xp = get_xp(device)
        self._scores = None
        self._ids = self._xp.empty(0, dtype=self._xp.uint64)
        self.total = 0

    def update(self, scores, ids):
        xp, scores, ids = _inputs(scores, ids, self.device)
        if self._scores is not None and scores.dtype != self._scores.dtype:
            raise TypeError("streaming score dtype must stay constant")
        self.total += int(scores.size)
        if self._scores is not None:
            scores = xp.concatenate((self._scores, scores))
            ids = xp.concatenate((self._ids, ids))
        order = _order(xp, scores, ids)[:min(self.k, scores.size)]
        self._scores, self._ids = scores[order], ids[order]
        return self

    def result(self):
        if self._scores is None:
            return self._xp.empty(0, dtype=self._xp.float64), self._ids.copy()
        return self._scores.copy(), self._ids.copy()
