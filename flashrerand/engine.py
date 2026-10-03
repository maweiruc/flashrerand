"""Bounded-batch orchestration; all selection is over the full candidate pool."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import numbers
import time
from typing import Any

import numpy as np

from .arrays import get_xp, synchronize, to_numpy
from .core import PreparedProblem, score
from .samplers import generate, pack
from .selection import StreamingTopK, topk

REPLAY_VERSION = "philox4x32-10-candidate-v1"


def _integer(name, value, minimum, maximum):
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise ValueError(f"{name} must be an integer")
    value = int(value)
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
    return value


def array_hash(array):
    a = np.ascontiguousarray(to_numpy(array))
    h = hashlib.sha256()
    h.update(str(a.shape).encode())
    h.update(a.dtype.str.encode())
    h.update(a.tobytes())
    return h.hexdigest()


@dataclass
class SampleResult:
    scores: np.ndarray
    candidate_ids: np.ndarray
    assignments: np.ndarray | None
    metadata: dict[str, Any]
    timings: dict[str, float]

    def replay_record(self):
        """JSON-safe record; does not contain the original covariate data."""
        return {**self.metadata, "candidate_ids": [int(i) for i in self.candidate_ids],
                "scores": self.scores.tolist()}


def replay(record, *, device="cpu"):
    """Recover assignments. Scores require the original X and numeric backend."""
    if record.get("replay_version") != REPLAY_VERSION:
        raise ValueError("Unsupported or missing replay_version")
    ids = record.get("candidate_ids")
    if ids is None:
        raise ValueError("Replay record must contain candidate_ids")
    # Validate before casting: a negative Python integer must not wrap to uint64.
    ids = [_integer("candidate_id", i, 0, 2**64 - 1) for i in ids]
    return to_numpy(generate(record["n"], record["n1"], np.asarray(ids, dtype=np.uint64),
                             record["seed"], sampler=record["sampler"], device=device))


def sample_topk(problem: PreparedProblem, n1: int, candidates: int, k: int, *,
                seed: int = 0, sampler: str = "keys", balance: str = "gemm",
                selection: str = "stream", batch_size: int = 256,
                output: str = "assignments", start_id: int = 0,
                profile: bool = False, validation: str = "checked") -> SampleResult:
    """Return the global smallest (score, ID) pairs; never deduplicates W.

    ``profile=True`` synchronizes each component. Use it for a separate
    diagnostic pass, not for an uninstrumented end-to-end speedup claim.
    Returned arrays live on the host. Complete W reconstruction is timed.
    """
    n1 = _integer("n1", n1, 1, problem.n - 1)
    candidates = _integer("candidates", candidates, 1, 2**63 - 1)
    k = _integer("k", k, 1, candidates)
    batch_size = _integer("batch_size", batch_size, 1, 2**31 - 1)
    seed = _integer("seed", seed, 0, 2**64 - 1)
    start_id = _integer("start_id", start_id, 0, 2**64 - 1)
    if start_id + candidates > 2**64:
        raise ValueError("candidate ID range overflows uint64")
    if selection not in ("all", "stream"):
        raise ValueError("selection must be 'all' or 'stream'")
    if output not in ("assignments", "records"):
        raise ValueError("output must be 'assignments' or 'records'")
    if balance not in ("gemm", "reduce", "tiled", "packed", "gather", "fused"):
        raise ValueError("Unknown balance backend")
    if sampler not in ("keys", "sequential", "repair"):
        raise ValueError("Unknown sampler")
    if validation not in ("checked", "deferred"):
        raise ValueError("validation must be checked or deferred")
    if validation == "deferred" and (problem.device != "cuda" or sampler != "repair" or
                                      balance != "gemm" or selection != "all"):
        raise ValueError("deferred validation currently requires CUDA repair + GEMM + all selection")
    if balance == "fused" and (problem.device != "cuda" or sampler != "sequential"):
        raise ValueError("fused requires device='cuda' and sampler='sequential'")
    if problem.device == "cpu" and balance in ("tiled", "packed"):
        raise ValueError(f"balance='{balance}' requires device='cuda'")
    if problem.device == "cuda" and balance == "gather":
        raise ValueError("gather is currently a CPU reference backend only")
    if problem.device == "cuda" and balance != "gemm" and problem.whitening != "pre":
        raise ValueError("custom CUDA scoring requires whitening='pre'")
    xp = get_xp(problem.device)
    timings = {}

    def measured(name, call):
        if not profile:
            return call()
        synchronize(problem.device)
        begin = time.perf_counter()
        result = call()
        synchronize(problem.device)
        timings[name] = timings.get(name, 0.0) + time.perf_counter() - begin
        return result

    synchronize(problem.device)
    begin = time.perf_counter()
    rng_error = None
    if validation == "deferred":
        from .arrays import cuda_math_metadata
        from .cuda_backend import generate_cuda, _check_rng
        from .generated_gemm import score_generated
        problem.metadata["cuda_math"] = cuda_math_metadata()
        rng_error = xp.zeros(1, dtype=xp.uint32)
    stream = StreamingTopK(k, device=problem.device) if selection == "stream" else None
    all_scores, all_ids = [], []
    for offset in range(0, candidates, batch_size):
        size = min(batch_size, candidates - offset)
        # Construct offsets first; arange with stop=2**64 is not portable.
        ids = np.arange(size, dtype=np.uint64) + np.uint64(start_id + offset)
        device_ids = xp.asarray(ids)
        if balance == "fused":
            from .cuda_backend import fused_sequential_cuda
            values = measured("sample_and_balance", lambda: fused_sequential_cuda(
                problem, n1, device_ids, seed))
        else:
            if validation == "deferred":
                w = measured("sample", lambda: generate_cuda(problem.n, n1, device_ids, seed,
                    sampler="repair", _error_out=rng_error))
            else:
                w = measured("sample", lambda: generate(problem.n, n1, ids, seed,
                                                         sampler=sampler, device=problem.device))
            representation = "dense"
            if balance == "packed":
                w = measured("representation", lambda: pack(w, device=problem.device))
                representation = "packed"
            if validation == "deferred":
                values = measured("balance", lambda: score_generated(problem, w, n1))
            else:
                values = measured("balance", lambda: score(problem, w, n1,
                    backend=balance, representation=representation))
            del w
        if stream is not None:
            measured("selection", lambda: stream.update(values, device_ids))
        else:
            all_scores.append(values)
            all_ids.append(device_ids)
    if rng_error is not None:
        measured("rng_validation", lambda: _check_rng(rng_error))
    if stream is not None:
        selected_scores, selected_ids = stream.result()
    else:
        selected_scores, selected_ids = measured("selection", lambda: topk(
            xp.concatenate(all_scores), xp.concatenate(all_ids), k))
    host_scores, host_ids = measured("records_to_host", lambda: (
        to_numpy(selected_scores), to_numpy(selected_ids)))
    assignments = None
    if output == "assignments":
        def materialize():
            host_w = np.empty((k, problem.n), dtype=np.uint8)
            for start in range(0, k, batch_size):
                end = min(start + batch_size, k)
                host_w[start:end] = to_numpy(generate(
                    problem.n, n1, host_ids[start:end], seed,
                    sampler=sampler, device=problem.device))
            return host_w
        assignments = measured("materialize_and_transfer", materialize)
    synchronize(problem.device)
    timings["total"] = time.perf_counter() - begin
    metadata = {
        "replay_version": REPLAY_VERSION, "n": problem.n, "p": problem.p,
        "n1": n1, "candidates": candidates, "k": k, "batch_size": batch_size,
        "seed": seed, "start_id": start_id, "sampler": sampler,
        "balance": balance, "selection": selection, "device": problem.device,
        "criterion": problem.criterion, "precision": problem.precision,
        "whitening": problem.whitening, "method": problem.method,
        "output": output, "output_location": "host", "status": "complete",
        "prepared": problem.metadata, "profile_instrumented": profile,
        "validation": validation,
    }
    return SampleResult(np.asarray(host_scores), np.asarray(host_ids, dtype=np.uint64),
                        assignments, metadata, timings)
