"""Bounded-batch orchestration; all selection is over the full candidate pool."""
from __future__ import annotations

from dataclasses import dataclass
from copy import deepcopy
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
                profile: bool = False, validation: str = "checked",
                _empty_output: bool = False, _event_profile=None,
                _score_callback=None, _packed_repair: bool = False,
                _deferred_sequential: bool = False,
                _packed_workspace: bool = False,
                _repair_fused: bool = False) -> SampleResult:
    """Return the global smallest (score, ID) pairs; never deduplicates W.

    ``profile=True`` synchronizes each component. Use it for a separate
    diagnostic pass, not for an uninstrumented end-to-end speedup claim.
    Returned arrays live on the host. Complete W reconstruction is timed.
    Private round-one controls leave the public default unchanged. An event
    callback records diagnostic ranges without synchronizing each batch.
    Round two's private sequential control gives materialized GEMM and fused
    scoring the same whole-pool RNG/finite checks, including checked replay.
    Private packed workspace storage belongs to one complete search; its
    allocation and initialization are included in the measured call.
    The repair-fused prototype retains only shared packed candidate state
    during scoring; selected assignments use the checked packed replay path.
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
    if not isinstance(_repair_fused, (bool, np.bool_)):
        raise TypeError("_repair_fused must be boolean")
    if _repair_fused and (problem.device != "cuda" or problem.criterion != "full" or
            problem.precision != "fp64" or problem.whitening != "pre" or
            sampler != "repair" or balance != "gemm" or validation != "deferred" or
            selection != "all" or _packed_repair or _packed_workspace or
            _deferred_sequential or _score_callback is not None):
        raise ValueError("repair fused requires CUDA/full/FP64/pre repair/GEMM/deferred/all "
                         "and no packed, sequential, workspace or scorer override")
    if _repair_fused and (problem.n > 1024 or problem.p > 32):
        raise ValueError("repair fused currently supports n <= 1024 and p <= 32")
    if not isinstance(_deferred_sequential, (bool, np.bool_)):
        raise TypeError("_deferred_sequential must be boolean")
    if _deferred_sequential and (validation != "deferred" or problem.device != "cuda" or
            sampler != "sequential" or balance not in ("gemm", "fused") or selection != "all" or
            problem.criterion != "full" or problem.precision != "fp64" or problem.whitening != "pre" or
            _packed_repair or _score_callback is not None):
        raise ValueError("deferred sequential requires CUDA/full/FP64/pre, sequential GEMM or fused, "
                         "deferred/all selection, and no packed repair or scorer override")
    if validation == "deferred" and (problem.device != "cuda" or sampler != "repair" or
                                      balance != "gemm" or selection != "all") and not _deferred_sequential:
        raise ValueError("deferred validation currently requires CUDA repair + GEMM + all selection")
    if balance == "fused" and (problem.device != "cuda" or sampler != "sequential"):
        raise ValueError("fused requires device='cuda' and sampler='sequential'")
    if problem.device == "cpu" and balance in ("tiled", "packed"):
        raise ValueError(f"balance='{balance}' requires device='cuda'")
    if problem.device == "cuda" and balance == "gather":
        raise ValueError("gather is currently a CPU reference backend only")
    if problem.device == "cuda" and balance != "gemm" and problem.whitening != "pre":
        raise ValueError("custom CUDA scoring requires whitening='pre'")
    if not isinstance(_empty_output, (bool, np.bool_)):
        raise TypeError("_empty_output must be boolean")
    if _empty_output and problem.device != "cuda":
        raise ValueError("_empty_output requires CUDA")
    if _event_profile is not None:
        if not callable(_event_profile) or problem.device != "cuda" or profile:
            raise ValueError("event profiling requires a CUDA callback and profile=False")
    if _score_callback is not None and (not callable(_score_callback) or
            validation != "deferred" or problem.whitening != "pre"):
        raise ValueError("a trusted scorer requires deferred validation and pre-whitening")
    if not isinstance(_packed_repair, (bool, np.bool_)):
        raise TypeError("_packed_repair must be boolean")
    if _packed_repair and (validation != "deferred" or problem.whitening != "pre" or
            problem.precision != "fp64" or problem.criterion != "full" or
            _score_callback is not None):
        raise ValueError("PK1 requires deferred CUDA repair/GEMM/all, full FP64 pre-whitening, and no scorer override")
    if not isinstance(_packed_workspace, (bool, np.bool_)):
        raise TypeError("_packed_workspace must be boolean")
    if _packed_workspace and not _packed_repair:
        raise ValueError("packed workspace requires _packed_repair=True")
    xp = get_xp(problem.device)
    timings = {}

    def measured(name, call):
        if _event_profile is not None:
            return _event_profile(name, call)
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
    prepared_metadata = deepcopy(problem.metadata)
    if validation == "deferred":
        from .arrays import cuda_math_metadata
        from .cuda_backend import generate_cuda, _check_rng
        from .generated_gemm import score_generated
        prepared_metadata["cuda_math"] = cuda_math_metadata()
        rng_error = xp.zeros(1, dtype=xp.uint32)
        if _packed_repair or _repair_fused:
            from .cuda_backend import generate_packed_repair_cuda, score_fixed_w_cuda
    stream = StreamingTopK(k, device=problem.device) if selection == "stream" else None
    all_scores, all_ids = [], []
    workspace = None
    if _packed_workspace:
        def allocate_workspace():
            capacity = min(candidates, batch_size)
            return {
                "packed": xp.empty((capacity, (problem.n + 7) // 8), dtype=xp.uint8),
                "counts": xp.empty(capacity, dtype=xp.int64),
                "scores": xp.empty(candidates, dtype=xp.float64),
            }
        workspace = measured("workspace_setup", allocate_workspace)
        host_pool_ids = measured("ids_host", lambda: np.arange(candidates, dtype=np.uint64) + np.uint64(start_id))
        workspace["ids"] = measured("ids_to_device", lambda: xp.asarray(host_pool_ids))
        del host_pool_ids
    for offset in range(0, candidates, batch_size):
        size = min(batch_size, candidates - offset)
        # Construct offsets first; arange with stop=2**64 is not portable.
        if workspace is None:
            ids = measured("ids_host", lambda: np.arange(size, dtype=np.uint64) + np.uint64(start_id + offset))
            device_ids = measured("ids_to_device", lambda: xp.asarray(ids))
        else:
            device_ids = workspace["ids"][offset:offset + size]
        if _repair_fused:
            from .cuda_backend import fused_repair_cuda
            values = measured("sample_and_balance", lambda: fused_repair_cuda(
                problem, n1, device_ids, seed, _error_out=rng_error))
        elif balance == "fused":
            from .cuda_backend import fused_sequential_cuda
            if _deferred_sequential:
                values = measured("sample_and_balance", lambda: fused_sequential_cuda(
                    problem, n1, device_ids, seed, _error_out=rng_error))
            else:
                values = measured("sample_and_balance", lambda: fused_sequential_cuda(
                    problem, n1, device_ids, seed))
        else:
            if validation == "deferred":
                if _packed_repair:
                    if workspace is None:
                        w = measured("sample", lambda: generate_packed_repair_cuda(
                            problem.n, n1, device_ids, seed, _error_out=rng_error,
                            _profile=_event_profile))
                    else:
                        w = measured("sample", lambda: generate_packed_repair_cuda(
                            problem.n, n1, device_ids, seed, _error_out=rng_error,
                            _profile=_event_profile, _out=workspace["packed"][:size],
                            _counts_out=workspace["counts"][:size]))
                else:
                    w = measured("sample", lambda: generate_cuda(problem.n, n1, device_ids, seed,
                        sampler=sampler, _error_out=rng_error,
                        _empty_output=_empty_output, _profile=_event_profile))
            elif problem.device == "cuda" and (_empty_output or _event_profile is not None):
                from .cuda_backend import generate_cuda
                w = measured("sample", lambda: generate_cuda(problem.n, n1, device_ids, seed,
                    sampler=sampler, _empty_output=_empty_output, _profile=_event_profile))
            else:
                w = measured("sample", lambda: generate(problem.n, n1, ids, seed,
                                                         sampler=sampler, device=problem.device))
            representation = "dense"
            if balance == "packed":
                w = measured("representation", lambda: pack(w, device=problem.device))
                representation = "packed"
            if validation == "deferred":
                if _packed_repair:
                    if workspace is None:
                        values = measured("balance", lambda: score_fixed_w_cuda(
                            problem, w, n1, representation="packed", candidate_tile=8,
                            subject_tile=64))
                    else:
                        values = measured("balance", lambda: score_fixed_w_cuda(
                            problem, w, n1, representation="packed", candidate_tile=8,
                            subject_tile=64, _out=workspace["scores"][offset:offset + size]))
                elif _score_callback is not None:
                    values = measured("balance", lambda: _score_callback(problem, w, n1))
                elif _event_profile is None:
                    values = measured("balance", lambda: score_generated(problem, w, n1))
                else:
                    values = measured("balance", lambda: score_generated(problem, w, n1,
                        _profile=_event_profile))
            else:
                values = measured("balance", lambda: score(problem, w, n1,
                    backend=balance, representation=representation))
            del w
        if stream is not None:
            measured("selection", lambda: stream.update(values, device_ids))
        elif workspace is None:
            all_scores.append(values)
            all_ids.append(device_ids)
    if rng_error is not None:
        measured("rng_validation", lambda: _check_rng(rng_error))
    if stream is not None:
        selected_scores, selected_ids = stream.result()
    elif workspace is not None:
        selected_scores, selected_ids = measured("selection", lambda: topk(
            workspace["scores"], workspace["ids"], k))
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
                if _packed_repair or _repair_fused:
                    def selected_replay():
                        if workspace is not None:
                            packed = generate_packed_repair_cuda(problem.n, n1,
                                xp.asarray(host_ids[start:end]), seed, _error_out=rng_error,
                                _out=workspace["packed"][:end - start],
                                _counts_out=workspace["counts"][:end - start])
                            _check_rng(rng_error)
                            return packed
                        return generate_packed_repair_cuda(problem.n, n1,
                            xp.asarray(host_ids[start:end]), seed)
                elif _deferred_sequential:
                    from .cuda_backend import generate_cuda
                    def selected_replay():
                        return generate_cuda(problem.n, n1, xp.asarray(host_ids[start:end]), seed,
                                             sampler="sequential", _empty_output=_empty_output)
                elif problem.device == "cuda" and (_empty_output or _event_profile is not None):
                    from .cuda_backend import generate_cuda
                    def selected_replay():
                        return generate_cuda(problem.n, n1, xp.asarray(host_ids[start:end]), seed,
                                             sampler=sampler, _empty_output=_empty_output)
                else:
                    def selected_replay():
                        return generate(problem.n, n1, host_ids[start:end], seed,
                                        sampler=sampler, device=problem.device)
                selected_w = measured("selected_replay", selected_replay)
                host_selected = measured("assignments_to_host", lambda: to_numpy(selected_w))
                if _packed_repair or _repair_fused:
                    host_selected = measured("unpack_on_host", lambda: np.unpackbits(
                        host_selected, axis=1, count=problem.n, bitorder="little"))
                host_w[start:end] = host_selected
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
        "prepared": prepared_metadata, "profile_instrumented": profile,
        "validation": validation,
    }
    if _empty_output or _event_profile is not None or _score_callback is not None or _packed_repair:
        metadata["round1"] = {"empty_sampler_output": bool(_empty_output),
                              "event_profile_instrumented": _event_profile is not None,
                              "trusted_scorer_override": _score_callback is not None,
                              "packed_repair": bool(_packed_repair)}
    if _packed_repair:
        metadata["actual_sampler"] = "philox_repair_direct_packed"
        metadata["actual_balance"] = "fp64_packed_tiled_c8_s64"
        metadata["assignment_representation"] = "packed_uint8_little_endian"
        metadata["selected_replay"] = "packed_to_host_then_numpy_unpackbits"
    if _deferred_sequential:
        metadata["round2"] = {"deferred_sequential": True}
        metadata["actual_sampler"] = "philox_sequential"
        metadata["actual_balance"] = "fp64_gemm" if balance == "gemm" else "fp64_fused_sequential"
        metadata["selected_replay"] = "checked_sequential_dense_to_host"
    if _repair_fused:
        metadata["actual_sampler"] = "philox_repair_fused_shared_packed"
        metadata["actual_balance"] = "fp64_fused_repair_c8_s64"
        metadata["assignment_representation"] = "shared_packed_uint32_little_endian"
        metadata["selected_replay"] = "packed_to_host_then_numpy_unpackbits"
        metadata["experimental"] = True
        metadata["round3"] = {"repair_fused": True, "global_assignment_buffer": False,
                              "candidate_tile": 8, "subject_tile": 64}
    if _packed_workspace:
        metadata["round2"] = {"packed_workspace": True,
            "workspace_ownership": "one search, current device and stream; no cache",
            "workspace_setup_included": True,
            "workspace_bytes": {name: int(value.nbytes) for name, value in workspace.items()},
            "workspace_score_regions": "nonoverlapping full candidate pool",
            "workspace_selected_replay": "reused packed/count scratch with checked sticky RNG"}
    return SampleResult(np.asarray(host_scores), np.asarray(host_ids, dtype=np.uint64),
                        assignments, metadata, timings)
