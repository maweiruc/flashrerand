"""Repair-first fused correctness, replay, resource and failure-path gates."""
import os
import unittest
from copy import deepcopy
from dataclasses import replace
from unittest import mock

import numpy as np

from flashrerand import cuda_backend as cuda
from flashrerand.core import prepare
from flashrerand.engine import replay, sample_topk
from flashrerand.rng import philox_words
from flashrerand.samplers import generate

ATOL, RTOL = 1e-10, 1e-11
try:
    import cupy as cp
    if cp.cuda.runtime.getDeviceCount() < 1 or float(cp.arange(4).sum()) != 6:
        raise RuntimeError("CUDA probe failed")
    CUDA_AVAILABLE, CUDA_REASON = True, ""
except Exception as exc:
    cp = None
    CUDA_AVAILABLE, CUDA_REASON = False, f"CUDA unavailable: {exc}; repair-fused GPU gates NOT executed"


def oracle(x, w, n1):
    covariance = np.atleast_2d(np.cov(x, rowvar=False, ddof=1))
    delta = np.asarray([x[row == 1].mean(0) - x[row == 0].mean(0) for row in w])
    return n1 * (len(x) - n1) / len(x) * np.sum(
        delta * np.linalg.solve(covariance, delta.T).T, axis=1)


def search(problem, **kwargs):
    options = dict(n1=problem.n // 2, candidates=35, k=7, seed=29,
                   sampler="repair", balance="gemm", selection="all", batch_size=8,
                   validation="deferred", output="assignments", _repair_fused=True)
    options.update(kwargs)
    return sample_topk(problem, **options)


class FusedContractTests(unittest.TestCase):
    def test_cuda_requirement(self):
        if os.environ.get("FLASHRERAND_REQUIRE_CUDA") == "1":
            self.assertTrue(CUDA_AVAILABLE, CUDA_REASON)
        elif not CUDA_AVAILABLE:
            self.skipTest(CUDA_REASON)

    def test_constraints_before_device_access(self):
        cpu = prepare(np.random.default_rng(8).normal(size=(17, 3)), criterion="full", precision="fp64")
        with self.assertRaisesRegex(ValueError, "CUDA"):
            search(cpu)
        gpu = replace(cpu, device="cuda")
        for flag in (1, "true", None):
            with self.subTest(flag=flag), self.assertRaises(TypeError):
                search(gpu, _repair_fused=flag)
        for changes in ({"sampler": "sequential"}, {"sampler": "keys"}, {"balance": "fused"},
                        {"selection": "stream"}, {"validation": "checked"}, {"_packed_repair": True},
                        {"_packed_workspace": True}, {"_deferred_sequential": True},
                        {"_score_callback": lambda *_: None}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                search(gpu, **changes)
        for invalid in (replace(gpu, criterion="diagonal"), replace(gpu, precision="fp32"),
                        replace(gpu, whitening="post"), replace(gpu, n=1025), replace(gpu, p=33)):
            with self.assertRaises(ValueError):
                search(invalid)
            with self.assertRaises(ValueError):
                cuda.fused_repair_cuda(invalid, 8, np.arange(2, dtype=np.uint64), 0)
        for limits in ((), (0, 1), (2**24, 1, False), (0, 2**32 + 1, False), (0, 1, 1)):
            with self.assertRaises(ValueError):
                cuda.fused_repair_cuda(gpu, 8, np.arange(2, dtype=np.uint64), 0, _test_limits=limits)


@unittest.skipUnless(CUDA_AVAILABLE, CUDA_REASON)
class RepairFusedGpuTests(unittest.TestCase):
    def setUp(self):
        self.x = np.random.default_rng(23).normal(size=(67, 5))
        self.problem = prepare(self.x, criterion="full", precision="fp64", device="cuda",
                               prepare_backend="cuda-qr")

    def test_full_candidate_bits_tails_extremes_high_ids_and_c1_score_order(self):
        ids = np.array([0, 2**32 + 3, 2**53 + 1, 2**63 + 7, 2**64 - 1,
                        19, 0, 19, 103], dtype=np.uint64)
        for n, p in ((3, 1), (7, 3), (8, 3), (9, 3), (31, 5), (32, 5), (33, 5),
                     (63, 5), (64, 5), (65, 5), (1023, 32), (1024, 32)):
            x = np.random.default_rng(n).normal(size=(n, p))
            problem = prepare(x, criterion="full", precision="fp64", device="cuda", prepare_backend="cuda-qr")
            for n1 in sorted({1, n // 2, n - 1}):
                expected = generate(n, n1, ids, 2**64 - 1, "repair", device="cpu")
                packed = cuda.generate_packed_repair_cuda(n, n1, ids, 2**64 - 1)
                expected_scores = cp.asnumpy(cuda.score_fixed_w_cuda(problem, packed, n1,
                    representation="packed", candidate_tile=8, subject_tile=64))
                for count in (1, 7, 8, 9):
                    with self.subTest(n=n, p=p, n1=n1, count=count):
                        debug = cp.full((count, (n + 7) // 8), 255, dtype=cp.uint8)
                        scores = cuda.fused_repair_cuda(problem, n1, ids[:count], 2**64 - 1,
                                                        _debug_packed_out=debug)
                        host = cp.asnumpy(debug)
                        np.testing.assert_array_equal(host, np.packbits(expected[:count], axis=1, bitorder="little"))
                        np.testing.assert_array_equal(cp.asnumpy(scores), expected_scores[:count])
                        np.testing.assert_allclose(cp.asnumpy(scores), oracle(x, expected[:count], n1), atol=ATOL, rtol=RTOL)
                        if n % 8:
                            self.assertTrue(np.all(host[:, -1] >> (n % 8) == 0))
            self.assertEqual(cuda.fused_repair_cuda(problem, 1, ids[:0], 0).shape, (0,))

    def test_complete_pool_topk_and_checked_replay_without_dense_intermediate(self):
        # A constant shift must cancel through the correction vector.
        self.problem.z = cp.ascontiguousarray(self.problem.z + cp.linspace(.25, .75, 5))
        self.problem.correction = self.problem.z.sum(0)
        start, count, n1 = 2**64 - 35, 35, 33
        ids = np.arange(count, dtype=np.uint64) + np.uint64(start)
        expected_w = generate(67, n1, ids, 29, "repair", device="cpu")
        reference = oracle(self.x, expected_w, n1)
        original = cuda.fused_repair_cuda
        original_metadata = deepcopy(self.problem.metadata)
        saved = []
        for batch in (7, 8, 9, 17):
            for k in (1, 7, count):
                parts = []
                def capture(*args, **kwargs):
                    values = original(*args, **kwargs)
                    parts.append(cp.asnumpy(values))
                    return values
                with self.subTest(batch=batch, k=k), \
                     mock.patch.object(cuda, "fused_repair_cuda", side_effect=capture), \
                     mock.patch.object(cuda, "generate_cuda", side_effect=AssertionError("dense W forbidden")), \
                     mock.patch("flashrerand.generated_gemm.score_generated", side_effect=AssertionError("GEMM forbidden")):
                    result = search(self.problem, n1=n1, start_id=start, batch_size=batch, k=k)
                scores = np.concatenate(parts)
                np.testing.assert_allclose(scores, reference, atol=ATOL, rtol=RTOL)
                order = np.lexsort((ids, scores))[:k]
                np.testing.assert_array_equal(result.candidate_ids, ids[order])
                np.testing.assert_array_equal(result.assignments, expected_w[order])
                np.testing.assert_array_equal(result.scores, scores[order])
                np.testing.assert_array_equal(replay(result.replay_record(), device="cpu"), result.assignments)
                self.assertEqual(result.metadata["actual_balance"], "fp64_fused_repair_c8_s64")
                self.assertTrue(result.metadata["experimental"])
                self.assertFalse(result.metadata["round3"]["global_assignment_buffer"])
                self.assertEqual(self.problem.metadata, original_metadata)
                for previous, snapshot in saved:
                    self.assertEqual(previous.metadata, snapshot)
                saved.append((result, deepcopy(result.metadata)))
        self.problem.metadata["later_mutation"] = True
        self.assertNotIn("later_mutation", saved[0][0].metadata["prepared"])

    def test_production_allocates_only_scores_and_error_flag_and_defers_checks(self):
        ids = cp.arange(9, dtype=cp.uint64)
        cuda.fused_repair_cuda(self.problem, 33, ids, 29)
        original_empty, original_zeros = cp.empty, cp.zeros
        allocations = []
        def allocated(factory):
            def wrapped(*args, **kwargs):
                out = factory(*args, **kwargs)
                allocations.append((out.shape, str(out.dtype)))
                return out
            return wrapped
        with mock.patch.object(cp, "empty", side_effect=allocated(original_empty)), \
             mock.patch.object(cp, "zeros", side_effect=allocated(original_zeros)):
            cuda.fused_repair_cuda(self.problem, 33, ids, 29)
        self.assertEqual(allocations, [((9,), "float64"), ((1,), "uint32")])
        flag = cp.asarray([2], dtype=cp.uint32)
        with mock.patch.object(cuda, "_check_rng", side_effect=AssertionError("unexpected RNG read")), \
             mock.patch.object(cp, "isfinite", side_effect=AssertionError("unexpected finite read")):
            scores = cuda.fused_repair_cuda(self.problem, 33, ids, 29, _error_out=flag)
            empty = cuda.fused_repair_cuda(self.problem, 33, ids[:0], 29, _error_out=flag)
        self.assertEqual(empty.shape, (0,))
        self.assertEqual(int(flag.item()), 2)
        self.assertTrue(np.all(np.isfinite(cp.asnumpy(scores))))

    def test_bounded_and_trial_exhaustion_preserve_barriers_and_production_module(self):
        ids = np.arange(9, dtype=np.uint64)
        # Include one unmodified candidate beside candidates requiring repair.
        bits = philox_words(29, ids, 0, 3)[:, 0] & np.uint32((1 << 31) - 1)
        counts = np.array([int(word).bit_count() for word in bits])
        n1 = int(counts[0])
        self.assertTrue(np.any(counts != n1))
        x = np.random.default_rng(31).normal(size=(31, 5))
        problem = prepare(x,
                          criterion="full", precision="fp64", device="cuda", prepare_backend="cuda-qr")
        expected_first = oracle(x, generate(31, n1, ids[:1], 29, "repair", device="cpu"), n1)
        for limits, code in (((0, 2**32, True), 1), ((2**24 - 1, 0, False), 2)):
            for count in (7, 8, 9):
                with self.subTest(limits=limits, count=count):
                    flag = cp.zeros(1, dtype=cp.uint32)
                    values = cuda.fused_repair_cuda(problem, n1, ids[:count], 29,
                        _error_out=flag, _test_limits=limits)
                    cp.cuda.get_current_stream().synchronize()
                    self.assertEqual(int(flag.item()), code)
                    np.testing.assert_allclose(cp.asnumpy(values)[:1], expected_first, atol=ATOL, rtol=RTOL)
                    with self.assertRaisesRegex(RuntimeError, f"CUDA error {code}"):
                        cuda.fused_repair_cuda(problem, n1, ids[:count], 29, _test_limits=limits)
        # Test-only macros must never contaminate the normal compiled module.
        debug = cp.empty((9, 4), dtype=cp.uint8)
        cuda.fused_repair_cuda(problem, n1, ids, 29, _debug_packed_out=debug)
        expected = generate(31, n1, ids, 29, "repair", device="cpu")
        np.testing.assert_array_equal(cp.asnumpy(debug), np.packbits(expected, axis=1, bitorder="little"))

    def test_sticky_failure_and_nonfinite_unselected_scores_are_not_hidden(self):
        original, pointers = cuda.fused_repair_cuda, []
        def injected(*args, **kwargs):
            values = original(*args, **kwargs)
            flag = kwargs["_error_out"]
            pointers.append(flag.data.ptr)
            if len(pointers) == 1:
                flag[0] = cp.uint32(2)
            return values
        with mock.patch.object(cuda, "fused_repair_cuda", side_effect=injected), \
             mock.patch("flashrerand.engine.topk", side_effect=AssertionError("selection after RNG error")):
            with self.assertRaisesRegex(RuntimeError, "RNG"):
                search(self.problem, candidates=17, k=1, batch_size=7)
        self.assertEqual(len(pointers), 3)
        self.assertEqual(len(set(pointers)), 1)
        for invalid in (np.nan, np.inf, -np.inf):
            def nonfinite(*args, **kwargs):
                values = original(*args, **kwargs)
                values[-1] = invalid
                return values
            with mock.patch.object(cuda, "fused_repair_cuda", side_effect=nonfinite):
                with self.assertRaisesRegex(ValueError, "finite"):
                    search(self.problem, candidates=17, k=1, batch_size=7)
        # Replay owns a separate checked flag, including selected-output errors.
        with mock.patch.object(cuda, "generate_packed_repair_cuda", side_effect=RuntimeError("RNG replay failed")):
            with self.assertRaisesRegex(RuntimeError, "RNG replay"):
                search(self.problem, candidates=9, k=1)

    def test_debug_flag_validation_and_array_device_ownership(self):
        ids = cp.arange(9, dtype=cp.uint64)
        for flag in (cp.zeros(2, cp.uint32), cp.zeros(1, cp.uint64), np.zeros(1, np.uint32)):
            with self.assertRaises(ValueError):
                cuda.fused_repair_cuda(self.problem, 33, ids, 29, _error_out=flag)
        for debug in (cp.empty((9, 8), cp.uint8), cp.empty((9, 9), cp.uint32),
                      np.empty((9, 9), np.uint8), cp.empty((9, 18), cp.uint8)[:, ::2]):
            with self.assertRaises(ValueError):
                cuda.fused_repair_cuda(self.problem, 33, ids, 29, _debug_packed_out=debug)
        aliased = self.problem.z.view(cp.uint8).reshape(-1)[:81].reshape(9, 9)
        with self.assertRaisesRegex(ValueError, "overlap"):
            cuda.fused_repair_cuda(self.problem, 33, ids, 29, _debug_packed_out=aliased)
        if cp.cuda.runtime.getDeviceCount() > 1:
            with cp.cuda.Device(1):
                with self.assertRaisesRegex(ValueError, "current CUDA device"):
                    cuda.fused_repair_cuda(self.problem, 33, ids, 29)

    def test_exact_ties_and_records_profile_path(self):
        problem = prepare(np.array([[0.], [0.], [1.], [1.]]), criterion="full", precision="fp64",
                          device="cuda", prepare_backend="cuda-qr")
        start = 2**64 - 17
        result = search(problem, n1=1, candidates=17, start_id=start)
        np.testing.assert_array_equal(result.candidate_ids, np.arange(7, dtype=np.uint64) + np.uint64(start))
        self.assertTrue(np.all(result.scores == result.scores[0]))
        records = search(problem, n1=1, candidates=17, output="records", profile=True)
        self.assertIsNone(records.assignments)
        self.assertIn("sample_and_balance", records.timings)


if __name__ == "__main__":
    unittest.main()
