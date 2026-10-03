"""CUDA gates. FLASHRERAND_REQUIRE_CUDA=1 makes unavailable CUDA a failure."""
import os
import unittest
from unittest import mock

import numpy as np

from flashrerand import rerandomize
from flashrerand.core import prepare, score
from flashrerand.engine import replay
from flashrerand.samplers import generate
from flashrerand.selection import topk


REQUIRE_CUDA = os.environ.get("FLASHRERAND_REQUIRE_CUDA") == "1"
try:
    import cupy as cp
    if cp.cuda.runtime.getDeviceCount() < 1:
        raise RuntimeError("no CUDA device")
    probe = cp.arange(4, dtype=cp.float64).sum()
    cp.cuda.get_current_stream().synchronize()
    if float(probe) != 6:
        raise RuntimeError("CUDA arithmetic probe failed")
    CUDA_AVAILABLE, CUDA_REASON = True, ""
except Exception as exc:
    cp = None
    CUDA_AVAILABLE = False
    CUDA_REASON = f"CUDA unavailable ({type(exc).__name__}: {exc}); GPU tests were not executed"

ATOL, RTOL = 1e-10, 1e-11


def oracle(x, w, n1):
    covariance = np.atleast_2d(np.cov(x, rowvar=False, ddof=1))
    values = []
    for row in w:
        delta = x[row == 1].mean(0) - x[row == 0].mean(0)
        values.append(n1 * (len(x) - n1) / len(x) * delta @ np.linalg.solve(covariance, delta))
    return np.asarray(values)


class CudaRequirement(unittest.TestCase):
    def test_cuda_requirement(self):
        if REQUIRE_CUDA:
            self.assertTrue(CUDA_AVAILABLE, CUDA_REASON)
        elif not CUDA_AVAILABLE:
            self.skipTest(CUDA_REASON)


@unittest.skipUnless(CUDA_AVAILABLE, CUDA_REASON)
class GpuTests(unittest.TestCase):
    def test_public_gpu_rejects_invalid_covariates(self):
        x = np.random.default_rng(12).normal(size=(17, 3))
        duplicate = x.copy(); duplicate[:, 2] = duplicate[:, 0]
        bad_condition = x.copy(); bad_condition[:, 2] *= 1e-6
        nonfinite = x.copy(); nonfinite[0, 0] = np.nan
        for value in (duplicate, bad_condition, nonfinite, np.ones_like(x)):
            with self.subTest(value=value[0].tolist()), self.assertRaises(ValueError):
                rerandomize(value, n1=8, candidates=17, device="cuda")

    def test_repair_cpu_gpu_exact_with_high_ids_and_batch_replay(self):
        ids = np.array([0, 1, 2**32 + 7, 2**53 + 1, 2**63 + 11, 2**64 - 1], dtype=np.uint64)
        for n, n1 in ((33, 16), (17, 1), (17, 16)):
            with self.subTest(n=n, n1=n1):
                cpu = generate(n, n1, ids, 2**64 - 1, sampler="repair", device="cpu")
                gpu = cp.asnumpy(generate(n, n1, ids, 2**64 - 1, sampler="repair", device="cuda"))
                parts = [cp.asnumpy(generate(n, n1, part, 2**64 - 1, sampler="repair", device="cuda"))
                         for part in (ids[:2], ids[2:])]
                np.testing.assert_array_equal(gpu, cpu)
                np.testing.assert_array_equal(np.concatenate(parts), cpu)
                self.assertTrue(np.all(gpu.sum(1) == n1))

    def test_fixed_pool_oracle_and_public_topk_with_tail_batches(self):
        x = np.random.default_rng(31).normal(size=(37, 5))
        c, n1, seed = 35, 18, 29
        ids = np.arange(c, dtype=np.uint64)
        w = generate(len(x), n1, ids, seed, sampler="repair", device="cpu")
        expected = oracle(x, w, n1)
        problem = prepare(x, criterion="full", precision="fp64", device="cuda", prepare_backend="cuda-qr")
        actual = cp.asnumpy(score(problem, cp.asarray(w), n1))
        np.testing.assert_allclose(actual, expected, atol=ATOL, rtol=RTOL)
        order = np.lexsort((ids, expected))
        previous = {}
        for batch in (8, 17):
            for k in (1, 7, c):
                with self.subTest(batch=batch, k=k):
                    result = rerandomize(x, n1=n1, candidates=c, k=k, seed=seed,
                                         device="cuda", batch_size=batch)
                    self.assertEqual(result.scores.dtype, np.dtype("float64"))
                    self.assertEqual(result.candidate_ids.dtype, np.dtype("uint64"))
                    self.assertEqual(result.assignments.shape, (k, len(x)))
                    self.assertTrue(np.all(result.assignments.sum(1) == n1))
                    np.testing.assert_array_equal(result.assignments, w[result.candidate_ids])
                    np.testing.assert_array_equal(replay(result.replay_record(), device="cpu"), result.assignments)
                    np.testing.assert_allclose(result.scores, expected[result.candidate_ids], atol=ATOL, rtol=RTOL)
                    np.testing.assert_array_equal(np.lexsort((result.candidate_ids, result.scores)), np.arange(k))
                    if k < c:
                        gap = expected[order[k]] - expected[order[k - 1]]
                        self.assertGreater(gap, 2 * (ATOL + RTOL * np.max(np.abs(expected))))
                    np.testing.assert_array_equal(np.sort(result.candidate_ids), np.sort(ids[order[:k]]))
                    if k in previous:
                        np.testing.assert_array_equal(result.candidate_ids, previous[k])
                    previous[k] = result.candidate_ids

    def test_uint64_score_ties_keep_integer_order(self):
        ids = np.array([2**64 - 1, 2**53 + 1, 2**53, 7, 0], dtype=np.uint64)
        values = np.array([2., -0., 0., -1., 2.])
        scores, selected = topk(cp.asarray(values), cp.asarray(ids), len(ids))
        order = np.lexsort((ids, values))
        np.testing.assert_array_equal(cp.asnumpy(selected), ids[order])
        np.testing.assert_array_equal(cp.asnumpy(scores), values[order])

    def test_deferred_rng_error_remains_sticky_across_batches(self):
        from flashrerand import cuda_backend
        original, flags = cuda_backend.generate_cuda, []
        def injected(*args, **kwargs):
            w = original(*args, **kwargs)
            flag = kwargs.get("_error_out")
            if flag is not None:
                flags.append(flag.data.ptr)
                if len(flags) == 1:
                    flag[0] = cp.uint32(2)
            return w
        x = np.random.default_rng(8).normal(size=(17, 3))
        with mock.patch("flashrerand.cuda_backend.generate_cuda", side_effect=injected):
            with self.assertRaisesRegex(RuntimeError, "RNG"):
                rerandomize(x, n1=8, candidates=17, k=1, batch_size=7, device="cuda")
        self.assertEqual(len(flags), 3)
        self.assertEqual(len(set(flags)), 1)

    def test_deferred_nonfinite_scores_never_return_a_result(self):
        from flashrerand import generated_gemm
        original = generated_gemm.score_generated
        calls = 0
        def injected(problem, w, n1):
            nonlocal calls
            calls += 1
            values = original(problem, w, n1)
            if calls == 1:
                values[0] = cp.nan
            return values
        x = np.random.default_rng(9).normal(size=(17, 3))
        with mock.patch("flashrerand.generated_gemm.score_generated", side_effect=injected):
            with self.assertRaisesRegex(ValueError, "finite"):
                rerandomize(x, n1=8, candidates=17, k=1, batch_size=7, device="cuda")


if __name__ == "__main__":
    unittest.main()
