"""CPU gates; imports the installed package without altering sys.path."""
import itertools
import unittest
from unittest import mock

import numpy as np

from flashrerand import SampleResult, rerandomize
from flashrerand.core import prepare, score
from flashrerand.rng import known_zero_vector
from flashrerand.samplers import generate
from flashrerand.selection import topk


ATOL, RTOL = 1e-10, 1e-11


def oracle(x, w, n1):
    """Group means + sample covariance solve, independent of whitening/GEMM."""
    x = np.asarray(x, dtype=np.float64)
    covariance = np.atleast_2d(np.cov(x, rowvar=False, ddof=1))
    values = []
    for row in w:
        delta = x[row == 1].mean(0) - x[row == 0].mean(0)
        values.append(n1 * (len(x) - n1) / len(x) * delta @ np.linalg.solve(covariance, delta))
    return np.asarray(values)


class CpuTests(unittest.TestCase):
    def test_philox_published_vector_and_uint64_ties(self):
        self.assertEqual(known_zero_vector(), (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8))
        ids = np.array([2**64 - 1, 2**53 + 1, 2**53, 7, 0], dtype=np.uint64)
        values = np.array([2., -0., 0., -1., 2.])
        selected_scores, selected_ids = topk(values, ids, len(ids))
        order = np.lexsort((ids, values))
        np.testing.assert_array_equal(selected_ids, ids[order])
        np.testing.assert_array_equal(selected_scores, values[order])

    def test_exhaustive_fixed_count_pool_against_independent_oracle(self):
        x = np.random.default_rng(104).normal(size=(8, 3))
        rows = []
        for treated in itertools.combinations(range(8), 3):
            row = np.zeros(8, dtype=np.uint8)
            row[list(treated)] = 1
            rows.append(row)
        w = np.stack(rows)
        p = prepare(x, criterion="full", precision="fp64", prepare_backend="cpu")
        actual = score(p, w, 3)
        expected = oracle(x, w, 3)
        self.assertEqual(len(w), 56)
        np.testing.assert_allclose(actual, expected, atol=ATOL, rtol=RTOL)
        ids = np.arange(len(w), dtype=np.uint64)
        order = np.lexsort((ids, expected))
        k = 5
        tolerance = ATOL + RTOL * np.max(np.abs(expected))
        self.assertGreater(expected[order[k]] - expected[order[k - 1]], 2 * tolerance)
        _, selected = topk(actual, ids, k)
        np.testing.assert_array_equal(np.sort(selected), np.sort(ids[order[:k]]))

    def test_public_cpu_full_pool_topk_k1_tail_and_batch_invariance(self):
        x = np.random.default_rng(31).normal(size=(37, 5))
        c, n1, seed = 35, 18, 29
        ids = np.arange(c, dtype=np.uint64)
        w = generate(len(x), n1, ids, seed, sampler="keys", device="cpu")
        expected = oracle(x, w, n1)
        order = np.lexsort((ids, expected))
        previous = {}
        for batch in (8, 17):  # Both tail shapes, including a final batch of one.
            for k in (1, 7, c):
                with self.subTest(batch=batch, k=k):
                    result = rerandomize(x, n1=n1, candidates=c, k=k, seed=seed,
                                         device="cpu", batch_size=batch)
                    self.assertIsInstance(result, SampleResult)
                    self.assertEqual(result.assignments.shape, (k, len(x)))
                    self.assertEqual(result.scores.dtype, np.dtype("float64"))
                    self.assertEqual(result.candidate_ids.dtype, np.dtype("uint64"))
                    self.assertTrue(np.all(result.assignments.sum(1) == n1))
                    np.testing.assert_array_equal(result.assignments, w[result.candidate_ids])
                    np.testing.assert_allclose(result.scores, expected[result.candidate_ids], atol=ATOL, rtol=RTOL)
                    np.testing.assert_array_equal(np.lexsort((result.candidate_ids, result.scores)), np.arange(k))
                    if k < c:
                        gap = expected[order[k]] - expected[order[k - 1]]
                        self.assertGreater(gap, 2 * (ATOL + RTOL * np.max(np.abs(expected))))
                    np.testing.assert_array_equal(np.sort(result.candidate_ids), np.sort(ids[order[:k]]))
                    if k in previous:
                        np.testing.assert_array_equal(result.candidate_ids, previous[k])
                    previous[k] = result.candidate_ids

    def test_public_defaults_and_invalid_arguments(self):
        x = np.random.default_rng(6).normal(size=(13, 3))
        default = rerandomize(x, n1=6, candidates=11, device="cpu")
        self.assertEqual(default.assignments.shape, (1, 13))
        for changes in ({"n1": 0}, {"n1": 13}, {"n1": True}, {"n1": 1.5},
                        {"candidates": 0}, {"candidates": True}, {"k": 0}, {"k": 12},
                        {"batch_size": 0}, {"seed": -1}, {"seed": 2**64}, {"device": "gpu"}):
            args = dict(n1=6, candidates=11, device="cpu")
            args.update(changes)
            with self.subTest(changes=changes), self.assertRaises((TypeError, ValueError)):
                rerandomize(x, **args)

    def test_invalid_covariates_rank_and_condition(self):
        x = np.random.default_rng(12).normal(size=(17, 3))
        duplicate = x.copy(); duplicate[:, 2] = duplicate[:, 0]
        bad_condition = x.copy(); bad_condition[:, 2] *= 1e-6
        nonfinite = x.copy(); nonfinite[0, 0] = np.nan
        for bad in (np.ones((17, 3)), duplicate, bad_condition, nonfinite,
                    x.ravel(), np.ones((2, 2)), x.astype(complex)):
            with self.subTest(shape=bad.shape), self.assertRaises((TypeError, ValueError)):
                rerandomize(bad, n1=6, candidates=11, device="cpu")
        with self.assertRaisesRegex(ValueError, "condition_limit"):
            rerandomize(bad_condition, n1=6, candidates=11, device="cpu")

    def test_requested_cuda_failure_never_silently_runs_on_cpu(self):
        x = np.random.default_rng(8).normal(size=(13, 3))
        with mock.patch("flashrerand.core.get_xp", side_effect=RuntimeError("CUDA unavailable in test")):
            with self.assertRaisesRegex(RuntimeError, "CUDA"):
                rerandomize(x, n1=6, candidates=11, device="cuda")


if __name__ == "__main__":
    unittest.main()
