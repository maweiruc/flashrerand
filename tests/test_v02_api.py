"""Public execution policy, cache ownership, and actual CPU/CUDA API gates."""

from copy import deepcopy
import inspect
import json
import os
import subprocess
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

import flashrerand
from flashrerand import rerandomize
from flashrerand.core import PreparedProblem, prepare
from flashrerand.engine import REPLAY_VERSION, SampleResult
from flashrerand import execution
from flashrerand.samplers import generate


ATOL, RTOL = 1e-10, 1e-11


def oracle(x, w, n1):
    covariance = np.atleast_2d(np.cov(x, rowvar=False, ddof=1))
    values = []
    for row in w:
        delta = x[row == 1].mean(0) - x[row == 0].mean(0)
        values.append(n1 * (len(x) - n1) / len(x) * delta @ np.linalg.solve(covariance, delta))
    return np.asarray(values)


def fake_problem(n=1024, p=16, device="cuda"):
    value = np.ones((1, 1))
    return PreparedProblem(n, p, device, "fp64", "full", "pre", "cholesky",
                           value, value, value, value, value,
                           dict(prepare_backend="cuda-qr", experimental_prepare_flags={}))


HARDWARE = dict(index=0, name="NVIDIA A100-PCIE-40GB", compute_capability=[8, 0],
                memory_bytes=40 * 1024**3, cupy_version="13.6.0")


def fake_search(problem, **options):
    return SampleResult(np.zeros(1), np.zeros(1, dtype=np.uint64), np.zeros((1, 2), dtype=np.uint8),
                        dict(replay_version=REPLAY_VERSION, seed=options["seed"],
                             batch_size=options["batch_size"], prepared=problem.metadata),
                        dict(total=0.0))


class ApiCpuTests(unittest.TestCase):
    def test_signature_and_version(self):
        signature = inspect.signature(rerandomize)
        self.assertEqual(list(signature.parameters), ["X", "n1", "candidates", "k", "seed",
                         "batch_size", "device", "sampler", "backend"])
        for name in list(signature.parameters)[1:]:
            self.assertEqual(signature.parameters[name].kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertEqual(signature.parameters["sampler"].default, "auto")
        self.assertEqual(signature.parameters["backend"].default, "auto")
        self.assertEqual(flashrerand.__version__, "0.2.0")

    def test_cpu_import_and_execution_do_not_import_cupy(self):
        code = """import importlib.abc, sys
class NoCupy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'cupy' or fullname.startswith('cupy.'):
            raise AssertionError('CPU attempted to import CuPy')
sys.meta_path.insert(0, NoCupy())
import numpy as np
from flashrerand import rerandomize
r = rerandomize(np.random.default_rng(7).normal(size=(13, 2)), n1=6,
                candidates=5, device='cpu')
assert r.metadata['execution']['actual_backend'] == 'gemm'
assert 'cupy' not in sys.modules
"""
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_all_supported_cpu_options_match_independent_oracle(self):
        x = np.random.default_rng(69).normal(size=(19, 4))
        w = generate(19, 9, np.arange(23, dtype=np.uint64), 42, sampler="keys", device="cpu")
        expected = oracle(x, w, 9)
        for sampler in ("auto", "keys"):
            for backend in ("auto", "gemm"):
                with self.subTest(sampler=sampler, backend=backend):
                    result = rerandomize(x, n1=9, candidates=23, k=23, seed=42,
                        batch_size=7, device="cpu", sampler=sampler, backend=backend)
                    np.testing.assert_array_equal(result.assignments, w[result.candidate_ids])
                    np.testing.assert_allclose(result.scores, expected[result.candidate_ids], atol=ATOL, rtol=RTOL)
                    execution_info = result.metadata["execution"]
                    self.assertEqual(execution_info["actual_sampler"], "keys")
                    self.assertEqual(execution_info["actual_backend"], "gemm")
                    self.assertEqual(execution_info["requested_backend"], backend)
                    self.assertFalse(execution_info["experimental"])
                    self.assertGreaterEqual(result.timings["total"], result.timings["search"])
                    self.assertGreaterEqual(result.timings["search"], result.timings["engine"])
                    json.dumps(result.replay_record(), allow_nan=False)

    def test_unsupported_combinations_fail_before_preprocessing(self):
        combinations = [("cpu", s, b) for s in ("auto", "keys", "repair", "sequential")
                        for b in ("auto", "gemm", "packed", "fused")
                        if s not in ("auto", "keys") or b not in ("auto", "gemm")]
        combinations += [("cuda", "keys", b) for b in ("auto", "gemm", "packed", "fused")]
        combinations += [("cuda", "sequential", "packed"), ("gpu", "auto", "auto"),
                         ("cpu", None, "auto"), ("cpu", "auto", None)]
        with mock.patch("flashrerand._prepare") as prepare_mock:
            for device, sampler, backend in combinations:
                with self.subTest(device=device, sampler=sampler, backend=backend):
                    with self.assertRaises(ValueError):
                        rerandomize(None, n1=1, candidates=1, device=device,
                                    sampler=sampler, backend=backend)
            prepare_mock.assert_not_called()

    def test_result_snapshots_do_not_alias_prepared_or_later_results(self):
        x = np.random.default_rng(81).normal(size=(17, 3))
        problem = prepare(x, criterion="full", precision="fp64", prepare_backend="cpu")
        first = execution.sample_prepared(problem, n1=8, candidates=11, k=3)
        frozen = json.dumps(first.replay_record(), sort_keys=True)
        problem.metadata["experimental_prepare_flags"]["compact"] = True
        second = execution.sample_prepared(problem, n1=8, candidates=11, k=3)
        second.metadata["prepared"]["experimental_prepare_flags"]["compact"] = False
        second.metadata["execution"]["preprocessing"]["experimental_flags"]["compact"] = False
        self.assertEqual(json.dumps(first.replay_record(), sort_keys=True), frozen)
        self.assertTrue(problem.metadata["experimental_prepare_flags"]["compact"])
        first.assignments[:] = 0
        self.assertTrue(np.all(second.assignments.sum(axis=1) == 8))

    def test_prepared_id_validation_and_high_id_snapshot(self):
        x = np.random.default_rng(91).normal(size=(17, 3))
        problem = prepare(x, criterion="full", precision="fp64", prepare_backend="cpu")
        result = execution.sample_prepared(problem, n1=8, candidates=3, k=3,
                                          _start_id=2**64 - 3)
        self.assertEqual(result.metadata["execution"]["candidate_id_range"],
                         dict(start=2**64 - 3, stop_exclusive=2**64))
        json.dumps(result.replay_record(), allow_nan=False)
        for changes in (dict(_start_id=-1), dict(_start_id=2**64 - 2), dict(seed=True),
                        dict(n1=0), dict(batch_size=0), dict(k=4), dict(candidates=True)):
            options = dict(n1=8, candidates=3)
            options.update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                execution.sample_prepared(problem, **options)


class DispatchTests(unittest.TestCase):
    def choose(self, problem=None, **changes):
        options = dict(sampler="repair", backend="auto", n1=512, candidates=8192,
                       k=64, batch_size=2048, hardware=deepcopy(HARDWARE))
        options.update(changes)
        with mock.patch.object(execution, "AUTO_PACKED_ENABLED", True):
            return execution._choose_backend(problem or fake_problem(), **options)

    def test_frozen_region_and_each_fallback_dimension(self):
        self.assertEqual(self.choose(), ("packed", "inside_frozen_packed_region"))
        for problem in (fake_problem(511, 16), fake_problem(1025, 16),
                        fake_problem(1024, 7), fake_problem(1024, 33)):
            self.assertEqual(self.choose(problem)[0], "gemm")
        for n in (512, 641, 897, 1023, 1024):
            for p in (8, 32):
                self.assertEqual(self.choose(fake_problem(n, p), n1=n // 2)[0], "packed")
        for changes in (dict(n1=511), dict(candidates=2050), dict(candidates=65537),
                        dict(k=65), dict(batch_size=512), dict(batch_size=4096),
                        dict(hardware=dict(name="NVIDIA A100-SXM4-40GB", compute_capability=[8, 0])),
                        dict(hardware=dict(name=HARDWARE["name"], compute_capability=[9, 0]))):
            with self.subTest(changes=changes):
                self.assertEqual(self.choose(**changes)[0], "gemm")
        for field, value in (("method", "qr"), ("precision", "fp32"), ("criterion", "diagonal")):
            problem = fake_problem()
            setattr(problem, field, value)
            self.assertEqual(self.choose(problem)[0], "gemm")
        for key, value in (("prepare_backend", "cuda"), ("experimental_prepare_flags", {"compact": True})):
            problem = fake_problem()
            problem.metadata[key] = value
            self.assertEqual(self.choose(problem)[0], "gemm")

    def test_explicit_override_and_sequential_auto_never_substitute_sampler(self):
        for sampler, backend, expected, flag in (
            ("auto", "fused", "repair", "_repair_fused"),
            ("repair", "packed", "repair", "_packed_repair"),
            ("sequential", "fused", "sequential", "_deferred_sequential"),
            ("sequential", "auto", "sequential", "_deferred_sequential"),
        ):
            with self.subTest(sampler=sampler, backend=backend), \
                    mock.patch.object(execution, "_hardware", return_value=deepcopy(HARDWARE)), \
                    mock.patch.object(execution, "sample_topk", side_effect=fake_search) as search:
                result = execution.sample_prepared(fake_problem(), n1=512, candidates=8192,
                                                   sampler=sampler, backend=backend)
                options = search.call_args.kwargs
                self.assertEqual(options["sampler"], expected)
                self.assertTrue(options[flag])
                info = result.metadata["execution"]
                self.assertEqual(info["actual_sampler"], expected)
                self.assertEqual(info["actual_backend"], "gemm" if backend == "auto" else backend)
                self.assertEqual(info["experimental"], backend == "fused")
                self.assertFalse(info["workspace_reuse"])

    def test_disabled_policy_is_global_gemm_without_changing_forced_packed(self):
        with mock.patch.object(execution, "AUTO_PACKED_ENABLED", False), \
                mock.patch.object(execution, "_hardware", return_value=deepcopy(HARDWARE)), \
                mock.patch.object(execution, "sample_topk", side_effect=fake_search):
            auto = execution.sample_prepared(fake_problem(), n1=512, candidates=8192)
            forced = execution.sample_prepared(fake_problem(), n1=512, candidates=8192, backend="packed")
        self.assertEqual(auto.metadata["execution"]["actual_backend"], "gemm")
        self.assertEqual(auto.metadata["execution"]["reason"], "packed_auto_disabled")
        self.assertEqual(forced.metadata["execution"]["actual_backend"], "packed")

    def test_fused_limits_and_engine_errors_are_not_fallbacks(self):
        with mock.patch.object(execution, "_hardware", return_value=deepcopy(HARDWARE)), \
                mock.patch.object(execution, "sample_topk", side_effect=RuntimeError("sticky RNG")) as search:
            for problem, sampler in ((fake_problem(1025, 16), "repair"),
                                     (fake_problem(1024, 33), "repair"),
                                     (fake_problem(2048, 1025), "sequential")):
                with self.subTest(n=problem.n, p=problem.p), self.assertRaisesRegex(ValueError, "fused requires"):
                    execution.sample_prepared(problem, n1=512, candidates=8192,
                                              sampler=sampler, backend="fused")
            search.assert_not_called()
            with self.assertRaisesRegex(RuntimeError, "sticky RNG"):
                execution.sample_prepared(fake_problem(), n1=512, candidates=8192)
            self.assertEqual(search.call_count, 1)

    def test_hardware_cache_is_immutable_process_and_device_specific(self):
        class Array:
            def __init__(self, device):
                self.device = SimpleNamespace(id=device)
        runtime = SimpleNamespace(
            getDevice=mock.Mock(return_value=0),
            getDeviceProperties=mock.Mock(return_value=dict(name=b"NVIDIA A100-PCIE-40GB", major=8,
                                                           minor=0, totalGlobalMem=40 * 1024**3)))
        fake_cp = SimpleNamespace(ndarray=Array, __version__="13.6.0", cuda=SimpleNamespace(runtime=runtime))
        problem = fake_problem()
        for field in ("x", "xc", "z", "factor", "correction"):
            setattr(problem, field, Array(0))
        execution._device_properties.cache_clear()
        self.addCleanup(execution._device_properties.cache_clear)
        with mock.patch.object(execution, "get_xp", return_value=fake_cp), \
                mock.patch.object(execution.os, "getpid", return_value=100) as pid:
            first = execution._hardware(problem)
            first["compute_capability"][0] = 9
            self.assertEqual(execution._hardware(problem)["compute_capability"], [8, 0])
            self.assertEqual(runtime.getDeviceProperties.call_count, 1)
            self.assertEqual(runtime.getDevice.call_count, 2)
            pid.return_value = 101
            execution._hardware(problem)
            self.assertEqual(runtime.getDeviceProperties.call_count, 2)
            runtime.getDevice.return_value = 1
            with self.assertRaisesRegex(ValueError, "current CUDA device"):
                execution._hardware(problem)
            self.assertEqual(runtime.getDeviceProperties.call_count, 2)
            for field in ("x", "xc", "z", "factor", "correction"):
                setattr(problem, field, Array(1))
            self.assertEqual(execution._hardware(problem)["index"], 1)
            self.assertEqual(runtime.getDeviceProperties.call_count, 3)
            problem.z = np.zeros((1, 1))
            with self.assertRaisesRegex(ValueError, "current CUDA device"):
                execution._hardware(problem)


class ApiCudaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import cupy as cp
            if cp.cuda.runtime.getDeviceCount() < 1:
                raise RuntimeError("no CUDA device")
            cp.cuda.get_current_stream().synchronize()
            cls.cp = cp
        except Exception as exc:
            reason = f"CUDA unavailable ({type(exc).__name__}: {exc}); API GPU tests were not executed"
            if os.environ.get("FLASHRERAND_REQUIRE_CUDA") == "1":
                raise RuntimeError(reason) from exc
            raise unittest.SkipTest(reason) from exc

    def test_public_repair_backends_full_pool_and_metadata(self):
        x = np.random.default_rng(504).normal(size=(37, 5))
        w = generate(37, 18, np.arange(19, dtype=np.uint64), 42, sampler="repair", device="cpu")
        expected = oracle(x, w, 18)
        for backend in ("auto", "gemm", "packed", "fused"):
            with self.subTest(backend=backend):
                result = rerandomize(x, n1=18, candidates=19, k=19, seed=42,
                                     batch_size=8, sampler="auto", backend=backend)
                np.testing.assert_array_equal(result.assignments, w[result.candidate_ids])
                np.testing.assert_allclose(result.scores, expected[result.candidate_ids], atol=ATOL, rtol=RTOL)
                info = result.metadata["execution"]
                self.assertEqual(info["actual_sampler"], "repair")
                self.assertEqual(info["actual_backend"], "gemm" if backend == "auto" else backend)
                self.assertEqual(info["experimental"], backend == "fused")
                self.assertIn("cupy_version", info["hardware"])
                json.dumps(result.replay_record(), allow_nan=False)

    def test_public_sequential_backends_same_candidate_stream(self):
        x = np.random.default_rng(505).normal(size=(33, 4))
        w = generate(33, 16, np.arange(17, dtype=np.uint64), 44, sampler="sequential", device="cpu")
        expected = oracle(x, w, 16)
        for backend in ("auto", "gemm", "fused"):
            result = rerandomize(x, n1=16, candidates=17, k=17, seed=44,
                                 batch_size=8, sampler="sequential", backend=backend)
            np.testing.assert_array_equal(result.assignments, w[result.candidate_ids])
            np.testing.assert_allclose(result.scores, expected[result.candidate_ids], atol=ATOL, rtol=RTOL)
            self.assertEqual(result.metadata["execution"]["actual_sampler"], "sequential")
            self.assertEqual(result.metadata["execution"]["actual_backend"], "gemm" if backend == "auto" else backend)

    def test_actual_auto_hardware_route_and_snapshot(self):
        x = np.random.default_rng(506).normal(size=(512, 8))
        problem = prepare(x, criterion="full", precision="fp64", device="cuda", prepare_backend="cuda-qr")
        first = execution.sample_prepared(problem, n1=256, candidates=2051, k=17, seed=42)
        frozen = json.dumps(first.replay_record(), sort_keys=True)
        hardware = first.metadata["execution"]["hardware"]
        expected, reason = execution._choose_backend(problem, sampler="repair", backend="auto",
            n1=256, candidates=2051, k=17, batch_size=2048, hardware=hardware)
        self.assertEqual(first.metadata["execution"]["actual_backend"], expected)
        self.assertEqual(first.metadata["execution"]["reason"], reason)
        second = execution.sample_prepared(problem, n1=256, candidates=2051, k=17, seed=43)
        second.metadata["execution"]["hardware"]["compute_capability"][0] = -1
        second.metadata["prepared"]["experimental_prepare_flags"]["compact"] = True
        self.assertEqual(json.dumps(first.replay_record(), sort_keys=True), frozen)
        self.assertFalse(problem.metadata["experimental_prepare_flags"]["compact"])
        np.testing.assert_allclose(first.scores, oracle(x, first.assignments, 256), atol=ATOL, rtol=RTOL)


if __name__ == "__main__":
    unittest.main()
