# FlashRerand V0

## Overview

GPU-accelerated rerandomization in Python. Generate a fixed budget of assignments,
with exactly `n1` treated units each, and return the global best `k` by
full-covariance Mahalanobis balance (FP64, sample covariance with `ddof=1`).
Set `k=1` for best-choice rerandomization; this is a fixed-budget search, not
threshold-based acceptance. Lower scores are better; ties use candidate IDs.

## Install

Python 3.10+. Choose CPU or CUDA (an NVIDIA GPU with CUDA 12.x is required):

```sh
python -m pip install .             # CPU: NumPy
python -m pip install ".[cuda12]"  # GPU: CuPy 13.6.0
```

## Example

```python
import numpy as np
from flashrerand import rerandomize

X = np.random.default_rng(2026).normal(size=(1024, 16))
result = rerandomize(X, n1=512, candidates=1024, k=32,
                    seed=42, device="cuda")

print(result.assignments.shape)  # (32, 1024), each row has 512 treated units
print(result.scores[:5])         # Lowest balance scores
```

Use `device="cpu"` for CPU execution. Results also include `candidate_ids` and
`metadata`; all returned arrays reside on the host. CUDA unavailability raises
an error. CPU and GPU use different samplers, so equal seeds do not imply equal
candidate pools.

## Performance

**48.4–611.7× faster than the included NumPy baseline** in the three measured
workloads below. These are complete API calls, including preparation and host
output, not isolated kernel timings or comparisons against the fastest CPU tool.

![FP64 rerandomization: NumPy baseline and CUDA](assets/performance.svg)

Measured on a Xeon Gold 6342 and one A100 40 GB, using balanced allocation and
normal covariates. CPU/GPU host thread limits were 8/1; active thread counts
were not measured. CPU uses NumPy keys + GEMM; GPU uses repair + GEMM.

Each seed (42, 43, 44) uses 2 warmups and 5 timed calls. Bars are medians of seed
medians; speedups are medians of paired-seed ratios, which can differ from the
ratio of displayed times. First calls are recorded separately, with caches
possibly present. [Raw timings, settings and environment](assets/benchmark.json).

## Tests and reproduction

All **13 tests passed on A100**, including GPU execution and compilation from an
empty kernel cache. CPU-only checks also passed on macOS. Tests compare scores
with an independent CPU formula (`atol=1e-10, rtol=1e-11`) and check small full
candidate pools. Benchmark validation checks selected outputs only.

```sh
python -m unittest discover -s tests -v
# Require GPU coverage instead of allowing skipped CUDA tests:
FLASHRERAND_REQUIRE_CUDA=1 python -m unittest discover -s tests -v

python -m pip install ".[benchmark]"  # Optional plotting dependency
python benchmark.py run --output results/benchmark.json
python benchmark.py plot results/benchmark.json --output results/performance.svg
```

Add `--smoke` to `run` for a small trial. Use new output paths for each run;
existing results are never overwritten. Keep `.private.log` diagnostics private.

## Notes and license

Inputs must be finite and have full column rank after centering, with
`1 <= p < n-1`, `1 <= n1 < n`, and `1 <= k <= candidates`. Constant columns and
covariance condition numbers above `1e10` are rejected; no ridge is added.
Repeated assignments are allowed. Close scores may rank differently across
floating-point paths. V0 uses one CUDA device; performance depends on hardware
and workload. Internal module APIs are not stable public interfaces.

[MIT License](LICENSE).
