# gpu/linalg_public_api_throughput: CUDA public API benchmark

Measured cells: **3**. Unsupported cells: **1**, including **1** numerically verified but timing-excluded cells. Unsupported cells do not count as measured coverage.

Raw run: `tests/fixtures/render_run/gpu_linalg_public_api_throughput/20261102_210935/`.

Large-shape view of the same CUDA public linalg surface. `large` is a shape class, not a measured classification: a memory-bound row takes the same time here as at a small shape.

Values are median (IQR) **µs per operation**. Samples time batches of declared operations plus device synchronization; retained output memory is limited to 256 MiB. Batch counts and raw durations are preserved in JSONL.

Setup, input transfers, library/plan initialization, JIT compilation, validation downloads and output destruction are outside timing. Intrinsic output allocation is inside timing; output-reuse rows instead use preallocated destinations across samples. Accumulating destinations are restored from prepared GPU seeds and synchronized before every batch clock, including initialization and calibration. Direct and eager each use one entered session across samples; direct FFT uses cached `FftExecutor`. JAX eager and compiled JIT are separate arms.

All measured output elements are checked against independently generated NumPy/SciPy reference results before and after sampling. Unsupported rows include unavailable APIs and APIs whose internal setup cannot be excluded; these are not performance measurements. Short batches limited by output memory remain identifiable in the sampling table.

| Case | tenferro-cuda-direct | pytorch-cuda |
| --- | ---: | ---: |
| cholesky_f32_512x512 | 71.370 (0.124) | 68.870 (0.319) |
| householder_r_default_f32_512x256 | 34.698 (0.000) | unsupported |

## Sampling

| Case | Backend | Ops/sample | Samples | Median batch ms | Verification error/tolerance |
| --- | --- | ---: | ---: | ---: | ---: |
| cholesky_f32_512x512 | pytorch-cuda | 4 | 3 | 0.275 | 0.25 |
| cholesky_f32_512x512 | tenferro-cuda-direct | 4 | 3 | 0.285 | 0.25 |
| householder_r_default_f32_512x256 | pytorch-cuda | — | — | — | state projections need an extra uploaded operand; excluded from standard timing |
| householder_r_default_f32_512x256 | tenferro-cuda-direct | 4 | 3 | 0.139 | 0.25 |

## Run metadata

```yaml
target_profile: nvidia-gpu
suite_id: gpu/linalg_public_api_throughput
suite_file: benchmarks/gpu/linalg_public_api_throughput.yaml
timestamp: '2026-11-02T21:09:35Z'
host:
  cpu: AMD EPYC 7713P 64-Core Processor
environment:
  hostname: fixture-host
cuda:
  device_name: NVIDIA A100 80GB PCIe
tenferro_rs:
  commit: 433121a7756fdb4f109b02f0422d3714a04514c6
```
