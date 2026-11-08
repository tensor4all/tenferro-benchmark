# gpu/isolated_ops: CUDA public API benchmark

Measured cells: **94**. Unsupported cells: **5**, including **3** numerically verified but timing-excluded cells. Unsupported cells do not count as measured coverage.

Logical traffic: this run measured a plain device copy of 67108864 bytes at **1417.7 GB/s** on this device. Every record carries the logical bytes the case must move per operation and the ratio of the case's implied rate to that copy rate, in the sampling table below. Logical bytes are the declared array sizes, not device DRAM traffic: a case whose operands stay in cache moves less DRAM than its arrays suggest, so a ratio above 1 is recorded, not rejected.

Raw run: `data/results/nvidia-gpu/gpu/isolated_ops/20261108_090627/`.

Isolated public APIs and explicit unsupported timing contracts. Issue 155.

This run did not compile: no NVRTC module was built in any of its arms. That describes this run only, and does not show that the CUDA artifact cache was warm.

Values are median (IQR) **µs per operation**. Samples time batches of declared operations plus device synchronization; retained output memory is limited to 256 MiB. Batch counts and raw durations are preserved in JSONL.

Cases marked † (**7** of 33) were sampled with a different number of operations per sample on different arms (see Ops/sample in the sampling table). Each value is still per operation, but the arms amortise launch and synchronization over different batch sizes, so a ratio between their columns is approximate and is not presented as a like-for-like comparison.

Setup, input transfers, library/plan initialization, JIT compilation, validation downloads and output destruction are outside timing. Intrinsic output allocation is inside timing; output-reuse rows instead use preallocated destinations across samples. Accumulating destinations are restored from prepared GPU seeds and synchronized before every batch clock, including initialization and calibration. Direct and eager each use one entered session across samples; direct FFT uses cached `FftExecutor`. JAX eager and compiled JIT are separate arms.

All measured output elements are checked against independently generated NumPy/SciPy reference results before and after sampling. Unsupported rows include unavailable APIs and APIs whose internal setup cannot be excluded; these are not performance measurements. Short batches limited by output memory remain identifiable in the sampling table.

| Case | tenferro-cuda-eager | pytorch-cuda | jax-cuda |
| --- | ---: | ---: | ---: |
| add_f64_n1048576 | 88.514 (0.200) | 12.128 (0.179) | 38.998 (2.060) |
| sub_f64_n1048576 | 88.587 (0.418) | 12.289 (0.139) | 40.872 (2.421) |
| mul_f64_n1048576 | 89.593 (0.649) | 12.162 (0.172) | 39.261 (4.688) |
| div_f64_n1048576 | 89.801 (0.883) | 13.462 (0.133) | 32.418 (7.867) |
| rem_f64_n1048576 | 120.978 (0.806) | 12.428 (0.131) | 27.441 (2.625) |
| pow_f64_n1048576 | 122.249 (0.625) | 29.230 (0.083) | 35.982 (1.407) |
| maximum_f64_n1048576 | 108.696 (0.759) | 12.672 (0.173) | 37.810 (3.446) |
| minimum_f64_n1048576 | 108.635 (0.967) | 12.459 (0.201) | 38.254 (2.648) |
| neg_f64_n1048576 | 52.806 (0.206) | 7.722 (0.054) | 34.683 (1.882) |
| abs_f64_n1048576 | 52.732 (0.213) | 7.798 (0.120) | 32.115 (1.653) |
| sign_f64_n1048576 | 52.746 (0.190) | 7.714 (0.035) | 30.897 (1.663) |
| conj_c64_n1048576 | 85.547 (0.386) | 25.359 (0.257) | 34.285 (1.943) |
| exp_f64_n1048576 | 53.251 (0.291) | 8.110 (0.061) | 28.385 (2.117) |
| expm1_f64_n1048576 | 54.471 (0.526) | 9.809 (0.120) | 27.488 (1.862) |
| log_f64_n1048576 | 57.348 (0.218) | 12.513 (0.150) | 26.883 (2.996) |
| log1p_f64_n1048576 | 57.361 (0.246) | 13.097 (0.051) | 28.500 (2.209) |
| erf_f64_n1048576 | 59.188 (0.455) | 14.324 (0.139) | 44.007 (2.280) |
| sin_f64_n1048576 | 56.210 (0.333) | 11.120 (0.058) | 26.671 (3.693) |
| cos_f64_n1048576 | 56.330 (0.325) | 11.140 (0.066) | 28.522 (3.098) |
| tanh_f64_n1048576 | 53.366 (0.243) | 8.249 (0.069) | 28.693 (2.657) |
| sqrt_f64_n1048576 | 52.806 (0.136) | 7.820 (0.105) | 26.445 (1.711) |
| rsqrt_f64_n1048576 | 52.780 (0.182) | 7.867 (0.100) | 43.148 (2.506) |
| compare_f64_n1048576 † | 81.271 (0.225) | 8.861 (0.055) | 28.673 (0.964) |
| where_select_f64_n1048576 | 114.504 (0.281) | 14.332 (0.135) | 31.588 (5.417) |
| clamp_f64_n1048576 † | 150.205 (0.947) | 10.243 (0.056) | 29.811 (2.611) |
| reduce_sum_f64_n1048576 † | 69.136 (0.572) | 7.004 (0.014) | 30.084 (1.020) |
| reduce_prod_f64_n1048576 † | 68.650 (0.552) | 7.110 (0.008) | 28.876 (0.872) |
| reduce_max_f64_n1048576 † | 68.665 (0.643) | 7.184 (0.008) | 30.786 (1.413) |
| reduce_min_f64_n1048576 † | 68.857 (0.669) | 7.160 (0.010) | 29.419 (0.759) |
| reduce_sum_squares_f64_n1048576 | 68.842 (0.303) | unsupported | unsupported |
| reduce_mean_f64_n1048576 † | unsupported | 6.953 (0.008) | 32.389 (1.755) |
| scale_real_f64_n1048576 | unsupported | 9.489 (0.068) | 31.964 (6.333) |
| scale_complex_c64_n1048576 | unsupported | 25.751 (0.295) | 36.983 (1.951) |

## Sampling

| Case | Backend | Ops/sample | Samples | Median batch ms | Bytes/op | Copy ratio | Verification error/tolerance |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| abs_f64_n1048576 | jax-cuda | 32 | 15 | 1.028 | 16777216 | 0.368 | 0 |
| abs_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.250 | 16777216 | 1.518 | 0 |
| abs_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.687 | 16777216 | 0.224 | 0 |
| add_f64_n1048576 | jax-cuda | 32 | 15 | 1.248 | 25165824 | 0.455 | 0 |
| add_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.388 | 25165824 | 1.464 | 0 |
| add_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.832 | 25165824 | 0.201 | 0 |
| clamp_f64_n1048576 | jax-cuda | 32 | 15 | 0.954 | 16777232 | 0.397 | 0 |
| clamp_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.328 | 16777232 | 1.155 | 0 |
| clamp_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.403 | 16777232 | 0.079 | 0 |
| compare_f64_n1048576 | jax-cuda | 64 | 15 | 1.835 | 17825792 | 0.439 | 0 |
| compare_f64_n1048576 | pytorch-cuda | 4 | 15 | 0.035 | 17825792 | 1.419 | 0 |
| compare_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.601 | 17825792 | 0.155 | 0 |
| conj_c64_n1048576 | jax-cuda | 16 | 15 | 0.549 | 33554432 | 0.690 | 0 |
| conj_c64_n1048576 | pytorch-cuda | 16 | 15 | 0.406 | 33554432 | 0.933 | 0 |
| conj_c64_n1048576 | tenferro-cuda-eager | 16 | 15 | 1.369 | 33554432 | 0.277 | 0 |
| cos_f64_n1048576 | jax-cuda | 32 | 15 | 0.913 | 16777216 | 0.415 | 5.905e-08 |
| cos_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.356 | 16777216 | 1.062 | 5.905e-08 |
| cos_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.803 | 16777216 | 0.210 | 5.905e-08 |
| div_f64_n1048576 | jax-cuda | 32 | 15 | 1.037 | 25165824 | 0.548 | 0 |
| div_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.431 | 25165824 | 1.319 | 0 |
| div_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.874 | 25165824 | 0.198 | 0 |
| erf_f64_n1048576 | jax-cuda | 32 | 15 | 1.408 | 16777216 | 0.269 | 8.536e-08 |
| erf_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.458 | 16777216 | 0.826 | 8.536e-08 |
| erf_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.894 | 16777216 | 0.200 | 8.536e-08 |
| exp_f64_n1048576 | jax-cuda | 32 | 15 | 0.908 | 16777216 | 0.417 | 1.093e-07 |
| exp_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.260 | 16777216 | 1.459 | 1.093e-07 |
| exp_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.704 | 16777216 | 0.222 | 1.093e-07 |
| expm1_f64_n1048576 | jax-cuda | 32 | 15 | 0.880 | 16777216 | 0.431 | 7.39e-08 |
| expm1_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.314 | 16777216 | 1.206 | 7.39e-08 |
| expm1_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.743 | 16777216 | 0.217 | 7.39e-08 |
| log1p_f64_n1048576 | jax-cuda | 32 | 15 | 0.912 | 16777216 | 0.415 | 1.106e-07 |
| log1p_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.419 | 16777216 | 0.904 | 1.106e-07 |
| log1p_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.836 | 16777216 | 0.206 | 1.106e-07 |
| log_f64_n1048576 | jax-cuda | 32 | 15 | 0.860 | 16777216 | 0.440 | 4.308e-08 |
| log_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.400 | 16777216 | 0.946 | 4.308e-08 |
| log_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.835 | 16777216 | 0.206 | 4.308e-08 |
| maximum_f64_n1048576 | jax-cuda | 32 | 15 | 1.210 | 25165824 | 0.469 | 0 |
| maximum_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.406 | 25165824 | 1.401 | 0 |
| maximum_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.478 | 25165824 | 0.163 | 0 |
| minimum_f64_n1048576 | jax-cuda | 32 | 15 | 1.224 | 25165824 | 0.464 | 0 |
| minimum_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.399 | 25165824 | 1.425 | 0 |
| minimum_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.476 | 25165824 | 0.163 | 0 |
| mul_f64_n1048576 | jax-cuda | 32 | 15 | 1.256 | 25165824 | 0.452 | 0 |
| mul_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.389 | 25165824 | 1.460 | 0 |
| mul_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.867 | 25165824 | 0.198 | 0 |
| neg_f64_n1048576 | jax-cuda | 32 | 15 | 1.110 | 16777216 | 0.341 | 0 |
| neg_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.247 | 16777216 | 1.533 | 0 |
| neg_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.690 | 16777216 | 0.224 | 0 |
| pow_f64_n1048576 | jax-cuda | 32 | 15 | 1.151 | 25165824 | 0.493 | 1.48e-07 |
| pow_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.935 | 25165824 | 0.607 | 1.48e-07 |
| pow_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.912 | 25165824 | 0.145 | 1.48e-07 |
| reduce_max_f64_n1048576 | jax-cuda | 128 | 15 | 3.941 | 8396800 | 0.192 | 0 |
| reduce_max_f64_n1048576 | pytorch-cuda | 512 | 15 | 3.678 | 8396800 | 0.824 | 0 |
| reduce_max_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.197 | 8396800 | 0.086 | 0 |
| reduce_mean_f64_n1048576 | jax-cuda | 64 | 15 | 2.073 | 8396800 | 0.183 | 6.5e-09 |
| reduce_mean_f64_n1048576 | pytorch-cuda | 512 | 15 | 3.560 | 8396800 | 0.852 | 3.9e-09 |
| reduce_mean_f64_n1048576 | tenferro-cuda-eager | — | — | — | 8396800 | unavailable | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| reduce_min_f64_n1048576 | jax-cuda | 64 | 15 | 1.883 | 8396800 | 0.201 | 0 |
| reduce_min_f64_n1048576 | pytorch-cuda | 512 | 15 | 3.666 | 8396800 | 0.827 | 0 |
| reduce_min_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.203 | 8396800 | 0.086 | 0 |
| reduce_prod_f64_n1048576 | jax-cuda | 128 | 15 | 3.696 | 8396800 | 0.205 | 4.436e-06 |
| reduce_prod_f64_n1048576 | pytorch-cuda | 512 | 15 | 3.640 | 8396800 | 0.833 | 4.216e-06 |
| reduce_prod_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.197 | 8396800 | 0.086 | 3.55e-06 |
| reduce_sum_f64_n1048576 | jax-cuda | 128 | 15 | 3.851 | 8396800 | 0.197 | 4.449e-06 |
| reduce_sum_f64_n1048576 | pytorch-cuda | 512 | 15 | 3.586 | 8396800 | 0.846 | 2.961e-06 |
| reduce_sum_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.212 | 8396800 | 0.086 | 3.391e-06 |
| reduce_sum_squares_f64_n1048576 | jax-cuda | — | — | — | 8396800 | unavailable | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_sum_squares_f64_n1048576 | pytorch-cuda | — | — | — | 8396800 | unavailable | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_sum_squares_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.203 | 8396800 | 0.086 | 3.294e-07 |
| rem_f64_n1048576 | jax-cuda | 32 | 15 | 0.878 | 25165824 | 0.647 | 0 |
| rem_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.398 | 25165824 | 1.428 | 0 |
| rem_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.871 | 25165824 | 0.147 | 0 |
| rsqrt_f64_n1048576 | jax-cuda | 32 | 15 | 1.381 | 16777216 | 0.274 | 6.504e-08 |
| rsqrt_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.252 | 16777216 | 1.504 | 6.504e-08 |
| rsqrt_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.689 | 16777216 | 0.224 | 6.504e-08 |
| scale_complex_c64_n1048576 | jax-cuda | 16 | 15 | 0.592 | 33554448 | 0.640 | 4.129e-08 |
| scale_complex_c64_n1048576 | pytorch-cuda | 16 | 15 | 0.412 | 33554448 | 0.919 | 0 |
| scale_complex_c64_n1048576 | tenferro-cuda-eager | — | — | — | 33554448 | unavailable | Eager scale host-placement bug fixed by upstream 917d5341. API now uploads its internal scalar through constant_from_host; no public preuploaded factor parameter. Numerical validation only; internal scalar transfer excludes standard timing. |
| scale_real_f64_n1048576 | jax-cuda | 32 | 15 | 1.023 | 16777224 | 0.370 | 0 |
| scale_real_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.304 | 16777224 | 1.247 | 0 |
| scale_real_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777224 | unavailable | Eager scale host-placement bug fixed by upstream 917d5341. API now uploads its internal scalar through constant_from_host; no public preuploaded factor parameter. Numerical validation only; internal scalar transfer excludes standard timing. |
| sign_f64_n1048576 | jax-cuda | 32 | 15 | 0.989 | 16777216 | 0.383 | 0 |
| sign_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.247 | 16777216 | 1.534 | 0 |
| sign_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.688 | 16777216 | 0.224 | 0 |
| sin_f64_n1048576 | jax-cuda | 32 | 15 | 0.853 | 16777216 | 0.444 | 4.153e-08 |
| sin_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.356 | 16777216 | 1.064 | 4.153e-08 |
| sin_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.799 | 16777216 | 0.211 | 4.153e-08 |
| sqrt_f64_n1048576 | jax-cuda | 32 | 15 | 0.846 | 16777216 | 0.447 | 0 |
| sqrt_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.250 | 16777216 | 1.513 | 0 |
| sqrt_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.690 | 16777216 | 0.224 | 0 |
| sub_f64_n1048576 | jax-cuda | 32 | 15 | 1.308 | 25165824 | 0.434 | 0 |
| sub_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.393 | 25165824 | 1.444 | 0 |
| sub_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.835 | 25165824 | 0.200 | 0 |
| tanh_f64_n1048576 | jax-cuda | 32 | 15 | 0.918 | 16777216 | 0.412 | 4.437e-08 |
| tanh_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.264 | 16777216 | 1.435 | 4.437e-08 |
| tanh_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.708 | 16777216 | 0.222 | 4.437e-08 |
| where_select_f64_n1048576 | jax-cuda | 32 | 15 | 1.011 | 26214400 | 0.585 | 0 |
| where_select_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.459 | 26214400 | 1.290 | 0 |
| where_select_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.664 | 26214400 | 0.161 | 0 |

## Run metadata

```yaml
schema_version: 1
target_profile: nvidia-gpu
suite_id: gpu/isolated_ops
suite_file: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/data/results/nvidia-gpu/gpu/routine_selection/20261108_090627/gpu_isolated_ops.yaml
timestamp: '2026-11-08T09:06:27+00:00'
tenferro_rs:
  path: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/extern/tenferro-rs
  commit: 9a850b06155405acc99ecd0648829e9e6d3866ba
  dirty: false
  features:
  - cuda
  resolved_path: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/extern/tenferro-rs
harness:
  path: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark
  commit: 44b5fe8b1b4b339b68f9e9f7dd56d03f6cb08303
  dirty: false
resolved_deps:
  strided-perm:
  - 0.8.0
  - 0.12.0
  strided-view:
  - 0.7.1
  threadteam:
  - 0.1.0
  - 0.5.1
host:
  cpu: AMD EPYC 9554 64-Core Processor
  logical_cpus: 252
  power:
    smt:
      active: '0'
      control: notsupported
  numa:
    nodes: 2
    cpulists:
      '0': 0-125
      '1': 126-251
  smt_siblings:
  - '0'
  - '1'
  - '2'
  - '3'
  - '4'
  - '5'
  - '6'
  - '7'
  - '8'
  - '9'
  - '10'
  - '11'
  - '12'
  - '13'
  - '14'
  - '15'
  - '16'
  - '17'
  - '18'
  - '19'
  - '20'
  - '21'
  - '22'
  - '23'
  - '24'
  - '25'
  - '26'
  - '27'
  - '28'
  - '29'
  - '30'
  - '31'
  - '32'
  - '33'
  - '34'
  - '35'
  - '36'
  - '37'
  - '38'
  - '39'
  - '40'
  - '41'
  - '42'
  - '43'
  - '44'
  - '45'
  - '46'
  - '47'
  - '48'
  - '49'
  - '50'
  - '51'
  - '52'
  - '53'
  - '54'
  - '55'
  - '56'
  - '57'
  - '58'
  - '59'
  - '60'
  - '61'
  - '62'
  - '63'
  - '64'
  - '65'
  - '66'
  - '67'
  - '68'
  - '69'
  - '70'
  - '71'
  - '72'
  - '73'
  - '74'
  - '75'
  - '76'
  - '77'
  - '78'
  - '79'
  - '80'
  - '81'
  - '82'
  - '83'
  - '84'
  - '85'
  - '86'
  - '87'
  - '88'
  - '89'
  - '90'
  - '91'
  - '92'
  - '93'
  - '94'
  - '95'
  - '96'
  - '97'
  - '98'
  - '99'
  - '100'
  - '101'
  - '102'
  - '103'
  - '104'
  - '105'
  - '106'
  - '107'
  - '108'
  - '109'
  - '110'
  - '111'
  - '112'
  - '113'
  - '114'
  - '115'
  - '116'
  - '117'
  - '118'
  - '119'
  - '120'
  - '121'
  - '122'
  - '123'
  - '124'
  - '125'
  - '126'
  - '127'
  - '128'
  - '129'
  - '130'
  - '131'
  - '132'
  - '133'
  - '134'
  - '135'
  - '136'
  - '137'
  - '138'
  - '139'
  - '140'
  - '141'
  - '142'
  - '143'
  - '144'
  - '145'
  - '146'
  - '147'
  - '148'
  - '149'
  - '150'
  - '151'
  - '152'
  - '153'
  - '154'
  - '155'
  - '156'
  - '157'
  - '158'
  - '159'
  - '160'
  - '161'
  - '162'
  - '163'
  - '164'
  - '165'
  - '166'
  - '167'
  - '168'
  - '169'
  - '170'
  - '171'
  - '172'
  - '173'
  - '174'
  - '175'
  - '176'
  - '177'
  - '178'
  - '179'
  - '180'
  - '181'
  - '182'
  - '183'
  - '184'
  - '185'
  - '186'
  - '187'
  - '188'
  - '189'
  - '190'
  - '191'
  - '192'
  - '193'
  - '194'
  - '195'
  - '196'
  - '197'
  - '198'
  - '199'
  - '200'
  - '201'
  - '202'
  - '203'
  - '204'
  - '205'
  - '206'
  - '207'
  - '208'
  - '209'
  - '210'
  - '211'
  - '212'
  - '213'
  - '214'
  - '215'
  - '216'
  - '217'
  - '218'
  - '219'
  - '220'
  - '221'
  - '222'
  - '223'
  - '224'
  - '225'
  - '226'
  - '227'
  - '228'
  - '229'
  - '230'
  - '231'
  - '232'
  - '233'
  - '234'
  - '235'
  - '236'
  - '237'
  - '238'
  - '239'
  - '240'
  - '241'
  - '242'
  - '243'
  - '244'
  - '245'
  - '246'
  - '247'
  - '248'
  - '249'
  - '250'
  - '251'
  l3_domains:
  - 0-125
  - 126-251
build:
  toolchain:
    rustc: null
    rustc_release: null
    rustflags: null
    target_cpu: null
environment:
  container: devcontainer
  hostname: 053e3b59f698
  os: Linux-6.8.0-147-generic-x86_64-with-glibc2.39
  arch: x86_64
  cpu: x86_64
  affinity:
    pinning: none
    requested_measured_cpus: null
    observer_cpus: null
    override_of: null
    thread_budget: '1'
    measured_cpus: 0-251
    l3_domains:
    - 0-125
    - 126-251
    cpu_boost: null
    scaling_governor: null
    scaling_driver: null
  env:
    OMP_NUM_THREADS: '1'
    RAYON_NUM_THREADS: '1'
    BENCHMARK_TARGET_PROFILE: nvidia-gpu
    OPENBLAS_NUM_THREADS: '1'
    MATMUL_NUM_THREADS: '1'
    CUDA_HOME: /opt/tenferro-benchmark-ci/cuda
    CUDA_PATH: /opt/tenferro-benchmark-ci/cuda
    TENFERRO_CUTENSOR_PATH: /opt/tenferro-benchmark-ci/cutensor/lib/libcutensor.so.2
    GPU_RUNTIME_SHA256: ecc740128a444e2b7d8a80ea018e5fbeb2da956dc2669f05886d8d7c97b49c67
    UV_PROJECT_ENVIRONMENT: /opt/tenferro-benchmark-ci/venv
    BENCH_RUNS: '15'
    BENCH_WARMUPS: '3'
timing_policy:
  version: 2
  scope: operation execution and completion synchronization
  setup: fixtures, H2D, runtime/session/handles, plans and compilation outside every
    timer
  cleanup: returned outputs retained until after timer stops
  minimum_untimed_priming_runs: 1
  short_operation_sampling: adaptive batches target 2 ms, at most 4096 operations
    and 256 MiB retained output; output-reuse destinations prepared before timing
  trace_execution: 'tenferro trace excluded: this suite compares immediate per-op
    calls, while trace executes a prepared graph in one caller-held session (run_prepared_in);
    JAX JIT compilation outside timing'
  setup_diagnostics: false
blas:
  implementation: none
python_backends:
  pytorch:
    available: true
    version: 2.11.0+cu128
    file: /opt/tenferro-benchmark-ci/venv/lib/python3.12/site-packages/torch/__init__.py
    provider: mkl
    blas_info: mkl
    lapack_info: mkl
    config: "PyTorch built with:\n  - GCC 13.3\n  - C++ Version: 201703\n  - Intel(R)\
      \ oneAPI Math Kernel Library Version 2024.2-Product Build 20240605 for Intel(R)\
      \ 64 architecture applications\n  - Intel(R) MKL-DNN v3.10.2 (Git Hash f1d471933dc852f956fd05389f9313c7148783d5)\n\
      \  - OpenMP 201511 (a.k.a. OpenMP 4.5)\n  - LAPACK is enabled (usually provided\
      \ by MKL)\n  - NNPACK is enabled\n  - CPU capability usage: AVX512\n  - CUDA\
      \ Runtime 12.8\n  - NVCC architecture flags: -gencode;arch=compute_75,code=sm_75;-gencode;arch=compute_80,code=sm_80;-gencode;arch=compute_86,code=sm_86;-gencode;arch=compute_90,code=sm_90;-gencode;arch=compute_100,code=sm_100;-gencode;arch=compute_120,code=sm_120\n\
      \  - CuDNN 91.9  (built against CUDA 12.9)\n  - Magma 2.6.1\n  - Build settings:\
      \ BLAS_INFO=mkl, BUILD_TYPE=Release, COMMIT_SHA=70d99e998b4955e0049d13a98d77ae1b14db1f45,\
      \ CUDA_VERSION=12.8, CUDNN_VERSION=9.19.0, CXX_COMPILER=/opt/rh/gcc-toolset-13/root/usr/bin/c++,\
      \ CXX_FLAGS= -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -DNDEBUG -DUSE_KINETO\
      \ -DLIBKINETO_NOROCTRACER -DLIBKINETO_NOXPUPTI=ON -DUSE_FBGEMM -DUSE_MSLK -DUSE_PYTORCH_QNNPACK\
      \ -DUSE_XNNPACK -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -O2 -fPIC -DC10_NODEPRECATED\
      \ -Wall -Wextra -Werror=return-type -Werror=non-virtual-dtor -Werror=range-loop-construct\
      \ -Werror=bool-operation -Wnarrowing -Wno-missing-field-initializers -Wno-unknown-pragmas\
      \ -Wno-unused-parameter -Wno-strict-overflow -Wno-strict-aliasing -Wno-stringop-overflow\
      \ -Wsuggest-override -Wno-psabi -Wno-error=old-style-cast -faligned-new -Wno-maybe-uninitialized\
      \ -fno-math-errno -fno-trapping-math -Werror=format -Wno-dangling-reference\
      \ -Wno-error=dangling-reference -Wno-stringop-overflow, LAPACK_INFO=mkl, PERF_WITH_AVX=1,\
      \ PERF_WITH_AVX2=1, TORCH_VERSION=2.11.0, USE_CUDA=ON, USE_CUDNN=ON, USE_CUSPARSELT=1,\
      \ USE_GFLAGS=OFF, USE_GLOG=OFF, USE_GLOO=ON, USE_MKL=ON, USE_MKLDNN=ON, USE_MPI=OFF,\
      \ USE_NCCL=1, USE_NNPACK=ON, USE_OPENMP=ON, USE_ROCM=OFF, USE_ROCM_KERNEL_ASSERT=OFF,\
      \ USE_XCCL=OFF, USE_XPU=OFF, \n"
    library: /opt/tenferro-benchmark-ci/venv/lib/python3.12/site-packages/torch/lib/libtorch_cpu.so
    linked_libraries:
    - /opt/tenferro-benchmark-ci/venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1
  jax:
    available: true
    version: 0.10.1
    jaxlib_version: 0.10.1
    backend: gpu
    file: /opt/tenferro-benchmark-ci/venv/lib/python3.12/site-packages/jaxlib/__init__.py
    provider: xla_gpu
    dot_backend: xla_gpu
    lapack_provider: none_detected
    library: /opt/tenferro-benchmark-ci/venv/lib/python3.12/site-packages/jaxlib/_ifrt_proxy.so
    linked_libraries: []
julia:
  available: false
  reason: julia not found on PATH
cuda:
  device_ordinal: 0
  device_name: NVIDIA A100 80GB PCIe
  device_uuid: GPU-12b45076-79d5-7b84-cb5c-7c6300074c8a
  device_memory_bytes: 85899345920
  driver_version: 595.91.07
  runtime_version: '12.8'
  cuda_version: '13.2'
  toolkit_version: null
  toolkit_root: /opt/tenferro-benchmark-ci/cuda
  cutensor_version: 2.2.0
  cusolver_version: 11.7.3
  cudnn_version: '91900'
  artifact_cache:
    path: /root/.cache/tenferro-benchmark/cuda-artifacts/12.8.93/sm_80
    regime: warm
gpu_memory_calibration:
  kind: device_copy
  method: torch.Tensor.copy_ between two contiguous device buffers, one copy per sample,
    device synchronized before and after the clock
  device_ordinal: 0
  device_name: NVIDIA A100 80GB PCIe
  buffer_bytes: 67108864
  traffic_bytes_per_copy: 134217728
  traffic_model: one read plus one write of buffer_bytes per copy
  warmups: 3
  sample_ms:
  - 0.09352
  - 0.09323
  - 0.097937
  - 0.094453
  - 0.092049
  - 0.098227
  - 0.094233
  - 0.092028
  - 0.097346
  - 0.095104
  - 0.094672
  - 0.095884
  - 0.095895
  - 0.095263
  - 0.094512
  median_ms: 0.094672
  bytes_per_second: 1417713030251.8167
  torch_version: 2.11.0+cu128
  cuda_version: '12.8'
  timestamp: '2026-11-08T09:18:06.967676+00:00'
gpu_assignment:
  role: benchmark
  physical_device_index: 0
  device_uuid: GPU-12b45076-79d5-7b84-cb5c-7c6300074c8a
  cuda_visible_devices: GPU-12b45076-79d5-7b84-cb5c-7c6300074c8a
  process_device_ordinal: 0
  verification_only: false
collection:
  command: BENCHMARK_TARGET_PROFILE=nvidia-gpu GPU_BENCH_DEVICE=0 BENCH_RUNS=15 BENCH_WARMUPS=3
    ./scripts/run_gpu_public_api.sh isolated_ops
  devcontainer_config: .devcontainer/cuda/devcontainer.json
  expected_cases: 33
  selected_cases: 33
  sha256:
    .devcontainer/cuda/devcontainer.json: 0bf1465bbd1a4c9f896d5320838f2d5a488632a59af7f9b59cc050e95e0b8e09
    .devcontainer/cuda/Dockerfile: 578d9815e5145aa1651ac7e93833468c345ee8bf5e95bfa7717551580ae9feca
    src/bin/benchmark_gpu_public_api.rs: 28b921283d595842fc599670df7fadfa2b226573a287cb9fd90799300ca81e6c
    src/bin/benchmark_gpu_public_api/lifetime.rs: 34be366f56aac88f5f3b86ade8018a15dad2bca85cea74f56cb2fe121917d0a9
    src/bin/benchmark_gpu_public_api/linalg.rs: 9c3d7916a3281d2dfd1caa68551dc9c81c79dab6b7525324339e5c4816581e6d
    src/bin/benchmark_gpu_public_api/multi.rs: 949eadd6272c17ba4e380a228547a03608d328a5ef852eb95294731c288f24d9
    src/bin/benchmark_gpu_public_api/plan.rs: 70fce8a6e9ed7518779a51a9d8692e010fa1e470e95eee0ce5276f595737fde8
    src/bin/benchmark_gpu_public_api/read_layout.rs: fc42e2160094ba5f9c7dd6facb5667a76e10b73f971ef9facfb1f64723e42467
    src/bin/benchmark_gpu_public_api/reuse.rs: a5a4aa0edb0a1158f6669b642f4bc25a7079314fc2a711ef3ef57cc19c0a35b2
    src/bin/benchmark_gpu_public_api/runtime_ops.rs: c9d61b1e99bda517a59a6e34f625fd4d651bbbb19f6dc06ba4d0d1c0d50dfd6a
    src/bin/benchmark_gpu_public_api/state.rs: a8bbf8985bbc58c435e3c4dda7135161a141a90a10fa50d4f95128397ce949cb
    src/bin/benchmark_gpu_public_api/structural.rs: 8b61ac01b58edc090eafbc124ec135d5efeb4979631a939656ba78dd6e5c042a
    /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/scripts/benchmark_gpu_public_api.py: 4f94c67a046ec57f95823fd01f3f73a710d97745fe6a58a3c6106dfec870d65c
    /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/scripts/gpu_public_api_ops.py: e88823db3f4a1fe43dc13c4743c9d78ece6af76a968724091af318dd8d7f2528
    scripts/run_gpu_public_api.sh: 788b9fde4f8ca4162921baef1203b2e31bb747ec30327a425de3442cf872df52
    data/results/nvidia-gpu/gpu/isolated_ops/20261108_090627/fixtures/manifest.json: 3c1a75f9c57996b228c04da1d4cbf34e4137a074e44ddca9f1c37e5824580bd8
    target/release/benchmark_gpu_public_api: 62f92e0c72d6fb46531ee8106f112567dfd32ddb482255a7cc41ba6819929786
  reference_numpy_version: 2.4.2
  measurement_cpu:
    policy: 'disabled (GPU_BENCH_CPU_SELECT unset): arms run on the allocation unpinned'
    arms:
      jax-cuda:
        arm: jax-cuda
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-08T09:19:18Z'
      pytorch-cuda:
        arm: pytorch-cuda
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-08T09:18:51Z'
      tenferro-cuda-eager:
        arm: tenferro-cuda-eager
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-08T09:18:23Z'
  compile:
    case_wall_seconds: 54.729
    compiling_processes: 0
    modules_compiled: 0
    compile_time_ms: null
    compile_time_scope: 'not measured: no compile interval was recorded; `compile_modules`
      counts the NVRTC modules and the untimed wall is the case wall'
    untimed_wall_seconds: 0.0
```

## Untimed work over the recorded budget

Each row records the wall clock it spent outside its timed region. These cases spent more than the recorded budget ratio (0.2) there. The threshold has no measured basis, so this is a recorded warning: it does not change a cell, its acceptance or the exit status of the run.

| Case | Backend | Warning |
| --- | --- | --- |
| add_f64_n1048576 | tenferro-cuda-eager | add_f64_n1048576 untimed 149.7 ms is 352% of the 42.5 ms timed region (budget 20%) |
| sub_f64_n1048576 | tenferro-cuda-eager | sub_f64_n1048576 untimed 70.1 ms is 164% of the 42.6 ms timed region (budget 20%) |
| mul_f64_n1048576 | tenferro-cuda-eager | mul_f64_n1048576 untimed 85.2 ms is 197% of the 43.2 ms timed region (budget 20%) |
| div_f64_n1048576 | tenferro-cuda-eager | div_f64_n1048576 untimed 86.1 ms is 200% of the 43.1 ms timed region (budget 20%) |
| rem_f64_n1048576 | tenferro-cuda-eager | rem_f64_n1048576 untimed 87.4 ms is 150% of the 58.1 ms timed region (budget 20%) |
| pow_f64_n1048576 | tenferro-cuda-eager | pow_f64_n1048576 untimed 96.7 ms is 165% of the 58.6 ms timed region (budget 20%) |
| maximum_f64_n1048576 | tenferro-cuda-eager | maximum_f64_n1048576 untimed 85.2 ms is 163% of the 52.1 ms timed region (budget 20%) |
| minimum_f64_n1048576 | tenferro-cuda-eager | minimum_f64_n1048576 untimed 85.1 ms is 163% of the 52.1 ms timed region (budget 20%) |
| neg_f64_n1048576 | tenferro-cuda-eager | neg_f64_n1048576 untimed 74.3 ms is 293% of the 25.4 ms timed region (budget 20%) |
| abs_f64_n1048576 | tenferro-cuda-eager | abs_f64_n1048576 untimed 62.4 ms is 246% of the 25.3 ms timed region (budget 20%) |
| sign_f64_n1048576 | tenferro-cuda-eager | sign_f64_n1048576 untimed 62.2 ms is 245% of the 25.3 ms timed region (budget 20%) |
| conj_c64_n1048576 | tenferro-cuda-eager | conj_c64_n1048576 untimed 94.6 ms is 460% of the 20.6 ms timed region (budget 20%) |
| exp_f64_n1048576 | tenferro-cuda-eager | exp_f64_n1048576 untimed 74.6 ms is 292% of the 25.6 ms timed region (budget 20%) |
| expm1_f64_n1048576 | tenferro-cuda-eager | expm1_f64_n1048576 untimed 62.8 ms is 240% of the 26.2 ms timed region (budget 20%) |
| log_f64_n1048576 | tenferro-cuda-eager | log_f64_n1048576 untimed 63.3 ms is 230% of the 27.5 ms timed region (budget 20%) |
| log1p_f64_n1048576 | tenferro-cuda-eager | log1p_f64_n1048576 untimed 65.5 ms is 238% of the 27.5 ms timed region (budget 20%) |
| erf_f64_n1048576 | tenferro-cuda-eager | erf_f64_n1048576 untimed 68.5 ms is 242% of the 28.4 ms timed region (budget 20%) |
| sin_f64_n1048576 | tenferro-cuda-eager | sin_f64_n1048576 untimed 62.6 ms is 232% of the 27.0 ms timed region (budget 20%) |
| cos_f64_n1048576 | tenferro-cuda-eager | cos_f64_n1048576 untimed 62.9 ms is 233% of the 27.0 ms timed region (budget 20%) |
| tanh_f64_n1048576 | tenferro-cuda-eager | tanh_f64_n1048576 untimed 64.4 ms is 251% of the 25.7 ms timed region (budget 20%) |
| sqrt_f64_n1048576 | tenferro-cuda-eager | sqrt_f64_n1048576 untimed 62.2 ms is 245% of the 25.4 ms timed region (budget 20%) |
| rsqrt_f64_n1048576 | tenferro-cuda-eager | rsqrt_f64_n1048576 untimed 65.9 ms is 259% of the 25.4 ms timed region (budget 20%) |
| compare_f64_n1048576 | tenferro-cuda-eager | compare_f64_n1048576 untimed 47.4 ms is 121% of the 39.0 ms timed region (budget 20%) |
| where_select_f64_n1048576 | tenferro-cuda-eager | where_select_f64_n1048576 untimed 84.6 ms is 154% of the 55.0 ms timed region (budget 20%) |
| clamp_f64_n1048576 | tenferro-cuda-eager | clamp_f64_n1048576 untimed 75.6 ms is 209% of the 36.2 ms timed region (budget 20%) |
| reduce_sum_f64_n1048576 | tenferro-cuda-eager | reduce_sum_f64_n1048576 untimed 8.3 ms is 25% of the 33.3 ms timed region (budget 20%) |
| reduce_prod_f64_n1048576 | tenferro-cuda-eager | reduce_prod_f64_n1048576 untimed 7.1 ms is 22% of the 32.9 ms timed region (budget 20%) |
| reduce_max_f64_n1048576 | tenferro-cuda-eager | reduce_max_f64_n1048576 untimed 7.0 ms is 21% of the 32.9 ms timed region (budget 20%) |
| reduce_min_f64_n1048576 | tenferro-cuda-eager | reduce_min_f64_n1048576 untimed 7.1 ms is 22% of the 33.0 ms timed region (budget 20%) |
| reduce_sum_squares_f64_n1048576 | tenferro-cuda-eager | reduce_sum_squares_f64_n1048576 untimed 7.3 ms is 22% of the 33.0 ms timed region (budget 20%) |
| add_f64_n1048576 | pytorch-cuda | add_f64_n1048576 untimed 1012.3 ms is 17406% of the 5.8 ms timed region (budget 20%) |
| sub_f64_n1048576 | pytorch-cuda | sub_f64_n1048576 untimed 930.8 ms is 15834% of the 5.9 ms timed region (budget 20%) |
| mul_f64_n1048576 | pytorch-cuda | mul_f64_n1048576 untimed 912.6 ms is 15534% of the 5.9 ms timed region (budget 20%) |
| div_f64_n1048576 | pytorch-cuda | div_f64_n1048576 untimed 958.0 ms is 14814% of the 6.5 ms timed region (budget 20%) |
| rem_f64_n1048576 | pytorch-cuda | rem_f64_n1048576 untimed 920.3 ms is 15435% of the 6.0 ms timed region (budget 20%) |
| pow_f64_n1048576 | pytorch-cuda | pow_f64_n1048576 untimed 944.5 ms is 6724% of the 14.0 ms timed region (budget 20%) |
| maximum_f64_n1048576 | pytorch-cuda | maximum_f64_n1048576 untimed 978.2 ms is 16011% of the 6.1 ms timed region (budget 20%) |
| minimum_f64_n1048576 | pytorch-cuda | minimum_f64_n1048576 untimed 907.3 ms is 15153% of the 6.0 ms timed region (budget 20%) |
| neg_f64_n1048576 | pytorch-cuda | neg_f64_n1048576 untimed 942.0 ms is 25347% of the 3.7 ms timed region (budget 20%) |
| abs_f64_n1048576 | pytorch-cuda | abs_f64_n1048576 untimed 1083.8 ms is 28912% of the 3.7 ms timed region (budget 20%) |
| sign_f64_n1048576 | pytorch-cuda | sign_f64_n1048576 untimed 917.5 ms is 24741% of the 3.7 ms timed region (budget 20%) |
| conj_c64_n1048576 | pytorch-cuda | conj_c64_n1048576 untimed 1031.7 ms is 16993% of the 6.1 ms timed region (budget 20%) |
| exp_f64_n1048576 | pytorch-cuda | exp_f64_n1048576 untimed 924.6 ms is 23709% of the 3.9 ms timed region (budget 20%) |
| expm1_f64_n1048576 | pytorch-cuda | expm1_f64_n1048576 untimed 914.2 ms is 19355% of the 4.7 ms timed region (budget 20%) |
| log_f64_n1048576 | pytorch-cuda | log_f64_n1048576 untimed 932.2 ms is 15442% of the 6.0 ms timed region (budget 20%) |
| log1p_f64_n1048576 | pytorch-cuda | log1p_f64_n1048576 untimed 893.7 ms is 14212% of the 6.3 ms timed region (budget 20%) |
| erf_f64_n1048576 | pytorch-cuda | erf_f64_n1048576 untimed 908.4 ms is 13193% of the 6.9 ms timed region (budget 20%) |
| sin_f64_n1048576 | pytorch-cuda | sin_f64_n1048576 untimed 1101.7 ms is 20649% of the 5.3 ms timed region (budget 20%) |
| cos_f64_n1048576 | pytorch-cuda | cos_f64_n1048576 untimed 905.0 ms is 16910% of the 5.4 ms timed region (budget 20%) |
| tanh_f64_n1048576 | pytorch-cuda | tanh_f64_n1048576 untimed 939.7 ms is 23688% of the 4.0 ms timed region (budget 20%) |
| sqrt_f64_n1048576 | pytorch-cuda | sqrt_f64_n1048576 untimed 948.4 ms is 25199% of the 3.8 ms timed region (budget 20%) |
| rsqrt_f64_n1048576 | pytorch-cuda | rsqrt_f64_n1048576 untimed 886.5 ms is 23434% of the 3.8 ms timed region (budget 20%) |
| compare_f64_n1048576 | pytorch-cuda | compare_f64_n1048576 untimed 896.9 ms is 168131% of the 0.5 ms timed region (budget 20%) |
| where_select_f64_n1048576 | pytorch-cuda | where_select_f64_n1048576 untimed 904.2 ms is 13153% of the 6.9 ms timed region (budget 20%) |
| clamp_f64_n1048576 | pytorch-cuda | clamp_f64_n1048576 untimed 891.9 ms is 18122% of the 4.9 ms timed region (budget 20%) |
| reduce_sum_f64_n1048576 | pytorch-cuda | reduce_sum_f64_n1048576 untimed 33.3 ms is 62% of the 53.8 ms timed region (budget 20%) |
| reduce_prod_f64_n1048576 | pytorch-cuda | reduce_prod_f64_n1048576 untimed 24.1 ms is 44% of the 54.6 ms timed region (budget 20%) |
| reduce_max_f64_n1048576 | pytorch-cuda | reduce_max_f64_n1048576 untimed 27.7 ms is 50% of the 55.2 ms timed region (budget 20%) |
| reduce_min_f64_n1048576 | pytorch-cuda | reduce_min_f64_n1048576 untimed 31.9 ms is 58% of the 55.0 ms timed region (budget 20%) |
| reduce_mean_f64_n1048576 | pytorch-cuda | reduce_mean_f64_n1048576 untimed 27.2 ms is 51% of the 53.4 ms timed region (budget 20%) |
| scale_real_f64_n1048576 | pytorch-cuda | scale_real_f64_n1048576 untimed 971.2 ms is 21285% of the 4.6 ms timed region (budget 20%) |
| scale_complex_c64_n1048576 | pytorch-cuda | scale_complex_c64_n1048576 untimed 961.9 ms is 15502% of the 6.2 ms timed region (budget 20%) |
| add_f64_n1048576 | jax-cuda | add_f64_n1048576 untimed 1108.3 ms is 5973% of the 18.6 ms timed region (budget 20%) |
| sub_f64_n1048576 | jax-cuda | sub_f64_n1048576 untimed 902.1 ms is 4663% of the 19.3 ms timed region (budget 20%) |
| mul_f64_n1048576 | jax-cuda | mul_f64_n1048576 untimed 882.1 ms is 4683% of the 18.8 ms timed region (budget 20%) |
| div_f64_n1048576 | jax-cuda | div_f64_n1048576 untimed 890.1 ms is 5493% of the 16.2 ms timed region (budget 20%) |
| rem_f64_n1048576 | jax-cuda | rem_f64_n1048576 untimed 943.8 ms is 6975% of the 13.5 ms timed region (budget 20%) |
| pow_f64_n1048576 | jax-cuda | pow_f64_n1048576 untimed 893.3 ms is 5159% of the 17.3 ms timed region (budget 20%) |
| maximum_f64_n1048576 | jax-cuda | maximum_f64_n1048576 untimed 885.9 ms is 4886% of the 18.1 ms timed region (budget 20%) |
| minimum_f64_n1048576 | jax-cuda | minimum_f64_n1048576 untimed 872.6 ms is 4708% of the 18.5 ms timed region (budget 20%) |
| neg_f64_n1048576 | jax-cuda | neg_f64_n1048576 untimed 890.2 ms is 5313% of the 16.8 ms timed region (budget 20%) |
| abs_f64_n1048576 | jax-cuda | abs_f64_n1048576 untimed 881.7 ms is 5649% of the 15.6 ms timed region (budget 20%) |
| sign_f64_n1048576 | jax-cuda | sign_f64_n1048576 untimed 888.9 ms is 5891% of the 15.1 ms timed region (budget 20%) |
| conj_c64_n1048576 | jax-cuda | conj_c64_n1048576 untimed 919.6 ms is 11039% of the 8.3 ms timed region (budget 20%) |
| exp_f64_n1048576 | jax-cuda | exp_f64_n1048576 untimed 891.7 ms is 6539% of the 13.6 ms timed region (budget 20%) |
| expm1_f64_n1048576 | jax-cuda | expm1_f64_n1048576 untimed 894.7 ms is 6706% of the 13.3 ms timed region (budget 20%) |
| log_f64_n1048576 | jax-cuda | log_f64_n1048576 untimed 908.5 ms is 6960% of the 13.1 ms timed region (budget 20%) |
| log1p_f64_n1048576 | jax-cuda | log1p_f64_n1048576 untimed 902.6 ms is 6690% of the 13.5 ms timed region (budget 20%) |
| erf_f64_n1048576 | jax-cuda | erf_f64_n1048576 untimed 899.6 ms is 4274% of the 21.1 ms timed region (budget 20%) |
| sin_f64_n1048576 | jax-cuda | sin_f64_n1048576 untimed 908.3 ms is 7010% of the 13.0 ms timed region (budget 20%) |
| cos_f64_n1048576 | jax-cuda | cos_f64_n1048576 untimed 904.8 ms is 6667% of the 13.6 ms timed region (budget 20%) |
| tanh_f64_n1048576 | jax-cuda | tanh_f64_n1048576 untimed 887.5 ms is 6386% of the 13.9 ms timed region (budget 20%) |
| sqrt_f64_n1048576 | jax-cuda | sqrt_f64_n1048576 untimed 884.7 ms is 6851% of the 12.9 ms timed region (budget 20%) |
| rsqrt_f64_n1048576 | jax-cuda | rsqrt_f64_n1048576 untimed 894.3 ms is 4295% of the 20.8 ms timed region (budget 20%) |
| compare_f64_n1048576 | jax-cuda | compare_f64_n1048576 untimed 895.2 ms is 3275% of the 27.3 ms timed region (budget 20%) |
| where_select_f64_n1048576 | jax-cuda | where_select_f64_n1048576 untimed 895.5 ms is 6023% of the 14.9 ms timed region (budget 20%) |
| clamp_f64_n1048576 | jax-cuda | clamp_f64_n1048576 untimed 894.0 ms is 6220% of the 14.4 ms timed region (budget 20%) |
| reduce_sum_f64_n1048576 | jax-cuda | reduce_sum_f64_n1048576 untimed 75.3 ms is 130% of the 58.0 ms timed region (budget 20%) |
| reduce_prod_f64_n1048576 | jax-cuda | reduce_prod_f64_n1048576 untimed 170.9 ms is 307% of the 55.7 ms timed region (budget 20%) |
| reduce_max_f64_n1048576 | jax-cuda | reduce_max_f64_n1048576 untimed 81.6 ms is 139% of the 58.8 ms timed region (budget 20%) |
| reduce_min_f64_n1048576 | jax-cuda | reduce_min_f64_n1048576 untimed 74.4 ms is 265% of the 28.1 ms timed region (budget 20%) |
| reduce_mean_f64_n1048576 | jax-cuda | reduce_mean_f64_n1048576 untimed 66.5 ms is 214% of the 31.0 ms timed region (budget 20%) |
| scale_real_f64_n1048576 | jax-cuda | scale_real_f64_n1048576 untimed 880.4 ms is 5728% of the 15.4 ms timed region (budget 20%) |
| scale_complex_c64_n1048576 | jax-cuda | scale_complex_c64_n1048576 untimed 925.8 ms is 10343% of the 9.0 ms timed region (budget 20%) |