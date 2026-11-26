# gpu/linalg_public_api_throughput: CUDA public API benchmark

Measured cells: **58**. Unsupported cells: **6**, including **0** numerically verified but timing-excluded cells. Unsupported cells do not count as measured coverage.

Logical traffic: this run measured a plain device copy of 67108864 bytes at **1449.0 GB/s** on this device. Every record carries the logical bytes the case must move per operation and the ratio of the case's implied rate to that copy rate, in the sampling table below. Logical bytes are the declared array sizes, not device DRAM traffic: a case whose operands stay in cache moves less DRAM than its arrays suggest, so a ratio above 1 is recorded, not rejected.

Raw run: `data/results/nvidia-gpu/gpu/linalg_public_api_throughput/20261126_062950/`.

Large-shape view of the CUDA public linalg API surface: the 82 cases whose largest matrix dimension is at least 256. `large` is a shape class, not a measured classification: no row is asserted to be compute- or bandwidth-bound, and a memory-bound row takes the same time here as at a small shape (`householder_r_default_f32` measured 34.698 µs at 512x256 and 35.223 µs at 64x32). These rows are never comparable with, or averaged together with, the per-call overhead rows of `gpu/linalg_public_api_latency` (tenferro-rs #2087). Rows here are not lower than on pages published before this harness change for any timing reason: the sample clock starts inside `measure`, so engine construction was never part of a case median. What changed is repeated **untimed setup** work, which was removed: the CUDA engine is opened once per arm and held for the whole run instead of once per case, which shortens the untimed wall clock around the cases and moves no operation median on this page. The same problems, protocol and backends are split by shape class between the two suites. Issue 155.

This run compiled **1 NVRTC modules** in 1 of its arms, because the CUDA artifact cache started cold. The untimed wall of the cases that compiled is **2.5 s** of its 31.3 s of recorded case wall clock, and that wall also holds their fixture read, transfers and verification, so it is not a compile time.

Values are median (IQR) **µs per operation**. Samples time batches of declared operations plus device synchronization; retained output memory is limited to 256 MiB. Batch counts and raw durations are preserved in JSONL.

Cases marked † (**4** of 16) were sampled with a different number of operations per sample on different arms (see Ops/sample in the sampling table). Each value is still per operation, but the arms amortise launch and synchronization over different batch sizes, so a ratio between their columns is approximate and is not presented as a like-for-like comparison.

Setup, input transfers, library/plan initialization, JIT compilation, validation downloads and output destruction are outside timing. Intrinsic output allocation is inside timing; output-reuse rows instead use preallocated destinations across samples. Accumulating destinations are restored from prepared GPU seeds and synchronized before every batch clock, including initialization and calibration. Direct and eager each use one entered session across samples; direct FFT uses cached `FftExecutor`. JAX eager and compiled JIT are separate arms.

All measured output elements are checked against independently generated NumPy/SciPy reference results before and after sampling. Unsupported rows include unavailable APIs and APIs whose internal setup cannot be excluded; these are not performance measurements. Short batches limited by output memory remain identifiable in the sampling table.

| Case | tenferro-cuda-direct | tenferro-cuda-eager | pytorch-cuda | jax-cuda |
| --- | ---: | ---: | ---: | ---: |
| cholesky_f64_512x512 | 331.263 (1.617) | 351.238 (1.761) | 326.317 (1.736) | 305.257 (0.497) |
| svdvals_f64_512x512 | 77823.150 (25.260) | 77840.990 (43.230) | 84678.155 (53.285) | 77795.420 (26.216) |
| eigvalsh_f64_512x512 | 5558.252 (7.620) | 5591.882 (12.880) | 5576.313 (13.120) | 6088.173 (59.770) |
| pinv_f64_512x512 | 78172.341 (51.520) | 78201.441 (19.441) | 84947.616 (23.579) | 78149.921 (38.930) |
| inv_f64_512x512 | 1898.789 (5.435) | 1921.639 (7.548) | 1785.394 (6.495) | 1824.924 (16.205) |
| det_f64_512x512 | 1568.704 (11.475) | 1579.339 (5.758) | 1304.223 (4.825) | 1358.038 (28.743) |
| norm_f64_512x512 † | 17.027 (0.029) | 17.134 (0.525) | 11.082 (0.157) | 107.821 (27.516) |
| triangular_solve_f64_512x512 † | 193.149 (0.259) | 211.199 (0.388) | 307.098 (0.170) | 218.390 (4.268) |
| svd_full_tall_f64_512x256 | 19596.413 (6.140) | 19628.593 (14.945) | 19923.424 (31.555) | 19617.883 (32.750) |
| slogdet_positive_f64_256x256 | 639.634 (4.014) | 668.904 (5.130) | 562.406 (0.333) | 548.244 (2.228) |
| householder_qr_tall_f64_512x256 | 1337.148 (6.337) | 1397.463 (5.978) | 1322.563 (5.247) | 1449.513 (30.960) |
| householder_from_factors_default_f64_512x256 | 1477.298 (8.562) | 1542.843 (13.390) | unsupported | unsupported |
| householder_append_default_f64_512x256 | 791.002 (4.668) | 819.210 (6.190) | unsupported | unsupported |
| householder_r_default_f64_512x256 † | 16.611 (0.128) | 24.256 (0.460) | 5.480 (0.091) | 65.938 (16.241) |
| householder_q_columns_prefix_f64_512x256 † | 158.706 (0.223) | 160.387 (0.402) | 75.262 (2.745) | 152.318 (45.521) |
| lu_tall_f64_s256 | 786.569 (1.211) | 795.777 (1.866) | unsupported | unsupported |

## Sampling

| Case | Backend | Ops/sample | Samples | Median batch ms | Bytes/op | Copy ratio | Verification error/tolerance |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cholesky_f64_512x512 | jax-cuda | 8 | 15 | 2.442 | 4194304 | 0.009 | 4.418e-07 |
| cholesky_f64_512x512 | pytorch-cuda | 8 | 15 | 2.611 | 4194304 | 0.009 | 4.418e-07 |
| cholesky_f64_512x512 | tenferro-cuda-direct | 8 | 15 | 2.650 | 4194304 | 0.009 | 4.418e-07 |
| cholesky_f64_512x512 | tenferro-cuda-eager | 8 | 15 | 2.810 | 4194304 | 0.008 | 4.418e-07 |
| det_f64_512x512 | jax-cuda | 2 | 15 | 2.716 | 2097160 | 0.001 | 1.426e-05 |
| det_f64_512x512 | pytorch-cuda | 2 | 15 | 2.608 | 2097160 | 0.001 | 1.369e-05 |
| det_f64_512x512 | tenferro-cuda-direct | 2 | 15 | 3.137 | 2097160 | 0.001 | 1.426e-05 |
| det_f64_512x512 | tenferro-cuda-eager | 2 | 15 | 3.159 | 2097160 | 0.001 | 1.426e-05 |
| eigvalsh_f64_512x512 | jax-cuda | 1 | 15 | 6.088 | 2101248 | 0.000 | 1.55e-06 |
| eigvalsh_f64_512x512 | pytorch-cuda | 1 | 15 | 5.576 | 2101248 | 0.000 | 1.55e-06 |
| eigvalsh_f64_512x512 | tenferro-cuda-direct | 1 | 15 | 5.558 | 2101248 | 0.000 | 1.55e-06 |
| eigvalsh_f64_512x512 | tenferro-cuda-eager | 1 | 15 | 5.592 | 2101248 | 0.000 | 1.55e-06 |
| householder_append_default_f64_512x256 | jax-cuda | — | — | — | 3147776 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_f64_512x256 | pytorch-cuda | — | — | — | 3147776 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_f64_512x256 | tenferro-cuda-direct | 4 | 15 | 3.164 | 3147776 | 0.003 | 9.992e-07 |
| householder_append_default_f64_512x256 | tenferro-cuda-eager | 4 | 15 | 3.277 | 3147776 | 0.003 | 9.992e-07 |
| householder_from_factors_default_f64_512x256 | jax-cuda | — | — | — | 3672064 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_from_factors_default_f64_512x256 | pytorch-cuda | — | — | — | 3672064 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_from_factors_default_f64_512x256 | tenferro-cuda-direct | 2 | 15 | 2.955 | 3672064 | 0.002 | 8.882e-07 |
| householder_from_factors_default_f64_512x256 | tenferro-cuda-eager | 2 | 15 | 3.086 | 3672064 | 0.002 | 8.882e-07 |
| householder_q_columns_prefix_f64_512x256 | jax-cuda | 16 | 15 | 2.437 | 1081344 | 0.005 | 1.136e-07 |
| householder_q_columns_prefix_f64_512x256 | pytorch-cuda | 32 | 15 | 2.408 | 1081344 | 0.010 | 1.136e-07 |
| householder_q_columns_prefix_f64_512x256 | tenferro-cuda-direct | 16 | 15 | 2.539 | 1081344 | 0.005 | 1.136e-07 |
| householder_q_columns_prefix_f64_512x256 | tenferro-cuda-eager | 16 | 15 | 2.566 | 1081344 | 0.005 | 1.136e-07 |
| householder_qr_tall_f64_512x256 | jax-cuda | 2 | 15 | 2.899 | 2099200 | 0.001 | 1.887e-06 |
| householder_qr_tall_f64_512x256 | pytorch-cuda | 2 | 15 | 2.645 | 2099200 | 0.001 | 1.887e-06 |
| householder_qr_tall_f64_512x256 | tenferro-cuda-direct | 2 | 15 | 2.674 | 2099200 | 0.001 | 9.992e-07 |
| householder_qr_tall_f64_512x256 | tenferro-cuda-eager | 2 | 15 | 2.795 | 2099200 | 0.001 | 9.992e-07 |
| householder_r_default_f64_512x256 | jax-cuda | 32 | 15 | 2.110 | 1572864 | 0.016 | 9.976e-07 |
| householder_r_default_f64_512x256 | pytorch-cuda | 128 | 15 | 0.701 | 1572864 | 0.198 | 9.976e-07 |
| householder_r_default_f64_512x256 | tenferro-cuda-direct | 128 | 15 | 2.126 | 1572864 | 0.065 | 9.976e-07 |
| householder_r_default_f64_512x256 | tenferro-cuda-eager | 128 | 15 | 3.105 | 1572864 | 0.045 | 9.976e-07 |
| inv_f64_512x512 | jax-cuda | 2 | 15 | 3.650 | 4194304 | 0.002 | 1.85e-06 |
| inv_f64_512x512 | pytorch-cuda | 2 | 15 | 3.571 | 4194304 | 0.002 | 1.85e-06 |
| inv_f64_512x512 | tenferro-cuda-direct | 2 | 15 | 3.798 | 4194304 | 0.002 | 1.85e-06 |
| inv_f64_512x512 | tenferro-cuda-eager | 2 | 15 | 3.843 | 4194304 | 0.002 | 1.85e-06 |
| lu_tall_f64_s256 | jax-cuda | — | — | — | 4718600 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_tall_f64_s256 | pytorch-cuda | — | — | — | 4718600 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_tall_f64_s256 | tenferro-cuda-direct | 4 | 15 | 3.146 | 4718600 | 0.004 | 1.102e-06 |
| lu_tall_f64_s256 | tenferro-cuda-eager | 4 | 15 | 3.183 | 4718600 | 0.004 | 1.102e-06 |
| norm_f64_512x512 | jax-cuda | 32 | 15 | 3.450 | 2097160 | 0.013 | 1.909e-07 |
| norm_f64_512x512 | pytorch-cuda | 256 | 15 | 2.837 | 2097160 | 0.131 | 1.909e-07 |
| norm_f64_512x512 | tenferro-cuda-direct | 128 | 15 | 2.179 | 2097160 | 0.085 | 0 |
| norm_f64_512x512 | tenferro-cuda-eager | 128 | 15 | 2.193 | 2097160 | 0.084 | 0 |
| pinv_f64_512x512 | jax-cuda | 1 | 15 | 78.150 | 4194320 | 0.000 | 8.975e-06 |
| pinv_f64_512x512 | pytorch-cuda | 1 | 15 | 84.948 | 4194320 | 0.000 | 8.747e-06 |
| pinv_f64_512x512 | tenferro-cuda-direct | 1 | 15 | 78.172 | 4194320 | 0.000 | 8.975e-06 |
| pinv_f64_512x512 | tenferro-cuda-eager | 1 | 15 | 78.201 | 4194320 | 0.000 | 8.975e-06 |
| slogdet_positive_f64_256x256 | jax-cuda | 4 | 15 | 2.193 | 524304 | 0.001 | 1.864e-06 |
| slogdet_positive_f64_256x256 | pytorch-cuda | 4 | 15 | 2.250 | 524304 | 0.001 | 1.708e-06 |
| slogdet_positive_f64_256x256 | tenferro-cuda-direct | 4 | 15 | 2.559 | 524304 | 0.001 | 1.864e-06 |
| slogdet_positive_f64_256x256 | tenferro-cuda-eager | 4 | 15 | 2.676 | 524304 | 0.001 | 1.864e-06 |
| svd_full_tall_f64_512x256 | jax-cuda | 1 | 15 | 19.618 | 3672064 | 0.000 | 7.76e-05 |
| svd_full_tall_f64_512x256 | pytorch-cuda | 1 | 15 | 19.923 | 3672064 | 0.000 | 7.76e-05 |
| svd_full_tall_f64_512x256 | tenferro-cuda-direct | 1 | 15 | 19.596 | 3672064 | 0.000 | 7.76e-05 |
| svd_full_tall_f64_512x256 | tenferro-cuda-eager | 1 | 15 | 19.629 | 3672064 | 0.000 | 7.76e-05 |
| svdvals_f64_512x512 | jax-cuda | 1 | 15 | 77.795 | 2101248 | 0.000 | 0.0002063 |
| svdvals_f64_512x512 | pytorch-cuda | 1 | 15 | 84.678 | 2101248 | 0.000 | 0.0002066 |
| svdvals_f64_512x512 | tenferro-cuda-direct | 1 | 15 | 77.823 | 2101248 | 0.000 | 0.0002063 |
| svdvals_f64_512x512 | tenferro-cuda-eager | 1 | 15 | 77.841 | 2101248 | 0.000 | 0.0002063 |
| triangular_solve_f64_512x512 | jax-cuda | 16 | 15 | 3.494 | 2162688 | 0.007 | 1.157e-06 |
| triangular_solve_f64_512x512 | pytorch-cuda | 8 | 15 | 2.457 | 2162688 | 0.005 | 6.241e-07 |
| triangular_solve_f64_512x512 | tenferro-cuda-direct | 16 | 15 | 3.090 | 2162688 | 0.008 | 1.157e-06 |
| triangular_solve_f64_512x512 | tenferro-cuda-eager | 16 | 15 | 3.379 | 2162688 | 0.007 | 1.157e-06 |

## Run metadata

```yaml
schema_version: 1
target_profile: nvidia-gpu
suite_id: gpu/linalg_public_api_throughput
suite_file: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/data/results/nvidia-gpu/gpu/routine_selection/20261126_062950/gpu_linalg_public_api_throughput.yaml
timestamp: '2026-11-26T06:29:50+00:00'
tenferro_rs:
  path: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/extern/tenferro-rs
  commit: 1ba9f33521691170c763dfce9b6c9bb56ca523c2
  dirty: false
  features:
  - cuda
  resolved_path: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/extern/tenferro-rs
harness:
  path: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark
  commit: cc9867403ef65b20357ed79bc78ec5fc2864c4a9
  dirty: false
resolved_deps:
  strided-perm:
  - 0.8.0
  - 0.15.1
  strided-view:
  - 0.7.1
  - 0.8.1
  threadteam:
  - 0.1.0
  - 0.6.1
host:
  cpu: AMD EPYC 7543 32-Core Processor
  logical_cpus: 128
  power:
    cpufreq_boost: '1'
    governor: schedutil
    scaling_max_freq_khz: '2800000'
    smt:
      active: '1'
      control: 'on'
  numa:
    nodes: 2
    cpulists:
      '0': 0-31,64-95
      '1': 32-63,96-127
  smt_siblings:
  - 0,64
  - 1,65
  - 2,66
  - 3,67
  - 4,68
  - 5,69
  - 6,70
  - 7,71
  - 8,72
  - 9,73
  - 10,74
  - 11,75
  - 12,76
  - 13,77
  - 14,78
  - 15,79
  - 16,80
  - 17,81
  - 18,82
  - 19,83
  - 20,84
  - 21,85
  - 22,86
  - 23,87
  - 24,88
  - 25,89
  - 26,90
  - 27,91
  - 28,92
  - 29,93
  - 30,94
  - 31,95
  - 32,96
  - 33,97
  - 34,98
  - 35,99
  - 36,100
  - 37,101
  - 38,102
  - 39,103
  - 40,104
  - 41,105
  - 42,106
  - 43,107
  - 44,108
  - 45,109
  - 46,110
  - 47,111
  - 48,112
  - 49,113
  - 50,114
  - 51,115
  - 52,116
  - 53,117
  - 54,118
  - 55,119
  - 56,120
  - 57,121
  - 58,122
  - 59,123
  - 60,124
  - 61,125
  - 62,126
  - 63,127
  l3_domains:
  - 0-3,64-67
  - 4-7,68-71
  - 8-11,72-75
  - 12-15,76-79
  - 16-19,80-83
  - 20-23,84-87
  - 24-27,88-91
  - 28-31,92-95
  - 32-35,96-99
  - 36-39,100-103
  - 40-43,104-107
  - 44-47,108-111
  - 48-51,112-115
  - 52-55,116-119
  - 56-59,120-123
  - 60-63,124-127
build:
  toolchain:
    rustc: null
    rustc_release: null
    rustflags: null
    target_cpu: null
environment:
  container: devcontainer
  hostname: d73a3a75b493
  os: Linux-6.8.0-138-generic-x86_64-with-glibc2.39
  arch: x86_64
  cpu: x86_64
  affinity:
    pinning: none
    requested_measured_cpus: null
    observer_cpus: null
    override_of: null
    thread_budget: '1'
    measured_cpus: 0-127
    l3_domains:
    - 0-3,64-67
    - 4-7,68-71
    - 8-11,72-75
    - 12-15,76-79
    - 16-19,80-83
    - 20-23,84-87
    - 24-27,88-91
    - 28-31,92-95
    - 32-35,96-99
    - 36-39,100-103
    - 40-43,104-107
    - 44-47,108-111
    - 48-51,112-115
    - 52-55,116-119
    - 56-59,120-123
    - 60-63,124-127
    cpu_boost: enabled
    scaling_governor: schedutil
    scaling_driver: acpi-cpufreq
  env:
    OMP_NUM_THREADS: '1'
    RAYON_NUM_THREADS: '1'
    BENCHMARK_TARGET_PROFILE: nvidia-gpu
    OPENBLAS_NUM_THREADS: '1'
    MATMUL_NUM_THREADS: '1'
    CUDA_HOME: /opt/tenferro-benchmark-ci/cuda
    CUDA_PATH: /opt/tenferro-benchmark-ci/cuda
    TENFERRO_CUTENSOR_PATH: /opt/tenferro-benchmark-ci/cutensor/lib/libcutensor.so.2
    GPU_RUNTIME_SHA256: a261fcdef53ecfd3885b6d565dc834faef59287ce2da88c3d6754be96741f84d
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
      \ by MKL)\n  - NNPACK is enabled\n  - CPU capability usage: AVX2\n  - CUDA Runtime\
      \ 12.8\n  - NVCC architecture flags: -gencode;arch=compute_75,code=sm_75;-gencode;arch=compute_80,code=sm_80;-gencode;arch=compute_86,code=sm_86;-gencode;arch=compute_90,code=sm_90;-gencode;arch=compute_100,code=sm_100;-gencode;arch=compute_120,code=sm_120\n\
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
  device_uuid: GPU-e0c82f3d-3a11-a28c-b42e-1c92d26527fd
  device_memory_bytes: 85899345920
  driver_version: 595.112.07
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
  - 0.09188
  - 0.09227
  - 0.09285
  - 0.092931
  - 0.09208
  - 0.09272
  - 0.09149
  - 0.09302
  - 0.093461
  - 0.09251
  - 0.09295
  - 0.09263
  - 0.09243
  - 0.092131
  - 0.09291
  median_ms: 0.09263
  bytes_per_second: 1448966080103.6382
  torch_version: 2.11.0+cu128
  cuda_version: '12.8'
  timestamp: '2026-11-26T06:41:42.249046+00:00'
gpu_assignment:
  role: benchmark
  physical_device_index: 0
  device_uuid: GPU-e0c82f3d-3a11-a28c-b42e-1c92d26527fd
  cuda_visible_devices: GPU-e0c82f3d-3a11-a28c-b42e-1c92d26527fd
  process_device_ordinal: 0
  verification_only: false
collection:
  command: BENCHMARK_TARGET_PROFILE=nvidia-gpu GPU_BENCH_DEVICE=0 BENCH_RUNS=15 BENCH_WARMUPS=3
    ./scripts/run_gpu_public_api.sh linalg_public_api_throughput
  devcontainer_config: .devcontainer/cuda/devcontainer.json
  expected_cases: 16
  selected_cases: 16
  sha256:
    .devcontainer/cuda/devcontainer.json: 402f3763533ea1531d0e400193395302213fd1b151a42baaeaebeafca3acabcd
    .devcontainer/cuda/Dockerfile: 6eff822595fa44f53a6fd9eebe0f4e3cc9e28362d9cfa8cac977a21b10a823f2
    src/bin/benchmark_gpu_public_api.rs: f807adb35a2c03383fde11227a47dffbf0ec5cc08c45ce298f7d9aa6341bbb67
    src/bin/benchmark_gpu_public_api/lifetime.rs: 34be366f56aac88f5f3b86ade8018a15dad2bca85cea74f56cb2fe121917d0a9
    src/bin/benchmark_gpu_public_api/linalg.rs: e8121a793ab45e565d150633b7e9a473b4d4f2c457ae36baed344f2c809fcad5
    src/bin/benchmark_gpu_public_api/multi.rs: 950cb1b3c1a7cca37665b58e3de074bd906dafb8c84b17a51c27ce797a73afa8
    src/bin/benchmark_gpu_public_api/plan.rs: 108edbad472e42b1a734552826d6d233085b581a3484cc4444471435160a6fd8
    src/bin/benchmark_gpu_public_api/read_layout.rs: f51757a34c90ce68e6fae0994df74169ed0438022031465349c7379c7b2c85e0
    src/bin/benchmark_gpu_public_api/reuse.rs: 53bee3baaeba93f4e1ee57d0d1b9553c189edd51d1940ea630ad0dbaa05b11eb
    src/bin/benchmark_gpu_public_api/runtime_ops.rs: 88038508963fbc13303fb54b3898808e4303131321cabedb9f38255941ead438
    src/bin/benchmark_gpu_public_api/state.rs: e1cc2781ce7392154f32cc19372c9fc41ae29765aedc544c7ae7bf02f873c12b
    src/bin/benchmark_gpu_public_api/structural.rs: 09b3cbf0dc0579978344c1aa738f6367112a0c25236924441f690a59de3fb80e
    /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/scripts/benchmark_gpu_public_api.py: d1ff68f86496d991bcbc730138fe6e3076001b3f23cc1a94c44b96f6857cb8ed
    /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/scripts/gpu_public_api_ops.py: e88823db3f4a1fe43dc13c4743c9d78ece6af76a968724091af318dd8d7f2528
    scripts/run_gpu_public_api.sh: 7995bcf409258185e07210969a738896f64860b67df251fcd1100d11128a5431
    data/results/nvidia-gpu/gpu/linalg_public_api_throughput/20261126_062950/fixtures/manifest.json: 8a74129fa252ab33835079b24a6ac6f6ab5b3b4e7796933304a8b929baf2626e
    target/release/benchmark_gpu_public_api: 3ff693b2ae7411a94a3e3337550156543b79bbe33a2720b00f7fb8181bdf1fd4
  reference_numpy_version: 2.4.2
  measurement_cpu:
    policy: 'disabled (GPU_BENCH_CPU_SELECT unset): arms run on the allocation unpinned'
    arms:
      jax-cuda:
        arm: jax-cuda
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-26T06:42:21Z'
      pytorch-cuda:
        arm: pytorch-cuda
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-26T06:42:11Z'
      tenferro-cuda-direct:
        arm: tenferro-cuda-direct
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-26T06:41:53Z'
      tenferro-cuda-eager:
        arm: tenferro-cuda-eager
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-26T06:42:02Z'
  compile:
    case_wall_seconds: 31.302
    compiling_processes: 1
    modules_compiled: 1
    compile_time_ms: null
    compile_time_scope: 'not measured: no compile interval was recorded; `compile_modules`
      counts the NVRTC modules and the untimed wall is the case wall'
    untimed_wall_seconds: 2.459
```

## Untimed work over the recorded budget

Each row records the wall clock it spent outside its timed region. These cases spent more than the recorded budget ratio (0.2) there. The threshold has no measured basis, so this is a recorded warning: it does not change a cell, its acceptance or the exit status of the run.

| Case | Backend | Warning |
| --- | --- | --- |
| cholesky_f64_512x512 | tenferro-cuda-direct | cholesky_f64_512x512 untimed 182.5 ms is 455% of the 40.1 ms timed region (budget 20%) |
| svdvals_f64_512x512 | tenferro-cuda-direct | svdvals_f64_512x512 untimed 549.7 ms is 47% of the 1168.3 ms timed region (budget 20%) |
| eigvalsh_f64_512x512 | tenferro-cuda-direct | eigvalsh_f64_512x512 untimed 57.1 ms is 68% of the 83.4 ms timed region (budget 20%) |
| pinv_f64_512x512 | tenferro-cuda-direct | pinv_f64_512x512 untimed 2459.1 ms is 209% of the 1174.9 ms timed region (budget 20%) |
| inv_f64_512x512 | tenferro-cuda-direct | inv_f64_512x512 untimed 42.7 ms is 75% of the 57.0 ms timed region (budget 20%) |
| det_f64_512x512 | tenferro-cuda-direct | det_f64_512x512 untimed 16.7 ms is 34% of the 49.0 ms timed region (budget 20%) |
| norm_f64_512x512 | tenferro-cuda-direct | norm_f64_512x512 untimed 8.9 ms is 27% of the 32.7 ms timed region (budget 20%) |
| svd_full_tall_f64_512x256 | tenferro-cuda-direct | svd_full_tall_f64_512x256 untimed 701.1 ms is 239% of the 294.0 ms timed region (budget 20%) |
| slogdet_positive_f64_256x256 | tenferro-cuda-direct | slogdet_positive_f64_256x256 untimed 9.4 ms is 24% of the 38.4 ms timed region (budget 20%) |
| householder_qr_tall_f64_512x256 | tenferro-cuda-direct | householder_qr_tall_f64_512x256 untimed 460.4 ms is 1147% of the 40.1 ms timed region (budget 20%) |
| householder_from_factors_default_f64_512x256 | tenferro-cuda-direct | householder_from_factors_default_f64_512x256 untimed 440.3 ms is 994% of the 44.3 ms timed region (budget 20%) |
| householder_append_default_f64_512x256 | tenferro-cuda-direct | householder_append_default_f64_512x256 untimed 448.9 ms is 945% of the 47.5 ms timed region (budget 20%) |
| householder_r_default_f64_512x256 | tenferro-cuda-direct | householder_r_default_f64_512x256 untimed 12.3 ms is 38% of the 32.0 ms timed region (budget 20%) |
| householder_q_columns_prefix_f64_512x256 | tenferro-cuda-direct | householder_q_columns_prefix_f64_512x256 untimed 13.8 ms is 36% of the 38.1 ms timed region (budget 20%) |
| lu_tall_f64_s256 | tenferro-cuda-direct | lu_tall_f64_s256 untimed 27.9 ms is 59% of the 47.3 ms timed region (budget 20%) |
| cholesky_f64_512x512 | tenferro-cuda-eager | cholesky_f64_512x512 untimed 160.0 ms is 380% of the 42.1 ms timed region (budget 20%) |
| svdvals_f64_512x512 | tenferro-cuda-eager | svdvals_f64_512x512 untimed 549.4 ms is 47% of the 1170.7 ms timed region (budget 20%) |
| eigvalsh_f64_512x512 | tenferro-cuda-eager | eigvalsh_f64_512x512 untimed 56.4 ms is 67% of the 84.0 ms timed region (budget 20%) |
| pinv_f64_512x512 | tenferro-cuda-eager | pinv_f64_512x512 untimed 2559.2 ms is 218% of the 1173.1 ms timed region (budget 20%) |
| inv_f64_512x512 | tenferro-cuda-eager | inv_f64_512x512 untimed 43.9 ms is 76% of the 57.7 ms timed region (budget 20%) |
| det_f64_512x512 | tenferro-cuda-eager | det_f64_512x512 untimed 16.0 ms is 34% of the 47.4 ms timed region (budget 20%) |
| norm_f64_512x512 | tenferro-cuda-eager | norm_f64_512x512 untimed 7.3 ms is 22% of the 33.4 ms timed region (budget 20%) |
| triangular_solve_f64_512x512 | tenferro-cuda-eager | triangular_solve_f64_512x512 untimed 10.2 ms is 20% of the 50.7 ms timed region (budget 20%) |
| svd_full_tall_f64_512x256 | tenferro-cuda-eager | svd_full_tall_f64_512x256 untimed 739.5 ms is 251% of the 294.4 ms timed region (budget 20%) |
| slogdet_positive_f64_256x256 | tenferro-cuda-eager | slogdet_positive_f64_256x256 untimed 10.4 ms is 26% of the 40.1 ms timed region (budget 20%) |
| householder_qr_tall_f64_512x256 | tenferro-cuda-eager | householder_qr_tall_f64_512x256 untimed 470.2 ms is 1122% of the 41.9 ms timed region (budget 20%) |
| householder_from_factors_default_f64_512x256 | tenferro-cuda-eager | householder_from_factors_default_f64_512x256 untimed 445.2 ms is 963% of the 46.2 ms timed region (budget 20%) |
| householder_append_default_f64_512x256 | tenferro-cuda-eager | householder_append_default_f64_512x256 untimed 451.6 ms is 914% of the 49.4 ms timed region (budget 20%) |
| householder_r_default_f64_512x256 | tenferro-cuda-eager | householder_r_default_f64_512x256 untimed 12.8 ms is 27% of the 46.6 ms timed region (budget 20%) |
| householder_q_columns_prefix_f64_512x256 | tenferro-cuda-eager | householder_q_columns_prefix_f64_512x256 untimed 13.9 ms is 36% of the 38.5 ms timed region (budget 20%) |
| lu_tall_f64_s256 | tenferro-cuda-eager | lu_tall_f64_s256 untimed 31.2 ms is 65% of the 47.9 ms timed region (budget 20%) |
| cholesky_f64_512x512 | pytorch-cuda | cholesky_f64_512x512 untimed 196.8 ms is 499% of the 39.5 ms timed region (budget 20%) |
| svdvals_f64_512x512 | pytorch-cuda | svdvals_f64_512x512 untimed 604.3 ms is 47% of the 1277.5 ms timed region (budget 20%) |
| eigvalsh_f64_512x512 | pytorch-cuda | eigvalsh_f64_512x512 untimed 61.8 ms is 74% of the 83.7 ms timed region (budget 20%) |
| pinv_f64_512x512 | pytorch-cuda | pinv_f64_512x512 untimed 710.5 ms is 56% of the 1279.4 ms timed region (budget 20%) |
| inv_f64_512x512 | pytorch-cuda | inv_f64_512x512 untimed 69.6 ms is 130% of the 53.5 ms timed region (budget 20%) |
| det_f64_512x512 | pytorch-cuda | det_f64_512x512 untimed 572.3 ms is 1460% of the 39.2 ms timed region (budget 20%) |
| norm_f64_512x512 | pytorch-cuda | norm_f64_512x512 untimed 41.5 ms is 97% of the 42.7 ms timed region (budget 20%) |
| triangular_solve_f64_512x512 | pytorch-cuda | triangular_solve_f64_512x512 untimed 14.7 ms is 40% of the 36.9 ms timed region (budget 20%) |
| svd_full_tall_f64_512x256 | pytorch-cuda | svd_full_tall_f64_512x256 untimed 215.9 ms is 72% of the 298.8 ms timed region (budget 20%) |
| slogdet_positive_f64_256x256 | pytorch-cuda | slogdet_positive_f64_256x256 untimed 26.1 ms is 77% of the 33.7 ms timed region (budget 20%) |
| householder_qr_tall_f64_512x256 | pytorch-cuda | householder_qr_tall_f64_512x256 untimed 225.9 ms is 570% of the 39.7 ms timed region (budget 20%) |
| householder_r_default_f64_512x256 | pytorch-cuda | householder_r_default_f64_512x256 untimed 16.1 ms is 153% of the 10.5 ms timed region (budget 20%) |
| householder_q_columns_prefix_f64_512x256 | pytorch-cuda | householder_q_columns_prefix_f64_512x256 untimed 20.6 ms is 57% of the 36.4 ms timed region (budget 20%) |
| cholesky_f64_512x512 | jax-cuda | cholesky_f64_512x512 untimed 457.9 ms is 1248% of the 36.7 ms timed region (budget 20%) |
| svdvals_f64_512x512 | jax-cuda | svdvals_f64_512x512 untimed 664.5 ms is 57% of the 1169.9 ms timed region (budget 20%) |
| eigvalsh_f64_512x512 | jax-cuda | eigvalsh_f64_512x512 untimed 187.1 ms is 204% of the 91.8 ms timed region (budget 20%) |
| pinv_f64_512x512 | jax-cuda | pinv_f64_512x512 untimed 1149.4 ms is 98% of the 1172.9 ms timed region (budget 20%) |
| inv_f64_512x512 | jax-cuda | inv_f64_512x512 untimed 233.6 ms is 427% of the 54.7 ms timed region (budget 20%) |
| det_f64_512x512 | jax-cuda | det_f64_512x512 untimed 249.0 ms is 609% of the 40.9 ms timed region (budget 20%) |
| norm_f64_512x512 | jax-cuda | norm_f64_512x512 untimed 175.7 ms is 384% of the 45.7 ms timed region (budget 20%) |
| triangular_solve_f64_512x512 | jax-cuda | triangular_solve_f64_512x512 untimed 138.7 ms is 260% of the 53.4 ms timed region (budget 20%) |
| svd_full_tall_f64_512x256 | jax-cuda | svd_full_tall_f64_512x256 untimed 351.4 ms is 119% of the 294.3 ms timed region (budget 20%) |
| slogdet_positive_f64_256x256 | jax-cuda | slogdet_positive_f64_256x256 untimed 238.0 ms is 719% of the 33.1 ms timed region (budget 20%) |
| householder_qr_tall_f64_512x256 | jax-cuda | householder_qr_tall_f64_512x256 untimed 332.8 ms is 770% of the 43.2 ms timed region (budget 20%) |
| householder_r_default_f64_512x256 | jax-cuda | householder_r_default_f64_512x256 untimed 239.1 ms is 725% of the 33.0 ms timed region (budget 20%) |
| householder_q_columns_prefix_f64_512x256 | jax-cuda | householder_q_columns_prefix_f64_512x256 untimed 213.8 ms is 547% of the 39.1 ms timed region (budget 20%) |