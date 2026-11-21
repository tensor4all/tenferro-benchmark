# gpu/linalg_public_api_throughput: CUDA public API benchmark

Measured cells: **49**. Unsupported cells: **15**, including **6** numerically verified but timing-excluded cells. Unsupported cells do not count as measured coverage.

Logical traffic: this run measured a plain device copy of 67108864 bytes at **1398.1 GB/s** on this device. Every record carries the logical bytes the case must move per operation and the ratio of the case's implied rate to that copy rate, in the sampling table below. Logical bytes are the declared array sizes, not device DRAM traffic: a case whose operands stay in cache moves less DRAM than its arrays suggest, so a ratio above 1 is recorded, not rejected.

Raw run: `data/results/nvidia-gpu/gpu/linalg_public_api_throughput/20261121_082311/`.

Large-shape view of the CUDA public linalg API surface: the 82 cases whose largest matrix dimension is at least 256. `large` is a shape class, not a measured classification: no row is asserted to be compute- or bandwidth-bound, and a memory-bound row takes the same time here as at a small shape (`householder_r_default_f32` measured 34.698 µs at 512x256 and 35.223 µs at 64x32). These rows are never comparable with, or averaged together with, the per-call overhead rows of `gpu/linalg_public_api_latency` (tenferro-rs #2087). Rows here are not lower than on pages published before this harness change for any timing reason: the sample clock starts inside `measure`, so engine construction was never part of a case median. What changed is repeated **untimed setup** work, which was removed: the CUDA engine is opened once per arm and held for the whole run instead of once per case, which shortens the untimed wall clock around the cases and moves no operation median on this page. The same problems, protocol and backends are split by shape class between the two suites. Issue 155.

This run compiled **1 NVRTC modules** in 1 of its arms, because the CUDA artifact cache started cold. The untimed wall of the cases that compiled is **0.0 s** of its 23.5 s of recorded case wall clock, and that wall also holds their fixture read, transfers and verification, so it is not a compile time.

Values are median (IQR) **µs per operation**. Samples time batches of declared operations plus device synchronization; retained output memory is limited to 256 MiB. Batch counts and raw durations are preserved in JSONL.

Cases marked † (**4** of 16) were sampled with a different number of operations per sample on different arms (see Ops/sample in the sampling table). Each value is still per operation, but the arms amortise launch and synchronization over different batch sizes, so a ratio between their columns is approximate and is not presented as a like-for-like comparison.

Setup, input transfers, library/plan initialization, JIT compilation, validation downloads and output destruction are outside timing. Intrinsic output allocation is inside timing; output-reuse rows instead use preallocated destinations across samples. Accumulating destinations are restored from prepared GPU seeds and synchronized before every batch clock, including initialization and calibration. Direct and eager each use one entered session across samples; direct FFT uses cached `FftExecutor`. JAX eager and compiled JIT are separate arms.

All measured output elements are checked against independently generated NumPy/SciPy reference results before and after sampling. Unsupported rows include unavailable APIs and APIs whose internal setup cannot be excluded; these are not performance measurements. Short batches limited by output memory remain identifiable in the sampling table.

| Case | tenferro-cuda-direct | tenferro-cuda-eager | pytorch-cuda | jax-cuda |
| --- | ---: | ---: | ---: | ---: |
| cholesky_f64_512x512 | 334.300 (2.106) | 380.089 (1.567) | 325.338 (1.469) | 310.663 (6.585) |
| svdvals_f64_512x512 | 78297.669 (125.961) | unsupported | 85113.691 (18.625) | 78532.271 (111.801) |
| eigvalsh_f64_512x512 | 5487.950 (17.245) | 5521.140 (11.990) | 5490.119 (10.825) | 6164.716 (43.980) |
| pinv_f64_512x512 | unsupported | unsupported | 85288.142 (47.910) | 78812.004 (46.186) |
| inv_f64_512x512 | unsupported | unsupported | 1799.306 (5.565) | 1833.632 (13.093) |
| det_f64_512x512 | 1725.910 (7.842) | 1814.596 (23.947) | 1318.057 (8.447) | 1374.558 (13.535) |
| norm_f64_512x512 † | 44.480 (1.085) | 64.190 (0.596) | 9.898 (0.151) | 121.495 (5.406) |
| triangular_solve_f64_512x512 † | 193.465 (0.365) | 239.283 (7.130) | 307.890 (3.336) | 220.478 (1.143) |
| svd_full_tall_f64_512x256 | 19715.128 (19.570) | 19729.378 (48.470) | 20038.102 (18.735) | 19845.510 (59.301) |
| slogdet_positive_f64_256x256 | 789.517 (7.169) | 815.880 (13.234) | 565.388 (2.453) | 581.665 (10.648) |
| householder_qr_tall_f64_512x256 | 1340.677 (5.665) | 1364.242 (4.336) | 1314.467 (6.218) | 1398.463 (19.837) |
| householder_from_factors_default_f64_512x256 | 1522.199 (10.340) | 1584.504 (9.235) | unsupported | unsupported |
| householder_append_default_f64_512x256 | unsupported | unsupported | unsupported | unsupported |
| householder_r_default_f64_512x256 † | 38.383 (0.429) | 48.875 (1.236) | 5.995 (0.278) | 83.620 (23.414) |
| householder_q_columns_prefix_f64_512x256 † | unsupported | unsupported | 68.608 (0.530) | 286.099 (82.654) |
| lu_tall_f64_s256 | 841.670 (2.974) | 886.806 (1.751) | unsupported | unsupported |

## Sampling

| Case | Backend | Ops/sample | Samples | Median batch ms | Bytes/op | Copy ratio | Verification error/tolerance |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cholesky_f64_512x512 | jax-cuda | 8 | 15 | 2.485 | 4194304 | 0.010 | 4.418e-07 |
| cholesky_f64_512x512 | pytorch-cuda | 8 | 15 | 2.603 | 4194304 | 0.009 | 4.418e-07 |
| cholesky_f64_512x512 | tenferro-cuda-direct | 8 | 15 | 2.674 | 4194304 | 0.009 | 4.418e-07 |
| cholesky_f64_512x512 | tenferro-cuda-eager | 8 | 15 | 3.041 | 4194304 | 0.008 | 4.418e-07 |
| det_f64_512x512 | jax-cuda | 2 | 15 | 2.749 | 2097160 | 0.001 | 1.426e-05 |
| det_f64_512x512 | pytorch-cuda | 2 | 15 | 2.636 | 2097160 | 0.001 | 1.369e-05 |
| det_f64_512x512 | tenferro-cuda-direct | 2 | 15 | 3.452 | 2097160 | 0.001 | 1.426e-05 |
| det_f64_512x512 | tenferro-cuda-eager | 2 | 15 | 3.629 | 2097160 | 0.001 | 1.426e-05 |
| eigvalsh_f64_512x512 | jax-cuda | 1 | 15 | 6.165 | 2101248 | 0.000 | 1.55e-06 |
| eigvalsh_f64_512x512 | pytorch-cuda | 1 | 15 | 5.490 | 2101248 | 0.000 | 1.55e-06 |
| eigvalsh_f64_512x512 | tenferro-cuda-direct | 1 | 15 | 5.488 | 2101248 | 0.000 | 1.55e-06 |
| eigvalsh_f64_512x512 | tenferro-cuda-eager | 1 | 15 | 5.521 | 2101248 | 0.000 | 1.55e-06 |
| householder_append_default_f64_512x256 | jax-cuda | — | — | — | 3147776 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_f64_512x256 | pytorch-cuda | — | — | — | 3147776 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_f64_512x256 | tenferro-cuda-direct | — | — | — | 3147776 | unavailable | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_append_default_f64_512x256 | tenferro-cuda-eager | — | — | — | 3147776 | unavailable | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_from_factors_default_f64_512x256 | jax-cuda | — | — | — | 3672064 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_from_factors_default_f64_512x256 | pytorch-cuda | — | — | — | 3672064 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_from_factors_default_f64_512x256 | tenferro-cuda-direct | 2 | 15 | 3.044 | 3672064 | 0.002 | 8.882e-07 |
| householder_from_factors_default_f64_512x256 | tenferro-cuda-eager | 2 | 15 | 3.169 | 3672064 | 0.002 | 8.882e-07 |
| householder_q_columns_prefix_f64_512x256 | jax-cuda | 8 | 15 | 2.289 | 1081344 | 0.003 | 1.136e-07 |
| householder_q_columns_prefix_f64_512x256 | pytorch-cuda | 32 | 15 | 2.195 | 1081344 | 0.011 | 1.136e-07 |
| householder_q_columns_prefix_f64_512x256 | tenferro-cuda-direct | — | — | — | 1081344 | unavailable | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_prefix_f64_512x256 | tenferro-cuda-eager | — | — | — | 1081344 | unavailable | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_qr_tall_f64_512x256 | jax-cuda | 2 | 15 | 2.797 | 2099200 | 0.001 | 1.887e-06 |
| householder_qr_tall_f64_512x256 | pytorch-cuda | 2 | 15 | 2.629 | 2099200 | 0.001 | 1.887e-06 |
| householder_qr_tall_f64_512x256 | tenferro-cuda-direct | 2 | 15 | 2.681 | 2099200 | 0.001 | 9.992e-07 |
| householder_qr_tall_f64_512x256 | tenferro-cuda-eager | 2 | 15 | 2.728 | 2099200 | 0.001 | 9.992e-07 |
| householder_r_default_f64_512x256 | jax-cuda | 32 | 15 | 2.676 | 1572864 | 0.013 | 9.976e-07 |
| householder_r_default_f64_512x256 | pytorch-cuda | 8 | 15 | 0.048 | 1572864 | 0.188 | 9.976e-07 |
| householder_r_default_f64_512x256 | tenferro-cuda-direct | 64 | 15 | 2.457 | 1572864 | 0.029 | 9.976e-07 |
| householder_r_default_f64_512x256 | tenferro-cuda-eager | 64 | 15 | 3.128 | 1572864 | 0.023 | 9.976e-07 |
| inv_f64_512x512 | jax-cuda | 2 | 15 | 3.667 | 4194304 | 0.002 | 1.85e-06 |
| inv_f64_512x512 | pytorch-cuda | 2 | 15 | 3.599 | 4194304 | 0.002 | 1.85e-06 |
| inv_f64_512x512 | tenferro-cuda-direct | — | — | — | 4194304 | unavailable | CUDA concrete route fails at revision 8b13220c: inv_owned/eye_like passes a host scalar to broadcast_in_dim. No public caller parameter can replace this internal operand; reproduced in raw run 20261013_094906. |
| inv_f64_512x512 | tenferro-cuda-eager | — | — | — | 4194304 | unavailable | Public linalg composite constructs scalar/identity operands per call; cannot move internal setup outside the timer. |
| lu_tall_f64_s256 | jax-cuda | — | — | — | 4718600 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_tall_f64_s256 | pytorch-cuda | — | — | — | 4718600 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_tall_f64_s256 | tenferro-cuda-direct | 4 | 15 | 3.367 | 4718600 | 0.004 | 1.102e-06 |
| lu_tall_f64_s256 | tenferro-cuda-eager | 4 | 15 | 3.547 | 4718600 | 0.004 | 1.102e-06 |
| norm_f64_512x512 | jax-cuda | 16 | 15 | 1.944 | 2097160 | 0.012 | 1.909e-07 |
| norm_f64_512x512 | pytorch-cuda | 256 | 15 | 2.534 | 2097160 | 0.152 | 1.909e-07 |
| norm_f64_512x512 | tenferro-cuda-direct | 64 | 15 | 2.847 | 2097160 | 0.034 | 0 |
| norm_f64_512x512 | tenferro-cuda-eager | 32 | 15 | 2.054 | 2097160 | 0.023 | 0 |
| pinv_f64_512x512 | jax-cuda | 1 | 15 | 78.812 | 4194320 | 0.000 | 8.975e-06 |
| pinv_f64_512x512 | pytorch-cuda | 1 | 15 | 85.288 | 4194320 | 0.000 | 8.747e-06 |
| pinv_f64_512x512 | tenferro-cuda-direct | — | — | — | 4194320 | unavailable | CUDA concrete route fails at revision 8b13220c: pinv_from_svd passes a host threshold scalar to mul_read. No public caller parameter can replace this internal operand; reproduced in raw run 20261013_094906. |
| pinv_f64_512x512 | tenferro-cuda-eager | — | — | — | 4194320 | unavailable | Public linalg composite constructs scalar/identity operands per call; cannot move internal setup outside the timer. |
| slogdet_positive_f64_256x256 | jax-cuda | 4 | 15 | 2.327 | 524304 | 0.001 | 1.864e-06 |
| slogdet_positive_f64_256x256 | pytorch-cuda | 4 | 15 | 2.262 | 524304 | 0.001 | 1.708e-06 |
| slogdet_positive_f64_256x256 | tenferro-cuda-direct | 4 | 15 | 3.158 | 524304 | 0.000 | 1.864e-06 |
| slogdet_positive_f64_256x256 | tenferro-cuda-eager | 4 | 15 | 3.264 | 524304 | 0.000 | 1.864e-06 |
| svd_full_tall_f64_512x256 | jax-cuda | 1 | 15 | 19.846 | 3672064 | 0.000 | 7.76e-05 |
| svd_full_tall_f64_512x256 | pytorch-cuda | 1 | 15 | 20.038 | 3672064 | 0.000 | 7.76e-05 |
| svd_full_tall_f64_512x256 | tenferro-cuda-direct | 1 | 15 | 19.715 | 3672064 | 0.000 | 7.76e-05 |
| svd_full_tall_f64_512x256 | tenferro-cuda-eager | 1 | 15 | 19.729 | 3672064 | 0.000 | 7.76e-05 |
| svdvals_f64_512x512 | jax-cuda | 1 | 15 | 78.532 | 2101248 | 0.000 | 0.0002063 |
| svdvals_f64_512x512 | pytorch-cuda | 1 | 15 | 85.114 | 2101248 | 0.000 | 0.0002066 |
| svdvals_f64_512x512 | tenferro-cuda-direct | 1 | 15 | 78.298 | 2101248 | 0.000 | 0.0002063 |
| svdvals_f64_512x512 | tenferro-cuda-eager | — | — | — | 2101248 | unavailable | EagerSessionLinalgExt has no values-only svdvals entry point; returning and discarding singular vectors is not equivalent. |
| triangular_solve_f64_512x512 | jax-cuda | 16 | 15 | 3.528 | 2162688 | 0.007 | 1.157e-06 |
| triangular_solve_f64_512x512 | pytorch-cuda | 8 | 15 | 2.463 | 2162688 | 0.005 | 6.241e-07 |
| triangular_solve_f64_512x512 | tenferro-cuda-direct | 16 | 15 | 3.095 | 2162688 | 0.008 | 1.157e-06 |
| triangular_solve_f64_512x512 | tenferro-cuda-eager | 16 | 15 | 3.829 | 2162688 | 0.006 | 1.157e-06 |

## Run metadata

```yaml
schema_version: 1
target_profile: nvidia-gpu
suite_id: gpu/linalg_public_api_throughput
suite_file: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/data/results/nvidia-gpu/gpu/routine_selection/20261121_082311/gpu_linalg_public_api_throughput.yaml
timestamp: '2026-11-21T08:23:11+00:00'
tenferro_rs:
  path: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/extern/tenferro-rs
  commit: ce9125d0cff4ee40841120759b5df45252c27f22
  dirty: false
  features:
  - cuda
  resolved_path: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/extern/tenferro-rs
harness:
  path: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark
  commit: a741cd656e4d51e77d8369154bf789e5f0147f5c
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
  hostname: 7894e3c40841
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
    GPU_RUNTIME_SHA256: 3d603f753ad72a63b928ce8e1b63847b8a21f94f0919276124993a6548126185
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
  device_uuid: GPU-98d8618e-77c8-160a-92f2-117e34fd43df
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
  - 0.094901
  - 0.094331
  - 0.102771
  - 0.099951
  - 0.094801
  - 0.097391
  - 0.096001
  - 0.096041
  - 0.096
  - 0.096581
  - 0.094301
  - 0.095951
  - 0.096071
  - 0.095181
  - 0.095641
  median_ms: 0.096
  bytes_per_second: 1398101333333.3333
  torch_version: 2.11.0+cu128
  cuda_version: '12.8'
  timestamp: '2026-11-21T08:32:54.812838+00:00'
gpu_assignment:
  role: benchmark
  physical_device_index: 0
  device_uuid: GPU-98d8618e-77c8-160a-92f2-117e34fd43df
  cuda_visible_devices: GPU-98d8618e-77c8-160a-92f2-117e34fd43df
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
    src/bin/benchmark_gpu_public_api.rs: a5ed803c89dc3085b18ed3220a6d484facf179c9b55fe13042beb19d83090196
    src/bin/benchmark_gpu_public_api/lifetime.rs: 34be366f56aac88f5f3b86ade8018a15dad2bca85cea74f56cb2fe121917d0a9
    src/bin/benchmark_gpu_public_api/linalg.rs: 3d5bedcc3620f75cab2687fdf49210df324021d043b23aec6e169642ceff15ba
    src/bin/benchmark_gpu_public_api/multi.rs: 950cb1b3c1a7cca37665b58e3de074bd906dafb8c84b17a51c27ce797a73afa8
    src/bin/benchmark_gpu_public_api/plan.rs: 016e752c59edefc3ded1405c36325e8680736860cb65954919ea4cf0f3eed061
    src/bin/benchmark_gpu_public_api/read_layout.rs: fc42e2160094ba5f9c7dd6facb5667a76e10b73f971ef9facfb1f64723e42467
    src/bin/benchmark_gpu_public_api/reuse.rs: e9b588e54fd067e570b557a4b06e40f0b9b432d665f50b64ee7f2ff7f461754b
    src/bin/benchmark_gpu_public_api/runtime_ops.rs: 88038508963fbc13303fb54b3898808e4303131321cabedb9f38255941ead438
    src/bin/benchmark_gpu_public_api/state.rs: e1cc2781ce7392154f32cc19372c9fc41ae29765aedc544c7ae7bf02f873c12b
    src/bin/benchmark_gpu_public_api/structural.rs: 10850458a640d11e79262c215072482ee463bc4ca0604f40d82ae56f345f6564
    /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/scripts/benchmark_gpu_public_api.py: d1ff68f86496d991bcbc730138fe6e3076001b3f23cc1a94c44b96f6857cb8ed
    /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/scripts/gpu_public_api_ops.py: e88823db3f4a1fe43dc13c4743c9d78ece6af76a968724091af318dd8d7f2528
    scripts/run_gpu_public_api.sh: 7995bcf409258185e07210969a738896f64860b67df251fcd1100d11128a5431
    data/results/nvidia-gpu/gpu/linalg_public_api_throughput/20261121_082311/fixtures/manifest.json: 8ac918ec3b3c102e5ae205c7547632fe9d4d7bbfd0f6cd3c1b00f3667f37aefc
    target/release/benchmark_gpu_public_api: c67dd59529f9c6644708d8bc0a5a5fabc82f29fff70f5532c20f6a3066720813
  reference_numpy_version: 2.4.2
  measurement_cpu:
    policy: 'disabled (GPU_BENCH_CPU_SELECT unset): arms run on the allocation unpinned'
    arms:
      jax-cuda:
        arm: jax-cuda
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-21T08:33:25Z'
      pytorch-cuda:
        arm: pytorch-cuda
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-21T08:33:16Z'
      tenferro-cuda-direct:
        arm: tenferro-cuda-direct
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-21T08:33:01Z'
      tenferro-cuda-eager:
        arm: tenferro-cuda-eager
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-21T08:33:07Z'
  compile:
    case_wall_seconds: 23.547
    compiling_processes: 1
    modules_compiled: 1
    compile_time_ms: null
    compile_time_scope: 'not measured: no compile interval was recorded; `compile_modules`
      counts the NVRTC modules and the untimed wall is the case wall'
    untimed_wall_seconds: 0.0
```

## Untimed work over the recorded budget

Each row records the wall clock it spent outside its timed region. These cases spent more than the recorded budget ratio (0.2) there. The threshold has no measured basis, so this is a recorded warning: it does not change a cell, its acceptance or the exit status of the run.

| Case | Backend | Warning |
| --- | --- | --- |
| cholesky_f64_512x512 | tenferro-cuda-direct | cholesky_f64_512x512 untimed 183.3 ms is 456% of the 40.2 ms timed region (budget 20%) |
| svdvals_f64_512x512 | tenferro-cuda-direct | svdvals_f64_512x512 untimed 552.4 ms is 47% of the 1183.0 ms timed region (budget 20%) |
| eigvalsh_f64_512x512 | tenferro-cuda-direct | eigvalsh_f64_512x512 untimed 60.5 ms is 73% of the 82.4 ms timed region (budget 20%) |
| det_f64_512x512 | tenferro-cuda-direct | det_f64_512x512 untimed 28.5 ms is 55% of the 51.8 ms timed region (budget 20%) |
| triangular_solve_f64_512x512 | tenferro-cuda-direct | triangular_solve_f64_512x512 untimed 11.3 ms is 24% of the 46.4 ms timed region (budget 20%) |
| svd_full_tall_f64_512x256 | tenferro-cuda-direct | svd_full_tall_f64_512x256 untimed 741.8 ms is 251% of the 295.7 ms timed region (budget 20%) |
| slogdet_positive_f64_256x256 | tenferro-cuda-direct | slogdet_positive_f64_256x256 untimed 11.3 ms is 24% of the 47.4 ms timed region (budget 20%) |
| householder_qr_tall_f64_512x256 | tenferro-cuda-direct | householder_qr_tall_f64_512x256 untimed 448.3 ms is 1115% of the 40.2 ms timed region (budget 20%) |
| householder_from_factors_default_f64_512x256 | tenferro-cuda-direct | householder_from_factors_default_f64_512x256 untimed 453.2 ms is 990% of the 45.8 ms timed region (budget 20%) |
| householder_r_default_f64_512x256 | tenferro-cuda-direct | householder_r_default_f64_512x256 untimed 9.5 ms is 25% of the 37.7 ms timed region (budget 20%) |
| lu_tall_f64_s256 | tenferro-cuda-direct | lu_tall_f64_s256 untimed 42.8 ms is 85% of the 50.5 ms timed region (budget 20%) |
| cholesky_f64_512x512 | tenferro-cuda-eager | cholesky_f64_512x512 untimed 154.7 ms is 339% of the 45.6 ms timed region (budget 20%) |
| eigvalsh_f64_512x512 | tenferro-cuda-eager | eigvalsh_f64_512x512 untimed 53.2 ms is 64% of the 82.8 ms timed region (budget 20%) |
| det_f64_512x512 | tenferro-cuda-eager | det_f64_512x512 untimed 18.8 ms is 34% of the 54.5 ms timed region (budget 20%) |
| norm_f64_512x512 | tenferro-cuda-eager | norm_f64_512x512 untimed 6.6 ms is 21% of the 31.2 ms timed region (budget 20%) |
| svd_full_tall_f64_512x256 | tenferro-cuda-eager | svd_full_tall_f64_512x256 untimed 727.5 ms is 243% of the 299.3 ms timed region (budget 20%) |
| slogdet_positive_f64_256x256 | tenferro-cuda-eager | slogdet_positive_f64_256x256 untimed 12.5 ms is 25% of the 49.7 ms timed region (budget 20%) |
| householder_qr_tall_f64_512x256 | tenferro-cuda-eager | householder_qr_tall_f64_512x256 untimed 449.2 ms is 1098% of the 40.9 ms timed region (budget 20%) |
| householder_from_factors_default_f64_512x256 | tenferro-cuda-eager | householder_from_factors_default_f64_512x256 untimed 452.0 ms is 947% of the 47.7 ms timed region (budget 20%) |
| householder_r_default_f64_512x256 | tenferro-cuda-eager | householder_r_default_f64_512x256 untimed 11.9 ms is 25% of the 48.0 ms timed region (budget 20%) |
| lu_tall_f64_s256 | tenferro-cuda-eager | lu_tall_f64_s256 untimed 30.9 ms is 58% of the 53.3 ms timed region (budget 20%) |
| cholesky_f64_512x512 | pytorch-cuda | cholesky_f64_512x512 untimed 203.0 ms is 516% of the 39.4 ms timed region (budget 20%) |
| svdvals_f64_512x512 | pytorch-cuda | svdvals_f64_512x512 untimed 604.5 ms is 47% of the 1282.7 ms timed region (budget 20%) |
| eigvalsh_f64_512x512 | pytorch-cuda | eigvalsh_f64_512x512 untimed 57.8 ms is 70% of the 82.4 ms timed region (budget 20%) |
| pinv_f64_512x512 | pytorch-cuda | pinv_f64_512x512 untimed 712.7 ms is 56% of the 1279.6 ms timed region (budget 20%) |
| inv_f64_512x512 | pytorch-cuda | inv_f64_512x512 untimed 66.7 ms is 123% of the 54.0 ms timed region (budget 20%) |
| det_f64_512x512 | pytorch-cuda | det_f64_512x512 untimed 548.5 ms is 1384% of the 39.6 ms timed region (budget 20%) |
| norm_f64_512x512 | pytorch-cuda | norm_f64_512x512 untimed 38.2 ms is 100% of the 38.2 ms timed region (budget 20%) |
| triangular_solve_f64_512x512 | pytorch-cuda | triangular_solve_f64_512x512 untimed 13.7 ms is 37% of the 37.4 ms timed region (budget 20%) |
| svd_full_tall_f64_512x256 | pytorch-cuda | svd_full_tall_f64_512x256 untimed 214.1 ms is 71% of the 300.7 ms timed region (budget 20%) |
| slogdet_positive_f64_256x256 | pytorch-cuda | slogdet_positive_f64_256x256 untimed 25.1 ms is 74% of the 34.2 ms timed region (budget 20%) |
| householder_qr_tall_f64_512x256 | pytorch-cuda | householder_qr_tall_f64_512x256 untimed 221.6 ms is 561% of the 39.5 ms timed region (budget 20%) |
| householder_r_default_f64_512x256 | pytorch-cuda | householder_r_default_f64_512x256 untimed 11.9 ms is 1594% of the 0.7 ms timed region (budget 20%) |
| householder_q_columns_prefix_f64_512x256 | pytorch-cuda | householder_q_columns_prefix_f64_512x256 untimed 19.2 ms is 58% of the 33.1 ms timed region (budget 20%) |
| cholesky_f64_512x512 | jax-cuda | cholesky_f64_512x512 untimed 459.3 ms is 1231% of the 37.3 ms timed region (budget 20%) |
| svdvals_f64_512x512 | jax-cuda | svdvals_f64_512x512 untimed 658.4 ms is 56% of the 1183.2 ms timed region (budget 20%) |
| eigvalsh_f64_512x512 | jax-cuda | eigvalsh_f64_512x512 untimed 188.4 ms is 203% of the 92.6 ms timed region (budget 20%) |
| pinv_f64_512x512 | jax-cuda | pinv_f64_512x512 untimed 1131.6 ms is 96% of the 1183.2 ms timed region (budget 20%) |
| inv_f64_512x512 | jax-cuda | inv_f64_512x512 untimed 252.3 ms is 459% of the 55.0 ms timed region (budget 20%) |
| det_f64_512x512 | jax-cuda | det_f64_512x512 untimed 211.4 ms is 510% of the 41.4 ms timed region (budget 20%) |
| norm_f64_512x512 | jax-cuda | norm_f64_512x512 untimed 195.9 ms is 671% of the 29.2 ms timed region (budget 20%) |
| triangular_solve_f64_512x512 | jax-cuda | triangular_solve_f64_512x512 untimed 138.5 ms is 261% of the 53.1 ms timed region (budget 20%) |
| svd_full_tall_f64_512x256 | jax-cuda | svd_full_tall_f64_512x256 untimed 383.9 ms is 129% of the 297.9 ms timed region (budget 20%) |
| slogdet_positive_f64_256x256 | jax-cuda | slogdet_positive_f64_256x256 untimed 258.3 ms is 743% of the 34.8 ms timed region (budget 20%) |
| householder_qr_tall_f64_512x256 | jax-cuda | householder_qr_tall_f64_512x256 untimed 348.0 ms is 829% of the 42.0 ms timed region (budget 20%) |
| householder_r_default_f64_512x256 | jax-cuda | householder_r_default_f64_512x256 untimed 229.0 ms is 530% of the 43.2 ms timed region (budget 20%) |
| householder_q_columns_prefix_f64_512x256 | jax-cuda | householder_q_columns_prefix_f64_512x256 untimed 166.9 ms is 543% of the 30.8 ms timed region (budget 20%) |