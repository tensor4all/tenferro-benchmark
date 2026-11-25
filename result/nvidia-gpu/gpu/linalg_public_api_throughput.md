# gpu/linalg_public_api_throughput: CUDA public API benchmark

Measured cells: **53**. Unsupported cells: **11**, including **2** numerically verified but timing-excluded cells. Unsupported cells do not count as measured coverage.

Logical traffic: this run measured a plain device copy of 67108864 bytes at **1392.0 GB/s** on this device. Every record carries the logical bytes the case must move per operation and the ratio of the case's implied rate to that copy rate, in the sampling table below. Logical bytes are the declared array sizes, not device DRAM traffic: a case whose operands stay in cache moves less DRAM than its arrays suggest, so a ratio above 1 is recorded, not rejected.

Raw run: `data/results/nvidia-gpu/gpu/linalg_public_api_throughput/20261125_070903/`.

Large-shape view of the CUDA public linalg API surface: the 82 cases whose largest matrix dimension is at least 256. `large` is a shape class, not a measured classification: no row is asserted to be compute- or bandwidth-bound, and a memory-bound row takes the same time here as at a small shape (`householder_r_default_f32` measured 34.698 µs at 512x256 and 35.223 µs at 64x32). These rows are never comparable with, or averaged together with, the per-call overhead rows of `gpu/linalg_public_api_latency` (tenferro-rs #2087). Rows here are not lower than on pages published before this harness change for any timing reason: the sample clock starts inside `measure`, so engine construction was never part of a case median. What changed is repeated **untimed setup** work, which was removed: the CUDA engine is opened once per arm and held for the whole run instead of once per case, which shortens the untimed wall clock around the cases and moves no operation median on this page. The same problems, protocol and backends are split by shape class between the two suites. Issue 155.

This run compiled **8 NVRTC modules** in 2 of its arms, because the CUDA artifact cache started cold. The untimed wall of the cases that compiled is **181.6 s** of its 213.3 s of recorded case wall clock, and that wall also holds their fixture read, transfers and verification, so it is not a compile time.

Values are median (IQR) **µs per operation**. Samples time batches of declared operations plus device synchronization; retained output memory is limited to 256 MiB. Batch counts and raw durations are preserved in JSONL.

Cases marked † (**4** of 16) were sampled with a different number of operations per sample on different arms (see Ops/sample in the sampling table). Each value is still per operation, but the arms amortise launch and synchronization over different batch sizes, so a ratio between their columns is approximate and is not presented as a like-for-like comparison.

Setup, input transfers, library/plan initialization, JIT compilation, validation downloads and output destruction are outside timing. Intrinsic output allocation is inside timing; output-reuse rows instead use preallocated destinations across samples. Accumulating destinations are restored from prepared GPU seeds and synchronized before every batch clock, including initialization and calibration. Direct and eager each use one entered session across samples; direct FFT uses cached `FftExecutor`. JAX eager and compiled JIT are separate arms.

All measured output elements are checked against independently generated NumPy/SciPy reference results before and after sampling. Unsupported rows include unavailable APIs and APIs whose internal setup cannot be excluded; these are not performance measurements. Short batches limited by output memory remain identifiable in the sampling table.

| Case | tenferro-cuda-direct | tenferro-cuda-eager | pytorch-cuda | jax-cuda |
| --- | ---: | ---: | ---: | ---: |
| cholesky_f64_512x512 | 331.417 (2.562) | 356.072 (3.691) | 330.832 (2.726) | 305.352 (1.780) |
| svdvals_f64_512x512 | 77714.777 (50.229) | unsupported | 84845.642 (187.447) | 78586.806 (106.564) |
| eigvalsh_f64_512x512 | 5642.195 (92.551) | 5516.915 (65.236) | 5625.889 (21.115) | 6156.769 (64.046) |
| pinv_f64_512x512 | unsupported | unsupported | 85334.728 (130.259) | 79703.582 (152.850) |
| inv_f64_512x512 | unsupported | unsupported | 1816.146 (15.463) | 1803.101 (9.642) |
| det_f64_512x512 | 1592.049 (4.770) | 1598.647 (12.360) | 1322.754 (3.477) | 1370.155 (15.658) |
| norm_f64_512x512 † | 18.044 (0.342) | 18.619 (1.356) | 11.787 (0.324) | 49.201 (1.117) |
| triangular_solve_f64_512x512 † | 193.791 (0.493) | 211.859 (1.087) | 309.944 (0.521) | 215.734 (2.933) |
| svd_full_tall_f64_512x256 | 19804.948 (131.376) | 19945.875 (89.036) | 20962.588 (130.413) | 20364.359 (96.661) |
| slogdet_positive_f64_256x256 | 663.784 (4.699) | 700.200 (9.299) | 574.685 (1.959) | 561.163 (3.600) |
| householder_qr_tall_f64_512x256 | 1344.098 (3.375) | 1407.020 (10.465) | 1341.089 (11.422) | 1372.220 (11.045) |
| householder_from_factors_default_f64_512x256 | 1493.264 (12.145) | 1574.627 (19.885) | unsupported | unsupported |
| householder_append_default_f64_512x256 | 797.565 (9.016) | 860.704 (8.828) | unsupported | unsupported |
| householder_r_default_f64_512x256 † | 17.802 (0.588) | 27.673 (0.883) | 5.837 (0.190) | 43.045 (1.563) |
| householder_q_columns_prefix_f64_512x256 † | 154.801 (0.256) | 174.935 (6.332) | 83.583 (2.399) | 143.884 (10.132) |
| lu_tall_f64_s256 | 790.065 (5.004) | 811.471 (5.814) | unsupported | unsupported |

## Sampling

| Case | Backend | Ops/sample | Samples | Median batch ms | Bytes/op | Copy ratio | Verification error/tolerance |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cholesky_f64_512x512 | jax-cuda | 8 | 15 | 2.443 | 4194304 | 0.010 | 4.418e-07 |
| cholesky_f64_512x512 | pytorch-cuda | 8 | 15 | 2.647 | 4194304 | 0.009 | 4.418e-07 |
| cholesky_f64_512x512 | tenferro-cuda-direct | 8 | 15 | 2.651 | 4194304 | 0.009 | 4.418e-07 |
| cholesky_f64_512x512 | tenferro-cuda-eager | 8 | 15 | 2.849 | 4194304 | 0.008 | 4.418e-07 |
| det_f64_512x512 | jax-cuda | 2 | 15 | 2.740 | 2097160 | 0.001 | 1.426e-05 |
| det_f64_512x512 | pytorch-cuda | 2 | 15 | 2.646 | 2097160 | 0.001 | 1.369e-05 |
| det_f64_512x512 | tenferro-cuda-direct | 2 | 15 | 3.184 | 2097160 | 0.001 | 1.426e-05 |
| det_f64_512x512 | tenferro-cuda-eager | 2 | 15 | 3.197 | 2097160 | 0.001 | 1.426e-05 |
| eigvalsh_f64_512x512 | jax-cuda | 1 | 15 | 6.157 | 2101248 | 0.000 | 1.55e-06 |
| eigvalsh_f64_512x512 | pytorch-cuda | 1 | 15 | 5.626 | 2101248 | 0.000 | 1.55e-06 |
| eigvalsh_f64_512x512 | tenferro-cuda-direct | 1 | 15 | 5.642 | 2101248 | 0.000 | 1.55e-06 |
| eigvalsh_f64_512x512 | tenferro-cuda-eager | 1 | 15 | 5.517 | 2101248 | 0.000 | 1.55e-06 |
| householder_append_default_f64_512x256 | jax-cuda | — | — | — | 3147776 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_f64_512x256 | pytorch-cuda | — | — | — | 3147776 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_f64_512x256 | tenferro-cuda-direct | 4 | 15 | 3.190 | 3147776 | 0.003 | 9.992e-07 |
| householder_append_default_f64_512x256 | tenferro-cuda-eager | 4 | 15 | 3.443 | 3147776 | 0.003 | 9.992e-07 |
| householder_from_factors_default_f64_512x256 | jax-cuda | — | — | — | 3672064 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_from_factors_default_f64_512x256 | pytorch-cuda | — | — | — | 3672064 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_from_factors_default_f64_512x256 | tenferro-cuda-direct | 2 | 15 | 2.987 | 3672064 | 0.002 | 8.882e-07 |
| householder_from_factors_default_f64_512x256 | tenferro-cuda-eager | 2 | 15 | 3.149 | 3672064 | 0.002 | 8.882e-07 |
| householder_q_columns_prefix_f64_512x256 | jax-cuda | 16 | 15 | 2.302 | 1081344 | 0.005 | 1.136e-07 |
| householder_q_columns_prefix_f64_512x256 | pytorch-cuda | 32 | 15 | 2.675 | 1081344 | 0.009 | 1.136e-07 |
| householder_q_columns_prefix_f64_512x256 | tenferro-cuda-direct | 16 | 15 | 2.477 | 1081344 | 0.005 | 1.136e-07 |
| householder_q_columns_prefix_f64_512x256 | tenferro-cuda-eager | 16 | 15 | 2.799 | 1081344 | 0.004 | 1.136e-07 |
| householder_qr_tall_f64_512x256 | jax-cuda | 2 | 15 | 2.744 | 2099200 | 0.001 | 1.887e-06 |
| householder_qr_tall_f64_512x256 | pytorch-cuda | 2 | 15 | 2.682 | 2099200 | 0.001 | 1.887e-06 |
| householder_qr_tall_f64_512x256 | tenferro-cuda-direct | 2 | 15 | 2.688 | 2099200 | 0.001 | 9.992e-07 |
| householder_qr_tall_f64_512x256 | tenferro-cuda-eager | 2 | 15 | 2.814 | 2099200 | 0.001 | 9.992e-07 |
| householder_r_default_f64_512x256 | jax-cuda | 64 | 15 | 2.755 | 1572864 | 0.026 | 9.976e-07 |
| householder_r_default_f64_512x256 | pytorch-cuda | 128 | 15 | 0.747 | 1572864 | 0.194 | 9.976e-07 |
| householder_r_default_f64_512x256 | tenferro-cuda-direct | 128 | 15 | 2.279 | 1572864 | 0.063 | 9.976e-07 |
| householder_r_default_f64_512x256 | tenferro-cuda-eager | 128 | 15 | 3.542 | 1572864 | 0.041 | 9.976e-07 |
| inv_f64_512x512 | jax-cuda | 2 | 15 | 3.606 | 4194304 | 0.002 | 1.85e-06 |
| inv_f64_512x512 | pytorch-cuda | 2 | 15 | 3.632 | 4194304 | 0.002 | 1.85e-06 |
| inv_f64_512x512 | tenferro-cuda-direct | — | — | — | 4194304 | unavailable | CUDA concrete route fails at revision 8b13220c: inv_owned/eye_like passes a host scalar to broadcast_in_dim. No public caller parameter can replace this internal operand; reproduced in raw run 20261013_094906. |
| inv_f64_512x512 | tenferro-cuda-eager | — | — | — | 4194304 | unavailable | Public linalg composite constructs scalar/identity operands per call; cannot move internal setup outside the timer. |
| lu_tall_f64_s256 | jax-cuda | — | — | — | 4718600 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_tall_f64_s256 | pytorch-cuda | — | — | — | 4718600 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_tall_f64_s256 | tenferro-cuda-direct | 4 | 15 | 3.160 | 4718600 | 0.004 | 1.102e-06 |
| lu_tall_f64_s256 | tenferro-cuda-eager | 4 | 15 | 3.246 | 4718600 | 0.004 | 1.102e-06 |
| norm_f64_512x512 | jax-cuda | 64 | 15 | 3.149 | 2097160 | 0.031 | 1.909e-07 |
| norm_f64_512x512 | pytorch-cuda | 256 | 15 | 3.018 | 2097160 | 0.128 | 1.909e-07 |
| norm_f64_512x512 | tenferro-cuda-direct | 128 | 15 | 2.310 | 2097160 | 0.083 | 0 |
| norm_f64_512x512 | tenferro-cuda-eager | 128 | 15 | 2.383 | 2097160 | 0.081 | 0 |
| pinv_f64_512x512 | jax-cuda | 1 | 15 | 79.704 | 4194320 | 0.000 | 8.975e-06 |
| pinv_f64_512x512 | pytorch-cuda | 1 | 15 | 85.335 | 4194320 | 0.000 | 8.747e-06 |
| pinv_f64_512x512 | tenferro-cuda-direct | — | — | — | 4194320 | unavailable | CUDA concrete route fails at revision 8b13220c: pinv_from_svd passes a host threshold scalar to mul_read. No public caller parameter can replace this internal operand; reproduced in raw run 20261013_094906. |
| pinv_f64_512x512 | tenferro-cuda-eager | — | — | — | 4194320 | unavailable | Public linalg composite constructs scalar/identity operands per call; cannot move internal setup outside the timer. |
| slogdet_positive_f64_256x256 | jax-cuda | 4 | 15 | 2.245 | 524304 | 0.001 | 1.864e-06 |
| slogdet_positive_f64_256x256 | pytorch-cuda | 4 | 15 | 2.299 | 524304 | 0.001 | 1.708e-06 |
| slogdet_positive_f64_256x256 | tenferro-cuda-direct | 4 | 15 | 2.655 | 524304 | 0.001 | 1.864e-06 |
| slogdet_positive_f64_256x256 | tenferro-cuda-eager | 4 | 15 | 2.801 | 524304 | 0.001 | 1.864e-06 |
| svd_full_tall_f64_512x256 | jax-cuda | 1 | 15 | 20.364 | 3672064 | 0.000 | 7.76e-05 |
| svd_full_tall_f64_512x256 | pytorch-cuda | 1 | 15 | 20.963 | 3672064 | 0.000 | 7.76e-05 |
| svd_full_tall_f64_512x256 | tenferro-cuda-direct | 1 | 15 | 19.805 | 3672064 | 0.000 | 7.76e-05 |
| svd_full_tall_f64_512x256 | tenferro-cuda-eager | 1 | 15 | 19.946 | 3672064 | 0.000 | 7.76e-05 |
| svdvals_f64_512x512 | jax-cuda | 1 | 15 | 78.587 | 2101248 | 0.000 | 0.0002063 |
| svdvals_f64_512x512 | pytorch-cuda | 1 | 15 | 84.846 | 2101248 | 0.000 | 0.0002066 |
| svdvals_f64_512x512 | tenferro-cuda-direct | 1 | 15 | 77.715 | 2101248 | 0.000 | 0.0002063 |
| svdvals_f64_512x512 | tenferro-cuda-eager | — | — | — | 2101248 | unavailable | EagerSessionLinalgExt has no values-only svdvals entry point; returning and discarding singular vectors is not equivalent. |
| triangular_solve_f64_512x512 | jax-cuda | 16 | 15 | 3.452 | 2162688 | 0.007 | 1.157e-06 |
| triangular_solve_f64_512x512 | pytorch-cuda | 8 | 15 | 2.480 | 2162688 | 0.005 | 6.241e-07 |
| triangular_solve_f64_512x512 | tenferro-cuda-direct | 16 | 15 | 3.101 | 2162688 | 0.008 | 1.157e-06 |
| triangular_solve_f64_512x512 | tenferro-cuda-eager | 16 | 15 | 3.390 | 2162688 | 0.007 | 1.157e-06 |

## Run metadata

```yaml
schema_version: 1
target_profile: nvidia-gpu
suite_id: gpu/linalg_public_api_throughput
suite_file: /home/shinaoka/tensor4all/tenferro-benchmark-2701/data/results/nvidia-gpu/gpu/routine_selection/20261125_070903/gpu_linalg_public_api_throughput.yaml
timestamp: '2026-11-25T07:09:03+00:00'
tenferro_rs:
  path: /home/shinaoka/tensor4all/tenferro-benchmark-2701/extern/tenferro-rs
  commit: 1ba9f33521691170c763dfce9b6c9bb56ca523c2
  dirty: false
  features:
  - cuda
  resolved_path: /home/shinaoka/tensor4all/tenferro-benchmark-2701/extern/tenferro-rs
harness:
  path: /home/shinaoka/tensor4all/tenferro-benchmark-2701
  commit: acdeaf6e4f6845a465352d7b7c3b67c7dbcf005a
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
  cpu: AMD EPYC 7713P 64-Core Processor
  logical_cpus: 64
  power:
    cpufreq_boost: '1'
    governor: schedutil
    scaling_max_freq_khz: '2000000'
    smt:
      active: '0'
      control: notsupported
  numa:
    nodes: 1
    cpulists:
      '0': 0-63
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
  l3_domains:
  - 0-7
  - 8-15
  - 16-23
  - 24-31
  - 32-39
  - 40-47
  - 48-55
  - 56-63
build:
  toolchain:
    rustc: null
    rustc_release: null
    rustflags: null
    target_cpu: null
environment:
  container: devcontainer
  hostname: primerose
  os: Linux-6.8.0-147-generic-x86_64-with-glibc2.39
  arch: x86_64
  cpu: x86_64
  affinity:
    pinning: none
    requested_measured_cpus: null
    observer_cpus: null
    override_of: null
    thread_budget: '1'
    measured_cpus: 27,29
    l3_domains:
    - 0-7
    - 8-15
    - 16-23
    - 24-31
    - 32-39
    - 40-47
    - 48-55
    - 56-63
    cpu_boost: enabled
    scaling_governor: schedutil
    scaling_driver: acpi-cpufreq
  env:
    OMP_NUM_THREADS: '1'
    OMP_THREAD_LIMIT: '1'
    RAYON_NUM_THREADS: '1'
    BENCHMARK_TARGET_PROFILE: nvidia-gpu
    OPENBLAS_NUM_THREADS: '1'
    MATMUL_NUM_THREADS: '1'
    MKL_NUM_THREADS: '1'
    CUDA_HOME: /opt/tenferro-benchmark-ci/cuda
    CUDA_PATH: /opt/tenferro-benchmark-ci/cuda
    TENFERRO_CUTENSOR_PATH: /opt/tenferro-benchmark-ci/cutensor/lib/libcutensor.so.2
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
  device_uuid: GPU-530977e1-4968-9283-4129-9fbec3e66542
  device_memory_bytes: 85899345920
  driver_version: 580.178.04
  runtime_version: '12.8'
  cuda_version: '13.0'
  toolkit_version: null
  toolkit_root: /opt/tenferro-benchmark-ci/cuda
  cutensor_version: 2.2.0
  cusolver_version: 11.7.3
  cudnn_version: '91900'
  artifact_cache:
    path: /home/shinaoka/tensor4all/.t4a-509-gpu-runtime/home/.cache/tenferro-benchmark/cuda-artifacts/12.8.93/sm_80
    regime: cold
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
  - 0.096351
  - 0.09405
  - 0.098211
  - 0.097471
  - 0.09389
  - 0.098131
  - 0.09726
  - 0.095491
  - 0.09642
  - 0.096841
  - 0.095951
  - 0.09704
  - 0.097351
  - 0.09557
  - 0.095531
  median_ms: 0.09642
  bytes_per_second: 1392011283965.9822
  torch_version: 2.11.0+cu128
  cuda_version: '12.8'
  timestamp: '2026-11-25T07:09:20.646777+00:00'
gpu_assignment:
  role: benchmark
  physical_device_index: 0
  device_uuid: GPU-530977e1-4968-9283-4129-9fbec3e66542
  cuda_visible_devices: GPU-530977e1-4968-9283-4129-9fbec3e66542
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
    src/bin/benchmark_gpu_public_api.rs: b85ad2743b1594fe0a647253a17fa3bb59800202fa8ed4016d0255f0170bf5b3
    src/bin/benchmark_gpu_public_api/lifetime.rs: 34be366f56aac88f5f3b86ade8018a15dad2bca85cea74f56cb2fe121917d0a9
    src/bin/benchmark_gpu_public_api/linalg.rs: 3d5bedcc3620f75cab2687fdf49210df324021d043b23aec6e169642ceff15ba
    src/bin/benchmark_gpu_public_api/multi.rs: 950cb1b3c1a7cca37665b58e3de074bd906dafb8c84b17a51c27ce797a73afa8
    src/bin/benchmark_gpu_public_api/plan.rs: 016e752c59edefc3ded1405c36325e8680736860cb65954919ea4cf0f3eed061
    src/bin/benchmark_gpu_public_api/read_layout.rs: fc42e2160094ba5f9c7dd6facb5667a76e10b73f971ef9facfb1f64723e42467
    src/bin/benchmark_gpu_public_api/runtime_ops.rs: 88038508963fbc13303fb54b3898808e4303131321cabedb9f38255941ead438
    src/bin/benchmark_gpu_public_api/state.rs: e1cc2781ce7392154f32cc19372c9fc41ae29765aedc544c7ae7bf02f873c12b
    src/bin/benchmark_gpu_public_api/structural.rs: 10850458a640d11e79262c215072482ee463bc4ca0604f40d82ae56f345f6564
    src/bin/benchmark_gpu_public_api/reuse.rs: 8934cd12e336764b5ed831f0c92d55f2827142de939a1f33daa5d9ee57b9f482
    /home/shinaoka/tensor4all/tenferro-benchmark-2701/scripts/benchmark_gpu_public_api.py: d1ff68f86496d991bcbc730138fe6e3076001b3f23cc1a94c44b96f6857cb8ed
    /home/shinaoka/tensor4all/tenferro-benchmark-2701/scripts/gpu_public_api_ops.py: e88823db3f4a1fe43dc13c4743c9d78ece6af76a968724091af318dd8d7f2528
    scripts/run_gpu_public_api.sh: 7995bcf409258185e07210969a738896f64860b67df251fcd1100d11128a5431
    data/results/nvidia-gpu/gpu/linalg_public_api_throughput/20261125_070903/fixtures/manifest.json: f00b94c1ee004d7b9c56463b7ff2314ce698db43cf403aeab4fba564a31668a2
    target/release/benchmark_gpu_public_api: 250c27c741c6e6364e4ff58525a6341f7e6a3c3b81e6f429f4c5662a4d8f0f86
  reference_numpy_version: 2.4.2
  measurement_cpu:
    policy: 'disabled (GPU_BENCH_CPU_SELECT unset): arms run on the allocation unpinned'
    arms:
      jax-cuda:
        arm: jax-cuda
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-25T07:13:06Z'
      pytorch-cuda:
        arm: pytorch-cuda
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-25T07:12:52Z'
      tenferro-cuda-direct:
        arm: tenferro-cuda-direct
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-25T07:10:49Z'
      tenferro-cuda-eager:
        arm: tenferro-cuda-eager
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-25T07:12:40Z'
  compile:
    case_wall_seconds: 213.294
    compiling_processes: 2
    modules_compiled: 8
    compile_time_ms: null
    compile_time_scope: 'not measured: no compile interval was recorded; `compile_modules`
      counts the NVRTC modules and the untimed wall is the case wall'
    untimed_wall_seconds: 181.636
```

## Untimed work over the recorded budget

Each row records the wall clock it spent outside its timed region. These cases spent more than the recorded budget ratio (0.2) there. The threshold has no measured basis, so this is a recorded warning: it does not change a cell, its acceptance or the exit status of the run.

| Case | Backend | Warning |
| --- | --- | --- |
| cholesky_f64_512x512 | tenferro-cuda-direct | cholesky_f64_512x512 untimed 3593.1 ms is 9028% of the 39.8 ms timed region (budget 20%) |
| svdvals_f64_512x512 | tenferro-cuda-direct | svdvals_f64_512x512 untimed 567.4 ms is 49% of the 1165.9 ms timed region (budget 20%) |
| eigvalsh_f64_512x512 | tenferro-cuda-direct | eigvalsh_f64_512x512 untimed 68.0 ms is 80% of the 84.9 ms timed region (budget 20%) |
| det_f64_512x512 | tenferro-cuda-direct | det_f64_512x512 untimed 62132.8 ms is 130220% of the 47.7 ms timed region (budget 20%) |
| norm_f64_512x512 | tenferro-cuda-direct | norm_f64_512x512 untimed 10.7 ms is 31% of the 34.8 ms timed region (budget 20%) |
| triangular_solve_f64_512x512 | tenferro-cuda-direct | triangular_solve_f64_512x512 untimed 17.1 ms is 37% of the 46.5 ms timed region (budget 20%) |
| svd_full_tall_f64_512x256 | tenferro-cuda-direct | svd_full_tall_f64_512x256 untimed 931.2 ms is 314% of the 297.0 ms timed region (budget 20%) |
| slogdet_positive_f64_256x256 | tenferro-cuda-direct | slogdet_positive_f64_256x256 untimed 10.8 ms is 27% of the 39.9 ms timed region (budget 20%) |
| householder_qr_tall_f64_512x256 | tenferro-cuda-direct | householder_qr_tall_f64_512x256 untimed 13884.7 ms is 34446% of the 40.3 ms timed region (budget 20%) |
| householder_from_factors_default_f64_512x256 | tenferro-cuda-direct | householder_from_factors_default_f64_512x256 untimed 570.4 ms is 1267% of the 45.0 ms timed region (budget 20%) |
| householder_append_default_f64_512x256 | tenferro-cuda-direct | householder_append_default_f64_512x256 untimed 584.5 ms is 1218% of the 48.0 ms timed region (budget 20%) |
| householder_r_default_f64_512x256 | tenferro-cuda-direct | householder_r_default_f64_512x256 untimed 12.7 ms is 37% of the 34.2 ms timed region (budget 20%) |
| householder_q_columns_prefix_f64_512x256 | tenferro-cuda-direct | householder_q_columns_prefix_f64_512x256 untimed 23.5 ms is 63% of the 37.1 ms timed region (budget 20%) |
| lu_tall_f64_s256 | tenferro-cuda-direct | lu_tall_f64_s256 untimed 39.4 ms is 83% of the 47.5 ms timed region (budget 20%) |
| cholesky_f64_512x512 | tenferro-cuda-eager | cholesky_f64_512x512 untimed 102025.8 ms is 238835% of the 42.7 ms timed region (budget 20%) |
| eigvalsh_f64_512x512 | tenferro-cuda-eager | eigvalsh_f64_512x512 untimed 51.4 ms is 62% of the 82.9 ms timed region (budget 20%) |
| det_f64_512x512 | tenferro-cuda-eager | det_f64_512x512 untimed 18.0 ms is 38% of the 47.9 ms timed region (budget 20%) |
| norm_f64_512x512 | tenferro-cuda-eager | norm_f64_512x512 untimed 9.9 ms is 27% of the 36.3 ms timed region (budget 20%) |
| triangular_solve_f64_512x512 | tenferro-cuda-eager | triangular_solve_f64_512x512 untimed 11.4 ms is 22% of the 50.9 ms timed region (budget 20%) |
| svd_full_tall_f64_512x256 | tenferro-cuda-eager | svd_full_tall_f64_512x256 untimed 903.3 ms is 301% of the 299.7 ms timed region (budget 20%) |
| slogdet_positive_f64_256x256 | tenferro-cuda-eager | slogdet_positive_f64_256x256 untimed 11.1 ms is 27% of the 41.9 ms timed region (budget 20%) |
| householder_qr_tall_f64_512x256 | tenferro-cuda-eager | householder_qr_tall_f64_512x256 untimed 600.7 ms is 1423% of the 42.2 ms timed region (budget 20%) |
| householder_from_factors_default_f64_512x256 | tenferro-cuda-eager | householder_from_factors_default_f64_512x256 untimed 625.5 ms is 1328% of the 47.1 ms timed region (budget 20%) |
| householder_append_default_f64_512x256 | tenferro-cuda-eager | householder_append_default_f64_512x256 untimed 621.7 ms is 1200% of the 51.8 ms timed region (budget 20%) |
| householder_r_default_f64_512x256 | tenferro-cuda-eager | householder_r_default_f64_512x256 untimed 24.4 ms is 46% of the 53.3 ms timed region (budget 20%) |
| householder_q_columns_prefix_f64_512x256 | tenferro-cuda-eager | householder_q_columns_prefix_f64_512x256 untimed 28.0 ms is 67% of the 42.0 ms timed region (budget 20%) |
| lu_tall_f64_s256 | tenferro-cuda-eager | lu_tall_f64_s256 untimed 72.1 ms is 148% of the 48.7 ms timed region (budget 20%) |
| cholesky_f64_512x512 | pytorch-cuda | cholesky_f64_512x512 untimed 579.7 ms is 1458% of the 39.7 ms timed region (budget 20%) |
| svdvals_f64_512x512 | pytorch-cuda | svdvals_f64_512x512 untimed 627.1 ms is 49% of the 1273.1 ms timed region (budget 20%) |
| eigvalsh_f64_512x512 | pytorch-cuda | eigvalsh_f64_512x512 untimed 83.0 ms is 98% of the 84.5 ms timed region (budget 20%) |
| pinv_f64_512x512 | pytorch-cuda | pinv_f64_512x512 untimed 809.5 ms is 63% of the 1280.8 ms timed region (budget 20%) |
| inv_f64_512x512 | pytorch-cuda | inv_f64_512x512 untimed 119.8 ms is 220% of the 54.5 ms timed region (budget 20%) |
| det_f64_512x512 | pytorch-cuda | det_f64_512x512 untimed 948.7 ms is 2390% of the 39.7 ms timed region (budget 20%) |
| norm_f64_512x512 | pytorch-cuda | norm_f64_512x512 untimed 67.9 ms is 150% of the 45.3 ms timed region (budget 20%) |
| triangular_solve_f64_512x512 | pytorch-cuda | triangular_solve_f64_512x512 untimed 24.7 ms is 66% of the 37.2 ms timed region (budget 20%) |
| svd_full_tall_f64_512x256 | pytorch-cuda | svd_full_tall_f64_512x256 untimed 327.3 ms is 104% of the 314.6 ms timed region (budget 20%) |
| slogdet_positive_f64_256x256 | pytorch-cuda | slogdet_positive_f64_256x256 untimed 70.5 ms is 204% of the 34.5 ms timed region (budget 20%) |
| householder_qr_tall_f64_512x256 | pytorch-cuda | householder_qr_tall_f64_512x256 untimed 360.8 ms is 897% of the 40.2 ms timed region (budget 20%) |
| householder_r_default_f64_512x256 | pytorch-cuda | householder_r_default_f64_512x256 untimed 23.2 ms is 208% of the 11.2 ms timed region (budget 20%) |
| householder_q_columns_prefix_f64_512x256 | pytorch-cuda | householder_q_columns_prefix_f64_512x256 untimed 41.4 ms is 103% of the 40.3 ms timed region (budget 20%) |
| cholesky_f64_512x512 | jax-cuda | cholesky_f64_512x512 untimed 1251.8 ms is 3415% of the 36.7 ms timed region (budget 20%) |
| svdvals_f64_512x512 | jax-cuda | svdvals_f64_512x512 untimed 700.7 ms is 59% of the 1178.9 ms timed region (budget 20%) |
| eigvalsh_f64_512x512 | jax-cuda | eigvalsh_f64_512x512 untimed 239.4 ms is 259% of the 92.3 ms timed region (budget 20%) |
| pinv_f64_512x512 | jax-cuda | pinv_f64_512x512 untimed 2916.7 ms is 244% of the 1195.7 ms timed region (budget 20%) |
| inv_f64_512x512 | jax-cuda | inv_f64_512x512 untimed 456.6 ms is 842% of the 54.2 ms timed region (budget 20%) |
| det_f64_512x512 | jax-cuda | det_f64_512x512 untimed 383.5 ms is 933% of the 41.1 ms timed region (budget 20%) |
| norm_f64_512x512 | jax-cuda | norm_f64_512x512 untimed 214.4 ms is 454% of the 47.2 ms timed region (budget 20%) |
| triangular_solve_f64_512x512 | jax-cuda | triangular_solve_f64_512x512 untimed 254.6 ms is 490% of the 51.9 ms timed region (budget 20%) |
| svd_full_tall_f64_512x256 | jax-cuda | svd_full_tall_f64_512x256 untimed 591.5 ms is 193% of the 305.9 ms timed region (budget 20%) |
| slogdet_positive_f64_256x256 | jax-cuda | slogdet_positive_f64_256x256 untimed 451.3 ms is 1341% of the 33.7 ms timed region (budget 20%) |
| householder_qr_tall_f64_512x256 | jax-cuda | householder_qr_tall_f64_512x256 untimed 391.0 ms is 950% of the 41.1 ms timed region (budget 20%) |
| householder_r_default_f64_512x256 | jax-cuda | householder_r_default_f64_512x256 untimed 316.9 ms is 762% of the 41.6 ms timed region (budget 20%) |
| householder_q_columns_prefix_f64_512x256 | jax-cuda | householder_q_columns_prefix_f64_512x256 untimed 283.2 ms is 819% of the 34.6 ms timed region (budget 20%) |