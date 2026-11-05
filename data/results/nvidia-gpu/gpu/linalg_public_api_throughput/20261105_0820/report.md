# gpu/linalg_public_api_throughput: CUDA public API benchmark

Measured cells: **49**. Unsupported cells: **15**, including **6** numerically verified but timing-excluded cells. Unsupported cells do not count as measured coverage.

Logical traffic: this run measured a plain device copy of 67108864 bytes at **1417.2 GB/s** on this device. Every record carries the logical bytes the case must move per operation and the ratio of the case's implied rate to that copy rate, in the sampling table below. Logical bytes are the declared array sizes, not device DRAM traffic: a case whose operands stay in cache moves less DRAM than its arrays suggest, so a ratio above 1 is recorded, not rejected.

Raw run: `data/results/nvidia-gpu/gpu/linalg_public_api_throughput/20261105_0820/`.

Large-shape view of the CUDA public linalg API surface: the 82 cases whose largest matrix dimension is at least 256. `large` is a shape class, not a measured classification: no row is asserted to be compute- or bandwidth-bound, and a memory-bound row takes the same time here as at a small shape (`householder_r_default_f32` measured 34.698 µs at 512x256 and 35.223 µs at 64x32). These rows are never comparable with, or averaged together with, the per-call overhead rows of `gpu/linalg_public_api_latency` (tenferro-rs #2087). Rows here are lower than on pages published before this harness change because the harness used to open the CUDA engine once per case inside the timed region and now opens it once per arm, so the difference is removed harness cost and not a change in the kernels. The same problems, protocol and backends are split by shape class between the two suites. Issue 155.

Values are median (IQR) **µs per operation**. Samples time batches of declared operations plus device synchronization; retained output memory is limited to 256 MiB. Batch counts and raw durations are preserved in JSONL.

Setup, input transfers, library/plan initialization, JIT compilation, validation downloads and output destruction are outside timing. Intrinsic output allocation is inside timing; output-reuse rows instead use preallocated destinations across samples. Accumulating destinations are restored from prepared GPU seeds and synchronized before every batch clock, including initialization and calibration. Direct and eager each use one entered session across samples; direct FFT uses cached `FftExecutor`. JAX eager and compiled JIT are separate arms.

All measured output elements are checked against independently generated NumPy/SciPy reference results before and after sampling. Unsupported rows include unavailable APIs and APIs whose internal setup cannot be excluded; these are not performance measurements. Short batches limited by output memory remain identifiable in the sampling table.

| Case | tenferro-cuda-direct | tenferro-cuda-eager | pytorch-cuda | jax-cuda |
| --- | ---: | ---: | ---: | ---: |
| cholesky_f64_512x512 | 2286.308 (26.546) | 2677.190 (228.568) | 331.410 (1.440) | 301.092 (1.334) |
| svdvals_f64_512x512 | 77906.682 (9.561) | unsupported | 84489.526 (28.421) | 77955.028 (38.487) |
| eigvalsh_f64_512x512 | 5589.774 (20.045) | 6348.330 (46.202) | 5595.640 (15.251) | 6059.887 (95.378) |
| pinv_f64_512x512 | unsupported | unsupported | 84760.315 (29.571) | 78265.488 (37.912) |
| inv_f64_512x512 | unsupported | unsupported | 1804.683 (2.915) | 1847.770 (4.757) |
| det_f64_512x512 | 1818.064 (6.863) | 1895.746 (7.038) | 1313.309 (4.733) | 1346.771 (60.275) |
| norm_f64_512x512 | 96.421 (0.289) | 130.924 (1.513) | 10.873 (0.131) | 56.680 (12.099) |
| triangular_solve_f64_512x512 | 194.015 (0.829) | 249.865 (1.460) | 308.728 (0.138) | 217.352 (0.557) |
| svd_full_tall_f64_512x256 | 19632.296 (9.755) | 19716.772 (18.567) | 19896.178 (15.911) | 19675.679 (11.726) |
| slogdet_positive_f64_256x256 | 839.654 (13.025) | 916.424 (12.258) | 569.264 (0.983) | 555.308 (3.063) |
| householder_qr_tall_f64_512x256 | 1334.745 (5.680) | 1390.152 (7.840) | 1333.030 (4.778) | 1429.444 (30.841) |
| householder_from_factors_default_f64_512x256 | 1573.074 (20.313) | 1665.258 (18.136) | unsupported | unsupported |
| householder_append_default_f64_512x256 | unsupported | unsupported | unsupported | unsupported |
| householder_r_default_f64_512x256 | 39.412 (0.340) | 56.072 (1.095) | 5.686 (0.123) | 42.076 (0.304) |
| householder_q_columns_prefix_f64_512x256 | unsupported | unsupported | 83.767 (3.140) | 239.429 (105.966) |
| lu_tall_f64_s256 | 847.467 (1.987) | 899.851 (11.224) | unsupported | unsupported |

## Sampling

| Case | Backend | Ops/sample | Samples | Median batch ms | Bytes/op | Copy ratio | Verification error/tolerance |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cholesky_f64_512x512 | jax-cuda | 8 | 15 | 2.409 | 4194304 | 0.010 | 4.418e-07 |
| cholesky_f64_512x512 | pytorch-cuda | 8 | 15 | 2.651 | 4194304 | 0.009 | 4.418e-07 |
| cholesky_f64_512x512 | tenferro-cuda-direct | 1 | 15 | 2.286 | 4194304 | 0.001 | 4.414e-07 |
| cholesky_f64_512x512 | tenferro-cuda-eager | 1 | 15 | 2.677 | 4194304 | 0.001 | 4.414e-07 |
| det_f64_512x512 | jax-cuda | 2 | 15 | 2.694 | 2097160 | 0.001 | 1.426e-05 |
| det_f64_512x512 | pytorch-cuda | 2 | 15 | 2.627 | 2097160 | 0.001 | 1.369e-05 |
| det_f64_512x512 | tenferro-cuda-direct | 2 | 15 | 3.636 | 2097160 | 0.001 | 1.426e-05 |
| det_f64_512x512 | tenferro-cuda-eager | 2 | 15 | 3.791 | 2097160 | 0.001 | 1.426e-05 |
| eigvalsh_f64_512x512 | jax-cuda | 1 | 15 | 6.060 | 2101248 | 0.000 | 1.55e-06 |
| eigvalsh_f64_512x512 | pytorch-cuda | 1 | 15 | 5.596 | 2101248 | 0.000 | 1.55e-06 |
| eigvalsh_f64_512x512 | tenferro-cuda-direct | 1 | 15 | 5.590 | 2101248 | 0.000 | 1.55e-06 |
| eigvalsh_f64_512x512 | tenferro-cuda-eager | 1 | 15 | 6.348 | 2101248 | 0.000 | 1.55e-06 |
| householder_append_default_f64_512x256 | jax-cuda | — | — | — | 3147776 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_f64_512x256 | pytorch-cuda | — | — | — | 3147776 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_f64_512x256 | tenferro-cuda-direct | — | — | — | 3147776 | unavailable | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_append_default_f64_512x256 | tenferro-cuda-eager | — | — | — | 3147776 | unavailable | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_from_factors_default_f64_512x256 | jax-cuda | — | — | — | 3672064 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_from_factors_default_f64_512x256 | pytorch-cuda | — | — | — | 3672064 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_from_factors_default_f64_512x256 | tenferro-cuda-direct | 2 | 15 | 3.146 | 3672064 | 0.002 | 8.882e-07 |
| householder_from_factors_default_f64_512x256 | tenferro-cuda-eager | 2 | 15 | 3.331 | 3672064 | 0.002 | 8.882e-07 |
| householder_q_columns_prefix_f64_512x256 | jax-cuda | 8 | 15 | 1.915 | 1081344 | 0.003 | 1.136e-07 |
| householder_q_columns_prefix_f64_512x256 | pytorch-cuda | 32 | 15 | 2.681 | 1081344 | 0.009 | 1.136e-07 |
| householder_q_columns_prefix_f64_512x256 | tenferro-cuda-direct | — | — | — | 1081344 | unavailable | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_q_columns_prefix_f64_512x256 | tenferro-cuda-eager | — | — | — | 1081344 | unavailable | CUDA reflector application constructs/uploads BLAS scalar operands per call; public state API cannot hoist that input setup outside standard timing. |
| householder_qr_tall_f64_512x256 | jax-cuda | 2 | 15 | 2.859 | 2099200 | 0.001 | 1.887e-06 |
| householder_qr_tall_f64_512x256 | pytorch-cuda | 2 | 15 | 2.666 | 2099200 | 0.001 | 1.887e-06 |
| householder_qr_tall_f64_512x256 | tenferro-cuda-direct | 2 | 15 | 2.669 | 2099200 | 0.001 | 9.992e-07 |
| householder_qr_tall_f64_512x256 | tenferro-cuda-eager | 2 | 15 | 2.780 | 2099200 | 0.001 | 9.992e-07 |
| householder_r_default_f64_512x256 | jax-cuda | 64 | 15 | 2.693 | 1572864 | 0.026 | 9.976e-07 |
| householder_r_default_f64_512x256 | pytorch-cuda | 128 | 15 | 0.728 | 1572864 | 0.195 | 9.976e-07 |
| householder_r_default_f64_512x256 | tenferro-cuda-direct | 64 | 15 | 2.522 | 1572864 | 0.028 | 9.976e-07 |
| householder_r_default_f64_512x256 | tenferro-cuda-eager | 64 | 15 | 3.589 | 1572864 | 0.020 | 9.976e-07 |
| inv_f64_512x512 | jax-cuda | 2 | 15 | 3.696 | 4194304 | 0.002 | 1.85e-06 |
| inv_f64_512x512 | pytorch-cuda | 2 | 15 | 3.609 | 4194304 | 0.002 | 1.85e-06 |
| inv_f64_512x512 | tenferro-cuda-direct | — | — | — | 4194304 | unavailable | CUDA concrete route fails at revision 8b13220c: inv_owned/eye_like passes a host scalar to broadcast_in_dim. No public caller parameter can replace this internal operand; reproduced in raw run 20261013_094906. |
| inv_f64_512x512 | tenferro-cuda-eager | — | — | — | 4194304 | unavailable | Public linalg composite constructs scalar/identity operands per call; cannot move internal setup outside the timer. |
| lu_tall_f64_s256 | jax-cuda | — | — | — | 4718600 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_tall_f64_s256 | pytorch-cuda | — | — | — | 4718600 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_tall_f64_s256 | tenferro-cuda-direct | 4 | 15 | 3.390 | 4718600 | 0.004 | 1.102e-06 |
| lu_tall_f64_s256 | tenferro-cuda-eager | 4 | 15 | 3.599 | 4718600 | 0.004 | 1.102e-06 |
| norm_f64_512x512 | jax-cuda | 32 | 15 | 1.814 | 2097160 | 0.026 | 1.909e-07 |
| norm_f64_512x512 | pytorch-cuda | 256 | 15 | 2.784 | 2097160 | 0.136 | 1.909e-07 |
| norm_f64_512x512 | tenferro-cuda-direct | 32 | 15 | 3.085 | 2097160 | 0.015 | 1.909e-07 |
| norm_f64_512x512 | tenferro-cuda-eager | 16 | 15 | 2.095 | 2097160 | 0.011 | 1.909e-07 |
| pinv_f64_512x512 | jax-cuda | 1 | 15 | 78.265 | 4194320 | 0.000 | 8.975e-06 |
| pinv_f64_512x512 | pytorch-cuda | 1 | 15 | 84.760 | 4194320 | 0.000 | 8.747e-06 |
| pinv_f64_512x512 | tenferro-cuda-direct | — | — | — | 4194320 | unavailable | CUDA concrete route fails at revision 8b13220c: pinv_from_svd passes a host threshold scalar to mul_read. No public caller parameter can replace this internal operand; reproduced in raw run 20261013_094906. |
| pinv_f64_512x512 | tenferro-cuda-eager | — | — | — | 4194320 | unavailable | Public linalg composite constructs scalar/identity operands per call; cannot move internal setup outside the timer. |
| slogdet_positive_f64_256x256 | jax-cuda | 4 | 15 | 2.221 | 524304 | 0.001 | 1.864e-06 |
| slogdet_positive_f64_256x256 | pytorch-cuda | 4 | 15 | 2.277 | 524304 | 0.001 | 1.708e-06 |
| slogdet_positive_f64_256x256 | tenferro-cuda-direct | 4 | 15 | 3.359 | 524304 | 0.000 | 1.864e-06 |
| slogdet_positive_f64_256x256 | tenferro-cuda-eager | 4 | 15 | 3.666 | 524304 | 0.000 | 1.864e-06 |
| svd_full_tall_f64_512x256 | jax-cuda | 1 | 15 | 19.676 | 3672064 | 0.000 | 7.76e-05 |
| svd_full_tall_f64_512x256 | pytorch-cuda | 1 | 15 | 19.896 | 3672064 | 0.000 | 7.76e-05 |
| svd_full_tall_f64_512x256 | tenferro-cuda-direct | 1 | 15 | 19.632 | 3672064 | 0.000 | 7.76e-05 |
| svd_full_tall_f64_512x256 | tenferro-cuda-eager | 1 | 15 | 19.717 | 3672064 | 0.000 | 7.76e-05 |
| svdvals_f64_512x512 | jax-cuda | 1 | 15 | 77.955 | 2101248 | 0.000 | 0.0002063 |
| svdvals_f64_512x512 | pytorch-cuda | 1 | 15 | 84.490 | 2101248 | 0.000 | 0.0002066 |
| svdvals_f64_512x512 | tenferro-cuda-direct | 1 | 15 | 77.907 | 2101248 | 0.000 | 0.0002063 |
| svdvals_f64_512x512 | tenferro-cuda-eager | — | — | — | 2101248 | unavailable | EagerSessionLinalgExt has no values-only svdvals entry point; returning and discarding singular vectors is not equivalent. |
| triangular_solve_f64_512x512 | jax-cuda | 16 | 15 | 3.478 | 2162688 | 0.007 | 1.157e-06 |
| triangular_solve_f64_512x512 | pytorch-cuda | 8 | 15 | 2.470 | 2162688 | 0.005 | 6.241e-07 |
| triangular_solve_f64_512x512 | tenferro-cuda-direct | 16 | 15 | 3.104 | 2162688 | 0.008 | 1.157e-06 |
| triangular_solve_f64_512x512 | tenferro-cuda-eager | 16 | 15 | 3.998 | 2162688 | 0.006 | 1.157e-06 |

## Run metadata

```yaml
schema_version: 1
target_profile: nvidia-gpu
suite_id: gpu/linalg_public_api_throughput
suite_file: /home/shinaoka/work/wallclock/data/results/nvidia-gpu/gpu/routine_selection/20261105_0820/gpu_linalg_public_api_throughput.yaml
timestamp: '2026-11-05T08:02:00+00:00'
tenferro_rs:
  path: /home/shinaoka/work/wallclock/extern/tenferro-rs
  commit: a000e5c4929d5bebb91e783c329b27106baed12e
  dirty: false
  features:
  - cuda
  resolved_path: /home/shinaoka/work/wallclock/extern/tenferro-rs
harness:
  path: /home/shinaoka/work/wallclock
  commit: 18ae0c54c3f6cc4493988efe87a65148334a5c77
  dirty: false
resolved_deps:
  strided-perm:
  - 0.8.0
  - 0.10.0
  strided-view:
  - 0.7.0
  threadteam:
  - 0.1.0
  - 0.3.0
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
    rustc: 'rustc 1.99.1 (58cf8f9c3 2026-10-13)

      binary: rustc

      commit-hash: 58cf8f9c35b355199727627d07ae34a4e2846d67

      commit-date: 2026-10-13

      host: x86_64-unknown-linux-gnu

      release: 1.99.1

      LLVM version: 23.1.1

      '
    rustc_release: 1.99.1
    rustflags: null
    target_cpu: null
environment:
  container: none
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
    measured_cpus: 0-63
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
    RAYON_NUM_THREADS: '1'
    BENCHMARK_TARGET_PROFILE: nvidia-gpu
    OPENBLAS_NUM_THREADS: '1'
    MATMUL_NUM_THREADS: '1'
    CUDA_HOME: /usr/local/cuda-12.6
    CUDA_PATH: /usr/local/cuda-12.6
    TENFERRO_CUTENSOR_PATH: /usr/lib/x86_64-linux-gnu/libcutensor/12/libcutensor.so.2
    UV_PROJECT_ENVIRONMENT: /home/shinaoka/tensor4all/tenferro-benchmark-gpuq/.venv
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
    version: 2.12.0+cu130
    file: /home/shinaoka/tensor4all/tenferro-benchmark-gpuq/.venv/lib/python3.14/site-packages/torch/__init__.py
    provider: mkl
    blas_info: mkl
    lapack_info: mkl
    config: "PyTorch built with:\n  - GCC 13.3\n  - C++ Version: 202002\n  - Intel(R)\
      \ oneAPI Math Kernel Library Version 2024.2-Product Build 20240605 for Intel(R)\
      \ 64 architecture applications\n  - Intel(R) MKL-DNN v3.11.2 (Git Hash 03c022d3ffdcee958cfacbe720048e725fdf644c)\n\
      \  - OpenMP 201511 (a.k.a. OpenMP 4.5)\n  - LAPACK is enabled (usually provided\
      \ by MKL)\n  - NNPACK is enabled\n  - CPU capability usage: AVX2\n  - CUDA Runtime\
      \ 13.0\n  - NVCC architecture flags: -gencode;arch=compute_75,code=sm_75;-gencode;arch=compute_80,code=sm_80;-gencode;arch=compute_86,code=sm_86;-gencode;arch=compute_90,code=sm_90;-gencode;arch=compute_100,code=sm_100;-gencode;arch=compute_120,code=sm_120\n\
      \  - CuDNN 92.7  (built against CUDA 12.9)\n    - Built with CuDNN 92.0\n  -\
      \ Magma 2.6.1\n  - Build settings: BLAS_INFO=mkl, BUILD_TYPE=Release, COMMIT_SHA=7661cd9c6b841b62b7f411aa52ec51f05457263b,\
      \ CUDA_FLAGS= -DLIBCUDACXX_ENABLE_SIMPLIFIED_COMPLEX_OPERATIONS -Xfatbin -compress-all\
      \ -DONNX_NAMESPACE=onnx_torch -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80\
      \ -gencode arch=compute_86,code=sm_86 -gencode arch=compute_90,code=sm_90 -gencode\
      \ arch=compute_100,code=sm_100 -gencode arch=compute_120,code=sm_120 -Xcudafe\
      \ --diag_suppress=cc_clobber_ignored,--diag_suppress=field_without_dll_interface,--diag_suppress=base_class_has_different_dll_interface,--diag_suppress=dll_interface_conflict_none_assumed,--diag_suppress=dll_interface_conflict_dllexport_assumed,--diag_suppress=bad_friend_decl\
      \ --expt-relaxed-constexpr --expt-extended-lambda -Xfatbin -compress-all --threads\
      \ 2 -compress-mode=size -Wno-deprecated-gpu-targets --expt-extended-lambda -DCUB_WRAPPED_NAMESPACE=at_cuda_detail\
      \ -DDISABLE_CUSPARSE_DEPRECATED -DCUDA_HAS_FP16=1 -D__CUDA_NO_HALF_OPERATORS__\
      \ -D__CUDA_NO_HALF_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ -D__CUDA_NO_BFLOAT16_CONVERSIONS__\
      \ -DC10_NODEPRECATED, CUDA_VERSION=13.0, CUDNN_VERSION=9.20.0, CXX_COMPILER=/opt/rh/gcc-toolset-13/root/usr/bin/c++,\
      \ CXX_FLAGS= -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -DNDEBUG -DUSE_KINETO\
      \ -DLIBKINETO_NOROCTRACER -DLIBKINETO_NOXPUPTI=ON -DUSE_FBGEMM -DUSE_MSLK -DUSE_PYTORCH_QNNPACK\
      \ -DUSE_XNNPACK -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -O2 -fPIC -DC10_NODEPRECATED\
      \ -Wall -Wextra -Werror=return-type -Werror=non-virtual-dtor -Werror=range-loop-construct\
      \ -Werror=bool-operation -Wnarrowing -Wno-missing-field-initializers -Wno-unknown-pragmas\
      \ -Wno-unused-parameter -Wno-strict-overflow -Wno-strict-aliasing -Wno-stringop-overflow\
      \ -Wsuggest-override -Wno-psabi -Wno-error=old-style-cast -faligned-new -Wno-maybe-uninitialized\
      \ -fno-math-errno -fno-trapping-math -Werror=format -Wno-dangling-reference\
      \ -Wno-error=dangling-reference -Wno-stringop-overflow, LAPACK_INFO=mkl, PERF_WITH_AVX=1,\
      \ PERF_WITH_AVX2=1, TORCH_VERSION=2.12.0, USE_CUDA=ON, USE_CUDNN=ON, USE_CUSPARSELT=1,\
      \ USE_GFLAGS=OFF, USE_GLOG=OFF, USE_GLOO=ON, USE_MKL=ON, USE_MKLDNN=ON, USE_MPI=OFF,\
      \ USE_NCCL=1, USE_NNPACK=ON, USE_OPENMP=ON, USE_ROCM=OFF, USE_ROCM_KERNEL_ASSERT=OFF,\
      \ USE_XCCL=OFF, USE_XPU=OFF, \n"
    library: /home/shinaoka/tensor4all/tenferro-benchmark-gpuq/.venv/lib/python3.14/site-packages/torch/lib/libtorch_cpu.so
    linked_libraries:
    - /home/shinaoka/tensor4all/tenferro-benchmark-gpuq/.venv/lib/python3.14/site-packages/torch/lib/libgomp.so.1
  jax:
    available: true
    version: 0.10.1
    jaxlib_version: 0.10.1
    backend: gpu
    file: /home/shinaoka/tensor4all/tenferro-benchmark-gpuq/.venv/lib/python3.14/site-packages/jaxlib/__init__.py
    provider: xla_gpu
    dot_backend: xla_gpu
    lapack_provider: none_detected
    library: /home/shinaoka/tensor4all/tenferro-benchmark-gpuq/.venv/lib/python3.14/site-packages/jaxlib/_ifrt_proxy.so
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
  runtime_version: '12.6'
  cuda_version: '13.0'
  toolkit_version: '12.6'
  toolkit_root: /usr/local/cuda-12.6
  cutensor_version: 2.8.0
  cusolver_version: 11.7.1
  cudnn_version: '92700'
  artifact_cache:
    path: null
    regime: 'off'
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
  - 0.093783
  - 0.092824
  - 0.096194
  - 0.094703
  - 0.092683
  - 0.096664
  - 0.094734
  - 0.093253
  - 0.095663
  - 0.094634
  - 0.094324
  - 0.095083
  - 0.094873
  - 0.093464
  - 0.096044
  median_ms: 0.094703
  bytes_per_second: 1417248957266.4011
  torch_version: 2.12.0+cu130
  cuda_version: '13.0'
  timestamp: '2026-11-05T07:47:44.722590+00:00'
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
    .devcontainer/cuda/devcontainer.json: 0bf1465bbd1a4c9f896d5320838f2d5a488632a59af7f9b59cc050e95e0b8e09
    .devcontainer/cuda/Dockerfile: 578d9815e5145aa1651ac7e93833468c345ee8bf5e95bfa7717551580ae9feca
    src/bin/benchmark_gpu_public_api.rs: 4e8f3c519e9210dbc8725dac34da23e543bf462d10c4a202c61e80b59ed108dd
    src/bin/benchmark_gpu_public_api/lifetime.rs: 34be366f56aac88f5f3b86ade8018a15dad2bca85cea74f56cb2fe121917d0a9
    src/bin/benchmark_gpu_public_api/linalg.rs: 9c3d7916a3281d2dfd1caa68551dc9c81c79dab6b7525324339e5c4816581e6d
    src/bin/benchmark_gpu_public_api/multi.rs: 949eadd6272c17ba4e380a228547a03608d328a5ef852eb95294731c288f24d9
    src/bin/benchmark_gpu_public_api/plan.rs: 70fce8a6e9ed7518779a51a9d8692e010fa1e470e95eee0ce5276f595737fde8
    src/bin/benchmark_gpu_public_api/read_layout.rs: fc42e2160094ba5f9c7dd6facb5667a76e10b73f971ef9facfb1f64723e42467
    src/bin/benchmark_gpu_public_api/reuse.rs: a5a4aa0edb0a1158f6669b642f4bc25a7079314fc2a711ef3ef57cc19c0a35b2
    src/bin/benchmark_gpu_public_api/runtime_ops.rs: c9d61b1e99bda517a59a6e34f625fd4d651bbbb19f6dc06ba4d0d1c0d50dfd6a
    src/bin/benchmark_gpu_public_api/state.rs: a8bbf8985bbc58c435e3c4dda7135161a141a90a10fa50d4f95128397ce949cb
    src/bin/benchmark_gpu_public_api/structural.rs: 8b61ac01b58edc090eafbc124ec135d5efeb4979631a939656ba78dd6e5c042a
    /home/shinaoka/work/wallclock/scripts/benchmark_gpu_public_api.py: 1919e33cc5f2d0208901837f80cb7c0600426462f998be6dd933062d9f6bb345
    /home/shinaoka/work/wallclock/scripts/gpu_public_api_ops.py: e88823db3f4a1fe43dc13c4743c9d78ece6af76a968724091af318dd8d7f2528
    scripts/run_gpu_public_api.sh: 31587e7a83440d99417be00110c4dd12c5cc1143e620bf24b1c03fe5f76bfb5e
    data/results/nvidia-gpu/gpu/linalg_public_api_throughput/20261105_0820/fixtures/manifest.json: 8ac918ec3b3c102e5ae205c7547632fe9d4d7bbfd0f6cd3c1b00f3667f37aefc
    target/release/benchmark_gpu_public_api: efd5ab70e2c4de27c458d808d308b8bf3415012551220c291451a2591d08880f
  reference_numpy_version: 2.4.2
```

## Untimed work over the recorded budget

Each row records the wall clock it spent outside its timed region. These cases spent more than the recorded budget ratio (0.2) there. The threshold has no measured basis, so this is a recorded warning: it does not change a cell, its acceptance or the exit status of the run.

| Case | Backend | Warning |
| --- | --- | --- |
| cholesky_f64_512x512 | tenferro-cuda-direct | cholesky_f64_512x512 untimed 2268.1 ms is 6569% of the 34.5 ms timed region (budget 20%) |
| svdvals_f64_512x512 | tenferro-cuda-direct | svdvals_f64_512x512 untimed 551.8 ms is 47% of the 1168.7 ms timed region (budget 20%) |
| eigvalsh_f64_512x512 | tenferro-cuda-direct | eigvalsh_f64_512x512 untimed 43.2 ms is 52% of the 83.8 ms timed region (budget 20%) |
| det_f64_512x512 | tenferro-cuda-direct | det_f64_512x512 untimed 66612.5 ms is 122160% of the 54.5 ms timed region (budget 20%) |
| triangular_solve_f64_512x512 | tenferro-cuda-direct | triangular_solve_f64_512x512 untimed 9.8 ms is 21% of the 46.6 ms timed region (budget 20%) |
| svd_full_tall_f64_512x256 | tenferro-cuda-direct | svd_full_tall_f64_512x256 untimed 826.9 ms is 281% of the 294.5 ms timed region (budget 20%) |
| slogdet_positive_f64_256x256 | tenferro-cuda-direct | slogdet_positive_f64_256x256 untimed 11.5 ms is 23% of the 50.4 ms timed region (budget 20%) |
| householder_qr_tall_f64_512x256 | tenferro-cuda-direct | householder_qr_tall_f64_512x256 untimed 5738.7 ms is 14328% of the 40.1 ms timed region (budget 20%) |
| householder_from_factors_default_f64_512x256 | tenferro-cuda-direct | householder_from_factors_default_f64_512x256 untimed 588.5 ms is 1246% of the 47.2 ms timed region (budget 20%) |
| householder_r_default_f64_512x256 | tenferro-cuda-direct | householder_r_default_f64_512x256 untimed 9.5 ms is 25% of the 37.8 ms timed region (budget 20%) |
| lu_tall_f64_s256 | tenferro-cuda-direct | lu_tall_f64_s256 untimed 19.1 ms is 38% of the 50.8 ms timed region (budget 20%) |
| cholesky_f64_512x512 | tenferro-cuda-eager | cholesky_f64_512x512 untimed 57201.4 ms is 147098% of the 38.9 ms timed region (budget 20%) |
| eigvalsh_f64_512x512 | tenferro-cuda-eager | eigvalsh_f64_512x512 untimed 50.1 ms is 53% of the 95.0 ms timed region (budget 20%) |
| det_f64_512x512 | tenferro-cuda-eager | det_f64_512x512 untimed 18.5 ms is 33% of the 56.9 ms timed region (budget 20%) |
| norm_f64_512x512 | tenferro-cuda-eager | norm_f64_512x512 untimed 6.9 ms is 22% of the 31.3 ms timed region (budget 20%) |
| svd_full_tall_f64_512x256 | tenferro-cuda-eager | svd_full_tall_f64_512x256 untimed 961.5 ms is 325% of the 295.7 ms timed region (budget 20%) |
| slogdet_positive_f64_256x256 | tenferro-cuda-eager | slogdet_positive_f64_256x256 untimed 12.8 ms is 23% of the 54.8 ms timed region (budget 20%) |
| householder_qr_tall_f64_512x256 | tenferro-cuda-eager | householder_qr_tall_f64_512x256 untimed 6519.2 ms is 15627% of the 41.7 ms timed region (budget 20%) |
| householder_from_factors_default_f64_512x256 | tenferro-cuda-eager | householder_from_factors_default_f64_512x256 untimed 648.9 ms is 1296% of the 50.1 ms timed region (budget 20%) |
| householder_r_default_f64_512x256 | tenferro-cuda-eager | householder_r_default_f64_512x256 untimed 13.9 ms is 26% of the 54.1 ms timed region (budget 20%) |
| lu_tall_f64_s256 | tenferro-cuda-eager | lu_tall_f64_s256 untimed 24.5 ms is 45% of the 53.9 ms timed region (budget 20%) |
| cholesky_f64_512x512 | pytorch-cuda | cholesky_f64_512x512 untimed 522.8 ms is 1312% of the 39.8 ms timed region (budget 20%) |
| svdvals_f64_512x512 | pytorch-cuda | svdvals_f64_512x512 untimed 600.6 ms is 47% of the 1267.7 ms timed region (budget 20%) |
| eigvalsh_f64_512x512 | pytorch-cuda | eigvalsh_f64_512x512 untimed 67.6 ms is 81% of the 83.9 ms timed region (budget 20%) |
| pinv_f64_512x512 | pytorch-cuda | pinv_f64_512x512 untimed 1022.1 ms is 80% of the 1271.7 ms timed region (budget 20%) |
| inv_f64_512x512 | pytorch-cuda | inv_f64_512x512 untimed 384.0 ms is 709% of the 54.2 ms timed region (budget 20%) |
| det_f64_512x512 | pytorch-cuda | det_f64_512x512 untimed 82.2 ms is 209% of the 39.4 ms timed region (budget 20%) |
| norm_f64_512x512 | pytorch-cuda | norm_f64_512x512 untimed 47.8 ms is 115% of the 41.7 ms timed region (budget 20%) |
| triangular_solve_f64_512x512 | pytorch-cuda | triangular_solve_f64_512x512 untimed 20.6 ms is 56% of the 37.0 ms timed region (budget 20%) |
| svd_full_tall_f64_512x256 | pytorch-cuda | svd_full_tall_f64_512x256 untimed 594.1 ms is 199% of the 298.5 ms timed region (budget 20%) |
| slogdet_positive_f64_256x256 | pytorch-cuda | slogdet_positive_f64_256x256 untimed 26.8 ms is 79% of the 34.2 ms timed region (budget 20%) |
| householder_qr_tall_f64_512x256 | pytorch-cuda | householder_qr_tall_f64_512x256 untimed 448.2 ms is 1121% of the 40.0 ms timed region (budget 20%) |
| householder_r_default_f64_512x256 | pytorch-cuda | householder_r_default_f64_512x256 untimed 97.2 ms is 889% of the 10.9 ms timed region (budget 20%) |
| householder_q_columns_prefix_f64_512x256 | pytorch-cuda | householder_q_columns_prefix_f64_512x256 untimed 26.8 ms is 67% of the 40.0 ms timed region (budget 20%) |
| cholesky_f64_512x512 | jax-cuda | cholesky_f64_512x512 untimed 953.4 ms is 2637% of the 36.1 ms timed region (budget 20%) |
| svdvals_f64_512x512 | jax-cuda | svdvals_f64_512x512 untimed 724.8 ms is 62% of the 1169.9 ms timed region (budget 20%) |
| eigvalsh_f64_512x512 | jax-cuda | eigvalsh_f64_512x512 untimed 200.9 ms is 221% of the 91.1 ms timed region (budget 20%) |
| pinv_f64_512x512 | jax-cuda | pinv_f64_512x512 untimed 1581.1 ms is 135% of the 1174.1 ms timed region (budget 20%) |
| inv_f64_512x512 | jax-cuda | inv_f64_512x512 untimed 512.4 ms is 925% of the 55.4 ms timed region (budget 20%) |
| det_f64_512x512 | jax-cuda | det_f64_512x512 untimed 249.9 ms is 611% of the 40.9 ms timed region (budget 20%) |
| norm_f64_512x512 | jax-cuda | norm_f64_512x512 untimed 189.2 ms is 681% of the 27.8 ms timed region (budget 20%) |
| triangular_solve_f64_512x512 | jax-cuda | triangular_solve_f64_512x512 untimed 172.8 ms is 332% of the 52.1 ms timed region (budget 20%) |
| svd_full_tall_f64_512x256 | jax-cuda | svd_full_tall_f64_512x256 untimed 716.0 ms is 243% of the 295.1 ms timed region (budget 20%) |
| slogdet_positive_f64_256x256 | jax-cuda | slogdet_positive_f64_256x256 untimed 277.5 ms is 831% of the 33.4 ms timed region (budget 20%) |
| householder_qr_tall_f64_512x256 | jax-cuda | householder_qr_tall_f64_512x256 untimed 504.2 ms is 1183% of the 42.6 ms timed region (budget 20%) |
| householder_r_default_f64_512x256 | jax-cuda | householder_r_default_f64_512x256 untimed 388.1 ms is 963% of the 40.3 ms timed region (budget 20%) |
| householder_q_columns_prefix_f64_512x256 | jax-cuda | householder_q_columns_prefix_f64_512x256 untimed 307.5 ms is 1313% of the 23.4 ms timed region (budget 20%) |