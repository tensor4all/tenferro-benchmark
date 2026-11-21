# gpu/structural: CUDA public API benchmark

Measured cells: **60**. Unsupported cells: **4**, including **0** numerically verified but timing-excluded cells. Unsupported cells do not count as measured coverage.

Logical traffic: this run measured a plain device copy of 67108864 bytes at **1420.7 GB/s** on this device. Every record carries the logical bytes the case must move per operation and the ratio of the case's implied rate to that copy rate, in the sampling table below. Logical bytes are the declared array sizes, not device DRAM traffic: a case whose operands stay in cache moves less DRAM than its arrays suggest, so a ratio above 1 is recorded, not rejected.

Raw run: `data/results/nvidia-gpu/gpu/structural/20261121_082311/`.

Isolated public APIs and explicit unsupported timing contracts. Issue 155.

This run compiled **1 NVRTC modules** in 1 of its arms, because the CUDA artifact cache started cold. The untimed wall of the cases that compiled is **35.8 s** of its 44.6 s of recorded case wall clock, and that wall also holds their fixture read, transfers and verification, so it is not a compile time.

Values are median (IQR) **µs per operation**. Samples time batches of declared operations plus device synchronization; retained output memory is limited to 256 MiB. Batch counts and raw durations are preserved in JSONL.

Cases marked † (**7** of 16) were sampled with a different number of operations per sample on different arms (see Ops/sample in the sampling table). Each value is still per operation, but the arms amortise launch and synchronization over different batch sizes, so a ratio between their columns is approximate and is not presented as a like-for-like comparison.

Setup, input transfers, library/plan initialization, JIT compilation, validation downloads and output destruction are outside timing. Intrinsic output allocation is inside timing; output-reuse rows instead use preallocated destinations across samples. Accumulating destinations are restored from prepared GPU seeds and synchronized before every batch clock, including initialization and calibration. Direct and eager each use one entered session across samples; direct FFT uses cached `FftExecutor`. JAX eager and compiled JIT are separate arms.

All measured output elements are checked against independently generated NumPy/SciPy reference results before and after sampling. Unsupported rows include unavailable APIs and APIs whose internal setup cannot be excluded; these are not performance measurements. Short batches limited by output memory remain identifiable in the sampling table.

Reshape-copy cases add a leading singleton dimension and materialize the full output. Direct Tensor.reshape and EagerSession.reshape copy device data; PyTorch calls public view_copy. Direct/PyTorch initial/final allocation-pointer checks are untimed and reject input aliasing. Eager reshape copies its input before returning a descriptor view; original CUDA placement/shape/dtype are checked and a public inspection copy is made outside timing. JAX reshape has no established matching owned-output guarantee and remains unsupported; reshape-plus-copy composition is not substituted.

Stack selects actual public Tensor.stack/EagerSession.stack with two distinct inputs and axis 0 or 2. Their intrinsic intermediate allocations are part of this public stack operation. PyTorch/JAX use native stack; final outputs are fully verified. Direct Rust initial/final pointer checks reject aliasing either input outside timing; eager original descriptors are checked before any untimed inspection copy.

| Case | tenferro-cuda-direct | tenferro-cuda-eager | pytorch-cuda | jax-cuda |
| --- | ---: | ---: | ---: | ---: |
| pad_default_f64_n1048576 † | 39.312 (0.265) | 42.376 (0.828) | 15.765 (0.202) | 78.348 (2.835) |
| slice_default_f64_n1048576 † | 23.495 (0.110) | unsupported | 9.721 (0.387) | 79.229 (2.375) |
| reverse_default_f64_n1048576 | 22.466 (0.699) | 23.334 (0.300) | 10.700 (0.098) | 60.795 (0.972) |
| concatenate_default_f64_n1048576 | 52.613 (4.213) | 51.881 (0.231) | 25.617 (0.297) | 131.350 (10.931) |
| extract_diag_default_f64_n1048576 † | 18.663 (1.486) | 20.138 (0.123) | 7.362 (0.054) | 62.680 (1.052) |
| embed_diag_default_f64_n1024 | 21.416 (0.220) | 22.713 (0.205) | 11.430 (0.301) | 67.005 (3.392) |
| tril_default_f64_n1048576 | 26.885 (0.193) | 26.824 (0.198) | 11.014 (0.028) | 59.528 (0.753) |
| triu_default_f64_n1048576 | 26.097 (0.108) | 26.817 (0.260) | 11.148 (0.041) | 69.862 (2.682) |
| take_along_axis_broadcast_indices_f64_n1048576 † | 62.284 (0.865) | 67.892 (0.122) | 16.872 (0.533) | 73.131 (5.601) |
| scatter_default_f64_n1048576 | 63.488 (0.258) | 66.143 (0.332) | 22.927 (0.079) | 127.456 (9.106) |
| cast_default_f64_n1048576 † | 23.475 (1.140) | 46.532 (0.543) | 9.288 (0.035) | 101.292 (4.848) |
| broadcast_in_dim_f64_n1048576 | 33.615 (0.140) | unsupported | 7.839 (0.153) | 94.513 (0.943) |
| dynamic_slice_in_bounds_f64_n1048576 † | 21.192 (0.175) | 22.971 (0.085) | unsupported | 105.088 (2.402) |
| gather_axis0_f64_n1048576 † | 41.506 (0.079) | 44.909 (1.185) | 9.889 (0.203) | 102.053 (11.785) |
| reshape_copy_f64_n1048576 | 24.857 (0.185) | 23.405 (0.229) | 8.881 (0.191) | unsupported |
| stack_axis0_f64_n1048576 | 115.844 (0.632) | 122.024 (1.567) | 25.558 (0.178) | 111.824 (1.948) |

## Sampling

| Case | Backend | Ops/sample | Samples | Median batch ms | Bytes/op | Copy ratio | Verification error/tolerance |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| broadcast_in_dim_f64_n1048576 | jax-cuda | 32 | 15 | 3.024 | 8396800 | 0.063 | 0 |
| broadcast_in_dim_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.251 | 8396800 | 0.754 | 0 |
| broadcast_in_dim_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.076 | 8396800 | 0.176 | 0 |
| broadcast_in_dim_f64_n1048576 | tenferro-cuda-eager | — | — | — | 8396800 | unavailable | Eager broadcast returns a lazy broadcast view, not the materialized full output required by this case. |
| cast_default_f64_n1048576 | jax-cuda | 32 | 15 | 3.241 | 12582912 | 0.087 | 0 |
| cast_default_f64_n1048576 | pytorch-cuda | 64 | 15 | 0.594 | 12582912 | 0.954 | 0 |
| cast_default_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.502 | 12582912 | 0.377 | 0 |
| cast_default_f64_n1048576 | tenferro-cuda-eager | 64 | 15 | 2.978 | 12582912 | 0.190 | 0 |
| concatenate_default_f64_n1048576 | jax-cuda | 16 | 15 | 2.102 | 33554432 | 0.180 | 0 |
| concatenate_default_f64_n1048576 | pytorch-cuda | 16 | 15 | 0.410 | 33554432 | 0.922 | 0 |
| concatenate_default_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 0.842 | 33554432 | 0.449 | 0 |
| concatenate_default_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 0.830 | 33554432 | 0.455 | 0 |
| dynamic_slice_in_bounds_f64_n1048576 | jax-cuda | 32 | 15 | 3.363 | 10485776 | 0.070 | 0 |
| dynamic_slice_in_bounds_f64_n1048576 | pytorch-cuda | — | — | — | 10485776 | unavailable | No matched single materializing multidimensional slice API with CUDA-resident start indices and clamping; index construction/composed slicing is not substituted. |
| dynamic_slice_in_bounds_f64_n1048576 | tenferro-cuda-direct | 128 | 15 | 2.713 | 10485776 | 0.348 | 0 |
| dynamic_slice_in_bounds_f64_n1048576 | tenferro-cuda-eager | 128 | 15 | 2.940 | 10485776 | 0.321 | 0 |
| embed_diag_default_f64_n1024 | jax-cuda | 32 | 15 | 2.144 | 8396800 | 0.088 | 0 |
| embed_diag_default_f64_n1024 | pytorch-cuda | 32 | 15 | 0.366 | 8396800 | 0.517 | 0 |
| embed_diag_default_f64_n1024 | tenferro-cuda-direct | 32 | 15 | 0.685 | 8396800 | 0.276 | 0 |
| embed_diag_default_f64_n1024 | tenferro-cuda-eager | 32 | 15 | 0.727 | 8396800 | 0.260 | 0 |
| extract_diag_default_f64_n1048576 | jax-cuda | 64 | 15 | 4.011 | 8396800 | 0.094 | 0 |
| extract_diag_default_f64_n1048576 | pytorch-cuda | 512 | 15 | 3.769 | 8396800 | 0.803 | 0 |
| extract_diag_default_f64_n1048576 | tenferro-cuda-direct | 128 | 15 | 2.389 | 8396800 | 0.317 | 0 |
| extract_diag_default_f64_n1048576 | tenferro-cuda-eager | 128 | 15 | 2.578 | 8396800 | 0.293 | 0 |
| gather_axis0_f64_n1048576 | jax-cuda | 32 | 15 | 3.266 | 12587008 | 0.087 | 0 |
| gather_axis0_f64_n1048576 | pytorch-cuda | 64 | 15 | 0.633 | 12587008 | 0.896 | 0 |
| gather_axis0_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.656 | 12587008 | 0.213 | 0 |
| gather_axis0_f64_n1048576 | tenferro-cuda-eager | 64 | 15 | 2.874 | 12587008 | 0.197 | 0 |
| pad_default_f64_n1048576 | jax-cuda | 31 | 15 | 2.429 | 16908288 | 0.152 | 0 |
| pad_default_f64_n1048576 | pytorch-cuda | 31 | 15 | 0.489 | 16908288 | 0.755 | 0 |
| pad_default_f64_n1048576 | tenferro-cuda-direct | 31 | 15 | 1.219 | 16908288 | 0.303 | 0 |
| pad_default_f64_n1048576 | tenferro-cuda-eager | 2 | 15 | 0.085 | 16908288 | 0.281 | 0 |
| reshape_copy_f64_n1048576 | jax-cuda | — | — | — | 16777216 | unavailable | No established single-call JAX reshape API that guarantees owned output materialization; native reshape may alias input. Reshape plus copy is a different composed contract. |
| reshape_copy_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.284 | 16777216 | 1.330 | 0 |
| reshape_copy_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.795 | 16777216 | 0.475 | 0 |
| reshape_copy_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.749 | 16777216 | 0.505 | 0 |
| reverse_default_f64_n1048576 | jax-cuda | 32 | 15 | 1.945 | 16777216 | 0.194 | 0 |
| reverse_default_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.342 | 16777216 | 1.104 | 0 |
| reverse_default_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.719 | 16777216 | 0.526 | 0 |
| reverse_default_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.747 | 16777216 | 0.506 | 0 |
| scatter_default_f64_n1048576 | jax-cuda | 32 | 15 | 4.079 | 25165824 | 0.139 | 0 |
| scatter_default_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.734 | 25165824 | 0.773 | 0 |
| scatter_default_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.032 | 25165824 | 0.279 | 0 |
| scatter_default_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.117 | 25165824 | 0.268 | 0 |
| slice_default_f64_n1048576 | jax-cuda | 32 | 15 | 2.535 | 12582912 | 0.112 | 0 |
| slice_default_f64_n1048576 | pytorch-cuda | 64 | 15 | 0.622 | 12582912 | 0.911 | 0 |
| slice_default_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.504 | 12582912 | 0.377 | 0 |
| slice_default_f64_n1048576 | tenferro-cuda-eager | — | — | — | 12582912 | unavailable | EagerSession::slice returns a lazy metadata view; it is not equivalent to the materializing direct slice / torch.slice_copy / JAX slice contract. No copy is added inside its timer. |
| stack_axis0_f64_n1048576 | jax-cuda | 16 | 15 | 1.789 | 33554432 | 0.211 | 0 |
| stack_axis0_f64_n1048576 | pytorch-cuda | 16 | 15 | 0.409 | 33554432 | 0.924 | 0 |
| stack_axis0_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 1.853 | 33554432 | 0.204 | 0 |
| stack_axis0_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 1.952 | 33554432 | 0.194 | 0 |
| take_along_axis_broadcast_indices_f64_n1048576 | jax-cuda | 32 | 15 | 2.340 | 12587008 | 0.121 | 0 |
| take_along_axis_broadcast_indices_f64_n1048576 | pytorch-cuda | 64 | 15 | 1.080 | 12587008 | 0.525 | 0 |
| take_along_axis_broadcast_indices_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.993 | 12587008 | 0.142 | 0 |
| take_along_axis_broadcast_indices_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.173 | 12587008 | 0.130 | 0 |
| tril_default_f64_n1048576 | jax-cuda | 32 | 15 | 1.905 | 16777216 | 0.198 | 0 |
| tril_default_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.352 | 16777216 | 1.072 | 0 |
| tril_default_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.860 | 16777216 | 0.439 | 0 |
| tril_default_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.858 | 16777216 | 0.440 | 0 |
| triu_default_f64_n1048576 | jax-cuda | 32 | 15 | 2.236 | 16777216 | 0.169 | 0 |
| triu_default_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.357 | 16777216 | 1.059 | 0 |
| triu_default_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.835 | 16777216 | 0.452 | 0 |
| triu_default_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.858 | 16777216 | 0.440 | 0 |

## Run metadata

```yaml
schema_version: 1
target_profile: nvidia-gpu
suite_id: gpu/structural
suite_file: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/data/results/nvidia-gpu/gpu/routine_selection/20261121_082311/gpu_structural.yaml
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
  - 0.093801
  - 0.093301
  - 0.096251
  - 0.094811
  - 0.093461
  - 0.09439
  - 0.09599
  - 0.093821
  - 0.095131
  - 0.095041
  - 0.094051
  - 0.096051
  - 0.094471
  - 0.094081
  - 0.09532
  median_ms: 0.094471
  bytes_per_second: 1420729409024.992
  torch_version: 2.11.0+cu128
  cuda_version: '12.8'
  timestamp: '2026-11-21T08:34:19.353401+00:00'
gpu_assignment:
  role: benchmark
  physical_device_index: 0
  device_uuid: GPU-98d8618e-77c8-160a-92f2-117e34fd43df
  cuda_visible_devices: GPU-98d8618e-77c8-160a-92f2-117e34fd43df
  process_device_ordinal: 0
  verification_only: false
collection:
  command: BENCHMARK_TARGET_PROFILE=nvidia-gpu GPU_BENCH_DEVICE=0 BENCH_RUNS=15 BENCH_WARMUPS=3
    ./scripts/run_gpu_public_api.sh structural
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
    data/results/nvidia-gpu/gpu/structural/20261121_082311/fixtures/manifest.json: a320918ec2ed38973c5dcd570dd59f21d7fabb60d8cfb2d95483f723ae9fa443
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
        finished_at: '2026-11-21T08:35:11Z'
      pytorch-cuda:
        arm: pytorch-cuda
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-21T08:35:05Z'
      tenferro-cuda-direct:
        arm: tenferro-cuda-direct
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-21T08:34:58Z'
      tenferro-cuda-eager:
        arm: tenferro-cuda-eager
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-21T08:35:01Z'
  compile:
    case_wall_seconds: 44.641
    compiling_processes: 1
    modules_compiled: 1
    compile_time_ms: null
    compile_time_scope: 'not measured: no compile interval was recorded; `compile_modules`
      counts the NVRTC modules and the untimed wall is the case wall'
    untimed_wall_seconds: 35.818
```

## Untimed work over the recorded budget

Each row records the wall clock it spent outside its timed region. These cases spent more than the recorded budget ratio (0.2) there. The threshold has no measured basis, so this is a recorded warning: it does not change a cell, its acceptance or the exit status of the run.

| Case | Backend | Warning |
| --- | --- | --- |
| pad_default_f64_n1048576 | tenferro-cuda-direct | pad_default_f64_n1048576 untimed 163.3 ms is 881% of the 18.5 ms timed region (budget 20%) |
| slice_default_f64_n1048576 | tenferro-cuda-direct | slice_default_f64_n1048576 untimed 38.3 ms is 170% of the 22.6 ms timed region (budget 20%) |
| reverse_default_f64_n1048576 | tenferro-cuda-direct | reverse_default_f64_n1048576 untimed 85.5 ms is 772% of the 11.1 ms timed region (budget 20%) |
| concatenate_default_f64_n1048576 | tenferro-cuda-direct | concatenate_default_f64_n1048576 untimed 183.9 ms is 1425% of the 12.9 ms timed region (budget 20%) |
| extract_diag_default_f64_n1048576 | tenferro-cuda-direct | extract_diag_default_f64_n1048576 untimed 9.2 ms is 25% of the 37.2 ms timed region (budget 20%) |
| embed_diag_default_f64_n1024 | tenferro-cuda-direct | embed_diag_default_f64_n1024 untimed 67.1 ms is 652% of the 10.3 ms timed region (budget 20%) |
| tril_default_f64_n1048576 | tenferro-cuda-direct | tril_default_f64_n1048576 untimed 89.0 ms is 689% of the 12.9 ms timed region (budget 20%) |
| triu_default_f64_n1048576 | tenferro-cuda-direct | triu_default_f64_n1048576 untimed 78.5 ms is 626% of the 12.5 ms timed region (budget 20%) |
| take_along_axis_broadcast_indices_f64_n1048576 | tenferro-cuda-direct | take_along_axis_broadcast_indices_f64_n1048576 untimed 35817.9 ms is 119459% of the 30.0 ms timed region (budget 20%) |
| scatter_default_f64_n1048576 | tenferro-cuda-direct | scatter_default_f64_n1048576 untimed 46.7 ms is 153% of the 30.5 ms timed region (budget 20%) |
| cast_default_f64_n1048576 | tenferro-cuda-direct | cast_default_f64_n1048576 untimed 31.1 ms is 138% of the 22.5 ms timed region (budget 20%) |
| broadcast_in_dim_f64_n1048576 | tenferro-cuda-direct | broadcast_in_dim_f64_n1048576 untimed 34.1 ms is 211% of the 16.1 ms timed region (budget 20%) |
| dynamic_slice_in_bounds_f64_n1048576 | tenferro-cuda-direct | dynamic_slice_in_bounds_f64_n1048576 untimed 20.0 ms is 49% of the 40.8 ms timed region (budget 20%) |
| gather_axis0_f64_n1048576 | tenferro-cuda-direct | gather_axis0_f64_n1048576 untimed 23.7 ms is 59% of the 39.9 ms timed region (budget 20%) |
| reshape_copy_f64_n1048576 | tenferro-cuda-direct | reshape_copy_f64_n1048576 untimed 34.0 ms is 282% of the 12.1 ms timed region (budget 20%) |
| stack_axis0_f64_n1048576 | tenferro-cuda-direct | stack_axis0_f64_n1048576 untimed 65.3 ms is 235% of the 27.8 ms timed region (budget 20%) |
| pad_default_f64_n1048576 | tenferro-cuda-eager | pad_default_f64_n1048576 untimed 149.1 ms is 11579% of the 1.3 ms timed region (budget 20%) |
| reverse_default_f64_n1048576 | tenferro-cuda-eager | reverse_default_f64_n1048576 untimed 91.3 ms is 814% of the 11.2 ms timed region (budget 20%) |
| concatenate_default_f64_n1048576 | tenferro-cuda-eager | concatenate_default_f64_n1048576 untimed 173.6 ms is 1393% of the 12.5 ms timed region (budget 20%) |
| extract_diag_default_f64_n1048576 | tenferro-cuda-eager | extract_diag_default_f64_n1048576 untimed 13.5 ms is 35% of the 38.8 ms timed region (budget 20%) |
| embed_diag_default_f64_n1024 | tenferro-cuda-eager | embed_diag_default_f64_n1024 untimed 70.5 ms is 645% of the 10.9 ms timed region (budget 20%) |
| tril_default_f64_n1048576 | tenferro-cuda-eager | tril_default_f64_n1048576 untimed 79.4 ms is 615% of the 12.9 ms timed region (budget 20%) |
| triu_default_f64_n1048576 | tenferro-cuda-eager | triu_default_f64_n1048576 untimed 84.3 ms is 644% of the 13.1 ms timed region (budget 20%) |
| take_along_axis_broadcast_indices_f64_n1048576 | tenferro-cuda-eager | take_along_axis_broadcast_indices_f64_n1048576 untimed 39.0 ms is 119% of the 32.6 ms timed region (budget 20%) |
| scatter_default_f64_n1048576 | tenferro-cuda-eager | scatter_default_f64_n1048576 untimed 81.3 ms is 256% of the 31.8 ms timed region (budget 20%) |
| cast_default_f64_n1048576 | tenferro-cuda-eager | cast_default_f64_n1048576 untimed 68.0 ms is 153% of the 44.6 ms timed region (budget 20%) |
| dynamic_slice_in_bounds_f64_n1048576 | tenferro-cuda-eager | dynamic_slice_in_bounds_f64_n1048576 untimed 21.6 ms is 49% of the 44.1 ms timed region (budget 20%) |
| gather_axis0_f64_n1048576 | tenferro-cuda-eager | gather_axis0_f64_n1048576 untimed 25.6 ms is 59% of the 43.3 ms timed region (budget 20%) |
| reshape_copy_f64_n1048576 | tenferro-cuda-eager | reshape_copy_f64_n1048576 untimed 70.8 ms is 628% of the 11.3 ms timed region (budget 20%) |
| stack_axis0_f64_n1048576 | tenferro-cuda-eager | stack_axis0_f64_n1048576 untimed 173.5 ms is 586% of the 29.6 ms timed region (budget 20%) |
| pad_default_f64_n1048576 | pytorch-cuda | pad_default_f64_n1048576 untimed 270.3 ms is 3668% of the 7.4 ms timed region (budget 20%) |
| slice_default_f64_n1048576 | pytorch-cuda | slice_default_f64_n1048576 untimed 64.5 ms is 663% of the 9.7 ms timed region (budget 20%) |
| reverse_default_f64_n1048576 | pytorch-cuda | reverse_default_f64_n1048576 untimed 143.8 ms is 2804% of the 5.1 ms timed region (budget 20%) |
| concatenate_default_f64_n1048576 | pytorch-cuda | concatenate_default_f64_n1048576 untimed 296.1 ms is 4825% of the 6.1 ms timed region (budget 20%) |
| extract_diag_default_f64_n1048576 | pytorch-cuda | extract_diag_default_f64_n1048576 untimed 21.1 ms is 37% of the 56.6 ms timed region (budget 20%) |
| embed_diag_default_f64_n1024 | pytorch-cuda | embed_diag_default_f64_n1024 untimed 141.8 ms is 2564% of the 5.5 ms timed region (budget 20%) |
| tril_default_f64_n1048576 | pytorch-cuda | tril_default_f64_n1048576 untimed 148.2 ms is 2803% of the 5.3 ms timed region (budget 20%) |
| triu_default_f64_n1048576 | pytorch-cuda | triu_default_f64_n1048576 untimed 131.8 ms is 2458% of the 5.4 ms timed region (budget 20%) |
| take_along_axis_broadcast_indices_f64_n1048576 | pytorch-cuda | take_along_axis_broadcast_indices_f64_n1048576 untimed 107.6 ms is 657% of the 16.4 ms timed region (budget 20%) |
| scatter_default_f64_n1048576 | pytorch-cuda | scatter_default_f64_n1048576 untimed 70.2 ms is 638% of the 11.0 ms timed region (budget 20%) |
| cast_default_f64_n1048576 | pytorch-cuda | cast_default_f64_n1048576 untimed 41.3 ms is 462% of the 8.9 ms timed region (budget 20%) |
| broadcast_in_dim_f64_n1048576 | pytorch-cuda | broadcast_in_dim_f64_n1048576 untimed 135.2 ms is 3573% of the 3.8 ms timed region (budget 20%) |
| gather_axis0_f64_n1048576 | pytorch-cuda | gather_axis0_f64_n1048576 untimed 64.9 ms is 683% of the 9.5 ms timed region (budget 20%) |
| reshape_copy_f64_n1048576 | pytorch-cuda | reshape_copy_f64_n1048576 untimed 133.0 ms is 3103% of the 4.3 ms timed region (budget 20%) |
| stack_axis0_f64_n1048576 | pytorch-cuda | stack_axis0_f64_n1048576 untimed 461.2 ms is 7315% of the 6.3 ms timed region (budget 20%) |
| pad_default_f64_n1048576 | jax-cuda | pad_default_f64_n1048576 untimed 402.3 ms is 1085% of the 37.1 ms timed region (budget 20%) |
| slice_default_f64_n1048576 | jax-cuda | slice_default_f64_n1048576 untimed 104.2 ms is 275% of the 37.8 ms timed region (budget 20%) |
| reverse_default_f64_n1048576 | jax-cuda | reverse_default_f64_n1048576 untimed 179.2 ms is 612% of the 29.3 ms timed region (budget 20%) |
| concatenate_default_f64_n1048576 | jax-cuda | concatenate_default_f64_n1048576 untimed 365.9 ms is 1167% of the 31.4 ms timed region (budget 20%) |
| extract_diag_default_f64_n1048576 | jax-cuda | extract_diag_default_f64_n1048576 untimed 68.6 ms is 114% of the 60.2 ms timed region (budget 20%) |
| embed_diag_default_f64_n1024 | jax-cuda | embed_diag_default_f64_n1024 untimed 213.9 ms is 632% of the 33.8 ms timed region (budget 20%) |
| tril_default_f64_n1048576 | jax-cuda | tril_default_f64_n1048576 untimed 199.5 ms is 699% of the 28.5 ms timed region (budget 20%) |
| triu_default_f64_n1048576 | jax-cuda | triu_default_f64_n1048576 untimed 187.9 ms is 556% of the 33.8 ms timed region (budget 20%) |
| take_along_axis_broadcast_indices_f64_n1048576 | jax-cuda | take_along_axis_broadcast_indices_f64_n1048576 untimed 135.7 ms is 388% of the 34.9 ms timed region (budget 20%) |
| scatter_default_f64_n1048576 | jax-cuda | scatter_default_f64_n1048576 untimed 132.2 ms is 214% of the 61.8 ms timed region (budget 20%) |
| cast_default_f64_n1048576 | jax-cuda | cast_default_f64_n1048576 untimed 87.0 ms is 180% of the 48.2 ms timed region (budget 20%) |
| broadcast_in_dim_f64_n1048576 | jax-cuda | broadcast_in_dim_f64_n1048576 untimed 203.7 ms is 450% of the 45.3 ms timed region (budget 20%) |
| dynamic_slice_in_bounds_f64_n1048576 | jax-cuda | dynamic_slice_in_bounds_f64_n1048576 untimed 118.4 ms is 228% of the 51.8 ms timed region (budget 20%) |
| gather_axis0_f64_n1048576 | jax-cuda | gather_axis0_f64_n1048576 untimed 113.4 ms is 233% of the 48.7 ms timed region (budget 20%) |
| stack_axis0_f64_n1048576 | jax-cuda | stack_axis0_f64_n1048576 untimed 502.3 ms is 1865% of the 26.9 ms timed region (budget 20%) |