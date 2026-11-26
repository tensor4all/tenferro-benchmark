# gpu/structural: CUDA public API benchmark

Measured cells: **60**. Unsupported cells: **4**, including **0** numerically verified but timing-excluded cells. Unsupported cells do not count as measured coverage.

Logical traffic: this run measured a plain device copy of 67108864 bytes at **1467.5 GB/s** on this device. Every record carries the logical bytes the case must move per operation and the ratio of the case's implied rate to that copy rate, in the sampling table below. Logical bytes are the declared array sizes, not device DRAM traffic: a case whose operands stay in cache moves less DRAM than its arrays suggest, so a ratio above 1 is recorded, not rejected.

Raw run: `data/results/nvidia-gpu/gpu/structural/20261126_062950/`.

Isolated public APIs and explicit unsupported timing contracts. Issue 155.

This run compiled **1 NVRTC modules** in 1 of its arms, because the CUDA artifact cache started cold. The untimed wall of the cases that compiled is **35.9 s** of its 43.9 s of recorded case wall clock, and that wall also holds their fixture read, transfers and verification, so it is not a compile time.

Values are median (IQR) **µs per operation**. Samples time batches of declared operations plus device synchronization; retained output memory is limited to 256 MiB. Batch counts and raw durations are preserved in JSONL.

Cases marked † (**5** of 16) were sampled with a different number of operations per sample on different arms (see Ops/sample in the sampling table). Each value is still per operation, but the arms amortise launch and synchronization over different batch sizes, so a ratio between their columns is approximate and is not presented as a like-for-like comparison.

Setup, input transfers, library/plan initialization, JIT compilation, validation downloads and output destruction are outside timing. Intrinsic output allocation is inside timing; output-reuse rows instead use preallocated destinations across samples. Accumulating destinations are restored from prepared GPU seeds and synchronized before every batch clock, including initialization and calibration. Direct and eager each use one entered session across samples; direct FFT uses cached `FftExecutor`. JAX eager and compiled JIT are separate arms.

All measured output elements are checked against independently generated NumPy/SciPy reference results before and after sampling. Unsupported rows include unavailable APIs and APIs whose internal setup cannot be excluded; these are not performance measurements. Short batches limited by output memory remain identifiable in the sampling table.

Reshape-copy cases add a leading singleton dimension and materialize the full output. Direct Tensor.reshape and EagerSession.reshape copy device data; PyTorch calls public view_copy. Direct/PyTorch initial/final allocation-pointer checks are untimed and reject input aliasing. Eager reshape copies its input before returning a descriptor view; original CUDA placement/shape/dtype are checked and a public inspection copy is made outside timing. JAX reshape has no established matching owned-output guarantee and remains unsupported; reshape-plus-copy composition is not substituted.

Stack selects actual public Tensor.stack/EagerSession.stack with two distinct inputs and axis 0 or 2. Their intrinsic intermediate allocations are part of this public stack operation. PyTorch/JAX use native stack; final outputs are fully verified. Direct Rust initial/final pointer checks reject aliasing either input outside timing; eager original descriptors are checked before any untimed inspection copy.

| Case | tenferro-cuda-direct | tenferro-cuda-eager | pytorch-cuda | jax-cuda |
| --- | ---: | ---: | ---: | ---: |
| pad_default_f64_n1048576 | 28.490 (0.032) | 28.713 (0.059) | 15.230 (0.092) | 77.389 (1.215) |
| slice_default_f64_n1048576 † | 12.554 (0.024) | unsupported | 8.380 (0.197) | 80.164 (2.767) |
| reverse_default_f64_n1048576 | 10.760 (0.067) | 10.908 (0.033) | 10.621 (0.088) | 64.915 (0.770) |
| concatenate_default_f64_n1048576 | 39.234 (0.102) | 39.688 (0.100) | 24.443 (0.125) | 100.675 (2.414) |
| extract_diag_default_f64_n1048576 † | 6.521 (0.107) | 7.748 (0.058) | 6.534 (0.040) | 54.278 (0.477) |
| embed_diag_default_f64_n1024 | 10.822 (0.271) | 11.113 (0.077) | 10.664 (0.175) | 53.799 (2.954) |
| tril_default_f64_n1048576 | 15.558 (0.023) | 15.768 (0.028) | 10.888 (0.048) | 51.724 (6.378) |
| triu_default_f64_n1048576 | 15.509 (0.047) | 15.755 (0.053) | 11.022 (0.023) | 54.198 (5.013) |
| take_along_axis_broadcast_indices_f64_n1048576 | 35.759 (0.019) | 35.054 (0.122) | 15.998 (0.178) | 41.981 (0.694) |
| scatter_default_f64_n1048576 | 51.765 (0.075) | 52.102 (0.096) | 22.831 (0.115) | 100.048 (2.593) |
| cast_default_f64_n1048576 † | 12.526 (0.037) | 22.566 (0.043) | 9.077 (0.039) | 80.191 (1.850) |
| broadcast_in_dim_f64_n1048576 | 22.854 (0.071) | unsupported | 7.603 (0.057) | 68.220 (0.637) |
| dynamic_slice_in_bounds_f64_n1048576 † | 9.694 (0.017) | 9.961 (1.935) | unsupported | 86.094 (0.734) |
| gather_axis0_f64_n1048576 † | 29.513 (0.022) | 29.740 (0.032) | 8.862 (0.167) | 75.566 (0.825) |
| reshape_copy_f64_n1048576 | 14.582 (0.034) | 10.079 (0.057) | 8.771 (0.071) | unsupported |
| stack_axis0_f64_n1048576 | 81.689 (0.113) | 76.170 (0.094) | 24.254 (0.182) | 94.153 (1.328) |

## Sampling

| Case | Backend | Ops/sample | Samples | Median batch ms | Bytes/op | Copy ratio | Verification error/tolerance |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| broadcast_in_dim_f64_n1048576 | jax-cuda | 32 | 15 | 2.183 | 8396800 | 0.084 | 0 |
| broadcast_in_dim_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.243 | 8396800 | 0.753 | 0 |
| broadcast_in_dim_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.731 | 8396800 | 0.250 | 0 |
| broadcast_in_dim_f64_n1048576 | tenferro-cuda-eager | — | — | — | 8396800 | unavailable | Eager broadcast returns a lazy broadcast view, not the materialized full output required by this case. |
| cast_default_f64_n1048576 | jax-cuda | 32 | 15 | 2.566 | 12582912 | 0.107 | 0 |
| cast_default_f64_n1048576 | pytorch-cuda | 64 | 15 | 0.581 | 12582912 | 0.945 | 0 |
| cast_default_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 0.802 | 12582912 | 0.685 | 0 |
| cast_default_f64_n1048576 | tenferro-cuda-eager | 64 | 15 | 1.444 | 12582912 | 0.380 | 0 |
| concatenate_default_f64_n1048576 | jax-cuda | 16 | 15 | 1.611 | 33554432 | 0.227 | 0 |
| concatenate_default_f64_n1048576 | pytorch-cuda | 16 | 15 | 0.391 | 33554432 | 0.935 | 0 |
| concatenate_default_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 0.628 | 33554432 | 0.583 | 0 |
| concatenate_default_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 0.635 | 33554432 | 0.576 | 0 |
| dynamic_slice_in_bounds_f64_n1048576 | jax-cuda | 32 | 15 | 2.755 | 10485776 | 0.083 | 0 |
| dynamic_slice_in_bounds_f64_n1048576 | pytorch-cuda | — | — | — | 10485776 | unavailable | No matched single materializing multidimensional slice API with CUDA-resident start indices and clamping; index construction/composed slicing is not substituted. |
| dynamic_slice_in_bounds_f64_n1048576 | tenferro-cuda-direct | 128 | 15 | 1.241 | 10485776 | 0.737 | 0 |
| dynamic_slice_in_bounds_f64_n1048576 | tenferro-cuda-eager | 128 | 15 | 1.275 | 10485776 | 0.717 | 0 |
| embed_diag_default_f64_n1024 | jax-cuda | 32 | 15 | 1.722 | 8396800 | 0.106 | 0 |
| embed_diag_default_f64_n1024 | pytorch-cuda | 32 | 15 | 0.341 | 8396800 | 0.537 | 0 |
| embed_diag_default_f64_n1024 | tenferro-cuda-direct | 32 | 15 | 0.346 | 8396800 | 0.529 | 0 |
| embed_diag_default_f64_n1024 | tenferro-cuda-eager | 32 | 15 | 0.356 | 8396800 | 0.515 | 0 |
| extract_diag_default_f64_n1048576 | jax-cuda | 64 | 15 | 3.474 | 8396800 | 0.105 | 0 |
| extract_diag_default_f64_n1048576 | pytorch-cuda | 512 | 15 | 3.345 | 8396800 | 0.876 | 0 |
| extract_diag_default_f64_n1048576 | tenferro-cuda-direct | 512 | 15 | 3.339 | 8396800 | 0.877 | 0 |
| extract_diag_default_f64_n1048576 | tenferro-cuda-eager | 256 | 15 | 1.983 | 8396800 | 0.738 | 0 |
| gather_axis0_f64_n1048576 | jax-cuda | 32 | 15 | 2.418 | 12587008 | 0.114 | 0 |
| gather_axis0_f64_n1048576 | pytorch-cuda | 64 | 15 | 0.567 | 12587008 | 0.968 | 0 |
| gather_axis0_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.889 | 12587008 | 0.291 | 0 |
| gather_axis0_f64_n1048576 | tenferro-cuda-eager | 64 | 15 | 1.903 | 12587008 | 0.288 | 0 |
| pad_default_f64_n1048576 | jax-cuda | 31 | 15 | 2.399 | 16908288 | 0.149 | 0 |
| pad_default_f64_n1048576 | pytorch-cuda | 31 | 15 | 0.472 | 16908288 | 0.757 | 0 |
| pad_default_f64_n1048576 | tenferro-cuda-direct | 31 | 15 | 0.883 | 16908288 | 0.404 | 0 |
| pad_default_f64_n1048576 | tenferro-cuda-eager | 31 | 15 | 0.890 | 16908288 | 0.401 | 0 |
| reshape_copy_f64_n1048576 | jax-cuda | — | — | — | 16777216 | unavailable | No established single-call JAX reshape API that guarantees owned output materialization; native reshape may alias input. Reshape plus copy is a different composed contract. |
| reshape_copy_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.281 | 16777216 | 1.303 | 0 |
| reshape_copy_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.467 | 16777216 | 0.784 | 0 |
| reshape_copy_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.323 | 16777216 | 1.134 | 0 |
| reverse_default_f64_n1048576 | jax-cuda | 32 | 15 | 2.077 | 16777216 | 0.176 | 0 |
| reverse_default_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.340 | 16777216 | 1.076 | 0 |
| reverse_default_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.344 | 16777216 | 1.063 | 0 |
| reverse_default_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.349 | 16777216 | 1.048 | 0 |
| scatter_default_f64_n1048576 | jax-cuda | 32 | 15 | 3.202 | 25165824 | 0.171 | 0 |
| scatter_default_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.731 | 25165824 | 0.751 | 0 |
| scatter_default_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.656 | 25165824 | 0.331 | 0 |
| scatter_default_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.667 | 25165824 | 0.329 | 0 |
| slice_default_f64_n1048576 | jax-cuda | 32 | 15 | 2.565 | 12582912 | 0.107 | 0 |
| slice_default_f64_n1048576 | pytorch-cuda | 64 | 15 | 0.536 | 12582912 | 1.023 | 0 |
| slice_default_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 0.803 | 12582912 | 0.683 | 0 |
| slice_default_f64_n1048576 | tenferro-cuda-eager | — | — | — | 12582912 | unavailable | EagerSession::slice returns a lazy metadata view; it is not equivalent to the materializing direct slice / torch.slice_copy / JAX slice contract. No copy is added inside its timer. |
| stack_axis0_f64_n1048576 | jax-cuda | 16 | 15 | 1.506 | 33554432 | 0.243 | 0 |
| stack_axis0_f64_n1048576 | pytorch-cuda | 16 | 15 | 0.388 | 33554432 | 0.943 | 0 |
| stack_axis0_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 1.307 | 33554432 | 0.280 | 0 |
| stack_axis0_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 1.219 | 33554432 | 0.300 | 0 |
| take_along_axis_broadcast_indices_f64_n1048576 | jax-cuda | 64 | 15 | 2.687 | 12587008 | 0.204 | 0 |
| take_along_axis_broadcast_indices_f64_n1048576 | pytorch-cuda | 64 | 15 | 1.024 | 12587008 | 0.536 | 0 |
| take_along_axis_broadcast_indices_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.289 | 12587008 | 0.240 | 0 |
| take_along_axis_broadcast_indices_f64_n1048576 | tenferro-cuda-eager | 64 | 15 | 2.243 | 12587008 | 0.245 | 0 |
| tril_default_f64_n1048576 | jax-cuda | 32 | 15 | 1.655 | 16777216 | 0.221 | 0 |
| tril_default_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.348 | 16777216 | 1.050 | 0 |
| tril_default_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.498 | 16777216 | 0.735 | 0 |
| tril_default_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.505 | 16777216 | 0.725 | 0 |
| triu_default_f64_n1048576 | jax-cuda | 32 | 15 | 1.734 | 16777216 | 0.211 | 0 |
| triu_default_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.353 | 16777216 | 1.037 | 0 |
| triu_default_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.496 | 16777216 | 0.737 | 0 |
| triu_default_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.504 | 16777216 | 0.726 | 0 |

## Run metadata

```yaml
schema_version: 1
target_profile: nvidia-gpu
suite_id: gpu/structural
suite_file: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/data/results/nvidia-gpu/gpu/routine_selection/20261126_062950/gpu_structural.yaml
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
  - 0.09065
  - 0.093
  - 0.09148
  - 0.09079
  - 0.091751
  - 0.09146
  - 0.09016
  - 0.0922
  - 0.09218
  - 0.090831
  - 0.09207
  - 0.09111
  - 0.09172
  - 0.09098
  - 0.090711
  median_ms: 0.09146
  bytes_per_second: 1467501946205.9917
  torch_version: 2.11.0+cu128
  cuda_version: '12.8'
  timestamp: '2026-11-26T06:43:17.342876+00:00'
gpu_assignment:
  role: benchmark
  physical_device_index: 0
  device_uuid: GPU-e0c82f3d-3a11-a28c-b42e-1c92d26527fd
  cuda_visible_devices: GPU-e0c82f3d-3a11-a28c-b42e-1c92d26527fd
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
    data/results/nvidia-gpu/gpu/structural/20261126_062950/fixtures/manifest.json: a320918ec2ed38973c5dcd570dd59f21d7fabb60d8cfb2d95483f723ae9fa443
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
        finished_at: '2026-11-26T06:44:08Z'
      pytorch-cuda:
        arm: pytorch-cuda
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-26T06:44:03Z'
      tenferro-cuda-direct:
        arm: tenferro-cuda-direct
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-26T06:43:56Z'
      tenferro-cuda-eager:
        arm: tenferro-cuda-eager
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-26T06:43:58Z'
  compile:
    case_wall_seconds: 43.915
    compiling_processes: 1
    modules_compiled: 1
    compile_time_ms: null
    compile_time_scope: 'not measured: no compile interval was recorded; `compile_modules`
      counts the NVRTC modules and the untimed wall is the case wall'
    untimed_wall_seconds: 35.894
```

## Untimed work over the recorded budget

Each row records the wall clock it spent outside its timed region. These cases spent more than the recorded budget ratio (0.2) there. The threshold has no measured basis, so this is a recorded warning: it does not change a cell, its acceptance or the exit status of the run.

| Case | Backend | Warning |
| --- | --- | --- |
| pad_default_f64_n1048576 | tenferro-cuda-direct | pad_default_f64_n1048576 untimed 154.7 ms is 1168% of the 13.3 ms timed region (budget 20%) |
| slice_default_f64_n1048576 | tenferro-cuda-direct | slice_default_f64_n1048576 untimed 29.2 ms is 243% of the 12.0 ms timed region (budget 20%) |
| reverse_default_f64_n1048576 | tenferro-cuda-direct | reverse_default_f64_n1048576 untimed 73.2 ms is 1407% of the 5.2 ms timed region (budget 20%) |
| concatenate_default_f64_n1048576 | tenferro-cuda-direct | concatenate_default_f64_n1048576 untimed 160.3 ms is 1701% of the 9.4 ms timed region (budget 20%) |
| extract_diag_default_f64_n1048576 | tenferro-cuda-direct | extract_diag_default_f64_n1048576 untimed 15.7 ms is 31% of the 50.8 ms timed region (budget 20%) |
| embed_diag_default_f64_n1024 | tenferro-cuda-direct | embed_diag_default_f64_n1024 untimed 66.8 ms is 1241% of the 5.4 ms timed region (budget 20%) |
| tril_default_f64_n1048576 | tenferro-cuda-direct | tril_default_f64_n1048576 untimed 74.7 ms is 993% of the 7.5 ms timed region (budget 20%) |
| triu_default_f64_n1048576 | tenferro-cuda-direct | triu_default_f64_n1048576 untimed 71.3 ms is 957% of the 7.5 ms timed region (budget 20%) |
| take_along_axis_broadcast_indices_f64_n1048576 | tenferro-cuda-direct | take_along_axis_broadcast_indices_f64_n1048576 untimed 35894.3 ms is 104571% of the 34.3 ms timed region (budget 20%) |
| scatter_default_f64_n1048576 | tenferro-cuda-direct | scatter_default_f64_n1048576 untimed 41.6 ms is 166% of the 25.0 ms timed region (budget 20%) |
| cast_default_f64_n1048576 | tenferro-cuda-direct | cast_default_f64_n1048576 untimed 25.3 ms is 211% of the 12.0 ms timed region (budget 20%) |
| broadcast_in_dim_f64_n1048576 | tenferro-cuda-direct | broadcast_in_dim_f64_n1048576 untimed 26.1 ms is 237% of the 11.0 ms timed region (budget 20%) |
| dynamic_slice_in_bounds_f64_n1048576 | tenferro-cuda-direct | dynamic_slice_in_bounds_f64_n1048576 untimed 13.5 ms is 73% of the 18.6 ms timed region (budget 20%) |
| gather_axis0_f64_n1048576 | tenferro-cuda-direct | gather_axis0_f64_n1048576 untimed 17.9 ms is 63% of the 28.3 ms timed region (budget 20%) |
| reshape_copy_f64_n1048576 | tenferro-cuda-direct | reshape_copy_f64_n1048576 untimed 26.3 ms is 375% of the 7.0 ms timed region (budget 20%) |
| stack_axis0_f64_n1048576 | tenferro-cuda-direct | stack_axis0_f64_n1048576 untimed 59.6 ms is 304% of the 19.6 ms timed region (budget 20%) |
| pad_default_f64_n1048576 | tenferro-cuda-eager | pad_default_f64_n1048576 untimed 157.2 ms is 1178% of the 13.4 ms timed region (budget 20%) |
| reverse_default_f64_n1048576 | tenferro-cuda-eager | reverse_default_f64_n1048576 untimed 81.6 ms is 1559% of the 5.2 ms timed region (budget 20%) |
| concatenate_default_f64_n1048576 | tenferro-cuda-eager | concatenate_default_f64_n1048576 untimed 171.1 ms is 1796% of the 9.5 ms timed region (budget 20%) |
| extract_diag_default_f64_n1048576 | tenferro-cuda-eager | extract_diag_default_f64_n1048576 untimed 10.9 ms is 36% of the 30.1 ms timed region (budget 20%) |
| embed_diag_default_f64_n1024 | tenferro-cuda-eager | embed_diag_default_f64_n1024 untimed 64.6 ms is 1211% of the 5.3 ms timed region (budget 20%) |
| tril_default_f64_n1048576 | tenferro-cuda-eager | tril_default_f64_n1048576 untimed 74.9 ms is 989% of the 7.6 ms timed region (budget 20%) |
| triu_default_f64_n1048576 | tenferro-cuda-eager | triu_default_f64_n1048576 untimed 71.4 ms is 943% of the 7.6 ms timed region (budget 20%) |
| take_along_axis_broadcast_indices_f64_n1048576 | tenferro-cuda-eager | take_along_axis_broadcast_indices_f64_n1048576 untimed 38.4 ms is 113% of the 33.9 ms timed region (budget 20%) |
| scatter_default_f64_n1048576 | tenferro-cuda-eager | scatter_default_f64_n1048576 untimed 79.7 ms is 319% of the 25.0 ms timed region (budget 20%) |
| cast_default_f64_n1048576 | tenferro-cuda-eager | cast_default_f64_n1048576 untimed 73.0 ms is 337% of the 21.7 ms timed region (budget 20%) |
| dynamic_slice_in_bounds_f64_n1048576 | tenferro-cuda-eager | dynamic_slice_in_bounds_f64_n1048576 untimed 18.5 ms is 87% of the 21.2 ms timed region (budget 20%) |
| gather_axis0_f64_n1048576 | tenferro-cuda-eager | gather_axis0_f64_n1048576 untimed 22.8 ms is 80% of the 28.6 ms timed region (budget 20%) |
| reshape_copy_f64_n1048576 | tenferro-cuda-eager | reshape_copy_f64_n1048576 untimed 66.3 ms is 1371% of the 4.8 ms timed region (budget 20%) |
| stack_axis0_f64_n1048576 | tenferro-cuda-eager | stack_axis0_f64_n1048576 untimed 154.8 ms is 847% of the 18.3 ms timed region (budget 20%) |
| pad_default_f64_n1048576 | pytorch-cuda | pad_default_f64_n1048576 untimed 272.5 ms is 3827% of the 7.1 ms timed region (budget 20%) |
| slice_default_f64_n1048576 | pytorch-cuda | slice_default_f64_n1048576 untimed 59.1 ms is 721% of the 8.2 ms timed region (budget 20%) |
| reverse_default_f64_n1048576 | pytorch-cuda | reverse_default_f64_n1048576 untimed 138.5 ms is 2613% of the 5.3 ms timed region (budget 20%) |
| concatenate_default_f64_n1048576 | pytorch-cuda | concatenate_default_f64_n1048576 untimed 281.7 ms is 4806% of the 5.9 ms timed region (budget 20%) |
| extract_diag_default_f64_n1048576 | pytorch-cuda | extract_diag_default_f64_n1048576 untimed 18.8 ms is 37% of the 50.2 ms timed region (budget 20%) |
| embed_diag_default_f64_n1024 | pytorch-cuda | embed_diag_default_f64_n1024 untimed 132.1 ms is 2567% of the 5.1 ms timed region (budget 20%) |
| tril_default_f64_n1048576 | pytorch-cuda | tril_default_f64_n1048576 untimed 144.0 ms is 2755% of the 5.2 ms timed region (budget 20%) |
| triu_default_f64_n1048576 | pytorch-cuda | triu_default_f64_n1048576 untimed 127.5 ms is 2412% of the 5.3 ms timed region (budget 20%) |
| take_along_axis_broadcast_indices_f64_n1048576 | pytorch-cuda | take_along_axis_broadcast_indices_f64_n1048576 untimed 95.8 ms is 623% of the 15.4 ms timed region (budget 20%) |
| scatter_default_f64_n1048576 | pytorch-cuda | scatter_default_f64_n1048576 untimed 66.4 ms is 605% of the 11.0 ms timed region (budget 20%) |
| cast_default_f64_n1048576 | pytorch-cuda | cast_default_f64_n1048576 untimed 54.5 ms is 624% of the 8.7 ms timed region (budget 20%) |
| broadcast_in_dim_f64_n1048576 | pytorch-cuda | broadcast_in_dim_f64_n1048576 untimed 132.5 ms is 3627% of the 3.7 ms timed region (budget 20%) |
| gather_axis0_f64_n1048576 | pytorch-cuda | gather_axis0_f64_n1048576 untimed 56.7 ms is 667% of the 8.5 ms timed region (budget 20%) |
| reshape_copy_f64_n1048576 | pytorch-cuda | reshape_copy_f64_n1048576 untimed 134.2 ms is 3188% of the 4.2 ms timed region (budget 20%) |
| stack_axis0_f64_n1048576 | pytorch-cuda | stack_axis0_f64_n1048576 untimed 447.6 ms is 7671% of the 5.8 ms timed region (budget 20%) |
| pad_default_f64_n1048576 | jax-cuda | pad_default_f64_n1048576 untimed 376.7 ms is 1044% of the 36.1 ms timed region (budget 20%) |
| slice_default_f64_n1048576 | jax-cuda | slice_default_f64_n1048576 untimed 95.9 ms is 249% of the 38.5 ms timed region (budget 20%) |
| reverse_default_f64_n1048576 | jax-cuda | reverse_default_f64_n1048576 untimed 172.8 ms is 553% of the 31.2 ms timed region (budget 20%) |
| concatenate_default_f64_n1048576 | jax-cuda | concatenate_default_f64_n1048576 untimed 322.1 ms is 1317% of the 24.5 ms timed region (budget 20%) |
| extract_diag_default_f64_n1048576 | jax-cuda | extract_diag_default_f64_n1048576 untimed 64.1 ms is 121% of the 52.9 ms timed region (budget 20%) |
| embed_diag_default_f64_n1024 | jax-cuda | embed_diag_default_f64_n1024 untimed 168.6 ms is 649% of the 26.0 ms timed region (budget 20%) |
| tril_default_f64_n1048576 | jax-cuda | tril_default_f64_n1048576 untimed 186.9 ms is 730% of the 25.6 ms timed region (budget 20%) |
| triu_default_f64_n1048576 | jax-cuda | triu_default_f64_n1048576 untimed 173.9 ms is 674% of the 25.8 ms timed region (budget 20%) |
| take_along_axis_broadcast_indices_f64_n1048576 | jax-cuda | take_along_axis_broadcast_indices_f64_n1048576 untimed 126.7 ms is 299% of the 42.4 ms timed region (budget 20%) |
| scatter_default_f64_n1048576 | jax-cuda | scatter_default_f64_n1048576 untimed 108.2 ms is 219% of the 49.3 ms timed region (budget 20%) |
| cast_default_f64_n1048576 | jax-cuda | cast_default_f64_n1048576 untimed 77.2 ms is 200% of the 38.6 ms timed region (budget 20%) |
| broadcast_in_dim_f64_n1048576 | jax-cuda | broadcast_in_dim_f64_n1048576 untimed 174.8 ms is 533% of the 32.8 ms timed region (budget 20%) |
| dynamic_slice_in_bounds_f64_n1048576 | jax-cuda | dynamic_slice_in_bounds_f64_n1048576 untimed 109.5 ms is 262% of the 41.7 ms timed region (budget 20%) |
| gather_axis0_f64_n1048576 | jax-cuda | gather_axis0_f64_n1048576 untimed 101.1 ms is 278% of the 36.3 ms timed region (budget 20%) |
| stack_axis0_f64_n1048576 | jax-cuda | stack_axis0_f64_n1048576 untimed 485.3 ms is 2129% of the 22.8 ms timed region (budget 20%) |