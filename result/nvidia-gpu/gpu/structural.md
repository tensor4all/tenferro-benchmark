# gpu/structural: CUDA public API benchmark

Measured cells: **60**. Unsupported cells: **4**, including **0** numerically verified but timing-excluded cells. Unsupported cells do not count as measured coverage.

Logical traffic: this run measured a plain device copy of 67108864 bytes at **1033.2 GB/s** on this device. Every record carries the logical bytes the case must move per operation and the ratio of the case's implied rate to that copy rate, in the sampling table below. Logical bytes are the declared array sizes, not device DRAM traffic: a case whose operands stay in cache moves less DRAM than its arrays suggest, so a ratio above 1 is recorded, not rejected.

Raw run: `data/results/nvidia-gpu/gpu/structural/20261114_082852/`.

Isolated public APIs and explicit unsupported timing contracts. Issue 155.

This run compiled **1 NVRTC modules** in 1 of its arms, because the CUDA artifact cache started cold. The untimed wall of the cases that compiled is **71.0 s** of its 86.9 s of recorded case wall clock, and that wall also holds their fixture read, transfers and verification, so it is not a compile time.

Values are median (IQR) **µs per operation**. Samples time batches of declared operations plus device synchronization; retained output memory is limited to 256 MiB. Batch counts and raw durations are preserved in JSONL.

Cases marked † (**16** of 16) were sampled with a different number of operations per sample on different arms (see Ops/sample in the sampling table). Each value is still per operation, but the arms amortise launch and synchronization over different batch sizes, so a ratio between their columns is approximate and is not presented as a like-for-like comparison.

Setup, input transfers, library/plan initialization, JIT compilation, validation downloads and output destruction are outside timing. Intrinsic output allocation is inside timing; output-reuse rows instead use preallocated destinations across samples. Accumulating destinations are restored from prepared GPU seeds and synchronized before every batch clock, including initialization and calibration. Direct and eager each use one entered session across samples; direct FFT uses cached `FftExecutor`. JAX eager and compiled JIT are separate arms.

All measured output elements are checked against independently generated NumPy/SciPy reference results before and after sampling. Unsupported rows include unavailable APIs and APIs whose internal setup cannot be excluded; these are not performance measurements. Short batches limited by output memory remain identifiable in the sampling table.

Reshape-copy cases add a leading singleton dimension and materialize the full output. Direct Tensor.reshape and EagerSession.reshape copy device data; PyTorch calls public view_copy. Direct/PyTorch initial/final allocation-pointer checks are untimed and reject input aliasing. Eager reshape copies its input before returning a descriptor view; original CUDA placement/shape/dtype are checked and a public inspection copy is made outside timing. JAX reshape has no established matching owned-output guarantee and remains unsupported; reshape-plus-copy composition is not substituted.

Stack selects actual public Tensor.stack/EagerSession.stack with two distinct inputs and axis 0 or 2. Their intrinsic intermediate allocations are part of this public stack operation. PyTorch/JAX use native stack; final outputs are fully verified. Direct Rust initial/final pointer checks reject aliasing either input outside timing; eager original descriptors are checked before any untimed inspection copy.

| Case | tenferro-cuda-direct | tenferro-cuda-eager | pytorch-cuda | jax-cuda |
| --- | ---: | ---: | ---: | ---: |
| pad_default_f64_n1048576 † | 60.965 (5.038) | 136.278 (43.109) | 376.026 (304.298) | 2042.997 (794.653) |
| slice_default_f64_n1048576 † | 45.414 (3.208) | unsupported | 173.137 (163.009) | 378.854 (132.279) |
| reverse_default_f64_n1048576 † | 46.226 (2.606) | 155.187 (34.973) | 164.197 (132.244) | 1450.045 (641.772) |
| concatenate_default_f64_n1048576 † | 81.452 (7.630) | 182.670 (34.475) | 149.810 (152.114) | 1832.915 (301.993) |
| extract_diag_default_f64_n1048576 † | 75.638 (24.846) | 137.378 (53.184) | 22.397 (13.250) | 1955.835 (523.669) |
| embed_diag_default_f64_n1024 † | 49.884 (6.910) | 166.170 (65.502) | 66.559 (20.280) | 721.949 (170.867) |
| tril_default_f64_n1048576 † | 39.958 (3.069) | 130.987 (53.114) | 124.317 (28.198) | 2016.457 (527.860) |
| triu_default_f64_n1048576 † | 42.446 (3.405) | 230.616 (271.473) | 56.327 (28.260) | 2043.277 (1190.096) |
| take_along_axis_broadcast_indices_f64_n1048576 † | 384.896 (371.381) | 330.943 (111.502) | 128.263 (153.948) | 600.635 (207.778) |
| scatter_default_f64_n1048576 † | 150.508 (36.972) | 155.292 (27.727) | 55.384 (38.749) | 1504.940 (635.555) |
| cast_default_f64_n1048576 † | 106.744 (42.608) | 194.665 (53.743) | 30.001 (9.952) | 1117.530 (939.330) |
| broadcast_in_dim_f64_n1048576 † | 175.721 (72.766) | unsupported | 10.441 (10.633) | 372.883 (111.694) |
| dynamic_slice_in_bounds_f64_n1048576 † | 133.083 (32.852) | 124.964 (20.752) | unsupported | 785.994 (346.971) |
| gather_axis0_f64_n1048576 † | 161.037 (57.960) | 166.956 (36.833) | 16.033 (8.510) | 562.990 (176.698) |
| reshape_copy_f64_n1048576 † | 167.287 (214.596) | 146.440 (50.050) | 30.905 (19.040) | unsupported |
| stack_axis0_f64_n1048576 † | 628.571 (445.937) | 383.603 (44.000) | 60.937 (33.742) | 1956.952 (541.829) |

## Sampling

| Case | Backend | Ops/sample | Samples | Median batch ms | Bytes/op | Copy ratio | Verification error/tolerance |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| broadcast_in_dim_f64_n1048576 | jax-cuda | 8 | 15 | 2.983 | 8396800 | 0.022 | 0 |
| broadcast_in_dim_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.334 | 8396800 | 0.778 | 0 |
| broadcast_in_dim_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.812 | 8396800 | 0.046 | 0 |
| broadcast_in_dim_f64_n1048576 | tenferro-cuda-eager | — | — | — | 8396800 | unavailable | Eager broadcast returns a lazy broadcast view, not the materialized full output required by this case. |
| cast_default_f64_n1048576 | jax-cuda | 4 | 15 | 4.470 | 12582912 | 0.011 | 0 |
| cast_default_f64_n1048576 | pytorch-cuda | 64 | 15 | 1.920 | 12582912 | 0.406 | 0 |
| cast_default_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.416 | 12582912 | 0.114 | 0 |
| cast_default_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 3.115 | 12582912 | 0.063 | 0 |
| concatenate_default_f64_n1048576 | jax-cuda | 1 | 15 | 1.833 | 33554432 | 0.018 | 0 |
| concatenate_default_f64_n1048576 | pytorch-cuda | 4 | 15 | 0.599 | 33554432 | 0.217 | 0 |
| concatenate_default_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 1.303 | 33554432 | 0.399 | 0 |
| concatenate_default_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.923 | 33554432 | 0.178 | 0 |
| dynamic_slice_in_bounds_f64_n1048576 | jax-cuda | 2 | 15 | 1.572 | 10485776 | 0.013 | 0 |
| dynamic_slice_in_bounds_f64_n1048576 | pytorch-cuda | — | — | — | 10485776 | unavailable | No matched single materializing multidimensional slice API with CUDA-resident start indices and clamping; index construction/composed slicing is not substituted. |
| dynamic_slice_in_bounds_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.129 | 10485776 | 0.076 | 0 |
| dynamic_slice_in_bounds_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 1.999 | 10485776 | 0.081 | 0 |
| embed_diag_default_f64_n1024 | jax-cuda | 4 | 15 | 2.888 | 8396800 | 0.011 | 0 |
| embed_diag_default_f64_n1024 | pytorch-cuda | 16 | 15 | 1.065 | 8396800 | 0.122 | 0 |
| embed_diag_default_f64_n1024 | tenferro-cuda-direct | 32 | 15 | 1.596 | 8396800 | 0.163 | 0 |
| embed_diag_default_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 2.659 | 8396800 | 0.049 | 0 |
| extract_diag_default_f64_n1048576 | jax-cuda | 1 | 15 | 1.956 | 8396800 | 0.004 | 0 |
| extract_diag_default_f64_n1048576 | pytorch-cuda | 64 | 15 | 1.433 | 8396800 | 0.363 | 0 |
| extract_diag_default_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 4.841 | 8396800 | 0.107 | 0 |
| extract_diag_default_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.198 | 8396800 | 0.059 | 0 |
| gather_axis0_f64_n1048576 | jax-cuda | 4 | 15 | 2.252 | 12587008 | 0.022 | 0 |
| gather_axis0_f64_n1048576 | pytorch-cuda | 64 | 15 | 1.026 | 12587008 | 0.760 | 0 |
| gather_axis0_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.577 | 12587008 | 0.076 | 0 |
| gather_axis0_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 5.343 | 12587008 | 0.073 | 0 |
| pad_default_f64_n1048576 | jax-cuda | 1 | 15 | 2.043 | 16908288 | 0.008 | 0 |
| pad_default_f64_n1048576 | pytorch-cuda | 2 | 15 | 0.752 | 16908288 | 0.044 | 0 |
| pad_default_f64_n1048576 | tenferro-cuda-direct | 31 | 15 | 1.890 | 16908288 | 0.268 | 0 |
| pad_default_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.180 | 16908288 | 0.120 | 0 |
| reshape_copy_f64_n1048576 | jax-cuda | — | — | — | 16777216 | unavailable | No established single-call JAX reshape API that guarantees owned output materialization; native reshape may alias input. Reshape plus copy is a different composed contract. |
| reshape_copy_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.989 | 16777216 | 0.525 | 0 |
| reshape_copy_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.677 | 16777216 | 0.097 | 0 |
| reshape_copy_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.343 | 16777216 | 0.111 | 0 |
| reverse_default_f64_n1048576 | jax-cuda | 2 | 15 | 2.900 | 16777216 | 0.011 | 0 |
| reverse_default_f64_n1048576 | pytorch-cuda | 4 | 15 | 0.657 | 16777216 | 0.099 | 0 |
| reverse_default_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.479 | 16777216 | 0.351 | 0 |
| reverse_default_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.483 | 16777216 | 0.105 | 0 |
| scatter_default_f64_n1048576 | jax-cuda | 1 | 15 | 1.505 | 25165824 | 0.016 | 0 |
| scatter_default_f64_n1048576 | pytorch-cuda | 32 | 15 | 1.772 | 25165824 | 0.440 | 0 |
| scatter_default_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.408 | 25165824 | 0.162 | 0 |
| scatter_default_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.485 | 25165824 | 0.157 | 0 |
| slice_default_f64_n1048576 | jax-cuda | 8 | 15 | 3.031 | 12582912 | 0.032 | 0 |
| slice_default_f64_n1048576 | pytorch-cuda | 4 | 15 | 0.693 | 12582912 | 0.070 | 0 |
| slice_default_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.907 | 12582912 | 0.268 | 0 |
| slice_default_f64_n1048576 | tenferro-cuda-eager | — | — | — | 12582912 | unavailable | EagerSession::slice returns a lazy metadata view; it is not equivalent to the materializing direct slice / torch.slice_copy / JAX slice contract. No copy is added inside its timer. |
| stack_axis0_f64_n1048576 | jax-cuda | 1 | 15 | 1.957 | 33554432 | 0.017 | 0 |
| stack_axis0_f64_n1048576 | pytorch-cuda | 16 | 15 | 0.975 | 33554432 | 0.533 | 0 |
| stack_axis0_f64_n1048576 | tenferro-cuda-direct | 2 | 15 | 1.257 | 33554432 | 0.052 | 0 |
| stack_axis0_f64_n1048576 | tenferro-cuda-eager | 8 | 15 | 3.069 | 33554432 | 0.085 | 0 |
| take_along_axis_broadcast_indices_f64_n1048576 | jax-cuda | 4 | 15 | 2.403 | 12587008 | 0.020 | 0 |
| take_along_axis_broadcast_indices_f64_n1048576 | pytorch-cuda | 8 | 15 | 1.026 | 12587008 | 0.095 | 0 |
| take_along_axis_broadcast_indices_f64_n1048576 | tenferro-cuda-direct | 4 | 15 | 1.540 | 12587008 | 0.032 | 0 |
| take_along_axis_broadcast_indices_f64_n1048576 | tenferro-cuda-eager | 8 | 15 | 2.648 | 12587008 | 0.037 | 0 |
| tril_default_f64_n1048576 | jax-cuda | 1 | 15 | 2.016 | 16777216 | 0.008 | 0 |
| tril_default_f64_n1048576 | pytorch-cuda | 8 | 15 | 0.995 | 16777216 | 0.131 | 0 |
| tril_default_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.279 | 16777216 | 0.406 | 0 |
| tril_default_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.096 | 16777216 | 0.124 | 0 |
| triu_default_f64_n1048576 | jax-cuda | 1 | 15 | 2.043 | 16777216 | 0.008 | 0 |
| triu_default_f64_n1048576 | pytorch-cuda | 16 | 15 | 0.901 | 16777216 | 0.288 | 0 |
| triu_default_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.358 | 16777216 | 0.383 | 0 |
| triu_default_f64_n1048576 | tenferro-cuda-eager | 4 | 15 | 0.922 | 16777216 | 0.070 | 0 |

## Run metadata

```yaml
schema_version: 1
target_profile: nvidia-gpu
suite_id: gpu/structural
suite_file: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/data/results/nvidia-gpu/gpu/routine_selection/20261114_082852/gpu_structural.yaml
timestamp: '2026-11-14T08:28:52+00:00'
tenferro_rs:
  path: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/extern/tenferro-rs
  commit: 9a850b06155405acc99ecd0648829e9e6d3866ba
  dirty: false
  features:
  - cuda
  resolved_path: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/extern/tenferro-rs
harness:
  path: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark
  commit: 76cb137693f0b16de2556cc47d32686deaa05863
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
  cpu: AMD EPYC-Genoa Processor
  logical_cpus: 116
  power:
    smt:
      active: '1'
      control: 'on'
  numa:
    nodes: 1
    cpulists:
      '0': 0-115
  smt_siblings:
  - 0-1
  - 2-3
  - 4-5
  - 6-7
  - 8-9
  - 10-11
  - 12-13
  - 14-15
  - 16-17
  - 18-19
  - 20-21
  - 22-23
  - 24-25
  - 26-27
  - 28-29
  - 30-31
  - 32-33
  - 34-35
  - 36-37
  - 38-39
  - 40-41
  - 42-43
  - 44-45
  - 46-47
  - 48-49
  - 50-51
  - 52-53
  - 54-55
  - 56-57
  - 58-59
  - 60-61
  - 62-63
  - 64-65
  - 66-67
  - 68-69
  - 70-71
  - 72-73
  - 74-75
  - 76-77
  - 78-79
  - 80-81
  - 82-83
  - 84-85
  - 86-87
  - 88-89
  - 90-91
  - 92-93
  - 94-95
  - 96-97
  - 98-99
  - 100-101
  - 102-103
  - 104-105
  - 106-107
  - 108-109
  - 110-111
  - 112-113
  - 114-115
  l3_domains:
  - 0-115
build:
  toolchain:
    rustc: null
    rustc_release: null
    rustflags: null
    target_cpu: null
environment:
  container: devcontainer
  hostname: dba65816ff33
  os: Linux-6.8.0-147-generic-x86_64-with-glibc2.39
  arch: x86_64
  cpu: x86_64
  affinity:
    pinning: none
    requested_measured_cpus: null
    observer_cpus: null
    override_of: null
    thread_budget: '1'
    measured_cpus: 0-115
    l3_domains:
    - 0-115
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
    GPU_RUNTIME_SHA256: f876805d95374305db5787a72cee40a6848d0fc64ee71d7b306c264971c36869
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
  device_uuid: GPU-c70851ec-1874-5bb5-8a0a-85e4eed17212
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
  - 0.222654
  - 0.134375
  - 0.125435
  - 0.129625
  - 0.12739
  - 0.280761
  - 0.191365
  - 0.133537
  - 0.146387
  - 0.129904
  - 0.233829
  - 0.122921
  - 0.129905
  - 0.125715
  - 0.126273
  median_ms: 0.129905
  bytes_per_second: 1033199091643.8936
  torch_version: 2.11.0+cu128
  cuda_version: '12.8'
  timestamp: '2026-11-14T08:46:09.618873+00:00'
gpu_assignment:
  role: benchmark
  physical_device_index: 0
  device_uuid: GPU-c70851ec-1874-5bb5-8a0a-85e4eed17212
  cuda_visible_devices: GPU-c70851ec-1874-5bb5-8a0a-85e4eed17212
  process_device_ordinal: 0
  verification_only: false
collection:
  command: BENCHMARK_TARGET_PROFILE=nvidia-gpu GPU_BENCH_DEVICE=0 BENCH_RUNS=15 BENCH_WARMUPS=3
    ./scripts/run_gpu_public_api.sh structural
  devcontainer_config: .devcontainer/cuda/devcontainer.json
  expected_cases: 16
  selected_cases: 16
  sha256:
    .devcontainer/cuda/devcontainer.json: 0bf1465bbd1a4c9f896d5320838f2d5a488632a59af7f9b59cc050e95e0b8e09
    .devcontainer/cuda/Dockerfile: 578d9815e5145aa1651ac7e93833468c345ee8bf5e95bfa7717551580ae9feca
    src/bin/benchmark_gpu_public_api.rs: 3c723c1d0c1658d8499a7713ced3b30357447f661dae37fca7fa4ce2c72de804
    src/bin/benchmark_gpu_public_api/lifetime.rs: 34be366f56aac88f5f3b86ade8018a15dad2bca85cea74f56cb2fe121917d0a9
    src/bin/benchmark_gpu_public_api/linalg.rs: 9c3d7916a3281d2dfd1caa68551dc9c81c79dab6b7525324339e5c4816581e6d
    src/bin/benchmark_gpu_public_api/multi.rs: 512b659c731722a152dbdf2e41569974aa84442dfc1f739afbe234e08c71c93b
    src/bin/benchmark_gpu_public_api/plan.rs: 70fce8a6e9ed7518779a51a9d8692e010fa1e470e95eee0ce5276f595737fde8
    src/bin/benchmark_gpu_public_api/read_layout.rs: fc42e2160094ba5f9c7dd6facb5667a76e10b73f971ef9facfb1f64723e42467
    src/bin/benchmark_gpu_public_api/reuse.rs: a5a4aa0edb0a1158f6669b642f4bc25a7079314fc2a711ef3ef57cc19c0a35b2
    src/bin/benchmark_gpu_public_api/runtime_ops.rs: c9d61b1e99bda517a59a6e34f625fd4d651bbbb19f6dc06ba4d0d1c0d50dfd6a
    src/bin/benchmark_gpu_public_api/state.rs: a8bbf8985bbc58c435e3c4dda7135161a141a90a10fa50d4f95128397ce949cb
    src/bin/benchmark_gpu_public_api/structural.rs: 8b61ac01b58edc090eafbc124ec135d5efeb4979631a939656ba78dd6e5c042a
    /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/scripts/benchmark_gpu_public_api.py: d1ff68f86496d991bcbc730138fe6e3076001b3f23cc1a94c44b96f6857cb8ed
    /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/scripts/gpu_public_api_ops.py: e88823db3f4a1fe43dc13c4743c9d78ece6af76a968724091af318dd8d7f2528
    scripts/run_gpu_public_api.sh: b0e1cdcefc95a1f2170739eeffbc2e9600eb3acd363566b605458401b04ec37e
    data/results/nvidia-gpu/gpu/structural/20261114_082852/fixtures/manifest.json: a320918ec2ed38973c5dcd570dd59f21d7fabb60d8cfb2d95483f723ae9fa443
    target/release/benchmark_gpu_public_api: 25f583eab659e7cc2c88a104c8203f0c2f400fd4316bfed75f479ee99d3f16bb
  reference_numpy_version: 2.4.2
  measurement_cpu:
    policy: 'disabled (GPU_BENCH_CPU_SELECT unset): arms run on the allocation unpinned'
    arms:
      jax-cuda:
        arm: jax-cuda
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-14T08:47:57Z'
      pytorch-cuda:
        arm: pytorch-cuda
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-14T08:47:45Z'
      tenferro-cuda-direct:
        arm: tenferro-cuda-direct
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-14T08:47:26Z'
      tenferro-cuda-eager:
        arm: tenferro-cuda-eager
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-14T08:47:30Z'
  compile:
    case_wall_seconds: 86.887
    compiling_processes: 1
    modules_compiled: 1
    compile_time_ms: null
    compile_time_scope: 'not measured: no compile interval was recorded; `compile_modules`
      counts the NVRTC modules and the untimed wall is the case wall'
    untimed_wall_seconds: 70.981
```

## Untimed work over the recorded budget

Each row records the wall clock it spent outside its timed region. These cases spent more than the recorded budget ratio (0.2) there. The threshold has no measured basis, so this is a recorded warning: it does not change a cell, its acceptance or the exit status of the run.

| Case | Backend | Warning |
| --- | --- | --- |
| pad_default_f64_n1048576 | tenferro-cuda-direct | pad_default_f64_n1048576 untimed 160.8 ms is 555% of the 29.0 ms timed region (budget 20%) |
| slice_default_f64_n1048576 | tenferro-cuda-direct | slice_default_f64_n1048576 untimed 38.5 ms is 88% of the 43.6 ms timed region (budget 20%) |
| reverse_default_f64_n1048576 | tenferro-cuda-direct | reverse_default_f64_n1048576 untimed 105.9 ms is 468% of the 22.6 ms timed region (budget 20%) |
| concatenate_default_f64_n1048576 | tenferro-cuda-direct | concatenate_default_f64_n1048576 untimed 221.9 ms is 1130% of the 19.6 ms timed region (budget 20%) |
| embed_diag_default_f64_n1024 | tenferro-cuda-direct | embed_diag_default_f64_n1024 untimed 99.4 ms is 298% of the 33.4 ms timed region (budget 20%) |
| tril_default_f64_n1048576 | tenferro-cuda-direct | tril_default_f64_n1048576 untimed 100.7 ms is 510% of the 19.8 ms timed region (budget 20%) |
| triu_default_f64_n1048576 | tenferro-cuda-direct | triu_default_f64_n1048576 untimed 73.7 ms is 174% of the 42.3 ms timed region (budget 20%) |
| take_along_axis_broadcast_indices_f64_n1048576 | tenferro-cuda-direct | take_along_axis_broadcast_indices_f64_n1048576 untimed 70980.7 ms is 194920% of the 36.4 ms timed region (budget 20%) |
| scatter_default_f64_n1048576 | tenferro-cuda-direct | scatter_default_f64_n1048576 untimed 59.0 ms is 141% of the 41.9 ms timed region (budget 20%) |
| cast_default_f64_n1048576 | tenferro-cuda-direct | cast_default_f64_n1048576 untimed 51.9 ms is 90% of the 57.4 ms timed region (budget 20%) |
| broadcast_in_dim_f64_n1048576 | tenferro-cuda-direct | broadcast_in_dim_f64_n1048576 untimed 41.2 ms is 88% of the 47.0 ms timed region (budget 20%) |
| dynamic_slice_in_bounds_f64_n1048576 | tenferro-cuda-direct | dynamic_slice_in_bounds_f64_n1048576 untimed 19.6 ms is 61% of the 32.2 ms timed region (budget 20%) |
| gather_axis0_f64_n1048576 | tenferro-cuda-direct | gather_axis0_f64_n1048576 untimed 30.9 ms is 57% of the 54.5 ms timed region (budget 20%) |
| reshape_copy_f64_n1048576 | tenferro-cuda-direct | reshape_copy_f64_n1048576 untimed 58.9 ms is 99% of the 59.2 ms timed region (budget 20%) |
| stack_axis0_f64_n1048576 | tenferro-cuda-direct | stack_axis0_f64_n1048576 untimed 91.6 ms is 405% of the 22.6 ms timed region (budget 20%) |
| pad_default_f64_n1048576 | tenferro-cuda-eager | pad_default_f64_n1048576 untimed 267.4 ms is 666% of the 40.2 ms timed region (budget 20%) |
| reverse_default_f64_n1048576 | tenferro-cuda-eager | reverse_default_f64_n1048576 untimed 127.8 ms is 324% of the 39.5 ms timed region (budget 20%) |
| concatenate_default_f64_n1048576 | tenferro-cuda-eager | concatenate_default_f64_n1048576 untimed 268.1 ms is 600% of the 44.7 ms timed region (budget 20%) |
| extract_diag_default_f64_n1048576 | tenferro-cuda-eager | extract_diag_default_f64_n1048576 untimed 16.4 ms is 42% of the 38.6 ms timed region (budget 20%) |
| embed_diag_default_f64_n1024 | tenferro-cuda-eager | embed_diag_default_f64_n1024 untimed 113.9 ms is 281% of the 40.6 ms timed region (budget 20%) |
| tril_default_f64_n1048576 | tenferro-cuda-eager | tril_default_f64_n1048576 untimed 154.8 ms is 445% of the 34.8 ms timed region (budget 20%) |
| triu_default_f64_n1048576 | tenferro-cuda-eager | triu_default_f64_n1048576 untimed 154.6 ms is 720% of the 21.5 ms timed region (budget 20%) |
| take_along_axis_broadcast_indices_f64_n1048576 | tenferro-cuda-eager | take_along_axis_broadcast_indices_f64_n1048576 untimed 57.9 ms is 124% of the 46.6 ms timed region (budget 20%) |
| scatter_default_f64_n1048576 | tenferro-cuda-eager | scatter_default_f64_n1048576 untimed 179.3 ms is 423% of the 42.4 ms timed region (budget 20%) |
| cast_default_f64_n1048576 | tenferro-cuda-eager | cast_default_f64_n1048576 untimed 118.0 ms is 237% of the 49.9 ms timed region (budget 20%) |
| dynamic_slice_in_bounds_f64_n1048576 | tenferro-cuda-eager | dynamic_slice_in_bounds_f64_n1048576 untimed 23.8 ms is 75% of the 31.7 ms timed region (budget 20%) |
| gather_axis0_f64_n1048576 | tenferro-cuda-eager | gather_axis0_f64_n1048576 untimed 38.6 ms is 45% of the 85.5 ms timed region (budget 20%) |
| reshape_copy_f64_n1048576 | tenferro-cuda-eager | reshape_copy_f64_n1048576 untimed 119.4 ms is 323% of the 36.9 ms timed region (budget 20%) |
| stack_axis0_f64_n1048576 | tenferro-cuda-eager | stack_axis0_f64_n1048576 untimed 241.8 ms is 536% of the 45.1 ms timed region (budget 20%) |
| pad_default_f64_n1048576 | pytorch-cuda | pad_default_f64_n1048576 untimed 652.1 ms is 6615% of the 9.9 ms timed region (budget 20%) |
| slice_default_f64_n1048576 | pytorch-cuda | slice_default_f64_n1048576 untimed 141.6 ms is 1450% of the 9.8 ms timed region (budget 20%) |
| reverse_default_f64_n1048576 | pytorch-cuda | reverse_default_f64_n1048576 untimed 302.5 ms is 3034% of the 10.0 ms timed region (budget 20%) |
| concatenate_default_f64_n1048576 | pytorch-cuda | concatenate_default_f64_n1048576 untimed 572.0 ms is 5753% of the 9.9 ms timed region (budget 20%) |
| extract_diag_default_f64_n1048576 | pytorch-cuda | extract_diag_default_f64_n1048576 untimed 65.7 ms is 283% of the 23.3 ms timed region (budget 20%) |
| embed_diag_default_f64_n1024 | pytorch-cuda | embed_diag_default_f64_n1024 untimed 277.9 ms is 1398% of the 19.9 ms timed region (budget 20%) |
| tril_default_f64_n1048576 | pytorch-cuda | tril_default_f64_n1048576 untimed 324.0 ms is 2068% of the 15.7 ms timed region (budget 20%) |
| triu_default_f64_n1048576 | pytorch-cuda | triu_default_f64_n1048576 untimed 328.8 ms is 2005% of the 16.4 ms timed region (budget 20%) |
| take_along_axis_broadcast_indices_f64_n1048576 | pytorch-cuda | take_along_axis_broadcast_indices_f64_n1048576 untimed 226.4 ms is 1164% of the 19.4 ms timed region (budget 20%) |
| scatter_default_f64_n1048576 | pytorch-cuda | scatter_default_f64_n1048576 untimed 216.5 ms is 728% of the 29.8 ms timed region (budget 20%) |
| cast_default_f64_n1048576 | pytorch-cuda | cast_default_f64_n1048576 untimed 122.3 ms is 440% of the 27.8 ms timed region (budget 20%) |
| broadcast_in_dim_f64_n1048576 | pytorch-cuda | broadcast_in_dim_f64_n1048576 untimed 217.0 ms is 3025% of the 7.2 ms timed region (budget 20%) |
| gather_axis0_f64_n1048576 | pytorch-cuda | gather_axis0_f64_n1048576 untimed 104.2 ms is 463% of the 22.5 ms timed region (budget 20%) |
| reshape_copy_f64_n1048576 | pytorch-cuda | reshape_copy_f64_n1048576 untimed 217.2 ms is 1173% of the 18.5 ms timed region (budget 20%) |
| stack_axis0_f64_n1048576 | pytorch-cuda | stack_axis0_f64_n1048576 untimed 468.1 ms is 3233% of the 14.5 ms timed region (budget 20%) |
| pad_default_f64_n1048576 | jax-cuda | pad_default_f64_n1048576 untimed 900.7 ms is 3084% of the 29.2 ms timed region (budget 20%) |
| slice_default_f64_n1048576 | jax-cuda | slice_default_f64_n1048576 untimed 255.2 ms is 582% of the 43.9 ms timed region (budget 20%) |
| reverse_default_f64_n1048576 | jax-cuda | reverse_default_f64_n1048576 untimed 474.9 ms is 1151% of the 41.3 ms timed region (budget 20%) |
| concatenate_default_f64_n1048576 | jax-cuda | concatenate_default_f64_n1048576 untimed 514.8 ms is 1978% of the 26.0 ms timed region (budget 20%) |
| extract_diag_default_f64_n1048576 | jax-cuda | extract_diag_default_f64_n1048576 untimed 257.0 ms is 696% of the 36.9 ms timed region (budget 20%) |
| embed_diag_default_f64_n1024 | jax-cuda | embed_diag_default_f64_n1024 untimed 528.4 ms is 1135% of the 46.5 ms timed region (budget 20%) |
| tril_default_f64_n1048576 | jax-cuda | tril_default_f64_n1048576 untimed 353.2 ms is 1221% of the 28.9 ms timed region (budget 20%) |
| triu_default_f64_n1048576 | jax-cuda | triu_default_f64_n1048576 untimed 442.2 ms is 1354% of the 32.7 ms timed region (budget 20%) |
| take_along_axis_broadcast_indices_f64_n1048576 | jax-cuda | take_along_axis_broadcast_indices_f64_n1048576 untimed 333.8 ms is 674% of the 49.6 ms timed region (budget 20%) |
| scatter_default_f64_n1048576 | jax-cuda | scatter_default_f64_n1048576 untimed 366.8 ms is 1174% of the 31.2 ms timed region (budget 20%) |
| cast_default_f64_n1048576 | jax-cuda | cast_default_f64_n1048576 untimed 197.7 ms is 216% of the 91.6 ms timed region (budget 20%) |
| broadcast_in_dim_f64_n1048576 | jax-cuda | broadcast_in_dim_f64_n1048576 untimed 416.6 ms is 867% of the 48.0 ms timed region (budget 20%) |
| dynamic_slice_in_bounds_f64_n1048576 | jax-cuda | dynamic_slice_in_bounds_f64_n1048576 untimed 471.4 ms is 1948% of the 24.2 ms timed region (budget 20%) |
| gather_axis0_f64_n1048576 | jax-cuda | gather_axis0_f64_n1048576 untimed 305.6 ms is 658% of the 46.4 ms timed region (budget 20%) |
| stack_axis0_f64_n1048576 | jax-cuda | stack_axis0_f64_n1048576 untimed 688.9 ms is 2177% of the 31.6 ms timed region (budget 20%) |