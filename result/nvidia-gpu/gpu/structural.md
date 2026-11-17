# gpu/structural: CUDA public API benchmark

Measured cells: **60**. Unsupported cells: **4**, including **0** numerically verified but timing-excluded cells. Unsupported cells do not count as measured coverage.

Logical traffic: this run measured a plain device copy of 67108864 bytes at **1448.0 GB/s** on this device. Every record carries the logical bytes the case must move per operation and the ratio of the case's implied rate to that copy rate, in the sampling table below. Logical bytes are the declared array sizes, not device DRAM traffic: a case whose operands stay in cache moves less DRAM than its arrays suggest, so a ratio above 1 is recorded, not rejected.

Raw run: `data/results/nvidia-gpu/gpu/structural/20261117_091552/`.

Isolated public APIs and explicit unsupported timing contracts. Issue 155.

This run compiled **1 NVRTC modules** in 1 of its arms, because the CUDA artifact cache started cold. The untimed wall of the cases that compiled is **41.5 s** of its 52.4 s of recorded case wall clock, and that wall also holds their fixture read, transfers and verification, so it is not a compile time.

Values are median (IQR) **µs per operation**. Samples time batches of declared operations plus device synchronization; retained output memory is limited to 256 MiB. Batch counts and raw durations are preserved in JSONL.

Cases marked † (**6** of 16) were sampled with a different number of operations per sample on different arms (see Ops/sample in the sampling table). Each value is still per operation, but the arms amortise launch and synchronization over different batch sizes, so a ratio between their columns is approximate and is not presented as a like-for-like comparison.

Setup, input transfers, library/plan initialization, JIT compilation, validation downloads and output destruction are outside timing. Intrinsic output allocation is inside timing; output-reuse rows instead use preallocated destinations across samples. Accumulating destinations are restored from prepared GPU seeds and synchronized before every batch clock, including initialization and calibration. Direct and eager each use one entered session across samples; direct FFT uses cached `FftExecutor`. JAX eager and compiled JIT are separate arms.

All measured output elements are checked against independently generated NumPy/SciPy reference results before and after sampling. Unsupported rows include unavailable APIs and APIs whose internal setup cannot be excluded; these are not performance measurements. Short batches limited by output memory remain identifiable in the sampling table.

Reshape-copy cases add a leading singleton dimension and materialize the full output. Direct Tensor.reshape and EagerSession.reshape copy device data; PyTorch calls public view_copy. Direct/PyTorch initial/final allocation-pointer checks are untimed and reject input aliasing. Eager reshape copies its input before returning a descriptor view; original CUDA placement/shape/dtype are checked and a public inspection copy is made outside timing. JAX reshape has no established matching owned-output guarantee and remains unsupported; reshape-plus-copy composition is not substituted.

Stack selects actual public Tensor.stack/EagerSession.stack with two distinct inputs and axis 0 or 2. Their intrinsic intermediate allocations are part of this public stack operation. PyTorch/JAX use native stack; final outputs are fully verified. Direct Rust initial/final pointer checks reject aliasing either input outside timing; eager original descriptors are checked before any untimed inspection copy.

| Case | tenferro-cuda-direct | tenferro-cuda-eager | pytorch-cuda | jax-cuda |
| --- | ---: | ---: | ---: | ---: |
| pad_default_f64_n1048576 | 50.154 (4.457) | 48.422 (4.038) | 18.714 (1.515) | 99.278 (4.697) |
| slice_default_f64_n1048576 † | 32.144 (1.197) | unsupported | 10.048 (0.223) | 93.552 (2.683) |
| reverse_default_f64_n1048576 | 29.140 (6.604) | 29.437 (0.785) | 12.288 (0.473) | 71.494 (3.769) |
| concatenate_default_f64_n1048576 | 63.822 (7.886) | 67.240 (11.419) | 25.922 (0.502) | 119.708 (4.334) |
| extract_diag_default_f64_n1048576 † | 24.307 (1.127) | 27.663 (3.365) | 7.434 (0.057) | 63.424 (1.711) |
| embed_diag_default_f64_n1024 | 28.986 (4.566) | 28.636 (5.785) | 12.916 (0.217) | 60.056 (3.338) |
| tril_default_f64_n1048576 | 33.001 (3.720) | 32.894 (3.816) | 13.568 (1.722) | 51.140 (4.832) |
| triu_default_f64_n1048576 | 33.112 (4.207) | 35.091 (4.759) | 13.558 (0.217) | 50.356 (1.432) |
| take_along_axis_broadcast_indices_f64_n1048576 † | 75.432 (3.747) | 89.111 (4.812) | 19.717 (1.019) | 53.783 (0.822) |
| scatter_default_f64_n1048576 | 81.338 (2.890) | 82.582 (5.531) | 26.889 (1.945) | 113.193 (3.213) |
| cast_default_f64_n1048576 † | 30.964 (1.644) | 61.215 (3.640) | 11.110 (1.467) | 90.300 (2.040) |
| broadcast_in_dim_f64_n1048576 | 45.627 (4.375) | unsupported | 9.301 (0.185) | 82.074 (2.369) |
| dynamic_slice_in_bounds_f64_n1048576 † | 28.579 (1.036) | 29.830 (3.073) | unsupported | 98.799 (4.579) |
| gather_axis0_f64_n1048576 † | 53.536 (2.019) | 58.573 (3.895) | 10.609 (0.228) | 92.349 (5.722) |
| reshape_copy_f64_n1048576 | 31.364 (0.569) | 28.538 (3.894) | 10.035 (0.201) | unsupported |
| stack_axis0_f64_n1048576 | 147.424 (7.629) | 154.834 (6.732) | 25.465 (0.158) | 104.777 (0.838) |

## Sampling

| Case | Backend | Ops/sample | Samples | Median batch ms | Bytes/op | Copy ratio | Verification error/tolerance |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| broadcast_in_dim_f64_n1048576 | jax-cuda | 32 | 15 | 2.626 | 8396800 | 0.071 | 0 |
| broadcast_in_dim_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.298 | 8396800 | 0.623 | 0 |
| broadcast_in_dim_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.460 | 8396800 | 0.127 | 0 |
| broadcast_in_dim_f64_n1048576 | tenferro-cuda-eager | — | — | — | 8396800 | unavailable | Eager broadcast returns a lazy broadcast view, not the materialized full output required by this case. |
| cast_default_f64_n1048576 | jax-cuda | 32 | 15 | 2.890 | 12582912 | 0.096 | 0 |
| cast_default_f64_n1048576 | pytorch-cuda | 64 | 15 | 0.711 | 12582912 | 0.782 | 0 |
| cast_default_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.982 | 12582912 | 0.281 | 0 |
| cast_default_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.959 | 12582912 | 0.142 | 0 |
| concatenate_default_f64_n1048576 | jax-cuda | 16 | 15 | 1.915 | 33554432 | 0.194 | 0 |
| concatenate_default_f64_n1048576 | pytorch-cuda | 16 | 15 | 0.415 | 33554432 | 0.894 | 0 |
| concatenate_default_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 1.021 | 33554432 | 0.363 | 0 |
| concatenate_default_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 1.076 | 33554432 | 0.345 | 0 |
| dynamic_slice_in_bounds_f64_n1048576 | jax-cuda | 32 | 15 | 3.162 | 10485776 | 0.073 | 0 |
| dynamic_slice_in_bounds_f64_n1048576 | pytorch-cuda | — | — | — | 10485776 | unavailable | No matched single materializing multidimensional slice API with CUDA-resident start indices and clamping; index construction/composed slicing is not substituted. |
| dynamic_slice_in_bounds_f64_n1048576 | tenferro-cuda-direct | 128 | 15 | 3.658 | 10485776 | 0.253 | 0 |
| dynamic_slice_in_bounds_f64_n1048576 | tenferro-cuda-eager | 64 | 15 | 1.909 | 10485776 | 0.243 | 0 |
| embed_diag_default_f64_n1024 | jax-cuda | 32 | 15 | 1.922 | 8396800 | 0.097 | 0 |
| embed_diag_default_f64_n1024 | pytorch-cuda | 32 | 15 | 0.413 | 8396800 | 0.449 | 0 |
| embed_diag_default_f64_n1024 | tenferro-cuda-direct | 32 | 15 | 0.928 | 8396800 | 0.200 | 0 |
| embed_diag_default_f64_n1024 | tenferro-cuda-eager | 32 | 15 | 0.916 | 8396800 | 0.202 | 0 |
| extract_diag_default_f64_n1048576 | jax-cuda | 32 | 15 | 2.030 | 8396800 | 0.091 | 0 |
| extract_diag_default_f64_n1048576 | pytorch-cuda | 512 | 15 | 3.806 | 8396800 | 0.780 | 0 |
| extract_diag_default_f64_n1048576 | tenferro-cuda-direct | 128 | 15 | 3.111 | 8396800 | 0.239 | 0 |
| extract_diag_default_f64_n1048576 | tenferro-cuda-eager | 128 | 15 | 3.541 | 8396800 | 0.210 | 0 |
| gather_axis0_f64_n1048576 | jax-cuda | 32 | 15 | 2.955 | 12587008 | 0.094 | 0 |
| gather_axis0_f64_n1048576 | pytorch-cuda | 64 | 15 | 0.679 | 12587008 | 0.819 | 0 |
| gather_axis0_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.426 | 12587008 | 0.162 | 0 |
| gather_axis0_f64_n1048576 | tenferro-cuda-eager | 64 | 15 | 3.749 | 12587008 | 0.148 | 0 |
| pad_default_f64_n1048576 | jax-cuda | 31 | 15 | 3.078 | 16908288 | 0.118 | 0 |
| pad_default_f64_n1048576 | pytorch-cuda | 31 | 15 | 0.580 | 16908288 | 0.624 | 0 |
| pad_default_f64_n1048576 | tenferro-cuda-direct | 31 | 15 | 1.555 | 16908288 | 0.233 | 0 |
| pad_default_f64_n1048576 | tenferro-cuda-eager | 31 | 15 | 1.501 | 16908288 | 0.241 | 0 |
| reshape_copy_f64_n1048576 | jax-cuda | — | — | — | 16777216 | unavailable | No established single-call JAX reshape API that guarantees owned output materialization; native reshape may alias input. Reshape plus copy is a different composed contract. |
| reshape_copy_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.321 | 16777216 | 1.155 | 0 |
| reshape_copy_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.004 | 16777216 | 0.369 | 0 |
| reshape_copy_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.913 | 16777216 | 0.406 | 0 |
| reverse_default_f64_n1048576 | jax-cuda | 32 | 15 | 2.288 | 16777216 | 0.162 | 0 |
| reverse_default_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.393 | 16777216 | 0.943 | 0 |
| reverse_default_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.932 | 16777216 | 0.398 | 0 |
| reverse_default_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.942 | 16777216 | 0.394 | 0 |
| scatter_default_f64_n1048576 | jax-cuda | 32 | 15 | 3.622 | 25165824 | 0.154 | 0 |
| scatter_default_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.860 | 25165824 | 0.646 | 0 |
| scatter_default_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.603 | 25165824 | 0.214 | 0 |
| scatter_default_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.643 | 25165824 | 0.210 | 0 |
| slice_default_f64_n1048576 | jax-cuda | 32 | 15 | 2.994 | 12582912 | 0.093 | 0 |
| slice_default_f64_n1048576 | pytorch-cuda | 64 | 15 | 0.643 | 12582912 | 0.865 | 0 |
| slice_default_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.057 | 12582912 | 0.270 | 0 |
| slice_default_f64_n1048576 | tenferro-cuda-eager | — | — | — | 12582912 | unavailable | EagerSession::slice returns a lazy metadata view; it is not equivalent to the materializing direct slice / torch.slice_copy / JAX slice contract. No copy is added inside its timer. |
| stack_axis0_f64_n1048576 | jax-cuda | 16 | 15 | 1.676 | 33554432 | 0.221 | 0 |
| stack_axis0_f64_n1048576 | pytorch-cuda | 16 | 15 | 0.407 | 33554432 | 0.910 | 0 |
| stack_axis0_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.359 | 33554432 | 0.157 | 0 |
| stack_axis0_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.477 | 33554432 | 0.150 | 0 |
| take_along_axis_broadcast_indices_f64_n1048576 | jax-cuda | 64 | 15 | 3.442 | 12587008 | 0.162 | 0 |
| take_along_axis_broadcast_indices_f64_n1048576 | pytorch-cuda | 64 | 15 | 1.262 | 12587008 | 0.441 | 0 |
| take_along_axis_broadcast_indices_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.414 | 12587008 | 0.115 | 0 |
| take_along_axis_broadcast_indices_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.852 | 12587008 | 0.098 | 0 |
| tril_default_f64_n1048576 | jax-cuda | 32 | 15 | 1.636 | 16777216 | 0.227 | 0 |
| tril_default_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.434 | 16777216 | 0.854 | 0 |
| tril_default_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.056 | 16777216 | 0.351 | 0 |
| tril_default_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.053 | 16777216 | 0.352 | 0 |
| triu_default_f64_n1048576 | jax-cuda | 32 | 15 | 1.611 | 16777216 | 0.230 | 0 |
| triu_default_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.434 | 16777216 | 0.855 | 0 |
| triu_default_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.060 | 16777216 | 0.350 | 0 |
| triu_default_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.123 | 16777216 | 0.330 | 0 |

## Run metadata

```yaml
schema_version: 1
target_profile: nvidia-gpu
suite_id: gpu/structural
suite_file: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/data/results/nvidia-gpu/gpu/routine_selection/20261117_091552/gpu_structural.yaml
timestamp: '2026-11-17T09:15:52+00:00'
tenferro_rs:
  path: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/extern/tenferro-rs
  commit: ce9125d0cff4ee40841120759b5df45252c27f22
  dirty: false
  features:
  - cuda
  resolved_path: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/extern/tenferro-rs
harness:
  path: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark
  commit: 2e63ea288fb120038b7a88288485bf2cb7e76866
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
  cpu: AMD EPYC 7713 64-Core Processor
  logical_cpus: 128
  power:
    cpufreq_boost: '1'
    governor: schedutil
    scaling_max_freq_khz: '2000000'
    smt:
      active: '0'
      control: notsupported
  numa:
    nodes: 2
    cpulists:
      '0': 0-63
      '1': 64-127
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
  l3_domains:
  - 0-7
  - 8-15
  - 16-23
  - 24-31
  - 32-39
  - 40-47
  - 48-55
  - 56-63
  - 64-71
  - 72-79
  - 80-87
  - 88-95
  - 96-103
  - 104-111
  - 112-119
  - 120-127
build:
  toolchain:
    rustc: null
    rustc_release: null
    rustflags: null
    target_cpu: null
environment:
  container: devcontainer
  hostname: b8e17af19547
  os: Linux-6.8.0-147-generic-x86_64-with-glibc2.39
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
    - 0-7
    - 8-15
    - 16-23
    - 24-31
    - 32-39
    - 40-47
    - 48-55
    - 56-63
    - 64-71
    - 72-79
    - 80-87
    - 88-95
    - 96-103
    - 104-111
    - 112-119
    - 120-127
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
  device_name: NVIDIA A100-SXM4-80GB
  device_uuid: GPU-704abe6c-29a2-539e-7ccd-01f9de4eed3b
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
  device_name: NVIDIA A100-SXM4-80GB
  buffer_bytes: 67108864
  traffic_bytes_per_copy: 134217728
  traffic_model: one read plus one write of buffer_bytes per copy
  warmups: 3
  sample_ms:
  - 0.09343
  - 0.09155
  - 0.09247
  - 0.0982
  - 0.09269
  - 0.091049
  - 0.09239
  - 0.09502
  - 0.09532
  - 0.09225
  - 0.092109
  - 0.09379
  - 0.09318
  - 0.09191
  - 0.09403
  median_ms: 0.09269
  bytes_per_second: 1448028136800.0864
  torch_version: 2.11.0+cu128
  cuda_version: '12.8'
  timestamp: '2026-11-17T09:37:06.533277+00:00'
gpu_assignment:
  role: benchmark
  physical_device_index: 0
  device_uuid: GPU-704abe6c-29a2-539e-7ccd-01f9de4eed3b
  cuda_visible_devices: GPU-704abe6c-29a2-539e-7ccd-01f9de4eed3b
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
    data/results/nvidia-gpu/gpu/structural/20261117_091552/fixtures/manifest.json: a320918ec2ed38973c5dcd570dd59f21d7fabb60d8cfb2d95483f723ae9fa443
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
        finished_at: '2026-11-17T09:38:08Z'
      pytorch-cuda:
        arm: pytorch-cuda
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-17T09:38:01Z'
      tenferro-cuda-direct:
        arm: tenferro-cuda-direct
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-17T09:37:53Z'
      tenferro-cuda-eager:
        arm: tenferro-cuda-eager
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-17T09:37:56Z'
  compile:
    case_wall_seconds: 52.353
    compiling_processes: 1
    modules_compiled: 1
    compile_time_ms: null
    compile_time_scope: 'not measured: no compile interval was recorded; `compile_modules`
      counts the NVRTC modules and the untimed wall is the case wall'
    untimed_wall_seconds: 41.544
```

## Untimed work over the recorded budget

Each row records the wall clock it spent outside its timed region. These cases spent more than the recorded budget ratio (0.2) there. The threshold has no measured basis, so this is a recorded warning: it does not change a cell, its acceptance or the exit status of the run.

| Case | Backend | Warning |
| --- | --- | --- |
| pad_default_f64_n1048576 | tenferro-cuda-direct | pad_default_f64_n1048576 untimed 182.2 ms is 748% of the 24.4 ms timed region (budget 20%) |
| slice_default_f64_n1048576 | tenferro-cuda-direct | slice_default_f64_n1048576 untimed 44.3 ms is 143% of the 31.0 ms timed region (budget 20%) |
| reverse_default_f64_n1048576 | tenferro-cuda-direct | reverse_default_f64_n1048576 untimed 92.5 ms is 613% of the 15.1 ms timed region (budget 20%) |
| concatenate_default_f64_n1048576 | tenferro-cuda-direct | concatenate_default_f64_n1048576 untimed 303.0 ms is 1890% of the 16.0 ms timed region (budget 20%) |
| extract_diag_default_f64_n1048576 | tenferro-cuda-direct | extract_diag_default_f64_n1048576 untimed 44.2 ms is 95% of the 46.7 ms timed region (budget 20%) |
| embed_diag_default_f64_n1024 | tenferro-cuda-direct | embed_diag_default_f64_n1024 untimed 90.7 ms is 633% of the 14.3 ms timed region (budget 20%) |
| tril_default_f64_n1048576 | tenferro-cuda-direct | tril_default_f64_n1048576 untimed 93.6 ms is 563% of the 16.6 ms timed region (budget 20%) |
| triu_default_f64_n1048576 | tenferro-cuda-direct | triu_default_f64_n1048576 untimed 89.1 ms is 537% of the 16.6 ms timed region (budget 20%) |
| take_along_axis_broadcast_indices_f64_n1048576 | tenferro-cuda-direct | take_along_axis_broadcast_indices_f64_n1048576 untimed 41543.6 ms is 111640% of the 37.2 ms timed region (budget 20%) |
| scatter_default_f64_n1048576 | tenferro-cuda-direct | scatter_default_f64_n1048576 untimed 43.7 ms is 110% of the 39.5 ms timed region (budget 20%) |
| cast_default_f64_n1048576 | tenferro-cuda-direct | cast_default_f64_n1048576 untimed 36.0 ms is 120% of the 30.0 ms timed region (budget 20%) |
| broadcast_in_dim_f64_n1048576 | tenferro-cuda-direct | broadcast_in_dim_f64_n1048576 untimed 38.3 ms is 176% of the 21.7 ms timed region (budget 20%) |
| dynamic_slice_in_bounds_f64_n1048576 | tenferro-cuda-direct | dynamic_slice_in_bounds_f64_n1048576 untimed 20.4 ms is 37% of the 55.1 ms timed region (budget 20%) |
| gather_axis0_f64_n1048576 | tenferro-cuda-direct | gather_axis0_f64_n1048576 untimed 27.3 ms is 53% of the 52.0 ms timed region (budget 20%) |
| reshape_copy_f64_n1048576 | tenferro-cuda-direct | reshape_copy_f64_n1048576 untimed 45.0 ms is 289% of the 15.6 ms timed region (budget 20%) |
| stack_axis0_f64_n1048576 | tenferro-cuda-direct | stack_axis0_f64_n1048576 untimed 71.9 ms is 205% of the 35.1 ms timed region (budget 20%) |
| pad_default_f64_n1048576 | tenferro-cuda-eager | pad_default_f64_n1048576 untimed 183.0 ms is 772% of the 23.7 ms timed region (budget 20%) |
| reverse_default_f64_n1048576 | tenferro-cuda-eager | reverse_default_f64_n1048576 untimed 97.0 ms is 670% of the 14.5 ms timed region (budget 20%) |
| concatenate_default_f64_n1048576 | tenferro-cuda-eager | concatenate_default_f64_n1048576 untimed 179.4 ms is 1102% of the 16.3 ms timed region (budget 20%) |
| extract_diag_default_f64_n1048576 | tenferro-cuda-eager | extract_diag_default_f64_n1048576 untimed 12.2 ms is 23% of the 52.7 ms timed region (budget 20%) |
| embed_diag_default_f64_n1024 | tenferro-cuda-eager | embed_diag_default_f64_n1024 untimed 85.4 ms is 498% of the 17.2 ms timed region (budget 20%) |
| tril_default_f64_n1048576 | tenferro-cuda-eager | tril_default_f64_n1048576 untimed 92.1 ms is 546% of the 16.9 ms timed region (budget 20%) |
| triu_default_f64_n1048576 | tenferro-cuda-eager | triu_default_f64_n1048576 untimed 103.0 ms is 590% of the 17.5 ms timed region (budget 20%) |
| take_along_axis_broadcast_indices_f64_n1048576 | tenferro-cuda-eager | take_along_axis_broadcast_indices_f64_n1048576 untimed 44.2 ms is 97% of the 45.4 ms timed region (budget 20%) |
| scatter_default_f64_n1048576 | tenferro-cuda-eager | scatter_default_f64_n1048576 untimed 105.7 ms is 262% of the 40.3 ms timed region (budget 20%) |
| cast_default_f64_n1048576 | tenferro-cuda-eager | cast_default_f64_n1048576 untimed 78.2 ms is 260% of the 30.1 ms timed region (budget 20%) |
| dynamic_slice_in_bounds_f64_n1048576 | tenferro-cuda-eager | dynamic_slice_in_bounds_f64_n1048576 untimed 20.8 ms is 72% of the 28.8 ms timed region (budget 20%) |
| gather_axis0_f64_n1048576 | tenferro-cuda-eager | gather_axis0_f64_n1048576 untimed 33.2 ms is 58% of the 57.7 ms timed region (budget 20%) |
| reshape_copy_f64_n1048576 | tenferro-cuda-eager | reshape_copy_f64_n1048576 untimed 89.7 ms is 625% of the 14.4 ms timed region (budget 20%) |
| stack_axis0_f64_n1048576 | tenferro-cuda-eager | stack_axis0_f64_n1048576 untimed 220.6 ms is 593% of the 37.2 ms timed region (budget 20%) |
| pad_default_f64_n1048576 | pytorch-cuda | pad_default_f64_n1048576 untimed 293.0 ms is 3186% of the 9.2 ms timed region (budget 20%) |
| slice_default_f64_n1048576 | pytorch-cuda | slice_default_f64_n1048576 untimed 71.0 ms is 731% of the 9.7 ms timed region (budget 20%) |
| reverse_default_f64_n1048576 | pytorch-cuda | reverse_default_f64_n1048576 untimed 151.9 ms is 2495% of the 6.1 ms timed region (budget 20%) |
| concatenate_default_f64_n1048576 | pytorch-cuda | concatenate_default_f64_n1048576 untimed 316.9 ms is 5037% of the 6.3 ms timed region (budget 20%) |
| extract_diag_default_f64_n1048576 | pytorch-cuda | extract_diag_default_f64_n1048576 untimed 23.3 ms is 41% of the 57.2 ms timed region (budget 20%) |
| embed_diag_default_f64_n1024 | pytorch-cuda | embed_diag_default_f64_n1024 untimed 136.6 ms is 2200% of the 6.2 ms timed region (budget 20%) |
| tril_default_f64_n1048576 | pytorch-cuda | tril_default_f64_n1048576 untimed 144.8 ms is 2124% of the 6.8 ms timed region (budget 20%) |
| triu_default_f64_n1048576 | pytorch-cuda | triu_default_f64_n1048576 untimed 148.5 ms is 2228% of the 6.7 ms timed region (budget 20%) |
| take_along_axis_broadcast_indices_f64_n1048576 | pytorch-cuda | take_along_axis_broadcast_indices_f64_n1048576 untimed 104.0 ms is 540% of the 19.3 ms timed region (budget 20%) |
| scatter_default_f64_n1048576 | pytorch-cuda | scatter_default_f64_n1048576 untimed 72.5 ms is 542% of the 13.4 ms timed region (budget 20%) |
| cast_default_f64_n1048576 | pytorch-cuda | cast_default_f64_n1048576 untimed 44.5 ms is 392% of the 11.4 ms timed region (budget 20%) |
| broadcast_in_dim_f64_n1048576 | pytorch-cuda | broadcast_in_dim_f64_n1048576 untimed 147.2 ms is 3141% of the 4.7 ms timed region (budget 20%) |
| gather_axis0_f64_n1048576 | pytorch-cuda | gather_axis0_f64_n1048576 untimed 69.4 ms is 681% of the 10.2 ms timed region (budget 20%) |
| reshape_copy_f64_n1048576 | pytorch-cuda | reshape_copy_f64_n1048576 untimed 151.0 ms is 3051% of the 4.9 ms timed region (budget 20%) |
| stack_axis0_f64_n1048576 | pytorch-cuda | stack_axis0_f64_n1048576 untimed 517.8 ms is 8464% of the 6.1 ms timed region (budget 20%) |
| pad_default_f64_n1048576 | jax-cuda | pad_default_f64_n1048576 untimed 560.8 ms is 1191% of the 47.1 ms timed region (budget 20%) |
| slice_default_f64_n1048576 | jax-cuda | slice_default_f64_n1048576 untimed 122.2 ms is 270% of the 45.2 ms timed region (budget 20%) |
| reverse_default_f64_n1048576 | jax-cuda | reverse_default_f64_n1048576 untimed 349.3 ms is 997% of the 35.0 ms timed region (budget 20%) |
| concatenate_default_f64_n1048576 | jax-cuda | concatenate_default_f64_n1048576 untimed 733.5 ms is 2543% of the 28.8 ms timed region (budget 20%) |
| extract_diag_default_f64_n1048576 | jax-cuda | extract_diag_default_f64_n1048576 untimed 74.6 ms is 237% of the 31.4 ms timed region (budget 20%) |
| embed_diag_default_f64_n1024 | jax-cuda | embed_diag_default_f64_n1024 untimed 221.8 ms is 773% of the 28.7 ms timed region (budget 20%) |
| tril_default_f64_n1048576 | jax-cuda | tril_default_f64_n1048576 untimed 256.5 ms is 1008% of the 25.5 ms timed region (budget 20%) |
| triu_default_f64_n1048576 | jax-cuda | triu_default_f64_n1048576 untimed 278.0 ms is 1151% of the 24.1 ms timed region (budget 20%) |
| take_along_axis_broadcast_indices_f64_n1048576 | jax-cuda | take_along_axis_broadcast_indices_f64_n1048576 untimed 126.4 ms is 242% of the 52.3 ms timed region (budget 20%) |
| scatter_default_f64_n1048576 | jax-cuda | scatter_default_f64_n1048576 untimed 236.0 ms is 421% of the 56.1 ms timed region (budget 20%) |
| cast_default_f64_n1048576 | jax-cuda | cast_default_f64_n1048576 untimed 86.3 ms is 194% of the 44.5 ms timed region (budget 20%) |
| broadcast_in_dim_f64_n1048576 | jax-cuda | broadcast_in_dim_f64_n1048576 untimed 201.5 ms is 504% of the 40.0 ms timed region (budget 20%) |
| dynamic_slice_in_bounds_f64_n1048576 | jax-cuda | dynamic_slice_in_bounds_f64_n1048576 untimed 117.8 ms is 244% of the 48.3 ms timed region (budget 20%) |
| gather_axis0_f64_n1048576 | jax-cuda | gather_axis0_f64_n1048576 untimed 109.6 ms is 245% of the 44.7 ms timed region (budget 20%) |
| stack_axis0_f64_n1048576 | jax-cuda | stack_axis0_f64_n1048576 untimed 741.3 ms is 2930% of the 25.3 ms timed region (budget 20%) |