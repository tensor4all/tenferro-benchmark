# gpu/nn: CUDA public API benchmark

Measured cells: **21**. Unsupported cells: **23**, including **20** numerically verified but timing-excluded cells. Unsupported cells do not count as measured coverage.

Logical traffic: this run measured a plain device copy of 67108864 bytes at **1411.2 GB/s** on this device. Every record carries the logical bytes the case must move per operation and the ratio of the case's implied rate to that copy rate, in the sampling table below. Logical bytes are the declared array sizes, not device DRAM traffic: a case whose operands stay in cache moves less DRAM than its arrays suggest, so a ratio above 1 is recorded, not rejected.

Raw run: `data/results/nvidia-gpu/gpu/nn/20261121_082311/`.

Isolated public APIs and explicit unsupported timing contracts. Issue 155.

This run compiled **1 NVRTC modules** in 1 of its arms, because the CUDA artifact cache started cold. The untimed wall of the cases that compiled is **0.2 s** of its 7.2 s of recorded case wall clock, and that wall also holds their fixture read, transfers and verification, so it is not a compile time.

Values are median (IQR) **µs per operation**. Samples time batches of declared operations plus device synchronization; retained output memory is limited to 256 MiB. Batch counts and raw durations are preserved in JSONL.

Cases marked † (**3** of 11) were sampled with a different number of operations per sample on different arms (see Ops/sample in the sampling table). Each value is still per operation, but the arms amortise launch and synchronization over different batch sizes, so a ratio between their columns is approximate and is not presented as a like-for-like comparison.

Setup, input transfers, library/plan initialization, JIT compilation, validation downloads and output destruction are outside timing. Intrinsic output allocation is inside timing; output-reuse rows instead use preallocated destinations across samples. Accumulating destinations are restored from prepared GPU seeds and synchronized before every batch clock, including initialization and calibration. Direct and eager each use one entered session across samples; direct FFT uses cached `FftExecutor`. JAX eager and compiled JIT are separate arms.

All measured output elements are checked against independently generated NumPy/SciPy reference results before and after sampling. Unsupported rows include unavailable APIs and APIs whose internal setup cannot be excluded; these are not performance measurements. Short batches limited by output memory remain identifiable in the sampling table.

| Case | tenferro-cuda-direct | tenferro-cuda-eager | pytorch-cuda | jax-cuda |
| --- | ---: | ---: | ---: | ---: |
| sigmoid_f64_n1048576 | unsupported | unsupported | 10.022 (0.056) | 65.176 (2.360) |
| silu_f64_n1048576 | unsupported | unsupported | 11.166 (0.060) | 58.888 (1.442) |
| softplus_f64_n1048576 | 38.553 (0.429) | 61.205 (0.495) | 25.796 (0.064) | 65.964 (2.139) |
| gelu_f64_n1048576 † | unsupported | unsupported | 15.061 (0.044) | 671.298 (14.345) |
| gelu_tanh_f64_n1048576 † | unsupported | unsupported | 9.439 (0.051) | 926.401 (55.309) |
| softmax_f64_n1048576 † | unsupported | unsupported | 16.561 (0.039) | 437.580 (55.997) |
| log_softmax_f64_n1048576 | unsupported | unsupported | 13.389 (0.019) | 67.558 (9.260) |
| masked_softmax_f64_n1048576 | unsupported | unsupported | unsupported | 706.289 (39.482) |
| masked_log_softmax_f64_n1048576 | unsupported | unsupported | unsupported | 57.138 (4.953) |
| layer_norm_f64_n1048576 | unsupported | unsupported | 30.436 (0.092) | 111.669 (2.730) |
| rms_norm_f64_n1048576 | unsupported | unsupported | 30.343 (0.128) | unsupported |

## Sampling

| Case | Backend | Ops/sample | Samples | Median batch ms | Bytes/op | Copy ratio | Verification error/tolerance |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gelu_f64_n1048576 | jax-cuda | 4 | 15 | 2.685 | 16777216 | 0.018 | 4.81e-08 |
| gelu_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.482 | 16777216 | 0.789 | 4.899e-08 |
| gelu_f64_n1048576 | tenferro-cuda-direct | — | — | — | 16777216 | unavailable | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| gelu_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| gelu_tanh_f64_n1048576 | jax-cuda | 4 | 15 | 3.706 | 16777216 | 0.013 | 4.349e-08 |
| gelu_tanh_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.302 | 16777216 | 1.259 | 4.349e-08 |
| gelu_tanh_f64_n1048576 | tenferro-cuda-direct | — | — | — | 16777216 | unavailable | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| gelu_tanh_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| layer_norm_f64_n1048576 | jax-cuda | 32 | 15 | 3.573 | 16777216 | 0.106 | 3.398e-07 |
| layer_norm_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.974 | 16777216 | 0.391 | 3.138e-07 |
| layer_norm_f64_n1048576 | tenferro-cuda-direct | — | — | — | 16777216 | unavailable | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| layer_norm_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| log_softmax_f64_n1048576 | jax-cuda | 32 | 15 | 2.162 | 16777216 | 0.176 | 2.377e-07 |
| log_softmax_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.428 | 16777216 | 0.888 | 2.377e-07 |
| log_softmax_f64_n1048576 | tenferro-cuda-direct | — | — | — | 16777216 | unavailable | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| log_softmax_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| masked_log_softmax_f64_n1048576 | jax-cuda | 32 | 15 | 1.828 | 17825792 | 0.221 | 2.489e-07 |
| masked_log_softmax_f64_n1048576 | pytorch-cuda | — | — | — | 17825792 | unavailable | No equivalent stable single PyTorch public masked-softmax operation; masked_fill plus softmax would change the operation contract. |
| masked_log_softmax_f64_n1048576 | tenferro-cuda-direct | — | — | — | 17825792 | unavailable | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| masked_log_softmax_f64_n1048576 | tenferro-cuda-eager | — | — | — | 17825792 | unavailable | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| masked_softmax_f64_n1048576 | jax-cuda | 4 | 15 | 2.825 | 17825792 | 0.018 | 8.659e-10 |
| masked_softmax_f64_n1048576 | pytorch-cuda | — | — | — | 17825792 | unavailable | No equivalent stable single PyTorch public masked-softmax operation; masked_fill plus softmax would change the operation contract. |
| masked_softmax_f64_n1048576 | tenferro-cuda-direct | — | — | — | 17825792 | unavailable | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| masked_softmax_f64_n1048576 | tenferro-cuda-eager | — | — | — | 17825792 | unavailable | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| rms_norm_f64_n1048576 | jax-cuda | — | — | — | 16777216 | unavailable | No equivalent single public JAX RMS-normalization operation; a composition is not timed as the missing API. |
| rms_norm_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.971 | 16777216 | 0.392 | 2.654e-07 |
| rms_norm_f64_n1048576 | tenferro-cuda-direct | — | — | — | 16777216 | unavailable | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| rms_norm_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| sigmoid_f64_n1048576 | jax-cuda | 32 | 15 | 2.086 | 16777216 | 0.182 | 7.964e-08 |
| sigmoid_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.321 | 16777216 | 1.186 | 7.964e-08 |
| sigmoid_f64_n1048576 | tenferro-cuda-direct | — | — | — | 16777216 | unavailable | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| sigmoid_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| silu_f64_n1048576 | jax-cuda | 32 | 15 | 1.884 | 16777216 | 0.202 | 4.848e-08 |
| silu_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.357 | 16777216 | 1.065 | 4.848e-08 |
| silu_f64_n1048576 | tenferro-cuda-direct | — | — | — | 16777216 | unavailable | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| silu_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| softmax_f64_n1048576 | jax-cuda | 8 | 15 | 3.501 | 16777216 | 0.027 | 6.499e-10 |
| softmax_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.530 | 16777216 | 0.718 | 8.661e-10 |
| softmax_f64_n1048576 | tenferro-cuda-direct | — | — | — | 16777216 | unavailable | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| softmax_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| softplus_f64_n1048576 | jax-cuda | 32 | 15 | 2.111 | 16777216 | 0.180 | 1.467e-07 |
| softplus_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.825 | 16777216 | 0.461 | 1.467e-07 |
| softplus_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.234 | 16777216 | 0.308 | 1.467e-07 |
| softplus_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.959 | 16777216 | 0.194 | 1.467e-07 |

## Run metadata

```yaml
schema_version: 1
target_profile: nvidia-gpu
suite_id: gpu/nn
suite_file: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/data/results/nvidia-gpu/gpu/routine_selection/20261121_082311/gpu_nn.yaml
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
  - 0.09434
  - 0.091691
  - 0.097541
  - 0.095811
  - 0.092861
  - 0.095001
  - 0.095191
  - 0.093631
  - 0.09511
  - 0.095521
  - 0.094061
  - 0.095831
  - 0.096871
  - 0.097291
  - 0.095051
  median_ms: 0.09511
  bytes_per_second: 1411184186731.1533
  torch_version: 2.11.0+cu128
  cuda_version: '12.8'
  timestamp: '2026-11-21T08:33:55.526079+00:00'
gpu_assignment:
  role: benchmark
  physical_device_index: 0
  device_uuid: GPU-98d8618e-77c8-160a-92f2-117e34fd43df
  cuda_visible_devices: GPU-98d8618e-77c8-160a-92f2-117e34fd43df
  process_device_ordinal: 0
  verification_only: false
collection:
  command: BENCHMARK_TARGET_PROFILE=nvidia-gpu GPU_BENCH_DEVICE=0 BENCH_RUNS=15 BENCH_WARMUPS=3
    ./scripts/run_gpu_public_api.sh nn
  devcontainer_config: .devcontainer/cuda/devcontainer.json
  expected_cases: 11
  selected_cases: 11
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
    data/results/nvidia-gpu/gpu/nn/20261121_082311/fixtures/manifest.json: 6019f8eb30ec023ad89d15545a6d050a2188b62ec40b3d90c773ad6e58eabea1
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
        finished_at: '2026-11-21T08:34:09Z'
      pytorch-cuda:
        arm: pytorch-cuda
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-21T08:34:03Z'
      tenferro-cuda-direct:
        arm: tenferro-cuda-direct
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-21T08:33:58Z'
      tenferro-cuda-eager:
        arm: tenferro-cuda-eager
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-21T08:33:59Z'
  compile:
    case_wall_seconds: 7.249
    compiling_processes: 1
    modules_compiled: 1
    compile_time_ms: null
    compile_time_scope: 'not measured: no compile interval was recorded; `compile_modules`
      counts the NVRTC modules and the untimed wall is the case wall'
    untimed_wall_seconds: 0.152
```

## Untimed work over the recorded budget

Each row records the wall clock it spent outside its timed region. These cases spent more than the recorded budget ratio (0.2) there. The threshold has no measured basis, so this is a recorded warning: it does not change a cell, its acceptance or the exit status of the run.

| Case | Backend | Warning |
| --- | --- | --- |
| softplus_f64_n1048576 | tenferro-cuda-direct | softplus_f64_n1048576 untimed 151.7 ms is 810% of the 18.7 ms timed region (budget 20%) |
| softplus_f64_n1048576 | tenferro-cuda-eager | softplus_f64_n1048576 untimed 88.8 ms is 302% of the 29.4 ms timed region (budget 20%) |
| sigmoid_f64_n1048576 | pytorch-cuda | sigmoid_f64_n1048576 untimed 259.5 ms is 5389% of the 4.8 ms timed region (budget 20%) |
| silu_f64_n1048576 | pytorch-cuda | silu_f64_n1048576 untimed 141.0 ms is 2626% of the 5.4 ms timed region (budget 20%) |
| softplus_f64_n1048576 | pytorch-cuda | softplus_f64_n1048576 untimed 139.3 ms is 1124% of the 12.4 ms timed region (budget 20%) |
| gelu_f64_n1048576 | pytorch-cuda | gelu_f64_n1048576 untimed 143.9 ms is 1990% of the 7.2 ms timed region (budget 20%) |
| gelu_tanh_f64_n1048576 | pytorch-cuda | gelu_tanh_f64_n1048576 untimed 141.0 ms is 3109% of the 4.5 ms timed region (budget 20%) |
| softmax_f64_n1048576 | pytorch-cuda | softmax_f64_n1048576 untimed 155.8 ms is 1960% of the 8.0 ms timed region (budget 20%) |
| log_softmax_f64_n1048576 | pytorch-cuda | log_softmax_f64_n1048576 untimed 138.9 ms is 2157% of the 6.4 ms timed region (budget 20%) |
| layer_norm_f64_n1048576 | pytorch-cuda | layer_norm_f64_n1048576 untimed 159.8 ms is 1093% of the 14.6 ms timed region (budget 20%) |
| rms_norm_f64_n1048576 | pytorch-cuda | rms_norm_f64_n1048576 untimed 181.2 ms is 1228% of the 14.8 ms timed region (budget 20%) |
| sigmoid_f64_n1048576 | jax-cuda | sigmoid_f64_n1048576 untimed 414.4 ms is 1320% of the 31.4 ms timed region (budget 20%) |
| silu_f64_n1048576 | jax-cuda | silu_f64_n1048576 untimed 215.6 ms is 760% of the 28.4 ms timed region (budget 20%) |
| softplus_f64_n1048576 | jax-cuda | softplus_f64_n1048576 untimed 252.5 ms is 804% of the 31.4 ms timed region (budget 20%) |
| gelu_f64_n1048576 | jax-cuda | gelu_f64_n1048576 untimed 515.3 ms is 1275% of the 40.4 ms timed region (budget 20%) |
| gelu_tanh_f64_n1048576 | jax-cuda | gelu_tanh_f64_n1048576 untimed 393.7 ms is 712% of the 55.3 ms timed region (budget 20%) |
| softmax_f64_n1048576 | jax-cuda | softmax_f64_n1048576 untimed 756.6 ms is 1423% of the 53.2 ms timed region (budget 20%) |
| log_softmax_f64_n1048576 | jax-cuda | log_softmax_f64_n1048576 untimed 276.5 ms is 882% of the 31.4 ms timed region (budget 20%) |
| masked_softmax_f64_n1048576 | jax-cuda | masked_softmax_f64_n1048576 untimed 623.6 ms is 1448% of the 43.1 ms timed region (budget 20%) |
| masked_log_softmax_f64_n1048576 | jax-cuda | masked_log_softmax_f64_n1048576 untimed 294.5 ms is 1010% of the 29.2 ms timed region (budget 20%) |
| layer_norm_f64_n1048576 | jax-cuda | layer_norm_f64_n1048576 untimed 239.1 ms is 442% of the 54.0 ms timed region (budget 20%) |