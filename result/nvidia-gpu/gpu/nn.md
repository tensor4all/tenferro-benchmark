# gpu/nn: CUDA public API benchmark

Measured cells: **41**. Unsupported cells: **3**, including **0** numerically verified but timing-excluded cells. Unsupported cells do not count as measured coverage.

Logical traffic: this run measured a plain device copy of 67108864 bytes at **1449.3 GB/s** on this device. Every record carries the logical bytes the case must move per operation and the ratio of the case's implied rate to that copy rate, in the sampling table below. Logical bytes are the declared array sizes, not device DRAM traffic: a case whose operands stay in cache moves less DRAM than its arrays suggest, so a ratio above 1 is recorded, not rejected.

Raw run: `data/results/nvidia-gpu/gpu/nn/20261126_062950/`.

Isolated public APIs and explicit unsupported timing contracts. Issue 155.

This run compiled **1 NVRTC modules** in 1 of its arms, because the CUDA artifact cache started cold. The untimed wall of the cases that compiled is **0.1 s** of its 9.1 s of recorded case wall clock, and that wall also holds their fixture read, transfers and verification, so it is not a compile time.

Values are median (IQR) **µs per operation**. Samples time batches of declared operations plus device synchronization; retained output memory is limited to 256 MiB. Batch counts and raw durations are preserved in JSONL.

Cases marked † (**6** of 11) were sampled with a different number of operations per sample on different arms (see Ops/sample in the sampling table). Each value is still per operation, but the arms amortise launch and synchronization over different batch sizes, so a ratio between their columns is approximate and is not presented as a like-for-like comparison.

Setup, input transfers, library/plan initialization, JIT compilation, validation downloads and output destruction are outside timing. Intrinsic output allocation is inside timing; output-reuse rows instead use preallocated destinations across samples. Accumulating destinations are restored from prepared GPU seeds and synchronized before every batch clock, including initialization and calibration. Direct and eager each use one entered session across samples; direct FFT uses cached `FftExecutor`. JAX eager and compiled JIT are separate arms.

All measured output elements are checked against independently generated NumPy/SciPy reference results before and after sampling. Unsupported rows include unavailable APIs and APIs whose internal setup cannot be excluded; these are not performance measurements. Short batches limited by output memory remain identifiable in the sampling table.

| Case | tenferro-cuda-direct | tenferro-cuda-eager | pytorch-cuda | jax-cuda |
| --- | ---: | ---: | ---: | ---: |
| sigmoid_f64_n1048576 | 90.400 (0.108) | 90.555 (0.070) | 9.975 (0.056) | 64.373 (1.923) |
| silu_f64_n1048576 | 100.142 (0.174) | 99.667 (0.287) | 11.151 (0.052) | 56.215 (1.268) |
| softplus_f64_n1048576 | 28.231 (0.125) | 28.175 (0.053) | 25.655 (0.032) | 54.749 (2.499) |
| gelu_f64_n1048576 † | 73.687 (0.091) | 73.884 (0.117) | 14.989 (0.057) | 636.637 (70.330) |
| gelu_tanh_f64_n1048576 † | 113.322 (0.097) | 112.927 (0.369) | 9.415 (0.040) | 923.090 (109.065) |
| softmax_f64_n1048576 † | 90.152 (0.091) | 90.755 (0.861) | 16.461 (0.021) | 521.979 (18.565) |
| log_softmax_f64_n1048576 | 92.313 (0.201) | 92.646 (0.094) | 12.877 (0.018) | 60.894 (2.656) |
| masked_softmax_f64_n1048576 † | 115.738 (0.122) | 116.138 (0.797) | unsupported | 865.929 (149.195) |
| masked_log_softmax_f64_n1048576 † | 113.806 (0.208) | 114.688 (0.356) | unsupported | 62.228 (1.544) |
| layer_norm_f64_n1048576 † | 87.567 (0.108) | 87.883 (3.653) | 30.623 (0.124) | 148.960 (16.101) |
| rms_norm_f64_n1048576 | 48.657 (0.112) | 82.424 (33.140) | 30.416 (0.149) | unsupported |

## Sampling

| Case | Backend | Ops/sample | Samples | Median batch ms | Bytes/op | Copy ratio | Verification error/tolerance |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gelu_f64_n1048576 | jax-cuda | 4 | 15 | 2.547 | 16777216 | 0.018 | 4.81e-08 |
| gelu_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.480 | 16777216 | 0.772 | 4.899e-08 |
| gelu_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.358 | 16777216 | 0.157 | 4.899e-08 |
| gelu_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.364 | 16777216 | 0.157 | 4.899e-08 |
| gelu_tanh_f64_n1048576 | jax-cuda | 4 | 15 | 3.692 | 16777216 | 0.013 | 4.349e-08 |
| gelu_tanh_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.301 | 16777216 | 1.230 | 4.349e-08 |
| gelu_tanh_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.626 | 16777216 | 0.102 | 4.349e-08 |
| gelu_tanh_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.614 | 16777216 | 0.103 | 4.349e-08 |
| layer_norm_f64_n1048576 | jax-cuda | 16 | 15 | 2.383 | 16777216 | 0.078 | 3.398e-07 |
| layer_norm_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.980 | 16777216 | 0.378 | 3.138e-07 |
| layer_norm_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.802 | 16777216 | 0.132 | 3.251e-07 |
| layer_norm_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.812 | 16777216 | 0.132 | 3.251e-07 |
| log_softmax_f64_n1048576 | jax-cuda | 32 | 15 | 1.949 | 16777216 | 0.190 | 2.377e-07 |
| log_softmax_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.412 | 16777216 | 0.899 | 2.377e-07 |
| log_softmax_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.954 | 16777216 | 0.125 | 2.377e-07 |
| log_softmax_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.965 | 16777216 | 0.125 | 2.377e-07 |
| masked_log_softmax_f64_n1048576 | jax-cuda | 32 | 15 | 1.991 | 17825792 | 0.198 | 2.489e-07 |
| masked_log_softmax_f64_n1048576 | pytorch-cuda | — | — | — | 17825792 | unavailable | No equivalent stable single PyTorch public masked-softmax operation; masked_fill plus softmax would change the operation contract. |
| masked_log_softmax_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.642 | 17825792 | 0.108 | 2.489e-07 |
| masked_log_softmax_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 1.835 | 17825792 | 0.107 | 2.489e-07 |
| masked_softmax_f64_n1048576 | jax-cuda | 4 | 15 | 3.464 | 17825792 | 0.014 | 8.659e-10 |
| masked_softmax_f64_n1048576 | pytorch-cuda | — | — | — | 17825792 | unavailable | No equivalent stable single PyTorch public masked-softmax operation; masked_fill plus softmax would change the operation contract. |
| masked_softmax_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.704 | 17825792 | 0.106 | 1.299e-09 |
| masked_softmax_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.716 | 17825792 | 0.106 | 1.299e-09 |
| rms_norm_f64_n1048576 | jax-cuda | — | — | — | 16777216 | unavailable | No equivalent single public JAX RMS-normalization operation; a composition is not timed as the missing API. |
| rms_norm_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.973 | 16777216 | 0.381 | 2.654e-07 |
| rms_norm_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.557 | 16777216 | 0.238 | 2.71e-07 |
| rms_norm_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.638 | 16777216 | 0.140 | 2.71e-07 |
| sigmoid_f64_n1048576 | jax-cuda | 32 | 15 | 2.060 | 16777216 | 0.180 | 7.964e-08 |
| sigmoid_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.319 | 16777216 | 1.161 | 7.964e-08 |
| sigmoid_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.893 | 16777216 | 0.128 | 8.044e-08 |
| sigmoid_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.898 | 16777216 | 0.128 | 8.044e-08 |
| silu_f64_n1048576 | jax-cuda | 32 | 15 | 1.799 | 16777216 | 0.206 | 4.848e-08 |
| silu_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.357 | 16777216 | 1.038 | 4.848e-08 |
| silu_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.205 | 16777216 | 0.116 | 4.912e-08 |
| silu_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.189 | 16777216 | 0.116 | 4.912e-08 |
| softmax_f64_n1048576 | jax-cuda | 4 | 15 | 2.088 | 16777216 | 0.022 | 6.499e-10 |
| softmax_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.527 | 16777216 | 0.703 | 8.661e-10 |
| softmax_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.885 | 16777216 | 0.128 | 1.083e-09 |
| softmax_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.904 | 16777216 | 0.128 | 1.083e-09 |
| softplus_f64_n1048576 | jax-cuda | 32 | 15 | 1.752 | 16777216 | 0.211 | 1.467e-07 |
| softplus_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.821 | 16777216 | 0.451 | 1.467e-07 |
| softplus_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.903 | 16777216 | 0.410 | 1.467e-07 |
| softplus_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.902 | 16777216 | 0.411 | 1.467e-07 |

## Run metadata

```yaml
schema_version: 1
target_profile: nvidia-gpu
suite_id: gpu/nn
suite_file: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/data/results/nvidia-gpu/gpu/routine_selection/20261126_062950/gpu_nn.yaml
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
  - 0.09283
  - 0.09247
  - 0.09347
  - 0.09311
  - 0.09228
  - 0.092311
  - 0.09157
  - 0.09387
  - 0.09332
  - 0.0923
  - 0.092851
  - 0.09261
  - 0.09229
  - 0.09294
  - 0.09177
  median_ms: 0.09261
  bytes_per_second: 1449278997948.3857
  torch_version: 2.11.0+cu128
  cuda_version: '12.8'
  timestamp: '2026-11-26T06:42:51.237421+00:00'
gpu_assignment:
  role: benchmark
  physical_device_index: 0
  device_uuid: GPU-e0c82f3d-3a11-a28c-b42e-1c92d26527fd
  cuda_visible_devices: GPU-e0c82f3d-3a11-a28c-b42e-1c92d26527fd
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
    data/results/nvidia-gpu/gpu/nn/20261126_062950/fixtures/manifest.json: 06ffcecd4fd9ca2058834e8c222d97a2dc8d86d5506100cc2b091e779e957bbf
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
        finished_at: '2026-11-26T06:43:07Z'
      pytorch-cuda:
        arm: pytorch-cuda
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-26T06:43:01Z'
      tenferro-cuda-direct:
        arm: tenferro-cuda-direct
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-26T06:42:55Z'
      tenferro-cuda-eager:
        arm: tenferro-cuda-eager
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-26T06:42:57Z'
  compile:
    case_wall_seconds: 9.121
    compiling_processes: 1
    modules_compiled: 1
    compile_time_ms: null
    compile_time_scope: 'not measured: no compile interval was recorded; `compile_modules`
      counts the NVRTC modules and the untimed wall is the case wall'
    untimed_wall_seconds: 0.149
```

## Untimed work over the recorded budget

Each row records the wall clock it spent outside its timed region. These cases spent more than the recorded budget ratio (0.2) there. The threshold has no measured basis, so this is a recorded warning: it does not change a cell, its acceptance or the exit status of the run.

| Case | Backend | Warning |
| --- | --- | --- |
| sigmoid_f64_n1048576 | tenferro-cuda-direct | sigmoid_f64_n1048576 untimed 164.2 ms is 377% of the 43.6 ms timed region (budget 20%) |
| silu_f64_n1048576 | tenferro-cuda-direct | silu_f64_n1048576 untimed 93.7 ms is 194% of the 48.3 ms timed region (budget 20%) |
| softplus_f64_n1048576 | tenferro-cuda-direct | softplus_f64_n1048576 untimed 149.0 ms is 1099% of the 13.6 ms timed region (budget 20%) |
| gelu_f64_n1048576 | tenferro-cuda-direct | gelu_f64_n1048576 untimed 94.2 ms is 266% of the 35.4 ms timed region (budget 20%) |
| gelu_tanh_f64_n1048576 | tenferro-cuda-direct | gelu_tanh_f64_n1048576 untimed 87.6 ms is 161% of the 54.4 ms timed region (budget 20%) |
| softmax_f64_n1048576 | tenferro-cuda-direct | softmax_f64_n1048576 untimed 115.3 ms is 265% of the 43.5 ms timed region (budget 20%) |
| log_softmax_f64_n1048576 | tenferro-cuda-direct | log_softmax_f64_n1048576 untimed 83.7 ms is 188% of the 44.5 ms timed region (budget 20%) |
| masked_softmax_f64_n1048576 | tenferro-cuda-direct | masked_softmax_f64_n1048576 untimed 96.2 ms is 172% of the 55.8 ms timed region (budget 20%) |
| masked_log_softmax_f64_n1048576 | tenferro-cuda-direct | masked_log_softmax_f64_n1048576 untimed 68.3 ms is 125% of the 54.8 ms timed region (budget 20%) |
| layer_norm_f64_n1048576 | tenferro-cuda-direct | layer_norm_f64_n1048576 untimed 98.6 ms is 234% of the 42.2 ms timed region (budget 20%) |
| rms_norm_f64_n1048576 | tenferro-cuda-direct | rms_norm_f64_n1048576 untimed 96.3 ms is 409% of the 23.5 ms timed region (budget 20%) |
| sigmoid_f64_n1048576 | tenferro-cuda-eager | sigmoid_f64_n1048576 untimed 170.5 ms is 391% of the 43.7 ms timed region (budget 20%) |
| silu_f64_n1048576 | tenferro-cuda-eager | silu_f64_n1048576 untimed 98.2 ms is 204% of the 48.0 ms timed region (budget 20%) |
| softplus_f64_n1048576 | tenferro-cuda-eager | softplus_f64_n1048576 untimed 87.9 ms is 649% of the 13.5 ms timed region (budget 20%) |
| gelu_f64_n1048576 | tenferro-cuda-eager | gelu_f64_n1048576 untimed 96.9 ms is 272% of the 35.7 ms timed region (budget 20%) |
| gelu_tanh_f64_n1048576 | tenferro-cuda-eager | gelu_tanh_f64_n1048576 untimed 85.2 ms is 155% of the 54.8 ms timed region (budget 20%) |
| softmax_f64_n1048576 | tenferro-cuda-eager | softmax_f64_n1048576 untimed 115.0 ms is 252% of the 45.6 ms timed region (budget 20%) |
| log_softmax_f64_n1048576 | tenferro-cuda-eager | log_softmax_f64_n1048576 untimed 90.8 ms is 204% of the 44.5 ms timed region (budget 20%) |
| masked_softmax_f64_n1048576 | tenferro-cuda-eager | masked_softmax_f64_n1048576 untimed 99.3 ms is 173% of the 57.3 ms timed region (budget 20%) |
| masked_log_softmax_f64_n1048576 | tenferro-cuda-eager | masked_log_softmax_f64_n1048576 untimed 67.3 ms is 241% of the 27.9 ms timed region (budget 20%) |
| layer_norm_f64_n1048576 | tenferro-cuda-eager | layer_norm_f64_n1048576 untimed 106.8 ms is 241% of the 44.4 ms timed region (budget 20%) |
| rms_norm_f64_n1048576 | tenferro-cuda-eager | rms_norm_f64_n1048576 untimed 101.3 ms is 304% of the 33.3 ms timed region (budget 20%) |
| sigmoid_f64_n1048576 | pytorch-cuda | sigmoid_f64_n1048576 untimed 264.5 ms is 5521% of the 4.8 ms timed region (budget 20%) |
| silu_f64_n1048576 | pytorch-cuda | silu_f64_n1048576 untimed 141.3 ms is 2640% of the 5.4 ms timed region (budget 20%) |
| softplus_f64_n1048576 | pytorch-cuda | softplus_f64_n1048576 untimed 135.5 ms is 1100% of the 12.3 ms timed region (budget 20%) |
| gelu_f64_n1048576 | pytorch-cuda | gelu_f64_n1048576 untimed 135.4 ms is 1881% of the 7.2 ms timed region (budget 20%) |
| gelu_tanh_f64_n1048576 | pytorch-cuda | gelu_tanh_f64_n1048576 untimed 130.3 ms is 2875% of the 4.5 ms timed region (budget 20%) |
| softmax_f64_n1048576 | pytorch-cuda | softmax_f64_n1048576 untimed 148.1 ms is 1874% of the 7.9 ms timed region (budget 20%) |
| log_softmax_f64_n1048576 | pytorch-cuda | log_softmax_f64_n1048576 untimed 135.7 ms is 2190% of the 6.2 ms timed region (budget 20%) |
| layer_norm_f64_n1048576 | pytorch-cuda | layer_norm_f64_n1048576 untimed 142.9 ms is 970% of the 14.7 ms timed region (budget 20%) |
| rms_norm_f64_n1048576 | pytorch-cuda | rms_norm_f64_n1048576 untimed 138.4 ms is 946% of the 14.6 ms timed region (budget 20%) |
| sigmoid_f64_n1048576 | jax-cuda | sigmoid_f64_n1048576 untimed 418.2 ms is 1346% of the 31.1 ms timed region (budget 20%) |
| silu_f64_n1048576 | jax-cuda | silu_f64_n1048576 untimed 224.4 ms is 833% of the 26.9 ms timed region (budget 20%) |
| softplus_f64_n1048576 | jax-cuda | softplus_f64_n1048576 untimed 254.4 ms is 957% of the 26.6 ms timed region (budget 20%) |
| gelu_f64_n1048576 | jax-cuda | gelu_f64_n1048576 untimed 520.5 ms is 1354% of the 38.4 ms timed region (budget 20%) |
| gelu_tanh_f64_n1048576 | jax-cuda | gelu_tanh_f64_n1048576 untimed 387.2 ms is 657% of the 59.0 ms timed region (budget 20%) |
| softmax_f64_n1048576 | jax-cuda | softmax_f64_n1048576 untimed 780.2 ms is 2453% of the 31.8 ms timed region (budget 20%) |
| log_softmax_f64_n1048576 | jax-cuda | log_softmax_f64_n1048576 untimed 301.4 ms is 1029% of the 29.3 ms timed region (budget 20%) |
| masked_softmax_f64_n1048576 | jax-cuda | masked_softmax_f64_n1048576 untimed 643.1 ms is 1258% of the 51.1 ms timed region (budget 20%) |
| masked_log_softmax_f64_n1048576 | jax-cuda | masked_log_softmax_f64_n1048576 untimed 335.6 ms is 1116% of the 30.1 ms timed region (budget 20%) |
| layer_norm_f64_n1048576 | jax-cuda | layer_norm_f64_n1048576 untimed 269.7 ms is 717% of the 37.6 ms timed region (budget 20%) |