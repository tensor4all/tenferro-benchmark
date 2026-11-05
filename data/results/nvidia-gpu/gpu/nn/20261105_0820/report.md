# gpu/nn: CUDA public API benchmark

Measured cells: **21**. Unsupported cells: **23**, including **20** numerically verified but timing-excluded cells. Unsupported cells do not count as measured coverage.

Logical traffic: this run measured a plain device copy of 67108864 bytes at **1410.8 GB/s** on this device. Every record carries the logical bytes the case must move per operation and the ratio of the case's implied rate to that copy rate, in the sampling table below. Logical bytes are the declared array sizes, not device DRAM traffic: a case whose operands stay in cache moves less DRAM than its arrays suggest, so a ratio above 1 is recorded, not rejected.

Raw run: `data/results/nvidia-gpu/gpu/nn/20261105_0820/`.

Isolated public APIs and explicit unsupported timing contracts. Issue 155.

Values are median (IQR) **µs per operation**. Samples time batches of declared operations plus device synchronization; retained output memory is limited to 256 MiB. Batch counts and raw durations are preserved in JSONL.

Setup, input transfers, library/plan initialization, JIT compilation, validation downloads and output destruction are outside timing. Intrinsic output allocation is inside timing; output-reuse rows instead use preallocated destinations across samples. Accumulating destinations are restored from prepared GPU seeds and synchronized before every batch clock, including initialization and calibration. Direct and eager each use one entered session across samples; direct FFT uses cached `FftExecutor`. JAX eager and compiled JIT are separate arms.

All measured output elements are checked against independently generated NumPy/SciPy reference results before and after sampling. Unsupported rows include unavailable APIs and APIs whose internal setup cannot be excluded; these are not performance measurements. Short batches limited by output memory remain identifiable in the sampling table.

| Case | tenferro-cuda-direct | tenferro-cuda-eager | pytorch-cuda | jax-cuda |
| --- | ---: | ---: | ---: | ---: |
| sigmoid_f64_n1048576 | unsupported | unsupported | 10.050 (0.073) | 41.868 (1.248) |
| silu_f64_n1048576 | unsupported | unsupported | 10.963 (0.038) | 42.038 (1.042) |
| softplus_f64_n1048576 | 200.081 (0.948) | 324.001 (1.074) | 25.635 (0.054) | 37.510 (0.436) |
| gelu_f64_n1048576 | unsupported | unsupported | 15.086 (0.045) | 510.764 (5.578) |
| gelu_tanh_f64_n1048576 | unsupported | unsupported | 9.378 (0.056) | 743.055 (16.812) |
| softmax_f64_n1048576 | unsupported | unsupported | 16.464 (0.035) | 357.736 (3.849) |
| log_softmax_f64_n1048576 | unsupported | unsupported | 13.250 (0.031) | 39.770 (0.307) |
| masked_softmax_f64_n1048576 | unsupported | unsupported | unsupported | 548.935 (8.784) |
| masked_log_softmax_f64_n1048576 | unsupported | unsupported | unsupported | 40.012 (1.132) |
| layer_norm_f64_n1048576 | unsupported | unsupported | 30.741 (0.063) | 94.790 (0.709) |
| rms_norm_f64_n1048576 | unsupported | unsupported | 30.664 (0.053) | unsupported |

## Sampling

| Case | Backend | Ops/sample | Samples | Median batch ms | Bytes/op | Copy ratio | Verification error/tolerance |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gelu_f64_n1048576 | jax-cuda | 4 | 15 | 2.043 | 16777216 | 0.023 | 4.81e-08 |
| gelu_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.483 | 16777216 | 0.788 | 4.899e-08 |
| gelu_f64_n1048576 | tenferro-cuda-direct | — | — | — | 16777216 | unavailable | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| gelu_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| gelu_tanh_f64_n1048576 | jax-cuda | 4 | 15 | 2.972 | 16777216 | 0.016 | 4.349e-08 |
| gelu_tanh_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.300 | 16777216 | 1.268 | 4.349e-08 |
| gelu_tanh_f64_n1048576 | tenferro-cuda-direct | — | — | — | 16777216 | unavailable | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| gelu_tanh_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| layer_norm_f64_n1048576 | jax-cuda | 32 | 15 | 3.033 | 16777216 | 0.125 | 3.398e-07 |
| layer_norm_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.984 | 16777216 | 0.387 | 3.138e-07 |
| layer_norm_f64_n1048576 | tenferro-cuda-direct | — | — | — | 16777216 | unavailable | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| layer_norm_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| log_softmax_f64_n1048576 | jax-cuda | 32 | 15 | 1.273 | 16777216 | 0.299 | 2.377e-07 |
| log_softmax_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.424 | 16777216 | 0.897 | 2.377e-07 |
| log_softmax_f64_n1048576 | tenferro-cuda-direct | — | — | — | 16777216 | unavailable | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| log_softmax_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| masked_log_softmax_f64_n1048576 | jax-cuda | 32 | 15 | 1.280 | 17825792 | 0.316 | 2.489e-07 |
| masked_log_softmax_f64_n1048576 | pytorch-cuda | — | — | — | 17825792 | unavailable | No equivalent stable single PyTorch public masked-softmax operation; masked_fill plus softmax would change the operation contract. |
| masked_log_softmax_f64_n1048576 | tenferro-cuda-direct | — | — | — | 17825792 | unavailable | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| masked_log_softmax_f64_n1048576 | tenferro-cuda-eager | — | — | — | 17825792 | unavailable | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| masked_softmax_f64_n1048576 | jax-cuda | 4 | 15 | 2.196 | 17825792 | 0.023 | 8.659e-10 |
| masked_softmax_f64_n1048576 | pytorch-cuda | — | — | — | 17825792 | unavailable | No equivalent stable single PyTorch public masked-softmax operation; masked_fill plus softmax would change the operation contract. |
| masked_softmax_f64_n1048576 | tenferro-cuda-direct | — | — | — | 17825792 | unavailable | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| masked_softmax_f64_n1048576 | tenferro-cuda-eager | — | — | — | 17825792 | unavailable | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| rms_norm_f64_n1048576 | jax-cuda | — | — | — | 16777216 | unavailable | No equivalent single public JAX RMS-normalization operation; a composition is not timed as the missing API. |
| rms_norm_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.981 | 16777216 | 0.388 | 2.654e-07 |
| rms_norm_f64_n1048576 | tenferro-cuda-direct | — | — | — | 16777216 | unavailable | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| rms_norm_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| sigmoid_f64_n1048576 | jax-cuda | 32 | 15 | 1.340 | 16777216 | 0.284 | 7.964e-08 |
| sigmoid_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.322 | 16777216 | 1.183 | 7.964e-08 |
| sigmoid_f64_n1048576 | tenferro-cuda-direct | — | — | — | 16777216 | unavailable | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| sigmoid_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| silu_f64_n1048576 | jax-cuda | 32 | 15 | 1.345 | 16777216 | 0.283 | 4.848e-08 |
| silu_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.351 | 16777216 | 1.085 | 4.848e-08 |
| silu_f64_n1048576 | tenferro-cuda-direct | — | — | — | 16777216 | unavailable | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| silu_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| softmax_f64_n1048576 | jax-cuda | 8 | 15 | 2.862 | 16777216 | 0.033 | 6.499e-10 |
| softmax_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.527 | 16777216 | 0.722 | 8.661e-10 |
| softmax_f64_n1048576 | tenferro-cuda-direct | — | — | — | 16777216 | unavailable | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| softmax_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| softplus_f64_n1048576 | jax-cuda | 32 | 15 | 1.200 | 16777216 | 0.317 | 1.467e-07 |
| softplus_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.820 | 16777216 | 0.464 | 1.467e-07 |
| softplus_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 3.201 | 16777216 | 0.059 | 1.467e-07 |
| softplus_f64_n1048576 | tenferro-cuda-eager | 8 | 15 | 2.592 | 16777216 | 0.037 | 1.467e-07 |

## Run metadata

```yaml
schema_version: 1
target_profile: nvidia-gpu
suite_id: gpu/nn
suite_file: /home/shinaoka/work/wallclock/data/results/nvidia-gpu/gpu/routine_selection/20261105_0820/gpu_nn.yaml
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
  - 0.093193
  - 0.095133
  - 0.098864
  - 0.094934
  - 0.092193
  - 0.096034
  - 0.095943
  - 0.092823
  - 0.097984
  - 0.094553
  - 0.093753
  - 0.097384
  - 0.096283
  - 0.093914
  - 0.095534
  median_ms: 0.095133
  bytes_per_second: 1410843009260.7192
  torch_version: 2.12.0+cu130
  cuda_version: '13.0'
  timestamp: '2026-11-05T07:52:16.001883+00:00'
gpu_assignment:
  role: benchmark
  physical_device_index: 0
  device_uuid: GPU-530977e1-4968-9283-4129-9fbec3e66542
  cuda_visible_devices: GPU-530977e1-4968-9283-4129-9fbec3e66542
  process_device_ordinal: 0
  verification_only: false
collection:
  command: BENCHMARK_TARGET_PROFILE=nvidia-gpu GPU_BENCH_DEVICE=0 BENCH_RUNS=15 BENCH_WARMUPS=3
    ./scripts/run_gpu_public_api.sh nn
  devcontainer_config: .devcontainer/cuda/devcontainer.json
  expected_cases: 11
  selected_cases: 11
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
    data/results/nvidia-gpu/gpu/nn/20261105_0820/fixtures/manifest.json: 6019f8eb30ec023ad89d15545a6d050a2188b62ec40b3d90c773ad6e58eabea1
    target/release/benchmark_gpu_public_api: efd5ab70e2c4de27c458d808d308b8bf3415012551220c291451a2591d08880f
  reference_numpy_version: 2.4.2
```

## Untimed work over the recorded budget

Each row records the wall clock it spent outside its timed region. These cases spent more than the recorded budget ratio (0.2) there. The threshold has no measured basis, so this is a recorded warning: it does not change a cell, its acceptance or the exit status of the run.

| Case | Backend | Warning |
| --- | --- | --- |
| softplus_f64_n1048576 | tenferro-cuda-direct | softplus_f64_n1048576 untimed 46.8 ms is 97% of the 48.0 ms timed region (budget 20%) |
| softplus_f64_n1048576 | tenferro-cuda-eager | softplus_f64_n1048576 untimed 51.5 ms is 132% of the 39.0 ms timed region (budget 20%) |
| sigmoid_f64_n1048576 | pytorch-cuda | sigmoid_f64_n1048576 untimed 1369.4 ms is 28401% of the 4.8 ms timed region (budget 20%) |
| silu_f64_n1048576 | pytorch-cuda | silu_f64_n1048576 untimed 1118.0 ms is 21235% of the 5.3 ms timed region (budget 20%) |
| softplus_f64_n1048576 | pytorch-cuda | softplus_f64_n1048576 untimed 1161.1 ms is 9436% of the 12.3 ms timed region (budget 20%) |
| gelu_f64_n1048576 | pytorch-cuda | gelu_f64_n1048576 untimed 1296.1 ms is 17876% of the 7.3 ms timed region (budget 20%) |
| gelu_tanh_f64_n1048576 | pytorch-cuda | gelu_tanh_f64_n1048576 untimed 1295.1 ms is 28732% of the 4.5 ms timed region (budget 20%) |
| softmax_f64_n1048576 | pytorch-cuda | softmax_f64_n1048576 untimed 1379.6 ms is 17454% of the 7.9 ms timed region (budget 20%) |
| log_softmax_f64_n1048576 | pytorch-cuda | log_softmax_f64_n1048576 untimed 1374.1 ms is 21610% of the 6.4 ms timed region (budget 20%) |
| layer_norm_f64_n1048576 | pytorch-cuda | layer_norm_f64_n1048576 untimed 1424.3 ms is 9653% of the 14.8 ms timed region (budget 20%) |
| rms_norm_f64_n1048576 | pytorch-cuda | rms_norm_f64_n1048576 untimed 1337.0 ms is 9089% of the 14.7 ms timed region (budget 20%) |
| sigmoid_f64_n1048576 | jax-cuda | sigmoid_f64_n1048576 untimed 1545.7 ms is 7740% of the 20.0 ms timed region (budget 20%) |
| silu_f64_n1048576 | jax-cuda | silu_f64_n1048576 untimed 1297.9 ms is 6493% of the 20.0 ms timed region (budget 20%) |
| softplus_f64_n1048576 | jax-cuda | softplus_f64_n1048576 untimed 1362.3 ms is 7545% of the 18.1 ms timed region (budget 20%) |
| gelu_f64_n1048576 | jax-cuda | gelu_f64_n1048576 untimed 1653.2 ms is 5400% of the 30.6 ms timed region (budget 20%) |
| gelu_tanh_f64_n1048576 | jax-cuda | gelu_tanh_f64_n1048576 untimed 1613.2 ms is 3647% of the 44.2 ms timed region (budget 20%) |
| softmax_f64_n1048576 | jax-cuda | softmax_f64_n1048576 untimed 2193.1 ms is 5107% of the 42.9 ms timed region (budget 20%) |
| log_softmax_f64_n1048576 | jax-cuda | log_softmax_f64_n1048576 untimed 1439.9 ms is 7556% of the 19.1 ms timed region (budget 20%) |
| masked_softmax_f64_n1048576 | jax-cuda | masked_softmax_f64_n1048576 untimed 1883.0 ms is 5700% of the 33.0 ms timed region (budget 20%) |
| masked_log_softmax_f64_n1048576 | jax-cuda | masked_log_softmax_f64_n1048576 untimed 338.2 ms is 1762% of the 19.2 ms timed region (budget 20%) |
| layer_norm_f64_n1048576 | jax-cuda | layer_norm_f64_n1048576 untimed 1382.9 ms is 3040% of the 45.5 ms timed region (budget 20%) |