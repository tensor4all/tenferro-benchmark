# gpu/nn: CUDA public API benchmark

Measured cells: **21**. Unsupported cells: **23**, including **20** numerically verified but timing-excluded cells. Unsupported cells do not count as measured coverage.

Logical traffic: this run measured a plain device copy of 67108864 bytes at **1438.8 GB/s** on this device. Every record carries the logical bytes the case must move per operation and the ratio of the case's implied rate to that copy rate, in the sampling table below. Logical bytes are the declared array sizes, not device DRAM traffic: a case whose operands stay in cache moves less DRAM than its arrays suggest, so a ratio above 1 is recorded, not rejected.

Raw run: `data/results/nvidia-gpu/gpu/nn/20261108_115237/`.

Isolated public APIs and explicit unsupported timing contracts. Issue 155.

This run did not compile: no NVRTC module was built in any of its arms. That describes this run only, and does not show that the CUDA artifact cache was warm.

Values are median (IQR) **µs per operation**. Samples time batches of declared operations plus device synchronization; retained output memory is limited to 256 MiB. Batch counts and raw durations are preserved in JSONL.

Cases marked † (**5** of 11) were sampled with a different number of operations per sample on different arms (see Ops/sample in the sampling table). Each value is still per operation, but the arms amortise launch and synchronization over different batch sizes, so a ratio between their columns is approximate and is not presented as a like-for-like comparison.

Setup, input transfers, library/plan initialization, JIT compilation, validation downloads and output destruction are outside timing. Intrinsic output allocation is inside timing; output-reuse rows instead use preallocated destinations across samples. Accumulating destinations are restored from prepared GPU seeds and synchronized before every batch clock, including initialization and calibration. Direct and eager each use one entered session across samples; direct FFT uses cached `FftExecutor`. JAX eager and compiled JIT are separate arms.

All measured output elements are checked against independently generated NumPy/SciPy reference results before and after sampling. Unsupported rows include unavailable APIs and APIs whose internal setup cannot be excluded; these are not performance measurements. Short batches limited by output memory remain identifiable in the sampling table.

| Case | tenferro-cuda-direct | tenferro-cuda-eager | pytorch-cuda | jax-cuda |
| --- | ---: | ---: | ---: | ---: |
| sigmoid_f64_n1048576 | unsupported | unsupported | 10.243 (0.087) | 81.742 (3.243) |
| silu_f64_n1048576 | unsupported | unsupported | 11.409 (0.061) | 68.351 (1.550) |
| softplus_f64_n1048576 † | 206.728 (2.740) | 343.737 (8.788) | 25.997 (0.086) | 78.621 (10.145) |
| gelu_f64_n1048576 † | unsupported | unsupported | 15.250 (0.106) | 816.243 (63.426) |
| gelu_tanh_f64_n1048576 † | unsupported | unsupported | 9.669 (0.057) | 1233.566 (74.748) |
| softmax_f64_n1048576 † | unsupported | unsupported | 16.751 (0.098) | 523.392 (11.879) |
| log_softmax_f64_n1048576 | unsupported | unsupported | 13.554 (0.068) | 76.612 (10.165) |
| masked_softmax_f64_n1048576 | unsupported | unsupported | unsupported | 1006.348 (132.714) |
| masked_log_softmax_f64_n1048576 | unsupported | unsupported | unsupported | 67.869 (0.678) |
| layer_norm_f64_n1048576 † | unsupported | unsupported | 30.851 (0.078) | 206.058 (5.188) |
| rms_norm_f64_n1048576 | unsupported | unsupported | 30.723 (0.087) | unsupported |

## Sampling

| Case | Backend | Ops/sample | Samples | Median batch ms | Bytes/op | Copy ratio | Verification error/tolerance |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gelu_f64_n1048576 | jax-cuda | 4 | 15 | 3.265 | 16777216 | 0.014 | 4.81e-08 |
| gelu_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.488 | 16777216 | 0.765 | 4.899e-08 |
| gelu_f64_n1048576 | tenferro-cuda-direct | — | — | — | 16777216 | unavailable | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| gelu_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| gelu_tanh_f64_n1048576 | jax-cuda | 2 | 15 | 2.467 | 16777216 | 0.009 | 4.349e-08 |
| gelu_tanh_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.309 | 16777216 | 1.206 | 4.349e-08 |
| gelu_tanh_f64_n1048576 | tenferro-cuda-direct | — | — | — | 16777216 | unavailable | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| gelu_tanh_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| layer_norm_f64_n1048576 | jax-cuda | 16 | 15 | 3.297 | 16777216 | 0.057 | 3.398e-07 |
| layer_norm_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.987 | 16777216 | 0.378 | 3.138e-07 |
| layer_norm_f64_n1048576 | tenferro-cuda-direct | — | — | — | 16777216 | unavailable | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| layer_norm_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| log_softmax_f64_n1048576 | jax-cuda | 32 | 15 | 2.452 | 16777216 | 0.152 | 2.377e-07 |
| log_softmax_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.434 | 16777216 | 0.860 | 2.377e-07 |
| log_softmax_f64_n1048576 | tenferro-cuda-direct | — | — | — | 16777216 | unavailable | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| log_softmax_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| masked_log_softmax_f64_n1048576 | jax-cuda | 32 | 15 | 2.172 | 17825792 | 0.183 | 2.489e-07 |
| masked_log_softmax_f64_n1048576 | pytorch-cuda | — | — | — | 17825792 | unavailable | No equivalent stable single PyTorch public masked-softmax operation; masked_fill plus softmax would change the operation contract. |
| masked_log_softmax_f64_n1048576 | tenferro-cuda-direct | — | — | — | 17825792 | unavailable | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| masked_log_softmax_f64_n1048576 | tenferro-cuda-eager | — | — | — | 17825792 | unavailable | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| masked_softmax_f64_n1048576 | jax-cuda | 4 | 15 | 4.025 | 17825792 | 0.012 | 8.659e-10 |
| masked_softmax_f64_n1048576 | pytorch-cuda | — | — | — | 17825792 | unavailable | No equivalent stable single PyTorch public masked-softmax operation; masked_fill plus softmax would change the operation contract. |
| masked_softmax_f64_n1048576 | tenferro-cuda-direct | — | — | — | 17825792 | unavailable | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| masked_softmax_f64_n1048576 | tenferro-cuda-eager | — | — | — | 17825792 | unavailable | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| rms_norm_f64_n1048576 | jax-cuda | — | — | — | 16777216 | unavailable | No equivalent single public JAX RMS-normalization operation; a composition is not timed as the missing API. |
| rms_norm_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.983 | 16777216 | 0.380 | 2.654e-07 |
| rms_norm_f64_n1048576 | tenferro-cuda-direct | — | — | — | 16777216 | unavailable | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| rms_norm_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| sigmoid_f64_n1048576 | jax-cuda | 32 | 15 | 2.616 | 16777216 | 0.143 | 7.964e-08 |
| sigmoid_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.328 | 16777216 | 1.138 | 7.964e-08 |
| sigmoid_f64_n1048576 | tenferro-cuda-direct | — | — | — | 16777216 | unavailable | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| sigmoid_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| silu_f64_n1048576 | jax-cuda | 32 | 15 | 2.187 | 16777216 | 0.171 | 4.848e-08 |
| silu_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.365 | 16777216 | 1.022 | 4.848e-08 |
| silu_f64_n1048576 | tenferro-cuda-direct | — | — | — | 16777216 | unavailable | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| silu_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| softmax_f64_n1048576 | jax-cuda | 4 | 15 | 2.094 | 16777216 | 0.022 | 6.499e-10 |
| softmax_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.536 | 16777216 | 0.696 | 8.661e-10 |
| softmax_f64_n1048576 | tenferro-cuda-direct | — | — | — | 16777216 | unavailable | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| softmax_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| softplus_f64_n1048576 | jax-cuda | 32 | 15 | 2.516 | 16777216 | 0.148 | 1.467e-07 |
| softplus_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.832 | 16777216 | 0.449 | 1.467e-07 |
| softplus_f64_n1048576 | tenferro-cuda-direct | 8 | 15 | 1.654 | 16777216 | 0.056 | 1.467e-07 |
| softplus_f64_n1048576 | tenferro-cuda-eager | 4 | 15 | 1.375 | 16777216 | 0.034 | 1.467e-07 |

## Run metadata

```yaml
schema_version: 1
target_profile: nvidia-gpu
suite_id: gpu/nn
suite_file: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/data/results/nvidia-gpu/gpu/routine_selection/20261108_115237/gpu_nn.yaml
timestamp: '2026-11-08T11:52:37+00:00'
tenferro_rs:
  path: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/extern/tenferro-rs
  commit: 9a850b06155405acc99ecd0648829e9e6d3866ba
  dirty: false
  features:
  - cuda
  resolved_path: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/extern/tenferro-rs
harness:
  path: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark
  commit: cf44136e09063a2ca09d2fe0dbb38e8849cb4522
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
  cpu: AMD EPYC 7763 64-Core Processor
  logical_cpus: 252
  power:
    smt:
      active: '0'
      control: notsupported
  numa:
    nodes: 1
    cpulists:
      '0': 0-251
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
  - '128'
  - '129'
  - '130'
  - '131'
  - '132'
  - '133'
  - '134'
  - '135'
  - '136'
  - '137'
  - '138'
  - '139'
  - '140'
  - '141'
  - '142'
  - '143'
  - '144'
  - '145'
  - '146'
  - '147'
  - '148'
  - '149'
  - '150'
  - '151'
  - '152'
  - '153'
  - '154'
  - '155'
  - '156'
  - '157'
  - '158'
  - '159'
  - '160'
  - '161'
  - '162'
  - '163'
  - '164'
  - '165'
  - '166'
  - '167'
  - '168'
  - '169'
  - '170'
  - '171'
  - '172'
  - '173'
  - '174'
  - '175'
  - '176'
  - '177'
  - '178'
  - '179'
  - '180'
  - '181'
  - '182'
  - '183'
  - '184'
  - '185'
  - '186'
  - '187'
  - '188'
  - '189'
  - '190'
  - '191'
  - '192'
  - '193'
  - '194'
  - '195'
  - '196'
  - '197'
  - '198'
  - '199'
  - '200'
  - '201'
  - '202'
  - '203'
  - '204'
  - '205'
  - '206'
  - '207'
  - '208'
  - '209'
  - '210'
  - '211'
  - '212'
  - '213'
  - '214'
  - '215'
  - '216'
  - '217'
  - '218'
  - '219'
  - '220'
  - '221'
  - '222'
  - '223'
  - '224'
  - '225'
  - '226'
  - '227'
  - '228'
  - '229'
  - '230'
  - '231'
  - '232'
  - '233'
  - '234'
  - '235'
  - '236'
  - '237'
  - '238'
  - '239'
  - '240'
  - '241'
  - '242'
  - '243'
  - '244'
  - '245'
  - '246'
  - '247'
  - '248'
  - '249'
  - '250'
  - '251'
  l3_domains:
  - 0-125
  - 126-251
build:
  toolchain:
    rustc: null
    rustc_release: null
    rustflags: null
    target_cpu: null
environment:
  container: devcontainer
  hostname: 941a9a84f64c
  os: Linux-6.8.0-138-generic-x86_64-with-glibc2.39
  arch: x86_64
  cpu: x86_64
  affinity:
    pinning: none
    requested_measured_cpus: null
    observer_cpus: null
    override_of: null
    thread_budget: '1'
    measured_cpus: 0-251
    l3_domains:
    - 0-125
    - 126-251
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
    GPU_RUNTIME_SHA256: ecc740128a444e2b7d8a80ea018e5fbeb2da956dc2669f05886d8d7c97b49c67
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
  device_uuid: GPU-e8de7039-b03e-2d5e-3147-bf92141dbc74
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
  - 0.092705
  - 0.093285
  - 0.093978
  - 0.094808
  - 0.093957
  - 0.096402
  - 0.093826
  - 0.092735
  - 0.092443
  - 0.092294
  - 0.093176
  - 0.092353
  - 0.092746
  - 0.094216
  - 0.093645
  median_ms: 0.093285
  bytes_per_second: 1438792174518.9473
  torch_version: 2.11.0+cu128
  cuda_version: '12.8'
  timestamp: '2026-11-08T12:06:00.446079+00:00'
gpu_assignment:
  role: benchmark
  physical_device_index: 0
  device_uuid: GPU-e8de7039-b03e-2d5e-3147-bf92141dbc74
  cuda_visible_devices: GPU-e8de7039-b03e-2d5e-3147-bf92141dbc74
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
    src/bin/benchmark_gpu_public_api.rs: 28b921283d595842fc599670df7fadfa2b226573a287cb9fd90799300ca81e6c
    src/bin/benchmark_gpu_public_api/lifetime.rs: 34be366f56aac88f5f3b86ade8018a15dad2bca85cea74f56cb2fe121917d0a9
    src/bin/benchmark_gpu_public_api/linalg.rs: 9c3d7916a3281d2dfd1caa68551dc9c81c79dab6b7525324339e5c4816581e6d
    src/bin/benchmark_gpu_public_api/multi.rs: 949eadd6272c17ba4e380a228547a03608d328a5ef852eb95294731c288f24d9
    src/bin/benchmark_gpu_public_api/plan.rs: 70fce8a6e9ed7518779a51a9d8692e010fa1e470e95eee0ce5276f595737fde8
    src/bin/benchmark_gpu_public_api/read_layout.rs: fc42e2160094ba5f9c7dd6facb5667a76e10b73f971ef9facfb1f64723e42467
    src/bin/benchmark_gpu_public_api/reuse.rs: a5a4aa0edb0a1158f6669b642f4bc25a7079314fc2a711ef3ef57cc19c0a35b2
    src/bin/benchmark_gpu_public_api/runtime_ops.rs: c9d61b1e99bda517a59a6e34f625fd4d651bbbb19f6dc06ba4d0d1c0d50dfd6a
    src/bin/benchmark_gpu_public_api/state.rs: a8bbf8985bbc58c435e3c4dda7135161a141a90a10fa50d4f95128397ce949cb
    src/bin/benchmark_gpu_public_api/structural.rs: 8b61ac01b58edc090eafbc124ec135d5efeb4979631a939656ba78dd6e5c042a
    /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/scripts/benchmark_gpu_public_api.py: 32aa2ab45453502fc16479f429191aa24f867c8d5ad5f59d5a2752a3da413ee5
    /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/scripts/gpu_public_api_ops.py: e88823db3f4a1fe43dc13c4743c9d78ece6af76a968724091af318dd8d7f2528
    scripts/run_gpu_public_api.sh: 257228952e48c61fa1714d0cb2d6fdfdb9b0cf3898f59307ba9e72e513aa1b50
    data/results/nvidia-gpu/gpu/nn/20261108_115237/fixtures/manifest.json: 6019f8eb30ec023ad89d15545a6d050a2188b62ec40b3d90c773ad6e58eabea1
    target/release/benchmark_gpu_public_api: 62f92e0c72d6fb46531ee8106f112567dfd32ddb482255a7cc41ba6819929786
  reference_numpy_version: 2.4.2
  measurement_cpu:
    policy: 'disabled (GPU_BENCH_CPU_SELECT unset): arms run on the allocation unpinned'
    arms:
      jax-cuda:
        arm: jax-cuda
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-08T12:06:18Z'
      pytorch-cuda:
        arm: pytorch-cuda
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-08T12:06:10Z'
      tenferro-cuda-direct:
        arm: tenferro-cuda-direct
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-08T12:06:04Z'
      tenferro-cuda-eager:
        arm: tenferro-cuda-eager
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-08T12:06:05Z'
  compile:
    case_wall_seconds: 9.022
    compiling_processes: 0
    modules_compiled: 0
    compile_time_ms: null
    compile_time_scope: 'not measured: no compile interval was recorded; `compile_modules`
      counts the NVRTC modules and the untimed wall is the case wall'
    untimed_wall_seconds: 0.0
```

## Untimed work over the recorded budget

Each row records the wall clock it spent outside its timed region. These cases spent more than the recorded budget ratio (0.2) there. The threshold has no measured basis, so this is a recorded warning: it does not change a cell, its acceptance or the exit status of the run.

| Case | Backend | Warning |
| --- | --- | --- |
| softplus_f64_n1048576 | tenferro-cuda-direct | softplus_f64_n1048576 untimed 104.4 ms is 422% of the 24.7 ms timed region (budget 20%) |
| softplus_f64_n1048576 | tenferro-cuda-eager | softplus_f64_n1048576 untimed 105.2 ms is 509% of the 20.7 ms timed region (budget 20%) |
| sigmoid_f64_n1048576 | pytorch-cuda | sigmoid_f64_n1048576 untimed 378.1 ms is 7663% of the 4.9 ms timed region (budget 20%) |
| silu_f64_n1048576 | pytorch-cuda | silu_f64_n1048576 untimed 201.6 ms is 3663% of the 5.5 ms timed region (budget 20%) |
| softplus_f64_n1048576 | pytorch-cuda | softplus_f64_n1048576 untimed 205.2 ms is 1621% of the 12.7 ms timed region (budget 20%) |
| gelu_f64_n1048576 | pytorch-cuda | gelu_f64_n1048576 untimed 200.6 ms is 2731% of the 7.3 ms timed region (budget 20%) |
| gelu_tanh_f64_n1048576 | pytorch-cuda | gelu_tanh_f64_n1048576 untimed 188.2 ms is 4051% of the 4.6 ms timed region (budget 20%) |
| softmax_f64_n1048576 | pytorch-cuda | softmax_f64_n1048576 untimed 206.0 ms is 2560% of the 8.0 ms timed region (budget 20%) |
| log_softmax_f64_n1048576 | pytorch-cuda | log_softmax_f64_n1048576 untimed 197.5 ms is 3030% of the 6.5 ms timed region (budget 20%) |
| layer_norm_f64_n1048576 | pytorch-cuda | layer_norm_f64_n1048576 untimed 201.0 ms is 1357% of the 14.8 ms timed region (budget 20%) |
| rms_norm_f64_n1048576 | pytorch-cuda | rms_norm_f64_n1048576 untimed 190.3 ms is 1290% of the 14.7 ms timed region (budget 20%) |
| sigmoid_f64_n1048576 | jax-cuda | sigmoid_f64_n1048576 untimed 578.6 ms is 1461% of the 39.6 ms timed region (budget 20%) |
| silu_f64_n1048576 | jax-cuda | silu_f64_n1048576 untimed 258.2 ms is 789% of the 32.7 ms timed region (budget 20%) |
| softplus_f64_n1048576 | jax-cuda | softplus_f64_n1048576 untimed 287.9 ms is 764% of the 37.7 ms timed region (budget 20%) |
| gelu_f64_n1048576 | jax-cuda | gelu_f64_n1048576 untimed 585.4 ms is 1180% of the 49.6 ms timed region (budget 20%) |
| gelu_tanh_f64_n1048576 | jax-cuda | gelu_tanh_f64_n1048576 untimed 397.8 ms is 1057% of the 37.6 ms timed region (budget 20%) |
| softmax_f64_n1048576 | jax-cuda | softmax_f64_n1048576 untimed 837.7 ms is 2684% of the 31.2 ms timed region (budget 20%) |
| log_softmax_f64_n1048576 | jax-cuda | log_softmax_f64_n1048576 untimed 351.8 ms is 983% of the 35.8 ms timed region (budget 20%) |
| masked_softmax_f64_n1048576 | jax-cuda | masked_softmax_f64_n1048576 untimed 675.8 ms is 1162% of the 58.2 ms timed region (budget 20%) |
| masked_log_softmax_f64_n1048576 | jax-cuda | masked_log_softmax_f64_n1048576 untimed 369.1 ms is 1135% of the 32.5 ms timed region (budget 20%) |
| layer_norm_f64_n1048576 | jax-cuda | layer_norm_f64_n1048576 untimed 293.1 ms is 603% of the 48.6 ms timed region (budget 20%) |