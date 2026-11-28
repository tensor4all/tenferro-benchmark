# gpu/nn: CUDA public API benchmark

Measured cells: **41**. Unsupported cells: **3**, including **0** numerically verified but timing-excluded cells. Unsupported cells do not count as measured coverage.

Logical traffic: this run measured a plain device copy of 67108864 bytes at **1413.5 GB/s** on this device. Every record carries the logical bytes the case must move per operation and the ratio of the case's implied rate to that copy rate, in the sampling table below. Logical bytes are the declared array sizes, not device DRAM traffic: a case whose operands stay in cache moves less DRAM than its arrays suggest, so a ratio above 1 is recorded, not rejected.

Raw run: `data/results/nvidia-gpu/gpu/nn/20261128_082346/`.

Isolated public APIs and explicit unsupported timing contracts. Issue 155.

This run compiled **1 NVRTC modules** in 1 of its arms, because the CUDA artifact cache started cold. The untimed wall of the cases that compiled is **0.1 s** of its 7.8 s of recorded case wall clock, and that wall also holds their fixture read, transfers and verification, so it is not a compile time.

Values are median (IQR) **µs per operation**. Samples time batches of declared operations plus device synchronization; retained output memory is limited to 256 MiB. Batch counts and raw durations are preserved in JSONL.

Cases marked † (**6** of 11) were sampled with a different number of operations per sample on different arms (see Ops/sample in the sampling table). Each value is still per operation, but the arms amortise launch and synchronization over different batch sizes, so a ratio between their columns is approximate and is not presented as a like-for-like comparison.

Setup, input transfers, library/plan initialization, JIT compilation, validation downloads and output destruction are outside timing. Intrinsic output allocation is inside timing; output-reuse rows instead use preallocated destinations across samples. Accumulating destinations are restored from prepared GPU seeds and synchronized before every batch clock, including initialization and calibration. Direct and eager each use one entered session across samples; direct FFT uses cached `FftExecutor`. JAX eager and compiled JIT are separate arms.

All measured output elements are checked against independently generated NumPy/SciPy reference results before and after sampling. Unsupported rows include unavailable APIs and APIs whose internal setup cannot be excluded; these are not performance measurements. Short batches limited by output memory remain identifiable in the sampling table.

| Case | tenferro-cuda-direct | tenferro-cuda-eager | pytorch-cuda | jax-cuda |
| --- | ---: | ---: | ---: | ---: |
| sigmoid_f64_n1048576 † | 90.845 (0.168) | 90.305 (0.057) | 9.929 (0.060) | 58.278 (5.720) |
| silu_f64_n1048576 | 101.202 (0.176) | 99.598 (0.397) | 11.093 (0.034) | 48.363 (3.264) |
| softplus_f64_n1048576 | 28.399 (0.033) | 28.187 (0.031) | 25.698 (0.048) | 44.800 (1.253) |
| gelu_f64_n1048576 † | 73.942 (0.063) | 73.490 (0.104) | 14.938 (0.063) | 554.109 (11.224) |
| gelu_tanh_f64_n1048576 † | 114.079 (0.037) | 112.775 (0.091) | 9.429 (0.084) | 874.972 (21.686) |
| softmax_f64_n1048576 † | 90.525 (0.084) | 90.052 (0.163) | 16.336 (0.028) | 456.826 (8.428) |
| log_softmax_f64_n1048576 | 92.664 (0.119) | 91.614 (0.189) | 12.712 (0.045) | 45.532 (3.258) |
| masked_softmax_f64_n1048576 † | 116.188 (0.124) | 115.976 (0.263) | unsupported | 710.940 (15.913) |
| masked_log_softmax_f64_n1048576 | 114.135 (0.096) | 113.399 (0.171) | unsupported | 43.376 (1.498) |
| layer_norm_f64_n1048576 † | 87.720 (0.116) | 86.437 (0.089) | 30.490 (0.058) | 149.511 (5.885) |
| rms_norm_f64_n1048576 | 48.706 (0.135) | 47.988 (0.121) | 30.345 (0.045) | unsupported |

## Sampling

| Case | Backend | Ops/sample | Samples | Median batch ms | Bytes/op | Copy ratio | Verification error/tolerance |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gelu_f64_n1048576 | jax-cuda | 4 | 15 | 2.216 | 16777216 | 0.021 | 4.81e-08 |
| gelu_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.478 | 16777216 | 0.795 | 4.899e-08 |
| gelu_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.366 | 16777216 | 0.161 | 4.899e-08 |
| gelu_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.352 | 16777216 | 0.162 | 4.899e-08 |
| gelu_tanh_f64_n1048576 | jax-cuda | 4 | 15 | 3.500 | 16777216 | 0.014 | 4.349e-08 |
| gelu_tanh_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.302 | 16777216 | 1.259 | 4.349e-08 |
| gelu_tanh_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.651 | 16777216 | 0.104 | 4.349e-08 |
| gelu_tanh_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.609 | 16777216 | 0.105 | 4.349e-08 |
| layer_norm_f64_n1048576 | jax-cuda | 16 | 15 | 2.392 | 16777216 | 0.079 | 3.398e-07 |
| layer_norm_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.976 | 16777216 | 0.389 | 3.138e-07 |
| layer_norm_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.807 | 16777216 | 0.135 | 3.251e-07 |
| layer_norm_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.766 | 16777216 | 0.137 | 3.251e-07 |
| log_softmax_f64_n1048576 | jax-cuda | 32 | 15 | 1.457 | 16777216 | 0.261 | 2.377e-07 |
| log_softmax_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.407 | 16777216 | 0.934 | 2.377e-07 |
| log_softmax_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.965 | 16777216 | 0.128 | 2.377e-07 |
| log_softmax_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.932 | 16777216 | 0.130 | 2.377e-07 |
| masked_log_softmax_f64_n1048576 | jax-cuda | 32 | 15 | 1.388 | 17825792 | 0.291 | 2.489e-07 |
| masked_log_softmax_f64_n1048576 | pytorch-cuda | — | — | — | 17825792 | unavailable | No equivalent stable single PyTorch public masked-softmax operation; masked_fill plus softmax would change the operation contract. |
| masked_log_softmax_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.652 | 17825792 | 0.110 | 2.489e-07 |
| masked_log_softmax_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.629 | 17825792 | 0.111 | 2.489e-07 |
| masked_softmax_f64_n1048576 | jax-cuda | 4 | 15 | 2.844 | 17825792 | 0.018 | 8.658e-10 |
| masked_softmax_f64_n1048576 | pytorch-cuda | — | — | — | 17825792 | unavailable | No equivalent stable single PyTorch public masked-softmax operation; masked_fill plus softmax would change the operation contract. |
| masked_softmax_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.718 | 17825792 | 0.109 | 1.299e-09 |
| masked_softmax_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.711 | 17825792 | 0.109 | 1.299e-09 |
| rms_norm_f64_n1048576 | jax-cuda | — | — | — | 16777216 | unavailable | No equivalent single public JAX RMS-normalization operation; a composition is not timed as the missing API. |
| rms_norm_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.971 | 16777216 | 0.391 | 2.654e-07 |
| rms_norm_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.559 | 16777216 | 0.244 | 2.71e-07 |
| rms_norm_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.536 | 16777216 | 0.247 | 2.71e-07 |
| sigmoid_f64_n1048576 | jax-cuda | 8 | 15 | 0.466 | 16777216 | 0.204 | 7.964e-08 |
| sigmoid_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.318 | 16777216 | 1.195 | 7.964e-08 |
| sigmoid_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.907 | 16777216 | 0.131 | 8.044e-08 |
| sigmoid_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.890 | 16777216 | 0.131 | 8.044e-08 |
| silu_f64_n1048576 | jax-cuda | 32 | 15 | 1.548 | 16777216 | 0.245 | 4.848e-08 |
| silu_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.355 | 16777216 | 1.070 | 4.848e-08 |
| silu_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 3.238 | 16777216 | 0.117 | 4.912e-08 |
| silu_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.187 | 16777216 | 0.119 | 4.912e-08 |
| softmax_f64_n1048576 | jax-cuda | 8 | 15 | 3.655 | 16777216 | 0.026 | 8.661e-10 |
| softmax_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.523 | 16777216 | 0.727 | 8.661e-10 |
| softmax_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.897 | 16777216 | 0.131 | 1.083e-09 |
| softmax_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.882 | 16777216 | 0.132 | 1.083e-09 |
| softplus_f64_n1048576 | jax-cuda | 32 | 15 | 1.434 | 16777216 | 0.265 | 1.467e-07 |
| softplus_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.822 | 16777216 | 0.462 | 1.467e-07 |
| softplus_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.909 | 16777216 | 0.418 | 1.467e-07 |
| softplus_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.902 | 16777216 | 0.421 | 1.467e-07 |

## Run metadata

```yaml
schema_version: 1
target_profile: nvidia-gpu
suite_id: gpu/nn
suite_file: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/data/results/nvidia-gpu/gpu/routine_selection/20261128_082346/gpu_nn.yaml
timestamp: '2026-11-28T08:23:46+00:00'
tenferro_rs:
  path: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/extern/tenferro-rs
  commit: 1ba9f33521691170c763dfce9b6c9bb56ca523c2
  dirty: false
  features:
  - cuda
  resolved_path: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/extern/tenferro-rs
harness:
  path: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark
  commit: 28b35e0408e272b66d7460e3fa12213799c543f9
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
  cpu: AMD EPYC 9554 64-Core Processor
  logical_cpus: 252
  power:
    smt:
      active: '0'
      control: notsupported
  numa:
    nodes: 2
    cpulists:
      '0': 0-125
      '1': 126-251
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
  hostname: 32b786430f5d
  os: Linux-6.8.0-149-generic-x86_64-with-glibc2.39
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
  device_uuid: GPU-12b45076-79d5-7b84-cb5c-7c6300074c8a
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
  - 0.09318
  - 0.09261
  - 0.097206
  - 0.094582
  - 0.090757
  - 0.097306
  - 0.095204
  - 0.092089
  - 0.097136
  - 0.094952
  - 0.09318
  - 0.095996
  - 0.096235
  - 0.094223
  - 0.096965
  median_ms: 0.094952
  bytes_per_second: 1413532395315.5278
  torch_version: 2.11.0+cu128
  cuda_version: '12.8'
  timestamp: '2026-11-28T08:34:56.483386+00:00'
gpu_assignment:
  role: benchmark
  physical_device_index: 0
  device_uuid: GPU-12b45076-79d5-7b84-cb5c-7c6300074c8a
  cuda_visible_devices: GPU-12b45076-79d5-7b84-cb5c-7c6300074c8a
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
    data/results/nvidia-gpu/gpu/nn/20261128_082346/fixtures/manifest.json: 06ffcecd4fd9ca2058834e8c222d97a2dc8d86d5506100cc2b091e779e957bbf
    target/release/benchmark_gpu_public_api: 54dda9d9f7540806d7ff5a5d68e1312bc33b0d58871d8e3e9467a819e7c59318
  reference_numpy_version: 2.4.2
  measurement_cpu:
    policy: 'disabled (GPU_BENCH_CPU_SELECT unset): arms run on the allocation unpinned'
    arms:
      jax-cuda:
        arm: jax-cuda
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-28T08:35:10Z'
      pytorch-cuda:
        arm: pytorch-cuda
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-28T08:35:05Z'
      tenferro-cuda-direct:
        arm: tenferro-cuda-direct
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-28T08:35:00Z'
      tenferro-cuda-eager:
        arm: tenferro-cuda-eager
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-28T08:35:02Z'
  compile:
    case_wall_seconds: 7.802
    compiling_processes: 1
    modules_compiled: 1
    compile_time_ms: null
    compile_time_scope: 'not measured: no compile interval was recorded; `compile_modules`
      counts the NVRTC modules and the untimed wall is the case wall'
    untimed_wall_seconds: 0.129
```

## Untimed work over the recorded budget

Each row records the wall clock it spent outside its timed region. These cases spent more than the recorded budget ratio (0.2) there. The threshold has no measured basis, so this is a recorded warning: it does not change a cell, its acceptance or the exit status of the run.

| Case | Backend | Warning |
| --- | --- | --- |
| sigmoid_f64_n1048576 | tenferro-cuda-direct | sigmoid_f64_n1048576 untimed 159.5 ms is 366% of the 43.6 ms timed region (budget 20%) |
| silu_f64_n1048576 | tenferro-cuda-direct | silu_f64_n1048576 untimed 85.1 ms is 175% of the 48.6 ms timed region (budget 20%) |
| softplus_f64_n1048576 | tenferro-cuda-direct | softplus_f64_n1048576 untimed 129.2 ms is 947% of the 13.6 ms timed region (budget 20%) |
| gelu_f64_n1048576 | tenferro-cuda-direct | gelu_f64_n1048576 untimed 86.8 ms is 244% of the 35.5 ms timed region (budget 20%) |
| gelu_tanh_f64_n1048576 | tenferro-cuda-direct | gelu_tanh_f64_n1048576 untimed 76.5 ms is 140% of the 54.8 ms timed region (budget 20%) |
| softmax_f64_n1048576 | tenferro-cuda-direct | softmax_f64_n1048576 untimed 117.9 ms is 271% of the 43.5 ms timed region (budget 20%) |
| log_softmax_f64_n1048576 | tenferro-cuda-direct | log_softmax_f64_n1048576 untimed 84.2 ms is 189% of the 44.5 ms timed region (budget 20%) |
| masked_softmax_f64_n1048576 | tenferro-cuda-direct | masked_softmax_f64_n1048576 untimed 88.8 ms is 159% of the 55.8 ms timed region (budget 20%) |
| masked_log_softmax_f64_n1048576 | tenferro-cuda-direct | masked_log_softmax_f64_n1048576 untimed 67.1 ms is 122% of the 54.8 ms timed region (budget 20%) |
| layer_norm_f64_n1048576 | tenferro-cuda-direct | layer_norm_f64_n1048576 untimed 95.2 ms is 226% of the 42.1 ms timed region (budget 20%) |
| rms_norm_f64_n1048576 | tenferro-cuda-direct | rms_norm_f64_n1048576 untimed 90.5 ms is 387% of the 23.4 ms timed region (budget 20%) |
| sigmoid_f64_n1048576 | tenferro-cuda-eager | sigmoid_f64_n1048576 untimed 149.7 ms is 345% of the 43.4 ms timed region (budget 20%) |
| silu_f64_n1048576 | tenferro-cuda-eager | silu_f64_n1048576 untimed 90.2 ms is 189% of the 47.8 ms timed region (budget 20%) |
| softplus_f64_n1048576 | tenferro-cuda-eager | softplus_f64_n1048576 untimed 77.3 ms is 571% of the 13.5 ms timed region (budget 20%) |
| gelu_f64_n1048576 | tenferro-cuda-eager | gelu_f64_n1048576 untimed 80.6 ms is 228% of the 35.3 ms timed region (budget 20%) |
| gelu_tanh_f64_n1048576 | tenferro-cuda-eager | gelu_tanh_f64_n1048576 untimed 79.5 ms is 147% of the 54.1 ms timed region (budget 20%) |
| softmax_f64_n1048576 | tenferro-cuda-eager | softmax_f64_n1048576 untimed 94.6 ms is 219% of the 43.3 ms timed region (budget 20%) |
| log_softmax_f64_n1048576 | tenferro-cuda-eager | log_softmax_f64_n1048576 untimed 95.0 ms is 216% of the 44.0 ms timed region (budget 20%) |
| masked_softmax_f64_n1048576 | tenferro-cuda-eager | masked_softmax_f64_n1048576 untimed 88.1 ms is 158% of the 55.7 ms timed region (budget 20%) |
| masked_log_softmax_f64_n1048576 | tenferro-cuda-eager | masked_log_softmax_f64_n1048576 untimed 65.5 ms is 120% of the 54.4 ms timed region (budget 20%) |
| layer_norm_f64_n1048576 | tenferro-cuda-eager | layer_norm_f64_n1048576 untimed 92.9 ms is 224% of the 41.5 ms timed region (budget 20%) |
| rms_norm_f64_n1048576 | tenferro-cuda-eager | rms_norm_f64_n1048576 untimed 89.6 ms is 388% of the 23.1 ms timed region (budget 20%) |
| sigmoid_f64_n1048576 | pytorch-cuda | sigmoid_f64_n1048576 untimed 208.3 ms is 4359% of the 4.8 ms timed region (budget 20%) |
| silu_f64_n1048576 | pytorch-cuda | silu_f64_n1048576 untimed 111.7 ms is 2101% of the 5.3 ms timed region (budget 20%) |
| softplus_f64_n1048576 | pytorch-cuda | softplus_f64_n1048576 untimed 112.0 ms is 908% of the 12.3 ms timed region (budget 20%) |
| gelu_f64_n1048576 | pytorch-cuda | gelu_f64_n1048576 untimed 117.0 ms is 1630% of the 7.2 ms timed region (budget 20%) |
| gelu_tanh_f64_n1048576 | pytorch-cuda | gelu_tanh_f64_n1048576 untimed 113.3 ms is 2505% of the 4.5 ms timed region (budget 20%) |
| softmax_f64_n1048576 | pytorch-cuda | softmax_f64_n1048576 untimed 131.2 ms is 1672% of the 7.8 ms timed region (budget 20%) |
| log_softmax_f64_n1048576 | pytorch-cuda | log_softmax_f64_n1048576 untimed 116.9 ms is 1914% of the 6.1 ms timed region (budget 20%) |
| layer_norm_f64_n1048576 | pytorch-cuda | layer_norm_f64_n1048576 untimed 124.3 ms is 849% of the 14.6 ms timed region (budget 20%) |
| rms_norm_f64_n1048576 | pytorch-cuda | rms_norm_f64_n1048576 untimed 114.6 ms is 787% of the 14.6 ms timed region (budget 20%) |
| sigmoid_f64_n1048576 | jax-cuda | sigmoid_f64_n1048576 untimed 364.8 ms is 5000% of the 7.3 ms timed region (budget 20%) |
| silu_f64_n1048576 | jax-cuda | silu_f64_n1048576 untimed 198.2 ms is 854% of the 23.2 ms timed region (budget 20%) |
| softplus_f64_n1048576 | jax-cuda | softplus_f64_n1048576 untimed 195.9 ms is 899% of the 21.8 ms timed region (budget 20%) |
| gelu_f64_n1048576 | jax-cuda | gelu_f64_n1048576 untimed 421.5 ms is 1264% of the 33.4 ms timed region (budget 20%) |
| gelu_tanh_f64_n1048576 | jax-cuda | gelu_tanh_f64_n1048576 untimed 288.9 ms is 545% of the 53.0 ms timed region (budget 20%) |
| softmax_f64_n1048576 | jax-cuda | softmax_f64_n1048576 untimed 608.7 ms is 1106% of the 55.0 ms timed region (budget 20%) |
| log_softmax_f64_n1048576 | jax-cuda | log_softmax_f64_n1048576 untimed 244.9 ms is 1120% of the 21.9 ms timed region (budget 20%) |
| masked_softmax_f64_n1048576 | jax-cuda | masked_softmax_f64_n1048576 untimed 470.4 ms is 1108% of the 42.5 ms timed region (budget 20%) |
| masked_log_softmax_f64_n1048576 | jax-cuda | masked_log_softmax_f64_n1048576 untimed 265.8 ms is 1258% of the 21.1 ms timed region (budget 20%) |
| layer_norm_f64_n1048576 | jax-cuda | layer_norm_f64_n1048576 untimed 200.8 ms is 559% of the 35.9 ms timed region (budget 20%) |