# gpu/nn: CUDA public API benchmark

Measured cells: **21**. Unsupported cells: **23**, including **20** numerically verified but timing-excluded cells. Unsupported cells do not count as measured coverage.

Logical traffic: this run measured a plain device copy of 67108864 bytes at **1425.8 GB/s** on this device. Every record carries the logical bytes the case must move per operation and the ratio of the case's implied rate to that copy rate, in the sampling table below. Logical bytes are the declared array sizes, not device DRAM traffic: a case whose operands stay in cache moves less DRAM than its arrays suggest, so a ratio above 1 is recorded, not rejected.

Raw run: `data/results/nvidia-gpu/gpu/nn/20261107_201728/`.

Isolated public APIs and explicit unsupported timing contracts. Issue 155.

This run did not compile: no NVRTC module was built in any of its arms. That describes this run only, and does not show that the CUDA artifact cache was warm.

Values are median (IQR) **µs per operation**. Samples time batches of declared operations plus device synchronization; retained output memory is limited to 256 MiB. Batch counts and raw durations are preserved in JSONL.

Cases marked † (**4** of 11) were sampled with a different number of operations per sample on different arms (see Ops/sample in the sampling table). Each value is still per operation, but the arms amortise launch and synchronization over different batch sizes, so a ratio between their columns is approximate and is not presented as a like-for-like comparison.

Setup, input transfers, library/plan initialization, JIT compilation, validation downloads and output destruction are outside timing. Intrinsic output allocation is inside timing; output-reuse rows instead use preallocated destinations across samples. Accumulating destinations are restored from prepared GPU seeds and synchronized before every batch clock, including initialization and calibration. Direct and eager each use one entered session across samples; direct FFT uses cached `FftExecutor`. JAX eager and compiled JIT are separate arms.

All measured output elements are checked against independently generated NumPy/SciPy reference results before and after sampling. Unsupported rows include unavailable APIs and APIs whose internal setup cannot be excluded; these are not performance measurements. Short batches limited by output memory remain identifiable in the sampling table.

| Case | tenferro-cuda-direct | tenferro-cuda-eager | pytorch-cuda | jax-cuda |
| --- | ---: | ---: | ---: | ---: |
| sigmoid_f64_n1048576 | unsupported | unsupported | 12.196 (0.110) | 50.554 (1.493) |
| silu_f64_n1048576 | unsupported | unsupported | 13.670 (0.084) | 50.160 (1.459) |
| softplus_f64_n1048576 † | 262.319 (0.620) | 431.990 (2.631) | 31.335 (0.023) | 47.519 (1.744) |
| gelu_f64_n1048576 † | unsupported | unsupported | 18.352 (0.052) | 562.268 (30.453) |
| gelu_tanh_f64_n1048576 † | unsupported | unsupported | 11.582 (0.053) | 847.488 (19.638) |
| softmax_f64_n1048576 † | unsupported | unsupported | 20.183 (0.039) | 391.898 (7.166) |
| log_softmax_f64_n1048576 | unsupported | unsupported | 15.863 (0.051) | 48.980 (2.316) |
| masked_softmax_f64_n1048576 | unsupported | unsupported | unsupported | 642.853 (21.864) |
| masked_log_softmax_f64_n1048576 | unsupported | unsupported | unsupported | 46.404 (1.106) |
| layer_norm_f64_n1048576 | unsupported | unsupported | 36.836 (0.056) | 127.409 (6.635) |
| rms_norm_f64_n1048576 | unsupported | unsupported | 36.662 (0.080) | unsupported |

## Sampling

| Case | Backend | Ops/sample | Samples | Median batch ms | Bytes/op | Copy ratio | Verification error/tolerance |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| gelu_f64_n1048576 | jax-cuda | 4 | 15 | 2.249 | 16777216 | 0.021 | 4.81e-08 |
| gelu_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.587 | 16777216 | 0.641 | 4.899e-08 |
| gelu_f64_n1048576 | tenferro-cuda-direct | — | — | — | 16777216 | unavailable | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| gelu_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| gelu_tanh_f64_n1048576 | jax-cuda | 4 | 15 | 3.390 | 16777216 | 0.014 | 4.349e-08 |
| gelu_tanh_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.371 | 16777216 | 1.016 | 4.349e-08 |
| gelu_tanh_f64_n1048576 | tenferro-cuda-direct | — | — | — | 16777216 | unavailable | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| gelu_tanh_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| layer_norm_f64_n1048576 | jax-cuda | 32 | 15 | 4.077 | 16777216 | 0.092 | 3.398e-07 |
| layer_norm_f64_n1048576 | pytorch-cuda | 32 | 15 | 1.179 | 16777216 | 0.319 | 3.138e-07 |
| layer_norm_f64_n1048576 | tenferro-cuda-direct | — | — | — | 16777216 | unavailable | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| layer_norm_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| log_softmax_f64_n1048576 | jax-cuda | 32 | 15 | 1.567 | 16777216 | 0.240 | 2.377e-07 |
| log_softmax_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.508 | 16777216 | 0.742 | 2.377e-07 |
| log_softmax_f64_n1048576 | tenferro-cuda-direct | — | — | — | 16777216 | unavailable | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| log_softmax_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| masked_log_softmax_f64_n1048576 | jax-cuda | 32 | 15 | 1.485 | 17825792 | 0.269 | 2.489e-07 |
| masked_log_softmax_f64_n1048576 | pytorch-cuda | — | — | — | 17825792 | unavailable | No equivalent stable single PyTorch public masked-softmax operation; masked_fill plus softmax would change the operation contract. |
| masked_log_softmax_f64_n1048576 | tenferro-cuda-direct | — | — | — | 17825792 | unavailable | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| masked_log_softmax_f64_n1048576 | tenferro-cuda-eager | — | — | — | 17825792 | unavailable | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| masked_softmax_f64_n1048576 | jax-cuda | 4 | 15 | 2.571 | 17825792 | 0.019 | 8.659e-10 |
| masked_softmax_f64_n1048576 | pytorch-cuda | — | — | — | 17825792 | unavailable | No equivalent stable single PyTorch public masked-softmax operation; masked_fill plus softmax would change the operation contract. |
| masked_softmax_f64_n1048576 | tenferro-cuda-direct | — | — | — | 17825792 | unavailable | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| masked_softmax_f64_n1048576 | tenferro-cuda-eager | — | — | — | 17825792 | unavailable | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| rms_norm_f64_n1048576 | jax-cuda | — | — | — | 16777216 | unavailable | No equivalent single public JAX RMS-normalization operation; a composition is not timed as the missing API. |
| rms_norm_f64_n1048576 | pytorch-cuda | 32 | 15 | 1.173 | 16777216 | 0.321 | 2.654e-07 |
| rms_norm_f64_n1048576 | tenferro-cuda-direct | — | — | — | 16777216 | unavailable | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| rms_norm_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| sigmoid_f64_n1048576 | jax-cuda | 32 | 15 | 1.618 | 16777216 | 0.233 | 7.964e-08 |
| sigmoid_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.390 | 16777216 | 0.965 | 7.964e-08 |
| sigmoid_f64_n1048576 | tenferro-cuda-direct | — | — | — | 16777216 | unavailable | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| sigmoid_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| silu_f64_n1048576 | jax-cuda | 32 | 15 | 1.605 | 16777216 | 0.235 | 4.848e-08 |
| silu_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.437 | 16777216 | 0.861 | 4.848e-08 |
| silu_f64_n1048576 | tenferro-cuda-direct | — | — | — | 16777216 | unavailable | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| silu_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| softmax_f64_n1048576 | jax-cuda | 8 | 15 | 3.135 | 16777216 | 0.030 | 6.499e-10 |
| softmax_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.646 | 16777216 | 0.583 | 8.661e-10 |
| softmax_f64_n1048576 | tenferro-cuda-direct | — | — | — | 16777216 | unavailable | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| softmax_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777216 | unavailable | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| softplus_f64_n1048576 | jax-cuda | 32 | 15 | 1.521 | 16777216 | 0.248 | 1.467e-07 |
| softplus_f64_n1048576 | pytorch-cuda | 32 | 15 | 1.003 | 16777216 | 0.376 | 1.467e-07 |
| softplus_f64_n1048576 | tenferro-cuda-direct | 8 | 15 | 2.099 | 16777216 | 0.045 | 1.467e-07 |
| softplus_f64_n1048576 | tenferro-cuda-eager | 4 | 15 | 1.728 | 16777216 | 0.027 | 1.467e-07 |

## Run metadata

```yaml
schema_version: 1
target_profile: nvidia-gpu
suite_id: gpu/nn
suite_file: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/data/results/nvidia-gpu/gpu/routine_selection/20261107_201728/gpu_nn.yaml
timestamp: '2026-11-07T20:17:28+00:00'
tenferro_rs:
  path: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/extern/tenferro-rs
  commit: ffd352bdb9f98537be747101bc278db8a50a1595
  dirty: false
  features:
  - cuda
  resolved_path: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/extern/tenferro-rs
harness:
  path: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark
  commit: e7860ca369657a92a929026378d2a4c11adaae16
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
  cpu: AMD EPYC 7742 64-Core Processor
  logical_cpus: 256
  power:
    cpufreq_boost: '1'
    governor: schedutil
    scaling_max_freq_khz: '2250000'
    smt:
      active: '1'
      control: 'on'
  numa:
    nodes: 8
    cpulists:
      '0': 0-15,128-143
      '1': 16-31,144-159
      '2': 32-47,160-175
      '3': 48-63,176-191
      '4': 64-79,192-207
      '5': 80-95,208-223
      '6': 96-111,224-239
      '7': 112-127,240-255
  smt_siblings:
  - 0,128
  - 1,129
  - 2,130
  - 3,131
  - 4,132
  - 5,133
  - 6,134
  - 7,135
  - 8,136
  - 9,137
  - 10,138
  - 11,139
  - 12,140
  - 13,141
  - 14,142
  - 15,143
  - 16,144
  - 17,145
  - 18,146
  - 19,147
  - 20,148
  - 21,149
  - 22,150
  - 23,151
  - 24,152
  - 25,153
  - 26,154
  - 27,155
  - 28,156
  - 29,157
  - 30,158
  - 31,159
  - 32,160
  - 33,161
  - 34,162
  - 35,163
  - 36,164
  - 37,165
  - 38,166
  - 39,167
  - 40,168
  - 41,169
  - 42,170
  - 43,171
  - 44,172
  - 45,173
  - 46,174
  - 47,175
  - 48,176
  - 49,177
  - 50,178
  - 51,179
  - 52,180
  - 53,181
  - 54,182
  - 55,183
  - 56,184
  - 57,185
  - 58,186
  - 59,187
  - 60,188
  - 61,189
  - 62,190
  - 63,191
  - 64,192
  - 65,193
  - 66,194
  - 67,195
  - 68,196
  - 69,197
  - 70,198
  - 71,199
  - 72,200
  - 73,201
  - 74,202
  - 75,203
  - 76,204
  - 77,205
  - 78,206
  - 79,207
  - 80,208
  - 81,209
  - 82,210
  - 83,211
  - 84,212
  - 85,213
  - 86,214
  - 87,215
  - 88,216
  - 89,217
  - 90,218
  - 91,219
  - 92,220
  - 93,221
  - 94,222
  - 95,223
  - 96,224
  - 97,225
  - 98,226
  - 99,227
  - 100,228
  - 101,229
  - 102,230
  - 103,231
  - 104,232
  - 105,233
  - 106,234
  - 107,235
  - 108,236
  - 109,237
  - 110,238
  - 111,239
  - 112,240
  - 113,241
  - 114,242
  - 115,243
  - 116,244
  - 117,245
  - 118,246
  - 119,247
  - 120,248
  - 121,249
  - 122,250
  - 123,251
  - 124,252
  - 125,253
  - 126,254
  - 127,255
  l3_domains:
  - 0-3,128-131
  - 4-7,132-135
  - 8-11,136-139
  - 12-15,140-143
  - 16-19,144-147
  - 20-23,148-151
  - 24-27,152-155
  - 28-31,156-159
  - 32-35,160-163
  - 36-39,164-167
  - 40-43,168-171
  - 44-47,172-175
  - 48-51,176-179
  - 52-55,180-183
  - 56-59,184-187
  - 60-63,188-191
  - 64-67,192-195
  - 68-71,196-199
  - 72-75,200-203
  - 76-79,204-207
  - 80-83,208-211
  - 84-87,212-215
  - 88-91,216-219
  - 92-95,220-223
  - 96-99,224-227
  - 100-103,228-231
  - 104-107,232-235
  - 108-111,236-239
  - 112-115,240-243
  - 116-119,244-247
  - 120-123,248-251
  - 124-127,252-255
build:
  toolchain:
    rustc: null
    rustc_release: null
    rustflags: null
    target_cpu: null
environment:
  container: devcontainer
  hostname: 86889f3412c3
  os: Linux-6.8.0-138-generic-x86_64-with-glibc2.39
  arch: x86_64
  cpu: x86_64
  affinity:
    pinning: none
    requested_measured_cpus: null
    observer_cpus: null
    override_of: null
    thread_budget: '1'
    measured_cpus: 0-255
    l3_domains:
    - 0-3,128-131
    - 4-7,132-135
    - 8-11,136-139
    - 12-15,140-143
    - 16-19,144-147
    - 20-23,148-151
    - 24-27,152-155
    - 28-31,156-159
    - 32-35,160-163
    - 36-39,164-167
    - 40-43,168-171
    - 44-47,172-175
    - 48-51,176-179
    - 52-55,180-183
    - 56-59,184-187
    - 60-63,188-191
    - 64-67,192-195
    - 68-71,196-199
    - 72-75,200-203
    - 76-79,204-207
    - 80-83,208-211
    - 84-87,212-215
    - 88-91,216-219
    - 92-95,220-223
    - 96-99,224-227
    - 100-103,228-231
    - 104-107,232-235
    - 108-111,236-239
    - 112-115,240-243
    - 116-119,244-247
    - 120-123,248-251
    - 124-127,252-255
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
  device_name: NVIDIA A100-SXM4-80GB
  device_uuid: GPU-3f509591-db26-6474-253e-fcb29d2b374b
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
  - 0.097164
  - 0.092485
  - 0.091824
  - 0.096453
  - 0.097244
  - 0.092075
  - 0.093056
  - 0.095711
  - 0.098486
  - 0.094007
  - 0.093837
  - 0.094138
  - 0.095431
  - 0.094619
  - 0.092996
  median_ms: 0.094138
  bytes_per_second: 1425755040472.4978
  torch_version: 2.11.0+cu128
  cuda_version: '12.8'
  timestamp: '2026-11-07T20:32:20.348310+00:00'
gpu_assignment:
  role: benchmark
  physical_device_index: 0
  device_uuid: GPU-3f509591-db26-6474-253e-fcb29d2b374b
  cuda_visible_devices: GPU-3f509591-db26-6474-253e-fcb29d2b374b
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
    /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/scripts/benchmark_gpu_public_api.py: 1919e33cc5f2d0208901837f80cb7c0600426462f998be6dd933062d9f6bb345
    /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/scripts/gpu_public_api_ops.py: e88823db3f4a1fe43dc13c4743c9d78ece6af76a968724091af318dd8d7f2528
    scripts/run_gpu_public_api.sh: 788b9fde4f8ca4162921baef1203b2e31bb747ec30327a425de3442cf872df52
    data/results/nvidia-gpu/gpu/nn/20261107_201728/fixtures/manifest.json: 6019f8eb30ec023ad89d15545a6d050a2188b62ec40b3d90c773ad6e58eabea1
    target/release/benchmark_gpu_public_api: 94cf356a747431a654002f07d7e16069b1849c82e215e2fe5d75a94346f4ae17
  reference_numpy_version: 2.4.2
  measurement_cpu:
    policy: 'disabled (GPU_BENCH_CPU_SELECT unset): arms run on the allocation unpinned'
    arms:
      jax-cuda:
        arm: jax-cuda
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-07T20:33:09Z'
      pytorch-cuda:
        arm: pytorch-cuda
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-07T20:32:49Z'
      tenferro-cuda-direct:
        arm: tenferro-cuda-direct
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-07T20:32:31Z'
      tenferro-cuda-eager:
        arm: tenferro-cuda-eager
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-07T20:32:32Z'
  compile:
    case_wall_seconds: 32.437
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
| softplus_f64_n1048576 | tenferro-cuda-direct | softplus_f64_n1048576 untimed 86.2 ms is 274% of the 31.5 ms timed region (budget 20%) |
| softplus_f64_n1048576 | tenferro-cuda-eager | softplus_f64_n1048576 untimed 87.3 ms is 336% of the 26.0 ms timed region (budget 20%) |
| sigmoid_f64_n1048576 | pytorch-cuda | sigmoid_f64_n1048576 untimed 1637.0 ms is 27932% of the 5.9 ms timed region (budget 20%) |
| silu_f64_n1048576 | pytorch-cuda | silu_f64_n1048576 untimed 1510.6 ms is 23027% of the 6.6 ms timed region (budget 20%) |
| softplus_f64_n1048576 | pytorch-cuda | softplus_f64_n1048576 untimed 1503.1 ms is 9992% of the 15.0 ms timed region (budget 20%) |
| gelu_f64_n1048576 | pytorch-cuda | gelu_f64_n1048576 untimed 1486.7 ms is 16862% of the 8.8 ms timed region (budget 20%) |
| gelu_tanh_f64_n1048576 | pytorch-cuda | gelu_tanh_f64_n1048576 untimed 1468.4 ms is 26380% of the 5.6 ms timed region (budget 20%) |
| softmax_f64_n1048576 | pytorch-cuda | softmax_f64_n1048576 untimed 1484.6 ms is 15334% of the 9.7 ms timed region (budget 20%) |
| log_softmax_f64_n1048576 | pytorch-cuda | log_softmax_f64_n1048576 untimed 1460.8 ms is 19171% of the 7.6 ms timed region (budget 20%) |
| layer_norm_f64_n1048576 | pytorch-cuda | layer_norm_f64_n1048576 untimed 1474.4 ms is 8336% of the 17.7 ms timed region (budget 20%) |
| rms_norm_f64_n1048576 | pytorch-cuda | rms_norm_f64_n1048576 untimed 1470.8 ms is 8313% of the 17.7 ms timed region (budget 20%) |
| sigmoid_f64_n1048576 | jax-cuda | sigmoid_f64_n1048576 untimed 1898.8 ms is 7822% of the 24.3 ms timed region (budget 20%) |
| silu_f64_n1048576 | jax-cuda | silu_f64_n1048576 untimed 1558.7 ms is 6416% of the 24.3 ms timed region (budget 20%) |
| softplus_f64_n1048576 | jax-cuda | softplus_f64_n1048576 untimed 1559.9 ms is 6840% of the 22.8 ms timed region (budget 20%) |
| gelu_f64_n1048576 | jax-cuda | gelu_f64_n1048576 untimed 1948.8 ms is 5308% of the 36.7 ms timed region (budget 20%) |
| gelu_tanh_f64_n1048576 | jax-cuda | gelu_tanh_f64_n1048576 untimed 1826.5 ms is 3579% of the 51.0 ms timed region (budget 20%) |
| softmax_f64_n1048576 | jax-cuda | softmax_f64_n1048576 untimed 2425.4 ms is 5134% of the 47.2 ms timed region (budget 20%) |
| log_softmax_f64_n1048576 | jax-cuda | log_softmax_f64_n1048576 untimed 1649.4 ms is 7008% of the 23.5 ms timed region (budget 20%) |
| masked_softmax_f64_n1048576 | jax-cuda | masked_softmax_f64_n1048576 untimed 2098.0 ms is 5382% of the 39.0 ms timed region (budget 20%) |
| masked_log_softmax_f64_n1048576 | jax-cuda | masked_log_softmax_f64_n1048576 untimed 445.6 ms is 1969% of the 22.6 ms timed region (budget 20%) |
| layer_norm_f64_n1048576 | jax-cuda | layer_norm_f64_n1048576 untimed 1592.4 ms is 2607% of the 61.1 ms timed region (budget 20%) |