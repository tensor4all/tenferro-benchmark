# gpu/lstsq: CUDA public API benchmark

Measured cells: **0**. Unsupported cells: **40**, including **8** numerically verified but timing-excluded cells. Unsupported cells do not count as measured coverage.

Raw run: `data/results/nvidia-gpu/gpu/lstsq/20261013_174738/`.

Values are median (IQR) **µs per operation**. Samples time batches of declared operations plus device synchronization; retained output memory is limited to 256 MiB. Batch counts and raw durations are preserved in JSONL.

Setup, input transfers, library/plan initialization, JIT compilation, validation downloads and output destruction are outside timing. Intrinsic output allocation is inside timing; output-reuse rows instead use preallocated destinations across samples. Accumulating destinations are restored from prepared GPU seeds and synchronized before every batch clock, including initialization and calibration. Direct and eager each use one entered session across samples; direct FFT uses cached `FftExecutor`. JAX eager and compiled JIT are separate arms.

All measured output elements are checked against NumPy/SciPy double-precision reference results before and after sampling. Unsupported rows include unavailable APIs and APIs whose internal setup cannot be excluded; these are not performance measurements. Short batches limited by output memory remain identifiable in the sampling table.

| Case | tenferro-cuda-direct | tenferro-cuda-eager | pytorch-cuda | jax-cuda | jax-cuda-jit |
| --- | ---: | ---: | ---: | ---: | ---: |
| lstsq_tall_f32_s32_rhs8 | unsupported | unsupported | unsupported | unsupported | unsupported |
| lstsq_tall_f64_s32_rhs8 | unsupported | unsupported | unsupported | unsupported | unsupported |
| lstsq_tall_c32_s32_rhs8 | unsupported | unsupported | unsupported | unsupported | unsupported |
| lstsq_tall_c64_s32_rhs8 | unsupported | unsupported | unsupported | unsupported | unsupported |
| lstsq_tall_f32_s256_rhs8 | unsupported | unsupported | unsupported | unsupported | unsupported |
| lstsq_tall_f64_s256_rhs8 | unsupported | unsupported | unsupported | unsupported | unsupported |
| lstsq_tall_c32_s256_rhs8 | unsupported | unsupported | unsupported | unsupported | unsupported |
| lstsq_tall_c64_s256_rhs8 | unsupported | unsupported | unsupported | unsupported | unsupported |

## Sampling

| Case | Backend | Ops/sample | Samples | Median batch ms | Verification error/tolerance |
| --- | --- | ---: | ---: | ---: | ---: |
| lstsq_tall_f32_s32_rhs8 | tenferro-cuda-direct | — | — | — | Current TensorLinalgExt has no direct lstsq operation; only the public eager composite is available. |
| lstsq_tall_f64_s32_rhs8 | tenferro-cuda-direct | — | — | — | Current TensorLinalgExt has no direct lstsq operation; only the public eager composite is available. |
| lstsq_tall_c32_s32_rhs8 | tenferro-cuda-direct | — | — | — | Current TensorLinalgExt has no direct lstsq operation; only the public eager composite is available. |
| lstsq_tall_c64_s32_rhs8 | tenferro-cuda-direct | — | — | — | Current TensorLinalgExt has no direct lstsq operation; only the public eager composite is available. |
| lstsq_tall_f32_s256_rhs8 | tenferro-cuda-direct | — | — | — | Current TensorLinalgExt has no direct lstsq operation; only the public eager composite is available. |
| lstsq_tall_f64_s256_rhs8 | tenferro-cuda-direct | — | — | — | Current TensorLinalgExt has no direct lstsq operation; only the public eager composite is available. |
| lstsq_tall_c32_s256_rhs8 | tenferro-cuda-direct | — | — | — | Current TensorLinalgExt has no direct lstsq operation; only the public eager composite is available. |
| lstsq_tall_c64_s256_rhs8 | tenferro-cuda-direct | — | — | — | Current TensorLinalgExt has no direct lstsq operation; only the public eager composite is available. |
| lstsq_tall_f32_s32_rhs8 | tenferro-cuda-eager | — | — | — | Public EagerSession lstsq invokes QR whose CUDA Q application uploads scalar operands internally; no public interface can hoist this input preparation out of standard operation timing. |
| lstsq_tall_f64_s32_rhs8 | tenferro-cuda-eager | — | — | — | Public EagerSession lstsq invokes QR whose CUDA Q application uploads scalar operands internally; no public interface can hoist this input preparation out of standard operation timing. |
| lstsq_tall_c32_s32_rhs8 | tenferro-cuda-eager | — | — | — | Public EagerSession lstsq invokes QR whose CUDA Q application uploads scalar operands internally; no public interface can hoist this input preparation out of standard operation timing. |
| lstsq_tall_c64_s32_rhs8 | tenferro-cuda-eager | — | — | — | Public EagerSession lstsq invokes QR whose CUDA Q application uploads scalar operands internally; no public interface can hoist this input preparation out of standard operation timing. |
| lstsq_tall_f32_s256_rhs8 | tenferro-cuda-eager | — | — | — | Public EagerSession lstsq invokes QR whose CUDA Q application uploads scalar operands internally; no public interface can hoist this input preparation out of standard operation timing. |
| lstsq_tall_f64_s256_rhs8 | tenferro-cuda-eager | — | — | — | Public EagerSession lstsq invokes QR whose CUDA Q application uploads scalar operands internally; no public interface can hoist this input preparation out of standard operation timing. |
| lstsq_tall_c32_s256_rhs8 | tenferro-cuda-eager | — | — | — | Public EagerSession lstsq invokes QR whose CUDA Q application uploads scalar operands internally; no public interface can hoist this input preparation out of standard operation timing. |
| lstsq_tall_c64_s256_rhs8 | tenferro-cuda-eager | — | — | — | Public EagerSession lstsq invokes QR whose CUDA Q application uploads scalar operands internally; no public interface can hoist this input preparation out of standard operation timing. |
| lstsq_tall_f32_s32_rhs8 | pytorch-cuda | — | — | — | No verified single public solution-only least-squares API with the same requested outputs; native candidate APIs also return residual/rank/singular-value outputs. |
| lstsq_tall_f64_s32_rhs8 | pytorch-cuda | — | — | — | No verified single public solution-only least-squares API with the same requested outputs; native candidate APIs also return residual/rank/singular-value outputs. |
| lstsq_tall_c32_s32_rhs8 | pytorch-cuda | — | — | — | No verified single public solution-only least-squares API with the same requested outputs; native candidate APIs also return residual/rank/singular-value outputs. |
| lstsq_tall_c64_s32_rhs8 | pytorch-cuda | — | — | — | No verified single public solution-only least-squares API with the same requested outputs; native candidate APIs also return residual/rank/singular-value outputs. |
| lstsq_tall_f32_s256_rhs8 | pytorch-cuda | — | — | — | No verified single public solution-only least-squares API with the same requested outputs; native candidate APIs also return residual/rank/singular-value outputs. |
| lstsq_tall_f64_s256_rhs8 | pytorch-cuda | — | — | — | No verified single public solution-only least-squares API with the same requested outputs; native candidate APIs also return residual/rank/singular-value outputs. |
| lstsq_tall_c32_s256_rhs8 | pytorch-cuda | — | — | — | No verified single public solution-only least-squares API with the same requested outputs; native candidate APIs also return residual/rank/singular-value outputs. |
| lstsq_tall_c64_s256_rhs8 | pytorch-cuda | — | — | — | No verified single public solution-only least-squares API with the same requested outputs; native candidate APIs also return residual/rank/singular-value outputs. |
| lstsq_tall_f32_s32_rhs8 | jax-cuda | — | — | — | No verified single public solution-only least-squares API with the same requested outputs; native candidate APIs also return residual/rank/singular-value outputs. |
| lstsq_tall_f64_s32_rhs8 | jax-cuda | — | — | — | No verified single public solution-only least-squares API with the same requested outputs; native candidate APIs also return residual/rank/singular-value outputs. |
| lstsq_tall_c32_s32_rhs8 | jax-cuda | — | — | — | No verified single public solution-only least-squares API with the same requested outputs; native candidate APIs also return residual/rank/singular-value outputs. |
| lstsq_tall_c64_s32_rhs8 | jax-cuda | — | — | — | No verified single public solution-only least-squares API with the same requested outputs; native candidate APIs also return residual/rank/singular-value outputs. |
| lstsq_tall_f32_s256_rhs8 | jax-cuda | — | — | — | No verified single public solution-only least-squares API with the same requested outputs; native candidate APIs also return residual/rank/singular-value outputs. |
| lstsq_tall_f64_s256_rhs8 | jax-cuda | — | — | — | No verified single public solution-only least-squares API with the same requested outputs; native candidate APIs also return residual/rank/singular-value outputs. |
| lstsq_tall_c32_s256_rhs8 | jax-cuda | — | — | — | No verified single public solution-only least-squares API with the same requested outputs; native candidate APIs also return residual/rank/singular-value outputs. |
| lstsq_tall_c64_s256_rhs8 | jax-cuda | — | — | — | No verified single public solution-only least-squares API with the same requested outputs; native candidate APIs also return residual/rank/singular-value outputs. |
| lstsq_tall_f32_s32_rhs8 | jax-cuda-jit | — | — | — | No verified single public solution-only least-squares API with the same requested outputs; native candidate APIs also return residual/rank/singular-value outputs. |
| lstsq_tall_f64_s32_rhs8 | jax-cuda-jit | — | — | — | No verified single public solution-only least-squares API with the same requested outputs; native candidate APIs also return residual/rank/singular-value outputs. |
| lstsq_tall_c32_s32_rhs8 | jax-cuda-jit | — | — | — | No verified single public solution-only least-squares API with the same requested outputs; native candidate APIs also return residual/rank/singular-value outputs. |
| lstsq_tall_c64_s32_rhs8 | jax-cuda-jit | — | — | — | No verified single public solution-only least-squares API with the same requested outputs; native candidate APIs also return residual/rank/singular-value outputs. |
| lstsq_tall_f32_s256_rhs8 | jax-cuda-jit | — | — | — | No verified single public solution-only least-squares API with the same requested outputs; native candidate APIs also return residual/rank/singular-value outputs. |
| lstsq_tall_f64_s256_rhs8 | jax-cuda-jit | — | — | — | No verified single public solution-only least-squares API with the same requested outputs; native candidate APIs also return residual/rank/singular-value outputs. |
| lstsq_tall_c32_s256_rhs8 | jax-cuda-jit | — | — | — | No verified single public solution-only least-squares API with the same requested outputs; native candidate APIs also return residual/rank/singular-value outputs. |
| lstsq_tall_c64_s256_rhs8 | jax-cuda-jit | — | — | — | No verified single public solution-only least-squares API with the same requested outputs; native candidate APIs also return residual/rank/singular-value outputs. |

## Run metadata

```yaml
schema_version: 1
target_profile: nvidia-gpu
suite_id: gpu/lstsq
suite_file: benchmarks/gpu/lstsq.yaml
timestamp: '2026-10-13T17:47:39Z'
tenferro_rs:
  path: extern/tenferro-rs
  commit: dfbdf6b66d40162f4731da116ec42ed034da0dce
  dirty: false
  features:
  - cuda
  resolved_path: /workspaces/tenferro-benchmark/extern/tenferro-rs
harness:
  path: /workspaces/tenferro-benchmark
  commit: 7829c17d2c169731f540021a31e7ba30a81a9380
  dirty: false
environment:
  hostname: 737f43342de3
  os: Linux-6.8.0-146-generic-x86_64-with-glibc2.39
  arch: x86_64
  cpu: x86_64
  env:
    OMP_NUM_THREADS: '1'
    RAYON_NUM_THREADS: '1'
    BENCHMARK_TARGET_PROFILE: nvidia-gpu
    OPENBLAS_ROOT: /opt/openblas
    OPENBLAS_NUM_THREADS: '1'
    CUDA_HOME: /usr/local/cuda
    CUDA_PATH: /usr/local/cuda
    TENFERRO_CUTENSOR_PATH: /usr/lib/x86_64-linux-gnu/libcutensor/12/libcutensor.so.2
    BENCH_RUNS: '15'
    BENCH_WARMUPS: '3'
    USE_CUDA: '1'
timing_policy:
  version: 2
  scope: operation execution and completion synchronization
  setup: fixtures, H2D, runtime/session/handles, plans and compilation outside every
    timer
  cleanup: returned outputs retained until after timer stops
  minimum_untimed_priming_runs: 1
  short_operation_sampling: adaptive batches target 2 ms, at most 4096 operations
    and 256 MiB retained output; output-reuse destinations prepared before timing
  trace_execution: 'tenferro trace excluded: run_prepared cannot borrow the entered
    session; JAX JIT compilation outside timing'
  setup_diagnostics: false
blas:
  implementation: none
python_backends:
  pytorch:
    available: true
    version: 2.12.0+cu130
    file: /workspaces/tenferro-benchmark/.venv/lib/python3.12/site-packages/torch/__init__.py
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
    library: /workspaces/tenferro-benchmark/.venv/lib/python3.12/site-packages/torch/lib/libtorch_cpu.so
    linked_libraries:
    - /workspaces/tenferro-benchmark/.venv/lib/python3.12/site-packages/torch/lib/libgomp.so.1
  jax:
    available: true
    version: 0.10.1
    jaxlib_version: 0.10.1
    backend: gpu
    file: /workspaces/tenferro-benchmark/.venv/lib/python3.12/site-packages/jaxlib/__init__.py
    provider: xla_gpu
    dot_backend: xla_gpu
    lapack_provider: none_detected
    library: /workspaces/tenferro-benchmark/.venv/lib/python3.12/site-packages/jaxlib/_ifrt_proxy.so
    linked_libraries: []
julia:
  available: false
  reason: julia not found on PATH
cuda:
  device_ordinal: 0
  device_name: NVIDIA GeForce RTX 3060
  device_uuid: GPU-a78d5217-eba3-72c2-3d5b-8ae496ebbc2e
  device_memory_bytes: 12884901888
  driver_version: 580.178.04
  runtime_version: '12.9'
  cuda_version: '13.0'
  cudnn_version: '92700'
gpu_assignment:
  role: benchmark
  physical_device_index: 0
  device_uuid: GPU-a78d5217-eba3-72c2-3d5b-8ae496ebbc2e
  cuda_visible_devices: GPU-a78d5217-eba3-72c2-3d5b-8ae496ebbc2e
  process_device_ordinal: 0
  verification_only: false
collection:
  command: BENCHMARK_TARGET_PROFILE=nvidia-gpu GPU_BENCH_DEVICE=0 BENCH_RUNS=15 BENCH_WARMUPS=3
    ./scripts/run_gpu_public_api.sh lstsq
  devcontainer_config: .devcontainer/cuda/devcontainer.json
  expected_cases: 8
  selected_cases: 8
  sha256:
    .devcontainer/cuda/devcontainer.json: 76569d9e898967d094988483a7adf6ab96eb426793db7248545d50e155f19cf1
    .devcontainer/cuda/Dockerfile: 1cadf2f890f01187553bab0f136dc817f8964eff6a35b1806b1448a7383d21de
    src/bin/benchmark_gpu_public_api.rs: 897fe5518ea92f3922c9bd33f306f9e6441de45204bcb5e4338aa5ed0ab01df6
    src/bin/benchmark_gpu_public_api/runtime_ops.rs: d7b64dccce85c804b8f67543fefca89836f045702bdcf15b5441c7bb8543eb21
    src/bin/benchmark_gpu_public_api/multi.rs: 1b477fef78260cf28aa31e515d57ab411f6241b91ffb30d625939538fdbc6bab
    src/bin/benchmark_gpu_public_api/read_layout.rs: bd7ce4a42576d1937a7048b7b002cac4838a7cd3c53628bc9aa6aebb0c5c1128
    src/bin/benchmark_gpu_public_api/structural.rs: 8121b12b5fed3b0acbc71e2218856173d53a5acc93092b93c7fd3309d4f45962
    src/bin/benchmark_gpu_public_api/linalg.rs: 19cd0d5e23b63d2b525611bffa02a62dbe4c4ed77dd2604ff379fa80130e392e
    src/bin/benchmark_gpu_public_api/state.rs: af8e9c9b4eb272af9a751d2a0aef9e1f8ff32c5efbd3ffffcf5f76e1e2c9704a
    src/bin/benchmark_gpu_public_api/reuse.rs: a305c36801d5bee0d9843ab6d224918d67b5330006ef997a37d12a36bc95ef7f
    /workspaces/tenferro-benchmark/scripts/benchmark_gpu_public_api.py: bbf982ad1b55067bf8dabc8676aa8ffe2fe5a1ff769f979b66eca4e2d5eba6e6
    /workspaces/tenferro-benchmark/scripts/gpu_public_api_ops.py: 8f9f7dae4e87904ae8ae03f32d943f5b09cbeb3679e28bace12ac7f91cbfee33
    scripts/run_gpu_public_api.sh: dd5e4cc47e85fcb5afdc56d460136b2a399a6a7d6a62005de8ba94afb0119ecc
    data/results/nvidia-gpu/gpu/lstsq/20261013_174738/fixtures/manifest.json: 7595bb050518b743d83b4a89ec25c7874a0745e7313761b7f6a6a604747a08db
    target/release/benchmark_gpu_public_api: 9b59be2d494b2b0c9421591356e5413a1d306a675e0a472c88b49aceb9d00621
  reference_numpy_version: 2.4.2
```
