# gpu/fft: CUDA public API benchmark

Raw run: `data/results/nvidia-gpu/gpu/fft/20261013_133616/`.

Values are median (IQR) **µs per operation**. Samples time batches of declared operations plus device synchronization; retained output memory is limited to 256 MiB. Batch counts and raw durations are preserved in JSONL.

Setup, input transfers, library/plan initialization, JIT compilation, validation downloads and output destruction are outside timing. Intrinsic output allocation is inside timing; `_into` rows instead reuse preallocated destinations across samples. Direct and eager each use one entered session across samples; direct FFT uses cached `FftExecutor`. JAX eager and compiled JIT are separate arms.

All measured output elements are checked against NumPy/SciPy double-precision reference results before and after sampling. Unsupported rows include unavailable APIs and APIs whose internal setup cannot be excluded; these are not performance measurements. Short batches limited by output memory remain identifiable in the sampling table.

| Case | tenferro-cuda-direct | tenferro-cuda-eager | pytorch-cuda | jax-cuda | jax-cuda-jit |
| --- | ---: | ---: | ---: | ---: | ---: |
| fft_c32_n1024 | 54.668 (0.758) | 137.450 (21.328) | 20.358 (0.367) | 1680.556 (274.721) | 90.660 (2.242) |
| fft_c64_n1024 | 66.118 (0.818) | 157.875 (3.201) | 20.823 (0.478) | 1573.035 (33.691) | 75.235 (1.546) |
| ifft_c32_n1024 | 195.464 (3.000) | 325.953 (26.754) | 38.516 (0.604) | 1655.850 (11.491) | 80.050 (3.617) |
| ifft_c64_n1024 | 210.301 (2.529) | 380.225 (11.043) | 41.372 (2.876) | 1618.430 (25.014) | 84.819 (3.876) |
| rfft_f32_n1024 | 84.933 (1.205) | 238.673 (1.953) | 21.822 (0.256) | 1620.170 (29.090) | 72.283 (2.537) |
| rfft_f64_n1024 | 87.233 (9.790) | 244.200 (3.073) | 21.900 (0.471) | 1667.784 (60.934) | 76.290 (1.831) |
| irfft_c32_n1024 | 195.769 (3.788) | 430.739 (12.700) | 61.277 (3.343) | 1666.104 (9.878) | 90.534 (1.227) |
| irfft_c64_n1024 | 215.084 (6.204) | 372.479 (8.541) | 60.712 (0.537) | 1640.465 (18.583) | 93.443 (2.856) |
| fft_c32_n1048576 | 206.541 (2.558) | 268.856 (1.415) | 136.778 (0.287) | 1336.155 (45.464) | 145.150 (1.438) |
| fft_c64_n1048576 | 1453.752 (4.519) | 1468.964 (4.258) | 1390.381 (3.905) | 2222.888 (50.793) | 1569.043 (26.529) |
| ifft_c32_n1048576 | 355.921 (6.124) | 382.598 (13.072) | 188.545 (0.667) | 1342.159 (31.228) | 199.166 (2.395) |
| ifft_c64_n1048576 | 1657.959 (7.268) | 1606.867 (9.400) | 1415.801 (1.234) | 2293.682 (66.643) | 1667.914 (10.607) |
| rfft_f32_n1048576 | 154.934 (2.469) | 219.110 (1.953) | 90.121 (0.129) | 1413.943 (78.037) | 96.817 (0.750) |
| rfft_f64_n1048576 | 860.122 (2.750) | 948.760 (14.438) | 757.983 (1.585) | 1337.880 (27.379) | 893.938 (8.538) |
| irfft_c32_n1048576 | 277.637 (4.095) | 322.897 (5.548) | 142.688 (0.716) | 1371.984 (31.629) | 152.744 (1.602) |
| irfft_c64_n1048576 | 995.459 (3.049) | 1025.049 (12.040) | 878.454 (0.925) | 1450.392 (20.827) | 1033.113 (10.938) |

## Sampling

| Case | Backend | Ops/sample | Samples | Median batch ms | Verification error/tolerance |
| --- | --- | ---: | ---: | ---: | ---: |
| fft_c32_n1024 | tenferro-cuda-direct | 64 | 15 | 3.499 | 0.01405 |
| fft_c64_n1024 | tenferro-cuda-direct | 32 | 15 | 2.116 | 1.351e-05 |
| ifft_c32_n1024 | tenferro-cuda-direct | 16 | 15 | 3.127 | 0.02634 |
| ifft_c64_n1024 | tenferro-cuda-direct | 16 | 15 | 3.365 | 4.043e-08 |
| rfft_f32_n1024 | tenferro-cuda-direct | 32 | 15 | 2.718 | 0.01301 |
| rfft_f64_n1024 | tenferro-cuda-direct | 32 | 15 | 2.791 | 1.518e-05 |
| irfft_c32_n1024 | tenferro-cuda-direct | 16 | 15 | 3.132 | 0.01787 |
| irfft_c64_n1024 | tenferro-cuda-direct | 16 | 15 | 3.441 | 3.809e-08 |
| fft_c32_n1048576 | tenferro-cuda-direct | 8 | 15 | 1.652 | 0.5149 |
| fft_c64_n1048576 | tenferro-cuda-direct | 2 | 15 | 2.908 | 0.01641 |
| ifft_c32_n1048576 | tenferro-cuda-direct | 8 | 15 | 2.847 | 0.633 |
| ifft_c64_n1048576 | tenferro-cuda-direct | 2 | 15 | 3.316 | 4.754e-08 |
| rfft_f32_n1048576 | tenferro-cuda-direct | 16 | 15 | 2.479 | 0.3419 |
| rfft_f64_n1048576 | tenferro-cuda-direct | 4 | 15 | 3.440 | 0.01077 |
| irfft_c32_n1048576 | tenferro-cuda-direct | 8 | 15 | 2.221 | 0.8898 |
| irfft_c64_n1048576 | tenferro-cuda-direct | 4 | 15 | 3.982 | 6.867e-08 |
| fft_c32_n1024 | tenferro-cuda-eager | 32 | 15 | 4.398 | 0.01405 |
| fft_c64_n1024 | tenferro-cuda-eager | 16 | 15 | 2.526 | 1.351e-05 |
| ifft_c32_n1024 | tenferro-cuda-eager | 16 | 15 | 5.215 | 0.02634 |
| ifft_c64_n1024 | tenferro-cuda-eager | 8 | 15 | 3.042 | 4.043e-08 |
| rfft_f32_n1024 | tenferro-cuda-eager | 8 | 15 | 1.909 | 0.01301 |
| rfft_f64_n1024 | tenferro-cuda-eager | 8 | 15 | 1.954 | 1.518e-05 |
| irfft_c32_n1024 | tenferro-cuda-eager | 8 | 15 | 3.446 | 0.01787 |
| irfft_c64_n1024 | tenferro-cuda-eager | 8 | 15 | 2.980 | 3.809e-08 |
| fft_c32_n1048576 | tenferro-cuda-eager | 8 | 15 | 2.151 | 0.5149 |
| fft_c64_n1048576 | tenferro-cuda-eager | 2 | 15 | 2.938 | 0.01641 |
| ifft_c32_n1048576 | tenferro-cuda-eager | 8 | 15 | 3.061 | 0.633 |
| ifft_c64_n1048576 | tenferro-cuda-eager | 2 | 15 | 3.214 | 4.754e-08 |
| rfft_f32_n1048576 | tenferro-cuda-eager | 16 | 15 | 3.506 | 0.3419 |
| rfft_f64_n1048576 | tenferro-cuda-eager | 4 | 15 | 3.795 | 0.01077 |
| irfft_c32_n1048576 | tenferro-cuda-eager | 8 | 15 | 2.583 | 0.8898 |
| irfft_c64_n1048576 | tenferro-cuda-eager | 2 | 15 | 2.050 | 6.867e-08 |
| fft_c32_n1024 | pytorch-cuda | 128 | 15 | 2.606 | 0.01405 |
| fft_c64_n1024 | pytorch-cuda | 128 | 15 | 2.665 | 1.351e-05 |
| ifft_c32_n1024 | pytorch-cuda | 64 | 15 | 2.465 | 0.02634 |
| ifft_c64_n1024 | pytorch-cuda | 64 | 15 | 2.648 | 4.043e-08 |
| rfft_f32_n1024 | pytorch-cuda | 128 | 15 | 2.793 | 0.01301 |
| rfft_f64_n1024 | pytorch-cuda | 128 | 15 | 2.803 | 1.518e-05 |
| irfft_c32_n1024 | pytorch-cuda | 64 | 15 | 3.922 | 0.01787 |
| irfft_c64_n1024 | pytorch-cuda | 64 | 15 | 3.886 | 3.809e-08 |
| fft_c32_n1048576 | pytorch-cuda | 16 | 15 | 2.188 | 0.5149 |
| fft_c64_n1048576 | pytorch-cuda | 2 | 15 | 2.781 | 0.01641 |
| ifft_c32_n1048576 | pytorch-cuda | 16 | 15 | 3.017 | 0.633 |
| ifft_c64_n1048576 | pytorch-cuda | 2 | 15 | 2.832 | 4.754e-08 |
| rfft_f32_n1048576 | pytorch-cuda | 32 | 15 | 2.884 | 0.3419 |
| rfft_f64_n1048576 | pytorch-cuda | 4 | 15 | 3.032 | 0.01077 |
| irfft_c32_n1048576 | pytorch-cuda | 16 | 15 | 2.283 | 0.8898 |
| irfft_c64_n1048576 | pytorch-cuda | 4 | 15 | 3.514 | 6.867e-08 |
| fft_c32_n1024 | jax-cuda | 2 | 15 | 3.361 | 0.01405 |
| fft_c64_n1024 | jax-cuda | 2 | 15 | 3.146 | 1.351e-05 |
| ifft_c32_n1024 | jax-cuda | 2 | 15 | 3.312 | 0.02634 |
| ifft_c64_n1024 | jax-cuda | 2 | 15 | 3.237 | 4.043e-08 |
| rfft_f32_n1024 | jax-cuda | 2 | 15 | 3.240 | 0.01301 |
| rfft_f64_n1024 | jax-cuda | 2 | 15 | 3.336 | 1.518e-05 |
| irfft_c32_n1024 | jax-cuda | 2 | 15 | 3.332 | 0.01787 |
| irfft_c64_n1024 | jax-cuda | 2 | 15 | 3.281 | 3.809e-08 |
| fft_c32_n1048576 | jax-cuda | 2 | 15 | 2.672 | 0.5149 |
| fft_c64_n1048576 | jax-cuda | 1 | 15 | 2.223 | 0.01641 |
| ifft_c32_n1048576 | jax-cuda | 2 | 15 | 2.684 | 0.633 |
| ifft_c64_n1048576 | jax-cuda | 1 | 15 | 2.294 | 4.754e-08 |
| rfft_f32_n1048576 | jax-cuda | 2 | 15 | 2.828 | 0.3419 |
| rfft_f64_n1048576 | jax-cuda | 2 | 15 | 2.676 | 0.01077 |
| irfft_c32_n1048576 | jax-cuda | 2 | 15 | 2.744 | 0.8898 |
| irfft_c64_n1048576 | jax-cuda | 2 | 15 | 2.901 | 6.867e-08 |
| fft_c32_n1024 | jax-cuda-jit | 32 | 15 | 2.901 | 0.01405 |
| fft_c64_n1024 | jax-cuda-jit | 32 | 15 | 2.408 | 1.351e-05 |
| ifft_c32_n1024 | jax-cuda-jit | 32 | 15 | 2.562 | 0.02634 |
| ifft_c64_n1024 | jax-cuda-jit | 32 | 15 | 2.714 | 4.043e-08 |
| rfft_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.313 | 0.01301 |
| rfft_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.441 | 1.518e-05 |
| irfft_c32_n1024 | jax-cuda-jit | 32 | 15 | 2.897 | 0.01787 |
| irfft_c64_n1024 | jax-cuda-jit | 32 | 15 | 2.990 | 3.809e-08 |
| fft_c32_n1048576 | jax-cuda-jit | 16 | 15 | 2.322 | 0.5149 |
| fft_c64_n1048576 | jax-cuda-jit | 2 | 15 | 3.138 | 0.01641 |
| ifft_c32_n1048576 | jax-cuda-jit | 16 | 15 | 3.187 | 0.633 |
| ifft_c64_n1048576 | jax-cuda-jit | 2 | 15 | 3.336 | 4.754e-08 |
| rfft_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.098 | 0.3419 |
| rfft_f64_n1048576 | jax-cuda-jit | 4 | 15 | 3.576 | 0.01077 |
| irfft_c32_n1048576 | jax-cuda-jit | 16 | 15 | 2.444 | 0.8898 |
| irfft_c64_n1048576 | jax-cuda-jit | 2 | 15 | 2.066 | 6.867e-08 |

## Run metadata

```yaml
schema_version: 1
target_profile: nvidia-gpu
suite_id: gpu/fft
suite_file: benchmarks/gpu/fft.yaml
timestamp: '2026-10-13T13:36:17Z'
tenferro_rs:
  path: extern/tenferro-rs
  commit: dfbdf6b66d40162f4731da116ec42ed034da0dce
  dirty: false
  features:
  - cuda
  resolved_path: /workspaces/tenferro-benchmark/extern/tenferro-rs
harness:
  path: /workspaces/tenferro-benchmark
  commit: 2858dcda46699788bc1df84d923a95ed93ef6469
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
    ./scripts/run_gpu_public_api.sh fft
  devcontainer_config: .devcontainer/cuda/devcontainer.json
  expected_cases: 16
  selected_cases: 16
  sha256:
    .devcontainer/cuda/devcontainer.json: 76569d9e898967d094988483a7adf6ab96eb426793db7248545d50e155f19cf1
    .devcontainer/cuda/Dockerfile: 1cadf2f890f01187553bab0f136dc817f8964eff6a35b1806b1448a7383d21de
    src/bin/benchmark_gpu_public_api.rs: 249184323d09433d1fbf401284e39fa797c0252dceb20351bbcc151551d1b32c
    src/bin/benchmark_gpu_public_api/runtime_ops.rs: d7b64dccce85c804b8f67543fefca89836f045702bdcf15b5441c7bb8543eb21
    src/bin/benchmark_gpu_public_api/structural.rs: 382fb5ffb6d86ac1a4a66723376808e9c95c8a8731570a5617f2ffed2a6e40a7
    src/bin/benchmark_gpu_public_api/linalg.rs: 84162923fa0d0772c88ec9c4eab169ad3a75e5d16ff76961d1541670dd8565c1
    src/bin/benchmark_gpu_public_api/reuse.rs: ad72b565373904275128d7beb8673bada08d922e8f310be6cedb0d2d8187898f
    /workspaces/tenferro-benchmark/scripts/benchmark_gpu_public_api.py: 4d05b9facc0b7e5a3ff31146fc42f98d680010264c6606a3154e9da2dfa6a0cc
    /workspaces/tenferro-benchmark/scripts/gpu_public_api_ops.py: 71cb26510a35647e7aadc46827ca00d06a06e283eacfcba8ebe93d2e8e8add86
    scripts/run_gpu_public_api.sh: 3516e06358fc3c2e36a066067cdffcab85673c7e78bf9a244dde318d109dd772
    data/results/nvidia-gpu/gpu/fft/20261013_133616/fixtures/manifest.json: 6f14a232def70d9c1acea54911211620ba8165c34541f631b0f76c95792ea041
    target/release/benchmark_gpu_public_api: b505a2bdb3d0c5a7bffd1390af659a154753ea998ec68becd4779e137c9c7150
  reference_numpy_version: 2.4.2
```
