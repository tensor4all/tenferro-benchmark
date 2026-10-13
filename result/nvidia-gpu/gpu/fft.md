# gpu/fft: CUDA public API benchmark

Raw run: `data/results/nvidia-gpu/gpu/fft/20261013_101241/`.

Values are median (IQR) **µs per operation**. Samples time batches of declared operations plus device synchronization; retained output memory is limited to 256 MiB. Batch counts and raw durations are preserved in JSONL.

Setup, input transfers, library/plan initialization, JIT compilation, validation downloads and output destruction are outside timing. Intrinsic output allocation is inside timing; `_into` rows instead reuse preallocated destinations across samples. Direct and eager each use one entered session across samples; direct FFT uses cached `FftExecutor`. JAX eager and compiled JIT are separate arms.

All measured output elements are checked against NumPy/SciPy double-precision reference results before and after sampling. Unsupported rows include unavailable APIs and APIs whose internal setup cannot be excluded; these are not performance measurements. Short batches limited by output memory remain identifiable in the sampling table.

| Case | tenferro-cuda-direct | tenferro-cuda-eager | pytorch-cuda | jax-cuda | jax-cuda-jit |
| --- | ---: | ---: | ---: | ---: | ---: |
| fft_c32_n1024 | 59.930 (3.969) | 119.448 (12.099) | 19.234 (0.159) | 1717.178 (15.762) | 93.870 (4.510) |
| fft_c64_n1024 | 65.377 (1.201) | 133.718 (5.108) | 19.311 (0.177) | 1324.043 (47.685) | 76.574 (3.699) |
| ifft_c32_n1024 | 145.813 (7.201) | 276.932 (5.181) | 36.419 (0.827) | 1281.086 (29.784) | 78.744 (1.977) |
| ifft_c64_n1024 | 183.675 (5.063) | 353.444 (36.347) | 37.098 (0.392) | 1324.200 (23.475) | 82.710 (2.743) |
| rfft_f32_n1024 | 85.733 (2.804) | 210.074 (6.658) | 20.984 (0.120) | 1254.760 (22.225) | 72.132 (3.170) |
| rfft_f64_n1024 | 79.605 (1.849) | 205.973 (5.072) | 22.112 (1.022) | 1284.644 (30.381) | 75.052 (1.795) |
| irfft_c32_n1024 | 195.748 (6.499) | 356.069 (10.310) | 57.547 (0.504) | 1342.056 (48.127) | 107.537 (0.797) |
| irfft_c64_n1024 | 212.811 (6.536) | 357.297 (14.808) | 57.259 (0.483) | 1337.714 (31.560) | 105.970 (0.270) |
| fft_c32_n1048576 | 220.810 (6.715) | 328.408 (5.130) | 136.650 (0.401) | 1431.358 (26.591) | 147.448 (3.017) |
| fft_c64_n1048576 | 1576.406 (7.675) | 1629.492 (4.407) | 1390.619 (2.206) | 2233.170 (57.444) | 1582.531 (12.865) |
| ifft_c32_n1048576 | 444.583 (10.883) | 506.813 (6.136) | 188.578 (0.197) | 1352.459 (26.574) | 201.743 (3.056) |
| ifft_c64_n1048576 | 1880.793 (11.741) | 1887.643 (12.618) | 1406.236 (2.716) | 2365.144 (49.416) | 1687.375 (23.507) |
| rfft_f32_n1048576 | 191.642 (1.822) | 254.498 (1.538) | 90.031 (0.270) | 1383.299 (74.953) | 96.632 (1.287) |
| rfft_f64_n1048576 | 967.442 (9.584) | 977.056 (36.227) | 756.770 (0.795) | 1469.574 (75.881) | 893.578 (9.919) |
| irfft_c32_n1048576 | 358.567 (22.008) | 387.329 (13.561) | 142.174 (0.154) | 1425.134 (15.392) | 152.696 (2.481) |
| irfft_c64_n1048576 | 1232.523 (14.655) | 1297.332 (21.550) | 878.088 (0.628) | 1517.361 (28.190) | 1085.271 (15.695) |

## Sampling

| Case | Backend | Ops/sample | Samples | Median batch ms | Verification error/tolerance |
| --- | --- | ---: | ---: | ---: | ---: |
| fft_c32_n1024 | tenferro-cuda-direct | 32 | 15 | 1.918 | 0.01405 |
| fft_c64_n1024 | tenferro-cuda-direct | 32 | 15 | 2.092 | 1.351e-05 |
| ifft_c32_n1024 | tenferro-cuda-direct | 32 | 15 | 4.666 | 0.02634 |
| ifft_c64_n1024 | tenferro-cuda-direct | 16 | 15 | 2.939 | 4.043e-08 |
| rfft_f32_n1024 | tenferro-cuda-direct | 32 | 15 | 2.743 | 0.01301 |
| rfft_f64_n1024 | tenferro-cuda-direct | 32 | 15 | 2.547 | 1.518e-05 |
| irfft_c32_n1024 | tenferro-cuda-direct | 16 | 15 | 3.132 | 0.01787 |
| irfft_c64_n1024 | tenferro-cuda-direct | 16 | 15 | 3.405 | 3.809e-08 |
| fft_c32_n1048576 | tenferro-cuda-direct | 16 | 15 | 3.533 | 0.5149 |
| fft_c64_n1048576 | tenferro-cuda-direct | 2 | 15 | 3.153 | 0.01641 |
| ifft_c32_n1048576 | tenferro-cuda-direct | 8 | 15 | 3.557 | 0.633 |
| ifft_c64_n1048576 | tenferro-cuda-direct | 2 | 15 | 3.762 | 4.754e-08 |
| rfft_f32_n1048576 | tenferro-cuda-direct | 16 | 15 | 3.066 | 0.3419 |
| rfft_f64_n1048576 | tenferro-cuda-direct | 4 | 15 | 3.870 | 0.01077 |
| irfft_c32_n1048576 | tenferro-cuda-direct | 8 | 15 | 2.869 | 0.8898 |
| irfft_c64_n1048576 | tenferro-cuda-direct | 2 | 15 | 2.465 | 6.867e-08 |
| fft_c32_n1024 | tenferro-cuda-eager | 32 | 15 | 3.822 | 0.01405 |
| fft_c64_n1024 | tenferro-cuda-eager | 16 | 15 | 2.139 | 1.351e-05 |
| ifft_c32_n1024 | tenferro-cuda-eager | 8 | 15 | 2.215 | 0.02634 |
| ifft_c64_n1024 | tenferro-cuda-eager | 8 | 15 | 2.828 | 4.043e-08 |
| rfft_f32_n1024 | tenferro-cuda-eager | 16 | 15 | 3.361 | 0.01301 |
| rfft_f64_n1024 | tenferro-cuda-eager | 16 | 15 | 3.296 | 1.518e-05 |
| irfft_c32_n1024 | tenferro-cuda-eager | 8 | 15 | 2.849 | 0.01787 |
| irfft_c64_n1024 | tenferro-cuda-eager | 8 | 15 | 2.858 | 3.809e-08 |
| fft_c32_n1048576 | tenferro-cuda-eager | 8 | 15 | 2.627 | 0.5149 |
| fft_c64_n1048576 | tenferro-cuda-eager | 2 | 15 | 3.259 | 0.01641 |
| ifft_c32_n1048576 | tenferro-cuda-eager | 4 | 15 | 2.027 | 0.633 |
| ifft_c64_n1048576 | tenferro-cuda-eager | 2 | 15 | 3.775 | 4.754e-08 |
| rfft_f32_n1048576 | tenferro-cuda-eager | 8 | 15 | 2.036 | 0.3419 |
| rfft_f64_n1048576 | tenferro-cuda-eager | 4 | 15 | 3.908 | 0.01077 |
| irfft_c32_n1048576 | tenferro-cuda-eager | 8 | 15 | 3.099 | 0.8898 |
| irfft_c64_n1048576 | tenferro-cuda-eager | 2 | 15 | 2.595 | 6.867e-08 |
| fft_c32_n1024 | pytorch-cuda | 128 | 15 | 2.462 | 0.01405 |
| fft_c64_n1024 | pytorch-cuda | 128 | 15 | 2.472 | 1.351e-05 |
| ifft_c32_n1024 | pytorch-cuda | 64 | 15 | 2.331 | 0.02634 |
| ifft_c64_n1024 | pytorch-cuda | 64 | 15 | 2.374 | 4.043e-08 |
| rfft_f32_n1024 | pytorch-cuda | 128 | 15 | 2.686 | 0.01301 |
| rfft_f64_n1024 | pytorch-cuda | 128 | 15 | 2.830 | 1.518e-05 |
| irfft_c32_n1024 | pytorch-cuda | 64 | 15 | 3.683 | 0.01787 |
| irfft_c64_n1024 | pytorch-cuda | 64 | 15 | 3.665 | 3.809e-08 |
| fft_c32_n1048576 | pytorch-cuda | 16 | 15 | 2.186 | 0.5149 |
| fft_c64_n1048576 | pytorch-cuda | 2 | 15 | 2.781 | 0.01641 |
| ifft_c32_n1048576 | pytorch-cuda | 16 | 15 | 3.017 | 0.633 |
| ifft_c64_n1048576 | pytorch-cuda | 2 | 15 | 2.812 | 4.754e-08 |
| rfft_f32_n1048576 | pytorch-cuda | 32 | 15 | 2.881 | 0.3419 |
| rfft_f64_n1048576 | pytorch-cuda | 4 | 15 | 3.027 | 0.01077 |
| irfft_c32_n1048576 | pytorch-cuda | 16 | 15 | 2.275 | 0.8898 |
| irfft_c64_n1048576 | pytorch-cuda | 4 | 15 | 3.512 | 6.867e-08 |
| fft_c32_n1024 | jax-cuda | 2 | 15 | 3.434 | 0.01405 |
| fft_c64_n1024 | jax-cuda | 2 | 15 | 2.648 | 1.351e-05 |
| ifft_c32_n1024 | jax-cuda | 2 | 15 | 2.562 | 0.02634 |
| ifft_c64_n1024 | jax-cuda | 2 | 15 | 2.648 | 4.043e-08 |
| rfft_f32_n1024 | jax-cuda | 2 | 15 | 2.510 | 0.01301 |
| rfft_f64_n1024 | jax-cuda | 2 | 15 | 2.569 | 1.518e-05 |
| irfft_c32_n1024 | jax-cuda | 2 | 15 | 2.684 | 0.01787 |
| irfft_c64_n1024 | jax-cuda | 2 | 15 | 2.675 | 3.809e-08 |
| fft_c32_n1048576 | jax-cuda | 2 | 15 | 2.863 | 0.5149 |
| fft_c64_n1048576 | jax-cuda | 1 | 15 | 2.233 | 0.01641 |
| ifft_c32_n1048576 | jax-cuda | 2 | 15 | 2.705 | 0.633 |
| ifft_c64_n1048576 | jax-cuda | 1 | 15 | 2.365 | 4.754e-08 |
| rfft_f32_n1048576 | jax-cuda | 2 | 15 | 2.767 | 0.3419 |
| rfft_f64_n1048576 | jax-cuda | 2 | 15 | 2.939 | 0.01077 |
| irfft_c32_n1048576 | jax-cuda | 2 | 15 | 2.850 | 0.8898 |
| irfft_c64_n1048576 | jax-cuda | 2 | 15 | 3.035 | 6.867e-08 |
| fft_c32_n1024 | jax-cuda-jit | 32 | 15 | 3.004 | 0.01405 |
| fft_c64_n1024 | jax-cuda-jit | 32 | 15 | 2.450 | 1.351e-05 |
| ifft_c32_n1024 | jax-cuda-jit | 32 | 15 | 2.520 | 0.02634 |
| ifft_c64_n1024 | jax-cuda-jit | 32 | 15 | 2.647 | 4.043e-08 |
| rfft_f32_n1024 | jax-cuda-jit | 32 | 15 | 2.308 | 0.01301 |
| rfft_f64_n1024 | jax-cuda-jit | 32 | 15 | 2.402 | 1.518e-05 |
| irfft_c32_n1024 | jax-cuda-jit | 32 | 15 | 3.441 | 0.01787 |
| irfft_c64_n1024 | jax-cuda-jit | 32 | 15 | 3.391 | 3.809e-08 |
| fft_c32_n1048576 | jax-cuda-jit | 16 | 15 | 2.359 | 0.5149 |
| fft_c64_n1048576 | jax-cuda-jit | 2 | 15 | 3.165 | 0.01641 |
| ifft_c32_n1048576 | jax-cuda-jit | 16 | 15 | 3.228 | 0.633 |
| ifft_c64_n1048576 | jax-cuda-jit | 2 | 15 | 3.375 | 4.754e-08 |
| rfft_f32_n1048576 | jax-cuda-jit | 32 | 15 | 3.092 | 0.3419 |
| rfft_f64_n1048576 | jax-cuda-jit | 4 | 15 | 3.574 | 0.01077 |
| irfft_c32_n1048576 | jax-cuda-jit | 16 | 15 | 2.443 | 0.8898 |
| irfft_c64_n1048576 | jax-cuda-jit | 2 | 15 | 2.171 | 6.867e-08 |

## Run metadata

```yaml
schema_version: 1
target_profile: nvidia-gpu
suite_id: gpu/fft
suite_file: benchmarks/gpu/fft.yaml
timestamp: '2026-10-13T10:12:41Z'
tenferro_rs:
  path: extern/tenferro-rs
  commit: 8b13220c1e0397a29a27be65557c24105352a35b
  dirty: false
  features:
  - cuda
  resolved_path: /workspaces/tenferro-benchmark/extern/tenferro-rs
harness:
  path: /workspaces/tenferro-benchmark
  commit: ed0ed4c79805014f2bba4644383450a6b9bf9361
  dirty: false
environment:
  hostname: 33ed5e46905b
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
    .devcontainer/cuda/devcontainer.json: 6d3b10caecad15c0a513120ce6282ebb13c65be81c40e29f78a329439c50f083
    .devcontainer/cuda/Dockerfile: 1cadf2f890f01187553bab0f136dc817f8964eff6a35b1806b1448a7383d21de
    src/bin/benchmark_gpu_public_api.rs: f29f1ad0dd99e8e5d24fffc68b6fe30673b0c42d34b5fe4a5557b7f774c2af15
    src/bin/benchmark_gpu_public_api/runtime_ops.rs: 86f2856411d902d27dcc5c8e66415eb933e057e04b9fad86be55569004572a36
    src/bin/benchmark_gpu_public_api/structural.rs: b5d780e867a5748be4f29f245d367833a8cd5ee0ebd440926d8b3003b432b0e4
    src/bin/benchmark_gpu_public_api/linalg.rs: 7d4e29355a973f3c60c2af53fd6ceac597d85fe21f9ab5f6bf164d0afe0a6566
    src/bin/benchmark_gpu_public_api/reuse.rs: ad72b565373904275128d7beb8673bada08d922e8f310be6cedb0d2d8187898f
    /workspaces/tenferro-benchmark/scripts/benchmark_gpu_public_api.py: 24cd5cec53eaecc8b6b4a22a1cd69e8cbe430ab6c9945d48df4c03d3b2655999
    /workspaces/tenferro-benchmark/scripts/gpu_public_api_ops.py: 0a9e8d89405fc802e392e047d8b7c587e38ab1850d831735dd53fe13e87dabd3
    scripts/run_gpu_public_api.sh: 3516e06358fc3c2e36a066067cdffcab85673c7e78bf9a244dde318d109dd772
    data/results/nvidia-gpu/gpu/fft/20261013_101241/fixtures/manifest.json: 6f14a232def70d9c1acea54911211620ba8165c34541f631b0f76c95792ea041
    target/release/benchmark_gpu_public_api: 91380afb8faffb97ca96d131ef98cc8e2656aababe594e82d65a07c9bb1bed38
  reference_numpy_version: 2.4.2
```
