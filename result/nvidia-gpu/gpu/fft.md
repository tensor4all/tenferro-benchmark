# gpu/fft: CUDA public API benchmark

Measured cells: **80**. Unsupported cells: **0**, including **0** numerically verified but timing-excluded cells. Unsupported cells do not count as measured coverage.

Raw run: `data/results/nvidia-gpu/gpu/fft/20261102_205658/`.

Values are median (IQR) **µs per operation**. Samples time batches of declared operations plus device synchronization; retained output memory is limited to 256 MiB. Batch counts and raw durations are preserved in JSONL.

Setup, input transfers, library/plan initialization, JIT compilation, validation downloads and output destruction are outside timing. Intrinsic output allocation is inside timing; output-reuse rows instead use preallocated destinations across samples. Accumulating destinations are restored from prepared GPU seeds and synchronized before every batch clock, including initialization and calibration. Direct and eager each use one entered session across samples; direct FFT uses cached `FftExecutor`. JAX eager and compiled JIT are separate arms.

All measured output elements are checked against independently generated NumPy/SciPy reference results before and after sampling. Unsupported rows include unavailable APIs and APIs whose internal setup cannot be excluded; these are not performance measurements. Short batches limited by output memory remain identifiable in the sampling table.

| Case | tenferro-cuda-direct | tenferro-cuda-eager | pytorch-cuda | jax-cuda | jax-cuda-jit |
| --- | ---: | ---: | ---: | ---: | ---: |
| fft_c32_n1024 | 19.118 (1.386) | 48.535 (0.436) | 9.260 (0.028) | 706.324 (125.247) | 42.212 (0.489) |
| fft_c64_n1024 | 20.136 (1.653) | 48.620 (0.135) | 9.363 (0.045) | 704.797 (1.918) | 41.620 (0.395) |
| ifft_c32_n1024 | 51.634 (0.603) | 83.906 (0.579) | 18.073 (0.080) | 718.764 (27.868) | 46.378 (0.326) |
| ifft_c64_n1024 | 52.226 (0.278) | 84.969 (1.182) | 18.439 (0.182) | 706.801 (3.364) | 47.476 (0.260) |
| rfft_f32_n1024 | 19.957 (0.377) | 48.693 (0.258) | 9.589 (0.039) | 702.486 (1.864) | 41.994 (0.399) |
| rfft_f64_n1024 | 21.155 (0.419) | 50.868 (0.203) | 9.642 (0.139) | 704.017 (2.629) | 41.884 (0.227) |
| irfft_c32_n1024 | 50.346 (0.571) | 85.870 (0.667) | 30.745 (0.177) | 716.965 (2.397) | 56.528 (0.266) |
| irfft_c64_n1024 | 51.901 (1.210) | 84.900 (0.889) | 31.449 (0.113) | 715.352 (3.034) | 56.295 (0.150) |
| fft_c32_n1048576 | 247.677 (0.228) | 137.783 (2.709) | 40.126 (0.073) | 718.413 (4.706) | 44.693 (0.312) |
| fft_c64_n1048576 | 496.784 (0.524) | 314.375 (4.280) | 99.949 (0.249) | 714.287 (3.368) | 106.033 (0.358) |
| ifft_c32_n1048576 | 103.809 (0.449) | 143.949 (3.164) | 47.701 (0.100) | 722.895 (3.612) | 52.319 (0.265) |
| ifft_c64_n1048576 | 273.854 (58.619) | 596.401 (6.639) | 111.534 (0.150) | 720.265 (3.253) | 117.166 (0.349) |
| rfft_f32_n1048576 | 56.274 (0.481) | 80.932 (1.907) | 28.974 (0.044) | 714.314 (3.894) | 46.077 (0.463) |
| rfft_f64_n1048576 | 232.007 (0.362) | 113.620 (0.372) | 50.420 (0.037) | 718.845 (3.274) | 54.587 (0.073) |
| irfft_c32_n1048576 | 87.394 (0.300) | 123.194 (0.770) | 42.714 (0.079) | 740.083 (6.628) | 67.757 (0.398) |
| irfft_c64_n1048576 | 115.227 (0.610) | 149.159 (6.229) | 71.486 (0.081) | 738.607 (6.036) | 75.511 (0.117) |

## Sampling

| Case | Backend | Ops/sample | Samples | Median batch ms | Verification error/tolerance |
| --- | --- | ---: | ---: | ---: | ---: |
| fft_c32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.447 | 0.01405 |
| fft_c64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.577 | 1.351e-05 |
| ifft_c32_n1024 | tenferro-cuda-direct | 64 | 15 | 3.305 | 0.02634 |
| ifft_c64_n1024 | tenferro-cuda-direct | 64 | 15 | 3.342 | 4.043e-08 |
| rfft_f32_n1024 | tenferro-cuda-direct | 128 | 15 | 2.554 | 0.01301 |
| rfft_f64_n1024 | tenferro-cuda-direct | 128 | 15 | 2.708 | 1.518e-05 |
| irfft_c32_n1024 | tenferro-cuda-direct | 64 | 15 | 3.222 | 0.01787 |
| irfft_c64_n1024 | tenferro-cuda-direct | 64 | 15 | 3.322 | 3.809e-08 |
| fft_c32_n1048576 | tenferro-cuda-direct | 32 | 15 | 7.926 | 0.5149 |
| fft_c64_n1048576 | tenferro-cuda-direct | 16 | 15 | 7.949 | 0.01641 |
| ifft_c32_n1048576 | tenferro-cuda-direct | 16 | 15 | 1.661 | 0.633 |
| ifft_c64_n1048576 | tenferro-cuda-direct | 16 | 15 | 4.382 | 4.754e-08 |
| rfft_f32_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.801 | 0.3419 |
| rfft_f64_n1048576 | tenferro-cuda-direct | 31 | 15 | 7.192 | 0.01077 |
| irfft_c32_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.797 | 0.8898 |
| irfft_c64_n1048576 | tenferro-cuda-direct | 16 | 15 | 1.844 | 6.867e-08 |
| fft_c32_n1024 | tenferro-cuda-eager | 64 | 15 | 3.106 | 0.01405 |
| fft_c64_n1024 | tenferro-cuda-eager | 64 | 15 | 3.112 | 1.351e-05 |
| ifft_c32_n1024 | tenferro-cuda-eager | 32 | 15 | 2.685 | 0.02634 |
| ifft_c64_n1024 | tenferro-cuda-eager | 32 | 15 | 2.719 | 4.043e-08 |
| rfft_f32_n1024 | tenferro-cuda-eager | 64 | 15 | 3.116 | 0.01301 |
| rfft_f64_n1024 | tenferro-cuda-eager | 64 | 15 | 3.256 | 1.518e-05 |
| irfft_c32_n1024 | tenferro-cuda-eager | 32 | 15 | 2.748 | 0.01787 |
| irfft_c64_n1024 | tenferro-cuda-eager | 32 | 15 | 2.717 | 3.809e-08 |
| fft_c32_n1048576 | tenferro-cuda-eager | 32 | 15 | 4.409 | 0.5149 |
| fft_c64_n1048576 | tenferro-cuda-eager | 16 | 15 | 5.030 | 0.01641 |
| ifft_c32_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.303 | 0.633 |
| ifft_c64_n1048576 | tenferro-cuda-eager | 16 | 15 | 9.542 | 4.754e-08 |
| rfft_f32_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.590 | 0.3419 |
| rfft_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 1.818 | 0.01077 |
| irfft_c32_n1048576 | tenferro-cuda-eager | 16 | 15 | 1.971 | 0.8898 |
| irfft_c64_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.387 | 6.867e-08 |
| fft_c32_n1024 | pytorch-cuda | 256 | 15 | 2.371 | 0.01405 |
| fft_c64_n1024 | pytorch-cuda | 256 | 15 | 2.397 | 1.351e-05 |
| ifft_c32_n1024 | pytorch-cuda | 128 | 15 | 2.313 | 0.02634 |
| ifft_c64_n1024 | pytorch-cuda | 128 | 15 | 2.360 | 4.043e-08 |
| rfft_f32_n1024 | pytorch-cuda | 256 | 15 | 2.455 | 0.01301 |
| rfft_f64_n1024 | pytorch-cuda | 256 | 15 | 2.468 | 1.518e-05 |
| irfft_c32_n1024 | pytorch-cuda | 128 | 15 | 3.935 | 0.01787 |
| irfft_c64_n1024 | pytorch-cuda | 64 | 15 | 2.013 | 3.809e-08 |
| fft_c32_n1048576 | pytorch-cuda | 32 | 15 | 1.284 | 0.5149 |
| fft_c64_n1048576 | pytorch-cuda | 16 | 15 | 1.599 | 0.01641 |
| ifft_c32_n1048576 | pytorch-cuda | 32 | 15 | 1.526 | 0.633 |
| ifft_c64_n1048576 | pytorch-cuda | 16 | 15 | 1.785 | 4.754e-08 |
| rfft_f32_n1048576 | pytorch-cuda | 63 | 15 | 1.825 | 0.3419 |
| rfft_f64_n1048576 | pytorch-cuda | 31 | 15 | 1.563 | 0.01077 |
| irfft_c32_n1048576 | pytorch-cuda | 64 | 15 | 2.734 | 0.8898 |
| irfft_c64_n1048576 | pytorch-cuda | 32 | 15 | 2.288 | 6.867e-08 |
| fft_c32_n1024 | jax-cuda | 4 | 15 | 2.825 | 0.01405 |
| fft_c64_n1024 | jax-cuda | 4 | 15 | 2.819 | 1.351e-05 |
| ifft_c32_n1024 | jax-cuda | 4 | 15 | 2.875 | 0.02634 |
| ifft_c64_n1024 | jax-cuda | 4 | 15 | 2.827 | 4.043e-08 |
| rfft_f32_n1024 | jax-cuda | 4 | 15 | 2.810 | 0.01301 |
| rfft_f64_n1024 | jax-cuda | 4 | 15 | 2.816 | 1.518e-05 |
| irfft_c32_n1024 | jax-cuda | 4 | 15 | 2.868 | 0.01787 |
| irfft_c64_n1024 | jax-cuda | 4 | 15 | 2.861 | 3.809e-08 |
| fft_c32_n1048576 | jax-cuda | 2 | 15 | 1.437 | 0.5149 |
| fft_c64_n1048576 | jax-cuda | 4 | 15 | 2.857 | 0.01641 |
| ifft_c32_n1048576 | jax-cuda | 4 | 15 | 2.892 | 0.633 |
| ifft_c64_n1048576 | jax-cuda | 4 | 15 | 2.881 | 4.754e-08 |
| rfft_f32_n1048576 | jax-cuda | 4 | 15 | 2.857 | 0.3419 |
| rfft_f64_n1048576 | jax-cuda | 4 | 15 | 2.875 | 0.01077 |
| irfft_c32_n1048576 | jax-cuda | 4 | 15 | 2.960 | 0.8898 |
| irfft_c64_n1048576 | jax-cuda | 4 | 15 | 2.954 | 6.867e-08 |
| fft_c32_n1024 | jax-cuda-jit | 64 | 15 | 2.702 | 0.01405 |
| fft_c64_n1024 | jax-cuda-jit | 64 | 15 | 2.664 | 1.351e-05 |
| ifft_c32_n1024 | jax-cuda-jit | 64 | 15 | 2.968 | 0.02634 |
| ifft_c64_n1024 | jax-cuda-jit | 64 | 15 | 3.038 | 4.043e-08 |
| rfft_f32_n1024 | jax-cuda-jit | 64 | 15 | 2.688 | 0.01301 |
| rfft_f64_n1024 | jax-cuda-jit | 64 | 15 | 2.681 | 1.518e-05 |
| irfft_c32_n1024 | jax-cuda-jit | 64 | 15 | 3.618 | 0.01787 |
| irfft_c64_n1024 | jax-cuda-jit | 64 | 15 | 3.603 | 3.809e-08 |
| fft_c32_n1048576 | jax-cuda-jit | 32 | 15 | 1.430 | 0.5149 |
| fft_c64_n1048576 | jax-cuda-jit | 16 | 15 | 1.697 | 0.01641 |
| ifft_c32_n1048576 | jax-cuda-jit | 32 | 15 | 1.674 | 0.633 |
| ifft_c64_n1048576 | jax-cuda-jit | 16 | 15 | 1.875 | 4.754e-08 |
| rfft_f32_n1048576 | jax-cuda-jit | 63 | 15 | 2.903 | 0.3419 |
| rfft_f64_n1048576 | jax-cuda-jit | 31 | 15 | 1.692 | 0.01077 |
| irfft_c32_n1048576 | jax-cuda-jit | 32 | 15 | 2.168 | 0.8898 |
| irfft_c64_n1048576 | jax-cuda-jit | 32 | 15 | 2.416 | 6.867e-08 |

## Run metadata

```yaml
schema_version: 1
target_profile: nvidia-gpu
suite_id: gpu/fft
suite_file: benchmarks/gpu/fft.yaml
timestamp: '2026-11-02T20:58:07Z'
tenferro_rs:
  path: extern/tenferro-rs
  commit: 433121a7756fdb4f109b02f0422d3714a04514c6
  dirty: false
  features:
  - cuda
  resolved_path: /home/shinaoka/projects/tensor4all/tenferro-batched-gemm
harness:
  path: /workspaces/tenferro-benchmark
  commit: 8987033651c18a03e47e16cdd365df807b77b6be
  dirty: true
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
  hostname: 0fe295fad79b
  os: Linux-6.8.0-147-generic-x86_64-with-glibc2.39
  arch: x86_64
  cpu: x86_64
  affinity:
    pinning: none
    requested_measured_cpus: null
    observer_cpus: null
    override_of: null
    thread_budget: '1'
    measured_cpus: 0-7
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
  device_name: NVIDIA A100 80GB PCIe
  device_uuid: GPU-530977e1-4968-9283-4129-9fbec3e66542
  device_memory_bytes: 85899345920
  driver_version: 580.178.04
  runtime_version: '12.9'
  cuda_version: '13.0'
  cudnn_version: '92700'
gpu_assignment:
  role: benchmark
  physical_device_index: 0
  device_uuid: GPU-530977e1-4968-9283-4129-9fbec3e66542
  cuda_visible_devices: GPU-530977e1-4968-9283-4129-9fbec3e66542
  process_device_ordinal: 0
  verification_only: false
collection:
  command: BENCHMARK_TARGET_PROFILE=nvidia-gpu GPU_BENCH_DEVICE=0 BENCH_RUNS=15 BENCH_WARMUPS=3
    ./scripts/run_gpu_public_api.sh fft
  devcontainer_config: .devcontainer/cuda/devcontainer.json
  expected_cases: 16
  selected_cases: 16
  sha256:
    .devcontainer/cuda/devcontainer.json: 0bf1465bbd1a4c9f896d5320838f2d5a488632a59af7f9b59cc050e95e0b8e09
    .devcontainer/cuda/Dockerfile: 578d9815e5145aa1651ac7e93833468c345ee8bf5e95bfa7717551580ae9feca
    src/bin/benchmark_gpu_public_api.rs: 1a5f3469fc9ffde64a79e05eb9853632366855ece3280ba83e2ba21fe3fb2cdc
    src/bin/benchmark_gpu_public_api/lifetime.rs: 34be366f56aac88f5f3b86ade8018a15dad2bca85cea74f56cb2fe121917d0a9
    src/bin/benchmark_gpu_public_api/linalg.rs: 9c3d7916a3281d2dfd1caa68551dc9c81c79dab6b7525324339e5c4816581e6d
    src/bin/benchmark_gpu_public_api/multi.rs: 291e63a517b6f7108527254a7f83751420aa1d4e73395aa60f0691312b2a3773
    src/bin/benchmark_gpu_public_api/plan.rs: 2d87b9d73bf25df8797133cf0a8dc8b1fa974fc3bee055ad92e42802c9ebf643
    src/bin/benchmark_gpu_public_api/read_layout.rs: fc42e2160094ba5f9c7dd6facb5667a76e10b73f971ef9facfb1f64723e42467
    src/bin/benchmark_gpu_public_api/reuse.rs: a5a4aa0edb0a1158f6669b642f4bc25a7079314fc2a711ef3ef57cc19c0a35b2
    src/bin/benchmark_gpu_public_api/runtime_ops.rs: c9d61b1e99bda517a59a6e34f625fd4d651bbbb19f6dc06ba4d0d1c0d50dfd6a
    src/bin/benchmark_gpu_public_api/state.rs: e11a863aee7a85822c474eb771b45ec402e21dbe92173d178c42119ab8ca307e
    src/bin/benchmark_gpu_public_api/structural.rs: 8b61ac01b58edc090eafbc124ec135d5efeb4979631a939656ba78dd6e5c042a
    /workspaces/tenferro-benchmark/scripts/benchmark_gpu_public_api.py: ddc112ea5d568a0399378f2be8b31c91b3144fe7d286e8094d27ca5ccda959bf
    /workspaces/tenferro-benchmark/scripts/gpu_public_api_ops.py: e88823db3f4a1fe43dc13c4743c9d78ece6af76a968724091af318dd8d7f2528
    scripts/run_gpu_public_api.sh: 9cbcb473cba55b1605bab07fd3ff14247859fd7fe9e73d5378dd70108313256b
    data/results/nvidia-gpu/gpu/fft/20261102_205658/fixtures/manifest.json: 6f14a232def70d9c1acea54911211620ba8165c34541f631b0f76c95792ea041
    target/release/benchmark_gpu_public_api: c0a0be40dd246cd4cb7ca59cac19bacd1549a4f3a593aea91e887cf41f6e4aa6
  reference_numpy_version: 2.4.2
```
