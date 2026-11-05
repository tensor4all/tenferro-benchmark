# gpu/structural: CUDA public API benchmark

Measured cells: **60**. Unsupported cells: **4**, including **0** numerically verified but timing-excluded cells. Unsupported cells do not count as measured coverage.

Logical traffic: this run measured a plain device copy of 67108864 bytes at **1405.1 GB/s** on this device. Every record carries the logical bytes the case must move per operation and the ratio of the case's implied rate to that copy rate, in the sampling table below. Logical bytes are the declared array sizes, not device DRAM traffic: a case whose operands stay in cache moves less DRAM than its arrays suggest, so a ratio above 1 is recorded, not rejected.

Raw run: `data/results/nvidia-gpu/gpu/structural/20261105_0820/`.

Isolated public APIs and explicit unsupported timing contracts. Issue 155.

Values are median (IQR) **µs per operation**. Samples time batches of declared operations plus device synchronization; retained output memory is limited to 256 MiB. Batch counts and raw durations are preserved in JSONL.

Setup, input transfers, library/plan initialization, JIT compilation, validation downloads and output destruction are outside timing. Intrinsic output allocation is inside timing; output-reuse rows instead use preallocated destinations across samples. Accumulating destinations are restored from prepared GPU seeds and synchronized before every batch clock, including initialization and calibration. Direct and eager each use one entered session across samples; direct FFT uses cached `FftExecutor`. JAX eager and compiled JIT are separate arms.

All measured output elements are checked against independently generated NumPy/SciPy reference results before and after sampling. Unsupported rows include unavailable APIs and APIs whose internal setup cannot be excluded; these are not performance measurements. Short batches limited by output memory remain identifiable in the sampling table.

Reshape-copy cases add a leading singleton dimension and materialize the full output. Direct Tensor.reshape and EagerSession.reshape copy device data; PyTorch calls public view_copy. Direct/PyTorch initial/final allocation-pointer checks are untimed and reject input aliasing. Eager reshape copies its input before returning a descriptor view; original CUDA placement/shape/dtype are checked and a public inspection copy is made outside timing. JAX reshape has no established matching owned-output guarantee and remains unsupported; reshape-plus-copy composition is not substituted.

Stack selects actual public Tensor.stack/EagerSession.stack with two distinct inputs and axis 0 or 2. Their intrinsic intermediate allocations are part of this public stack operation. PyTorch/JAX use native stack; final outputs are fully verified. Direct Rust initial/final pointer checks reject aliasing either input outside timing; eager original descriptors are checked before any untimed inspection copy.

| Case | tenferro-cuda-direct | tenferro-cuda-eager | pytorch-cuda | jax-cuda |
| --- | ---: | ---: | ---: | ---: |
| pad_default_f64_n1048576 | 39.630 (0.252) | 41.667 (0.331) | 15.528 (0.375) | 67.589 (2.130) |
| slice_default_f64_n1048576 | 24.324 (0.181) | unsupported | 11.696 (0.154) | 75.072 (0.861) |
| reverse_default_f64_n1048576 | 27.138 (0.291) | 29.181 (0.349) | 10.634 (0.071) | 51.523 (0.329) |
| concatenate_default_f64_n1048576 | 50.342 (0.212) | 52.388 (0.289) | 25.833 (0.297) | 115.675 (10.410) |
| extract_diag_default_f64_n1048576 | 18.786 (0.187) | 20.150 (0.833) | 8.403 (0.104) | 47.543 (3.727) |
| embed_diag_default_f64_n1024 | 21.635 (0.236) | 23.484 (0.435) | 12.471 (0.274) | 58.772 (1.560) |
| tril_default_f64_n1048576 | 26.302 (0.106) | 28.583 (0.171) | 10.840 (0.048) | 47.846 (0.430) |
| triu_default_f64_n1048576 | 26.308 (0.129) | 28.323 (0.388) | 10.972 (0.036) | 47.643 (0.822) |
| take_along_axis_broadcast_indices_f64_n1048576 | 61.944 (0.307) | 74.621 (7.408) | 17.688 (0.165) | 52.223 (0.738) |
| scatter_default_f64_n1048576 | 62.617 (0.395) | 65.513 (0.231) | 22.764 (0.117) | 115.325 (0.726) |
| cast_default_f64_n1048576 | 22.412 (0.198) | 51.442 (0.297) | 9.129 (0.022) | 91.712 (1.033) |
| broadcast_in_dim_f64_n1048576 | 33.418 (0.123) | unsupported | 8.775 (0.275) | 70.885 (0.540) |
| dynamic_slice_in_bounds_f64_n1048576 | 21.551 (0.242) | 23.563 (0.209) | unsupported | 95.658 (4.151) |
| gather_axis0_f64_n1048576 | 42.314 (0.132) | 45.134 (0.096) | 11.484 (0.198) | 86.961 (1.140) |
| reshape_copy_f64_n1048576 | 25.060 (0.154) | 30.136 (0.121) | 9.453 (0.397) | unsupported |
| stack_axis0_f64_n1048576 | 116.327 (0.233) | 131.010 (5.530) | 25.411 (0.263) | 102.886 (7.018) |

## Sampling

| Case | Backend | Ops/sample | Samples | Median batch ms | Bytes/op | Copy ratio | Verification error/tolerance |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| broadcast_in_dim_f64_n1048576 | jax-cuda | 32 | 15 | 2.268 | 8396800 | 0.084 | 0 |
| broadcast_in_dim_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.281 | 8396800 | 0.681 | 0 |
| broadcast_in_dim_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 1.069 | 8396800 | 0.179 | 0 |
| broadcast_in_dim_f64_n1048576 | tenferro-cuda-eager | — | — | — | 8396800 | unavailable | Eager broadcast returns a lazy broadcast view, not the materialized full output required by this case. |
| cast_default_f64_n1048576 | jax-cuda | 32 | 15 | 2.935 | 12582912 | 0.098 | 0 |
| cast_default_f64_n1048576 | pytorch-cuda | 64 | 15 | 0.584 | 12582912 | 0.981 | 0 |
| cast_default_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.434 | 12582912 | 0.400 | 0 |
| cast_default_f64_n1048576 | tenferro-cuda-eager | 64 | 15 | 3.292 | 12582912 | 0.174 | 0 |
| concatenate_default_f64_n1048576 | jax-cuda | 16 | 15 | 1.851 | 33554432 | 0.206 | 0 |
| concatenate_default_f64_n1048576 | pytorch-cuda | 16 | 15 | 0.413 | 33554432 | 0.924 | 0 |
| concatenate_default_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 0.805 | 33554432 | 0.474 | 0 |
| concatenate_default_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 0.838 | 33554432 | 0.456 | 0 |
| dynamic_slice_in_bounds_f64_n1048576 | jax-cuda | 32 | 15 | 3.061 | 10485776 | 0.078 | 0 |
| dynamic_slice_in_bounds_f64_n1048576 | pytorch-cuda | — | — | — | 10485776 | unavailable | No matched single materializing multidimensional slice API with CUDA-resident start indices and clamping; index construction/composed slicing is not substituted. |
| dynamic_slice_in_bounds_f64_n1048576 | tenferro-cuda-direct | 128 | 15 | 2.758 | 10485776 | 0.346 | 0 |
| dynamic_slice_in_bounds_f64_n1048576 | tenferro-cuda-eager | 128 | 15 | 3.016 | 10485776 | 0.317 | 0 |
| embed_diag_default_f64_n1024 | jax-cuda | 32 | 15 | 1.881 | 8396800 | 0.102 | 0 |
| embed_diag_default_f64_n1024 | pytorch-cuda | 32 | 15 | 0.399 | 8396800 | 0.479 | 0 |
| embed_diag_default_f64_n1024 | tenferro-cuda-direct | 32 | 15 | 0.692 | 8396800 | 0.276 | 0 |
| embed_diag_default_f64_n1024 | tenferro-cuda-eager | 32 | 15 | 0.751 | 8396800 | 0.254 | 0 |
| extract_diag_default_f64_n1048576 | jax-cuda | 64 | 15 | 3.043 | 8396800 | 0.126 | 0 |
| extract_diag_default_f64_n1048576 | pytorch-cuda | 256 | 15 | 2.151 | 8396800 | 0.711 | 0 |
| extract_diag_default_f64_n1048576 | tenferro-cuda-direct | 128 | 15 | 2.405 | 8396800 | 0.318 | 0 |
| extract_diag_default_f64_n1048576 | tenferro-cuda-eager | 128 | 15 | 2.579 | 8396800 | 0.297 | 0 |
| gather_axis0_f64_n1048576 | jax-cuda | 32 | 15 | 2.783 | 12587008 | 0.103 | 0 |
| gather_axis0_f64_n1048576 | pytorch-cuda | 64 | 15 | 0.735 | 12587008 | 0.780 | 0 |
| gather_axis0_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 2.708 | 12587008 | 0.212 | 0 |
| gather_axis0_f64_n1048576 | tenferro-cuda-eager | 64 | 15 | 2.889 | 12587008 | 0.198 | 0 |
| pad_default_f64_n1048576 | jax-cuda | 31 | 15 | 2.095 | 16908288 | 0.178 | 0 |
| pad_default_f64_n1048576 | pytorch-cuda | 31 | 15 | 0.481 | 16908288 | 0.775 | 0 |
| pad_default_f64_n1048576 | tenferro-cuda-direct | 31 | 15 | 1.229 | 16908288 | 0.304 | 0 |
| pad_default_f64_n1048576 | tenferro-cuda-eager | 31 | 15 | 1.292 | 16908288 | 0.289 | 0 |
| reshape_copy_f64_n1048576 | jax-cuda | — | — | — | 16777216 | unavailable | No established single-call JAX reshape API that guarantees owned output materialization; native reshape may alias input. Reshape plus copy is a different composed contract. |
| reshape_copy_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.303 | 16777216 | 1.263 | 0 |
| reshape_copy_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.802 | 16777216 | 0.476 | 0 |
| reshape_copy_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.964 | 16777216 | 0.396 | 0 |
| reverse_default_f64_n1048576 | jax-cuda | 32 | 15 | 1.649 | 16777216 | 0.232 | 0 |
| reverse_default_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.340 | 16777216 | 1.123 | 0 |
| reverse_default_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.868 | 16777216 | 0.440 | 0 |
| reverse_default_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.934 | 16777216 | 0.409 | 0 |
| scatter_default_f64_n1048576 | jax-cuda | 32 | 15 | 3.690 | 25165824 | 0.155 | 0 |
| scatter_default_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.728 | 25165824 | 0.787 | 0 |
| scatter_default_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 2.004 | 25165824 | 0.286 | 0 |
| scatter_default_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.096 | 25165824 | 0.273 | 0 |
| slice_default_f64_n1048576 | jax-cuda | 32 | 15 | 2.402 | 12582912 | 0.119 | 0 |
| slice_default_f64_n1048576 | pytorch-cuda | 64 | 15 | 0.749 | 12582912 | 0.766 | 0 |
| slice_default_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 1.557 | 12582912 | 0.368 | 0 |
| slice_default_f64_n1048576 | tenferro-cuda-eager | — | — | — | 12582912 | unavailable | EagerSession::slice returns a lazy metadata view; it is not equivalent to the materializing direct slice / torch.slice_copy / JAX slice contract. No copy is added inside its timer. |
| stack_axis0_f64_n1048576 | jax-cuda | 16 | 15 | 1.646 | 33554432 | 0.232 | 0 |
| stack_axis0_f64_n1048576 | pytorch-cuda | 16 | 15 | 0.407 | 33554432 | 0.940 | 0 |
| stack_axis0_f64_n1048576 | tenferro-cuda-direct | 16 | 15 | 1.861 | 33554432 | 0.205 | 0 |
| stack_axis0_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.096 | 33554432 | 0.182 | 0 |
| take_along_axis_broadcast_indices_f64_n1048576 | jax-cuda | 64 | 15 | 3.342 | 12587008 | 0.172 | 0 |
| take_along_axis_broadcast_indices_f64_n1048576 | pytorch-cuda | 64 | 15 | 1.132 | 12587008 | 0.506 | 0 |
| take_along_axis_broadcast_indices_f64_n1048576 | tenferro-cuda-direct | 64 | 15 | 3.964 | 12587008 | 0.145 | 0 |
| take_along_axis_broadcast_indices_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.388 | 12587008 | 0.120 | 0 |
| tril_default_f64_n1048576 | jax-cuda | 32 | 15 | 1.531 | 16777216 | 0.250 | 0 |
| tril_default_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.347 | 16777216 | 1.101 | 0 |
| tril_default_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.842 | 16777216 | 0.454 | 0 |
| tril_default_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.915 | 16777216 | 0.418 | 0 |
| triu_default_f64_n1048576 | jax-cuda | 32 | 15 | 1.525 | 16777216 | 0.251 | 0 |
| triu_default_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.351 | 16777216 | 1.088 | 0 |
| triu_default_f64_n1048576 | tenferro-cuda-direct | 32 | 15 | 0.842 | 16777216 | 0.454 | 0 |
| triu_default_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.906 | 16777216 | 0.422 | 0 |

## Run metadata

```yaml
schema_version: 1
target_profile: nvidia-gpu
suite_id: gpu/structural
suite_file: /home/shinaoka/work/wallclock/data/results/nvidia-gpu/gpu/routine_selection/20261105_0820/gpu_structural.yaml
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
  - 0.095673
  - 0.093814
  - 0.097763
  - 0.096483
  - 0.093764
  - 0.097833
  - 0.095793
  - 0.093934
  - 0.094783
  - 0.095664
  - 0.094384
  - 0.094893
  - 0.096094
  - 0.095103
  - 0.095524
  median_ms: 0.095524
  bytes_per_second: 1405068129475.3152
  torch_version: 2.12.0+cu130
  cuda_version: '13.0'
  timestamp: '2026-11-05T07:54:53.288620+00:00'
gpu_assignment:
  role: benchmark
  physical_device_index: 0
  device_uuid: GPU-530977e1-4968-9283-4129-9fbec3e66542
  cuda_visible_devices: GPU-530977e1-4968-9283-4129-9fbec3e66542
  process_device_ordinal: 0
  verification_only: false
collection:
  command: BENCHMARK_TARGET_PROFILE=nvidia-gpu GPU_BENCH_DEVICE=0 BENCH_RUNS=15 BENCH_WARMUPS=3
    ./scripts/run_gpu_public_api.sh structural
  devcontainer_config: .devcontainer/cuda/devcontainer.json
  expected_cases: 16
  selected_cases: 16
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
    data/results/nvidia-gpu/gpu/structural/20261105_0820/fixtures/manifest.json: 27449ceb91a10b308247afb22c9b261ceeafa3bfdbc21adad0a4af193ed9234c
    target/release/benchmark_gpu_public_api: efd5ab70e2c4de27c458d808d308b8bf3415012551220c291451a2591d08880f
  reference_numpy_version: 2.4.2
```

## Untimed work over the recorded budget

Each row records the wall clock it spent outside its timed region. These cases spent more than the recorded budget ratio (0.2) there. The threshold has no measured basis, so this is a recorded warning: it does not change a cell, its acceptance or the exit status of the run.

| Case | Backend | Warning |
| --- | --- | --- |
| pad_default_f64_n1048576 | tenferro-cuda-direct | pad_default_f64_n1048576 untimed 5551.5 ms is 30111% of the 18.4 ms timed region (budget 20%) |
| slice_default_f64_n1048576 | tenferro-cuda-direct | slice_default_f64_n1048576 untimed 29.1 ms is 125% of the 23.3 ms timed region (budget 20%) |
| reverse_default_f64_n1048576 | tenferro-cuda-direct | reverse_default_f64_n1048576 untimed 48258.3 ms is 366972% of the 13.2 ms timed region (budget 20%) |
| concatenate_default_f64_n1048576 | tenferro-cuda-direct | concatenate_default_f64_n1048576 untimed 72.1 ms is 596% of the 12.1 ms timed region (budget 20%) |
| extract_diag_default_f64_n1048576 | tenferro-cuda-direct | extract_diag_default_f64_n1048576 untimed 2075.3 ms is 5750% of the 36.1 ms timed region (budget 20%) |
| embed_diag_default_f64_n1024 | tenferro-cuda-direct | embed_diag_default_f64_n1024 untimed 30.4 ms is 293% of the 10.4 ms timed region (budget 20%) |
| tril_default_f64_n1048576 | tenferro-cuda-direct | tril_default_f64_n1048576 untimed 33.0 ms is 261% of the 12.6 ms timed region (budget 20%) |
| triu_default_f64_n1048576 | tenferro-cuda-direct | triu_default_f64_n1048576 untimed 32.8 ms is 260% of the 12.6 ms timed region (budget 20%) |
| take_along_axis_broadcast_indices_f64_n1048576 | tenferro-cuda-direct | take_along_axis_broadcast_indices_f64_n1048576 untimed 39719.9 ms is 66768% of the 59.5 ms timed region (budget 20%) |
| scatter_default_f64_n1048576 | tenferro-cuda-direct | scatter_default_f64_n1048576 untimed 41.5 ms is 138% of the 30.1 ms timed region (budget 20%) |
| cast_default_f64_n1048576 | tenferro-cuda-direct | cast_default_f64_n1048576 untimed 140.5 ms is 646% of the 21.8 ms timed region (budget 20%) |
| broadcast_in_dim_f64_n1048576 | tenferro-cuda-direct | broadcast_in_dim_f64_n1048576 untimed 33.2 ms is 207% of the 16.0 ms timed region (budget 20%) |
| dynamic_slice_in_bounds_f64_n1048576 | tenferro-cuda-direct | dynamic_slice_in_bounds_f64_n1048576 untimed 18.0 ms is 43% of the 41.6 ms timed region (budget 20%) |
| gather_axis0_f64_n1048576 | tenferro-cuda-direct | gather_axis0_f64_n1048576 untimed 24.2 ms is 60% of the 40.6 ms timed region (budget 20%) |
| reshape_copy_f64_n1048576 | tenferro-cuda-direct | reshape_copy_f64_n1048576 untimed 36.2 ms is 301% of the 12.0 ms timed region (budget 20%) |
| stack_axis0_f64_n1048576 | tenferro-cuda-direct | stack_axis0_f64_n1048576 untimed 82.0 ms is 293% of the 27.9 ms timed region (budget 20%) |
| pad_default_f64_n1048576 | tenferro-cuda-eager | pad_default_f64_n1048576 untimed 66656.9 ms is 343636% of the 19.4 ms timed region (budget 20%) |
| reverse_default_f64_n1048576 | tenferro-cuda-eager | reverse_default_f64_n1048576 untimed 66.9 ms is 476% of the 14.1 ms timed region (budget 20%) |
| concatenate_default_f64_n1048576 | tenferro-cuda-eager | concatenate_default_f64_n1048576 untimed 102.2 ms is 813% of the 12.6 ms timed region (budget 20%) |
| extract_diag_default_f64_n1048576 | tenferro-cuda-eager | extract_diag_default_f64_n1048576 untimed 2895.3 ms is 7450% of the 38.9 ms timed region (budget 20%) |
| embed_diag_default_f64_n1024 | tenferro-cuda-eager | embed_diag_default_f64_n1024 untimed 51.6 ms is 457% of the 11.3 ms timed region (budget 20%) |
| tril_default_f64_n1048576 | tenferro-cuda-eager | tril_default_f64_n1048576 untimed 68.1 ms is 496% of the 13.7 ms timed region (budget 20%) |
| triu_default_f64_n1048576 | tenferro-cuda-eager | triu_default_f64_n1048576 untimed 67.6 ms is 498% of the 13.6 ms timed region (budget 20%) |
| take_along_axis_broadcast_indices_f64_n1048576 | tenferro-cuda-eager | take_along_axis_broadcast_indices_f64_n1048576 untimed 58169.7 ms is 157742% of the 36.9 ms timed region (budget 20%) |
| scatter_default_f64_n1048576 | tenferro-cuda-eager | scatter_default_f64_n1048576 untimed 44.8 ms is 142% of the 31.5 ms timed region (budget 20%) |
| cast_default_f64_n1048576 | tenferro-cuda-eager | cast_default_f64_n1048576 untimed 150.2 ms is 303% of the 49.6 ms timed region (budget 20%) |
| dynamic_slice_in_bounds_f64_n1048576 | tenferro-cuda-eager | dynamic_slice_in_bounds_f64_n1048576 untimed 18.7 ms is 41% of the 45.5 ms timed region (budget 20%) |
| gather_axis0_f64_n1048576 | tenferro-cuda-eager | gather_axis0_f64_n1048576 untimed 23.7 ms is 55% of the 43.4 ms timed region (budget 20%) |
| reshape_copy_f64_n1048576 | tenferro-cuda-eager | reshape_copy_f64_n1048576 untimed 33.9 ms is 234% of the 14.5 ms timed region (budget 20%) |
| stack_axis0_f64_n1048576 | tenferro-cuda-eager | stack_axis0_f64_n1048576 untimed 73.1 ms is 230% of the 31.8 ms timed region (budget 20%) |
| pad_default_f64_n1048576 | pytorch-cuda | pad_default_f64_n1048576 untimed 1768.2 ms is 24405% of the 7.2 ms timed region (budget 20%) |
| slice_default_f64_n1048576 | pytorch-cuda | slice_default_f64_n1048576 untimed 856.6 ms is 7637% of the 11.2 ms timed region (budget 20%) |
| reverse_default_f64_n1048576 | pytorch-cuda | reverse_default_f64_n1048576 untimed 1568.2 ms is 30715% of the 5.1 ms timed region (budget 20%) |
| concatenate_default_f64_n1048576 | pytorch-cuda | concatenate_default_f64_n1048576 untimed 2965.5 ms is 47786% of the 6.2 ms timed region (budget 20%) |
| extract_diag_default_f64_n1048576 | pytorch-cuda | extract_diag_default_f64_n1048576 untimed 22.7 ms is 70% of the 32.3 ms timed region (budget 20%) |
| embed_diag_default_f64_n1024 | pytorch-cuda | embed_diag_default_f64_n1024 untimed 1416.7 ms is 23350% of the 6.1 ms timed region (budget 20%) |
| tril_default_f64_n1048576 | pytorch-cuda | tril_default_f64_n1048576 untimed 1451.0 ms is 27803% of the 5.2 ms timed region (budget 20%) |
| triu_default_f64_n1048576 | pytorch-cuda | triu_default_f64_n1048576 untimed 1452.7 ms is 27601% of the 5.3 ms timed region (budget 20%) |
| take_along_axis_broadcast_indices_f64_n1048576 | pytorch-cuda | take_along_axis_broadcast_indices_f64_n1048576 untimed 693.8 ms is 4089% of the 17.0 ms timed region (budget 20%) |
| scatter_default_f64_n1048576 | pytorch-cuda | scatter_default_f64_n1048576 untimed 1296.5 ms is 11645% of the 11.1 ms timed region (budget 20%) |
| cast_default_f64_n1048576 | pytorch-cuda | cast_default_f64_n1048576 untimed 1445.7 ms is 16474% of the 8.8 ms timed region (budget 20%) |
| broadcast_in_dim_f64_n1048576 | pytorch-cuda | broadcast_in_dim_f64_n1048576 untimed 1477.5 ms is 33997% of the 4.3 ms timed region (budget 20%) |
| gather_axis0_f64_n1048576 | pytorch-cuda | gather_axis0_f64_n1048576 untimed 728.5 ms is 6593% of the 11.0 ms timed region (budget 20%) |
| reshape_copy_f64_n1048576 | pytorch-cuda | reshape_copy_f64_n1048576 untimed 1488.5 ms is 32343% of the 4.6 ms timed region (budget 20%) |
| stack_axis0_f64_n1048576 | pytorch-cuda | stack_axis0_f64_n1048576 untimed 2962.5 ms is 48708% of the 6.1 ms timed region (budget 20%) |
| pad_default_f64_n1048576 | jax-cuda | pad_default_f64_n1048576 untimed 1760.9 ms is 5446% of the 32.3 ms timed region (budget 20%) |
| slice_default_f64_n1048576 | jax-cuda | slice_default_f64_n1048576 untimed 735.3 ms is 2040% of the 36.0 ms timed region (budget 20%) |
| reverse_default_f64_n1048576 | jax-cuda | reverse_default_f64_n1048576 untimed 1459.6 ms is 5902% of the 24.7 ms timed region (budget 20%) |
| concatenate_default_f64_n1048576 | jax-cuda | concatenate_default_f64_n1048576 untimed 2962.4 ms is 10774% of the 27.5 ms timed region (budget 20%) |
| extract_diag_default_f64_n1048576 | jax-cuda | extract_diag_default_f64_n1048576 untimed 113.8 ms is 243% of the 46.8 ms timed region (budget 20%) |
| embed_diag_default_f64_n1024 | jax-cuda | embed_diag_default_f64_n1024 untimed 1576.3 ms is 5585% of the 28.2 ms timed region (budget 20%) |
| tril_default_f64_n1048576 | jax-cuda | tril_default_f64_n1048576 untimed 1612.3 ms is 7000% of the 23.0 ms timed region (budget 20%) |
| triu_default_f64_n1048576 | jax-cuda | triu_default_f64_n1048576 untimed 1600.4 ms is 6995% of the 22.9 ms timed region (budget 20%) |
| take_along_axis_broadcast_indices_f64_n1048576 | jax-cuda | take_along_axis_broadcast_indices_f64_n1048576 untimed 863.4 ms is 1721% of the 50.2 ms timed region (budget 20%) |
| scatter_default_f64_n1048576 | jax-cuda | scatter_default_f64_n1048576 untimed 1430.8 ms is 2579% of the 55.5 ms timed region (budget 20%) |
| cast_default_f64_n1048576 | jax-cuda | cast_default_f64_n1048576 untimed 1388.2 ms is 3150% of the 44.1 ms timed region (budget 20%) |
| broadcast_in_dim_f64_n1048576 | jax-cuda | broadcast_in_dim_f64_n1048576 untimed 1485.7 ms is 4369% of the 34.0 ms timed region (budget 20%) |
| dynamic_slice_in_bounds_f64_n1048576 | jax-cuda | dynamic_slice_in_bounds_f64_n1048576 untimed 520.0 ms is 1109% of the 46.9 ms timed region (budget 20%) |
| gather_axis0_f64_n1048576 | jax-cuda | gather_axis0_f64_n1048576 untimed 783.9 ms is 1881% of the 41.7 ms timed region (budget 20%) |
| stack_axis0_f64_n1048576 | jax-cuda | stack_axis0_f64_n1048576 untimed 3013.8 ms is 11898% of the 25.3 ms timed region (budget 20%) |