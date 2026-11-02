# gpu/lstsq: CUDA public API benchmark

Measured cells: **0**. Unsupported cells: **40**, including **8** numerically verified but timing-excluded cells. Unsupported cells do not count as measured coverage.

Raw run: `data/results/nvidia-gpu/gpu/lstsq/20261102_211841/`.

Values are median (IQR) **µs per operation**. Samples time batches of declared operations plus device synchronization; retained output memory is limited to 256 MiB. Batch counts and raw durations are preserved in JSONL.

Setup, input transfers, library/plan initialization, JIT compilation, validation downloads and output destruction are outside timing. Intrinsic output allocation is inside timing; output-reuse rows instead use preallocated destinations across samples. Accumulating destinations are restored from prepared GPU seeds and synchronized before every batch clock, including initialization and calibration. Direct and eager each use one entered session across samples; direct FFT uses cached `FftExecutor`. JAX eager and compiled JIT are separate arms.

All measured output elements are checked against independently generated NumPy/SciPy reference results before and after sampling. Unsupported rows include unavailable APIs and APIs whose internal setup cannot be excluded; these are not performance measurements. Short batches limited by output memory remain identifiable in the sampling table.

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
timestamp: '2026-11-02T21:18:42Z'
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
    ./scripts/run_gpu_public_api.sh lstsq
  devcontainer_config: .devcontainer/cuda/devcontainer.json
  expected_cases: 8
  selected_cases: 8
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
    data/results/nvidia-gpu/gpu/lstsq/20261102_211841/fixtures/manifest.json: 7595bb050518b743d83b4a89ec25c7874a0745e7313761b7f6a6a604747a08db
    target/release/benchmark_gpu_public_api: c0a0be40dd246cd4cb7ca59cac19bacd1549a4f3a593aea91e887cf41f6e4aa6
  reference_numpy_version: 2.4.2
```
