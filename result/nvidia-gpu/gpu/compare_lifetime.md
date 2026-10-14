# Small CUDA compare: lifetime diagnostic

Diagnostic only: these nonstationary samples are not steady-state operation coverage or a regression/speedup claim. Physical GPU1 was used for debugging; no GPU0 publication timings are replaced.

Raw diagnostic: `data/verification/nvidia-gpu/gpu/compare_recheck/20261014_010820/`. Three independent processes ran the same case sequence, each with 3 warmups and 30 batched samples. Setup and validation are untimed; each batch retains outputs through native device synchronization and clock stop.

Sequence: `compare_f32_n1024`, `compare_f64_n1024`, `compare_eq_f32_n1024`, `compare_lt_f32_n1024`, `compare_le_f32_n1024`, `compare_ge_f32_n1024`, `compare_eq_f64_n1024`, `compare_lt_f64_n1024`, `compare_le_f64_n1024`, `compare_ge_f64_n1024`.

Each case creates a backend, but process-global device allocation state can survive between cases. First/last columns describe the first/last ten samples in that process; they are not balanced candidate/baseline comparisons.

| Trial | Case | Ops/sample | Median µs/op | First ten median batch ms | Last ten median batch ms | Last/first |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | compare_f32_n1024 | 128 | 108.954 | 6.275 | 18.212 | 2.902 |
| 1 | compare_f64_n1024 | 16 | 185.200 | 2.883 | 3.059 | 1.061 |
| 1 | compare_eq_f32_n1024 | 16 | 202.251 | 3.135 | 3.306 | 1.054 |
| 1 | compare_lt_f32_n1024 | 16 | 218.918 | 3.416 | 3.574 | 1.046 |
| 1 | compare_le_f32_n1024 | 16 | 236.779 | 3.688 | 4.069 | 1.103 |
| 1 | compare_ge_f32_n1024 | 8 | 266.379 | 2.180 | 2.051 | 0.941 |
| 1 | compare_eq_f64_n1024 | 8 | 263.098 | 2.076 | 2.124 | 1.023 |
| 1 | compare_lt_f64_n1024 | 8 | 273.986 | 2.221 | 2.195 | 0.989 |
| 1 | compare_le_f64_n1024 | 8 | 280.300 | 2.215 | 2.278 | 1.029 |
| 1 | compare_ge_f64_n1024 | 8 | 289.479 | 2.303 | 2.333 | 1.013 |
| 2 | compare_f32_n1024 | 256 | 154.847 | 17.231 | 61.237 | 3.554 |
| 2 | compare_f64_n1024 | 32 | 109.481 | 3.087 | 3.809 | 1.234 |
| 2 | compare_eq_f32_n1024 | 16 | 146.553 | 2.248 | 2.433 | 1.082 |
| 2 | compare_lt_f32_n1024 | 16 | 164.178 | 2.527 | 2.692 | 1.065 |
| 2 | compare_le_f32_n1024 | 16 | 181.419 | 2.806 | 2.957 | 1.054 |
| 2 | compare_ge_f32_n1024 | 16 | 211.920 | 3.046 | 3.700 | 1.215 |
| 2 | compare_eq_f64_n1024 | 16 | 213.030 | 3.413 | 3.459 | 1.013 |
| 2 | compare_lt_f64_n1024 | 16 | 229.588 | 3.696 | 3.682 | 0.996 |
| 2 | compare_le_f64_n1024 | 16 | 246.937 | 3.973 | 3.960 | 0.997 |
| 2 | compare_ge_f64_n1024 | 8 | 262.492 | 2.072 | 2.135 | 1.031 |
| 3 | compare_f32_n1024 | 128 | 84.828 | 6.184 | 16.760 | 2.710 |
| 3 | compare_f64_n1024 | 16 | 180.378 | 2.761 | 2.986 | 1.081 |
| 3 | compare_eq_f32_n1024 | 16 | 208.666 | 3.234 | 3.476 | 1.075 |
| 3 | compare_lt_f32_n1024 | 16 | 220.137 | 3.479 | 3.565 | 1.025 |
| 3 | compare_le_f32_n1024 | 16 | 229.122 | 3.548 | 3.718 | 1.048 |
| 3 | compare_ge_f32_n1024 | 8 | 247.410 | 1.963 | 1.986 | 1.012 |
| 3 | compare_eq_f64_n1024 | 8 | 255.944 | 2.029 | 2.059 | 1.015 |
| 3 | compare_lt_f64_n1024 | 8 | 264.853 | 2.079 | 2.124 | 1.022 |
| 3 | compare_le_f64_n1024 | 8 | 275.361 | 2.231 | 2.181 | 0.978 |
| 3 | compare_ge_f64_n1024 | 8 | 279.981 | 2.207 | 2.248 | 1.019 |

Source observation: at the recorded tenferro revision, `cubecl/dispatch.rs::alloc_bool_output` calls `client.empty` directly; numeric `alloc_output` first consults `take_output_handle`. This difference is a hypothesis for allocator/lifetime investigation, not an established cause. The separate untimed public-counter pass is published in `compare_lifetime_counters.md`. Public output-pool statistics and clear controls remain unavailable; upstream tenferro-rs #1928 tracks the investigation. These observations do not establish the cause or a memory leak.

Run metadata was collected immediately after the probes with the same checkout and GPU environment; it is not a per-phase timing record.

```yaml
schema_version: 1
target_profile: nvidia-gpu
suite_id: gpu/isolated_ops
suite_file: benchmarks/gpu/isolated_ops.yaml
timestamp: '2026-10-14T01:08:20Z'
tenferro_rs:
  path: extern/tenferro-rs
  commit: 9655b4e0daced17e0aa35a30d35710b069e10644
  dirty: false
  features:
  - cuda
  resolved_path: /workspaces/tenferro-benchmark/extern/tenferro-confirm-baseline
harness:
  path: /workspaces/tenferro-benchmark/.worktrees/gpu-vendors
  commit: b66fb4ec081468dd62f07ab392e25d0200665aff
  dirty: false
environment:
  hostname: 737f43342de3
  os: Linux-6.8.0-146-generic-x86_64-with-glibc2.39
  arch: x86_64
  cpu: x86_64
  env:
    OMP_NUM_THREADS: '1'
    RAYON_NUM_THREADS: '1'
    OPENBLAS_ROOT: /opt/openblas
    OPENBLAS_NUM_THREADS: '1'
    CUDA_HOME: /usr/local/cuda
    CUDA_PATH: /usr/local/cuda
    TENFERRO_CUTENSOR_PATH: /usr/lib/x86_64-linux-gnu/libcutensor/12/libcutensor.so.2
    CARGO_TARGET_DIR: /workspaces/tenferro-benchmark/target
    BENCH_RUNS: '30'
    BENCH_WARMUPS: '3'
    USE_CUDA: '1'
timing_policy:
  version: 2
  scope: operation execution and completion synchronization
  setup: inputs, transfers, runtimes/handles, plans and compilation outside timer;
    public API session entry may remain inside
  cleanup: returned outputs retained until after timer stops
  minimum_untimed_priming_runs: 1
  short_operation_sampling: steady-state GPU cases use one interval over many operations
    with a device synchronize at the batch edge; session/lifecycle diagnostics batch
    their calls where the API allows, and issue-workflow diagnostics whose per-call
    precondition cannot be batched sample one isolated call
  trace_execution: prepare_compiled outside timing; run_prepared inside
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
  device_ordinal: 1
  device_name: NVIDIA GeForce RTX 3060
  device_uuid: GPU-ed896702-ca5c-93f6-2537-06fffdac4a91
  device_memory_bytes: 12884901888
  driver_version: 580.178.04
  runtime_version: '12.9'
  cuda_version: '13.0'
  cudnn_version: '92700'
```
