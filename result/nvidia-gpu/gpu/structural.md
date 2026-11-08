# gpu/structural: CUDA public API benchmark

Measured cells: **44**. Unsupported cells: **4**, including **0** numerically verified but timing-excluded cells. Unsupported cells do not count as measured coverage.

Logical traffic: this run measured a plain device copy of 67108864 bytes at **1410.7 GB/s** on this device. Every record carries the logical bytes the case must move per operation and the ratio of the case's implied rate to that copy rate, in the sampling table below. Logical bytes are the declared array sizes, not device DRAM traffic: a case whose operands stay in cache moves less DRAM than its arrays suggest, so a ratio above 1 is recorded, not rejected.

Raw run: `data/results/nvidia-gpu/gpu/structural/20261108_090627/`.

Isolated public APIs and explicit unsupported timing contracts. Issue 155.

This run compiled **1 NVRTC modules** in 1 of its arms, because the CUDA artifact cache started cold. The untimed wall of the cases that compiled is **46.2 s** of its 74.8 s of recorded case wall clock, and that wall also holds their fixture read, transfers and verification, so it is not a compile time.

Values are median (IQR) **µs per operation**. Samples time batches of declared operations plus device synchronization; retained output memory is limited to 256 MiB. Batch counts and raw durations are preserved in JSONL.

Cases marked † (**3** of 16) were sampled with a different number of operations per sample on different arms (see Ops/sample in the sampling table). Each value is still per operation, but the arms amortise launch and synchronization over different batch sizes, so a ratio between their columns is approximate and is not presented as a like-for-like comparison.

Setup, input transfers, library/plan initialization, JIT compilation, validation downloads and output destruction are outside timing. Intrinsic output allocation is inside timing; output-reuse rows instead use preallocated destinations across samples. Accumulating destinations are restored from prepared GPU seeds and synchronized before every batch clock, including initialization and calibration. Direct and eager each use one entered session across samples; direct FFT uses cached `FftExecutor`. JAX eager and compiled JIT are separate arms.

All measured output elements are checked against independently generated NumPy/SciPy reference results before and after sampling. Unsupported rows include unavailable APIs and APIs whose internal setup cannot be excluded; these are not performance measurements. Short batches limited by output memory remain identifiable in the sampling table.

Reshape-copy cases add a leading singleton dimension and materialize the full output. Direct Tensor.reshape and EagerSession.reshape copy device data; PyTorch calls public view_copy. Direct/PyTorch initial/final allocation-pointer checks are untimed and reject input aliasing. Eager reshape copies its input before returning a descriptor view; original CUDA placement/shape/dtype are checked and a public inspection copy is made outside timing. JAX reshape has no established matching owned-output guarantee and remains unsupported; reshape-plus-copy composition is not substituted.

Stack selects actual public Tensor.stack/EagerSession.stack with two distinct inputs and axis 0 or 2. Their intrinsic intermediate allocations are part of this public stack operation. PyTorch/JAX use native stack; final outputs are fully verified. Direct Rust initial/final pointer checks reject aliasing either input outside timing; eager original descriptors are checked before any untimed inspection copy.

| Case | tenferro-cuda-eager | pytorch-cuda | jax-cuda |
| --- | ---: | ---: | ---: |
| pad_default_f64_n1048576 | 40.422 (0.232) | 15.106 (0.186) | 53.175 (2.279) |
| slice_default_f64_n1048576 | unsupported | 6.876 (0.181) | 45.527 (2.389) |
| reverse_default_f64_n1048576 | 27.873 (0.216) | 10.543 (0.039) | 29.663 (0.647) |
| concatenate_default_f64_n1048576 | 51.935 (0.233) | 25.577 (0.326) | 58.209 (2.436) |
| extract_diag_default_f64_n1048576 † | 18.041 (0.221) | 5.713 (0.055) | 30.346 (1.033) |
| embed_diag_default_f64_n1024 | 22.368 (0.310) | 10.275 (0.042) | 30.135 (1.283) |
| tril_default_f64_n1048576 | 27.387 (0.263) | 10.792 (0.035) | 25.230 (0.860) |
| triu_default_f64_n1048576 | 27.368 (0.554) | 10.934 (0.038) | 24.531 (1.580) |
| take_along_axis_broadcast_indices_f64_n1048576 † | 63.365 (0.374) | 15.432 (0.136) | 29.421 (1.169) |
| scatter_default_f64_n1048576 | 65.824 (0.224) | 22.673 (0.085) | 76.290 (1.397) |
| cast_default_f64_n1048576 | 50.913 (0.163) | 8.999 (0.030) | 54.018 (0.605) |
| broadcast_in_dim_f64_n1048576 | unsupported | 7.429 (0.046) | 43.990 (1.904) |
| dynamic_slice_in_bounds_f64_n1048576 † | 21.377 (0.116) | unsupported | 62.850 (0.952) |
| gather_axis0_f64_n1048576 | 42.849 (0.175) | 7.116 (0.308) | 52.745 (0.812) |
| reshape_copy_f64_n1048576 | 28.801 (0.123) | 8.652 (0.088) | unsupported |
| stack_axis0_f64_n1048576 | 125.062 (0.824) | 25.290 (0.180) | 62.073 (2.113) |

## Sampling

| Case | Backend | Ops/sample | Samples | Median batch ms | Bytes/op | Copy ratio | Verification error/tolerance |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| broadcast_in_dim_f64_n1048576 | jax-cuda | 32 | 15 | 1.408 | 8396800 | 0.135 | 0 |
| broadcast_in_dim_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.238 | 8396800 | 0.801 | 0 |
| broadcast_in_dim_f64_n1048576 | tenferro-cuda-eager | — | — | — | 8396800 | unavailable | Eager broadcast returns a lazy broadcast view, not the materialized full output required by this case. |
| cast_default_f64_n1048576 | jax-cuda | 64 | 15 | 3.457 | 12582912 | 0.165 | 0 |
| cast_default_f64_n1048576 | pytorch-cuda | 64 | 15 | 0.576 | 12582912 | 0.991 | 0 |
| cast_default_f64_n1048576 | tenferro-cuda-eager | 64 | 15 | 3.258 | 12582912 | 0.175 | 0 |
| concatenate_default_f64_n1048576 | jax-cuda | 16 | 15 | 0.931 | 33554432 | 0.409 | 0 |
| concatenate_default_f64_n1048576 | pytorch-cuda | 16 | 15 | 0.409 | 33554432 | 0.930 | 0 |
| concatenate_default_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 0.831 | 33554432 | 0.458 | 0 |
| dynamic_slice_in_bounds_f64_n1048576 | jax-cuda | 32 | 15 | 2.011 | 10485776 | 0.118 | 0 |
| dynamic_slice_in_bounds_f64_n1048576 | pytorch-cuda | — | — | — | 10485776 | unavailable | No matched single materializing multidimensional slice API with CUDA-resident start indices and clamping; index construction/composed slicing is not substituted. |
| dynamic_slice_in_bounds_f64_n1048576 | tenferro-cuda-eager | 128 | 15 | 2.736 | 10485776 | 0.348 | 0 |
| embed_diag_default_f64_n1024 | jax-cuda | 32 | 15 | 0.964 | 8396800 | 0.198 | 0 |
| embed_diag_default_f64_n1024 | pytorch-cuda | 32 | 15 | 0.329 | 8396800 | 0.579 | 0 |
| embed_diag_default_f64_n1024 | tenferro-cuda-eager | 32 | 15 | 0.716 | 8396800 | 0.266 | 0 |
| extract_diag_default_f64_n1048576 | jax-cuda | 64 | 15 | 1.942 | 8396800 | 0.196 | 0 |
| extract_diag_default_f64_n1048576 | pytorch-cuda | 512 | 15 | 2.925 | 8396800 | 1.042 | 0 |
| extract_diag_default_f64_n1048576 | tenferro-cuda-eager | 128 | 15 | 2.309 | 8396800 | 0.330 | 0 |
| gather_axis0_f64_n1048576 | jax-cuda | 64 | 15 | 3.376 | 12587008 | 0.169 | 0 |
| gather_axis0_f64_n1048576 | pytorch-cuda | 64 | 15 | 0.455 | 12587008 | 1.254 | 0 |
| gather_axis0_f64_n1048576 | tenferro-cuda-eager | 64 | 15 | 2.742 | 12587008 | 0.208 | 0 |
| pad_default_f64_n1048576 | jax-cuda | 31 | 15 | 1.648 | 16908288 | 0.225 | 0 |
| pad_default_f64_n1048576 | pytorch-cuda | 31 | 15 | 0.468 | 16908288 | 0.793 | 0 |
| pad_default_f64_n1048576 | tenferro-cuda-eager | 31 | 15 | 1.253 | 16908288 | 0.297 | 0 |
| reshape_copy_f64_n1048576 | jax-cuda | — | — | — | 16777216 | unavailable | No established single-call JAX reshape API that guarantees owned output materialization; native reshape may alias input. Reshape plus copy is a different composed contract. |
| reshape_copy_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.277 | 16777216 | 1.375 | 0 |
| reshape_copy_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.922 | 16777216 | 0.413 | 0 |
| reverse_default_f64_n1048576 | jax-cuda | 32 | 15 | 0.949 | 16777216 | 0.401 | 0 |
| reverse_default_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.337 | 16777216 | 1.128 | 0 |
| reverse_default_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.892 | 16777216 | 0.427 | 0 |
| scatter_default_f64_n1048576 | jax-cuda | 32 | 15 | 2.441 | 25165824 | 0.234 | 0 |
| scatter_default_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.726 | 25165824 | 0.787 | 0 |
| scatter_default_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.106 | 25165824 | 0.271 | 0 |
| slice_default_f64_n1048576 | jax-cuda | 64 | 15 | 2.914 | 12582912 | 0.196 | 0 |
| slice_default_f64_n1048576 | pytorch-cuda | 64 | 15 | 0.440 | 12582912 | 1.297 | 0 |
| slice_default_f64_n1048576 | tenferro-cuda-eager | — | — | — | 12582912 | unavailable | EagerSession::slice returns a lazy metadata view; it is not equivalent to the materializing direct slice / torch.slice_copy / JAX slice contract. No copy is added inside its timer. |
| stack_axis0_f64_n1048576 | jax-cuda | 16 | 15 | 0.993 | 33554432 | 0.383 | 0 |
| stack_axis0_f64_n1048576 | pytorch-cuda | 16 | 15 | 0.405 | 33554432 | 0.941 | 0 |
| stack_axis0_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.001 | 33554432 | 0.190 | 0 |
| take_along_axis_broadcast_indices_f64_n1048576 | jax-cuda | 64 | 15 | 1.883 | 12587008 | 0.303 | 0 |
| take_along_axis_broadcast_indices_f64_n1048576 | pytorch-cuda | 64 | 15 | 0.988 | 12587008 | 0.578 | 0 |
| take_along_axis_broadcast_indices_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.028 | 12587008 | 0.141 | 0 |
| tril_default_f64_n1048576 | jax-cuda | 32 | 15 | 0.807 | 16777216 | 0.471 | 0 |
| tril_default_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.345 | 16777216 | 1.102 | 0 |
| tril_default_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.876 | 16777216 | 0.434 | 0 |
| triu_default_f64_n1048576 | jax-cuda | 32 | 15 | 0.785 | 16777216 | 0.485 | 0 |
| triu_default_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.350 | 16777216 | 1.088 | 0 |
| triu_default_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 0.876 | 16777216 | 0.435 | 0 |

## Run metadata

```yaml
schema_version: 1
target_profile: nvidia-gpu
suite_id: gpu/structural
suite_file: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/data/results/nvidia-gpu/gpu/routine_selection/20261108_090627/gpu_structural.yaml
timestamp: '2026-11-08T09:06:27+00:00'
tenferro_rs:
  path: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/extern/tenferro-rs
  commit: 9a850b06155405acc99ecd0648829e9e6d3866ba
  dirty: false
  features:
  - cuda
  resolved_path: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/extern/tenferro-rs
harness:
  path: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark
  commit: 44b5fe8b1b4b339b68f9e9f7dd56d03f6cb08303
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
  hostname: 053e3b59f698
  os: Linux-6.8.0-147-generic-x86_64-with-glibc2.39
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
  - 0.095143
  - 0.091569
  - 0.096425
  - 0.095864
  - 0.104007
  - 0.097255
  - 0.096815
  - 0.094223
  - 0.094362
  - 0.094653
  - 0.094192
  - 0.095203
  - 0.096554
  - 0.094844
  - 0.094822
  median_ms: 0.095143
  bytes_per_second: 1410694722680.5967
  torch_version: 2.11.0+cu128
  cuda_version: '12.8'
  timestamp: '2026-11-08T09:16:31.883538+00:00'
gpu_assignment:
  role: benchmark
  physical_device_index: 0
  device_uuid: GPU-12b45076-79d5-7b84-cb5c-7c6300074c8a
  cuda_visible_devices: GPU-12b45076-79d5-7b84-cb5c-7c6300074c8a
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
    /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/scripts/benchmark_gpu_public_api.py: 4f94c67a046ec57f95823fd01f3f73a710d97745fe6a58a3c6106dfec870d65c
    /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/scripts/gpu_public_api_ops.py: e88823db3f4a1fe43dc13c4743c9d78ece6af76a968724091af318dd8d7f2528
    scripts/run_gpu_public_api.sh: 788b9fde4f8ca4162921baef1203b2e31bb747ec30327a425de3442cf872df52
    data/results/nvidia-gpu/gpu/structural/20261108_090627/fixtures/manifest.json: 27449ceb91a10b308247afb22c9b261ceeafa3bfdbc21adad0a4af193ed9234c
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
        finished_at: '2026-11-08T09:17:58Z'
      pytorch-cuda:
        arm: pytorch-cuda
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-08T09:17:43Z'
      tenferro-cuda-eager:
        arm: tenferro-cuda-eager
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-08T09:17:27Z'
  compile:
    case_wall_seconds: 74.85
    compiling_processes: 1
    modules_compiled: 1
    compile_time_ms: null
    compile_time_scope: 'not measured: no compile interval was recorded; `compile_modules`
      counts the NVRTC modules and the untimed wall is the case wall'
    untimed_wall_seconds: 46.231
```

## Untimed work over the recorded budget

Each row records the wall clock it spent outside its timed region. These cases spent more than the recorded budget ratio (0.2) there. The threshold has no measured basis, so this is a recorded warning: it does not change a cell, its acceptance or the exit status of the run.

| Case | Backend | Warning |
| --- | --- | --- |
| pad_default_f64_n1048576 | tenferro-cuda-eager | pad_default_f64_n1048576 untimed 133.8 ms is 710% of the 18.8 ms timed region (budget 20%) |
| reverse_default_f64_n1048576 | tenferro-cuda-eager | reverse_default_f64_n1048576 untimed 73.6 ms is 549% of the 13.4 ms timed region (budget 20%) |
| concatenate_default_f64_n1048576 | tenferro-cuda-eager | concatenate_default_f64_n1048576 untimed 155.3 ms is 1244% of the 12.5 ms timed region (budget 20%) |
| extract_diag_default_f64_n1048576 | tenferro-cuda-eager | extract_diag_default_f64_n1048576 untimed 9.1 ms is 26% of the 35.2 ms timed region (budget 20%) |
| embed_diag_default_f64_n1024 | tenferro-cuda-eager | embed_diag_default_f64_n1024 untimed 73.0 ms is 669% of the 10.9 ms timed region (budget 20%) |
| tril_default_f64_n1048576 | tenferro-cuda-eager | tril_default_f64_n1048576 untimed 74.0 ms is 564% of the 13.1 ms timed region (budget 20%) |
| triu_default_f64_n1048576 | tenferro-cuda-eager | triu_default_f64_n1048576 untimed 82.4 ms is 495% of the 16.7 ms timed region (budget 20%) |
| take_along_axis_broadcast_indices_f64_n1048576 | tenferro-cuda-eager | take_along_axis_broadcast_indices_f64_n1048576 untimed 46230.8 ms is 151964% of the 30.4 ms timed region (budget 20%) |
| scatter_default_f64_n1048576 | tenferro-cuda-eager | scatter_default_f64_n1048576 untimed 42.7 ms is 135% of the 31.6 ms timed region (budget 20%) |
| cast_default_f64_n1048576 | tenferro-cuda-eager | cast_default_f64_n1048576 untimed 30.6 ms is 63% of the 48.9 ms timed region (budget 20%) |
| dynamic_slice_in_bounds_f64_n1048576 | tenferro-cuda-eager | dynamic_slice_in_bounds_f64_n1048576 untimed 18.2 ms is 44% of the 41.1 ms timed region (budget 20%) |
| gather_axis0_f64_n1048576 | tenferro-cuda-eager | gather_axis0_f64_n1048576 untimed 19.2 ms is 47% of the 41.1 ms timed region (budget 20%) |
| reshape_copy_f64_n1048576 | tenferro-cuda-eager | reshape_copy_f64_n1048576 untimed 25.9 ms is 187% of the 13.8 ms timed region (budget 20%) |
| stack_axis0_f64_n1048576 | tenferro-cuda-eager | stack_axis0_f64_n1048576 untimed 51.6 ms is 172% of the 30.0 ms timed region (budget 20%) |
| pad_default_f64_n1048576 | pytorch-cuda | pad_default_f64_n1048576 untimed 1105.2 ms is 15665% of the 7.1 ms timed region (budget 20%) |
| slice_default_f64_n1048576 | pytorch-cuda | slice_default_f64_n1048576 untimed 484.1 ms is 7310% of the 6.6 ms timed region (budget 20%) |
| reverse_default_f64_n1048576 | pytorch-cuda | reverse_default_f64_n1048576 untimed 910.2 ms is 17974% of the 5.1 ms timed region (budget 20%) |
| concatenate_default_f64_n1048576 | pytorch-cuda | concatenate_default_f64_n1048576 untimed 1993.2 ms is 32506% of the 6.1 ms timed region (budget 20%) |
| extract_diag_default_f64_n1048576 | pytorch-cuda | extract_diag_default_f64_n1048576 untimed 25.0 ms is 56% of the 44.3 ms timed region (budget 20%) |
| embed_diag_default_f64_n1024 | pytorch-cuda | embed_diag_default_f64_n1024 untimed 897.7 ms is 18149% of the 4.9 ms timed region (budget 20%) |
| tril_default_f64_n1048576 | pytorch-cuda | tril_default_f64_n1048576 untimed 898.2 ms is 17347% of the 5.2 ms timed region (budget 20%) |
| triu_default_f64_n1048576 | pytorch-cuda | triu_default_f64_n1048576 untimed 917.4 ms is 17446% of the 5.3 ms timed region (budget 20%) |
| take_along_axis_broadcast_indices_f64_n1048576 | pytorch-cuda | take_along_axis_broadcast_indices_f64_n1048576 untimed 493.9 ms is 3286% of the 15.0 ms timed region (budget 20%) |
| scatter_default_f64_n1048576 | pytorch-cuda | scatter_default_f64_n1048576 untimed 867.6 ms is 7971% of the 10.9 ms timed region (budget 20%) |
| cast_default_f64_n1048576 | pytorch-cuda | cast_default_f64_n1048576 untimed 834.8 ms is 9657% of the 8.6 ms timed region (budget 20%) |
| broadcast_in_dim_f64_n1048576 | pytorch-cuda | broadcast_in_dim_f64_n1048576 untimed 904.8 ms is 25396% of the 3.6 ms timed region (budget 20%) |
| gather_axis0_f64_n1048576 | pytorch-cuda | gather_axis0_f64_n1048576 untimed 455.6 ms is 6510% of the 7.0 ms timed region (budget 20%) |
| reshape_copy_f64_n1048576 | pytorch-cuda | reshape_copy_f64_n1048576 untimed 910.5 ms is 21891% of the 4.2 ms timed region (budget 20%) |
| stack_axis0_f64_n1048576 | pytorch-cuda | stack_axis0_f64_n1048576 untimed 2059.7 ms is 33781% of the 6.1 ms timed region (budget 20%) |
| pad_default_f64_n1048576 | jax-cuda | pad_default_f64_n1048576 untimed 1144.4 ms is 4639% of the 24.7 ms timed region (budget 20%) |
| slice_default_f64_n1048576 | jax-cuda | slice_default_f64_n1048576 untimed 501.1 ms is 1131% of the 44.3 ms timed region (budget 20%) |
| reverse_default_f64_n1048576 | jax-cuda | reverse_default_f64_n1048576 untimed 932.4 ms is 6551% of the 14.2 ms timed region (budget 20%) |
| concatenate_default_f64_n1048576 | jax-cuda | concatenate_default_f64_n1048576 untimed 1836.2 ms is 12974% of the 14.2 ms timed region (budget 20%) |
| extract_diag_default_f64_n1048576 | jax-cuda | extract_diag_default_f64_n1048576 untimed 61.2 ms is 208% of the 29.4 ms timed region (budget 20%) |
| embed_diag_default_f64_n1024 | jax-cuda | embed_diag_default_f64_n1024 untimed 940.7 ms is 6523% of the 14.4 ms timed region (budget 20%) |
| tril_default_f64_n1048576 | jax-cuda | tril_default_f64_n1048576 untimed 936.5 ms is 7709% of the 12.1 ms timed region (budget 20%) |
| triu_default_f64_n1048576 | jax-cuda | triu_default_f64_n1048576 untimed 928.6 ms is 7681% of the 12.1 ms timed region (budget 20%) |
| take_along_axis_broadcast_indices_f64_n1048576 | jax-cuda | take_along_axis_broadcast_indices_f64_n1048576 untimed 505.1 ms is 1809% of the 27.9 ms timed region (budget 20%) |
| scatter_default_f64_n1048576 | jax-cuda | scatter_default_f64_n1048576 untimed 886.3 ms is 2375% of the 37.3 ms timed region (budget 20%) |
| cast_default_f64_n1048576 | jax-cuda | cast_default_f64_n1048576 untimed 868.5 ms is 1672% of the 51.9 ms timed region (budget 20%) |
| broadcast_in_dim_f64_n1048576 | jax-cuda | broadcast_in_dim_f64_n1048576 untimed 952.1 ms is 4502% of the 21.2 ms timed region (budget 20%) |
| dynamic_slice_in_bounds_f64_n1048576 | jax-cuda | dynamic_slice_in_bounds_f64_n1048576 untimed 304.8 ms is 1005% of the 30.3 ms timed region (budget 20%) |
| gather_axis0_f64_n1048576 | jax-cuda | gather_axis0_f64_n1048576 untimed 505.2 ms is 995% of the 50.8 ms timed region (budget 20%) |
| stack_axis0_f64_n1048576 | jax-cuda | stack_axis0_f64_n1048576 untimed 1872.5 ms is 12661% of the 14.8 ms timed region (budget 20%) |