# gpu/isolated_ops: CUDA public API benchmark

Measured cells: **62**. Unsupported cells: **4**, including **3** numerically verified but timing-excluded cells. Unsupported cells do not count as measured coverage.

Logical traffic: this run measured a plain device copy of 67108864 bytes at **1410.1 GB/s** on this device. Every record carries the logical bytes the case must move per operation and the ratio of the case's implied rate to that copy rate, in the sampling table below. Logical bytes are the declared array sizes, not device DRAM traffic: a case whose operands stay in cache moves less DRAM than its arrays suggest, so a ratio above 1 is recorded, not rejected.

Raw run: `data/results/nvidia-gpu/gpu/isolated_ops/20261108_110139/`.

Isolated public APIs and explicit unsupported timing contracts. Issue 155.

This run did not compile: no NVRTC module was built in any of its arms. That describes this run only, and does not show that the CUDA artifact cache was warm.

Values are median (IQR) **µs per operation**. Samples time batches of declared operations plus device synchronization; retained output memory is limited to 256 MiB. Batch counts and raw durations are preserved in JSONL.

Cases marked † (**6** of 33) were sampled with a different number of operations per sample on different arms (see Ops/sample in the sampling table). Each value is still per operation, but the arms amortise launch and synchronization over different batch sizes, so a ratio between their columns is approximate and is not presented as a like-for-like comparison.

Setup, input transfers, library/plan initialization, JIT compilation, validation downloads and output destruction are outside timing. Intrinsic output allocation is inside timing; output-reuse rows instead use preallocated destinations across samples. Accumulating destinations are restored from prepared GPU seeds and synchronized before every batch clock, including initialization and calibration. Direct and eager each use one entered session across samples; direct FFT uses cached `FftExecutor`. JAX eager and compiled JIT are separate arms.

All measured output elements are checked against independently generated NumPy/SciPy reference results before and after sampling. Unsupported rows include unavailable APIs and APIs whose internal setup cannot be excluded; these are not performance measurements. Short batches limited by output memory remain identifiable in the sampling table.

| Case | tenferro-cuda-eager | pytorch-cuda |
| --- | ---: | ---: |
| add_f64_n1048576 | 88.584 (0.297) | 12.101 (0.079) |
| sub_f64_n1048576 | 88.500 (0.454) | 12.311 (0.079) |
| mul_f64_n1048576 | 88.538 (0.421) | 12.285 (0.157) |
| div_f64_n1048576 | 88.658 (0.401) | 13.604 (0.151) |
| rem_f64_n1048576 | 120.569 (0.784) | 12.381 (0.149) |
| pow_f64_n1048576 | 121.934 (0.930) | 29.042 (0.026) |
| maximum_f64_n1048576 | 107.950 (0.620) | 12.336 (0.224) |
| minimum_f64_n1048576 | 108.389 (0.613) | 12.457 (0.181) |
| neg_f64_n1048576 | 52.607 (0.352) | 7.667 (0.033) |
| abs_f64_n1048576 | 52.531 (0.493) | 7.738 (0.060) |
| sign_f64_n1048576 | 52.741 (0.487) | 7.648 (0.048) |
| conj_c64_n1048576 | 85.528 (0.226) | 25.178 (0.414) |
| exp_f64_n1048576 | 52.980 (0.314) | 8.092 (0.052) |
| expm1_f64_n1048576 | 54.454 (0.410) | 9.792 (0.051) |
| log_f64_n1048576 | 57.107 (0.433) | 12.483 (0.066) |
| log1p_f64_n1048576 | 57.214 (0.188) | 12.999 (0.067) |
| erf_f64_n1048576 | 59.064 (0.139) | 14.249 (0.046) |
| sin_f64_n1048576 | 55.801 (0.335) | 11.013 (0.030) |
| cos_f64_n1048576 | 55.794 (0.290) | 11.051 (0.047) |
| tanh_f64_n1048576 | 53.192 (0.534) | 8.137 (0.038) |
| sqrt_f64_n1048576 | 52.799 (9.245) | 7.728 (0.026) |
| rsqrt_f64_n1048576 | 52.756 (0.395) | 7.807 (0.066) |
| compare_f64_n1048576 † | 81.005 (0.342) | 6.935 (0.015) |
| where_select_f64_n1048576 | 114.416 (0.502) | 14.287 (0.104) |
| clamp_f64_n1048576 † | 147.758 (4.961) | 10.180 (0.033) |
| reduce_sum_f64_n1048576 † | 68.956 (0.516) | 6.992 (0.007) |
| reduce_prod_f64_n1048576 † | 68.311 (0.403) | 7.073 (0.008) |
| reduce_max_f64_n1048576 † | 68.122 (0.369) | 7.179 (0.005) |
| reduce_min_f64_n1048576 † | 68.333 (0.372) | 7.168 (0.005) |
| reduce_sum_squares_f64_n1048576 | 68.519 (0.550) | unsupported |
| reduce_mean_f64_n1048576 | unsupported | 6.962 (0.012) |
| scale_real_f64_n1048576 | unsupported | 9.445 (0.038) |
| scale_complex_c64_n1048576 | unsupported | 25.648 (0.438) |

## Sampling

| Case | Backend | Ops/sample | Samples | Median batch ms | Bytes/op | Copy ratio | Verification error/tolerance |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| abs_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.248 | 16777216 | 1.538 | 0 |
| abs_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.681 | 16777216 | 0.226 | 0 |
| add_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.387 | 25165824 | 1.475 | 0 |
| add_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.835 | 25165824 | 0.201 | 0 |
| clamp_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.326 | 16777232 | 1.169 | 0 |
| clamp_f64_n1048576 | tenferro-cuda-eager | 16 | 15 | 2.364 | 16777232 | 0.081 | 0 |
| compare_f64_n1048576 | pytorch-cuda | 64 | 15 | 0.444 | 17825792 | 1.823 | 0 |
| compare_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.592 | 17825792 | 0.156 | 0 |
| conj_c64_n1048576 | pytorch-cuda | 16 | 15 | 0.403 | 33554432 | 0.945 | 0 |
| conj_c64_n1048576 | tenferro-cuda-eager | 16 | 15 | 1.368 | 33554432 | 0.278 | 0 |
| cos_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.354 | 16777216 | 1.077 | 5.905e-08 |
| cos_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.785 | 16777216 | 0.213 | 5.905e-08 |
| div_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.435 | 25165824 | 1.312 | 0 |
| div_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.837 | 25165824 | 0.201 | 0 |
| erf_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.456 | 16777216 | 0.835 | 8.536e-08 |
| erf_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.890 | 16777216 | 0.201 | 8.536e-08 |
| exp_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.259 | 16777216 | 1.470 | 1.093e-07 |
| exp_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.695 | 16777216 | 0.225 | 1.093e-07 |
| expm1_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.313 | 16777216 | 1.215 | 7.39e-08 |
| expm1_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.743 | 16777216 | 0.218 | 7.39e-08 |
| log1p_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.416 | 16777216 | 0.915 | 1.106e-07 |
| log1p_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.831 | 16777216 | 0.208 | 1.106e-07 |
| log_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.399 | 16777216 | 0.953 | 4.308e-08 |
| log_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.827 | 16777216 | 0.208 | 4.308e-08 |
| maximum_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.395 | 25165824 | 1.447 | 0 |
| maximum_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.454 | 25165824 | 0.165 | 0 |
| minimum_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.399 | 25165824 | 1.433 | 0 |
| minimum_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.468 | 25165824 | 0.165 | 0 |
| mul_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.393 | 25165824 | 1.453 | 0 |
| mul_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.833 | 25165824 | 0.202 | 0 |
| neg_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.245 | 16777216 | 1.552 | 0 |
| neg_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.683 | 16777216 | 0.226 | 0 |
| pow_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.929 | 25165824 | 0.615 | 1.48e-07 |
| pow_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.902 | 25165824 | 0.146 | 1.48e-07 |
| reduce_max_f64_n1048576 | pytorch-cuda | 512 | 15 | 3.676 | 8396800 | 0.829 | 0 |
| reduce_max_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.180 | 8396800 | 0.087 | 0 |
| reduce_mean_f64_n1048576 | pytorch-cuda | 512 | 15 | 3.565 | 8396800 | 0.855 | 3.9e-09 |
| reduce_mean_f64_n1048576 | tenferro-cuda-eager | — | — | — | 8396800 | unavailable | CUDA implementation checked numerically only: API creates and uploads scalar constants per call; public API cannot hoist this setup outside the timer. |
| reduce_min_f64_n1048576 | pytorch-cuda | 512 | 15 | 3.670 | 8396800 | 0.831 | 0 |
| reduce_min_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.187 | 8396800 | 0.087 | 0 |
| reduce_prod_f64_n1048576 | pytorch-cuda | 512 | 15 | 3.622 | 8396800 | 0.842 | 4.216e-06 |
| reduce_prod_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.186 | 8396800 | 0.087 | 3.55e-06 |
| reduce_sum_f64_n1048576 | pytorch-cuda | 512 | 15 | 3.580 | 8396800 | 0.852 | 2.961e-06 |
| reduce_sum_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.207 | 8396800 | 0.086 | 3.391e-06 |
| reduce_sum_squares_f64_n1048576 | pytorch-cuda | — | — | — | 8396800 | unavailable | No equivalent single axis-wise sum-of-squares API; multiplying then reducing is not used as a synthetic reference arm. |
| reduce_sum_squares_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.193 | 8396800 | 0.087 | 3.294e-07 |
| rem_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.396 | 25165824 | 1.441 | 0 |
| rem_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.858 | 25165824 | 0.148 | 0 |
| rsqrt_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.250 | 16777216 | 1.524 | 6.504e-08 |
| rsqrt_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.688 | 16777216 | 0.226 | 6.504e-08 |
| scale_complex_c64_n1048576 | pytorch-cuda | 16 | 15 | 0.410 | 33554448 | 0.928 | 0 |
| scale_complex_c64_n1048576 | tenferro-cuda-eager | — | — | — | 33554448 | unavailable | Eager scale host-placement bug fixed by upstream 917d5341. API now uploads its internal scalar through constant_from_host; no public preuploaded factor parameter. Numerical validation only; internal scalar transfer excludes standard timing. |
| scale_real_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.302 | 16777224 | 1.260 | 0 |
| scale_real_f64_n1048576 | tenferro-cuda-eager | — | — | — | 16777224 | unavailable | Eager scale host-placement bug fixed by upstream 917d5341. API now uploads its internal scalar through constant_from_host; no public preuploaded factor parameter. Numerical validation only; internal scalar transfer excludes standard timing. |
| sign_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.245 | 16777216 | 1.556 | 0 |
| sign_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.688 | 16777216 | 0.226 | 0 |
| sin_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.352 | 16777216 | 1.080 | 4.153e-08 |
| sin_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.786 | 16777216 | 0.213 | 4.153e-08 |
| sqrt_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.247 | 16777216 | 1.540 | 0 |
| sqrt_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.690 | 16777216 | 0.225 | 0 |
| sub_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.394 | 25165824 | 1.450 | 0 |
| sub_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 2.832 | 25165824 | 0.202 | 0 |
| tanh_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.260 | 16777216 | 1.462 | 4.437e-08 |
| tanh_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 1.702 | 16777216 | 0.224 | 4.437e-08 |
| where_select_f64_n1048576 | pytorch-cuda | 32 | 15 | 0.457 | 26214400 | 1.301 | 0 |
| where_select_f64_n1048576 | tenferro-cuda-eager | 32 | 15 | 3.661 | 26214400 | 0.162 | 0 |

## Run metadata

```yaml
schema_version: 1
target_profile: nvidia-gpu
suite_id: gpu/isolated_ops
suite_file: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/data/results/nvidia-gpu/gpu/routine_selection/20261108_110139/gpu_isolated_ops.yaml
timestamp: '2026-11-08T11:01:39+00:00'
tenferro_rs:
  path: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/extern/tenferro-rs
  commit: 9a850b06155405acc99ecd0648829e9e6d3866ba
  dirty: false
  features:
  - cuda
  resolved_path: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/extern/tenferro-rs
harness:
  path: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark
  commit: 41f2cdf8bbdf08664bbffe7f2f31efd86ecea146
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
  hostname: 0db5d65f81dc
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
    GPU_RUNTIME_SHA256: 7cac8477f65d07d1a9f43b33d9f8fc8f336e03e4302840fc3eea4b16076e41cf
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
    backend: cpu
    file: /opt/tenferro-benchmark-ci/venv/lib/python3.12/site-packages/jaxlib/__init__.py
    provider: xla_cpu
    dot_backend: xla_cpu
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
  - 0.093431
  - 0.090306
  - 0.098258
  - 0.096385
  - 0.094172
  - 0.095805
  - 0.095183
  - 0.092939
  - 0.096136
  - 0.095443
  - 0.09351
  - 0.095434
  - 0.094913
  - 0.094331
  - 0.095555
  median_ms: 0.095183
  bytes_per_second: 1410101887942.1746
  torch_version: 2.11.0+cu128
  cuda_version: '12.8'
  timestamp: '2026-11-08T11:09:03.949627+00:00'
gpu_assignment:
  role: benchmark
  physical_device_index: 0
  device_uuid: GPU-12b45076-79d5-7b84-cb5c-7c6300074c8a
  cuda_visible_devices: GPU-12b45076-79d5-7b84-cb5c-7c6300074c8a
  process_device_ordinal: 0
  verification_only: false
collection:
  command: BENCHMARK_TARGET_PROFILE=nvidia-gpu GPU_BENCH_DEVICE=0 BENCH_RUNS=15 BENCH_WARMUPS=3
    ./scripts/run_gpu_public_api.sh isolated_ops
  devcontainer_config: .devcontainer/cuda/devcontainer.json
  expected_cases: 33
  selected_cases: 33
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
    /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/scripts/benchmark_gpu_public_api.py: 32aa2ab45453502fc16479f429191aa24f867c8d5ad5f59d5a2752a3da413ee5
    /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/scripts/gpu_public_api_ops.py: e88823db3f4a1fe43dc13c4743c9d78ece6af76a968724091af318dd8d7f2528
    scripts/run_gpu_public_api.sh: 257228952e48c61fa1714d0cb2d6fdfdb9b0cf3898f59307ba9e72e513aa1b50
    data/results/nvidia-gpu/gpu/isolated_ops/20261108_110139/fixtures/manifest.json: 3c1a75f9c57996b228c04da1d4cbf34e4137a074e44ddca9f1c37e5824580bd8
    target/release/benchmark_gpu_public_api: 62f92e0c72d6fb46531ee8106f112567dfd32ddb482255a7cc41ba6819929786
  reference_numpy_version: 2.4.2
  measurement_cpu:
    policy: 'disabled (GPU_BENCH_CPU_SELECT unset): arms run on the allocation unpinned'
    arms:
      pytorch-cuda:
        arm: pytorch-cuda
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-08T11:09:14Z'
      tenferro-cuda-eager:
        arm: tenferro-cuda-eager
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-08T11:09:09Z'
  compile:
    case_wall_seconds: 5.458
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
| add_f64_n1048576 | tenferro-cuda-eager | add_f64_n1048576 untimed 143.1 ms is 336% of the 42.5 ms timed region (budget 20%) |
| sub_f64_n1048576 | tenferro-cuda-eager | sub_f64_n1048576 untimed 67.1 ms is 158% of the 42.5 ms timed region (budget 20%) |
| mul_f64_n1048576 | tenferro-cuda-eager | mul_f64_n1048576 untimed 80.8 ms is 190% of the 42.5 ms timed region (budget 20%) |
| div_f64_n1048576 | tenferro-cuda-eager | div_f64_n1048576 untimed 80.6 ms is 189% of the 42.6 ms timed region (budget 20%) |
| rem_f64_n1048576 | tenferro-cuda-eager | rem_f64_n1048576 untimed 82.5 ms is 143% of the 57.9 ms timed region (budget 20%) |
| pow_f64_n1048576 | tenferro-cuda-eager | pow_f64_n1048576 untimed 89.7 ms is 153% of the 58.5 ms timed region (budget 20%) |
| maximum_f64_n1048576 | tenferro-cuda-eager | maximum_f64_n1048576 untimed 81.3 ms is 157% of the 51.8 ms timed region (budget 20%) |
| minimum_f64_n1048576 | tenferro-cuda-eager | minimum_f64_n1048576 untimed 81.4 ms is 157% of the 52.0 ms timed region (budget 20%) |
| neg_f64_n1048576 | tenferro-cuda-eager | neg_f64_n1048576 untimed 71.6 ms is 284% of the 25.2 ms timed region (budget 20%) |
| abs_f64_n1048576 | tenferro-cuda-eager | abs_f64_n1048576 untimed 59.9 ms is 237% of the 25.2 ms timed region (budget 20%) |
| sign_f64_n1048576 | tenferro-cuda-eager | sign_f64_n1048576 untimed 59.5 ms is 235% of the 25.3 ms timed region (budget 20%) |
| conj_c64_n1048576 | tenferro-cuda-eager | conj_c64_n1048576 untimed 92.1 ms is 448% of the 20.5 ms timed region (budget 20%) |
| exp_f64_n1048576 | tenferro-cuda-eager | exp_f64_n1048576 untimed 71.9 ms is 282% of the 25.5 ms timed region (budget 20%) |
| expm1_f64_n1048576 | tenferro-cuda-eager | expm1_f64_n1048576 untimed 61.5 ms is 235% of the 26.2 ms timed region (budget 20%) |
| log_f64_n1048576 | tenferro-cuda-eager | log_f64_n1048576 untimed 61.4 ms is 224% of the 27.5 ms timed region (budget 20%) |
| log1p_f64_n1048576 | tenferro-cuda-eager | log1p_f64_n1048576 untimed 62.9 ms is 229% of the 27.4 ms timed region (budget 20%) |
| erf_f64_n1048576 | tenferro-cuda-eager | erf_f64_n1048576 untimed 67.1 ms is 236% of the 28.4 ms timed region (budget 20%) |
| sin_f64_n1048576 | tenferro-cuda-eager | sin_f64_n1048576 untimed 61.1 ms is 223% of the 27.5 ms timed region (budget 20%) |
| cos_f64_n1048576 | tenferro-cuda-eager | cos_f64_n1048576 untimed 60.7 ms is 226% of the 26.8 ms timed region (budget 20%) |
| tanh_f64_n1048576 | tenferro-cuda-eager | tanh_f64_n1048576 untimed 63.2 ms is 248% of the 25.5 ms timed region (budget 20%) |
| sqrt_f64_n1048576 | tenferro-cuda-eager | sqrt_f64_n1048576 untimed 60.6 ms is 218% of the 27.8 ms timed region (budget 20%) |
| rsqrt_f64_n1048576 | tenferro-cuda-eager | rsqrt_f64_n1048576 untimed 64.5 ms is 255% of the 25.3 ms timed region (budget 20%) |
| compare_f64_n1048576 | tenferro-cuda-eager | compare_f64_n1048576 untimed 45.8 ms is 118% of the 38.9 ms timed region (budget 20%) |
| where_select_f64_n1048576 | tenferro-cuda-eager | where_select_f64_n1048576 untimed 82.0 ms is 149% of the 55.0 ms timed region (budget 20%) |
| clamp_f64_n1048576 | tenferro-cuda-eager | clamp_f64_n1048576 untimed 73.2 ms is 204% of the 35.9 ms timed region (budget 20%) |
| reduce_sum_f64_n1048576 | tenferro-cuda-eager | reduce_sum_f64_n1048576 untimed 8.4 ms is 25% of the 33.1 ms timed region (budget 20%) |
| reduce_prod_f64_n1048576 | tenferro-cuda-eager | reduce_prod_f64_n1048576 untimed 7.2 ms is 21% of the 33.7 ms timed region (budget 20%) |
| reduce_max_f64_n1048576 | tenferro-cuda-eager | reduce_max_f64_n1048576 untimed 8.6 ms is 26% of the 32.8 ms timed region (budget 20%) |
| reduce_min_f64_n1048576 | tenferro-cuda-eager | reduce_min_f64_n1048576 untimed 7.1 ms is 22% of the 32.8 ms timed region (budget 20%) |
| reduce_sum_squares_f64_n1048576 | tenferro-cuda-eager | reduce_sum_squares_f64_n1048576 untimed 7.1 ms is 22% of the 32.9 ms timed region (budget 20%) |
| add_f64_n1048576 | pytorch-cuda | add_f64_n1048576 untimed 184.1 ms is 3187% of the 5.8 ms timed region (budget 20%) |
| sub_f64_n1048576 | pytorch-cuda | sub_f64_n1048576 untimed 65.9 ms is 1116% of the 5.9 ms timed region (budget 20%) |
| mul_f64_n1048576 | pytorch-cuda | mul_f64_n1048576 untimed 60.9 ms is 1033% of the 5.9 ms timed region (budget 20%) |
| div_f64_n1048576 | pytorch-cuda | div_f64_n1048576 untimed 66.2 ms is 1014% of the 6.5 ms timed region (budget 20%) |
| rem_f64_n1048576 | pytorch-cuda | rem_f64_n1048576 untimed 82.2 ms is 1386% of the 5.9 ms timed region (budget 20%) |
| pow_f64_n1048576 | pytorch-cuda | pow_f64_n1048576 untimed 86.7 ms is 622% of the 13.9 ms timed region (budget 20%) |
| maximum_f64_n1048576 | pytorch-cuda | maximum_f64_n1048576 untimed 70.8 ms is 1192% of the 5.9 ms timed region (budget 20%) |
| minimum_f64_n1048576 | pytorch-cuda | minimum_f64_n1048576 untimed 54.7 ms is 913% of the 6.0 ms timed region (budget 20%) |
| neg_f64_n1048576 | pytorch-cuda | neg_f64_n1048576 untimed 62.4 ms is 1696% of the 3.7 ms timed region (budget 20%) |
| abs_f64_n1048576 | pytorch-cuda | abs_f64_n1048576 untimed 57.8 ms is 1549% of the 3.7 ms timed region (budget 20%) |
| sign_f64_n1048576 | pytorch-cuda | sign_f64_n1048576 untimed 55.6 ms is 1513% of the 3.7 ms timed region (budget 20%) |
| conj_c64_n1048576 | pytorch-cuda | conj_c64_n1048576 untimed 95.5 ms is 1580% of the 6.0 ms timed region (budget 20%) |
| exp_f64_n1048576 | pytorch-cuda | exp_f64_n1048576 untimed 71.6 ms is 1838% of the 3.9 ms timed region (budget 20%) |
| expm1_f64_n1048576 | pytorch-cuda | expm1_f64_n1048576 untimed 56.6 ms is 1206% of the 4.7 ms timed region (budget 20%) |
| log_f64_n1048576 | pytorch-cuda | log_f64_n1048576 untimed 59.3 ms is 990% of the 6.0 ms timed region (budget 20%) |
| log1p_f64_n1048576 | pytorch-cuda | log1p_f64_n1048576 untimed 57.2 ms is 916% of the 6.2 ms timed region (budget 20%) |
| erf_f64_n1048576 | pytorch-cuda | erf_f64_n1048576 untimed 61.5 ms is 900% of the 6.8 ms timed region (budget 20%) |
| sin_f64_n1048576 | pytorch-cuda | sin_f64_n1048576 untimed 57.2 ms is 1080% of the 5.3 ms timed region (budget 20%) |
| cos_f64_n1048576 | pytorch-cuda | cos_f64_n1048576 untimed 59.1 ms is 1112% of the 5.3 ms timed region (budget 20%) |
| tanh_f64_n1048576 | pytorch-cuda | tanh_f64_n1048576 untimed 51.4 ms is 1316% of the 3.9 ms timed region (budget 20%) |
| sqrt_f64_n1048576 | pytorch-cuda | sqrt_f64_n1048576 untimed 56.5 ms is 1524% of the 3.7 ms timed region (budget 20%) |
| rsqrt_f64_n1048576 | pytorch-cuda | rsqrt_f64_n1048576 untimed 69.7 ms is 1856% of the 3.8 ms timed region (budget 20%) |
| compare_f64_n1048576 | pytorch-cuda | compare_f64_n1048576 untimed 53.6 ms is 804% of the 6.7 ms timed region (budget 20%) |
| where_select_f64_n1048576 | pytorch-cuda | where_select_f64_n1048576 untimed 75.3 ms is 1095% of the 6.9 ms timed region (budget 20%) |
| clamp_f64_n1048576 | pytorch-cuda | clamp_f64_n1048576 untimed 50.1 ms is 1027% of the 4.9 ms timed region (budget 20%) |
| reduce_sum_f64_n1048576 | pytorch-cuda | reduce_sum_f64_n1048576 untimed 27.9 ms is 52% of the 53.7 ms timed region (budget 20%) |
| reduce_prod_f64_n1048576 | pytorch-cuda | reduce_prod_f64_n1048576 untimed 23.7 ms is 44% of the 54.3 ms timed region (budget 20%) |
| reduce_max_f64_n1048576 | pytorch-cuda | reduce_max_f64_n1048576 untimed 25.4 ms is 46% of the 55.2 ms timed region (budget 20%) |
| reduce_min_f64_n1048576 | pytorch-cuda | reduce_min_f64_n1048576 untimed 24.8 ms is 45% of the 55.1 ms timed region (budget 20%) |
| reduce_mean_f64_n1048576 | pytorch-cuda | reduce_mean_f64_n1048576 untimed 29.1 ms is 54% of the 53.5 ms timed region (budget 20%) |
| scale_real_f64_n1048576 | pytorch-cuda | scale_real_f64_n1048576 untimed 59.7 ms is 1317% of the 4.5 ms timed region (budget 20%) |
| scale_complex_c64_n1048576 | pytorch-cuda | scale_complex_c64_n1048576 untimed 103.4 ms is 1681% of the 6.2 ms timed region (budget 20%) |