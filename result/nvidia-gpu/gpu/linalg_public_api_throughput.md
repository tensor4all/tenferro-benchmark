# gpu/linalg_public_api_throughput: CUDA public API benchmark

Measured cells: **58**. Unsupported cells: **6**, including **0** numerically verified but timing-excluded cells. Unsupported cells do not count as measured coverage.

Logical traffic: this run measured a plain device copy of 67108864 bytes at **1412.8 GB/s** on this device. Every record carries the logical bytes the case must move per operation and the ratio of the case's implied rate to that copy rate, in the sampling table below. Logical bytes are the declared array sizes, not device DRAM traffic: a case whose operands stay in cache moves less DRAM than its arrays suggest, so a ratio above 1 is recorded, not rejected.

Raw run: `data/results/nvidia-gpu/gpu/linalg_public_api_throughput/20261128_082346/`.

Large-shape view of the CUDA public linalg API surface: the 82 cases whose largest matrix dimension is at least 256. `large` is a shape class, not a measured classification: no row is asserted to be compute- or bandwidth-bound, and a memory-bound row takes the same time here as at a small shape (`householder_r_default_f32` measured 34.698 µs at 512x256 and 35.223 µs at 64x32). These rows are never comparable with, or averaged together with, the per-call overhead rows of `gpu/linalg_public_api_latency` (tenferro-rs #2087). Rows here are not lower than on pages published before this harness change for any timing reason: the sample clock starts inside `measure`, so engine construction was never part of a case median. What changed is repeated **untimed setup** work, which was removed: the CUDA engine is opened once per arm and held for the whole run instead of once per case, which shortens the untimed wall clock around the cases and moves no operation median on this page. The same problems, protocol and backends are split by shape class between the two suites. Issue 155.

This run compiled **1 NVRTC modules** in 1 of its arms, because the CUDA artifact cache started cold. The untimed wall of the cases that compiled is **2.8 s** of its 30.6 s of recorded case wall clock, and that wall also holds their fixture read, transfers and verification, so it is not a compile time.

Values are median (IQR) **µs per operation**. Samples time batches of declared operations plus device synchronization; retained output memory is limited to 256 MiB. Batch counts and raw durations are preserved in JSONL.

Cases marked † (**4** of 16) were sampled with a different number of operations per sample on different arms (see Ops/sample in the sampling table). Each value is still per operation, but the arms amortise launch and synchronization over different batch sizes, so a ratio between their columns is approximate and is not presented as a like-for-like comparison.

Setup, input transfers, library/plan initialization, JIT compilation, validation downloads and output destruction are outside timing. Intrinsic output allocation is inside timing; output-reuse rows instead use preallocated destinations across samples. Accumulating destinations are restored from prepared GPU seeds and synchronized before every batch clock, including initialization and calibration. Direct and eager each use one entered session across samples; direct FFT uses cached `FftExecutor`. JAX eager and compiled JIT are separate arms.

All measured output elements are checked against independently generated NumPy/SciPy reference results before and after sampling. Unsupported rows include unavailable APIs and APIs whose internal setup cannot be excluded; these are not performance measurements. Short batches limited by output memory remain identifiable in the sampling table.

| Case | tenferro-cuda-direct | tenferro-cuda-eager | pytorch-cuda | jax-cuda |
| --- | ---: | ---: | ---: | ---: |
| cholesky_f64_512x512 | 319.998 (1.059) | 331.107 (1.982) | 318.529 (1.328) | 307.573 (1.702) |
| svdvals_f64_512x512 | 77928.659 (53.226) | 77780.536 (24.496) | 84884.689 (41.792) | 78091.846 (42.664) |
| eigvalsh_f64_512x512 | 5590.995 (19.139) | 5502.442 (13.800) | 5614.781 (20.475) | 6102.054 (22.865) |
| pinv_f64_512x512 | 78445.837 (67.768) | 78226.757 (33.510) | 85173.434 (27.837) | 78464.586 (40.746) |
| inv_f64_512x512 | 1901.938 (5.210) | 1919.214 (6.677) | 1808.521 (10.521) | 1825.793 (7.283) |
| det_f64_512x512 | 1571.996 (7.834) | 1577.549 (6.592) | 1320.382 (8.959) | 1367.849 (12.711) |
| norm_f64_512x512 † | 17.039 (0.041) | 16.756 (0.019) | 12.496 (0.787) | 48.050 (2.431) |
| triangular_solve_f64_512x512 † | 195.761 (0.619) | 212.320 (0.248) | 309.557 (1.525) | 214.261 (0.790) |
| svd_full_tall_f64_512x256 | 19649.564 (59.105) | 19627.992 (18.928) | 20030.217 (60.836) | 19728.583 (22.513) |
| slogdet_positive_f64_256x256 | 643.478 (59.835) | 643.145 (2.775) | 566.716 (2.282) | 561.245 (3.141) |
| householder_qr_tall_f64_512x256 | 1328.725 (12.393) | 1358.470 (4.094) | 1349.376 (12.120) | 1406.271 (10.886) |
| householder_from_factors_default_f64_512x256 | 1472.596 (13.901) | 1488.220 (7.659) | unsupported | unsupported |
| householder_append_default_f64_512x256 | 780.059 (5.154) | 802.598 (4.312) | unsupported | unsupported |
| householder_r_default_f64_512x256 † | 15.245 (0.103) | 17.123 (0.171) | 6.721 (0.494) | 33.390 (0.909) |
| householder_q_columns_prefix_f64_512x256 † | 158.027 (0.380) | 156.763 (0.127) | 70.165 (9.734) | 108.548 (2.764) |
| lu_tall_f64_s256 | 802.360 (2.583) | 790.852 (1.681) | unsupported | unsupported |

## Sampling

| Case | Backend | Ops/sample | Samples | Median batch ms | Bytes/op | Copy ratio | Verification error/tolerance |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| cholesky_f64_512x512 | jax-cuda | 8 | 15 | 2.461 | 4194304 | 0.010 | 4.418e-07 |
| cholesky_f64_512x512 | pytorch-cuda | 8 | 15 | 2.548 | 4194304 | 0.009 | 4.418e-07 |
| cholesky_f64_512x512 | tenferro-cuda-direct | 8 | 15 | 2.560 | 4194304 | 0.009 | 4.418e-07 |
| cholesky_f64_512x512 | tenferro-cuda-eager | 8 | 15 | 2.649 | 4194304 | 0.009 | 4.418e-07 |
| det_f64_512x512 | jax-cuda | 2 | 15 | 2.736 | 2097160 | 0.001 | 1.426e-05 |
| det_f64_512x512 | pytorch-cuda | 2 | 15 | 2.641 | 2097160 | 0.001 | 1.369e-05 |
| det_f64_512x512 | tenferro-cuda-direct | 2 | 15 | 3.144 | 2097160 | 0.001 | 1.426e-05 |
| det_f64_512x512 | tenferro-cuda-eager | 2 | 15 | 3.155 | 2097160 | 0.001 | 1.426e-05 |
| eigvalsh_f64_512x512 | jax-cuda | 1 | 15 | 6.102 | 2101248 | 0.000 | 1.767e-06 |
| eigvalsh_f64_512x512 | pytorch-cuda | 1 | 15 | 5.615 | 2101248 | 0.000 | 1.767e-06 |
| eigvalsh_f64_512x512 | tenferro-cuda-direct | 1 | 15 | 5.591 | 2101248 | 0.000 | 1.767e-06 |
| eigvalsh_f64_512x512 | tenferro-cuda-eager | 1 | 15 | 5.502 | 2101248 | 0.000 | 1.767e-06 |
| householder_append_default_f64_512x256 | jax-cuda | — | — | — | 3147776 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_f64_512x256 | pytorch-cuda | — | — | — | 3147776 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_append_default_f64_512x256 | tenferro-cuda-direct | 4 | 15 | 3.120 | 3147776 | 0.003 | 9.992e-07 |
| householder_append_default_f64_512x256 | tenferro-cuda-eager | 4 | 15 | 3.210 | 3147776 | 0.003 | 9.992e-07 |
| householder_from_factors_default_f64_512x256 | jax-cuda | — | — | — | 3672064 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_from_factors_default_f64_512x256 | pytorch-cuda | — | — | — | 3672064 | unavailable | No verified single public QR state initialization/update counterpart; factor multiplication or full refactorization is not substituted. |
| householder_from_factors_default_f64_512x256 | tenferro-cuda-direct | 2 | 15 | 2.945 | 3672064 | 0.002 | 8.882e-07 |
| householder_from_factors_default_f64_512x256 | tenferro-cuda-eager | 2 | 15 | 2.976 | 3672064 | 0.002 | 8.882e-07 |
| householder_q_columns_prefix_f64_512x256 | jax-cuda | 32 | 15 | 3.474 | 1081344 | 0.007 | 1.136e-07 |
| householder_q_columns_prefix_f64_512x256 | pytorch-cuda | 32 | 15 | 2.245 | 1081344 | 0.011 | 1.136e-07 |
| householder_q_columns_prefix_f64_512x256 | tenferro-cuda-direct | 16 | 15 | 2.528 | 1081344 | 0.005 | 1.136e-07 |
| householder_q_columns_prefix_f64_512x256 | tenferro-cuda-eager | 16 | 15 | 2.508 | 1081344 | 0.005 | 1.136e-07 |
| householder_qr_tall_f64_512x256 | jax-cuda | 2 | 15 | 2.813 | 2099200 | 0.001 | 1.887e-06 |
| householder_qr_tall_f64_512x256 | pytorch-cuda | 2 | 15 | 2.699 | 2099200 | 0.001 | 1.887e-06 |
| householder_qr_tall_f64_512x256 | tenferro-cuda-direct | 2 | 15 | 2.657 | 2099200 | 0.001 | 9.992e-07 |
| householder_qr_tall_f64_512x256 | tenferro-cuda-eager | 2 | 15 | 2.717 | 2099200 | 0.001 | 9.992e-07 |
| householder_r_default_f64_512x256 | jax-cuda | 64 | 15 | 2.137 | 1572864 | 0.033 | 9.976e-07 |
| householder_r_default_f64_512x256 | pytorch-cuda | 32 | 15 | 0.215 | 1572864 | 0.166 | 9.976e-07 |
| householder_r_default_f64_512x256 | tenferro-cuda-direct | 128 | 15 | 1.951 | 1572864 | 0.073 | 9.976e-07 |
| householder_r_default_f64_512x256 | tenferro-cuda-eager | 128 | 15 | 2.192 | 1572864 | 0.065 | 9.976e-07 |
| inv_f64_512x512 | jax-cuda | 2 | 15 | 3.652 | 4194304 | 0.002 | 1.85e-06 |
| inv_f64_512x512 | pytorch-cuda | 2 | 15 | 3.617 | 4194304 | 0.002 | 1.85e-06 |
| inv_f64_512x512 | tenferro-cuda-direct | 2 | 15 | 3.804 | 4194304 | 0.002 | 1.85e-06 |
| inv_f64_512x512 | tenferro-cuda-eager | 2 | 15 | 3.838 | 4194304 | 0.002 | 1.85e-06 |
| lu_tall_f64_s256 | jax-cuda | — | — | — | 4718600 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_tall_f64_s256 | pytorch-cuda | — | — | — | 4718600 | unavailable | No verified single public API returns the matching P/L/U/parity contract; three-output LU plus parity computation is a different operation. |
| lu_tall_f64_s256 | tenferro-cuda-direct | 4 | 15 | 3.209 | 4718600 | 0.004 | 1.104e-06 |
| lu_tall_f64_s256 | tenferro-cuda-eager | 4 | 15 | 3.163 | 4718600 | 0.004 | 1.104e-06 |
| norm_f64_512x512 | jax-cuda | 64 | 15 | 3.075 | 2097160 | 0.031 | 1.909e-07 |
| norm_f64_512x512 | pytorch-cuda | 256 | 15 | 3.199 | 2097160 | 0.119 | 1.909e-07 |
| norm_f64_512x512 | tenferro-cuda-direct | 128 | 15 | 2.181 | 2097160 | 0.087 | 0 |
| norm_f64_512x512 | tenferro-cuda-eager | 128 | 15 | 2.145 | 2097160 | 0.089 | 0 |
| pinv_f64_512x512 | jax-cuda | 1 | 15 | 78.465 | 4194320 | 0.000 | 9.031e-06 |
| pinv_f64_512x512 | pytorch-cuda | 1 | 15 | 85.173 | 4194320 | 0.000 | 8.691e-06 |
| pinv_f64_512x512 | tenferro-cuda-direct | 1 | 15 | 78.446 | 4194320 | 0.000 | 9.031e-06 |
| pinv_f64_512x512 | tenferro-cuda-eager | 1 | 15 | 78.227 | 4194320 | 0.000 | 9.031e-06 |
| slogdet_positive_f64_256x256 | jax-cuda | 4 | 15 | 2.245 | 524304 | 0.001 | 1.243e-06 |
| slogdet_positive_f64_256x256 | pytorch-cuda | 4 | 15 | 2.267 | 524304 | 0.001 | 1.087e-06 |
| slogdet_positive_f64_256x256 | tenferro-cuda-direct | 4 | 15 | 2.574 | 524304 | 0.001 | 1.243e-06 |
| slogdet_positive_f64_256x256 | tenferro-cuda-eager | 4 | 15 | 2.573 | 524304 | 0.001 | 1.243e-06 |
| svd_full_tall_f64_512x256 | jax-cuda | 1 | 15 | 19.729 | 3672064 | 0.000 | 7.76e-05 |
| svd_full_tall_f64_512x256 | pytorch-cuda | 1 | 15 | 20.030 | 3672064 | 0.000 | 7.76e-05 |
| svd_full_tall_f64_512x256 | tenferro-cuda-direct | 1 | 15 | 19.650 | 3672064 | 0.000 | 7.76e-05 |
| svd_full_tall_f64_512x256 | tenferro-cuda-eager | 1 | 15 | 19.628 | 3672064 | 0.000 | 7.76e-05 |
| svdvals_f64_512x512 | jax-cuda | 1 | 15 | 78.092 | 2101248 | 0.000 | 0.0002061 |
| svdvals_f64_512x512 | pytorch-cuda | 1 | 15 | 84.885 | 2101248 | 0.000 | 0.0002065 |
| svdvals_f64_512x512 | tenferro-cuda-direct | 1 | 15 | 77.929 | 2101248 | 0.000 | 0.0002061 |
| svdvals_f64_512x512 | tenferro-cuda-eager | 1 | 15 | 77.781 | 2101248 | 0.000 | 0.0002061 |
| triangular_solve_f64_512x512 | jax-cuda | 16 | 15 | 3.428 | 2162688 | 0.007 | 1.157e-06 |
| triangular_solve_f64_512x512 | pytorch-cuda | 8 | 15 | 2.476 | 2162688 | 0.005 | 9.176e-07 |
| triangular_solve_f64_512x512 | tenferro-cuda-direct | 16 | 15 | 3.132 | 2162688 | 0.008 | 1.157e-06 |
| triangular_solve_f64_512x512 | tenferro-cuda-eager | 16 | 15 | 3.397 | 2162688 | 0.007 | 1.157e-06 |

## Run metadata

```yaml
schema_version: 1
target_profile: nvidia-gpu
suite_id: gpu/linalg_public_api_throughput
suite_file: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/data/results/nvidia-gpu/gpu/routine_selection/20261128_082346/gpu_linalg_public_api_throughput.yaml
timestamp: '2026-11-28T08:23:46+00:00'
tenferro_rs:
  path: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/extern/tenferro-rs
  commit: 1ba9f33521691170c763dfce9b6c9bb56ca523c2
  dirty: false
  features:
  - cuda
  resolved_path: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/extern/tenferro-rs
harness:
  path: /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark
  commit: 28b35e0408e272b66d7460e3fa12213799c543f9
  dirty: false
resolved_deps:
  strided-perm:
  - 0.8.0
  - 0.15.1
  strided-view:
  - 0.7.1
  - 0.8.1
  threadteam:
  - 0.1.0
  - 0.6.1
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
  hostname: 32b786430f5d
  os: Linux-6.8.0-149-generic-x86_64-with-glibc2.39
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
    GPU_RUNTIME_SHA256: a261fcdef53ecfd3885b6d565dc834faef59287ce2da88c3d6754be96741f84d
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
  driver_version: 595.112.07
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
  - 0.093442
  - 0.090046
  - 0.098539
  - 0.098768
  - 0.09265
  - 0.096735
  - 0.095003
  - 0.0924
  - 0.096064
  - 0.094853
  - 0.092178
  - 0.096656
  - 0.09986
  - 0.096505
  - 0.094754
  median_ms: 0.095003
  bytes_per_second: 1412773575571.2979
  torch_version: 2.11.0+cu128
  cuda_version: '12.8'
  timestamp: '2026-11-28T08:33:50.665294+00:00'
gpu_assignment:
  role: benchmark
  physical_device_index: 0
  device_uuid: GPU-12b45076-79d5-7b84-cb5c-7c6300074c8a
  cuda_visible_devices: GPU-12b45076-79d5-7b84-cb5c-7c6300074c8a
  process_device_ordinal: 0
  verification_only: false
collection:
  command: BENCHMARK_TARGET_PROFILE=nvidia-gpu GPU_BENCH_DEVICE=0 BENCH_RUNS=15 BENCH_WARMUPS=3
    ./scripts/run_gpu_public_api.sh linalg_public_api_throughput
  devcontainer_config: .devcontainer/cuda/devcontainer.json
  expected_cases: 16
  selected_cases: 16
  sha256:
    .devcontainer/cuda/devcontainer.json: 402f3763533ea1531d0e400193395302213fd1b151a42baaeaebeafca3acabcd
    .devcontainer/cuda/Dockerfile: 6eff822595fa44f53a6fd9eebe0f4e3cc9e28362d9cfa8cac977a21b10a823f2
    src/bin/benchmark_gpu_public_api.rs: f807adb35a2c03383fde11227a47dffbf0ec5cc08c45ce298f7d9aa6341bbb67
    src/bin/benchmark_gpu_public_api/lifetime.rs: 34be366f56aac88f5f3b86ade8018a15dad2bca85cea74f56cb2fe121917d0a9
    src/bin/benchmark_gpu_public_api/linalg.rs: e8121a793ab45e565d150633b7e9a473b4d4f2c457ae36baed344f2c809fcad5
    src/bin/benchmark_gpu_public_api/multi.rs: 950cb1b3c1a7cca37665b58e3de074bd906dafb8c84b17a51c27ce797a73afa8
    src/bin/benchmark_gpu_public_api/plan.rs: 108edbad472e42b1a734552826d6d233085b581a3484cc4444471435160a6fd8
    src/bin/benchmark_gpu_public_api/read_layout.rs: f51757a34c90ce68e6fae0994df74169ed0438022031465349c7379c7b2c85e0
    src/bin/benchmark_gpu_public_api/reuse.rs: 53bee3baaeba93f4e1ee57d0d1b9553c189edd51d1940ea630ad0dbaa05b11eb
    src/bin/benchmark_gpu_public_api/runtime_ops.rs: 88038508963fbc13303fb54b3898808e4303131321cabedb9f38255941ead438
    src/bin/benchmark_gpu_public_api/state.rs: e1cc2781ce7392154f32cc19372c9fc41ae29765aedc544c7ae7bf02f873c12b
    src/bin/benchmark_gpu_public_api/structural.rs: 09b3cbf0dc0579978344c1aa738f6367112a0c25236924441f690a59de3fb80e
    /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/scripts/benchmark_gpu_public_api.py: d1ff68f86496d991bcbc730138fe6e3076001b3f23cc1a94c44b96f6857cb8ed
    /tmp/actions-runner/_work/tenferro-benchmark/tenferro-benchmark/scripts/gpu_public_api_ops.py: e88823db3f4a1fe43dc13c4743c9d78ece6af76a968724091af318dd8d7f2528
    scripts/run_gpu_public_api.sh: 7995bcf409258185e07210969a738896f64860b67df251fcd1100d11128a5431
    data/results/nvidia-gpu/gpu/linalg_public_api_throughput/20261128_082346/fixtures/manifest.json: 8a74129fa252ab33835079b24a6ac6f6ab5b3b4e7796933304a8b929baf2626e
    target/release/benchmark_gpu_public_api: 54dda9d9f7540806d7ff5a5d68e1312bc33b0d58871d8e3e9467a819e7c59318
  reference_numpy_version: 2.4.2
  measurement_cpu:
    policy: 'disabled (GPU_BENCH_CPU_SELECT unset): arms run on the allocation unpinned'
    arms:
      jax-cuda:
        arm: jax-cuda
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-28T08:34:28Z'
      pytorch-cuda:
        arm: pytorch-cuda
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-28T08:34:19Z'
      tenferro-cuda-direct:
        arm: tenferro-cuda-direct
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-28T08:34:01Z'
      tenferro-cuda-eager:
        arm: tenferro-cuda-eager
        status: disabled
        reason: 'GPU_BENCH_CPU_SELECT is not 1: the arm runs on the allocation unpinned'
        exit_code: 0
        finished_at: '2026-11-28T08:34:10Z'
  compile:
    case_wall_seconds: 30.593
    compiling_processes: 1
    modules_compiled: 1
    compile_time_ms: null
    compile_time_scope: 'not measured: no compile interval was recorded; `compile_modules`
      counts the NVRTC modules and the untimed wall is the case wall'
    untimed_wall_seconds: 2.756
```

## Untimed work over the recorded budget

Each row records the wall clock it spent outside its timed region. These cases spent more than the recorded budget ratio (0.2) there. The threshold has no measured basis, so this is a recorded warning: it does not change a cell, its acceptance or the exit status of the run.

| Case | Backend | Warning |
| --- | --- | --- |
| cholesky_f64_512x512 | tenferro-cuda-direct | cholesky_f64_512x512 untimed 145.8 ms is 379% of the 38.4 ms timed region (budget 20%) |
| svdvals_f64_512x512 | tenferro-cuda-direct | svdvals_f64_512x512 untimed 549.2 ms is 47% of the 1170.1 ms timed region (budget 20%) |
| eigvalsh_f64_512x512 | tenferro-cuda-direct | eigvalsh_f64_512x512 untimed 56.3 ms is 66% of the 84.7 ms timed region (budget 20%) |
| pinv_f64_512x512 | tenferro-cuda-direct | pinv_f64_512x512 untimed 2756.0 ms is 234% of the 1176.9 ms timed region (budget 20%) |
| inv_f64_512x512 | tenferro-cuda-direct | inv_f64_512x512 untimed 36.7 ms is 64% of the 57.1 ms timed region (budget 20%) |
| det_f64_512x512 | tenferro-cuda-direct | det_f64_512x512 untimed 15.8 ms is 33% of the 47.2 ms timed region (budget 20%) |
| norm_f64_512x512 | tenferro-cuda-direct | norm_f64_512x512 untimed 6.9 ms is 21% of the 32.7 ms timed region (budget 20%) |
| svd_full_tall_f64_512x256 | tenferro-cuda-direct | svd_full_tall_f64_512x256 untimed 693.1 ms is 235% of the 295.4 ms timed region (budget 20%) |
| slogdet_positive_f64_256x256 | tenferro-cuda-direct | slogdet_positive_f64_256x256 untimed 10.2 ms is 26% of the 39.8 ms timed region (budget 20%) |
| householder_qr_tall_f64_512x256 | tenferro-cuda-direct | householder_qr_tall_f64_512x256 untimed 421.0 ms is 1049% of the 40.1 ms timed region (budget 20%) |
| householder_from_factors_default_f64_512x256 | tenferro-cuda-direct | householder_from_factors_default_f64_512x256 untimed 423.5 ms is 953% of the 44.4 ms timed region (budget 20%) |
| householder_append_default_f64_512x256 | tenferro-cuda-direct | householder_append_default_f64_512x256 untimed 421.7 ms is 899% of the 46.9 ms timed region (budget 20%) |
| householder_r_default_f64_512x256 | tenferro-cuda-direct | householder_r_default_f64_512x256 untimed 9.1 ms is 30% of the 30.2 ms timed region (budget 20%) |
| householder_q_columns_prefix_f64_512x256 | tenferro-cuda-direct | householder_q_columns_prefix_f64_512x256 untimed 12.4 ms is 33% of the 37.9 ms timed region (budget 20%) |
| lu_tall_f64_s256 | tenferro-cuda-direct | lu_tall_f64_s256 untimed 27.1 ms is 55% of the 49.4 ms timed region (budget 20%) |
| cholesky_f64_512x512 | tenferro-cuda-eager | cholesky_f64_512x512 untimed 157.3 ms is 396% of the 39.8 ms timed region (budget 20%) |
| svdvals_f64_512x512 | tenferro-cuda-eager | svdvals_f64_512x512 untimed 548.3 ms is 47% of the 1169.9 ms timed region (budget 20%) |
| eigvalsh_f64_512x512 | tenferro-cuda-eager | eigvalsh_f64_512x512 untimed 55.2 ms is 67% of the 82.7 ms timed region (budget 20%) |
| pinv_f64_512x512 | tenferro-cuda-eager | pinv_f64_512x512 untimed 2545.8 ms is 217% of the 1173.5 ms timed region (budget 20%) |
| inv_f64_512x512 | tenferro-cuda-eager | inv_f64_512x512 untimed 36.2 ms is 63% of the 57.6 ms timed region (budget 20%) |
| det_f64_512x512 | tenferro-cuda-eager | det_f64_512x512 untimed 17.2 ms is 36% of the 47.3 ms timed region (budget 20%) |
| norm_f64_512x512 | tenferro-cuda-eager | norm_f64_512x512 untimed 6.9 ms is 21% of the 32.2 ms timed region (budget 20%) |
| svd_full_tall_f64_512x256 | tenferro-cuda-eager | svd_full_tall_f64_512x256 untimed 653.7 ms is 222% of the 294.4 ms timed region (budget 20%) |
| slogdet_positive_f64_256x256 | tenferro-cuda-eager | slogdet_positive_f64_256x256 untimed 9.8 ms is 26% of the 38.6 ms timed region (budget 20%) |
| householder_qr_tall_f64_512x256 | tenferro-cuda-eager | householder_qr_tall_f64_512x256 untimed 397.0 ms is 974% of the 40.8 ms timed region (budget 20%) |
| householder_from_factors_default_f64_512x256 | tenferro-cuda-eager | householder_from_factors_default_f64_512x256 untimed 396.1 ms is 886% of the 44.7 ms timed region (budget 20%) |
| householder_append_default_f64_512x256 | tenferro-cuda-eager | householder_append_default_f64_512x256 untimed 395.0 ms is 819% of the 48.2 ms timed region (budget 20%) |
| householder_r_default_f64_512x256 | tenferro-cuda-eager | householder_r_default_f64_512x256 untimed 10.7 ms is 32% of the 33.0 ms timed region (budget 20%) |
| householder_q_columns_prefix_f64_512x256 | tenferro-cuda-eager | householder_q_columns_prefix_f64_512x256 untimed 12.7 ms is 34% of the 37.6 ms timed region (budget 20%) |
| lu_tall_f64_s256 | tenferro-cuda-eager | lu_tall_f64_s256 untimed 25.4 ms is 53% of the 47.5 ms timed region (budget 20%) |
| cholesky_f64_512x512 | pytorch-cuda | cholesky_f64_512x512 untimed 189.1 ms is 495% of the 38.2 ms timed region (budget 20%) |
| svdvals_f64_512x512 | pytorch-cuda | svdvals_f64_512x512 untimed 606.6 ms is 47% of the 1285.0 ms timed region (budget 20%) |
| eigvalsh_f64_512x512 | pytorch-cuda | eigvalsh_f64_512x512 untimed 67.0 ms is 80% of the 84.2 ms timed region (budget 20%) |
| pinv_f64_512x512 | pytorch-cuda | pinv_f64_512x512 untimed 738.2 ms is 58% of the 1278.6 ms timed region (budget 20%) |
| inv_f64_512x512 | pytorch-cuda | inv_f64_512x512 untimed 84.1 ms is 155% of the 54.2 ms timed region (budget 20%) |
| det_f64_512x512 | pytorch-cuda | det_f64_512x512 untimed 691.5 ms is 1746% of the 39.6 ms timed region (budget 20%) |
| norm_f64_512x512 | pytorch-cuda | norm_f64_512x512 untimed 47.9 ms is 99% of the 48.5 ms timed region (budget 20%) |
| triangular_solve_f64_512x512 | pytorch-cuda | triangular_solve_f64_512x512 untimed 14.0 ms is 38% of the 37.1 ms timed region (budget 20%) |
| svd_full_tall_f64_512x256 | pytorch-cuda | svd_full_tall_f64_512x256 untimed 228.1 ms is 76% of the 300.4 ms timed region (budget 20%) |
| slogdet_positive_f64_256x256 | pytorch-cuda | slogdet_positive_f64_256x256 untimed 29.7 ms is 87% of the 34.0 ms timed region (budget 20%) |
| householder_qr_tall_f64_512x256 | pytorch-cuda | householder_qr_tall_f64_512x256 untimed 260.0 ms is 642% of the 40.5 ms timed region (budget 20%) |
| householder_r_default_f64_512x256 | pytorch-cuda | householder_r_default_f64_512x256 untimed 12.7 ms is 381% of the 3.3 ms timed region (budget 20%) |
| householder_q_columns_prefix_f64_512x256 | pytorch-cuda | householder_q_columns_prefix_f64_512x256 untimed 22.8 ms is 65% of the 35.1 ms timed region (budget 20%) |
| cholesky_f64_512x512 | jax-cuda | cholesky_f64_512x512 untimed 368.5 ms is 999% of the 36.9 ms timed region (budget 20%) |
| svdvals_f64_512x512 | jax-cuda | svdvals_f64_512x512 untimed 637.7 ms is 54% of the 1171.2 ms timed region (budget 20%) |
| eigvalsh_f64_512x512 | jax-cuda | eigvalsh_f64_512x512 untimed 164.4 ms is 180% of the 91.5 ms timed region (budget 20%) |
| pinv_f64_512x512 | jax-cuda | pinv_f64_512x512 untimed 1000.9 ms is 85% of the 1177.0 ms timed region (budget 20%) |
| inv_f64_512x512 | jax-cuda | inv_f64_512x512 untimed 168.7 ms is 308% of the 54.8 ms timed region (budget 20%) |
| det_f64_512x512 | jax-cuda | det_f64_512x512 untimed 214.1 ms is 522% of the 41.0 ms timed region (budget 20%) |
| norm_f64_512x512 | jax-cuda | norm_f64_512x512 untimed 135.1 ms is 294% of the 45.9 ms timed region (budget 20%) |
| triangular_solve_f64_512x512 | jax-cuda | triangular_solve_f64_512x512 untimed 132.4 ms is 258% of the 51.4 ms timed region (budget 20%) |
| svd_full_tall_f64_512x256 | jax-cuda | svd_full_tall_f64_512x256 untimed 313.4 ms is 106% of the 295.9 ms timed region (budget 20%) |
| slogdet_positive_f64_256x256 | jax-cuda | slogdet_positive_f64_256x256 untimed 167.7 ms is 498% of the 33.7 ms timed region (budget 20%) |
| householder_qr_tall_f64_512x256 | jax-cuda | householder_qr_tall_f64_512x256 untimed 243.1 ms is 577% of the 42.2 ms timed region (budget 20%) |
| householder_r_default_f64_512x256 | jax-cuda | householder_r_default_f64_512x256 untimed 148.8 ms is 465% of the 32.0 ms timed region (budget 20%) |
| householder_q_columns_prefix_f64_512x256 | jax-cuda | householder_q_columns_prefix_f64_512x256 untimed 153.1 ms is 286% of the 53.5 ms timed region (budget 20%) |