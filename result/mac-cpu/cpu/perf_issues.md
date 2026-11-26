# CPU Performance-Issue Workloads

- Suite: `cpu/perf_issues`
- Raw run: `data/results/mac-cpu/cpu/perf_issues/20261126_185530`

## CPU Information

- Model: `Apple M1 (Virtual)`
- Vendor: `Apple`
- Logical CPUs: `3`
- Physical CPUs: `3`
- Sockets: `1`
- Cores per socket: `3`
- Threads per core: `1`
- NUMA nodes: `1`
- Performance levels: `Standard: 3 physical / 3 logical (L1i 192 KiB, L1d 128 KiB, L2 12 MiB, 3 CPUs/L2)`
- Python platform: `macOS-26.7.2-arm64-arm-64bit-Mach-O`
- Container: `none`

## run_t1

Full metadata: `data/results/mac-cpu/cpu/perf_issues/20261126_185530/run_t1.yaml`

```yaml
timestamp: '2026-11-26T18:55:31Z'
tenferro_rs:
  path: /Users/runner/work/tenferro-benchmark/tenferro-benchmark/extern/tenferro-rs
  commit: 1ba9f33521691170c763dfce9b6c9bb56ca523c2
  dirty: false
  features:
  - native
  resolved_path: /Users/runner/work/tenferro-benchmark/tenferro-benchmark/extern/tenferro-rs
environment:
  container: none
  hostname: iad20-fj923-dad4e7a6-f252-45b3-932c-43b555f813c5-02FFA1A64A01.local
  os: macOS-26.7.2-arm64-arm-64bit-Mach-O
  arch: arm64
  cpu: arm
  affinity:
    pinning: none
    requested_measured_cpus: null
    observer_cpus: null
    override_of: null
    thread_budget: '1'
    measured_cpus: null
    l3_domains: null
    cpu_boost: null
    scaling_governor: null
    scaling_driver: null
  env:
    OMP_NUM_THREADS: '1'
    OMP_THREAD_LIMIT: '1'
    OMP_DYNAMIC: 'FALSE'
    RAYON_NUM_THREADS: '1'
    BENCHMARK_TARGET_PROFILE: mac-cpu
    BENCHMARK_COMMIT: 9b03dd718025b3350d93eaab13a8b93dcc66cab2
    OPENBLAS_NUM_THREADS: '1'
    MATMUL_NUM_THREADS: '1'
    GOTO_NUM_THREADS: '1'
    MKL_NUM_THREADS: '1'
    MKL_DYNAMIC: 'FALSE'
    VECLIB_MAXIMUM_THREADS: '1'
    VECLIB_NUM_THREADS: '1'
    NUMEXPR_NUM_THREADS: '1'
    BLIS_NUM_THREADS: '1'
    PJRT_NPROC: '1'
    XLA_FLAGS: --xla_cpu_multi_thread_eigen=false --xla_cpu_experimental_ynn_fusion_type=
      intra_op_parallelism_threads=1
    JULIA_NUM_THREADS: '1'
    PUBLICATION_GATE_PROFILE: quick
    BENCHMARK_BUILD_PROFILE: release
    RUSTC_VERSION: rustc 1.100.0 (0ef1bc386 2026-11-09)
    BENCH_INSTANCE: null
```

## run_t4

Full metadata: `data/results/mac-cpu/cpu/perf_issues/20261126_185530/run_t4.yaml`

```yaml
timestamp: '2026-11-26T18:56:23Z'
tenferro_rs:
  path: /Users/runner/work/tenferro-benchmark/tenferro-benchmark/extern/tenferro-rs
  commit: 1ba9f33521691170c763dfce9b6c9bb56ca523c2
  dirty: false
  features:
  - native
  resolved_path: /Users/runner/work/tenferro-benchmark/tenferro-benchmark/extern/tenferro-rs
environment:
  container: none
  hostname: iad20-fj923-dad4e7a6-f252-45b3-932c-43b555f813c5-02FFA1A64A01.local
  os: macOS-26.7.2-arm64-arm-64bit-Mach-O
  arch: arm64
  cpu: arm
  affinity:
    pinning: none
    requested_measured_cpus: null
    observer_cpus: null
    override_of: null
    thread_budget: '4'
    measured_cpus: null
    l3_domains: null
    cpu_boost: null
    scaling_governor: null
    scaling_driver: null
  env:
    OMP_NUM_THREADS: '4'
    OMP_THREAD_LIMIT: '4'
    OMP_DYNAMIC: 'FALSE'
    RAYON_NUM_THREADS: '4'
    BENCHMARK_TARGET_PROFILE: mac-cpu
    BENCHMARK_COMMIT: 9b03dd718025b3350d93eaab13a8b93dcc66cab2
    OPENBLAS_NUM_THREADS: '4'
    MATMUL_NUM_THREADS: '4'
    GOTO_NUM_THREADS: '4'
    MKL_NUM_THREADS: '4'
    MKL_DYNAMIC: 'FALSE'
    VECLIB_MAXIMUM_THREADS: '4'
    VECLIB_NUM_THREADS: '4'
    NUMEXPR_NUM_THREADS: '4'
    BLIS_NUM_THREADS: '4'
    PJRT_NPROC: '4'
    XLA_FLAGS: --xla_cpu_multi_thread_eigen=true --xla_cpu_experimental_ynn_fusion_type=
      intra_op_parallelism_threads=4
    JULIA_NUM_THREADS: '4'
    PUBLICATION_GATE_PROFILE: quick
    BENCHMARK_BUILD_PROFILE: release
    RUSTC_VERSION: rustc 1.100.0 (0ef1bc386 2026-11-09)
    BENCH_INSTANCE: null
```

## Case status

- Manifest: `cpu/perf_issues` version 25, coverage `quick`, effort `standard`
- Expected 683, selected 683, executed 495, unsupported 0, failed 0, skipped 188, missing 0, noisy 199.
- Complete: **NO** — only executed cases count as covered; unsupported, skipped, failed, missing and unselected cases never do.

| expected | selected | executed | unsupported | failed | skipped | missing | noisy |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 683 | 683 | 495 | 0 | 0 | 188 | 0 | 199 |

- Skipped: `activation_erf_f32_eager-single-call_1024x64@t1`, `activation_erf_f32_pytorch_1024x64@t1`, `activation_sigmoid_f32_eager-single-call_1024x64@t1`, `activation_sigmoid_f32_pytorch_1024x64@t1`, `activation_silu_f32_eager-single-call_1024x64@t1`, `activation_silu_f32_pytorch_1024x64@t1`, `activation_softplus_f32_eager-single-call_1024x64@t1`, `activation_softplus_f32_pytorch_1024x64@t1`, `activation_gelu_tanh_f32_eager-single-call_1024x64@t1`, `activation_gelu_tanh_f32_pytorch_1024x64@t1`, `followup_structural_broadcast_in_dim_direct-shared@t1`, `followup_structural_broadcast_in_dim_pytorch@t1`, `followup_real_tanh_direct-shared@t1`, `followup_real_tanh_jax@t1`, `followup_real_log1p_direct-shared@t1`, `followup_real_log1p_pytorch@t1`, `followup_real_cos_direct-shared@t1`, `followup_real_cos_pytorch@t1`, `followup_real_sin_direct-shared@t1`, `followup_real_sin_pytorch@t1`, `followup_real_exp_direct-shared@t1`, `followup_real_exp_jax@t1`, `followup_real_expm1_direct-shared@t1`, `followup_real_expm1_jax@t1`, `followup_real_pow_direct-shared@t1`, `followup_real_pow_pytorch@t1`, `followup_complex_exp_direct-shared@t1`, `followup_complex_exp_julia-base@t1`, `followup_real_log_direct-shared@t1`, `followup_real_log_pytorch@t1`, `followup_real_reduce_max_axis1_direct-shared@t1`, `followup_real_reduce_max_axis1_julia-base@t1`, `followup_real_reduce_prod_axis1_direct-shared@t1`, `followup_real_reduce_prod_axis1_pytorch@t1`, `followup_real_reduce_sum_axis1_direct-shared@t1`, `followup_real_reduce_sum_axis1_pytorch@t1`, `followup_real_reduce_max_axis0_direct-shared@t1`, `followup_real_reduce_max_axis0_julia-base@t1`, `followup_real_reduce_min_axis1_direct-shared@t1`, `followup_real_reduce_min_axis1_julia-base@t1` (+148 more)
- Noisy: `two_operand_einsum_f64_2x2_subscripts@t1`, `two_operand_einsum_f64_2x2_dot-general@t1`, `two_operand_einsum_f64_2x2_pytorch@t1`, `two_operand_einsum_f64_16x16_prepared@t1`, `two_operand_einsum_f64_16x16_pytorch@t1`, `two_operand_einsum_c64_2x2_string@t1`, `two_operand_einsum_c64_2x2_subscripts@t1`, `two_operand_einsum_c64_16x16_string@t1`, `two_operand_einsum_c64_16x16_prepared@t1`, `two_operand_einsum_c64_16x16_dot-general@t1`, `two_operand_einsum_c64_16x16_pytorch@t1`, `two_operand_einsum_f64_abc_cef_abef_string@t1`, `two_operand_einsum_f64_abc_cef_abef_subscripts@t1`, `two_operand_einsum_f64_abc_cef_abef_prepared@t1`, `two_operand_einsum_f64_abc_cef_abef_dot-general@t1`, `two_operand_einsum_f64_abc_cef_abef_pytorch@t1`, `thin_c64_gemm_1x141_141x282_t1@t1`, `thin_c64_gemm_1x141_141x282_t1_faer-direct@t1`, `thin_c64_gemm_1x141_141x282_t1_pytorch@t1`, `thin_c64_gemm_2x141_141x282_t1@t1`, `thin_c64_gemm_2x141_141x282_t1_faer-direct@t1`, `thin_c64_gemm_2x141_141x282_t1_pytorch@t1`, `thin_c64_gemm_8x141_141x282_t1@t1`, `decode_proj_f32_pytorch_in1024_out1024_len64@t1`, `decode_proj_f32_faer-direct_in1024_out1024_len512@t1`, `decode_proj_f32_host-sgemm_in1024_out4096_len8@t1`, `decode_proj_f32_pytorch_in1024_out4096_len8@t1`, `decode_proj_f32_faer-direct_in1024_out4096_len512@t1`, `conj_dot_f64_pytorch-vdot_len100@t1`, `conj_dot_f64_portable-loop_len10000@t1`, `conj_dot_f64_portable-loop_len1000000@t1`, `conj_dot_f64_pytorch-vdot_len1000000@t1`, `conj_dot_c64_dot-general-rank1_len100@t1`, `conj_dot_c64_dot-general-rank1_len10000@t1`, `conj_dot_c64_dot-general-rank1_len1000000@t1`, `small_solve_f64_solve-batched_k16_b64@t1`, `small_solve_f64_pytorch-solve-batched_k16_b64@t1`, `small_solve_f64_solve-loop_k64_b64@t1`, `small_solve_f64_solve-batched_k64_b64@t1`, `layer_norm_f32_pytorch-fused_d1024_len8_b1@t1` (+159 more)

Numerical checks: **495/683 recorded rows passed** (case × thread count).

## Steady-state operation rows

Each batch is one wall-clock interval over many operations; every output is retained until the clock stops. Median and inclusive IQR are per operation, in nanoseconds; CoV is sample standard deviation / mean (NOISY means CoV > 10%, a descriptive label only). Setup (inputs, backends, runtimes, session entry, tracing/compilation, preparation) is outside timing; faer-direct and host-sgemm write into a preallocated output, the tenferro and PyTorch arms allocate their output. The thin_complex_gemm faer-direct arm instead allocates its output inside timing to match its product references. `reference_for` in the case file names the case each reference arm is compared with. GPU batches end with one device synchronize (a per-call synchronize where the transfer is the operation); the Threads column is the host thread setting.

### Compute- and bandwidth-bound operations

| Case ID | Issues | Arm | Backend | Dtype | Workload | Threads | Ops/batch | Median ns/op | IQR ns | CoV | Status |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| softmax_f64_pool_serial_1024x4096 | #2273 #2274 #2276 | serial | tenferro-rs | f64 | rows=1024, cols=4096, axis=0 | 1 | 1 | 15324959.0 | 173354.0 | 2.1% | ok |
| softmax_f64_pool_pytorch_t1_1024x4096 | #2273 #2274 #2276 | pytorch | pytorch-cpu | f64 | rows=1024, cols=4096, axis=0 | 1 | 1 | 17659125.0 | 616021.0 | 2.0% | ok |
| softmax_f64_pool_pool4_1024x4096 | #2273 #2274 #2276 | pool4 | tenferro-rs | f64 | rows=1024, cols=4096, axis=0 | 4 | 1 | 7355750.0 | 478979.0 | 8.7% | ok |
| softmax_f64_pool_pytorch_t4_1024x4096 | #2273 #2274 #2276 | pytorch | pytorch-cpu | f64 | rows=1024, cols=4096, axis=0 | 4 | 1 | 11236666.0 | 3243229.5 | 16.9% | NOISY |

### Latency and per-call overhead

These rows measure dispatch/operation latency; they do not measure compute throughput.

| Case ID | Issues | Arm | Backend | Dtype | Workload | Threads | Ops/batch | Median ns/op | IQR ns | CoV | Status |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| two_operand_einsum_f64_2x2_string | #2325 #2324 | string | tenferro-rs | f64 | m=2, k=2, n=2, format=ij,jk->ik | 1 | 1024 | 2055.3 | 167.7 | 5.1% | ok |
| two_operand_einsum_f64_2x2_subscripts | #2325 #2324 | subscripts | tenferro-rs | f64 | m=2, k=2, n=2, format=ij,jk->ik | 1 | 2048 | 1383.4 | 116.3 | 16.6% | NOISY |
| two_operand_einsum_f64_2x2_prepared | #2325 #2324 | prepared | tenferro-rs | f64 | m=2, k=2, n=2, format=ij,jk->ik | 1 | 2048 | 1002.9 | 21.1 | 4.9% | ok |
| two_operand_einsum_f64_2x2_dot-general | #2325 #2324 | dot-general | tenferro-rs | f64 | m=2, k=2, n=2, format=ij,jk->ik | 1 | 4096 | 944.7 | 90.3 | 11.3% | NOISY |
| two_operand_einsum_f64_2x2_pytorch | #2325 #2324 | pytorch | pytorch-cpu | f64 | m=2, k=2, n=2, format=ij,jk->ik | 1 | 256 | 7990.9 | 2577.5 | 62.1% | NOISY |
| two_operand_einsum_f64_16x16_string | #2325 #2324 | string | tenferro-rs | f64 | m=16, k=16, n=16, format=ij,jk->ik | 1 | 1024 | 2858.7 | 316.0 | 9.1% | ok |
| two_operand_einsum_f64_16x16_subscripts | #2325 #2324 | subscripts | tenferro-rs | f64 | m=16, k=16, n=16, format=ij,jk->ik | 1 | 1024 | 1944.9 | 95.6 | 5.9% | ok |
| two_operand_einsum_f64_16x16_prepared | #2325 #2324 | prepared | tenferro-rs | f64 | m=16, k=16, n=16, format=ij,jk->ik | 1 | 2048 | 1364.1 | 133.6 | 14.7% | NOISY |
| two_operand_einsum_f64_16x16_dot-general | #2325 #2324 | dot-general | tenferro-rs | f64 | m=16, k=16, n=16, format=ij,jk->ik | 1 | 1024 | 1480.8 | 89.9 | 4.1% | ok |
| two_operand_einsum_f64_16x16_pytorch | #2325 #2324 | pytorch | pytorch-cpu | f64 | m=16, k=16, n=16, format=ij,jk->ik | 1 | 256 | 10099.3 | 536.1 | 19.9% | NOISY |
| two_operand_einsum_c64_2x2_string | #2325 #2324 | string | tenferro-rs | c64 | m=2, k=2, n=2, format=ij,jk->ik | 1 | 1024 | 1932.0 | 370.2 | 11.3% | NOISY |
| two_operand_einsum_c64_2x2_subscripts | #2325 #2324 | subscripts | tenferro-rs | c64 | m=2, k=2, n=2, format=ij,jk->ik | 1 | 1024 | 1357.6 | 180.5 | 18.2% | NOISY |
| two_operand_einsum_c64_2x2_prepared | #2325 #2324 | prepared | tenferro-rs | c64 | m=2, k=2, n=2, format=ij,jk->ik | 1 | 2048 | 931.7 | 33.7 | 2.2% | ok |
| two_operand_einsum_c64_2x2_dot-general | #2325 #2324 | dot-general | tenferro-rs | c64 | m=2, k=2, n=2, format=ij,jk->ik | 1 | 4096 | 829.2 | 11.8 | 1.6% | ok |
| two_operand_einsum_c64_2x2_pytorch | #2325 #2324 | pytorch | pytorch-cpu | c64 | m=2, k=2, n=2, format=ij,jk->ik | 1 | 512 | 6636.6 | 33.1 | 0.8% | ok |
| two_operand_einsum_c64_16x16_string | #2325 #2324 | string | tenferro-rs | c64 | m=16, k=16, n=16, format=ij,jk->ik | 1 | 512 | 3106.4 | 70.5 | 16.1% | NOISY |
| two_operand_einsum_c64_16x16_subscripts | #2325 #2324 | subscripts | tenferro-rs | c64 | m=16, k=16, n=16, format=ij,jk->ik | 1 | 512 | 2975.3 | 343.3 | 8.3% | ok |
| two_operand_einsum_c64_16x16_prepared | #2325 #2324 | prepared | tenferro-rs | c64 | m=16, k=16, n=16, format=ij,jk->ik | 1 | 1024 | 2509.9 | 347.0 | 39.0% | NOISY |
| two_operand_einsum_c64_16x16_dot-general | #2325 #2324 | dot-general | tenferro-rs | c64 | m=16, k=16, n=16, format=ij,jk->ik | 1 | 1024 | 2377.6 | 512.5 | 29.4% | NOISY |
| two_operand_einsum_c64_16x16_pytorch | #2325 #2324 | pytorch | pytorch-cpu | c64 | m=16, k=16, n=16, format=ij,jk->ik | 1 | 256 | 13743.2 | 15256.6 | 51.1% | NOISY |
| two_operand_einsum_f64_abc_cef_abef_string | #2325 #2324 | string | tenferro-rs | f64 | m=8, k=2, n=16, format=abc,cef->abef | 1 | 1024 | 3721.4 | 2202.8 | 30.4% | NOISY |
| two_operand_einsum_f64_abc_cef_abef_subscripts | #2325 #2324 | subscripts | tenferro-rs | f64 | m=8, k=2, n=16, format=abc,cef->abef | 1 | 1024 | 1791.5 | 398.6 | 30.4% | NOISY |
| two_operand_einsum_f64_abc_cef_abef_prepared | #2325 #2324 | prepared | tenferro-rs | f64 | m=8, k=2, n=16, format=abc,cef->abef | 1 | 1024 | 1279.9 | 107.4 | 12.7% | NOISY |
| two_operand_einsum_f64_abc_cef_abef_dot-general | #2325 #2324 | dot-general | tenferro-rs | f64 | m=8, k=2, n=16, format=abc,cef->abef | 1 | 1024 | 1184.4 | 110.2 | 29.9% | NOISY |
| two_operand_einsum_f64_abc_cef_abef_pytorch | #2325 #2324 | pytorch | pytorch-cpu | f64 | m=8, k=2, n=16, format=abc,cef->abef | 1 | 256 | 10909.3 | 5452.8 | 35.7% | NOISY |
| thin_c64_gemm_1x141_141x282_t1 | #2524 #2416 #1099 | dot-general | tenferro-rs | c64 | m=1, k=141, n=282 | 1 | 64 | 29408.9 | 2192.4 | 14.7% | NOISY |
| thin_c64_gemm_1x141_141x282_t1_faer-direct | #2524 #2416 #1099 | faer-direct | faer-direct | c64 | m=1, k=141, n=282 | 1 | 128 | 30215.2 | 13546.9 | 32.5% | NOISY |
| thin_c64_gemm_1x141_141x282_t1_pytorch | #2524 #2416 #1099 | pytorch | pytorch-cpu | c64 | m=1, k=141, n=282 | 1 | 64 | 65998.7 | 31386.4 | 28.7% | NOISY |
| thin_c64_gemm_2x141_141x282_t1 | #2524 #2416 #1099 | dot-general | tenferro-rs | c64 | m=2, k=141, n=282 | 1 | 64 | 34658.2 | 2854.5 | 16.8% | NOISY |
| thin_c64_gemm_2x141_141x282_t1_faer-direct | #2524 #2416 #1099 | faer-direct | faer-direct | c64 | m=2, k=141, n=282 | 1 | 64 | 30669.9 | 7172.9 | 26.1% | NOISY |
| thin_c64_gemm_2x141_141x282_t1_pytorch | #2524 #2416 #1099 | pytorch | pytorch-cpu | c64 | m=2, k=141, n=282 | 1 | 64 | 44480.5 | 11663.4 | 29.8% | NOISY |
| thin_c64_gemm_8x141_141x282_t1 | #2524 #2416 #1099 | dot-general | tenferro-rs | c64 | m=8, k=141, n=282 | 1 | 16 | 115513.0 | 10852.8 | 28.1% | NOISY |
| thin_c64_gemm_8x141_141x282_t1_faer-direct | #2524 #2416 #1099 | faer-direct | faer-direct | c64 | m=8, k=141, n=282 | 1 | 32 | 107122.4 | 2561.8 | 1.4% | ok |
| thin_c64_gemm_8x141_141x282_t1_pytorch | #2524 #2416 #1099 | pytorch | pytorch-cpu | c64 | m=8, k=141, n=282 | 1 | 64 | 45934.2 | 562.8 | 1.1% | ok |
| decode_proj_f32_eager-shared_in1024_out1024_len8 | #1992 #2003 #1995 #2229 | eager-shared | tenferro-rs | f32 | in=1024, out=1024, len=8 | 1 | 8 | 400802.1 | 11466.1 | 3.4% | ok |
| decode_proj_f32_faer-direct_in1024_out1024_len8 | #1992 #2003 #1995 #2229 | faer-direct | faer-direct | f32 | in=1024, out=1024, len=8 | 1 | 8 | 357036.5 | 666.6 | 0.4% | ok |
| decode_proj_f32_host-sgemm_in1024_out1024_len8 | #1992 #2003 #1995 #2229 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=1024, len=8 | 1 | 8 | 381869.8 | 25713.6 | 4.6% | ok |
| decode_proj_f32_pytorch_in1024_out1024_len8 | #1992 #2003 #1995 #2229 | pytorch | pytorch-cpu | f32 | in=1024, out=1024, len=8 | 1 | 16 | 150195.3 | 14660.2 | 8.3% | ok |
| decode_proj_f32_eager-shared_in1024_out1024_len64 | #1992 #2003 #1995 #2229 | eager-shared | tenferro-rs | f32 | in=1024, out=1024, len=64 | 1 | 1 | 2006917.0 | 72249.5 | 4.4% | ok |
| decode_proj_f32_faer-direct_in1024_out1024_len64 | #1992 #2003 #1995 #2229 | faer-direct | faer-direct | f32 | in=1024, out=1024, len=64 | 1 | 2 | 2005083.5 | 169718.8 | 6.5% | ok |
| decode_proj_f32_host-sgemm_in1024_out1024_len64 | #1992 #2003 #1995 #2229 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=1024, len=64 | 1 | 2 | 1739354.5 | 66406.2 | 2.9% | ok |
| decode_proj_f32_pytorch_in1024_out1024_len64 | #1992 #2003 #1995 #2229 | pytorch | pytorch-cpu | f32 | in=1024, out=1024, len=64 | 1 | 16 | 179052.1 | 19489.6 | 15.6% | NOISY |
| decode_proj_f32_eager-shared_in1024_out1024_len512 | #1992 #2003 #1995 #2229 | eager-shared | tenferro-rs | f32 | in=1024, out=1024, len=512 | 1 | 1 | 14276500.0 | 553958.5 | 8.4% | ok |
| decode_proj_f32_faer-direct_in1024_out1024_len512 | #1992 #2003 #1995 #2229 | faer-direct | faer-direct | f32 | in=1024, out=1024, len=512 | 1 | 1 | 12795584.0 | 1565582.5 | 10.2% | NOISY |
| decode_proj_f32_host-sgemm_in1024_out1024_len512 | #1992 #2003 #1995 #2229 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=1024, len=512 | 1 | 1 | 12550583.0 | 540958.5 | 3.8% | ok |
| decode_proj_f32_pytorch_in1024_out1024_len512 | #1992 #2003 #1995 #2229 | pytorch | pytorch-cpu | f32 | in=1024, out=1024, len=512 | 1 | 2 | 1407604.5 | 106687.2 | 6.4% | ok |
| decode_proj_f32_eager-shared_in1024_out4096_len8 | #1992 #2003 #1995 #2229 | eager-shared | tenferro-rs | f32 | in=1024, out=4096, len=8 | 1 | 2 | 1792416.5 | 196031.5 | 6.3% | ok |
| decode_proj_f32_faer-direct_in1024_out4096_len8 | #1992 #2003 #1995 #2229 | faer-direct | faer-direct | f32 | in=1024, out=4096, len=8 | 1 | 2 | 1464333.0 | 10333.8 | 1.6% | ok |
| decode_proj_f32_host-sgemm_in1024_out4096_len8 | #1992 #2003 #1995 #2229 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=4096, len=8 | 1 | 2 | 1563500.0 | 22114.8 | 11.1% | NOISY |
| decode_proj_f32_pytorch_in1024_out4096_len8 | #1992 #2003 #1995 #2229 | pytorch | pytorch-cpu | f32 | in=1024, out=4096, len=8 | 1 | 2 | 1359687.5 | 354844.0 | 19.4% | NOISY |
| decode_proj_f32_eager-shared_in1024_out4096_len64 | #1992 #2003 #1995 #2229 | eager-shared | tenferro-rs | f32 | in=1024, out=4096, len=64 | 1 | 1 | 8621166.0 | 288562.5 | 2.7% | ok |
| decode_proj_f32_faer-direct_in1024_out4096_len64 | #1992 #2003 #1995 #2229 | faer-direct | faer-direct | f32 | in=1024, out=4096, len=64 | 1 | 1 | 8253167.0 | 730937.0 | 5.2% | ok |
| decode_proj_f32_host-sgemm_in1024_out4096_len64 | #1992 #2003 #1995 #2229 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=4096, len=64 | 1 | 1 | 8006834.0 | 509812.5 | 4.6% | ok |
| decode_proj_f32_pytorch_in1024_out4096_len64 | #1992 #2003 #1995 #2229 | pytorch | pytorch-cpu | f32 | in=1024, out=4096, len=64 | 1 | 4 | 987323.0 | 18323.0 | 1.2% | ok |
| decode_proj_f32_eager-shared_in1024_out4096_len512 | #1992 #2003 #1995 #2229 | eager-shared | tenferro-rs | f32 | in=1024, out=4096, len=512 | 1 | 1 | 50946500.0 | 3795062.5 | 5.4% | ok |
| decode_proj_f32_faer-direct_in1024_out4096_len512 | #1992 #2003 #1995 #2229 | faer-direct | faer-direct | f32 | in=1024, out=4096, len=512 | 1 | 1 | 51019291.0 | 9662333.5 | 10.4% | NOISY |
| decode_proj_f32_host-sgemm_in1024_out4096_len512 | #1992 #2003 #1995 #2229 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=4096, len=512 | 1 | 1 | 48641417.0 | 6029396.5 | 8.3% | ok |
| decode_proj_f32_pytorch_in1024_out4096_len512 | #1992 #2003 #1995 #2229 | pytorch | pytorch-cpu | f32 | in=1024, out=4096, len=512 | 1 | 1 | 5342875.0 | 692437.0 | 7.3% | ok |
| decode_proj_view_operand_f32_prepared_in1024_out1024_len8_t1 | #2216 | prepared-view | tenferro-rs | f32 | in=1024, out=1024, len=8 | 1 | 8 | 476109.4 | 44335.9 | 9.0% | ok |
| decode_proj_contiguous_operand_f32_prepared_in1024_out1024_len8_t1 | #2216 | prepared-contiguous | tenferro-rs | f32 | in=1024, out=1024, len=8 | 1 | 8 | 403437.5 | 21421.9 | 4.2% | ok |
| decode_proj_view_operand_f32_pytorch_in1024_out1024_len8_t1 | #2216 | pytorch | pytorch-cpu | f32 | in=1024, out=1024, len=8 | 1 | 16 | 132231.8 | 11691.4 | 7.1% | ok |
| gemm_c64_mm256_t1 | #1900 | dot-general-shared | tenferro-rs | c64 | n=256 | 1 | 1 | 4302333.0 | 73041.5 | 2.7% | ok |
| gemm_f64_mm256_t1 | #1900 | dot-general-shared | tenferro-rs | f64 | n=256 | 1 | 4 | 836343.8 | 13494.6 | 2.1% | ok |
| conj_dot_f64_dot-general-rank1_len100 | #1615 | dot-general-rank1 | tenferro-rs | f64 | len=100 | 1 | 4096 | 838.3 | 10.5 | 2.3% | ok |
| conj_dot_f64_portable-loop_len100 | #1615 | portable-loop | rust-loop | f64 | len=100 | 1 | 65536 | 57.1 | 0.6 | 1.5% | ok |
| conj_dot_f64_pytorch-vdot_len100 | #1615 | pytorch-vdot | pytorch-cpu | f64 | len=100 | 1 | 4096 | 638.5 | 35.9 | 16.6% | NOISY |
| conj_dot_f64_dot-general-rank1_len10000 | #1615 | dot-general-rank1 | tenferro-rs | f64 | len=10000 | 1 | 1024 | 3136.4 | 159.5 | 3.6% | ok |
| conj_dot_f64_portable-loop_len10000 | #1615 | portable-loop | rust-loop | f64 | len=10000 | 1 | 256 | 10211.3 | 687.6 | 14.3% | NOISY |
| conj_dot_f64_pytorch-vdot_len10000 | #1615 | pytorch-vdot | pytorch-cpu | f64 | len=10000 | 1 | 1024 | 3086.2 | 122.1 | 8.0% | ok |
| conj_dot_f64_dot-general-rank1_len1000000 | #1615 | dot-general-rank1 | tenferro-rs | f64 | len=1000000 | 1 | 8 | 355885.5 | 27932.4 | 4.7% | ok |
| conj_dot_f64_portable-loop_len1000000 | #1615 | portable-loop | rust-loop | f64 | len=1000000 | 1 | 2 | 1055292.0 | 32583.2 | 11.7% | NOISY |
| conj_dot_f64_pytorch-vdot_len1000000 | #1615 | pytorch-vdot | pytorch-cpu | f64 | len=1000000 | 1 | 8 | 389750.0 | 54726.6 | 14.9% | NOISY |
| conj_dot_c64_dot-general-rank1_len100 | #1615 | dot-general-rank1 | tenferro-rs | c64 | len=100 | 1 | 4096 | 916.2 | 80.0 | 30.6% | NOISY |
| conj_dot_c64_portable-loop_len100 | #1615 | portable-loop | rust-loop | c64 | len=100 | 1 | 32768 | 91.9 | 1.7 | 2.9% | ok |
| conj_dot_c64_pytorch-vdot_len100 | #1615 | pytorch-vdot | pytorch-cpu | c64 | len=100 | 1 | 2048 | 595.5 | 9.5 | 4.1% | ok |
| conj_dot_c64_dot-general-rank1_len10000 | #1615 | dot-general-rank1 | tenferro-rs | c64 | len=10000 | 1 | 512 | 7424.6 | 768.4 | 38.3% | NOISY |
| conj_dot_c64_portable-loop_len10000 | #1615 | portable-loop | rust-loop | c64 | len=10000 | 1 | 256 | 10406.9 | 474.9 | 3.5% | ok |
| conj_dot_c64_pytorch-vdot_len10000 | #1615 | pytorch-vdot | pytorch-cpu | c64 | len=10000 | 1 | 512 | 5328.9 | 237.7 | 2.9% | ok |
| conj_dot_c64_dot-general-rank1_len1000000 | #1615 | dot-general-rank1 | tenferro-rs | c64 | len=1000000 | 1 | 4 | 839260.2 | 40687.5 | 28.4% | NOISY |
| conj_dot_c64_portable-loop_len1000000 | #1615 | portable-loop | rust-loop | c64 | len=1000000 | 1 | 2 | 1039416.5 | 20969.0 | 1.6% | ok |
| conj_dot_c64_pytorch-vdot_len1000000 | #1615 | pytorch-vdot | pytorch-cpu | c64 | len=1000000 | 1 | 4 | 729041.5 | 66104.1 | 7.3% | ok |
| small_solve_f64_solve-loop_k16_b64 | #2007 #1878 | solve-loop | tenferro-rs | f64 | k=16, batch=64 | 1 | 16 | 172664.1 | 4283.8 | 3.1% | ok |
| small_solve_f64_triangular-solve-loop_k16_b64 | #2007 #1878 | triangular-solve-loop | tenferro-rs | f64 | k=16, batch=64 | 1 | 32 | 83918.0 | 1696.0 | 2.0% | ok |
| small_solve_f64_solve-batched_k16_b64 | #2007 #1878 | solve-batched | tenferro-rs | f64 | k=16, batch=64 | 1 | 32 | 114546.9 | 2036.5 | 37.2% | NOISY |
| small_solve_f64_pytorch-solve-batched_k16_b64 | #2007 #1878 | pytorch-solve-batched | pytorch-cpu | f64 | k=16, batch=64 | 1 | 32 | 104795.6 | 1742.8 | 25.6% | NOISY |
| small_solve_f64_pytorch-solve-triangular-batched_k16_b64 | #2007 #1878 | pytorch-solve-triangular-batched | pytorch-cpu | f64 | k=16, batch=64 | 1 | 64 | 24145.8 | 595.0 | 3.1% | ok |
| small_solve_f64_solve-loop_k64_b64 | #2007 #1878 | solve-loop | tenferro-rs | f64 | k=64, batch=64 | 1 | 1 | 2863625.0 | 77395.5 | 17.8% | NOISY |
| small_solve_f64_triangular-solve-loop_k64_b64 | #2007 #1878 | triangular-solve-loop | tenferro-rs | f64 | k=64, batch=64 | 1 | 2 | 1070625.0 | 25916.5 | 1.9% | ok |
| small_solve_f64_solve-batched_k64_b64 | #2007 #1878 | solve-batched | tenferro-rs | f64 | k=64, batch=64 | 1 | 1 | 2686083.0 | 73417.0 | 17.5% | NOISY |
| small_solve_f64_pytorch-solve-batched_k64_b64 | #2007 #1878 | pytorch-solve-batched | pytorch-cpu | f64 | k=64, batch=64 | 1 | 2 | 1589083.5 | 74531.0 | 3.7% | ok |
| small_solve_f64_pytorch-solve-triangular-batched_k64_b64 | #2007 #1878 | pytorch-solve-triangular-batched | pytorch-cpu | f64 | k=64, batch=64 | 1 | 8 | 384104.1 | 17802.1 | 4.0% | ok |
| tanh_chain_f32_compiled-prepared_k10_1024x64 | #1990 | compiled-prepared | tenferro-rs | f32 | rows=1024, cols=64, chain=10 | 1 | 1 | 2399375.0 | 67312.5 | 4.3% | ok |
| tanh_chain_f32_eager-shared_k10_1024x64 | #1990 | eager-shared | tenferro-rs | f32 | rows=1024, cols=64, chain=10 | 1 | 1 | 2379166.0 | 19750.0 | 0.9% | ok |
| tanh_chain_f32_compiled-prepared_k100_1024x64 | #1990 | compiled-prepared | tenferro-rs | f32 | rows=1024, cols=64, chain=100 | 1 | 1 | 23616875.0 | 333355.0 | 1.7% | ok |
| tanh_chain_f32_eager-shared_k100_1024x64 | #1990 | eager-shared | tenferro-rs | f32 | rows=1024, cols=64, chain=100 | 1 | 1 | 24801375.0 | 1332250.5 | 3.7% | ok |
| tanh_chain_f32_compiled-prepared_k200_1024x64 | #1990 | compiled-prepared | tenferro-rs | f32 | rows=1024, cols=64, chain=200 | 1 | 1 | 47200125.0 | 2620166.5 | 3.9% | ok |
| tanh_chain_f32_eager-shared_k200_1024x64 | #1990 | eager-shared | tenferro-rs | f32 | rows=1024, cols=64, chain=200 | 1 | 1 | 46838583.0 | 1683771.0 | 2.1% | ok |
| layer_norm_f32_eager-composed_d1024_len8_b1 | #2006 | eager-composed | tenferro-rs | f32 | norm=layer_norm, d=1024, len=8, batch=1 | 1 | 64 | 39688.2 | 3707.7 | 4.9% | ok |
| layer_norm_f32_pytorch-fused_d1024_len8_b1 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=layer_norm, d=1024, len=8, batch=1 | 1 | 256 | 5437.8 | 104.7 | 46.9% | NOISY |
| layer_norm_f32_eager-single-call_d1024_len8_b1 | #2006 | eager-single-call | tenferro-rs | f32 | norm=layer_norm, d=1024, len=8, batch=1 | 1 | 64 | 44408.8 | 1599.0 | 8.2% | ok |
| layer_norm_f32_eager-composed_d1024_len8_b8 | #2006 | eager-composed | tenferro-rs | f32 | norm=layer_norm, d=1024, len=8, batch=8 | 1 | 32 | 77777.3 | 2188.8 | 3.3% | ok |
| layer_norm_f32_pytorch-fused_d1024_len8_b8 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=layer_norm, d=1024, len=8, batch=8 | 1 | 64 | 31880.9 | 1839.8 | 4.2% | ok |
| layer_norm_f32_eager-single-call_d1024_len8_b8 | #2006 | eager-single-call | tenferro-rs | f32 | norm=layer_norm, d=1024, len=8, batch=8 | 1 | 32 | 84778.7 | 6016.3 | 5.6% | ok |
| layer_norm_f32_eager-composed_d1024_len64_b1 | #2006 | eager-composed | tenferro-rs | f32 | norm=layer_norm, d=1024, len=64, batch=1 | 1 | 32 | 75203.1 | 1039.1 | 1.3% | ok |
| layer_norm_f32_pytorch-fused_d1024_len64_b1 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=layer_norm, d=1024, len=64, batch=1 | 1 | 64 | 31357.4 | 621.1 | 3.7% | ok |
| layer_norm_f32_eager-single-call_d1024_len64_b1 | #2006 | eager-single-call | tenferro-rs | f32 | norm=layer_norm, d=1024, len=64, batch=1 | 1 | 32 | 86774.8 | 7925.2 | 8.1% | ok |
| layer_norm_f32_eager-composed_d1024_len64_b8 | #2006 | eager-composed | tenferro-rs | f32 | norm=layer_norm, d=1024, len=64, batch=8 | 1 | 4 | 444989.5 | 14427.0 | 3.1% | ok |
| layer_norm_f32_pytorch-fused_d1024_len64_b8 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=layer_norm, d=1024, len=64, batch=8 | 1 | 8 | 244520.9 | 15682.2 | 6.7% | ok |
| layer_norm_f32_eager-single-call_d1024_len64_b8 | #2006 | eager-single-call | tenferro-rs | f32 | norm=layer_norm, d=1024, len=64, batch=8 | 1 | 4 | 409968.8 | 8630.4 | 3.3% | ok |
| rms_norm_f32_eager-composed_d1024_len8_b1 | #2006 | eager-composed | tenferro-rs | f32 | norm=rms_norm, d=1024, len=8, batch=1 | 1 | 128 | 20931.3 | 711.4 | 4.0% | ok |
| rms_norm_f32_pytorch-fused_d1024_len8_b1 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=rms_norm, d=1024, len=8, batch=1 | 1 | 128 | 10373.7 | 318.0 | 4.6% | ok |
| rms_norm_f32_eager-single-call_d1024_len8_b1 | #2006 | eager-single-call | tenferro-rs | f32 | norm=rms_norm, d=1024, len=8, batch=1 | 1 | 128 | 26806.6 | 777.3 | 6.4% | ok |
| rms_norm_f32_eager-composed_d1024_len8_b8 | #2006 | eager-composed | tenferro-rs | f32 | norm=rms_norm, d=1024, len=8, batch=8 | 1 | 64 | 43294.9 | 2068.0 | 12.8% | NOISY |
| rms_norm_f32_pytorch-fused_d1024_len8_b8 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=rms_norm, d=1024, len=8, batch=8 | 1 | 64 | 32530.0 | 464.8 | 6.2% | ok |
| rms_norm_f32_eager-single-call_d1024_len8_b8 | #2006 | eager-single-call | tenferro-rs | f32 | norm=rms_norm, d=1024, len=8, batch=8 | 1 | 32 | 47574.2 | 1597.0 | 11.6% | NOISY |
| rms_norm_f32_eager-composed_d1024_len64_b1 | #2006 | eager-composed | tenferro-rs | f32 | norm=rms_norm, d=1024, len=64, batch=1 | 1 | 64 | 42544.9 | 989.9 | 2.5% | ok |
| rms_norm_f32_pytorch-fused_d1024_len64_b1 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=rms_norm, d=1024, len=64, batch=1 | 1 | 64 | 32510.4 | 845.7 | 3.3% | ok |
| rms_norm_f32_eager-single-call_d1024_len64_b1 | #2006 | eager-single-call | tenferro-rs | f32 | norm=rms_norm, d=1024, len=64, batch=1 | 1 | 32 | 49029.9 | 2285.2 | 5.3% | ok |
| rms_norm_f32_eager-composed_d1024_len64_b8 | #2006 | eager-composed | tenferro-rs | f32 | norm=rms_norm, d=1024, len=64, batch=8 | 1 | 8 | 223432.4 | 6125.1 | 2.7% | ok |
| rms_norm_f32_pytorch-fused_d1024_len64_b8 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=rms_norm, d=1024, len=64, batch=8 | 1 | 8 | 216531.2 | 84239.6 | 34.4% | NOISY |
| rms_norm_f32_eager-single-call_d1024_len64_b8 | #2006 | eager-single-call | tenferro-rs | f32 | norm=rms_norm, d=1024, len=64, batch=8 | 1 | 8 | 271802.1 | 80677.1 | 53.4% | NOISY |
| small_contraction_abcd-dbef-acef_f64_shared-session-default-pool_d4 | #1885 #1897 #1901 | shared-session-default-pool | tenferro-rs | f64 | extent=4 | 1 | 1024 | 3230.1 | 1162.3 | 34.0% | NOISY |
| activation_erf_f32_eager-single-call_1024x64 | #1975 #2069 #2032 | eager-single-call | tenferro-rs | f32 | op=erf, rows=1024, cols=64 | 1 | — | — | — | — | SKIPPED |
| activation_erf_f32_session-single-call_1024x64 | #1975 #2069 #2032 | session-single-call | tenferro-rs | f32 | op=erf, rows=1024, cols=64 | 1 | 8 | 499067.6 | 51046.8 | 38.6% | NOISY |
| activation_erf_f32_pytorch_1024x64 | #1975 #2069 #2032 | pytorch | pytorch-cpu | f32 | op=erf, rows=1024, cols=64 | 1 | — | — | — | — | SKIPPED |
| activation_sigmoid_f32_eager-single-call_1024x64 | #1975 #2069 #2032 | eager-single-call | tenferro-rs | f32 | op=sigmoid, rows=1024, cols=64 | 1 | — | — | — | — | SKIPPED |
| activation_sigmoid_f32_session-single-call_1024x64 | #1975 #2069 #2032 | session-single-call | tenferro-rs | f32 | op=sigmoid, rows=1024, cols=64 | 1 | 4 | 54593.8 | 43213.4 | 72.4% | NOISY |
| activation_sigmoid_f32_eager-composed_1024x64 | #1975 #2069 #2032 | eager-composed | tenferro-rs | f32 | op=sigmoid, rows=1024, cols=64 | 1 | 8 | 67849.0 | 8583.3 | 66.6% | NOISY |
| activation_sigmoid_f32_pytorch_1024x64 | #1975 #2069 #2032 | pytorch | pytorch-cpu | f32 | op=sigmoid, rows=1024, cols=64 | 1 | — | — | — | — | SKIPPED |
| activation_silu_f32_eager-single-call_1024x64 | #1975 #2069 #2032 | eager-single-call | tenferro-rs | f32 | op=silu, rows=1024, cols=64 | 1 | — | — | — | — | SKIPPED |
| activation_silu_f32_session-single-call_1024x64 | #1975 #2069 #2032 | session-single-call | tenferro-rs | f32 | op=silu, rows=1024, cols=64 | 1 | 32 | 57300.8 | 10816.4 | 14.1% | NOISY |
| activation_silu_f32_eager-composed_1024x64 | #1975 #2069 #2032 | eager-composed | tenferro-rs | f32 | op=silu, rows=1024, cols=64 | 1 | 8 | 90875.0 | 8070.3 | 47.6% | NOISY |
| activation_silu_f32_pytorch_1024x64 | #1975 #2069 #2032 | pytorch | pytorch-cpu | f32 | op=silu, rows=1024, cols=64 | 1 | — | — | — | — | SKIPPED |
| activation_softplus_f32_eager-single-call_1024x64 | #1975 #2069 #2032 | eager-single-call | tenferro-rs | f32 | op=softplus, rows=1024, cols=64 | 1 | — | — | — | — | SKIPPED |
| activation_softplus_f32_session-single-call_1024x64 | #1975 #2069 #2032 | session-single-call | tenferro-rs | f32 | op=softplus, rows=1024, cols=64 | 1 | 8 | 233260.5 | 4101.5 | 27.9% | NOISY |
| activation_softplus_f32_eager-composed_1024x64 | #1975 #2069 #2032 | eager-composed | tenferro-rs | f32 | op=softplus, rows=1024, cols=64 | 1 | 8 | 198182.2 | 15781.2 | 60.1% | NOISY |
| activation_softplus_f32_pytorch_1024x64 | #1975 #2069 #2032 | pytorch | pytorch-cpu | f32 | op=softplus, rows=1024, cols=64 | 1 | — | — | — | — | SKIPPED |
| activation_gelu_f32_eager-single-call_1024x64 | #1975 #2030 #2069 #2228 | eager-single-call | tenferro-rs | f32 | op=gelu, rows=1024, cols=64 | 1 | 32 | 85738.3 | 21161.5 | 41.3% | NOISY |
| activation_gelu_f32_session-single-call_1024x64 | #1975 #2030 #2069 #2228 | session-single-call | tenferro-rs | f32 | op=gelu, rows=1024, cols=64 | 1 | 16 | 84242.2 | 4899.7 | 9.7% | ok |
| activation_gelu_f32_pytorch_1024x64 | #1975 #2030 #2069 #2228 | pytorch | pytorch-cpu | f32 | op=gelu, rows=1024, cols=64 | 1 | 16 | 93888.1 | 16322.9 | 60.4% | NOISY |
| activation_gelu_tanh_f32_eager-single-call_1024x64 | #1975 #2069 #2032 | eager-single-call | tenferro-rs | f32 | op=gelu_tanh, rows=1024, cols=64 | 1 | — | — | — | — | SKIPPED |
| activation_gelu_tanh_f32_session-single-call_1024x64 | #1975 #2069 #2032 | session-single-call | tenferro-rs | f32 | op=gelu_tanh, rows=1024, cols=64 | 1 | 8 | 279223.9 | 6458.2 | 3.6% | ok |
| activation_gelu_tanh_f32_eager-composed_1024x64 | #1975 #2069 #2032 | eager-composed | tenferro-rs | f32 | op=gelu_tanh, rows=1024, cols=64 | 1 | 8 | 345557.2 | 91036.4 | 39.8% | NOISY |
| activation_gelu_tanh_f32_pytorch_1024x64 | #1975 #2069 #2032 | pytorch | pytorch-cpu | f32 | op=gelu_tanh, rows=1024, cols=64 | 1 | — | — | — | — | SKIPPED |
| softmax_f32_eager-single-call_len64_b8 | #1976 #2031 #2231 | eager-single-call | tenferro-rs | f32 | op=softmax, len=64, batch=8, axis=0 | 1 | 64 | 37152.3 | 3090.2 | 17.4% | NOISY |
| softmax_f32_session-single-call_len64_b8 | #1976 #2031 #2231 | session-single-call | tenferro-rs | f32 | op=softmax, len=64, batch=8, axis=0 | 1 | 64 | 35835.9 | 1146.5 | 30.8% | NOISY |
| softmax_f32_eager-composed_len64_b8 | #1976 #2031 #2231 | eager-composed | tenferro-rs | f32 | op=softmax, len=64, batch=8, axis=0 | 1 | 8 | 71713.5 | 2929.8 | 59.2% | NOISY |
| softmax_f32_pytorch_len64_b8 | #1976 #2031 #2231 | pytorch | pytorch-cpu | f32 | op=softmax, len=64, batch=8, axis=0 | 1 | 64 | 53184.2 | 4154.6 | 40.8% | NOISY |
| softmax_f32_eager-single-call_len512_b8 | #1976 #2231 | eager-single-call | tenferro-rs | f32 | op=softmax, len=512, batch=8, axis=0 | 1 | 1 | 2782417.0 | 281521.5 | 51.7% | NOISY |
| softmax_f32_session-single-call_len512_b8 | #1976 #2231 | session-single-call | tenferro-rs | f32 | op=softmax, len=512, batch=8, axis=0 | 1 | 1 | 2743458.0 | 97000.0 | 29.3% | NOISY |
| softmax_f32_eager-composed_len512_b8 | #1976 #2231 | eager-composed | tenferro-rs | f32 | op=softmax, len=512, batch=8, axis=0 | 1 | 1 | 4315666.0 | 2561625.0 | 39.0% | NOISY |
| softmax_f32_pytorch_len512_b8 | #1976 #2231 | pytorch | pytorch-cpu | f32 | op=softmax, len=512, batch=8, axis=0 | 1 | 1 | 3429458.0 | 46292.0 | 8.0% | ok |
| log_softmax_f32_eager-single-call_len64_b8 | #1976 | eager-single-call | tenferro-rs | f32 | op=log_softmax, len=64, batch=8, axis=0 | 1 | 32 | 87227.9 | 1515.6 | 1.2% | ok |
| log_softmax_f32_session-single-call_len64_b8 | #1976 | session-single-call | tenferro-rs | f32 | op=log_softmax, len=64, batch=8, axis=0 | 1 | 32 | 70949.2 | 1675.1 | 3.0% | ok |
| log_softmax_f32_eager-composed_len64_b8 | #1976 | eager-composed | tenferro-rs | f32 | op=log_softmax, len=64, batch=8, axis=0 | 1 | 32 | 73079.4 | 2367.9 | 6.2% | ok |
| log_softmax_f32_pytorch_len64_b8 | #1976 | pytorch | pytorch-cpu | f32 | op=log_softmax, len=64, batch=8, axis=0 | 1 | 64 | 47740.9 | 619.1 | 41.2% | NOISY |
| log_softmax_f32_eager-single-call_len512_b8 | #1976 | eager-single-call | tenferro-rs | f32 | op=log_softmax, len=512, batch=8, axis=0 | 1 | 1 | 3464875.0 | 1568854.0 | 47.5% | NOISY |
| log_softmax_f32_session-single-call_len512_b8 | #1976 | session-single-call | tenferro-rs | f32 | op=log_softmax, len=512, batch=8, axis=0 | 1 | 1 | 2888083.0 | 158897.0 | 3.1% | ok |
| log_softmax_f32_eager-composed_len512_b8 | #1976 | eager-composed | tenferro-rs | f32 | op=log_softmax, len=512, batch=8, axis=0 | 1 | 1 | 3136709.0 | 1786875.0 | 48.9% | NOISY |
| log_softmax_f32_pytorch_len512_b8 | #1976 | pytorch | pytorch-cpu | f32 | op=log_softmax, len=512, batch=8, axis=0 | 1 | 1 | 3041333.0 | 1433353.5 | 35.9% | NOISY |
| masked_softmax_f32_eager-single-call_len64_b8 | #1976 | eager-single-call | tenferro-rs | f32 | op=masked_softmax, len=64, batch=8, axis=0 | 1 | 16 | 111723.9 | 3770.8 | 4.6% | ok |
| masked_softmax_f32_session-single-call_len64_b8 | #1976 | session-single-call | tenferro-rs | f32 | op=masked_softmax, len=64, batch=8, axis=0 | 1 | 32 | 92647.1 | 1401.0 | 2.9% | ok |
| masked_softmax_f32_eager-composed_len64_b8 | #1976 | eager-composed | tenferro-rs | f32 | op=masked_softmax, len=64, batch=8, axis=0 | 1 | 32 | 87554.7 | 22195.3 | 88.0% | NOISY |
| masked_softmax_f32_pytorch_len64_b8 | #1976 | pytorch | pytorch-cpu | f32 | op=masked_softmax, len=64, batch=8, axis=0 | 1 | 32 | 76665.4 | 30485.7 | 41.5% | NOISY |
| masked_softmax_f32_eager-single-call_len512_b8 | #1976 | eager-single-call | tenferro-rs | f32 | op=masked_softmax, len=512, batch=8, axis=0 | 1 | 1 | 4276167.0 | 744916.5 | 34.0% | NOISY |
| masked_softmax_f32_session-single-call_len512_b8 | #1976 | session-single-call | tenferro-rs | f32 | op=masked_softmax, len=512, batch=8, axis=0 | 1 | 1 | 4417792.0 | 392917.0 | 11.2% | NOISY |
| masked_softmax_f32_eager-composed_len512_b8 | #1976 | eager-composed | tenferro-rs | f32 | op=masked_softmax, len=512, batch=8, axis=0 | 1 | 1 | 3848875.0 | 458270.0 | 18.3% | NOISY |
| masked_softmax_f32_pytorch_len512_b8 | #1976 | pytorch | pytorch-cpu | f32 | op=masked_softmax, len=512, batch=8, axis=0 | 1 | 1 | 5244500.0 | 3291416.5 | 39.2% | NOISY |
| softmax_f32_eager-single-call_len64_b8_t1 | #2233 #2235 | eager-single-call | tenferro-rs | f32 | op=softmax, len=64, batch=8, axis=0 | 1 | 64 | 37670.6 | 9692.1 | 44.1% | NOISY |
| softmax_f32_eager-composed_len64_b8_t1 | #2233 #2235 | eager-composed | tenferro-rs | f32 | op=softmax, len=64, batch=8, axis=0 | 1 | 32 | 75194.0 | 2091.1 | 8.1% | ok |
| softmax_f32_pytorch_len64_b8_t1 | #2233 #2235 | pytorch | pytorch-cpu | f32 | op=softmax, len=64, batch=8, axis=0 | 1 | 2 | 60042.0 | 90562.5 | 109.1% | NOISY |
| reduce_mean_f32_eager-single-call_1024x64_axis0 | #1976 | eager-single-call | tenferro-rs | f32 | rows=1024, cols=64, axis=0 | 1 | 256 | 13899.2 | 2034.3 | 38.2% | NOISY |
| reduce_mean_f32_session-single-call_1024x64_axis0 | #1976 | session-single-call | tenferro-rs | f32 | rows=1024, cols=64, axis=0 | 1 | 256 | 8210.0 | 1201.1 | 22.8% | NOISY |
| reduce_mean_f32_eager-composed_1024x64_axis0 | #1976 | eager-composed | tenferro-rs | f32 | rows=1024, cols=64, axis=0 | 1 | 256 | 13253.4 | 1564.5 | 25.8% | NOISY |
| reduce_mean_f32_pytorch_1024x64_axis0 | #1976 | pytorch | pytorch-cpu | f32 | rows=1024, cols=64, axis=0 | 1 | 128 | 6884.4 | 1571.3 | 53.6% | NOISY |
| take_along_axis_rows_f64_eager-single-call_n64_b256 | #2008 | eager-single-call | tenferro-rs | f64 | n=64, batch=256, axis=0 | 1 | 1 | 6729708.0 | 3017770.5 | 28.3% | NOISY |
| take_along_axis_rows_f64_session-single-call_n64_b256 | #2008 | session-single-call | tenferro-rs | f64 | n=64, batch=256, axis=0 | 1 | 1 | 6081083.0 | 211937.5 | 3.7% | ok |
| take_along_axis_rows_f64_eager-gather-prebuilt_n64_b256 | #2008 | eager-gather-prebuilt | tenferro-rs | f64 | n=64, batch=256, axis=0 | 1 | 1 | 5881708.0 | 230104.0 | 10.1% | NOISY |
| take_along_axis_rows_f64_pytorch_n64_b256 | #2008 | pytorch | pytorch-cpu | f64 | n=64, batch=256, axis=0 | 1 | 1 | 2265458.0 | 133771.0 | 3.9% | ok |
| lstsq_f64_eager-shared_m768_n384_rhs16 | #2027 | eager-shared | tenferro-rs | f64 | operation=lstsq | 1 | 1 | 18480208.0 | 869312.0 | 7.6% | ok |
| lstsq_f64_pytorch_m768_n384_rhs16 | #2027 | pytorch | pytorch-cpu | f64 | operation=lstsq | 1 | 1 | 5306333.0 | 104603.5 | 3.2% | ok |
| triangular_f64_concrete-shared_n4096_rhs64 | #2028 | concrete-shared | tenferro-rs | f64 | operation=triangular | 1 | 1 | 39035750.0 | 1235937.5 | 2.5% | ok |
| triangular_f64_pytorch_n4096_rhs64 | #2028 | pytorch | pytorch-cpu | f64 | operation=triangular | 1 | 1 | 13279958.0 | 344229.5 | 1.7% | ok |
| reshape_f64_concrete-shared_n33554432_to8192x4096 | #2029 | concrete-shared | tenferro-rs | f64 | operation=reshape | 1 | 1 | 11065625.0 | 1172209.0 | 10.8% | NOISY |
| reshape_f64_pytorch_n33554432_to8192x4096 | #2029 | pytorch | pytorch-cpu | f64 | operation=reshape | 1 | 1 | 11731791.0 | 175833.5 | 2.7% | ok |
| metadata_transpose_view_f64_rank8_t1 | #2180 | metadata-clone | tenferro-rs | f64 | shape=[2, 2, 2, 2, 2, 2, 2, 2] | 1 | 32768 | 55.4 | 1.3 | 1.7% | ok |
| metadata_transpose_view_f64_rank8_t1_julia-base | #2180 | julia-base | julia-base | f64 | shape=[2, 2, 2, 2, 2, 2, 2, 2] | 1 | 1048576 | 0.4 | 0.0 | 14.9% | NOISY |
| followup_structural_broadcast_in_dim_direct-shared | #2021 | direct-shared | tenferro-rs | f64 | case_id=structural.broadcast_in_dim, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [8192, 1], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | SKIPPED |
| followup_structural_broadcast_in_dim_pytorch | #2021 | pytorch | pytorch-cpu | f64 | case_id=structural.broadcast_in_dim, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [8192, 1], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | SKIPPED |
| followup_real_tanh_direct-shared | #2033 | direct-shared | tenferro-rs | f64 | case_id=real.tanh, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [8388608], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | SKIPPED |
| followup_real_tanh_jax | #2033 | jax | jax-cpu | f64 | case_id=real.tanh, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [8388608], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | SKIPPED |
| followup_real_log1p_direct-shared | #2033 | direct-shared | tenferro-rs | f64 | case_id=real.log1p, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [4194304], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | SKIPPED |
| followup_real_log1p_pytorch | #2033 | pytorch | pytorch-cpu | f64 | case_id=real.log1p, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [4194304], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | SKIPPED |
| followup_real_cos_direct-shared | #2033 | direct-shared | tenferro-rs | f64 | case_id=real.cos, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [8388608], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | SKIPPED |
| followup_real_cos_pytorch | #2033 | pytorch | pytorch-cpu | f64 | case_id=real.cos, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [8388608], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | SKIPPED |
| followup_real_sin_direct-shared | #2033 | direct-shared | tenferro-rs | f64 | case_id=real.sin, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [8388608], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | SKIPPED |
| followup_real_sin_pytorch | #2033 | pytorch | pytorch-cpu | f64 | case_id=real.sin, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [8388608], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | SKIPPED |
| followup_real_exp_direct-shared | #2033 | direct-shared | tenferro-rs | f64 | case_id=real.exp, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [8388608], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | SKIPPED |
| followup_real_exp_jax | #2033 | jax | jax-cpu | f64 | case_id=real.exp, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [8388608], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | SKIPPED |
| followup_real_expm1_direct-shared | #2033 | direct-shared | tenferro-rs | f64 | case_id=real.expm1, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [4194304], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | SKIPPED |
| followup_real_expm1_jax | #2033 | jax | jax-cpu | f64 | case_id=real.expm1, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [4194304], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | SKIPPED |
| followup_real_pow_direct-shared | #2033 | direct-shared | tenferro-rs | f64 | case_id=real.pow, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [4194304], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | SKIPPED |
| followup_real_pow_pytorch | #2033 | pytorch | pytorch-cpu | f64 | case_id=real.pow, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [4194304], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | SKIPPED |
| followup_complex_exp_direct-shared | #2033 | direct-shared | tenferro-rs | c64 | case_id=complex.exp, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [4194304], 'dtype': 'c64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | SKIPPED |
| followup_complex_exp_julia-base | #2033 | julia-base | julia-base | c64 | case_id=complex.exp, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [4194304], 'dtype': 'c64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | SKIPPED |
| followup_real_log_direct-shared | #2033 | direct-shared | tenferro-rs | f64 | case_id=real.log, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [8388608], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | SKIPPED |
| followup_real_log_pytorch | #2033 | pytorch | pytorch-cpu | f64 | case_id=real.log, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [8388608], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | SKIPPED |
| followup_real_reduce_max_axis1_direct-shared | #2034 | direct-shared | tenferro-rs | f64 | case_id=real.reduce_max_axis1, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [2048, 2048], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | SKIPPED |
| followup_real_reduce_max_axis1_julia-base | #2034 | julia-base | julia-base | f64 | case_id=real.reduce_max_axis1, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [2048, 2048], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | SKIPPED |
| followup_real_reduce_prod_axis1_direct-shared | #2034 | direct-shared | tenferro-rs | f64 | case_id=real.reduce_prod_axis1, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [2048, 2048], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | SKIPPED |
| followup_real_reduce_prod_axis1_pytorch | #2034 | pytorch | pytorch-cpu | f64 | case_id=real.reduce_prod_axis1, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [2048, 2048], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | SKIPPED |
| followup_real_reduce_sum_axis1_direct-shared | #2034 | direct-shared | tenferro-rs | f64 | case_id=real.reduce_sum_axis1, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [2048, 2048], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | SKIPPED |
| followup_real_reduce_sum_axis1_pytorch | #2034 | pytorch | pytorch-cpu | f64 | case_id=real.reduce_sum_axis1, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [2048, 2048], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | SKIPPED |
| followup_real_reduce_max_axis0_direct-shared | #2034 | direct-shared | tenferro-rs | f64 | case_id=real.reduce_max_axis0, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [2048, 2048], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | SKIPPED |
| followup_real_reduce_max_axis0_julia-base | #2034 | julia-base | julia-base | f64 | case_id=real.reduce_max_axis0, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [2048, 2048], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | SKIPPED |
| followup_real_reduce_min_axis1_direct-shared | #2034 | direct-shared | tenferro-rs | f64 | case_id=real.reduce_min_axis1, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [4096, 4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | SKIPPED |
| followup_real_reduce_min_axis1_julia-base | #2034 | julia-base | julia-base | f64 | case_id=real.reduce_min_axis1, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [4096, 4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | SKIPPED |
| followup_real_reduce_min_axis0_direct-shared | #2034 | direct-shared | tenferro-rs | f64 | case_id=real.reduce_min_axis0, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [2048, 2048], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | SKIPPED |
| followup_real_reduce_min_axis0_julia-base | #2034 | julia-base | julia-base | f64 | case_id=real.reduce_min_axis0, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [2048, 2048], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | SKIPPED |
| followup_real_reduce_max_all_direct-shared | #2034 | direct-shared | tenferro-rs | f64 | case_id=real.reduce_max_all, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [8192, 4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | SKIPPED |
| followup_real_reduce_max_all_julia-base | #2034 | julia-base | julia-base | f64 | case_id=real.reduce_max_all, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [8192, 4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | SKIPPED |
| followup_real_reduce_min_all_direct-shared | #2034 | direct-shared | tenferro-rs | f64 | case_id=real.reduce_min_all, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [8192, 4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | SKIPPED |
| followup_real_reduce_min_all_julia-base | #2034 | julia-base | julia-base | f64 | case_id=real.reduce_min_all, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [8192, 4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | SKIPPED |
| followup_complex_norm_fro_direct-shared | #2035 | direct-shared | tenferro-rs | c64 | case_id=complex.norm_fro, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [2048, 1536], 'dtype': 'c64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | SKIPPED |
| followup_complex_norm_fro_jax | #2035 | jax | jax-cpu | c64 | case_id=complex.norm_fro, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [2048, 1536], 'dtype': 'c64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | SKIPPED |
| followup_index_gather_direct-shared | #2036 #1719 | direct-shared | tenferro-rs | f64 | case_id=index.gather, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [262144], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | SKIPPED |
| followup_index_gather_jax | #2036 #1719 | jax | jax-cpu | f64 | case_id=index.gather, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [262144], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | SKIPPED |
| followup_index_scatter_direct-shared | #2036 #1719 | direct-shared | tenferro-rs | f64 | case_id=index.scatter, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [262144], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | SKIPPED |
| followup_index_scatter_pytorch | #2036 #1719 | pytorch | pytorch-cpu | f64 | case_id=index.scatter, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [262144], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | SKIPPED |
| followup_index_dynamic_update_slice_direct-shared | #2036 #1719 | direct-shared | tenferro-rs | f64 | case_id=index.dynamic_update_slice, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [2097152], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | SKIPPED |
| followup_index_dynamic_update_slice_jax | #2036 #1719 | jax | jax-cpu | f64 | case_id=index.dynamic_update_slice, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [2097152], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | SKIPPED |
| followup_perm_reverse_15d_3_materialize | #2037 #2228 | materialize | tenferro-rs | f64 | case_id=perm.reverse_15d_3, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | SKIPPED |
| followup_perm_reverse_15d_3_strided-rs | #2037 #2228 | strided-rs | strided-rs | f64 | case_id=perm.reverse_15d_3, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | SKIPPED |
| followup_structural_transpose_direct-shared | #2037 #2228 | direct-shared | tenferro-rs | f64 | case_id=structural.transpose, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [4096, 4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | SKIPPED |
| followup_structural_transpose_pytorch | #2037 #2228 | pytorch | pytorch-cpu | f64 | case_id=structural.transpose, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [4096, 4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | SKIPPED |
| followup_perm_reverse_23d_2_materialize | #2037 #2228 | materialize | tenferro-rs | f64 | case_id=perm.reverse_23d_2, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | SKIPPED |
| followup_perm_reverse_23d_2_strided-rs | #2037 #2228 | strided-rs | strided-rs | f64 | case_id=perm.reverse_23d_2, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | SKIPPED |
| followup_perm_transpose_3d_256_201_materialize | #2037 #2228 | materialize | tenferro-rs | f64 | case_id=perm.transpose_3d_256_201, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [256, 256, 256], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | SKIPPED |
| followup_perm_transpose_3d_256_201_strided-rs | #2037 #2228 | strided-rs | strided-rs | f64 | case_id=perm.transpose_3d_256_201, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [256, 256, 256], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | SKIPPED |
| followup_perm_cyclic_15d_3_materialize | #2037 #2228 | materialize | tenferro-rs | f64 | case_id=perm.cyclic_15d_3, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | SKIPPED |
| followup_perm_cyclic_15d_3_strided-rs | #2037 #2228 | strided-rs | strided-rs | f64 | case_id=perm.cyclic_15d_3, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | SKIPPED |
| followup_perm_transpose_3d_256_102_materialize | #2037 #2228 | materialize | tenferro-rs | f64 | case_id=perm.transpose_3d_256_102, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [256, 256, 256], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | SKIPPED |
| followup_perm_transpose_3d_256_102_strided-rs | #2037 #2228 | strided-rs | strided-rs | f64 | case_id=perm.transpose_3d_256_102, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [256, 256, 256], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | SKIPPED |
| followup_perm_tn_light_415_24d_scattered_to_colmajor_materialize | #2037 #2228 | materialize | tenferro-rs | f64 | case_id=perm.tn_light_415_24d_scattered_to_colmajor, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | SKIPPED |
| followup_perm_tn_light_415_24d_scattered_to_colmajor_strided-rs | #2037 #2228 | strided-rs | strided-rs | f64 | case_id=perm.tn_light_415_24d_scattered_to_colmajor, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | SKIPPED |
| followup_perm_tn_light_415_24d_contiguous_same_perm_materialize | #2037 #2228 | materialize | tenferro-rs | f64 | case_id=perm.tn_light_415_24d_contiguous_same_perm, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | SKIPPED |
| followup_perm_tn_light_415_24d_contiguous_same_perm_strided-rs | #2037 #2228 | strided-rs | strided-rs | f64 | case_id=perm.tn_light_415_24d_contiguous_same_perm, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | SKIPPED |
| followup_perm_rotation_6d_32_32_32_32_16_16_materialize | #2037 #2228 | materialize | tenferro-rs | f64 | case_id=perm.rotation_6d_32_32_32_32_16_16, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [32, 32, 32, 32, 16, 16], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | SKIPPED |
| followup_perm_rotation_6d_32_32_32_32_16_16_strided-rs | #2037 #2228 | strided-rs | strided-rs | f64 | case_id=perm.rotation_6d_32_32_32_32_16_16, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [32, 32, 32, 32, 16, 16], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | SKIPPED |
| followup_structural_extract_diagonal_direct-shared | #2038 #2228 | direct-shared | tenferro-rs | f64 | case_id=structural.extract_diagonal, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [8388608, 2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | SKIPPED |
| followup_structural_extract_diagonal_jax | #2038 #2228 | jax | jax-cpu | f64 | case_id=structural.extract_diagonal, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [8388608, 2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | SKIPPED |
| followup_fft_irfft_direct-shared | #2039 #2230 #2228 | direct-shared | tenferro-rs | c32 | case_id=fft.irfft, rust_path=direct-shared, reference_path=mkl-dfti, fixture={'input_shape': [524289], 'dtype': 'c32', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | SKIPPED |
| followup_fft_irfft_mkl-dfti | #2039 #2230 #2228 | mkl-dfti | mkl-dfti | c32 | case_id=fft.irfft, rust_path=direct-shared, reference_path=mkl-dfti, fixture={'input_shape': [524289], 'dtype': 'c32', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | SKIPPED |
| followup_fft_rfft_direct-shared | #2039 #2230 #2228 | direct-shared | tenferro-rs | f32 | case_id=fft.rfft, rust_path=direct-shared, reference_path=mkl-dfti, fixture={'input_shape': [1048576], 'dtype': 'f32', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | SKIPPED |
| followup_fft_rfft_mkl-dfti | #2039 #2230 #2228 | mkl-dfti | mkl-dfti | f32 | case_id=fft.rfft, rust_path=direct-shared, reference_path=mkl-dfti, fixture={'input_shape': [1048576], 'dtype': 'f32', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | SKIPPED |
| followup_fft_fft_direct-shared | #2039 #2228 | direct-shared | tenferro-rs | c64 | case_id=fft.fft, rust_path=direct-shared, reference_path=mkl-dfti, fixture={'input_shape': [1048576], 'dtype': 'c64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | SKIPPED |
| followup_fft_fft_mkl-dfti | #2039 #2228 | mkl-dfti | mkl-dfti | c64 | case_id=fft.fft, rust_path=direct-shared, reference_path=mkl-dfti, fixture={'input_shape': [1048576], 'dtype': 'c64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | SKIPPED |
| followup_fft_ifft_direct-shared | #2039 #2228 | direct-shared | tenferro-rs | c32 | case_id=fft.ifft, rust_path=direct-shared, reference_path=mkl-dfti, fixture={'input_shape': [1048576], 'dtype': 'c32', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | SKIPPED |
| followup_fft_ifft_mkl-dfti | #2039 #2228 | mkl-dfti | mkl-dfti | c32 | case_id=fft.ifft, rust_path=direct-shared, reference_path=mkl-dfti, fixture={'input_shape': [1048576], 'dtype': 'c32', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | SKIPPED |
| followup_metadata_transpose_view_metadata | #2040 #1719 | metadata | tenferro-rs | f64 | case_id=metadata.transpose_view, rust_path=metadata, reference_path=julia-base, fixture={'input_shape': [2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 1 | — | — | — | — | SKIPPED |
| followup_metadata_transpose_view_julia-base | #2040 #1719 | julia-base | julia-base | f64 | case_id=metadata.transpose_view, rust_path=metadata, reference_path=julia-base, fixture={'input_shape': [2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 1 | — | — | — | — | SKIPPED |
| followup_metadata_transpose_view_metadata-host-dyn | #2040 #1719 | metadata-host-dyn | tenferro-rs | f64 | case_id=metadata.transpose_view, rust_path=metadata-host-dyn, reference_path=julia-base, fixture={'input_shape': [2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 1 | — | — | — | — | SKIPPED |
| followup_metadata_transpose_view_metadata-host-static | #2040 #1719 | metadata-host-static | tenferro-rs | f64 | case_id=metadata.transpose_view, rust_path=metadata-host-static, reference_path=julia-base, fixture={'input_shape': [2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 1 | — | — | — | — | SKIPPED |
| followup_metadata_slice_view_metadata | #2040 #1719 | metadata | tenferro-rs | f64 | case_id=metadata.slice_view, rust_path=metadata, reference_path=julia-base, fixture={'input_shape': [4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 1 | — | — | — | — | SKIPPED |
| followup_metadata_slice_view_julia-base | #2040 #1719 | julia-base | julia-base | f64 | case_id=metadata.slice_view, rust_path=metadata, reference_path=julia-base, fixture={'input_shape': [4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 1 | — | — | — | — | SKIPPED |
| followup_metadata_slice_view_metadata-host-dyn | #2040 #1719 | metadata-host-dyn | tenferro-rs | f64 | case_id=metadata.slice_view, rust_path=metadata-host-dyn, reference_path=julia-base, fixture={'input_shape': [4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 1 | — | — | — | — | SKIPPED |
| followup_metadata_slice_view_metadata-host-static | #2040 #1719 | metadata-host-static | tenferro-rs | f64 | case_id=metadata.slice_view, rust_path=metadata-host-static, reference_path=julia-base, fixture={'input_shape': [4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 1 | — | — | — | — | SKIPPED |
| followup_metadata_reshape_view_metadata | #2040 #1719 | metadata | tenferro-rs | f64 | case_id=metadata.reshape_view, rust_path=metadata, reference_path=julia-base, fixture={'input_shape': [1024], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 1 | — | — | — | — | SKIPPED |
| followup_metadata_reshape_view_julia-base | #2040 #1719 | julia-base | julia-base | f64 | case_id=metadata.reshape_view, rust_path=metadata, reference_path=julia-base, fixture={'input_shape': [1024], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 1 | — | — | — | — | SKIPPED |
| followup_metadata_reshape_view_metadata-host-dyn | #2040 #1719 | metadata-host-dyn | tenferro-rs | f64 | case_id=metadata.reshape_view, rust_path=metadata-host-dyn, reference_path=julia-base, fixture={'input_shape': [1024], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 1 | — | — | — | — | SKIPPED |
| followup_metadata_reshape_view_metadata-host-static | #2040 #1719 | metadata-host-static | tenferro-rs | f64 | case_id=metadata.reshape_view, rust_path=metadata-host-static, reference_path=julia-base, fixture={'input_shape': [1024], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 1 | — | — | — | — | SKIPPED |
| typed_elemwise_f64_add_held_len16 | #2139 | tenferro | tenferro-rs | f64 | n=16, route=held | 1 | 4096 | 709.7 | 44.0 | 3.3% | ok |
| typed_elemwise_f64_add_held_len16_pytorch | #2139 | pytorch | pytorch-cpu | f64 | n=16, route=held | 1 | 4096 | 759.3 | 83.4 | 280.9% | NOISY |
| typed_elemwise_f64_add_into_held_len16 | #2139 | tenferro | tenferro-rs | f64 | n=16, route=into_held | 1 | 4096 | 547.8 | 8.6 | 1.2% | ok |
| typed_elemwise_f64_add_into_held_len16_pytorch | #2139 | pytorch | pytorch-cpu | f64 | n=16, route=into_held | 1 | 4096 | 786.9 | 34.4 | 6.8% | ok |
| typed_elemwise_f64_add_held_len256 | #2139 | tenferro | tenferro-rs | f64 | n=256, route=held | 1 | 4096 | 773.7 | 8.9 | 1.5% | ok |
| typed_elemwise_f64_add_held_len256_pytorch | #2139 | pytorch | pytorch-cpu | f64 | n=256, route=held | 1 | 2048 | 818.9 | 47.4 | 5.8% | ok |
| typed_elemwise_f64_add_into_held_len256 | #2139 | tenferro | tenferro-rs | f64 | n=256, route=into_held | 1 | 4096 | 586.5 | 12.8 | 3.3% | ok |
| typed_elemwise_f64_add_into_held_len256_pytorch | #2139 | pytorch | pytorch-cpu | f64 | n=256, route=into_held | 1 | 1024 | 858.8 | 124.6 | 40.2% | NOISY |
| typed_elemwise_f64_add_held_len4096 | #2139 | tenferro | tenferro-rs | f64 | n=4096, route=held | 1 | 512 | 1453.3 | 115.2 | 8.9% | ok |
| typed_elemwise_f64_add_held_len4096_pytorch | #2139 | pytorch | pytorch-cpu | f64 | n=4096, route=held | 1 | 1024 | 1646.9 | 49.3 | 3.3% | ok |
| typed_elemwise_f64_add_into_held_len4096 | #2139 | tenferro | tenferro-rs | f64 | n=4096, route=into_held | 1 | 2048 | 1067.4 | 10.7 | 1.3% | ok |
| typed_elemwise_f64_add_into_held_len4096_pytorch | #2139 | pytorch | pytorch-cpu | f64 | n=4096, route=into_held | 1 | 2048 | 1450.5 | 12.4 | 1.7% | ok |
| tiny_matmul_2x2_typed_held | #2139 | tenferro | tenferro-rs | f64 | n=2 | 1 | 4096 | 838.1 | 7.7 | 2.6% | ok |
| tiny_matmul_2x2_typed_held_pytorch | #2139 | pytorch | pytorch-cpu | f64 | n=2 | 1 | 4096 | 707.9 | 8.8 | 1.0% | ok |
| analytic_neg_f64_sweep_8MiB_t1 | #2139 | tenferro | tenferro-rs | f64 | n=1048576 | 1 | 4 | 254552.0 | 33406.2 | 26.2% | NOISY |
| analytic_neg_f64_sweep_8MiB_t1_pytorch | #2139 | pytorch | pytorch-cpu | f64 | n=1048576 | 1 | 8 | 255958.4 | 10085.9 | 9.6% | ok |
| analytic_neg_f64_sweep_32MiB_t1 | #2139 | tenferro | tenferro-rs | f64 | n=4194304 | 1 | 2 | 1737125.0 | 337780.8 | 13.3% | NOISY |
| analytic_neg_f64_sweep_32MiB_t1_pytorch | #2139 | pytorch | pytorch-cpu | f64 | n=4194304 | 1 | 2 | 1367625.0 | 83833.5 | 5.0% | ok |
| gemm_bound_f64_t1_t8_scaling | #2141 | tenferro | tenferro-rs | f64 | n=1536 | 1 | 1 | 182087458.0 | 15162958.5 | 6.1% | ok |
| gemm_bound_f64_t1_t8_scaling_pytorch | #2141 | pytorch | pytorch-cpu | f64 | n=1536 | 1 | 1 | 40323458.0 | 698583.5 | 5.7% | ok |
| lu_batched_f64_factor_n8_b1024 | #2142 | tenferro | tenferro-rs | f64 | n=8, batch=1024, op=factor | 1 | 8 | 245072.9 | 5031.2 | 2.7% | ok |
| lu_batched_f64_factor_n8_b1024_host | #2142 | host | rust-loop | f64 | n=8, batch=1024, op=factor | 1 | 8 | 245255.1 | 6450.6 | 4.1% | ok |
| lu_batched_f64_solve_n8_b1024 | #2142 | tenferro | tenferro-rs | f64 | n=8, batch=1024, op=solve | 1 | 16 | 134820.3 | 2121.1 | 1.1% | ok |
| lu_batched_f64_solve_n8_b1024_pytorch | #2142 | pytorch | pytorch-cpu | f64 | n=8, batch=1024, op=solve | 1 | 32 | 99138.0 | 789.7 | 3.9% | ok |
| lu_batched_f64_factor_n16_b1024 | #2142 | tenferro | tenferro-rs | f64 | n=16, batch=1024, op=factor | 1 | 4 | 805062.5 | 47661.5 | 4.6% | ok |
| lu_batched_f64_factor_n16_b1024_host | #2142 | host | rust-loop | f64 | n=16, batch=1024, op=factor | 1 | 2 | 1087771.0 | 19301.8 | 2.8% | ok |
| lu_batched_f64_solve_n16_b1024 | #2142 | tenferro | tenferro-rs | f64 | n=16, batch=1024, op=solve | 1 | 8 | 435281.2 | 6112.1 | 1.5% | ok |
| lu_batched_f64_solve_n16_b1024_pytorch | #2142 | pytorch | pytorch-cpu | f64 | n=16, batch=1024, op=solve | 1 | 16 | 219919.3 | 2214.8 | 2.1% | ok |
| axpby_accum_strided_dest_f64_block | #2143 | tenferro | tenferro-rs | f64 | n=64 | 1 | 512 | 4554.5 | 396.2 | 6.3% | ok |
| axpby_accum_strided_dest_f64_block_host | #2143 | host | rust-loop | f64 | n=64 | 1 | 512 | 4922.3 | 696.2 | 17.2% | NOISY |
| small_grad_qr_f64_trace-prepared_n4 | #2186 | trace-prepared | tenferro-rs | f64 | op=qr, n=4 | 1 | 128 | 26008.8 | 1328.3 | 12.6% | NOISY |
| small_grad_qr_f64_pytorch_n4 | #2186 | pytorch | pytorch-cpu | f64 | op=qr, n=4 | 1 | 64 | 44497.4 | 11923.8 | 35.2% | NOISY |
| small_grad_qr_f64_trace-prepared_n16 | #2186 | trace-prepared | tenferro-rs | f64 | op=qr, n=16 | 1 | 64 | 35904.3 | 635.1 | 1.6% | ok |
| small_grad_qr_f64_pytorch_n16 | #2186 | pytorch | pytorch-cpu | f64 | op=qr, n=16 | 1 | 64 | 54406.2 | 280.3 | 0.7% | ok |
| small_grad_eigh_f64_trace-prepared_n4 | #2186 | trace-prepared | tenferro-rs | f64 | op=eigh, n=4 | 1 | 64 | 35342.5 | 1381.8 | 10.1% | NOISY |
| small_grad_eigh_f64_pytorch_n4 | #2186 | pytorch | pytorch-cpu | f64 | op=eigh, n=4 | 1 | 64 | 43727.2 | 3066.1 | 5.7% | ok |
| small_grad_eigh_f64_trace-prepared_n16 | #2186 | trace-prepared | tenferro-rs | f64 | op=eigh, n=16 | 1 | 32 | 69695.3 | 4608.1 | 4.7% | ok |
| small_grad_eigh_f64_pytorch_n16 | #2186 | pytorch | pytorch-cpu | f64 | op=eigh, n=16 | 1 | 32 | 68261.7 | 2594.4 | 5.8% | ok |
| small_grad_lu_f64_trace-prepared_n4 | #2186 | trace-prepared | tenferro-rs | f64 | op=lu, n=4 | 1 | 128 | 17779.3 | 680.8 | 3.6% | ok |
| small_grad_lu_f64_pytorch_n4 | #2186 | pytorch | pytorch-cpu | f64 | op=lu, n=4 | 1 | 64 | 44897.1 | 2008.8 | 7.0% | ok |
| small_grad_lu_f64_trace-prepared_n16 | #2186 | trace-prepared | tenferro-rs | f64 | op=lu, n=16 | 1 | 128 | 24214.5 | 2580.7 | 21.4% | NOISY |
| small_grad_lu_f64_pytorch_n16 | #2186 | pytorch | pytorch-cpu | f64 | op=lu, n=16 | 1 | 64 | 56863.9 | 7700.8 | 32.7% | NOISY |
| small_grad_svd_f64_trace-prepared_n4 | #2186 | trace-prepared | tenferro-rs | f64 | op=svd, n=4 | 1 | 64 | 53371.8 | 954.4 | 1.3% | ok |
| small_grad_svd_f64_pytorch_n4 | #2186 | pytorch | pytorch-cpu | f64 | op=svd, n=4 | 1 | 32 | 60013.0 | 1682.3 | 3.2% | ok |
| small_grad_svd_f64_trace-prepared_n16 | #2186 | trace-prepared | tenferro-rs | f64 | op=svd, n=16 | 1 | 32 | 107373.7 | 1580.1 | 1.1% | ok |
| small_grad_svd_f64_pytorch_n16 | #2186 | pytorch | pytorch-cpu | f64 | op=svd, n=16 | 1 | 32 | 108026.0 | 3026.0 | 7.0% | ok |
| small_grad_solve_f64_trace-prepared_n4 | #2186 | trace-prepared | tenferro-rs | f64 | op=solve, n=4 | 1 | 256 | 15213.2 | 469.6 | 4.7% | ok |
| small_grad_solve_f64_pytorch_n4 | #2186 | pytorch | pytorch-cpu | f64 | op=solve, n=4 | 1 | 64 | 37830.7 | 1930.3 | 6.2% | ok |
| small_grad_solve_f64_trace-prepared_n16 | #2186 | trace-prepared | tenferro-rs | f64 | op=solve, n=16 | 1 | 128 | 19555.3 | 861.7 | 3.9% | ok |
| small_grad_solve_f64_pytorch_n16 | #2186 | pytorch | pytorch-cpu | f64 | op=solve, n=16 | 1 | 64 | 43308.6 | 424.5 | 3.8% | ok |
| fft_axis0_c64_fft-executor_n128 | #1765 | fft-executor | tenferro-rs | c64 | n=128 | 1 | 1 | 7657208.0 | 105500.0 | 11.0% | NOISY |
| fft_axis0_c64_pytorch_n128 | #1765 | pytorch | pytorch-cpu | c64 | n=128 | 1 | 1 | 21968750.0 | 6168687.5 | 15.5% | NOISY |
| degenerate_dot_hadamard_f64_dot-general-shared_n1024 | #2111 | dot-general-shared | tenferro-rs | f64 | form=hadamard, n=1024 | 1 | 4 | 489739.5 | 22687.5 | 4.4% | ok |
| degenerate_dot_hadamard_f64_pytorch_n1024 | #2111 | pytorch | pytorch-cpu | f64 | form=hadamard, n=1024 | 1 | 2 | 438541.5 | 16406.2 | 3.3% | ok |
| degenerate_dot_outer_f64_dot-general-shared_n1024 | #2111 | dot-general-shared | tenferro-rs | f64 | form=outer, n=1024 | 1 | 4 | 335604.0 | 15198.0 | 6.6% | ok |
| degenerate_dot_outer_f64_pytorch_n1024 | #2111 | pytorch | pytorch-cpu | f64 | form=outer, n=1024 | 1 | 4 | 197843.8 | 6630.2 | 4.7% | ok |
| degenerate_dot_rank0_f64_dot-general-shared_n1024 | #2111 | dot-general-shared | tenferro-rs | f64 | form=rank0, n=1024 | 1 | 4 | 631552.2 | 108770.8 | 9.8% | ok |
| degenerate_dot_rank0_f64_pytorch_n1024 | #2111 | pytorch | pytorch-cpu | f64 | form=rank0, n=1024 | 1 | 4 | 290291.8 | 26968.8 | 13.2% | NOISY |
| erased_add_f64_session-add_n256 | #1766 | session-add | tenferro-rs | f64 | n=256 | 1 | 4096 | 682.5 | 46.1 | 7.2% | ok |
| erased_add_f64_pytorch_n256 | #1766 | pytorch | pytorch-cpu | f64 | n=256 | 1 | 2048 | 914.9 | 43.2 | 7.0% | ok |
| erased_add_f64_session-add_n4096 | #1766 | session-add | tenferro-rs | f64 | n=4096 | 1 | 1024 | 1656.8 | 49.5 | 4.7% | ok |
| erased_add_f64_pytorch_n4096 | #1766 | pytorch | pytorch-cpu | f64 | n=4096 | 1 | 512 | 1755.2 | 55.0 | 3.4% | ok |
| erased_add_f64_session-add_n1048576 | #1766 | session-add | tenferro-rs | f64 | n=1048576 | 1 | 2 | 567416.5 | 48468.8 | 7.3% | ok |
| erased_add_f64_pytorch_n1048576 | #1766 | pytorch | pytorch-cpu | f64 | n=1048576 | 1 | 2 | 529125.0 | 55531.5 | 11.7% | NOISY |
| axpby_strided_source_f64_1024x1024_t1 | #2220 #2214 | strided-source | tenferro-rs | f64 | n=1024 | 1 | 2 | 1645521.0 | 52718.8 | 5.7% | ok |
| axpby_contiguous_source_f64_1024x1024_t1 | #2220 #2214 | contiguous-source | tenferro-rs | f64 | n=1024 | 1 | 4 | 514375.0 | 18088.5 | 6.5% | ok |
| axpby_strided_source_f64_1024x1024_host_t1 | #2220 #2214 | host | rust-loop | f64 | n=1024 | 1 | 2 | 1662104.0 | 60271.0 | 2.6% | ok |
| erased_construct_f64_col_n8 | #2177 | col-major | tenferro-rs | f64 | n=8, layout=col | 1 | 32768 | 56.7 | 1.7 | 5.5% | ok |
| erased_construct_f64_row_n8 | #2177 | row-major | tenferro-rs | f64 | n=8, layout=row | 1 | 16384 | 130.0 | 4.3 | 3.3% | ok |
| erased_construct_f64_col_n4096 | #2177 | col-major | tenferro-rs | f64 | n=4096, layout=col | 1 | 16384 | 68.4 | 8.3 | 8.9% | ok |
| erased_construct_f64_row_n4096 | #2177 | row-major | tenferro-rs | f64 | n=4096, layout=row | 1 | 128 | 16703.8 | 1213.1 | 8.6% | ok |
| erased_construct_f64_col_n65536 | #2177 | col-major | tenferro-rs | f64 | n=65536, layout=col | 1 | 1024 | 99.0 | 15.0 | 16.0% | NOISY |
| erased_construct_f64_row_n65536 | #2177 | row-major | tenferro-rs | f64 | n=65536, layout=row | 1 | 16 | 244054.7 | 5018.2 | 1.7% | ok |
| eager_small_leaf_f64_eager-held-session | #1704 | eager-held-session | tenferro-rs | f64 | op=leaf | 1 | 1024 | 1835.2 | 116.6 | 13.5% | NOISY |
| eager_small_add_f64_eager-held-session | #1667 | eager-held-session | tenferro-rs | f64 | op=add | 1 | 2048 | 1437.5 | 53.7 | 4.4% | ok |
| eager_small_add_f64_pytorch | #1667 | pytorch | pytorch-cpu | f64 | op=add | 1 | 4096 | 786.7 | 17.5 | 2.1% | ok |
| eager_small_gemm_f64_eager-held-session | #1904 | eager-held-session | tenferro-rs | f64 | op=gemm | 1 | 1024 | 2113.6 | 164.8 | 6.4% | ok |
| eager_small_gemm_f64_pytorch | #1904 | pytorch | pytorch-cpu | f64 | op=gemm | 1 | 4096 | 846.2 | 27.3 | 3.4% | ok |
| trace_metadata_trace-fold_f32_1024x64_k50 | #2226 #2238 | trace-fold | tenferro-rs | f32 | rows=1024, cols=64, steps=50 | 1 | 2 | 1856875.0 | 50365.0 | 10.0% | NOISY |
| trace_metadata_trace-outputs_f32_1024x64_k50 | #2226 #2238 | trace-outputs | tenferro-rs | f32 | rows=1024, cols=64, steps=50 | 1 | 2 | 1926666.5 | 163969.0 | 5.8% | ok |
| trace_metadata_eager_f32_1024x64_k50 | #2226 #2238 | eager | tenferro-rs | f32 | rows=1024, cols=64, steps=50 | 1 | 1 | 2075750.0 | 165688.0 | 5.0% | ok |
| trace_metadata_pytorch_f32_1024x64_k50 | #2226 #2238 | pytorch | pytorch-cpu | f32 | rows=1024, cols=64, steps=50 | 1 | 2 | 1416166.5 | 101510.5 | 4.2% | ok |
| trace_metadata_fixed_neg_f32_8 | #2226 #2238 | trace-fold | tenferro-rs | f32 | rows=8, cols=1, steps=0 | 1 | 2048 | 1458.3 | 170.7 | 16.5% | NOISY |
| two_operand_einsum_f64_2x2_string | #2325 #2324 | string | tenferro-rs | f64 | m=2, k=2, n=2, format=ij,jk->ik | 4 | 1024 | 2033.1 | 177.2 | 5.5% | ok |
| two_operand_einsum_f64_2x2_subscripts | #2325 #2324 | subscripts | tenferro-rs | f64 | m=2, k=2, n=2, format=ij,jk->ik | 4 | 2048 | 1320.2 | 66.9 | 9.1% | ok |
| two_operand_einsum_f64_2x2_prepared | #2325 #2324 | prepared | tenferro-rs | f64 | m=2, k=2, n=2, format=ij,jk->ik | 4 | 2048 | 1013.5 | 50.8 | 18.9% | NOISY |
| two_operand_einsum_f64_2x2_dot-general | #2325 #2324 | dot-general | tenferro-rs | f64 | m=2, k=2, n=2, format=ij,jk->ik | 4 | 2048 | 939.0 | 25.7 | 13.4% | NOISY |
| two_operand_einsum_f64_2x2_pytorch | #2325 #2324 | pytorch | pytorch-cpu | f64 | m=2, k=2, n=2, format=ij,jk->ik | 4 | 512 | 7514.1 | 310.5 | 6.0% | ok |
| two_operand_einsum_f64_16x16_string | #2325 #2324 | string | tenferro-rs | f64 | m=16, k=16, n=16, format=ij,jk->ik | 4 | 1024 | 2598.1 | 316.2 | 13.9% | NOISY |
| two_operand_einsum_f64_16x16_subscripts | #2325 #2324 | subscripts | tenferro-rs | f64 | m=16, k=16, n=16, format=ij,jk->ik | 4 | 1024 | 1975.5 | 768.1 | 34.2% | NOISY |
| two_operand_einsum_f64_16x16_prepared | #2325 #2324 | prepared | tenferro-rs | f64 | m=16, k=16, n=16, format=ij,jk->ik | 4 | 2048 | 1568.4 | 175.1 | 12.6% | NOISY |
| two_operand_einsum_f64_16x16_dot-general | #2325 #2324 | dot-general | tenferro-rs | f64 | m=16, k=16, n=16, format=ij,jk->ik | 4 | 2048 | 1417.0 | 28.6 | 3.3% | ok |
| two_operand_einsum_f64_16x16_pytorch | #2325 #2324 | pytorch | pytorch-cpu | f64 | m=16, k=16, n=16, format=ij,jk->ik | 4 | 256 | 10083.2 | 203.1 | 7.7% | ok |
| two_operand_einsum_c64_2x2_string | #2325 #2324 | string | tenferro-rs | c64 | m=2, k=2, n=2, format=ij,jk->ik | 4 | 1024 | 2109.7 | 93.3 | 11.8% | NOISY |
| two_operand_einsum_c64_2x2_subscripts | #2325 #2324 | subscripts | tenferro-rs | c64 | m=2, k=2, n=2, format=ij,jk->ik | 4 | 1024 | 1395.4 | 120.5 | 15.0% | NOISY |
| two_operand_einsum_c64_2x2_prepared | #2325 #2324 | prepared | tenferro-rs | c64 | m=2, k=2, n=2, format=ij,jk->ik | 4 | 2048 | 1020.8 | 19.5 | 2.5% | ok |
| two_operand_einsum_c64_2x2_dot-general | #2325 #2324 | dot-general | tenferro-rs | c64 | m=2, k=2, n=2, format=ij,jk->ik | 4 | 2048 | 950.7 | 13.1 | 2.2% | ok |
| two_operand_einsum_c64_2x2_pytorch | #2325 #2324 | pytorch | pytorch-cpu | c64 | m=2, k=2, n=2, format=ij,jk->ik | 4 | 512 | 7413.2 | 164.0 | 11.6% | NOISY |
| two_operand_einsum_c64_16x16_string | #2325 #2324 | string | tenferro-rs | c64 | m=16, k=16, n=16, format=ij,jk->ik | 4 | 256 | 3509.0 | 352.5 | 13.5% | NOISY |
| two_operand_einsum_c64_16x16_subscripts | #2325 #2324 | subscripts | tenferro-rs | c64 | m=16, k=16, n=16, format=ij,jk->ik | 4 | 1024 | 2556.2 | 104.3 | 8.8% | ok |
| two_operand_einsum_c64_16x16_prepared | #2325 #2324 | prepared | tenferro-rs | c64 | m=16, k=16, n=16, format=ij,jk->ik | 4 | 1024 | 2125.9 | 69.7 | 2.0% | ok |
| two_operand_einsum_c64_16x16_dot-general | #2325 #2324 | dot-general | tenferro-rs | c64 | m=16, k=16, n=16, format=ij,jk->ik | 4 | 1024 | 1985.1 | 12.2 | 1.6% | ok |
| two_operand_einsum_c64_16x16_pytorch | #2325 #2324 | pytorch | pytorch-cpu | c64 | m=16, k=16, n=16, format=ij,jk->ik | 4 | 256 | 11438.5 | 763.6 | 8.5% | ok |
| two_operand_einsum_f64_abc_cef_abef_string | #2325 #2324 | string | tenferro-rs | f64 | m=8, k=2, n=16, format=abc,cef->abef | 4 | 1024 | 2889.0 | 1208.5 | 40.7% | NOISY |
| two_operand_einsum_f64_abc_cef_abef_subscripts | #2325 #2324 | subscripts | tenferro-rs | f64 | m=8, k=2, n=16, format=abc,cef->abef | 4 | 2048 | 1473.2 | 79.6 | 10.1% | NOISY |
| two_operand_einsum_f64_abc_cef_abef_prepared | #2325 #2324 | prepared | tenferro-rs | f64 | m=8, k=2, n=16, format=abc,cef->abef | 4 | 2048 | 1067.6 | 27.3 | 1.6% | ok |
| two_operand_einsum_f64_abc_cef_abef_dot-general | #2325 #2324 | dot-general | tenferro-rs | f64 | m=8, k=2, n=16, format=abc,cef->abef | 4 | 2048 | 990.2 | 10.1 | 0.9% | ok |
| two_operand_einsum_f64_abc_cef_abef_pytorch | #2325 #2324 | pytorch | pytorch-cpu | f64 | m=8, k=2, n=16, format=abc,cef->abef | 4 | 256 | 8534.5 | 171.2 | 3.2% | ok |
| thin_c64_gemm_8x141_141x282_t4 | #2524 #2416 #1099 | dot-general | tenferro-rs | c64 | m=8, k=141, n=282 | 4 | 32 | 87846.3 | 29036.5 | 23.6% | NOISY |
| thin_c64_gemm_8x141_141x282_t4_pytorch | #2524 #2416 #1099 | pytorch | pytorch-cpu | c64 | m=8, k=141, n=282 | 4 | 64 | 42981.8 | 958.0 | 2.4% | ok |
| decode_proj_f32_eager-shared_in1024_out1024_len8 | #1992 #2003 #1995 #2229 | eager-shared | tenferro-rs | f32 | in=1024, out=1024, len=8 | 4 | 8 | 236859.4 | 44075.6 | 11.0% | NOISY |
| decode_proj_f32_faer-direct_in1024_out1024_len8 | #1992 #2003 #1995 #2229 | faer-direct | faer-direct | f32 | in=1024, out=1024, len=8 | 4 | 8 | 241369.8 | 21252.7 | 5.7% | ok |
| decode_proj_f32_host-sgemm_in1024_out1024_len8 | #1992 #2003 #1995 #2229 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=1024, len=8 | 4 | 8 | 389583.4 | 28078.1 | 5.5% | ok |
| decode_proj_f32_pytorch_in1024_out1024_len8 | #1992 #2003 #1995 #2229 | pytorch | pytorch-cpu | f32 | in=1024, out=1024, len=8 | 4 | 16 | 124205.8 | 6110.7 | 4.9% | ok |
| decode_proj_f32_eager-shared_in1024_out1024_len64 | #1992 #2003 #1995 #2229 | eager-shared | tenferro-rs | f32 | in=1024, out=1024, len=64 | 4 | 2 | 1158208.5 | 273219.0 | 27.4% | NOISY |
| decode_proj_f32_faer-direct_in1024_out1024_len64 | #1992 #2003 #1995 #2229 | faer-direct | faer-direct | f32 | in=1024, out=1024, len=64 | 4 | 2 | 1033250.0 | 71541.8 | 3.6% | ok |
| decode_proj_f32_host-sgemm_in1024_out1024_len64 | #1992 #2003 #1995 #2229 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=1024, len=64 | 4 | 2 | 1132854.5 | 24114.8 | 1.9% | ok |
| decode_proj_f32_pytorch_in1024_out1024_len64 | #1992 #2003 #1995 #2229 | pytorch | pytorch-cpu | f32 | in=1024, out=1024, len=64 | 4 | 16 | 203442.7 | 26647.1 | 7.4% | ok |
| decode_proj_f32_eager-shared_in1024_out1024_len512 | #1992 #2003 #1995 #2229 | eager-shared | tenferro-rs | f32 | in=1024, out=1024, len=512 | 4 | 1 | 6879917.0 | 267271.0 | 4.1% | ok |
| decode_proj_f32_faer-direct_in1024_out1024_len512 | #1992 #2003 #1995 #2229 | faer-direct | faer-direct | f32 | in=1024, out=1024, len=512 | 4 | 1 | 6834000.0 | 223624.5 | 11.0% | NOISY |
| decode_proj_f32_host-sgemm_in1024_out1024_len512 | #1992 #2003 #1995 #2229 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=1024, len=512 | 4 | 1 | 6915333.0 | 869375.0 | 14.1% | NOISY |
| decode_proj_f32_pytorch_in1024_out1024_len512 | #1992 #2003 #1995 #2229 | pytorch | pytorch-cpu | f32 | in=1024, out=1024, len=512 | 4 | 2 | 1132395.5 | 96875.0 | 7.6% | ok |
| decode_proj_f32_eager-shared_in1024_out4096_len8 | #1992 #2003 #1995 #2229 | eager-shared | tenferro-rs | f32 | in=1024, out=4096, len=8 | 4 | 2 | 1496375.0 | 695573.0 | 31.9% | NOISY |
| decode_proj_f32_faer-direct_in1024_out4096_len8 | #1992 #2003 #1995 #2229 | faer-direct | faer-direct | f32 | in=1024, out=4096, len=8 | 4 | 2 | 1007604.5 | 90791.5 | 10.2% | NOISY |
| decode_proj_f32_host-sgemm_in1024_out4096_len8 | #1992 #2003 #1995 #2229 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=4096, len=8 | 4 | 1 | 3219125.0 | 374166.5 | 9.1% | ok |
| decode_proj_f32_pytorch_in1024_out4096_len8 | #1992 #2003 #1995 #2229 | pytorch | pytorch-cpu | f32 | in=1024, out=4096, len=8 | 4 | 2 | 1069791.5 | 116948.0 | 29.1% | NOISY |
| decode_proj_f32_eager-shared_in1024_out4096_len64 | #1992 #2003 #1995 #2229 | eager-shared | tenferro-rs | f32 | in=1024, out=4096, len=64 | 4 | 1 | 4353500.0 | 256083.0 | 4.7% | ok |
| decode_proj_f32_faer-direct_in1024_out4096_len64 | #1992 #2003 #1995 #2229 | faer-direct | faer-direct | f32 | in=1024, out=4096, len=64 | 4 | 1 | 4606417.0 | 625104.0 | 8.7% | ok |
| decode_proj_f32_host-sgemm_in1024_out4096_len64 | #1992 #2003 #1995 #2229 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=4096, len=64 | 4 | 1 | 8034417.0 | 1411729.0 | 13.8% | NOISY |
| decode_proj_f32_pytorch_in1024_out4096_len64 | #1992 #2003 #1995 #2229 | pytorch | pytorch-cpu | f32 | in=1024, out=4096, len=64 | 4 | 2 | 1279917.0 | 152864.8 | 8.0% | ok |
| decode_proj_f32_eager-shared_in1024_out4096_len512 | #1992 #2003 #1995 #2229 | eager-shared | tenferro-rs | f32 | in=1024, out=4096, len=512 | 4 | 1 | 31808459.0 | 6980937.0 | 19.1% | NOISY |
| decode_proj_f32_faer-direct_in1024_out4096_len512 | #1992 #2003 #1995 #2229 | faer-direct | faer-direct | f32 | in=1024, out=4096, len=512 | 4 | 1 | 29797166.0 | 6298395.0 | 15.0% | NOISY |
| decode_proj_f32_host-sgemm_in1024_out4096_len512 | #1992 #2003 #1995 #2229 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=4096, len=512 | 4 | 1 | 25949750.0 | 6014375.5 | 20.4% | NOISY |
| decode_proj_f32_pytorch_in1024_out4096_len512 | #1992 #2003 #1995 #2229 | pytorch | pytorch-cpu | f32 | in=1024, out=4096, len=512 | 4 | 1 | 5840000.0 | 773854.0 | 15.8% | NOISY |
| decode_proj_view_operand_f32_prepared_in1024_out1024_len8_t4 | #2216 | prepared-view | tenferro-rs | f32 | in=1024, out=1024, len=8 | 4 | 8 | 374854.2 | 105450.5 | 17.6% | NOISY |
| decode_proj_contiguous_operand_f32_prepared_in1024_out1024_len8_t4 | #2216 | prepared-contiguous | tenferro-rs | f32 | in=1024, out=1024, len=8 | 4 | 16 | 287838.5 | 93592.5 | 17.0% | NOISY |
| decode_proj_view_operand_f32_pytorch_in1024_out1024_len8_t4 | #2216 | pytorch | pytorch-cpu | f32 | in=1024, out=1024, len=8 | 4 | 16 | 130559.9 | 3937.5 | 19.5% | NOISY |
| gemm_c64_mm256_t1 | #1900 | dot-general-shared | tenferro-rs | c64 | n=256 | 4 | 1 | 4372333.0 | 175958.5 | 10.7% | NOISY |
| gemm_f64_mm256_t1 | #1900 | dot-general-shared | tenferro-rs | f64 | n=256 | 4 | 4 | 812812.5 | 23499.9 | 3.3% | ok |
| conj_dot_f64_dot-general-rank1_len100 | #1615 | dot-general-rank1 | tenferro-rs | f64 | len=100 | 4 | 4096 | 883.5 | 9.7 | 2.2% | ok |
| conj_dot_f64_portable-loop_len100 | #1615 | portable-loop | rust-loop | f64 | len=100 | 4 | 65536 | 55.9 | 1.2 | 1.4% | ok |
| conj_dot_f64_pytorch-vdot_len100 | #1615 | pytorch-vdot | pytorch-cpu | f64 | len=100 | 4 | 4096 | 658.6 | 124.6 | 52.4% | NOISY |
| conj_dot_f64_dot-general-rank1_len10000 | #1615 | dot-general-rank1 | tenferro-rs | f64 | len=10000 | 4 | 1024 | 3232.2 | 65.2 | 6.2% | ok |
| conj_dot_f64_portable-loop_len10000 | #1615 | portable-loop | rust-loop | f64 | len=10000 | 4 | 256 | 10185.7 | 120.2 | 1.2% | ok |
| conj_dot_f64_pytorch-vdot_len10000 | #1615 | pytorch-vdot | pytorch-cpu | f64 | len=10000 | 4 | 1024 | 3051.3 | 174.1 | 7.7% | ok |
| conj_dot_f64_dot-general-rank1_len1000000 | #1615 | dot-general-rank1 | tenferro-rs | f64 | len=1000000 | 4 | 8 | 269171.9 | 24455.8 | 5.4% | ok |
| conj_dot_f64_portable-loop_len1000000 | #1615 | portable-loop | rust-loop | f64 | len=1000000 | 4 | 2 | 1029167.0 | 26854.2 | 2.1% | ok |
| conj_dot_f64_pytorch-vdot_len1000000 | #1615 | pytorch-vdot | pytorch-cpu | f64 | len=1000000 | 4 | 4 | 349771.0 | 87776.1 | 22.4% | NOISY |
| conj_dot_c64_dot-general-rank1_len100 | #1615 | dot-general-rank1 | tenferro-rs | c64 | len=100 | 4 | 4096 | 928.2 | 43.0 | 7.7% | ok |
| conj_dot_c64_portable-loop_len100 | #1615 | portable-loop | rust-loop | c64 | len=100 | 4 | 32768 | 89.0 | 2.9 | 2.5% | ok |
| conj_dot_c64_pytorch-vdot_len100 | #1615 | pytorch-vdot | pytorch-cpu | c64 | len=100 | 4 | 4096 | 578.5 | 53.9 | 25.5% | NOISY |
| conj_dot_c64_dot-general-rank1_len10000 | #1615 | dot-general-rank1 | tenferro-rs | c64 | len=10000 | 4 | 512 | 6665.9 | 64.9 | 1.9% | ok |
| conj_dot_c64_portable-loop_len10000 | #1615 | portable-loop | rust-loop | c64 | len=10000 | 4 | 256 | 10154.5 | 214.2 | 2.3% | ok |
| conj_dot_c64_pytorch-vdot_len10000 | #1615 | pytorch-vdot | pytorch-cpu | c64 | len=10000 | 4 | 512 | 5529.0 | 1238.0 | 44.5% | NOISY |
| conj_dot_c64_dot-general-rank1_len1000000 | #1615 | dot-general-rank1 | tenferro-rs | c64 | len=1000000 | 4 | 4 | 718437.5 | 36343.6 | 5.9% | ok |
| conj_dot_c64_portable-loop_len1000000 | #1615 | portable-loop | rust-loop | c64 | len=1000000 | 4 | 2 | 1065521.0 | 31895.8 | 5.6% | ok |
| conj_dot_c64_pytorch-vdot_len1000000 | #1615 | pytorch-vdot | pytorch-cpu | c64 | len=1000000 | 4 | 4 | 809583.5 | 145619.9 | 14.7% | NOISY |
| small_solve_f64_solve-loop_k16_b64 | #2007 #1878 | solve-loop | tenferro-rs | f64 | k=16, batch=64 | 4 | 16 | 164140.6 | 8410.2 | 4.0% | ok |
| small_solve_f64_triangular-solve-loop_k16_b64 | #2007 #1878 | triangular-solve-loop | tenferro-rs | f64 | k=16, batch=64 | 4 | 32 | 82502.6 | 4304.7 | 13.9% | NOISY |
| small_solve_f64_solve-batched_k16_b64 | #2007 #1878 | solve-batched | tenferro-rs | f64 | k=16, batch=64 | 4 | 32 | 90789.1 | 25824.9 | 18.0% | NOISY |
| small_solve_f64_pytorch-solve-batched_k16_b64 | #2007 #1878 | pytorch-solve-batched | pytorch-cpu | f64 | k=16, batch=64 | 4 | 16 | 120882.8 | 30937.5 | 17.9% | NOISY |
| small_solve_f64_pytorch-solve-triangular-batched_k16_b64 | #2007 #1878 | pytorch-solve-triangular-batched | pytorch-cpu | f64 | k=16, batch=64 | 4 | 64 | 21744.8 | 368.2 | 2.4% | ok |
| small_solve_f64_solve-loop_k64_b64 | #2007 #1878 | solve-loop | tenferro-rs | f64 | k=64, batch=64 | 4 | 1 | 3018584.0 | 433791.5 | 8.5% | ok |
| small_solve_f64_triangular-solve-loop_k64_b64 | #2007 #1878 | triangular-solve-loop | tenferro-rs | f64 | k=64, batch=64 | 4 | 2 | 1106583.0 | 133291.5 | 12.5% | NOISY |
| small_solve_f64_solve-batched_k64_b64 | #2007 #1878 | solve-batched | tenferro-rs | f64 | k=64, batch=64 | 4 | 2 | 1446354.0 | 49416.5 | 2.1% | ok |
| small_solve_f64_pytorch-solve-batched_k64_b64 | #2007 #1878 | pytorch-solve-batched | pytorch-cpu | f64 | k=64, batch=64 | 4 | 2 | 1179917.0 | 44948.0 | 5.2% | ok |
| small_solve_f64_pytorch-solve-triangular-batched_k64_b64 | #2007 #1878 | pytorch-solve-triangular-batched | pytorch-cpu | f64 | k=64, batch=64 | 4 | 8 | 363224.0 | 7684.9 | 1.5% | ok |
| tanh_chain_f32_compiled-prepared_k10_1024x64 | #1990 | compiled-prepared | tenferro-rs | f32 | rows=1024, cols=64, chain=10 | 4 | 2 | 1182854.0 | 35114.5 | 12.4% | NOISY |
| tanh_chain_f32_eager-shared_k10_1024x64 | #1990 | eager-shared | tenferro-rs | f32 | rows=1024, cols=64, chain=10 | 4 | 1 | 2262667.0 | 8291.5 | 1.4% | ok |
| tanh_chain_f32_compiled-prepared_k100_1024x64 | #1990 | compiled-prepared | tenferro-rs | f32 | rows=1024, cols=64, chain=100 | 4 | 1 | 10548875.0 | 1144979.0 | 7.8% | ok |
| tanh_chain_f32_eager-shared_k100_1024x64 | #1990 | eager-shared | tenferro-rs | f32 | rows=1024, cols=64, chain=100 | 4 | 1 | 23210000.0 | 525916.5 | 1.4% | ok |
| tanh_chain_f32_compiled-prepared_k200_1024x64 | #1990 | compiled-prepared | tenferro-rs | f32 | rows=1024, cols=64, chain=200 | 4 | 1 | 20007000.0 | 2209791.5 | 9.0% | ok |
| tanh_chain_f32_eager-shared_k200_1024x64 | #1990 | eager-shared | tenferro-rs | f32 | rows=1024, cols=64, chain=200 | 4 | 1 | 46812959.0 | 1146499.5 | 3.7% | ok |
| layer_norm_f32_eager-composed_d1024_len8_b1 | #2006 | eager-composed | tenferro-rs | f32 | norm=layer_norm, d=1024, len=8, batch=1 | 4 | 64 | 41682.3 | 1586.3 | 2.7% | ok |
| layer_norm_f32_pytorch-fused_d1024_len8_b1 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=layer_norm, d=1024, len=8, batch=1 | 4 | 64 | 33540.4 | 3600.3 | 8.4% | ok |
| layer_norm_f32_eager-single-call_d1024_len8_b1 | #2006 | eager-single-call | tenferro-rs | f32 | norm=layer_norm, d=1024, len=8, batch=1 | 4 | 64 | 47382.8 | 2533.2 | 28.4% | NOISY |
| layer_norm_f32_eager-composed_d1024_len8_b8 | #2006 | eager-composed | tenferro-rs | f32 | norm=layer_norm, d=1024, len=8, batch=8 | 4 | 32 | 83830.7 | 5141.3 | 4.7% | ok |
| layer_norm_f32_pytorch-fused_d1024_len8_b8 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=layer_norm, d=1024, len=8, batch=8 | 4 | 32 | 89407.6 | 4851.6 | 4.0% | ok |
| layer_norm_f32_eager-single-call_d1024_len8_b8 | #2006 | eager-single-call | tenferro-rs | f32 | norm=layer_norm, d=1024, len=8, batch=8 | 4 | 32 | 91312.5 | 3663.4 | 22.2% | NOISY |
| layer_norm_f32_eager-composed_d1024_len64_b1 | #2006 | eager-composed | tenferro-rs | f32 | norm=layer_norm, d=1024, len=64, batch=1 | 4 | 16 | 86549.4 | 18114.6 | 44.9% | NOISY |
| layer_norm_f32_pytorch-fused_d1024_len64_b1 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=layer_norm, d=1024, len=64, batch=1 | 4 | 32 | 81428.4 | 21829.4 | 17.3% | NOISY |
| layer_norm_f32_eager-single-call_d1024_len64_b1 | #2006 | eager-single-call | tenferro-rs | f32 | norm=layer_norm, d=1024, len=64, batch=1 | 4 | 16 | 92242.2 | 5015.6 | 4.9% | ok |
| layer_norm_f32_eager-composed_d1024_len64_b8 | #2006 | eager-composed | tenferro-rs | f32 | norm=layer_norm, d=1024, len=64, batch=8 | 4 | 4 | 1037875.0 | 352671.8 | 25.0% | NOISY |
| layer_norm_f32_pytorch-fused_d1024_len64_b8 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=layer_norm, d=1024, len=64, batch=8 | 4 | 8 | 271234.4 | 33760.5 | 9.0% | ok |
| layer_norm_f32_eager-single-call_d1024_len64_b8 | #2006 | eager-single-call | tenferro-rs | f32 | norm=layer_norm, d=1024, len=64, batch=8 | 4 | 2 | 1018812.5 | 166333.2 | 16.6% | NOISY |
| rms_norm_f32_eager-composed_d1024_len8_b1 | #2006 | eager-composed | tenferro-rs | f32 | norm=rms_norm, d=1024, len=8, batch=1 | 4 | 128 | 22404.6 | 809.1 | 5.2% | ok |
| rms_norm_f32_pytorch-fused_d1024_len8_b1 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=rms_norm, d=1024, len=8, batch=1 | 4 | 128 | 9766.9 | 271.6 | 2.2% | ok |
| rms_norm_f32_eager-single-call_d1024_len8_b1 | #2006 | eager-single-call | tenferro-rs | f32 | norm=rms_norm, d=1024, len=8, batch=1 | 4 | 32 | 26260.4 | 871.8 | 3.5% | ok |
| rms_norm_f32_eager-composed_d1024_len8_b8 | #2006 | eager-composed | tenferro-rs | f32 | norm=rms_norm, d=1024, len=8, batch=8 | 4 | 16 | 40671.9 | 2808.6 | 20.2% | NOISY |
| rms_norm_f32_pytorch-fused_d1024_len8_b8 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=rms_norm, d=1024, len=8, batch=8 | 4 | 16 | 136786.4 | 5755.2 | 3.7% | ok |
| rms_norm_f32_eager-single-call_d1024_len8_b8 | #2006 | eager-single-call | tenferro-rs | f32 | norm=rms_norm, d=1024, len=8, batch=8 | 4 | 16 | 47434.9 | 3640.6 | 23.7% | NOISY |
| rms_norm_f32_eager-composed_d1024_len64_b1 | #2006 | eager-composed | tenferro-rs | f32 | norm=rms_norm, d=1024, len=64, batch=1 | 4 | 32 | 39781.2 | 876.3 | 1.9% | ok |
| rms_norm_f32_pytorch-fused_d1024_len64_b1 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=rms_norm, d=1024, len=64, batch=1 | 4 | 16 | 142239.6 | 11231.8 | 11.6% | NOISY |
| rms_norm_f32_eager-single-call_d1024_len64_b1 | #2006 | eager-single-call | tenferro-rs | f32 | norm=rms_norm, d=1024, len=64, batch=1 | 4 | 32 | 47177.1 | 744.8 | 1.9% | ok |
| rms_norm_f32_eager-composed_d1024_len64_b8 | #2006 | eager-composed | tenferro-rs | f32 | norm=rms_norm, d=1024, len=64, batch=8 | 4 | 4 | 313385.5 | 55224.0 | 20.1% | NOISY |
| rms_norm_f32_pytorch-fused_d1024_len64_b8 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=rms_norm, d=1024, len=64, batch=8 | 4 | 8 | 326078.1 | 28380.2 | 11.4% | NOISY |
| rms_norm_f32_eager-single-call_d1024_len64_b8 | #2006 | eager-single-call | tenferro-rs | f32 | norm=rms_norm, d=1024, len=64, batch=8 | 4 | 4 | 304323.0 | 200948.1 | 33.5% | NOISY |
| small_contraction_abcd-dbef-acef_f64_shared-session-default-pool_d4 | #1885 #1897 #1901 | shared-session-default-pool | tenferro-rs | f64 | extent=4 | 4 | 1024 | 2260.2 | 81.0 | 5.8% | ok |
| activation_erf_f32_eager-single-call_1024x64 | #1975 #2069 #2032 | eager-single-call | tenferro-rs | f32 | op=erf, rows=1024, cols=64 | 4 | — | — | — | — | SKIPPED |
| activation_erf_f32_session-single-call_1024x64 | #1975 #2069 #2032 | session-single-call | tenferro-rs | f32 | op=erf, rows=1024, cols=64 | 4 | 4 | 471385.5 | 157396.0 | 29.4% | NOISY |
| activation_erf_f32_pytorch_1024x64 | #1975 #2069 #2032 | pytorch | pytorch-cpu | f32 | op=erf, rows=1024, cols=64 | 4 | — | — | — | — | SKIPPED |
| activation_sigmoid_f32_eager-single-call_1024x64 | #1975 #2069 #2032 | eager-single-call | tenferro-rs | f32 | op=sigmoid, rows=1024, cols=64 | 4 | — | — | — | — | SKIPPED |
| activation_sigmoid_f32_session-single-call_1024x64 | #1975 #2069 #2032 | session-single-call | tenferro-rs | f32 | op=sigmoid, rows=1024, cols=64 | 4 | 32 | 30570.3 | 17003.3 | 38.4% | NOISY |
| activation_sigmoid_f32_eager-composed_1024x64 | #1975 #2069 #2032 | eager-composed | tenferro-rs | f32 | op=sigmoid, rows=1024, cols=64 | 4 | 32 | 74592.4 | 1587.2 | 1.6% | ok |
| activation_sigmoid_f32_pytorch_1024x64 | #1975 #2069 #2032 | pytorch | pytorch-cpu | f32 | op=sigmoid, rows=1024, cols=64 | 4 | — | — | — | — | SKIPPED |
| activation_silu_f32_eager-single-call_1024x64 | #1975 #2069 #2032 | eager-single-call | tenferro-rs | f32 | op=silu, rows=1024, cols=64 | 4 | — | — | — | — | SKIPPED |
| activation_silu_f32_session-single-call_1024x64 | #1975 #2069 #2032 | session-single-call | tenferro-rs | f32 | op=silu, rows=1024, cols=64 | 4 | 32 | 31100.3 | 12346.3 | 41.7% | NOISY |
| activation_silu_f32_eager-composed_1024x64 | #1975 #2069 #2032 | eager-composed | tenferro-rs | f32 | op=silu, rows=1024, cols=64 | 4 | 8 | 79291.6 | 26971.3 | 29.6% | NOISY |
| activation_silu_f32_pytorch_1024x64 | #1975 #2069 #2032 | pytorch | pytorch-cpu | f32 | op=silu, rows=1024, cols=64 | 4 | — | — | — | — | SKIPPED |
| activation_softplus_f32_eager-single-call_1024x64 | #1975 #2069 #2032 | eager-single-call | tenferro-rs | f32 | op=softplus, rows=1024, cols=64 | 4 | — | — | — | — | SKIPPED |
| activation_softplus_f32_session-single-call_1024x64 | #1975 #2069 #2032 | session-single-call | tenferro-rs | f32 | op=softplus, rows=1024, cols=64 | 4 | 16 | 171901.0 | 9979.2 | 6.3% | ok |
| activation_softplus_f32_eager-composed_1024x64 | #1975 #2069 #2032 | eager-composed | tenferro-rs | f32 | op=softplus, rows=1024, cols=64 | 4 | 16 | 191763.0 | 588.5 | 0.7% | ok |
| activation_softplus_f32_pytorch_1024x64 | #1975 #2069 #2032 | pytorch | pytorch-cpu | f32 | op=softplus, rows=1024, cols=64 | 4 | — | — | — | — | SKIPPED |
| activation_gelu_f32_eager-single-call_1024x64 | #1975 #2030 #2069 #2228 | eager-single-call | tenferro-rs | f32 | op=gelu, rows=1024, cols=64 | 4 | 32 | 65932.3 | 17619.1 | 22.7% | NOISY |
| activation_gelu_f32_session-single-call_1024x64 | #1975 #2030 #2069 #2228 | session-single-call | tenferro-rs | f32 | op=gelu, rows=1024, cols=64 | 4 | 32 | 73955.7 | 31850.9 | 23.8% | NOISY |
| activation_gelu_f32_pytorch_1024x64 | #1975 #2030 #2069 #2228 | pytorch | pytorch-cpu | f32 | op=gelu, rows=1024, cols=64 | 4 | 32 | 87002.6 | 7404.3 | 12.0% | NOISY |
| activation_gelu_tanh_f32_eager-single-call_1024x64 | #1975 #2069 #2032 | eager-single-call | tenferro-rs | f32 | op=gelu_tanh, rows=1024, cols=64 | 4 | — | — | — | — | SKIPPED |
| activation_gelu_tanh_f32_session-single-call_1024x64 | #1975 #2069 #2032 | session-single-call | tenferro-rs | f32 | op=gelu_tanh, rows=1024, cols=64 | 4 | 16 | 182414.1 | 3096.4 | 1.0% | ok |
| activation_gelu_tanh_f32_eager-composed_1024x64 | #1975 #2069 #2032 | eager-composed | tenferro-rs | f32 | op=gelu_tanh, rows=1024, cols=64 | 4 | 8 | 325322.9 | 3437.6 | 1.9% | ok |
| activation_gelu_tanh_f32_pytorch_1024x64 | #1975 #2069 #2032 | pytorch | pytorch-cpu | f32 | op=gelu_tanh, rows=1024, cols=64 | 4 | — | — | — | — | SKIPPED |
| softmax_f32_eager-single-call_len64_b8 | #1976 #2031 #2231 | eager-single-call | tenferro-rs | f32 | op=softmax, len=64, batch=8, axis=0 | 4 | 32 | 41812.5 | 6738.3 | 48.9% | NOISY |
| softmax_f32_session-single-call_len64_b8 | #1976 #2031 #2231 | session-single-call | tenferro-rs | f32 | op=softmax, len=64, batch=8, axis=0 | 4 | 64 | 24933.6 | 7474.6 | 25.3% | NOISY |
| softmax_f32_eager-composed_len64_b8 | #1976 #2031 #2231 | eager-composed | tenferro-rs | f32 | op=softmax, len=64, batch=8, axis=0 | 4 | 32 | 70182.3 | 1143.9 | 1.1% | ok |
| softmax_f32_pytorch_len64_b8 | #1976 #2031 #2231 | pytorch | pytorch-cpu | f32 | op=softmax, len=64, batch=8, axis=0 | 4 | 32 | 64048.2 | 10918.6 | 13.0% | NOISY |
| softmax_f32_eager-single-call_len512_b8 | #1976 #2231 | eager-single-call | tenferro-rs | f32 | op=softmax, len=512, batch=8, axis=0 | 4 | 1 | 3515750.0 | 1714354.0 | 62.1% | NOISY |
| softmax_f32_session-single-call_len512_b8 | #1976 #2231 | session-single-call | tenferro-rs | f32 | op=softmax, len=512, batch=8, axis=0 | 4 | 2 | 2244625.0 | 639219.0 | 23.7% | NOISY |
| softmax_f32_eager-composed_len512_b8 | #1976 #2231 | eager-composed | tenferro-rs | f32 | op=softmax, len=512, batch=8, axis=0 | 4 | 1 | 4089333.0 | 1301645.5 | 19.8% | NOISY |
| softmax_f32_pytorch_len512_b8 | #1976 #2231 | pytorch | pytorch-cpu | f32 | op=softmax, len=512, batch=8, axis=0 | 4 | 2 | 2144250.0 | 783656.0 | 22.8% | NOISY |
| log_softmax_f32_eager-single-call_len64_b8 | #1976 | eager-single-call | tenferro-rs | f32 | op=log_softmax, len=64, batch=8, axis=0 | 4 | 32 | 96378.9 | 3304.7 | 15.4% | NOISY |
| log_softmax_f32_session-single-call_len64_b8 | #1976 | session-single-call | tenferro-rs | f32 | op=log_softmax, len=64, batch=8, axis=0 | 4 | 32 | 80162.8 | 28953.1 | 32.3% | NOISY |
| log_softmax_f32_eager-composed_len64_b8 | #1976 | eager-composed | tenferro-rs | f32 | op=log_softmax, len=64, batch=8, axis=0 | 4 | 32 | 75731.8 | 23185.5 | 37.1% | NOISY |
| log_softmax_f32_pytorch_len64_b8 | #1976 | pytorch | pytorch-cpu | f32 | op=log_softmax, len=64, batch=8, axis=0 | 4 | 32 | 91849.0 | 8106.1 | 11.1% | NOISY |
| log_softmax_f32_eager-single-call_len512_b8 | #1976 | eager-single-call | tenferro-rs | f32 | op=log_softmax, len=512, batch=8, axis=0 | 4 | 1 | 3114542.0 | 220875.5 | 6.7% | ok |
| log_softmax_f32_session-single-call_len512_b8 | #1976 | session-single-call | tenferro-rs | f32 | op=log_softmax, len=512, batch=8, axis=0 | 4 | 1 | 3766084.0 | 1249312.5 | 25.9% | NOISY |
| log_softmax_f32_eager-composed_len512_b8 | #1976 | eager-composed | tenferro-rs | f32 | op=log_softmax, len=512, batch=8, axis=0 | 4 | 1 | 3506000.0 | 297229.5 | 8.1% | ok |
| log_softmax_f32_pytorch_len512_b8 | #1976 | pytorch | pytorch-cpu | f32 | op=log_softmax, len=512, batch=8, axis=0 | 4 | 1 | 2145458.0 | 24811.5 | 5.0% | ok |
| masked_softmax_f32_eager-single-call_len64_b8 | #1976 | eager-single-call | tenferro-rs | f32 | op=masked_softmax, len=64, batch=8, axis=0 | 4 | 16 | 113846.4 | 4442.7 | 46.0% | NOISY |
| masked_softmax_f32_session-single-call_len64_b8 | #1976 | session-single-call | tenferro-rs | f32 | op=masked_softmax, len=64, batch=8, axis=0 | 4 | 32 | 89045.6 | 1931.0 | 3.0% | ok |
| masked_softmax_f32_eager-composed_len64_b8 | #1976 | eager-composed | tenferro-rs | f32 | op=masked_softmax, len=64, batch=8, axis=0 | 4 | 32 | 77718.8 | 1659.5 | 5.4% | ok |
| masked_softmax_f32_pytorch_len64_b8 | #1976 | pytorch | pytorch-cpu | f32 | op=masked_softmax, len=64, batch=8, axis=0 | 4 | 32 | 115591.1 | 10805.3 | 6.3% | ok |
| masked_softmax_f32_eager-single-call_len512_b8 | #1976 | eager-single-call | tenferro-rs | f32 | op=masked_softmax, len=512, batch=8, axis=0 | 4 | 1 | 2777042.0 | 374374.5 | 9.7% | ok |
| masked_softmax_f32_session-single-call_len512_b8 | #1976 | session-single-call | tenferro-rs | f32 | op=masked_softmax, len=512, batch=8, axis=0 | 4 | 1 | 3748083.0 | 432625.5 | 18.9% | NOISY |
| masked_softmax_f32_eager-composed_len512_b8 | #1976 | eager-composed | tenferro-rs | f32 | op=masked_softmax, len=512, batch=8, axis=0 | 4 | 1 | 3862458.0 | 210958.0 | 5.8% | ok |
| masked_softmax_f32_pytorch_len512_b8 | #1976 | pytorch | pytorch-cpu | f32 | op=masked_softmax, len=512, batch=8, axis=0 | 4 | 1 | 2594458.0 | 79375.0 | 2.5% | ok |
| softmax_f32_eager-single-call_len64_b8_t4 | #2233 #2235 | eager-single-call | tenferro-rs | f32 | op=softmax, len=64, batch=8, axis=0 | 4 | 32 | 58492.2 | 15449.2 | 27.5% | NOISY |
| softmax_f32_eager-composed_len64_b8_t4 | #2233 #2235 | eager-composed | tenferro-rs | f32 | op=softmax, len=64, batch=8, axis=0 | 4 | 32 | 68220.0 | 1405.6 | 1.4% | ok |
| softmax_f32_pytorch_len64_b8_t4 | #2233 #2235 | pytorch | pytorch-cpu | f32 | op=softmax, len=64, batch=8, axis=0 | 4 | 32 | 77437.5 | 4974.0 | 6.5% | ok |
| reduce_mean_f32_eager-single-call_1024x64_axis0 | #1976 | eager-single-call | tenferro-rs | f32 | rows=1024, cols=64, axis=0 | 4 | 256 | 12093.9 | 95.3 | 13.0% | NOISY |
| reduce_mean_f32_session-single-call_1024x64_axis0 | #1976 | session-single-call | tenferro-rs | f32 | rows=1024, cols=64, axis=0 | 4 | 512 | 7463.9 | 105.8 | 1.3% | ok |
| reduce_mean_f32_eager-composed_1024x64_axis0 | #1976 | eager-composed | tenferro-rs | f32 | rows=1024, cols=64, axis=0 | 4 | 256 | 11881.8 | 186.5 | 1.7% | ok |
| reduce_mean_f32_pytorch_1024x64_axis0 | #1976 | pytorch | pytorch-cpu | f32 | rows=1024, cols=64, axis=0 | 4 | 64 | 36334.6 | 1808.9 | 4.2% | ok |
| take_along_axis_rows_f64_eager-single-call_n64_b256 | #2008 | eager-single-call | tenferro-rs | f64 | n=64, batch=256, axis=0 | 4 | 1 | 3915708.0 | 73958.5 | 6.8% | ok |
| take_along_axis_rows_f64_session-single-call_n64_b256 | #2008 | session-single-call | tenferro-rs | f64 | n=64, batch=256, axis=0 | 4 | 1 | 3791625.0 | 44812.5 | 1.4% | ok |
| take_along_axis_rows_f64_eager-gather-prebuilt_n64_b256 | #2008 | eager-gather-prebuilt | tenferro-rs | f64 | n=64, batch=256, axis=0 | 4 | 1 | 3731958.0 | 120645.5 | 7.3% | ok |
| take_along_axis_rows_f64_pytorch_n64_b256 | #2008 | pytorch | pytorch-cpu | f64 | n=64, batch=256, axis=0 | 4 | 2 | 1698792.0 | 112739.8 | 10.9% | NOISY |
| lstsq_f64_eager-shared_m768_n384_rhs16 | #2027 | eager-shared | tenferro-rs | f64 | operation=lstsq | 4 | 1 | 12181542.0 | 1816083.0 | 13.7% | NOISY |
| lstsq_f64_pytorch_m768_n384_rhs16 | #2027 | pytorch | pytorch-cpu | f64 | operation=lstsq | 4 | 1 | 4764250.0 | 304083.0 | 5.8% | ok |
| triangular_f64_concrete-shared_n4096_rhs64 | #2028 | concrete-shared | tenferro-rs | f64 | operation=triangular | 4 | 1 | 24444375.0 | 589063.5 | 2.5% | ok |
| triangular_f64_pytorch_n4096_rhs64 | #2028 | pytorch | pytorch-cpu | f64 | operation=triangular | 4 | 1 | 11759167.0 | 243208.5 | 3.5% | ok |
| reshape_f64_concrete-shared_n33554432_to8192x4096 | #2029 | concrete-shared | tenferro-rs | f64 | operation=reshape | 4 | 1 | 10863000.0 | 637688.0 | 7.6% | ok |
| reshape_f64_pytorch_n33554432_to8192x4096 | #2029 | pytorch | pytorch-cpu | f64 | operation=reshape | 4 | 1 | 10862958.0 | 647291.5 | 5.8% | ok |
| metadata_transpose_view_f64_rank8_t4 | #2180 | metadata-clone | tenferro-rs | f64 | shape=[2, 2, 2, 2, 2, 2, 2, 2] | 4 | 32768 | 53.0 | 0.5 | 2.1% | ok |
| metadata_transpose_view_f64_rank8_t4_julia-base | #2180 | julia-base | julia-base | f64 | shape=[2, 2, 2, 2, 2, 2, 2, 2] | 4 | 1048576 | 0.3 | 0.0 | 4.6% | ok |
| followup_structural_broadcast_in_dim_direct-shared | #2021 | direct-shared | tenferro-rs | f64 | case_id=structural.broadcast_in_dim, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [8192, 1], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | SKIPPED |
| followup_structural_broadcast_in_dim_pytorch | #2021 | pytorch | pytorch-cpu | f64 | case_id=structural.broadcast_in_dim, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [8192, 1], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | SKIPPED |
| followup_real_tanh_direct-shared | #2033 | direct-shared | tenferro-rs | f64 | case_id=real.tanh, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [8388608], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | SKIPPED |
| followup_real_tanh_jax | #2033 | jax | jax-cpu | f64 | case_id=real.tanh, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [8388608], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | SKIPPED |
| followup_real_log1p_direct-shared | #2033 | direct-shared | tenferro-rs | f64 | case_id=real.log1p, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [4194304], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | SKIPPED |
| followup_real_log1p_pytorch | #2033 | pytorch | pytorch-cpu | f64 | case_id=real.log1p, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [4194304], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | SKIPPED |
| followup_real_cos_direct-shared | #2033 | direct-shared | tenferro-rs | f64 | case_id=real.cos, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [8388608], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | SKIPPED |
| followup_real_cos_pytorch | #2033 | pytorch | pytorch-cpu | f64 | case_id=real.cos, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [8388608], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | SKIPPED |
| followup_real_sin_direct-shared | #2033 | direct-shared | tenferro-rs | f64 | case_id=real.sin, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [8388608], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | SKIPPED |
| followup_real_sin_pytorch | #2033 | pytorch | pytorch-cpu | f64 | case_id=real.sin, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [8388608], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | SKIPPED |
| followup_real_exp_direct-shared | #2033 | direct-shared | tenferro-rs | f64 | case_id=real.exp, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [8388608], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | SKIPPED |
| followup_real_exp_jax | #2033 | jax | jax-cpu | f64 | case_id=real.exp, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [8388608], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | SKIPPED |
| followup_real_expm1_direct-shared | #2033 | direct-shared | tenferro-rs | f64 | case_id=real.expm1, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [4194304], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | SKIPPED |
| followup_real_expm1_jax | #2033 | jax | jax-cpu | f64 | case_id=real.expm1, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [4194304], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | SKIPPED |
| followup_real_pow_direct-shared | #2033 | direct-shared | tenferro-rs | f64 | case_id=real.pow, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [4194304], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | SKIPPED |
| followup_real_pow_pytorch | #2033 | pytorch | pytorch-cpu | f64 | case_id=real.pow, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [4194304], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | SKIPPED |
| followup_complex_exp_direct-shared | #2033 | direct-shared | tenferro-rs | c64 | case_id=complex.exp, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [4194304], 'dtype': 'c64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | SKIPPED |
| followup_complex_exp_julia-base | #2033 | julia-base | julia-base | c64 | case_id=complex.exp, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [4194304], 'dtype': 'c64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | SKIPPED |
| followup_real_log_direct-shared | #2033 | direct-shared | tenferro-rs | f64 | case_id=real.log, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [8388608], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | SKIPPED |
| followup_real_log_pytorch | #2033 | pytorch | pytorch-cpu | f64 | case_id=real.log, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [8388608], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | SKIPPED |
| followup_real_reduce_max_axis1_direct-shared | #2034 | direct-shared | tenferro-rs | f64 | case_id=real.reduce_max_axis1, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [2048, 2048], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | SKIPPED |
| followup_real_reduce_max_axis1_julia-base | #2034 | julia-base | julia-base | f64 | case_id=real.reduce_max_axis1, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [2048, 2048], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | SKIPPED |
| followup_real_reduce_prod_axis1_direct-shared | #2034 | direct-shared | tenferro-rs | f64 | case_id=real.reduce_prod_axis1, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [2048, 2048], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | SKIPPED |
| followup_real_reduce_prod_axis1_pytorch | #2034 | pytorch | pytorch-cpu | f64 | case_id=real.reduce_prod_axis1, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [2048, 2048], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | SKIPPED |
| followup_real_reduce_sum_axis1_direct-shared | #2034 | direct-shared | tenferro-rs | f64 | case_id=real.reduce_sum_axis1, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [2048, 2048], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | SKIPPED |
| followup_real_reduce_sum_axis1_pytorch | #2034 | pytorch | pytorch-cpu | f64 | case_id=real.reduce_sum_axis1, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [2048, 2048], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | SKIPPED |
| followup_real_reduce_max_axis0_direct-shared | #2034 | direct-shared | tenferro-rs | f64 | case_id=real.reduce_max_axis0, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [2048, 2048], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | SKIPPED |
| followup_real_reduce_max_axis0_julia-base | #2034 | julia-base | julia-base | f64 | case_id=real.reduce_max_axis0, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [2048, 2048], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | SKIPPED |
| followup_real_reduce_min_axis1_direct-shared | #2034 | direct-shared | tenferro-rs | f64 | case_id=real.reduce_min_axis1, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [4096, 4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | SKIPPED |
| followup_real_reduce_min_axis1_julia-base | #2034 | julia-base | julia-base | f64 | case_id=real.reduce_min_axis1, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [4096, 4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | SKIPPED |
| followup_real_reduce_min_axis0_direct-shared | #2034 | direct-shared | tenferro-rs | f64 | case_id=real.reduce_min_axis0, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [2048, 2048], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | SKIPPED |
| followup_real_reduce_min_axis0_julia-base | #2034 | julia-base | julia-base | f64 | case_id=real.reduce_min_axis0, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [2048, 2048], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | SKIPPED |
| followup_real_reduce_max_all_direct-shared | #2034 | direct-shared | tenferro-rs | f64 | case_id=real.reduce_max_all, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [8192, 4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | SKIPPED |
| followup_real_reduce_max_all_julia-base | #2034 | julia-base | julia-base | f64 | case_id=real.reduce_max_all, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [8192, 4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | SKIPPED |
| followup_real_reduce_min_all_direct-shared | #2034 | direct-shared | tenferro-rs | f64 | case_id=real.reduce_min_all, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [8192, 4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | SKIPPED |
| followup_real_reduce_min_all_julia-base | #2034 | julia-base | julia-base | f64 | case_id=real.reduce_min_all, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [8192, 4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | SKIPPED |
| followup_complex_norm_fro_direct-shared | #2035 | direct-shared | tenferro-rs | c64 | case_id=complex.norm_fro, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [2048, 1536], 'dtype': 'c64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | SKIPPED |
| followup_complex_norm_fro_jax | #2035 | jax | jax-cpu | c64 | case_id=complex.norm_fro, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [2048, 1536], 'dtype': 'c64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | SKIPPED |
| followup_index_gather_direct-shared | #2036 #1719 | direct-shared | tenferro-rs | f64 | case_id=index.gather, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [262144], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | SKIPPED |
| followup_index_gather_jax | #2036 #1719 | jax | jax-cpu | f64 | case_id=index.gather, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [262144], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | SKIPPED |
| followup_index_scatter_direct-shared | #2036 #1719 | direct-shared | tenferro-rs | f64 | case_id=index.scatter, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [262144], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | SKIPPED |
| followup_index_scatter_pytorch | #2036 #1719 | pytorch | pytorch-cpu | f64 | case_id=index.scatter, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [262144], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | SKIPPED |
| followup_index_dynamic_update_slice_direct-shared | #2036 #1719 | direct-shared | tenferro-rs | f64 | case_id=index.dynamic_update_slice, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [2097152], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | SKIPPED |
| followup_index_dynamic_update_slice_jax | #2036 #1719 | jax | jax-cpu | f64 | case_id=index.dynamic_update_slice, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [2097152], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | SKIPPED |
| followup_perm_reverse_15d_3_materialize | #2037 #2228 | materialize | tenferro-rs | f64 | case_id=perm.reverse_15d_3, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | SKIPPED |
| followup_perm_reverse_15d_3_strided-rs | #2037 #2228 | strided-rs | strided-rs | f64 | case_id=perm.reverse_15d_3, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | SKIPPED |
| followup_structural_transpose_direct-shared | #2037 #2228 | direct-shared | tenferro-rs | f64 | case_id=structural.transpose, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [4096, 4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | SKIPPED |
| followup_structural_transpose_pytorch | #2037 #2228 | pytorch | pytorch-cpu | f64 | case_id=structural.transpose, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [4096, 4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | SKIPPED |
| followup_perm_reverse_23d_2_materialize | #2037 #2228 | materialize | tenferro-rs | f64 | case_id=perm.reverse_23d_2, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | SKIPPED |
| followup_perm_reverse_23d_2_strided-rs | #2037 #2228 | strided-rs | strided-rs | f64 | case_id=perm.reverse_23d_2, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | SKIPPED |
| followup_perm_transpose_3d_256_201_materialize | #2037 #2228 | materialize | tenferro-rs | f64 | case_id=perm.transpose_3d_256_201, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [256, 256, 256], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | SKIPPED |
| followup_perm_transpose_3d_256_201_strided-rs | #2037 #2228 | strided-rs | strided-rs | f64 | case_id=perm.transpose_3d_256_201, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [256, 256, 256], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | SKIPPED |
| followup_perm_cyclic_15d_3_materialize | #2037 #2228 | materialize | tenferro-rs | f64 | case_id=perm.cyclic_15d_3, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | SKIPPED |
| followup_perm_cyclic_15d_3_strided-rs | #2037 #2228 | strided-rs | strided-rs | f64 | case_id=perm.cyclic_15d_3, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | SKIPPED |
| followup_perm_transpose_3d_256_102_materialize | #2037 #2228 | materialize | tenferro-rs | f64 | case_id=perm.transpose_3d_256_102, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [256, 256, 256], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | SKIPPED |
| followup_perm_transpose_3d_256_102_strided-rs | #2037 #2228 | strided-rs | strided-rs | f64 | case_id=perm.transpose_3d_256_102, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [256, 256, 256], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | SKIPPED |
| followup_perm_tn_light_415_24d_scattered_to_colmajor_materialize | #2037 #2228 | materialize | tenferro-rs | f64 | case_id=perm.tn_light_415_24d_scattered_to_colmajor, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | SKIPPED |
| followup_perm_tn_light_415_24d_scattered_to_colmajor_strided-rs | #2037 #2228 | strided-rs | strided-rs | f64 | case_id=perm.tn_light_415_24d_scattered_to_colmajor, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | SKIPPED |
| followup_perm_tn_light_415_24d_contiguous_same_perm_materialize | #2037 #2228 | materialize | tenferro-rs | f64 | case_id=perm.tn_light_415_24d_contiguous_same_perm, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | SKIPPED |
| followup_perm_tn_light_415_24d_contiguous_same_perm_strided-rs | #2037 #2228 | strided-rs | strided-rs | f64 | case_id=perm.tn_light_415_24d_contiguous_same_perm, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | SKIPPED |
| followup_perm_rotation_6d_32_32_32_32_16_16_materialize | #2037 #2228 | materialize | tenferro-rs | f64 | case_id=perm.rotation_6d_32_32_32_32_16_16, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [32, 32, 32, 32, 16, 16], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | SKIPPED |
| followup_perm_rotation_6d_32_32_32_32_16_16_strided-rs | #2037 #2228 | strided-rs | strided-rs | f64 | case_id=perm.rotation_6d_32_32_32_32_16_16, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [32, 32, 32, 32, 16, 16], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | SKIPPED |
| followup_structural_extract_diagonal_direct-shared | #2038 #2228 | direct-shared | tenferro-rs | f64 | case_id=structural.extract_diagonal, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [8388608, 2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | SKIPPED |
| followup_structural_extract_diagonal_jax | #2038 #2228 | jax | jax-cpu | f64 | case_id=structural.extract_diagonal, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [8388608, 2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | SKIPPED |
| followup_fft_irfft_direct-shared | #2039 #2230 #2228 | direct-shared | tenferro-rs | c32 | case_id=fft.irfft, rust_path=direct-shared, reference_path=mkl-dfti, fixture={'input_shape': [524289], 'dtype': 'c32', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | SKIPPED |
| followup_fft_irfft_mkl-dfti | #2039 #2230 #2228 | mkl-dfti | mkl-dfti | c32 | case_id=fft.irfft, rust_path=direct-shared, reference_path=mkl-dfti, fixture={'input_shape': [524289], 'dtype': 'c32', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | SKIPPED |
| followup_fft_rfft_direct-shared | #2039 #2230 #2228 | direct-shared | tenferro-rs | f32 | case_id=fft.rfft, rust_path=direct-shared, reference_path=mkl-dfti, fixture={'input_shape': [1048576], 'dtype': 'f32', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | SKIPPED |
| followup_fft_rfft_mkl-dfti | #2039 #2230 #2228 | mkl-dfti | mkl-dfti | f32 | case_id=fft.rfft, rust_path=direct-shared, reference_path=mkl-dfti, fixture={'input_shape': [1048576], 'dtype': 'f32', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | SKIPPED |
| followup_fft_fft_direct-shared | #2039 #2228 | direct-shared | tenferro-rs | c64 | case_id=fft.fft, rust_path=direct-shared, reference_path=mkl-dfti, fixture={'input_shape': [1048576], 'dtype': 'c64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | SKIPPED |
| followup_fft_fft_mkl-dfti | #2039 #2228 | mkl-dfti | mkl-dfti | c64 | case_id=fft.fft, rust_path=direct-shared, reference_path=mkl-dfti, fixture={'input_shape': [1048576], 'dtype': 'c64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | SKIPPED |
| followup_fft_ifft_direct-shared | #2039 #2228 | direct-shared | tenferro-rs | c32 | case_id=fft.ifft, rust_path=direct-shared, reference_path=mkl-dfti, fixture={'input_shape': [1048576], 'dtype': 'c32', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | SKIPPED |
| followup_fft_ifft_mkl-dfti | #2039 #2228 | mkl-dfti | mkl-dfti | c32 | case_id=fft.ifft, rust_path=direct-shared, reference_path=mkl-dfti, fixture={'input_shape': [1048576], 'dtype': 'c32', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | SKIPPED |
| followup_metadata_transpose_view_metadata | #2040 #1719 | metadata | tenferro-rs | f64 | case_id=metadata.transpose_view, rust_path=metadata, reference_path=julia-base, fixture={'input_shape': [2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 4 | — | — | — | — | SKIPPED |
| followup_metadata_transpose_view_julia-base | #2040 #1719 | julia-base | julia-base | f64 | case_id=metadata.transpose_view, rust_path=metadata, reference_path=julia-base, fixture={'input_shape': [2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 4 | — | — | — | — | SKIPPED |
| followup_metadata_transpose_view_metadata-host-dyn | #2040 #1719 | metadata-host-dyn | tenferro-rs | f64 | case_id=metadata.transpose_view, rust_path=metadata-host-dyn, reference_path=julia-base, fixture={'input_shape': [2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 4 | — | — | — | — | SKIPPED |
| followup_metadata_transpose_view_metadata-host-static | #2040 #1719 | metadata-host-static | tenferro-rs | f64 | case_id=metadata.transpose_view, rust_path=metadata-host-static, reference_path=julia-base, fixture={'input_shape': [2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 4 | — | — | — | — | SKIPPED |
| followup_metadata_slice_view_metadata | #2040 #1719 | metadata | tenferro-rs | f64 | case_id=metadata.slice_view, rust_path=metadata, reference_path=julia-base, fixture={'input_shape': [4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 4 | — | — | — | — | SKIPPED |
| followup_metadata_slice_view_julia-base | #2040 #1719 | julia-base | julia-base | f64 | case_id=metadata.slice_view, rust_path=metadata, reference_path=julia-base, fixture={'input_shape': [4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 4 | — | — | — | — | SKIPPED |
| followup_metadata_slice_view_metadata-host-dyn | #2040 #1719 | metadata-host-dyn | tenferro-rs | f64 | case_id=metadata.slice_view, rust_path=metadata-host-dyn, reference_path=julia-base, fixture={'input_shape': [4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 4 | — | — | — | — | SKIPPED |
| followup_metadata_slice_view_metadata-host-static | #2040 #1719 | metadata-host-static | tenferro-rs | f64 | case_id=metadata.slice_view, rust_path=metadata-host-static, reference_path=julia-base, fixture={'input_shape': [4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 4 | — | — | — | — | SKIPPED |
| followup_metadata_reshape_view_metadata | #2040 #1719 | metadata | tenferro-rs | f64 | case_id=metadata.reshape_view, rust_path=metadata, reference_path=julia-base, fixture={'input_shape': [1024], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 4 | — | — | — | — | SKIPPED |
| followup_metadata_reshape_view_julia-base | #2040 #1719 | julia-base | julia-base | f64 | case_id=metadata.reshape_view, rust_path=metadata, reference_path=julia-base, fixture={'input_shape': [1024], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 4 | — | — | — | — | SKIPPED |
| followup_metadata_reshape_view_metadata-host-dyn | #2040 #1719 | metadata-host-dyn | tenferro-rs | f64 | case_id=metadata.reshape_view, rust_path=metadata-host-dyn, reference_path=julia-base, fixture={'input_shape': [1024], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 4 | — | — | — | — | SKIPPED |
| followup_metadata_reshape_view_metadata-host-static | #2040 #1719 | metadata-host-static | tenferro-rs | f64 | case_id=metadata.reshape_view, rust_path=metadata-host-static, reference_path=julia-base, fixture={'input_shape': [1024], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 4 | — | — | — | — | SKIPPED |
| lu_batched_f64_factor_n8_b1024 | #2142 | tenferro | tenferro-rs | f64 | n=8, batch=1024, op=factor | 4 | 16 | 236533.8 | 6296.8 | 2.5% | ok |
| lu_batched_f64_factor_n8_b1024_host | #2142 | host | rust-loop | f64 | n=8, batch=1024, op=factor | 4 | 8 | 237229.2 | 5091.1 | 4.0% | ok |
| lu_batched_f64_solve_n8_b1024 | #2142 | tenferro | tenferro-rs | f64 | n=8, batch=1024, op=solve | 4 | 16 | 130015.6 | 2061.2 | 1.2% | ok |
| lu_batched_f64_solve_n8_b1024_pytorch | #2142 | pytorch | pytorch-cpu | f64 | n=8, batch=1024, op=solve | 4 | 32 | 97604.2 | 1175.8 | 1.1% | ok |
| lu_batched_f64_factor_n16_b1024 | #2142 | tenferro | tenferro-rs | f64 | n=16, batch=1024, op=factor | 4 | 4 | 463104.2 | 18208.2 | 13.0% | NOISY |
| lu_batched_f64_factor_n16_b1024_host | #2142 | host | rust-loop | f64 | n=16, batch=1024, op=factor | 4 | 2 | 1070521.0 | 55187.5 | 3.9% | ok |
| lu_batched_f64_solve_n16_b1024 | #2142 | tenferro | tenferro-rs | f64 | n=16, batch=1024, op=solve | 4 | 8 | 297026.0 | 3088.5 | 1.3% | ok |
| lu_batched_f64_solve_n16_b1024_pytorch | #2142 | pytorch | pytorch-cpu | f64 | n=16, batch=1024, op=solve | 4 | 16 | 211471.3 | 2561.2 | 1.3% | ok |
| small_grad_qr_f64_trace-prepared_n4 | #2186 | trace-prepared | tenferro-rs | f64 | op=qr, n=4 | 4 | 128 | 26803.1 | 694.7 | 1.8% | ok |
| small_grad_qr_f64_pytorch_n4 | #2186 | pytorch | pytorch-cpu | f64 | op=qr, n=4 | 4 | 32 | 137813.8 | 30126.9 | 18.9% | NOISY |
| small_grad_qr_f64_trace-prepared_n16 | #2186 | trace-prepared | tenferro-rs | f64 | op=qr, n=16 | 4 | 64 | 37139.3 | 725.6 | 1.8% | ok |
| small_grad_qr_f64_pytorch_n16 | #2186 | pytorch | pytorch-cpu | f64 | op=qr, n=16 | 4 | 32 | 100549.5 | 6484.4 | 3.9% | ok |
| small_grad_eigh_f64_trace-prepared_n4 | #2186 | trace-prepared | tenferro-rs | f64 | op=eigh, n=4 | 4 | 64 | 35709.6 | 1878.3 | 34.7% | NOISY |
| small_grad_eigh_f64_pytorch_n4 | #2186 | pytorch | pytorch-cpu | f64 | op=eigh, n=4 | 4 | 8 | 80755.1 | 37341.1 | 35.0% | NOISY |
| small_grad_eigh_f64_trace-prepared_n16 | #2186 | trace-prepared | tenferro-rs | f64 | op=eigh, n=16 | 4 | 32 | 72016.9 | 8299.5 | 12.9% | NOISY |
| small_grad_eigh_f64_pytorch_n16 | #2186 | pytorch | pytorch-cpu | f64 | op=eigh, n=16 | 4 | 32 | 63496.1 | 1653.0 | 10.9% | NOISY |
| small_grad_lu_f64_trace-prepared_n4 | #2186 | trace-prepared | tenferro-rs | f64 | op=lu, n=4 | 4 | 128 | 18178.4 | 302.9 | 1.2% | ok |
| small_grad_lu_f64_pytorch_n4 | #2186 | pytorch | pytorch-cpu | f64 | op=lu, n=4 | 4 | 16 | 156518.2 | 6584.6 | 3.2% | ok |
| small_grad_lu_f64_trace-prepared_n16 | #2186 | trace-prepared | tenferro-rs | f64 | op=lu, n=16 | 4 | 128 | 24263.3 | 418.8 | 1.4% | ok |
| small_grad_lu_f64_pytorch_n16 | #2186 | pytorch | pytorch-cpu | f64 | op=lu, n=16 | 4 | 16 | 157591.2 | 8507.8 | 10.9% | NOISY |
| small_grad_svd_f64_trace-prepared_n4 | #2186 | trace-prepared | tenferro-rs | f64 | op=svd, n=4 | 4 | 64 | 54730.5 | 732.1 | 1.3% | ok |
| small_grad_svd_f64_pytorch_n4 | #2186 | pytorch | pytorch-cpu | f64 | op=svd, n=4 | 4 | 64 | 57912.8 | 1214.2 | 2.3% | ok |
| small_grad_svd_f64_trace-prepared_n16 | #2186 | trace-prepared | tenferro-rs | f64 | op=svd, n=16 | 4 | 32 | 108824.2 | 3187.5 | 2.6% | ok |
| small_grad_svd_f64_pytorch_n16 | #2186 | pytorch | pytorch-cpu | f64 | op=svd, n=16 | 4 | 32 | 101549.5 | 1658.9 | 2.6% | ok |
| small_grad_solve_f64_trace-prepared_n4 | #2186 | trace-prepared | tenferro-rs | f64 | op=solve, n=4 | 4 | 128 | 15718.1 | 186.8 | 1.3% | ok |
| small_grad_solve_f64_pytorch_n4 | #2186 | pytorch | pytorch-cpu | f64 | op=solve, n=4 | 4 | 64 | 35671.2 | 650.4 | 3.1% | ok |
| small_grad_solve_f64_trace-prepared_n16 | #2186 | trace-prepared | tenferro-rs | f64 | op=solve, n=16 | 4 | 128 | 19992.8 | 307.9 | 2.0% | ok |
| small_grad_solve_f64_pytorch_n16 | #2186 | pytorch | pytorch-cpu | f64 | op=solve, n=16 | 4 | 64 | 43230.5 | 293.6 | 0.8% | ok |
| fft_axis0_c64_fft-executor_n128 | #1765 | fft-executor | tenferro-rs | c64 | n=128 | 4 | 1 | 4041500.0 | 82417.0 | 2.4% | ok |
| fft_axis0_c64_pytorch_n128 | #1765 | pytorch | pytorch-cpu | c64 | n=128 | 4 | 1 | 18870042.0 | 192416.5 | 1.8% | ok |
| degenerate_dot_hadamard_f64_dot-general-shared_n1024 | #2111 | dot-general-shared | tenferro-rs | f64 | form=hadamard, n=1024 | 4 | 4 | 421864.5 | 45109.2 | 7.5% | ok |
| degenerate_dot_hadamard_f64_pytorch_n1024 | #2111 | pytorch | pytorch-cpu | f64 | form=hadamard, n=1024 | 4 | 4 | 456541.8 | 10093.8 | 4.6% | ok |
| degenerate_dot_outer_f64_dot-general-shared_n1024 | #2111 | dot-general-shared | tenferro-rs | f64 | form=outer, n=1024 | 4 | 4 | 339927.0 | 5984.2 | 4.4% | ok |
| degenerate_dot_outer_f64_pytorch_n1024 | #2111 | pytorch | pytorch-cpu | f64 | form=outer, n=1024 | 4 | 8 | 187166.6 | 5557.2 | 7.3% | ok |
| degenerate_dot_rank0_f64_dot-general-shared_n1024 | #2111 | dot-general-shared | tenferro-rs | f64 | form=rank0, n=1024 | 4 | 1 | 3680459.0 | 62500.5 | 1.7% | ok |
| degenerate_dot_rank0_f64_pytorch_n1024 | #2111 | pytorch | pytorch-cpu | f64 | form=rank0, n=1024 | 4 | 8 | 200645.8 | 15739.5 | 6.2% | ok |
| erased_add_f64_session-add_n256 | #1766 | session-add | tenferro-rs | f64 | n=256 | 4 | 4096 | 651.7 | 25.5 | 5.6% | ok |
| erased_add_f64_pytorch_n256 | #1766 | pytorch | pytorch-cpu | f64 | n=256 | 4 | 2048 | 836.6 | 37.8 | 323.1% | NOISY |
| erased_add_f64_session-add_n4096 | #1766 | session-add | tenferro-rs | f64 | n=4096 | 4 | 1024 | 1794.3 | 220.3 | 12.3% | NOISY |
| erased_add_f64_pytorch_n4096 | #1766 | pytorch | pytorch-cpu | f64 | n=4096 | 4 | 256 | 1563.2 | 112.0 | 4.7% | ok |
| erased_add_f64_session-add_n1048576 | #1766 | session-add | tenferro-rs | f64 | n=1048576 | 4 | 4 | 590552.0 | 101390.8 | 11.3% | NOISY |
| erased_add_f64_pytorch_n1048576 | #1766 | pytorch | pytorch-cpu | f64 | n=1048576 | 4 | 2 | 924833.5 | 137552.2 | 13.8% | NOISY |
| axpby_strided_source_f64_1024x1024_t4 | #2220 #2214 | strided-source | tenferro-rs | f64 | n=1024 | 4 | 1 | 2674708.0 | 500771.0 | 36.0% | NOISY |
| axpby_contiguous_source_f64_1024x1024_t4 | #2220 #2214 | contiguous-source | tenferro-rs | f64 | n=1024 | 4 | 4 | 724229.0 | 115661.5 | 13.9% | NOISY |
| axpby_strided_source_f64_1024x1024_host_t4 | #2220 #2214 | host | rust-loop | f64 | n=1024 | 4 | 1 | 2256167.0 | 984500.0 | 35.9% | NOISY |
| erased_construct_f64_col_n8 | #2177 | col-major | tenferro-rs | f64 | n=8, layout=col | 4 | 8192 | 63.4 | 11.9 | 32.2% | NOISY |
| erased_construct_f64_row_n8 | #2177 | row-major | tenferro-rs | f64 | n=8, layout=row | 4 | 16384 | 170.2 | 41.7 | 21.0% | NOISY |
| erased_construct_f64_col_n4096 | #2177 | col-major | tenferro-rs | f64 | n=4096, layout=col | 4 | 16384 | 62.2 | 7.5 | 36.1% | NOISY |
| erased_construct_f64_row_n4096 | #2177 | row-major | tenferro-rs | f64 | n=4096, layout=row | 4 | 128 | 14736.0 | 983.2 | 33.5% | NOISY |
| erased_construct_f64_col_n65536 | #2177 | col-major | tenferro-rs | f64 | n=65536, layout=col | 4 | 1024 | 99.5 | 20.9 | 92.9% | NOISY |
| erased_construct_f64_row_n65536 | #2177 | row-major | tenferro-rs | f64 | n=65536, layout=row | 4 | 16 | 232598.9 | 6028.7 | 2.8% | ok |
| eager_small_leaf_f64_eager-held-session | #1704 | eager-held-session | tenferro-rs | f64 | op=leaf | 4 | 2048 | 1825.5 | 153.8 | 22.0% | NOISY |
| eager_small_add_f64_eager-held-session | #1667 | eager-held-session | tenferro-rs | f64 | op=add | 4 | 2048 | 1418.1 | 164.3 | 18.4% | NOISY |
| eager_small_add_f64_pytorch | #1667 | pytorch | pytorch-cpu | f64 | op=add | 4 | 2048 | 690.5 | 40.1 | 4.4% | ok |
| eager_small_gemm_f64_eager-held-session | #1904 | eager-held-session | tenferro-rs | f64 | op=gemm | 4 | 1024 | 1951.2 | 99.9 | 5.6% | ok |
| eager_small_gemm_f64_pytorch | #1904 | pytorch | pytorch-cpu | f64 | op=gemm | 4 | 4096 | 762.3 | 25.5 | 3.0% | ok |
| trace_metadata_trace-fold_f32_1024x64_k50 | #2226 #2238 | trace-fold | tenferro-rs | f32 | rows=1024, cols=64, steps=50 | 4 | 2 | 1378895.5 | 283927.0 | 24.6% | NOISY |
| trace_metadata_trace-outputs_f32_1024x64_k50 | #2226 #2238 | trace-outputs | tenferro-rs | f32 | rows=1024, cols=64, steps=50 | 4 | 2 | 1751937.5 | 114177.2 | 4.8% | ok |
| trace_metadata_eager_f32_1024x64_k50 | #2226 #2238 | eager | tenferro-rs | f32 | rows=1024, cols=64, steps=50 | 4 | 2 | 1882354.5 | 165666.8 | 6.3% | ok |
| trace_metadata_pytorch_f32_1024x64_k50 | #2226 #2238 | pytorch | pytorch-cpu | f32 | rows=1024, cols=64, steps=50 | 4 | 1 | 2384917.0 | 225645.5 | 7.8% | ok |
| trace_metadata_fixed_neg_f32_8 | #2226 #2238 | trace-fold | tenferro-rs | f32 | rows=8, cols=1, steps=0 | 4 | 2048 | 1520.3 | 162.9 | 11.6% | NOISY |


## Timing boundaries

- **activation / eager-composed** — inside: hand-written formula from pre-B1 eager primitives, intermediate and output allocation; outside: backend/runtime construction, input, constant, mask and index construction, eager session entry, correctness_check.
- **activation / eager-single-call** — inside: one activation call, intermediate and output allocation; outside: backend/runtime construction, input, constant, mask and index construction, eager session entry, correctness_check.
- **activation / pytorch** — inside: torch gelu, output_allocation; outside: input_construction, torch.set_num_threads, correctness_check.
- **activation / session-single-call** — inside: one activation call, intermediate and output allocation; outside: backend/runtime construction, input, constant, mask and index construction, with_backend_session entry (CpuBackend::new()), correctness_check.
- **axpby_strided_source / contiguous-source** — inside: Engine::axpby_read_into_accum (contiguous owned source, compact owned destination); outside: backend/session construction, input_construction, strided/offset view construction (metadata only), per-sample destination allocation and seed restoration, correctness_check.
- **axpby_strided_source / host** — inside: sequential Rust strided axpby loop into a compact owned destination; outside: backend/session construction, input_construction, strided/offset view construction (metadata only), per-sample destination allocation and seed restoration, correctness_check.
- **axpby_strided_source / strided-source** — inside: Engine::axpby_read_into_accum (strided/offset source view, compact owned destination); outside: backend/session construction, input_construction, strided/offset view construction (metadata only), per-sample destination allocation and seed restoration, correctness_check.
- **binary_einsum_shortcut / dot-general** — inside: declared allocating product API, intrinsic output allocation; outside: inputs and scalar reference, engine and session, subscripts and ConcreteEinsumPlan preparation, explicit initialization and numerical validation, output destruction.
- **binary_einsum_shortcut / prepared** — inside: declared allocating product API, intrinsic output allocation; outside: inputs and scalar reference, engine and session, subscripts and ConcreteEinsumPlan preparation, explicit initialization and numerical validation, output destruction.
- **binary_einsum_shortcut / pytorch** — inside: declared allocating PyTorch product, output_allocation; outside: input_construction, torch.set_num_threads, correctness_check.
- **binary_einsum_shortcut / string** — inside: declared allocating product API, intrinsic output allocation; outside: inputs and scalar reference, engine and session, subscripts and ConcreteEinsumPlan preparation, explicit initialization and numerical validation, output destruction.
- **binary_einsum_shortcut / subscripts** — inside: declared allocating product API, intrinsic output allocation; outside: inputs and scalar reference, engine and session, subscripts and ConcreteEinsumPlan preparation, explicit initialization and numerical validation, output destruction.
- **composed_norm / eager-composed** — inside: eager composition (reduce_sum, scale_real, broadcast_in_dim, sub/add/mul, reduce_sum_squares, rsqrt), intermediate and output allocation; outside: eager_runtime_construction, input/weight/bias/eps construction, eager session entry, correctness_check.
- **composed_norm / eager-single-call** — inside: one EagerSession layer_norm / rms_norm call (composite), intermediate and output allocation; outside: eager_runtime_construction, input/weight/bias/eps construction, eager session entry, correctness_check.
- **composed_norm / pytorch-fused** — inside: torch.nn.functional.layer_norm (fused), output_allocation; outside: input_construction, torch.set_num_threads, correctness_check.
- **composed_norm / pytorch-fused** — inside: torch.nn.functional.rms_norm (fused), output_allocation; outside: input_construction, torch.set_num_threads, correctness_check.
- **cpu_gap_mwe / concrete-shared** — inside: reshape, intrinsic output allocation; outside: fixture construction, runtime/backend construction, session entry, FFT plans and scratch priming, correctness validation, retention vector allocation, output destruction.
- **cpu_gap_mwe / concrete-shared** — inside: triangular, intrinsic output allocation; outside: fixture construction, runtime/backend construction, session entry, FFT plans and scratch priming, correctness validation, retention vector allocation, output destruction.
- **cpu_gap_mwe / eager-shared** — inside: lstsq, intrinsic output allocation; outside: fixture construction, runtime/backend construction, session entry, FFT plans and scratch priming, correctness validation, retention vector allocation, output destruction.
- **cpu_gap_mwe / pytorch** — inside: lstsq, intrinsic output allocation, Python dispatch; outside: input_construction, torch.set_num_threads, correctness_check.
- **cpu_gap_mwe / pytorch** — inside: reshape, intrinsic output allocation, Python dispatch; outside: input_construction, torch.set_num_threads, correctness_check.
- **cpu_gap_mwe / pytorch** — inside: triangular, intrinsic output allocation, Python dispatch; outside: input_construction, torch.set_num_threads, correctness_check.
- **decode_projection / eager-shared** — inside: eager_dot_general, output_allocation; outside: backend_construction, input_construction, correctness_check, eager_session_entry_exit.
- **decode_projection / faer-direct** — inside: faer_matmul_into_preallocated_output; outside: backend_construction, input_construction, correctness_check, output_allocation.
- **decode_projection / host-sgemm** — inside: matrixmultiply_sgemm_into_preallocated_output; outside: backend_construction, input_construction, correctness_check, output_allocation.
- **decode_projection / pytorch** — inside: torch.nn.functional.linear, output_allocation; outside: input_construction, torch.set_num_threads, correctness_check.
- **decode_projection_view_operand / prepared-contiguous** — inside: Session::dot_general_read_cached (prepared cache slot) on the borrowed caller-held session, output_allocation; outside: backend/session construction, input_construction, weight view construction (metadata only), correctness_check.
- **decode_projection_view_operand / prepared-view** — inside: Session::dot_general_read_cached (prepared cache slot) on the borrowed caller-held session, output_allocation; outside: backend/session construction, input_construction, weight view construction (metadata only), correctness_check.
- **decode_projection_view_operand / pytorch** — inside: torch.matmul(x, W.T), output_allocation; outside: input_construction, torch.set_num_threads, correctness_check.
- **degenerate_dot_general / dot-general-shared** — inside: Session::dot_general_read on the borrowed caller-held session, output_allocation; outside: backend/session construction, input_construction, correctness_check.
- **degenerate_dot_general / pytorch** — inside: torch hadamard (degenerate dot), output_allocation; outside: input_construction, torch.set_num_threads, correctness_check.
- **degenerate_dot_general / pytorch** — inside: torch outer (degenerate dot), output_allocation; outside: input_construction, torch.set_num_threads, correctness_check.
- **degenerate_dot_general / pytorch** — inside: torch rank0 (degenerate dot), output_allocation; outside: input_construction, torch.set_num_threads, correctness_check.
- **eager_small_call / eager-held-session** — inside: EagerSession::add on two held 8-element leaves; outside: backend/runtime construction, eager session entry (held, outside the clock), leaf input construction, correctness_check.
- **eager_small_call / eager-held-session** — inside: EagerSession::constant_from on a prepared 8-element payload; outside: backend/runtime construction, eager session entry (held, outside the clock), leaf input construction, correctness_check.
- **eager_small_call / eager-held-session** — inside: EagerSession::dot_general on two held 4x4 leaves; outside: backend/runtime construction, eager session entry (held, outside the clock), leaf input construction, correctness_check.
- **eager_small_call / pytorch** — inside: torch.add (8 elements), output_allocation; outside: input_construction, torch.set_num_threads, correctness_check.
- **eager_small_call / pytorch** — inside: torch.matmul (4x4), output_allocation; outside: input_construction, torch.set_num_threads, correctness_check.
- **erased_binary_add / pytorch** — inside: torch.add, output_allocation; outside: input_construction, torch.set_num_threads, correctness_check.
- **erased_binary_add / session-add** — inside: erased Tensor::add on the borrowed caller-held session, output_allocation; outside: backend/session construction, input_construction, correctness_check.
- **erased_construct / col-major** — inside: erased Tensor::from_vec_* on a prepared payload Vec; outside: payload clone and restore, shape_check, allocation counting pass (separate).
- **erased_construct / row-major** — inside: erased Tensor::from_vec_* on a prepared payload Vec; outside: payload clone and restore, shape_check, allocation counting pass (separate).
- **fft_axis_c2c / fft-executor** — inside: FftExecutor::fft over axis 0 on the borrowed caller-held session, output_allocation; outside: backend/session construction, input_construction, correctness_check.
- **fft_axis_c2c / pytorch** — inside: torch.fft.fft(dim=0), output_allocation; outside: input_construction, torch.set_num_threads, correctness_check.
- **gemm_mm256 / dot-general-shared** — inside: dot_general_read, output_allocation; outside: backend_construction (CpuBackend::with_threads(1)), input_construction, session_entry_exit, correctness_check.
- **metadata_clone / julia-base** — inside: PermutedDimsArray metadata descriptor; outside: owner and transposed view creation, alias/layout checks, JIT compilation, output retention allocation and cleanup.
- **metadata_clone / metadata-clone** — inside: clone of prepared borrowed rank-8 transpose-view metadata; outside: owner allocation and values, transpose-view construction, alias and layout validation, output destruction.
- **rank1_dot / dot-general-rank1** — inside: dot_general_read (rank-1, rank-0 tensor output), output_allocation; outside: backend_construction, input_construction, operand_conjugation, session_entry_exit, correctness_check.
- **rank1_dot / portable-loop** — inside: sequential Rust loop over borrowed slices, scalar result; outside: input_construction, correctness_check.
- **rank1_dot / pytorch-vdot** — inside: torch.vdot (BLAS ddot/zdotc), scalar tensor allocation; outside: input_construction, torch.set_num_threads, correctness_check.
- **reduce_mean / eager-composed** — inside: reduce_sum + scale_real, intermediate and output allocation; outside: backend/runtime construction, input, constant, mask and index construction, eager session entry, correctness_check.
- **reduce_mean / eager-single-call** — inside: one reduce_mean call, intermediate and output allocation; outside: backend/runtime construction, input, constant, mask and index construction, eager session entry, correctness_check.
- **reduce_mean / pytorch** — inside: torch.mean, output_allocation; outside: input_construction, torch.set_num_threads, correctness_check.
- **reduce_mean / session-single-call** — inside: one reduce_mean call, intermediate and output allocation; outside: backend/runtime construction, input, constant, mask and index construction, with_backend_session entry (CpuBackend::new()), correctness_check.
- **small_contraction_pool_hop / shared-session-default-pool** — inside: dot_general_read, output_allocation; outside: backend_construction, input_construction, correctness_check.
- **small_grad_trace / pytorch** — inside: torch.linalg.eigh + autograd.grad, output_allocation; outside: input_construction, torch.set_num_threads, correctness_check.
- **small_grad_trace / pytorch** — inside: torch.linalg.lu + autograd.grad, output_allocation; outside: input_construction, torch.set_num_threads, correctness_check.
- **small_grad_trace / pytorch** — inside: torch.linalg.qr + autograd.grad, output_allocation; outside: input_construction, torch.set_num_threads, correctness_check.
- **small_grad_trace / pytorch** — inside: torch.linalg.solve + autograd.grad, output_allocation; outside: input_construction, torch.set_num_threads, correctness_check.
- **small_grad_trace / pytorch** — inside: torch.linalg.svd + autograd.grad, output_allocation; outside: input_construction, torch.set_num_threads, correctness_check.
- **small_grad_trace / trace-prepared** — inside: Runtime::run of the prepared gradient program on the caller-held engine, output_allocation; outside: backend/runtime construction, trace, AD transform and compile, prepare, input_construction, finite-difference correctness check.
- **small_solves / pytorch-solve-batched** — inside: torch.linalg.solve on (batch, k, k), output_allocation; outside: input_construction, torch.set_num_threads, correctness_check.
- **small_solves / pytorch-solve-triangular-batched** — inside: torch.linalg.solve_triangular on (batch, k, k), output_allocation; outside: input_construction, torch.set_num_threads, correctness_check.
- **small_solves / solve-batched** — inside: one batched solve call ([k, k, batch]), output_allocation; outside: backend_construction, input_construction (items pre-sliced), session_entry_exit, correctness_check.
- **small_solves / solve-loop** — inside: batch x rank-2 solve calls, output_allocation per item; outside: backend_construction, input_construction (items pre-sliced), session_entry_exit, correctness_check.
- **small_solves / triangular-solve-loop** — inside: batch x rank-2 solve calls, output_allocation per item; outside: backend_construction, input_construction (items pre-sliced), session_entry_exit, correctness_check.
- **softmax / eager-composed** — inside: max-subtracted composition (reduce_max, broadcast_in_dim, sub, exp, reduce_sum, div/log; select when masked), intermediate and output allocation; outside: backend/runtime construction, input, constant, mask and index construction, eager session entry, correctness_check.
- **softmax / eager-single-call** — inside: one softmax-family call (composite, all-masked guard included), intermediate and output allocation; outside: backend/runtime construction, input, constant, mask and index construction, eager session entry, correctness_check.
- **softmax / pytorch** — inside: torch log_softmax, output_allocation; outside: input_construction, torch.set_num_threads, correctness_check.
- **softmax / pytorch** — inside: torch masked_softmax, output_allocation; outside: input_construction, torch.set_num_threads, correctness_check.
- **softmax / pytorch** — inside: torch softmax, output_allocation; outside: input_construction, torch.set_num_threads, correctness_check.
- **softmax / session-single-call** — inside: one softmax-family call (composite, all-masked guard included), intermediate and output allocation; outside: backend/runtime construction, input, constant, mask and index construction, with_backend_session entry (CpuBackend::new()), correctness_check.
- **softmax_pool / pool4** — inside: axis-0 softmax, output allocation; outside: input and engine construction, session entry, correctness check.
- **softmax_pool / pytorch** — inside: torch softmax, output_allocation; outside: input_construction, torch.set_num_threads, correctness_check.
- **softmax_pool / serial** — inside: axis-0 softmax, output allocation; outside: input and engine construction, session entry, correctness check.
- **take_along_axis / eager-gather-prebuilt** — inside: gather with prebuilt (row, batch) index tuples, intermediate and output allocation; outside: backend/runtime construction, input, constant, mask and index construction, eager session entry, correctness_check.
- **take_along_axis / eager-single-call** — inside: one take_along_axis call (index tuples built inside, then gather), intermediate and output allocation; outside: backend/runtime construction, input, constant, mask and index construction, eager session entry, correctness_check.
- **take_along_axis / pytorch** — inside: torch.take_along_dim, output_allocation; outside: input_construction, torch.set_num_threads, correctness_check.
- **take_along_axis / session-single-call** — inside: one take_along_axis call (index tuples built inside, then gather), intermediate and output allocation; outside: backend/runtime construction, input, constant, mask and index construction, with_backend_session entry (CpuBackend::new()), correctness_check.
- **tanh_chain / compiled-prepared** — inside: Runtime::run on the caller-held engine, fused region execution, output_allocation; outside: backend/runtime construction, trace and compile, prepare, input_construction, eager session entry (eager arm), correctness_check.
- **tanh_chain / eager-shared** — inside: chain of eager tanh ops, intermediate and output allocation; outside: backend/runtime construction, trace and compile, prepare, input_construction, eager session entry (eager arm), correctness_check.
- **thin_complex_gemm / dot-general** — inside: declared allocating product API, intrinsic output allocation; outside: inputs and scalar reference, engine and session, subscripts and ConcreteEinsumPlan preparation, explicit initialization and numerical validation, output destruction.
- **thin_complex_gemm / faer-direct** — inside: declared allocating product API, intrinsic output allocation; outside: inputs and scalar reference, engine and session, subscripts and ConcreteEinsumPlan preparation, explicit initialization and numerical validation, output destruction.
- **thin_complex_gemm / pytorch** — inside: declared allocating PyTorch product, output_allocation; outside: input_construction, torch.set_num_threads, correctness_check.
- **trace_metadata / eager** — inside: triu transpose chain and neg, output allocation; outside: fixture, engine, session, tracing, compilation and preparation, correctness check.
- **trace_metadata / pytorch** — inside: triu and transpose chain, neg, output allocation; outside: input_construction, torch.set_num_threads, correctness_check.
- **trace_metadata / trace-fold** — inside: triu transpose chain and neg, output allocation; outside: fixture, engine, session, tracing, compilation and preparation, correctness check.
- **trace_metadata / trace-outputs** — inside: triu transpose chain and neg, output allocation; outside: fixture, engine, session, tracing, compilation and preparation, correctness check.
- **umbrella_add / pytorch** — inside: torch.add allocating; no session concept; outside: input_construction, torch.set_num_threads, correctness_check.
- **umbrella_add / pytorch** — inside: torch.add out= caller output; no session concept; outside: input_construction, torch.set_num_threads, correctness_check.
- **umbrella_add / tenferro** — inside: add_read_into with typed reads and caller output; outside: backend/thread-pool initialization and explicit priming, input/output preparation and per-sample restoration, entered held session (except explicitly named entry diagnostics), validation, allocation counter queries, output destruction.
- **umbrella_add / tenferro** — inside: typed allocating add in held session; outside: backend/thread-pool initialization and explicit priming, input/output preparation and per-sample restoration, entered held session (except explicitly named entry diagnostics), validation, allocation counter queries, output destruction.
- **umbrella_axpby / host** — inside: strided alpha*x+beta*y, no staging; outside: backend/thread-pool initialization and explicit priming, input/output preparation and per-sample restoration, entered held session (except explicitly named entry diagnostics), validation, allocation counter queries, output destruction.
- **umbrella_axpby / tenferro** — inside: strided alpha*x+beta*y, no staging; outside: backend/thread-pool initialization and explicit priming, input/output preparation and per-sample restoration, entered held session (except explicitly named entry diagnostics), validation, allocation counter queries, output destruction.
- **umbrella_gemm / pytorch** — inside: torch.matmul allocating; equivalent prepared binary contraction; outside: input_construction, torch.set_num_threads, correctness_check.
- **umbrella_gemm / tenferro** — inside: prepared allocating einsum execution; outside: backend/thread-pool initialization and explicit priming, input/output preparation and per-sample restoration, entered held session (except explicitly named entry diagnostics), validation, allocation counter queries, output destruction.
- **umbrella_lu / host** — inside: independent scalar partial-pivot packed LU, pivots and parity; outside: backend/thread-pool initialization and explicit priming, input/output preparation and per-sample restoration, entered held session (except explicitly named entry diagnostics), validation, allocation counter queries, output destruction.
- **umbrella_lu / pytorch** — inside: torch.linalg.lu_solve only; factorization prepared outside timing; outside: input_construction, torch.set_num_threads, correctness_check.
- **umbrella_lu / tenferro** — inside: public LinalgBackend packed LU, pivots and parity; outside: backend/thread-pool initialization and explicit priming, input/output preparation and per-sample restoration, entered held session (except explicitly named entry diagnostics), validation, allocation counter queries, output destruction.
- **umbrella_lu / tenferro** — inside: public LinalgBackend prepared LU solve only; outside: backend/thread-pool initialization and explicit priming, input/output preparation and per-sample restoration, entered held session (except explicitly named entry diagnostics), validation, allocation counter queries, output destruction.
- **umbrella_matmul / pytorch** — inside: torch.matmul allocating; outside: input_construction, torch.set_num_threads, correctness_check.
- **umbrella_matmul / tenferro** — inside: typed allocating matmul in held session; outside: backend/thread-pool initialization and explicit priming, input/output preparation and per-sample restoration, entered held session (except explicitly named entry diagnostics), validation, allocation counter queries, output destruction.
- **umbrella_neg / pytorch** — inside: torch.neg allocating; outside: input_construction, torch.set_num_threads, correctness_check.
- **umbrella_neg / tenferro** — inside: typed allocating neg; outside: backend/thread-pool initialization and explicit priming, input/output preparation and per-sample restoration, entered held session (except explicitly named entry diagnostics), validation, allocation counter queries, output destruction.

### SKIPPED: activation_erf_f32_eager-single-call_1024x64 (1 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: activation_erf_f32_pytorch_1024x64 (1 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: activation_sigmoid_f32_eager-single-call_1024x64 (1 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: activation_sigmoid_f32_pytorch_1024x64 (1 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: activation_silu_f32_eager-single-call_1024x64 (1 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: activation_silu_f32_pytorch_1024x64 (1 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: activation_softplus_f32_eager-single-call_1024x64 (1 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: activation_softplus_f32_pytorch_1024x64 (1 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: activation_gelu_tanh_f32_eager-single-call_1024x64 (1 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: activation_gelu_tanh_f32_pytorch_1024x64 (1 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_structural_broadcast_in_dim_direct-shared (1 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_structural_broadcast_in_dim_pytorch (1 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_real_tanh_direct-shared (1 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_real_tanh_jax (1 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_real_log1p_direct-shared (1 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_real_log1p_pytorch (1 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_real_cos_direct-shared (1 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_real_cos_pytorch (1 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_real_sin_direct-shared (1 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_real_sin_pytorch (1 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_real_exp_direct-shared (1 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_real_exp_jax (1 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_real_expm1_direct-shared (1 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_real_expm1_jax (1 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_real_pow_direct-shared (1 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_real_pow_pytorch (1 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_complex_exp_direct-shared (1 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_complex_exp_julia-base (1 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_real_log_direct-shared (1 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_real_log_pytorch (1 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_real_reduce_max_axis1_direct-shared (1 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_real_reduce_max_axis1_julia-base (1 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_real_reduce_prod_axis1_direct-shared (1 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_real_reduce_prod_axis1_pytorch (1 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_real_reduce_sum_axis1_direct-shared (1 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_real_reduce_sum_axis1_pytorch (1 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_real_reduce_max_axis0_direct-shared (1 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_real_reduce_max_axis0_julia-base (1 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_real_reduce_min_axis1_direct-shared (1 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_real_reduce_min_axis1_julia-base (1 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_real_reduce_min_axis0_direct-shared (1 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_real_reduce_min_axis0_julia-base (1 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_real_reduce_max_all_direct-shared (1 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_real_reduce_max_all_julia-base (1 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_real_reduce_min_all_direct-shared (1 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_real_reduce_min_all_julia-base (1 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_complex_norm_fro_direct-shared (1 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_complex_norm_fro_jax (1 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_index_gather_direct-shared (1 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_index_gather_jax (1 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_index_scatter_direct-shared (1 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_index_scatter_pytorch (1 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_index_dynamic_update_slice_direct-shared (1 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_index_dynamic_update_slice_jax (1 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_perm_reverse_15d_3_materialize (1 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_perm_reverse_15d_3_strided-rs (1 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_structural_transpose_direct-shared (1 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_structural_transpose_pytorch (1 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_perm_reverse_23d_2_materialize (1 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_perm_reverse_23d_2_strided-rs (1 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_perm_transpose_3d_256_201_materialize (1 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_perm_transpose_3d_256_201_strided-rs (1 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_perm_cyclic_15d_3_materialize (1 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_perm_cyclic_15d_3_strided-rs (1 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_perm_transpose_3d_256_102_materialize (1 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_perm_transpose_3d_256_102_strided-rs (1 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_perm_tn_light_415_24d_scattered_to_colmajor_materialize (1 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_perm_tn_light_415_24d_scattered_to_colmajor_strided-rs (1 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_perm_tn_light_415_24d_contiguous_same_perm_materialize (1 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_perm_tn_light_415_24d_contiguous_same_perm_strided-rs (1 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_perm_rotation_6d_32_32_32_32_16_16_materialize (1 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_perm_rotation_6d_32_32_32_32_16_16_strided-rs (1 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_structural_extract_diagonal_direct-shared (1 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_structural_extract_diagonal_jax (1 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_fft_irfft_direct-shared (1 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_fft_irfft_mkl-dfti (1 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_fft_rfft_direct-shared (1 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_fft_rfft_mkl-dfti (1 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_fft_fft_direct-shared (1 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_fft_fft_mkl-dfti (1 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_fft_ifft_direct-shared (1 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_fft_ifft_mkl-dfti (1 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_metadata_transpose_view_metadata (1 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_metadata_transpose_view_julia-base (1 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_metadata_transpose_view_metadata-host-dyn (1 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_metadata_transpose_view_metadata-host-static (1 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_metadata_slice_view_metadata (1 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_metadata_slice_view_julia-base (1 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_metadata_slice_view_metadata-host-dyn (1 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_metadata_slice_view_metadata-host-static (1 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_metadata_reshape_view_metadata (1 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_metadata_reshape_view_julia-base (1 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_metadata_reshape_view_metadata-host-dyn (1 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_metadata_reshape_view_metadata-host-static (1 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: activation_erf_f32_eager-single-call_1024x64 (4 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: activation_erf_f32_pytorch_1024x64 (4 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: activation_sigmoid_f32_eager-single-call_1024x64 (4 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: activation_sigmoid_f32_pytorch_1024x64 (4 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: activation_silu_f32_eager-single-call_1024x64 (4 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: activation_silu_f32_pytorch_1024x64 (4 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: activation_softplus_f32_eager-single-call_1024x64 (4 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: activation_softplus_f32_pytorch_1024x64 (4 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: activation_gelu_tanh_f32_eager-single-call_1024x64 (4 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: activation_gelu_tanh_f32_pytorch_1024x64 (4 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_structural_broadcast_in_dim_direct-shared (4 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_structural_broadcast_in_dim_pytorch (4 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_real_tanh_direct-shared (4 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_real_tanh_jax (4 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_real_log1p_direct-shared (4 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_real_log1p_pytorch (4 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_real_cos_direct-shared (4 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_real_cos_pytorch (4 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_real_sin_direct-shared (4 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_real_sin_pytorch (4 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_real_exp_direct-shared (4 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_real_exp_jax (4 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_real_expm1_direct-shared (4 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_real_expm1_jax (4 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_real_pow_direct-shared (4 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_real_pow_pytorch (4 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_complex_exp_direct-shared (4 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_complex_exp_julia-base (4 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_real_log_direct-shared (4 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_real_log_pytorch (4 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_real_reduce_max_axis1_direct-shared (4 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_real_reduce_max_axis1_julia-base (4 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_real_reduce_prod_axis1_direct-shared (4 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_real_reduce_prod_axis1_pytorch (4 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_real_reduce_sum_axis1_direct-shared (4 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_real_reduce_sum_axis1_pytorch (4 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_real_reduce_max_axis0_direct-shared (4 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_real_reduce_max_axis0_julia-base (4 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_real_reduce_min_axis1_direct-shared (4 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_real_reduce_min_axis1_julia-base (4 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_real_reduce_min_axis0_direct-shared (4 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_real_reduce_min_axis0_julia-base (4 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_real_reduce_max_all_direct-shared (4 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_real_reduce_max_all_julia-base (4 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_real_reduce_min_all_direct-shared (4 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_real_reduce_min_all_julia-base (4 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_complex_norm_fro_direct-shared (4 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_complex_norm_fro_jax (4 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_index_gather_direct-shared (4 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_index_gather_jax (4 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_index_scatter_direct-shared (4 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_index_scatter_pytorch (4 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_index_dynamic_update_slice_direct-shared (4 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_index_dynamic_update_slice_jax (4 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_perm_reverse_15d_3_materialize (4 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_perm_reverse_15d_3_strided-rs (4 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_structural_transpose_direct-shared (4 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_structural_transpose_pytorch (4 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_perm_reverse_23d_2_materialize (4 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_perm_reverse_23d_2_strided-rs (4 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_perm_transpose_3d_256_201_materialize (4 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_perm_transpose_3d_256_201_strided-rs (4 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_perm_cyclic_15d_3_materialize (4 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_perm_cyclic_15d_3_strided-rs (4 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_perm_transpose_3d_256_102_materialize (4 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_perm_transpose_3d_256_102_strided-rs (4 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_perm_tn_light_415_24d_scattered_to_colmajor_materialize (4 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_perm_tn_light_415_24d_scattered_to_colmajor_strided-rs (4 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_perm_tn_light_415_24d_contiguous_same_perm_materialize (4 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_perm_tn_light_415_24d_contiguous_same_perm_strided-rs (4 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_perm_rotation_6d_32_32_32_32_16_16_materialize (4 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_perm_rotation_6d_32_32_32_32_16_16_strided-rs (4 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_structural_extract_diagonal_direct-shared (4 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_structural_extract_diagonal_jax (4 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_fft_irfft_direct-shared (4 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_fft_irfft_mkl-dfti (4 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_fft_rfft_direct-shared (4 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_fft_rfft_mkl-dfti (4 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_fft_fft_direct-shared (4 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_fft_fft_mkl-dfti (4 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_fft_ifft_direct-shared (4 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_fft_ifft_mkl-dfti (4 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_metadata_transpose_view_metadata (4 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_metadata_transpose_view_julia-base (4 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_metadata_transpose_view_metadata-host-dyn (4 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_metadata_transpose_view_metadata-host-static (4 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_metadata_slice_view_metadata (4 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_metadata_slice_view_julia-base (4 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_metadata_slice_view_metadata-host-dyn (4 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_metadata_slice_view_metadata-host-static (4 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_metadata_reshape_view_metadata (4 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_metadata_reshape_view_julia-base (4 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_metadata_reshape_view_metadata-host-dyn (4 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```

### SKIPPED: followup_metadata_reshape_view_metadata-host-static (4 threads)
```
cpu_followup MWE needs the Linux system-MKL build; skipped on darwin
```
