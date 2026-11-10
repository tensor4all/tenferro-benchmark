# CPU Performance-Issue Workloads

- Suite: `cpu/perf_issues`
- Raw run: `data/results/mac-cpu/cpu/perf_issues/20261109_184821`

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

Full metadata: `data/results/mac-cpu/cpu/perf_issues/20261109_184821/run_t1.yaml`

```yaml
timestamp: '2026-11-09T18:48:21Z'
tenferro_rs:
  path: /Users/runner/work/tenferro-benchmark/tenferro-benchmark/extern/tenferro-rs
  commit: 9a850b06155405acc99ecd0648829e9e6d3866ba
  dirty: false
  features:
  - blas-accelerate
  resolved_path: /Users/runner/work/tenferro-benchmark/tenferro-benchmark/extern/tenferro-rs
blas:
  implementation: accelerate
  version: unknown
  library: /System/Library/Frameworks/Accelerate.framework
environment:
  container: none
  hostname: sjc22-bt146-f3598dff-34be-4214-a267-4b39e1fa9959-1633B0D39AC3.local
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
    BENCHMARK_COMMIT: ad9aa8a2fc06c36f8ce1d1dfb1ce9fb1cd9a9c46
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
    RUSTC_VERSION: rustc 1.99.1 (58cf8f9c3 2026-10-13)
    BENCH_INSTANCE: null
```

## run_t4

Full metadata: `data/results/mac-cpu/cpu/perf_issues/20261109_184821/run_t4.yaml`

```yaml
timestamp: '2026-11-09T18:49:23Z'
tenferro_rs:
  path: /Users/runner/work/tenferro-benchmark/tenferro-benchmark/extern/tenferro-rs
  commit: 9a850b06155405acc99ecd0648829e9e6d3866ba
  dirty: false
  features:
  - blas-accelerate
  resolved_path: /Users/runner/work/tenferro-benchmark/tenferro-benchmark/extern/tenferro-rs
blas:
  implementation: accelerate
  version: unknown
  library: /System/Library/Frameworks/Accelerate.framework
environment:
  container: none
  hostname: sjc22-bt146-f3598dff-34be-4214-a267-4b39e1fa9959-1633B0D39AC3.local
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
    BENCHMARK_COMMIT: ad9aa8a2fc06c36f8ce1d1dfb1ce9fb1cd9a9c46
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
    RUSTC_VERSION: rustc 1.99.1 (58cf8f9c3 2026-10-13)
    BENCH_INSTANCE: null
```

## Case status

- Manifest: `cpu/perf_issues` version 22, coverage `quick`, effort `standard`
- Expected 681, selected 681, executed 493, unsupported 0, failed 0, skipped 188, missing 0, noisy 421.
- Complete: **NO** — only executed cases count as covered; unsupported, skipped, failed, missing and unselected cases never do.

| expected | selected | executed | unsupported | failed | skipped | missing | noisy |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 681 | 681 | 493 | 0 | 0 | 188 | 0 | 421 |

- Skipped: `activation_erf_f32_eager-single-call_1024x64@t1`, `activation_erf_f32_pytorch_1024x64@t1`, `activation_sigmoid_f32_eager-single-call_1024x64@t1`, `activation_sigmoid_f32_pytorch_1024x64@t1`, `activation_silu_f32_eager-single-call_1024x64@t1`, `activation_silu_f32_pytorch_1024x64@t1`, `activation_softplus_f32_eager-single-call_1024x64@t1`, `activation_softplus_f32_pytorch_1024x64@t1`, `activation_gelu_tanh_f32_eager-single-call_1024x64@t1`, `activation_gelu_tanh_f32_pytorch_1024x64@t1`, `followup_structural_broadcast_in_dim_direct-shared@t1`, `followup_structural_broadcast_in_dim_pytorch@t1`, `followup_real_tanh_direct-shared@t1`, `followup_real_tanh_jax@t1`, `followup_real_log1p_direct-shared@t1`, `followup_real_log1p_pytorch@t1`, `followup_real_cos_direct-shared@t1`, `followup_real_cos_pytorch@t1`, `followup_real_sin_direct-shared@t1`, `followup_real_sin_pytorch@t1`, `followup_real_exp_direct-shared@t1`, `followup_real_exp_jax@t1`, `followup_real_expm1_direct-shared@t1`, `followup_real_expm1_jax@t1`, `followup_real_pow_direct-shared@t1`, `followup_real_pow_pytorch@t1`, `followup_complex_exp_direct-shared@t1`, `followup_complex_exp_julia-base@t1`, `followup_real_log_direct-shared@t1`, `followup_real_log_pytorch@t1`, `followup_real_reduce_max_axis1_direct-shared@t1`, `followup_real_reduce_max_axis1_julia-base@t1`, `followup_real_reduce_prod_axis1_direct-shared@t1`, `followup_real_reduce_prod_axis1_pytorch@t1`, `followup_real_reduce_sum_axis1_direct-shared@t1`, `followup_real_reduce_sum_axis1_pytorch@t1`, `followup_real_reduce_max_axis0_direct-shared@t1`, `followup_real_reduce_max_axis0_julia-base@t1`, `followup_real_reduce_min_axis1_direct-shared@t1`, `followup_real_reduce_min_axis1_julia-base@t1` (+148 more)
- Noisy: `two_operand_einsum_f64_2x2_string@t1`, `two_operand_einsum_f64_2x2_subscripts@t1`, `two_operand_einsum_f64_2x2_prepared@t1`, `two_operand_einsum_f64_2x2_dot-general@t1`, `two_operand_einsum_f64_2x2_pytorch@t1`, `two_operand_einsum_f64_16x16_string@t1`, `two_operand_einsum_f64_16x16_subscripts@t1`, `two_operand_einsum_f64_16x16_prepared@t1`, `two_operand_einsum_f64_16x16_dot-general@t1`, `two_operand_einsum_f64_16x16_pytorch@t1`, `two_operand_einsum_c64_2x2_string@t1`, `two_operand_einsum_c64_2x2_subscripts@t1`, `two_operand_einsum_c64_2x2_prepared@t1`, `two_operand_einsum_c64_2x2_dot-general@t1`, `two_operand_einsum_c64_2x2_pytorch@t1`, `two_operand_einsum_c64_16x16_string@t1`, `two_operand_einsum_c64_16x16_subscripts@t1`, `two_operand_einsum_c64_16x16_prepared@t1`, `two_operand_einsum_c64_16x16_pytorch@t1`, `two_operand_einsum_f64_abc_cef_abef_string@t1`, `two_operand_einsum_f64_abc_cef_abef_subscripts@t1`, `two_operand_einsum_f64_abc_cef_abef_prepared@t1`, `two_operand_einsum_f64_abc_cef_abef_dot-general@t1`, `two_operand_einsum_f64_abc_cef_abef_pytorch@t1`, `thin_c64_gemm_1x141_141x282_t1_faer-direct@t1`, `thin_c64_gemm_1x141_141x282_t1_pytorch@t1`, `thin_c64_gemm_2x141_141x282_t1@t1`, `thin_c64_gemm_2x141_141x282_t1_faer-direct@t1`, `thin_c64_gemm_2x141_141x282_t1_pytorch@t1`, `thin_c64_gemm_8x141_141x282_t1_faer-direct@t1`, `thin_c64_gemm_8x141_141x282_t1_pytorch@t1`, `decode_proj_f32_eager-shared_in1024_out1024_len8@t1`, `decode_proj_f32_faer-direct_in1024_out1024_len8@t1`, `decode_proj_f32_pytorch_in1024_out1024_len8@t1`, `decode_proj_f32_eager-shared_in1024_out1024_len64@t1`, `decode_proj_f32_pytorch_in1024_out1024_len64@t1`, `decode_proj_f32_eager-shared_in1024_out1024_len512@t1`, `decode_proj_f32_faer-direct_in1024_out1024_len512@t1`, `decode_proj_f32_host-sgemm_in1024_out1024_len512@t1`, `decode_proj_f32_pytorch_in1024_out1024_len512@t1` (+381 more)

Numerical checks: **493/681 recorded rows passed** (case × thread count).

## Steady-state operation rows

Each batch is one wall-clock interval over many operations; every output is retained until the clock stops. Median and inclusive IQR are per operation, in nanoseconds; CoV is sample standard deviation / mean (NOISY means CoV > 10%, a descriptive label only). Setup (inputs, backends, runtimes, session entry, tracing/compilation, preparation) is outside timing; faer-direct and host-sgemm write into a preallocated output, the tenferro and PyTorch arms allocate their output. The thin_complex_gemm faer-direct arm instead allocates its output inside timing to match its product references. `reference_for` in the case file names the case each reference arm is compared with. GPU batches end with one device synchronize (a per-call synchronize where the transfer is the operation); the Threads column is the host thread setting.

### Compute- and bandwidth-bound operations

| Case ID | Issues | Arm | Backend | Dtype | Workload | Threads | Ops/batch | Median ns/op | IQR ns | CoV | Status |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| softmax_f64_pool_serial_1024x4096 | #2273 #2274 #2276 | serial | tenferro-rs | f64 | rows=1024, cols=4096, axis=0 | 1 | 1 | 16103542.0 | 1357957.5 | 9.7% | ok |
| softmax_f64_pool_pytorch_t1_1024x4096 | #2273 #2274 #2276 | pytorch | pytorch-cpu | f64 | rows=1024, cols=4096, axis=0 | 1 | 1 | 20059833.0 | 1309646.0 | 5.8% | ok |
| softmax_f64_pool_pool4_1024x4096 | #2273 #2274 #2276 | pool4 | tenferro-rs | f64 | rows=1024, cols=4096, axis=0 | 4 | 1 | 10537625.0 | 1916084.0 | 13.6% | NOISY |
| softmax_f64_pool_pytorch_t4_1024x4096 | #2273 #2274 #2276 | pytorch | pytorch-cpu | f64 | rows=1024, cols=4096, axis=0 | 4 | 1 | 13149208.0 | 2704291.5 | 24.0% | NOISY |

### Latency and per-call overhead

These rows measure dispatch/operation latency; they do not measure compute throughput.

| Case ID | Issues | Arm | Backend | Dtype | Workload | Threads | Ops/batch | Median ns/op | IQR ns | CoV | Status |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| two_operand_einsum_f64_2x2_string | #2325 #2324 | string | tenferro-rs | f64 | m=2, k=2, n=2, format=ij,jk->ik | 1 | 1024 | 2612.9 | 943.5 | 37.9% | NOISY |
| two_operand_einsum_f64_2x2_subscripts | #2325 #2324 | subscripts | tenferro-rs | f64 | m=2, k=2, n=2, format=ij,jk->ik | 1 | 2048 | 1905.3 | 598.9 | 27.7% | NOISY |
| two_operand_einsum_f64_2x2_prepared | #2325 #2324 | prepared | tenferro-rs | f64 | m=2, k=2, n=2, format=ij,jk->ik | 1 | 2048 | 1197.1 | 372.7 | 20.3% | NOISY |
| two_operand_einsum_f64_2x2_dot-general | #2325 #2324 | dot-general | tenferro-rs | f64 | m=2, k=2, n=2, format=ij,jk->ik | 1 | 1024 | 1124.6 | 334.7 | 34.9% | NOISY |
| two_operand_einsum_f64_2x2_pytorch | #2325 #2324 | pytorch | pytorch-cpu | f64 | m=2, k=2, n=2, format=ij,jk->ik | 1 | 256 | 14229.8 | 10550.8 | 41.9% | NOISY |
| two_operand_einsum_f64_16x16_string | #2325 #2324 | string | tenferro-rs | f64 | m=16, k=16, n=16, format=ij,jk->ik | 1 | 1024 | 3480.1 | 1098.5 | 19.6% | NOISY |
| two_operand_einsum_f64_16x16_subscripts | #2325 #2324 | subscripts | tenferro-rs | f64 | m=16, k=16, n=16, format=ij,jk->ik | 1 | 1024 | 1869.6 | 544.0 | 26.7% | NOISY |
| two_operand_einsum_f64_16x16_prepared | #2325 #2324 | prepared | tenferro-rs | f64 | m=16, k=16, n=16, format=ij,jk->ik | 1 | 2048 | 1787.8 | 725.7 | 21.9% | NOISY |
| two_operand_einsum_f64_16x16_dot-general | #2325 #2324 | dot-general | tenferro-rs | f64 | m=16, k=16, n=16, format=ij,jk->ik | 1 | 2048 | 1678.0 | 557.3 | 24.2% | NOISY |
| two_operand_einsum_f64_16x16_pytorch | #2325 #2324 | pytorch | pytorch-cpu | f64 | m=16, k=16, n=16, format=ij,jk->ik | 1 | 256 | 13097.3 | 2546.1 | 17.2% | NOISY |
| two_operand_einsum_c64_2x2_string | #2325 #2324 | string | tenferro-rs | c64 | m=2, k=2, n=2, format=ij,jk->ik | 1 | 1024 | 2552.6 | 441.2 | 15.1% | NOISY |
| two_operand_einsum_c64_2x2_subscripts | #2325 #2324 | subscripts | tenferro-rs | c64 | m=2, k=2, n=2, format=ij,jk->ik | 1 | 2048 | 1765.2 | 276.4 | 15.2% | NOISY |
| two_operand_einsum_c64_2x2_prepared | #2325 #2324 | prepared | tenferro-rs | c64 | m=2, k=2, n=2, format=ij,jk->ik | 1 | 2048 | 1382.3 | 288.0 | 15.2% | NOISY |
| two_operand_einsum_c64_2x2_dot-general | #2325 #2324 | dot-general | tenferro-rs | c64 | m=2, k=2, n=2, format=ij,jk->ik | 1 | 1024 | 1519.4 | 264.5 | 12.6% | NOISY |
| two_operand_einsum_c64_2x2_pytorch | #2325 #2324 | pytorch | pytorch-cpu | c64 | m=2, k=2, n=2, format=ij,jk->ik | 1 | 256 | 11104.3 | 2420.7 | 18.1% | NOISY |
| two_operand_einsum_c64_16x16_string | #2325 #2324 | string | tenferro-rs | c64 | m=16, k=16, n=16, format=ij,jk->ik | 1 | 512 | 5734.8 | 713.3 | 14.0% | NOISY |
| two_operand_einsum_c64_16x16_subscripts | #2325 #2324 | subscripts | tenferro-rs | c64 | m=16, k=16, n=16, format=ij,jk->ik | 1 | 512 | 3277.2 | 1059.7 | 27.5% | NOISY |
| two_operand_einsum_c64_16x16_prepared | #2325 #2324 | prepared | tenferro-rs | c64 | m=16, k=16, n=16, format=ij,jk->ik | 1 | 512 | 3614.7 | 755.6 | 12.3% | NOISY |
| two_operand_einsum_c64_16x16_dot-general | #2325 #2324 | dot-general | tenferro-rs | c64 | m=16, k=16, n=16, format=ij,jk->ik | 1 | 512 | 3462.2 | 266.6 | 9.5% | ok |
| two_operand_einsum_c64_16x16_pytorch | #2325 #2324 | pytorch | pytorch-cpu | c64 | m=16, k=16, n=16, format=ij,jk->ik | 1 | 128 | 13024.1 | 4921.7 | 32.4% | NOISY |
| two_operand_einsum_f64_abc_cef_abef_string | #2325 #2324 | string | tenferro-rs | f64 | m=8, k=2, n=16, format=abc,cef->abef | 1 | 1024 | 3071.2 | 696.3 | 19.3% | NOISY |
| two_operand_einsum_f64_abc_cef_abef_subscripts | #2325 #2324 | subscripts | tenferro-rs | f64 | m=8, k=2, n=16, format=abc,cef->abef | 1 | 2048 | 1916.6 | 640.9 | 23.8% | NOISY |
| two_operand_einsum_f64_abc_cef_abef_prepared | #2325 #2324 | prepared | tenferro-rs | f64 | m=8, k=2, n=16, format=abc,cef->abef | 1 | 2048 | 1445.1 | 338.6 | 14.6% | NOISY |
| two_operand_einsum_f64_abc_cef_abef_dot-general | #2325 #2324 | dot-general | tenferro-rs | f64 | m=8, k=2, n=16, format=abc,cef->abef | 1 | 2048 | 1319.4 | 399.5 | 18.4% | NOISY |
| two_operand_einsum_f64_abc_cef_abef_pytorch | #2325 #2324 | pytorch | pytorch-cpu | f64 | m=8, k=2, n=16, format=abc,cef->abef | 1 | 256 | 11591.0 | 2888.6 | 16.9% | NOISY |
| thin_c64_gemm_1x141_141x282_t1 | #2416 #1099 | dot-general | tenferro-rs | c64 | m=1, k=141, n=282 | 1 | 64 | 36903.6 | 3286.5 | 8.4% | ok |
| thin_c64_gemm_1x141_141x282_t1_faer-direct | #2416 #1099 | faer-direct | faer-direct | c64 | m=1, k=141, n=282 | 1 | 128 | 35598.0 | 8628.9 | 25.7% | NOISY |
| thin_c64_gemm_1x141_141x282_t1_pytorch | #2416 #1099 | pytorch | pytorch-cpu | c64 | m=1, k=141, n=282 | 1 | 32 | 93070.3 | 38154.3 | 32.1% | NOISY |
| thin_c64_gemm_2x141_141x282_t1 | #2416 #1099 | dot-general | tenferro-rs | c64 | m=2, k=141, n=282 | 1 | 32 | 75957.0 | 18381.5 | 15.7% | NOISY |
| thin_c64_gemm_2x141_141x282_t1_faer-direct | #2416 #1099 | faer-direct | faer-direct | c64 | m=2, k=141, n=282 | 1 | 128 | 33639.3 | 6038.4 | 11.1% | NOISY |
| thin_c64_gemm_2x141_141x282_t1_pytorch | #2416 #1099 | pytorch | pytorch-cpu | c64 | m=2, k=141, n=282 | 1 | 32 | 69466.2 | 21576.8 | 29.1% | NOISY |
| thin_c64_gemm_8x141_141x282_t1 | #2416 #1099 | dot-general | tenferro-rs | c64 | m=8, k=141, n=282 | 1 | 16 | 149294.2 | 14631.5 | 7.3% | ok |
| thin_c64_gemm_8x141_141x282_t1_faer-direct | #2416 #1099 | faer-direct | faer-direct | c64 | m=8, k=141, n=282 | 1 | 16 | 145601.6 | 93561.2 | 29.8% | NOISY |
| thin_c64_gemm_8x141_141x282_t1_pytorch | #2416 #1099 | pytorch | pytorch-cpu | c64 | m=8, k=141, n=282 | 1 | 64 | 54796.2 | 16502.6 | 19.3% | NOISY |
| decode_proj_f32_eager-shared_in1024_out1024_len8 | #1992 #2003 #1995 #2229 | eager-shared | tenferro-rs | f32 | in=1024, out=1024, len=8 | 1 | 16 | 158549.5 | 37187.5 | 24.8% | NOISY |
| decode_proj_f32_faer-direct_in1024_out1024_len8 | #1992 #2003 #1995 #2229 | faer-direct | faer-direct | f32 | in=1024, out=1024, len=8 | 1 | 8 | 442697.9 | 270468.8 | 33.9% | NOISY |
| decode_proj_f32_host-sgemm_in1024_out1024_len8 | #1992 #2003 #1995 #2229 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=1024, len=8 | 1 | 8 | 434921.9 | 48812.4 | 7.4% | ok |
| decode_proj_f32_pytorch_in1024_out1024_len8 | #1992 #2003 #1995 #2229 | pytorch | pytorch-cpu | f32 | in=1024, out=1024, len=8 | 1 | 16 | 167692.7 | 53606.8 | 19.4% | NOISY |
| decode_proj_f32_eager-shared_in1024_out1024_len64 | #1992 #2003 #1995 #2229 | eager-shared | tenferro-rs | f32 | in=1024, out=1024, len=64 | 1 | 8 | 240197.9 | 71776.0 | 25.7% | NOISY |
| decode_proj_f32_faer-direct_in1024_out1024_len64 | #1992 #2003 #1995 #2229 | faer-direct | faer-direct | f32 | in=1024, out=1024, len=64 | 1 | 2 | 2056625.0 | 196708.5 | 9.6% | ok |
| decode_proj_f32_host-sgemm_in1024_out1024_len64 | #1992 #2003 #1995 #2229 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=1024, len=64 | 1 | 1 | 1936209.0 | 152625.0 | 9.2% | ok |
| decode_proj_f32_pytorch_in1024_out1024_len64 | #1992 #2003 #1995 #2229 | pytorch | pytorch-cpu | f32 | in=1024, out=1024, len=64 | 1 | 8 | 257234.4 | 40174.4 | 18.5% | NOISY |
| decode_proj_f32_eager-shared_in1024_out1024_len512 | #1992 #2003 #1995 #2229 | eager-shared | tenferro-rs | f32 | in=1024, out=1024, len=512 | 1 | 2 | 2403125.0 | 2657062.8 | 52.5% | NOISY |
| decode_proj_f32_faer-direct_in1024_out1024_len512 | #1992 #2003 #1995 #2229 | faer-direct | faer-direct | f32 | in=1024, out=1024, len=512 | 1 | 1 | 13811458.0 | 1612709.0 | 15.3% | NOISY |
| decode_proj_f32_host-sgemm_in1024_out1024_len512 | #1992 #2003 #1995 #2229 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=1024, len=512 | 1 | 1 | 16655834.0 | 3788041.5 | 16.9% | NOISY |
| decode_proj_f32_pytorch_in1024_out1024_len512 | #1992 #2003 #1995 #2229 | pytorch | pytorch-cpu | f32 | in=1024, out=1024, len=512 | 1 | 2 | 1781520.5 | 730593.8 | 32.6% | NOISY |
| decode_proj_f32_eager-shared_in1024_out4096_len8 | #1992 #2003 #1995 #2229 | eager-shared | tenferro-rs | f32 | in=1024, out=4096, len=8 | 1 | 2 | 2042458.0 | 1015864.8 | 35.6% | NOISY |
| decode_proj_f32_faer-direct_in1024_out4096_len8 | #1992 #2003 #1995 #2229 | faer-direct | faer-direct | f32 | in=1024, out=4096, len=8 | 1 | 1 | 2397417.0 | 460458.5 | 12.9% | NOISY |
| decode_proj_f32_host-sgemm_in1024_out4096_len8 | #1992 #2003 #1995 #2229 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=4096, len=8 | 1 | 1 | 3914875.0 | 1555812.5 | 27.3% | NOISY |
| decode_proj_f32_pytorch_in1024_out4096_len8 | #1992 #2003 #1995 #2229 | pytorch | pytorch-cpu | f32 | in=1024, out=4096, len=8 | 1 | 2 | 2302541.5 | 1350072.8 | 40.1% | NOISY |
| decode_proj_f32_eager-shared_in1024_out4096_len64 | #1992 #2003 #1995 #2229 | eager-shared | tenferro-rs | f32 | in=1024, out=4096, len=64 | 1 | 1 | 1923875.0 | 476396.5 | 15.6% | NOISY |
| decode_proj_f32_faer-direct_in1024_out4096_len64 | #1992 #2003 #1995 #2229 | faer-direct | faer-direct | f32 | in=1024, out=4096, len=64 | 1 | 1 | 9595917.0 | 1209854.0 | 8.3% | ok |
| decode_proj_f32_host-sgemm_in1024_out4096_len64 | #1992 #2003 #1995 #2229 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=4096, len=64 | 1 | 1 | 10465750.0 | 1356708.0 | 8.9% | ok |
| decode_proj_f32_pytorch_in1024_out4096_len64 | #1992 #2003 #1995 #2229 | pytorch | pytorch-cpu | f32 | in=1024, out=4096, len=64 | 1 | 2 | 2120833.5 | 776729.2 | 29.3% | NOISY |
| decode_proj_f32_eager-shared_in1024_out4096_len512 | #1992 #2003 #1995 #2229 | eager-shared | tenferro-rs | f32 | in=1024, out=4096, len=512 | 1 | 1 | 9585708.0 | 2876479.5 | 20.7% | NOISY |
| decode_proj_f32_faer-direct_in1024_out4096_len512 | #1992 #2003 #1995 #2229 | faer-direct | faer-direct | f32 | in=1024, out=4096, len=512 | 1 | 1 | 77376625.0 | 11772583.0 | 9.7% | ok |
| decode_proj_f32_host-sgemm_in1024_out4096_len512 | #1992 #2003 #1995 #2229 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=4096, len=512 | 1 | 1 | 64326875.0 | 7298687.5 | 10.1% | NOISY |
| decode_proj_f32_pytorch_in1024_out4096_len512 | #1992 #2003 #1995 #2229 | pytorch | pytorch-cpu | f32 | in=1024, out=4096, len=512 | 1 | 1 | 10717666.0 | 2705188.0 | 22.0% | NOISY |
| decode_proj_view_operand_f32_prepared_in1024_out1024_len8_t1 | #2216 | prepared-view | tenferro-rs | f32 | in=1024, out=1024, len=8 | 1 | 16 | 256630.2 | 102061.2 | 33.6% | NOISY |
| decode_proj_contiguous_operand_f32_prepared_in1024_out1024_len8_t1 | #2216 | prepared-contiguous | tenferro-rs | f32 | in=1024, out=1024, len=8 | 1 | 4 | 229708.2 | 731969.0 | 86.1% | NOISY |
| decode_proj_view_operand_f32_pytorch_in1024_out1024_len8_t1 | #2216 | pytorch | pytorch-cpu | f32 | in=1024, out=1024, len=8 | 1 | 8 | 216906.2 | 108974.0 | 31.7% | NOISY |
| gemm_c64_mm256_t1 | #1900 | dot-general-shared | tenferro-rs | c64 | n=256 | 1 | 1 | 1153000.0 | 362083.5 | 21.9% | NOISY |
| gemm_f64_mm256_t1 | #1900 | dot-general-shared | tenferro-rs | f64 | n=256 | 1 | 8 | 161255.2 | 27138.0 | 17.8% | NOISY |
| conj_dot_f64_dot-general-rank1_len100 | #1615 | dot-general-rank1 | tenferro-rs | f64 | len=100 | 1 | 2048 | 1277.3 | 455.6 | 27.0% | NOISY |
| conj_dot_f64_portable-loop_len100 | #1615 | portable-loop | rust-loop | f64 | len=100 | 1 | 65536 | 74.6 | 13.8 | 15.5% | NOISY |
| conj_dot_f64_pytorch-vdot_len100 | #1615 | pytorch-vdot | pytorch-cpu | f64 | len=100 | 1 | 2048 | 871.9 | 307.6 | 54.9% | NOISY |
| conj_dot_f64_dot-general-rank1_len10000 | #1615 | dot-general-rank1 | tenferro-rs | f64 | len=10000 | 1 | 128 | 18923.5 | 2625.8 | 13.3% | NOISY |
| conj_dot_f64_portable-loop_len10000 | #1615 | portable-loop | rust-loop | f64 | len=10000 | 1 | 256 | 11820.8 | 1615.1 | 9.4% | ok |
| conj_dot_f64_pytorch-vdot_len10000 | #1615 | pytorch-vdot | pytorch-cpu | f64 | len=10000 | 1 | 512 | 4301.5 | 1284.2 | 23.5% | NOISY |
| conj_dot_f64_dot-general-rank1_len1000000 | #1615 | dot-general-rank1 | tenferro-rs | f64 | len=1000000 | 1 | 2 | 1680291.5 | 187604.2 | 10.9% | NOISY |
| conj_dot_f64_portable-loop_len1000000 | #1615 | portable-loop | rust-loop | f64 | len=1000000 | 1 | 2 | 1131791.5 | 288177.0 | 15.5% | NOISY |
| conj_dot_f64_pytorch-vdot_len1000000 | #1615 | pytorch-vdot | pytorch-cpu | f64 | len=1000000 | 1 | 4 | 570583.2 | 287073.0 | 32.7% | NOISY |
| conj_dot_c64_dot-general-rank1_len100 | #1615 | dot-general-rank1 | tenferro-rs | c64 | len=100 | 1 | 1024 | 1895.1 | 821.5 | 27.1% | NOISY |
| conj_dot_c64_portable-loop_len100 | #1615 | portable-loop | rust-loop | c64 | len=100 | 1 | 16384 | 114.8 | 44.7 | 38.9% | NOISY |
| conj_dot_c64_pytorch-vdot_len100 | #1615 | pytorch-vdot | pytorch-cpu | c64 | len=100 | 1 | 4096 | 895.9 | 149.3 | 36.5% | NOISY |
| conj_dot_c64_dot-general-rank1_len10000 | #1615 | dot-general-rank1 | tenferro-rs | c64 | len=10000 | 1 | 64 | 79867.2 | 16545.6 | 14.0% | NOISY |
| conj_dot_c64_portable-loop_len10000 | #1615 | portable-loop | rust-loop | c64 | len=10000 | 1 | 128 | 12965.2 | 3406.7 | 37.6% | NOISY |
| conj_dot_c64_pytorch-vdot_len10000 | #1615 | pytorch-vdot | pytorch-cpu | c64 | len=10000 | 1 | 512 | 6816.8 | 3284.3 | 28.3% | NOISY |
| conj_dot_c64_dot-general-rank1_len1000000 | #1615 | dot-general-rank1 | tenferro-rs | c64 | len=1000000 | 1 | 1 | 7452666.0 | 1687980.0 | 17.3% | NOISY |
| conj_dot_c64_portable-loop_len1000000 | #1615 | portable-loop | rust-loop | c64 | len=1000000 | 1 | 2 | 1348666.5 | 342937.8 | 35.2% | NOISY |
| conj_dot_c64_pytorch-vdot_len1000000 | #1615 | pytorch-vdot | pytorch-cpu | c64 | len=1000000 | 1 | 2 | 1133791.5 | 136531.2 | 8.8% | ok |
| small_solve_f64_solve-loop_k16_b64 | #2007 #1878 | solve-loop | tenferro-rs | f64 | k=16, batch=64 | 1 | 4 | 217791.8 | 81729.1 | 47.6% | NOISY |
| small_solve_f64_triangular-solve-loop_k16_b64 | #2007 #1878 | triangular-solve-loop | tenferro-rs | f64 | k=16, batch=64 | 1 | 16 | 69182.3 | 28602.9 | 26.2% | NOISY |
| small_solve_f64_solve-batched_k16_b64 | #2007 #1878 | solve-batched | tenferro-rs | f64 | k=16, batch=64 | 1 | 8 | 167520.9 | 88343.8 | 32.0% | NOISY |
| small_solve_f64_pytorch-solve-batched_k16_b64 | #2007 #1878 | pytorch-solve-batched | pytorch-cpu | f64 | k=16, batch=64 | 1 | 16 | 176666.7 | 73651.1 | 34.1% | NOISY |
| small_solve_f64_pytorch-solve-triangular-batched_k16_b64 | #2007 #1878 | pytorch-solve-triangular-batched | pytorch-cpu | f64 | k=16, batch=64 | 1 | 32 | 33979.2 | 17077.5 | 45.4% | NOISY |
| small_solve_f64_solve-loop_k64_b64 | #2007 #1878 | solve-loop | tenferro-rs | f64 | k=64, batch=64 | 1 | 2 | 1884333.5 | 381885.5 | 21.9% | NOISY |
| small_solve_f64_triangular-solve-loop_k64_b64 | #2007 #1878 | triangular-solve-loop | tenferro-rs | f64 | k=64, batch=64 | 1 | 2 | 855937.5 | 642250.0 | 48.1% | NOISY |
| small_solve_f64_solve-batched_k64_b64 | #2007 #1878 | solve-batched | tenferro-rs | f64 | k=64, batch=64 | 1 | 2 | 2882895.5 | 1066979.0 | 21.4% | NOISY |
| small_solve_f64_pytorch-solve-batched_k64_b64 | #2007 #1878 | pytorch-solve-batched | pytorch-cpu | f64 | k=64, batch=64 | 1 | 1 | 3729042.0 | 2801583.0 | 37.6% | NOISY |
| small_solve_f64_pytorch-solve-triangular-batched_k64_b64 | #2007 #1878 | pytorch-solve-triangular-batched | pytorch-cpu | f64 | k=64, batch=64 | 1 | 2 | 1099770.5 | 887322.8 | 48.6% | NOISY |
| tanh_chain_f32_compiled-prepared_k10_1024x64 | #1990 | compiled-prepared | tenferro-rs | f32 | rows=1024, cols=64, chain=10 | 1 | 1 | 4809584.0 | 2903083.5 | 40.3% | NOISY |
| tanh_chain_f32_eager-shared_k10_1024x64 | #1990 | eager-shared | tenferro-rs | f32 | rows=1024, cols=64, chain=10 | 1 | 1 | 2704125.0 | 2760104.0 | 44.1% | NOISY |
| tanh_chain_f32_compiled-prepared_k100_1024x64 | #1990 | compiled-prepared | tenferro-rs | f32 | rows=1024, cols=64, chain=100 | 1 | 1 | 45124750.0 | 7236771.0 | 21.9% | NOISY |
| tanh_chain_f32_eager-shared_k100_1024x64 | #1990 | eager-shared | tenferro-rs | f32 | rows=1024, cols=64, chain=100 | 1 | 1 | 42847208.0 | 18545833.0 | 22.1% | NOISY |
| tanh_chain_f32_compiled-prepared_k200_1024x64 | #1990 | compiled-prepared | tenferro-rs | f32 | rows=1024, cols=64, chain=200 | 1 | 1 | 80861000.0 | 14642000.5 | 15.2% | NOISY |
| tanh_chain_f32_eager-shared_k200_1024x64 | #1990 | eager-shared | tenferro-rs | f32 | rows=1024, cols=64, chain=200 | 1 | 1 | 73104834.0 | 18737938.5 | 14.9% | NOISY |
| layer_norm_f32_eager-composed_d1024_len8_b1 | #2006 | eager-composed | tenferro-rs | f32 | norm=layer_norm, d=1024, len=8, batch=1 | 1 | 64 | 61628.3 | 32884.4 | 37.6% | NOISY |
| layer_norm_f32_pytorch-fused_d1024_len8_b1 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=layer_norm, d=1024, len=8, batch=1 | 1 | 64 | 6421.2 | 4357.8 | 50.9% | NOISY |
| layer_norm_f32_eager-single-call_d1024_len8_b1 | #2006 | eager-single-call | tenferro-rs | f32 | norm=layer_norm, d=1024, len=8, batch=1 | 1 | 32 | 71253.9 | 18669.3 | 26.9% | NOISY |
| layer_norm_f32_eager-composed_d1024_len8_b8 | #2006 | eager-composed | tenferro-rs | f32 | norm=layer_norm, d=1024, len=8, batch=8 | 1 | 16 | 130401.1 | 47477.9 | 33.7% | NOISY |
| layer_norm_f32_pytorch-fused_d1024_len8_b8 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=layer_norm, d=1024, len=8, batch=8 | 1 | 32 | 93093.8 | 28378.3 | 30.1% | NOISY |
| layer_norm_f32_eager-single-call_d1024_len8_b8 | #2006 | eager-single-call | tenferro-rs | f32 | norm=layer_norm, d=1024, len=8, batch=8 | 1 | 8 | 137031.2 | 111289.1 | 47.3% | NOISY |
| layer_norm_f32_eager-composed_d1024_len64_b1 | #2006 | eager-composed | tenferro-rs | f32 | norm=layer_norm, d=1024, len=64, batch=1 | 1 | 8 | 117026.0 | 43817.8 | 28.9% | NOISY |
| layer_norm_f32_pytorch-fused_d1024_len64_b1 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=layer_norm, d=1024, len=64, batch=1 | 1 | 64 | 37913.4 | 11333.3 | 31.1% | NOISY |
| layer_norm_f32_eager-single-call_d1024_len64_b1 | #2006 | eager-single-call | tenferro-rs | f32 | norm=layer_norm, d=1024, len=64, batch=1 | 1 | 16 | 175976.6 | 152466.2 | 41.3% | NOISY |
| layer_norm_f32_eager-composed_d1024_len64_b8 | #2006 | eager-composed | tenferro-rs | f32 | norm=layer_norm, d=1024, len=64, batch=8 | 1 | 2 | 843625.0 | 695667.0 | 44.5% | NOISY |
| layer_norm_f32_pytorch-fused_d1024_len64_b8 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=layer_norm, d=1024, len=64, batch=8 | 1 | 8 | 275229.1 | 173885.5 | 49.3% | NOISY |
| layer_norm_f32_eager-single-call_d1024_len64_b8 | #2006 | eager-single-call | tenferro-rs | f32 | norm=layer_norm, d=1024, len=64, batch=8 | 1 | 1 | 2387625.0 | 2309896.0 | 73.9% | NOISY |
| rms_norm_f32_eager-composed_d1024_len8_b1 | #2006 | eager-composed | tenferro-rs | f32 | norm=rms_norm, d=1024, len=8, batch=1 | 1 | 64 | 21904.3 | 13016.6 | 54.5% | NOISY |
| rms_norm_f32_pytorch-fused_d1024_len8_b1 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=rms_norm, d=1024, len=8, batch=1 | 1 | 128 | 16779.6 | 11790.9 | 40.4% | NOISY |
| rms_norm_f32_eager-single-call_d1024_len8_b1 | #2006 | eager-single-call | tenferro-rs | f32 | norm=rms_norm, d=1024, len=8, batch=1 | 1 | 16 | 42645.8 | 2731.8 | 20.9% | NOISY |
| rms_norm_f32_eager-composed_d1024_len8_b8 | #2006 | eager-composed | tenferro-rs | f32 | norm=rms_norm, d=1024, len=8, batch=8 | 1 | 4 | 92385.2 | 47317.8 | 69.5% | NOISY |
| rms_norm_f32_pytorch-fused_d1024_len8_b8 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=rms_norm, d=1024, len=8, batch=8 | 1 | 16 | 40203.1 | 17378.9 | 54.7% | NOISY |
| rms_norm_f32_eager-single-call_d1024_len8_b8 | #2006 | eager-single-call | tenferro-rs | f32 | norm=rms_norm, d=1024, len=8, batch=8 | 1 | 4 | 92531.2 | 90088.6 | 67.5% | NOISY |
| rms_norm_f32_eager-composed_d1024_len64_b1 | #2006 | eager-composed | tenferro-rs | f32 | norm=rms_norm, d=1024, len=64, batch=1 | 1 | 16 | 48164.1 | 19781.3 | 46.9% | NOISY |
| rms_norm_f32_pytorch-fused_d1024_len64_b1 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=rms_norm, d=1024, len=64, batch=1 | 1 | 64 | 41561.2 | 19905.6 | 36.8% | NOISY |
| rms_norm_f32_eager-single-call_d1024_len64_b1 | #2006 | eager-single-call | tenferro-rs | f32 | norm=rms_norm, d=1024, len=64, batch=1 | 1 | 32 | 48674.5 | 10722.0 | 21.3% | NOISY |
| rms_norm_f32_eager-composed_d1024_len64_b8 | #2006 | eager-composed | tenferro-rs | f32 | norm=rms_norm, d=1024, len=64, batch=8 | 1 | 4 | 274760.5 | 99479.1 | 35.4% | NOISY |
| rms_norm_f32_pytorch-fused_d1024_len64_b8 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=rms_norm, d=1024, len=64, batch=8 | 1 | 4 | 222020.8 | 43463.4 | 22.8% | NOISY |
| rms_norm_f32_eager-single-call_d1024_len64_b8 | #2006 | eager-single-call | tenferro-rs | f32 | norm=rms_norm, d=1024, len=64, batch=8 | 1 | 4 | 376500.0 | 69354.1 | 36.3% | NOISY |
| small_contraction_abcd-dbef-acef_f64_shared-session-default-pool_d4 | #1885 #1897 #1901 | shared-session-default-pool | tenferro-rs | f64 | extent=4 | 1 | 512 | 2494.1 | 71.0 | 7.3% | ok |
| activation_erf_f32_eager-single-call_1024x64 | #1975 #2069 #2032 | eager-single-call | tenferro-rs | f32 | op=erf, rows=1024, cols=64 | 1 | — | — | — | — | SKIPPED |
| activation_erf_f32_session-single-call_1024x64 | #1975 #2069 #2032 | session-single-call | tenferro-rs | f32 | op=erf, rows=1024, cols=64 | 1 | 4 | 504510.5 | 33739.4 | 6.6% | ok |
| activation_erf_f32_pytorch_1024x64 | #1975 #2069 #2032 | pytorch | pytorch-cpu | f32 | op=erf, rows=1024, cols=64 | 1 | — | — | — | — | SKIPPED |
| activation_sigmoid_f32_eager-single-call_1024x64 | #1975 #2069 #2032 | eager-single-call | tenferro-rs | f32 | op=sigmoid, rows=1024, cols=64 | 1 | — | — | — | — | SKIPPED |
| activation_sigmoid_f32_session-single-call_1024x64 | #1975 #2069 #2032 | session-single-call | tenferro-rs | f32 | op=sigmoid, rows=1024, cols=64 | 1 | 32 | 55951.8 | 12354.2 | 23.4% | NOISY |
| activation_sigmoid_f32_eager-composed_1024x64 | #1975 #2069 #2032 | eager-composed | tenferro-rs | f32 | op=sigmoid, rows=1024, cols=64 | 1 | 4 | 78948.0 | 9896.0 | 9.1% | ok |
| activation_sigmoid_f32_pytorch_1024x64 | #1975 #2069 #2032 | pytorch | pytorch-cpu | f32 | op=sigmoid, rows=1024, cols=64 | 1 | — | — | — | — | SKIPPED |
| activation_silu_f32_eager-single-call_1024x64 | #1975 #2069 #2032 | eager-single-call | tenferro-rs | f32 | op=silu, rows=1024, cols=64 | 1 | — | — | — | — | SKIPPED |
| activation_silu_f32_session-single-call_1024x64 | #1975 #2069 #2032 | session-single-call | tenferro-rs | f32 | op=silu, rows=1024, cols=64 | 1 | 16 | 57843.8 | 9013.0 | 20.8% | NOISY |
| activation_silu_f32_eager-composed_1024x64 | #1975 #2069 #2032 | eager-composed | tenferro-rs | f32 | op=silu, rows=1024, cols=64 | 1 | 8 | 86239.6 | 6203.2 | 18.1% | NOISY |
| activation_silu_f32_pytorch_1024x64 | #1975 #2069 #2032 | pytorch | pytorch-cpu | f32 | op=silu, rows=1024, cols=64 | 1 | — | — | — | — | SKIPPED |
| activation_softplus_f32_eager-single-call_1024x64 | #1975 #2069 #2032 | eager-single-call | tenferro-rs | f32 | op=softplus, rows=1024, cols=64 | 1 | — | — | — | — | SKIPPED |
| activation_softplus_f32_session-single-call_1024x64 | #1975 #2069 #2032 | session-single-call | tenferro-rs | f32 | op=softplus, rows=1024, cols=64 | 1 | 8 | 257020.9 | 68849.0 | 36.5% | NOISY |
| activation_softplus_f32_eager-composed_1024x64 | #1975 #2069 #2032 | eager-composed | tenferro-rs | f32 | op=softplus, rows=1024, cols=64 | 1 | 8 | 219046.9 | 75098.9 | 29.9% | NOISY |
| activation_softplus_f32_pytorch_1024x64 | #1975 #2069 #2032 | pytorch | pytorch-cpu | f32 | op=softplus, rows=1024, cols=64 | 1 | — | — | — | — | SKIPPED |
| activation_gelu_f32_eager-single-call_1024x64 | #1975 #2030 #2069 #2228 | eager-single-call | tenferro-rs | f32 | op=gelu, rows=1024, cols=64 | 1 | 4 | 95083.5 | 17343.8 | 11.7% | NOISY |
| activation_gelu_f32_session-single-call_1024x64 | #1975 #2030 #2069 #2228 | session-single-call | tenferro-rs | f32 | op=gelu, rows=1024, cols=64 | 1 | 8 | 91036.5 | 37401.0 | 55.1% | NOISY |
| activation_gelu_f32_pytorch_1024x64 | #1975 #2030 #2069 #2228 | pytorch | pytorch-cpu | f32 | op=gelu, rows=1024, cols=64 | 1 | 32 | 103235.7 | 20953.8 | 17.3% | NOISY |
| activation_gelu_tanh_f32_eager-single-call_1024x64 | #1975 #2069 #2032 | eager-single-call | tenferro-rs | f32 | op=gelu_tanh, rows=1024, cols=64 | 1 | — | — | — | — | SKIPPED |
| activation_gelu_tanh_f32_session-single-call_1024x64 | #1975 #2069 #2032 | session-single-call | tenferro-rs | f32 | op=gelu_tanh, rows=1024, cols=64 | 1 | 8 | 396291.8 | 179635.4 | 30.1% | NOISY |
| activation_gelu_tanh_f32_eager-composed_1024x64 | #1975 #2069 #2032 | eager-composed | tenferro-rs | f32 | op=gelu_tanh, rows=1024, cols=64 | 1 | 4 | 397239.5 | 102427.0 | 22.4% | NOISY |
| activation_gelu_tanh_f32_pytorch_1024x64 | #1975 #2069 #2032 | pytorch | pytorch-cpu | f32 | op=gelu_tanh, rows=1024, cols=64 | 1 | — | — | — | — | SKIPPED |
| softmax_f32_eager-single-call_len64_b8 | #1976 #2031 #2231 | eager-single-call | tenferro-rs | f32 | op=softmax, len=64, batch=8, axis=0 | 1 | 32 | 39570.3 | 6295.6 | 13.8% | NOISY |
| softmax_f32_session-single-call_len64_b8 | #1976 #2031 #2231 | session-single-call | tenferro-rs | f32 | op=softmax, len=64, batch=8, axis=0 | 1 | 32 | 36773.4 | 1576.2 | 28.4% | NOISY |
| softmax_f32_eager-composed_len64_b8 | #1976 #2031 #2231 | eager-composed | tenferro-rs | f32 | op=softmax, len=64, batch=8, axis=0 | 1 | 32 | 89121.1 | 27246.8 | 16.4% | NOISY |
| softmax_f32_pytorch_len64_b8 | #1976 #2031 #2231 | pytorch | pytorch-cpu | f32 | op=softmax, len=64, batch=8, axis=0 | 1 | 32 | 54809.9 | 8382.2 | 22.0% | NOISY |
| softmax_f32_eager-single-call_len512_b8 | #1976 #2231 | eager-single-call | tenferro-rs | f32 | op=softmax, len=512, batch=8, axis=0 | 1 | 1 | 2796792.0 | 198416.0 | 8.4% | ok |
| softmax_f32_session-single-call_len512_b8 | #1976 #2231 | session-single-call | tenferro-rs | f32 | op=softmax, len=512, batch=8, axis=0 | 1 | 1 | 2901708.0 | 165020.5 | 21.8% | NOISY |
| softmax_f32_eager-composed_len512_b8 | #1976 #2231 | eager-composed | tenferro-rs | f32 | op=softmax, len=512, batch=8, axis=0 | 1 | 1 | 3587959.0 | 863770.5 | 13.2% | NOISY |
| softmax_f32_pytorch_len512_b8 | #1976 #2231 | pytorch | pytorch-cpu | f32 | op=softmax, len=512, batch=8, axis=0 | 1 | 1 | 6625834.0 | 1878687.0 | 25.5% | NOISY |
| log_softmax_f32_eager-single-call_len64_b8 | #1976 | eager-single-call | tenferro-rs | f32 | op=log_softmax, len=64, batch=8, axis=0 | 1 | 8 | 175302.0 | 143622.3 | 40.6% | NOISY |
| log_softmax_f32_session-single-call_len64_b8 | #1976 | session-single-call | tenferro-rs | f32 | op=log_softmax, len=64, batch=8, axis=0 | 1 | 16 | 182645.8 | 122867.2 | 43.1% | NOISY |
| log_softmax_f32_eager-composed_len64_b8 | #1976 | eager-composed | tenferro-rs | f32 | op=log_softmax, len=64, batch=8, axis=0 | 1 | 32 | 98015.6 | 45992.2 | 33.0% | NOISY |
| log_softmax_f32_pytorch_len64_b8 | #1976 | pytorch | pytorch-cpu | f32 | op=log_softmax, len=64, batch=8, axis=0 | 1 | 32 | 83832.0 | 53939.5 | 43.9% | NOISY |
| log_softmax_f32_eager-single-call_len512_b8 | #1976 | eager-single-call | tenferro-rs | f32 | op=log_softmax, len=512, batch=8, axis=0 | 1 | 1 | 3638625.0 | 1154875.0 | 21.7% | NOISY |
| log_softmax_f32_session-single-call_len512_b8 | #1976 | session-single-call | tenferro-rs | f32 | op=log_softmax, len=512, batch=8, axis=0 | 1 | 1 | 3429292.0 | 1726625.5 | 22.5% | NOISY |
| log_softmax_f32_eager-composed_len512_b8 | #1976 | eager-composed | tenferro-rs | f32 | op=log_softmax, len=512, batch=8, axis=0 | 1 | 1 | 4311666.0 | 1200708.0 | 34.2% | NOISY |
| log_softmax_f32_pytorch_len512_b8 | #1976 | pytorch | pytorch-cpu | f32 | op=log_softmax, len=512, batch=8, axis=0 | 1 | 1 | 4575708.0 | 1605312.5 | 31.5% | NOISY |
| masked_softmax_f32_eager-single-call_len64_b8 | #1976 | eager-single-call | tenferro-rs | f32 | op=masked_softmax, len=64, batch=8, axis=0 | 1 | 4 | 212083.5 | 214442.9 | 45.0% | NOISY |
| masked_softmax_f32_session-single-call_len64_b8 | #1976 | session-single-call | tenferro-rs | f32 | op=masked_softmax, len=64, batch=8, axis=0 | 1 | 8 | 103573.0 | 21156.2 | 32.0% | NOISY |
| masked_softmax_f32_eager-composed_len64_b8 | #1976 | eager-composed | tenferro-rs | f32 | op=masked_softmax, len=64, batch=8, axis=0 | 1 | 16 | 129210.9 | 58673.2 | 30.9% | NOISY |
| masked_softmax_f32_pytorch_len64_b8 | #1976 | pytorch | pytorch-cpu | f32 | op=masked_softmax, len=64, batch=8, axis=0 | 1 | 16 | 96312.5 | 34162.8 | 36.8% | NOISY |
| masked_softmax_f32_eager-single-call_len512_b8 | #1976 | eager-single-call | tenferro-rs | f32 | op=masked_softmax, len=512, batch=8, axis=0 | 1 | 1 | 6907542.0 | 3709396.0 | 28.0% | NOISY |
| masked_softmax_f32_session-single-call_len512_b8 | #1976 | session-single-call | tenferro-rs | f32 | op=masked_softmax, len=512, batch=8, axis=0 | 1 | 1 | 5152917.0 | 1209374.5 | 34.1% | NOISY |
| masked_softmax_f32_eager-composed_len512_b8 | #1976 | eager-composed | tenferro-rs | f32 | op=masked_softmax, len=512, batch=8, axis=0 | 1 | 1 | 7339166.0 | 3182395.5 | 27.5% | NOISY |
| masked_softmax_f32_pytorch_len512_b8 | #1976 | pytorch | pytorch-cpu | f32 | op=masked_softmax, len=512, batch=8, axis=0 | 1 | 1 | 8834625.0 | 2440292.0 | 25.7% | NOISY |
| softmax_f32_eager-single-call_len64_b8_t1 | #2233 #2235 | eager-single-call | tenferro-rs | f32 | op=softmax, len=64, batch=8, axis=0 | 1 | 8 | 39432.2 | 39265.6 | 53.7% | NOISY |
| softmax_f32_eager-composed_len64_b8_t1 | #2233 #2235 | eager-composed | tenferro-rs | f32 | op=softmax, len=64, batch=8, axis=0 | 1 | 4 | 245687.5 | 74411.5 | 38.1% | NOISY |
| softmax_f32_pytorch_len64_b8_t1 | #2233 #2235 | pytorch | pytorch-cpu | f32 | op=softmax, len=64, batch=8, axis=0 | 1 | 16 | 107708.3 | 62062.5 | 49.4% | NOISY |
| reduce_mean_f32_eager-single-call_1024x64_axis0 | #1976 | eager-single-call | tenferro-rs | f32 | rows=1024, cols=64, axis=0 | 1 | 64 | 13175.8 | 6632.8 | 58.8% | NOISY |
| reduce_mean_f32_session-single-call_1024x64_axis0 | #1976 | session-single-call | tenferro-rs | f32 | rows=1024, cols=64, axis=0 | 1 | 128 | 11069.7 | 14592.0 | 53.2% | NOISY |
| reduce_mean_f32_eager-composed_1024x64_axis0 | #1976 | eager-composed | tenferro-rs | f32 | rows=1024, cols=64, axis=0 | 1 | 128 | 17679.4 | 7367.2 | 29.5% | NOISY |
| reduce_mean_f32_pytorch_1024x64_axis0 | #1976 | pytorch | pytorch-cpu | f32 | rows=1024, cols=64, axis=0 | 1 | 256 | 15714.4 | 9113.5 | 38.0% | NOISY |
| take_along_axis_rows_f64_eager-single-call_n64_b256 | #2008 | eager-single-call | tenferro-rs | f64 | n=64, batch=256, axis=0 | 1 | 1 | 10774500.0 | 5988292.0 | 28.7% | NOISY |
| take_along_axis_rows_f64_session-single-call_n64_b256 | #2008 | session-single-call | tenferro-rs | f64 | n=64, batch=256, axis=0 | 1 | 1 | 8554500.0 | 2040291.0 | 15.3% | NOISY |
| take_along_axis_rows_f64_eager-gather-prebuilt_n64_b256 | #2008 | eager-gather-prebuilt | tenferro-rs | f64 | n=64, batch=256, axis=0 | 1 | 1 | 10211250.0 | 2243833.5 | 13.7% | NOISY |
| take_along_axis_rows_f64_pytorch_n64_b256 | #2008 | pytorch | pytorch-cpu | f64 | n=64, batch=256, axis=0 | 1 | 1 | 2834333.0 | 1774959.0 | 33.4% | NOISY |
| lstsq_f64_eager-shared_m768_n384_rhs16 | #2027 | eager-shared | tenferro-rs | f64 | operation=lstsq | 1 | 1 | 11997958.0 | 2982355.0 | 18.3% | NOISY |
| lstsq_f64_pytorch_m768_n384_rhs16 | #2027 | pytorch | pytorch-cpu | f64 | operation=lstsq | 1 | 1 | 8033375.0 | 1341333.0 | 14.3% | NOISY |
| triangular_f64_concrete-shared_n4096_rhs64 | #2028 | concrete-shared | tenferro-rs | f64 | operation=triangular | 1 | 1 | 24609333.0 | 4258125.5 | 14.3% | NOISY |
| triangular_f64_pytorch_n4096_rhs64 | #2028 | pytorch | pytorch-cpu | f64 | operation=triangular | 1 | 1 | 23366625.0 | 7952396.0 | 21.1% | NOISY |
| reshape_f64_concrete-shared_n33554432_to8192x4096 | #2029 | concrete-shared | tenferro-rs | f64 | operation=reshape | 1 | 1 | 18410041.0 | 4540333.5 | 19.4% | NOISY |
| reshape_f64_pytorch_n33554432_to8192x4096 | #2029 | pytorch | pytorch-cpu | f64 | operation=reshape | 1 | 1 | 15079583.0 | 1716188.0 | 24.3% | NOISY |
| metadata_transpose_view_f64_rank8_t1 | #2180 | metadata-clone | tenferro-rs | f64 | shape=[2, 2, 2, 2, 2, 2, 2, 2] | 1 | 16384 | 69.2 | 9.2 | 26.0% | NOISY |
| metadata_transpose_view_f64_rank8_t1_julia-base | #2180 | julia-base | julia-base | f64 | shape=[2, 2, 2, 2, 2, 2, 2, 2] | 1 | 1048576 | 0.6 | 0.3 | 31.3% | NOISY |
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
| typed_elemwise_f64_add_held_len16 | #2139 | tenferro | tenferro-rs | f64 | n=16, route=held | 1 | 2048 | 1558.3 | 606.1 | 27.5% | NOISY |
| typed_elemwise_f64_add_held_len16_pytorch | #2139 | pytorch | pytorch-cpu | f64 | n=16, route=held | 1 | 1024 | 1297.4 | 978.2 | 53.6% | NOISY |
| typed_elemwise_f64_add_into_held_len16 | #2139 | tenferro | tenferro-rs | f64 | n=16, route=into_held | 1 | 2048 | 985.0 | 631.2 | 35.2% | NOISY |
| typed_elemwise_f64_add_into_held_len16_pytorch | #2139 | pytorch | pytorch-cpu | f64 | n=16, route=into_held | 1 | 2048 | 1561.1 | 838.0 | 35.9% | NOISY |
| typed_elemwise_f64_add_held_len256 | #2139 | tenferro | tenferro-rs | f64 | n=256, route=held | 1 | 2048 | 820.9 | 331.6 | 37.0% | NOISY |
| typed_elemwise_f64_add_held_len256_pytorch | #2139 | pytorch | pytorch-cpu | f64 | n=256, route=held | 1 | 2048 | 881.9 | 417.6 | 101.0% | NOISY |
| typed_elemwise_f64_add_into_held_len256 | #2139 | tenferro | tenferro-rs | f64 | n=256, route=into_held | 1 | 4096 | 1011.2 | 484.2 | 30.6% | NOISY |
| typed_elemwise_f64_add_into_held_len256_pytorch | #2139 | pytorch | pytorch-cpu | f64 | n=256, route=into_held | 1 | 4096 | 1242.1 | 396.7 | 35.2% | NOISY |
| typed_elemwise_f64_add_held_len4096 | #2139 | tenferro | tenferro-rs | f64 | n=4096, route=held | 1 | 512 | 3630.1 | 1912.0 | 33.6% | NOISY |
| typed_elemwise_f64_add_held_len4096_pytorch | #2139 | pytorch | pytorch-cpu | f64 | n=4096, route=held | 1 | 256 | 1633.0 | 1429.8 | 65.6% | NOISY |
| typed_elemwise_f64_add_into_held_len4096 | #2139 | tenferro | tenferro-rs | f64 | n=4096, route=into_held | 1 | 1024 | 2591.5 | 1884.2 | 46.0% | NOISY |
| typed_elemwise_f64_add_into_held_len4096_pytorch | #2139 | pytorch | pytorch-cpu | f64 | n=4096, route=into_held | 1 | 2048 | 2573.2 | 1904.4 | 45.8% | NOISY |
| tiny_matmul_2x2_typed_held | #2139 | tenferro | tenferro-rs | f64 | n=2 | 1 | 4096 | 989.0 | 506.2 | 29.9% | NOISY |
| tiny_matmul_2x2_typed_held_pytorch | #2139 | pytorch | pytorch-cpu | f64 | n=2 | 1 | 4096 | 1073.9 | 534.1 | 282.6% | NOISY |
| analytic_neg_f64_sweep_8MiB_t1 | #2139 | tenferro | tenferro-rs | f64 | n=1048576 | 1 | 2 | 280146.0 | 28791.8 | 18.2% | NOISY |
| analytic_neg_f64_sweep_8MiB_t1_pytorch | #2139 | pytorch | pytorch-cpu | f64 | n=1048576 | 1 | 8 | 399432.2 | 201505.2 | 41.8% | NOISY |
| analytic_neg_f64_sweep_32MiB_t1 | #2139 | tenferro | tenferro-rs | f64 | n=4194304 | 1 | 1 | 2752792.0 | 1910354.5 | 36.2% | NOISY |
| analytic_neg_f64_sweep_32MiB_t1_pytorch | #2139 | pytorch | pytorch-cpu | f64 | n=4194304 | 1 | 1 | 1822459.0 | 403667.0 | 16.1% | NOISY |
| gemm_bound_f64_t1_t8_scaling | #2141 | tenferro | tenferro-rs | f64 | n=1536 | 1 | 1 | 55804500.0 | 10996687.5 | 12.4% | NOISY |
| gemm_bound_f64_t1_t8_scaling_pytorch | #2141 | pytorch | pytorch-cpu | f64 | n=1536 | 1 | 1 | 54541709.0 | 6339208.0 | 12.1% | NOISY |
| lu_batched_f64_factor_n8_b1024 | #2142 | tenferro | tenferro-rs | f64 | n=8, batch=1024, op=factor | 1 | 8 | 362578.1 | 358906.2 | 39.4% | NOISY |
| lu_batched_f64_factor_n8_b1024_host | #2142 | host | rust-loop | f64 | n=8, batch=1024, op=factor | 1 | 8 | 276937.5 | 400143.3 | 52.1% | NOISY |
| lu_batched_f64_solve_n8_b1024 | #2142 | tenferro | tenferro-rs | f64 | n=8, batch=1024, op=solve | 1 | 8 | 115401.1 | 28312.6 | 46.6% | NOISY |
| lu_batched_f64_solve_n8_b1024_pytorch | #2142 | pytorch | pytorch-cpu | f64 | n=8, batch=1024, op=solve | 1 | 32 | 96231.8 | 38118.5 | 57.6% | NOISY |
| lu_batched_f64_factor_n16_b1024 | #2142 | tenferro | tenferro-rs | f64 | n=16, batch=1024, op=factor | 1 | 2 | 868875.0 | 283406.2 | 41.7% | NOISY |
| lu_batched_f64_factor_n16_b1024_host | #2142 | host | rust-loop | f64 | n=16, batch=1024, op=factor | 1 | 2 | 1206458.5 | 121062.8 | 26.8% | NOISY |
| lu_batched_f64_solve_n16_b1024 | #2142 | tenferro | tenferro-rs | f64 | n=16, batch=1024, op=solve | 1 | 16 | 269755.2 | 72147.2 | 29.8% | NOISY |
| lu_batched_f64_solve_n16_b1024_pytorch | #2142 | pytorch | pytorch-cpu | f64 | n=16, batch=1024, op=solve | 1 | 16 | 220369.8 | 27670.6 | 21.9% | NOISY |
| axpby_accum_strided_dest_f64_block | #2143 | tenferro | tenferro-rs | f64 | n=64 | 1 | 512 | 5797.4 | 1000.3 | 33.8% | NOISY |
| axpby_accum_strided_dest_f64_block_host | #2143 | host | rust-loop | f64 | n=64 | 1 | 512 | 4862.0 | 468.8 | 32.0% | NOISY |
| small_grad_qr_f64_trace-prepared_n4 | #2186 | trace-prepared | tenferro-rs | f64 | op=qr, n=4 | 1 | 128 | 32014.0 | 3829.4 | 15.6% | NOISY |
| small_grad_qr_f64_pytorch_n4 | #2186 | pytorch | pytorch-cpu | f64 | op=qr, n=4 | 1 | 64 | 61966.8 | 27466.1 | 38.3% | NOISY |
| small_grad_qr_f64_trace-prepared_n16 | #2186 | trace-prepared | tenferro-rs | f64 | op=qr, n=16 | 1 | 64 | 42936.8 | 4238.0 | 7.1% | ok |
| small_grad_qr_f64_pytorch_n16 | #2186 | pytorch | pytorch-cpu | f64 | op=qr, n=16 | 1 | 64 | 56357.4 | 17453.8 | 25.6% | NOISY |
| small_grad_eigh_f64_trace-prepared_n4 | #2186 | trace-prepared | tenferro-rs | f64 | op=eigh, n=4 | 1 | 64 | 35379.6 | 1831.4 | 44.6% | NOISY |
| small_grad_eigh_f64_pytorch_n4 | #2186 | pytorch | pytorch-cpu | f64 | op=eigh, n=4 | 1 | 64 | 44611.3 | 2918.9 | 48.8% | NOISY |
| small_grad_eigh_f64_trace-prepared_n16 | #2186 | trace-prepared | tenferro-rs | f64 | op=eigh, n=16 | 1 | 32 | 76168.0 | 2662.8 | 8.8% | ok |
| small_grad_eigh_f64_pytorch_n16 | #2186 | pytorch | pytorch-cpu | f64 | op=eigh, n=16 | 1 | 16 | 70286.4 | 8597.6 | 34.4% | NOISY |
| small_grad_lu_f64_trace-prepared_n4 | #2186 | trace-prepared | tenferro-rs | f64 | op=lu, n=4 | 1 | 128 | 18745.8 | 4412.4 | 41.8% | NOISY |
| small_grad_lu_f64_pytorch_n4 | #2186 | pytorch | pytorch-cpu | f64 | op=lu, n=4 | 1 | 64 | 51229.8 | 22079.8 | 29.6% | NOISY |
| small_grad_lu_f64_trace-prepared_n16 | #2186 | trace-prepared | tenferro-rs | f64 | op=lu, n=16 | 1 | 128 | 25930.3 | 2685.2 | 28.0% | NOISY |
| small_grad_lu_f64_pytorch_n16 | #2186 | pytorch | pytorch-cpu | f64 | op=lu, n=16 | 1 | 16 | 57989.6 | 12988.3 | 59.9% | NOISY |
| small_grad_svd_f64_trace-prepared_n4 | #2186 | trace-prepared | tenferro-rs | f64 | op=svd, n=4 | 1 | 32 | 61544.3 | 23756.5 | 30.0% | NOISY |
| small_grad_svd_f64_pytorch_n4 | #2186 | pytorch | pytorch-cpu | f64 | op=svd, n=4 | 1 | 32 | 66153.7 | 48886.7 | 61.0% | NOISY |
| small_grad_svd_f64_trace-prepared_n16 | #2186 | trace-prepared | tenferro-rs | f64 | op=svd, n=16 | 1 | 16 | 143791.7 | 24114.6 | 36.6% | NOISY |
| small_grad_svd_f64_pytorch_n16 | #2186 | pytorch | pytorch-cpu | f64 | op=svd, n=16 | 1 | 32 | 126091.1 | 35978.5 | 35.9% | NOISY |
| small_grad_solve_f64_trace-prepared_n4 | #2186 | trace-prepared | tenferro-rs | f64 | op=solve, n=4 | 1 | 256 | 14504.2 | 648.9 | 35.8% | NOISY |
| small_grad_solve_f64_pytorch_n4 | #2186 | pytorch | pytorch-cpu | f64 | op=solve, n=4 | 1 | 64 | 37500.6 | 7806.6 | 46.6% | NOISY |
| small_grad_solve_f64_trace-prepared_n16 | #2186 | trace-prepared | tenferro-rs | f64 | op=solve, n=16 | 1 | 128 | 20834.3 | 1918.9 | 27.0% | NOISY |
| small_grad_solve_f64_pytorch_n16 | #2186 | pytorch | pytorch-cpu | f64 | op=solve, n=16 | 1 | 64 | 43640.6 | 555.3 | 31.5% | NOISY |
| fft_axis0_c64_fft-executor_n128 | #1765 | fft-executor | tenferro-rs | c64 | n=128 | 1 | 1 | 8248291.0 | 781896.0 | 11.4% | NOISY |
| fft_axis0_c64_pytorch_n128 | #1765 | pytorch | pytorch-cpu | c64 | n=128 | 1 | 1 | 21102291.0 | 2091021.0 | 7.0% | ok |
| degenerate_dot_hadamard_f64_dot-general-shared_n1024 | #2111 | dot-general-shared | tenferro-rs | f64 | form=hadamard, n=1024 | 1 | 2 | 609666.5 | 86510.2 | 10.7% | NOISY |
| degenerate_dot_hadamard_f64_pytorch_n1024 | #2111 | pytorch | pytorch-cpu | f64 | form=hadamard, n=1024 | 1 | 2 | 526666.5 | 28729.2 | 5.5% | ok |
| degenerate_dot_outer_f64_dot-general-shared_n1024 | #2111 | dot-general-shared | tenferro-rs | f64 | form=outer, n=1024 | 1 | 4 | 329281.2 | 26078.0 | 5.6% | ok |
| degenerate_dot_outer_f64_pytorch_n1024 | #2111 | pytorch | pytorch-cpu | f64 | form=outer, n=1024 | 1 | 4 | 204427.0 | 9047.0 | 3.4% | ok |
| degenerate_dot_rank0_f64_dot-general-shared_n1024 | #2111 | dot-general-shared | tenferro-rs | f64 | form=rank0, n=1024 | 1 | 2 | 385958.0 | 55093.5 | 14.5% | NOISY |
| degenerate_dot_rank0_f64_pytorch_n1024 | #2111 | pytorch | pytorch-cpu | f64 | form=rank0, n=1024 | 1 | 4 | 289218.8 | 15286.5 | 4.7% | ok |
| erased_add_f64_session-add_n256 | #1766 | session-add | tenferro-rs | f64 | n=256 | 1 | 4096 | 752.9 | 162.4 | 11.8% | NOISY |
| erased_add_f64_pytorch_n256 | #1766 | pytorch | pytorch-cpu | f64 | n=256 | 1 | 2048 | 992.3 | 120.5 | 24.9% | NOISY |
| erased_add_f64_session-add_n4096 | #1766 | session-add | tenferro-rs | f64 | n=4096 | 1 | 256 | 2448.9 | 2967.3 | 47.8% | NOISY |
| erased_add_f64_pytorch_n4096 | #1766 | pytorch | pytorch-cpu | f64 | n=4096 | 1 | 256 | 1840.8 | 652.8 | 45.7% | NOISY |
| erased_add_f64_session-add_n1048576 | #1766 | session-add | tenferro-rs | f64 | n=1048576 | 1 | 2 | 706312.5 | 217687.8 | 21.3% | NOISY |
| erased_add_f64_pytorch_n1048576 | #1766 | pytorch | pytorch-cpu | f64 | n=1048576 | 1 | 2 | 688854.0 | 149666.8 | 18.7% | NOISY |
| axpby_strided_source_f64_1024x1024_t1 | #2220 #2214 | strided-source | tenferro-rs | f64 | n=1024 | 1 | 1 | 4018750.0 | 2550895.5 | 50.7% | NOISY |
| axpby_contiguous_source_f64_1024x1024_t1 | #2220 #2214 | contiguous-source | tenferro-rs | f64 | n=1024 | 1 | 4 | 542531.2 | 126911.5 | 18.8% | NOISY |
| axpby_strided_source_f64_1024x1024_host_t1 | #2220 #2214 | host | rust-loop | f64 | n=1024 | 1 | 2 | 1691917.0 | 201562.5 | 26.5% | NOISY |
| erased_construct_f64_col_n8 | #2177 | col-major | tenferro-rs | f64 | n=8, layout=col | 1 | 16384 | 60.1 | 5.2 | 13.0% | NOISY |
| erased_construct_f64_row_n8 | #2177 | row-major | tenferro-rs | f64 | n=8, layout=row | 1 | 8192 | 153.5 | 16.7 | 8.3% | ok |
| erased_construct_f64_col_n4096 | #2177 | col-major | tenferro-rs | f64 | n=4096, layout=col | 1 | 16384 | 80.8 | 34.0 | 33.7% | NOISY |
| erased_construct_f64_row_n4096 | #2177 | row-major | tenferro-rs | f64 | n=4096, layout=row | 1 | 64 | 14968.1 | 938.8 | 5.0% | ok |
| erased_construct_f64_col_n65536 | #2177 | col-major | tenferro-rs | f64 | n=65536, layout=col | 1 | 1024 | 99.9 | 13.6 | 45.2% | NOISY |
| erased_construct_f64_row_n65536 | #2177 | row-major | tenferro-rs | f64 | n=65536, layout=row | 1 | 16 | 246023.4 | 22420.6 | 10.1% | NOISY |
| eager_small_leaf_f64_eager-held-session | #1704 | eager-held-session | tenferro-rs | f64 | op=leaf | 1 | 2048 | 1748.1 | 137.3 | 17.2% | NOISY |
| eager_small_add_f64_eager-held-session | #1667 | eager-held-session | tenferro-rs | f64 | op=add | 1 | 1024 | 1554.3 | 114.9 | 27.5% | NOISY |
| eager_small_add_f64_pytorch | #1667 | pytorch | pytorch-cpu | f64 | op=add | 1 | 4096 | 841.0 | 186.8 | 40.9% | NOISY |
| eager_small_gemm_f64_eager-held-session | #1904 | eager-held-session | tenferro-rs | f64 | op=gemm | 1 | 1024 | 2150.9 | 180.9 | 10.3% | NOISY |
| eager_small_gemm_f64_pytorch | #1904 | pytorch | pytorch-cpu | f64 | op=gemm | 1 | 1024 | 1053.8 | 395.8 | 34.0% | NOISY |
| trace_metadata_trace-fold_f32_1024x64_k50 | #2226 #2238 | trace-fold | tenferro-rs | f32 | rows=1024, cols=64, steps=50 | 1 | 2 | 1930687.5 | 375312.5 | 14.9% | NOISY |
| trace_metadata_trace-outputs_f32_1024x64_k50 | #2226 #2238 | trace-outputs | tenferro-rs | f32 | rows=1024, cols=64, steps=50 | 1 | 2 | 2034208.5 | 123437.5 | 7.4% | ok |
| trace_metadata_eager_f32_1024x64_k50 | #2226 #2238 | eager | tenferro-rs | f32 | rows=1024, cols=64, steps=50 | 1 | 1 | 2281917.0 | 130103.5 | 6.0% | ok |
| trace_metadata_pytorch_f32_1024x64_k50 | #2226 #2238 | pytorch | pytorch-cpu | f32 | rows=1024, cols=64, steps=50 | 1 | 2 | 1435042.0 | 98114.8 | 7.9% | ok |
| trace_metadata_fixed_neg_f32_8 | #2226 #2238 | trace-fold | tenferro-rs | f32 | rows=8, cols=1, steps=0 | 1 | 2048 | 1540.8 | 102.1 | 8.3% | ok |
| two_operand_einsum_f64_2x2_string | #2325 #2324 | string | tenferro-rs | f64 | m=2, k=2, n=2, format=ij,jk->ik | 4 | 1024 | 2624.8 | 275.7 | 16.1% | NOISY |
| two_operand_einsum_f64_2x2_subscripts | #2325 #2324 | subscripts | tenferro-rs | f64 | m=2, k=2, n=2, format=ij,jk->ik | 4 | 512 | 1917.7 | 193.5 | 43.8% | NOISY |
| two_operand_einsum_f64_2x2_prepared | #2325 #2324 | prepared | tenferro-rs | f64 | m=2, k=2, n=2, format=ij,jk->ik | 4 | 2048 | 1378.2 | 61.0 | 12.3% | NOISY |
| two_operand_einsum_f64_2x2_dot-general | #2325 #2324 | dot-general | tenferro-rs | f64 | m=2, k=2, n=2, format=ij,jk->ik | 4 | 2048 | 1412.2 | 96.3 | 12.2% | NOISY |
| two_operand_einsum_f64_2x2_pytorch | #2325 #2324 | pytorch | pytorch-cpu | f64 | m=2, k=2, n=2, format=ij,jk->ik | 4 | 256 | 11849.3 | 6214.7 | 28.2% | NOISY |
| two_operand_einsum_f64_16x16_string | #2325 #2324 | string | tenferro-rs | f64 | m=16, k=16, n=16, format=ij,jk->ik | 4 | 512 | 4091.5 | 1133.0 | 35.0% | NOISY |
| two_operand_einsum_f64_16x16_subscripts | #2325 #2324 | subscripts | tenferro-rs | f64 | m=16, k=16, n=16, format=ij,jk->ik | 4 | 512 | 2828.0 | 2960.8 | 50.7% | NOISY |
| two_operand_einsum_f64_16x16_prepared | #2325 #2324 | prepared | tenferro-rs | f64 | m=16, k=16, n=16, format=ij,jk->ik | 4 | 512 | 2557.1 | 655.7 | 39.5% | NOISY |
| two_operand_einsum_f64_16x16_dot-general | #2325 #2324 | dot-general | tenferro-rs | f64 | m=16, k=16, n=16, format=ij,jk->ik | 4 | 1024 | 2564.9 | 789.4 | 51.3% | NOISY |
| two_operand_einsum_f64_16x16_pytorch | #2325 #2324 | pytorch | pytorch-cpu | f64 | m=16, k=16, n=16, format=ij,jk->ik | 4 | 16 | 41067.7 | 17658.9 | 33.7% | NOISY |
| two_operand_einsum_c64_2x2_string | #2325 #2324 | string | tenferro-rs | c64 | m=2, k=2, n=2, format=ij,jk->ik | 4 | 512 | 4759.2 | 2302.1 | 42.0% | NOISY |
| two_operand_einsum_c64_2x2_subscripts | #2325 #2324 | subscripts | tenferro-rs | c64 | m=2, k=2, n=2, format=ij,jk->ik | 4 | 1024 | 2918.7 | 1112.5 | 27.7% | NOISY |
| two_operand_einsum_c64_2x2_prepared | #2325 #2324 | prepared | tenferro-rs | c64 | m=2, k=2, n=2, format=ij,jk->ik | 4 | 1024 | 2292.2 | 1770.1 | 49.2% | NOISY |
| two_operand_einsum_c64_2x2_dot-general | #2325 #2324 | dot-general | tenferro-rs | c64 | m=2, k=2, n=2, format=ij,jk->ik | 4 | 1024 | 1935.8 | 983.2 | 47.7% | NOISY |
| two_operand_einsum_c64_2x2_pytorch | #2325 #2324 | pytorch | pytorch-cpu | c64 | m=2, k=2, n=2, format=ij,jk->ik | 4 | 256 | 8591.1 | 605.5 | 59.4% | NOISY |
| two_operand_einsum_c64_16x16_string | #2325 #2324 | string | tenferro-rs | c64 | m=16, k=16, n=16, format=ij,jk->ik | 4 | 512 | 6497.9 | 1825.6 | 36.0% | NOISY |
| two_operand_einsum_c64_16x16_subscripts | #2325 #2324 | subscripts | tenferro-rs | c64 | m=16, k=16, n=16, format=ij,jk->ik | 4 | 512 | 3894.0 | 106.2 | 3.0% | ok |
| two_operand_einsum_c64_16x16_prepared | #2325 #2324 | prepared | tenferro-rs | c64 | m=16, k=16, n=16, format=ij,jk->ik | 4 | 512 | 3396.1 | 118.6 | 21.3% | NOISY |
| two_operand_einsum_c64_16x16_dot-general | #2325 #2324 | dot-general | tenferro-rs | c64 | m=16, k=16, n=16, format=ij,jk->ik | 4 | 512 | 3392.9 | 71.0 | 3.3% | ok |
| two_operand_einsum_c64_16x16_pytorch | #2325 #2324 | pytorch | pytorch-cpu | c64 | m=16, k=16, n=16, format=ij,jk->ik | 4 | 128 | 13049.8 | 311.2 | 1.6% | ok |
| two_operand_einsum_f64_abc_cef_abef_string | #2325 #2324 | string | tenferro-rs | f64 | m=8, k=2, n=16, format=abc,cef->abef | 4 | 1024 | 3283.3 | 146.2 | 7.8% | ok |
| two_operand_einsum_f64_abc_cef_abef_subscripts | #2325 #2324 | subscripts | tenferro-rs | f64 | m=8, k=2, n=16, format=abc,cef->abef | 4 | 1024 | 2046.3 | 117.6 | 11.1% | NOISY |
| two_operand_einsum_f64_abc_cef_abef_prepared | #2325 #2324 | prepared | tenferro-rs | f64 | m=8, k=2, n=16, format=abc,cef->abef | 4 | 512 | 2115.0 | 777.2 | 21.3% | NOISY |
| two_operand_einsum_f64_abc_cef_abef_dot-general | #2325 #2324 | dot-general | tenferro-rs | f64 | m=8, k=2, n=16, format=abc,cef->abef | 4 | 512 | 1606.2 | 67.0 | 23.1% | NOISY |
| two_operand_einsum_f64_abc_cef_abef_pytorch | #2325 #2324 | pytorch | pytorch-cpu | f64 | m=8, k=2, n=16, format=abc,cef->abef | 4 | 256 | 10267.7 | 424.6 | 28.0% | NOISY |
| decode_proj_f32_eager-shared_in1024_out1024_len8 | #1992 #2003 #1995 #2229 | eager-shared | tenferro-rs | f32 | in=1024, out=1024, len=8 | 4 | 16 | 156044.3 | 8622.4 | 5.9% | ok |
| decode_proj_f32_faer-direct_in1024_out1024_len8 | #1992 #2003 #1995 #2229 | faer-direct | faer-direct | f32 | in=1024, out=1024, len=8 | 4 | 16 | 270325.5 | 31029.9 | 26.2% | NOISY |
| decode_proj_f32_host-sgemm_in1024_out1024_len8 | #1992 #2003 #1995 #2229 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=1024, len=8 | 4 | 4 | 522708.2 | 248098.9 | 36.4% | NOISY |
| decode_proj_f32_pytorch_in1024_out1024_len8 | #1992 #2003 #1995 #2229 | pytorch | pytorch-cpu | f32 | in=1024, out=1024, len=8 | 4 | 8 | 177661.5 | 67598.9 | 30.2% | NOISY |
| decode_proj_f32_eager-shared_in1024_out1024_len64 | #1992 #2003 #1995 #2229 | eager-shared | tenferro-rs | f32 | in=1024, out=1024, len=64 | 4 | 2 | 256645.5 | 223593.5 | 80.5% | NOISY |
| decode_proj_f32_faer-direct_in1024_out1024_len64 | #1992 #2003 #1995 #2229 | faer-direct | faer-direct | f32 | in=1024, out=1024, len=64 | 4 | 1 | 1189042.0 | 250791.5 | 12.7% | NOISY |
| decode_proj_f32_host-sgemm_in1024_out1024_len64 | #1992 #2003 #1995 #2229 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=1024, len=64 | 4 | 2 | 2071229.0 | 350104.5 | 17.6% | NOISY |
| decode_proj_f32_pytorch_in1024_out1024_len64 | #1992 #2003 #1995 #2229 | pytorch | pytorch-cpu | f32 | in=1024, out=1024, len=64 | 4 | 16 | 332265.6 | 180449.2 | 43.1% | NOISY |
| decode_proj_f32_eager-shared_in1024_out1024_len512 | #1992 #2003 #1995 #2229 | eager-shared | tenferro-rs | f32 | in=1024, out=1024, len=512 | 4 | 2 | 1783771.0 | 568208.5 | 30.6% | NOISY |
| decode_proj_f32_faer-direct_in1024_out1024_len512 | #1992 #2003 #1995 #2229 | faer-direct | faer-direct | f32 | in=1024, out=1024, len=512 | 4 | 1 | 6590875.0 | 789937.0 | 11.3% | NOISY |
| decode_proj_f32_host-sgemm_in1024_out1024_len512 | #1992 #2003 #1995 #2229 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=1024, len=512 | 4 | 1 | 8183583.0 | 3153374.5 | 24.5% | NOISY |
| decode_proj_f32_pytorch_in1024_out1024_len512 | #1992 #2003 #1995 #2229 | pytorch | pytorch-cpu | f32 | in=1024, out=1024, len=512 | 4 | 2 | 1392354.5 | 79687.5 | 11.5% | NOISY |
| decode_proj_f32_eager-shared_in1024_out4096_len8 | #1992 #2003 #1995 #2229 | eager-shared | tenferro-rs | f32 | in=1024, out=4096, len=8 | 4 | 2 | 1017979.0 | 160229.5 | 10.5% | NOISY |
| decode_proj_f32_faer-direct_in1024_out4096_len8 | #1992 #2003 #1995 #2229 | faer-direct | faer-direct | f32 | in=1024, out=4096, len=8 | 4 | 2 | 969854.0 | 128854.2 | 8.6% | ok |
| decode_proj_f32_host-sgemm_in1024_out4096_len8 | #1992 #2003 #1995 #2229 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=4096, len=8 | 4 | 1 | 2182708.0 | 296021.0 | 9.7% | ok |
| decode_proj_f32_pytorch_in1024_out4096_len8 | #1992 #2003 #1995 #2229 | pytorch | pytorch-cpu | f32 | in=1024, out=4096, len=8 | 4 | 2 | 1234437.5 | 81771.0 | 9.2% | ok |
| decode_proj_f32_eager-shared_in1024_out4096_len64 | #1992 #2003 #1995 #2229 | eager-shared | tenferro-rs | f32 | in=1024, out=4096, len=64 | 4 | 1 | 1948166.0 | 1512521.0 | 41.4% | NOISY |
| decode_proj_f32_faer-direct_in1024_out4096_len64 | #1992 #2003 #1995 #2229 | faer-direct | faer-direct | f32 | in=1024, out=4096, len=64 | 4 | 1 | 8894959.0 | 2127792.0 | 20.9% | NOISY |
| decode_proj_f32_host-sgemm_in1024_out4096_len64 | #1992 #2003 #1995 #2229 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=4096, len=64 | 4 | 1 | 12444792.0 | 2810937.5 | 16.7% | NOISY |
| decode_proj_f32_pytorch_in1024_out4096_len64 | #1992 #2003 #1995 #2229 | pytorch | pytorch-cpu | f32 | in=1024, out=4096, len=64 | 4 | 2 | 1558792.0 | 221499.8 | 23.4% | NOISY |
| decode_proj_f32_eager-shared_in1024_out4096_len512 | #1992 #2003 #1995 #2229 | eager-shared | tenferro-rs | f32 | in=1024, out=4096, len=512 | 4 | 1 | 6589667.0 | 1655270.5 | 18.0% | NOISY |
| decode_proj_f32_faer-direct_in1024_out4096_len512 | #1992 #2003 #1995 #2229 | faer-direct | faer-direct | f32 | in=1024, out=4096, len=512 | 4 | 1 | 26301541.0 | 8198708.5 | 19.4% | NOISY |
| decode_proj_f32_host-sgemm_in1024_out4096_len512 | #1992 #2003 #1995 #2229 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=4096, len=512 | 4 | 1 | 25778375.0 | 8222187.5 | 26.0% | NOISY |
| decode_proj_f32_pytorch_in1024_out4096_len512 | #1992 #2003 #1995 #2229 | pytorch | pytorch-cpu | f32 | in=1024, out=4096, len=512 | 4 | 1 | 9810083.0 | 7115562.0 | 52.2% | NOISY |
| decode_proj_view_operand_f32_prepared_in1024_out1024_len8_t4 | #2216 | prepared-view | tenferro-rs | f32 | in=1024, out=1024, len=8 | 4 | 16 | 167411.4 | 5220.0 | 19.5% | NOISY |
| decode_proj_contiguous_operand_f32_prepared_in1024_out1024_len8_t4 | #2216 | prepared-contiguous | tenferro-rs | f32 | in=1024, out=1024, len=8 | 4 | 4 | 143333.2 | 12338.8 | 103.5% | NOISY |
| decode_proj_view_operand_f32_pytorch_in1024_out1024_len8_t4 | #2216 | pytorch | pytorch-cpu | f32 | in=1024, out=1024, len=8 | 4 | 16 | 178098.9 | 147345.1 | 77.8% | NOISY |
| gemm_c64_mm256_t1 | #1900 | dot-general-shared | tenferro-rs | c64 | n=256 | 4 | 2 | 728604.0 | 344187.8 | 71.7% | NOISY |
| gemm_f64_mm256_t1 | #1900 | dot-general-shared | tenferro-rs | f64 | n=256 | 4 | 16 | 180484.4 | 150378.9 | 62.1% | NOISY |
| conj_dot_f64_dot-general-rank1_len100 | #1615 | dot-general-rank1 | tenferro-rs | f64 | len=100 | 4 | 1024 | 1648.4 | 464.6 | 19.7% | NOISY |
| conj_dot_f64_portable-loop_len100 | #1615 | portable-loop | rust-loop | f64 | len=100 | 4 | 32768 | 62.9 | 41.7 | 37.9% | NOISY |
| conj_dot_f64_pytorch-vdot_len100 | #1615 | pytorch-vdot | pytorch-cpu | f64 | len=100 | 4 | 2048 | 804.2 | 42.4 | 47.5% | NOISY |
| conj_dot_f64_dot-general-rank1_len10000 | #1615 | dot-general-rank1 | tenferro-rs | f64 | len=10000 | 4 | 128 | 16020.5 | 958.8 | 17.6% | NOISY |
| conj_dot_f64_portable-loop_len10000 | #1615 | portable-loop | rust-loop | f64 | len=10000 | 4 | 256 | 10773.9 | 1513.1 | 10.2% | NOISY |
| conj_dot_f64_pytorch-vdot_len10000 | #1615 | pytorch-vdot | pytorch-cpu | f64 | len=10000 | 4 | 512 | 3454.1 | 1345.4 | 28.8% | NOISY |
| conj_dot_f64_dot-general-rank1_len1000000 | #1615 | dot-general-rank1 | tenferro-rs | f64 | len=1000000 | 4 | 1 | 1911792.0 | 549687.5 | 21.8% | NOISY |
| conj_dot_f64_portable-loop_len1000000 | #1615 | portable-loop | rust-loop | f64 | len=1000000 | 4 | 2 | 1206791.5 | 376177.0 | 29.2% | NOISY |
| conj_dot_f64_pytorch-vdot_len1000000 | #1615 | pytorch-vdot | pytorch-cpu | f64 | len=1000000 | 4 | 4 | 642698.0 | 293161.4 | 24.2% | NOISY |
| conj_dot_c64_dot-general-rank1_len100 | #1615 | dot-general-rank1 | tenferro-rs | c64 | len=100 | 4 | 1024 | 2745.8 | 2208.5 | 46.0% | NOISY |
| conj_dot_c64_portable-loop_len100 | #1615 | portable-loop | rust-loop | c64 | len=100 | 4 | 16384 | 180.0 | 138.2 | 40.4% | NOISY |
| conj_dot_c64_pytorch-vdot_len100 | #1615 | pytorch-vdot | pytorch-cpu | c64 | len=100 | 4 | 4096 | 1138.3 | 804.9 | 38.7% | NOISY |
| conj_dot_c64_dot-general-rank1_len10000 | #1615 | dot-general-rank1 | tenferro-rs | c64 | len=10000 | 4 | 32 | 65065.1 | 37731.1 | 26.2% | NOISY |
| conj_dot_c64_portable-loop_len10000 | #1615 | portable-loop | rust-loop | c64 | len=10000 | 4 | 256 | 10940.8 | 1967.9 | 22.1% | NOISY |
| conj_dot_c64_pytorch-vdot_len10000 | #1615 | pytorch-vdot | pytorch-cpu | c64 | len=10000 | 4 | 512 | 9834.2 | 6194.7 | 33.5% | NOISY |
| conj_dot_c64_dot-general-rank1_len1000000 | #1615 | dot-general-rank1 | tenferro-rs | c64 | len=1000000 | 4 | 1 | 7685250.0 | 3050124.5 | 55.9% | NOISY |
| conj_dot_c64_portable-loop_len1000000 | #1615 | portable-loop | rust-loop | c64 | len=1000000 | 4 | 2 | 1140729.5 | 96302.0 | 12.4% | NOISY |
| conj_dot_c64_pytorch-vdot_len1000000 | #1615 | pytorch-vdot | pytorch-cpu | c64 | len=1000000 | 4 | 2 | 1277333.0 | 555812.5 | 25.1% | NOISY |
| small_solve_f64_solve-loop_k16_b64 | #2007 #1878 | solve-loop | tenferro-rs | f64 | k=16, batch=64 | 4 | 4 | 325739.5 | 271921.8 | 41.7% | NOISY |
| small_solve_f64_triangular-solve-loop_k16_b64 | #2007 #1878 | triangular-solve-loop | tenferro-rs | f64 | k=16, batch=64 | 4 | 16 | 97888.1 | 4915.3 | 7.5% | ok |
| small_solve_f64_solve-batched_k16_b64 | #2007 #1878 | solve-batched | tenferro-rs | f64 | k=16, batch=64 | 4 | 32 | 92298.2 | 11411.5 | 23.6% | NOISY |
| small_solve_f64_pytorch-solve-batched_k16_b64 | #2007 #1878 | pytorch-solve-batched | pytorch-cpu | f64 | k=16, batch=64 | 4 | 8 | 203208.4 | 57598.9 | 17.9% | NOISY |
| small_solve_f64_pytorch-solve-triangular-batched_k16_b64 | #2007 #1878 | pytorch-solve-triangular-batched | pytorch-cpu | f64 | k=16, batch=64 | 4 | 32 | 25139.3 | 4488.3 | 20.9% | NOISY |
| small_solve_f64_solve-loop_k64_b64 | #2007 #1878 | solve-loop | tenferro-rs | f64 | k=64, batch=64 | 4 | 2 | 1714021.0 | 93510.5 | 13.7% | NOISY |
| small_solve_f64_triangular-solve-loop_k64_b64 | #2007 #1878 | triangular-solve-loop | tenferro-rs | f64 | k=64, batch=64 | 4 | 2 | 476187.5 | 71083.0 | 17.0% | NOISY |
| small_solve_f64_solve-batched_k64_b64 | #2007 #1878 | solve-batched | tenferro-rs | f64 | k=64, batch=64 | 4 | 2 | 1725041.5 | 22614.5 | 1.7% | ok |
| small_solve_f64_pytorch-solve-batched_k64_b64 | #2007 #1878 | pytorch-solve-batched | pytorch-cpu | f64 | k=64, batch=64 | 4 | 2 | 1565750.0 | 312323.0 | 12.6% | NOISY |
| small_solve_f64_pytorch-solve-triangular-batched_k64_b64 | #2007 #1878 | pytorch-solve-triangular-batched | pytorch-cpu | f64 | k=64, batch=64 | 4 | 8 | 440291.6 | 105648.4 | 14.3% | NOISY |
| tanh_chain_f32_compiled-prepared_k10_1024x64 | #1990 | compiled-prepared | tenferro-rs | f32 | rows=1024, cols=64, chain=10 | 4 | 2 | 1227417.0 | 55490.0 | 19.8% | NOISY |
| tanh_chain_f32_eager-shared_k10_1024x64 | #1990 | eager-shared | tenferro-rs | f32 | rows=1024, cols=64, chain=10 | 4 | 1 | 2515291.0 | 475520.5 | 16.1% | NOISY |
| tanh_chain_f32_compiled-prepared_k100_1024x64 | #1990 | compiled-prepared | tenferro-rs | f32 | rows=1024, cols=64, chain=100 | 4 | 1 | 10166459.0 | 660479.5 | 4.8% | ok |
| tanh_chain_f32_eager-shared_k100_1024x64 | #1990 | eager-shared | tenferro-rs | f32 | rows=1024, cols=64, chain=100 | 4 | 1 | 24958084.0 | 2944666.5 | 15.3% | NOISY |
| tanh_chain_f32_compiled-prepared_k200_1024x64 | #1990 | compiled-prepared | tenferro-rs | f32 | rows=1024, cols=64, chain=200 | 4 | 1 | 27836042.0 | 5262980.0 | 20.2% | NOISY |
| tanh_chain_f32_eager-shared_k200_1024x64 | #1990 | eager-shared | tenferro-rs | f32 | rows=1024, cols=64, chain=200 | 4 | 1 | 54216167.0 | 8721646.0 | 15.7% | NOISY |
| layer_norm_f32_eager-composed_d1024_len8_b1 | #2006 | eager-composed | tenferro-rs | f32 | norm=layer_norm, d=1024, len=8, batch=1 | 4 | 64 | 48589.8 | 4077.2 | 13.1% | NOISY |
| layer_norm_f32_pytorch-fused_d1024_len8_b1 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=layer_norm, d=1024, len=8, batch=1 | 4 | 64 | 31219.4 | 2722.6 | 20.9% | NOISY |
| layer_norm_f32_eager-single-call_d1024_len8_b1 | #2006 | eager-single-call | tenferro-rs | f32 | norm=layer_norm, d=1024, len=8, batch=1 | 4 | 32 | 52569.0 | 2547.5 | 6.7% | ok |
| layer_norm_f32_eager-composed_d1024_len8_b8 | #2006 | eager-composed | tenferro-rs | f32 | norm=layer_norm, d=1024, len=8, batch=8 | 4 | 2 | 104333.5 | 37021.0 | 34.6% | NOISY |
| layer_norm_f32_pytorch-fused_d1024_len8_b8 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=layer_norm, d=1024, len=8, batch=8 | 4 | 32 | 50385.4 | 3670.6 | 10.2% | NOISY |
| layer_norm_f32_eager-single-call_d1024_len8_b8 | #2006 | eager-single-call | tenferro-rs | f32 | norm=layer_norm, d=1024, len=8, batch=8 | 4 | 16 | 91906.2 | 4208.3 | 5.0% | ok |
| layer_norm_f32_eager-composed_d1024_len64_b1 | #2006 | eager-composed | tenferro-rs | f32 | norm=layer_norm, d=1024, len=64, batch=1 | 4 | 16 | 91958.3 | 14730.5 | 37.5% | NOISY |
| layer_norm_f32_pytorch-fused_d1024_len64_b1 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=layer_norm, d=1024, len=64, batch=1 | 4 | 32 | 88423.2 | 22818.4 | 24.3% | NOISY |
| layer_norm_f32_eager-single-call_d1024_len64_b1 | #2006 | eager-single-call | tenferro-rs | f32 | norm=layer_norm, d=1024, len=64, batch=1 | 4 | 16 | 90752.6 | 2044.3 | 6.1% | ok |
| layer_norm_f32_eager-composed_d1024_len64_b8 | #2006 | eager-composed | tenferro-rs | f32 | norm=layer_norm, d=1024, len=64, batch=8 | 4 | 4 | 927729.2 | 104823.0 | 21.1% | NOISY |
| layer_norm_f32_pytorch-fused_d1024_len64_b8 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=layer_norm, d=1024, len=64, batch=8 | 4 | 8 | 174703.1 | 33786.4 | 11.1% | NOISY |
| layer_norm_f32_eager-single-call_d1024_len64_b8 | #2006 | eager-single-call | tenferro-rs | f32 | norm=layer_norm, d=1024, len=64, batch=8 | 4 | 2 | 743333.0 | 236927.0 | 19.3% | NOISY |
| rms_norm_f32_eager-composed_d1024_len8_b1 | #2006 | eager-composed | tenferro-rs | f32 | norm=rms_norm, d=1024, len=8, batch=1 | 4 | 64 | 25664.7 | 1683.3 | 5.4% | ok |
| rms_norm_f32_pytorch-fused_d1024_len8_b1 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=rms_norm, d=1024, len=8, batch=1 | 4 | 64 | 16162.8 | 11445.0 | 63.0% | NOISY |
| rms_norm_f32_eager-single-call_d1024_len8_b1 | #2006 | eager-single-call | tenferro-rs | f32 | norm=rms_norm, d=1024, len=8, batch=1 | 4 | 64 | 34158.2 | 4897.8 | 13.1% | NOISY |
| rms_norm_f32_eager-composed_d1024_len8_b8 | #2006 | eager-composed | tenferro-rs | f32 | norm=rms_norm, d=1024, len=8, batch=8 | 4 | 32 | 45951.8 | 2736.3 | 4.4% | ok |
| rms_norm_f32_pytorch-fused_d1024_len8_b8 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=rms_norm, d=1024, len=8, batch=8 | 4 | 16 | 250455.8 | 21723.9 | 6.8% | ok |
| rms_norm_f32_eager-single-call_d1024_len8_b8 | #2006 | eager-single-call | tenferro-rs | f32 | norm=rms_norm, d=1024, len=8, batch=8 | 4 | 8 | 100541.6 | 85479.1 | 49.4% | NOISY |
| rms_norm_f32_eager-composed_d1024_len64_b1 | #2006 | eager-composed | tenferro-rs | f32 | norm=rms_norm, d=1024, len=64, batch=1 | 4 | 16 | 63994.8 | 64699.2 | 52.7% | NOISY |
| rms_norm_f32_pytorch-fused_d1024_len64_b1 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=rms_norm, d=1024, len=64, batch=1 | 4 | 8 | 168583.4 | 20257.8 | 23.1% | NOISY |
| rms_norm_f32_eager-single-call_d1024_len64_b1 | #2006 | eager-single-call | tenferro-rs | f32 | norm=rms_norm, d=1024, len=64, batch=1 | 4 | 16 | 61505.2 | 13925.8 | 15.7% | NOISY |
| rms_norm_f32_eager-composed_d1024_len64_b8 | #2006 | eager-composed | tenferro-rs | f32 | norm=rms_norm, d=1024, len=64, batch=8 | 4 | 2 | 818000.0 | 182083.2 | 21.8% | NOISY |
| rms_norm_f32_pytorch-fused_d1024_len64_b8 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=rms_norm, d=1024, len=64, batch=8 | 4 | 4 | 591510.5 | 99994.8 | 11.0% | NOISY |
| rms_norm_f32_eager-single-call_d1024_len64_b8 | #2006 | eager-single-call | tenferro-rs | f32 | norm=rms_norm, d=1024, len=64, batch=8 | 4 | 2 | 610687.5 | 115010.5 | 15.0% | NOISY |
| small_contraction_abcd-dbef-acef_f64_shared-session-default-pool_d4 | #1885 #1897 #1901 | shared-session-default-pool | tenferro-rs | f64 | extent=4 | 4 | 512 | 3582.6 | 425.0 | 8.1% | ok |
| activation_erf_f32_eager-single-call_1024x64 | #1975 #2069 #2032 | eager-single-call | tenferro-rs | f32 | op=erf, rows=1024, cols=64 | 4 | — | — | — | — | SKIPPED |
| activation_erf_f32_session-single-call_1024x64 | #1975 #2069 #2032 | session-single-call | tenferro-rs | f32 | op=erf, rows=1024, cols=64 | 4 | 4 | 584437.5 | 144140.8 | 29.3% | NOISY |
| activation_erf_f32_pytorch_1024x64 | #1975 #2069 #2032 | pytorch | pytorch-cpu | f32 | op=erf, rows=1024, cols=64 | 4 | — | — | — | — | SKIPPED |
| activation_sigmoid_f32_eager-single-call_1024x64 | #1975 #2069 #2032 | eager-single-call | tenferro-rs | f32 | op=sigmoid, rows=1024, cols=64 | 4 | — | — | — | — | SKIPPED |
| activation_sigmoid_f32_session-single-call_1024x64 | #1975 #2069 #2032 | session-single-call | tenferro-rs | f32 | op=sigmoid, rows=1024, cols=64 | 4 | 16 | 82666.7 | 20205.7 | 18.4% | NOISY |
| activation_sigmoid_f32_eager-composed_1024x64 | #1975 #2069 #2032 | eager-composed | tenferro-rs | f32 | op=sigmoid, rows=1024, cols=64 | 4 | 16 | 89502.6 | 10960.9 | 6.8% | ok |
| activation_sigmoid_f32_pytorch_1024x64 | #1975 #2069 #2032 | pytorch | pytorch-cpu | f32 | op=sigmoid, rows=1024, cols=64 | 4 | — | — | — | — | SKIPPED |
| activation_silu_f32_eager-single-call_1024x64 | #1975 #2069 #2032 | eager-single-call | tenferro-rs | f32 | op=silu, rows=1024, cols=64 | 4 | — | — | — | — | SKIPPED |
| activation_silu_f32_session-single-call_1024x64 | #1975 #2069 #2032 | session-single-call | tenferro-rs | f32 | op=silu, rows=1024, cols=64 | 4 | 16 | 86351.6 | 14118.5 | 25.3% | NOISY |
| activation_silu_f32_eager-composed_1024x64 | #1975 #2069 #2032 | eager-composed | tenferro-rs | f32 | op=silu, rows=1024, cols=64 | 4 | 16 | 108583.3 | 15781.2 | 12.2% | NOISY |
| activation_silu_f32_pytorch_1024x64 | #1975 #2069 #2032 | pytorch | pytorch-cpu | f32 | op=silu, rows=1024, cols=64 | 4 | — | — | — | — | SKIPPED |
| activation_softplus_f32_eager-single-call_1024x64 | #1975 #2069 #2032 | eager-single-call | tenferro-rs | f32 | op=softplus, rows=1024, cols=64 | 4 | — | — | — | — | SKIPPED |
| activation_softplus_f32_session-single-call_1024x64 | #1975 #2069 #2032 | session-single-call | tenferro-rs | f32 | op=softplus, rows=1024, cols=64 | 4 | 8 | 270713.5 | 15895.9 | 6.0% | ok |
| activation_softplus_f32_eager-composed_1024x64 | #1975 #2069 #2032 | eager-composed | tenferro-rs | f32 | op=softplus, rows=1024, cols=64 | 4 | 8 | 233390.6 | 38757.8 | 11.8% | NOISY |
| activation_softplus_f32_pytorch_1024x64 | #1975 #2069 #2032 | pytorch | pytorch-cpu | f32 | op=softplus, rows=1024, cols=64 | 4 | — | — | — | — | SKIPPED |
| activation_gelu_f32_eager-single-call_1024x64 | #1975 #2030 #2069 #2228 | eager-single-call | tenferro-rs | f32 | op=gelu, rows=1024, cols=64 | 4 | 16 | 138705.8 | 14867.2 | 6.5% | ok |
| activation_gelu_f32_session-single-call_1024x64 | #1975 #2030 #2069 #2228 | session-single-call | tenferro-rs | f32 | op=gelu, rows=1024, cols=64 | 4 | 16 | 133760.4 | 14121.1 | 10.4% | NOISY |
| activation_gelu_f32_pytorch_1024x64 | #1975 #2030 #2069 #2228 | pytorch | pytorch-cpu | f32 | op=gelu, rows=1024, cols=64 | 4 | 16 | 160682.3 | 9889.2 | 5.4% | ok |
| activation_gelu_tanh_f32_eager-single-call_1024x64 | #1975 #2069 #2032 | eager-single-call | tenferro-rs | f32 | op=gelu_tanh, rows=1024, cols=64 | 4 | — | — | — | — | SKIPPED |
| activation_gelu_tanh_f32_session-single-call_1024x64 | #1975 #2069 #2032 | session-single-call | tenferro-rs | f32 | op=gelu_tanh, rows=1024, cols=64 | 4 | 8 | 181203.1 | 27526.0 | 18.4% | NOISY |
| activation_gelu_tanh_f32_eager-composed_1024x64 | #1975 #2069 #2032 | eager-composed | tenferro-rs | f32 | op=gelu_tanh, rows=1024, cols=64 | 4 | 4 | 404781.2 | 90677.1 | 16.7% | NOISY |
| activation_gelu_tanh_f32_pytorch_1024x64 | #1975 #2069 #2032 | pytorch | pytorch-cpu | f32 | op=gelu_tanh, rows=1024, cols=64 | 4 | — | — | — | — | SKIPPED |
| softmax_f32_eager-single-call_len64_b8 | #1976 #2031 #2231 | eager-single-call | tenferro-rs | f32 | op=softmax, len=64, batch=8, axis=0 | 4 | 32 | 66290.4 | 14270.2 | 17.7% | NOISY |
| softmax_f32_session-single-call_len64_b8 | #1976 #2031 #2231 | session-single-call | tenferro-rs | f32 | op=softmax, len=64, batch=8, axis=0 | 4 | 32 | 36119.8 | 6051.4 | 13.4% | NOISY |
| softmax_f32_eager-composed_len64_b8 | #1976 #2031 #2231 | eager-composed | tenferro-rs | f32 | op=softmax, len=64, batch=8, axis=0 | 4 | 32 | 79888.0 | 7762.3 | 8.6% | ok |
| softmax_f32_pytorch_len64_b8 | #1976 #2031 #2231 | pytorch | pytorch-cpu | f32 | op=softmax, len=64, batch=8, axis=0 | 4 | 32 | 121595.1 | 28812.5 | 26.9% | NOISY |
| softmax_f32_eager-single-call_len512_b8 | #1976 #2231 | eager-single-call | tenferro-rs | f32 | op=softmax, len=512, batch=8, axis=0 | 4 | 2 | 1905437.5 | 561218.2 | 15.9% | NOISY |
| softmax_f32_session-single-call_len512_b8 | #1976 #2231 | session-single-call | tenferro-rs | f32 | op=softmax, len=512, batch=8, axis=0 | 4 | 2 | 1900750.0 | 181895.8 | 10.2% | NOISY |
| softmax_f32_eager-composed_len512_b8 | #1976 #2231 | eager-composed | tenferro-rs | f32 | op=softmax, len=512, batch=8, axis=0 | 4 | 1 | 3113125.0 | 476521.0 | 12.6% | NOISY |
| softmax_f32_pytorch_len512_b8 | #1976 #2231 | pytorch | pytorch-cpu | f32 | op=softmax, len=512, batch=8, axis=0 | 4 | 1 | 2515625.0 | 582770.5 | 34.5% | NOISY |
| log_softmax_f32_eager-single-call_len64_b8 | #1976 | eager-single-call | tenferro-rs | f32 | op=log_softmax, len=64, batch=8, axis=0 | 4 | 16 | 97958.3 | 6002.6 | 22.3% | NOISY |
| log_softmax_f32_session-single-call_len64_b8 | #1976 | session-single-call | tenferro-rs | f32 | op=log_softmax, len=64, batch=8, axis=0 | 4 | 32 | 73118.5 | 3509.8 | 8.0% | ok |
| log_softmax_f32_eager-composed_len64_b8 | #1976 | eager-composed | tenferro-rs | f32 | op=log_softmax, len=64, batch=8, axis=0 | 4 | 8 | 83078.1 | 19036.5 | 37.4% | NOISY |
| log_softmax_f32_pytorch_len64_b8 | #1976 | pytorch | pytorch-cpu | f32 | op=log_softmax, len=64, batch=8, axis=0 | 4 | 32 | 79165.4 | 49856.1 | 40.4% | NOISY |
| log_softmax_f32_eager-single-call_len512_b8 | #1976 | eager-single-call | tenferro-rs | f32 | op=log_softmax, len=512, batch=8, axis=0 | 4 | 1 | 2521750.0 | 309500.0 | 19.3% | NOISY |
| log_softmax_f32_session-single-call_len512_b8 | #1976 | session-single-call | tenferro-rs | f32 | op=log_softmax, len=512, batch=8, axis=0 | 4 | 1 | 3234041.0 | 308709.0 | 36.7% | NOISY |
| log_softmax_f32_eager-composed_len512_b8 | #1976 | eager-composed | tenferro-rs | f32 | op=log_softmax, len=512, batch=8, axis=0 | 4 | 1 | 3649875.0 | 444604.5 | 8.6% | ok |
| log_softmax_f32_pytorch_len512_b8 | #1976 | pytorch | pytorch-cpu | f32 | op=log_softmax, len=512, batch=8, axis=0 | 4 | 2 | 2542041.5 | 668260.5 | 28.1% | NOISY |
| masked_softmax_f32_eager-single-call_len64_b8 | #1976 | eager-single-call | tenferro-rs | f32 | op=masked_softmax, len=64, batch=8, axis=0 | 4 | 16 | 143044.2 | 37231.7 | 14.1% | NOISY |
| masked_softmax_f32_session-single-call_len64_b8 | #1976 | session-single-call | tenferro-rs | f32 | op=masked_softmax, len=64, batch=8, axis=0 | 4 | 16 | 101354.2 | 13729.2 | 13.9% | NOISY |
| masked_softmax_f32_eager-composed_len64_b8 | #1976 | eager-composed | tenferro-rs | f32 | op=masked_softmax, len=64, batch=8, axis=0 | 4 | 16 | 93721.4 | 20773.4 | 25.5% | NOISY |
| masked_softmax_f32_pytorch_len64_b8 | #1976 | pytorch | pytorch-cpu | f32 | op=masked_softmax, len=64, batch=8, axis=0 | 4 | 16 | 145520.9 | 22334.7 | 20.5% | NOISY |
| masked_softmax_f32_eager-single-call_len512_b8 | #1976 | eager-single-call | tenferro-rs | f32 | op=masked_softmax, len=512, batch=8, axis=0 | 4 | 1 | 4929375.0 | 664812.5 | 11.6% | NOISY |
| masked_softmax_f32_session-single-call_len512_b8 | #1976 | session-single-call | tenferro-rs | f32 | op=masked_softmax, len=512, batch=8, axis=0 | 4 | 1 | 3447917.0 | 286562.5 | 9.2% | ok |
| masked_softmax_f32_eager-composed_len512_b8 | #1976 | eager-composed | tenferro-rs | f32 | op=masked_softmax, len=512, batch=8, axis=0 | 4 | 1 | 4136583.0 | 799000.0 | 13.5% | NOISY |
| masked_softmax_f32_pytorch_len512_b8 | #1976 | pytorch | pytorch-cpu | f32 | op=masked_softmax, len=512, batch=8, axis=0 | 4 | 1 | 3474209.0 | 914313.5 | 15.3% | NOISY |
| softmax_f32_eager-single-call_len64_b8_t4 | #2233 #2235 | eager-single-call | tenferro-rs | f32 | op=softmax, len=64, batch=8, axis=0 | 4 | 32 | 39843.8 | 9386.7 | 16.8% | NOISY |
| softmax_f32_eager-composed_len64_b8_t4 | #2233 #2235 | eager-composed | tenferro-rs | f32 | op=softmax, len=64, batch=8, axis=0 | 4 | 32 | 81230.5 | 12947.3 | 11.5% | NOISY |
| softmax_f32_pytorch_len64_b8_t4 | #2233 #2235 | pytorch | pytorch-cpu | f32 | op=softmax, len=64, batch=8, axis=0 | 4 | 32 | 87625.0 | 21124.3 | 21.6% | NOISY |
| reduce_mean_f32_eager-single-call_1024x64_axis0 | #1976 | eager-single-call | tenferro-rs | f32 | rows=1024, cols=64, axis=0 | 4 | 256 | 15848.0 | 4602.1 | 23.3% | NOISY |
| reduce_mean_f32_session-single-call_1024x64_axis0 | #1976 | session-single-call | tenferro-rs | f32 | rows=1024, cols=64, axis=0 | 4 | 256 | 8800.9 | 724.9 | 18.0% | NOISY |
| reduce_mean_f32_eager-composed_1024x64_axis0 | #1976 | eager-composed | tenferro-rs | f32 | rows=1024, cols=64, axis=0 | 4 | 128 | 14258.1 | 2644.5 | 21.4% | NOISY |
| reduce_mean_f32_pytorch_1024x64_axis0 | #1976 | pytorch | pytorch-cpu | f32 | rows=1024, cols=64, axis=0 | 4 | 32 | 38957.0 | 5935.6 | 10.6% | NOISY |
| take_along_axis_rows_f64_eager-single-call_n64_b256 | #2008 | eager-single-call | tenferro-rs | f64 | n=64, batch=256, axis=0 | 4 | 1 | 4233833.0 | 205416.5 | 4.4% | ok |
| take_along_axis_rows_f64_session-single-call_n64_b256 | #2008 | session-single-call | tenferro-rs | f64 | n=64, batch=256, axis=0 | 4 | 1 | 4461667.0 | 852541.5 | 14.9% | NOISY |
| take_along_axis_rows_f64_eager-gather-prebuilt_n64_b256 | #2008 | eager-gather-prebuilt | tenferro-rs | f64 | n=64, batch=256, axis=0 | 4 | 1 | 4392833.0 | 422083.0 | 7.8% | ok |
| take_along_axis_rows_f64_pytorch_n64_b256 | #2008 | pytorch | pytorch-cpu | f64 | n=64, batch=256, axis=0 | 4 | 1 | 2389916.0 | 286916.5 | 13.4% | NOISY |
| lstsq_f64_eager-shared_m768_n384_rhs16 | #2027 | eager-shared | tenferro-rs | f64 | operation=lstsq | 4 | 1 | 13052333.0 | 3182833.0 | 21.1% | NOISY |
| lstsq_f64_pytorch_m768_n384_rhs16 | #2027 | pytorch | pytorch-cpu | f64 | operation=lstsq | 4 | 1 | 6394250.0 | 1444020.5 | 27.8% | NOISY |
| triangular_f64_concrete-shared_n4096_rhs64 | #2028 | concrete-shared | tenferro-rs | f64 | operation=triangular | 4 | 1 | 21443875.0 | 6434187.0 | 18.1% | NOISY |
| triangular_f64_pytorch_n4096_rhs64 | #2028 | pytorch | pytorch-cpu | f64 | operation=triangular | 4 | 1 | 13428125.0 | 3058396.5 | 36.8% | NOISY |
| reshape_f64_concrete-shared_n33554432_to8192x4096 | #2029 | concrete-shared | tenferro-rs | f64 | operation=reshape | 4 | 1 | 13125125.0 | 2720541.5 | 16.0% | NOISY |
| reshape_f64_pytorch_n33554432_to8192x4096 | #2029 | pytorch | pytorch-cpu | f64 | operation=reshape | 4 | 1 | 11089333.0 | 698813.0 | 11.6% | NOISY |
| metadata_transpose_view_f64_rank8_t4 | #2180 | metadata-clone | tenferro-rs | f64 | shape=[2, 2, 2, 2, 2, 2, 2, 2] | 4 | 16384 | 57.2 | 5.3 | 9.0% | ok |
| metadata_transpose_view_f64_rank8_t4_julia-base | #2180 | julia-base | julia-base | f64 | shape=[2, 2, 2, 2, 2, 2, 2, 2] | 4 | 1048576 | 0.4 | 0.1 | 16.5% | NOISY |
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
| lu_batched_f64_factor_n8_b1024 | #2142 | tenferro | tenferro-rs | f64 | n=8, batch=1024, op=factor | 4 | 8 | 576265.6 | 165070.4 | 24.7% | NOISY |
| lu_batched_f64_factor_n8_b1024_host | #2142 | host | rust-loop | f64 | n=8, batch=1024, op=factor | 4 | 8 | 250005.2 | 20307.2 | 4.4% | ok |
| lu_batched_f64_solve_n8_b1024 | #2142 | tenferro | tenferro-rs | f64 | n=8, batch=1024, op=solve | 4 | 16 | 104906.2 | 4425.8 | 3.4% | ok |
| lu_batched_f64_solve_n8_b1024_pytorch | #2142 | pytorch | pytorch-cpu | f64 | n=8, batch=1024, op=solve | 4 | 32 | 83115.9 | 3658.2 | 17.5% | NOISY |
| lu_batched_f64_factor_n16_b1024 | #2142 | tenferro | tenferro-rs | f64 | n=16, batch=1024, op=factor | 4 | 2 | 822250.0 | 162833.2 | 22.9% | NOISY |
| lu_batched_f64_factor_n16_b1024_host | #2142 | host | rust-loop | f64 | n=16, batch=1024, op=factor | 4 | 2 | 1096375.0 | 99500.0 | 19.4% | NOISY |
| lu_batched_f64_solve_n16_b1024 | #2142 | tenferro | tenferro-rs | f64 | n=16, batch=1024, op=solve | 4 | 16 | 233093.8 | 15869.8 | 9.2% | ok |
| lu_batched_f64_solve_n16_b1024_pytorch | #2142 | pytorch | pytorch-cpu | f64 | n=16, batch=1024, op=solve | 4 | 16 | 195455.8 | 19684.9 | 15.8% | NOISY |
| small_grad_qr_f64_trace-prepared_n4 | #2186 | trace-prepared | tenferro-rs | f64 | op=qr, n=4 | 4 | 32 | 47645.8 | 15877.0 | 46.5% | NOISY |
| small_grad_qr_f64_pytorch_n4 | #2186 | pytorch | pytorch-cpu | f64 | op=qr, n=4 | 4 | 16 | 216856.8 | 74842.4 | 34.2% | NOISY |
| small_grad_qr_f64_trace-prepared_n16 | #2186 | trace-prepared | tenferro-rs | f64 | op=qr, n=16 | 4 | 32 | 62755.2 | 12795.6 | 35.2% | NOISY |
| small_grad_qr_f64_pytorch_n16 | #2186 | pytorch | pytorch-cpu | f64 | op=qr, n=16 | 4 | 16 | 123705.8 | 8342.4 | 9.3% | ok |
| small_grad_eigh_f64_trace-prepared_n4 | #2186 | trace-prepared | tenferro-rs | f64 | op=eigh, n=4 | 4 | 64 | 52011.7 | 14828.8 | 18.6% | NOISY |
| small_grad_eigh_f64_pytorch_n4 | #2186 | pytorch | pytorch-cpu | f64 | op=eigh, n=4 | 4 | 32 | 46019.5 | 3979.2 | 16.8% | NOISY |
| small_grad_eigh_f64_trace-prepared_n16 | #2186 | trace-prepared | tenferro-rs | f64 | op=eigh, n=16 | 4 | 32 | 91880.2 | 3856.8 | 3.4% | ok |
| small_grad_eigh_f64_pytorch_n16 | #2186 | pytorch | pytorch-cpu | f64 | op=eigh, n=16 | 4 | 32 | 72466.2 | 10656.2 | 28.8% | NOISY |
| small_grad_lu_f64_trace-prepared_n4 | #2186 | trace-prepared | tenferro-rs | f64 | op=lu, n=4 | 4 | 64 | 41251.3 | 34964.8 | 54.6% | NOISY |
| small_grad_lu_f64_pytorch_n4 | #2186 | pytorch | pytorch-cpu | f64 | op=lu, n=4 | 4 | 2 | 187854.5 | 145312.2 | 39.0% | NOISY |
| small_grad_lu_f64_trace-prepared_n16 | #2186 | trace-prepared | tenferro-rs | f64 | op=lu, n=16 | 4 | 64 | 36718.8 | 7542.6 | 20.9% | NOISY |
| small_grad_lu_f64_pytorch_n16 | #2186 | pytorch | pytorch-cpu | f64 | op=lu, n=16 | 4 | 8 | 257505.1 | 50093.8 | 14.9% | NOISY |
| small_grad_svd_f64_trace-prepared_n4 | #2186 | trace-prepared | tenferro-rs | f64 | op=svd, n=4 | 4 | 32 | 81240.9 | 11498.7 | 19.8% | NOISY |
| small_grad_svd_f64_pytorch_n4 | #2186 | pytorch | pytorch-cpu | f64 | op=svd, n=4 | 4 | 32 | 63054.7 | 2150.4 | 32.4% | NOISY |
| small_grad_svd_f64_trace-prepared_n16 | #2186 | trace-prepared | tenferro-rs | f64 | op=svd, n=16 | 4 | 16 | 173421.9 | 2815.1 | 2.5% | ok |
| small_grad_svd_f64_pytorch_n16 | #2186 | pytorch | pytorch-cpu | f64 | op=svd, n=16 | 4 | 32 | 113320.3 | 5185.6 | 6.8% | ok |
| small_grad_solve_f64_trace-prepared_n4 | #2186 | trace-prepared | tenferro-rs | f64 | op=solve, n=4 | 4 | 128 | 21512.7 | 292.3 | 2.2% | ok |
| small_grad_solve_f64_pytorch_n4 | #2186 | pytorch | pytorch-cpu | f64 | op=solve, n=4 | 4 | 64 | 38367.8 | 6856.1 | 11.2% | NOISY |
| small_grad_solve_f64_trace-prepared_n16 | #2186 | trace-prepared | tenferro-rs | f64 | op=solve, n=16 | 4 | 128 | 29654.3 | 1631.8 | 7.8% | ok |
| small_grad_solve_f64_pytorch_n16 | #2186 | pytorch | pytorch-cpu | f64 | op=solve, n=16 | 4 | 64 | 53314.5 | 9421.2 | 11.1% | NOISY |
| fft_axis0_c64_fft-executor_n128 | #1765 | fft-executor | tenferro-rs | c64 | n=128 | 4 | 1 | 6563459.0 | 4347833.0 | 29.5% | NOISY |
| fft_axis0_c64_pytorch_n128 | #1765 | pytorch | pytorch-cpu | c64 | n=128 | 4 | 1 | 22709666.0 | 8838375.5 | 21.9% | NOISY |
| degenerate_dot_hadamard_f64_dot-general-shared_n1024 | #2111 | dot-general-shared | tenferro-rs | f64 | form=hadamard, n=1024 | 4 | 2 | 892958.0 | 147604.0 | 16.3% | NOISY |
| degenerate_dot_hadamard_f64_pytorch_n1024 | #2111 | pytorch | pytorch-cpu | f64 | form=hadamard, n=1024 | 4 | 2 | 909521.0 | 104552.0 | 11.9% | NOISY |
| degenerate_dot_outer_f64_dot-general-shared_n1024 | #2111 | dot-general-shared | tenferro-rs | f64 | form=outer, n=1024 | 4 | 2 | 311104.5 | 32458.5 | 12.4% | NOISY |
| degenerate_dot_outer_f64_pytorch_n1024 | #2111 | pytorch | pytorch-cpu | f64 | form=outer, n=1024 | 4 | 4 | 456718.8 | 112822.9 | 16.5% | NOISY |
| degenerate_dot_rank0_f64_dot-general-shared_n1024 | #2111 | dot-general-shared | tenferro-rs | f64 | form=rank0, n=1024 | 4 | 4 | 445760.5 | 155088.5 | 27.4% | NOISY |
| degenerate_dot_rank0_f64_pytorch_n1024 | #2111 | pytorch | pytorch-cpu | f64 | form=rank0, n=1024 | 4 | 4 | 498031.2 | 84338.4 | 21.8% | NOISY |
| erased_add_f64_session-add_n256 | #1766 | session-add | tenferro-rs | f64 | n=256 | 4 | 1024 | 1581.3 | 624.1 | 31.9% | NOISY |
| erased_add_f64_pytorch_n256 | #1766 | pytorch | pytorch-cpu | f64 | n=256 | 4 | 2048 | 1167.7 | 192.2 | 20.2% | NOISY |
| erased_add_f64_session-add_n4096 | #1766 | session-add | tenferro-rs | f64 | n=4096 | 4 | 512 | 2649.1 | 889.1 | 35.9% | NOISY |
| erased_add_f64_pytorch_n4096 | #1766 | pytorch | pytorch-cpu | f64 | n=4096 | 4 | 256 | 1896.6 | 418.1 | 25.0% | NOISY |
| erased_add_f64_session-add_n1048576 | #1766 | session-add | tenferro-rs | f64 | n=1048576 | 4 | 2 | 689250.0 | 248729.2 | 22.7% | NOISY |
| erased_add_f64_pytorch_n1048576 | #1766 | pytorch | pytorch-cpu | f64 | n=1048576 | 4 | 2 | 739708.0 | 110458.5 | 11.3% | NOISY |
| axpby_strided_source_f64_1024x1024_t4 | #2220 #2214 | strided-source | tenferro-rs | f64 | n=1024 | 4 | 2 | 1214312.5 | 639125.0 | 27.5% | NOISY |
| axpby_contiguous_source_f64_1024x1024_t4 | #2220 #2214 | contiguous-source | tenferro-rs | f64 | n=1024 | 4 | 4 | 596250.0 | 97120.0 | 12.3% | NOISY |
| axpby_strided_source_f64_1024x1024_host_t4 | #2220 #2214 | host | rust-loop | f64 | n=1024 | 4 | 1 | 2138084.0 | 307104.5 | 15.7% | NOISY |
| erased_construct_f64_col_n8 | #2177 | col-major | tenferro-rs | f64 | n=8, layout=col | 4 | 32768 | 78.8 | 36.8 | 38.8% | NOISY |
| erased_construct_f64_row_n8 | #2177 | row-major | tenferro-rs | f64 | n=8, layout=row | 4 | 16384 | 201.6 | 51.1 | 17.5% | NOISY |
| erased_construct_f64_col_n4096 | #2177 | col-major | tenferro-rs | f64 | n=4096, layout=col | 4 | 16384 | 68.3 | 69.2 | 38.1% | NOISY |
| erased_construct_f64_row_n4096 | #2177 | row-major | tenferro-rs | f64 | n=4096, layout=row | 4 | 32 | 13932.3 | 347.0 | 10.6% | NOISY |
| erased_construct_f64_col_n65536 | #2177 | col-major | tenferro-rs | f64 | n=65536, layout=col | 4 | 1024 | 100.3 | 30.4 | 29.8% | NOISY |
| erased_construct_f64_row_n65536 | #2177 | row-major | tenferro-rs | f64 | n=65536, layout=row | 4 | 16 | 244502.6 | 22800.8 | 14.8% | NOISY |
| eager_small_leaf_f64_eager-held-session | #1704 | eager-held-session | tenferro-rs | f64 | op=leaf | 4 | 1024 | 2071.2 | 157.1 | 24.6% | NOISY |
| eager_small_add_f64_eager-held-session | #1667 | eager-held-session | tenferro-rs | f64 | op=add | 4 | 1024 | 2137.9 | 167.8 | 9.7% | ok |
| eager_small_add_f64_pytorch | #1667 | pytorch | pytorch-cpu | f64 | op=add | 4 | 2048 | 966.7 | 256.3 | 315.0% | NOISY |
| eager_small_gemm_f64_eager-held-session | #1904 | eager-held-session | tenferro-rs | f64 | op=gemm | 4 | 1024 | 3081.2 | 1048.3 | 19.6% | NOISY |
| eager_small_gemm_f64_pytorch | #1904 | pytorch | pytorch-cpu | f64 | op=gemm | 4 | 2048 | 912.3 | 64.3 | 6.5% | ok |
| trace_metadata_trace-fold_f32_1024x64_k50 | #2226 #2238 | trace-fold | tenferro-rs | f32 | rows=1024, cols=64, steps=50 | 4 | 2 | 2003104.5 | 171177.0 | 5.6% | ok |
| trace_metadata_trace-outputs_f32_1024x64_k50 | #2226 #2238 | trace-outputs | tenferro-rs | f32 | rows=1024, cols=64, steps=50 | 4 | 1 | 3255917.0 | 1265479.5 | 32.5% | NOISY |
| trace_metadata_eager_f32_1024x64_k50 | #2226 #2238 | eager | tenferro-rs | f32 | rows=1024, cols=64, steps=50 | 4 | 1 | 3274333.0 | 1456604.5 | 23.5% | NOISY |
| trace_metadata_pytorch_f32_1024x64_k50 | #2226 #2238 | pytorch | pytorch-cpu | f32 | rows=1024, cols=64, steps=50 | 4 | 1 | 4234792.0 | 161396.5 | 10.3% | NOISY |
| trace_metadata_fixed_neg_f32_8 | #2226 #2238 | trace-fold | tenferro-rs | f32 | rows=8, cols=1, steps=0 | 4 | 1024 | 2541.3 | 354.1 | 9.2% | ok |


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
