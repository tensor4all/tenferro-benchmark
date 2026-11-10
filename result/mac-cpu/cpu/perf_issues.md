# CPU Performance-Issue Workloads

- Suite: `cpu/perf_issues`
- Raw run: `data/results/mac-cpu/cpu/perf_issues/20261110_180020`

## CPU Information

- Model: `Apple M2 Pro (Virtual)`
- Vendor: `Apple`
- Logical CPUs: `5`
- Physical CPUs: `5`
- Sockets: `1`
- Cores per socket: `5`
- Threads per core: `1`
- NUMA nodes: `1`
- Performance levels: `Standard: 5 physical / 5 logical (L1i 128 KiB, L1d 64 KiB, L2 4 MiB, 5 CPUs/L2)`
- Python platform: `macOS-26.7.2-arm64-arm-64bit-Mach-O`
- Container: `none`

## run_t1

Full metadata: `data/results/mac-cpu/cpu/perf_issues/20261110_180020/run_t1.yaml`

```yaml
timestamp: '2026-11-10T18:00:20Z'
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
  hostname: dsm12-be220-026d960e-6330-44e9-b70f-e8169c749221-360E892740E3.local
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
    BENCHMARK_COMMIT: 15cf8f674203aeca079999fdb339cf4d35801b88
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

Full metadata: `data/results/mac-cpu/cpu/perf_issues/20261110_180020/run_t4.yaml`

```yaml
timestamp: '2026-11-10T18:01:00Z'
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
  hostname: dsm12-be220-026d960e-6330-44e9-b70f-e8169c749221-360E892740E3.local
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
    BENCHMARK_COMMIT: 15cf8f674203aeca079999fdb339cf4d35801b88
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
- Expected 681, selected 681, executed 493, unsupported 0, failed 0, skipped 188, missing 0, noisy 91.
- Complete: **NO** — only executed cases count as covered; unsupported, skipped, failed, missing and unselected cases never do.

| expected | selected | executed | unsupported | failed | skipped | missing | noisy |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 681 | 681 | 493 | 0 | 0 | 188 | 0 | 91 |

- Skipped: `activation_erf_f32_eager-single-call_1024x64@t1`, `activation_erf_f32_pytorch_1024x64@t1`, `activation_sigmoid_f32_eager-single-call_1024x64@t1`, `activation_sigmoid_f32_pytorch_1024x64@t1`, `activation_silu_f32_eager-single-call_1024x64@t1`, `activation_silu_f32_pytorch_1024x64@t1`, `activation_softplus_f32_eager-single-call_1024x64@t1`, `activation_softplus_f32_pytorch_1024x64@t1`, `activation_gelu_tanh_f32_eager-single-call_1024x64@t1`, `activation_gelu_tanh_f32_pytorch_1024x64@t1`, `followup_structural_broadcast_in_dim_direct-shared@t1`, `followup_structural_broadcast_in_dim_pytorch@t1`, `followup_real_tanh_direct-shared@t1`, `followup_real_tanh_jax@t1`, `followup_real_log1p_direct-shared@t1`, `followup_real_log1p_pytorch@t1`, `followup_real_cos_direct-shared@t1`, `followup_real_cos_pytorch@t1`, `followup_real_sin_direct-shared@t1`, `followup_real_sin_pytorch@t1`, `followup_real_exp_direct-shared@t1`, `followup_real_exp_jax@t1`, `followup_real_expm1_direct-shared@t1`, `followup_real_expm1_jax@t1`, `followup_real_pow_direct-shared@t1`, `followup_real_pow_pytorch@t1`, `followup_complex_exp_direct-shared@t1`, `followup_complex_exp_julia-base@t1`, `followup_real_log_direct-shared@t1`, `followup_real_log_pytorch@t1`, `followup_real_reduce_max_axis1_direct-shared@t1`, `followup_real_reduce_max_axis1_julia-base@t1`, `followup_real_reduce_prod_axis1_direct-shared@t1`, `followup_real_reduce_prod_axis1_pytorch@t1`, `followup_real_reduce_sum_axis1_direct-shared@t1`, `followup_real_reduce_sum_axis1_pytorch@t1`, `followup_real_reduce_max_axis0_direct-shared@t1`, `followup_real_reduce_max_axis0_julia-base@t1`, `followup_real_reduce_min_axis1_direct-shared@t1`, `followup_real_reduce_min_axis1_julia-base@t1` (+148 more)
- Noisy: `two_operand_einsum_f64_2x2_string@t1`, `two_operand_einsum_f64_2x2_prepared@t1`, `two_operand_einsum_f64_16x16_prepared@t1`, `conj_dot_c64_pytorch-vdot_len100@t1`, `tanh_chain_f32_eager-shared_k10_1024x64@t1`, `layer_norm_f32_eager-composed_d1024_len8_b1@t1`, `layer_norm_f32_pytorch-fused_d1024_len8_b1@t1`, `layer_norm_f32_eager-composed_d1024_len8_b8@t1`, `layer_norm_f32_eager-single-call_d1024_len8_b8@t1`, `rms_norm_f32_eager-single-call_d1024_len8_b1@t1`, `activation_softplus_f32_session-single-call_1024x64@t1`, `softmax_f32_eager-composed_len64_b8@t1`, `reduce_mean_f32_eager-composed_1024x64_axis0@t1`, `reduce_mean_f32_pytorch_1024x64_axis0@t1`, `take_along_axis_rows_f64_eager-gather-prebuilt_n64_b256@t1`, `analytic_neg_f64_sweep_8MiB_t1@t1`, `lu_batched_f64_solve_n8_b1024@t1`, `small_grad_qr_f64_pytorch_n4@t1`, `degenerate_dot_rank0_f64_pytorch_n1024@t1`, `erased_construct_f64_col_n65536@t1`, `two_operand_einsum_f64_2x2_string@t4`, `two_operand_einsum_f64_2x2_dot-general@t4`, `two_operand_einsum_f64_16x16_prepared@t4`, `two_operand_einsum_c64_2x2_prepared@t4`, `two_operand_einsum_c64_16x16_string@t4`, `decode_proj_f32_faer-direct_in1024_out1024_len64@t4`, `decode_proj_f32_faer-direct_in1024_out1024_len512@t4`, `decode_proj_f32_eager-shared_in1024_out4096_len64@t4`, `decode_proj_f32_faer-direct_in1024_out4096_len64@t4`, `decode_proj_f32_pytorch_in1024_out4096_len64@t4`, `decode_proj_f32_faer-direct_in1024_out4096_len512@t4`, `conj_dot_c64_pytorch-vdot_len100@t4`, `small_solve_f64_triangular-solve-loop_k64_b64@t4`, `tanh_chain_f32_compiled-prepared_k10_1024x64@t4`, `tanh_chain_f32_compiled-prepared_k100_1024x64@t4`, `tanh_chain_f32_compiled-prepared_k200_1024x64@t4`, `layer_norm_f32_eager-single-call_d1024_len64_b1@t4`, `layer_norm_f32_eager-single-call_d1024_len64_b8@t4`, `rms_norm_f32_eager-composed_d1024_len8_b1@t4`, `rms_norm_f32_eager-single-call_d1024_len64_b1@t4` (+51 more)

Numerical checks: **493/681 recorded rows passed** (case × thread count).

## Steady-state operation rows

Each batch is one wall-clock interval over many operations; every output is retained until the clock stops. Median and inclusive IQR are per operation, in nanoseconds; CoV is sample standard deviation / mean (NOISY means CoV > 10%, a descriptive label only). Setup (inputs, backends, runtimes, session entry, tracing/compilation, preparation) is outside timing; faer-direct and host-sgemm write into a preallocated output, the tenferro and PyTorch arms allocate their output. The thin_complex_gemm faer-direct arm instead allocates its output inside timing to match its product references. `reference_for` in the case file names the case each reference arm is compared with. GPU batches end with one device synchronize (a per-call synchronize where the transfer is the operation); the Threads column is the host thread setting.

### Compute- and bandwidth-bound operations

| Case ID | Issues | Arm | Backend | Dtype | Workload | Threads | Ops/batch | Median ns/op | IQR ns | CoV | Status |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| softmax_f64_pool_serial_1024x4096 | #2273 #2274 #2276 | serial | tenferro-rs | f64 | rows=1024, cols=4096, axis=0 | 1 | 1 | 14955666.0 | 527417.5 | 2.6% | ok |
| softmax_f64_pool_pytorch_t1_1024x4096 | #2273 #2274 #2276 | pytorch | pytorch-cpu | f64 | rows=1024, cols=4096, axis=0 | 1 | 1 | 16320000.0 | 812667.0 | 4.6% | ok |
| softmax_f64_pool_pool4_1024x4096 | #2273 #2274 #2276 | pool4 | tenferro-rs | f64 | rows=1024, cols=4096, axis=0 | 4 | 1 | 3794958.0 | 242625.0 | 44.8% | NOISY |
| softmax_f64_pool_pytorch_t4_1024x4096 | #2273 #2274 #2276 | pytorch | pytorch-cpu | f64 | rows=1024, cols=4096, axis=0 | 4 | 1 | 6753167.0 | 2165292.0 | 22.4% | NOISY |

### Latency and per-call overhead

These rows measure dispatch/operation latency; they do not measure compute throughput.

| Case ID | Issues | Arm | Backend | Dtype | Workload | Threads | Ops/batch | Median ns/op | IQR ns | CoV | Status |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| two_operand_einsum_f64_2x2_string | #2325 #2324 | string | tenferro-rs | f64 | m=2, k=2, n=2, format=ij,jk->ik | 1 | 1024 | 1842.1 | 322.9 | 10.1% | NOISY |
| two_operand_einsum_f64_2x2_subscripts | #2325 #2324 | subscripts | tenferro-rs | f64 | m=2, k=2, n=2, format=ij,jk->ik | 1 | 2048 | 1064.6 | 106.7 | 6.5% | ok |
| two_operand_einsum_f64_2x2_prepared | #2325 #2324 | prepared | tenferro-rs | f64 | m=2, k=2, n=2, format=ij,jk->ik | 1 | 4096 | 784.3 | 146.5 | 10.4% | NOISY |
| two_operand_einsum_f64_2x2_dot-general | #2325 #2324 | dot-general | tenferro-rs | f64 | m=2, k=2, n=2, format=ij,jk->ik | 1 | 4096 | 797.6 | 8.0 | 1.1% | ok |
| two_operand_einsum_f64_2x2_pytorch | #2325 #2324 | pytorch | pytorch-cpu | f64 | m=2, k=2, n=2, format=ij,jk->ik | 1 | 256 | 8029.3 | 784.9 | 5.8% | ok |
| two_operand_einsum_f64_16x16_string | #2325 #2324 | string | tenferro-rs | f64 | m=16, k=16, n=16, format=ij,jk->ik | 1 | 1024 | 2602.9 | 199.4 | 5.4% | ok |
| two_operand_einsum_f64_16x16_subscripts | #2325 #2324 | subscripts | tenferro-rs | f64 | m=16, k=16, n=16, format=ij,jk->ik | 1 | 2048 | 1606.5 | 65.1 | 3.4% | ok |
| two_operand_einsum_f64_16x16_prepared | #2325 #2324 | prepared | tenferro-rs | f64 | m=16, k=16, n=16, format=ij,jk->ik | 1 | 2048 | 1333.8 | 61.4 | 12.3% | NOISY |
| two_operand_einsum_f64_16x16_dot-general | #2325 #2324 | dot-general | tenferro-rs | f64 | m=16, k=16, n=16, format=ij,jk->ik | 1 | 2048 | 1350.1 | 51.9 | 2.8% | ok |
| two_operand_einsum_f64_16x16_pytorch | #2325 #2324 | pytorch | pytorch-cpu | f64 | m=16, k=16, n=16, format=ij,jk->ik | 1 | 256 | 10927.6 | 567.1 | 4.7% | ok |
| two_operand_einsum_c64_2x2_string | #2325 #2324 | string | tenferro-rs | c64 | m=2, k=2, n=2, format=ij,jk->ik | 1 | 1024 | 1910.0 | 116.6 | 4.8% | ok |
| two_operand_einsum_c64_2x2_subscripts | #2325 #2324 | subscripts | tenferro-rs | c64 | m=2, k=2, n=2, format=ij,jk->ik | 1 | 2048 | 1168.0 | 114.6 | 5.0% | ok |
| two_operand_einsum_c64_2x2_prepared | #2325 #2324 | prepared | tenferro-rs | c64 | m=2, k=2, n=2, format=ij,jk->ik | 1 | 2048 | 962.8 | 105.3 | 6.7% | ok |
| two_operand_einsum_c64_2x2_dot-general | #2325 #2324 | dot-general | tenferro-rs | c64 | m=2, k=2, n=2, format=ij,jk->ik | 1 | 2048 | 1058.8 | 56.2 | 4.7% | ok |
| two_operand_einsum_c64_2x2_pytorch | #2325 #2324 | pytorch | pytorch-cpu | c64 | m=2, k=2, n=2, format=ij,jk->ik | 1 | 512 | 7936.1 | 709.3 | 7.2% | ok |
| two_operand_einsum_c64_16x16_string | #2325 #2324 | string | tenferro-rs | c64 | m=16, k=16, n=16, format=ij,jk->ik | 1 | 512 | 4026.3 | 117.2 | 2.8% | ok |
| two_operand_einsum_c64_16x16_subscripts | #2325 #2324 | subscripts | tenferro-rs | c64 | m=16, k=16, n=16, format=ij,jk->ik | 1 | 1024 | 3092.0 | 123.6 | 2.8% | ok |
| two_operand_einsum_c64_16x16_prepared | #2325 #2324 | prepared | tenferro-rs | c64 | m=16, k=16, n=16, format=ij,jk->ik | 1 | 1024 | 2622.5 | 190.9 | 4.5% | ok |
| two_operand_einsum_c64_16x16_dot-general | #2325 #2324 | dot-general | tenferro-rs | c64 | m=16, k=16, n=16, format=ij,jk->ik | 1 | 1024 | 2676.0 | 145.9 | 3.9% | ok |
| two_operand_einsum_c64_16x16_pytorch | #2325 #2324 | pytorch | pytorch-cpu | c64 | m=16, k=16, n=16, format=ij,jk->ik | 1 | 256 | 11568.0 | 307.0 | 2.2% | ok |
| two_operand_einsum_f64_abc_cef_abef_string | #2325 #2324 | string | tenferro-rs | f64 | m=8, k=2, n=16, format=abc,cef->abef | 1 | 1024 | 2462.8 | 109.7 | 5.0% | ok |
| two_operand_einsum_f64_abc_cef_abef_subscripts | #2325 #2324 | subscripts | tenferro-rs | f64 | m=8, k=2, n=16, format=abc,cef->abef | 1 | 2048 | 1317.7 | 166.5 | 7.2% | ok |
| two_operand_einsum_f64_abc_cef_abef_prepared | #2325 #2324 | prepared | tenferro-rs | f64 | m=8, k=2, n=16, format=abc,cef->abef | 1 | 2048 | 1038.8 | 128.6 | 7.2% | ok |
| two_operand_einsum_f64_abc_cef_abef_dot-general | #2325 #2324 | dot-general | tenferro-rs | f64 | m=8, k=2, n=16, format=abc,cef->abef | 1 | 2048 | 1003.2 | 99.3 | 7.7% | ok |
| two_operand_einsum_f64_abc_cef_abef_pytorch | #2325 #2324 | pytorch | pytorch-cpu | f64 | m=8, k=2, n=16, format=abc,cef->abef | 1 | 256 | 8819.3 | 696.0 | 4.4% | ok |
| thin_c64_gemm_1x141_141x282_t1 | #2416 #1099 | dot-general | tenferro-rs | c64 | m=1, k=141, n=282 | 1 | 128 | 24621.7 | 512.0 | 1.6% | ok |
| thin_c64_gemm_1x141_141x282_t1_faer-direct | #2416 #1099 | faer-direct | faer-direct | c64 | m=1, k=141, n=282 | 1 | 128 | 22038.1 | 641.8 | 4.9% | ok |
| thin_c64_gemm_1x141_141x282_t1_pytorch | #2416 #1099 | pytorch | pytorch-cpu | c64 | m=1, k=141, n=282 | 1 | 64 | 40627.0 | 1667.3 | 3.1% | ok |
| thin_c64_gemm_2x141_141x282_t1 | #2416 #1099 | dot-general | tenferro-rs | c64 | m=2, k=141, n=282 | 1 | 64 | 46844.4 | 4092.1 | 4.8% | ok |
| thin_c64_gemm_2x141_141x282_t1_faer-direct | #2416 #1099 | faer-direct | faer-direct | c64 | m=2, k=141, n=282 | 1 | 128 | 27505.9 | 2464.0 | 5.8% | ok |
| thin_c64_gemm_2x141_141x282_t1_pytorch | #2416 #1099 | pytorch | pytorch-cpu | c64 | m=2, k=141, n=282 | 1 | 64 | 38466.2 | 988.0 | 2.8% | ok |
| thin_c64_gemm_8x141_141x282_t1 | #2416 #1099 | dot-general | tenferro-rs | c64 | m=8, k=141, n=282 | 1 | 32 | 121197.9 | 4158.2 | 2.6% | ok |
| thin_c64_gemm_8x141_141x282_t1_faer-direct | #2416 #1099 | faer-direct | faer-direct | c64 | m=8, k=141, n=282 | 1 | 32 | 95446.6 | 9979.2 | 6.1% | ok |
| thin_c64_gemm_8x141_141x282_t1_pytorch | #2416 #1099 | pytorch | pytorch-cpu | c64 | m=8, k=141, n=282 | 1 | 64 | 44268.9 | 2392.6 | 4.3% | ok |
| decode_proj_f32_eager-shared_in1024_out1024_len8 | #1992 #2003 #1995 #2229 | eager-shared | tenferro-rs | f32 | in=1024, out=1024, len=8 | 1 | 32 | 112878.9 | 9951.8 | 6.9% | ok |
| decode_proj_f32_faer-direct_in1024_out1024_len8 | #1992 #2003 #1995 #2229 | faer-direct | faer-direct | f32 | in=1024, out=1024, len=8 | 1 | 8 | 362786.5 | 10833.3 | 6.6% | ok |
| decode_proj_f32_host-sgemm_in1024_out1024_len8 | #1992 #2003 #1995 #2229 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=1024, len=8 | 1 | 8 | 403125.0 | 15625.0 | 3.7% | ok |
| decode_proj_f32_pytorch_in1024_out1024_len8 | #1992 #2003 #1995 #2229 | pytorch | pytorch-cpu | f32 | in=1024, out=1024, len=8 | 1 | 32 | 119671.9 | 7052.7 | 5.2% | ok |
| decode_proj_f32_eager-shared_in1024_out1024_len64 | #1992 #2003 #1995 #2229 | eager-shared | tenferro-rs | f32 | in=1024, out=1024, len=64 | 1 | 16 | 168585.9 | 2050.8 | 4.9% | ok |
| decode_proj_f32_faer-direct_in1024_out1024_len64 | #1992 #2003 #1995 #2229 | faer-direct | faer-direct | f32 | in=1024, out=1024, len=64 | 1 | 2 | 1579312.5 | 159583.2 | 6.5% | ok |
| decode_proj_f32_host-sgemm_in1024_out1024_len64 | #1992 #2003 #1995 #2229 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=1024, len=64 | 1 | 2 | 1578604.0 | 64719.0 | 3.5% | ok |
| decode_proj_f32_pytorch_in1024_out1024_len64 | #1992 #2003 #1995 #2229 | pytorch | pytorch-cpu | f32 | in=1024, out=1024, len=64 | 1 | 16 | 168921.9 | 23864.6 | 9.1% | ok |
| decode_proj_f32_eager-shared_in1024_out1024_len512 | #1992 #2003 #1995 #2229 | eager-shared | tenferro-rs | f32 | in=1024, out=1024, len=512 | 1 | 2 | 896062.5 | 53906.0 | 5.5% | ok |
| decode_proj_f32_faer-direct_in1024_out1024_len512 | #1992 #2003 #1995 #2229 | faer-direct | faer-direct | f32 | in=1024, out=1024, len=512 | 1 | 1 | 12585750.0 | 781855.0 | 3.9% | ok |
| decode_proj_f32_host-sgemm_in1024_out1024_len512 | #1992 #2003 #1995 #2229 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=1024, len=512 | 1 | 1 | 12348291.0 | 900937.0 | 4.3% | ok |
| decode_proj_f32_pytorch_in1024_out1024_len512 | #1992 #2003 #1995 #2229 | pytorch | pytorch-cpu | f32 | in=1024, out=1024, len=512 | 1 | 2 | 925104.5 | 28947.8 | 2.8% | ok |
| decode_proj_f32_eager-shared_in1024_out4096_len8 | #1992 #2003 #1995 #2229 | eager-shared | tenferro-rs | f32 | in=1024, out=4096, len=8 | 1 | 4 | 623125.0 | 41927.0 | 8.7% | ok |
| decode_proj_f32_faer-direct_in1024_out4096_len8 | #1992 #2003 #1995 #2229 | faer-direct | faer-direct | f32 | in=1024, out=4096, len=8 | 1 | 2 | 1570458.5 | 117687.5 | 4.6% | ok |
| decode_proj_f32_host-sgemm_in1024_out4096_len8 | #1992 #2003 #1995 #2229 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=4096, len=8 | 1 | 2 | 1619458.5 | 65479.2 | 3.3% | ok |
| decode_proj_f32_pytorch_in1024_out4096_len8 | #1992 #2003 #1995 #2229 | pytorch | pytorch-cpu | f32 | in=1024, out=4096, len=8 | 1 | 4 | 620750.0 | 31577.9 | 3.7% | ok |
| decode_proj_f32_eager-shared_in1024_out4096_len64 | #1992 #2003 #1995 #2229 | eager-shared | tenferro-rs | f32 | in=1024, out=4096, len=64 | 1 | 2 | 882104.5 | 46385.2 | 3.9% | ok |
| decode_proj_f32_faer-direct_in1024_out4096_len64 | #1992 #2003 #1995 #2229 | faer-direct | faer-direct | f32 | in=1024, out=4096, len=64 | 1 | 1 | 7287000.0 | 718896.0 | 5.8% | ok |
| decode_proj_f32_host-sgemm_in1024_out4096_len64 | #1992 #2003 #1995 #2229 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=4096, len=64 | 1 | 1 | 7471542.0 | 372438.0 | 4.0% | ok |
| decode_proj_f32_pytorch_in1024_out4096_len64 | #1992 #2003 #1995 #2229 | pytorch | pytorch-cpu | f32 | in=1024, out=4096, len=64 | 1 | 4 | 939541.8 | 63786.5 | 5.8% | ok |
| decode_proj_f32_eager-shared_in1024_out4096_len512 | #1992 #2003 #1995 #2229 | eager-shared | tenferro-rs | f32 | in=1024, out=4096, len=512 | 1 | 1 | 3909000.0 | 181021.0 | 4.1% | ok |
| decode_proj_f32_faer-direct_in1024_out4096_len512 | #1992 #2003 #1995 #2229 | faer-direct | faer-direct | f32 | in=1024, out=4096, len=512 | 1 | 1 | 48269333.0 | 1439979.5 | 3.0% | ok |
| decode_proj_f32_host-sgemm_in1024_out4096_len512 | #1992 #2003 #1995 #2229 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=4096, len=512 | 1 | 1 | 48106541.0 | 2838187.5 | 4.2% | ok |
| decode_proj_f32_pytorch_in1024_out4096_len512 | #1992 #2003 #1995 #2229 | pytorch | pytorch-cpu | f32 | in=1024, out=4096, len=512 | 1 | 1 | 3669750.0 | 63312.5 | 1.4% | ok |
| decode_proj_view_operand_f32_prepared_in1024_out1024_len8_t1 | #2216 | prepared-view | tenferro-rs | f32 | in=1024, out=1024, len=8 | 1 | 32 | 91738.3 | 747.4 | 0.5% | ok |
| decode_proj_contiguous_operand_f32_prepared_in1024_out1024_len8_t1 | #2216 | prepared-contiguous | tenferro-rs | f32 | in=1024, out=1024, len=8 | 1 | 16 | 121919.2 | 1727.9 | 1.1% | ok |
| decode_proj_view_operand_f32_pytorch_in1024_out1024_len8_t1 | #2216 | pytorch | pytorch-cpu | f32 | in=1024, out=1024, len=8 | 1 | 32 | 105381.5 | 552.7 | 1.8% | ok |
| gemm_c64_mm256_t1 | #1900 | dot-general-shared | tenferro-rs | c64 | n=256 | 1 | 4 | 550781.2 | 11374.9 | 2.5% | ok |
| gemm_f64_mm256_t1 | #1900 | dot-general-shared | tenferro-rs | f64 | n=256 | 1 | 16 | 97848.9 | 1006.5 | 0.8% | ok |
| conj_dot_f64_dot-general-rank1_len100 | #1615 | dot-general-rank1 | tenferro-rs | f64 | len=100 | 1 | 4096 | 888.1 | 81.6 | 6.8% | ok |
| conj_dot_f64_portable-loop_len100 | #1615 | portable-loop | rust-loop | f64 | len=100 | 1 | 65536 | 54.7 | 4.5 | 5.1% | ok |
| conj_dot_f64_pytorch-vdot_len100 | #1615 | pytorch-vdot | pytorch-cpu | f64 | len=100 | 1 | 4096 | 764.6 | 68.9 | 7.4% | ok |
| conj_dot_f64_dot-general-rank1_len10000 | #1615 | dot-general-rank1 | tenferro-rs | f64 | len=10000 | 1 | 256 | 14351.4 | 1359.8 | 5.6% | ok |
| conj_dot_f64_portable-loop_len10000 | #1615 | portable-loop | rust-loop | f64 | len=10000 | 1 | 256 | 10623.2 | 915.1 | 5.6% | ok |
| conj_dot_f64_pytorch-vdot_len10000 | #1615 | pytorch-vdot | pytorch-cpu | f64 | len=10000 | 1 | 2048 | 1305.4 | 14.5 | 0.8% | ok |
| conj_dot_f64_dot-general-rank1_len1000000 | #1615 | dot-general-rank1 | tenferro-rs | f64 | len=1000000 | 1 | 2 | 1243104.0 | 174614.5 | 8.9% | ok |
| conj_dot_f64_portable-loop_len1000000 | #1615 | portable-loop | rust-loop | f64 | len=1000000 | 1 | 4 | 931083.5 | 43921.9 | 2.8% | ok |
| conj_dot_f64_pytorch-vdot_len1000000 | #1615 | pytorch-vdot | pytorch-cpu | f64 | len=1000000 | 1 | 16 | 134903.6 | 7962.3 | 4.9% | ok |
| conj_dot_c64_dot-general-rank1_len100 | #1615 | dot-general-rank1 | tenferro-rs | c64 | len=100 | 1 | 2048 | 1233.7 | 187.0 | 8.9% | ok |
| conj_dot_c64_portable-loop_len100 | #1615 | portable-loop | rust-loop | c64 | len=100 | 1 | 32768 | 88.9 | 10.0 | 6.9% | ok |
| conj_dot_c64_pytorch-vdot_len100 | #1615 | pytorch-vdot | pytorch-cpu | c64 | len=100 | 1 | 4096 | 713.2 | 82.6 | 221.0% | NOISY |
| conj_dot_c64_dot-general-rank1_len10000 | #1615 | dot-general-rank1 | tenferro-rs | c64 | len=10000 | 1 | 64 | 57750.6 | 4525.1 | 5.3% | ok |
| conj_dot_c64_portable-loop_len10000 | #1615 | portable-loop | rust-loop | c64 | len=10000 | 1 | 256 | 11042.0 | 683.2 | 5.6% | ok |
| conj_dot_c64_pytorch-vdot_len10000 | #1615 | pytorch-vdot | pytorch-cpu | c64 | len=10000 | 1 | 512 | 5101.6 | 749.5 | 7.2% | ok |
| conj_dot_c64_dot-general-rank1_len1000000 | #1615 | dot-general-rank1 | tenferro-rs | c64 | len=1000000 | 1 | 1 | 5233208.0 | 286771.0 | 3.7% | ok |
| conj_dot_c64_portable-loop_len1000000 | #1615 | portable-loop | rust-loop | c64 | len=1000000 | 1 | 4 | 1059781.2 | 104651.0 | 6.1% | ok |
| conj_dot_c64_pytorch-vdot_len1000000 | #1615 | pytorch-vdot | pytorch-cpu | c64 | len=1000000 | 1 | 4 | 661437.5 | 48817.8 | 5.2% | ok |
| small_solve_f64_solve-loop_k16_b64 | #2007 #1878 | solve-loop | tenferro-rs | f64 | k=16, batch=64 | 1 | 16 | 122611.9 | 2976.6 | 3.8% | ok |
| small_solve_f64_triangular-solve-loop_k16_b64 | #2007 #1878 | triangular-solve-loop | tenferro-rs | f64 | k=16, batch=64 | 1 | 32 | 56289.1 | 5001.3 | 6.0% | ok |
| small_solve_f64_solve-batched_k16_b64 | #2007 #1878 | solve-batched | tenferro-rs | f64 | k=16, batch=64 | 1 | 32 | 79635.4 | 5171.2 | 6.2% | ok |
| small_solve_f64_pytorch-solve-batched_k16_b64 | #2007 #1878 | pytorch-solve-batched | pytorch-cpu | f64 | k=16, batch=64 | 1 | 32 | 89257.8 | 2078.1 | 2.3% | ok |
| small_solve_f64_pytorch-solve-triangular-batched_k16_b64 | #2007 #1878 | pytorch-solve-triangular-batched | pytorch-cpu | f64 | k=16, batch=64 | 1 | 64 | 23787.1 | 2667.6 | 7.3% | ok |
| small_solve_f64_solve-loop_k64_b64 | #2007 #1878 | solve-loop | tenferro-rs | f64 | k=64, batch=64 | 1 | 2 | 1585646.0 | 33406.5 | 1.5% | ok |
| small_solve_f64_triangular-solve-loop_k64_b64 | #2007 #1878 | triangular-solve-loop | tenferro-rs | f64 | k=64, batch=64 | 1 | 4 | 308250.0 | 7104.2 | 1.9% | ok |
| small_solve_f64_solve-batched_k64_b64 | #2007 #1878 | solve-batched | tenferro-rs | f64 | k=64, batch=64 | 1 | 2 | 1515604.0 | 109916.5 | 6.2% | ok |
| small_solve_f64_pytorch-solve-batched_k64_b64 | #2007 #1878 | pytorch-solve-batched | pytorch-cpu | f64 | k=64, batch=64 | 1 | 2 | 1414771.0 | 90218.5 | 4.8% | ok |
| small_solve_f64_pytorch-solve-triangular-batched_k64_b64 | #2007 #1878 | pytorch-solve-triangular-batched | pytorch-cpu | f64 | k=64, batch=64 | 1 | 8 | 283187.5 | 12164.0 | 3.0% | ok |
| tanh_chain_f32_compiled-prepared_k10_1024x64 | #1990 | compiled-prepared | tenferro-rs | f32 | rows=1024, cols=64, chain=10 | 1 | 1 | 2183375.0 | 140104.0 | 5.2% | ok |
| tanh_chain_f32_eager-shared_k10_1024x64 | #1990 | eager-shared | tenferro-rs | f32 | rows=1024, cols=64, chain=10 | 1 | 1 | 2238584.0 | 249271.0 | 15.9% | NOISY |
| tanh_chain_f32_compiled-prepared_k100_1024x64 | #1990 | compiled-prepared | tenferro-rs | f32 | rows=1024, cols=64, chain=100 | 1 | 1 | 21436792.0 | 1184812.0 | 4.9% | ok |
| tanh_chain_f32_eager-shared_k100_1024x64 | #1990 | eager-shared | tenferro-rs | f32 | rows=1024, cols=64, chain=100 | 1 | 1 | 22420333.0 | 981936.5 | 4.5% | ok |
| tanh_chain_f32_compiled-prepared_k200_1024x64 | #1990 | compiled-prepared | tenferro-rs | f32 | rows=1024, cols=64, chain=200 | 1 | 1 | 43429709.0 | 3159562.5 | 7.2% | ok |
| tanh_chain_f32_eager-shared_k200_1024x64 | #1990 | eager-shared | tenferro-rs | f32 | rows=1024, cols=64, chain=200 | 1 | 1 | 44722500.0 | 1531334.0 | 3.4% | ok |
| layer_norm_f32_eager-composed_d1024_len8_b1 | #2006 | eager-composed | tenferro-rs | f32 | norm=layer_norm, d=1024, len=8, batch=1 | 1 | 64 | 39650.4 | 8907.2 | 28.4% | NOISY |
| layer_norm_f32_pytorch-fused_d1024_len8_b1 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=layer_norm, d=1024, len=8, batch=1 | 1 | 256 | 6439.3 | 2659.3 | 26.4% | NOISY |
| layer_norm_f32_eager-single-call_d1024_len8_b1 | #2006 | eager-single-call | tenferro-rs | f32 | norm=layer_norm, d=1024, len=8, batch=1 | 1 | 64 | 41862.6 | 5562.8 | 8.9% | ok |
| layer_norm_f32_eager-composed_d1024_len8_b8 | #2006 | eager-composed | tenferro-rs | f32 | norm=layer_norm, d=1024, len=8, batch=8 | 1 | 32 | 76660.2 | 8082.7 | 10.5% | NOISY |
| layer_norm_f32_pytorch-fused_d1024_len8_b8 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=layer_norm, d=1024, len=8, batch=8 | 1 | 64 | 30151.7 | 1063.8 | 2.9% | ok |
| layer_norm_f32_eager-single-call_d1024_len8_b8 | #2006 | eager-single-call | tenferro-rs | f32 | norm=layer_norm, d=1024, len=8, batch=8 | 1 | 32 | 83833.3 | 7103.5 | 17.7% | NOISY |
| layer_norm_f32_eager-composed_d1024_len64_b1 | #2006 | eager-composed | tenferro-rs | f32 | norm=layer_norm, d=1024, len=64, batch=1 | 1 | 32 | 82867.2 | 9328.2 | 7.4% | ok |
| layer_norm_f32_pytorch-fused_d1024_len64_b1 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=layer_norm, d=1024, len=64, batch=1 | 1 | 64 | 33653.6 | 3387.4 | 6.8% | ok |
| layer_norm_f32_eager-single-call_d1024_len64_b1 | #2006 | eager-single-call | tenferro-rs | f32 | norm=layer_norm, d=1024, len=64, batch=1 | 1 | 16 | 89177.1 | 6505.2 | 8.7% | ok |
| layer_norm_f32_eager-composed_d1024_len64_b8 | #2006 | eager-composed | tenferro-rs | f32 | norm=layer_norm, d=1024, len=64, batch=8 | 1 | 4 | 406093.8 | 19291.6 | 3.5% | ok |
| layer_norm_f32_pytorch-fused_d1024_len64_b8 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=layer_norm, d=1024, len=64, batch=8 | 1 | 8 | 238614.6 | 7796.8 | 4.8% | ok |
| layer_norm_f32_eager-single-call_d1024_len64_b8 | #2006 | eager-single-call | tenferro-rs | f32 | norm=layer_norm, d=1024, len=64, batch=8 | 1 | 4 | 399302.0 | 37359.1 | 7.5% | ok |
| rms_norm_f32_eager-composed_d1024_len8_b1 | #2006 | eager-composed | tenferro-rs | f32 | norm=rms_norm, d=1024, len=8, batch=1 | 1 | 128 | 21219.7 | 1153.3 | 4.6% | ok |
| rms_norm_f32_pytorch-fused_d1024_len8_b1 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=rms_norm, d=1024, len=8, batch=1 | 1 | 256 | 10845.1 | 383.7 | 2.7% | ok |
| rms_norm_f32_eager-single-call_d1024_len8_b1 | #2006 | eager-single-call | tenferro-rs | f32 | norm=rms_norm, d=1024, len=8, batch=1 | 1 | 128 | 26316.7 | 2642.7 | 32.6% | NOISY |
| rms_norm_f32_eager-composed_d1024_len8_b8 | #2006 | eager-composed | tenferro-rs | f32 | norm=rms_norm, d=1024, len=8, batch=8 | 1 | 32 | 43122.4 | 2248.7 | 3.7% | ok |
| rms_norm_f32_pytorch-fused_d1024_len8_b8 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=rms_norm, d=1024, len=8, batch=8 | 1 | 64 | 33524.7 | 1475.6 | 3.8% | ok |
| rms_norm_f32_eager-single-call_d1024_len8_b8 | #2006 | eager-single-call | tenferro-rs | f32 | norm=rms_norm, d=1024, len=8, batch=8 | 1 | 32 | 46955.7 | 2356.8 | 4.1% | ok |
| rms_norm_f32_eager-composed_d1024_len64_b1 | #2006 | eager-composed | tenferro-rs | f32 | norm=rms_norm, d=1024, len=64, batch=1 | 1 | 32 | 42462.2 | 4601.5 | 9.5% | ok |
| rms_norm_f32_pytorch-fused_d1024_len64_b1 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=rms_norm, d=1024, len=64, batch=1 | 1 | 64 | 34460.9 | 3115.9 | 9.1% | ok |
| rms_norm_f32_eager-single-call_d1024_len64_b1 | #2006 | eager-single-call | tenferro-rs | f32 | norm=rms_norm, d=1024, len=64, batch=1 | 1 | 32 | 42704.4 | 1299.5 | 5.1% | ok |
| rms_norm_f32_eager-composed_d1024_len64_b8 | #2006 | eager-composed | tenferro-rs | f32 | norm=rms_norm, d=1024, len=64, batch=8 | 1 | 8 | 189229.1 | 11791.7 | 5.9% | ok |
| rms_norm_f32_pytorch-fused_d1024_len64_b8 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=rms_norm, d=1024, len=64, batch=8 | 1 | 8 | 179114.6 | 7755.1 | 3.7% | ok |
| rms_norm_f32_eager-single-call_d1024_len64_b8 | #2006 | eager-single-call | tenferro-rs | f32 | norm=rms_norm, d=1024, len=64, batch=8 | 1 | 8 | 198348.9 | 5721.4 | 3.0% | ok |
| small_contraction_abcd-dbef-acef_f64_shared-session-default-pool_d4 | #1885 #1897 #1901 | shared-session-default-pool | tenferro-rs | f64 | extent=4 | 1 | 1024 | 2491.8 | 52.7 | 3.6% | ok |
| activation_erf_f32_eager-single-call_1024x64 | #1975 #2069 #2032 | eager-single-call | tenferro-rs | f32 | op=erf, rows=1024, cols=64 | 1 | — | — | — | — | SKIPPED |
| activation_erf_f32_session-single-call_1024x64 | #1975 #2069 #2032 | session-single-call | tenferro-rs | f32 | op=erf, rows=1024, cols=64 | 1 | 8 | 414843.8 | 4901.1 | 1.6% | ok |
| activation_erf_f32_pytorch_1024x64 | #1975 #2069 #2032 | pytorch | pytorch-cpu | f32 | op=erf, rows=1024, cols=64 | 1 | — | — | — | — | SKIPPED |
| activation_sigmoid_f32_eager-single-call_1024x64 | #1975 #2069 #2032 | eager-single-call | tenferro-rs | f32 | op=sigmoid, rows=1024, cols=64 | 1 | — | — | — | — | SKIPPED |
| activation_sigmoid_f32_session-single-call_1024x64 | #1975 #2069 #2032 | session-single-call | tenferro-rs | f32 | op=sigmoid, rows=1024, cols=64 | 1 | 32 | 48235.7 | 2277.4 | 3.5% | ok |
| activation_sigmoid_f32_eager-composed_1024x64 | #1975 #2069 #2032 | eager-composed | tenferro-rs | f32 | op=sigmoid, rows=1024, cols=64 | 1 | 32 | 72800.8 | 2274.7 | 3.7% | ok |
| activation_sigmoid_f32_pytorch_1024x64 | #1975 #2069 #2032 | pytorch | pytorch-cpu | f32 | op=sigmoid, rows=1024, cols=64 | 1 | — | — | — | — | SKIPPED |
| activation_silu_f32_eager-single-call_1024x64 | #1975 #2069 #2032 | eager-single-call | tenferro-rs | f32 | op=silu, rows=1024, cols=64 | 1 | — | — | — | — | SKIPPED |
| activation_silu_f32_session-single-call_1024x64 | #1975 #2069 #2032 | session-single-call | tenferro-rs | f32 | op=silu, rows=1024, cols=64 | 1 | 32 | 51174.5 | 1151.0 | 5.2% | ok |
| activation_silu_f32_eager-composed_1024x64 | #1975 #2069 #2032 | eager-composed | tenferro-rs | f32 | op=silu, rows=1024, cols=64 | 1 | 32 | 76971.3 | 3617.2 | 6.1% | ok |
| activation_silu_f32_pytorch_1024x64 | #1975 #2069 #2032 | pytorch | pytorch-cpu | f32 | op=silu, rows=1024, cols=64 | 1 | — | — | — | — | SKIPPED |
| activation_softplus_f32_eager-single-call_1024x64 | #1975 #2069 #2032 | eager-single-call | tenferro-rs | f32 | op=softplus, rows=1024, cols=64 | 1 | — | — | — | — | SKIPPED |
| activation_softplus_f32_session-single-call_1024x64 | #1975 #2069 #2032 | session-single-call | tenferro-rs | f32 | op=softplus, rows=1024, cols=64 | 1 | 16 | 213070.3 | 10451.8 | 12.7% | NOISY |
| activation_softplus_f32_eager-composed_1024x64 | #1975 #2069 #2032 | eager-composed | tenferro-rs | f32 | op=softplus, rows=1024, cols=64 | 1 | 16 | 182460.9 | 5072.9 | 3.3% | ok |
| activation_softplus_f32_pytorch_1024x64 | #1975 #2069 #2032 | pytorch | pytorch-cpu | f32 | op=softplus, rows=1024, cols=64 | 1 | — | — | — | — | SKIPPED |
| activation_gelu_f32_eager-single-call_1024x64 | #1975 #2030 #2069 #2228 | eager-single-call | tenferro-rs | f32 | op=gelu, rows=1024, cols=64 | 1 | 32 | 77854.2 | 4166.0 | 3.8% | ok |
| activation_gelu_f32_session-single-call_1024x64 | #1975 #2030 #2069 #2228 | session-single-call | tenferro-rs | f32 | op=gelu, rows=1024, cols=64 | 1 | 32 | 81725.3 | 4543.0 | 8.0% | ok |
| activation_gelu_f32_pytorch_1024x64 | #1975 #2030 #2069 #2228 | pytorch | pytorch-cpu | f32 | op=gelu, rows=1024, cols=64 | 1 | 32 | 84326.8 | 6588.5 | 5.2% | ok |
| activation_gelu_tanh_f32_eager-single-call_1024x64 | #1975 #2069 #2032 | eager-single-call | tenferro-rs | f32 | op=gelu_tanh, rows=1024, cols=64 | 1 | — | — | — | — | SKIPPED |
| activation_gelu_tanh_f32_session-single-call_1024x64 | #1975 #2069 #2032 | session-single-call | tenferro-rs | f32 | op=gelu_tanh, rows=1024, cols=64 | 1 | 8 | 248947.9 | 7848.9 | 2.6% | ok |
| activation_gelu_tanh_f32_eager-composed_1024x64 | #1975 #2069 #2032 | eager-composed | tenferro-rs | f32 | op=gelu_tanh, rows=1024, cols=64 | 1 | 8 | 309645.8 | 14255.2 | 3.5% | ok |
| activation_gelu_tanh_f32_pytorch_1024x64 | #1975 #2069 #2032 | pytorch | pytorch-cpu | f32 | op=gelu_tanh, rows=1024, cols=64 | 1 | — | — | — | — | SKIPPED |
| softmax_f32_eager-single-call_len64_b8 | #1976 #2031 #2231 | eager-single-call | tenferro-rs | f32 | op=softmax, len=64, batch=8, axis=0 | 1 | 64 | 33210.3 | 1556.6 | 6.6% | ok |
| softmax_f32_session-single-call_len64_b8 | #1976 #2031 #2231 | session-single-call | tenferro-rs | f32 | op=softmax, len=64, batch=8, axis=0 | 1 | 64 | 34011.1 | 1522.1 | 4.8% | ok |
| softmax_f32_eager-composed_len64_b8 | #1976 #2031 #2231 | eager-composed | tenferro-rs | f32 | op=softmax, len=64, batch=8, axis=0 | 1 | 32 | 70274.8 | 11981.1 | 21.3% | NOISY |
| softmax_f32_pytorch_len64_b8 | #1976 #2031 #2231 | pytorch | pytorch-cpu | f32 | op=softmax, len=64, batch=8, axis=0 | 1 | 64 | 46930.3 | 2197.3 | 4.3% | ok |
| softmax_f32_eager-single-call_len512_b8 | #1976 #2231 | eager-single-call | tenferro-rs | f32 | op=softmax, len=512, batch=8, axis=0 | 1 | 1 | 2489166.0 | 166146.5 | 4.1% | ok |
| softmax_f32_session-single-call_len512_b8 | #1976 #2231 | session-single-call | tenferro-rs | f32 | op=softmax, len=512, batch=8, axis=0 | 1 | 1 | 2570541.0 | 74063.0 | 2.1% | ok |
| softmax_f32_eager-composed_len512_b8 | #1976 #2231 | eager-composed | tenferro-rs | f32 | op=softmax, len=512, batch=8, axis=0 | 1 | 1 | 2618209.0 | 124624.5 | 4.4% | ok |
| softmax_f32_pytorch_len512_b8 | #1976 #2231 | pytorch | pytorch-cpu | f32 | op=softmax, len=512, batch=8, axis=0 | 1 | 1 | 3366667.0 | 211562.5 | 4.2% | ok |
| log_softmax_f32_eager-single-call_len64_b8 | #1976 | eager-single-call | tenferro-rs | f32 | op=log_softmax, len=64, batch=8, axis=0 | 1 | 32 | 81625.0 | 6402.3 | 4.9% | ok |
| log_softmax_f32_session-single-call_len64_b8 | #1976 | session-single-call | tenferro-rs | f32 | op=log_softmax, len=64, batch=8, axis=0 | 1 | 32 | 65638.0 | 2524.7 | 2.6% | ok |
| log_softmax_f32_eager-composed_len64_b8 | #1976 | eager-composed | tenferro-rs | f32 | op=log_softmax, len=64, batch=8, axis=0 | 1 | 32 | 70179.7 | 7456.4 | 6.0% | ok |
| log_softmax_f32_pytorch_len64_b8 | #1976 | pytorch | pytorch-cpu | f32 | op=log_softmax, len=64, batch=8, axis=0 | 1 | 64 | 42975.9 | 2757.5 | 4.2% | ok |
| log_softmax_f32_eager-single-call_len512_b8 | #1976 | eager-single-call | tenferro-rs | f32 | op=log_softmax, len=512, batch=8, axis=0 | 1 | 1 | 2640500.0 | 165000.0 | 4.9% | ok |
| log_softmax_f32_session-single-call_len512_b8 | #1976 | session-single-call | tenferro-rs | f32 | op=log_softmax, len=512, batch=8, axis=0 | 1 | 1 | 2613500.0 | 102791.5 | 4.0% | ok |
| log_softmax_f32_eager-composed_len512_b8 | #1976 | eager-composed | tenferro-rs | f32 | op=log_softmax, len=512, batch=8, axis=0 | 1 | 1 | 2537083.0 | 124687.0 | 3.2% | ok |
| log_softmax_f32_pytorch_len512_b8 | #1976 | pytorch | pytorch-cpu | f32 | op=log_softmax, len=512, batch=8, axis=0 | 1 | 1 | 2790458.0 | 163395.5 | 5.1% | ok |
| masked_softmax_f32_eager-single-call_len64_b8 | #1976 | eager-single-call | tenferro-rs | f32 | op=masked_softmax, len=64, batch=8, axis=0 | 1 | 32 | 106928.4 | 2742.9 | 3.4% | ok |
| masked_softmax_f32_session-single-call_len64_b8 | #1976 | session-single-call | tenferro-rs | f32 | op=masked_softmax, len=64, batch=8, axis=0 | 1 | 32 | 88244.8 | 2438.2 | 3.4% | ok |
| masked_softmax_f32_eager-composed_len64_b8 | #1976 | eager-composed | tenferro-rs | f32 | op=masked_softmax, len=64, batch=8, axis=0 | 1 | 32 | 75536.4 | 3948.6 | 5.6% | ok |
| masked_softmax_f32_pytorch_len64_b8 | #1976 | pytorch | pytorch-cpu | f32 | op=masked_softmax, len=64, batch=8, axis=0 | 1 | 32 | 69350.3 | 6585.9 | 5.7% | ok |
| masked_softmax_f32_eager-single-call_len512_b8 | #1976 | eager-single-call | tenferro-rs | f32 | op=masked_softmax, len=512, batch=8, axis=0 | 1 | 1 | 3762917.0 | 144437.5 | 4.1% | ok |
| masked_softmax_f32_session-single-call_len512_b8 | #1976 | session-single-call | tenferro-rs | f32 | op=masked_softmax, len=512, batch=8, axis=0 | 1 | 1 | 3943584.0 | 284749.5 | 5.0% | ok |
| masked_softmax_f32_eager-composed_len512_b8 | #1976 | eager-composed | tenferro-rs | f32 | op=masked_softmax, len=512, batch=8, axis=0 | 1 | 1 | 2661917.0 | 54625.5 | 2.2% | ok |
| masked_softmax_f32_pytorch_len512_b8 | #1976 | pytorch | pytorch-cpu | f32 | op=masked_softmax, len=512, batch=8, axis=0 | 1 | 1 | 4310416.0 | 91812.0 | 2.8% | ok |
| softmax_f32_eager-single-call_len64_b8_t1 | #2233 #2235 | eager-single-call | tenferro-rs | f32 | op=softmax, len=64, batch=8, axis=0 | 1 | 64 | 34817.1 | 1570.3 | 4.0% | ok |
| softmax_f32_eager-composed_len64_b8_t1 | #2233 #2235 | eager-composed | tenferro-rs | f32 | op=softmax, len=64, batch=8, axis=0 | 1 | 32 | 67904.9 | 1890.0 | 6.3% | ok |
| softmax_f32_pytorch_len64_b8_t1 | #2233 #2235 | pytorch | pytorch-cpu | f32 | op=softmax, len=64, batch=8, axis=0 | 1 | 64 | 46675.1 | 1095.1 | 2.0% | ok |
| reduce_mean_f32_eager-single-call_1024x64_axis0 | #1976 | eager-single-call | tenferro-rs | f32 | rows=1024, cols=64, axis=0 | 1 | 256 | 11435.5 | 335.5 | 4.1% | ok |
| reduce_mean_f32_session-single-call_1024x64_axis0 | #1976 | session-single-call | tenferro-rs | f32 | rows=1024, cols=64, axis=0 | 1 | 512 | 7180.7 | 844.6 | 8.5% | ok |
| reduce_mean_f32_eager-composed_1024x64_axis0 | #1976 | eager-composed | tenferro-rs | f32 | rows=1024, cols=64, axis=0 | 1 | 256 | 12644.4 | 1842.3 | 13.3% | NOISY |
| reduce_mean_f32_pytorch_1024x64_axis0 | #1976 | pytorch | pytorch-cpu | f32 | rows=1024, cols=64, axis=0 | 1 | 512 | 6743.5 | 471.8 | 10.5% | NOISY |
| take_along_axis_rows_f64_eager-single-call_n64_b256 | #2008 | eager-single-call | tenferro-rs | f64 | n=64, batch=256, axis=0 | 1 | 1 | 6408209.0 | 466666.0 | 5.7% | ok |
| take_along_axis_rows_f64_session-single-call_n64_b256 | #2008 | session-single-call | tenferro-rs | f64 | n=64, batch=256, axis=0 | 1 | 1 | 6091459.0 | 396812.0 | 4.8% | ok |
| take_along_axis_rows_f64_eager-gather-prebuilt_n64_b256 | #2008 | eager-gather-prebuilt | tenferro-rs | f64 | n=64, batch=256, axis=0 | 1 | 1 | 5975500.0 | 330166.5 | 13.2% | NOISY |
| take_along_axis_rows_f64_pytorch_n64_b256 | #2008 | pytorch | pytorch-cpu | f64 | n=64, batch=256, axis=0 | 1 | 1 | 2094750.0 | 169479.0 | 7.4% | ok |
| lstsq_f64_eager-shared_m768_n384_rhs16 | #2027 | eager-shared | tenferro-rs | f64 | operation=lstsq | 1 | 1 | 6740125.0 | 671625.5 | 6.6% | ok |
| lstsq_f64_pytorch_m768_n384_rhs16 | #2027 | pytorch | pytorch-cpu | f64 | operation=lstsq | 1 | 1 | 4446750.0 | 102374.5 | 2.9% | ok |
| triangular_f64_concrete-shared_n4096_rhs64 | #2028 | concrete-shared | tenferro-rs | f64 | operation=triangular | 1 | 1 | 13446291.0 | 667083.0 | 3.8% | ok |
| triangular_f64_pytorch_n4096_rhs64 | #2028 | pytorch | pytorch-cpu | f64 | operation=triangular | 1 | 1 | 9720333.0 | 774937.0 | 5.1% | ok |
| reshape_f64_concrete-shared_n33554432_to8192x4096 | #2029 | concrete-shared | tenferro-rs | f64 | operation=reshape | 1 | 1 | 6103417.0 | 238854.5 | 3.1% | ok |
| reshape_f64_pytorch_n33554432_to8192x4096 | #2029 | pytorch | pytorch-cpu | f64 | operation=reshape | 1 | 1 | 7488584.0 | 171750.0 | 4.6% | ok |
| metadata_transpose_view_f64_rank8_t1 | #2180 | metadata-clone | tenferro-rs | f64 | shape=[2, 2, 2, 2, 2, 2, 2, 2] | 1 | 16384 | 47.6 | 1.8 | 3.2% | ok |
| metadata_transpose_view_f64_rank8_t1_julia-base | #2180 | julia-base | julia-base | f64 | shape=[2, 2, 2, 2, 2, 2, 2, 2] | 1 | 1048576 | 0.4 | 0.1 | 9.8% | ok |
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
| typed_elemwise_f64_add_held_len16 | #2139 | tenferro | tenferro-rs | f64 | n=16, route=held | 1 | 4096 | 676.0 | 36.2 | 4.1% | ok |
| typed_elemwise_f64_add_held_len16_pytorch | #2139 | pytorch | pytorch-cpu | f64 | n=16, route=held | 1 | 4096 | 753.5 | 43.5 | 5.8% | ok |
| typed_elemwise_f64_add_into_held_len16 | #2139 | tenferro | tenferro-rs | f64 | n=16, route=into_held | 1 | 4096 | 502.6 | 23.0 | 4.0% | ok |
| typed_elemwise_f64_add_into_held_len16_pytorch | #2139 | pytorch | pytorch-cpu | f64 | n=16, route=into_held | 1 | 4096 | 813.0 | 46.5 | 3.5% | ok |
| typed_elemwise_f64_add_held_len256 | #2139 | tenferro | tenferro-rs | f64 | n=256, route=held | 1 | 4096 | 692.1 | 27.7 | 3.1% | ok |
| typed_elemwise_f64_add_held_len256_pytorch | #2139 | pytorch | pytorch-cpu | f64 | n=256, route=held | 1 | 2048 | 824.2 | 36.4 | 3.5% | ok |
| typed_elemwise_f64_add_into_held_len256 | #2139 | tenferro | tenferro-rs | f64 | n=256, route=into_held | 1 | 4096 | 526.7 | 24.1 | 2.7% | ok |
| typed_elemwise_f64_add_into_held_len256_pytorch | #2139 | pytorch | pytorch-cpu | f64 | n=256, route=into_held | 1 | 4096 | 808.3 | 13.4 | 1.8% | ok |
| typed_elemwise_f64_add_held_len4096 | #2139 | tenferro | tenferro-rs | f64 | n=4096, route=held | 1 | 512 | 1287.8 | 38.5 | 3.0% | ok |
| typed_elemwise_f64_add_held_len4096_pytorch | #2139 | pytorch | pytorch-cpu | f64 | n=4096, route=held | 1 | 512 | 1537.0 | 182.0 | 8.1% | ok |
| typed_elemwise_f64_add_into_held_len4096 | #2139 | tenferro | tenferro-rs | f64 | n=4096, route=into_held | 1 | 2048 | 1193.8 | 112.3 | 7.0% | ok |
| typed_elemwise_f64_add_into_held_len4096_pytorch | #2139 | pytorch | pytorch-cpu | f64 | n=4096, route=into_held | 1 | 2048 | 1558.2 | 97.1 | 5.1% | ok |
| tiny_matmul_2x2_typed_held | #2139 | tenferro | tenferro-rs | f64 | n=2 | 1 | 4096 | 812.7 | 31.2 | 2.7% | ok |
| tiny_matmul_2x2_typed_held_pytorch | #2139 | pytorch | pytorch-cpu | f64 | n=2 | 1 | 4096 | 855.0 | 53.9 | 4.8% | ok |
| analytic_neg_f64_sweep_8MiB_t1 | #2139 | tenferro | tenferro-rs | f64 | n=1048576 | 1 | 2 | 262708.0 | 115010.5 | 31.7% | NOISY |
| analytic_neg_f64_sweep_8MiB_t1_pytorch | #2139 | pytorch | pytorch-cpu | f64 | n=1048576 | 1 | 8 | 223932.2 | 21351.5 | 6.1% | ok |
| analytic_neg_f64_sweep_32MiB_t1 | #2139 | tenferro | tenferro-rs | f64 | n=4194304 | 1 | 2 | 896500.0 | 72041.8 | 4.6% | ok |
| analytic_neg_f64_sweep_32MiB_t1_pytorch | #2139 | pytorch | pytorch-cpu | f64 | n=4194304 | 1 | 2 | 1058104.0 | 42739.2 | 4.2% | ok |
| gemm_bound_f64_t1_t8_scaling | #2141 | tenferro | tenferro-rs | f64 | n=1536 | 1 | 1 | 28988625.0 | 806645.5 | 1.7% | ok |
| gemm_bound_f64_t1_t8_scaling_pytorch | #2141 | pytorch | pytorch-cpu | f64 | n=1536 | 1 | 1 | 25658792.0 | 2771209.0 | 5.2% | ok |
| lu_batched_f64_factor_n8_b1024 | #2142 | tenferro | tenferro-rs | f64 | n=8, batch=1024, op=factor | 1 | 8 | 298786.4 | 33151.1 | 7.8% | ok |
| lu_batched_f64_factor_n8_b1024_host | #2142 | host | rust-loop | f64 | n=8, batch=1024, op=factor | 1 | 8 | 256213.5 | 27151.0 | 6.4% | ok |
| lu_batched_f64_solve_n8_b1024 | #2142 | tenferro | tenferro-rs | f64 | n=8, batch=1024, op=solve | 1 | 32 | 98184.9 | 21117.2 | 12.2% | NOISY |
| lu_batched_f64_solve_n8_b1024_pytorch | #2142 | pytorch | pytorch-cpu | f64 | n=8, batch=1024, op=solve | 1 | 32 | 75565.1 | 5343.8 | 5.2% | ok |
| lu_batched_f64_factor_n16_b1024 | #2142 | tenferro | tenferro-rs | f64 | n=16, batch=1024, op=factor | 1 | 4 | 796666.5 | 42484.2 | 3.2% | ok |
| lu_batched_f64_factor_n16_b1024_host | #2142 | host | rust-loop | f64 | n=16, batch=1024, op=factor | 1 | 2 | 959937.5 | 74093.8 | 4.8% | ok |
| lu_batched_f64_solve_n16_b1024 | #2142 | tenferro | tenferro-rs | f64 | n=16, batch=1024, op=solve | 1 | 16 | 225872.4 | 24858.1 | 6.5% | ok |
| lu_batched_f64_solve_n16_b1024_pytorch | #2142 | pytorch | pytorch-cpu | f64 | n=16, batch=1024, op=solve | 1 | 16 | 178635.4 | 15671.9 | 5.1% | ok |
| axpby_accum_strided_dest_f64_block | #2143 | tenferro | tenferro-rs | f64 | n=64 | 1 | 512 | 4245.8 | 584.9 | 7.0% | ok |
| axpby_accum_strided_dest_f64_block_host | #2143 | host | rust-loop | f64 | n=64 | 1 | 512 | 4045.4 | 290.6 | 4.5% | ok |
| small_grad_qr_f64_trace-prepared_n4 | #2186 | trace-prepared | tenferro-rs | f64 | op=qr, n=4 | 1 | 128 | 22916.7 | 388.5 | 1.8% | ok |
| small_grad_qr_f64_pytorch_n4 | #2186 | pytorch | pytorch-cpu | f64 | op=qr, n=4 | 1 | 64 | 38527.3 | 4981.4 | 23.9% | NOISY |
| small_grad_qr_f64_trace-prepared_n16 | #2186 | trace-prepared | tenferro-rs | f64 | op=qr, n=16 | 1 | 64 | 36037.8 | 1757.2 | 6.0% | ok |
| small_grad_qr_f64_pytorch_n16 | #2186 | pytorch | pytorch-cpu | f64 | op=qr, n=16 | 1 | 64 | 52690.1 | 1821.3 | 4.0% | ok |
| small_grad_eigh_f64_trace-prepared_n4 | #2186 | trace-prepared | tenferro-rs | f64 | op=eigh, n=4 | 1 | 64 | 28119.8 | 1547.8 | 4.0% | ok |
| small_grad_eigh_f64_pytorch_n4 | #2186 | pytorch | pytorch-cpu | f64 | op=eigh, n=4 | 1 | 64 | 35764.3 | 496.1 | 1.4% | ok |
| small_grad_eigh_f64_trace-prepared_n16 | #2186 | trace-prepared | tenferro-rs | f64 | op=eigh, n=16 | 1 | 32 | 69912.8 | 640.6 | 2.6% | ok |
| small_grad_eigh_f64_pytorch_n16 | #2186 | pytorch | pytorch-cpu | f64 | op=eigh, n=16 | 1 | 32 | 58808.6 | 923.8 | 2.8% | ok |
| small_grad_lu_f64_trace-prepared_n4 | #2186 | trace-prepared | tenferro-rs | f64 | op=lu, n=4 | 1 | 128 | 14589.2 | 1057.8 | 3.9% | ok |
| small_grad_lu_f64_pytorch_n4 | #2186 | pytorch | pytorch-cpu | f64 | op=lu, n=4 | 1 | 64 | 38037.8 | 628.6 | 1.3% | ok |
| small_grad_lu_f64_trace-prepared_n16 | #2186 | trace-prepared | tenferro-rs | f64 | op=lu, n=16 | 1 | 128 | 23741.2 | 2161.6 | 4.9% | ok |
| small_grad_lu_f64_pytorch_n16 | #2186 | pytorch | pytorch-cpu | f64 | op=lu, n=16 | 1 | 64 | 49066.4 | 124.3 | 0.3% | ok |
| small_grad_svd_f64_trace-prepared_n4 | #2186 | trace-prepared | tenferro-rs | f64 | op=svd, n=4 | 1 | 64 | 45307.9 | 259.1 | 0.8% | ok |
| small_grad_svd_f64_pytorch_n4 | #2186 | pytorch | pytorch-cpu | f64 | op=svd, n=4 | 1 | 64 | 52049.5 | 3258.8 | 4.5% | ok |
| small_grad_svd_f64_trace-prepared_n16 | #2186 | trace-prepared | tenferro-rs | f64 | op=svd, n=16 | 1 | 32 | 113760.4 | 1371.7 | 2.6% | ok |
| small_grad_svd_f64_pytorch_n16 | #2186 | pytorch | pytorch-cpu | f64 | op=svd, n=16 | 1 | 32 | 96084.6 | 2289.7 | 4.7% | ok |
| small_grad_solve_f64_trace-prepared_n4 | #2186 | trace-prepared | tenferro-rs | f64 | op=solve, n=4 | 1 | 256 | 11823.9 | 93.8 | 0.7% | ok |
| small_grad_solve_f64_pytorch_n4 | #2186 | pytorch | pytorch-cpu | f64 | op=solve, n=4 | 1 | 64 | 30190.8 | 358.1 | 1.1% | ok |
| small_grad_solve_f64_trace-prepared_n16 | #2186 | trace-prepared | tenferro-rs | f64 | op=solve, n=16 | 1 | 128 | 19447.3 | 1779.9 | 5.2% | ok |
| small_grad_solve_f64_pytorch_n16 | #2186 | pytorch | pytorch-cpu | f64 | op=solve, n=16 | 1 | 64 | 38263.7 | 2867.5 | 6.7% | ok |
| fft_axis0_c64_fft-executor_n128 | #1765 | fft-executor | tenferro-rs | c64 | n=128 | 1 | 1 | 7278167.0 | 120647.0 | 2.3% | ok |
| fft_axis0_c64_pytorch_n128 | #1765 | pytorch | pytorch-cpu | c64 | n=128 | 1 | 1 | 18207125.0 | 1565458.0 | 6.6% | ok |
| degenerate_dot_hadamard_f64_dot-general-shared_n1024 | #2111 | dot-general-shared | tenferro-rs | f64 | form=hadamard, n=1024 | 1 | 4 | 355979.2 | 20838.6 | 5.9% | ok |
| degenerate_dot_hadamard_f64_pytorch_n1024 | #2111 | pytorch | pytorch-cpu | f64 | form=hadamard, n=1024 | 1 | 4 | 387687.5 | 12198.0 | 7.1% | ok |
| degenerate_dot_outer_f64_dot-general-shared_n1024 | #2111 | dot-general-shared | tenferro-rs | f64 | form=outer, n=1024 | 1 | 4 | 170104.0 | 3260.4 | 2.3% | ok |
| degenerate_dot_outer_f64_pytorch_n1024 | #2111 | pytorch | pytorch-cpu | f64 | form=outer, n=1024 | 1 | 4 | 179875.0 | 12270.9 | 5.6% | ok |
| degenerate_dot_rank0_f64_dot-general-shared_n1024 | #2111 | dot-general-shared | tenferro-rs | f64 | form=rank0, n=1024 | 1 | 4 | 340396.0 | 11338.6 | 4.6% | ok |
| degenerate_dot_rank0_f64_pytorch_n1024 | #2111 | pytorch | pytorch-cpu | f64 | form=rank0, n=1024 | 1 | 4 | 231239.8 | 45333.5 | 11.8% | NOISY |
| erased_add_f64_session-add_n256 | #1766 | session-add | tenferro-rs | f64 | n=256 | 1 | 4096 | 572.6 | 39.8 | 5.5% | ok |
| erased_add_f64_pytorch_n256 | #1766 | pytorch | pytorch-cpu | f64 | n=256 | 1 | 4096 | 844.6 | 67.1 | 4.5% | ok |
| erased_add_f64_session-add_n4096 | #1766 | session-add | tenferro-rs | f64 | n=4096 | 1 | 1024 | 1455.6 | 54.4 | 3.0% | ok |
| erased_add_f64_pytorch_n4096 | #1766 | pytorch | pytorch-cpu | f64 | n=4096 | 1 | 1024 | 1849.2 | 156.5 | 4.7% | ok |
| erased_add_f64_session-add_n1048576 | #1766 | session-add | tenferro-rs | f64 | n=1048576 | 1 | 4 | 426125.0 | 21552.1 | 4.5% | ok |
| erased_add_f64_pytorch_n1048576 | #1766 | pytorch | pytorch-cpu | f64 | n=1048576 | 1 | 4 | 412052.0 | 61015.6 | 9.6% | ok |
| axpby_strided_source_f64_1024x1024_t1 | #2220 #2214 | strided-source | tenferro-rs | f64 | n=1024 | 1 | 2 | 1162687.5 | 49791.8 | 3.7% | ok |
| axpby_contiguous_source_f64_1024x1024_t1 | #2220 #2214 | contiguous-source | tenferro-rs | f64 | n=1024 | 1 | 8 | 293906.2 | 9218.8 | 7.6% | ok |
| axpby_strided_source_f64_1024x1024_host_t1 | #2220 #2214 | host | rust-loop | f64 | n=1024 | 1 | 2 | 1295771.0 | 65656.5 | 2.9% | ok |
| erased_construct_f64_col_n8 | #2177 | col-major | tenferro-rs | f64 | n=8, layout=col | 1 | 32768 | 57.7 | 3.9 | 4.2% | ok |
| erased_construct_f64_row_n8 | #2177 | row-major | tenferro-rs | f64 | n=8, layout=row | 1 | 16384 | 122.6 | 6.0 | 3.0% | ok |
| erased_construct_f64_col_n4096 | #2177 | col-major | tenferro-rs | f64 | n=4096, layout=col | 1 | 16384 | 51.0 | 0.8 | 2.6% | ok |
| erased_construct_f64_row_n4096 | #2177 | row-major | tenferro-rs | f64 | n=4096, layout=row | 1 | 256 | 14047.2 | 467.6 | 3.9% | ok |
| erased_construct_f64_col_n65536 | #2177 | col-major | tenferro-rs | f64 | n=65536, layout=col | 1 | 1024 | 76.8 | 16.3 | 13.9% | NOISY |
| erased_construct_f64_row_n65536 | #2177 | row-major | tenferro-rs | f64 | n=65536, layout=row | 1 | 16 | 198239.6 | 7097.7 | 2.9% | ok |
| eager_small_leaf_f64_eager-held-session | #1704 | eager-held-session | tenferro-rs | f64 | op=leaf | 1 | 2048 | 1615.8 | 52.1 | 4.8% | ok |
| eager_small_add_f64_eager-held-session | #1667 | eager-held-session | tenferro-rs | f64 | op=add | 1 | 2048 | 1203.3 | 87.1 | 4.7% | ok |
| eager_small_add_f64_pytorch | #1667 | pytorch | pytorch-cpu | f64 | op=add | 1 | 4096 | 731.3 | 45.0 | 4.2% | ok |
| eager_small_gemm_f64_eager-held-session | #1904 | eager-held-session | tenferro-rs | f64 | op=gemm | 1 | 1024 | 1921.2 | 126.2 | 3.4% | ok |
| eager_small_gemm_f64_pytorch | #1904 | pytorch | pytorch-cpu | f64 | op=gemm | 1 | 4096 | 931.9 | 42.2 | 4.7% | ok |
| trace_metadata_trace-fold_f32_1024x64_k50 | #2226 #2238 | trace-fold | tenferro-rs | f32 | rows=1024, cols=64, steps=50 | 1 | 2 | 1574895.5 | 179218.5 | 7.6% | ok |
| trace_metadata_trace-outputs_f32_1024x64_k50 | #2226 #2238 | trace-outputs | tenferro-rs | f32 | rows=1024, cols=64, steps=50 | 1 | 2 | 1672771.0 | 108208.2 | 4.8% | ok |
| trace_metadata_eager_f32_1024x64_k50 | #2226 #2238 | eager | tenferro-rs | f32 | rows=1024, cols=64, steps=50 | 1 | 2 | 1979958.5 | 64458.5 | 5.7% | ok |
| trace_metadata_pytorch_f32_1024x64_k50 | #2226 #2238 | pytorch | pytorch-cpu | f32 | rows=1024, cols=64, steps=50 | 1 | 2 | 1320916.5 | 20823.2 | 2.5% | ok |
| trace_metadata_fixed_neg_f32_8 | #2226 #2238 | trace-fold | tenferro-rs | f32 | rows=8, cols=1, steps=0 | 1 | 2048 | 1309.4 | 130.6 | 5.1% | ok |
| two_operand_einsum_f64_2x2_string | #2325 #2324 | string | tenferro-rs | f64 | m=2, k=2, n=2, format=ij,jk->ik | 4 | 1024 | 2205.6 | 315.2 | 37.2% | NOISY |
| two_operand_einsum_f64_2x2_subscripts | #2325 #2324 | subscripts | tenferro-rs | f64 | m=2, k=2, n=2, format=ij,jk->ik | 4 | 2048 | 1394.7 | 181.6 | 6.6% | ok |
| two_operand_einsum_f64_2x2_prepared | #2325 #2324 | prepared | tenferro-rs | f64 | m=2, k=2, n=2, format=ij,jk->ik | 4 | 2048 | 1216.7 | 85.2 | 4.2% | ok |
| two_operand_einsum_f64_2x2_dot-general | #2325 #2324 | dot-general | tenferro-rs | f64 | m=2, k=2, n=2, format=ij,jk->ik | 4 | 2048 | 1316.5 | 191.3 | 25.2% | NOISY |
| two_operand_einsum_f64_2x2_pytorch | #2325 #2324 | pytorch | pytorch-cpu | f64 | m=2, k=2, n=2, format=ij,jk->ik | 4 | 256 | 7992.8 | 568.0 | 5.8% | ok |
| two_operand_einsum_f64_16x16_string | #2325 #2324 | string | tenferro-rs | f64 | m=16, k=16, n=16, format=ij,jk->ik | 4 | 1024 | 3171.0 | 249.6 | 5.2% | ok |
| two_operand_einsum_f64_16x16_subscripts | #2325 #2324 | subscripts | tenferro-rs | f64 | m=16, k=16, n=16, format=ij,jk->ik | 4 | 1024 | 2275.1 | 77.4 | 2.7% | ok |
| two_operand_einsum_f64_16x16_prepared | #2325 #2324 | prepared | tenferro-rs | f64 | m=16, k=16, n=16, format=ij,jk->ik | 4 | 1024 | 1922.2 | 142.9 | 33.1% | NOISY |
| two_operand_einsum_f64_16x16_dot-general | #2325 #2324 | dot-general | tenferro-rs | f64 | m=16, k=16, n=16, format=ij,jk->ik | 4 | 1024 | 1932.0 | 61.5 | 4.3% | ok |
| two_operand_einsum_f64_16x16_pytorch | #2325 #2324 | pytorch | pytorch-cpu | f64 | m=16, k=16, n=16, format=ij,jk->ik | 4 | 256 | 11962.1 | 60.6 | 0.4% | ok |
| two_operand_einsum_c64_2x2_string | #2325 #2324 | string | tenferro-rs | c64 | m=2, k=2, n=2, format=ij,jk->ik | 4 | 1024 | 2356.2 | 406.4 | 9.4% | ok |
| two_operand_einsum_c64_2x2_subscripts | #2325 #2324 | subscripts | tenferro-rs | c64 | m=2, k=2, n=2, format=ij,jk->ik | 4 | 1024 | 1739.3 | 85.6 | 5.2% | ok |
| two_operand_einsum_c64_2x2_prepared | #2325 #2324 | prepared | tenferro-rs | c64 | m=2, k=2, n=2, format=ij,jk->ik | 4 | 2048 | 1304.5 | 148.6 | 25.0% | NOISY |
| two_operand_einsum_c64_2x2_dot-general | #2325 #2324 | dot-general | tenferro-rs | c64 | m=2, k=2, n=2, format=ij,jk->ik | 4 | 2048 | 1432.2 | 34.8 | 3.5% | ok |
| two_operand_einsum_c64_2x2_pytorch | #2325 #2324 | pytorch | pytorch-cpu | c64 | m=2, k=2, n=2, format=ij,jk->ik | 4 | 512 | 7844.4 | 50.1 | 1.0% | ok |
| two_operand_einsum_c64_16x16_string | #2325 #2324 | string | tenferro-rs | c64 | m=16, k=16, n=16, format=ij,jk->ik | 4 | 512 | 4697.6 | 273.6 | 27.3% | NOISY |
| two_operand_einsum_c64_16x16_subscripts | #2325 #2324 | subscripts | tenferro-rs | c64 | m=16, k=16, n=16, format=ij,jk->ik | 4 | 512 | 3814.1 | 57.5 | 2.0% | ok |
| two_operand_einsum_c64_16x16_prepared | #2325 #2324 | prepared | tenferro-rs | c64 | m=16, k=16, n=16, format=ij,jk->ik | 4 | 1024 | 3202.2 | 87.2 | 1.7% | ok |
| two_operand_einsum_c64_16x16_dot-general | #2325 #2324 | dot-general | tenferro-rs | c64 | m=16, k=16, n=16, format=ij,jk->ik | 4 | 1024 | 3250.3 | 98.4 | 4.9% | ok |
| two_operand_einsum_c64_16x16_pytorch | #2325 #2324 | pytorch | pytorch-cpu | c64 | m=16, k=16, n=16, format=ij,jk->ik | 4 | 256 | 13147.0 | 84.1 | 0.8% | ok |
| two_operand_einsum_f64_abc_cef_abef_string | #2325 #2324 | string | tenferro-rs | f64 | m=8, k=2, n=16, format=abc,cef->abef | 4 | 1024 | 2709.3 | 251.7 | 7.0% | ok |
| two_operand_einsum_f64_abc_cef_abef_subscripts | #2325 #2324 | subscripts | tenferro-rs | f64 | m=8, k=2, n=16, format=abc,cef->abef | 4 | 2048 | 1621.4 | 127.3 | 4.3% | ok |
| two_operand_einsum_f64_abc_cef_abef_prepared | #2325 #2324 | prepared | tenferro-rs | f64 | m=8, k=2, n=16, format=abc,cef->abef | 4 | 2048 | 1353.9 | 74.9 | 3.1% | ok |
| two_operand_einsum_f64_abc_cef_abef_dot-general | #2325 #2324 | dot-general | tenferro-rs | f64 | m=8, k=2, n=16, format=abc,cef->abef | 4 | 2048 | 1349.6 | 66.9 | 2.7% | ok |
| two_operand_einsum_f64_abc_cef_abef_pytorch | #2325 #2324 | pytorch | pytorch-cpu | f64 | m=8, k=2, n=16, format=abc,cef->abef | 4 | 256 | 9198.4 | 83.7 | 3.2% | ok |
| decode_proj_f32_eager-shared_in1024_out1024_len8 | #1992 #2003 #1995 #2229 | eager-shared | tenferro-rs | f32 | in=1024, out=1024, len=8 | 4 | 32 | 122976.6 | 7990.9 | 5.0% | ok |
| decode_proj_f32_faer-direct_in1024_out1024_len8 | #1992 #2003 #1995 #2229 | faer-direct | faer-direct | f32 | in=1024, out=1024, len=8 | 4 | 16 | 129906.2 | 5147.2 | 3.8% | ok |
| decode_proj_f32_host-sgemm_in1024_out1024_len8 | #1992 #2003 #1995 #2229 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=1024, len=8 | 4 | 8 | 389729.2 | 18481.8 | 3.0% | ok |
| decode_proj_f32_pytorch_in1024_out1024_len8 | #1992 #2003 #1995 #2229 | pytorch | pytorch-cpu | f32 | in=1024, out=1024, len=8 | 4 | 32 | 118079.4 | 1217.5 | 1.1% | ok |
| decode_proj_f32_eager-shared_in1024_out1024_len64 | #1992 #2003 #1995 #2229 | eager-shared | tenferro-rs | f32 | in=1024, out=1024, len=64 | 4 | 16 | 167526.1 | 2447.9 | 1.5% | ok |
| decode_proj_f32_faer-direct_in1024_out1024_len64 | #1992 #2003 #1995 #2229 | faer-direct | faer-direct | f32 | in=1024, out=1024, len=64 | 4 | 4 | 674708.5 | 271916.6 | 18.4% | NOISY |
| decode_proj_f32_host-sgemm_in1024_out1024_len64 | #1992 #2003 #1995 #2229 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=1024, len=64 | 4 | 2 | 1102687.5 | 43114.5 | 2.4% | ok |
| decode_proj_f32_pytorch_in1024_out1024_len64 | #1992 #2003 #1995 #2229 | pytorch | pytorch-cpu | f32 | in=1024, out=1024, len=64 | 4 | 16 | 173158.8 | 2006.5 | 1.3% | ok |
| decode_proj_f32_eager-shared_in1024_out1024_len512 | #1992 #2003 #1995 #2229 | eager-shared | tenferro-rs | f32 | in=1024, out=1024, len=512 | 4 | 4 | 869573.0 | 10078.1 | 5.0% | ok |
| decode_proj_f32_faer-direct_in1024_out1024_len512 | #1992 #2003 #1995 #2229 | faer-direct | faer-direct | f32 | in=1024, out=1024, len=512 | 4 | 1 | 3284208.0 | 1550916.5 | 21.4% | NOISY |
| decode_proj_f32_host-sgemm_in1024_out1024_len512 | #1992 #2003 #1995 #2229 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=1024, len=512 | 4 | 1 | 5158041.0 | 170104.0 | 2.8% | ok |
| decode_proj_f32_pytorch_in1024_out1024_len512 | #1992 #2003 #1995 #2229 | pytorch | pytorch-cpu | f32 | in=1024, out=1024, len=512 | 4 | 2 | 929333.5 | 16843.5 | 1.2% | ok |
| decode_proj_f32_eager-shared_in1024_out4096_len8 | #1992 #2003 #1995 #2229 | eager-shared | tenferro-rs | f32 | in=1024, out=4096, len=8 | 4 | 4 | 631645.8 | 30390.6 | 7.4% | ok |
| decode_proj_f32_faer-direct_in1024_out4096_len8 | #1992 #2003 #1995 #2229 | faer-direct | faer-direct | f32 | in=1024, out=4096, len=8 | 4 | 4 | 643791.5 | 39645.9 | 4.2% | ok |
| decode_proj_f32_host-sgemm_in1024_out4096_len8 | #1992 #2003 #1995 #2229 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=4096, len=8 | 4 | 2 | 1879125.0 | 256844.0 | 8.7% | ok |
| decode_proj_f32_pytorch_in1024_out4096_len8 | #1992 #2003 #1995 #2229 | pytorch | pytorch-cpu | f32 | in=1024, out=4096, len=8 | 4 | 4 | 600979.2 | 16359.4 | 3.0% | ok |
| decode_proj_f32_eager-shared_in1024_out4096_len64 | #1992 #2003 #1995 #2229 | eager-shared | tenferro-rs | f32 | in=1024, out=4096, len=64 | 4 | 4 | 744625.0 | 52072.9 | 49.0% | NOISY |
| decode_proj_f32_faer-direct_in1024_out4096_len64 | #1992 #2003 #1995 #2229 | faer-direct | faer-direct | f32 | in=1024, out=4096, len=64 | 4 | 1 | 3137083.0 | 477230.0 | 15.1% | NOISY |
| decode_proj_f32_host-sgemm_in1024_out4096_len64 | #1992 #2003 #1995 #2229 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=4096, len=64 | 4 | 1 | 4765458.0 | 196771.5 | 2.8% | ok |
| decode_proj_f32_pytorch_in1024_out4096_len64 | #1992 #2003 #1995 #2229 | pytorch | pytorch-cpu | f32 | in=1024, out=4096, len=64 | 4 | 4 | 839666.8 | 90307.4 | 10.1% | NOISY |
| decode_proj_f32_eager-shared_in1024_out4096_len512 | #1992 #2003 #1995 #2229 | eager-shared | tenferro-rs | f32 | in=1024, out=4096, len=512 | 4 | 1 | 3653792.0 | 64999.0 | 3.7% | ok |
| decode_proj_f32_faer-direct_in1024_out4096_len512 | #1992 #2003 #1995 #2229 | faer-direct | faer-direct | f32 | in=1024, out=4096, len=512 | 4 | 1 | 18072834.0 | 1210896.0 | 14.1% | NOISY |
| decode_proj_f32_host-sgemm_in1024_out4096_len512 | #1992 #2003 #1995 #2229 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=4096, len=512 | 4 | 1 | 21291542.0 | 407041.5 | 1.5% | ok |
| decode_proj_f32_pytorch_in1024_out4096_len512 | #1992 #2003 #1995 #2229 | pytorch | pytorch-cpu | f32 | in=1024, out=4096, len=512 | 4 | 1 | 4163458.0 | 68937.0 | 2.7% | ok |
| decode_proj_view_operand_f32_prepared_in1024_out1024_len8_t4 | #2216 | prepared-view | tenferro-rs | f32 | in=1024, out=1024, len=8 | 4 | 32 | 99527.3 | 8417.3 | 4.9% | ok |
| decode_proj_contiguous_operand_f32_prepared_in1024_out1024_len8_t4 | #2216 | prepared-contiguous | tenferro-rs | f32 | in=1024, out=1024, len=8 | 4 | 16 | 118838.5 | 751.3 | 3.9% | ok |
| decode_proj_view_operand_f32_pytorch_in1024_out1024_len8_t4 | #2216 | pytorch | pytorch-cpu | f32 | in=1024, out=1024, len=8 | 4 | 32 | 115278.7 | 3778.0 | 5.6% | ok |
| gemm_c64_mm256_t1 | #1900 | dot-general-shared | tenferro-rs | c64 | n=256 | 4 | 4 | 588468.8 | 18854.1 | 2.2% | ok |
| gemm_f64_mm256_t1 | #1900 | dot-general-shared | tenferro-rs | f64 | n=256 | 4 | 16 | 106984.4 | 315.1 | 0.5% | ok |
| conj_dot_f64_dot-general-rank1_len100 | #1615 | dot-general-rank1 | tenferro-rs | f64 | len=100 | 4 | 2048 | 1283.7 | 77.7 | 3.5% | ok |
| conj_dot_f64_portable-loop_len100 | #1615 | portable-loop | rust-loop | f64 | len=100 | 4 | 65536 | 50.3 | 0.3 | 0.4% | ok |
| conj_dot_f64_pytorch-vdot_len100 | #1615 | pytorch-vdot | pytorch-cpu | f64 | len=100 | 4 | 4096 | 733.3 | 15.4 | 7.5% | ok |
| conj_dot_f64_dot-general-rank1_len10000 | #1615 | dot-general-rank1 | tenferro-rs | f64 | len=10000 | 4 | 256 | 13758.5 | 161.5 | 3.0% | ok |
| conj_dot_f64_portable-loop_len10000 | #1615 | portable-loop | rust-loop | f64 | len=10000 | 4 | 256 | 9714.2 | 257.0 | 2.4% | ok |
| conj_dot_f64_pytorch-vdot_len10000 | #1615 | pytorch-vdot | pytorch-cpu | f64 | len=10000 | 4 | 2048 | 1502.4 | 40.8 | 4.1% | ok |
| conj_dot_f64_dot-general-rank1_len1000000 | #1615 | dot-general-rank1 | tenferro-rs | f64 | len=1000000 | 4 | 2 | 1248271.0 | 11031.5 | 0.8% | ok |
| conj_dot_f64_portable-loop_len1000000 | #1615 | portable-loop | rust-loop | f64 | len=1000000 | 4 | 4 | 943583.2 | 29989.6 | 2.1% | ok |
| conj_dot_f64_pytorch-vdot_len1000000 | #1615 | pytorch-vdot | pytorch-cpu | f64 | len=1000000 | 4 | 32 | 125104.2 | 9646.5 | 9.8% | ok |
| conj_dot_c64_dot-general-rank1_len100 | #1615 | dot-general-rank1 | tenferro-rs | c64 | len=100 | 4 | 2048 | 1685.1 | 76.6 | 2.6% | ok |
| conj_dot_c64_portable-loop_len100 | #1615 | portable-loop | rust-loop | c64 | len=100 | 4 | 32768 | 83.8 | 1.5 | 2.2% | ok |
| conj_dot_c64_pytorch-vdot_len100 | #1615 | pytorch-vdot | pytorch-cpu | c64 | len=100 | 4 | 4096 | 641.7 | 16.8 | 213.7% | NOISY |
| conj_dot_c64_dot-general-rank1_len10000 | #1615 | dot-general-rank1 | tenferro-rs | c64 | len=10000 | 4 | 64 | 52471.3 | 693.0 | 2.6% | ok |
| conj_dot_c64_portable-loop_len10000 | #1615 | portable-loop | rust-loop | c64 | len=10000 | 4 | 256 | 10082.0 | 407.6 | 3.4% | ok |
| conj_dot_c64_pytorch-vdot_len10000 | #1615 | pytorch-vdot | pytorch-cpu | c64 | len=10000 | 4 | 512 | 5349.1 | 148.6 | 2.0% | ok |
| conj_dot_c64_dot-general-rank1_len1000000 | #1615 | dot-general-rank1 | tenferro-rs | c64 | len=1000000 | 4 | 1 | 5627375.0 | 91875.0 | 1.6% | ok |
| conj_dot_c64_portable-loop_len1000000 | #1615 | portable-loop | rust-loop | c64 | len=1000000 | 4 | 2 | 1020646.0 | 32114.5 | 2.5% | ok |
| conj_dot_c64_pytorch-vdot_len1000000 | #1615 | pytorch-vdot | pytorch-cpu | c64 | len=1000000 | 4 | 4 | 551333.5 | 8031.4 | 1.0% | ok |
| small_solve_f64_solve-loop_k16_b64 | #2007 #1878 | solve-loop | tenferro-rs | f64 | k=16, batch=64 | 4 | 16 | 153294.2 | 4079.5 | 2.1% | ok |
| small_solve_f64_triangular-solve-loop_k16_b64 | #2007 #1878 | triangular-solve-loop | tenferro-rs | f64 | k=16, batch=64 | 4 | 32 | 83302.1 | 4066.4 | 2.8% | ok |
| small_solve_f64_solve-batched_k16_b64 | #2007 #1878 | solve-batched | tenferro-rs | f64 | k=16, batch=64 | 4 | 32 | 81371.1 | 2218.1 | 4.8% | ok |
| small_solve_f64_pytorch-solve-batched_k16_b64 | #2007 #1878 | pytorch-solve-batched | pytorch-cpu | f64 | k=16, batch=64 | 4 | 16 | 122411.5 | 13802.1 | 8.0% | ok |
| small_solve_f64_pytorch-solve-triangular-batched_k16_b64 | #2007 #1878 | pytorch-solve-triangular-batched | pytorch-cpu | f64 | k=16, batch=64 | 4 | 128 | 21552.7 | 355.3 | 2.0% | ok |
| small_solve_f64_solve-loop_k64_b64 | #2007 #1878 | solve-loop | tenferro-rs | f64 | k=64, batch=64 | 4 | 2 | 1565792.0 | 22521.0 | 2.8% | ok |
| small_solve_f64_triangular-solve-loop_k64_b64 | #2007 #1878 | triangular-solve-loop | tenferro-rs | f64 | k=64, batch=64 | 4 | 2 | 375000.0 | 38719.0 | 19.6% | NOISY |
| small_solve_f64_solve-batched_k64_b64 | #2007 #1878 | solve-batched | tenferro-rs | f64 | k=64, batch=64 | 4 | 2 | 1660604.5 | 21969.0 | 0.9% | ok |
| small_solve_f64_pytorch-solve-batched_k64_b64 | #2007 #1878 | pytorch-solve-batched | pytorch-cpu | f64 | k=64, batch=64 | 4 | 2 | 1065854.5 | 71698.0 | 4.5% | ok |
| small_solve_f64_pytorch-solve-triangular-batched_k64_b64 | #2007 #1878 | pytorch-solve-triangular-batched | pytorch-cpu | f64 | k=64, batch=64 | 4 | 8 | 293739.6 | 19315.1 | 4.7% | ok |
| tanh_chain_f32_compiled-prepared_k10_1024x64 | #1990 | compiled-prepared | tenferro-rs | f32 | rows=1024, cols=64, chain=10 | 4 | 2 | 530895.5 | 219614.5 | 56.9% | NOISY |
| tanh_chain_f32_eager-shared_k10_1024x64 | #1990 | eager-shared | tenferro-rs | f32 | rows=1024, cols=64, chain=10 | 4 | 1 | 2046916.0 | 65520.5 | 2.1% | ok |
| tanh_chain_f32_compiled-prepared_k100_1024x64 | #1990 | compiled-prepared | tenferro-rs | f32 | rows=1024, cols=64, chain=100 | 4 | 1 | 5169333.0 | 48853.0 | 30.0% | NOISY |
| tanh_chain_f32_eager-shared_k100_1024x64 | #1990 | eager-shared | tenferro-rs | f32 | rows=1024, cols=64, chain=100 | 4 | 1 | 20764000.0 | 1074021.0 | 3.4% | ok |
| tanh_chain_f32_compiled-prepared_k200_1024x64 | #1990 | compiled-prepared | tenferro-rs | f32 | rows=1024, cols=64, chain=200 | 4 | 1 | 14046708.0 | 5012374.5 | 19.3% | NOISY |
| tanh_chain_f32_eager-shared_k200_1024x64 | #1990 | eager-shared | tenferro-rs | f32 | rows=1024, cols=64, chain=200 | 4 | 1 | 41213500.0 | 1116354.5 | 3.8% | ok |
| layer_norm_f32_eager-composed_d1024_len8_b1 | #2006 | eager-composed | tenferro-rs | f32 | norm=layer_norm, d=1024, len=8, batch=1 | 4 | 64 | 41194.7 | 1655.3 | 2.8% | ok |
| layer_norm_f32_pytorch-fused_d1024_len8_b1 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=layer_norm, d=1024, len=8, batch=1 | 4 | 64 | 36533.9 | 1865.6 | 8.7% | ok |
| layer_norm_f32_eager-single-call_d1024_len8_b1 | #2006 | eager-single-call | tenferro-rs | f32 | norm=layer_norm, d=1024, len=8, batch=1 | 4 | 64 | 47003.9 | 1101.6 | 1.7% | ok |
| layer_norm_f32_eager-composed_d1024_len8_b8 | #2006 | eager-composed | tenferro-rs | f32 | norm=layer_norm, d=1024, len=8, batch=8 | 4 | 32 | 81214.8 | 4479.8 | 6.1% | ok |
| layer_norm_f32_pytorch-fused_d1024_len8_b8 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=layer_norm, d=1024, len=8, batch=8 | 4 | 32 | 58592.4 | 3034.5 | 4.4% | ok |
| layer_norm_f32_eager-single-call_d1024_len8_b8 | #2006 | eager-single-call | tenferro-rs | f32 | norm=layer_norm, d=1024, len=8, batch=8 | 4 | 32 | 82196.6 | 2444.7 | 2.1% | ok |
| layer_norm_f32_eager-composed_d1024_len64_b1 | #2006 | eager-composed | tenferro-rs | f32 | norm=layer_norm, d=1024, len=64, batch=1 | 4 | 32 | 76713.6 | 2919.3 | 2.0% | ok |
| layer_norm_f32_pytorch-fused_d1024_len64_b1 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=layer_norm, d=1024, len=64, batch=1 | 4 | 64 | 51838.5 | 7760.1 | 9.9% | ok |
| layer_norm_f32_eager-single-call_d1024_len64_b1 | #2006 | eager-single-call | tenferro-rs | f32 | norm=layer_norm, d=1024, len=64, batch=1 | 4 | 32 | 79541.7 | 7849.0 | 32.4% | NOISY |
| layer_norm_f32_eager-composed_d1024_len64_b8 | #2006 | eager-composed | tenferro-rs | f32 | norm=layer_norm, d=1024, len=64, batch=8 | 4 | 8 | 284593.8 | 4546.9 | 2.1% | ok |
| layer_norm_f32_pytorch-fused_d1024_len64_b8 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=layer_norm, d=1024, len=64, batch=8 | 4 | 16 | 158500.0 | 6425.8 | 3.5% | ok |
| layer_norm_f32_eager-single-call_d1024_len64_b8 | #2006 | eager-single-call | tenferro-rs | f32 | norm=layer_norm, d=1024, len=64, batch=8 | 4 | 4 | 408114.5 | 161265.6 | 20.0% | NOISY |
| rms_norm_f32_eager-composed_d1024_len8_b1 | #2006 | eager-composed | tenferro-rs | f32 | norm=rms_norm, d=1024, len=8, batch=1 | 4 | 64 | 23883.5 | 6771.8 | 33.5% | NOISY |
| rms_norm_f32_pytorch-fused_d1024_len8_b1 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=rms_norm, d=1024, len=8, batch=1 | 4 | 256 | 9338.2 | 90.1 | 0.8% | ok |
| rms_norm_f32_eager-single-call_d1024_len8_b1 | #2006 | eager-single-call | tenferro-rs | f32 | norm=rms_norm, d=1024, len=8, batch=1 | 4 | 128 | 26390.0 | 1633.5 | 3.6% | ok |
| rms_norm_f32_eager-composed_d1024_len8_b8 | #2006 | eager-composed | tenferro-rs | f32 | norm=rms_norm, d=1024, len=8, batch=8 | 4 | 64 | 40483.1 | 1402.0 | 2.3% | ok |
| rms_norm_f32_pytorch-fused_d1024_len8_b8 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=rms_norm, d=1024, len=8, batch=8 | 4 | 16 | 194072.9 | 14161.4 | 4.8% | ok |
| rms_norm_f32_eager-single-call_d1024_len8_b8 | #2006 | eager-single-call | tenferro-rs | f32 | norm=rms_norm, d=1024, len=8, batch=8 | 4 | 32 | 44933.6 | 959.7 | 1.4% | ok |
| rms_norm_f32_eager-composed_d1024_len64_b1 | #2006 | eager-composed | tenferro-rs | f32 | norm=rms_norm, d=1024, len=64, batch=1 | 4 | 64 | 40763.0 | 491.2 | 2.5% | ok |
| rms_norm_f32_pytorch-fused_d1024_len64_b1 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=rms_norm, d=1024, len=64, batch=1 | 4 | 16 | 147507.8 | 15838.6 | 8.6% | ok |
| rms_norm_f32_eager-single-call_d1024_len64_b1 | #2006 | eager-single-call | tenferro-rs | f32 | norm=rms_norm, d=1024, len=64, batch=1 | 4 | 32 | 44880.2 | 3446.0 | 37.4% | NOISY |
| rms_norm_f32_eager-composed_d1024_len64_b8 | #2006 | eager-composed | tenferro-rs | f32 | norm=rms_norm, d=1024, len=64, batch=8 | 4 | 4 | 235895.8 | 95567.6 | 24.1% | NOISY |
| rms_norm_f32_pytorch-fused_d1024_len64_b8 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=rms_norm, d=1024, len=64, batch=8 | 4 | 8 | 290802.0 | 11885.4 | 5.9% | ok |
| rms_norm_f32_eager-single-call_d1024_len64_b8 | #2006 | eager-single-call | tenferro-rs | f32 | norm=rms_norm, d=1024, len=64, batch=8 | 4 | 8 | 179474.0 | 6039.1 | 2.8% | ok |
| small_contraction_abcd-dbef-acef_f64_shared-session-default-pool_d4 | #1885 #1897 #1901 | shared-session-default-pool | tenferro-rs | f64 | extent=4 | 4 | 1024 | 2934.1 | 275.4 | 5.1% | ok |
| activation_erf_f32_eager-single-call_1024x64 | #1975 #2069 #2032 | eager-single-call | tenferro-rs | f32 | op=erf, rows=1024, cols=64 | 4 | — | — | — | — | SKIPPED |
| activation_erf_f32_session-single-call_1024x64 | #1975 #2069 #2032 | session-single-call | tenferro-rs | f32 | op=erf, rows=1024, cols=64 | 4 | 8 | 393515.6 | 44330.7 | 10.8% | NOISY |
| activation_erf_f32_pytorch_1024x64 | #1975 #2069 #2032 | pytorch | pytorch-cpu | f32 | op=erf, rows=1024, cols=64 | 4 | — | — | — | — | SKIPPED |
| activation_sigmoid_f32_eager-single-call_1024x64 | #1975 #2069 #2032 | eager-single-call | tenferro-rs | f32 | op=sigmoid, rows=1024, cols=64 | 4 | — | — | — | — | SKIPPED |
| activation_sigmoid_f32_session-single-call_1024x64 | #1975 #2069 #2032 | session-single-call | tenferro-rs | f32 | op=sigmoid, rows=1024, cols=64 | 4 | 64 | 13709.6 | 1344.1 | 57.8% | NOISY |
| activation_sigmoid_f32_eager-composed_1024x64 | #1975 #2069 #2032 | eager-composed | tenferro-rs | f32 | op=sigmoid, rows=1024, cols=64 | 4 | 32 | 67710.9 | 2430.3 | 2.8% | ok |
| activation_sigmoid_f32_pytorch_1024x64 | #1975 #2069 #2032 | pytorch | pytorch-cpu | f32 | op=sigmoid, rows=1024, cols=64 | 4 | — | — | — | — | SKIPPED |
| activation_silu_f32_eager-single-call_1024x64 | #1975 #2069 #2032 | eager-single-call | tenferro-rs | f32 | op=silu, rows=1024, cols=64 | 4 | — | — | — | — | SKIPPED |
| activation_silu_f32_session-single-call_1024x64 | #1975 #2069 #2032 | session-single-call | tenferro-rs | f32 | op=silu, rows=1024, cols=64 | 4 | 128 | 13906.6 | 293.5 | 1.7% | ok |
| activation_silu_f32_eager-composed_1024x64 | #1975 #2069 #2032 | eager-composed | tenferro-rs | f32 | op=silu, rows=1024, cols=64 | 4 | 32 | 72985.7 | 3607.4 | 4.0% | ok |
| activation_silu_f32_pytorch_1024x64 | #1975 #2069 #2032 | pytorch | pytorch-cpu | f32 | op=silu, rows=1024, cols=64 | 4 | — | — | — | — | SKIPPED |
| activation_softplus_f32_eager-single-call_1024x64 | #1975 #2069 #2032 | eager-single-call | tenferro-rs | f32 | op=softplus, rows=1024, cols=64 | 4 | — | — | — | — | SKIPPED |
| activation_softplus_f32_session-single-call_1024x64 | #1975 #2069 #2032 | session-single-call | tenferro-rs | f32 | op=softplus, rows=1024, cols=64 | 4 | 32 | 102941.4 | 116770.2 | 54.4% | NOISY |
| activation_softplus_f32_eager-composed_1024x64 | #1975 #2069 #2032 | eager-composed | tenferro-rs | f32 | op=softplus, rows=1024, cols=64 | 4 | 16 | 184877.6 | 13240.8 | 4.2% | ok |
| activation_softplus_f32_pytorch_1024x64 | #1975 #2069 #2032 | pytorch | pytorch-cpu | f32 | op=softplus, rows=1024, cols=64 | 4 | — | — | — | — | SKIPPED |
| activation_gelu_f32_eager-single-call_1024x64 | #1975 #2030 #2069 #2228 | eager-single-call | tenferro-rs | f32 | op=gelu, rows=1024, cols=64 | 4 | 64 | 22584.0 | 31511.1 | 55.6% | NOISY |
| activation_gelu_f32_session-single-call_1024x64 | #1975 #2030 #2069 #2228 | session-single-call | tenferro-rs | f32 | op=gelu, rows=1024, cols=64 | 4 | 32 | 66760.4 | 1629.6 | 28.5% | NOISY |
| activation_gelu_f32_pytorch_1024x64 | #1975 #2030 #2069 #2228 | pytorch | pytorch-cpu | f32 | op=gelu, rows=1024, cols=64 | 4 | 32 | 95440.1 | 1479.8 | 1.1% | ok |
| activation_gelu_tanh_f32_eager-single-call_1024x64 | #1975 #2069 #2032 | eager-single-call | tenferro-rs | f32 | op=gelu_tanh, rows=1024, cols=64 | 4 | — | — | — | — | SKIPPED |
| activation_gelu_tanh_f32_session-single-call_1024x64 | #1975 #2069 #2032 | session-single-call | tenferro-rs | f32 | op=gelu_tanh, rows=1024, cols=64 | 4 | 32 | 71195.3 | 120776.0 | 54.1% | NOISY |
| activation_gelu_tanh_f32_eager-composed_1024x64 | #1975 #2069 #2032 | eager-composed | tenferro-rs | f32 | op=gelu_tanh, rows=1024, cols=64 | 4 | 8 | 312625.0 | 20200.5 | 13.2% | NOISY |
| activation_gelu_tanh_f32_pytorch_1024x64 | #1975 #2069 #2032 | pytorch | pytorch-cpu | f32 | op=gelu_tanh, rows=1024, cols=64 | 4 | — | — | — | — | SKIPPED |
| softmax_f32_eager-single-call_len64_b8 | #1976 #2031 #2231 | eager-single-call | tenferro-rs | f32 | op=softmax, len=64, batch=8, axis=0 | 4 | 64 | 27640.6 | 19039.1 | 37.1% | NOISY |
| softmax_f32_session-single-call_len64_b8 | #1976 #2031 #2231 | session-single-call | tenferro-rs | f32 | op=softmax, len=64, batch=8, axis=0 | 4 | 64 | 30826.8 | 1576.5 | 2.7% | ok |
| softmax_f32_eager-composed_len64_b8 | #1976 #2031 #2231 | eager-composed | tenferro-rs | f32 | op=softmax, len=64, batch=8, axis=0 | 4 | 32 | 68109.4 | 781.9 | 1.5% | ok |
| softmax_f32_pytorch_len64_b8 | #1976 #2031 #2231 | pytorch | pytorch-cpu | f32 | op=softmax, len=64, batch=8, axis=0 | 4 | 64 | 50927.7 | 4208.7 | 5.5% | ok |
| softmax_f32_eager-single-call_len512_b8 | #1976 #2231 | eager-single-call | tenferro-rs | f32 | op=softmax, len=512, batch=8, axis=0 | 4 | 2 | 735041.5 | 907687.5 | 56.9% | NOISY |
| softmax_f32_session-single-call_len512_b8 | #1976 #2231 | session-single-call | tenferro-rs | f32 | op=softmax, len=512, batch=8, axis=0 | 4 | 2 | 693083.0 | 293583.2 | 74.7% | NOISY |
| softmax_f32_eager-composed_len512_b8 | #1976 #2231 | eager-composed | tenferro-rs | f32 | op=softmax, len=512, batch=8, axis=0 | 4 | 2 | 1018416.5 | 670968.8 | 39.4% | NOISY |
| softmax_f32_pytorch_len512_b8 | #1976 #2231 | pytorch | pytorch-cpu | f32 | op=softmax, len=512, batch=8, axis=0 | 4 | 2 | 1327000.0 | 64989.5 | 3.1% | ok |
| log_softmax_f32_eager-single-call_len64_b8 | #1976 | eager-single-call | tenferro-rs | f32 | op=log_softmax, len=64, batch=8, axis=0 | 4 | 32 | 82972.7 | 3065.8 | 2.1% | ok |
| log_softmax_f32_session-single-call_len64_b8 | #1976 | session-single-call | tenferro-rs | f32 | op=log_softmax, len=64, batch=8, axis=0 | 4 | 32 | 66816.4 | 2164.1 | 2.1% | ok |
| log_softmax_f32_eager-composed_len64_b8 | #1976 | eager-composed | tenferro-rs | f32 | op=log_softmax, len=64, batch=8, axis=0 | 4 | 32 | 68282.6 | 1324.9 | 11.4% | NOISY |
| log_softmax_f32_pytorch_len64_b8 | #1976 | pytorch | pytorch-cpu | f32 | op=log_softmax, len=64, batch=8, axis=0 | 4 | 32 | 58166.7 | 4052.1 | 6.6% | ok |
| log_softmax_f32_eager-single-call_len512_b8 | #1976 | eager-single-call | tenferro-rs | f32 | op=log_softmax, len=512, batch=8, axis=0 | 4 | 2 | 980208.0 | 1178135.0 | 45.8% | NOISY |
| log_softmax_f32_session-single-call_len512_b8 | #1976 | session-single-call | tenferro-rs | f32 | op=log_softmax, len=512, batch=8, axis=0 | 4 | 2 | 1132333.0 | 1029739.5 | 37.7% | NOISY |
| log_softmax_f32_eager-composed_len512_b8 | #1976 | eager-composed | tenferro-rs | f32 | op=log_softmax, len=512, batch=8, axis=0 | 4 | 2 | 1118625.0 | 1002573.2 | 35.6% | NOISY |
| log_softmax_f32_pytorch_len512_b8 | #1976 | pytorch | pytorch-cpu | f32 | op=log_softmax, len=512, batch=8, axis=0 | 4 | 2 | 1130958.5 | 91666.5 | 7.6% | ok |
| masked_softmax_f32_eager-single-call_len64_b8 | #1976 | eager-single-call | tenferro-rs | f32 | op=masked_softmax, len=64, batch=8, axis=0 | 4 | 32 | 105571.6 | 1766.3 | 2.2% | ok |
| masked_softmax_f32_session-single-call_len64_b8 | #1976 | session-single-call | tenferro-rs | f32 | op=masked_softmax, len=64, batch=8, axis=0 | 4 | 32 | 90330.7 | 7135.4 | 19.3% | NOISY |
| masked_softmax_f32_eager-composed_len64_b8 | #1976 | eager-composed | tenferro-rs | f32 | op=masked_softmax, len=64, batch=8, axis=0 | 4 | 32 | 75851.6 | 1461.6 | 3.1% | ok |
| masked_softmax_f32_pytorch_len64_b8 | #1976 | pytorch | pytorch-cpu | f32 | op=masked_softmax, len=64, batch=8, axis=0 | 4 | 32 | 83839.8 | 14896.5 | 9.4% | ok |
| masked_softmax_f32_eager-single-call_len512_b8 | #1976 | eager-single-call | tenferro-rs | f32 | op=masked_softmax, len=512, batch=8, axis=0 | 4 | 2 | 2619708.5 | 1499489.8 | 40.5% | NOISY |
| masked_softmax_f32_session-single-call_len512_b8 | #1976 | session-single-call | tenferro-rs | f32 | op=masked_softmax, len=512, batch=8, axis=0 | 4 | 1 | 1164542.0 | 628313.0 | 40.3% | NOISY |
| masked_softmax_f32_eager-composed_len512_b8 | #1976 | eager-composed | tenferro-rs | f32 | op=masked_softmax, len=512, batch=8, axis=0 | 4 | 1 | 1183917.0 | 124625.0 | 35.9% | NOISY |
| masked_softmax_f32_pytorch_len512_b8 | #1976 | pytorch | pytorch-cpu | f32 | op=masked_softmax, len=512, batch=8, axis=0 | 4 | 2 | 1954479.5 | 37145.8 | 1.2% | ok |
| softmax_f32_eager-single-call_len64_b8_t4 | #2233 #2235 | eager-single-call | tenferro-rs | f32 | op=softmax, len=64, batch=8, axis=0 | 4 | 64 | 30535.8 | 6294.3 | 28.6% | NOISY |
| softmax_f32_eager-composed_len64_b8_t4 | #2233 #2235 | eager-composed | tenferro-rs | f32 | op=softmax, len=64, batch=8, axis=0 | 4 | 32 | 67571.6 | 599.6 | 2.2% | ok |
| softmax_f32_pytorch_len64_b8_t4 | #2233 #2235 | pytorch | pytorch-cpu | f32 | op=softmax, len=64, batch=8, axis=0 | 4 | 64 | 51881.5 | 3103.8 | 4.9% | ok |
| reduce_mean_f32_eager-single-call_1024x64_axis0 | #1976 | eager-single-call | tenferro-rs | f32 | rows=1024, cols=64, axis=0 | 4 | 256 | 12114.1 | 314.8 | 2.3% | ok |
| reduce_mean_f32_session-single-call_1024x64_axis0 | #1976 | session-single-call | tenferro-rs | f32 | rows=1024, cols=64, axis=0 | 4 | 512 | 7525.6 | 132.2 | 22.1% | NOISY |
| reduce_mean_f32_eager-composed_1024x64_axis0 | #1976 | eager-composed | tenferro-rs | f32 | rows=1024, cols=64, axis=0 | 4 | 256 | 11686.0 | 546.2 | 22.3% | NOISY |
| reduce_mean_f32_pytorch_1024x64_axis0 | #1976 | pytorch | pytorch-cpu | f32 | rows=1024, cols=64, axis=0 | 4 | 64 | 38224.0 | 2029.0 | 4.9% | ok |
| take_along_axis_rows_f64_eager-single-call_n64_b256 | #2008 | eager-single-call | tenferro-rs | f64 | n=64, batch=256, axis=0 | 4 | 1 | 3133250.0 | 1312771.5 | 27.3% | NOISY |
| take_along_axis_rows_f64_session-single-call_n64_b256 | #2008 | session-single-call | tenferro-rs | f64 | n=64, batch=256, axis=0 | 4 | 1 | 1823542.0 | 1360333.5 | 33.2% | NOISY |
| take_along_axis_rows_f64_eager-gather-prebuilt_n64_b256 | #2008 | eager-gather-prebuilt | tenferro-rs | f64 | n=64, batch=256, axis=0 | 4 | 2 | 1797208.0 | 1365020.5 | 29.6% | NOISY |
| take_along_axis_rows_f64_pytorch_n64_b256 | #2008 | pytorch | pytorch-cpu | f64 | n=64, batch=256, axis=0 | 4 | 2 | 917854.5 | 62083.5 | 6.2% | ok |
| lstsq_f64_eager-shared_m768_n384_rhs16 | #2027 | eager-shared | tenferro-rs | f64 | operation=lstsq | 4 | 1 | 9954375.0 | 3369187.0 | 24.0% | NOISY |
| lstsq_f64_pytorch_m768_n384_rhs16 | #2027 | pytorch | pytorch-cpu | f64 | operation=lstsq | 4 | 1 | 4119750.0 | 133166.0 | 4.9% | ok |
| triangular_f64_concrete-shared_n4096_rhs64 | #2028 | concrete-shared | tenferro-rs | f64 | operation=triangular | 4 | 1 | 12989417.0 | 1053979.0 | 4.5% | ok |
| triangular_f64_pytorch_n4096_rhs64 | #2028 | pytorch | pytorch-cpu | f64 | operation=triangular | 4 | 1 | 9220417.0 | 511437.0 | 4.0% | ok |
| reshape_f64_concrete-shared_n33554432_to8192x4096 | #2029 | concrete-shared | tenferro-rs | f64 | operation=reshape | 4 | 1 | 4452333.0 | 939146.5 | 14.0% | NOISY |
| reshape_f64_pytorch_n33554432_to8192x4096 | #2029 | pytorch | pytorch-cpu | f64 | operation=reshape | 4 | 1 | 4848958.0 | 427583.5 | 8.3% | ok |
| metadata_transpose_view_f64_rank8_t4 | #2180 | metadata-clone | tenferro-rs | f64 | shape=[2, 2, 2, 2, 2, 2, 2, 2] | 4 | 32768 | 51.9 | 1.2 | 2.5% | ok |
| metadata_transpose_view_f64_rank8_t4_julia-base | #2180 | julia-base | julia-base | f64 | shape=[2, 2, 2, 2, 2, 2, 2, 2] | 4 | 1048576 | 0.3 | 0.0 | 1.4% | ok |
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
| lu_batched_f64_factor_n8_b1024 | #2142 | tenferro | tenferro-rs | f64 | n=8, batch=1024, op=factor | 4 | 8 | 266994.9 | 11664.1 | 6.6% | ok |
| lu_batched_f64_factor_n8_b1024_host | #2142 | host | rust-loop | f64 | n=8, batch=1024, op=factor | 4 | 8 | 230062.5 | 18117.2 | 4.9% | ok |
| lu_batched_f64_solve_n8_b1024 | #2142 | tenferro | tenferro-rs | f64 | n=8, batch=1024, op=solve | 4 | 32 | 97662.8 | 16901.0 | 10.6% | NOISY |
| lu_batched_f64_solve_n8_b1024_pytorch | #2142 | pytorch | pytorch-cpu | f64 | n=8, batch=1024, op=solve | 4 | 32 | 76514.3 | 5072.3 | 4.0% | ok |
| lu_batched_f64_factor_n16_b1024 | #2142 | tenferro | tenferro-rs | f64 | n=16, batch=1024, op=factor | 4 | 4 | 731125.0 | 44468.6 | 30.4% | NOISY |
| lu_batched_f64_factor_n16_b1024_host | #2142 | host | rust-loop | f64 | n=16, batch=1024, op=factor | 4 | 2 | 964312.5 | 32208.2 | 2.8% | ok |
| lu_batched_f64_solve_n16_b1024 | #2142 | tenferro | tenferro-rs | f64 | n=16, batch=1024, op=solve | 4 | 16 | 190000.0 | 23496.1 | 28.3% | NOISY |
| lu_batched_f64_solve_n16_b1024_pytorch | #2142 | pytorch | pytorch-cpu | f64 | n=16, batch=1024, op=solve | 4 | 16 | 175315.1 | 2424.5 | 2.2% | ok |
| small_grad_qr_f64_trace-prepared_n4 | #2186 | trace-prepared | tenferro-rs | f64 | op=qr, n=4 | 4 | 64 | 31863.9 | 2006.2 | 29.6% | NOISY |
| small_grad_qr_f64_pytorch_n4 | #2186 | pytorch | pytorch-cpu | f64 | op=qr, n=4 | 4 | 16 | 130872.4 | 31765.6 | 16.2% | NOISY |
| small_grad_qr_f64_trace-prepared_n16 | #2186 | trace-prepared | tenferro-rs | f64 | op=qr, n=16 | 4 | 64 | 47554.0 | 5867.8 | 30.0% | NOISY |
| small_grad_qr_f64_pytorch_n16 | #2186 | pytorch | pytorch-cpu | f64 | op=qr, n=16 | 4 | 16 | 125716.2 | 2588.5 | 3.3% | ok |
| small_grad_eigh_f64_trace-prepared_n4 | #2186 | trace-prepared | tenferro-rs | f64 | op=eigh, n=4 | 4 | 64 | 41817.1 | 2408.2 | 3.3% | ok |
| small_grad_eigh_f64_pytorch_n4 | #2186 | pytorch | pytorch-cpu | f64 | op=eigh, n=4 | 4 | 64 | 38593.1 | 1181.6 | 2.5% | ok |
| small_grad_eigh_f64_trace-prepared_n16 | #2186 | trace-prepared | tenferro-rs | f64 | op=eigh, n=16 | 4 | 32 | 83084.6 | 4845.7 | 4.2% | ok |
| small_grad_eigh_f64_pytorch_n16 | #2186 | pytorch | pytorch-cpu | f64 | op=eigh, n=16 | 4 | 64 | 60296.9 | 166.0 | 0.2% | ok |
| small_grad_lu_f64_trace-prepared_n4 | #2186 | trace-prepared | tenferro-rs | f64 | op=lu, n=4 | 4 | 128 | 21165.7 | 1167.3 | 3.4% | ok |
| small_grad_lu_f64_pytorch_n4 | #2186 | pytorch | pytorch-cpu | f64 | op=lu, n=4 | 4 | 16 | 193583.3 | 56720.0 | 15.1% | NOISY |
| small_grad_lu_f64_trace-prepared_n16 | #2186 | trace-prepared | tenferro-rs | f64 | op=lu, n=16 | 4 | 64 | 31520.2 | 946.6 | 4.1% | ok |
| small_grad_lu_f64_pytorch_n16 | #2186 | pytorch | pytorch-cpu | f64 | op=lu, n=16 | 4 | 16 | 180375.0 | 3757.8 | 2.0% | ok |
| small_grad_svd_f64_trace-prepared_n4 | #2186 | trace-prepared | tenferro-rs | f64 | op=svd, n=4 | 4 | 32 | 67890.6 | 6225.9 | 38.2% | NOISY |
| small_grad_svd_f64_pytorch_n4 | #2186 | pytorch | pytorch-cpu | f64 | op=svd, n=4 | 4 | 64 | 55702.5 | 502.3 | 0.6% | ok |
| small_grad_svd_f64_trace-prepared_n16 | #2186 | trace-prepared | tenferro-rs | f64 | op=svd, n=16 | 4 | 16 | 151125.0 | 11803.4 | 4.2% | ok |
| small_grad_svd_f64_pytorch_n16 | #2186 | pytorch | pytorch-cpu | f64 | op=svd, n=16 | 4 | 32 | 94027.3 | 864.6 | 2.6% | ok |
| small_grad_solve_f64_trace-prepared_n4 | #2186 | trace-prepared | tenferro-rs | f64 | op=solve, n=4 | 4 | 128 | 18322.3 | 1212.9 | 3.7% | ok |
| small_grad_solve_f64_pytorch_n4 | #2186 | pytorch | pytorch-cpu | f64 | op=solve, n=4 | 4 | 64 | 31701.8 | 430.7 | 0.9% | ok |
| small_grad_solve_f64_trace-prepared_n16 | #2186 | trace-prepared | tenferro-rs | f64 | op=solve, n=16 | 4 | 128 | 24241.2 | 3068.0 | 6.1% | ok |
| small_grad_solve_f64_pytorch_n16 | #2186 | pytorch | pytorch-cpu | f64 | op=solve, n=16 | 4 | 64 | 39737.0 | 982.1 | 2.1% | ok |
| fft_axis0_c64_fft-executor_n128 | #1765 | fft-executor | tenferro-rs | c64 | n=128 | 4 | 1 | 3821208.0 | 2007229.0 | 33.3% | NOISY |
| fft_axis0_c64_pytorch_n128 | #1765 | pytorch | pytorch-cpu | c64 | n=128 | 4 | 1 | 18334917.0 | 1353166.5 | 5.0% | ok |
| degenerate_dot_hadamard_f64_dot-general-shared_n1024 | #2111 | dot-general-shared | tenferro-rs | f64 | form=hadamard, n=1024 | 4 | 4 | 199916.5 | 7557.2 | 79.7% | NOISY |
| degenerate_dot_hadamard_f64_pytorch_n1024 | #2111 | pytorch | pytorch-cpu | f64 | form=hadamard, n=1024 | 4 | 4 | 188291.8 | 17948.0 | 13.2% | NOISY |
| degenerate_dot_outer_f64_dot-general-shared_n1024 | #2111 | dot-general-shared | tenferro-rs | f64 | form=outer, n=1024 | 4 | 8 | 203369.8 | 13229.1 | 4.6% | ok |
| degenerate_dot_outer_f64_pytorch_n1024 | #2111 | pytorch | pytorch-cpu | f64 | form=outer, n=1024 | 4 | 8 | 176812.5 | 14669.3 | 5.3% | ok |
| degenerate_dot_rank0_f64_dot-general-shared_n1024 | #2111 | dot-general-shared | tenferro-rs | f64 | form=rank0, n=1024 | 4 | 4 | 251031.2 | 361520.8 | 48.2% | NOISY |
| degenerate_dot_rank0_f64_pytorch_n1024 | #2111 | pytorch | pytorch-cpu | f64 | form=rank0, n=1024 | 4 | 4 | 175000.0 | 14551.9 | 6.0% | ok |
| erased_add_f64_session-add_n256 | #1766 | session-add | tenferro-rs | f64 | n=256 | 4 | 2048 | 1028.7 | 40.3 | 15.5% | NOISY |
| erased_add_f64_pytorch_n256 | #1766 | pytorch | pytorch-cpu | f64 | n=256 | 4 | 2048 | 807.2 | 20.2 | 6.0% | ok |
| erased_add_f64_session-add_n4096 | #1766 | session-add | tenferro-rs | f64 | n=4096 | 4 | 1024 | 1702.6 | 94.7 | 4.1% | ok |
| erased_add_f64_pytorch_n4096 | #1766 | pytorch | pytorch-cpu | f64 | n=4096 | 4 | 1024 | 1614.3 | 120.1 | 4.3% | ok |
| erased_add_f64_session-add_n1048576 | #1766 | session-add | tenferro-rs | f64 | n=1048576 | 4 | 8 | 100020.9 | 54263.1 | 33.1% | NOISY |
| erased_add_f64_pytorch_n1048576 | #1766 | pytorch | pytorch-cpu | f64 | n=1048576 | 4 | 4 | 201031.2 | 43187.4 | 23.7% | NOISY |
| axpby_strided_source_f64_1024x1024_t4 | #2220 #2214 | strided-source | tenferro-rs | f64 | n=1024 | 4 | 2 | 724166.5 | 362500.0 | 38.5% | NOISY |
| axpby_contiguous_source_f64_1024x1024_t4 | #2220 #2214 | contiguous-source | tenferro-rs | f64 | n=1024 | 4 | 16 | 158752.6 | 30126.3 | 13.3% | NOISY |
| axpby_strided_source_f64_1024x1024_host_t4 | #2220 #2214 | host | rust-loop | f64 | n=1024 | 4 | 2 | 1234042.0 | 29645.8 | 3.7% | ok |
| erased_construct_f64_col_n8 | #2177 | col-major | tenferro-rs | f64 | n=8, layout=col | 4 | 32768 | 53.2 | 1.3 | 2.9% | ok |
| erased_construct_f64_row_n8 | #2177 | row-major | tenferro-rs | f64 | n=8, layout=row | 4 | 16384 | 130.4 | 7.4 | 5.8% | ok |
| erased_construct_f64_col_n4096 | #2177 | col-major | tenferro-rs | f64 | n=4096, layout=col | 4 | 16384 | 59.6 | 6.6 | 8.5% | ok |
| erased_construct_f64_row_n4096 | #2177 | row-major | tenferro-rs | f64 | n=4096, layout=row | 4 | 256 | 12483.2 | 105.8 | 2.5% | ok |
| erased_construct_f64_col_n65536 | #2177 | col-major | tenferro-rs | f64 | n=65536, layout=col | 4 | 1024 | 78.8 | 13.7 | 10.4% | NOISY |
| erased_construct_f64_row_n65536 | #2177 | row-major | tenferro-rs | f64 | n=65536, layout=row | 4 | 16 | 226687.5 | 21533.8 | 5.8% | ok |
| eager_small_leaf_f64_eager-held-session | #1704 | eager-held-session | tenferro-rs | f64 | op=leaf | 4 | 2048 | 1626.0 | 56.5 | 4.9% | ok |
| eager_small_add_f64_eager-held-session | #1667 | eager-held-session | tenferro-rs | f64 | op=add | 4 | 2048 | 1681.9 | 97.9 | 5.1% | ok |
| eager_small_add_f64_pytorch | #1667 | pytorch | pytorch-cpu | f64 | op=add | 4 | 4096 | 801.8 | 42.8 | 4.8% | ok |
| eager_small_gemm_f64_eager-held-session | #1904 | eager-held-session | tenferro-rs | f64 | op=gemm | 4 | 1024 | 2300.5 | 113.4 | 5.9% | ok |
| eager_small_gemm_f64_pytorch | #1904 | pytorch | pytorch-cpu | f64 | op=gemm | 4 | 2048 | 922.8 | 69.9 | 3.9% | ok |
| trace_metadata_trace-fold_f32_1024x64_k50 | #2226 #2238 | trace-fold | tenferro-rs | f32 | rows=1024, cols=64, steps=50 | 4 | 2 | 1534541.5 | 175468.8 | 14.1% | NOISY |
| trace_metadata_trace-outputs_f32_1024x64_k50 | #2226 #2238 | trace-outputs | tenferro-rs | f32 | rows=1024, cols=64, steps=50 | 4 | 1 | 1469959.0 | 182500.5 | 62.8% | NOISY |
| trace_metadata_eager_f32_1024x64_k50 | #2226 #2238 | eager | tenferro-rs | f32 | rows=1024, cols=64, steps=50 | 4 | 1 | 2028625.0 | 156832.5 | 22.2% | NOISY |
| trace_metadata_pytorch_f32_1024x64_k50 | #2226 #2238 | pytorch | pytorch-cpu | f32 | rows=1024, cols=64, steps=50 | 4 | 1 | 2918000.0 | 189665.5 | 6.8% | ok |
| trace_metadata_fixed_neg_f32_8 | #2226 #2238 | trace-fold | tenferro-rs | f32 | rows=8, cols=1, steps=0 | 4 | 2048 | 1774.7 | 122.3 | 5.3% | ok |


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
