# CPU Performance-Issue Workloads

- Suite: `cpu/perf_issues`
- Raw run: `data/results/zen5-cpu/cpu/perf_issues/20261112_155453`

## CPU Information

- Model: `AMD Ryzen AI 9 HX 470 w/ Radeon 890M`
- Vendor: `AuthenticAMD`
- Logical CPUs: `24`
- Sockets: `1`
- Cores per socket: `12`
- Threads per core: `2`
- NUMA nodes: `1`
- Transparent huge pages: `madvise (always [madvise] never)`
- Transparent huge page defrag: `madvise (always defer defer+madvise [madvise] never)`
- Python platform: `Linux-7.0.0-38-generic-x86_64-with-glibc2.39`
- L3 domains (shared CPUs): `0-3,12-15; 4-11,16-23`
- CPU boost: `disabled`
- Container: `none`

## run_t1

Full metadata: `data/results/zen5-cpu/cpu/perf_issues/20261112_155453/run_t1.yaml`

```yaml
timestamp: '2026-11-12T15:54:54Z'
tenferro_rs:
  path: /home/shinaoka/projects/tensor4all/tenferro-benchmark/extern/tenferro-rs
  commit: 9a850b06155405acc99ecd0648829e9e6d3866ba
  dirty: false
  features:
  - native
  resolved_path: /home/shinaoka/projects/tensor4all/tenferro-benchmark/extern/tenferro-rs
blas:
  implementation: none
environment:
  container: none
  hostname: shinaoka-EVO-X1Pro
  os: Linux-7.0.0-38-generic-x86_64-with-glibc2.39
  arch: x86_64
  cpu: x86_64
  affinity:
    pinning: taskset
    requested_measured_cpus: '0'
    observer_cpus: 4-11,16-23
    override_of: null
    thread_budget: '1'
    measured_cpus: '0'
    l3_domains:
    - 0-3,12-15
    - 4-11,16-23
    cpu_boost: disabled
    scaling_governor: performance
    scaling_driver: amd-pstate-epp
  env:
    OMP_NUM_THREADS: '1'
    OMP_THREAD_LIMIT: '1'
    OMP_DYNAMIC: 'FALSE'
    RAYON_NUM_THREADS: '1'
    BENCHMARK_TARGET_PROFILE: zen5-cpu
    BENCHMARK_COMMIT: 918046a6065be464f13b727654679c3a00650c06
    OPENBLAS_ROOT: /usr
    MKLROOT: /opt/intel/oneapi/mkl/latest
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
    PUBLICATION_GATE_PROFILE: full
    BENCHMARK_BUILD_PROFILE: release
    RUSTC_VERSION: rustc 1.99.0 (b940084d7 2026-09-28)
    BENCH_INSTANCE: null
```

## run_t4

Full metadata: `data/results/zen5-cpu/cpu/perf_issues/20261112_155453/run_t4.yaml`

```yaml
timestamp: '2026-11-12T15:59:04Z'
tenferro_rs:
  path: /home/shinaoka/projects/tensor4all/tenferro-benchmark/extern/tenferro-rs
  commit: 9a850b06155405acc99ecd0648829e9e6d3866ba
  dirty: false
  features:
  - native
  resolved_path: /home/shinaoka/projects/tensor4all/tenferro-benchmark/extern/tenferro-rs
blas:
  implementation: none
environment:
  container: none
  hostname: shinaoka-EVO-X1Pro
  os: Linux-7.0.0-38-generic-x86_64-with-glibc2.39
  arch: x86_64
  cpu: x86_64
  affinity:
    pinning: taskset
    requested_measured_cpus: 0-3
    observer_cpus: 4-11,16-23
    override_of: null
    thread_budget: '4'
    measured_cpus: 0-3
    l3_domains:
    - 0-3,12-15
    - 4-11,16-23
    cpu_boost: disabled
    scaling_governor: performance
    scaling_driver: amd-pstate-epp
  env:
    OMP_NUM_THREADS: '4'
    OMP_THREAD_LIMIT: '4'
    OMP_DYNAMIC: 'FALSE'
    RAYON_NUM_THREADS: '4'
    BENCHMARK_TARGET_PROFILE: zen5-cpu
    BENCHMARK_COMMIT: 918046a6065be464f13b727654679c3a00650c06
    OPENBLAS_ROOT: /usr
    MKLROOT: /opt/intel/oneapi/mkl/latest
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
    PUBLICATION_GATE_PROFILE: full
    BENCHMARK_BUILD_PROFILE: release
    RUSTC_VERSION: rustc 1.99.0 (b940084d7 2026-09-28)
    BENCH_INSTANCE: null
```

## run_t8

Full metadata: `data/results/zen5-cpu/cpu/perf_issues/20261112_155453/run_t8.yaml`

```yaml
timestamp: '2026-11-12T16:02:44Z'
tenferro_rs:
  path: /home/shinaoka/projects/tensor4all/tenferro-benchmark/extern/tenferro-rs
  commit: 9a850b06155405acc99ecd0648829e9e6d3866ba
  dirty: false
  features:
  - native
  resolved_path: /home/shinaoka/projects/tensor4all/tenferro-benchmark/extern/tenferro-rs
blas:
  implementation: none
environment:
  container: none
  hostname: shinaoka-EVO-X1Pro
  os: Linux-7.0.0-38-generic-x86_64-with-glibc2.39
  arch: x86_64
  cpu: x86_64
  affinity:
    pinning: taskset
    requested_measured_cpus: 4-11
    observer_cpus: 0-3,12-15
    override_of: null
    thread_budget: '8'
    measured_cpus: 4-11
    l3_domains:
    - 0-3,12-15
    - 4-11,16-23
    cpu_boost: disabled
    scaling_governor: performance
    scaling_driver: amd-pstate-epp
  env:
    OMP_NUM_THREADS: '8'
    OMP_THREAD_LIMIT: '8'
    OMP_DYNAMIC: 'FALSE'
    RAYON_NUM_THREADS: '8'
    BENCHMARK_TARGET_PROFILE: zen5-cpu
    BENCHMARK_COMMIT: 918046a6065be464f13b727654679c3a00650c06
    OPENBLAS_ROOT: /usr
    MKLROOT: /opt/intel/oneapi/mkl/latest
    OPENBLAS_NUM_THREADS: '8'
    MATMUL_NUM_THREADS: '8'
    GOTO_NUM_THREADS: '8'
    MKL_NUM_THREADS: '8'
    MKL_DYNAMIC: 'FALSE'
    VECLIB_MAXIMUM_THREADS: '8'
    VECLIB_NUM_THREADS: '8'
    NUMEXPR_NUM_THREADS: '8'
    BLIS_NUM_THREADS: '8'
    PJRT_NPROC: '8'
    XLA_FLAGS: --xla_cpu_multi_thread_eigen=true --xla_cpu_experimental_ynn_fusion_type=
      intra_op_parallelism_threads=8
    JULIA_NUM_THREADS: '8'
    PUBLICATION_GATE_PROFILE: full
    BENCHMARK_BUILD_PROFILE: release
    RUSTC_VERSION: rustc 1.99.0 (b940084d7 2026-09-28)
    BENCH_INSTANCE: null
```

## Case status

- Manifest: `cpu/perf_issues` version 22, coverage `quick`, effort `standard`
- Expected 957, selected 957, executed 675, unsupported 0, failed 282, skipped 0, missing 0, noisy 23.
- Complete: **NO** — only executed cases count as covered; unsupported, skipped, failed, missing and unselected cases never do.

| expected | selected | executed | unsupported | failed | skipped | missing | noisy |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 957 | 957 | 675 | 0 | 282 | 0 | 0 | 23 |

- Failed: `activation_erf_f32_eager-single-call_1024x64@t1`, `activation_erf_f32_pytorch_1024x64@t1`, `activation_sigmoid_f32_eager-single-call_1024x64@t1`, `activation_sigmoid_f32_pytorch_1024x64@t1`, `activation_silu_f32_eager-single-call_1024x64@t1`, `activation_silu_f32_pytorch_1024x64@t1`, `activation_softplus_f32_eager-single-call_1024x64@t1`, `activation_softplus_f32_pytorch_1024x64@t1`, `activation_gelu_tanh_f32_eager-single-call_1024x64@t1`, `activation_gelu_tanh_f32_pytorch_1024x64@t1`, `followup_structural_broadcast_in_dim_direct-shared@t1`, `followup_structural_broadcast_in_dim_pytorch@t1`, `followup_real_tanh_direct-shared@t1`, `followup_real_tanh_jax@t1`, `followup_real_log1p_direct-shared@t1`, `followup_real_log1p_pytorch@t1`, `followup_real_cos_direct-shared@t1`, `followup_real_cos_pytorch@t1`, `followup_real_sin_direct-shared@t1`, `followup_real_sin_pytorch@t1`, `followup_real_exp_direct-shared@t1`, `followup_real_exp_jax@t1`, `followup_real_expm1_direct-shared@t1`, `followup_real_expm1_jax@t1`, `followup_real_pow_direct-shared@t1`, `followup_real_pow_pytorch@t1`, `followup_complex_exp_direct-shared@t1`, `followup_complex_exp_julia-base@t1`, `followup_real_log_direct-shared@t1`, `followup_real_log_pytorch@t1`, `followup_real_reduce_max_axis1_direct-shared@t1`, `followup_real_reduce_max_axis1_julia-base@t1`, `followup_real_reduce_prod_axis1_direct-shared@t1`, `followup_real_reduce_prod_axis1_pytorch@t1`, `followup_real_reduce_sum_axis1_direct-shared@t1`, `followup_real_reduce_sum_axis1_pytorch@t1`, `followup_real_reduce_max_axis0_direct-shared@t1`, `followup_real_reduce_max_axis0_julia-base@t1`, `followup_real_reduce_min_axis1_direct-shared@t1`, `followup_real_reduce_min_axis1_julia-base@t1` (+242 more)
- Noisy: `layer_norm_f32_eager-single-call_d1024_len8_b1@t1`, `metadata_transpose_view_f64_rank8_t1@t1`, `typed_elemwise_f64_add_held_len16_pytorch@t1`, `lu_batched_f64_factor_n16_b1024_host@t1`, `lu_batched_f64_solve_n16_b1024@t1`, `axpby_strided_source_f64_1024x1024_t1@t1`, `erased_construct_f64_col_n8@t1`, `decode_proj_f32_faer-direct_in1024_out1024_len8@t4`, `decode_proj_f32_faer-direct_in1024_out1024_len64@t4`, `conj_dot_f64_pytorch-vdot_len10000@t4`, `conj_dot_c64_pytorch-vdot_len100@t4`, `tanh_chain_f32_compiled-prepared_k10_1024x64@t4`, `metadata_transpose_view_f64_rank8_t4@t4`, `small_grad_qr_f64_trace-prepared_n16@t4`, `degenerate_dot_outer_f64_pytorch_n1024@t4`, `erased_construct_f64_col_n8@t4`, `conj_dot_f64_pytorch-vdot_len1000000@t8`, `conj_dot_c64_pytorch-vdot_len100@t8`, `small_solve_f64_pytorch-solve-triangular-batched_k16_b64@t8`, `layer_norm_f32_eager-single-call_d1024_len8_b8@t8`, `softmax_f32_eager-single-call_len512_b8@t8`, `softmax_f32_session-single-call_len512_b8@t8`, `erased_construct_f64_col_n8@t8`

Numerical checks: **675/957 recorded rows passed** (case × thread count).

## Steady-state operation rows

Each batch is one wall-clock interval over many operations; every output is retained until the clock stops. Median and inclusive IQR are per operation, in nanoseconds; CoV is sample standard deviation / mean (NOISY means CoV > 10%, a descriptive label only). Setup (inputs, backends, runtimes, session entry, tracing/compilation, preparation) is outside timing; faer-direct and host-sgemm write into a preallocated output, the tenferro and PyTorch arms allocate their output. The thin_complex_gemm faer-direct arm instead allocates its output inside timing to match its product references. `reference_for` in the case file names the case each reference arm is compared with. GPU batches end with one device synchronize (a per-call synchronize where the transfer is the operation); the Threads column is the host thread setting.

### Compute- and bandwidth-bound operations

| Case ID | Issues | Arm | Backend | Dtype | Workload | Threads | Ops/batch | Median ns/op | IQR ns | CoV | Status |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| softmax_f64_pool_serial_1024x4096 | #2273 #2274 #2276 | serial | tenferro-rs | f64 | rows=1024, cols=4096, axis=0 | 1 | 1 | 17700946.0 | 23949.5 | 0.1% | ok |
| softmax_f64_pool_pytorch_t1_1024x4096 | #2273 #2274 #2276 | pytorch | pytorch-cpu | f64 | rows=1024, cols=4096, axis=0 | 1 | 1 | 19929798.0 | 121130.0 | 0.4% | ok |
| softmax_f64_pool_pool4_1024x4096 | #2273 #2274 #2276 | pool4 | tenferro-rs | f64 | rows=1024, cols=4096, axis=0 | 4 | 1 | 4489787.0 | 18044.0 | 0.5% | ok |
| softmax_f64_pool_pytorch_t4_1024x4096 | #2273 #2274 #2276 | pytorch | pytorch-cpu | f64 | rows=1024, cols=4096, axis=0 | 4 | 1 | 5638458.0 | 188506.0 | 2.7% | ok |

### Latency and per-call overhead

These rows measure dispatch/operation latency; they do not measure compute throughput.

| Case ID | Issues | Arm | Backend | Dtype | Workload | Threads | Ops/batch | Median ns/op | IQR ns | CoV | Status |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| two_operand_einsum_f64_2x2_string | #2325 #2324 | string | tenferro-rs | f64 | m=2, k=2, n=2, format=ij,jk->ik | 1 | 512 | 3854.3 | 9.9 | 0.6% | ok |
| two_operand_einsum_f64_2x2_subscripts | #2325 #2324 | subscripts | tenferro-rs | f64 | m=2, k=2, n=2, format=ij,jk->ik | 1 | 1024 | 2613.5 | 12.5 | 0.8% | ok |
| two_operand_einsum_f64_2x2_prepared | #2325 #2324 | prepared | tenferro-rs | f64 | m=2, k=2, n=2, format=ij,jk->ik | 1 | 1024 | 2107.8 | 7.8 | 1.0% | ok |
| two_operand_einsum_f64_2x2_dot-general | #2325 #2324 | dot-general | tenferro-rs | f64 | m=2, k=2, n=2, format=ij,jk->ik | 1 | 1024 | 2031.3 | 7.8 | 0.9% | ok |
| two_operand_einsum_f64_2x2_pytorch | #2325 #2324 | pytorch | pytorch-cpu | f64 | m=2, k=2, n=2, format=ij,jk->ik | 1 | 256 | 13880.8 | 162.8 | 0.7% | ok |
| two_operand_einsum_f64_16x16_string | #2325 #2324 | string | tenferro-rs | f64 | m=16, k=16, n=16, format=ij,jk->ik | 1 | 512 | 4059.2 | 12.2 | 0.7% | ok |
| two_operand_einsum_f64_16x16_subscripts | #2325 #2324 | subscripts | tenferro-rs | f64 | m=16, k=16, n=16, format=ij,jk->ik | 1 | 1024 | 2859.1 | 7.7 | 1.0% | ok |
| two_operand_einsum_f64_16x16_prepared | #2325 #2324 | prepared | tenferro-rs | f64 | m=16, k=16, n=16, format=ij,jk->ik | 1 | 1024 | 2354.6 | 11.3 | 0.9% | ok |
| two_operand_einsum_f64_16x16_dot-general | #2325 #2324 | dot-general | tenferro-rs | f64 | m=16, k=16, n=16, format=ij,jk->ik | 1 | 1024 | 2323.2 | 7.6 | 1.5% | ok |
| two_operand_einsum_f64_16x16_pytorch | #2325 #2324 | pytorch | pytorch-cpu | f64 | m=16, k=16, n=16, format=ij,jk->ik | 1 | 128 | 15142.2 | 110.9 | 0.7% | ok |
| two_operand_einsum_c64_2x2_string | #2325 #2324 | string | tenferro-rs | c64 | m=2, k=2, n=2, format=ij,jk->ik | 1 | 512 | 3845.6 | 6.3 | 0.6% | ok |
| two_operand_einsum_c64_2x2_subscripts | #2325 #2324 | subscripts | tenferro-rs | c64 | m=2, k=2, n=2, format=ij,jk->ik | 1 | 1024 | 2587.1 | 6.4 | 0.8% | ok |
| two_operand_einsum_c64_2x2_prepared | #2325 #2324 | prepared | tenferro-rs | c64 | m=2, k=2, n=2, format=ij,jk->ik | 1 | 1024 | 2072.4 | 3.3 | 1.0% | ok |
| two_operand_einsum_c64_2x2_dot-general | #2325 #2324 | dot-general | tenferro-rs | c64 | m=2, k=2, n=2, format=ij,jk->ik | 1 | 1024 | 2047.9 | 4.4 | 1.1% | ok |
| two_operand_einsum_c64_2x2_pytorch | #2325 #2324 | pytorch | pytorch-cpu | c64 | m=2, k=2, n=2, format=ij,jk->ik | 1 | 256 | 13849.7 | 72.4 | 0.7% | ok |
| two_operand_einsum_c64_16x16_string | #2325 #2324 | string | tenferro-rs | c64 | m=16, k=16, n=16, format=ij,jk->ik | 1 | 512 | 4948.5 | 20.4 | 0.6% | ok |
| two_operand_einsum_c64_16x16_subscripts | #2325 #2324 | subscripts | tenferro-rs | c64 | m=16, k=16, n=16, format=ij,jk->ik | 1 | 512 | 3711.4 | 25.6 | 0.9% | ok |
| two_operand_einsum_c64_16x16_prepared | #2325 #2324 | prepared | tenferro-rs | c64 | m=16, k=16, n=16, format=ij,jk->ik | 1 | 1024 | 3213.8 | 13.8 | 0.9% | ok |
| two_operand_einsum_c64_16x16_dot-general | #2325 #2324 | dot-general | tenferro-rs | c64 | m=16, k=16, n=16, format=ij,jk->ik | 1 | 1024 | 3175.8 | 7.8 | 0.8% | ok |
| two_operand_einsum_c64_16x16_pytorch | #2325 #2324 | pytorch | pytorch-cpu | c64 | m=16, k=16, n=16, format=ij,jk->ik | 1 | 128 | 15869.2 | 67.3 | 0.4% | ok |
| two_operand_einsum_f64_abc_cef_abef_string | #2325 #2324 | string | tenferro-rs | f64 | m=8, k=2, n=16, format=abc,cef->abef | 1 | 512 | 4544.6 | 18.9 | 0.7% | ok |
| two_operand_einsum_f64_abc_cef_abef_subscripts | #2325 #2324 | subscripts | tenferro-rs | f64 | m=8, k=2, n=16, format=abc,cef->abef | 1 | 1024 | 2737.1 | 11.8 | 0.8% | ok |
| two_operand_einsum_f64_abc_cef_abef_prepared | #2325 #2324 | prepared | tenferro-rs | f64 | m=8, k=2, n=16, format=abc,cef->abef | 1 | 1024 | 2201.0 | 9.2 | 1.0% | ok |
| two_operand_einsum_f64_abc_cef_abef_dot-general | #2325 #2324 | dot-general | tenferro-rs | f64 | m=8, k=2, n=16, format=abc,cef->abef | 1 | 1024 | 2166.2 | 5.7 | 1.2% | ok |
| two_operand_einsum_f64_abc_cef_abef_pytorch | #2325 #2324 | pytorch | pytorch-cpu | f64 | m=8, k=2, n=16, format=abc,cef->abef | 1 | 128 | 16979.7 | 218.3 | 0.8% | ok |
| thin_c64_gemm_1x141_141x282_t1 | #2416 #1099 | dot-general | tenferro-rs | c64 | m=1, k=141, n=282 | 1 | 128 | 20718.7 | 77.1 | 0.3% | ok |
| thin_c64_gemm_1x141_141x282_t1_faer-direct | #2416 #1099 | faer-direct | faer-direct | c64 | m=1, k=141, n=282 | 1 | 128 | 18161.2 | 47.4 | 0.2% | ok |
| thin_c64_gemm_1x141_141x282_t1_pytorch | #2416 #1099 | pytorch | pytorch-cpu | c64 | m=1, k=141, n=282 | 1 | 64 | 49919.2 | 60.5 | 0.3% | ok |
| thin_c64_gemm_2x141_141x282_t1 | #2416 #1099 | dot-general | tenferro-rs | c64 | m=2, k=141, n=282 | 1 | 64 | 50472.0 | 120.9 | 0.3% | ok |
| thin_c64_gemm_2x141_141x282_t1_faer-direct | #2416 #1099 | faer-direct | faer-direct | c64 | m=2, k=141, n=282 | 1 | 64 | 50781.7 | 172.8 | 7.2% | ok |
| thin_c64_gemm_2x141_141x282_t1_pytorch | #2416 #1099 | pytorch | pytorch-cpu | c64 | m=2, k=141, n=282 | 1 | 32 | 69955.9 | 178.5 | 0.3% | ok |
| thin_c64_gemm_8x141_141x282_t1 | #2416 #1099 | dot-general | tenferro-rs | c64 | m=8, k=141, n=282 | 1 | 16 | 131139.1 | 436.5 | 0.2% | ok |
| thin_c64_gemm_8x141_141x282_t1_faer-direct | #2416 #1099 | faer-direct | faer-direct | c64 | m=8, k=141, n=282 | 1 | 32 | 91400.8 | 140.4 | 0.2% | ok |
| thin_c64_gemm_8x141_141x282_t1_pytorch | #2416 #1099 | pytorch | pytorch-cpu | c64 | m=8, k=141, n=282 | 1 | 32 | 125374.7 | 610.0 | 0.5% | ok |
| decode_proj_f32_eager-shared_in1024_out1024_len8 | #1992 #2003 #1995 #2229 | eager-shared | tenferro-rs | f32 | in=1024, out=1024, len=8 | 1 | 4 | 587423.8 | 5487.6 | 0.7% | ok |
| decode_proj_f32_faer-direct_in1024_out1024_len8 | #1992 #2003 #1995 #2229 | faer-direct | faer-direct | f32 | in=1024, out=1024, len=8 | 1 | 4 | 584550.8 | 3304.9 | 0.4% | ok |
| decode_proj_f32_host-sgemm_in1024_out1024_len8 | #1992 #2003 #1995 #2229 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=1024, len=8 | 1 | 2 | 1468349.0 | 1570.5 | 0.2% | ok |
| decode_proj_f32_pytorch_in1024_out1024_len8 | #1992 #2003 #1995 #2229 | pytorch | pytorch-cpu | f32 | in=1024, out=1024, len=8 | 1 | 4 | 558662.5 | 1852.2 | 0.7% | ok |
| decode_proj_f32_eager-shared_in1024_out1024_len64 | #1992 #2003 #1995 #2229 | eager-shared | tenferro-rs | f32 | in=1024, out=1024, len=64 | 1 | 1 | 2176172.0 | 6822.5 | 0.4% | ok |
| decode_proj_f32_faer-direct_in1024_out1024_len64 | #1992 #2003 #1995 #2229 | faer-direct | faer-direct | f32 | in=1024, out=1024, len=64 | 1 | 1 | 2169429.0 | 4478.5 | 0.3% | ok |
| decode_proj_f32_host-sgemm_in1024_out1024_len64 | #1992 #2003 #1995 #2229 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=1024, len=64 | 1 | 1 | 3074765.0 | 12218.0 | 0.4% | ok |
| decode_proj_f32_pytorch_in1024_out1024_len64 | #1992 #2003 #1995 #2229 | pytorch | pytorch-cpu | f32 | in=1024, out=1024, len=64 | 1 | 1 | 2753557.0 | 9932.5 | 1.4% | ok |
| decode_proj_f32_eager-shared_in1024_out1024_len512 | #1992 #2003 #1995 #2229 | eager-shared | tenferro-rs | f32 | in=1024, out=1024, len=512 | 1 | 1 | 17256009.0 | 18479.5 | 0.1% | ok |
| decode_proj_f32_faer-direct_in1024_out1024_len512 | #1992 #2003 #1995 #2229 | faer-direct | faer-direct | f32 | in=1024, out=1024, len=512 | 1 | 1 | 17271888.0 | 28568.5 | 1.3% | ok |
| decode_proj_f32_host-sgemm_in1024_out1024_len512 | #1992 #2003 #1995 #2229 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=1024, len=512 | 1 | 1 | 18443630.0 | 24065.0 | 0.1% | ok |
| decode_proj_f32_pytorch_in1024_out1024_len512 | #1992 #2003 #1995 #2229 | pytorch | pytorch-cpu | f32 | in=1024, out=1024, len=512 | 1 | 1 | 20123918.0 | 70441.0 | 0.2% | ok |
| decode_proj_f32_eager-shared_in1024_out4096_len8 | #1992 #2003 #1995 #2229 | eager-shared | tenferro-rs | f32 | in=1024, out=4096, len=8 | 1 | 1 | 2331581.0 | 12193.0 | 0.6% | ok |
| decode_proj_f32_faer-direct_in1024_out4096_len8 | #1992 #2003 #1995 #2229 | faer-direct | faer-direct | f32 | in=1024, out=4096, len=8 | 1 | 1 | 2332603.0 | 14597.5 | 0.5% | ok |
| decode_proj_f32_host-sgemm_in1024_out4096_len8 | #1992 #2003 #1995 #2229 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=4096, len=8 | 1 | 1 | 5996605.0 | 18399.5 | 0.6% | ok |
| decode_proj_f32_pytorch_in1024_out4096_len8 | #1992 #2003 #1995 #2229 | pytorch | pytorch-cpu | f32 | in=1024, out=4096, len=8 | 1 | 1 | 2470139.0 | 41026.5 | 2.8% | ok |
| decode_proj_f32_eager-shared_in1024_out4096_len64 | #1992 #2003 #1995 #2229 | eager-shared | tenferro-rs | f32 | in=1024, out=4096, len=64 | 1 | 1 | 8716579.0 | 60206.5 | 0.7% | ok |
| decode_proj_f32_faer-direct_in1024_out4096_len64 | #1992 #2003 #1995 #2229 | faer-direct | faer-direct | f32 | in=1024, out=4096, len=64 | 1 | 1 | 8691162.0 | 6887.5 | 0.1% | ok |
| decode_proj_f32_host-sgemm_in1024_out4096_len64 | #1992 #2003 #1995 #2229 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=4096, len=64 | 1 | 1 | 12583319.0 | 32365.0 | 0.2% | ok |
| decode_proj_f32_pytorch_in1024_out4096_len64 | #1992 #2003 #1995 #2229 | pytorch | pytorch-cpu | f32 | in=1024, out=4096, len=64 | 1 | 1 | 11253173.0 | 47288.0 | 0.4% | ok |
| decode_proj_f32_eager-shared_in1024_out4096_len512 | #1992 #2003 #1995 #2229 | eager-shared | tenferro-rs | f32 | in=1024, out=4096, len=512 | 1 | 1 | 69833955.0 | 93925.5 | 0.3% | ok |
| decode_proj_f32_faer-direct_in1024_out4096_len512 | #1992 #2003 #1995 #2229 | faer-direct | faer-direct | f32 | in=1024, out=4096, len=512 | 1 | 1 | 69659772.0 | 43155.0 | 0.3% | ok |
| decode_proj_f32_host-sgemm_in1024_out4096_len512 | #1992 #2003 #1995 #2229 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=4096, len=512 | 1 | 1 | 74084340.0 | 97637.0 | 0.1% | ok |
| decode_proj_f32_pytorch_in1024_out4096_len512 | #1992 #2003 #1995 #2229 | pytorch | pytorch-cpu | f32 | in=1024, out=4096, len=512 | 1 | 1 | 81537610.0 | 229512.5 | 0.5% | ok |
| decode_proj_view_operand_f32_prepared_in1024_out1024_len8_t1 | #2216 | prepared-view | tenferro-rs | f32 | in=1024, out=1024, len=8 | 1 | 4 | 567065.8 | 4609.9 | 2.0% | ok |
| decode_proj_contiguous_operand_f32_prepared_in1024_out1024_len8_t1 | #2216 | prepared-contiguous | tenferro-rs | f32 | in=1024, out=1024, len=8 | 1 | 4 | 575075.8 | 2739.1 | 0.6% | ok |
| decode_proj_view_operand_f32_pytorch_in1024_out1024_len8_t1 | #2216 | pytorch | pytorch-cpu | f32 | in=1024, out=1024, len=8 | 1 | 4 | 514064.8 | 2574.8 | 0.9% | ok |
| gemm_c64_mm256_t1 | #1900 | dot-general-shared | tenferro-rs | c64 | n=256 | 1 | 1 | 4437252.0 | 5871.0 | 0.2% | ok |
| gemm_f64_mm256_t1 | #1900 | dot-general-shared | tenferro-rs | f64 | n=256 | 1 | 2 | 1074436.0 | 2181.5 | 0.2% | ok |
| conj_dot_f64_dot-general-rank1_len100 | #1615 | dot-general-rank1 | tenferro-rs | f64 | len=100 | 1 | 1024 | 1864.3 | 6.1 | 1.0% | ok |
| conj_dot_f64_portable-loop_len100 | #1615 | portable-loop | rust-loop | f64 | len=100 | 1 | 32768 | 89.6 | 0.1 | 0.4% | ok |
| conj_dot_f64_pytorch-vdot_len100 | #1615 | pytorch-vdot | pytorch-cpu | f64 | len=100 | 1 | 2048 | 1677.7 | 2.5 | 5.1% | ok |
| conj_dot_f64_dot-general-rank1_len10000 | #1615 | dot-general-rank1 | tenferro-rs | f64 | len=10000 | 1 | 512 | 4456.7 | 7.7 | 0.4% | ok |
| conj_dot_f64_portable-loop_len10000 | #1615 | portable-loop | rust-loop | f64 | len=10000 | 1 | 256 | 15040.0 | 14.1 | 0.1% | ok |
| conj_dot_f64_pytorch-vdot_len10000 | #1615 | pytorch-vdot | pytorch-cpu | f64 | len=10000 | 1 | 512 | 4246.7 | 11.2 | 0.4% | ok |
| conj_dot_f64_dot-general-rank1_len1000000 | #1615 | dot-general-rank1 | tenferro-rs | f64 | len=1000000 | 1 | 8 | 370933.1 | 10683.0 | 3.4% | ok |
| conj_dot_f64_portable-loop_len1000000 | #1615 | portable-loop | rust-loop | f64 | len=1000000 | 1 | 2 | 1547842.0 | 5467.2 | 0.5% | ok |
| conj_dot_f64_pytorch-vdot_len1000000 | #1615 | pytorch-vdot | pytorch-cpu | f64 | len=1000000 | 1 | 8 | 308349.1 | 18344.2 | 4.2% | ok |
| conj_dot_c64_dot-general-rank1_len100 | #1615 | dot-general-rank1 | tenferro-rs | c64 | len=100 | 1 | 1024 | 1943.1 | 4.0 | 1.5% | ok |
| conj_dot_c64_portable-loop_len100 | #1615 | portable-loop | rust-loop | c64 | len=100 | 1 | 16384 | 143.9 | 0.3 | 0.6% | ok |
| conj_dot_c64_pytorch-vdot_len100 | #1615 | pytorch-vdot | pytorch-cpu | c64 | len=100 | 1 | 2048 | 1564.2 | 4.0 | 0.7% | ok |
| conj_dot_c64_dot-general-rank1_len10000 | #1615 | dot-general-rank1 | tenferro-rs | c64 | len=10000 | 1 | 128 | 15794.4 | 26.1 | 3.9% | ok |
| conj_dot_c64_portable-loop_len10000 | #1615 | portable-loop | rust-loop | c64 | len=10000 | 1 | 256 | 15279.7 | 32.5 | 0.3% | ok |
| conj_dot_c64_pytorch-vdot_len10000 | #1615 | pytorch-vdot | pytorch-cpu | c64 | len=10000 | 1 | 256 | 9548.2 | 69.2 | 0.9% | ok |
| conj_dot_c64_dot-general-rank1_len1000000 | #1615 | dot-general-rank1 | tenferro-rs | c64 | len=1000000 | 1 | 2 | 1466370.5 | 5122.0 | 0.5% | ok |
| conj_dot_c64_portable-loop_len1000000 | #1615 | portable-loop | rust-loop | c64 | len=1000000 | 1 | 2 | 1551799.5 | 6732.5 | 0.4% | ok |
| conj_dot_c64_pytorch-vdot_len1000000 | #1615 | pytorch-vdot | pytorch-cpu | c64 | len=1000000 | 1 | 4 | 865621.8 | 7366.2 | 0.6% | ok |
| small_solve_f64_solve-loop_k16_b64 | #2007 #1878 | solve-loop | tenferro-rs | f64 | k=16, batch=64 | 1 | 8 | 329457.2 | 497.8 | 0.3% | ok |
| small_solve_f64_triangular-solve-loop_k16_b64 | #2007 #1878 | triangular-solve-loop | tenferro-rs | f64 | k=16, batch=64 | 1 | 16 | 173221.2 | 301.8 | 0.4% | ok |
| small_solve_f64_solve-batched_k16_b64 | #2007 #1878 | solve-batched | tenferro-rs | f64 | k=16, batch=64 | 1 | 16 | 187063.3 | 754.2 | 0.3% | ok |
| small_solve_f64_pytorch-solve-batched_k16_b64 | #2007 #1878 | pytorch-solve-batched | pytorch-cpu | f64 | k=16, batch=64 | 1 | 16 | 236119.6 | 325.9 | 0.1% | ok |
| small_solve_f64_pytorch-solve-triangular-batched_k16_b64 | #2007 #1878 | pytorch-solve-triangular-batched | pytorch-cpu | f64 | k=16, batch=64 | 1 | 32 | 92531.7 | 101.4 | 0.1% | ok |
| small_solve_f64_solve-loop_k64_b64 | #2007 #1878 | solve-loop | tenferro-rs | f64 | k=64, batch=64 | 1 | 1 | 3341953.0 | 7974.5 | 0.2% | ok |
| small_solve_f64_triangular-solve-loop_k64_b64 | #2007 #1878 | triangular-solve-loop | tenferro-rs | f64 | k=64, batch=64 | 1 | 2 | 1687983.0 | 2702.2 | 0.3% | ok |
| small_solve_f64_solve-batched_k64_b64 | #2007 #1878 | solve-batched | tenferro-rs | f64 | k=64, batch=64 | 1 | 1 | 3164222.0 | 6272.0 | 0.3% | ok |
| small_solve_f64_pytorch-solve-batched_k64_b64 | #2007 #1878 | pytorch-solve-batched | pytorch-cpu | f64 | k=64, batch=64 | 1 | 1 | 6109856.0 | 18293.5 | 0.4% | ok |
| small_solve_f64_pytorch-solve-triangular-batched_k64_b64 | #2007 #1878 | pytorch-solve-triangular-batched | pytorch-cpu | f64 | k=64, batch=64 | 1 | 2 | 1949620.5 | 3123.0 | 0.1% | ok |
| tanh_chain_f32_compiled-prepared_k10_1024x64 | #1990 | compiled-prepared | tenferro-rs | f32 | rows=1024, cols=64, chain=10 | 1 | 2 | 1724896.5 | 6148.8 | 0.5% | ok |
| tanh_chain_f32_eager-shared_k10_1024x64 | #1990 | eager-shared | tenferro-rs | f32 | rows=1024, cols=64, chain=10 | 1 | 2 | 1729340.0 | 7526.5 | 0.4% | ok |
| tanh_chain_f32_compiled-prepared_k100_1024x64 | #1990 | compiled-prepared | tenferro-rs | f32 | rows=1024, cols=64, chain=100 | 1 | 1 | 17322765.0 | 96946.0 | 1.7% | ok |
| tanh_chain_f32_eager-shared_k100_1024x64 | #1990 | eager-shared | tenferro-rs | f32 | rows=1024, cols=64, chain=100 | 1 | 1 | 17216838.0 | 41828.0 | 0.3% | ok |
| tanh_chain_f32_compiled-prepared_k200_1024x64 | #1990 | compiled-prepared | tenferro-rs | f32 | rows=1024, cols=64, chain=200 | 1 | 1 | 34672369.0 | 86050.0 | 0.3% | ok |
| tanh_chain_f32_eager-shared_k200_1024x64 | #1990 | eager-shared | tenferro-rs | f32 | rows=1024, cols=64, chain=200 | 1 | 1 | 34416443.0 | 40275.0 | 0.1% | ok |
| layer_norm_f32_eager-composed_d1024_len8_b1 | #2006 | eager-composed | tenferro-rs | f32 | norm=layer_norm, d=1024, len=8, batch=1 | 1 | 32 | 66084.3 | 416.7 | 0.6% | ok |
| layer_norm_f32_pytorch-fused_d1024_len8_b1 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=layer_norm, d=1024, len=8, batch=1 | 1 | 256 | 8041.6 | 20.0 | 0.3% | ok |
| layer_norm_f32_eager-single-call_d1024_len8_b1 | #2006 | eager-single-call | tenferro-rs | f32 | norm=layer_norm, d=1024, len=8, batch=1 | 1 | 32 | 68126.2 | 510.9 | 14.1% | NOISY |
| layer_norm_f32_eager-composed_d1024_len8_b8 | #2006 | eager-composed | tenferro-rs | f32 | norm=layer_norm, d=1024, len=8, batch=8 | 1 | 16 | 122185.6 | 1210.3 | 1.0% | ok |
| layer_norm_f32_pytorch-fused_d1024_len8_b8 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=layer_norm, d=1024, len=8, batch=8 | 1 | 64 | 34006.3 | 98.5 | 0.4% | ok |
| layer_norm_f32_eager-single-call_d1024_len8_b8 | #2006 | eager-single-call | tenferro-rs | f32 | norm=layer_norm, d=1024, len=8, batch=8 | 1 | 16 | 122019.1 | 1204.8 | 1.0% | ok |
| layer_norm_f32_eager-composed_d1024_len64_b1 | #2006 | eager-composed | tenferro-rs | f32 | norm=layer_norm, d=1024, len=64, batch=1 | 1 | 16 | 121567.0 | 1286.8 | 0.8% | ok |
| layer_norm_f32_pytorch-fused_d1024_len64_b1 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=layer_norm, d=1024, len=64, batch=1 | 1 | 64 | 34444.8 | 82.1 | 0.3% | ok |
| layer_norm_f32_eager-single-call_d1024_len64_b1 | #2006 | eager-single-call | tenferro-rs | f32 | norm=layer_norm, d=1024, len=64, batch=1 | 1 | 16 | 117444.9 | 2467.1 | 1.3% | ok |
| layer_norm_f32_eager-composed_d1024_len64_b8 | #2006 | eager-composed | tenferro-rs | f32 | norm=layer_norm, d=1024, len=64, batch=8 | 1 | 4 | 602499.2 | 9799.4 | 1.6% | ok |
| layer_norm_f32_pytorch-fused_d1024_len64_b8 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=layer_norm, d=1024, len=64, batch=8 | 1 | 16 | 820397.6 | 7454.5 | 0.8% | ok |
| layer_norm_f32_eager-single-call_d1024_len64_b8 | #2006 | eager-single-call | tenferro-rs | f32 | norm=layer_norm, d=1024, len=64, batch=8 | 1 | 4 | 597101.8 | 5036.9 | 1.0% | ok |
| rms_norm_f32_eager-composed_d1024_len8_b1 | #2006 | eager-composed | tenferro-rs | f32 | norm=rms_norm, d=1024, len=8, batch=1 | 1 | 64 | 34179.8 | 124.8 | 0.3% | ok |
| rms_norm_f32_pytorch-fused_d1024_len8_b1 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=rms_norm, d=1024, len=8, batch=1 | 1 | 128 | 18230.4 | 78.0 | 0.3% | ok |
| rms_norm_f32_eager-single-call_d1024_len8_b1 | #2006 | eager-single-call | tenferro-rs | f32 | norm=rms_norm, d=1024, len=8, batch=1 | 1 | 64 | 40321.3 | 148.5 | 0.4% | ok |
| rms_norm_f32_eager-composed_d1024_len8_b8 | #2006 | eager-composed | tenferro-rs | f32 | norm=rms_norm, d=1024, len=8, batch=8 | 1 | 32 | 59634.2 | 265.6 | 0.5% | ok |
| rms_norm_f32_pytorch-fused_d1024_len8_b8 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=rms_norm, d=1024, len=8, batch=8 | 1 | 32 | 39285.0 | 243.0 | 0.8% | ok |
| rms_norm_f32_eager-single-call_d1024_len8_b8 | #2006 | eager-single-call | tenferro-rs | f32 | norm=rms_norm, d=1024, len=8, batch=8 | 1 | 32 | 66563.0 | 742.0 | 0.7% | ok |
| rms_norm_f32_eager-composed_d1024_len64_b1 | #2006 | eager-composed | tenferro-rs | f32 | norm=rms_norm, d=1024, len=64, batch=1 | 1 | 32 | 60474.2 | 2139.3 | 2.0% | ok |
| rms_norm_f32_pytorch-fused_d1024_len64_b1 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=rms_norm, d=1024, len=64, batch=1 | 1 | 64 | 39549.1 | 431.4 | 1.4% | ok |
| rms_norm_f32_eager-single-call_d1024_len64_b1 | #2006 | eager-single-call | tenferro-rs | f32 | norm=rms_norm, d=1024, len=64, batch=1 | 1 | 32 | 67150.3 | 1178.0 | 0.9% | ok |
| rms_norm_f32_eager-composed_d1024_len64_b8 | #2006 | eager-composed | tenferro-rs | f32 | norm=rms_norm, d=1024, len=64, batch=8 | 1 | 4 | 300008.5 | 11178.2 | 3.1% | ok |
| rms_norm_f32_pytorch-fused_d1024_len64_b8 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=rms_norm, d=1024, len=64, batch=8 | 1 | 8 | 1400723.1 | 5797.6 | 0.4% | ok |
| rms_norm_f32_eager-single-call_d1024_len64_b8 | #2006 | eager-single-call | tenferro-rs | f32 | norm=rms_norm, d=1024, len=64, batch=8 | 1 | 4 | 311079.2 | 2767.6 | 0.9% | ok |
| small_contraction_abcd-dbef-acef_f64_shared-session-default-pool_d4 | #1885 #1897 #1901 | shared-session-default-pool | tenferro-rs | f64 | extent=4 | 1 | 512 | 3905.4 | 21.6 | 0.4% | ok |
| activation_erf_f32_eager-single-call_1024x64 | #1975 #2069 #2032 | eager-single-call | tenferro-rs | f32 | op=erf, rows=1024, cols=64 | 1 | — | — | — | — | FAILED |
| activation_erf_f32_session-single-call_1024x64 | #1975 #2069 #2032 | session-single-call | tenferro-rs | f32 | op=erf, rows=1024, cols=64 | 1 | 8 | 290728.9 | 526.0 | 0.1% | ok |
| activation_erf_f32_pytorch_1024x64 | #1975 #2069 #2032 | pytorch | pytorch-cpu | f32 | op=erf, rows=1024, cols=64 | 1 | — | — | — | — | FAILED |
| activation_sigmoid_f32_eager-single-call_1024x64 | #1975 #2069 #2032 | eager-single-call | tenferro-rs | f32 | op=sigmoid, rows=1024, cols=64 | 1 | — | — | — | — | FAILED |
| activation_sigmoid_f32_session-single-call_1024x64 | #1975 #2069 #2032 | session-single-call | tenferro-rs | f32 | op=sigmoid, rows=1024, cols=64 | 1 | 32 | 43933.4 | 320.4 | 0.7% | ok |
| activation_sigmoid_f32_eager-composed_1024x64 | #1975 #2069 #2032 | eager-composed | tenferro-rs | f32 | op=sigmoid, rows=1024, cols=64 | 1 | 32 | 78674.3 | 261.6 | 0.6% | ok |
| activation_sigmoid_f32_pytorch_1024x64 | #1975 #2069 #2032 | pytorch | pytorch-cpu | f32 | op=sigmoid, rows=1024, cols=64 | 1 | — | — | — | — | FAILED |
| activation_silu_f32_eager-single-call_1024x64 | #1975 #2069 #2032 | eager-single-call | tenferro-rs | f32 | op=silu, rows=1024, cols=64 | 1 | — | — | — | — | FAILED |
| activation_silu_f32_session-single-call_1024x64 | #1975 #2069 #2032 | session-single-call | tenferro-rs | f32 | op=silu, rows=1024, cols=64 | 1 | 32 | 46026.0 | 200.5 | 0.5% | ok |
| activation_silu_f32_eager-composed_1024x64 | #1975 #2069 #2032 | eager-composed | tenferro-rs | f32 | op=silu, rows=1024, cols=64 | 1 | 16 | 98308.0 | 540.4 | 0.3% | ok |
| activation_silu_f32_pytorch_1024x64 | #1975 #2069 #2032 | pytorch | pytorch-cpu | f32 | op=silu, rows=1024, cols=64 | 1 | — | — | — | — | FAILED |
| activation_softplus_f32_eager-single-call_1024x64 | #1975 #2069 #2032 | eager-single-call | tenferro-rs | f32 | op=softplus, rows=1024, cols=64 | 1 | — | — | — | — | FAILED |
| activation_softplus_f32_session-single-call_1024x64 | #1975 #2069 #2032 | session-single-call | tenferro-rs | f32 | op=softplus, rows=1024, cols=64 | 1 | 16 | 160058.6 | 263.3 | 0.3% | ok |
| activation_softplus_f32_eager-composed_1024x64 | #1975 #2069 #2032 | eager-composed | tenferro-rs | f32 | op=softplus, rows=1024, cols=64 | 1 | 16 | 174556.9 | 658.8 | 1.9% | ok |
| activation_softplus_f32_pytorch_1024x64 | #1975 #2069 #2032 | pytorch | pytorch-cpu | f32 | op=softplus, rows=1024, cols=64 | 1 | — | — | — | — | FAILED |
| activation_gelu_f32_eager-single-call_1024x64 | #1975 #2030 #2069 #2228 | eager-single-call | tenferro-rs | f32 | op=gelu, rows=1024, cols=64 | 1 | 32 | 65224.2 | 277.8 | 0.3% | ok |
| activation_gelu_f32_session-single-call_1024x64 | #1975 #2030 #2069 #2228 | session-single-call | tenferro-rs | f32 | op=gelu, rows=1024, cols=64 | 1 | 32 | 64573.0 | 143.8 | 0.3% | ok |
| activation_gelu_f32_pytorch_1024x64 | #1975 #2030 #2069 #2228 | pytorch | pytorch-cpu | f32 | op=gelu, rows=1024, cols=64 | 1 | 32 | 55702.2 | 180.5 | 1.3% | ok |
| activation_gelu_tanh_f32_eager-single-call_1024x64 | #1975 #2069 #2032 | eager-single-call | tenferro-rs | f32 | op=gelu_tanh, rows=1024, cols=64 | 1 | — | — | — | — | FAILED |
| activation_gelu_tanh_f32_session-single-call_1024x64 | #1975 #2069 #2032 | session-single-call | tenferro-rs | f32 | op=gelu_tanh, rows=1024, cols=64 | 1 | 16 | 191309.4 | 430.2 | 0.2% | ok |
| activation_gelu_tanh_f32_eager-composed_1024x64 | #1975 #2069 #2032 | eager-composed | tenferro-rs | f32 | op=gelu_tanh, rows=1024, cols=64 | 1 | 8 | 325710.2 | 1839.6 | 0.5% | ok |
| activation_gelu_tanh_f32_pytorch_1024x64 | #1975 #2069 #2032 | pytorch | pytorch-cpu | f32 | op=gelu_tanh, rows=1024, cols=64 | 1 | — | — | — | — | FAILED |
| softmax_f32_eager-single-call_len64_b8 | #1976 #2031 #2231 | eager-single-call | tenferro-rs | f32 | op=softmax, len=64, batch=8, axis=0 | 1 | 64 | 31373.4 | 336.2 | 1.4% | ok |
| softmax_f32_session-single-call_len64_b8 | #1976 #2031 #2231 | session-single-call | tenferro-rs | f32 | op=softmax, len=64, batch=8, axis=0 | 1 | 64 | 30569.7 | 103.8 | 0.4% | ok |
| softmax_f32_eager-composed_len64_b8 | #1976 #2031 #2231 | eager-composed | tenferro-rs | f32 | op=softmax, len=64, batch=8, axis=0 | 1 | 16 | 139923.7 | 733.2 | 0.4% | ok |
| softmax_f32_pytorch_len64_b8 | #1976 #2031 #2231 | pytorch | pytorch-cpu | f32 | op=softmax, len=64, batch=8, axis=0 | 1 | 64 | 43315.3 | 49.7 | 0.6% | ok |
| softmax_f32_eager-single-call_len512_b8 | #1976 #2231 | eager-single-call | tenferro-rs | f32 | op=softmax, len=512, batch=8, axis=0 | 1 | 2 | 1897894.0 | 17277.2 | 0.9% | ok |
| softmax_f32_session-single-call_len512_b8 | #1976 #2231 | session-single-call | tenferro-rs | f32 | op=softmax, len=512, batch=8, axis=0 | 1 | 2 | 1887149.0 | 17830.8 | 1.6% | ok |
| softmax_f32_eager-composed_len512_b8 | #1976 #2231 | eager-composed | tenferro-rs | f32 | op=softmax, len=512, batch=8, axis=0 | 1 | 1 | 3931271.0 | 11660.5 | 0.8% | ok |
| softmax_f32_pytorch_len512_b8 | #1976 #2231 | pytorch | pytorch-cpu | f32 | op=softmax, len=512, batch=8, axis=0 | 1 | 1 | 2179358.0 | 17858.0 | 0.8% | ok |
| log_softmax_f32_eager-single-call_len64_b8 | #1976 | eager-single-call | tenferro-rs | f32 | op=log_softmax, len=64, batch=8, axis=0 | 1 | 16 | 147283.0 | 11490.2 | 3.8% | ok |
| log_softmax_f32_session-single-call_len64_b8 | #1976 | session-single-call | tenferro-rs | f32 | op=log_softmax, len=64, batch=8, axis=0 | 1 | 16 | 121994.7 | 670.0 | 0.4% | ok |
| log_softmax_f32_eager-composed_len64_b8 | #1976 | eager-composed | tenferro-rs | f32 | op=log_softmax, len=64, batch=8, axis=0 | 1 | 16 | 125601.4 | 948.3 | 0.8% | ok |
| log_softmax_f32_pytorch_len64_b8 | #1976 | pytorch | pytorch-cpu | f32 | op=log_softmax, len=64, batch=8, axis=0 | 1 | 64 | 36999.8 | 661.0 | 1.2% | ok |
| log_softmax_f32_eager-single-call_len512_b8 | #1976 | eager-single-call | tenferro-rs | f32 | op=log_softmax, len=512, batch=8, axis=0 | 1 | 1 | 3654344.0 | 68302.5 | 1.9% | ok |
| log_softmax_f32_session-single-call_len512_b8 | #1976 | session-single-call | tenferro-rs | f32 | op=log_softmax, len=512, batch=8, axis=0 | 1 | 1 | 3709598.0 | 44984.0 | 1.1% | ok |
| log_softmax_f32_eager-composed_len512_b8 | #1976 | eager-composed | tenferro-rs | f32 | op=log_softmax, len=512, batch=8, axis=0 | 1 | 1 | 3882590.0 | 258501.0 | 3.4% | ok |
| log_softmax_f32_pytorch_len512_b8 | #1976 | pytorch | pytorch-cpu | f32 | op=log_softmax, len=512, batch=8, axis=0 | 1 | 1 | 2067801.0 | 27766.0 | 1.0% | ok |
| masked_softmax_f32_eager-single-call_len64_b8 | #1976 | eager-single-call | tenferro-rs | f32 | op=masked_softmax, len=64, batch=8, axis=0 | 1 | 8 | 250713.1 | 856.6 | 1.2% | ok |
| masked_softmax_f32_session-single-call_len64_b8 | #1976 | session-single-call | tenferro-rs | f32 | op=masked_softmax, len=64, batch=8, axis=0 | 1 | 16 | 222308.4 | 2019.4 | 0.8% | ok |
| masked_softmax_f32_eager-composed_len64_b8 | #1976 | eager-composed | tenferro-rs | f32 | op=masked_softmax, len=64, batch=8, axis=0 | 1 | 16 | 154388.7 | 1312.8 | 1.1% | ok |
| masked_softmax_f32_pytorch_len64_b8 | #1976 | pytorch | pytorch-cpu | f32 | op=masked_softmax, len=64, batch=8, axis=0 | 1 | 32 | 90216.1 | 5607.3 | 3.0% | ok |
| masked_softmax_f32_eager-single-call_len512_b8 | #1976 | eager-single-call | tenferro-rs | f32 | op=masked_softmax, len=512, batch=8, axis=0 | 1 | 1 | 9232542.0 | 89963.0 | 0.8% | ok |
| masked_softmax_f32_session-single-call_len512_b8 | #1976 | session-single-call | tenferro-rs | f32 | op=masked_softmax, len=512, batch=8, axis=0 | 1 | 1 | 9321658.0 | 104935.0 | 1.0% | ok |
| masked_softmax_f32_eager-composed_len512_b8 | #1976 | eager-composed | tenferro-rs | f32 | op=masked_softmax, len=512, batch=8, axis=0 | 1 | 1 | 4921304.0 | 44197.5 | 0.7% | ok |
| masked_softmax_f32_pytorch_len512_b8 | #1976 | pytorch | pytorch-cpu | f32 | op=masked_softmax, len=512, batch=8, axis=0 | 1 | 1 | 5033112.0 | 56881.5 | 1.3% | ok |
| softmax_f32_eager-single-call_len64_b8_t1 | #2233 #2235 | eager-single-call | tenferro-rs | f32 | op=softmax, len=64, batch=8, axis=0 | 1 | 64 | 31195.0 | 107.4 | 1.2% | ok |
| softmax_f32_eager-composed_len64_b8_t1 | #2233 #2235 | eager-composed | tenferro-rs | f32 | op=softmax, len=64, batch=8, axis=0 | 1 | 16 | 139850.4 | 695.3 | 0.5% | ok |
| softmax_f32_pytorch_len64_b8_t1 | #2233 #2235 | pytorch | pytorch-cpu | f32 | op=softmax, len=64, batch=8, axis=0 | 1 | 64 | 43611.8 | 70.8 | 0.1% | ok |
| reduce_mean_f32_eager-single-call_1024x64_axis0 | #1976 | eager-single-call | tenferro-rs | f32 | rows=1024, cols=64, axis=0 | 1 | 128 | 15879.3 | 99.0 | 0.4% | ok |
| reduce_mean_f32_session-single-call_1024x64_axis0 | #1976 | session-single-call | tenferro-rs | f32 | rows=1024, cols=64, axis=0 | 1 | 256 | 8542.9 | 14.0 | 0.2% | ok |
| reduce_mean_f32_eager-composed_1024x64_axis0 | #1976 | eager-composed | tenferro-rs | f32 | rows=1024, cols=64, axis=0 | 1 | 128 | 16254.7 | 68.1 | 0.6% | ok |
| reduce_mean_f32_pytorch_1024x64_axis0 | #1976 | pytorch | pytorch-cpu | f32 | rows=1024, cols=64, axis=0 | 1 | 256 | 8771.1 | 53.9 | 0.5% | ok |
| take_along_axis_rows_f64_eager-single-call_n64_b256 | #2008 | eager-single-call | tenferro-rs | f64 | n=64, batch=256, axis=0 | 1 | 1 | 9123890.0 | 37770.5 | 0.3% | ok |
| take_along_axis_rows_f64_session-single-call_n64_b256 | #2008 | session-single-call | tenferro-rs | f64 | n=64, batch=256, axis=0 | 1 | 1 | 8998126.0 | 66148.0 | 0.6% | ok |
| take_along_axis_rows_f64_eager-gather-prebuilt_n64_b256 | #2008 | eager-gather-prebuilt | tenferro-rs | f64 | n=64, batch=256, axis=0 | 1 | 1 | 8846794.0 | 74072.5 | 1.2% | ok |
| take_along_axis_rows_f64_pytorch_n64_b256 | #2008 | pytorch | pytorch-cpu | f64 | n=64, batch=256, axis=0 | 1 | 1 | 4594676.0 | 25021.5 | 0.5% | ok |
| lstsq_f64_eager-shared_m768_n384_rhs16 | #2027 | eager-shared | tenferro-rs | f64 | operation=lstsq | 1 | 1 | 22307369.0 | 53905.5 | 1.1% | ok |
| lstsq_f64_pytorch_m768_n384_rhs16 | #2027 | pytorch | pytorch-cpu | f64 | operation=lstsq | 1 | 1 | 12323679.0 | 19737.0 | 0.1% | ok |
| triangular_f64_concrete-shared_n4096_rhs64 | #2028 | concrete-shared | tenferro-rs | f64 | operation=triangular | 1 | 1 | 40672618.0 | 69424.0 | 0.2% | ok |
| triangular_f64_pytorch_n4096_rhs64 | #2028 | pytorch | pytorch-cpu | f64 | operation=triangular | 1 | 1 | 44399318.0 | 94962.5 | 0.8% | ok |
| reshape_f64_concrete-shared_n33554432_to8192x4096 | #2029 | concrete-shared | tenferro-rs | f64 | operation=reshape | 1 | 1 | 87493375.0 | 1174788.0 | 0.8% | ok |
| reshape_f64_pytorch_n33554432_to8192x4096 | #2029 | pytorch | pytorch-cpu | f64 | operation=reshape | 1 | 1 | 85532875.0 | 1907016.0 | 1.2% | ok |
| metadata_transpose_view_f64_rank8_t1 | #2180 | metadata-clone | tenferro-rs | f64 | shape=[2, 2, 2, 2, 2, 2, 2, 2] | 1 | 8192 | 80.3 | 0.4 | 45.1% | NOISY |
| metadata_transpose_view_f64_rank8_t1_julia-base | #2180 | julia-base | julia-base | f64 | shape=[2, 2, 2, 2, 2, 2, 2, 2] | 1 | 1048576 | 0.3 | 0.0 | 6.3% | ok |
| followup_structural_broadcast_in_dim_direct-shared | #2021 | direct-shared | tenferro-rs | f64 | case_id=structural.broadcast_in_dim, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [8192, 1], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | FAILED |
| followup_structural_broadcast_in_dim_pytorch | #2021 | pytorch | pytorch-cpu | f64 | case_id=structural.broadcast_in_dim, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [8192, 1], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | FAILED |
| followup_real_tanh_direct-shared | #2033 | direct-shared | tenferro-rs | f64 | case_id=real.tanh, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [8388608], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | FAILED |
| followup_real_tanh_jax | #2033 | jax | jax-cpu | f64 | case_id=real.tanh, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [8388608], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | FAILED |
| followup_real_log1p_direct-shared | #2033 | direct-shared | tenferro-rs | f64 | case_id=real.log1p, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [4194304], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | FAILED |
| followup_real_log1p_pytorch | #2033 | pytorch | pytorch-cpu | f64 | case_id=real.log1p, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [4194304], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | FAILED |
| followup_real_cos_direct-shared | #2033 | direct-shared | tenferro-rs | f64 | case_id=real.cos, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [8388608], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | FAILED |
| followup_real_cos_pytorch | #2033 | pytorch | pytorch-cpu | f64 | case_id=real.cos, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [8388608], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | FAILED |
| followup_real_sin_direct-shared | #2033 | direct-shared | tenferro-rs | f64 | case_id=real.sin, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [8388608], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | FAILED |
| followup_real_sin_pytorch | #2033 | pytorch | pytorch-cpu | f64 | case_id=real.sin, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [8388608], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | FAILED |
| followup_real_exp_direct-shared | #2033 | direct-shared | tenferro-rs | f64 | case_id=real.exp, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [8388608], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | FAILED |
| followup_real_exp_jax | #2033 | jax | jax-cpu | f64 | case_id=real.exp, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [8388608], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | FAILED |
| followup_real_expm1_direct-shared | #2033 | direct-shared | tenferro-rs | f64 | case_id=real.expm1, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [4194304], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | FAILED |
| followup_real_expm1_jax | #2033 | jax | jax-cpu | f64 | case_id=real.expm1, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [4194304], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | FAILED |
| followup_real_pow_direct-shared | #2033 | direct-shared | tenferro-rs | f64 | case_id=real.pow, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [4194304], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | FAILED |
| followup_real_pow_pytorch | #2033 | pytorch | pytorch-cpu | f64 | case_id=real.pow, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [4194304], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | FAILED |
| followup_complex_exp_direct-shared | #2033 | direct-shared | tenferro-rs | c64 | case_id=complex.exp, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [4194304], 'dtype': 'c64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | FAILED |
| followup_complex_exp_julia-base | #2033 | julia-base | julia-base | c64 | case_id=complex.exp, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [4194304], 'dtype': 'c64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | FAILED |
| followup_real_log_direct-shared | #2033 | direct-shared | tenferro-rs | f64 | case_id=real.log, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [8388608], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | FAILED |
| followup_real_log_pytorch | #2033 | pytorch | pytorch-cpu | f64 | case_id=real.log, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [8388608], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | FAILED |
| followup_real_reduce_max_axis1_direct-shared | #2034 | direct-shared | tenferro-rs | f64 | case_id=real.reduce_max_axis1, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [2048, 2048], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | FAILED |
| followup_real_reduce_max_axis1_julia-base | #2034 | julia-base | julia-base | f64 | case_id=real.reduce_max_axis1, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [2048, 2048], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | FAILED |
| followup_real_reduce_prod_axis1_direct-shared | #2034 | direct-shared | tenferro-rs | f64 | case_id=real.reduce_prod_axis1, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [2048, 2048], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | FAILED |
| followup_real_reduce_prod_axis1_pytorch | #2034 | pytorch | pytorch-cpu | f64 | case_id=real.reduce_prod_axis1, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [2048, 2048], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | FAILED |
| followup_real_reduce_sum_axis1_direct-shared | #2034 | direct-shared | tenferro-rs | f64 | case_id=real.reduce_sum_axis1, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [2048, 2048], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | FAILED |
| followup_real_reduce_sum_axis1_pytorch | #2034 | pytorch | pytorch-cpu | f64 | case_id=real.reduce_sum_axis1, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [2048, 2048], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | FAILED |
| followup_real_reduce_max_axis0_direct-shared | #2034 | direct-shared | tenferro-rs | f64 | case_id=real.reduce_max_axis0, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [2048, 2048], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | FAILED |
| followup_real_reduce_max_axis0_julia-base | #2034 | julia-base | julia-base | f64 | case_id=real.reduce_max_axis0, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [2048, 2048], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | FAILED |
| followup_real_reduce_min_axis1_direct-shared | #2034 | direct-shared | tenferro-rs | f64 | case_id=real.reduce_min_axis1, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [4096, 4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | FAILED |
| followup_real_reduce_min_axis1_julia-base | #2034 | julia-base | julia-base | f64 | case_id=real.reduce_min_axis1, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [4096, 4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | FAILED |
| followup_real_reduce_min_axis0_direct-shared | #2034 | direct-shared | tenferro-rs | f64 | case_id=real.reduce_min_axis0, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [2048, 2048], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | FAILED |
| followup_real_reduce_min_axis0_julia-base | #2034 | julia-base | julia-base | f64 | case_id=real.reduce_min_axis0, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [2048, 2048], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | FAILED |
| followup_real_reduce_max_all_direct-shared | #2034 | direct-shared | tenferro-rs | f64 | case_id=real.reduce_max_all, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [8192, 4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | FAILED |
| followup_real_reduce_max_all_julia-base | #2034 | julia-base | julia-base | f64 | case_id=real.reduce_max_all, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [8192, 4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | FAILED |
| followup_real_reduce_min_all_direct-shared | #2034 | direct-shared | tenferro-rs | f64 | case_id=real.reduce_min_all, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [8192, 4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | FAILED |
| followup_real_reduce_min_all_julia-base | #2034 | julia-base | julia-base | f64 | case_id=real.reduce_min_all, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [8192, 4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | FAILED |
| followup_complex_norm_fro_direct-shared | #2035 | direct-shared | tenferro-rs | c64 | case_id=complex.norm_fro, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [2048, 1536], 'dtype': 'c64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | FAILED |
| followup_complex_norm_fro_jax | #2035 | jax | jax-cpu | c64 | case_id=complex.norm_fro, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [2048, 1536], 'dtype': 'c64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | FAILED |
| followup_index_gather_direct-shared | #2036 #1719 | direct-shared | tenferro-rs | f64 | case_id=index.gather, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [262144], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | FAILED |
| followup_index_gather_jax | #2036 #1719 | jax | jax-cpu | f64 | case_id=index.gather, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [262144], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | FAILED |
| followup_index_scatter_direct-shared | #2036 #1719 | direct-shared | tenferro-rs | f64 | case_id=index.scatter, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [262144], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | FAILED |
| followup_index_scatter_pytorch | #2036 #1719 | pytorch | pytorch-cpu | f64 | case_id=index.scatter, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [262144], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | FAILED |
| followup_index_dynamic_update_slice_direct-shared | #2036 #1719 | direct-shared | tenferro-rs | f64 | case_id=index.dynamic_update_slice, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [2097152], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | FAILED |
| followup_index_dynamic_update_slice_jax | #2036 #1719 | jax | jax-cpu | f64 | case_id=index.dynamic_update_slice, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [2097152], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | FAILED |
| followup_perm_reverse_15d_3_materialize | #2037 #2228 | materialize | tenferro-rs | f64 | case_id=perm.reverse_15d_3, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | FAILED |
| followup_perm_reverse_15d_3_strided-rs | #2037 #2228 | strided-rs | strided-rs | f64 | case_id=perm.reverse_15d_3, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | FAILED |
| followup_structural_transpose_direct-shared | #2037 #2228 | direct-shared | tenferro-rs | f64 | case_id=structural.transpose, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [4096, 4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | FAILED |
| followup_structural_transpose_pytorch | #2037 #2228 | pytorch | pytorch-cpu | f64 | case_id=structural.transpose, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [4096, 4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | FAILED |
| followup_perm_reverse_23d_2_materialize | #2037 #2228 | materialize | tenferro-rs | f64 | case_id=perm.reverse_23d_2, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | FAILED |
| followup_perm_reverse_23d_2_strided-rs | #2037 #2228 | strided-rs | strided-rs | f64 | case_id=perm.reverse_23d_2, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | FAILED |
| followup_perm_transpose_3d_256_201_materialize | #2037 #2228 | materialize | tenferro-rs | f64 | case_id=perm.transpose_3d_256_201, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [256, 256, 256], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | FAILED |
| followup_perm_transpose_3d_256_201_strided-rs | #2037 #2228 | strided-rs | strided-rs | f64 | case_id=perm.transpose_3d_256_201, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [256, 256, 256], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | FAILED |
| followup_perm_cyclic_15d_3_materialize | #2037 #2228 | materialize | tenferro-rs | f64 | case_id=perm.cyclic_15d_3, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | FAILED |
| followup_perm_cyclic_15d_3_strided-rs | #2037 #2228 | strided-rs | strided-rs | f64 | case_id=perm.cyclic_15d_3, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | FAILED |
| followup_perm_transpose_3d_256_102_materialize | #2037 #2228 | materialize | tenferro-rs | f64 | case_id=perm.transpose_3d_256_102, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [256, 256, 256], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | FAILED |
| followup_perm_transpose_3d_256_102_strided-rs | #2037 #2228 | strided-rs | strided-rs | f64 | case_id=perm.transpose_3d_256_102, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [256, 256, 256], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | FAILED |
| followup_perm_tn_light_415_24d_scattered_to_colmajor_materialize | #2037 #2228 | materialize | tenferro-rs | f64 | case_id=perm.tn_light_415_24d_scattered_to_colmajor, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | FAILED |
| followup_perm_tn_light_415_24d_scattered_to_colmajor_strided-rs | #2037 #2228 | strided-rs | strided-rs | f64 | case_id=perm.tn_light_415_24d_scattered_to_colmajor, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | FAILED |
| followup_perm_tn_light_415_24d_contiguous_same_perm_materialize | #2037 #2228 | materialize | tenferro-rs | f64 | case_id=perm.tn_light_415_24d_contiguous_same_perm, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | FAILED |
| followup_perm_tn_light_415_24d_contiguous_same_perm_strided-rs | #2037 #2228 | strided-rs | strided-rs | f64 | case_id=perm.tn_light_415_24d_contiguous_same_perm, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | FAILED |
| followup_perm_rotation_6d_32_32_32_32_16_16_materialize | #2037 #2228 | materialize | tenferro-rs | f64 | case_id=perm.rotation_6d_32_32_32_32_16_16, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [32, 32, 32, 32, 16, 16], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | FAILED |
| followup_perm_rotation_6d_32_32_32_32_16_16_strided-rs | #2037 #2228 | strided-rs | strided-rs | f64 | case_id=perm.rotation_6d_32_32_32_32_16_16, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [32, 32, 32, 32, 16, 16], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | FAILED |
| followup_structural_extract_diagonal_direct-shared | #2038 #2228 | direct-shared | tenferro-rs | f64 | case_id=structural.extract_diagonal, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [8388608, 2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | FAILED |
| followup_structural_extract_diagonal_jax | #2038 #2228 | jax | jax-cpu | f64 | case_id=structural.extract_diagonal, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [8388608, 2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | FAILED |
| followup_fft_irfft_direct-shared | #2039 #2230 #2228 | direct-shared | tenferro-rs | c32 | case_id=fft.irfft, rust_path=direct-shared, reference_path=mkl-dfti, fixture={'input_shape': [524289], 'dtype': 'c32', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | FAILED |
| followup_fft_irfft_mkl-dfti | #2039 #2230 #2228 | mkl-dfti | mkl-dfti | c32 | case_id=fft.irfft, rust_path=direct-shared, reference_path=mkl-dfti, fixture={'input_shape': [524289], 'dtype': 'c32', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | FAILED |
| followup_fft_rfft_direct-shared | #2039 #2230 #2228 | direct-shared | tenferro-rs | f32 | case_id=fft.rfft, rust_path=direct-shared, reference_path=mkl-dfti, fixture={'input_shape': [1048576], 'dtype': 'f32', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | FAILED |
| followup_fft_rfft_mkl-dfti | #2039 #2230 #2228 | mkl-dfti | mkl-dfti | f32 | case_id=fft.rfft, rust_path=direct-shared, reference_path=mkl-dfti, fixture={'input_shape': [1048576], 'dtype': 'f32', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | FAILED |
| followup_fft_fft_direct-shared | #2039 #2228 | direct-shared | tenferro-rs | c64 | case_id=fft.fft, rust_path=direct-shared, reference_path=mkl-dfti, fixture={'input_shape': [1048576], 'dtype': 'c64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | FAILED |
| followup_fft_fft_mkl-dfti | #2039 #2228 | mkl-dfti | mkl-dfti | c64 | case_id=fft.fft, rust_path=direct-shared, reference_path=mkl-dfti, fixture={'input_shape': [1048576], 'dtype': 'c64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | FAILED |
| followup_fft_ifft_direct-shared | #2039 #2228 | direct-shared | tenferro-rs | c32 | case_id=fft.ifft, rust_path=direct-shared, reference_path=mkl-dfti, fixture={'input_shape': [1048576], 'dtype': 'c32', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | FAILED |
| followup_fft_ifft_mkl-dfti | #2039 #2228 | mkl-dfti | mkl-dfti | c32 | case_id=fft.ifft, rust_path=direct-shared, reference_path=mkl-dfti, fixture={'input_shape': [1048576], 'dtype': 'c32', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | — | — | — | — | FAILED |
| followup_metadata_transpose_view_metadata | #2040 #1719 | metadata | tenferro-rs | f64 | case_id=metadata.transpose_view, rust_path=metadata, reference_path=julia-base, fixture={'input_shape': [2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 1 | — | — | — | — | FAILED |
| followup_metadata_transpose_view_julia-base | #2040 #1719 | julia-base | julia-base | f64 | case_id=metadata.transpose_view, rust_path=metadata, reference_path=julia-base, fixture={'input_shape': [2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 1 | — | — | — | — | FAILED |
| followup_metadata_transpose_view_metadata-host-dyn | #2040 #1719 | metadata-host-dyn | tenferro-rs | f64 | case_id=metadata.transpose_view, rust_path=metadata-host-dyn, reference_path=julia-base, fixture={'input_shape': [2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 1 | — | — | — | — | FAILED |
| followup_metadata_transpose_view_metadata-host-static | #2040 #1719 | metadata-host-static | tenferro-rs | f64 | case_id=metadata.transpose_view, rust_path=metadata-host-static, reference_path=julia-base, fixture={'input_shape': [2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 1 | — | — | — | — | FAILED |
| followup_metadata_slice_view_metadata | #2040 #1719 | metadata | tenferro-rs | f64 | case_id=metadata.slice_view, rust_path=metadata, reference_path=julia-base, fixture={'input_shape': [4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 1 | — | — | — | — | FAILED |
| followup_metadata_slice_view_julia-base | #2040 #1719 | julia-base | julia-base | f64 | case_id=metadata.slice_view, rust_path=metadata, reference_path=julia-base, fixture={'input_shape': [4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 1 | — | — | — | — | FAILED |
| followup_metadata_slice_view_metadata-host-dyn | #2040 #1719 | metadata-host-dyn | tenferro-rs | f64 | case_id=metadata.slice_view, rust_path=metadata-host-dyn, reference_path=julia-base, fixture={'input_shape': [4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 1 | — | — | — | — | FAILED |
| followup_metadata_slice_view_metadata-host-static | #2040 #1719 | metadata-host-static | tenferro-rs | f64 | case_id=metadata.slice_view, rust_path=metadata-host-static, reference_path=julia-base, fixture={'input_shape': [4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 1 | — | — | — | — | FAILED |
| followup_metadata_reshape_view_metadata | #2040 #1719 | metadata | tenferro-rs | f64 | case_id=metadata.reshape_view, rust_path=metadata, reference_path=julia-base, fixture={'input_shape': [1024], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 1 | — | — | — | — | FAILED |
| followup_metadata_reshape_view_julia-base | #2040 #1719 | julia-base | julia-base | f64 | case_id=metadata.reshape_view, rust_path=metadata, reference_path=julia-base, fixture={'input_shape': [1024], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 1 | — | — | — | — | FAILED |
| followup_metadata_reshape_view_metadata-host-dyn | #2040 #1719 | metadata-host-dyn | tenferro-rs | f64 | case_id=metadata.reshape_view, rust_path=metadata-host-dyn, reference_path=julia-base, fixture={'input_shape': [1024], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 1 | — | — | — | — | FAILED |
| followup_metadata_reshape_view_metadata-host-static | #2040 #1719 | metadata-host-static | tenferro-rs | f64 | case_id=metadata.reshape_view, rust_path=metadata-host-static, reference_path=julia-base, fixture={'input_shape': [1024], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 1 | — | — | — | — | FAILED |
| typed_elemwise_f64_add_held_len16 | #2139 | tenferro | tenferro-rs | f64 | n=16, route=held | 1 | 2048 | 1427.1 | 6.1 | 1.4% | ok |
| typed_elemwise_f64_add_held_len16_pytorch | #2139 | pytorch | pytorch-cpu | f64 | n=16, route=held | 1 | 2048 | 1838.2 | 15.3 | 244.7% | NOISY |
| typed_elemwise_f64_add_into_held_len16 | #2139 | tenferro | tenferro-rs | f64 | n=16, route=into_held | 1 | 2048 | 1023.1 | 4.3 | 0.9% | ok |
| typed_elemwise_f64_add_into_held_len16_pytorch | #2139 | pytorch | pytorch-cpu | f64 | n=16, route=into_held | 1 | 2048 | 1772.0 | 2.6 | 0.1% | ok |
| typed_elemwise_f64_add_held_len256 | #2139 | tenferro | tenferro-rs | f64 | n=256, route=held | 1 | 1024 | 1610.9 | 14.1 | 1.9% | ok |
| typed_elemwise_f64_add_held_len256_pytorch | #2139 | pytorch | pytorch-cpu | f64 | n=256, route=held | 1 | 1024 | 1932.7 | 9.4 | 0.7% | ok |
| typed_elemwise_f64_add_into_held_len256 | #2139 | tenferro | tenferro-rs | f64 | n=256, route=into_held | 1 | 2048 | 1123.1 | 2.1 | 0.2% | ok |
| typed_elemwise_f64_add_into_held_len256_pytorch | #2139 | pytorch | pytorch-cpu | f64 | n=256, route=into_held | 1 | 2048 | 1835.2 | 9.0 | 0.4% | ok |
| typed_elemwise_f64_add_held_len4096 | #2139 | tenferro | tenferro-rs | f64 | n=4096, route=held | 1 | 256 | 2995.0 | 26.3 | 1.3% | ok |
| typed_elemwise_f64_add_held_len4096_pytorch | #2139 | pytorch | pytorch-cpu | f64 | n=4096, route=held | 1 | 1024 | 3705.0 | 39.0 | 1.1% | ok |
| typed_elemwise_f64_add_into_held_len4096 | #2139 | tenferro | tenferro-rs | f64 | n=4096, route=into_held | 1 | 1024 | 2162.5 | 14.0 | 0.5% | ok |
| typed_elemwise_f64_add_into_held_len4096_pytorch | #2139 | pytorch | pytorch-cpu | f64 | n=4096, route=into_held | 1 | 1024 | 2752.6 | 6.6 | 0.2% | ok |
| tiny_matmul_2x2_typed_held | #2139 | tenferro | tenferro-rs | f64 | n=2 | 1 | 1024 | 2143.6 | 17.5 | 1.0% | ok |
| tiny_matmul_2x2_typed_held_pytorch | #2139 | pytorch | pytorch-cpu | f64 | n=2 | 1 | 1024 | 1956.1 | 5.5 | 0.3% | ok |
| analytic_neg_f64_sweep_8MiB_t1 | #2139 | tenferro | tenferro-rs | f64 | n=1048576 | 1 | 2 | 342268.0 | 13512.5 | 8.5% | ok |
| analytic_neg_f64_sweep_8MiB_t1_pytorch | #2139 | pytorch | pytorch-cpu | f64 | n=1048576 | 1 | 8 | 1686803.5 | 28229.5 | 1.6% | ok |
| analytic_neg_f64_sweep_32MiB_t1 | #2139 | tenferro | tenferro-rs | f64 | n=4194304 | 1 | 2 | 1870489.5 | 25800.8 | 1.6% | ok |
| analytic_neg_f64_sweep_32MiB_t1_pytorch | #2139 | pytorch | pytorch-cpu | f64 | n=4194304 | 1 | 1 | 10431081.0 | 46431.5 | 0.4% | ok |
| gemm_bound_f64_t1_t8_scaling | #2141 | tenferro | tenferro-rs | f64 | n=1536 | 1 | 1 | 234960860.0 | 420614.0 | 0.3% | ok |
| gemm_bound_f64_t1_t8_scaling_pytorch | #2141 | pytorch | pytorch-cpu | f64 | n=1536 | 1 | 1 | 238752147.0 | 396222.5 | 0.2% | ok |
| lu_batched_f64_factor_n8_b1024 | #2142 | tenferro | tenferro-rs | f64 | n=8, batch=1024, op=factor | 1 | 4 | 613663.0 | 3236.0 | 0.8% | ok |
| lu_batched_f64_factor_n8_b1024_host | #2142 | host | rust-loop | f64 | n=8, batch=1024, op=factor | 1 | 4 | 528715.0 | 3539.1 | 0.9% | ok |
| lu_batched_f64_solve_n8_b1024 | #2142 | tenferro | tenferro-rs | f64 | n=8, batch=1024, op=solve | 1 | 8 | 248990.1 | 2065.0 | 1.2% | ok |
| lu_batched_f64_solve_n8_b1024_pytorch | #2142 | pytorch | pytorch-cpu | f64 | n=8, batch=1024, op=solve | 1 | 16 | 229357.9 | 604.6 | 0.2% | ok |
| lu_batched_f64_factor_n16_b1024 | #2142 | tenferro | tenferro-rs | f64 | n=16, batch=1024, op=factor | 1 | 2 | 1613530.5 | 4541.0 | 0.4% | ok |
| lu_batched_f64_factor_n16_b1024_host | #2142 | host | rust-loop | f64 | n=16, batch=1024, op=factor | 1 | 2 | 2106950.0 | 2995.5 | 17.7% | NOISY |
| lu_batched_f64_solve_n16_b1024 | #2142 | tenferro | tenferro-rs | f64 | n=16, batch=1024, op=solve | 1 | 2 | 1008869.5 | 5352.5 | 11.2% | NOISY |
| lu_batched_f64_solve_n16_b1024_pytorch | #2142 | pytorch | pytorch-cpu | f64 | n=16, batch=1024, op=solve | 1 | 8 | 500134.2 | 1544.2 | 8.1% | ok |
| axpby_accum_strided_dest_f64_block | #2143 | tenferro | tenferro-rs | f64 | n=64 | 1 | 256 | 8076.4 | 17.6 | 0.3% | ok |
| axpby_accum_strided_dest_f64_block_host | #2143 | host | rust-loop | f64 | n=64 | 1 | 512 | 5767.6 | 144.0 | 1.8% | ok |
| small_grad_qr_f64_trace-prepared_n4 | #2186 | trace-prepared | tenferro-rs | f64 | op=qr, n=4 | 1 | 64 | 46308.1 | 445.5 | 1.1% | ok |
| small_grad_qr_f64_pytorch_n4 | #2186 | pytorch | pytorch-cpu | f64 | op=qr, n=4 | 1 | 32 | 73595.8 | 1830.9 | 2.0% | ok |
| small_grad_qr_f64_trace-prepared_n16 | #2186 | trace-prepared | tenferro-rs | f64 | op=qr, n=16 | 1 | 32 | 66509.2 | 412.5 | 0.4% | ok |
| small_grad_qr_f64_pytorch_n16 | #2186 | pytorch | pytorch-cpu | f64 | op=qr, n=16 | 1 | 32 | 84421.0 | 493.0 | 0.6% | ok |
| small_grad_eigh_f64_trace-prepared_n4 | #2186 | trace-prepared | tenferro-rs | f64 | op=eigh, n=4 | 1 | 32 | 65025.5 | 225.8 | 0.5% | ok |
| small_grad_eigh_f64_pytorch_n4 | #2186 | pytorch | pytorch-cpu | f64 | op=eigh, n=4 | 1 | 32 | 73366.7 | 423.1 | 0.5% | ok |
| small_grad_eigh_f64_trace-prepared_n16 | #2186 | trace-prepared | tenferro-rs | f64 | op=eigh, n=16 | 1 | 32 | 120044.2 | 359.6 | 2.5% | ok |
| small_grad_eigh_f64_pytorch_n16 | #2186 | pytorch | pytorch-cpu | f64 | op=eigh, n=16 | 1 | 32 | 100041.1 | 383.2 | 0.4% | ok |
| small_grad_lu_f64_trace-prepared_n4 | #2186 | trace-prepared | tenferro-rs | f64 | op=lu, n=4 | 1 | 64 | 34389.2 | 172.6 | 0.5% | ok |
| small_grad_lu_f64_pytorch_n4 | #2186 | pytorch | pytorch-cpu | f64 | op=lu, n=4 | 1 | 32 | 80227.0 | 976.2 | 0.7% | ok |
| small_grad_lu_f64_trace-prepared_n16 | #2186 | trace-prepared | tenferro-rs | f64 | op=lu, n=16 | 1 | 64 | 42651.8 | 229.7 | 0.6% | ok |
| small_grad_lu_f64_pytorch_n16 | #2186 | pytorch | pytorch-cpu | f64 | op=lu, n=16 | 1 | 32 | 89534.0 | 413.9 | 0.6% | ok |
| small_grad_svd_f64_trace-prepared_n4 | #2186 | trace-prepared | tenferro-rs | f64 | op=svd, n=4 | 1 | 32 | 105664.3 | 322.8 | 0.2% | ok |
| small_grad_svd_f64_pytorch_n4 | #2186 | pytorch | pytorch-cpu | f64 | op=svd, n=4 | 1 | 32 | 111992.7 | 931.9 | 0.6% | ok |
| small_grad_svd_f64_trace-prepared_n16 | #2186 | trace-prepared | tenferro-rs | f64 | op=svd, n=16 | 1 | 16 | 205496.6 | 1256.1 | 0.4% | ok |
| small_grad_svd_f64_pytorch_n16 | #2186 | pytorch | pytorch-cpu | f64 | op=svd, n=16 | 1 | 16 | 173406.2 | 1196.9 | 0.7% | ok |
| small_grad_solve_f64_trace-prepared_n4 | #2186 | trace-prepared | tenferro-rs | f64 | op=solve, n=4 | 1 | 128 | 28567.4 | 34.5 | 0.1% | ok |
| small_grad_solve_f64_pytorch_n4 | #2186 | pytorch | pytorch-cpu | f64 | op=solve, n=4 | 1 | 32 | 61341.2 | 258.8 | 0.3% | ok |
| small_grad_solve_f64_trace-prepared_n16 | #2186 | trace-prepared | tenferro-rs | f64 | op=solve, n=16 | 1 | 64 | 36213.7 | 153.8 | 0.4% | ok |
| small_grad_solve_f64_pytorch_n16 | #2186 | pytorch | pytorch-cpu | f64 | op=solve, n=16 | 1 | 32 | 71025.2 | 339.2 | 0.8% | ok |
| fft_axis0_c64_fft-executor_n128 | #1765 | fft-executor | tenferro-rs | c64 | n=128 | 1 | 1 | 8740804.0 | 67841.5 | 0.5% | ok |
| fft_axis0_c64_pytorch_n128 | #1765 | pytorch | pytorch-cpu | c64 | n=128 | 1 | 1 | 23449204.0 | 162147.0 | 0.5% | ok |
| degenerate_dot_hadamard_f64_dot-general-shared_n1024 | #2111 | dot-general-shared | tenferro-rs | f64 | form=hadamard, n=1024 | 1 | 2 | 510721.0 | 7564.0 | 2.4% | ok |
| degenerate_dot_hadamard_f64_pytorch_n1024 | #2111 | pytorch | pytorch-cpu | f64 | form=hadamard, n=1024 | 1 | 2 | 513236.0 | 38281.5 | 9.4% | ok |
| degenerate_dot_outer_f64_dot-general-shared_n1024 | #2111 | dot-general-shared | tenferro-rs | f64 | form=outer, n=1024 | 1 | 2 | 362505.0 | 8427.8 | 3.6% | ok |
| degenerate_dot_outer_f64_pytorch_n1024 | #2111 | pytorch | pytorch-cpu | f64 | form=outer, n=1024 | 1 | 8 | 1597255.5 | 26417.3 | 1.2% | ok |
| degenerate_dot_rank0_f64_dot-general-shared_n1024 | #2111 | dot-general-shared | tenferro-rs | f64 | form=rank0, n=1024 | 1 | 2 | 769042.5 | 9880.8 | 1.4% | ok |
| degenerate_dot_rank0_f64_pytorch_n1024 | #2111 | pytorch | pytorch-cpu | f64 | form=rank0, n=1024 | 1 | 4 | 349235.5 | 14174.0 | 7.1% | ok |
| erased_add_f64_session-add_n256 | #1766 | session-add | tenferro-rs | f64 | n=256 | 1 | 2048 | 1185.5 | 23.0 | 5.0% | ok |
| erased_add_f64_pytorch_n256 | #1766 | pytorch | pytorch-cpu | f64 | n=256 | 1 | 1024 | 1922.8 | 14.8 | 1.4% | ok |
| erased_add_f64_session-add_n4096 | #1766 | session-add | tenferro-rs | f64 | n=4096 | 1 | 512 | 2839.5 | 81.7 | 3.5% | ok |
| erased_add_f64_pytorch_n4096 | #1766 | pytorch | pytorch-cpu | f64 | n=4096 | 1 | 256 | 3323.3 | 40.1 | 1.4% | ok |
| erased_add_f64_session-add_n1048576 | #1766 | session-add | tenferro-rs | f64 | n=1048576 | 1 | 2 | 511858.5 | 20418.0 | 4.0% | ok |
| erased_add_f64_pytorch_n1048576 | #1766 | pytorch | pytorch-cpu | f64 | n=1048576 | 1 | 2 | 517298.5 | 19278.5 | 8.0% | ok |
| axpby_strided_source_f64_1024x1024_t1 | #2220 #2214 | strided-source | tenferro-rs | f64 | n=1024 | 1 | 1 | 2347964.0 | 454787.5 | 12.1% | NOISY |
| axpby_contiguous_source_f64_1024x1024_t1 | #2220 #2214 | contiguous-source | tenferro-rs | f64 | n=1024 | 1 | 4 | 584321.0 | 11958.6 | 1.6% | ok |
| axpby_strided_source_f64_1024x1024_host_t1 | #2220 #2214 | host | rust-loop | f64 | n=1024 | 1 | 1 | 2261673.0 | 43230.0 | 1.3% | ok |
| erased_construct_f64_col_n8 | #2177 | col-major | tenferro-rs | f64 | n=8, layout=col | 1 | 16384 | 129.6 | 1.0 | 13.6% | NOISY |
| erased_construct_f64_row_n8 | #2177 | row-major | tenferro-rs | f64 | n=8, layout=row | 1 | 8192 | 249.9 | 3.4 | 7.5% | ok |
| erased_construct_f64_col_n4096 | #2177 | col-major | tenferro-rs | f64 | n=4096, layout=col | 1 | 16384 | 214.8 | 4.6 | 1.4% | ok |
| erased_construct_f64_row_n4096 | #2177 | row-major | tenferro-rs | f64 | n=4096, layout=row | 1 | 128 | 18023.3 | 1614.2 | 5.7% | ok |
| erased_construct_f64_col_n65536 | #2177 | col-major | tenferro-rs | f64 | n=65536, layout=col | 1 | 1024 | 132.3 | 2.6 | 1.8% | ok |
| erased_construct_f64_row_n65536 | #2177 | row-major | tenferro-rs | f64 | n=65536, layout=row | 1 | 8 | 309830.9 | 18624.0 | 4.8% | ok |
| eager_small_leaf_f64_eager-held-session | #1704 | eager-held-session | tenferro-rs | f64 | op=leaf | 1 | 1024 | 3443.9 | 59.4 | 1.8% | ok |
| eager_small_add_f64_eager-held-session | #1667 | eager-held-session | tenferro-rs | f64 | op=add | 1 | 1024 | 2552.3 | 4.3 | 0.2% | ok |
| eager_small_add_f64_pytorch | #1667 | pytorch | pytorch-cpu | f64 | op=add | 1 | 2048 | 1812.1 | 4.1 | 0.7% | ok |
| eager_small_gemm_f64_eager-held-session | #1904 | eager-held-session | tenferro-rs | f64 | op=gemm | 1 | 512 | 3955.4 | 15.2 | 0.5% | ok |
| eager_small_gemm_f64_pytorch | #1904 | pytorch | pytorch-cpu | f64 | op=gemm | 1 | 1024 | 2035.9 | 2.6 | 0.4% | ok |
| trace_metadata_trace-fold_f32_1024x64_k50 | #2226 #2238 | trace-fold | tenferro-rs | f32 | rows=1024, cols=64, steps=50 | 1 | 1 | 5928985.0 | 7253.5 | 0.1% | ok |
| trace_metadata_trace-outputs_f32_1024x64_k50 | #2226 #2238 | trace-outputs | tenferro-rs | f32 | rows=1024, cols=64, steps=50 | 1 | 1 | 3035024.0 | 469490.0 | 7.5% | ok |
| trace_metadata_eager_f32_1024x64_k50 | #2226 #2238 | eager | tenferro-rs | f32 | rows=1024, cols=64, steps=50 | 1 | 1 | 6494930.0 | 33823.0 | 1.3% | ok |
| trace_metadata_pytorch_f32_1024x64_k50 | #2226 #2238 | pytorch | pytorch-cpu | f32 | rows=1024, cols=64, steps=50 | 1 | 1 | 3177660.0 | 45254.0 | 1.2% | ok |
| trace_metadata_fixed_neg_f32_8 | #2226 #2238 | trace-fold | tenferro-rs | f32 | rows=8, cols=1, steps=0 | 1 | 1024 | 2838.4 | 17.5 | 0.7% | ok |
| two_operand_einsum_f64_2x2_string | #2325 #2324 | string | tenferro-rs | f64 | m=2, k=2, n=2, format=ij,jk->ik | 4 | 512 | 4158.2 | 35.7 | 1.1% | ok |
| two_operand_einsum_f64_2x2_subscripts | #2325 #2324 | subscripts | tenferro-rs | f64 | m=2, k=2, n=2, format=ij,jk->ik | 4 | 1024 | 2966.3 | 34.4 | 1.1% | ok |
| two_operand_einsum_f64_2x2_prepared | #2325 #2324 | prepared | tenferro-rs | f64 | m=2, k=2, n=2, format=ij,jk->ik | 4 | 1024 | 2466.1 | 12.6 | 0.6% | ok |
| two_operand_einsum_f64_2x2_dot-general | #2325 #2324 | dot-general | tenferro-rs | f64 | m=2, k=2, n=2, format=ij,jk->ik | 4 | 1024 | 2420.2 | 8.2 | 0.6% | ok |
| two_operand_einsum_f64_2x2_pytorch | #2325 #2324 | pytorch | pytorch-cpu | f64 | m=2, k=2, n=2, format=ij,jk->ik | 4 | 256 | 13952.7 | 161.4 | 0.7% | ok |
| two_operand_einsum_f64_16x16_string | #2325 #2324 | string | tenferro-rs | f64 | m=16, k=16, n=16, format=ij,jk->ik | 4 | 512 | 4337.4 | 20.1 | 0.7% | ok |
| two_operand_einsum_f64_16x16_subscripts | #2325 #2324 | subscripts | tenferro-rs | f64 | m=16, k=16, n=16, format=ij,jk->ik | 4 | 1024 | 3174.8 | 22.3 | 0.9% | ok |
| two_operand_einsum_f64_16x16_prepared | #2325 #2324 | prepared | tenferro-rs | f64 | m=16, k=16, n=16, format=ij,jk->ik | 4 | 1024 | 2703.0 | 11.9 | 3.3% | ok |
| two_operand_einsum_f64_16x16_dot-general | #2325 #2324 | dot-general | tenferro-rs | f64 | m=16, k=16, n=16, format=ij,jk->ik | 4 | 1024 | 2611.5 | 15.1 | 1.3% | ok |
| two_operand_einsum_f64_16x16_pytorch | #2325 #2324 | pytorch | pytorch-cpu | f64 | m=16, k=16, n=16, format=ij,jk->ik | 4 | 128 | 15405.1 | 198.6 | 1.0% | ok |
| two_operand_einsum_c64_2x2_string | #2325 #2324 | string | tenferro-rs | c64 | m=2, k=2, n=2, format=ij,jk->ik | 4 | 512 | 4115.0 | 11.9 | 0.5% | ok |
| two_operand_einsum_c64_2x2_subscripts | #2325 #2324 | subscripts | tenferro-rs | c64 | m=2, k=2, n=2, format=ij,jk->ik | 4 | 1024 | 2924.4 | 11.5 | 0.7% | ok |
| two_operand_einsum_c64_2x2_prepared | #2325 #2324 | prepared | tenferro-rs | c64 | m=2, k=2, n=2, format=ij,jk->ik | 4 | 1024 | 2438.7 | 14.0 | 1.1% | ok |
| two_operand_einsum_c64_2x2_dot-general | #2325 #2324 | dot-general | tenferro-rs | c64 | m=2, k=2, n=2, format=ij,jk->ik | 4 | 1024 | 2471.4 | 9.1 | 0.7% | ok |
| two_operand_einsum_c64_2x2_pytorch | #2325 #2324 | pytorch | pytorch-cpu | c64 | m=2, k=2, n=2, format=ij,jk->ik | 4 | 256 | 13918.1 | 112.0 | 0.8% | ok |
| two_operand_einsum_c64_16x16_string | #2325 #2324 | string | tenferro-rs | c64 | m=16, k=16, n=16, format=ij,jk->ik | 4 | 512 | 5369.0 | 15.4 | 0.6% | ok |
| two_operand_einsum_c64_16x16_subscripts | #2325 #2324 | subscripts | tenferro-rs | c64 | m=16, k=16, n=16, format=ij,jk->ik | 4 | 512 | 4055.0 | 9.9 | 0.5% | ok |
| two_operand_einsum_c64_16x16_prepared | #2325 #2324 | prepared | tenferro-rs | c64 | m=16, k=16, n=16, format=ij,jk->ik | 4 | 512 | 3587.3 | 7.1 | 0.6% | ok |
| two_operand_einsum_c64_16x16_dot-general | #2325 #2324 | dot-general | tenferro-rs | c64 | m=16, k=16, n=16, format=ij,jk->ik | 4 | 512 | 3520.5 | 6.9 | 0.6% | ok |
| two_operand_einsum_c64_16x16_pytorch | #2325 #2324 | pytorch | pytorch-cpu | c64 | m=16, k=16, n=16, format=ij,jk->ik | 4 | 128 | 16005.6 | 91.8 | 0.6% | ok |
| two_operand_einsum_f64_abc_cef_abef_string | #2325 #2324 | string | tenferro-rs | f64 | m=8, k=2, n=16, format=abc,cef->abef | 4 | 512 | 4875.5 | 9.4 | 0.6% | ok |
| two_operand_einsum_f64_abc_cef_abef_subscripts | #2325 #2324 | subscripts | tenferro-rs | f64 | m=8, k=2, n=16, format=abc,cef->abef | 4 | 1024 | 3144.1 | 19.2 | 0.9% | ok |
| two_operand_einsum_f64_abc_cef_abef_prepared | #2325 #2324 | prepared | tenferro-rs | f64 | m=8, k=2, n=16, format=abc,cef->abef | 4 | 1024 | 2562.9 | 12.2 | 0.8% | ok |
| two_operand_einsum_f64_abc_cef_abef_dot-general | #2325 #2324 | dot-general | tenferro-rs | f64 | m=8, k=2, n=16, format=abc,cef->abef | 4 | 1024 | 2520.6 | 6.0 | 0.8% | ok |
| two_operand_einsum_f64_abc_cef_abef_pytorch | #2325 #2324 | pytorch | pytorch-cpu | f64 | m=8, k=2, n=16, format=abc,cef->abef | 4 | 128 | 17179.4 | 165.9 | 0.8% | ok |
| decode_proj_f32_eager-shared_in1024_out1024_len8 | #1992 #2003 #1995 #2229 | eager-shared | tenferro-rs | f32 | in=1024, out=1024, len=8 | 4 | 16 | 157315.0 | 2001.2 | 0.9% | ok |
| decode_proj_f32_faer-direct_in1024_out1024_len8 | #1992 #2003 #1995 #2229 | faer-direct | faer-direct | f32 | in=1024, out=1024, len=8 | 4 | 16 | 209793.9 | 3815.6 | 33.1% | NOISY |
| decode_proj_f32_host-sgemm_in1024_out1024_len8 | #1992 #2003 #1995 #2229 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=1024, len=8 | 4 | 2 | 1208466.0 | 5204.8 | 0.5% | ok |
| decode_proj_f32_pytorch_in1024_out1024_len8 | #1992 #2003 #1995 #2229 | pytorch | pytorch-cpu | f32 | in=1024, out=1024, len=8 | 4 | 16 | 187124.2 | 1557.0 | 0.6% | ok |
| decode_proj_f32_eager-shared_in1024_out1024_len64 | #1992 #2003 #1995 #2229 | eager-shared | tenferro-rs | f32 | in=1024, out=1024, len=64 | 4 | 4 | 579657.5 | 1690.5 | 0.3% | ok |
| decode_proj_f32_faer-direct_in1024_out1024_len64 | #1992 #2003 #1995 #2229 | faer-direct | faer-direct | f32 | in=1024, out=1024, len=64 | 4 | 4 | 732329.2 | 37092.8 | 12.9% | NOISY |
| decode_proj_f32_host-sgemm_in1024_out1024_len64 | #1992 #2003 #1995 #2229 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=1024, len=64 | 4 | 1 | 2047074.0 | 11216.0 | 0.3% | ok |
| decode_proj_f32_pytorch_in1024_out1024_len64 | #1992 #2003 #1995 #2229 | pytorch | pytorch-cpu | f32 | in=1024, out=1024, len=64 | 4 | 4 | 951522.5 | 3376.2 | 0.3% | ok |
| decode_proj_f32_eager-shared_in1024_out1024_len512 | #1992 #2003 #1995 #2229 | eager-shared | tenferro-rs | f32 | in=1024, out=1024, len=512 | 4 | 1 | 4407992.0 | 9408.0 | 0.2% | ok |
| decode_proj_f32_faer-direct_in1024_out1024_len512 | #1992 #2003 #1995 #2229 | faer-direct | faer-direct | f32 | in=1024, out=1024, len=512 | 4 | 1 | 4449860.0 | 18820.0 | 0.3% | ok |
| decode_proj_f32_host-sgemm_in1024_out1024_len512 | #1992 #2003 #1995 #2229 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=1024, len=512 | 4 | 1 | 5446166.0 | 49583.0 | 0.6% | ok |
| decode_proj_f32_pytorch_in1024_out1024_len512 | #1992 #2003 #1995 #2229 | pytorch | pytorch-cpu | f32 | in=1024, out=1024, len=512 | 4 | 1 | 7034144.0 | 18224.0 | 0.2% | ok |
| decode_proj_f32_eager-shared_in1024_out4096_len8 | #1992 #2003 #1995 #2229 | eager-shared | tenferro-rs | f32 | in=1024, out=4096, len=8 | 4 | 4 | 620678.8 | 5580.5 | 8.5% | ok |
| decode_proj_f32_faer-direct_in1024_out4096_len8 | #1992 #2003 #1995 #2229 | faer-direct | faer-direct | f32 | in=1024, out=4096, len=8 | 4 | 4 | 657740.2 | 12131.4 | 1.4% | ok |
| decode_proj_f32_host-sgemm_in1024_out4096_len8 | #1992 #2003 #1995 #2229 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=4096, len=8 | 4 | 1 | 4967414.0 | 46977.0 | 1.0% | ok |
| decode_proj_f32_pytorch_in1024_out4096_len8 | #1992 #2003 #1995 #2229 | pytorch | pytorch-cpu | f32 | in=1024, out=4096, len=8 | 4 | 4 | 892011.8 | 20997.8 | 1.7% | ok |
| decode_proj_f32_eager-shared_in1024_out4096_len64 | #1992 #2003 #1995 #2229 | eager-shared | tenferro-rs | f32 | in=1024, out=4096, len=64 | 4 | 1 | 2230135.0 | 6682.0 | 0.4% | ok |
| decode_proj_f32_faer-direct_in1024_out4096_len64 | #1992 #2003 #1995 #2229 | faer-direct | faer-direct | f32 | in=1024, out=4096, len=64 | 4 | 1 | 2341412.0 | 42859.5 | 1.4% | ok |
| decode_proj_f32_host-sgemm_in1024_out4096_len64 | #1992 #2003 #1995 #2229 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=4096, len=64 | 4 | 1 | 8359263.0 | 20573.0 | 0.2% | ok |
| decode_proj_f32_pytorch_in1024_out4096_len64 | #1992 #2003 #1995 #2229 | pytorch | pytorch-cpu | f32 | in=1024, out=4096, len=64 | 4 | 1 | 3849242.0 | 23259.0 | 0.6% | ok |
| decode_proj_f32_eager-shared_in1024_out4096_len512 | #1992 #2003 #1995 #2229 | eager-shared | tenferro-rs | f32 | in=1024, out=4096, len=512 | 4 | 1 | 17634643.0 | 17326.5 | 0.1% | ok |
| decode_proj_f32_faer-direct_in1024_out4096_len512 | #1992 #2003 #1995 #2229 | faer-direct | faer-direct | f32 | in=1024, out=4096, len=512 | 4 | 1 | 17639021.0 | 22962.5 | 0.1% | ok |
| decode_proj_f32_host-sgemm_in1024_out4096_len512 | #1992 #2003 #1995 #2229 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=4096, len=512 | 4 | 1 | 22319731.0 | 283263.0 | 1.5% | ok |
| decode_proj_f32_pytorch_in1024_out4096_len512 | #1992 #2003 #1995 #2229 | pytorch | pytorch-cpu | f32 | in=1024, out=4096, len=512 | 4 | 1 | 27510122.0 | 60503.5 | 1.2% | ok |
| decode_proj_view_operand_f32_prepared_in1024_out1024_len8_t4 | #2216 | prepared-view | tenferro-rs | f32 | in=1024, out=1024, len=8 | 4 | 16 | 160339.9 | 177.5 | 0.1% | ok |
| decode_proj_contiguous_operand_f32_prepared_in1024_out1024_len8_t4 | #2216 | prepared-contiguous | tenferro-rs | f32 | in=1024, out=1024, len=8 | 4 | 16 | 154024.4 | 467.7 | 0.3% | ok |
| decode_proj_view_operand_f32_pytorch_in1024_out1024_len8_t4 | #2216 | pytorch | pytorch-cpu | f32 | in=1024, out=1024, len=8 | 4 | 16 | 185798.6 | 882.6 | 0.4% | ok |
| gemm_c64_mm256_t1 | #1900 | dot-general-shared | tenferro-rs | c64 | n=256 | 4 | 1 | 4458756.0 | 5479.5 | 0.2% | ok |
| gemm_f64_mm256_t1 | #1900 | dot-general-shared | tenferro-rs | f64 | n=256 | 4 | 2 | 1073786.0 | 1775.8 | 0.1% | ok |
| conj_dot_f64_dot-general-rank1_len100 | #1615 | dot-general-rank1 | tenferro-rs | f64 | len=100 | 4 | 1024 | 2223.2 | 8.8 | 1.5% | ok |
| conj_dot_f64_portable-loop_len100 | #1615 | portable-loop | rust-loop | f64 | len=100 | 4 | 32768 | 89.4 | 0.1 | 0.4% | ok |
| conj_dot_f64_pytorch-vdot_len100 | #1615 | pytorch-vdot | pytorch-cpu | f64 | len=100 | 4 | 2048 | 1665.2 | 11.4 | 5.5% | ok |
| conj_dot_f64_dot-general-rank1_len10000 | #1615 | dot-general-rank1 | tenferro-rs | f64 | len=10000 | 4 | 512 | 4816.9 | 16.9 | 0.9% | ok |
| conj_dot_f64_portable-loop_len10000 | #1615 | portable-loop | rust-loop | f64 | len=10000 | 4 | 256 | 15014.2 | 21.8 | 0.3% | ok |
| conj_dot_f64_pytorch-vdot_len10000 | #1615 | pytorch-vdot | pytorch-cpu | f64 | len=10000 | 4 | 1024 | 3326.1 | 46.9 | 10.8% | NOISY |
| conj_dot_f64_dot-general-rank1_len1000000 | #1615 | dot-general-rank1 | tenferro-rs | f64 | len=1000000 | 4 | 32 | 94197.4 | 598.0 | 1.0% | ok |
| conj_dot_f64_portable-loop_len1000000 | #1615 | portable-loop | rust-loop | f64 | len=1000000 | 4 | 2 | 1544091.5 | 1926.2 | 0.2% | ok |
| conj_dot_f64_pytorch-vdot_len1000000 | #1615 | pytorch-vdot | pytorch-cpu | f64 | len=1000000 | 4 | 32 | 92725.2 | 1130.2 | 1.2% | ok |
| conj_dot_c64_dot-general-rank1_len100 | #1615 | dot-general-rank1 | tenferro-rs | c64 | len=100 | 4 | 1024 | 2329.0 | 6.7 | 1.4% | ok |
| conj_dot_c64_portable-loop_len100 | #1615 | portable-loop | rust-loop | c64 | len=100 | 4 | 16384 | 145.0 | 0.3 | 0.5% | ok |
| conj_dot_c64_pytorch-vdot_len100 | #1615 | pytorch-vdot | pytorch-cpu | c64 | len=100 | 4 | 2048 | 1571.4 | 9.1 | 257.7% | NOISY |
| conj_dot_c64_dot-general-rank1_len10000 | #1615 | dot-general-rank1 | tenferro-rs | c64 | len=10000 | 4 | 128 | 16243.2 | 19.1 | 0.1% | ok |
| conj_dot_c64_portable-loop_len10000 | #1615 | portable-loop | rust-loop | c64 | len=10000 | 4 | 256 | 15320.0 | 21.8 | 0.1% | ok |
| conj_dot_c64_pytorch-vdot_len10000 | #1615 | pytorch-vdot | pytorch-cpu | c64 | len=10000 | 4 | 512 | 4470.2 | 39.0 | 0.5% | ok |
| conj_dot_c64_dot-general-rank1_len1000000 | #1615 | dot-general-rank1 | tenferro-rs | c64 | len=1000000 | 4 | 8 | 400304.5 | 15576.5 | 2.1% | ok |
| conj_dot_c64_portable-loop_len1000000 | #1615 | portable-loop | rust-loop | c64 | len=1000000 | 4 | 2 | 1548004.0 | 5339.8 | 0.2% | ok |
| conj_dot_c64_pytorch-vdot_len1000000 | #1615 | pytorch-vdot | pytorch-cpu | c64 | len=1000000 | 4 | 8 | 354187.4 | 8813.9 | 7.2% | ok |
| small_solve_f64_solve-loop_k16_b64 | #2007 #1878 | solve-loop | tenferro-rs | f64 | k=16, batch=64 | 4 | 8 | 360209.9 | 1603.6 | 0.6% | ok |
| small_solve_f64_triangular-solve-loop_k16_b64 | #2007 #1878 | triangular-solve-loop | tenferro-rs | f64 | k=16, batch=64 | 4 | 16 | 200848.6 | 433.0 | 0.4% | ok |
| small_solve_f64_solve-batched_k16_b64 | #2007 #1878 | solve-batched | tenferro-rs | f64 | k=16, batch=64 | 4 | 32 | 53588.3 | 673.0 | 1.2% | ok |
| small_solve_f64_pytorch-solve-batched_k16_b64 | #2007 #1878 | pytorch-solve-batched | pytorch-cpu | f64 | k=16, batch=64 | 4 | 16 | 182188.8 | 1227.0 | 0.4% | ok |
| small_solve_f64_pytorch-solve-triangular-batched_k16_b64 | #2007 #1878 | pytorch-solve-triangular-batched | pytorch-cpu | f64 | k=16, batch=64 | 4 | 32 | 92646.7 | 150.4 | 0.2% | ok |
| small_solve_f64_solve-loop_k64_b64 | #2007 #1878 | solve-loop | tenferro-rs | f64 | k=64, batch=64 | 4 | 1 | 3505250.0 | 7429.0 | 0.3% | ok |
| small_solve_f64_triangular-solve-loop_k64_b64 | #2007 #1878 | triangular-solve-loop | tenferro-rs | f64 | k=64, batch=64 | 4 | 2 | 1675917.5 | 4455.8 | 0.4% | ok |
| small_solve_f64_solve-batched_k64_b64 | #2007 #1878 | solve-batched | tenferro-rs | f64 | k=64, batch=64 | 4 | 2 | 855739.5 | 8693.8 | 1.2% | ok |
| small_solve_f64_pytorch-solve-batched_k64_b64 | #2007 #1878 | pytorch-solve-batched | pytorch-cpu | f64 | k=64, batch=64 | 4 | 1 | 2271812.0 | 6902.5 | 0.6% | ok |
| small_solve_f64_pytorch-solve-triangular-batched_k64_b64 | #2007 #1878 | pytorch-solve-triangular-batched | pytorch-cpu | f64 | k=64, batch=64 | 4 | 4 | 594337.2 | 2030.0 | 0.4% | ok |
| tanh_chain_f32_compiled-prepared_k10_1024x64 | #1990 | compiled-prepared | tenferro-rs | f32 | rows=1024, cols=64, chain=10 | 4 | 1 | 442434.0 | 6186.5 | 94.2% | NOISY |
| tanh_chain_f32_eager-shared_k10_1024x64 | #1990 | eager-shared | tenferro-rs | f32 | rows=1024, cols=64, chain=10 | 4 | 8 | 480664.2 | 1211.7 | 2.6% | ok |
| tanh_chain_f32_compiled-prepared_k100_1024x64 | #1990 | compiled-prepared | tenferro-rs | f32 | rows=1024, cols=64, chain=100 | 4 | 1 | 4370823.0 | 23038.5 | 3.4% | ok |
| tanh_chain_f32_eager-shared_k100_1024x64 | #1990 | eager-shared | tenferro-rs | f32 | rows=1024, cols=64, chain=100 | 4 | 1 | 4778833.0 | 9778.0 | 0.1% | ok |
| tanh_chain_f32_compiled-prepared_k200_1024x64 | #1990 | compiled-prepared | tenferro-rs | f32 | rows=1024, cols=64, chain=200 | 4 | 1 | 8704136.0 | 31334.0 | 0.3% | ok |
| tanh_chain_f32_eager-shared_k200_1024x64 | #1990 | eager-shared | tenferro-rs | f32 | rows=1024, cols=64, chain=200 | 4 | 1 | 9546005.0 | 18208.5 | 0.3% | ok |
| layer_norm_f32_eager-composed_d1024_len8_b1 | #2006 | eager-composed | tenferro-rs | f32 | norm=layer_norm, d=1024, len=8, batch=1 | 4 | 32 | 68258.8 | 246.2 | 0.6% | ok |
| layer_norm_f32_pytorch-fused_d1024_len8_b1 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=layer_norm, d=1024, len=8, batch=1 | 4 | 512 | 6622.1 | 66.6 | 1.3% | ok |
| layer_norm_f32_eager-single-call_d1024_len8_b1 | #2006 | eager-single-call | tenferro-rs | f32 | norm=layer_norm, d=1024, len=8, batch=1 | 4 | 32 | 71651.3 | 563.6 | 0.6% | ok |
| layer_norm_f32_eager-composed_d1024_len8_b8 | #2006 | eager-composed | tenferro-rs | f32 | norm=layer_norm, d=1024, len=8, batch=8 | 4 | 16 | 117683.0 | 2408.5 | 1.4% | ok |
| layer_norm_f32_pytorch-fused_d1024_len8_b8 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=layer_norm, d=1024, len=8, batch=8 | 4 | 128 | 13608.1 | 344.3 | 4.7% | ok |
| layer_norm_f32_eager-single-call_d1024_len8_b8 | #2006 | eager-single-call | tenferro-rs | f32 | norm=layer_norm, d=1024, len=8, batch=8 | 4 | 16 | 119641.6 | 1295.8 | 0.7% | ok |
| layer_norm_f32_eager-composed_d1024_len64_b1 | #2006 | eager-composed | tenferro-rs | f32 | norm=layer_norm, d=1024, len=64, batch=1 | 4 | 16 | 116944.1 | 1802.8 | 1.0% | ok |
| layer_norm_f32_pytorch-fused_d1024_len64_b1 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=layer_norm, d=1024, len=64, batch=1 | 4 | 256 | 30440.1 | 1119.6 | 2.4% | ok |
| layer_norm_f32_eager-single-call_d1024_len64_b1 | #2006 | eager-single-call | tenferro-rs | f32 | norm=layer_norm, d=1024, len=64, batch=1 | 4 | 16 | 119911.6 | 839.7 | 0.6% | ok |
| layer_norm_f32_eager-composed_d1024_len64_b8 | #2006 | eager-composed | tenferro-rs | f32 | norm=layer_norm, d=1024, len=64, batch=8 | 4 | 4 | 405836.2 | 4260.5 | 1.1% | ok |
| layer_norm_f32_pytorch-fused_d1024_len64_b8 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=layer_norm, d=1024, len=64, batch=8 | 4 | 32 | 277929.7 | 5568.0 | 2.0% | ok |
| layer_norm_f32_eager-single-call_d1024_len64_b8 | #2006 | eager-single-call | tenferro-rs | f32 | norm=layer_norm, d=1024, len=64, batch=8 | 4 | 4 | 380413.8 | 12621.1 | 2.6% | ok |
| rms_norm_f32_eager-composed_d1024_len8_b1 | #2006 | eager-composed | tenferro-rs | f32 | norm=rms_norm, d=1024, len=8, batch=1 | 4 | 64 | 36004.0 | 95.0 | 0.5% | ok |
| rms_norm_f32_pytorch-fused_d1024_len8_b1 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=rms_norm, d=1024, len=8, batch=1 | 4 | 128 | 18518.9 | 135.4 | 0.8% | ok |
| rms_norm_f32_eager-single-call_d1024_len8_b1 | #2006 | eager-single-call | tenferro-rs | f32 | norm=rms_norm, d=1024, len=8, batch=1 | 4 | 64 | 42594.0 | 114.7 | 0.5% | ok |
| rms_norm_f32_eager-composed_d1024_len8_b8 | #2006 | eager-composed | tenferro-rs | f32 | norm=rms_norm, d=1024, len=8, batch=8 | 4 | 32 | 57806.8 | 389.6 | 0.4% | ok |
| rms_norm_f32_pytorch-fused_d1024_len8_b8 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=rms_norm, d=1024, len=8, batch=8 | 4 | 32 | 41006.1 | 405.3 | 0.9% | ok |
| rms_norm_f32_eager-single-call_d1024_len8_b8 | #2006 | eager-single-call | tenferro-rs | f32 | norm=rms_norm, d=1024, len=8, batch=8 | 4 | 32 | 63815.2 | 256.7 | 0.3% | ok |
| rms_norm_f32_eager-composed_d1024_len64_b1 | #2006 | eager-composed | tenferro-rs | f32 | norm=rms_norm, d=1024, len=64, batch=1 | 4 | 32 | 57549.7 | 343.5 | 1.6% | ok |
| rms_norm_f32_pytorch-fused_d1024_len64_b1 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=rms_norm, d=1024, len=64, batch=1 | 4 | 64 | 41451.3 | 352.6 | 0.9% | ok |
| rms_norm_f32_eager-single-call_d1024_len64_b1 | #2006 | eager-single-call | tenferro-rs | f32 | norm=rms_norm, d=1024, len=64, batch=1 | 4 | 2 | 63383.5 | 1154.5 | 1.9% | ok |
| rms_norm_f32_eager-composed_d1024_len64_b8 | #2006 | eager-composed | tenferro-rs | f32 | norm=rms_norm, d=1024, len=64, batch=8 | 4 | 4 | 179694.8 | 2713.8 | 2.1% | ok |
| rms_norm_f32_pytorch-fused_d1024_len64_b8 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=rms_norm, d=1024, len=64, batch=8 | 4 | 8 | 522106.4 | 14772.6 | 5.0% | ok |
| rms_norm_f32_eager-single-call_d1024_len64_b8 | #2006 | eager-single-call | tenferro-rs | f32 | norm=rms_norm, d=1024, len=64, batch=8 | 4 | 8 | 212095.1 | 3677.4 | 1.1% | ok |
| small_contraction_abcd-dbef-acef_f64_shared-session-default-pool_d4 | #1885 #1897 #1901 | shared-session-default-pool | tenferro-rs | f64 | extent=4 | 4 | 512 | 4211.9 | 28.3 | 1.0% | ok |
| activation_erf_f32_eager-single-call_1024x64 | #1975 #2069 #2032 | eager-single-call | tenferro-rs | f32 | op=erf, rows=1024, cols=64 | 4 | — | — | — | — | FAILED |
| activation_erf_f32_session-single-call_1024x64 | #1975 #2069 #2032 | session-single-call | tenferro-rs | f32 | op=erf, rows=1024, cols=64 | 4 | 32 | 76574.2 | 337.4 | 0.5% | ok |
| activation_erf_f32_pytorch_1024x64 | #1975 #2069 #2032 | pytorch | pytorch-cpu | f32 | op=erf, rows=1024, cols=64 | 4 | — | — | — | — | FAILED |
| activation_sigmoid_f32_eager-single-call_1024x64 | #1975 #2069 #2032 | eager-single-call | tenferro-rs | f32 | op=sigmoid, rows=1024, cols=64 | 4 | — | — | — | — | FAILED |
| activation_sigmoid_f32_session-single-call_1024x64 | #1975 #2069 #2032 | session-single-call | tenferro-rs | f32 | op=sigmoid, rows=1024, cols=64 | 4 | 128 | 13001.6 | 142.6 | 1.8% | ok |
| activation_sigmoid_f32_eager-composed_1024x64 | #1975 #2069 #2032 | eager-composed | tenferro-rs | f32 | op=sigmoid, rows=1024, cols=64 | 4 | 64 | 37393.4 | 485.5 | 1.2% | ok |
| activation_sigmoid_f32_pytorch_1024x64 | #1975 #2069 #2032 | pytorch | pytorch-cpu | f32 | op=sigmoid, rows=1024, cols=64 | 4 | — | — | — | — | FAILED |
| activation_silu_f32_eager-single-call_1024x64 | #1975 #2069 #2032 | eager-single-call | tenferro-rs | f32 | op=silu, rows=1024, cols=64 | 4 | — | — | — | — | FAILED |
| activation_silu_f32_session-single-call_1024x64 | #1975 #2069 #2032 | session-single-call | tenferro-rs | f32 | op=silu, rows=1024, cols=64 | 4 | 128 | 13423.2 | 185.1 | 1.5% | ok |
| activation_silu_f32_eager-composed_1024x64 | #1975 #2069 #2032 | eager-composed | tenferro-rs | f32 | op=silu, rows=1024, cols=64 | 4 | 64 | 45841.2 | 264.9 | 0.6% | ok |
| activation_silu_f32_pytorch_1024x64 | #1975 #2069 #2032 | pytorch | pytorch-cpu | f32 | op=silu, rows=1024, cols=64 | 4 | — | — | — | — | FAILED |
| activation_softplus_f32_eager-single-call_1024x64 | #1975 #2069 #2032 | eager-single-call | tenferro-rs | f32 | op=softplus, rows=1024, cols=64 | 4 | — | — | — | — | FAILED |
| activation_softplus_f32_session-single-call_1024x64 | #1975 #2069 #2032 | session-single-call | tenferro-rs | f32 | op=softplus, rows=1024, cols=64 | 4 | 64 | 42237.8 | 1101.7 | 5.6% | ok |
| activation_softplus_f32_eager-composed_1024x64 | #1975 #2069 #2032 | eager-composed | tenferro-rs | f32 | op=softplus, rows=1024, cols=64 | 4 | 32 | 68624.7 | 544.9 | 0.6% | ok |
| activation_softplus_f32_pytorch_1024x64 | #1975 #2069 #2032 | pytorch | pytorch-cpu | f32 | op=softplus, rows=1024, cols=64 | 4 | — | — | — | — | FAILED |
| activation_gelu_f32_eager-single-call_1024x64 | #1975 #2030 #2069 #2228 | eager-single-call | tenferro-rs | f32 | op=gelu, rows=1024, cols=64 | 4 | 64 | 18404.9 | 340.8 | 2.6% | ok |
| activation_gelu_f32_session-single-call_1024x64 | #1975 #2030 #2069 #2228 | session-single-call | tenferro-rs | f32 | op=gelu, rows=1024, cols=64 | 4 | 64 | 17850.3 | 106.1 | 0.7% | ok |
| activation_gelu_f32_pytorch_1024x64 | #1975 #2030 #2069 #2228 | pytorch | pytorch-cpu | f32 | op=gelu, rows=1024, cols=64 | 4 | 64 | 21805.8 | 162.6 | 0.8% | ok |
| activation_gelu_tanh_f32_eager-single-call_1024x64 | #1975 #2069 #2032 | eager-single-call | tenferro-rs | f32 | op=gelu_tanh, rows=1024, cols=64 | 4 | — | — | — | — | FAILED |
| activation_gelu_tanh_f32_session-single-call_1024x64 | #1975 #2069 #2032 | session-single-call | tenferro-rs | f32 | op=gelu_tanh, rows=1024, cols=64 | 4 | 64 | 49544.6 | 101.3 | 0.2% | ok |
| activation_gelu_tanh_f32_eager-composed_1024x64 | #1975 #2069 #2032 | eager-composed | tenferro-rs | f32 | op=gelu_tanh, rows=1024, cols=64 | 4 | 16 | 138338.4 | 748.0 | 0.4% | ok |
| activation_gelu_tanh_f32_pytorch_1024x64 | #1975 #2069 #2032 | pytorch | pytorch-cpu | f32 | op=gelu_tanh, rows=1024, cols=64 | 4 | — | — | — | — | FAILED |
| softmax_f32_eager-single-call_len64_b8 | #1976 #2031 #2231 | eager-single-call | tenferro-rs | f32 | op=softmax, len=64, batch=8, axis=0 | 4 | 128 | 14407.9 | 219.9 | 1.3% | ok |
| softmax_f32_session-single-call_len64_b8 | #1976 #2031 #2231 | session-single-call | tenferro-rs | f32 | op=softmax, len=64, batch=8, axis=0 | 4 | 128 | 14013.5 | 70.4 | 1.3% | ok |
| softmax_f32_eager-composed_len64_b8 | #1976 #2031 #2231 | eager-composed | tenferro-rs | f32 | op=softmax, len=64, batch=8, axis=0 | 4 | 32 | 61706.8 | 351.4 | 1.1% | ok |
| softmax_f32_pytorch_len64_b8 | #1976 #2031 #2231 | pytorch | pytorch-cpu | f32 | op=softmax, len=64, batch=8, axis=0 | 4 | 256 | 13315.5 | 103.6 | 0.9% | ok |
| softmax_f32_eager-single-call_len512_b8 | #1976 #2231 | eager-single-call | tenferro-rs | f32 | op=softmax, len=512, batch=8, axis=0 | 4 | 2 | 501810.0 | 7629.2 | 1.5% | ok |
| softmax_f32_session-single-call_len512_b8 | #1976 #2231 | session-single-call | tenferro-rs | f32 | op=softmax, len=512, batch=8, axis=0 | 4 | 4 | 492622.8 | 7309.9 | 1.1% | ok |
| softmax_f32_eager-composed_len512_b8 | #1976 #2231 | eager-composed | tenferro-rs | f32 | op=softmax, len=512, batch=8, axis=0 | 4 | 2 | 1707856.5 | 10747.2 | 0.7% | ok |
| softmax_f32_pytorch_len512_b8 | #1976 #2231 | pytorch | pytorch-cpu | f32 | op=softmax, len=512, batch=8, axis=0 | 4 | 4 | 1308878.5 | 50408.8 | 2.9% | ok |
| log_softmax_f32_eager-single-call_len64_b8 | #1976 | eager-single-call | tenferro-rs | f32 | op=log_softmax, len=64, batch=8, axis=0 | 4 | 32 | 85506.8 | 327.6 | 0.3% | ok |
| log_softmax_f32_session-single-call_len64_b8 | #1976 | session-single-call | tenferro-rs | f32 | op=log_softmax, len=64, batch=8, axis=0 | 4 | 32 | 61290.4 | 347.2 | 0.5% | ok |
| log_softmax_f32_eager-composed_len64_b8 | #1976 | eager-composed | tenferro-rs | f32 | op=log_softmax, len=64, batch=8, axis=0 | 4 | 32 | 63708.4 | 663.4 | 1.2% | ok |
| log_softmax_f32_pytorch_len64_b8 | #1976 | pytorch | pytorch-cpu | f32 | op=log_softmax, len=64, batch=8, axis=0 | 4 | 256 | 13440.4 | 134.2 | 0.8% | ok |
| log_softmax_f32_eager-single-call_len512_b8 | #1976 | eager-single-call | tenferro-rs | f32 | op=log_softmax, len=512, batch=8, axis=0 | 4 | 2 | 1687484.0 | 33925.5 | 2.0% | ok |
| log_softmax_f32_session-single-call_len512_b8 | #1976 | session-single-call | tenferro-rs | f32 | op=log_softmax, len=512, batch=8, axis=0 | 4 | 2 | 1775418.0 | 18699.5 | 0.8% | ok |
| log_softmax_f32_eager-composed_len512_b8 | #1976 | eager-composed | tenferro-rs | f32 | op=log_softmax, len=512, batch=8, axis=0 | 4 | 2 | 1789499.0 | 22196.2 | 6.1% | ok |
| log_softmax_f32_pytorch_len512_b8 | #1976 | pytorch | pytorch-cpu | f32 | op=log_softmax, len=512, batch=8, axis=0 | 4 | 4 | 1204391.2 | 59545.9 | 4.6% | ok |
| masked_softmax_f32_eager-single-call_len64_b8 | #1976 | eager-single-call | tenferro-rs | f32 | op=masked_softmax, len=64, batch=8, axis=0 | 4 | 16 | 120254.7 | 778.3 | 0.4% | ok |
| masked_softmax_f32_session-single-call_len64_b8 | #1976 | session-single-call | tenferro-rs | f32 | op=masked_softmax, len=64, batch=8, axis=0 | 4 | 32 | 96629.4 | 772.8 | 0.5% | ok |
| masked_softmax_f32_eager-composed_len64_b8 | #1976 | eager-composed | tenferro-rs | f32 | op=masked_softmax, len=64, batch=8, axis=0 | 4 | 32 | 73601.8 | 1018.2 | 1.1% | ok |
| masked_softmax_f32_pytorch_len64_b8 | #1976 | pytorch | pytorch-cpu | f32 | op=masked_softmax, len=64, batch=8, axis=0 | 4 | 32 | 66992.9 | 487.0 | 0.9% | ok |
| masked_softmax_f32_eager-single-call_len512_b8 | #1976 | eager-single-call | tenferro-rs | f32 | op=masked_softmax, len=512, batch=8, axis=0 | 4 | 1 | 2860920.0 | 24270.0 | 1.0% | ok |
| masked_softmax_f32_session-single-call_len512_b8 | #1976 | session-single-call | tenferro-rs | f32 | op=masked_softmax, len=512, batch=8, axis=0 | 4 | 1 | 2905924.0 | 36091.5 | 2.9% | ok |
| masked_softmax_f32_eager-composed_len512_b8 | #1976 | eager-composed | tenferro-rs | f32 | op=masked_softmax, len=512, batch=8, axis=0 | 4 | 1 | 2376939.0 | 32601.5 | 0.9% | ok |
| masked_softmax_f32_pytorch_len512_b8 | #1976 | pytorch | pytorch-cpu | f32 | op=masked_softmax, len=512, batch=8, axis=0 | 4 | 2 | 3000340.0 | 101424.0 | 5.7% | ok |
| softmax_f32_eager-single-call_len64_b8_t4 | #2233 #2235 | eager-single-call | tenferro-rs | f32 | op=softmax, len=64, batch=8, axis=0 | 4 | 128 | 14503.4 | 88.3 | 1.1% | ok |
| softmax_f32_eager-composed_len64_b8_t4 | #2233 #2235 | eager-composed | tenferro-rs | f32 | op=softmax, len=64, batch=8, axis=0 | 4 | 32 | 61890.0 | 510.2 | 0.7% | ok |
| softmax_f32_pytorch_len64_b8_t4 | #2233 #2235 | pytorch | pytorch-cpu | f32 | op=softmax, len=64, batch=8, axis=0 | 4 | 256 | 13413.8 | 147.9 | 1.0% | ok |
| reduce_mean_f32_eager-single-call_1024x64_axis0 | #1976 | eager-single-call | tenferro-rs | f32 | rows=1024, cols=64, axis=0 | 4 | 128 | 15290.5 | 26.9 | 0.2% | ok |
| reduce_mean_f32_session-single-call_1024x64_axis0 | #1976 | session-single-call | tenferro-rs | f32 | rows=1024, cols=64, axis=0 | 4 | 256 | 7900.5 | 46.8 | 0.4% | ok |
| reduce_mean_f32_eager-composed_1024x64_axis0 | #1976 | eager-composed | tenferro-rs | f32 | rows=1024, cols=64, axis=0 | 4 | 128 | 15230.5 | 49.7 | 0.4% | ok |
| reduce_mean_f32_pytorch_1024x64_axis0 | #1976 | pytorch | pytorch-cpu | f32 | rows=1024, cols=64, axis=0 | 4 | 256 | 7954.1 | 66.3 | 0.6% | ok |
| take_along_axis_rows_f64_eager-single-call_n64_b256 | #2008 | eager-single-call | tenferro-rs | f64 | n=64, batch=256, axis=0 | 4 | 1 | 3108291.0 | 177696.5 | 3.5% | ok |
| take_along_axis_rows_f64_session-single-call_n64_b256 | #2008 | session-single-call | tenferro-rs | f64 | n=64, batch=256, axis=0 | 4 | 1 | 3032861.0 | 85699.5 | 2.3% | ok |
| take_along_axis_rows_f64_eager-gather-prebuilt_n64_b256 | #2008 | eager-gather-prebuilt | tenferro-rs | f64 | n=64, batch=256, axis=0 | 4 | 1 | 2870097.0 | 71879.0 | 2.1% | ok |
| take_along_axis_rows_f64_pytorch_n64_b256 | #2008 | pytorch | pytorch-cpu | f64 | n=64, batch=256, axis=0 | 4 | 2 | 1274189.0 | 9147.0 | 1.1% | ok |
| lstsq_f64_eager-shared_m768_n384_rhs16 | #2027 | eager-shared | tenferro-rs | f64 | operation=lstsq | 4 | 1 | 7830289.0 | 63228.0 | 2.3% | ok |
| lstsq_f64_pytorch_m768_n384_rhs16 | #2027 | pytorch | pytorch-cpu | f64 | operation=lstsq | 4 | 1 | 5177476.0 | 8997.5 | 0.1% | ok |
| triangular_f64_concrete-shared_n4096_rhs64 | #2028 | concrete-shared | tenferro-rs | f64 | operation=triangular | 4 | 1 | 15667110.0 | 17973.5 | 0.1% | ok |
| triangular_f64_pytorch_n4096_rhs64 | #2028 | pytorch | pytorch-cpu | f64 | operation=triangular | 4 | 1 | 16190084.0 | 167061.0 | 2.0% | ok |
| reshape_f64_concrete-shared_n33554432_to8192x4096 | #2029 | concrete-shared | tenferro-rs | f64 | operation=reshape | 4 | 1 | 23708141.0 | 153086.0 | 2.3% | ok |
| reshape_f64_pytorch_n33554432_to8192x4096 | #2029 | pytorch | pytorch-cpu | f64 | operation=reshape | 4 | 1 | 25717715.0 | 572506.5 | 9.5% | ok |
| metadata_transpose_view_f64_rank8_t4 | #2180 | metadata-clone | tenferro-rs | f64 | shape=[2, 2, 2, 2, 2, 2, 2, 2] | 4 | 8192 | 80.3 | 0.7 | 44.4% | NOISY |
| metadata_transpose_view_f64_rank8_t4_julia-base | #2180 | julia-base | julia-base | f64 | shape=[2, 2, 2, 2, 2, 2, 2, 2] | 4 | 1048576 | 0.3 | 0.0 | 7.4% | ok |
| followup_structural_broadcast_in_dim_direct-shared | #2021 | direct-shared | tenferro-rs | f64 | case_id=structural.broadcast_in_dim, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [8192, 1], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | FAILED |
| followup_structural_broadcast_in_dim_pytorch | #2021 | pytorch | pytorch-cpu | f64 | case_id=structural.broadcast_in_dim, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [8192, 1], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | FAILED |
| followup_real_tanh_direct-shared | #2033 | direct-shared | tenferro-rs | f64 | case_id=real.tanh, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [8388608], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | FAILED |
| followup_real_tanh_jax | #2033 | jax | jax-cpu | f64 | case_id=real.tanh, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [8388608], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | FAILED |
| followup_real_log1p_direct-shared | #2033 | direct-shared | tenferro-rs | f64 | case_id=real.log1p, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [4194304], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | FAILED |
| followup_real_log1p_pytorch | #2033 | pytorch | pytorch-cpu | f64 | case_id=real.log1p, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [4194304], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | FAILED |
| followup_real_cos_direct-shared | #2033 | direct-shared | tenferro-rs | f64 | case_id=real.cos, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [8388608], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | FAILED |
| followup_real_cos_pytorch | #2033 | pytorch | pytorch-cpu | f64 | case_id=real.cos, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [8388608], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | FAILED |
| followup_real_sin_direct-shared | #2033 | direct-shared | tenferro-rs | f64 | case_id=real.sin, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [8388608], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | FAILED |
| followup_real_sin_pytorch | #2033 | pytorch | pytorch-cpu | f64 | case_id=real.sin, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [8388608], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | FAILED |
| followup_real_exp_direct-shared | #2033 | direct-shared | tenferro-rs | f64 | case_id=real.exp, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [8388608], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | FAILED |
| followup_real_exp_jax | #2033 | jax | jax-cpu | f64 | case_id=real.exp, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [8388608], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | FAILED |
| followup_real_expm1_direct-shared | #2033 | direct-shared | tenferro-rs | f64 | case_id=real.expm1, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [4194304], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | FAILED |
| followup_real_expm1_jax | #2033 | jax | jax-cpu | f64 | case_id=real.expm1, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [4194304], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | FAILED |
| followup_real_pow_direct-shared | #2033 | direct-shared | tenferro-rs | f64 | case_id=real.pow, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [4194304], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | FAILED |
| followup_real_pow_pytorch | #2033 | pytorch | pytorch-cpu | f64 | case_id=real.pow, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [4194304], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | FAILED |
| followup_complex_exp_direct-shared | #2033 | direct-shared | tenferro-rs | c64 | case_id=complex.exp, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [4194304], 'dtype': 'c64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | FAILED |
| followup_complex_exp_julia-base | #2033 | julia-base | julia-base | c64 | case_id=complex.exp, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [4194304], 'dtype': 'c64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | FAILED |
| followup_real_log_direct-shared | #2033 | direct-shared | tenferro-rs | f64 | case_id=real.log, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [8388608], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | FAILED |
| followup_real_log_pytorch | #2033 | pytorch | pytorch-cpu | f64 | case_id=real.log, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [8388608], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | FAILED |
| followup_real_reduce_max_axis1_direct-shared | #2034 | direct-shared | tenferro-rs | f64 | case_id=real.reduce_max_axis1, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [2048, 2048], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | FAILED |
| followup_real_reduce_max_axis1_julia-base | #2034 | julia-base | julia-base | f64 | case_id=real.reduce_max_axis1, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [2048, 2048], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | FAILED |
| followup_real_reduce_prod_axis1_direct-shared | #2034 | direct-shared | tenferro-rs | f64 | case_id=real.reduce_prod_axis1, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [2048, 2048], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | FAILED |
| followup_real_reduce_prod_axis1_pytorch | #2034 | pytorch | pytorch-cpu | f64 | case_id=real.reduce_prod_axis1, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [2048, 2048], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | FAILED |
| followup_real_reduce_sum_axis1_direct-shared | #2034 | direct-shared | tenferro-rs | f64 | case_id=real.reduce_sum_axis1, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [2048, 2048], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | FAILED |
| followup_real_reduce_sum_axis1_pytorch | #2034 | pytorch | pytorch-cpu | f64 | case_id=real.reduce_sum_axis1, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [2048, 2048], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | FAILED |
| followup_real_reduce_max_axis0_direct-shared | #2034 | direct-shared | tenferro-rs | f64 | case_id=real.reduce_max_axis0, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [2048, 2048], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | FAILED |
| followup_real_reduce_max_axis0_julia-base | #2034 | julia-base | julia-base | f64 | case_id=real.reduce_max_axis0, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [2048, 2048], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | FAILED |
| followup_real_reduce_min_axis1_direct-shared | #2034 | direct-shared | tenferro-rs | f64 | case_id=real.reduce_min_axis1, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [4096, 4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | FAILED |
| followup_real_reduce_min_axis1_julia-base | #2034 | julia-base | julia-base | f64 | case_id=real.reduce_min_axis1, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [4096, 4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | FAILED |
| followup_real_reduce_min_axis0_direct-shared | #2034 | direct-shared | tenferro-rs | f64 | case_id=real.reduce_min_axis0, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [2048, 2048], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | FAILED |
| followup_real_reduce_min_axis0_julia-base | #2034 | julia-base | julia-base | f64 | case_id=real.reduce_min_axis0, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [2048, 2048], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | FAILED |
| followup_real_reduce_max_all_direct-shared | #2034 | direct-shared | tenferro-rs | f64 | case_id=real.reduce_max_all, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [8192, 4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | FAILED |
| followup_real_reduce_max_all_julia-base | #2034 | julia-base | julia-base | f64 | case_id=real.reduce_max_all, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [8192, 4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | FAILED |
| followup_real_reduce_min_all_direct-shared | #2034 | direct-shared | tenferro-rs | f64 | case_id=real.reduce_min_all, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [8192, 4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | FAILED |
| followup_real_reduce_min_all_julia-base | #2034 | julia-base | julia-base | f64 | case_id=real.reduce_min_all, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [8192, 4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | FAILED |
| followup_complex_norm_fro_direct-shared | #2035 | direct-shared | tenferro-rs | c64 | case_id=complex.norm_fro, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [2048, 1536], 'dtype': 'c64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | FAILED |
| followup_complex_norm_fro_jax | #2035 | jax | jax-cpu | c64 | case_id=complex.norm_fro, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [2048, 1536], 'dtype': 'c64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | FAILED |
| followup_index_gather_direct-shared | #2036 #1719 | direct-shared | tenferro-rs | f64 | case_id=index.gather, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [262144], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | FAILED |
| followup_index_gather_jax | #2036 #1719 | jax | jax-cpu | f64 | case_id=index.gather, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [262144], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | FAILED |
| followup_index_scatter_direct-shared | #2036 #1719 | direct-shared | tenferro-rs | f64 | case_id=index.scatter, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [262144], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | FAILED |
| followup_index_scatter_pytorch | #2036 #1719 | pytorch | pytorch-cpu | f64 | case_id=index.scatter, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [262144], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | FAILED |
| followup_index_dynamic_update_slice_direct-shared | #2036 #1719 | direct-shared | tenferro-rs | f64 | case_id=index.dynamic_update_slice, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [2097152], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | FAILED |
| followup_index_dynamic_update_slice_jax | #2036 #1719 | jax | jax-cpu | f64 | case_id=index.dynamic_update_slice, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [2097152], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | FAILED |
| followup_perm_reverse_15d_3_materialize | #2037 #2228 | materialize | tenferro-rs | f64 | case_id=perm.reverse_15d_3, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | FAILED |
| followup_perm_reverse_15d_3_strided-rs | #2037 #2228 | strided-rs | strided-rs | f64 | case_id=perm.reverse_15d_3, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | FAILED |
| followup_structural_transpose_direct-shared | #2037 #2228 | direct-shared | tenferro-rs | f64 | case_id=structural.transpose, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [4096, 4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | FAILED |
| followup_structural_transpose_pytorch | #2037 #2228 | pytorch | pytorch-cpu | f64 | case_id=structural.transpose, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [4096, 4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | FAILED |
| followup_perm_reverse_23d_2_materialize | #2037 #2228 | materialize | tenferro-rs | f64 | case_id=perm.reverse_23d_2, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | FAILED |
| followup_perm_reverse_23d_2_strided-rs | #2037 #2228 | strided-rs | strided-rs | f64 | case_id=perm.reverse_23d_2, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | FAILED |
| followup_perm_transpose_3d_256_201_materialize | #2037 #2228 | materialize | tenferro-rs | f64 | case_id=perm.transpose_3d_256_201, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [256, 256, 256], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | FAILED |
| followup_perm_transpose_3d_256_201_strided-rs | #2037 #2228 | strided-rs | strided-rs | f64 | case_id=perm.transpose_3d_256_201, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [256, 256, 256], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | FAILED |
| followup_perm_cyclic_15d_3_materialize | #2037 #2228 | materialize | tenferro-rs | f64 | case_id=perm.cyclic_15d_3, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | FAILED |
| followup_perm_cyclic_15d_3_strided-rs | #2037 #2228 | strided-rs | strided-rs | f64 | case_id=perm.cyclic_15d_3, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | FAILED |
| followup_perm_transpose_3d_256_102_materialize | #2037 #2228 | materialize | tenferro-rs | f64 | case_id=perm.transpose_3d_256_102, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [256, 256, 256], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | FAILED |
| followup_perm_transpose_3d_256_102_strided-rs | #2037 #2228 | strided-rs | strided-rs | f64 | case_id=perm.transpose_3d_256_102, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [256, 256, 256], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | FAILED |
| followup_perm_tn_light_415_24d_scattered_to_colmajor_materialize | #2037 #2228 | materialize | tenferro-rs | f64 | case_id=perm.tn_light_415_24d_scattered_to_colmajor, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | FAILED |
| followup_perm_tn_light_415_24d_scattered_to_colmajor_strided-rs | #2037 #2228 | strided-rs | strided-rs | f64 | case_id=perm.tn_light_415_24d_scattered_to_colmajor, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | FAILED |
| followup_perm_tn_light_415_24d_contiguous_same_perm_materialize | #2037 #2228 | materialize | tenferro-rs | f64 | case_id=perm.tn_light_415_24d_contiguous_same_perm, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | FAILED |
| followup_perm_tn_light_415_24d_contiguous_same_perm_strided-rs | #2037 #2228 | strided-rs | strided-rs | f64 | case_id=perm.tn_light_415_24d_contiguous_same_perm, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | FAILED |
| followup_perm_rotation_6d_32_32_32_32_16_16_materialize | #2037 #2228 | materialize | tenferro-rs | f64 | case_id=perm.rotation_6d_32_32_32_32_16_16, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [32, 32, 32, 32, 16, 16], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | FAILED |
| followup_perm_rotation_6d_32_32_32_32_16_16_strided-rs | #2037 #2228 | strided-rs | strided-rs | f64 | case_id=perm.rotation_6d_32_32_32_32_16_16, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [32, 32, 32, 32, 16, 16], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | FAILED |
| followup_structural_extract_diagonal_direct-shared | #2038 #2228 | direct-shared | tenferro-rs | f64 | case_id=structural.extract_diagonal, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [8388608, 2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | FAILED |
| followup_structural_extract_diagonal_jax | #2038 #2228 | jax | jax-cpu | f64 | case_id=structural.extract_diagonal, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [8388608, 2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | FAILED |
| followup_fft_irfft_direct-shared | #2039 #2230 #2228 | direct-shared | tenferro-rs | c32 | case_id=fft.irfft, rust_path=direct-shared, reference_path=mkl-dfti, fixture={'input_shape': [524289], 'dtype': 'c32', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | FAILED |
| followup_fft_irfft_mkl-dfti | #2039 #2230 #2228 | mkl-dfti | mkl-dfti | c32 | case_id=fft.irfft, rust_path=direct-shared, reference_path=mkl-dfti, fixture={'input_shape': [524289], 'dtype': 'c32', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | FAILED |
| followup_fft_rfft_direct-shared | #2039 #2230 #2228 | direct-shared | tenferro-rs | f32 | case_id=fft.rfft, rust_path=direct-shared, reference_path=mkl-dfti, fixture={'input_shape': [1048576], 'dtype': 'f32', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | FAILED |
| followup_fft_rfft_mkl-dfti | #2039 #2230 #2228 | mkl-dfti | mkl-dfti | f32 | case_id=fft.rfft, rust_path=direct-shared, reference_path=mkl-dfti, fixture={'input_shape': [1048576], 'dtype': 'f32', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | FAILED |
| followup_fft_fft_direct-shared | #2039 #2228 | direct-shared | tenferro-rs | c64 | case_id=fft.fft, rust_path=direct-shared, reference_path=mkl-dfti, fixture={'input_shape': [1048576], 'dtype': 'c64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | FAILED |
| followup_fft_fft_mkl-dfti | #2039 #2228 | mkl-dfti | mkl-dfti | c64 | case_id=fft.fft, rust_path=direct-shared, reference_path=mkl-dfti, fixture={'input_shape': [1048576], 'dtype': 'c64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | FAILED |
| followup_fft_ifft_direct-shared | #2039 #2228 | direct-shared | tenferro-rs | c32 | case_id=fft.ifft, rust_path=direct-shared, reference_path=mkl-dfti, fixture={'input_shape': [1048576], 'dtype': 'c32', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | FAILED |
| followup_fft_ifft_mkl-dfti | #2039 #2228 | mkl-dfti | mkl-dfti | c32 | case_id=fft.ifft, rust_path=direct-shared, reference_path=mkl-dfti, fixture={'input_shape': [1048576], 'dtype': 'c32', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | — | — | — | — | FAILED |
| followup_metadata_transpose_view_metadata | #2040 #1719 | metadata | tenferro-rs | f64 | case_id=metadata.transpose_view, rust_path=metadata, reference_path=julia-base, fixture={'input_shape': [2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 4 | — | — | — | — | FAILED |
| followup_metadata_transpose_view_julia-base | #2040 #1719 | julia-base | julia-base | f64 | case_id=metadata.transpose_view, rust_path=metadata, reference_path=julia-base, fixture={'input_shape': [2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 4 | — | — | — | — | FAILED |
| followup_metadata_transpose_view_metadata-host-dyn | #2040 #1719 | metadata-host-dyn | tenferro-rs | f64 | case_id=metadata.transpose_view, rust_path=metadata-host-dyn, reference_path=julia-base, fixture={'input_shape': [2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 4 | — | — | — | — | FAILED |
| followup_metadata_transpose_view_metadata-host-static | #2040 #1719 | metadata-host-static | tenferro-rs | f64 | case_id=metadata.transpose_view, rust_path=metadata-host-static, reference_path=julia-base, fixture={'input_shape': [2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 4 | — | — | — | — | FAILED |
| followup_metadata_slice_view_metadata | #2040 #1719 | metadata | tenferro-rs | f64 | case_id=metadata.slice_view, rust_path=metadata, reference_path=julia-base, fixture={'input_shape': [4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 4 | — | — | — | — | FAILED |
| followup_metadata_slice_view_julia-base | #2040 #1719 | julia-base | julia-base | f64 | case_id=metadata.slice_view, rust_path=metadata, reference_path=julia-base, fixture={'input_shape': [4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 4 | — | — | — | — | FAILED |
| followup_metadata_slice_view_metadata-host-dyn | #2040 #1719 | metadata-host-dyn | tenferro-rs | f64 | case_id=metadata.slice_view, rust_path=metadata-host-dyn, reference_path=julia-base, fixture={'input_shape': [4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 4 | — | — | — | — | FAILED |
| followup_metadata_slice_view_metadata-host-static | #2040 #1719 | metadata-host-static | tenferro-rs | f64 | case_id=metadata.slice_view, rust_path=metadata-host-static, reference_path=julia-base, fixture={'input_shape': [4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 4 | — | — | — | — | FAILED |
| followup_metadata_reshape_view_metadata | #2040 #1719 | metadata | tenferro-rs | f64 | case_id=metadata.reshape_view, rust_path=metadata, reference_path=julia-base, fixture={'input_shape': [1024], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 4 | — | — | — | — | FAILED |
| followup_metadata_reshape_view_julia-base | #2040 #1719 | julia-base | julia-base | f64 | case_id=metadata.reshape_view, rust_path=metadata, reference_path=julia-base, fixture={'input_shape': [1024], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 4 | — | — | — | — | FAILED |
| followup_metadata_reshape_view_metadata-host-dyn | #2040 #1719 | metadata-host-dyn | tenferro-rs | f64 | case_id=metadata.reshape_view, rust_path=metadata-host-dyn, reference_path=julia-base, fixture={'input_shape': [1024], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 4 | — | — | — | — | FAILED |
| followup_metadata_reshape_view_metadata-host-static | #2040 #1719 | metadata-host-static | tenferro-rs | f64 | case_id=metadata.reshape_view, rust_path=metadata-host-static, reference_path=julia-base, fixture={'input_shape': [1024], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 4 | — | — | — | — | FAILED |
| lu_batched_f64_factor_n8_b1024 | #2142 | tenferro | tenferro-rs | f64 | n=8, batch=1024, op=factor | 4 | 8 | 179929.0 | 2699.3 | 4.1% | ok |
| lu_batched_f64_factor_n8_b1024_host | #2142 | host | rust-loop | f64 | n=8, batch=1024, op=factor | 4 | 4 | 533464.2 | 5961.1 | 0.7% | ok |
| lu_batched_f64_solve_n8_b1024 | #2142 | tenferro | tenferro-rs | f64 | n=8, batch=1024, op=solve | 4 | 32 | 93641.1 | 394.7 | 0.9% | ok |
| lu_batched_f64_solve_n8_b1024_pytorch | #2142 | pytorch | pytorch-cpu | f64 | n=8, batch=1024, op=solve | 4 | 16 | 229657.4 | 784.6 | 0.3% | ok |
| lu_batched_f64_factor_n16_b1024 | #2142 | tenferro | tenferro-rs | f64 | n=16, batch=1024, op=factor | 4 | 4 | 474274.0 | 12714.9 | 2.2% | ok |
| lu_batched_f64_factor_n16_b1024_host | #2142 | host | rust-loop | f64 | n=16, batch=1024, op=factor | 4 | 2 | 2107757.5 | 20142.5 | 0.8% | ok |
| lu_batched_f64_solve_n16_b1024 | #2142 | tenferro | tenferro-rs | f64 | n=16, batch=1024, op=solve | 4 | 8 | 242941.5 | 1167.2 | 0.5% | ok |
| lu_batched_f64_solve_n16_b1024_pytorch | #2142 | pytorch | pytorch-cpu | f64 | n=16, batch=1024, op=solve | 4 | 8 | 498803.4 | 1027.6 | 0.2% | ok |
| small_grad_qr_f64_trace-prepared_n4 | #2186 | trace-prepared | tenferro-rs | f64 | op=qr, n=4 | 4 | 64 | 53756.9 | 144.1 | 0.4% | ok |
| small_grad_qr_f64_pytorch_n4 | #2186 | pytorch | pytorch-cpu | f64 | op=qr, n=4 | 4 | 32 | 75350.8 | 254.7 | 0.5% | ok |
| small_grad_qr_f64_trace-prepared_n16 | #2186 | trace-prepared | tenferro-rs | f64 | op=qr, n=16 | 4 | 32 | 74321.3 | 340.5 | 13.0% | NOISY |
| small_grad_qr_f64_pytorch_n16 | #2186 | pytorch | pytorch-cpu | f64 | op=qr, n=16 | 4 | 32 | 88076.6 | 505.5 | 4.5% | ok |
| small_grad_eigh_f64_trace-prepared_n4 | #2186 | trace-prepared | tenferro-rs | f64 | op=eigh, n=4 | 4 | 32 | 74133.5 | 1492.0 | 3.5% | ok |
| small_grad_eigh_f64_pytorch_n4 | #2186 | pytorch | pytorch-cpu | f64 | op=eigh, n=4 | 4 | 32 | 74755.9 | 535.5 | 0.9% | ok |
| small_grad_eigh_f64_trace-prepared_n16 | #2186 | trace-prepared | tenferro-rs | f64 | op=eigh, n=16 | 4 | 16 | 132610.2 | 1230.4 | 1.2% | ok |
| small_grad_eigh_f64_pytorch_n16 | #2186 | pytorch | pytorch-cpu | f64 | op=eigh, n=16 | 4 | 32 | 101132.8 | 1038.5 | 0.8% | ok |
| small_grad_lu_f64_trace-prepared_n4 | #2186 | trace-prepared | tenferro-rs | f64 | op=lu, n=4 | 4 | 64 | 40037.3 | 206.3 | 3.7% | ok |
| small_grad_lu_f64_pytorch_n4 | #2186 | pytorch | pytorch-cpu | f64 | op=lu, n=4 | 4 | 32 | 86078.8 | 1412.2 | 1.5% | ok |
| small_grad_lu_f64_trace-prepared_n16 | #2186 | trace-prepared | tenferro-rs | f64 | op=lu, n=16 | 4 | 64 | 47897.5 | 154.1 | 0.4% | ok |
| small_grad_lu_f64_pytorch_n16 | #2186 | pytorch | pytorch-cpu | f64 | op=lu, n=16 | 4 | 32 | 94562.2 | 456.8 | 1.0% | ok |
| small_grad_svd_f64_trace-prepared_n4 | #2186 | trace-prepared | tenferro-rs | f64 | op=svd, n=4 | 4 | 32 | 121067.8 | 531.3 | 0.5% | ok |
| small_grad_svd_f64_pytorch_n4 | #2186 | pytorch | pytorch-cpu | f64 | op=svd, n=4 | 4 | 32 | 113508.1 | 756.1 | 0.8% | ok |
| small_grad_svd_f64_trace-prepared_n16 | #2186 | trace-prepared | tenferro-rs | f64 | op=svd, n=16 | 4 | 16 | 224016.2 | 1067.3 | 0.6% | ok |
| small_grad_svd_f64_pytorch_n16 | #2186 | pytorch | pytorch-cpu | f64 | op=svd, n=16 | 4 | 16 | 172723.1 | 526.0 | 0.5% | ok |
| small_grad_solve_f64_trace-prepared_n4 | #2186 | trace-prepared | tenferro-rs | f64 | op=solve, n=4 | 4 | 64 | 33202.3 | 371.4 | 0.6% | ok |
| small_grad_solve_f64_pytorch_n4 | #2186 | pytorch | pytorch-cpu | f64 | op=solve, n=4 | 4 | 32 | 62358.4 | 165.8 | 0.3% | ok |
| small_grad_solve_f64_trace-prepared_n16 | #2186 | trace-prepared | tenferro-rs | f64 | op=solve, n=16 | 4 | 64 | 40825.5 | 177.4 | 0.3% | ok |
| small_grad_solve_f64_pytorch_n16 | #2186 | pytorch | pytorch-cpu | f64 | op=solve, n=16 | 4 | 32 | 71755.0 | 234.3 | 0.4% | ok |
| fft_axis0_c64_fft-executor_n128 | #1765 | fft-executor | tenferro-rs | c64 | n=128 | 4 | 1 | 4548847.0 | 105276.0 | 1.2% | ok |
| fft_axis0_c64_pytorch_n128 | #1765 | pytorch | pytorch-cpu | c64 | n=128 | 4 | 1 | 6916910.0 | 286204.0 | 3.1% | ok |
| degenerate_dot_hadamard_f64_dot-general-shared_n1024 | #2111 | dot-general-shared | tenferro-rs | f64 | form=hadamard, n=1024 | 4 | 4 | 328271.8 | 4040.0 | 1.7% | ok |
| degenerate_dot_hadamard_f64_pytorch_n1024 | #2111 | pytorch | pytorch-cpu | f64 | form=hadamard, n=1024 | 4 | 4 | 325564.0 | 15485.0 | 4.5% | ok |
| degenerate_dot_outer_f64_dot-general-shared_n1024 | #2111 | dot-general-shared | tenferro-rs | f64 | form=outer, n=1024 | 4 | 4 | 233966.2 | 7276.0 | 2.2% | ok |
| degenerate_dot_outer_f64_pytorch_n1024 | #2111 | pytorch | pytorch-cpu | f64 | form=outer, n=1024 | 4 | 4 | 163742.5 | 9293.6 | 14.6% | NOISY |
| degenerate_dot_rank0_f64_dot-general-shared_n1024 | #2111 | dot-general-shared | tenferro-rs | f64 | form=rank0, n=1024 | 4 | 1 | 9223027.0 | 175171.5 | 1.6% | ok |
| degenerate_dot_rank0_f64_pytorch_n1024 | #2111 | pytorch | pytorch-cpu | f64 | form=rank0, n=1024 | 4 | 8 | 620972.2 | 39014.7 | 4.1% | ok |
| erased_add_f64_session-add_n256 | #1766 | session-add | tenferro-rs | f64 | n=256 | 4 | 1024 | 1596.5 | 29.5 | 3.0% | ok |
| erased_add_f64_pytorch_n256 | #1766 | pytorch | pytorch-cpu | f64 | n=256 | 4 | 1024 | 1929.0 | 5.5 | 0.5% | ok |
| erased_add_f64_session-add_n4096 | #1766 | session-add | tenferro-rs | f64 | n=4096 | 4 | 256 | 3043.6 | 39.8 | 1.3% | ok |
| erased_add_f64_pytorch_n4096 | #1766 | pytorch | pytorch-cpu | f64 | n=4096 | 4 | 1024 | 3726.9 | 36.1 | 6.0% | ok |
| erased_add_f64_session-add_n1048576 | #1766 | session-add | tenferro-rs | f64 | n=1048576 | 4 | 4 | 329894.8 | 7423.9 | 2.7% | ok |
| erased_add_f64_pytorch_n1048576 | #1766 | pytorch | pytorch-cpu | f64 | n=1048576 | 4 | 4 | 335565.2 | 7534.2 | 4.3% | ok |
| axpby_strided_source_f64_1024x1024_t4 | #2220 #2214 | strided-source | tenferro-rs | f64 | n=1024 | 4 | 4 | 899543.8 | 55336.8 | 7.1% | ok |
| axpby_contiguous_source_f64_1024x1024_t4 | #2220 #2214 | contiguous-source | tenferro-rs | f64 | n=1024 | 4 | 8 | 314633.9 | 7837.8 | 3.2% | ok |
| axpby_strided_source_f64_1024x1024_host_t4 | #2220 #2214 | host | rust-loop | f64 | n=1024 | 4 | 1 | 2302911.0 | 53820.0 | 1.5% | ok |
| erased_construct_f64_col_n8 | #2177 | col-major | tenferro-rs | f64 | n=8, layout=col | 4 | 16384 | 132.5 | 0.6 | 13.8% | NOISY |
| erased_construct_f64_row_n8 | #2177 | row-major | tenferro-rs | f64 | n=8, layout=row | 4 | 8192 | 269.0 | 1.5 | 7.0% | ok |
| erased_construct_f64_col_n4096 | #2177 | col-major | tenferro-rs | f64 | n=4096, layout=col | 4 | 16384 | 221.9 | 0.4 | 1.1% | ok |
| erased_construct_f64_row_n4096 | #2177 | row-major | tenferro-rs | f64 | n=4096, layout=row | 4 | 128 | 19067.8 | 1302.0 | 4.0% | ok |
| erased_construct_f64_col_n65536 | #2177 | col-major | tenferro-rs | f64 | n=65536, layout=col | 4 | 1024 | 157.0 | 1.9 | 1.6% | ok |
| erased_construct_f64_row_n65536 | #2177 | row-major | tenferro-rs | f64 | n=65536, layout=row | 4 | 8 | 342909.0 | 61447.1 | 9.0% | ok |
| eager_small_leaf_f64_eager-held-session | #1704 | eager-held-session | tenferro-rs | f64 | op=leaf | 4 | 1024 | 3405.5 | 64.8 | 2.0% | ok |
| eager_small_add_f64_eager-held-session | #1667 | eager-held-session | tenferro-rs | f64 | op=add | 4 | 1024 | 2836.8 | 5.1 | 0.4% | ok |
| eager_small_add_f64_pytorch | #1667 | pytorch | pytorch-cpu | f64 | op=add | 4 | 2048 | 1820.1 | 10.5 | 0.8% | ok |
| eager_small_gemm_f64_eager-held-session | #1904 | eager-held-session | tenferro-rs | f64 | op=gemm | 4 | 512 | 4214.7 | 11.5 | 0.6% | ok |
| eager_small_gemm_f64_pytorch | #1904 | pytorch | pytorch-cpu | f64 | op=gemm | 4 | 1024 | 2040.1 | 5.9 | 0.3% | ok |
| trace_metadata_trace-fold_f32_1024x64_k50 | #2226 #2238 | trace-fold | tenferro-rs | f32 | rows=1024, cols=64, steps=50 | 4 | 2 | 1775043.0 | 5805.5 | 0.3% | ok |
| trace_metadata_trace-outputs_f32_1024x64_k50 | #2226 #2238 | trace-outputs | tenferro-rs | f32 | rows=1024, cols=64, steps=50 | 4 | 2 | 1677486.5 | 23273.5 | 1.1% | ok |
| trace_metadata_eager_f32_1024x64_k50 | #2226 #2238 | eager | tenferro-rs | f32 | rows=1024, cols=64, steps=50 | 4 | 1 | 2500129.0 | 7954.5 | 1.0% | ok |
| trace_metadata_pytorch_f32_1024x64_k50 | #2226 #2238 | pytorch | pytorch-cpu | f32 | rows=1024, cols=64, steps=50 | 4 | 2 | 1207886.0 | 3161.2 | 0.2% | ok |
| trace_metadata_fixed_neg_f32_8 | #2226 #2238 | trace-fold | tenferro-rs | f32 | rows=8, cols=1, steps=0 | 4 | 512 | 3055.3 | 31.4 | 0.7% | ok |
| decode_proj_f32_eager-shared_in1024_out1024_len8 | #1992 #2003 #1995 #2229 | eager-shared | tenferro-rs | f32 | in=1024, out=1024, len=8 | 8 | 32 | 87360.6 | 585.0 | 0.6% | ok |
| decode_proj_f32_faer-direct_in1024_out1024_len8 | #1992 #2003 #1995 #2229 | faer-direct | faer-direct | f32 | in=1024, out=1024, len=8 | 8 | 16 | 195886.2 | 3607.9 | 2.0% | ok |
| decode_proj_f32_host-sgemm_in1024_out1024_len8 | #1992 #2003 #1995 #2229 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=1024, len=8 | 8 | 2 | 1206127.5 | 12025.0 | 1.9% | ok |
| decode_proj_f32_pytorch_in1024_out1024_len8 | #1992 #2003 #1995 #2229 | pytorch | pytorch-cpu | f32 | in=1024, out=1024, len=8 | 8 | 32 | 98975.7 | 562.0 | 0.9% | ok |
| decode_proj_f32_eager-shared_in1024_out1024_len64 | #1992 #2003 #1995 #2229 | eager-shared | tenferro-rs | f32 | in=1024, out=1024, len=64 | 8 | 8 | 315034.5 | 1189.7 | 0.4% | ok |
| decode_proj_f32_faer-direct_in1024_out1024_len64 | #1992 #2003 #1995 #2229 | faer-direct | faer-direct | f32 | in=1024, out=1024, len=64 | 8 | 4 | 752595.0 | 14350.5 | 1.9% | ok |
| decode_proj_f32_host-sgemm_in1024_out1024_len64 | #1992 #2003 #1995 #2229 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=1024, len=64 | 8 | 1 | 2056412.0 | 18790.5 | 0.5% | ok |
| decode_proj_f32_pytorch_in1024_out1024_len64 | #1992 #2003 #1995 #2229 | pytorch | pytorch-cpu | f32 | in=1024, out=1024, len=64 | 8 | 8 | 384509.1 | 1527.2 | 0.4% | ok |
| decode_proj_f32_eager-shared_in1024_out1024_len512 | #1992 #2003 #1995 #2229 | eager-shared | tenferro-rs | f32 | in=1024, out=1024, len=512 | 8 | 1 | 2385185.0 | 4077.5 | 0.2% | ok |
| decode_proj_f32_faer-direct_in1024_out1024_len512 | #1992 #2003 #1995 #2229 | faer-direct | faer-direct | f32 | in=1024, out=1024, len=512 | 8 | 1 | 2634129.0 | 91084.5 | 2.2% | ok |
| decode_proj_f32_host-sgemm_in1024_out1024_len512 | #1992 #2003 #1995 #2229 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=1024, len=512 | 8 | 1 | 5547728.0 | 44753.0 | 2.6% | ok |
| decode_proj_f32_pytorch_in1024_out1024_len512 | #1992 #2003 #1995 #2229 | pytorch | pytorch-cpu | f32 | in=1024, out=1024, len=512 | 8 | 1 | 2649046.0 | 28272.0 | 1.3% | ok |
| decode_proj_f32_eager-shared_in1024_out4096_len8 | #1992 #2003 #1995 #2229 | eager-shared | tenferro-rs | f32 | in=1024, out=4096, len=8 | 8 | 8 | 367815.5 | 3788.9 | 4.6% | ok |
| decode_proj_f32_faer-direct_in1024_out4096_len8 | #1992 #2003 #1995 #2229 | faer-direct | faer-direct | f32 | in=1024, out=4096, len=8 | 8 | 8 | 475104.1 | 14631.0 | 4.1% | ok |
| decode_proj_f32_host-sgemm_in1024_out4096_len8 | #1992 #2003 #1995 #2229 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=4096, len=8 | 8 | 1 | 5045513.0 | 99165.0 | 1.6% | ok |
| decode_proj_f32_pytorch_in1024_out4096_len8 | #1992 #2003 #1995 #2229 | pytorch | pytorch-cpu | f32 | in=1024, out=4096, len=8 | 8 | 4 | 773519.0 | 10171.4 | 1.0% | ok |
| decode_proj_f32_eager-shared_in1024_out4096_len64 | #1992 #2003 #1995 #2229 | eager-shared | tenferro-rs | f32 | in=1024, out=4096, len=64 | 8 | 2 | 1186766.5 | 4751.5 | 0.9% | ok |
| decode_proj_f32_faer-direct_in1024_out4096_len64 | #1992 #2003 #1995 #2229 | faer-direct | faer-direct | f32 | in=1024, out=4096, len=64 | 8 | 2 | 1615511.0 | 34118.2 | 1.8% | ok |
| decode_proj_f32_host-sgemm_in1024_out4096_len64 | #1992 #2003 #1995 #2229 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=4096, len=64 | 8 | 1 | 8499109.0 | 51075.0 | 0.4% | ok |
| decode_proj_f32_pytorch_in1024_out4096_len64 | #1992 #2003 #1995 #2229 | pytorch | pytorch-cpu | f32 | in=1024, out=4096, len=64 | 8 | 2 | 2078252.5 | 69269.2 | 2.9% | ok |
| decode_proj_f32_eager-shared_in1024_out4096_len512 | #1992 #2003 #1995 #2229 | eager-shared | tenferro-rs | f32 | in=1024, out=4096, len=512 | 8 | 1 | 8933389.0 | 13264.5 | 0.5% | ok |
| decode_proj_f32_faer-direct_in1024_out4096_len512 | #1992 #2003 #1995 #2229 | faer-direct | faer-direct | f32 | in=1024, out=4096, len=512 | 8 | 1 | 9265247.0 | 34218.5 | 0.4% | ok |
| decode_proj_f32_host-sgemm_in1024_out4096_len512 | #1992 #2003 #1995 #2229 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=4096, len=512 | 8 | 1 | 22700756.0 | 96253.5 | 0.4% | ok |
| decode_proj_f32_pytorch_in1024_out4096_len512 | #1992 #2003 #1995 #2229 | pytorch | pytorch-cpu | f32 | in=1024, out=4096, len=512 | 8 | 1 | 12822215.0 | 157128.5 | 1.4% | ok |
| gemm_c64_mm256_t1 | #1900 | dot-general-shared | tenferro-rs | c64 | n=256 | 8 | 1 | 4441988.0 | 29881.0 | 0.4% | ok |
| gemm_f64_mm256_t1 | #1900 | dot-general-shared | tenferro-rs | f64 | n=256 | 8 | 2 | 1071016.0 | 2785.0 | 0.5% | ok |
| conj_dot_f64_dot-general-rank1_len100 | #1615 | dot-general-rank1 | tenferro-rs | f64 | len=100 | 8 | 1024 | 2289.0 | 9.4 | 1.5% | ok |
| conj_dot_f64_portable-loop_len100 | #1615 | portable-loop | rust-loop | f64 | len=100 | 8 | 32768 | 89.6 | 0.4 | 0.5% | ok |
| conj_dot_f64_pytorch-vdot_len100 | #1615 | pytorch-vdot | pytorch-cpu | f64 | len=100 | 8 | 2048 | 1684.5 | 15.5 | 1.8% | ok |
| conj_dot_f64_dot-general-rank1_len10000 | #1615 | dot-general-rank1 | tenferro-rs | f64 | len=10000 | 8 | 512 | 4910.7 | 19.9 | 0.7% | ok |
| conj_dot_f64_portable-loop_len10000 | #1615 | portable-loop | rust-loop | f64 | len=10000 | 8 | 256 | 15003.5 | 40.7 | 0.2% | ok |
| conj_dot_f64_pytorch-vdot_len10000 | #1615 | pytorch-vdot | pytorch-cpu | f64 | len=10000 | 8 | 1024 | 3368.9 | 33.1 | 0.7% | ok |
| conj_dot_f64_dot-general-rank1_len1000000 | #1615 | dot-general-rank1 | tenferro-rs | f64 | len=1000000 | 8 | 32 | 66337.1 | 1174.4 | 4.9% | ok |
| conj_dot_f64_portable-loop_len1000000 | #1615 | portable-loop | rust-loop | f64 | len=1000000 | 8 | 2 | 1514257.0 | 5022.0 | 0.5% | ok |
| conj_dot_f64_pytorch-vdot_len1000000 | #1615 | pytorch-vdot | pytorch-cpu | f64 | len=1000000 | 8 | 32 | 65200.2 | 1198.6 | 11.6% | NOISY |
| conj_dot_c64_dot-general-rank1_len100 | #1615 | dot-general-rank1 | tenferro-rs | c64 | len=100 | 8 | 1024 | 2327.4 | 14.2 | 1.8% | ok |
| conj_dot_c64_portable-loop_len100 | #1615 | portable-loop | rust-loop | c64 | len=100 | 8 | 16384 | 144.6 | 0.7 | 0.7% | ok |
| conj_dot_c64_pytorch-vdot_len100 | #1615 | pytorch-vdot | pytorch-cpu | c64 | len=100 | 8 | 2048 | 1587.1 | 8.9 | 236.9% | NOISY |
| conj_dot_c64_dot-general-rank1_len10000 | #1615 | dot-general-rank1 | tenferro-rs | c64 | len=10000 | 8 | 128 | 16233.1 | 52.2 | 0.2% | ok |
| conj_dot_c64_portable-loop_len10000 | #1615 | portable-loop | rust-loop | c64 | len=10000 | 8 | 256 | 15296.9 | 39.0 | 0.2% | ok |
| conj_dot_c64_pytorch-vdot_len10000 | #1615 | pytorch-vdot | pytorch-cpu | c64 | len=10000 | 8 | 512 | 4019.4 | 9.3 | 0.3% | ok |
| conj_dot_c64_dot-general-rank1_len1000000 | #1615 | dot-general-rank1 | tenferro-rs | c64 | len=1000000 | 8 | 4 | 516031.8 | 22115.0 | 4.2% | ok |
| conj_dot_c64_portable-loop_len1000000 | #1615 | portable-loop | rust-loop | c64 | len=1000000 | 8 | 2 | 1544558.5 | 2449.8 | 0.3% | ok |
| conj_dot_c64_pytorch-vdot_len1000000 | #1615 | pytorch-vdot | pytorch-cpu | c64 | len=1000000 | 8 | 4 | 506531.2 | 12721.2 | 2.5% | ok |
| small_solve_f64_solve-loop_k16_b64 | #2007 #1878 | solve-loop | tenferro-rs | f64 | k=16, batch=64 | 8 | 8 | 368846.1 | 1949.4 | 0.8% | ok |
| small_solve_f64_triangular-solve-loop_k16_b64 | #2007 #1878 | triangular-solve-loop | tenferro-rs | f64 | k=16, batch=64 | 8 | 16 | 206554.3 | 1151.8 | 0.4% | ok |
| small_solve_f64_solve-batched_k16_b64 | #2007 #1878 | solve-batched | tenferro-rs | f64 | k=16, batch=64 | 8 | 64 | 31682.2 | 1254.2 | 7.5% | ok |
| small_solve_f64_pytorch-solve-batched_k16_b64 | #2007 #1878 | pytorch-solve-batched | pytorch-cpu | f64 | k=16, batch=64 | 8 | 16 | 174975.5 | 948.9 | 0.6% | ok |
| small_solve_f64_pytorch-solve-triangular-batched_k16_b64 | #2007 #1878 | pytorch-solve-triangular-batched | pytorch-cpu | f64 | k=16, batch=64 | 8 | 32 | 93841.4 | 202.9 | 10.4% | NOISY |
| small_solve_f64_solve-loop_k64_b64 | #2007 #1878 | solve-loop | tenferro-rs | f64 | k=64, batch=64 | 8 | 1 | 3570815.0 | 14141.5 | 0.5% | ok |
| small_solve_f64_triangular-solve-loop_k64_b64 | #2007 #1878 | triangular-solve-loop | tenferro-rs | f64 | k=64, batch=64 | 8 | 2 | 1717440.5 | 14168.8 | 0.6% | ok |
| small_solve_f64_solve-batched_k64_b64 | #2007 #1878 | solve-batched | tenferro-rs | f64 | k=64, batch=64 | 8 | 4 | 474263.8 | 3109.5 | 0.5% | ok |
| small_solve_f64_pytorch-solve-batched_k64_b64 | #2007 #1878 | pytorch-solve-batched | pytorch-cpu | f64 | k=64, batch=64 | 8 | 2 | 1606549.0 | 12498.2 | 0.5% | ok |
| small_solve_f64_pytorch-solve-triangular-batched_k64_b64 | #2007 #1878 | pytorch-solve-triangular-batched | pytorch-cpu | f64 | k=64, batch=64 | 8 | 8 | 448587.2 | 13957.2 | 4.2% | ok |
| tanh_chain_f32_compiled-prepared_k10_1024x64 | #1990 | compiled-prepared | tenferro-rs | f32 | rows=1024, cols=64, chain=10 | 8 | 4 | 226464.5 | 1311.1 | 0.4% | ok |
| tanh_chain_f32_eager-shared_k10_1024x64 | #1990 | eager-shared | tenferro-rs | f32 | rows=1024, cols=64, chain=10 | 8 | 8 | 269638.8 | 1192.8 | 0.4% | ok |
| tanh_chain_f32_compiled-prepared_k100_1024x64 | #1990 | compiled-prepared | tenferro-rs | f32 | rows=1024, cols=64, chain=100 | 8 | 1 | 2199419.0 | 11892.0 | 2.2% | ok |
| tanh_chain_f32_eager-shared_k100_1024x64 | #1990 | eager-shared | tenferro-rs | f32 | rows=1024, cols=64, chain=100 | 8 | 1 | 2693370.0 | 8566.0 | 0.5% | ok |
| tanh_chain_f32_compiled-prepared_k200_1024x64 | #1990 | compiled-prepared | tenferro-rs | f32 | rows=1024, cols=64, chain=200 | 8 | 1 | 4361899.0 | 22687.0 | 1.8% | ok |
| tanh_chain_f32_eager-shared_k200_1024x64 | #1990 | eager-shared | tenferro-rs | f32 | rows=1024, cols=64, chain=200 | 8 | 1 | 5363796.0 | 22737.0 | 1.3% | ok |
| layer_norm_f32_eager-composed_d1024_len8_b1 | #2006 | eager-composed | tenferro-rs | f32 | norm=layer_norm, d=1024, len=8, batch=1 | 8 | 32 | 68400.0 | 895.3 | 1.0% | ok |
| layer_norm_f32_pytorch-fused_d1024_len8_b1 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=layer_norm, d=1024, len=8, batch=1 | 8 | 512 | 7015.7 | 161.2 | 1.5% | ok |
| layer_norm_f32_eager-single-call_d1024_len8_b1 | #2006 | eager-single-call | tenferro-rs | f32 | norm=layer_norm, d=1024, len=8, batch=1 | 8 | 32 | 72311.0 | 518.3 | 0.7% | ok |
| layer_norm_f32_eager-composed_d1024_len8_b8 | #2006 | eager-composed | tenferro-rs | f32 | norm=layer_norm, d=1024, len=8, batch=8 | 8 | 16 | 120462.6 | 1108.9 | 0.6% | ok |
| layer_norm_f32_pytorch-fused_d1024_len8_b8 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=layer_norm, d=1024, len=8, batch=8 | 8 | 128 | 14906.5 | 1534.7 | 6.7% | ok |
| layer_norm_f32_eager-single-call_d1024_len8_b8 | #2006 | eager-single-call | tenferro-rs | f32 | norm=layer_norm, d=1024, len=8, batch=8 | 8 | 4 | 120046.2 | 2497.4 | 59.4% | NOISY |
| layer_norm_f32_eager-composed_d1024_len64_b1 | #2006 | eager-composed | tenferro-rs | f32 | norm=layer_norm, d=1024, len=64, batch=1 | 8 | 16 | 119442.6 | 1190.7 | 3.0% | ok |
| layer_norm_f32_pytorch-fused_d1024_len64_b1 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=layer_norm, d=1024, len=64, batch=1 | 8 | 256 | 27271.7 | 945.9 | 2.7% | ok |
| layer_norm_f32_eager-single-call_d1024_len64_b1 | #2006 | eager-single-call | tenferro-rs | f32 | norm=layer_norm, d=1024, len=64, batch=1 | 8 | 16 | 122421.2 | 1037.2 | 0.5% | ok |
| layer_norm_f32_eager-composed_d1024_len64_b8 | #2006 | eager-composed | tenferro-rs | f32 | norm=layer_norm, d=1024, len=64, batch=8 | 8 | 4 | 516465.0 | 24690.8 | 2.6% | ok |
| layer_norm_f32_pytorch-fused_d1024_len64_b8 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=layer_norm, d=1024, len=64, batch=8 | 8 | 32 | 157437.8 | 10876.4 | 4.6% | ok |
| layer_norm_f32_eager-single-call_d1024_len64_b8 | #2006 | eager-single-call | tenferro-rs | f32 | norm=layer_norm, d=1024, len=64, batch=8 | 8 | 4 | 435489.0 | 12828.9 | 2.0% | ok |
| rms_norm_f32_eager-composed_d1024_len8_b1 | #2006 | eager-composed | tenferro-rs | f32 | norm=rms_norm, d=1024, len=8, batch=1 | 8 | 64 | 36848.2 | 199.2 | 0.5% | ok |
| rms_norm_f32_pytorch-fused_d1024_len8_b1 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=rms_norm, d=1024, len=8, batch=1 | 8 | 128 | 18316.1 | 152.2 | 1.3% | ok |
| rms_norm_f32_eager-single-call_d1024_len8_b1 | #2006 | eager-single-call | tenferro-rs | f32 | norm=rms_norm, d=1024, len=8, batch=1 | 8 | 64 | 42205.7 | 232.7 | 0.6% | ok |
| rms_norm_f32_eager-composed_d1024_len8_b8 | #2006 | eager-composed | tenferro-rs | f32 | norm=rms_norm, d=1024, len=8, batch=8 | 8 | 32 | 61609.8 | 1815.1 | 1.9% | ok |
| rms_norm_f32_pytorch-fused_d1024_len8_b8 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=rms_norm, d=1024, len=8, batch=8 | 8 | 32 | 43039.0 | 503.1 | 1.0% | ok |
| rms_norm_f32_eager-single-call_d1024_len8_b8 | #2006 | eager-single-call | tenferro-rs | f32 | norm=rms_norm, d=1024, len=8, batch=8 | 8 | 8 | 65813.9 | 1758.2 | 2.4% | ok |
| rms_norm_f32_eager-composed_d1024_len64_b1 | #2006 | eager-composed | tenferro-rs | f32 | norm=rms_norm, d=1024, len=64, batch=1 | 8 | 32 | 61155.8 | 1732.3 | 1.6% | ok |
| rms_norm_f32_pytorch-fused_d1024_len64_b1 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=rms_norm, d=1024, len=64, batch=1 | 8 | 64 | 43064.8 | 403.7 | 1.6% | ok |
| rms_norm_f32_eager-single-call_d1024_len64_b1 | #2006 | eager-single-call | tenferro-rs | f32 | norm=rms_norm, d=1024, len=64, batch=1 | 8 | 8 | 64870.9 | 905.6 | 1.5% | ok |
| rms_norm_f32_eager-composed_d1024_len64_b8 | #2006 | eager-composed | tenferro-rs | f32 | norm=rms_norm, d=1024, len=64, batch=8 | 8 | 4 | 230494.8 | 10688.6 | 3.1% | ok |
| rms_norm_f32_pytorch-fused_d1024_len64_b8 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=rms_norm, d=1024, len=64, batch=8 | 8 | 8 | 173008.6 | 18141.9 | 6.0% | ok |
| rms_norm_f32_eager-single-call_d1024_len64_b8 | #2006 | eager-single-call | tenferro-rs | f32 | norm=rms_norm, d=1024, len=64, batch=8 | 8 | 4 | 227158.2 | 10649.8 | 2.8% | ok |
| small_contraction_abcd-dbef-acef_f64_shared-session-default-pool_d4 | #1885 #1897 #1901 | shared-session-default-pool | tenferro-rs | f64 | extent=4 | 8 | 512 | 4249.3 | 31.2 | 1.3% | ok |
| activation_erf_f32_eager-single-call_1024x64 | #1975 #2069 #2032 | eager-single-call | tenferro-rs | f32 | op=erf, rows=1024, cols=64 | 8 | — | — | — | — | FAILED |
| activation_erf_f32_session-single-call_1024x64 | #1975 #2069 #2032 | session-single-call | tenferro-rs | f32 | op=erf, rows=1024, cols=64 | 8 | 64 | 41253.1 | 291.3 | 0.6% | ok |
| activation_erf_f32_pytorch_1024x64 | #1975 #2069 #2032 | pytorch | pytorch-cpu | f32 | op=erf, rows=1024, cols=64 | 8 | — | — | — | — | FAILED |
| activation_sigmoid_f32_eager-single-call_1024x64 | #1975 #2069 #2032 | eager-single-call | tenferro-rs | f32 | op=sigmoid, rows=1024, cols=64 | 8 | — | — | — | — | FAILED |
| activation_sigmoid_f32_session-single-call_1024x64 | #1975 #2069 #2032 | session-single-call | tenferro-rs | f32 | op=sigmoid, rows=1024, cols=64 | 8 | 128 | 9702.2 | 272.2 | 2.0% | ok |
| activation_sigmoid_f32_eager-composed_1024x64 | #1975 #2069 #2032 | eager-composed | tenferro-rs | f32 | op=sigmoid, rows=1024, cols=64 | 8 | 64 | 34367.5 | 355.0 | 1.7% | ok |
| activation_sigmoid_f32_pytorch_1024x64 | #1975 #2069 #2032 | pytorch | pytorch-cpu | f32 | op=sigmoid, rows=1024, cols=64 | 8 | — | — | — | — | FAILED |
| activation_silu_f32_eager-single-call_1024x64 | #1975 #2069 #2032 | eager-single-call | tenferro-rs | f32 | op=silu, rows=1024, cols=64 | 8 | — | — | — | — | FAILED |
| activation_silu_f32_session-single-call_1024x64 | #1975 #2069 #2032 | session-single-call | tenferro-rs | f32 | op=silu, rows=1024, cols=64 | 8 | 128 | 9857.8 | 336.0 | 4.1% | ok |
| activation_silu_f32_eager-composed_1024x64 | #1975 #2069 #2032 | eager-composed | tenferro-rs | f32 | op=silu, rows=1024, cols=64 | 8 | 64 | 42293.5 | 183.9 | 1.0% | ok |
| activation_silu_f32_pytorch_1024x64 | #1975 #2069 #2032 | pytorch | pytorch-cpu | f32 | op=silu, rows=1024, cols=64 | 8 | — | — | — | — | FAILED |
| activation_softplus_f32_eager-single-call_1024x64 | #1975 #2069 #2032 | eager-single-call | tenferro-rs | f32 | op=softplus, rows=1024, cols=64 | 8 | — | — | — | — | FAILED |
| activation_softplus_f32_session-single-call_1024x64 | #1975 #2069 #2032 | session-single-call | tenferro-rs | f32 | op=softplus, rows=1024, cols=64 | 8 | 64 | 22579.9 | 614.9 | 6.0% | ok |
| activation_softplus_f32_eager-composed_1024x64 | #1975 #2069 #2032 | eager-composed | tenferro-rs | f32 | op=softplus, rows=1024, cols=64 | 8 | 32 | 55057.6 | 170.3 | 0.8% | ok |
| activation_softplus_f32_pytorch_1024x64 | #1975 #2069 #2032 | pytorch | pytorch-cpu | f32 | op=softplus, rows=1024, cols=64 | 8 | — | — | — | — | FAILED |
| activation_gelu_f32_eager-single-call_1024x64 | #1975 #2030 #2069 #2228 | eager-single-call | tenferro-rs | f32 | op=gelu, rows=1024, cols=64 | 8 | 128 | 11505.4 | 395.7 | 6.4% | ok |
| activation_gelu_f32_session-single-call_1024x64 | #1975 #2030 #2069 #2228 | session-single-call | tenferro-rs | f32 | op=gelu, rows=1024, cols=64 | 8 | 128 | 10868.6 | 122.5 | 1.1% | ok |
| activation_gelu_f32_pytorch_1024x64 | #1975 #2030 #2069 #2228 | pytorch | pytorch-cpu | f32 | op=gelu, rows=1024, cols=64 | 8 | 128 | 19077.3 | 1268.1 | 4.2% | ok |
| activation_gelu_tanh_f32_eager-single-call_1024x64 | #1975 #2069 #2032 | eager-single-call | tenferro-rs | f32 | op=gelu_tanh, rows=1024, cols=64 | 8 | — | — | — | — | FAILED |
| activation_gelu_tanh_f32_session-single-call_1024x64 | #1975 #2069 #2032 | session-single-call | tenferro-rs | f32 | op=gelu_tanh, rows=1024, cols=64 | 8 | 64 | 26354.3 | 258.8 | 1.0% | ok |
| activation_gelu_tanh_f32_eager-composed_1024x64 | #1975 #2069 #2032 | eager-composed | tenferro-rs | f32 | op=gelu_tanh, rows=1024, cols=64 | 8 | 16 | 115239.1 | 966.8 | 0.6% | ok |
| activation_gelu_tanh_f32_pytorch_1024x64 | #1975 #2069 #2032 | pytorch | pytorch-cpu | f32 | op=gelu_tanh, rows=1024, cols=64 | 8 | — | — | — | — | FAILED |
| softmax_f32_eager-single-call_len64_b8 | #1976 #2031 #2231 | eager-single-call | tenferro-rs | f32 | op=softmax, len=64, batch=8, axis=0 | 8 | 128 | 14969.4 | 194.7 | 5.0% | ok |
| softmax_f32_session-single-call_len64_b8 | #1976 #2031 #2231 | session-single-call | tenferro-rs | f32 | op=softmax, len=64, batch=8, axis=0 | 8 | 128 | 14256.1 | 98.3 | 0.7% | ok |
| softmax_f32_eager-composed_len64_b8 | #1976 #2031 #2231 | eager-composed | tenferro-rs | f32 | op=softmax, len=64, batch=8, axis=0 | 8 | 32 | 63480.5 | 1217.8 | 2.8% | ok |
| softmax_f32_pytorch_len64_b8 | #1976 #2031 #2231 | pytorch | pytorch-cpu | f32 | op=softmax, len=64, batch=8, axis=0 | 8 | 256 | 9295.4 | 233.4 | 1.6% | ok |
| softmax_f32_eager-single-call_len512_b8 | #1976 #2231 | eager-single-call | tenferro-rs | f32 | op=softmax, len=512, batch=8, axis=0 | 8 | 4 | 335462.8 | 86879.5 | 14.5% | NOISY |
| softmax_f32_session-single-call_len512_b8 | #1976 #2231 | session-single-call | tenferro-rs | f32 | op=softmax, len=512, batch=8, axis=0 | 8 | 4 | 311437.8 | 4056.5 | 17.1% | NOISY |
| softmax_f32_eager-composed_len512_b8 | #1976 #2231 | eager-composed | tenferro-rs | f32 | op=softmax, len=512, batch=8, axis=0 | 8 | 2 | 1676870.0 | 33312.5 | 3.9% | ok |
| softmax_f32_pytorch_len512_b8 | #1976 #2231 | pytorch | pytorch-cpu | f32 | op=softmax, len=512, batch=8, axis=0 | 8 | 4 | 357215.8 | 11694.4 | 3.0% | ok |
| log_softmax_f32_eager-single-call_len64_b8 | #1976 | eager-single-call | tenferro-rs | f32 | op=log_softmax, len=64, batch=8, axis=0 | 8 | 32 | 86480.2 | 597.8 | 1.8% | ok |
| log_softmax_f32_session-single-call_len64_b8 | #1976 | session-single-call | tenferro-rs | f32 | op=log_softmax, len=64, batch=8, axis=0 | 8 | 32 | 61688.1 | 558.2 | 3.9% | ok |
| log_softmax_f32_eager-composed_len64_b8 | #1976 | eager-composed | tenferro-rs | f32 | op=log_softmax, len=64, batch=8, axis=0 | 8 | 32 | 63885.9 | 209.1 | 0.5% | ok |
| log_softmax_f32_pytorch_len64_b8 | #1976 | pytorch | pytorch-cpu | f32 | op=log_softmax, len=64, batch=8, axis=0 | 8 | 256 | 11277.3 | 781.7 | 3.7% | ok |
| log_softmax_f32_eager-single-call_len512_b8 | #1976 | eager-single-call | tenferro-rs | f32 | op=log_softmax, len=512, batch=8, axis=0 | 8 | 2 | 1538948.5 | 18779.8 | 1.3% | ok |
| log_softmax_f32_session-single-call_len512_b8 | #1976 | session-single-call | tenferro-rs | f32 | op=log_softmax, len=512, batch=8, axis=0 | 8 | 2 | 1774577.0 | 18860.5 | 2.4% | ok |
| log_softmax_f32_eager-composed_len512_b8 | #1976 | eager-composed | tenferro-rs | f32 | op=log_softmax, len=512, batch=8, axis=0 | 8 | 2 | 1759564.5 | 39045.2 | 2.6% | ok |
| log_softmax_f32_pytorch_len512_b8 | #1976 | pytorch | pytorch-cpu | f32 | op=log_softmax, len=512, batch=8, axis=0 | 8 | 4 | 338601.0 | 18077.5 | 2.7% | ok |
| masked_softmax_f32_eager-single-call_len64_b8 | #1976 | eager-single-call | tenferro-rs | f32 | op=masked_softmax, len=64, batch=8, axis=0 | 8 | 16 | 121960.4 | 1469.3 | 0.6% | ok |
| masked_softmax_f32_session-single-call_len64_b8 | #1976 | session-single-call | tenferro-rs | f32 | op=masked_softmax, len=64, batch=8, axis=0 | 8 | 32 | 96715.6 | 356.0 | 2.9% | ok |
| masked_softmax_f32_eager-composed_len64_b8 | #1976 | eager-composed | tenferro-rs | f32 | op=masked_softmax, len=64, batch=8, axis=0 | 8 | 32 | 74215.2 | 793.0 | 0.7% | ok |
| masked_softmax_f32_pytorch_len64_b8 | #1976 | pytorch | pytorch-cpu | f32 | op=masked_softmax, len=64, batch=8, axis=0 | 8 | 32 | 63369.0 | 796.8 | 0.8% | ok |
| masked_softmax_f32_eager-single-call_len512_b8 | #1976 | eager-single-call | tenferro-rs | f32 | op=masked_softmax, len=512, batch=8, axis=0 | 8 | 1 | 1863694.0 | 16916.0 | 2.3% | ok |
| masked_softmax_f32_session-single-call_len512_b8 | #1976 | session-single-call | tenferro-rs | f32 | op=masked_softmax, len=512, batch=8, axis=0 | 8 | 2 | 1913952.0 | 20856.5 | 4.5% | ok |
| masked_softmax_f32_eager-composed_len512_b8 | #1976 | eager-composed | tenferro-rs | f32 | op=masked_softmax, len=512, batch=8, axis=0 | 8 | 1 | 2247960.0 | 41001.0 | 3.5% | ok |
| masked_softmax_f32_pytorch_len512_b8 | #1976 | pytorch | pytorch-cpu | f32 | op=masked_softmax, len=512, batch=8, axis=0 | 8 | 2 | 895289.0 | 101589.5 | 8.9% | ok |
| reduce_mean_f32_eager-single-call_1024x64_axis0 | #1976 | eager-single-call | tenferro-rs | f32 | rows=1024, cols=64, axis=0 | 8 | 128 | 15424.2 | 134.7 | 0.5% | ok |
| reduce_mean_f32_session-single-call_1024x64_axis0 | #1976 | session-single-call | tenferro-rs | f32 | rows=1024, cols=64, axis=0 | 8 | 256 | 7987.0 | 52.2 | 0.5% | ok |
| reduce_mean_f32_eager-composed_1024x64_axis0 | #1976 | eager-composed | tenferro-rs | f32 | rows=1024, cols=64, axis=0 | 8 | 128 | 15622.8 | 91.7 | 0.5% | ok |
| reduce_mean_f32_pytorch_1024x64_axis0 | #1976 | pytorch | pytorch-cpu | f32 | rows=1024, cols=64, axis=0 | 8 | 256 | 8667.1 | 43.0 | 0.4% | ok |
| take_along_axis_rows_f64_eager-single-call_n64_b256 | #2008 | eager-single-call | tenferro-rs | f64 | n=64, batch=256, axis=0 | 8 | 2 | 1712236.0 | 37805.2 | 2.5% | ok |
| take_along_axis_rows_f64_session-single-call_n64_b256 | #2008 | session-single-call | tenferro-rs | f64 | n=64, batch=256, axis=0 | 8 | 2 | 1582649.5 | 55578.5 | 2.4% | ok |
| take_along_axis_rows_f64_eager-gather-prebuilt_n64_b256 | #2008 | eager-gather-prebuilt | tenferro-rs | f64 | n=64, batch=256, axis=0 | 8 | 2 | 1507750.0 | 64568.0 | 9.0% | ok |
| take_along_axis_rows_f64_pytorch_n64_b256 | #2008 | pytorch | pytorch-cpu | f64 | n=64, batch=256, axis=0 | 8 | 2 | 982255.5 | 45988.2 | 3.8% | ok |
| lstsq_f64_eager-shared_m768_n384_rhs16 | #2027 | eager-shared | tenferro-rs | f64 | operation=lstsq | 8 | 1 | 6111540.0 | 94711.5 | 3.0% | ok |
| lstsq_f64_pytorch_m768_n384_rhs16 | #2027 | pytorch | pytorch-cpu | f64 | operation=lstsq | 8 | 1 | 3085130.0 | 13099.0 | 0.3% | ok |
| triangular_f64_concrete-shared_n4096_rhs64 | #2028 | concrete-shared | tenferro-rs | f64 | operation=triangular | 8 | 1 | 11477470.0 | 232588.0 | 1.0% | ok |
| triangular_f64_pytorch_n4096_rhs64 | #2028 | pytorch | pytorch-cpu | f64 | operation=triangular | 8 | 1 | 14291385.0 | 90794.0 | 1.0% | ok |
| reshape_f64_concrete-shared_n33554432_to8192x4096 | #2029 | concrete-shared | tenferro-rs | f64 | operation=reshape | 8 | 1 | 16081331.0 | 391589.5 | 2.3% | ok |
| reshape_f64_pytorch_n33554432_to8192x4096 | #2029 | pytorch | pytorch-cpu | f64 | operation=reshape | 8 | 1 | 18074845.0 | 605979.0 | 2.7% | ok |
| followup_structural_broadcast_in_dim_direct-shared | #2021 | direct-shared | tenferro-rs | f64 | case_id=structural.broadcast_in_dim, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [8192, 1], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 8 | — | — | — | — | FAILED |
| followup_structural_broadcast_in_dim_pytorch | #2021 | pytorch | pytorch-cpu | f64 | case_id=structural.broadcast_in_dim, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [8192, 1], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 8 | — | — | — | — | FAILED |
| followup_real_tanh_direct-shared | #2033 | direct-shared | tenferro-rs | f64 | case_id=real.tanh, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [8388608], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 8 | — | — | — | — | FAILED |
| followup_real_tanh_jax | #2033 | jax | jax-cpu | f64 | case_id=real.tanh, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [8388608], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 8 | — | — | — | — | FAILED |
| followup_real_log1p_direct-shared | #2033 | direct-shared | tenferro-rs | f64 | case_id=real.log1p, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [4194304], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 8 | — | — | — | — | FAILED |
| followup_real_log1p_pytorch | #2033 | pytorch | pytorch-cpu | f64 | case_id=real.log1p, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [4194304], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 8 | — | — | — | — | FAILED |
| followup_real_cos_direct-shared | #2033 | direct-shared | tenferro-rs | f64 | case_id=real.cos, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [8388608], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 8 | — | — | — | — | FAILED |
| followup_real_cos_pytorch | #2033 | pytorch | pytorch-cpu | f64 | case_id=real.cos, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [8388608], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 8 | — | — | — | — | FAILED |
| followup_real_sin_direct-shared | #2033 | direct-shared | tenferro-rs | f64 | case_id=real.sin, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [8388608], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 8 | — | — | — | — | FAILED |
| followup_real_sin_pytorch | #2033 | pytorch | pytorch-cpu | f64 | case_id=real.sin, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [8388608], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 8 | — | — | — | — | FAILED |
| followup_real_exp_direct-shared | #2033 | direct-shared | tenferro-rs | f64 | case_id=real.exp, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [8388608], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 8 | — | — | — | — | FAILED |
| followup_real_exp_jax | #2033 | jax | jax-cpu | f64 | case_id=real.exp, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [8388608], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 8 | — | — | — | — | FAILED |
| followup_real_expm1_direct-shared | #2033 | direct-shared | tenferro-rs | f64 | case_id=real.expm1, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [4194304], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 8 | — | — | — | — | FAILED |
| followup_real_expm1_jax | #2033 | jax | jax-cpu | f64 | case_id=real.expm1, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [4194304], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 8 | — | — | — | — | FAILED |
| followup_real_pow_direct-shared | #2033 | direct-shared | tenferro-rs | f64 | case_id=real.pow, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [4194304], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 8 | — | — | — | — | FAILED |
| followup_real_pow_pytorch | #2033 | pytorch | pytorch-cpu | f64 | case_id=real.pow, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [4194304], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 8 | — | — | — | — | FAILED |
| followup_complex_exp_direct-shared | #2033 | direct-shared | tenferro-rs | c64 | case_id=complex.exp, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [4194304], 'dtype': 'c64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 8 | — | — | — | — | FAILED |
| followup_complex_exp_julia-base | #2033 | julia-base | julia-base | c64 | case_id=complex.exp, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [4194304], 'dtype': 'c64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 8 | — | — | — | — | FAILED |
| followup_real_log_direct-shared | #2033 | direct-shared | tenferro-rs | f64 | case_id=real.log, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [8388608], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 8 | — | — | — | — | FAILED |
| followup_real_log_pytorch | #2033 | pytorch | pytorch-cpu | f64 | case_id=real.log, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [8388608], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 8 | — | — | — | — | FAILED |
| followup_real_reduce_max_axis1_direct-shared | #2034 | direct-shared | tenferro-rs | f64 | case_id=real.reduce_max_axis1, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [2048, 2048], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 8 | — | — | — | — | FAILED |
| followup_real_reduce_max_axis1_julia-base | #2034 | julia-base | julia-base | f64 | case_id=real.reduce_max_axis1, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [2048, 2048], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 8 | — | — | — | — | FAILED |
| followup_real_reduce_prod_axis1_direct-shared | #2034 | direct-shared | tenferro-rs | f64 | case_id=real.reduce_prod_axis1, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [2048, 2048], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 8 | — | — | — | — | FAILED |
| followup_real_reduce_prod_axis1_pytorch | #2034 | pytorch | pytorch-cpu | f64 | case_id=real.reduce_prod_axis1, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [2048, 2048], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 8 | — | — | — | — | FAILED |
| followup_real_reduce_sum_axis1_direct-shared | #2034 | direct-shared | tenferro-rs | f64 | case_id=real.reduce_sum_axis1, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [2048, 2048], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 8 | — | — | — | — | FAILED |
| followup_real_reduce_sum_axis1_pytorch | #2034 | pytorch | pytorch-cpu | f64 | case_id=real.reduce_sum_axis1, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [2048, 2048], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 8 | — | — | — | — | FAILED |
| followup_real_reduce_max_axis0_direct-shared | #2034 | direct-shared | tenferro-rs | f64 | case_id=real.reduce_max_axis0, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [2048, 2048], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 8 | — | — | — | — | FAILED |
| followup_real_reduce_max_axis0_julia-base | #2034 | julia-base | julia-base | f64 | case_id=real.reduce_max_axis0, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [2048, 2048], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 8 | — | — | — | — | FAILED |
| followup_real_reduce_min_axis1_direct-shared | #2034 | direct-shared | tenferro-rs | f64 | case_id=real.reduce_min_axis1, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [4096, 4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 8 | — | — | — | — | FAILED |
| followup_real_reduce_min_axis1_julia-base | #2034 | julia-base | julia-base | f64 | case_id=real.reduce_min_axis1, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [4096, 4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 8 | — | — | — | — | FAILED |
| followup_real_reduce_min_axis0_direct-shared | #2034 | direct-shared | tenferro-rs | f64 | case_id=real.reduce_min_axis0, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [2048, 2048], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 8 | — | — | — | — | FAILED |
| followup_real_reduce_min_axis0_julia-base | #2034 | julia-base | julia-base | f64 | case_id=real.reduce_min_axis0, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [2048, 2048], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 8 | — | — | — | — | FAILED |
| followup_real_reduce_max_all_direct-shared | #2034 | direct-shared | tenferro-rs | f64 | case_id=real.reduce_max_all, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [8192, 4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 8 | — | — | — | — | FAILED |
| followup_real_reduce_max_all_julia-base | #2034 | julia-base | julia-base | f64 | case_id=real.reduce_max_all, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [8192, 4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 8 | — | — | — | — | FAILED |
| followup_real_reduce_min_all_direct-shared | #2034 | direct-shared | tenferro-rs | f64 | case_id=real.reduce_min_all, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [8192, 4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 8 | — | — | — | — | FAILED |
| followup_real_reduce_min_all_julia-base | #2034 | julia-base | julia-base | f64 | case_id=real.reduce_min_all, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [8192, 4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 8 | — | — | — | — | FAILED |
| followup_complex_norm_fro_direct-shared | #2035 | direct-shared | tenferro-rs | c64 | case_id=complex.norm_fro, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [2048, 1536], 'dtype': 'c64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 8 | — | — | — | — | FAILED |
| followup_complex_norm_fro_jax | #2035 | jax | jax-cpu | c64 | case_id=complex.norm_fro, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [2048, 1536], 'dtype': 'c64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 8 | — | — | — | — | FAILED |
| followup_index_gather_direct-shared | #2036 #1719 | direct-shared | tenferro-rs | f64 | case_id=index.gather, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [262144], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 8 | — | — | — | — | FAILED |
| followup_index_gather_jax | #2036 #1719 | jax | jax-cpu | f64 | case_id=index.gather, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [262144], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 8 | — | — | — | — | FAILED |
| followup_index_scatter_direct-shared | #2036 #1719 | direct-shared | tenferro-rs | f64 | case_id=index.scatter, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [262144], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 8 | — | — | — | — | FAILED |
| followup_index_scatter_pytorch | #2036 #1719 | pytorch | pytorch-cpu | f64 | case_id=index.scatter, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [262144], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 8 | — | — | — | — | FAILED |
| followup_index_dynamic_update_slice_direct-shared | #2036 #1719 | direct-shared | tenferro-rs | f64 | case_id=index.dynamic_update_slice, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [2097152], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 8 | — | — | — | — | FAILED |
| followup_index_dynamic_update_slice_jax | #2036 #1719 | jax | jax-cpu | f64 | case_id=index.dynamic_update_slice, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [2097152], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 8 | — | — | — | — | FAILED |
| followup_perm_reverse_15d_3_materialize | #2037 #2228 | materialize | tenferro-rs | f64 | case_id=perm.reverse_15d_3, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 8 | — | — | — | — | FAILED |
| followup_perm_reverse_15d_3_strided-rs | #2037 #2228 | strided-rs | strided-rs | f64 | case_id=perm.reverse_15d_3, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 8 | — | — | — | — | FAILED |
| followup_structural_transpose_direct-shared | #2037 #2228 | direct-shared | tenferro-rs | f64 | case_id=structural.transpose, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [4096, 4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 8 | — | — | — | — | FAILED |
| followup_structural_transpose_pytorch | #2037 #2228 | pytorch | pytorch-cpu | f64 | case_id=structural.transpose, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [4096, 4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 8 | — | — | — | — | FAILED |
| followup_perm_reverse_23d_2_materialize | #2037 #2228 | materialize | tenferro-rs | f64 | case_id=perm.reverse_23d_2, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 8 | — | — | — | — | FAILED |
| followup_perm_reverse_23d_2_strided-rs | #2037 #2228 | strided-rs | strided-rs | f64 | case_id=perm.reverse_23d_2, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 8 | — | — | — | — | FAILED |
| followup_perm_transpose_3d_256_201_materialize | #2037 #2228 | materialize | tenferro-rs | f64 | case_id=perm.transpose_3d_256_201, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [256, 256, 256], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 8 | — | — | — | — | FAILED |
| followup_perm_transpose_3d_256_201_strided-rs | #2037 #2228 | strided-rs | strided-rs | f64 | case_id=perm.transpose_3d_256_201, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [256, 256, 256], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 8 | — | — | — | — | FAILED |
| followup_perm_cyclic_15d_3_materialize | #2037 #2228 | materialize | tenferro-rs | f64 | case_id=perm.cyclic_15d_3, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 8 | — | — | — | — | FAILED |
| followup_perm_cyclic_15d_3_strided-rs | #2037 #2228 | strided-rs | strided-rs | f64 | case_id=perm.cyclic_15d_3, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 8 | — | — | — | — | FAILED |
| followup_perm_transpose_3d_256_102_materialize | #2037 #2228 | materialize | tenferro-rs | f64 | case_id=perm.transpose_3d_256_102, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [256, 256, 256], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 8 | — | — | — | — | FAILED |
| followup_perm_transpose_3d_256_102_strided-rs | #2037 #2228 | strided-rs | strided-rs | f64 | case_id=perm.transpose_3d_256_102, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [256, 256, 256], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 8 | — | — | — | — | FAILED |
| followup_perm_tn_light_415_24d_scattered_to_colmajor_materialize | #2037 #2228 | materialize | tenferro-rs | f64 | case_id=perm.tn_light_415_24d_scattered_to_colmajor, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 8 | — | — | — | — | FAILED |
| followup_perm_tn_light_415_24d_scattered_to_colmajor_strided-rs | #2037 #2228 | strided-rs | strided-rs | f64 | case_id=perm.tn_light_415_24d_scattered_to_colmajor, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 8 | — | — | — | — | FAILED |
| followup_perm_tn_light_415_24d_contiguous_same_perm_materialize | #2037 #2228 | materialize | tenferro-rs | f64 | case_id=perm.tn_light_415_24d_contiguous_same_perm, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 8 | — | — | — | — | FAILED |
| followup_perm_tn_light_415_24d_contiguous_same_perm_strided-rs | #2037 #2228 | strided-rs | strided-rs | f64 | case_id=perm.tn_light_415_24d_contiguous_same_perm, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 8 | — | — | — | — | FAILED |
| followup_perm_rotation_6d_32_32_32_32_16_16_materialize | #2037 #2228 | materialize | tenferro-rs | f64 | case_id=perm.rotation_6d_32_32_32_32_16_16, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [32, 32, 32, 32, 16, 16], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 8 | — | — | — | — | FAILED |
| followup_perm_rotation_6d_32_32_32_32_16_16_strided-rs | #2037 #2228 | strided-rs | strided-rs | f64 | case_id=perm.rotation_6d_32_32_32_32_16_16, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [32, 32, 32, 32, 16, 16], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 8 | — | — | — | — | FAILED |
| followup_structural_extract_diagonal_direct-shared | #2038 #2228 | direct-shared | tenferro-rs | f64 | case_id=structural.extract_diagonal, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [8388608, 2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 8 | — | — | — | — | FAILED |
| followup_structural_extract_diagonal_jax | #2038 #2228 | jax | jax-cpu | f64 | case_id=structural.extract_diagonal, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [8388608, 2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 8 | — | — | — | — | FAILED |
| followup_fft_irfft_direct-shared | #2039 #2230 #2228 | direct-shared | tenferro-rs | c32 | case_id=fft.irfft, rust_path=direct-shared, reference_path=mkl-dfti, fixture={'input_shape': [524289], 'dtype': 'c32', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 8 | — | — | — | — | FAILED |
| followup_fft_irfft_mkl-dfti | #2039 #2230 #2228 | mkl-dfti | mkl-dfti | c32 | case_id=fft.irfft, rust_path=direct-shared, reference_path=mkl-dfti, fixture={'input_shape': [524289], 'dtype': 'c32', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 8 | — | — | — | — | FAILED |
| followup_fft_rfft_direct-shared | #2039 #2230 #2228 | direct-shared | tenferro-rs | f32 | case_id=fft.rfft, rust_path=direct-shared, reference_path=mkl-dfti, fixture={'input_shape': [1048576], 'dtype': 'f32', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 8 | — | — | — | — | FAILED |
| followup_fft_rfft_mkl-dfti | #2039 #2230 #2228 | mkl-dfti | mkl-dfti | f32 | case_id=fft.rfft, rust_path=direct-shared, reference_path=mkl-dfti, fixture={'input_shape': [1048576], 'dtype': 'f32', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 8 | — | — | — | — | FAILED |
| followup_fft_fft_direct-shared | #2039 #2228 | direct-shared | tenferro-rs | c64 | case_id=fft.fft, rust_path=direct-shared, reference_path=mkl-dfti, fixture={'input_shape': [1048576], 'dtype': 'c64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 8 | — | — | — | — | FAILED |
| followup_fft_fft_mkl-dfti | #2039 #2228 | mkl-dfti | mkl-dfti | c64 | case_id=fft.fft, rust_path=direct-shared, reference_path=mkl-dfti, fixture={'input_shape': [1048576], 'dtype': 'c64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 8 | — | — | — | — | FAILED |
| followup_fft_ifft_direct-shared | #2039 #2228 | direct-shared | tenferro-rs | c32 | case_id=fft.ifft, rust_path=direct-shared, reference_path=mkl-dfti, fixture={'input_shape': [1048576], 'dtype': 'c32', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 8 | — | — | — | — | FAILED |
| followup_fft_ifft_mkl-dfti | #2039 #2228 | mkl-dfti | mkl-dfti | c32 | case_id=fft.ifft, rust_path=direct-shared, reference_path=mkl-dfti, fixture={'input_shape': [1048576], 'dtype': 'c32', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 8 | — | — | — | — | FAILED |
| followup_metadata_transpose_view_metadata | #2040 #1719 | metadata | tenferro-rs | f64 | case_id=metadata.transpose_view, rust_path=metadata, reference_path=julia-base, fixture={'input_shape': [2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 8 | — | — | — | — | FAILED |
| followup_metadata_transpose_view_julia-base | #2040 #1719 | julia-base | julia-base | f64 | case_id=metadata.transpose_view, rust_path=metadata, reference_path=julia-base, fixture={'input_shape': [2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 8 | — | — | — | — | FAILED |
| followup_metadata_transpose_view_metadata-host-dyn | #2040 #1719 | metadata-host-dyn | tenferro-rs | f64 | case_id=metadata.transpose_view, rust_path=metadata-host-dyn, reference_path=julia-base, fixture={'input_shape': [2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 8 | — | — | — | — | FAILED |
| followup_metadata_transpose_view_metadata-host-static | #2040 #1719 | metadata-host-static | tenferro-rs | f64 | case_id=metadata.transpose_view, rust_path=metadata-host-static, reference_path=julia-base, fixture={'input_shape': [2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 8 | — | — | — | — | FAILED |
| followup_metadata_slice_view_metadata | #2040 #1719 | metadata | tenferro-rs | f64 | case_id=metadata.slice_view, rust_path=metadata, reference_path=julia-base, fixture={'input_shape': [4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 8 | — | — | — | — | FAILED |
| followup_metadata_slice_view_julia-base | #2040 #1719 | julia-base | julia-base | f64 | case_id=metadata.slice_view, rust_path=metadata, reference_path=julia-base, fixture={'input_shape': [4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 8 | — | — | — | — | FAILED |
| followup_metadata_slice_view_metadata-host-dyn | #2040 #1719 | metadata-host-dyn | tenferro-rs | f64 | case_id=metadata.slice_view, rust_path=metadata-host-dyn, reference_path=julia-base, fixture={'input_shape': [4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 8 | — | — | — | — | FAILED |
| followup_metadata_slice_view_metadata-host-static | #2040 #1719 | metadata-host-static | tenferro-rs | f64 | case_id=metadata.slice_view, rust_path=metadata-host-static, reference_path=julia-base, fixture={'input_shape': [4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 8 | — | — | — | — | FAILED |
| followup_metadata_reshape_view_metadata | #2040 #1719 | metadata | tenferro-rs | f64 | case_id=metadata.reshape_view, rust_path=metadata, reference_path=julia-base, fixture={'input_shape': [1024], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 8 | — | — | — | — | FAILED |
| followup_metadata_reshape_view_julia-base | #2040 #1719 | julia-base | julia-base | f64 | case_id=metadata.reshape_view, rust_path=metadata, reference_path=julia-base, fixture={'input_shape': [1024], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 8 | — | — | — | — | FAILED |
| followup_metadata_reshape_view_metadata-host-dyn | #2040 #1719 | metadata-host-dyn | tenferro-rs | f64 | case_id=metadata.reshape_view, rust_path=metadata-host-dyn, reference_path=julia-base, fixture={'input_shape': [1024], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 8 | — | — | — | — | FAILED |
| followup_metadata_reshape_view_metadata-host-static | #2040 #1719 | metadata-host-static | tenferro-rs | f64 | case_id=metadata.reshape_view, rust_path=metadata-host-static, reference_path=julia-base, fixture={'input_shape': [1024], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 8 | — | — | — | — | FAILED |
| gemm_bound_f64_t1_t8_scaling | #2141 | tenferro | tenferro-rs | f64 | n=1536 | 8 | 1 | 30016590.0 | 93218.0 | 0.2% | ok |
| gemm_bound_f64_t1_t8_scaling_pytorch | #2141 | pytorch | pytorch-cpu | f64 | n=1536 | 8 | 1 | 36055076.0 | 162923.5 | 0.4% | ok |
| small_grad_qr_f64_trace-prepared_n4 | #2186 | trace-prepared | tenferro-rs | f64 | op=qr, n=4 | 8 | 64 | 55467.9 | 873.3 | 0.8% | ok |
| small_grad_qr_f64_pytorch_n4 | #2186 | pytorch | pytorch-cpu | f64 | op=qr, n=4 | 8 | 32 | 78108.4 | 447.1 | 2.5% | ok |
| small_grad_qr_f64_trace-prepared_n16 | #2186 | trace-prepared | tenferro-rs | f64 | op=qr, n=16 | 8 | 32 | 75759.7 | 509.1 | 0.4% | ok |
| small_grad_qr_f64_pytorch_n16 | #2186 | pytorch | pytorch-cpu | f64 | op=qr, n=16 | 8 | 32 | 89049.4 | 538.3 | 1.0% | ok |
| small_grad_eigh_f64_trace-prepared_n4 | #2186 | trace-prepared | tenferro-rs | f64 | op=eigh, n=4 | 8 | 32 | 76549.6 | 802.4 | 0.6% | ok |
| small_grad_eigh_f64_pytorch_n4 | #2186 | pytorch | pytorch-cpu | f64 | op=eigh, n=4 | 8 | 32 | 74927.8 | 776.3 | 1.0% | ok |
| small_grad_eigh_f64_trace-prepared_n16 | #2186 | trace-prepared | tenferro-rs | f64 | op=eigh, n=16 | 8 | 16 | 132523.2 | 781.1 | 0.5% | ok |
| small_grad_eigh_f64_pytorch_n16 | #2186 | pytorch | pytorch-cpu | f64 | op=eigh, n=16 | 8 | 32 | 100783.8 | 484.3 | 0.5% | ok |
| small_grad_lu_f64_trace-prepared_n4 | #2186 | trace-prepared | tenferro-rs | f64 | op=lu, n=4 | 8 | 64 | 40106.2 | 145.7 | 0.3% | ok |
| small_grad_lu_f64_pytorch_n4 | #2186 | pytorch | pytorch-cpu | f64 | op=lu, n=4 | 8 | 32 | 89496.5 | 824.0 | 1.3% | ok |
| small_grad_lu_f64_trace-prepared_n16 | #2186 | trace-prepared | tenferro-rs | f64 | op=lu, n=16 | 8 | 64 | 49388.1 | 152.0 | 0.3% | ok |
| small_grad_lu_f64_pytorch_n16 | #2186 | pytorch | pytorch-cpu | f64 | op=lu, n=16 | 8 | 32 | 97971.1 | 1919.8 | 1.3% | ok |
| small_grad_svd_f64_trace-prepared_n4 | #2186 | trace-prepared | tenferro-rs | f64 | op=svd, n=4 | 8 | 32 | 122419.8 | 337.8 | 0.2% | ok |
| small_grad_svd_f64_pytorch_n4 | #2186 | pytorch | pytorch-cpu | f64 | op=svd, n=4 | 8 | 32 | 112741.7 | 652.5 | 3.6% | ok |
| small_grad_svd_f64_trace-prepared_n16 | #2186 | trace-prepared | tenferro-rs | f64 | op=svd, n=16 | 8 | 16 | 224642.4 | 860.6 | 0.3% | ok |
| small_grad_svd_f64_pytorch_n16 | #2186 | pytorch | pytorch-cpu | f64 | op=svd, n=16 | 8 | 16 | 174459.5 | 2919.5 | 1.6% | ok |
| small_grad_solve_f64_trace-prepared_n4 | #2186 | trace-prepared | tenferro-rs | f64 | op=solve, n=4 | 8 | 64 | 33818.5 | 187.0 | 0.3% | ok |
| small_grad_solve_f64_pytorch_n4 | #2186 | pytorch | pytorch-cpu | f64 | op=solve, n=4 | 8 | 32 | 62143.3 | 449.9 | 0.6% | ok |
| small_grad_solve_f64_trace-prepared_n16 | #2186 | trace-prepared | tenferro-rs | f64 | op=solve, n=16 | 8 | 64 | 41631.2 | 130.6 | 0.4% | ok |
| small_grad_solve_f64_pytorch_n16 | #2186 | pytorch | pytorch-cpu | f64 | op=solve, n=16 | 8 | 32 | 71375.6 | 364.0 | 0.5% | ok |
| fft_axis0_c64_fft-executor_n128 | #1765 | fft-executor | tenferro-rs | c64 | n=128 | 8 | 1 | 3219311.0 | 53986.0 | 1.4% | ok |
| fft_axis0_c64_pytorch_n128 | #1765 | pytorch | pytorch-cpu | c64 | n=128 | 8 | 1 | 4151348.0 | 156872.0 | 3.5% | ok |
| degenerate_dot_hadamard_f64_dot-general-shared_n1024 | #2111 | dot-general-shared | tenferro-rs | f64 | form=hadamard, n=1024 | 8 | 4 | 426422.5 | 5681.9 | 3.3% | ok |
| degenerate_dot_hadamard_f64_pytorch_n1024 | #2111 | pytorch | pytorch-cpu | f64 | form=hadamard, n=1024 | 8 | 4 | 423474.2 | 11897.1 | 7.3% | ok |
| degenerate_dot_outer_f64_dot-general-shared_n1024 | #2111 | dot-general-shared | tenferro-rs | f64 | form=outer, n=1024 | 8 | 2 | 194592.5 | 9911.0 | 3.0% | ok |
| degenerate_dot_outer_f64_pytorch_n1024 | #2111 | pytorch | pytorch-cpu | f64 | form=outer, n=1024 | 8 | 8 | 443794.8 | 19876.4 | 4.4% | ok |
| degenerate_dot_rank0_f64_dot-general-shared_n1024 | #2111 | dot-general-shared | tenferro-rs | f64 | form=rank0, n=1024 | 8 | 1 | 15770426.0 | 603870.0 | 2.8% | ok |
| degenerate_dot_rank0_f64_pytorch_n1024 | #2111 | pytorch | pytorch-cpu | f64 | form=rank0, n=1024 | 8 | 4 | 311245.0 | 14483.2 | 3.8% | ok |
| erased_add_f64_session-add_n256 | #1766 | session-add | tenferro-rs | f64 | n=256 | 8 | 256 | 1669.6 | 18.4 | 1.8% | ok |
| erased_add_f64_pytorch_n256 | #1766 | pytorch | pytorch-cpu | f64 | n=256 | 8 | 1024 | 1914.4 | 7.3 | 3.8% | ok |
| erased_add_f64_session-add_n4096 | #1766 | session-add | tenferro-rs | f64 | n=4096 | 8 | 256 | 3238.7 | 73.5 | 1.6% | ok |
| erased_add_f64_pytorch_n4096 | #1766 | pytorch | pytorch-cpu | f64 | n=4096 | 8 | 1024 | 3757.2 | 53.9 | 1.3% | ok |
| erased_add_f64_session-add_n1048576 | #1766 | session-add | tenferro-rs | f64 | n=1048576 | 8 | 4 | 438850.5 | 4785.1 | 1.9% | ok |
| erased_add_f64_pytorch_n1048576 | #1766 | pytorch | pytorch-cpu | f64 | n=1048576 | 8 | 4 | 429943.8 | 9332.4 | 6.1% | ok |
| erased_construct_f64_col_n8 | #2177 | col-major | tenferro-rs | f64 | n=8, layout=col | 8 | 16384 | 126.4 | 1.9 | 15.6% | NOISY |
| erased_construct_f64_row_n8 | #2177 | row-major | tenferro-rs | f64 | n=8, layout=row | 8 | 8192 | 261.8 | 2.7 | 7.5% | ok |
| erased_construct_f64_col_n4096 | #2177 | col-major | tenferro-rs | f64 | n=4096, layout=col | 8 | 16384 | 214.6 | 3.8 | 1.3% | ok |
| erased_construct_f64_row_n4096 | #2177 | row-major | tenferro-rs | f64 | n=4096, layout=row | 8 | 128 | 18109.7 | 776.3 | 3.6% | ok |
| erased_construct_f64_col_n65536 | #2177 | col-major | tenferro-rs | f64 | n=65536, layout=col | 8 | 1024 | 155.6 | 22.4 | 7.3% | ok |
| erased_construct_f64_row_n65536 | #2177 | row-major | tenferro-rs | f64 | n=65536, layout=row | 8 | 8 | 299568.4 | 27831.9 | 5.4% | ok |
| eager_small_leaf_f64_eager-held-session | #1704 | eager-held-session | tenferro-rs | f64 | op=leaf | 8 | 1024 | 3470.6 | 28.5 | 1.7% | ok |
| eager_small_add_f64_eager-held-session | #1667 | eager-held-session | tenferro-rs | f64 | op=add | 8 | 1024 | 2924.5 | 13.6 | 0.3% | ok |
| eager_small_add_f64_pytorch | #1667 | pytorch | pytorch-cpu | f64 | op=add | 8 | 2048 | 1821.5 | 8.4 | 0.3% | ok |
| eager_small_gemm_f64_eager-held-session | #1904 | eager-held-session | tenferro-rs | f64 | op=gemm | 8 | 512 | 4303.4 | 29.9 | 0.6% | ok |
| eager_small_gemm_f64_pytorch | #1904 | pytorch | pytorch-cpu | f64 | op=gemm | 8 | 1024 | 2031.2 | 6.6 | 0.3% | ok |


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

### FAILED: activation_erf_f32_eager-single-call_1024x64 (1 threads)
```
RuntimeError('eager-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: activation_erf_f32_pytorch_1024x64 (1 threads)
```
RuntimeError('eager-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: activation_sigmoid_f32_eager-single-call_1024x64 (1 threads)
```
RuntimeError('eager-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: activation_sigmoid_f32_pytorch_1024x64 (1 threads)
```
RuntimeError('eager-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: activation_silu_f32_eager-single-call_1024x64 (1 threads)
```
RuntimeError('eager-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: activation_silu_f32_pytorch_1024x64 (1 threads)
```
RuntimeError('eager-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: activation_softplus_f32_eager-single-call_1024x64 (1 threads)
```
RuntimeError('eager-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: activation_softplus_f32_pytorch_1024x64 (1 threads)
```
RuntimeError('eager-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: activation_gelu_tanh_f32_eager-single-call_1024x64 (1 threads)
```
RuntimeError('eager-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: activation_gelu_tanh_f32_pytorch_1024x64 (1 threads)
```
RuntimeError('eager-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_structural_broadcast_in_dim_direct-shared (1 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_structural_broadcast_in_dim_pytorch (1 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_real_tanh_direct-shared (1 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_real_tanh_jax (1 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_real_log1p_direct-shared (1 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_real_log1p_pytorch (1 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_real_cos_direct-shared (1 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_real_cos_pytorch (1 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_real_sin_direct-shared (1 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_real_sin_pytorch (1 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_real_exp_direct-shared (1 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_real_exp_jax (1 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_real_expm1_direct-shared (1 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_real_expm1_jax (1 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_real_pow_direct-shared (1 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_real_pow_pytorch (1 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_complex_exp_direct-shared (1 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_complex_exp_julia-base (1 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_real_log_direct-shared (1 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_real_log_pytorch (1 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_real_reduce_max_axis1_direct-shared (1 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_real_reduce_max_axis1_julia-base (1 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_real_reduce_prod_axis1_direct-shared (1 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_real_reduce_prod_axis1_pytorch (1 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_real_reduce_sum_axis1_direct-shared (1 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_real_reduce_sum_axis1_pytorch (1 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_real_reduce_max_axis0_direct-shared (1 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_real_reduce_max_axis0_julia-base (1 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_real_reduce_min_axis1_direct-shared (1 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_real_reduce_min_axis1_julia-base (1 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_real_reduce_min_axis0_direct-shared (1 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_real_reduce_min_axis0_julia-base (1 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_real_reduce_max_all_direct-shared (1 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_real_reduce_max_all_julia-base (1 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_real_reduce_min_all_direct-shared (1 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_real_reduce_min_all_julia-base (1 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_complex_norm_fro_direct-shared (1 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_complex_norm_fro_jax (1 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_index_gather_direct-shared (1 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_index_gather_jax (1 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_index_scatter_direct-shared (1 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_index_scatter_pytorch (1 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_index_dynamic_update_slice_direct-shared (1 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_index_dynamic_update_slice_jax (1 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_perm_reverse_15d_3_materialize (1 threads)
```
RuntimeError('strided-rs: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_perm_reverse_15d_3_strided-rs (1 threads)
```
RuntimeError('strided-rs: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_structural_transpose_direct-shared (1 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_structural_transpose_pytorch (1 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_perm_reverse_23d_2_materialize (1 threads)
```
RuntimeError('strided-rs: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_perm_reverse_23d_2_strided-rs (1 threads)
```
RuntimeError('strided-rs: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_perm_transpose_3d_256_201_materialize (1 threads)
```
RuntimeError('strided-rs: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_perm_transpose_3d_256_201_strided-rs (1 threads)
```
RuntimeError('strided-rs: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_perm_cyclic_15d_3_materialize (1 threads)
```
RuntimeError('strided-rs: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_perm_cyclic_15d_3_strided-rs (1 threads)
```
RuntimeError('strided-rs: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_perm_transpose_3d_256_102_materialize (1 threads)
```
RuntimeError('strided-rs: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_perm_transpose_3d_256_102_strided-rs (1 threads)
```
RuntimeError('strided-rs: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_perm_tn_light_415_24d_scattered_to_colmajor_materialize (1 threads)
```
RuntimeError('strided-rs: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_perm_tn_light_415_24d_scattered_to_colmajor_strided-rs (1 threads)
```
RuntimeError('strided-rs: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_perm_tn_light_415_24d_contiguous_same_perm_materialize (1 threads)
```
RuntimeError('strided-rs: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_perm_tn_light_415_24d_contiguous_same_perm_strided-rs (1 threads)
```
RuntimeError('strided-rs: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_perm_rotation_6d_32_32_32_32_16_16_materialize (1 threads)
```
RuntimeError('strided-rs: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_perm_rotation_6d_32_32_32_32_16_16_strided-rs (1 threads)
```
RuntimeError('strided-rs: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_structural_extract_diagonal_direct-shared (1 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_structural_extract_diagonal_jax (1 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_fft_irfft_direct-shared (1 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_fft_irfft_mkl-dfti (1 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_fft_rfft_direct-shared (1 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_fft_rfft_mkl-dfti (1 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_fft_fft_direct-shared (1 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_fft_fft_mkl-dfti (1 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_fft_ifft_direct-shared (1 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_fft_ifft_mkl-dfti (1 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_metadata_transpose_view_metadata (1 threads)
```
RuntimeError('metadata: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_metadata_transpose_view_julia-base (1 threads)
```
RuntimeError('metadata: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_metadata_transpose_view_metadata-host-dyn (1 threads)
```
RuntimeError('metadata-host-dyn: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_metadata_transpose_view_metadata-host-static (1 threads)
```
RuntimeError('metadata-host-static: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_metadata_slice_view_metadata (1 threads)
```
RuntimeError('metadata: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_metadata_slice_view_julia-base (1 threads)
```
RuntimeError('metadata: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_metadata_slice_view_metadata-host-dyn (1 threads)
```
RuntimeError('metadata-host-dyn: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_metadata_slice_view_metadata-host-static (1 threads)
```
RuntimeError('metadata-host-static: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_metadata_reshape_view_metadata (1 threads)
```
RuntimeError('metadata: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_metadata_reshape_view_julia-base (1 threads)
```
RuntimeError('metadata: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_metadata_reshape_view_metadata-host-dyn (1 threads)
```
RuntimeError('metadata-host-dyn: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_metadata_reshape_view_metadata-host-static (1 threads)
```
RuntimeError('metadata-host-static: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: activation_erf_f32_eager-single-call_1024x64 (4 threads)
```
RuntimeError('eager-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: activation_erf_f32_pytorch_1024x64 (4 threads)
```
RuntimeError('eager-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: activation_sigmoid_f32_eager-single-call_1024x64 (4 threads)
```
RuntimeError('eager-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: activation_sigmoid_f32_pytorch_1024x64 (4 threads)
```
RuntimeError('eager-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: activation_silu_f32_eager-single-call_1024x64 (4 threads)
```
RuntimeError('eager-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: activation_silu_f32_pytorch_1024x64 (4 threads)
```
RuntimeError('eager-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: activation_softplus_f32_eager-single-call_1024x64 (4 threads)
```
RuntimeError('eager-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: activation_softplus_f32_pytorch_1024x64 (4 threads)
```
RuntimeError('eager-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: activation_gelu_tanh_f32_eager-single-call_1024x64 (4 threads)
```
RuntimeError('eager-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: activation_gelu_tanh_f32_pytorch_1024x64 (4 threads)
```
RuntimeError('eager-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_structural_broadcast_in_dim_direct-shared (4 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_structural_broadcast_in_dim_pytorch (4 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_real_tanh_direct-shared (4 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_real_tanh_jax (4 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_real_log1p_direct-shared (4 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_real_log1p_pytorch (4 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_real_cos_direct-shared (4 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_real_cos_pytorch (4 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_real_sin_direct-shared (4 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_real_sin_pytorch (4 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_real_exp_direct-shared (4 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_real_exp_jax (4 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_real_expm1_direct-shared (4 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_real_expm1_jax (4 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_real_pow_direct-shared (4 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_real_pow_pytorch (4 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_complex_exp_direct-shared (4 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_complex_exp_julia-base (4 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_real_log_direct-shared (4 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_real_log_pytorch (4 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_real_reduce_max_axis1_direct-shared (4 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_real_reduce_max_axis1_julia-base (4 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_real_reduce_prod_axis1_direct-shared (4 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_real_reduce_prod_axis1_pytorch (4 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_real_reduce_sum_axis1_direct-shared (4 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_real_reduce_sum_axis1_pytorch (4 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_real_reduce_max_axis0_direct-shared (4 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_real_reduce_max_axis0_julia-base (4 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_real_reduce_min_axis1_direct-shared (4 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_real_reduce_min_axis1_julia-base (4 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_real_reduce_min_axis0_direct-shared (4 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_real_reduce_min_axis0_julia-base (4 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_real_reduce_max_all_direct-shared (4 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_real_reduce_max_all_julia-base (4 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_real_reduce_min_all_direct-shared (4 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_real_reduce_min_all_julia-base (4 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_complex_norm_fro_direct-shared (4 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_complex_norm_fro_jax (4 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_index_gather_direct-shared (4 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_index_gather_jax (4 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_index_scatter_direct-shared (4 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_index_scatter_pytorch (4 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_index_dynamic_update_slice_direct-shared (4 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_index_dynamic_update_slice_jax (4 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_perm_reverse_15d_3_materialize (4 threads)
```
RuntimeError('strided-rs: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_perm_reverse_15d_3_strided-rs (4 threads)
```
RuntimeError('strided-rs: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_structural_transpose_direct-shared (4 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_structural_transpose_pytorch (4 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_perm_reverse_23d_2_materialize (4 threads)
```
RuntimeError('strided-rs: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_perm_reverse_23d_2_strided-rs (4 threads)
```
RuntimeError('strided-rs: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_perm_transpose_3d_256_201_materialize (4 threads)
```
RuntimeError('strided-rs: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_perm_transpose_3d_256_201_strided-rs (4 threads)
```
RuntimeError('strided-rs: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_perm_cyclic_15d_3_materialize (4 threads)
```
RuntimeError('strided-rs: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_perm_cyclic_15d_3_strided-rs (4 threads)
```
RuntimeError('strided-rs: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_perm_transpose_3d_256_102_materialize (4 threads)
```
RuntimeError('strided-rs: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_perm_transpose_3d_256_102_strided-rs (4 threads)
```
RuntimeError('strided-rs: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_perm_tn_light_415_24d_scattered_to_colmajor_materialize (4 threads)
```
RuntimeError('strided-rs: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_perm_tn_light_415_24d_scattered_to_colmajor_strided-rs (4 threads)
```
RuntimeError('strided-rs: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_perm_tn_light_415_24d_contiguous_same_perm_materialize (4 threads)
```
RuntimeError('strided-rs: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_perm_tn_light_415_24d_contiguous_same_perm_strided-rs (4 threads)
```
RuntimeError('strided-rs: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_perm_rotation_6d_32_32_32_32_16_16_materialize (4 threads)
```
RuntimeError('strided-rs: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_perm_rotation_6d_32_32_32_32_16_16_strided-rs (4 threads)
```
RuntimeError('strided-rs: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_structural_extract_diagonal_direct-shared (4 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_structural_extract_diagonal_jax (4 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_fft_irfft_direct-shared (4 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_fft_irfft_mkl-dfti (4 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_fft_rfft_direct-shared (4 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_fft_rfft_mkl-dfti (4 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_fft_fft_direct-shared (4 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_fft_fft_mkl-dfti (4 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_fft_ifft_direct-shared (4 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_fft_ifft_mkl-dfti (4 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_metadata_transpose_view_metadata (4 threads)
```
RuntimeError('metadata: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_metadata_transpose_view_julia-base (4 threads)
```
RuntimeError('metadata: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_metadata_transpose_view_metadata-host-dyn (4 threads)
```
RuntimeError('metadata-host-dyn: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_metadata_transpose_view_metadata-host-static (4 threads)
```
RuntimeError('metadata-host-static: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_metadata_slice_view_metadata (4 threads)
```
RuntimeError('metadata: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_metadata_slice_view_julia-base (4 threads)
```
RuntimeError('metadata: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_metadata_slice_view_metadata-host-dyn (4 threads)
```
RuntimeError('metadata-host-dyn: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_metadata_slice_view_metadata-host-static (4 threads)
```
RuntimeError('metadata-host-static: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_metadata_reshape_view_metadata (4 threads)
```
RuntimeError('metadata: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_metadata_reshape_view_julia-base (4 threads)
```
RuntimeError('metadata: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_metadata_reshape_view_metadata-host-dyn (4 threads)
```
RuntimeError('metadata-host-dyn: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_metadata_reshape_view_metadata-host-static (4 threads)
```
RuntimeError('metadata-host-static: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: activation_erf_f32_eager-single-call_1024x64 (8 threads)
```
RuntimeError('eager-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: activation_erf_f32_pytorch_1024x64 (8 threads)
```
RuntimeError('eager-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: activation_sigmoid_f32_eager-single-call_1024x64 (8 threads)
```
RuntimeError('eager-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: activation_sigmoid_f32_pytorch_1024x64 (8 threads)
```
RuntimeError('eager-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: activation_silu_f32_eager-single-call_1024x64 (8 threads)
```
RuntimeError('eager-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: activation_silu_f32_pytorch_1024x64 (8 threads)
```
RuntimeError('eager-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: activation_softplus_f32_eager-single-call_1024x64 (8 threads)
```
RuntimeError('eager-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: activation_softplus_f32_pytorch_1024x64 (8 threads)
```
RuntimeError('eager-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: activation_gelu_tanh_f32_eager-single-call_1024x64 (8 threads)
```
RuntimeError('eager-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: activation_gelu_tanh_f32_pytorch_1024x64 (8 threads)
```
RuntimeError('eager-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_structural_broadcast_in_dim_direct-shared (8 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_structural_broadcast_in_dim_pytorch (8 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_real_tanh_direct-shared (8 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_real_tanh_jax (8 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_real_log1p_direct-shared (8 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_real_log1p_pytorch (8 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_real_cos_direct-shared (8 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_real_cos_pytorch (8 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_real_sin_direct-shared (8 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_real_sin_pytorch (8 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_real_exp_direct-shared (8 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_real_exp_jax (8 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_real_expm1_direct-shared (8 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_real_expm1_jax (8 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_real_pow_direct-shared (8 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_real_pow_pytorch (8 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_complex_exp_direct-shared (8 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_complex_exp_julia-base (8 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_real_log_direct-shared (8 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_real_log_pytorch (8 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_real_reduce_max_axis1_direct-shared (8 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_real_reduce_max_axis1_julia-base (8 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_real_reduce_prod_axis1_direct-shared (8 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_real_reduce_prod_axis1_pytorch (8 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_real_reduce_sum_axis1_direct-shared (8 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_real_reduce_sum_axis1_pytorch (8 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_real_reduce_max_axis0_direct-shared (8 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_real_reduce_max_axis0_julia-base (8 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_real_reduce_min_axis1_direct-shared (8 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_real_reduce_min_axis1_julia-base (8 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_real_reduce_min_axis0_direct-shared (8 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_real_reduce_min_axis0_julia-base (8 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_real_reduce_max_all_direct-shared (8 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_real_reduce_max_all_julia-base (8 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_real_reduce_min_all_direct-shared (8 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_real_reduce_min_all_julia-base (8 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_complex_norm_fro_direct-shared (8 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_complex_norm_fro_jax (8 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_index_gather_direct-shared (8 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_index_gather_jax (8 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_index_scatter_direct-shared (8 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_index_scatter_pytorch (8 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_index_dynamic_update_slice_direct-shared (8 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_index_dynamic_update_slice_jax (8 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_perm_reverse_15d_3_materialize (8 threads)
```
RuntimeError('strided-rs: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_perm_reverse_15d_3_strided-rs (8 threads)
```
RuntimeError('strided-rs: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_structural_transpose_direct-shared (8 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_structural_transpose_pytorch (8 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_perm_reverse_23d_2_materialize (8 threads)
```
RuntimeError('strided-rs: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_perm_reverse_23d_2_strided-rs (8 threads)
```
RuntimeError('strided-rs: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_perm_transpose_3d_256_201_materialize (8 threads)
```
RuntimeError('strided-rs: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_perm_transpose_3d_256_201_strided-rs (8 threads)
```
RuntimeError('strided-rs: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_perm_cyclic_15d_3_materialize (8 threads)
```
RuntimeError('strided-rs: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_perm_cyclic_15d_3_strided-rs (8 threads)
```
RuntimeError('strided-rs: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_perm_transpose_3d_256_102_materialize (8 threads)
```
RuntimeError('strided-rs: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_perm_transpose_3d_256_102_strided-rs (8 threads)
```
RuntimeError('strided-rs: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_perm_tn_light_415_24d_scattered_to_colmajor_materialize (8 threads)
```
RuntimeError('strided-rs: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_perm_tn_light_415_24d_scattered_to_colmajor_strided-rs (8 threads)
```
RuntimeError('strided-rs: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_perm_tn_light_415_24d_contiguous_same_perm_materialize (8 threads)
```
RuntimeError('strided-rs: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_perm_tn_light_415_24d_contiguous_same_perm_strided-rs (8 threads)
```
RuntimeError('strided-rs: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_perm_rotation_6d_32_32_32_32_16_16_materialize (8 threads)
```
RuntimeError('strided-rs: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_perm_rotation_6d_32_32_32_32_16_16_strided-rs (8 threads)
```
RuntimeError('strided-rs: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_structural_extract_diagonal_direct-shared (8 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_structural_extract_diagonal_jax (8 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_fft_irfft_direct-shared (8 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_fft_irfft_mkl-dfti (8 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_fft_rfft_direct-shared (8 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_fft_rfft_mkl-dfti (8 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_fft_fft_direct-shared (8 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_fft_fft_mkl-dfti (8 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_fft_ifft_direct-shared (8 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_fft_ifft_mkl-dfti (8 threads)
```
RuntimeError('direct-shared: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_metadata_transpose_view_metadata (8 threads)
```
RuntimeError('metadata: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_metadata_transpose_view_julia-base (8 threads)
```
RuntimeError('metadata: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_metadata_transpose_view_metadata-host-dyn (8 threads)
```
RuntimeError('metadata-host-dyn: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_metadata_transpose_view_metadata-host-static (8 threads)
```
RuntimeError('metadata-host-static: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_metadata_slice_view_metadata (8 threads)
```
RuntimeError('metadata: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_metadata_slice_view_julia-base (8 threads)
```
RuntimeError('metadata: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_metadata_slice_view_metadata-host-dyn (8 threads)
```
RuntimeError('metadata-host-dyn: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_metadata_slice_view_metadata-host-static (8 threads)
```
RuntimeError('metadata-host-static: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_metadata_reshape_view_metadata (8 threads)
```
RuntimeError('metadata: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_metadata_reshape_view_julia-base (8 threads)
```
RuntimeError('metadata: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_metadata_reshape_view_metadata-host-dyn (8 threads)
```
RuntimeError('metadata-host-dyn: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```

### FAILED: followup_metadata_reshape_view_metadata-host-static (8 threads)
```
RuntimeError('metadata-host-static: /home/shinaoka/projects/tensor4all/tenferro-benchmark/target/followup-current/release/cpu-followup-mwe: error while loading shared libraries: libmkl_intel_lp64.so.3: cannot open shared object file: No such file or directory')
```
