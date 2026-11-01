# CPU Performance-Issue Workloads

- Suite: `cpu/perf_issues`
- Raw run: `data/results/amd-cpu/cpu/perf_issues/20261101_030047`

## CPU Information

- Model: `AMD EPYC 7763 64-Core Processor`
- Vendor: `AuthenticAMD`
- Logical CPUs: `4`
- Sockets: `1`
- Cores per socket: `2`
- Threads per core: `2`
- NUMA nodes: `1`
- Transparent huge pages: `madvise (always [madvise] never)`
- Transparent huge page defrag: `madvise (always defer defer+madvise [madvise] never)`
- Python platform: `Linux-7.0.0-1017-azure-x86_64-with-glibc2.39`
- L3 domains (shared CPUs): `0-3`

## run_t1

Full metadata: `data/results/amd-cpu/cpu/perf_issues/20261101_030047/run_t1.yaml`

```yaml
timestamp: '2026-11-01T03:00:47Z'
tenferro_rs:
  path: /workspaces/tenferro-benchmark/extern/tenferro-rs
  commit: a9f494cc4679387c75af698b299c316fd2295fe4
  dirty: false
  features:
  - blas-mkl
  resolved_path: /workspaces/tenferro-benchmark/extern/tenferro-rs
blas:
  implementation: mkl
  version: 2026.0.2
  root: /opt/intel/oneapi/mkl/latest
  library: /opt/intel/oneapi/mkl/latest/lib/libmkl_rt.so
environment:
  hostname: 8e87612b3072
  os: Linux-7.0.0-1017-azure-x86_64-with-glibc2.39
  arch: x86_64
  cpu: x86_64
  affinity:
    pinning: none
    requested_measured_cpus: null
    observer_cpus: null
    override_of: null
    thread_budget: '1'
    measured_cpus: 0-3
    l3_domains:
    - 0-3
    cpu_boost: null
    scaling_governor: null
    scaling_driver: null
  env:
    OMP_NUM_THREADS: '1'
    OMP_THREAD_LIMIT: '1'
    OMP_DYNAMIC: 'FALSE'
    RAYON_NUM_THREADS: '1'
    BENCHMARK_TARGET_PROFILE: amd-cpu
    BENCHMARK_COMMIT: 4cc17b1c7061855cfd9ddc2d549a09f50199de3c
    OPENBLAS_ROOT: /opt/openblas
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
    PUBLICATION_GATE_PROFILE: quick
    USE_CUDA: '0'
    BENCHMARK_BUILD_PROFILE: release
    RUSTC_VERSION: rustc 1.99.1 (58cf8f9c3 2026-10-13)
    BENCH_INSTANCE: null
```

## run_t4

Full metadata: `data/results/amd-cpu/cpu/perf_issues/20261101_030047/run_t4.yaml`

```yaml
timestamp: '2026-11-01T03:04:17Z'
tenferro_rs:
  path: /workspaces/tenferro-benchmark/extern/tenferro-rs
  commit: a9f494cc4679387c75af698b299c316fd2295fe4
  dirty: false
  features:
  - blas-mkl
  resolved_path: /workspaces/tenferro-benchmark/extern/tenferro-rs
blas:
  implementation: mkl
  version: 2026.0.2
  root: /opt/intel/oneapi/mkl/latest
  library: /opt/intel/oneapi/mkl/latest/lib/libmkl_rt.so
environment:
  hostname: 8e87612b3072
  os: Linux-7.0.0-1017-azure-x86_64-with-glibc2.39
  arch: x86_64
  cpu: x86_64
  affinity:
    pinning: none
    requested_measured_cpus: null
    observer_cpus: null
    override_of: null
    thread_budget: '4'
    measured_cpus: 0-3
    l3_domains:
    - 0-3
    cpu_boost: null
    scaling_governor: null
    scaling_driver: null
  env:
    OMP_NUM_THREADS: '4'
    OMP_THREAD_LIMIT: '4'
    OMP_DYNAMIC: 'FALSE'
    RAYON_NUM_THREADS: '4'
    BENCHMARK_TARGET_PROFILE: amd-cpu
    BENCHMARK_COMMIT: 4cc17b1c7061855cfd9ddc2d549a09f50199de3c
    OPENBLAS_ROOT: /opt/openblas
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
    PUBLICATION_GATE_PROFILE: quick
    USE_CUDA: '0'
    BENCHMARK_BUILD_PROFILE: release
    RUSTC_VERSION: rustc 1.99.1 (58cf8f9c3 2026-10-13)
    BENCH_INSTANCE: null
```

## Case status

- Manifest: `cpu/perf_issues` version 21, coverage `quick`, effort `standard`
- Expected 622, selected 622, executed 434, unsupported 0, failed 188, skipped 0, missing 0, noisy 48.
- Complete: **NO** — only executed cases count as covered; unsupported, skipped, failed, missing and unselected cases never do.

| expected | selected | executed | unsupported | failed | skipped | missing | noisy |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 622 | 622 | 434 | 0 | 188 | 0 | 0 | 48 |

- Failed: `activation_erf_f32_eager-single-call_1024x64@t1`, `activation_erf_f32_pytorch_1024x64@t1`, `activation_sigmoid_f32_eager-single-call_1024x64@t1`, `activation_sigmoid_f32_pytorch_1024x64@t1`, `activation_silu_f32_eager-single-call_1024x64@t1`, `activation_silu_f32_pytorch_1024x64@t1`, `activation_softplus_f32_eager-single-call_1024x64@t1`, `activation_softplus_f32_pytorch_1024x64@t1`, `activation_gelu_tanh_f32_eager-single-call_1024x64@t1`, `activation_gelu_tanh_f32_pytorch_1024x64@t1`, `followup_structural_broadcast_in_dim_direct-shared@t1`, `followup_structural_broadcast_in_dim_pytorch@t1`, `followup_real_tanh_direct-shared@t1`, `followup_real_tanh_jax@t1`, `followup_real_log1p_direct-shared@t1`, `followup_real_log1p_pytorch@t1`, `followup_real_cos_direct-shared@t1`, `followup_real_cos_pytorch@t1`, `followup_real_sin_direct-shared@t1`, `followup_real_sin_pytorch@t1`, `followup_real_exp_direct-shared@t1`, `followup_real_exp_jax@t1`, `followup_real_expm1_direct-shared@t1`, `followup_real_expm1_jax@t1`, `followup_real_pow_direct-shared@t1`, `followup_real_pow_pytorch@t1`, `followup_complex_exp_direct-shared@t1`, `followup_complex_exp_julia-base@t1`, `followup_real_log_direct-shared@t1`, `followup_real_log_pytorch@t1`, `followup_real_reduce_max_axis1_direct-shared@t1`, `followup_real_reduce_max_axis1_julia-base@t1`, `followup_real_reduce_prod_axis1_direct-shared@t1`, `followup_real_reduce_prod_axis1_pytorch@t1`, `followup_real_reduce_sum_axis1_direct-shared@t1`, `followup_real_reduce_sum_axis1_pytorch@t1`, `followup_real_reduce_max_axis0_direct-shared@t1`, `followup_real_reduce_max_axis0_julia-base@t1`, `followup_real_reduce_min_axis1_direct-shared@t1`, `followup_real_reduce_min_axis1_julia-base@t1` (+148 more)
- Noisy: `conj_dot_f64_pytorch-vdot_len10000@t1`, `small_solve_f64_triangular-solve-loop_k16_b64@t1`, `small_solve_f64_triangular-solve-loop_k64_b64@t1`, `metadata_transpose_view_f64_rank8_t1@t1`, `metadata_transpose_view_f64_rank8_t1_julia-base@t1`, `analytic_neg_f64_sweep_8MiB_t1@t1`, `degenerate_dot_outer_f64_dot-general-shared_n1024@t1`, `degenerate_dot_rank0_f64_pytorch_n1024@t1`, `axpby_strided_source_f64_1024x1024_t1@t1`, `erased_construct_f64_col_n8@t1`, `erased_construct_f64_row_n8@t1`, `decode_proj_f32_host-sgemm_in1024_out4096_len512@t4`, `conj_dot_f64_pytorch-vdot_len1000000@t4`, `conj_dot_c64_pytorch-vdot_len100@t4`, `conj_dot_c64_pytorch-vdot_len1000000@t4`, `small_solve_f64_solve-batched_k16_b64@t4`, `small_solve_f64_triangular-solve-loop_k64_b64@t4`, `layer_norm_f32_eager-composed_d1024_len8_b1@t4`, `layer_norm_f32_pytorch-fused_d1024_len8_b1@t4`, `layer_norm_f32_eager-single-call_d1024_len8_b1@t4`, `rms_norm_f32_eager-composed_d1024_len8_b1@t4`, `rms_norm_f32_eager-single-call_d1024_len8_b1@t4`, `rms_norm_f32_eager-single-call_d1024_len8_b8@t4`, `activation_sigmoid_f32_eager-composed_1024x64@t4`, `activation_silu_f32_eager-composed_1024x64@t4`, `softmax_f32_session-single-call_len512_b8@t4`, `log_softmax_f32_eager-single-call_len64_b8@t4`, `masked_softmax_f32_session-single-call_len64_b8@t4`, `masked_softmax_f32_pytorch_len512_b8@t4`, `reduce_mean_f32_session-single-call_1024x64_axis0@t4`, `lstsq_f64_eager-shared_m768_n384_rhs16@t4`, `triangular_f64_pytorch_n4096_rhs64@t4`, `metadata_transpose_view_f64_rank8_t4@t4`, `metadata_transpose_view_f64_rank8_t4_julia-base@t4`, `lu_batched_f64_solve_n16_b1024_pytorch@t4`, `degenerate_dot_hadamard_f64_dot-general-shared_n1024@t4`, `degenerate_dot_hadamard_f64_pytorch_n1024@t4`, `degenerate_dot_outer_f64_dot-general-shared_n1024@t4`, `degenerate_dot_outer_f64_pytorch_n1024@t4`, `degenerate_dot_rank0_f64_pytorch_n1024@t4` (+8 more)

Numerical checks: **434/622 recorded rows passed** (case × thread count).

## Steady-state operation rows

Each batch is one wall-clock interval over many operations; every output is retained until the clock stops. Median and inclusive IQR are per operation, in nanoseconds; CoV is sample standard deviation / mean (NOISY means CoV > 10%, a descriptive label only). Setup (inputs, backends, runtimes, session entry, tracing/compilation, preparation) is outside timing; faer-direct and host-sgemm write into a preallocated output, the tenferro and PyTorch arms allocate their output. `reference_for` in the case file names the case each reference arm is compared with. GPU batches end with one device synchronize (a per-call synchronize where the transfer is the operation); the Threads column is the host thread setting.

### Compute- and bandwidth-bound operations

| Case ID | Issues | Arm | Backend | Dtype | Workload | Threads | Ops/batch | Median ns/op | IQR ns | CoV | Status |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| softmax_f64_pool_serial_1024x4096 | #2273 #2274 #2276 | serial | tenferro-rs | f64 | rows=1024, cols=4096, axis=0 | 1 | 1 | 17708517.0 | 72024.0 | 0.3% | ok |
| softmax_f64_pool_pytorch_t1_1024x4096 | #2273 #2274 #2276 | pytorch | pytorch-cpu | f64 | rows=1024, cols=4096, axis=0 | 1 | 1 | 25629534.0 | 147975.5 | 0.5% | ok |
| softmax_f64_pool_pool4_1024x4096 | #2273 #2274 #2276 | pool4 | tenferro-rs | f64 | rows=1024, cols=4096, axis=0 | 4 | 1 | 6764894.0 | 62121.0 | 5.5% | ok |
| softmax_f64_pool_pytorch_t4_1024x4096 | #2273 #2274 #2276 | pytorch | pytorch-cpu | f64 | rows=1024, cols=4096, axis=0 | 4 | 1 | 9696979.0 | 186572.5 | 1.8% | ok |

### Latency and per-call overhead

These rows measure dispatch/operation latency; they do not measure compute throughput.

| Case ID | Issues | Arm | Backend | Dtype | Workload | Threads | Ops/batch | Median ns/op | IQR ns | CoV | Status |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| decode_proj_f32_eager-shared_in1024_out1024_len8 | #1992 #2003 #1995 #2229 | eager-shared | tenferro-rs | f32 | in=1024, out=1024, len=8 | 1 | 2 | 995705.5 | 3854.5 | 0.2% | ok |
| decode_proj_f32_faer-direct_in1024_out1024_len8 | #1992 #2003 #1995 #2229 | faer-direct | faer-direct | f32 | in=1024, out=1024, len=8 | 1 | 8 | 352409.1 | 2723.1 | 0.7% | ok |
| decode_proj_f32_host-sgemm_in1024_out1024_len8 | #1992 #2003 #1995 #2229 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=1024, len=8 | 1 | 8 | 486514.0 | 9194.6 | 1.3% | ok |
| decode_proj_f32_pytorch_in1024_out1024_len8 | #1992 #2003 #1995 #2229 | pytorch | pytorch-cpu | f32 | in=1024, out=1024, len=8 | 1 | 8 | 452609.5 | 12070.1 | 1.9% | ok |
| decode_proj_f32_eager-shared_in1024_out1024_len64 | #1992 #2003 #1995 #2229 | eager-shared | tenferro-rs | f32 | in=1024, out=1024, len=64 | 1 | 2 | 1651091.5 | 25670.2 | 1.1% | ok |
| decode_proj_f32_faer-direct_in1024_out1024_len64 | #1992 #2003 #1995 #2229 | faer-direct | faer-direct | f32 | in=1024, out=1024, len=64 | 1 | 2 | 1411715.5 | 10985.2 | 0.6% | ok |
| decode_proj_f32_host-sgemm_in1024_out1024_len64 | #1992 #2003 #1995 #2229 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=1024, len=64 | 1 | 2 | 1778990.0 | 5026.8 | 0.3% | ok |
| decode_proj_f32_pytorch_in1024_out1024_len64 | #1992 #2003 #1995 #2229 | pytorch | pytorch-cpu | f32 | in=1024, out=1024, len=64 | 1 | 2 | 1678808.0 | 10882.8 | 0.4% | ok |
| decode_proj_f32_eager-shared_in1024_out1024_len512 | #1992 #2003 #1995 #2229 | eager-shared | tenferro-rs | f32 | in=1024, out=1024, len=512 | 1 | 1 | 11401665.0 | 48159.5 | 0.3% | ok |
| decode_proj_f32_faer-direct_in1024_out1024_len512 | #1992 #2003 #1995 #2229 | faer-direct | faer-direct | f32 | in=1024, out=1024, len=512 | 1 | 1 | 11923888.0 | 60733.0 | 0.3% | ok |
| decode_proj_f32_host-sgemm_in1024_out1024_len512 | #1992 #2003 #1995 #2229 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=1024, len=512 | 1 | 1 | 12337338.0 | 50148.5 | 0.3% | ok |
| decode_proj_f32_pytorch_in1024_out1024_len512 | #1992 #2003 #1995 #2229 | pytorch | pytorch-cpu | f32 | in=1024, out=1024, len=512 | 1 | 1 | 11293333.0 | 48730.5 | 0.6% | ok |
| decode_proj_f32_eager-shared_in1024_out4096_len8 | #1992 #2003 #1995 #2229 | eager-shared | tenferro-rs | f32 | in=1024, out=4096, len=8 | 1 | 1 | 4135717.0 | 73827.5 | 1.1% | ok |
| decode_proj_f32_faer-direct_in1024_out4096_len8 | #1992 #2003 #1995 #2229 | faer-direct | faer-direct | f32 | in=1024, out=4096, len=8 | 1 | 2 | 1484291.0 | 6607.8 | 1.1% | ok |
| decode_proj_f32_host-sgemm_in1024_out4096_len8 | #1992 #2003 #1995 #2229 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=4096, len=8 | 1 | 1 | 2020525.0 | 20228.0 | 0.9% | ok |
| decode_proj_f32_pytorch_in1024_out4096_len8 | #1992 #2003 #1995 #2229 | pytorch | pytorch-cpu | f32 | in=1024, out=4096, len=8 | 1 | 1 | 1961304.0 | 112379.5 | 3.8% | ok |
| decode_proj_f32_eager-shared_in1024_out4096_len64 | #1992 #2003 #1995 #2229 | eager-shared | tenferro-rs | f32 | in=1024, out=4096, len=64 | 1 | 1 | 6615217.0 | 62543.5 | 1.5% | ok |
| decode_proj_f32_faer-direct_in1024_out4096_len64 | #1992 #2003 #1995 #2229 | faer-direct | faer-direct | f32 | in=1024, out=4096, len=64 | 1 | 1 | 5606807.0 | 16656.0 | 0.4% | ok |
| decode_proj_f32_host-sgemm_in1024_out4096_len64 | #1992 #2003 #1995 #2229 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=4096, len=64 | 1 | 1 | 7270076.0 | 140611.5 | 1.3% | ok |
| decode_proj_f32_pytorch_in1024_out4096_len64 | #1992 #2003 #1995 #2229 | pytorch | pytorch-cpu | f32 | in=1024, out=4096, len=64 | 1 | 1 | 6843301.0 | 103868.5 | 1.8% | ok |
| decode_proj_f32_eager-shared_in1024_out4096_len512 | #1992 #2003 #1995 #2229 | eager-shared | tenferro-rs | f32 | in=1024, out=4096, len=512 | 1 | 1 | 49455071.0 | 1401497.0 | 2.4% | ok |
| decode_proj_f32_faer-direct_in1024_out4096_len512 | #1992 #2003 #1995 #2229 | faer-direct | faer-direct | f32 | in=1024, out=4096, len=512 | 1 | 1 | 48822412.0 | 717211.5 | 0.9% | ok |
| decode_proj_f32_host-sgemm_in1024_out4096_len512 | #1992 #2003 #1995 #2229 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=4096, len=512 | 1 | 1 | 49859395.0 | 217575.0 | 0.3% | ok |
| decode_proj_f32_pytorch_in1024_out4096_len512 | #1992 #2003 #1995 #2229 | pytorch | pytorch-cpu | f32 | in=1024, out=4096, len=512 | 1 | 1 | 46035089.0 | 406832.5 | 0.8% | ok |
| decode_proj_view_operand_f32_prepared_in1024_out1024_len8_t1 | #2216 | prepared-view | tenferro-rs | f32 | in=1024, out=1024, len=8 | 1 | 1 | 3473943.0 | 7899.5 | 0.2% | ok |
| decode_proj_contiguous_operand_f32_prepared_in1024_out1024_len8_t1 | #2216 | prepared-contiguous | tenferro-rs | f32 | in=1024, out=1024, len=8 | 1 | 2 | 997358.5 | 2296.8 | 2.3% | ok |
| decode_proj_view_operand_f32_pytorch_in1024_out1024_len8_t1 | #2216 | pytorch | pytorch-cpu | f32 | in=1024, out=1024, len=8 | 1 | 8 | 462401.6 | 16334.2 | 2.4% | ok |
| gemm_c64_mm256_t1 | #1900 | dot-general-shared | tenferro-rs | c64 | n=256 | 1 | 1 | 2969934.0 | 110680.5 | 2.1% | ok |
| gemm_f64_mm256_t1 | #1900 | dot-general-shared | tenferro-rs | f64 | n=256 | 1 | 4 | 831527.0 | 13719.2 | 1.1% | ok |
| conj_dot_f64_dot-general-rank1_len100 | #1615 | dot-general-rank1 | tenferro-rs | f64 | len=100 | 1 | 2048 | 1289.0 | 10.3 | 3.0% | ok |
| conj_dot_f64_portable-loop_len100 | #1615 | portable-loop | rust-loop | f64 | len=100 | 1 | 32768 | 75.3 | 0.9 | 1.4% | ok |
| conj_dot_f64_pytorch-vdot_len100 | #1615 | pytorch-vdot | pytorch-cpu | f64 | len=100 | 1 | 2048 | 1348.7 | 2.5 | 1.7% | ok |
| conj_dot_f64_dot-general-rank1_len10000 | #1615 | dot-general-rank1 | tenferro-rs | f64 | len=10000 | 1 | 512 | 6052.9 | 20.7 | 0.8% | ok |
| conj_dot_f64_portable-loop_len10000 | #1615 | portable-loop | rust-loop | f64 | len=10000 | 1 | 256 | 9342.4 | 38.0 | 0.4% | ok |
| conj_dot_f64_pytorch-vdot_len10000 | #1615 | pytorch-vdot | pytorch-cpu | f64 | len=10000 | 1 | 1024 | 3069.3 | 27.7 | 182.6% | NOISY |
| conj_dot_f64_dot-general-rank1_len1000000 | #1615 | dot-general-rank1 | tenferro-rs | f64 | len=1000000 | 1 | 8 | 477463.4 | 2249.2 | 0.7% | ok |
| conj_dot_f64_portable-loop_len1000000 | #1615 | portable-loop | rust-loop | f64 | len=1000000 | 1 | 4 | 941123.5 | 5714.1 | 0.4% | ok |
| conj_dot_f64_pytorch-vdot_len1000000 | #1615 | pytorch-vdot | pytorch-cpu | f64 | len=1000000 | 1 | 8 | 277989.1 | 12420.1 | 4.0% | ok |
| conj_dot_c64_dot-general-rank1_len100 | #1615 | dot-general-rank1 | tenferro-rs | c64 | len=100 | 1 | 2048 | 1375.3 | 11.6 | 1.2% | ok |
| conj_dot_c64_portable-loop_len100 | #1615 | portable-loop | rust-loop | c64 | len=100 | 1 | 32768 | 110.9 | 4.2 | 3.1% | ok |
| conj_dot_c64_pytorch-vdot_len100 | #1615 | pytorch-vdot | pytorch-cpu | c64 | len=100 | 1 | 2048 | 1286.5 | 7.1 | 0.3% | ok |
| conj_dot_c64_dot-general-rank1_len10000 | #1615 | dot-general-rank1 | tenferro-rs | c64 | len=10000 | 1 | 256 | 10989.1 | 92.4 | 0.7% | ok |
| conj_dot_c64_portable-loop_len10000 | #1615 | portable-loop | rust-loop | c64 | len=10000 | 1 | 256 | 9787.5 | 90.6 | 1.0% | ok |
| conj_dot_c64_pytorch-vdot_len10000 | #1615 | pytorch-vdot | pytorch-cpu | c64 | len=10000 | 1 | 512 | 7453.2 | 39.8 | 0.7% | ok |
| conj_dot_c64_dot-general-rank1_len1000000 | #1615 | dot-general-rank1 | tenferro-rs | c64 | len=1000000 | 1 | 2 | 1092681.0 | 15847.0 | 1.3% | ok |
| conj_dot_c64_portable-loop_len1000000 | #1615 | portable-loop | rust-loop | c64 | len=1000000 | 1 | 2 | 1144497.5 | 24813.8 | 2.0% | ok |
| conj_dot_c64_pytorch-vdot_len1000000 | #1615 | pytorch-vdot | pytorch-cpu | c64 | len=1000000 | 1 | 2 | 946153.0 | 21437.8 | 2.6% | ok |
| small_solve_f64_solve-loop_k16_b64 | #2007 #1878 | solve-loop | tenferro-rs | f64 | k=16, batch=64 | 1 | 8 | 272158.2 | 3556.0 | 1.6% | ok |
| small_solve_f64_triangular-solve-loop_k16_b64 | #2007 #1878 | triangular-solve-loop | tenferro-rs | f64 | k=16, batch=64 | 1 | 16 | 176756.2 | 3922.4 | 22.8% | NOISY |
| small_solve_f64_solve-batched_k16_b64 | #2007 #1878 | solve-batched | tenferro-rs | f64 | k=16, batch=64 | 1 | 16 | 253985.8 | 1384.8 | 0.7% | ok |
| small_solve_f64_pytorch-solve-batched_k16_b64 | #2007 #1878 | pytorch-solve-batched | pytorch-cpu | f64 | k=16, batch=64 | 1 | 16 | 191397.9 | 2158.4 | 1.0% | ok |
| small_solve_f64_pytorch-solve-triangular-batched_k16_b64 | #2007 #1878 | pytorch-solve-triangular-batched | pytorch-cpu | f64 | k=16, batch=64 | 1 | 32 | 71384.3 | 635.2 | 0.6% | ok |
| small_solve_f64_solve-loop_k64_b64 | #2007 #1878 | solve-loop | tenferro-rs | f64 | k=64, batch=64 | 1 | 1 | 5002451.0 | 8059.5 | 0.2% | ok |
| small_solve_f64_triangular-solve-loop_k64_b64 | #2007 #1878 | triangular-solve-loop | tenferro-rs | f64 | k=64, batch=64 | 1 | 1 | 1511286.0 | 32580.5 | 13.5% | NOISY |
| small_solve_f64_solve-batched_k64_b64 | #2007 #1878 | solve-batched | tenferro-rs | f64 | k=64, batch=64 | 1 | 1 | 6559032.0 | 24926.5 | 0.5% | ok |
| small_solve_f64_pytorch-solve-batched_k64_b64 | #2007 #1878 | pytorch-solve-batched | pytorch-cpu | f64 | k=64, batch=64 | 1 | 1 | 4866428.0 | 101609.5 | 1.4% | ok |
| small_solve_f64_pytorch-solve-triangular-batched_k64_b64 | #2007 #1878 | pytorch-solve-triangular-batched | pytorch-cpu | f64 | k=64, batch=64 | 1 | 2 | 1501437.5 | 24082.5 | 3.1% | ok |
| tanh_chain_f32_compiled-prepared_k10_1024x64 | #1990 | compiled-prepared | tenferro-rs | f32 | rows=1024, cols=64, chain=10 | 1 | 2 | 1715251.0 | 11772.0 | 0.7% | ok |
| tanh_chain_f32_eager-shared_k10_1024x64 | #1990 | eager-shared | tenferro-rs | f32 | rows=1024, cols=64, chain=10 | 1 | 2 | 1705708.5 | 5790.8 | 0.4% | ok |
| tanh_chain_f32_compiled-prepared_k100_1024x64 | #1990 | compiled-prepared | tenferro-rs | f32 | rows=1024, cols=64, chain=100 | 1 | 1 | 17014904.0 | 37370.5 | 0.2% | ok |
| tanh_chain_f32_eager-shared_k100_1024x64 | #1990 | eager-shared | tenferro-rs | f32 | rows=1024, cols=64, chain=100 | 1 | 1 | 17055860.0 | 33818.0 | 0.5% | ok |
| tanh_chain_f32_compiled-prepared_k200_1024x64 | #1990 | compiled-prepared | tenferro-rs | f32 | rows=1024, cols=64, chain=200 | 1 | 1 | 34069421.0 | 31268.5 | 0.1% | ok |
| tanh_chain_f32_eager-shared_k200_1024x64 | #1990 | eager-shared | tenferro-rs | f32 | rows=1024, cols=64, chain=200 | 1 | 1 | 34141045.0 | 45284.0 | 0.3% | ok |
| layer_norm_f32_eager-composed_d1024_len8_b1 | #2006 | eager-composed | tenferro-rs | f32 | norm=layer_norm, d=1024, len=8, batch=1 | 1 | 32 | 70369.8 | 630.5 | 1.4% | ok |
| layer_norm_f32_pytorch-fused_d1024_len8_b1 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=layer_norm, d=1024, len=8, batch=1 | 1 | 512 | 7104.7 | 340.0 | 3.8% | ok |
| layer_norm_f32_eager-single-call_d1024_len8_b1 | #2006 | eager-single-call | tenferro-rs | f32 | norm=layer_norm, d=1024, len=8, batch=1 | 1 | 32 | 75480.6 | 775.7 | 0.8% | ok |
| layer_norm_f32_eager-composed_d1024_len8_b8 | #2006 | eager-composed | tenferro-rs | f32 | norm=layer_norm, d=1024, len=8, batch=8 | 1 | 16 | 120506.4 | 2064.2 | 2.5% | ok |
| layer_norm_f32_pytorch-fused_d1024_len8_b8 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=layer_norm, d=1024, len=8, batch=8 | 1 | 128 | 31516.1 | 1057.1 | 2.5% | ok |
| layer_norm_f32_eager-single-call_d1024_len8_b8 | #2006 | eager-single-call | tenferro-rs | f32 | norm=layer_norm, d=1024, len=8, batch=8 | 1 | 16 | 124454.4 | 4056.0 | 2.4% | ok |
| layer_norm_f32_eager-composed_d1024_len64_b1 | #2006 | eager-composed | tenferro-rs | f32 | norm=layer_norm, d=1024, len=64, batch=1 | 1 | 16 | 119285.4 | 4974.9 | 2.9% | ok |
| layer_norm_f32_pytorch-fused_d1024_len64_b1 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=layer_norm, d=1024, len=64, batch=1 | 1 | 64 | 27282.4 | 700.0 | 9.6% | ok |
| layer_norm_f32_eager-single-call_d1024_len64_b1 | #2006 | eager-single-call | tenferro-rs | f32 | norm=layer_norm, d=1024, len=64, batch=1 | 1 | 16 | 123527.6 | 2176.6 | 2.1% | ok |
| layer_norm_f32_eager-composed_d1024_len64_b8 | #2006 | eager-composed | tenferro-rs | f32 | norm=layer_norm, d=1024, len=64, batch=8 | 1 | 4 | 470895.0 | 16192.5 | 3.0% | ok |
| layer_norm_f32_pytorch-fused_d1024_len64_b8 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=layer_norm, d=1024, len=64, batch=8 | 1 | 16 | 226776.9 | 3381.6 | 1.5% | ok |
| layer_norm_f32_eager-single-call_d1024_len64_b8 | #2006 | eager-single-call | tenferro-rs | f32 | norm=layer_norm, d=1024, len=64, batch=8 | 1 | 4 | 459924.5 | 8984.1 | 2.8% | ok |
| rms_norm_f32_eager-composed_d1024_len8_b1 | #2006 | eager-composed | tenferro-rs | f32 | norm=rms_norm, d=1024, len=8, batch=1 | 1 | 64 | 36236.9 | 880.3 | 1.7% | ok |
| rms_norm_f32_pytorch-fused_d1024_len8_b1 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=rms_norm, d=1024, len=8, batch=1 | 1 | 128 | 19975.5 | 180.5 | 1.2% | ok |
| rms_norm_f32_eager-single-call_d1024_len8_b1 | #2006 | eager-single-call | tenferro-rs | f32 | norm=rms_norm, d=1024, len=8, batch=1 | 1 | 64 | 44045.7 | 485.1 | 1.2% | ok |
| rms_norm_f32_eager-composed_d1024_len8_b8 | #2006 | eager-composed | tenferro-rs | f32 | norm=rms_norm, d=1024, len=8, batch=8 | 1 | 16 | 63262.9 | 1724.5 | 2.4% | ok |
| rms_norm_f32_pytorch-fused_d1024_len8_b8 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=rms_norm, d=1024, len=8, batch=8 | 1 | 64 | 47345.1 | 701.8 | 2.0% | ok |
| rms_norm_f32_eager-single-call_d1024_len8_b8 | #2006 | eager-single-call | tenferro-rs | f32 | norm=rms_norm, d=1024, len=8, batch=8 | 1 | 16 | 68783.8 | 1828.7 | 2.7% | ok |
| rms_norm_f32_eager-composed_d1024_len64_b1 | #2006 | eager-composed | tenferro-rs | f32 | norm=rms_norm, d=1024, len=64, batch=1 | 1 | 16 | 59224.7 | 1676.6 | 3.5% | ok |
| rms_norm_f32_pytorch-fused_d1024_len64_b1 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=rms_norm, d=1024, len=64, batch=1 | 1 | 64 | 47538.0 | 1205.9 | 4.3% | ok |
| rms_norm_f32_eager-single-call_d1024_len64_b1 | #2006 | eager-single-call | tenferro-rs | f32 | norm=rms_norm, d=1024, len=64, batch=1 | 1 | 16 | 65999.2 | 1779.0 | 2.0% | ok |
| rms_norm_f32_eager-composed_d1024_len64_b8 | #2006 | eager-composed | tenferro-rs | f32 | norm=rms_norm, d=1024, len=64, batch=8 | 1 | 4 | 228823.8 | 10177.6 | 3.1% | ok |
| rms_norm_f32_pytorch-fused_d1024_len64_b8 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=rms_norm, d=1024, len=64, batch=8 | 1 | 8 | 316901.8 | 5530.3 | 2.0% | ok |
| rms_norm_f32_eager-single-call_d1024_len64_b8 | #2006 | eager-single-call | tenferro-rs | f32 | norm=rms_norm, d=1024, len=64, batch=8 | 1 | 4 | 240605.8 | 9739.4 | 3.5% | ok |
| small_contraction_abcd-dbef-acef_f64_shared-session-default-pool_d4 | #1885 #1897 #1901 | shared-session-default-pool | tenferro-rs | f64 | extent=4 | 1 | 512 | 3235.2 | 18.2 | 1.1% | ok |
| activation_erf_f32_eager-single-call_1024x64 | #1975 #2069 #2032 | eager-single-call | tenferro-rs | f32 | op=erf, rows=1024, cols=64 | 1 | — | — | — | — | FAILED |
| activation_erf_f32_session-single-call_1024x64 | #1975 #2069 #2032 | session-single-call | tenferro-rs | f32 | op=erf, rows=1024, cols=64 | 1 | 8 | 316352.0 | 1583.0 | 0.5% | ok |
| activation_erf_f32_pytorch_1024x64 | #1975 #2069 #2032 | pytorch | pytorch-cpu | f32 | op=erf, rows=1024, cols=64 | 1 | — | — | — | — | FAILED |
| activation_sigmoid_f32_eager-single-call_1024x64 | #1975 #2069 #2032 | eager-single-call | tenferro-rs | f32 | op=sigmoid, rows=1024, cols=64 | 1 | — | — | — | — | FAILED |
| activation_sigmoid_f32_session-single-call_1024x64 | #1975 #2069 #2032 | session-single-call | tenferro-rs | f32 | op=sigmoid, rows=1024, cols=64 | 1 | 32 | 53728.6 | 433.5 | 0.7% | ok |
| activation_sigmoid_f32_eager-composed_1024x64 | #1975 #2069 #2032 | eager-composed | tenferro-rs | f32 | op=sigmoid, rows=1024, cols=64 | 1 | 16 | 90322.7 | 3726.6 | 6.1% | ok |
| activation_sigmoid_f32_pytorch_1024x64 | #1975 #2069 #2032 | pytorch | pytorch-cpu | f32 | op=sigmoid, rows=1024, cols=64 | 1 | — | — | — | — | FAILED |
| activation_silu_f32_eager-single-call_1024x64 | #1975 #2069 #2032 | eager-single-call | tenferro-rs | f32 | op=silu, rows=1024, cols=64 | 1 | — | — | — | — | FAILED |
| activation_silu_f32_session-single-call_1024x64 | #1975 #2069 #2032 | session-single-call | tenferro-rs | f32 | op=silu, rows=1024, cols=64 | 1 | 32 | 55796.1 | 249.1 | 0.4% | ok |
| activation_silu_f32_eager-composed_1024x64 | #1975 #2069 #2032 | eager-composed | tenferro-rs | f32 | op=silu, rows=1024, cols=64 | 1 | 16 | 103333.8 | 1592.3 | 1.5% | ok |
| activation_silu_f32_pytorch_1024x64 | #1975 #2069 #2032 | pytorch | pytorch-cpu | f32 | op=silu, rows=1024, cols=64 | 1 | — | — | — | — | FAILED |
| activation_softplus_f32_eager-single-call_1024x64 | #1975 #2069 #2032 | eager-single-call | tenferro-rs | f32 | op=softplus, rows=1024, cols=64 | 1 | — | — | — | — | FAILED |
| activation_softplus_f32_session-single-call_1024x64 | #1975 #2069 #2032 | session-single-call | tenferro-rs | f32 | op=softplus, rows=1024, cols=64 | 1 | 16 | 183174.9 | 960.5 | 0.6% | ok |
| activation_softplus_f32_eager-composed_1024x64 | #1975 #2069 #2032 | eager-composed | tenferro-rs | f32 | op=softplus, rows=1024, cols=64 | 1 | 16 | 183898.2 | 2043.8 | 1.1% | ok |
| activation_softplus_f32_pytorch_1024x64 | #1975 #2069 #2032 | pytorch | pytorch-cpu | f32 | op=softplus, rows=1024, cols=64 | 1 | — | — | — | — | FAILED |
| activation_gelu_f32_eager-single-call_1024x64 | #1975 #2030 #2069 #2228 | eager-single-call | tenferro-rs | f32 | op=gelu, rows=1024, cols=64 | 1 | 16 | 80170.6 | 635.2 | 1.0% | ok |
| activation_gelu_f32_session-single-call_1024x64 | #1975 #2030 #2069 #2228 | session-single-call | tenferro-rs | f32 | op=gelu, rows=1024, cols=64 | 1 | 16 | 78296.4 | 648.7 | 0.8% | ok |
| activation_gelu_f32_pytorch_1024x64 | #1975 #2030 #2069 #2228 | pytorch | pytorch-cpu | f32 | op=gelu, rows=1024, cols=64 | 1 | 32 | 65277.9 | 526.6 | 0.8% | ok |
| activation_gelu_tanh_f32_eager-single-call_1024x64 | #1975 #2069 #2032 | eager-single-call | tenferro-rs | f32 | op=gelu_tanh, rows=1024, cols=64 | 1 | — | — | — | — | FAILED |
| activation_gelu_tanh_f32_session-single-call_1024x64 | #1975 #2069 #2032 | session-single-call | tenferro-rs | f32 | op=gelu_tanh, rows=1024, cols=64 | 1 | 8 | 197059.6 | 1782.1 | 1.5% | ok |
| activation_gelu_tanh_f32_eager-composed_1024x64 | #1975 #2069 #2032 | eager-composed | tenferro-rs | f32 | op=gelu_tanh, rows=1024, cols=64 | 1 | 8 | 328346.6 | 6335.5 | 1.1% | ok |
| activation_gelu_tanh_f32_pytorch_1024x64 | #1975 #2069 #2032 | pytorch | pytorch-cpu | f32 | op=gelu_tanh, rows=1024, cols=64 | 1 | — | — | — | — | FAILED |
| softmax_f32_eager-single-call_len64_b8 | #1976 #2031 #2231 | eager-single-call | tenferro-rs | f32 | op=softmax, len=64, batch=8, axis=0 | 1 | 32 | 31853.7 | 470.7 | 1.3% | ok |
| softmax_f32_session-single-call_len64_b8 | #1976 #2031 #2231 | session-single-call | tenferro-rs | f32 | op=softmax, len=64, batch=8, axis=0 | 1 | 32 | 31529.1 | 356.0 | 1.0% | ok |
| softmax_f32_eager-composed_len64_b8 | #1976 #2031 #2231 | eager-composed | tenferro-rs | f32 | op=softmax, len=64, batch=8, axis=0 | 1 | 16 | 107975.5 | 2061.7 | 1.5% | ok |
| softmax_f32_pytorch_len64_b8 | #1976 #2031 #2231 | pytorch | pytorch-cpu | f32 | op=softmax, len=64, batch=8, axis=0 | 1 | 64 | 40956.5 | 157.1 | 1.4% | ok |
| softmax_f32_eager-single-call_len512_b8 | #1976 #2231 | eager-single-call | tenferro-rs | f32 | op=softmax, len=512, batch=8, axis=0 | 1 | 2 | 2014137.0 | 25094.5 | 0.9% | ok |
| softmax_f32_session-single-call_len512_b8 | #1976 #2231 | session-single-call | tenferro-rs | f32 | op=softmax, len=512, batch=8, axis=0 | 1 | 1 | 2021506.0 | 60842.5 | 3.0% | ok |
| softmax_f32_eager-composed_len512_b8 | #1976 #2231 | eager-composed | tenferro-rs | f32 | op=softmax, len=512, batch=8, axis=0 | 1 | 1 | 3403961.0 | 142109.5 | 4.5% | ok |
| softmax_f32_pytorch_len512_b8 | #1976 #2231 | pytorch | pytorch-cpu | f32 | op=softmax, len=512, batch=8, axis=0 | 1 | 1 | 2223302.0 | 20648.5 | 1.7% | ok |
| log_softmax_f32_eager-single-call_len64_b8 | #1976 | eager-single-call | tenferro-rs | f32 | op=log_softmax, len=64, batch=8, axis=0 | 1 | 16 | 131646.6 | 2357.2 | 1.2% | ok |
| log_softmax_f32_session-single-call_len64_b8 | #1976 | session-single-call | tenferro-rs | f32 | op=log_softmax, len=64, batch=8, axis=0 | 1 | 16 | 100566.1 | 2382.6 | 1.9% | ok |
| log_softmax_f32_eager-composed_len64_b8 | #1976 | eager-composed | tenferro-rs | f32 | op=log_softmax, len=64, batch=8, axis=0 | 1 | 16 | 104176.6 | 2052.9 | 1.9% | ok |
| log_softmax_f32_pytorch_len64_b8 | #1976 | pytorch | pytorch-cpu | f32 | op=log_softmax, len=64, batch=8, axis=0 | 1 | 64 | 34921.2 | 627.3 | 2.5% | ok |
| log_softmax_f32_eager-single-call_len512_b8 | #1976 | eager-single-call | tenferro-rs | f32 | op=log_softmax, len=512, batch=8, axis=0 | 1 | 1 | 3102370.0 | 155911.0 | 3.2% | ok |
| log_softmax_f32_session-single-call_len512_b8 | #1976 | session-single-call | tenferro-rs | f32 | op=log_softmax, len=512, batch=8, axis=0 | 1 | 1 | 3372614.0 | 97412.0 | 2.7% | ok |
| log_softmax_f32_eager-composed_len512_b8 | #1976 | eager-composed | tenferro-rs | f32 | op=log_softmax, len=512, batch=8, axis=0 | 1 | 1 | 3342948.0 | 78537.5 | 2.7% | ok |
| log_softmax_f32_pytorch_len512_b8 | #1976 | pytorch | pytorch-cpu | f32 | op=log_softmax, len=512, batch=8, axis=0 | 1 | 1 | 2065979.0 | 30221.0 | 1.3% | ok |
| masked_softmax_f32_eager-single-call_len64_b8 | #1976 | eager-single-call | tenferro-rs | f32 | op=masked_softmax, len=64, batch=8, axis=0 | 1 | 8 | 237915.8 | 3274.2 | 1.0% | ok |
| masked_softmax_f32_session-single-call_len64_b8 | #1976 | session-single-call | tenferro-rs | f32 | op=masked_softmax, len=64, batch=8, axis=0 | 1 | 16 | 193298.3 | 2361.2 | 1.4% | ok |
| masked_softmax_f32_eager-composed_len64_b8 | #1976 | eager-composed | tenferro-rs | f32 | op=masked_softmax, len=64, batch=8, axis=0 | 1 | 16 | 121638.5 | 2945.2 | 3.4% | ok |
| masked_softmax_f32_pytorch_len64_b8 | #1976 | pytorch | pytorch-cpu | f32 | op=masked_softmax, len=64, batch=8, axis=0 | 1 | 32 | 87569.7 | 616.3 | 1.6% | ok |
| masked_softmax_f32_eager-single-call_len512_b8 | #1976 | eager-single-call | tenferro-rs | f32 | op=masked_softmax, len=512, batch=8, axis=0 | 1 | 1 | 8086567.0 | 89561.5 | 0.9% | ok |
| masked_softmax_f32_session-single-call_len512_b8 | #1976 | session-single-call | tenferro-rs | f32 | op=masked_softmax, len=512, batch=8, axis=0 | 1 | 1 | 8139065.0 | 99421.0 | 0.8% | ok |
| masked_softmax_f32_eager-composed_len512_b8 | #1976 | eager-composed | tenferro-rs | f32 | op=masked_softmax, len=512, batch=8, axis=0 | 1 | 1 | 4598448.0 | 53700.5 | 2.0% | ok |
| masked_softmax_f32_pytorch_len512_b8 | #1976 | pytorch | pytorch-cpu | f32 | op=masked_softmax, len=512, batch=8, axis=0 | 1 | 1 | 4940285.0 | 106939.5 | 1.7% | ok |
| softmax_f32_eager-single-call_len64_b8_t1 | #2233 #2235 | eager-single-call | tenferro-rs | f32 | op=softmax, len=64, batch=8, axis=0 | 1 | 32 | 32205.9 | 471.8 | 0.8% | ok |
| softmax_f32_eager-composed_len64_b8_t1 | #2233 #2235 | eager-composed | tenferro-rs | f32 | op=softmax, len=64, batch=8, axis=0 | 1 | 16 | 107361.9 | 1065.7 | 1.8% | ok |
| softmax_f32_pytorch_len64_b8_t1 | #2233 #2235 | pytorch | pytorch-cpu | f32 | op=softmax, len=64, batch=8, axis=0 | 1 | 64 | 41044.5 | 591.3 | 0.9% | ok |
| reduce_mean_f32_eager-single-call_1024x64_axis0 | #1976 | eager-single-call | tenferro-rs | f32 | rows=1024, cols=64, axis=0 | 1 | 128 | 17635.6 | 533.5 | 2.1% | ok |
| reduce_mean_f32_session-single-call_1024x64_axis0 | #1976 | session-single-call | tenferro-rs | f32 | rows=1024, cols=64, axis=0 | 1 | 256 | 8108.6 | 103.8 | 3.2% | ok |
| reduce_mean_f32_eager-composed_1024x64_axis0 | #1976 | eager-composed | tenferro-rs | f32 | rows=1024, cols=64, axis=0 | 1 | 128 | 17135.8 | 350.4 | 1.7% | ok |
| reduce_mean_f32_pytorch_1024x64_axis0 | #1976 | pytorch | pytorch-cpu | f32 | rows=1024, cols=64, axis=0 | 1 | 256 | 8968.8 | 118.4 | 1.3% | ok |
| take_along_axis_rows_f64_eager-single-call_n64_b256 | #2008 | eager-single-call | tenferro-rs | f64 | n=64, batch=256, axis=0 | 1 | 1 | 8395974.0 | 36618.0 | 0.4% | ok |
| take_along_axis_rows_f64_session-single-call_n64_b256 | #2008 | session-single-call | tenferro-rs | f64 | n=64, batch=256, axis=0 | 1 | 1 | 8159534.0 | 51056.0 | 0.5% | ok |
| take_along_axis_rows_f64_eager-gather-prebuilt_n64_b256 | #2008 | eager-gather-prebuilt | tenferro-rs | f64 | n=64, batch=256, axis=0 | 1 | 1 | 8069435.0 | 59521.5 | 0.7% | ok |
| take_along_axis_rows_f64_pytorch_n64_b256 | #2008 | pytorch | pytorch-cpu | f64 | n=64, batch=256, axis=0 | 1 | 1 | 3384155.0 | 74684.0 | 1.4% | ok |
| lstsq_f64_eager-shared_m768_n384_rhs16 | #2027 | eager-shared | tenferro-rs | f64 | operation=lstsq | 1 | 1 | 20428023.0 | 465057.0 | 1.3% | ok |
| lstsq_f64_pytorch_m768_n384_rhs16 | #2027 | pytorch | pytorch-cpu | f64 | operation=lstsq | 1 | 1 | 9108882.0 | 39538.5 | 0.5% | ok |
| triangular_f64_concrete-shared_n4096_rhs64 | #2028 | concrete-shared | tenferro-rs | f64 | operation=triangular | 1 | 1 | 34374959.0 | 431560.0 | 1.2% | ok |
| triangular_f64_pytorch_n4096_rhs64 | #2028 | pytorch | pytorch-cpu | f64 | operation=triangular | 1 | 1 | 31063749.0 | 332544.5 | 0.7% | ok |
| reshape_f64_concrete-shared_n33554432_to8192x4096 | #2029 | concrete-shared | tenferro-rs | f64 | operation=reshape | 1 | 1 | 142644879.0 | 4923924.0 | 3.0% | ok |
| reshape_f64_pytorch_n33554432_to8192x4096 | #2029 | pytorch | pytorch-cpu | f64 | operation=reshape | 1 | 1 | 134708150.0 | 2062903.0 | 1.3% | ok |
| metadata_transpose_view_f64_rank8_t1 | #2180 | metadata-clone | tenferro-rs | f64 | shape=[2, 2, 2, 2, 2, 2, 2, 2] | 1 | 8192 | 65.1 | 2.1 | 83.6% | NOISY |
| metadata_transpose_view_f64_rank8_t1_julia-base | #2180 | julia-base | julia-base | f64 | shape=[2, 2, 2, 2, 2, 2, 2, 2] | 1 | 1048576 | 0.3 | 0.1 | 14.7% | NOISY |
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
| typed_elemwise_f64_add_held_len16 | #2139 | tenferro | tenferro-rs | f64 | n=16, route=held | 1 | 2048 | 1131.3 | 8.7 | 3.5% | ok |
| typed_elemwise_f64_add_held_len16_pytorch | #2139 | pytorch | pytorch-cpu | f64 | n=16, route=held | 1 | 2048 | 1511.3 | 4.8 | 1.9% | ok |
| typed_elemwise_f64_add_into_held_len16 | #2139 | tenferro | tenferro-rs | f64 | n=16, route=into_held | 1 | 4096 | 793.9 | 4.3 | 0.9% | ok |
| typed_elemwise_f64_add_into_held_len16_pytorch | #2139 | pytorch | pytorch-cpu | f64 | n=16, route=into_held | 1 | 2048 | 1412.4 | 11.8 | 0.7% | ok |
| typed_elemwise_f64_add_held_len256 | #2139 | tenferro | tenferro-rs | f64 | n=256, route=held | 1 | 1024 | 1202.7 | 19.6 | 3.8% | ok |
| typed_elemwise_f64_add_held_len256_pytorch | #2139 | pytorch | pytorch-cpu | f64 | n=256, route=held | 1 | 2048 | 1595.2 | 43.0 | 3.0% | ok |
| typed_elemwise_f64_add_into_held_len256 | #2139 | tenferro | tenferro-rs | f64 | n=256, route=into_held | 1 | 4096 | 858.4 | 2.6 | 0.3% | ok |
| typed_elemwise_f64_add_into_held_len256_pytorch | #2139 | pytorch | pytorch-cpu | f64 | n=256, route=into_held | 1 | 2048 | 1462.6 | 9.7 | 0.5% | ok |
| typed_elemwise_f64_add_held_len4096 | #2139 | tenferro | tenferro-rs | f64 | n=4096, route=held | 1 | 256 | 2342.7 | 40.2 | 1.5% | ok |
| typed_elemwise_f64_add_held_len4096_pytorch | #2139 | pytorch | pytorch-cpu | f64 | n=4096, route=held | 1 | 1024 | 3325.7 | 239.6 | 4.4% | ok |
| typed_elemwise_f64_add_into_held_len4096 | #2139 | tenferro | tenferro-rs | f64 | n=4096, route=into_held | 1 | 2048 | 1812.7 | 13.1 | 1.2% | ok |
| typed_elemwise_f64_add_into_held_len4096_pytorch | #2139 | pytorch | pytorch-cpu | f64 | n=4096, route=into_held | 1 | 1024 | 2448.1 | 25.2 | 0.8% | ok |
| tiny_matmul_2x2_typed_held | #2139 | tenferro | tenferro-rs | f64 | n=2 | 1 | 2048 | 1425.5 | 2.3 | 0.8% | ok |
| tiny_matmul_2x2_typed_held_pytorch | #2139 | pytorch | pytorch-cpu | f64 | n=2 | 1 | 2048 | 1635.8 | 12.1 | 0.7% | ok |
| analytic_neg_f64_sweep_8MiB_t1 | #2139 | tenferro | tenferro-rs | f64 | n=1048576 | 1 | 2 | 347322.0 | 58534.0 | 12.3% | NOISY |
| analytic_neg_f64_sweep_8MiB_t1_pytorch | #2139 | pytorch | pytorch-cpu | f64 | n=1048576 | 1 | 4 | 492121.8 | 27199.5 | 4.5% | ok |
| analytic_neg_f64_sweep_32MiB_t1 | #2139 | tenferro | tenferro-rs | f64 | n=4194304 | 1 | 1 | 2354156.0 | 88409.0 | 3.0% | ok |
| analytic_neg_f64_sweep_32MiB_t1_pytorch | #2139 | pytorch | pytorch-cpu | f64 | n=4194304 | 1 | 1 | 15817902.0 | 166651.0 | 1.0% | ok |
| gemm_bound_f64_t1_t8_scaling | #2141 | tenferro | tenferro-rs | f64 | n=1536 | 1 | 1 | 158447766.0 | 912383.5 | 0.6% | ok |
| gemm_bound_f64_t1_t8_scaling_pytorch | #2141 | pytorch | pytorch-cpu | f64 | n=1536 | 1 | 1 | 157836397.0 | 656472.5 | 0.3% | ok |
| lu_batched_f64_factor_n8_b1024 | #2142 | tenferro | tenferro-rs | f64 | n=8, batch=1024, op=factor | 1 | 8 | 323985.0 | 1822.1 | 0.5% | ok |
| lu_batched_f64_factor_n8_b1024_host | #2142 | host | rust-loop | f64 | n=8, batch=1024, op=factor | 1 | 4 | 505384.0 | 2624.9 | 0.4% | ok |
| lu_batched_f64_solve_n8_b1024 | #2142 | tenferro | tenferro-rs | f64 | n=8, batch=1024, op=solve | 1 | 16 | 203161.1 | 737.3 | 0.7% | ok |
| lu_batched_f64_solve_n8_b1024_pytorch | #2142 | pytorch | pytorch-cpu | f64 | n=8, batch=1024, op=solve | 1 | 16 | 169466.4 | 381.0 | 0.3% | ok |
| lu_batched_f64_factor_n16_b1024 | #2142 | tenferro | tenferro-rs | f64 | n=16, batch=1024, op=factor | 1 | 2 | 1061317.5 | 13883.2 | 1.3% | ok |
| lu_batched_f64_factor_n16_b1024_host | #2142 | host | rust-loop | f64 | n=16, batch=1024, op=factor | 1 | 2 | 2174356.5 | 16092.0 | 0.8% | ok |
| lu_batched_f64_solve_n16_b1024 | #2142 | tenferro | tenferro-rs | f64 | n=16, batch=1024, op=solve | 1 | 8 | 305943.8 | 3188.4 | 0.7% | ok |
| lu_batched_f64_solve_n16_b1024_pytorch | #2142 | pytorch | pytorch-cpu | f64 | n=16, batch=1024, op=solve | 1 | 8 | 373670.0 | 2311.8 | 1.3% | ok |
| axpby_accum_strided_dest_f64_block | #2143 | tenferro | tenferro-rs | f64 | n=64 | 1 | 512 | 6152.8 | 141.7 | 1.4% | ok |
| axpby_accum_strided_dest_f64_block_host | #2143 | host | rust-loop | f64 | n=64 | 1 | 512 | 6108.6 | 63.8 | 1.7% | ok |
| small_grad_qr_f64_trace-prepared_n4 | #2186 | trace-prepared | tenferro-rs | f64 | op=qr, n=4 | 1 | 32 | 63164.2 | 1162.2 | 1.2% | ok |
| small_grad_qr_f64_pytorch_n4 | #2186 | pytorch | pytorch-cpu | f64 | op=qr, n=4 | 1 | 16 | 143999.0 | 3918.6 | 1.6% | ok |
| small_grad_qr_f64_trace-prepared_n16 | #2186 | trace-prepared | tenferro-rs | f64 | op=qr, n=16 | 1 | 32 | 75126.9 | 619.1 | 1.0% | ok |
| small_grad_qr_f64_pytorch_n16 | #2186 | pytorch | pytorch-cpu | f64 | op=qr, n=16 | 1 | 16 | 155867.3 | 3329.0 | 2.1% | ok |
| small_grad_eigh_f64_trace-prepared_n4 | #2186 | trace-prepared | tenferro-rs | f64 | op=eigh, n=4 | 1 | 32 | 73185.1 | 1647.9 | 1.5% | ok |
| small_grad_eigh_f64_pytorch_n4 | #2186 | pytorch | pytorch-cpu | f64 | op=eigh, n=4 | 1 | 16 | 146856.1 | 2706.3 | 4.6% | ok |
| small_grad_eigh_f64_trace-prepared_n16 | #2186 | trace-prepared | tenferro-rs | f64 | op=eigh, n=16 | 1 | 32 | 104843.5 | 917.7 | 0.6% | ok |
| small_grad_eigh_f64_pytorch_n16 | #2186 | pytorch | pytorch-cpu | f64 | op=eigh, n=16 | 1 | 16 | 171668.0 | 1764.8 | 1.1% | ok |
| small_grad_lu_f64_trace-prepared_n4 | #2186 | trace-prepared | tenferro-rs | f64 | op=lu, n=4 | 1 | 64 | 31640.5 | 326.5 | 0.9% | ok |
| small_grad_lu_f64_pytorch_n4 | #2186 | pytorch | pytorch-cpu | f64 | op=lu, n=4 | 1 | 16 | 154192.9 | 3366.3 | 1.3% | ok |
| small_grad_lu_f64_trace-prepared_n16 | #2186 | trace-prepared | tenferro-rs | f64 | op=lu, n=16 | 1 | 64 | 39082.2 | 767.3 | 1.5% | ok |
| small_grad_lu_f64_pytorch_n16 | #2186 | pytorch | pytorch-cpu | f64 | op=lu, n=16 | 1 | 16 | 162845.9 | 3007.8 | 1.6% | ok |
| small_grad_svd_f64_trace-prepared_n4 | #2186 | trace-prepared | tenferro-rs | f64 | op=svd, n=4 | 1 | 32 | 106553.2 | 1439.2 | 2.5% | ok |
| small_grad_svd_f64_pytorch_n4 | #2186 | pytorch | pytorch-cpu | f64 | op=svd, n=4 | 1 | 16 | 193728.4 | 1830.9 | 0.9% | ok |
| small_grad_svd_f64_trace-prepared_n16 | #2186 | trace-prepared | tenferro-rs | f64 | op=svd, n=16 | 1 | 16 | 156392.0 | 1740.4 | 1.1% | ok |
| small_grad_svd_f64_pytorch_n16 | #2186 | pytorch | pytorch-cpu | f64 | op=svd, n=16 | 1 | 16 | 239052.9 | 2160.3 | 0.9% | ok |
| small_grad_solve_f64_trace-prepared_n4 | #2186 | trace-prepared | tenferro-rs | f64 | op=solve, n=4 | 1 | 128 | 21866.6 | 242.6 | 1.0% | ok |
| small_grad_solve_f64_pytorch_n4 | #2186 | pytorch | pytorch-cpu | f64 | op=solve, n=4 | 1 | 16 | 124724.9 | 3817.8 | 2.6% | ok |
| small_grad_solve_f64_trace-prepared_n16 | #2186 | trace-prepared | tenferro-rs | f64 | op=solve, n=16 | 1 | 64 | 31912.0 | 440.9 | 1.6% | ok |
| small_grad_solve_f64_pytorch_n16 | #2186 | pytorch | pytorch-cpu | f64 | op=solve, n=16 | 1 | 16 | 136912.0 | 4177.8 | 2.4% | ok |
| fft_axis0_c64_fft-executor_n128 | #1765 | fft-executor | tenferro-rs | c64 | n=128 | 1 | 1 | 7412211.0 | 32816.5 | 0.3% | ok |
| fft_axis0_c64_pytorch_n128 | #1765 | pytorch | pytorch-cpu | c64 | n=128 | 1 | 1 | 28510834.0 | 634201.5 | 2.3% | ok |
| degenerate_dot_hadamard_f64_dot-general-shared_n1024 | #2111 | dot-general-shared | tenferro-rs | f64 | form=hadamard, n=1024 | 1 | 2 | 620331.0 | 54717.2 | 5.4% | ok |
| degenerate_dot_hadamard_f64_pytorch_n1024 | #2111 | pytorch | pytorch-cpu | f64 | form=hadamard, n=1024 | 1 | 2 | 586994.0 | 40057.2 | 5.3% | ok |
| degenerate_dot_outer_f64_dot-general-shared_n1024 | #2111 | dot-general-shared | tenferro-rs | f64 | form=outer, n=1024 | 1 | 2 | 224814.0 | 15494.0 | 11.0% | NOISY |
| degenerate_dot_outer_f64_pytorch_n1024 | #2111 | pytorch | pytorch-cpu | f64 | form=outer, n=1024 | 1 | 8 | 2466534.5 | 43992.6 | 1.1% | ok |
| degenerate_dot_rank0_f64_dot-general-shared_n1024 | #2111 | dot-general-shared | tenferro-rs | f64 | form=rank0, n=1024 | 1 | 2 | 1199405.0 | 64574.8 | 2.9% | ok |
| degenerate_dot_rank0_f64_pytorch_n1024 | #2111 | pytorch | pytorch-cpu | f64 | form=rank0, n=1024 | 1 | 4 | 467188.0 | 21052.9 | 10.5% | NOISY |
| erased_add_f64_session-add_n256 | #1766 | session-add | tenferro-rs | f64 | n=256 | 1 | 2048 | 931.1 | 16.2 | 5.9% | ok |
| erased_add_f64_pytorch_n256 | #1766 | pytorch | pytorch-cpu | f64 | n=256 | 1 | 2048 | 1587.7 | 12.8 | 2.0% | ok |
| erased_add_f64_session-add_n4096 | #1766 | session-add | tenferro-rs | f64 | n=4096 | 1 | 256 | 2077.9 | 51.1 | 2.8% | ok |
| erased_add_f64_pytorch_n4096 | #1766 | pytorch | pytorch-cpu | f64 | n=4096 | 1 | 1024 | 3502.3 | 340.0 | 4.9% | ok |
| erased_add_f64_session-add_n1048576 | #1766 | session-add | tenferro-rs | f64 | n=1048576 | 1 | 2 | 574510.5 | 76763.2 | 9.5% | ok |
| erased_add_f64_pytorch_n1048576 | #1766 | pytorch | pytorch-cpu | f64 | n=1048576 | 1 | 4 | 640440.8 | 20757.5 | 4.4% | ok |
| axpby_strided_source_f64_1024x1024_t1 | #2220 #2214 | strided-source | tenferro-rs | f64 | n=1024 | 1 | 2 | 2166492.0 | 416796.5 | 14.1% | NOISY |
| axpby_contiguous_source_f64_1024x1024_t1 | #2220 #2214 | contiguous-source | tenferro-rs | f64 | n=1024 | 1 | 4 | 584506.5 | 39906.9 | 4.5% | ok |
| axpby_strided_source_f64_1024x1024_host_t1 | #2220 #2214 | host | rust-loop | f64 | n=1024 | 1 | 2 | 1716884.0 | 46106.0 | 2.1% | ok |
| erased_construct_f64_col_n8 | #2177 | col-major | tenferro-rs | f64 | n=8, layout=col | 1 | 8192 | 116.2 | 0.7 | 25.9% | NOISY |
| erased_construct_f64_row_n8 | #2177 | row-major | tenferro-rs | f64 | n=8, layout=row | 1 | 8192 | 208.6 | 2.7 | 15.3% | NOISY |
| erased_construct_f64_col_n4096 | #2177 | col-major | tenferro-rs | f64 | n=4096, layout=col | 1 | 16384 | 234.0 | 4.9 | 1.3% | ok |
| erased_construct_f64_row_n4096 | #2177 | row-major | tenferro-rs | f64 | n=4096, layout=row | 1 | 128 | 19822.4 | 254.9 | 1.6% | ok |
| erased_construct_f64_col_n65536 | #2177 | col-major | tenferro-rs | f64 | n=65536, layout=col | 1 | 1024 | 99.5 | 7.0 | 6.7% | ok |
| erased_construct_f64_row_n65536 | #2177 | row-major | tenferro-rs | f64 | n=65536, layout=row | 1 | 8 | 336776.2 | 2338.1 | 0.7% | ok |
| eager_small_leaf_f64_eager-held-session | #1704 | eager-held-session | tenferro-rs | f64 | op=leaf | 1 | 512 | 3332.2 | 53.6 | 2.0% | ok |
| eager_small_add_f64_eager-held-session | #1667 | eager-held-session | tenferro-rs | f64 | op=add | 1 | 1024 | 1916.3 | 24.7 | 2.1% | ok |
| eager_small_add_f64_pytorch | #1667 | pytorch | pytorch-cpu | f64 | op=add | 1 | 2048 | 1495.7 | 19.0 | 1.0% | ok |
| eager_small_gemm_f64_eager-held-session | #1904 | eager-held-session | tenferro-rs | f64 | op=gemm | 1 | 1024 | 3007.6 | 13.5 | 0.7% | ok |
| eager_small_gemm_f64_pytorch | #1904 | pytorch | pytorch-cpu | f64 | op=gemm | 1 | 2048 | 1720.1 | 22.3 | 0.8% | ok |
| trace_metadata_trace-fold_f32_1024x64_k50 | #2226 #2238 | trace-fold | tenferro-rs | f32 | rows=1024, cols=64, steps=50 | 1 | 1 | 5193227.0 | 20177.5 | 0.3% | ok |
| trace_metadata_trace-outputs_f32_1024x64_k50 | #2226 #2238 | trace-outputs | tenferro-rs | f32 | rows=1024, cols=64, steps=50 | 1 | 1 | 2220908.0 | 42704.5 | 1.6% | ok |
| trace_metadata_eager_f32_1024x64_k50 | #2226 #2238 | eager | tenferro-rs | f32 | rows=1024, cols=64, steps=50 | 1 | 1 | 5802131.0 | 23504.5 | 0.3% | ok |
| trace_metadata_pytorch_f32_1024x64_k50 | #2226 #2238 | pytorch | pytorch-cpu | f32 | rows=1024, cols=64, steps=50 | 1 | 1 | 2575619.0 | 18308.5 | 1.7% | ok |
| trace_metadata_fixed_neg_f32_8 | #2226 #2238 | trace-fold | tenferro-rs | f32 | rows=8, cols=1, steps=0 | 1 | 1024 | 3139.9 | 9.2 | 0.6% | ok |
| decode_proj_f32_eager-shared_in1024_out1024_len8 | #1992 #2003 #1995 #2229 | eager-shared | tenferro-rs | f32 | in=1024, out=1024, len=8 | 4 | 2 | 1517958.0 | 71462.8 | 5.6% | ok |
| decode_proj_f32_faer-direct_in1024_out1024_len8 | #1992 #2003 #1995 #2229 | faer-direct | faer-direct | f32 | in=1024, out=1024, len=8 | 4 | 16 | 160901.1 | 4642.4 | 3.5% | ok |
| decode_proj_f32_host-sgemm_in1024_out1024_len8 | #1992 #2003 #1995 #2229 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=1024, len=8 | 4 | 8 | 526111.5 | 14371.8 | 3.0% | ok |
| decode_proj_f32_pytorch_in1024_out1024_len8 | #1992 #2003 #1995 #2229 | pytorch | pytorch-cpu | f32 | in=1024, out=1024, len=8 | 4 | 8 | 366301.2 | 12861.5 | 3.3% | ok |
| decode_proj_f32_eager-shared_in1024_out1024_len64 | #1992 #2003 #1995 #2229 | eager-shared | tenferro-rs | f32 | in=1024, out=1024, len=64 | 4 | 2 | 1134248.5 | 20834.0 | 1.6% | ok |
| decode_proj_f32_faer-direct_in1024_out1024_len64 | #1992 #2003 #1995 #2229 | faer-direct | faer-direct | f32 | in=1024, out=1024, len=64 | 4 | 4 | 764932.8 | 8391.8 | 0.8% | ok |
| decode_proj_f32_host-sgemm_in1024_out1024_len64 | #1992 #2003 #1995 #2229 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=1024, len=64 | 4 | 2 | 1148760.5 | 11654.2 | 2.1% | ok |
| decode_proj_f32_pytorch_in1024_out1024_len64 | #1992 #2003 #1995 #2229 | pytorch | pytorch-cpu | f32 | in=1024, out=1024, len=64 | 4 | 2 | 1165396.5 | 17472.5 | 1.8% | ok |
| decode_proj_f32_eager-shared_in1024_out1024_len512 | #1992 #2003 #1995 #2229 | eager-shared | tenferro-rs | f32 | in=1024, out=1024, len=512 | 4 | 1 | 7871847.0 | 243323.5 | 1.7% | ok |
| decode_proj_f32_faer-direct_in1024_out1024_len512 | #1992 #2003 #1995 #2229 | faer-direct | faer-direct | f32 | in=1024, out=1024, len=512 | 4 | 1 | 5684492.0 | 39428.5 | 0.5% | ok |
| decode_proj_f32_host-sgemm_in1024_out1024_len512 | #1992 #2003 #1995 #2229 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=1024, len=512 | 4 | 1 | 6648700.0 | 30342.5 | 0.4% | ok |
| decode_proj_f32_pytorch_in1024_out1024_len512 | #1992 #2003 #1995 #2229 | pytorch | pytorch-cpu | f32 | in=1024, out=1024, len=512 | 4 | 1 | 7998735.0 | 31228.5 | 0.5% | ok |
| decode_proj_f32_eager-shared_in1024_out4096_len8 | #1992 #2003 #1995 #2229 | eager-shared | tenferro-rs | f32 | in=1024, out=4096, len=8 | 4 | 1 | 6147175.0 | 40300.5 | 0.7% | ok |
| decode_proj_f32_faer-direct_in1024_out4096_len8 | #1992 #2003 #1995 #2229 | faer-direct | faer-direct | f32 | in=1024, out=4096, len=8 | 4 | 4 | 578966.2 | 7784.4 | 2.7% | ok |
| decode_proj_f32_host-sgemm_in1024_out4096_len8 | #1992 #2003 #1995 #2229 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=4096, len=8 | 4 | 1 | 2266113.0 | 24626.0 | 1.3% | ok |
| decode_proj_f32_pytorch_in1024_out4096_len8 | #1992 #2003 #1995 #2229 | pytorch | pytorch-cpu | f32 | in=1024, out=4096, len=8 | 4 | 2 | 1690480.0 | 64026.5 | 4.0% | ok |
| decode_proj_f32_eager-shared_in1024_out4096_len64 | #1992 #2003 #1995 #2229 | eager-shared | tenferro-rs | f32 | in=1024, out=4096, len=64 | 4 | 1 | 4474618.0 | 113356.0 | 2.5% | ok |
| decode_proj_f32_faer-direct_in1024_out4096_len64 | #1992 #2003 #1995 #2229 | faer-direct | faer-direct | f32 | in=1024, out=4096, len=64 | 4 | 1 | 2855922.0 | 13004.5 | 0.4% | ok |
| decode_proj_f32_host-sgemm_in1024_out4096_len64 | #1992 #2003 #1995 #2229 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=4096, len=64 | 4 | 1 | 4695590.0 | 97592.0 | 1.5% | ok |
| decode_proj_f32_pytorch_in1024_out4096_len64 | #1992 #2003 #1995 #2229 | pytorch | pytorch-cpu | f32 | in=1024, out=4096, len=64 | 4 | 1 | 4715247.0 | 87563.5 | 5.6% | ok |
| decode_proj_f32_eager-shared_in1024_out4096_len512 | #1992 #2003 #1995 #2229 | eager-shared | tenferro-rs | f32 | in=1024, out=4096, len=512 | 4 | 1 | 33242175.0 | 1774326.5 | 2.9% | ok |
| decode_proj_f32_faer-direct_in1024_out4096_len512 | #1992 #2003 #1995 #2229 | faer-direct | faer-direct | f32 | in=1024, out=4096, len=512 | 4 | 1 | 22628808.0 | 65395.0 | 0.2% | ok |
| decode_proj_f32_host-sgemm_in1024_out4096_len512 | #1992 #2003 #1995 #2229 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=4096, len=512 | 4 | 1 | 26941104.0 | 330446.5 | 10.2% | NOISY |
| decode_proj_f32_pytorch_in1024_out4096_len512 | #1992 #2003 #1995 #2229 | pytorch | pytorch-cpu | f32 | in=1024, out=4096, len=512 | 4 | 1 | 30606154.0 | 81797.5 | 0.4% | ok |
| decode_proj_view_operand_f32_prepared_in1024_out1024_len8_t4 | #2216 | prepared-view | tenferro-rs | f32 | in=1024, out=1024, len=8 | 4 | 1 | 4087197.0 | 14216.5 | 1.3% | ok |
| decode_proj_contiguous_operand_f32_prepared_in1024_out1024_len8_t4 | #2216 | prepared-contiguous | tenferro-rs | f32 | in=1024, out=1024, len=8 | 4 | 2 | 1512839.5 | 5976.5 | 0.4% | ok |
| decode_proj_view_operand_f32_pytorch_in1024_out1024_len8_t4 | #2216 | pytorch | pytorch-cpu | f32 | in=1024, out=1024, len=8 | 4 | 8 | 369926.9 | 17617.2 | 3.8% | ok |
| gemm_c64_mm256_t1 | #1900 | dot-general-shared | tenferro-rs | c64 | n=256 | 4 | 2 | 1956305.5 | 10517.0 | 0.4% | ok |
| gemm_f64_mm256_t1 | #1900 | dot-general-shared | tenferro-rs | f64 | n=256 | 4 | 4 | 471834.2 | 9979.9 | 1.1% | ok |
| conj_dot_f64_dot-general-rank1_len100 | #1615 | dot-general-rank1 | tenferro-rs | f64 | len=100 | 4 | 1024 | 2358.6 | 11.8 | 1.8% | ok |
| conj_dot_f64_portable-loop_len100 | #1615 | portable-loop | rust-loop | f64 | len=100 | 4 | 32768 | 75.5 | 0.8 | 1.3% | ok |
| conj_dot_f64_pytorch-vdot_len100 | #1615 | pytorch-vdot | pytorch-cpu | f64 | len=100 | 4 | 2048 | 1360.9 | 8.5 | 1.9% | ok |
| conj_dot_f64_dot-general-rank1_len10000 | #1615 | dot-general-rank1 | tenferro-rs | f64 | len=10000 | 4 | 256 | 7238.2 | 57.6 | 4.8% | ok |
| conj_dot_f64_portable-loop_len10000 | #1615 | portable-loop | rust-loop | f64 | len=10000 | 4 | 256 | 9321.8 | 31.7 | 0.3% | ok |
| conj_dot_f64_pytorch-vdot_len10000 | #1615 | pytorch-vdot | pytorch-cpu | f64 | len=10000 | 4 | 512 | 4633.2 | 45.6 | 0.6% | ok |
| conj_dot_f64_dot-general-rank1_len1000000 | #1615 | dot-general-rank1 | tenferro-rs | f64 | len=1000000 | 4 | 8 | 484685.6 | 5942.9 | 0.9% | ok |
| conj_dot_f64_portable-loop_len1000000 | #1615 | portable-loop | rust-loop | f64 | len=1000000 | 4 | 4 | 946729.0 | 3745.8 | 0.6% | ok |
| conj_dot_f64_pytorch-vdot_len1000000 | #1615 | pytorch-vdot | pytorch-cpu | f64 | len=1000000 | 4 | 8 | 155776.5 | 69964.1 | 28.2% | NOISY |
| conj_dot_c64_dot-general-rank1_len100 | #1615 | dot-general-rank1 | tenferro-rs | c64 | len=100 | 4 | 1024 | 2440.4 | 9.3 | 0.6% | ok |
| conj_dot_c64_portable-loop_len100 | #1615 | portable-loop | rust-loop | c64 | len=100 | 4 | 32768 | 110.7 | 2.0 | 1.7% | ok |
| conj_dot_c64_pytorch-vdot_len100 | #1615 | pytorch-vdot | pytorch-cpu | c64 | len=100 | 4 | 2048 | 1317.4 | 25.2 | 200.7% | NOISY |
| conj_dot_c64_dot-general-rank1_len10000 | #1615 | dot-general-rank1 | tenferro-rs | c64 | len=10000 | 4 | 256 | 12227.3 | 25.2 | 0.1% | ok |
| conj_dot_c64_portable-loop_len10000 | #1615 | portable-loop | rust-loop | c64 | len=10000 | 4 | 256 | 9784.9 | 151.5 | 1.6% | ok |
| conj_dot_c64_pytorch-vdot_len10000 | #1615 | pytorch-vdot | pytorch-cpu | c64 | len=10000 | 4 | 512 | 6747.8 | 68.4 | 6.6% | ok |
| conj_dot_c64_dot-general-rank1_len1000000 | #1615 | dot-general-rank1 | tenferro-rs | c64 | len=1000000 | 4 | 2 | 1102053.5 | 17612.5 | 1.4% | ok |
| conj_dot_c64_portable-loop_len1000000 | #1615 | portable-loop | rust-loop | c64 | len=1000000 | 4 | 2 | 1183776.0 | 40998.8 | 2.1% | ok |
| conj_dot_c64_pytorch-vdot_len1000000 | #1615 | pytorch-vdot | pytorch-cpu | c64 | len=1000000 | 4 | 8 | 397693.5 | 25051.7 | 15.6% | NOISY |
| small_solve_f64_solve-loop_k16_b64 | #2007 #1878 | solve-loop | tenferro-rs | f64 | k=16, batch=64 | 4 | 8 | 343021.6 | 1052.6 | 0.4% | ok |
| small_solve_f64_triangular-solve-loop_k16_b64 | #2007 #1878 | triangular-solve-loop | tenferro-rs | f64 | k=16, batch=64 | 4 | 16 | 251519.9 | 3818.9 | 1.1% | ok |
| small_solve_f64_solve-batched_k16_b64 | #2007 #1878 | solve-batched | tenferro-rs | f64 | k=16, batch=64 | 4 | 16 | 255831.7 | 1913.2 | 16.2% | NOISY |
| small_solve_f64_pytorch-solve-batched_k16_b64 | #2007 #1878 | pytorch-solve-batched | pytorch-cpu | f64 | k=16, batch=64 | 4 | 8 | 337581.5 | 22922.7 | 6.7% | ok |
| small_solve_f64_pytorch-solve-triangular-batched_k16_b64 | #2007 #1878 | pytorch-solve-triangular-batched | pytorch-cpu | f64 | k=16, batch=64 | 4 | 32 | 71667.9 | 1382.6 | 5.3% | ok |
| small_solve_f64_solve-loop_k64_b64 | #2007 #1878 | solve-loop | tenferro-rs | f64 | k=64, batch=64 | 4 | 1 | 5755074.0 | 78631.5 | 4.2% | ok |
| small_solve_f64_triangular-solve-loop_k64_b64 | #2007 #1878 | triangular-solve-loop | tenferro-rs | f64 | k=64, batch=64 | 4 | 1 | 1282650.0 | 21520.5 | 18.9% | NOISY |
| small_solve_f64_solve-batched_k64_b64 | #2007 #1878 | solve-batched | tenferro-rs | f64 | k=64, batch=64 | 4 | 1 | 7211207.0 | 89161.0 | 2.6% | ok |
| small_solve_f64_pytorch-solve-batched_k64_b64 | #2007 #1878 | pytorch-solve-batched | pytorch-cpu | f64 | k=64, batch=64 | 4 | 1 | 3412308.0 | 43887.0 | 3.2% | ok |
| small_solve_f64_pytorch-solve-triangular-batched_k64_b64 | #2007 #1878 | pytorch-solve-triangular-batched | pytorch-cpu | f64 | k=64, batch=64 | 4 | 4 | 934148.2 | 19540.1 | 1.3% | ok |
| tanh_chain_f32_compiled-prepared_k10_1024x64 | #1990 | compiled-prepared | tenferro-rs | f32 | rows=1024, cols=64, chain=10 | 4 | 4 | 677883.2 | 3755.9 | 9.7% | ok |
| tanh_chain_f32_eager-shared_k10_1024x64 | #1990 | eager-shared | tenferro-rs | f32 | rows=1024, cols=64, chain=10 | 4 | 2 | 916918.5 | 5906.2 | 0.8% | ok |
| tanh_chain_f32_compiled-prepared_k100_1024x64 | #1990 | compiled-prepared | tenferro-rs | f32 | rows=1024, cols=64, chain=100 | 4 | 1 | 6642337.0 | 23784.0 | 1.9% | ok |
| tanh_chain_f32_eager-shared_k100_1024x64 | #1990 | eager-shared | tenferro-rs | f32 | rows=1024, cols=64, chain=100 | 4 | 1 | 8950408.0 | 21626.0 | 0.2% | ok |
| tanh_chain_f32_compiled-prepared_k200_1024x64 | #1990 | compiled-prepared | tenferro-rs | f32 | rows=1024, cols=64, chain=200 | 4 | 1 | 13275597.0 | 36579.0 | 0.8% | ok |
| tanh_chain_f32_eager-shared_k200_1024x64 | #1990 | eager-shared | tenferro-rs | f32 | rows=1024, cols=64, chain=200 | 4 | 1 | 17906667.0 | 65952.5 | 0.5% | ok |
| layer_norm_f32_eager-composed_d1024_len8_b1 | #2006 | eager-composed | tenferro-rs | f32 | norm=layer_norm, d=1024, len=8, batch=1 | 4 | 16 | 87074.1 | 4743.5 | 14.1% | NOISY |
| layer_norm_f32_pytorch-fused_d1024_len8_b1 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=layer_norm, d=1024, len=8, batch=1 | 4 | 256 | 9127.3 | 70.5 | 14.3% | NOISY |
| layer_norm_f32_eager-single-call_d1024_len8_b1 | #2006 | eager-single-call | tenferro-rs | f32 | norm=layer_norm, d=1024, len=8, batch=1 | 4 | 16 | 93929.4 | 1876.0 | 12.7% | NOISY |
| layer_norm_f32_eager-composed_d1024_len8_b8 | #2006 | eager-composed | tenferro-rs | f32 | norm=layer_norm, d=1024, len=8, batch=8 | 4 | 8 | 167869.1 | 7375.1 | 9.5% | ok |
| layer_norm_f32_pytorch-fused_d1024_len8_b8 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=layer_norm, d=1024, len=8, batch=8 | 4 | 128 | 19078.4 | 451.9 | 3.0% | ok |
| layer_norm_f32_eager-single-call_d1024_len8_b8 | #2006 | eager-single-call | tenferro-rs | f32 | norm=layer_norm, d=1024, len=8, batch=8 | 4 | 8 | 194705.4 | 3684.4 | 4.1% | ok |
| layer_norm_f32_eager-composed_d1024_len64_b1 | #2006 | eager-composed | tenferro-rs | f32 | norm=layer_norm, d=1024, len=64, batch=1 | 4 | 8 | 212519.8 | 5001.8 | 1.9% | ok |
| layer_norm_f32_pytorch-fused_d1024_len64_b1 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=layer_norm, d=1024, len=64, batch=1 | 4 | 128 | 18449.2 | 308.2 | 2.2% | ok |
| layer_norm_f32_eager-single-call_d1024_len64_b1 | #2006 | eager-single-call | tenferro-rs | f32 | norm=layer_norm, d=1024, len=64, batch=1 | 4 | 8 | 191481.8 | 5154.0 | 4.4% | ok |
| layer_norm_f32_eager-composed_d1024_len64_b8 | #2006 | eager-composed | tenferro-rs | f32 | norm=layer_norm, d=1024, len=64, batch=8 | 4 | 2 | 588817.5 | 23075.5 | 3.2% | ok |
| layer_norm_f32_pytorch-fused_d1024_len64_b8 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=layer_norm, d=1024, len=64, batch=8 | 4 | 16 | 413188.1 | 10104.8 | 2.7% | ok |
| layer_norm_f32_eager-single-call_d1024_len64_b8 | #2006 | eager-single-call | tenferro-rs | f32 | norm=layer_norm, d=1024, len=64, batch=8 | 4 | 4 | 575597.5 | 18845.0 | 3.7% | ok |
| rms_norm_f32_eager-composed_d1024_len8_b1 | #2006 | eager-composed | tenferro-rs | f32 | norm=rms_norm, d=1024, len=8, batch=1 | 4 | 32 | 46909.9 | 804.9 | 12.3% | NOISY |
| rms_norm_f32_pytorch-fused_d1024_len8_b1 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=rms_norm, d=1024, len=8, batch=1 | 4 | 128 | 19927.0 | 628.9 | 2.9% | ok |
| rms_norm_f32_eager-single-call_d1024_len8_b1 | #2006 | eager-single-call | tenferro-rs | f32 | norm=rms_norm, d=1024, len=8, batch=1 | 4 | 32 | 54614.9 | 629.5 | 11.1% | NOISY |
| rms_norm_f32_eager-composed_d1024_len8_b8 | #2006 | eager-composed | tenferro-rs | f32 | norm=rms_norm, d=1024, len=8, batch=8 | 4 | 16 | 103268.7 | 3422.0 | 3.7% | ok |
| rms_norm_f32_pytorch-fused_d1024_len8_b8 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=rms_norm, d=1024, len=8, batch=8 | 4 | 32 | 55558.5 | 1163.6 | 2.9% | ok |
| rms_norm_f32_eager-single-call_d1024_len8_b8 | #2006 | eager-single-call | tenferro-rs | f32 | norm=rms_norm, d=1024, len=8, batch=8 | 4 | 16 | 89678.4 | 3371.3 | 14.7% | NOISY |
| rms_norm_f32_eager-composed_d1024_len64_b1 | #2006 | eager-composed | tenferro-rs | f32 | norm=rms_norm, d=1024, len=64, batch=1 | 4 | 16 | 82853.1 | 1560.7 | 7.9% | ok |
| rms_norm_f32_pytorch-fused_d1024_len64_b1 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=rms_norm, d=1024, len=64, batch=1 | 4 | 32 | 66569.7 | 818.5 | 1.0% | ok |
| rms_norm_f32_eager-single-call_d1024_len64_b1 | #2006 | eager-single-call | tenferro-rs | f32 | norm=rms_norm, d=1024, len=64, batch=1 | 4 | 16 | 111049.4 | 3028.8 | 2.2% | ok |
| rms_norm_f32_eager-composed_d1024_len64_b8 | #2006 | eager-composed | tenferro-rs | f32 | norm=rms_norm, d=1024, len=64, batch=8 | 4 | 4 | 256087.2 | 13894.8 | 6.1% | ok |
| rms_norm_f32_pytorch-fused_d1024_len64_b8 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=rms_norm, d=1024, len=64, batch=8 | 4 | 8 | 914166.0 | 59599.3 | 5.0% | ok |
| rms_norm_f32_eager-single-call_d1024_len64_b8 | #2006 | eager-single-call | tenferro-rs | f32 | norm=rms_norm, d=1024, len=64, batch=8 | 4 | 4 | 269785.2 | 7115.6 | 3.9% | ok |
| small_contraction_abcd-dbef-acef_f64_shared-session-default-pool_d4 | #1885 #1897 #1901 | shared-session-default-pool | tenferro-rs | f64 | extent=4 | 4 | 512 | 4448.6 | 41.3 | 0.7% | ok |
| activation_erf_f32_eager-single-call_1024x64 | #1975 #2069 #2032 | eager-single-call | tenferro-rs | f32 | op=erf, rows=1024, cols=64 | 4 | — | — | — | — | FAILED |
| activation_erf_f32_session-single-call_1024x64 | #1975 #2069 #2032 | session-single-call | tenferro-rs | f32 | op=erf, rows=1024, cols=64 | 4 | 16 | 162415.8 | 639.6 | 2.1% | ok |
| activation_erf_f32_pytorch_1024x64 | #1975 #2069 #2032 | pytorch | pytorch-cpu | f32 | op=erf, rows=1024, cols=64 | 4 | — | — | — | — | FAILED |
| activation_sigmoid_f32_eager-single-call_1024x64 | #1975 #2069 #2032 | eager-single-call | tenferro-rs | f32 | op=sigmoid, rows=1024, cols=64 | 4 | — | — | — | — | FAILED |
| activation_sigmoid_f32_session-single-call_1024x64 | #1975 #2069 #2032 | session-single-call | tenferro-rs | f32 | op=sigmoid, rows=1024, cols=64 | 4 | 64 | 27002.2 | 300.1 | 2.1% | ok |
| activation_sigmoid_f32_eager-composed_1024x64 | #1975 #2069 #2032 | eager-composed | tenferro-rs | f32 | op=sigmoid, rows=1024, cols=64 | 4 | 16 | 77845.1 | 5554.7 | 10.1% | NOISY |
| activation_sigmoid_f32_pytorch_1024x64 | #1975 #2069 #2032 | pytorch | pytorch-cpu | f32 | op=sigmoid, rows=1024, cols=64 | 4 | — | — | — | — | FAILED |
| activation_silu_f32_eager-single-call_1024x64 | #1975 #2069 #2032 | eager-single-call | tenferro-rs | f32 | op=silu, rows=1024, cols=64 | 4 | — | — | — | — | FAILED |
| activation_silu_f32_session-single-call_1024x64 | #1975 #2069 #2032 | session-single-call | tenferro-rs | f32 | op=silu, rows=1024, cols=64 | 4 | 64 | 28063.8 | 304.4 | 1.0% | ok |
| activation_silu_f32_eager-composed_1024x64 | #1975 #2069 #2032 | eager-composed | tenferro-rs | f32 | op=silu, rows=1024, cols=64 | 4 | 16 | 94700.9 | 49314.7 | 22.3% | NOISY |
| activation_silu_f32_pytorch_1024x64 | #1975 #2069 #2032 | pytorch | pytorch-cpu | f32 | op=silu, rows=1024, cols=64 | 4 | — | — | — | — | FAILED |
| activation_softplus_f32_eager-single-call_1024x64 | #1975 #2069 #2032 | eager-single-call | tenferro-rs | f32 | op=softplus, rows=1024, cols=64 | 4 | — | — | — | — | FAILED |
| activation_softplus_f32_session-single-call_1024x64 | #1975 #2069 #2032 | session-single-call | tenferro-rs | f32 | op=softplus, rows=1024, cols=64 | 4 | 32 | 73459.1 | 512.4 | 4.6% | ok |
| activation_softplus_f32_eager-composed_1024x64 | #1975 #2069 #2032 | eager-composed | tenferro-rs | f32 | op=softplus, rows=1024, cols=64 | 4 | 16 | 136644.6 | 1408.6 | 1.1% | ok |
| activation_softplus_f32_pytorch_1024x64 | #1975 #2069 #2032 | pytorch | pytorch-cpu | f32 | op=softplus, rows=1024, cols=64 | 4 | — | — | — | — | FAILED |
| activation_gelu_f32_eager-single-call_1024x64 | #1975 #2030 #2069 #2228 | eager-single-call | tenferro-rs | f32 | op=gelu, rows=1024, cols=64 | 4 | 64 | 38653.0 | 340.2 | 0.5% | ok |
| activation_gelu_f32_session-single-call_1024x64 | #1975 #2030 #2069 #2228 | session-single-call | tenferro-rs | f32 | op=gelu, rows=1024, cols=64 | 4 | 32 | 38046.5 | 802.3 | 8.3% | ok |
| activation_gelu_f32_pytorch_1024x64 | #1975 #2030 #2069 #2228 | pytorch | pytorch-cpu | f32 | op=gelu, rows=1024, cols=64 | 4 | 32 | 41440.0 | 695.5 | 1.5% | ok |
| activation_gelu_tanh_f32_eager-single-call_1024x64 | #1975 #2069 #2032 | eager-single-call | tenferro-rs | f32 | op=gelu_tanh, rows=1024, cols=64 | 4 | — | — | — | — | FAILED |
| activation_gelu_tanh_f32_session-single-call_1024x64 | #1975 #2069 #2032 | session-single-call | tenferro-rs | f32 | op=gelu_tanh, rows=1024, cols=64 | 4 | 32 | 76661.9 | 772.1 | 0.8% | ok |
| activation_gelu_tanh_f32_eager-composed_1024x64 | #1975 #2069 #2032 | eager-composed | tenferro-rs | f32 | op=gelu_tanh, rows=1024, cols=64 | 4 | 8 | 255317.0 | 3263.7 | 5.0% | ok |
| activation_gelu_tanh_f32_pytorch_1024x64 | #1975 #2069 #2032 | pytorch | pytorch-cpu | f32 | op=gelu_tanh, rows=1024, cols=64 | 4 | — | — | — | — | FAILED |
| softmax_f32_eager-single-call_len64_b8 | #1976 #2031 #2231 | eager-single-call | tenferro-rs | f32 | op=softmax, len=64, batch=8, axis=0 | 4 | 64 | 21679.8 | 163.2 | 0.7% | ok |
| softmax_f32_session-single-call_len64_b8 | #1976 #2031 #2231 | session-single-call | tenferro-rs | f32 | op=softmax, len=64, batch=8, axis=0 | 4 | 64 | 20692.6 | 462.9 | 1.3% | ok |
| softmax_f32_eager-composed_len64_b8 | #1976 #2031 #2231 | eager-composed | tenferro-rs | f32 | op=softmax, len=64, batch=8, axis=0 | 4 | 16 | 115507.1 | 3236.7 | 4.6% | ok |
| softmax_f32_pytorch_len64_b8 | #1976 #2031 #2231 | pytorch | pytorch-cpu | f32 | op=softmax, len=64, batch=8, axis=0 | 4 | 128 | 19120.1 | 173.4 | 1.2% | ok |
| softmax_f32_eager-single-call_len512_b8 | #1976 #2231 | eager-single-call | tenferro-rs | f32 | op=softmax, len=512, batch=8, axis=0 | 4 | 2 | 860758.5 | 19827.0 | 1.5% | ok |
| softmax_f32_session-single-call_len512_b8 | #1976 #2231 | session-single-call | tenferro-rs | f32 | op=softmax, len=512, batch=8, axis=0 | 4 | 2 | 845826.0 | 22151.2 | 12.5% | NOISY |
| softmax_f32_eager-composed_len512_b8 | #1976 #2231 | eager-composed | tenferro-rs | f32 | op=softmax, len=512, batch=8, axis=0 | 4 | 1 | 1977745.0 | 104605.5 | 3.4% | ok |
| softmax_f32_pytorch_len512_b8 | #1976 #2231 | pytorch | pytorch-cpu | f32 | op=softmax, len=512, batch=8, axis=0 | 4 | 2 | 953416.5 | 4303.2 | 0.4% | ok |
| log_softmax_f32_eager-single-call_len64_b8 | #1976 | eager-single-call | tenferro-rs | f32 | op=log_softmax, len=64, batch=8, axis=0 | 4 | 8 | 146762.2 | 3076.3 | 14.3% | NOISY |
| log_softmax_f32_session-single-call_len64_b8 | #1976 | session-single-call | tenferro-rs | f32 | op=log_softmax, len=64, batch=8, axis=0 | 4 | 16 | 111360.0 | 892.3 | 4.1% | ok |
| log_softmax_f32_eager-composed_len64_b8 | #1976 | eager-composed | tenferro-rs | f32 | op=log_softmax, len=64, batch=8, axis=0 | 4 | 16 | 116790.8 | 1988.0 | 3.9% | ok |
| log_softmax_f32_pytorch_len64_b8 | #1976 | pytorch | pytorch-cpu | f32 | op=log_softmax, len=64, batch=8, axis=0 | 4 | 128 | 20678.9 | 215.2 | 2.0% | ok |
| log_softmax_f32_eager-single-call_len512_b8 | #1976 | eager-single-call | tenferro-rs | f32 | op=log_softmax, len=512, batch=8, axis=0 | 4 | 1 | 1995528.0 | 181558.0 | 6.6% | ok |
| log_softmax_f32_session-single-call_len512_b8 | #1976 | session-single-call | tenferro-rs | f32 | op=log_softmax, len=512, batch=8, axis=0 | 4 | 2 | 1952688.5 | 45640.0 | 2.7% | ok |
| log_softmax_f32_eager-composed_len512_b8 | #1976 | eager-composed | tenferro-rs | f32 | op=log_softmax, len=512, batch=8, axis=0 | 4 | 1 | 1972746.0 | 68087.5 | 2.3% | ok |
| log_softmax_f32_pytorch_len512_b8 | #1976 | pytorch | pytorch-cpu | f32 | op=log_softmax, len=512, batch=8, axis=0 | 4 | 2 | 941364.0 | 33359.5 | 2.0% | ok |
| masked_softmax_f32_eager-single-call_len64_b8 | #1976 | eager-single-call | tenferro-rs | f32 | op=masked_softmax, len=64, batch=8, axis=0 | 4 | 8 | 251646.4 | 1613.6 | 5.4% | ok |
| masked_softmax_f32_session-single-call_len64_b8 | #1976 | session-single-call | tenferro-rs | f32 | op=masked_softmax, len=64, batch=8, axis=0 | 4 | 8 | 207763.4 | 3979.3 | 11.1% | NOISY |
| masked_softmax_f32_eager-composed_len64_b8 | #1976 | eager-composed | tenferro-rs | f32 | op=masked_softmax, len=64, batch=8, axis=0 | 4 | 16 | 132059.8 | 2355.0 | 1.3% | ok |
| masked_softmax_f32_pytorch_len64_b8 | #1976 | pytorch | pytorch-cpu | f32 | op=masked_softmax, len=64, batch=8, axis=0 | 4 | 32 | 95880.8 | 1642.8 | 9.6% | ok |
| masked_softmax_f32_eager-single-call_len512_b8 | #1976 | eager-single-call | tenferro-rs | f32 | op=masked_softmax, len=512, batch=8, axis=0 | 4 | 1 | 4140776.0 | 175833.5 | 4.3% | ok |
| masked_softmax_f32_session-single-call_len512_b8 | #1976 | session-single-call | tenferro-rs | f32 | op=masked_softmax, len=512, batch=8, axis=0 | 4 | 1 | 3862066.0 | 139701.0 | 2.2% | ok |
| masked_softmax_f32_eager-composed_len512_b8 | #1976 | eager-composed | tenferro-rs | f32 | op=masked_softmax, len=512, batch=8, axis=0 | 4 | 1 | 2691846.0 | 160905.0 | 5.1% | ok |
| masked_softmax_f32_pytorch_len512_b8 | #1976 | pytorch | pytorch-cpu | f32 | op=masked_softmax, len=512, batch=8, axis=0 | 4 | 1 | 2316366.0 | 38311.5 | 10.4% | NOISY |
| softmax_f32_eager-single-call_len64_b8_t4 | #2233 #2235 | eager-single-call | tenferro-rs | f32 | op=softmax, len=64, batch=8, axis=0 | 4 | 64 | 22197.1 | 271.2 | 1.3% | ok |
| softmax_f32_eager-composed_len64_b8_t4 | #2233 #2235 | eager-composed | tenferro-rs | f32 | op=softmax, len=64, batch=8, axis=0 | 4 | 16 | 117290.4 | 1560.8 | 4.6% | ok |
| softmax_f32_pytorch_len64_b8_t4 | #2233 #2235 | pytorch | pytorch-cpu | f32 | op=softmax, len=64, batch=8, axis=0 | 4 | 128 | 19744.6 | 157.2 | 1.0% | ok |
| reduce_mean_f32_eager-single-call_1024x64_axis0 | #1976 | eager-single-call | tenferro-rs | f32 | rows=1024, cols=64, axis=0 | 4 | 128 | 20240.1 | 449.2 | 3.8% | ok |
| reduce_mean_f32_session-single-call_1024x64_axis0 | #1976 | session-single-call | tenferro-rs | f32 | rows=1024, cols=64, axis=0 | 4 | 256 | 13231.6 | 929.6 | 15.2% | NOISY |
| reduce_mean_f32_eager-composed_1024x64_axis0 | #1976 | eager-composed | tenferro-rs | f32 | rows=1024, cols=64, axis=0 | 4 | 128 | 19170.4 | 415.7 | 5.4% | ok |
| reduce_mean_f32_pytorch_1024x64_axis0 | #1976 | pytorch | pytorch-cpu | f32 | rows=1024, cols=64, axis=0 | 4 | 256 | 12892.2 | 78.9 | 5.4% | ok |
| take_along_axis_rows_f64_eager-single-call_n64_b256 | #2008 | eager-single-call | tenferro-rs | f64 | n=64, batch=256, axis=0 | 4 | 1 | 5700932.0 | 16069.5 | 3.8% | ok |
| take_along_axis_rows_f64_session-single-call_n64_b256 | #2008 | session-single-call | tenferro-rs | f64 | n=64, batch=256, axis=0 | 4 | 1 | 5379233.0 | 30672.0 | 2.0% | ok |
| take_along_axis_rows_f64_eager-gather-prebuilt_n64_b256 | #2008 | eager-gather-prebuilt | tenferro-rs | f64 | n=64, batch=256, axis=0 | 4 | 1 | 5208956.0 | 20598.5 | 0.7% | ok |
| take_along_axis_rows_f64_pytorch_n64_b256 | #2008 | pytorch | pytorch-cpu | f64 | n=64, batch=256, axis=0 | 4 | 2 | 1683131.0 | 17743.0 | 4.0% | ok |
| lstsq_f64_eager-shared_m768_n384_rhs16 | #2027 | eager-shared | tenferro-rs | f64 | operation=lstsq | 4 | 1 | 36653214.0 | 4162220.0 | 10.9% | NOISY |
| lstsq_f64_pytorch_m768_n384_rhs16 | #2027 | pytorch | pytorch-cpu | f64 | operation=lstsq | 4 | 1 | 6286744.0 | 49983.0 | 2.5% | ok |
| triangular_f64_concrete-shared_n4096_rhs64 | #2028 | concrete-shared | tenferro-rs | f64 | operation=triangular | 4 | 1 | 24970656.0 | 342774.0 | 1.7% | ok |
| triangular_f64_pytorch_n4096_rhs64 | #2028 | pytorch | pytorch-cpu | f64 | operation=triangular | 4 | 1 | 47559496.0 | 27320214.5 | 34.5% | NOISY |
| reshape_f64_concrete-shared_n33554432_to8192x4096 | #2029 | concrete-shared | tenferro-rs | f64 | operation=reshape | 4 | 1 | 42882141.0 | 605743.5 | 2.7% | ok |
| reshape_f64_pytorch_n33554432_to8192x4096 | #2029 | pytorch | pytorch-cpu | f64 | operation=reshape | 4 | 1 | 44406785.0 | 1194744.5 | 3.0% | ok |
| metadata_transpose_view_f64_rank8_t4 | #2180 | metadata-clone | tenferro-rs | f64 | shape=[2, 2, 2, 2, 2, 2, 2, 2] | 4 | 8192 | 77.6 | 1.1 | 71.3% | NOISY |
| metadata_transpose_view_f64_rank8_t4_julia-base | #2180 | julia-base | julia-base | f64 | shape=[2, 2, 2, 2, 2, 2, 2, 2] | 4 | 1048576 | 0.3 | 0.1 | 15.2% | NOISY |
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
| lu_batched_f64_factor_n8_b1024 | #2142 | tenferro | tenferro-rs | f64 | n=8, batch=1024, op=factor | 4 | 8 | 322492.2 | 1874.7 | 1.0% | ok |
| lu_batched_f64_factor_n8_b1024_host | #2142 | host | rust-loop | f64 | n=8, batch=1024, op=factor | 4 | 4 | 509852.5 | 1502.9 | 0.8% | ok |
| lu_batched_f64_solve_n8_b1024 | #2142 | tenferro | tenferro-rs | f64 | n=8, batch=1024, op=solve | 4 | 16 | 203858.7 | 1195.3 | 0.5% | ok |
| lu_batched_f64_solve_n8_b1024_pytorch | #2142 | pytorch | pytorch-cpu | f64 | n=8, batch=1024, op=solve | 4 | 16 | 172875.2 | 1070.1 | 1.0% | ok |
| lu_batched_f64_factor_n16_b1024 | #2142 | tenferro | tenferro-rs | f64 | n=16, batch=1024, op=factor | 4 | 2 | 1039822.5 | 3216.0 | 0.4% | ok |
| lu_batched_f64_factor_n16_b1024_host | #2142 | host | rust-loop | f64 | n=16, batch=1024, op=factor | 4 | 2 | 2161643.5 | 6349.5 | 0.5% | ok |
| lu_batched_f64_solve_n16_b1024 | #2142 | tenferro | tenferro-rs | f64 | n=16, batch=1024, op=solve | 4 | 8 | 306163.1 | 1661.2 | 0.6% | ok |
| lu_batched_f64_solve_n16_b1024_pytorch | #2142 | pytorch | pytorch-cpu | f64 | n=16, batch=1024, op=solve | 4 | 4 | 379322.0 | 10860.1 | 11.5% | NOISY |
| small_grad_qr_f64_trace-prepared_n4 | #2186 | trace-prepared | tenferro-rs | f64 | op=qr, n=4 | 4 | 32 | 90261.7 | 950.2 | 0.6% | ok |
| small_grad_qr_f64_pytorch_n4 | #2186 | pytorch | pytorch-cpu | f64 | op=qr, n=4 | 4 | 16 | 184928.4 | 4751.3 | 5.5% | ok |
| small_grad_qr_f64_trace-prepared_n16 | #2186 | trace-prepared | tenferro-rs | f64 | op=qr, n=16 | 4 | 32 | 100903.0 | 972.0 | 1.7% | ok |
| small_grad_qr_f64_pytorch_n16 | #2186 | pytorch | pytorch-cpu | f64 | op=qr, n=16 | 4 | 16 | 203506.8 | 2706.3 | 1.5% | ok |
| small_grad_eigh_f64_trace-prepared_n4 | #2186 | trace-prepared | tenferro-rs | f64 | op=eigh, n=4 | 4 | 32 | 113776.7 | 2272.5 | 1.0% | ok |
| small_grad_eigh_f64_pytorch_n4 | #2186 | pytorch | pytorch-cpu | f64 | op=eigh, n=4 | 4 | 16 | 145503.6 | 1542.6 | 1.9% | ok |
| small_grad_eigh_f64_trace-prepared_n16 | #2186 | trace-prepared | tenferro-rs | f64 | op=eigh, n=16 | 4 | 16 | 138364.8 | 1917.6 | 1.1% | ok |
| small_grad_eigh_f64_pytorch_n16 | #2186 | pytorch | pytorch-cpu | f64 | op=eigh, n=16 | 4 | 16 | 171798.3 | 3680.9 | 4.4% | ok |
| small_grad_lu_f64_trace-prepared_n4 | #2186 | trace-prepared | tenferro-rs | f64 | op=lu, n=4 | 4 | 64 | 49544.2 | 319.6 | 0.5% | ok |
| small_grad_lu_f64_pytorch_n4 | #2186 | pytorch | pytorch-cpu | f64 | op=lu, n=4 | 4 | 16 | 205703.4 | 3853.7 | 1.0% | ok |
| small_grad_lu_f64_trace-prepared_n16 | #2186 | trace-prepared | tenferro-rs | f64 | op=lu, n=16 | 4 | 64 | 59316.2 | 471.8 | 1.1% | ok |
| small_grad_lu_f64_pytorch_n16 | #2186 | pytorch | pytorch-cpu | f64 | op=lu, n=16 | 4 | 16 | 216697.6 | 2167.2 | 3.2% | ok |
| small_grad_svd_f64_trace-prepared_n4 | #2186 | trace-prepared | tenferro-rs | f64 | op=svd, n=4 | 4 | 16 | 165206.0 | 4515.6 | 2.0% | ok |
| small_grad_svd_f64_pytorch_n4 | #2186 | pytorch | pytorch-cpu | f64 | op=svd, n=4 | 4 | 16 | 190433.0 | 3006.5 | 1.4% | ok |
| small_grad_svd_f64_trace-prepared_n16 | #2186 | trace-prepared | tenferro-rs | f64 | op=svd, n=16 | 4 | 16 | 219596.1 | 4480.5 | 1.4% | ok |
| small_grad_svd_f64_pytorch_n16 | #2186 | pytorch | pytorch-cpu | f64 | op=svd, n=16 | 4 | 16 | 239943.9 | 3275.2 | 1.6% | ok |
| small_grad_solve_f64_trace-prepared_n4 | #2186 | trace-prepared | tenferro-rs | f64 | op=solve, n=4 | 4 | 64 | 34293.8 | 445.2 | 0.9% | ok |
| small_grad_solve_f64_pytorch_n4 | #2186 | pytorch | pytorch-cpu | f64 | op=solve, n=4 | 4 | 16 | 123036.8 | 2204.7 | 1.3% | ok |
| small_grad_solve_f64_trace-prepared_n16 | #2186 | trace-prepared | tenferro-rs | f64 | op=solve, n=16 | 4 | 64 | 45359.0 | 600.4 | 0.9% | ok |
| small_grad_solve_f64_pytorch_n16 | #2186 | pytorch | pytorch-cpu | f64 | op=solve, n=16 | 4 | 16 | 137525.6 | 2334.3 | 2.1% | ok |
| fft_axis0_c64_fft-executor_n128 | #1765 | fft-executor | tenferro-rs | c64 | n=128 | 4 | 1 | 6277090.0 | 79879.5 | 1.1% | ok |
| fft_axis0_c64_pytorch_n128 | #1765 | pytorch | pytorch-cpu | c64 | n=128 | 4 | 1 | 9137421.0 | 179339.0 | 2.2% | ok |
| degenerate_dot_hadamard_f64_dot-general-shared_n1024 | #2111 | dot-general-shared | tenferro-rs | f64 | form=hadamard, n=1024 | 4 | 4 | 337884.8 | 44014.6 | 14.9% | NOISY |
| degenerate_dot_hadamard_f64_pytorch_n1024 | #2111 | pytorch | pytorch-cpu | f64 | form=hadamard, n=1024 | 4 | 4 | 293394.5 | 60379.0 | 15.9% | NOISY |
| degenerate_dot_outer_f64_dot-general-shared_n1024 | #2111 | dot-general-shared | tenferro-rs | f64 | form=outer, n=1024 | 4 | 2 | 229072.0 | 36735.8 | 18.2% | NOISY |
| degenerate_dot_outer_f64_pytorch_n1024 | #2111 | pytorch | pytorch-cpu | f64 | form=outer, n=1024 | 4 | 4 | 150017.2 | 23612.6 | 13.1% | NOISY |
| degenerate_dot_rank0_f64_dot-general-shared_n1024 | #2111 | dot-general-shared | tenferro-rs | f64 | form=rank0, n=1024 | 4 | 2 | 1185945.5 | 64638.5 | 4.0% | ok |
| degenerate_dot_rank0_f64_pytorch_n1024 | #2111 | pytorch | pytorch-cpu | f64 | form=rank0, n=1024 | 4 | 4 | 261357.2 | 60128.4 | 33.1% | NOISY |
| erased_add_f64_session-add_n256 | #1766 | session-add | tenferro-rs | f64 | n=256 | 4 | 1024 | 2034.7 | 17.9 | 8.7% | ok |
| erased_add_f64_pytorch_n256 | #1766 | pytorch | pytorch-cpu | f64 | n=256 | 4 | 2048 | 1615.2 | 47.1 | 4.0% | ok |
| erased_add_f64_session-add_n4096 | #1766 | session-add | tenferro-rs | f64 | n=4096 | 4 | 256 | 3262.3 | 73.8 | 16.0% | NOISY |
| erased_add_f64_pytorch_n4096 | #1766 | pytorch | pytorch-cpu | f64 | n=4096 | 4 | 512 | 2706.6 | 56.6 | 5.2% | ok |
| erased_add_f64_session-add_n1048576 | #1766 | session-add | tenferro-rs | f64 | n=1048576 | 4 | 4 | 307595.8 | 30846.1 | 15.4% | NOISY |
| erased_add_f64_pytorch_n1048576 | #1766 | pytorch | pytorch-cpu | f64 | n=1048576 | 4 | 4 | 252924.0 | 43126.9 | 19.8% | NOISY |
| axpby_strided_source_f64_1024x1024_t4 | #2220 #2214 | strided-source | tenferro-rs | f64 | n=1024 | 4 | 4 | 1089065.0 | 147465.9 | 11.8% | NOISY |
| axpby_contiguous_source_f64_1024x1024_t4 | #2220 #2214 | contiguous-source | tenferro-rs | f64 | n=1024 | 4 | 8 | 369094.2 | 6367.6 | 5.2% | ok |
| axpby_strided_source_f64_1024x1024_host_t4 | #2220 #2214 | host | rust-loop | f64 | n=1024 | 4 | 2 | 1590469.5 | 18254.2 | 1.0% | ok |
| erased_construct_f64_col_n8 | #2177 | col-major | tenferro-rs | f64 | n=8, layout=col | 4 | 16384 | 98.3 | 1.5 | 30.6% | NOISY |
| erased_construct_f64_row_n8 | #2177 | row-major | tenferro-rs | f64 | n=8, layout=row | 4 | 8192 | 208.2 | 5.4 | 15.7% | NOISY |
| erased_construct_f64_col_n4096 | #2177 | col-major | tenferro-rs | f64 | n=4096, layout=col | 4 | 16384 | 232.9 | 1.6 | 1.0% | ok |
| erased_construct_f64_row_n4096 | #2177 | row-major | tenferro-rs | f64 | n=4096, layout=row | 4 | 128 | 19803.4 | 127.8 | 1.5% | ok |
| erased_construct_f64_col_n65536 | #2177 | col-major | tenferro-rs | f64 | n=65536, layout=col | 4 | 1024 | 99.4 | 0.7 | 0.5% | ok |
| erased_construct_f64_row_n65536 | #2177 | row-major | tenferro-rs | f64 | n=65536, layout=row | 4 | 8 | 336267.9 | 2254.9 | 0.6% | ok |
| eager_small_leaf_f64_eager-held-session | #1704 | eager-held-session | tenferro-rs | f64 | op=leaf | 4 | 512 | 3178.3 | 94.8 | 4.5% | ok |
| eager_small_add_f64_eager-held-session | #1667 | eager-held-session | tenferro-rs | f64 | op=add | 4 | 512 | 3177.5 | 102.8 | 13.7% | NOISY |
| eager_small_add_f64_pytorch | #1667 | pytorch | pytorch-cpu | f64 | op=add | 4 | 2048 | 1526.1 | 11.3 | 0.7% | ok |
| eager_small_gemm_f64_eager-held-session | #1904 | eager-held-session | tenferro-rs | f64 | op=gemm | 4 | 512 | 4270.3 | 39.0 | 0.8% | ok |
| eager_small_gemm_f64_pytorch | #1904 | pytorch | pytorch-cpu | f64 | op=gemm | 4 | 2048 | 1704.4 | 11.9 | 0.6% | ok |
| trace_metadata_trace-fold_f32_1024x64_k50 | #2226 #2238 | trace-fold | tenferro-rs | f32 | rows=1024, cols=64, steps=50 | 4 | 2 | 1948340.5 | 18068.8 | 0.9% | ok |
| trace_metadata_trace-outputs_f32_1024x64_k50 | #2226 #2238 | trace-outputs | tenferro-rs | f32 | rows=1024, cols=64, steps=50 | 4 | 2 | 1958173.5 | 128010.8 | 4.9% | ok |
| trace_metadata_eager_f32_1024x64_k50 | #2226 #2238 | eager | tenferro-rs | f32 | rows=1024, cols=64, steps=50 | 4 | 1 | 2955337.0 | 31303.0 | 9.9% | ok |
| trace_metadata_pytorch_f32_1024x64_k50 | #2226 #2238 | pytorch | pytorch-cpu | f32 | rows=1024, cols=64, steps=50 | 4 | 2 | 1636709.5 | 11654.2 | 0.7% | ok |
| trace_metadata_fixed_neg_f32_8 | #2226 #2238 | trace-fold | tenferro-rs | f32 | rows=8, cols=1, steps=0 | 4 | 512 | 4555.1 | 20.4 | 10.2% | NOISY |


## Timing boundaries

- **activation / eager-composed** — inside: hand-written formula from pre-B1 eager primitives, intermediate and output allocation; outside: backend/runtime construction, input, constant, mask and index construction, eager session entry, correctness_check.
- **activation / eager-single-call** — inside: one activation call, intermediate and output allocation; outside: backend/runtime construction, input, constant, mask and index construction, eager session entry, correctness_check.
- **activation / pytorch** — inside: torch gelu, output_allocation; outside: input_construction, torch.set_num_threads, correctness_check.
- **activation / session-single-call** — inside: one activation call, intermediate and output allocation; outside: backend/runtime construction, input, constant, mask and index construction, with_backend_session entry (CpuBackend::new()), correctness_check.
- **axpby_strided_source / contiguous-source** — inside: Engine::axpby_read_into_accum (contiguous owned source, compact owned destination); outside: backend/session construction, input_construction, strided/offset view construction (metadata only), per-sample destination allocation and seed restoration, correctness_check.
- **axpby_strided_source / host** — inside: sequential Rust strided axpby loop into a compact owned destination; outside: backend/session construction, input_construction, strided/offset view construction (metadata only), per-sample destination allocation and seed restoration, correctness_check.
- **axpby_strided_source / strided-source** — inside: Engine::axpby_read_into_accum (strided/offset source view, compact owned destination); outside: backend/session construction, input_construction, strided/offset view construction (metadata only), per-sample destination allocation and seed restoration, correctness_check.
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
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: activation_erf_f32_pytorch_1024x64 (1 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: activation_sigmoid_f32_eager-single-call_1024x64 (1 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: activation_sigmoid_f32_pytorch_1024x64 (1 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: activation_silu_f32_eager-single-call_1024x64 (1 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: activation_silu_f32_pytorch_1024x64 (1 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: activation_softplus_f32_eager-single-call_1024x64 (1 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: activation_softplus_f32_pytorch_1024x64 (1 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: activation_gelu_tanh_f32_eager-single-call_1024x64 (1 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: activation_gelu_tanh_f32_pytorch_1024x64 (1 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_structural_broadcast_in_dim_direct-shared (1 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_structural_broadcast_in_dim_pytorch (1 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_real_tanh_direct-shared (1 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_real_tanh_jax (1 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_real_log1p_direct-shared (1 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_real_log1p_pytorch (1 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_real_cos_direct-shared (1 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_real_cos_pytorch (1 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_real_sin_direct-shared (1 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_real_sin_pytorch (1 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_real_exp_direct-shared (1 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_real_exp_jax (1 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_real_expm1_direct-shared (1 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_real_expm1_jax (1 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_real_pow_direct-shared (1 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_real_pow_pytorch (1 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_complex_exp_direct-shared (1 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_complex_exp_julia-base (1 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_real_log_direct-shared (1 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_real_log_pytorch (1 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_real_reduce_max_axis1_direct-shared (1 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_real_reduce_max_axis1_julia-base (1 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_real_reduce_prod_axis1_direct-shared (1 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_real_reduce_prod_axis1_pytorch (1 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_real_reduce_sum_axis1_direct-shared (1 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_real_reduce_sum_axis1_pytorch (1 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_real_reduce_max_axis0_direct-shared (1 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_real_reduce_max_axis0_julia-base (1 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_real_reduce_min_axis1_direct-shared (1 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_real_reduce_min_axis1_julia-base (1 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_real_reduce_min_axis0_direct-shared (1 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_real_reduce_min_axis0_julia-base (1 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_real_reduce_max_all_direct-shared (1 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_real_reduce_max_all_julia-base (1 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_real_reduce_min_all_direct-shared (1 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_real_reduce_min_all_julia-base (1 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_complex_norm_fro_direct-shared (1 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_complex_norm_fro_jax (1 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_index_gather_direct-shared (1 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_index_gather_jax (1 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_index_scatter_direct-shared (1 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_index_scatter_pytorch (1 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_index_dynamic_update_slice_direct-shared (1 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_index_dynamic_update_slice_jax (1 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_perm_reverse_15d_3_materialize (1 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_perm_reverse_15d_3_strided-rs (1 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_structural_transpose_direct-shared (1 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_structural_transpose_pytorch (1 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_perm_reverse_23d_2_materialize (1 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_perm_reverse_23d_2_strided-rs (1 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_perm_transpose_3d_256_201_materialize (1 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_perm_transpose_3d_256_201_strided-rs (1 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_perm_cyclic_15d_3_materialize (1 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_perm_cyclic_15d_3_strided-rs (1 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_perm_transpose_3d_256_102_materialize (1 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_perm_transpose_3d_256_102_strided-rs (1 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_perm_tn_light_415_24d_scattered_to_colmajor_materialize (1 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_perm_tn_light_415_24d_scattered_to_colmajor_strided-rs (1 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_perm_tn_light_415_24d_contiguous_same_perm_materialize (1 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_perm_tn_light_415_24d_contiguous_same_perm_strided-rs (1 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_perm_rotation_6d_32_32_32_32_16_16_materialize (1 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_perm_rotation_6d_32_32_32_32_16_16_strided-rs (1 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_structural_extract_diagonal_direct-shared (1 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_structural_extract_diagonal_jax (1 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_fft_irfft_direct-shared (1 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_fft_irfft_mkl-dfti (1 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_fft_rfft_direct-shared (1 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_fft_rfft_mkl-dfti (1 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_fft_fft_direct-shared (1 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_fft_fft_mkl-dfti (1 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_fft_ifft_direct-shared (1 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_fft_ifft_mkl-dfti (1 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_metadata_transpose_view_metadata (1 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_metadata_transpose_view_julia-base (1 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_metadata_transpose_view_metadata-host-dyn (1 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_metadata_transpose_view_metadata-host-static (1 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_metadata_slice_view_metadata (1 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_metadata_slice_view_julia-base (1 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_metadata_slice_view_metadata-host-dyn (1 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_metadata_slice_view_metadata-host-static (1 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_metadata_reshape_view_metadata (1 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_metadata_reshape_view_julia-base (1 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_metadata_reshape_view_metadata-host-dyn (1 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_metadata_reshape_view_metadata-host-static (1 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: activation_erf_f32_eager-single-call_1024x64 (4 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: activation_erf_f32_pytorch_1024x64 (4 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: activation_sigmoid_f32_eager-single-call_1024x64 (4 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: activation_sigmoid_f32_pytorch_1024x64 (4 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: activation_silu_f32_eager-single-call_1024x64 (4 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: activation_silu_f32_pytorch_1024x64 (4 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: activation_softplus_f32_eager-single-call_1024x64 (4 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: activation_softplus_f32_pytorch_1024x64 (4 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: activation_gelu_tanh_f32_eager-single-call_1024x64 (4 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: activation_gelu_tanh_f32_pytorch_1024x64 (4 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_structural_broadcast_in_dim_direct-shared (4 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_structural_broadcast_in_dim_pytorch (4 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_real_tanh_direct-shared (4 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_real_tanh_jax (4 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_real_log1p_direct-shared (4 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_real_log1p_pytorch (4 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_real_cos_direct-shared (4 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_real_cos_pytorch (4 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_real_sin_direct-shared (4 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_real_sin_pytorch (4 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_real_exp_direct-shared (4 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_real_exp_jax (4 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_real_expm1_direct-shared (4 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_real_expm1_jax (4 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_real_pow_direct-shared (4 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_real_pow_pytorch (4 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_complex_exp_direct-shared (4 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_complex_exp_julia-base (4 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_real_log_direct-shared (4 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_real_log_pytorch (4 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_real_reduce_max_axis1_direct-shared (4 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_real_reduce_max_axis1_julia-base (4 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_real_reduce_prod_axis1_direct-shared (4 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_real_reduce_prod_axis1_pytorch (4 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_real_reduce_sum_axis1_direct-shared (4 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_real_reduce_sum_axis1_pytorch (4 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_real_reduce_max_axis0_direct-shared (4 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_real_reduce_max_axis0_julia-base (4 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_real_reduce_min_axis1_direct-shared (4 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_real_reduce_min_axis1_julia-base (4 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_real_reduce_min_axis0_direct-shared (4 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_real_reduce_min_axis0_julia-base (4 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_real_reduce_max_all_direct-shared (4 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_real_reduce_max_all_julia-base (4 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_real_reduce_min_all_direct-shared (4 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_real_reduce_min_all_julia-base (4 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_complex_norm_fro_direct-shared (4 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_complex_norm_fro_jax (4 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_index_gather_direct-shared (4 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_index_gather_jax (4 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_index_scatter_direct-shared (4 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_index_scatter_pytorch (4 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_index_dynamic_update_slice_direct-shared (4 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_index_dynamic_update_slice_jax (4 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_perm_reverse_15d_3_materialize (4 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_perm_reverse_15d_3_strided-rs (4 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_structural_transpose_direct-shared (4 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_structural_transpose_pytorch (4 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_perm_reverse_23d_2_materialize (4 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_perm_reverse_23d_2_strided-rs (4 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_perm_transpose_3d_256_201_materialize (4 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_perm_transpose_3d_256_201_strided-rs (4 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_perm_cyclic_15d_3_materialize (4 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_perm_cyclic_15d_3_strided-rs (4 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_perm_transpose_3d_256_102_materialize (4 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_perm_transpose_3d_256_102_strided-rs (4 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_perm_tn_light_415_24d_scattered_to_colmajor_materialize (4 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_perm_tn_light_415_24d_scattered_to_colmajor_strided-rs (4 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_perm_tn_light_415_24d_contiguous_same_perm_materialize (4 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_perm_tn_light_415_24d_contiguous_same_perm_strided-rs (4 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_perm_rotation_6d_32_32_32_32_16_16_materialize (4 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_perm_rotation_6d_32_32_32_32_16_16_strided-rs (4 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_structural_extract_diagonal_direct-shared (4 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_structural_extract_diagonal_jax (4 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_fft_irfft_direct-shared (4 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_fft_irfft_mkl-dfti (4 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_fft_rfft_direct-shared (4 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_fft_rfft_mkl-dfti (4 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_fft_fft_direct-shared (4 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_fft_fft_mkl-dfti (4 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_fft_ifft_direct-shared (4 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_fft_ifft_mkl-dfti (4 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_metadata_transpose_view_metadata (4 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_metadata_transpose_view_julia-base (4 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_metadata_transpose_view_metadata-host-dyn (4 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_metadata_transpose_view_metadata-host-static (4 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_metadata_slice_view_metadata (4 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_metadata_slice_view_julia-base (4 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_metadata_slice_view_metadata-host-dyn (4 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_metadata_slice_view_metadata-host-static (4 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_metadata_reshape_view_metadata (4 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_metadata_reshape_view_julia-base (4 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_metadata_reshape_view_metadata-host-dyn (4 threads)
```
FileNotFoundError(2, 'No such file or directory')
```

### FAILED: followup_metadata_reshape_view_metadata-host-static (4 threads)
```
FileNotFoundError(2, 'No such file or directory')
```
