# CPU Performance-Issue Workloads

- Suite: `cpu/perf_issues`
- Raw run: `data/results/amd-cpu/cpu/perf_issues/20261109_195017`

## CPU Information

- Model: `AMD EPYC 9V74 80-Core Processor`
- Vendor: `AuthenticAMD`
- Logical CPUs: `4`
- Sockets: `1`
- Cores per socket: `2`
- Threads per core: `2`
- NUMA nodes: `1`
- Transparent huge pages: `madvise (always [madvise] never)`
- Transparent huge page defrag: `madvise (always defer defer+madvise [madvise] never)`
- Python platform: `Linux-7.0.0-1019-azure-x86_64-with-glibc2.39`
- L3 domains (shared CPUs): `0-3`
- Container: `devcontainer`

## run_t1

Full metadata: `data/results/amd-cpu/cpu/perf_issues/20261109_195017/run_t1.yaml`

```yaml
timestamp: '2026-11-09T19:50:17Z'
tenferro_rs:
  path: /workspaces/tenferro-benchmark/extern/tenferro-rs
  commit: 9a850b06155405acc99ecd0648829e9e6d3866ba
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
  container: devcontainer
  hostname: ef1fa85bba25
  os: Linux-7.0.0-1019-azure-x86_64-with-glibc2.39
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
    BENCHMARK_COMMIT: ad9aa8a2fc06c36f8ce1d1dfb1ce9fb1cd9a9c46
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

Full metadata: `data/results/amd-cpu/cpu/perf_issues/20261109_195017/run_t4.yaml`

```yaml
timestamp: '2026-11-09T19:57:17Z'
tenferro_rs:
  path: /workspaces/tenferro-benchmark/extern/tenferro-rs
  commit: 9a850b06155405acc99ecd0648829e9e6d3866ba
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
  container: devcontainer
  hostname: ef1fa85bba25
  os: Linux-7.0.0-1019-azure-x86_64-with-glibc2.39
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
    BENCHMARK_COMMIT: ad9aa8a2fc06c36f8ce1d1dfb1ce9fb1cd9a9c46
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

- Manifest: `cpu/perf_issues` version 22, coverage `quick`, effort `standard`
- Expected 681, selected 681, executed 681, unsupported 0, failed 0, skipped 0, missing 0, noisy 35.
- Complete: **yes** — only executed cases count as covered; unsupported, skipped, failed, missing and unselected cases never do.

| expected | selected | executed | unsupported | failed | skipped | missing | noisy |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 681 | 681 | 681 | 0 | 0 | 0 | 0 | 35 |

- Noisy: `decode_proj_f32_eager-shared_in1024_out4096_len512@t1`, `conj_dot_c64_pytorch-vdot_len100@t1`, `small_solve_f64_triangular-solve-loop_k64_b64@t1`, `rms_norm_f32_pytorch-fused_d1024_len8_b1@t1`, `metadata_transpose_view_f64_rank8_t1@t1`, `followup_metadata_slice_view_julia-base@t1`, `followup_metadata_slice_view_metadata-host-static@t1`, `followup_metadata_reshape_view_julia-base@t1`, `followup_metadata_reshape_view_metadata-host-static@t1`, `erased_construct_f64_col_n8@t1`, `erased_construct_f64_row_n8@t1`, `two_operand_einsum_f64_2x2_subscripts@t4`, `two_operand_einsum_c64_2x2_string@t4`, `decode_proj_f32_eager-shared_in1024_out4096_len512@t4`, `decode_proj_f32_pytorch_in1024_out4096_len512@t4`, `conj_dot_c64_pytorch-vdot_len100@t4`, `small_solve_f64_triangular-solve-loop_k64_b64@t4`, `activation_erf_f32_pytorch_1024x64@t4`, `activation_sigmoid_f32_pytorch_1024x64@t4`, `activation_silu_f32_pytorch_1024x64@t4`, `triangular_f64_pytorch_n4096_rhs64@t4`, `metadata_transpose_view_f64_rank8_t4@t4`, `followup_real_log_pytorch@t4`, `followup_real_reduce_min_all_julia-base@t4`, `followup_perm_reverse_15d_3_materialize@t4`, `followup_metadata_slice_view_julia-base@t4`, `followup_metadata_reshape_view_julia-base@t4`, `lu_batched_f64_factor_n8_b1024_host@t4`, `degenerate_dot_outer_f64_dot-general-shared_n1024@t4`, `degenerate_dot_outer_f64_pytorch_n1024@t4`, `degenerate_dot_rank0_f64_pytorch_n1024@t4`, `erased_add_f64_session-add_n256@t4`, `erased_add_f64_session-add_n4096@t4`, `erased_construct_f64_col_n8@t4`, `erased_construct_f64_row_n8@t4`

Numerical checks: **681/681 recorded rows passed** (case × thread count).

## Steady-state operation rows

Each batch is one wall-clock interval over many operations; every output is retained until the clock stops. Median and inclusive IQR are per operation, in nanoseconds; CoV is sample standard deviation / mean (NOISY means CoV > 10%, a descriptive label only). Setup (inputs, backends, runtimes, session entry, tracing/compilation, preparation) is outside timing; faer-direct and host-sgemm write into a preallocated output, the tenferro and PyTorch arms allocate their output. The thin_complex_gemm faer-direct arm instead allocates its output inside timing to match its product references. `reference_for` in the case file names the case each reference arm is compared with. GPU batches end with one device synchronize (a per-call synchronize where the transfer is the operation); the Threads column is the host thread setting.

### Compute- and bandwidth-bound operations

| Case ID | Issues | Arm | Backend | Dtype | Workload | Threads | Ops/batch | Median ns/op | IQR ns | CoV | Status |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| softmax_f64_pool_serial_1024x4096 | #2273 #2274 #2276 | serial | tenferro-rs | f64 | rows=1024, cols=4096, axis=0 | 1 | 1 | 10754083.0 | 62928.5 | 1.5% | ok |
| softmax_f64_pool_pytorch_t1_1024x4096 | #2273 #2274 #2276 | pytorch | pytorch-cpu | f64 | rows=1024, cols=4096, axis=0 | 1 | 1 | 19207383.0 | 562527.0 | 1.8% | ok |
| softmax_f64_pool_pool4_1024x4096 | #2273 #2274 #2276 | pool4 | tenferro-rs | f64 | rows=1024, cols=4096, axis=0 | 4 | 1 | 3969180.0 | 16379.0 | 0.8% | ok |
| softmax_f64_pool_pytorch_t4_1024x4096 | #2273 #2274 #2276 | pytorch | pytorch-cpu | f64 | rows=1024, cols=4096, axis=0 | 4 | 1 | 6956066.0 | 78165.5 | 2.2% | ok |

### Latency and per-call overhead

These rows measure dispatch/operation latency; they do not measure compute throughput.

| Case ID | Issues | Arm | Backend | Dtype | Workload | Threads | Ops/batch | Median ns/op | IQR ns | CoV | Status |
|---|---|---|---|---|---|---:|---:|---:|---:|---:|---|
| two_operand_einsum_f64_2x2_string | #2325 #2324 | string | tenferro-rs | f64 | m=2, k=2, n=2, format=ij,jk->ik | 1 | 1024 | 2393.3 | 5.1 | 0.2% | ok |
| two_operand_einsum_f64_2x2_subscripts | #2325 #2324 | subscripts | tenferro-rs | f64 | m=2, k=2, n=2, format=ij,jk->ik | 1 | 2048 | 1630.4 | 7.1 | 0.5% | ok |
| two_operand_einsum_f64_2x2_prepared | #2325 #2324 | prepared | tenferro-rs | f64 | m=2, k=2, n=2, format=ij,jk->ik | 1 | 2048 | 1394.3 | 5.9 | 0.5% | ok |
| two_operand_einsum_f64_2x2_dot-general | #2325 #2324 | dot-general | tenferro-rs | f64 | m=2, k=2, n=2, format=ij,jk->ik | 1 | 2048 | 1355.8 | 12.3 | 1.5% | ok |
| two_operand_einsum_f64_2x2_pytorch | #2325 #2324 | pytorch | pytorch-cpu | f64 | m=2, k=2, n=2, format=ij,jk->ik | 1 | 256 | 8765.0 | 72.7 | 0.6% | ok |
| two_operand_einsum_f64_16x16_string | #2325 #2324 | string | tenferro-rs | f64 | m=16, k=16, n=16, format=ij,jk->ik | 1 | 1024 | 3072.6 | 27.5 | 1.3% | ok |
| two_operand_einsum_f64_16x16_subscripts | #2325 #2324 | subscripts | tenferro-rs | f64 | m=16, k=16, n=16, format=ij,jk->ik | 1 | 1024 | 2320.8 | 32.3 | 1.3% | ok |
| two_operand_einsum_f64_16x16_prepared | #2325 #2324 | prepared | tenferro-rs | f64 | m=16, k=16, n=16, format=ij,jk->ik | 1 | 1024 | 2082.5 | 10.9 | 1.7% | ok |
| two_operand_einsum_f64_16x16_dot-general | #2325 #2324 | dot-general | tenferro-rs | f64 | m=16, k=16, n=16, format=ij,jk->ik | 1 | 1024 | 2028.3 | 30.1 | 1.7% | ok |
| two_operand_einsum_f64_16x16_pytorch | #2325 #2324 | pytorch | pytorch-cpu | f64 | m=16, k=16, n=16, format=ij,jk->ik | 1 | 256 | 9736.3 | 141.8 | 1.0% | ok |
| two_operand_einsum_c64_2x2_string | #2325 #2324 | string | tenferro-rs | c64 | m=2, k=2, n=2, format=ij,jk->ik | 1 | 1024 | 2280.7 | 21.4 | 1.8% | ok |
| two_operand_einsum_c64_2x2_subscripts | #2325 #2324 | subscripts | tenferro-rs | c64 | m=2, k=2, n=2, format=ij,jk->ik | 1 | 2048 | 1622.5 | 10.2 | 2.4% | ok |
| two_operand_einsum_c64_2x2_prepared | #2325 #2324 | prepared | tenferro-rs | c64 | m=2, k=2, n=2, format=ij,jk->ik | 1 | 2048 | 1373.1 | 6.2 | 2.7% | ok |
| two_operand_einsum_c64_2x2_dot-general | #2325 #2324 | dot-general | tenferro-rs | c64 | m=2, k=2, n=2, format=ij,jk->ik | 1 | 2048 | 1320.6 | 3.6 | 3.1% | ok |
| two_operand_einsum_c64_2x2_pytorch | #2325 #2324 | pytorch | pytorch-cpu | c64 | m=2, k=2, n=2, format=ij,jk->ik | 1 | 256 | 8778.8 | 179.9 | 2.7% | ok |
| two_operand_einsum_c64_16x16_string | #2325 #2324 | string | tenferro-rs | c64 | m=16, k=16, n=16, format=ij,jk->ik | 1 | 512 | 3374.5 | 35.9 | 1.0% | ok |
| two_operand_einsum_c64_16x16_subscripts | #2325 #2324 | subscripts | tenferro-rs | c64 | m=16, k=16, n=16, format=ij,jk->ik | 1 | 1024 | 2710.2 | 43.6 | 1.0% | ok |
| two_operand_einsum_c64_16x16_prepared | #2325 #2324 | prepared | tenferro-rs | c64 | m=16, k=16, n=16, format=ij,jk->ik | 1 | 1024 | 2452.4 | 26.4 | 1.1% | ok |
| two_operand_einsum_c64_16x16_dot-general | #2325 #2324 | dot-general | tenferro-rs | c64 | m=16, k=16, n=16, format=ij,jk->ik | 1 | 1024 | 2402.2 | 47.0 | 2.5% | ok |
| two_operand_einsum_c64_16x16_pytorch | #2325 #2324 | pytorch | pytorch-cpu | c64 | m=16, k=16, n=16, format=ij,jk->ik | 1 | 256 | 10041.9 | 261.1 | 4.1% | ok |
| two_operand_einsum_f64_abc_cef_abef_string | #2325 #2324 | string | tenferro-rs | f64 | m=8, k=2, n=16, format=abc,cef->abef | 1 | 1024 | 2809.5 | 23.6 | 0.8% | ok |
| two_operand_einsum_f64_abc_cef_abef_subscripts | #2325 #2324 | subscripts | tenferro-rs | f64 | m=8, k=2, n=16, format=abc,cef->abef | 1 | 1024 | 1892.5 | 8.5 | 0.7% | ok |
| two_operand_einsum_f64_abc_cef_abef_prepared | #2325 #2324 | prepared | tenferro-rs | f64 | m=8, k=2, n=16, format=abc,cef->abef | 1 | 1024 | 1603.1 | 12.8 | 1.0% | ok |
| two_operand_einsum_f64_abc_cef_abef_dot-general | #2325 #2324 | dot-general | tenferro-rs | f64 | m=8, k=2, n=16, format=abc,cef->abef | 1 | 1024 | 1519.2 | 7.0 | 1.2% | ok |
| two_operand_einsum_f64_abc_cef_abef_pytorch | #2325 #2324 | pytorch | pytorch-cpu | f64 | m=8, k=2, n=16, format=abc,cef->abef | 1 | 256 | 10749.7 | 110.0 | 0.7% | ok |
| thin_c64_gemm_1x141_141x282_t1 | #2416 #1099 | dot-general | tenferro-rs | c64 | m=1, k=141, n=282 | 1 | 128 | 16223.5 | 111.1 | 0.7% | ok |
| thin_c64_gemm_1x141_141x282_t1_faer-direct | #2416 #1099 | faer-direct | faer-direct | c64 | m=1, k=141, n=282 | 1 | 256 | 13058.4 | 131.6 | 0.7% | ok |
| thin_c64_gemm_1x141_141x282_t1_pytorch | #2416 #1099 | pytorch | pytorch-cpu | c64 | m=1, k=141, n=282 | 1 | 64 | 29564.8 | 242.9 | 1.2% | ok |
| thin_c64_gemm_2x141_141x282_t1 | #2416 #1099 | dot-general | tenferro-rs | c64 | m=2, k=141, n=282 | 1 | 64 | 31905.5 | 214.0 | 0.6% | ok |
| thin_c64_gemm_2x141_141x282_t1_faer-direct | #2416 #1099 | faer-direct | faer-direct | c64 | m=2, k=141, n=282 | 1 | 128 | 26413.1 | 247.1 | 3.1% | ok |
| thin_c64_gemm_2x141_141x282_t1_pytorch | #2416 #1099 | pytorch | pytorch-cpu | c64 | m=2, k=141, n=282 | 1 | 64 | 43979.2 | 416.9 | 0.7% | ok |
| thin_c64_gemm_8x141_141x282_t1 | #2416 #1099 | dot-general | tenferro-rs | c64 | m=8, k=141, n=282 | 1 | 8 | 254923.1 | 2151.2 | 7.6% | ok |
| thin_c64_gemm_8x141_141x282_t1_faer-direct | #2416 #1099 | faer-direct | faer-direct | c64 | m=8, k=141, n=282 | 1 | 64 | 61028.1 | 777.8 | 0.8% | ok |
| thin_c64_gemm_8x141_141x282_t1_pytorch | #2416 #1099 | pytorch | pytorch-cpu | c64 | m=8, k=141, n=282 | 1 | 32 | 66985.3 | 603.7 | 0.7% | ok |
| decode_proj_f32_eager-shared_in1024_out1024_len8 | #1992 #2003 #1995 #2229 | eager-shared | tenferro-rs | f32 | in=1024, out=1024, len=8 | 1 | 4 | 859425.8 | 3183.5 | 0.7% | ok |
| decode_proj_f32_faer-direct_in1024_out1024_len8 | #1992 #2003 #1995 #2229 | faer-direct | faer-direct | f32 | in=1024, out=1024, len=8 | 1 | 8 | 303557.6 | 2248.4 | 0.7% | ok |
| decode_proj_f32_host-sgemm_in1024_out1024_len8 | #1992 #2003 #1995 #2229 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=1024, len=8 | 1 | 2 | 1240084.0 | 5130.5 | 0.6% | ok |
| decode_proj_f32_pytorch_in1024_out1024_len8 | #1992 #2003 #1995 #2229 | pytorch | pytorch-cpu | f32 | in=1024, out=1024, len=8 | 1 | 8 | 374742.9 | 6201.0 | 1.3% | ok |
| decode_proj_f32_eager-shared_in1024_out1024_len64 | #1992 #2003 #1995 #2229 | eager-shared | tenferro-rs | f32 | in=1024, out=1024, len=64 | 1 | 2 | 1405307.5 | 6577.2 | 0.4% | ok |
| decode_proj_f32_faer-direct_in1024_out1024_len64 | #1992 #2003 #1995 #2229 | faer-direct | faer-direct | f32 | in=1024, out=1024, len=64 | 1 | 2 | 1189799.0 | 4932.5 | 0.3% | ok |
| decode_proj_f32_host-sgemm_in1024_out1024_len64 | #1992 #2003 #1995 #2229 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=1024, len=64 | 1 | 1 | 2154897.0 | 4381.5 | 0.3% | ok |
| decode_proj_f32_pytorch_in1024_out1024_len64 | #1992 #2003 #1995 #2229 | pytorch | pytorch-cpu | f32 | in=1024, out=1024, len=64 | 1 | 2 | 1523568.0 | 11364.0 | 0.5% | ok |
| decode_proj_f32_eager-shared_in1024_out1024_len512 | #1992 #2003 #1995 #2229 | eager-shared | tenferro-rs | f32 | in=1024, out=1024, len=512 | 1 | 1 | 10786419.0 | 127449.0 | 1.3% | ok |
| decode_proj_f32_faer-direct_in1024_out1024_len512 | #1992 #2003 #1995 #2229 | faer-direct | faer-direct | f32 | in=1024, out=1024, len=512 | 1 | 1 | 9734895.0 | 151248.5 | 0.9% | ok |
| decode_proj_f32_host-sgemm_in1024_out1024_len512 | #1992 #2003 #1995 #2229 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=1024, len=512 | 1 | 1 | 11096688.0 | 66989.0 | 0.8% | ok |
| decode_proj_f32_pytorch_in1024_out1024_len512 | #1992 #2003 #1995 #2229 | pytorch | pytorch-cpu | f32 | in=1024, out=1024, len=512 | 1 | 1 | 10567356.0 | 226259.5 | 1.7% | ok |
| decode_proj_f32_eager-shared_in1024_out4096_len8 | #1992 #2003 #1995 #2229 | eager-shared | tenferro-rs | f32 | in=1024, out=4096, len=8 | 1 | 1 | 3852596.0 | 497365.0 | 6.8% | ok |
| decode_proj_f32_faer-direct_in1024_out4096_len8 | #1992 #2003 #1995 #2229 | faer-direct | faer-direct | f32 | in=1024, out=4096, len=8 | 1 | 2 | 1298920.5 | 66904.0 | 4.9% | ok |
| decode_proj_f32_host-sgemm_in1024_out4096_len8 | #1992 #2003 #1995 #2229 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=4096, len=8 | 1 | 1 | 5059291.0 | 35808.0 | 0.6% | ok |
| decode_proj_f32_pytorch_in1024_out4096_len8 | #1992 #2003 #1995 #2229 | pytorch | pytorch-cpu | f32 | in=1024, out=4096, len=8 | 1 | 2 | 1873901.0 | 164728.5 | 5.6% | ok |
| decode_proj_f32_eager-shared_in1024_out4096_len64 | #1992 #2003 #1995 #2229 | eager-shared | tenferro-rs | f32 | in=1024, out=4096, len=64 | 1 | 1 | 6356739.0 | 383086.5 | 3.2% | ok |
| decode_proj_f32_faer-direct_in1024_out4096_len64 | #1992 #2003 #1995 #2229 | faer-direct | faer-direct | f32 | in=1024, out=4096, len=64 | 1 | 1 | 5039672.0 | 45863.5 | 0.6% | ok |
| decode_proj_f32_host-sgemm_in1024_out4096_len64 | #1992 #2003 #1995 #2229 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=4096, len=64 | 1 | 1 | 9136630.0 | 79232.0 | 1.0% | ok |
| decode_proj_f32_pytorch_in1024_out4096_len64 | #1992 #2003 #1995 #2229 | pytorch | pytorch-cpu | f32 | in=1024, out=4096, len=64 | 1 | 1 | 6789479.0 | 106583.0 | 2.8% | ok |
| decode_proj_f32_eager-shared_in1024_out4096_len512 | #1992 #2003 #1995 #2229 | eager-shared | tenferro-rs | f32 | in=1024, out=4096, len=512 | 1 | 1 | 46432067.0 | 11742268.0 | 18.1% | NOISY |
| decode_proj_f32_faer-direct_in1024_out4096_len512 | #1992 #2003 #1995 #2229 | faer-direct | faer-direct | f32 | in=1024, out=4096, len=512 | 1 | 1 | 39477594.0 | 244326.0 | 0.7% | ok |
| decode_proj_f32_host-sgemm_in1024_out4096_len512 | #1992 #2003 #1995 #2229 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=4096, len=512 | 1 | 1 | 44307115.0 | 259774.0 | 0.6% | ok |
| decode_proj_f32_pytorch_in1024_out4096_len512 | #1992 #2003 #1995 #2229 | pytorch | pytorch-cpu | f32 | in=1024, out=4096, len=512 | 1 | 1 | 44842667.0 | 1276518.0 | 2.2% | ok |
| decode_proj_view_operand_f32_prepared_in1024_out1024_len8_t1 | #2216 | prepared-view | tenferro-rs | f32 | in=1024, out=1024, len=8 | 1 | 1 | 2839369.0 | 21977.5 | 1.5% | ok |
| decode_proj_contiguous_operand_f32_prepared_in1024_out1024_len8_t1 | #2216 | prepared-contiguous | tenferro-rs | f32 | in=1024, out=1024, len=8 | 1 | 4 | 857861.0 | 1726.4 | 0.5% | ok |
| decode_proj_view_operand_f32_pytorch_in1024_out1024_len8_t1 | #2216 | pytorch | pytorch-cpu | f32 | in=1024, out=1024, len=8 | 1 | 8 | 372883.9 | 933.9 | 0.4% | ok |
| gemm_c64_mm256_t1 | #1900 | dot-general-shared | tenferro-rs | c64 | n=256 | 1 | 1 | 2633053.0 | 10350.5 | 0.3% | ok |
| gemm_f64_mm256_t1 | #1900 | dot-general-shared | tenferro-rs | f64 | n=256 | 1 | 4 | 695288.0 | 8910.9 | 0.8% | ok |
| conj_dot_f64_dot-general-rank1_len100 | #1615 | dot-general-rank1 | tenferro-rs | f64 | len=100 | 1 | 2048 | 1182.4 | 10.1 | 3.4% | ok |
| conj_dot_f64_portable-loop_len100 | #1615 | portable-loop | rust-loop | f64 | len=100 | 1 | 32768 | 58.7 | 0.4 | 1.0% | ok |
| conj_dot_f64_pytorch-vdot_len100 | #1615 | pytorch-vdot | pytorch-cpu | f64 | len=100 | 1 | 2048 | 1094.4 | 8.0 | 0.7% | ok |
| conj_dot_f64_dot-general-rank1_len10000 | #1615 | dot-general-rank1 | tenferro-rs | f64 | len=10000 | 1 | 512 | 5329.4 | 18.3 | 0.8% | ok |
| conj_dot_f64_portable-loop_len10000 | #1615 | portable-loop | rust-loop | f64 | len=10000 | 1 | 256 | 8189.3 | 37.4 | 0.4% | ok |
| conj_dot_f64_pytorch-vdot_len10000 | #1615 | pytorch-vdot | pytorch-cpu | f64 | len=10000 | 1 | 1024 | 2559.2 | 17.0 | 0.6% | ok |
| conj_dot_f64_dot-general-rank1_len1000000 | #1615 | dot-general-rank1 | tenferro-rs | f64 | len=1000000 | 1 | 8 | 417483.5 | 1750.7 | 0.4% | ok |
| conj_dot_f64_portable-loop_len1000000 | #1615 | portable-loop | rust-loop | f64 | len=1000000 | 1 | 4 | 823440.0 | 2160.8 | 4.7% | ok |
| conj_dot_f64_pytorch-vdot_len1000000 | #1615 | pytorch-vdot | pytorch-cpu | f64 | len=1000000 | 1 | 16 | 170517.1 | 4372.4 | 1.4% | ok |
| conj_dot_c64_dot-general-rank1_len100 | #1615 | dot-general-rank1 | tenferro-rs | c64 | len=100 | 1 | 2048 | 1216.0 | 12.4 | 4.5% | ok |
| conj_dot_c64_portable-loop_len100 | #1615 | portable-loop | rust-loop | c64 | len=100 | 1 | 32768 | 86.4 | 0.8 | 1.4% | ok |
| conj_dot_c64_pytorch-vdot_len100 | #1615 | pytorch-vdot | pytorch-cpu | c64 | len=100 | 1 | 2048 | 979.1 | 13.7 | 276.3% | NOISY |
| conj_dot_c64_dot-general-rank1_len10000 | #1615 | dot-general-rank1 | tenferro-rs | c64 | len=10000 | 1 | 256 | 9423.0 | 57.1 | 0.5% | ok |
| conj_dot_c64_portable-loop_len10000 | #1615 | portable-loop | rust-loop | c64 | len=10000 | 1 | 256 | 8320.7 | 65.3 | 1.1% | ok |
| conj_dot_c64_pytorch-vdot_len10000 | #1615 | pytorch-vdot | pytorch-cpu | c64 | len=10000 | 1 | 512 | 5685.3 | 24.5 | 0.5% | ok |
| conj_dot_c64_dot-general-rank1_len1000000 | #1615 | dot-general-rank1 | tenferro-rs | c64 | len=1000000 | 1 | 4 | 889938.2 | 7513.6 | 1.1% | ok |
| conj_dot_c64_portable-loop_len1000000 | #1615 | portable-loop | rust-loop | c64 | len=1000000 | 1 | 4 | 894224.8 | 2936.8 | 0.6% | ok |
| conj_dot_c64_pytorch-vdot_len1000000 | #1615 | pytorch-vdot | pytorch-cpu | c64 | len=1000000 | 1 | 4 | 674988.2 | 42258.8 | 3.3% | ok |
| small_solve_f64_solve-loop_k16_b64 | #2007 #1878 | solve-loop | tenferro-rs | f64 | k=16, batch=64 | 1 | 8 | 222940.9 | 1837.7 | 1.0% | ok |
| small_solve_f64_triangular-solve-loop_k16_b64 | #2007 #1878 | triangular-solve-loop | tenferro-rs | f64 | k=16, batch=64 | 1 | 16 | 172783.0 | 3416.0 | 1.8% | ok |
| small_solve_f64_solve-batched_k16_b64 | #2007 #1878 | solve-batched | tenferro-rs | f64 | k=16, batch=64 | 1 | 16 | 196356.6 | 1759.5 | 0.8% | ok |
| small_solve_f64_pytorch-solve-batched_k16_b64 | #2007 #1878 | pytorch-solve-batched | pytorch-cpu | f64 | k=16, batch=64 | 1 | 16 | 147052.4 | 1248.1 | 1.0% | ok |
| small_solve_f64_pytorch-solve-triangular-batched_k16_b64 | #2007 #1878 | pytorch-solve-triangular-batched | pytorch-cpu | f64 | k=16, batch=64 | 1 | 64 | 57066.7 | 781.1 | 1.7% | ok |
| small_solve_f64_solve-loop_k64_b64 | #2007 #1878 | solve-loop | tenferro-rs | f64 | k=64, batch=64 | 1 | 1 | 3901890.0 | 25808.0 | 1.2% | ok |
| small_solve_f64_triangular-solve-loop_k64_b64 | #2007 #1878 | triangular-solve-loop | tenferro-rs | f64 | k=64, batch=64 | 1 | 1 | 1239022.0 | 10300.0 | 14.5% | NOISY |
| small_solve_f64_solve-batched_k64_b64 | #2007 #1878 | solve-batched | tenferro-rs | f64 | k=64, batch=64 | 1 | 1 | 3835541.0 | 46538.5 | 1.0% | ok |
| small_solve_f64_pytorch-solve-batched_k64_b64 | #2007 #1878 | pytorch-solve-batched | pytorch-cpu | f64 | k=64, batch=64 | 1 | 1 | 3960866.0 | 26794.0 | 0.5% | ok |
| small_solve_f64_pytorch-solve-triangular-batched_k64_b64 | #2007 #1878 | pytorch-solve-triangular-batched | pytorch-cpu | f64 | k=64, batch=64 | 1 | 2 | 1272041.0 | 12150.5 | 1.8% | ok |
| tanh_chain_f32_compiled-prepared_k10_1024x64 | #1990 | compiled-prepared | tenferro-rs | f32 | rows=1024, cols=64, chain=10 | 1 | 2 | 952478.5 | 4514.0 | 2.1% | ok |
| tanh_chain_f32_eager-shared_k10_1024x64 | #1990 | eager-shared | tenferro-rs | f32 | rows=1024, cols=64, chain=10 | 1 | 2 | 950270.0 | 2726.5 | 0.3% | ok |
| tanh_chain_f32_compiled-prepared_k100_1024x64 | #1990 | compiled-prepared | tenferro-rs | f32 | rows=1024, cols=64, chain=100 | 1 | 1 | 9418927.0 | 21116.0 | 0.2% | ok |
| tanh_chain_f32_eager-shared_k100_1024x64 | #1990 | eager-shared | tenferro-rs | f32 | rows=1024, cols=64, chain=100 | 1 | 1 | 9481049.0 | 58998.0 | 0.3% | ok |
| tanh_chain_f32_compiled-prepared_k200_1024x64 | #1990 | compiled-prepared | tenferro-rs | f32 | rows=1024, cols=64, chain=200 | 1 | 1 | 18870313.0 | 34466.0 | 0.2% | ok |
| tanh_chain_f32_eager-shared_k200_1024x64 | #1990 | eager-shared | tenferro-rs | f32 | rows=1024, cols=64, chain=200 | 1 | 1 | 18989859.0 | 38206.0 | 0.2% | ok |
| layer_norm_f32_eager-composed_d1024_len8_b1 | #2006 | eager-composed | tenferro-rs | f32 | norm=layer_norm, d=1024, len=8, batch=1 | 1 | 64 | 44896.2 | 619.7 | 1.2% | ok |
| layer_norm_f32_pytorch-fused_d1024_len8_b1 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=layer_norm, d=1024, len=8, batch=1 | 1 | 512 | 6393.4 | 225.9 | 3.8% | ok |
| layer_norm_f32_eager-single-call_d1024_len8_b1 | #2006 | eager-single-call | tenferro-rs | f32 | norm=layer_norm, d=1024, len=8, batch=1 | 1 | 64 | 47565.8 | 176.0 | 0.9% | ok |
| layer_norm_f32_eager-composed_d1024_len8_b8 | #2006 | eager-composed | tenferro-rs | f32 | norm=layer_norm, d=1024, len=8, batch=8 | 1 | 16 | 82548.3 | 1721.3 | 1.5% | ok |
| layer_norm_f32_pytorch-fused_d1024_len8_b8 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=layer_norm, d=1024, len=8, batch=8 | 1 | 128 | 30413.6 | 1691.6 | 2.8% | ok |
| layer_norm_f32_eager-single-call_d1024_len8_b8 | #2006 | eager-single-call | tenferro-rs | f32 | norm=layer_norm, d=1024, len=8, batch=8 | 1 | 16 | 82820.6 | 2363.8 | 3.1% | ok |
| layer_norm_f32_eager-composed_d1024_len64_b1 | #2006 | eager-composed | tenferro-rs | f32 | norm=layer_norm, d=1024, len=64, batch=1 | 1 | 16 | 82878.1 | 760.5 | 1.2% | ok |
| layer_norm_f32_pytorch-fused_d1024_len64_b1 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=layer_norm, d=1024, len=64, batch=1 | 1 | 128 | 31195.9 | 880.1 | 2.9% | ok |
| layer_norm_f32_eager-single-call_d1024_len64_b1 | #2006 | eager-single-call | tenferro-rs | f32 | norm=layer_norm, d=1024, len=64, batch=1 | 1 | 16 | 83788.9 | 1814.2 | 2.5% | ok |
| layer_norm_f32_eager-composed_d1024_len64_b8 | #2006 | eager-composed | tenferro-rs | f32 | norm=layer_norm, d=1024, len=64, batch=8 | 1 | 4 | 402865.8 | 24426.0 | 4.0% | ok |
| layer_norm_f32_pytorch-fused_d1024_len64_b8 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=layer_norm, d=1024, len=64, batch=8 | 1 | 16 | 968640.9 | 25169.4 | 2.3% | ok |
| layer_norm_f32_eager-single-call_d1024_len64_b8 | #2006 | eager-single-call | tenferro-rs | f32 | norm=layer_norm, d=1024, len=64, batch=8 | 1 | 4 | 378439.8 | 9414.0 | 1.8% | ok |
| rms_norm_f32_eager-composed_d1024_len8_b1 | #2006 | eager-composed | tenferro-rs | f32 | norm=rms_norm, d=1024, len=8, batch=1 | 1 | 64 | 23368.0 | 520.8 | 1.3% | ok |
| rms_norm_f32_pytorch-fused_d1024_len8_b1 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=rms_norm, d=1024, len=8, batch=1 | 1 | 256 | 13680.1 | 715.3 | 14.7% | NOISY |
| rms_norm_f32_eager-single-call_d1024_len8_b1 | #2006 | eager-single-call | tenferro-rs | f32 | norm=rms_norm, d=1024, len=8, batch=1 | 1 | 64 | 27400.2 | 99.4 | 0.5% | ok |
| rms_norm_f32_eager-composed_d1024_len8_b8 | #2006 | eager-composed | tenferro-rs | f32 | norm=rms_norm, d=1024, len=8, batch=8 | 1 | 32 | 42005.2 | 1033.1 | 3.0% | ok |
| rms_norm_f32_pytorch-fused_d1024_len8_b8 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=rms_norm, d=1024, len=8, batch=8 | 1 | 64 | 39046.6 | 2670.3 | 5.9% | ok |
| rms_norm_f32_eager-single-call_d1024_len8_b8 | #2006 | eager-single-call | tenferro-rs | f32 | norm=rms_norm, d=1024, len=8, batch=8 | 1 | 32 | 47494.3 | 1847.6 | 3.3% | ok |
| rms_norm_f32_eager-composed_d1024_len64_b1 | #2006 | eager-composed | tenferro-rs | f32 | norm=rms_norm, d=1024, len=64, batch=1 | 1 | 32 | 42042.8 | 629.7 | 2.6% | ok |
| rms_norm_f32_pytorch-fused_d1024_len64_b1 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=rms_norm, d=1024, len=64, batch=1 | 1 | 64 | 38890.3 | 1797.9 | 2.8% | ok |
| rms_norm_f32_eager-single-call_d1024_len64_b1 | #2006 | eager-single-call | tenferro-rs | f32 | norm=rms_norm, d=1024, len=64, batch=1 | 1 | 32 | 47101.8 | 896.0 | 2.4% | ok |
| rms_norm_f32_eager-composed_d1024_len64_b8 | #2006 | eager-composed | tenferro-rs | f32 | norm=rms_norm, d=1024, len=64, batch=8 | 1 | 4 | 187432.2 | 8362.4 | 4.8% | ok |
| rms_norm_f32_pytorch-fused_d1024_len64_b8 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=rms_norm, d=1024, len=64, batch=8 | 1 | 8 | 1865831.4 | 97916.8 | 3.0% | ok |
| rms_norm_f32_eager-single-call_d1024_len64_b8 | #2006 | eager-single-call | tenferro-rs | f32 | norm=rms_norm, d=1024, len=64, batch=8 | 1 | 4 | 203731.2 | 9773.2 | 3.9% | ok |
| small_contraction_abcd-dbef-acef_f64_shared-session-default-pool_d4 | #1885 #1897 #1901 | shared-session-default-pool | tenferro-rs | f64 | extent=4 | 1 | 1024 | 2948.2 | 17.3 | 0.5% | ok |
| activation_erf_f32_eager-single-call_1024x64 | #1975 #2069 #2032 | eager-single-call | tenferro-rs | f32 | op=erf, rows=1024, cols=64 | 1 | 16 | 165229.3 | 834.4 | 0.4% | ok |
| activation_erf_f32_session-single-call_1024x64 | #1975 #2069 #2032 | session-single-call | tenferro-rs | f32 | op=erf, rows=1024, cols=64 | 1 | 16 | 164362.4 | 338.9 | 0.3% | ok |
| activation_erf_f32_pytorch_1024x64 | #1975 #2069 #2032 | pytorch | pytorch-cpu | f32 | op=erf, rows=1024, cols=64 | 1 | 16 | 123842.5 | 2699.3 | 2.7% | ok |
| activation_sigmoid_f32_eager-single-call_1024x64 | #1975 #2069 #2032 | eager-single-call | tenferro-rs | f32 | op=sigmoid, rows=1024, cols=64 | 1 | 32 | 26203.4 | 237.1 | 0.9% | ok |
| activation_sigmoid_f32_session-single-call_1024x64 | #1975 #2069 #2032 | session-single-call | tenferro-rs | f32 | op=sigmoid, rows=1024, cols=64 | 1 | 32 | 25812.6 | 198.1 | 0.7% | ok |
| activation_sigmoid_f32_eager-composed_1024x64 | #1975 #2069 #2032 | eager-composed | tenferro-rs | f32 | op=sigmoid, rows=1024, cols=64 | 1 | 32 | 49148.9 | 971.9 | 2.0% | ok |
| activation_sigmoid_f32_pytorch_1024x64 | #1975 #2069 #2032 | pytorch | pytorch-cpu | f32 | op=sigmoid, rows=1024, cols=64 | 1 | 32 | 88866.4 | 2513.4 | 3.9% | ok |
| activation_silu_f32_eager-single-call_1024x64 | #1975 #2069 #2032 | eager-single-call | tenferro-rs | f32 | op=silu, rows=1024, cols=64 | 1 | 32 | 27262.8 | 94.4 | 1.1% | ok |
| activation_silu_f32_session-single-call_1024x64 | #1975 #2069 #2032 | session-single-call | tenferro-rs | f32 | op=silu, rows=1024, cols=64 | 1 | 32 | 26662.9 | 179.0 | 0.5% | ok |
| activation_silu_f32_eager-composed_1024x64 | #1975 #2069 #2032 | eager-composed | tenferro-rs | f32 | op=silu, rows=1024, cols=64 | 1 | 32 | 59659.8 | 276.3 | 0.6% | ok |
| activation_silu_f32_pytorch_1024x64 | #1975 #2069 #2032 | pytorch | pytorch-cpu | f32 | op=silu, rows=1024, cols=64 | 1 | 32 | 90151.1 | 5327.9 | 5.7% | ok |
| activation_softplus_f32_eager-single-call_1024x64 | #1975 #2069 #2032 | eager-single-call | tenferro-rs | f32 | op=softplus, rows=1024, cols=64 | 1 | 16 | 91985.4 | 667.2 | 0.5% | ok |
| activation_softplus_f32_session-single-call_1024x64 | #1975 #2069 #2032 | session-single-call | tenferro-rs | f32 | op=softplus, rows=1024, cols=64 | 1 | 16 | 91499.7 | 395.9 | 0.3% | ok |
| activation_softplus_f32_eager-composed_1024x64 | #1975 #2069 #2032 | eager-composed | tenferro-rs | f32 | op=softplus, rows=1024, cols=64 | 1 | 16 | 105199.3 | 694.4 | 0.7% | ok |
| activation_softplus_f32_pytorch_1024x64 | #1975 #2069 #2032 | pytorch | pytorch-cpu | f32 | op=softplus, rows=1024, cols=64 | 1 | 16 | 139541.9 | 1386.4 | 2.8% | ok |
| activation_gelu_f32_eager-single-call_1024x64 | #1975 #2030 #2069 #2228 | eager-single-call | tenferro-rs | f32 | op=gelu, rows=1024, cols=64 | 1 | 32 | 37986.2 | 679.6 | 1.4% | ok |
| activation_gelu_f32_session-single-call_1024x64 | #1975 #2030 #2069 #2228 | session-single-call | tenferro-rs | f32 | op=gelu, rows=1024, cols=64 | 1 | 32 | 37580.2 | 350.3 | 0.6% | ok |
| activation_gelu_f32_pytorch_1024x64 | #1975 #2030 #2069 #2228 | pytorch | pytorch-cpu | f32 | op=gelu, rows=1024, cols=64 | 1 | 32 | 36474.6 | 227.5 | 0.8% | ok |
| activation_gelu_tanh_f32_eager-single-call_1024x64 | #1975 #2069 #2032 | eager-single-call | tenferro-rs | f32 | op=gelu_tanh, rows=1024, cols=64 | 1 | 16 | 106230.9 | 1044.0 | 0.9% | ok |
| activation_gelu_tanh_f32_session-single-call_1024x64 | #1975 #2069 #2032 | session-single-call | tenferro-rs | f32 | op=gelu_tanh, rows=1024, cols=64 | 1 | 16 | 105341.4 | 586.8 | 1.5% | ok |
| activation_gelu_tanh_f32_eager-composed_1024x64 | #1975 #2069 #2032 | eager-composed | tenferro-rs | f32 | op=gelu_tanh, rows=1024, cols=64 | 1 | 16 | 193634.4 | 1537.9 | 0.6% | ok |
| activation_gelu_tanh_f32_pytorch_1024x64 | #1975 #2069 #2032 | pytorch | pytorch-cpu | f32 | op=gelu_tanh, rows=1024, cols=64 | 1 | 16 | 152529.9 | 2277.4 | 1.7% | ok |
| softmax_f32_eager-single-call_len64_b8 | #1976 #2031 #2231 | eager-single-call | tenferro-rs | f32 | op=softmax, len=64, batch=8, axis=0 | 1 | 64 | 19415.3 | 289.8 | 1.6% | ok |
| softmax_f32_session-single-call_len64_b8 | #1976 #2031 #2231 | session-single-call | tenferro-rs | f32 | op=softmax, len=64, batch=8, axis=0 | 1 | 64 | 18931.8 | 169.4 | 0.8% | ok |
| softmax_f32_eager-composed_len64_b8 | #1976 #2031 #2231 | eager-composed | tenferro-rs | f32 | op=softmax, len=64, batch=8, axis=0 | 1 | 16 | 96612.9 | 1047.5 | 0.9% | ok |
| softmax_f32_pytorch_len64_b8 | #1976 #2031 #2231 | pytorch | pytorch-cpu | f32 | op=softmax, len=64, batch=8, axis=0 | 1 | 128 | 25771.9 | 343.5 | 1.5% | ok |
| softmax_f32_eager-single-call_len512_b8 | #1976 #2231 | eager-single-call | tenferro-rs | f32 | op=softmax, len=512, batch=8, axis=0 | 1 | 2 | 1205482.5 | 40625.2 | 4.5% | ok |
| softmax_f32_session-single-call_len512_b8 | #1976 #2231 | session-single-call | tenferro-rs | f32 | op=softmax, len=512, batch=8, axis=0 | 1 | 2 | 1247950.0 | 72660.0 | 3.3% | ok |
| softmax_f32_eager-composed_len512_b8 | #1976 #2231 | eager-composed | tenferro-rs | f32 | op=softmax, len=512, batch=8, axis=0 | 1 | 1 | 2411876.0 | 89152.5 | 2.4% | ok |
| softmax_f32_pytorch_len512_b8 | #1976 #2231 | pytorch | pytorch-cpu | f32 | op=softmax, len=512, batch=8, axis=0 | 1 | 2 | 1329751.5 | 7388.8 | 4.5% | ok |
| log_softmax_f32_eager-single-call_len64_b8 | #1976 | eager-single-call | tenferro-rs | f32 | op=log_softmax, len=64, batch=8, axis=0 | 1 | 16 | 105694.4 | 933.9 | 1.4% | ok |
| log_softmax_f32_session-single-call_len64_b8 | #1976 | session-single-call | tenferro-rs | f32 | op=log_softmax, len=64, batch=8, axis=0 | 1 | 32 | 89555.2 | 905.7 | 0.7% | ok |
| log_softmax_f32_eager-composed_len64_b8 | #1976 | eager-composed | tenferro-rs | f32 | op=log_softmax, len=64, batch=8, axis=0 | 1 | 32 | 92838.2 | 711.0 | 0.7% | ok |
| log_softmax_f32_pytorch_len64_b8 | #1976 | pytorch | pytorch-cpu | f32 | op=log_softmax, len=64, batch=8, axis=0 | 1 | 128 | 24866.9 | 403.1 | 8.3% | ok |
| log_softmax_f32_eager-single-call_len512_b8 | #1976 | eager-single-call | tenferro-rs | f32 | op=log_softmax, len=512, batch=8, axis=0 | 1 | 1 | 2182077.0 | 101445.0 | 2.5% | ok |
| log_softmax_f32_session-single-call_len512_b8 | #1976 | session-single-call | tenferro-rs | f32 | op=log_softmax, len=512, batch=8, axis=0 | 1 | 1 | 2322654.0 | 63359.0 | 2.4% | ok |
| log_softmax_f32_eager-composed_len512_b8 | #1976 | eager-composed | tenferro-rs | f32 | op=log_softmax, len=512, batch=8, axis=0 | 1 | 1 | 2437254.0 | 119011.0 | 3.4% | ok |
| log_softmax_f32_pytorch_len512_b8 | #1976 | pytorch | pytorch-cpu | f32 | op=log_softmax, len=512, batch=8, axis=0 | 1 | 2 | 1427130.0 | 22383.2 | 1.1% | ok |
| masked_softmax_f32_eager-single-call_len64_b8 | #1976 | eager-single-call | tenferro-rs | f32 | op=masked_softmax, len=64, batch=8, axis=0 | 1 | 16 | 192929.1 | 1709.8 | 0.7% | ok |
| masked_softmax_f32_session-single-call_len64_b8 | #1976 | session-single-call | tenferro-rs | f32 | op=masked_softmax, len=64, batch=8, axis=0 | 1 | 16 | 175309.8 | 1678.1 | 0.9% | ok |
| masked_softmax_f32_eager-composed_len64_b8 | #1976 | eager-composed | tenferro-rs | f32 | op=masked_softmax, len=64, batch=8, axis=0 | 1 | 16 | 105700.0 | 677.8 | 0.7% | ok |
| masked_softmax_f32_pytorch_len64_b8 | #1976 | pytorch | pytorch-cpu | f32 | op=masked_softmax, len=64, batch=8, axis=0 | 1 | 64 | 52926.4 | 619.4 | 0.7% | ok |
| masked_softmax_f32_eager-single-call_len512_b8 | #1976 | eager-single-call | tenferro-rs | f32 | op=masked_softmax, len=512, batch=8, axis=0 | 1 | 1 | 7140067.0 | 70635.0 | 0.9% | ok |
| masked_softmax_f32_session-single-call_len512_b8 | #1976 | session-single-call | tenferro-rs | f32 | op=masked_softmax, len=512, batch=8, axis=0 | 1 | 1 | 7261747.0 | 112556.0 | 1.2% | ok |
| masked_softmax_f32_eager-composed_len512_b8 | #1976 | eager-composed | tenferro-rs | f32 | op=masked_softmax, len=512, batch=8, axis=0 | 1 | 1 | 3227763.0 | 90098.0 | 2.5% | ok |
| masked_softmax_f32_pytorch_len512_b8 | #1976 | pytorch | pytorch-cpu | f32 | op=masked_softmax, len=512, batch=8, axis=0 | 1 | 1 | 3166872.0 | 35633.0 | 0.7% | ok |
| softmax_f32_eager-single-call_len64_b8_t1 | #2233 #2235 | eager-single-call | tenferro-rs | f32 | op=softmax, len=64, batch=8, axis=0 | 1 | 64 | 19277.4 | 140.8 | 1.0% | ok |
| softmax_f32_eager-composed_len64_b8_t1 | #2233 #2235 | eager-composed | tenferro-rs | f32 | op=softmax, len=64, batch=8, axis=0 | 1 | 32 | 96621.6 | 863.2 | 0.9% | ok |
| softmax_f32_pytorch_len64_b8_t1 | #2233 #2235 | pytorch | pytorch-cpu | f32 | op=softmax, len=64, batch=8, axis=0 | 1 | 128 | 26040.6 | 295.7 | 1.1% | ok |
| reduce_mean_f32_eager-single-call_1024x64_axis0 | #1976 | eager-single-call | tenferro-rs | f32 | rows=1024, cols=64, axis=0 | 1 | 256 | 10905.4 | 35.5 | 1.0% | ok |
| reduce_mean_f32_session-single-call_1024x64_axis0 | #1976 | session-single-call | tenferro-rs | f32 | rows=1024, cols=64, axis=0 | 1 | 512 | 5671.9 | 48.8 | 0.5% | ok |
| reduce_mean_f32_eager-composed_1024x64_axis0 | #1976 | eager-composed | tenferro-rs | f32 | rows=1024, cols=64, axis=0 | 1 | 256 | 10804.9 | 140.5 | 0.9% | ok |
| reduce_mean_f32_pytorch_1024x64_axis0 | #1976 | pytorch | pytorch-cpu | f32 | rows=1024, cols=64, axis=0 | 1 | 512 | 6228.6 | 14.3 | 1.2% | ok |
| take_along_axis_rows_f64_eager-single-call_n64_b256 | #2008 | eager-single-call | tenferro-rs | f64 | n=64, batch=256, axis=0 | 1 | 1 | 6519880.0 | 35487.0 | 0.8% | ok |
| take_along_axis_rows_f64_session-single-call_n64_b256 | #2008 | session-single-call | tenferro-rs | f64 | n=64, batch=256, axis=0 | 1 | 1 | 6555162.0 | 42242.0 | 0.7% | ok |
| take_along_axis_rows_f64_eager-gather-prebuilt_n64_b256 | #2008 | eager-gather-prebuilt | tenferro-rs | f64 | n=64, batch=256, axis=0 | 1 | 1 | 6329268.0 | 59267.5 | 1.1% | ok |
| take_along_axis_rows_f64_pytorch_n64_b256 | #2008 | pytorch | pytorch-cpu | f64 | n=64, batch=256, axis=0 | 1 | 1 | 2843063.0 | 26104.0 | 1.1% | ok |
| lstsq_f64_eager-shared_m768_n384_rhs16 | #2027 | eager-shared | tenferro-rs | f64 | operation=lstsq | 1 | 1 | 16161435.0 | 1448396.5 | 5.5% | ok |
| lstsq_f64_pytorch_m768_n384_rhs16 | #2027 | pytorch | pytorch-cpu | f64 | operation=lstsq | 1 | 1 | 7506128.0 | 88105.0 | 1.1% | ok |
| triangular_f64_concrete-shared_n4096_rhs64 | #2028 | concrete-shared | tenferro-rs | f64 | operation=triangular | 1 | 1 | 30845757.0 | 187111.0 | 0.8% | ok |
| triangular_f64_pytorch_n4096_rhs64 | #2028 | pytorch | pytorch-cpu | f64 | operation=triangular | 1 | 1 | 26217494.0 | 246985.0 | 0.8% | ok |
| reshape_f64_concrete-shared_n33554432_to8192x4096 | #2029 | concrete-shared | tenferro-rs | f64 | operation=reshape | 1 | 1 | 114803312.0 | 6185871.5 | 2.9% | ok |
| reshape_f64_pytorch_n33554432_to8192x4096 | #2029 | pytorch | pytorch-cpu | f64 | operation=reshape | 1 | 1 | 106042031.0 | 3148806.0 | 1.6% | ok |
| metadata_transpose_view_f64_rank8_t1 | #2180 | metadata-clone | tenferro-rs | f64 | shape=[2, 2, 2, 2, 2, 2, 2, 2] | 1 | 8192 | 58.5 | 1.8 | 81.0% | NOISY |
| metadata_transpose_view_f64_rank8_t1_julia-base | #2180 | julia-base | julia-base | f64 | shape=[2, 2, 2, 2, 2, 2, 2, 2] | 1 | 1048576 | 0.2 | 0.0 | 9.5% | ok |
| followup_structural_broadcast_in_dim_direct-shared | #2021 | direct-shared | tenferro-rs | f64 | case_id=structural.broadcast_in_dim, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [8192, 1], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 111226056.0 | 308511.0 | 0.3% | ok |
| followup_structural_broadcast_in_dim_pytorch | #2021 | pytorch | pytorch-cpu | f64 | case_id=structural.broadcast_in_dim, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [8192, 1], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 89991029.0 | 477616.5 | 0.5% | ok |
| followup_real_tanh_direct-shared | #2033 | direct-shared | tenferro-rs | f64 | case_id=real.tanh, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [8388608], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 27097496.0 | 15854.0 | 0.2% | ok |
| followup_real_tanh_jax | #2033 | jax | jax-cpu | f64 | case_id=real.tanh, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [8388608], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 30365458.0 | 145404.5 | 1.0% | ok |
| followup_real_log1p_direct-shared | #2033 | direct-shared | tenferro-rs | f64 | case_id=real.log1p, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [4194304], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 6864660.0 | 28096.0 | 0.3% | ok |
| followup_real_log1p_pytorch | #2033 | pytorch | pytorch-cpu | f64 | case_id=real.log1p, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [4194304], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 20246146.0 | 79962.5 | 0.5% | ok |
| followup_real_cos_direct-shared | #2033 | direct-shared | tenferro-rs | f64 | case_id=real.cos, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [8388608], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 13446321.0 | 21452.5 | 0.1% | ok |
| followup_real_cos_pytorch | #2033 | pytorch | pytorch-cpu | f64 | case_id=real.cos, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [8388608], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 48394670.0 | 776243.0 | 0.9% | ok |
| followup_real_sin_direct-shared | #2033 | direct-shared | tenferro-rs | f64 | case_id=real.sin, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [8388608], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 10793866.0 | 23555.0 | 0.2% | ok |
| followup_real_sin_pytorch | #2033 | pytorch | pytorch-cpu | f64 | case_id=real.sin, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [8388608], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 48317721.0 | 600617.5 | 0.6% | ok |
| followup_real_exp_direct-shared | #2033 | direct-shared | tenferro-rs | f64 | case_id=real.exp, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [8388608], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 7973399.0 | 25157.0 | 0.3% | ok |
| followup_real_exp_jax | #2033 | jax | jax-cpu | f64 | case_id=real.exp, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [8388608], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 33929177.0 | 299789.0 | 3.3% | ok |
| followup_real_expm1_direct-shared | #2033 | direct-shared | tenferro-rs | f64 | case_id=real.expm1, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [4194304], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 10739689.0 | 16895.0 | 0.7% | ok |
| followup_real_expm1_jax | #2033 | jax | jax-cpu | f64 | case_id=real.expm1, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [4194304], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 19589224.0 | 190707.5 | 0.7% | ok |
| followup_real_pow_direct-shared | #2033 | direct-shared | tenferro-rs | f64 | case_id=real.pow, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [4194304], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 43021383.0 | 139892.0 | 1.0% | ok |
| followup_real_pow_pytorch | #2033 | pytorch | pytorch-cpu | f64 | case_id=real.pow, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [4194304], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 34837684.0 | 199507.5 | 1.5% | ok |
| followup_complex_exp_direct-shared | #2033 | direct-shared | tenferro-rs | c64 | case_id=complex.exp, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [4194304], 'dtype': 'c64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 54703751.0 | 832250.5 | 7.5% | ok |
| followup_complex_exp_julia-base | #2033 | julia-base | julia-base | c64 | case_id=complex.exp, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [4194304], 'dtype': 'c64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 62110875.0 | 1159457.5 | 1.4% | ok |
| followup_real_log_direct-shared | #2033 | direct-shared | tenferro-rs | f64 | case_id=real.log, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [8388608], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 13707608.0 | 38992.5 | 2.8% | ok |
| followup_real_log_pytorch | #2033 | pytorch | pytorch-cpu | f64 | case_id=real.log, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [8388608], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 41636182.0 | 681748.0 | 2.2% | ok |
| followup_real_reduce_max_axis1_direct-shared | #2034 | direct-shared | tenferro-rs | f64 | case_id=real.reduce_max_axis1, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [2048, 2048], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 2 | 1784458.5 | 58174.5 | 3.3% | ok |
| followup_real_reduce_max_axis1_julia-base | #2034 | julia-base | julia-base | f64 | case_id=real.reduce_max_axis1, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [2048, 2048], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 2 | 1008205.5 | 29253.0 | 7.2% | ok |
| followup_real_reduce_prod_axis1_direct-shared | #2034 | direct-shared | tenferro-rs | f64 | case_id=real.reduce_prod_axis1, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [2048, 2048], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 2 | 1549355.0 | 53686.8 | 2.3% | ok |
| followup_real_reduce_prod_axis1_pytorch | #2034 | pytorch | pytorch-cpu | f64 | case_id=real.reduce_prod_axis1, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [2048, 2048], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 4 | 725762.8 | 24483.8 | 2.9% | ok |
| followup_real_reduce_sum_axis1_direct-shared | #2034 | direct-shared | tenferro-rs | f64 | case_id=real.reduce_sum_axis1, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [2048, 2048], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 2 | 1480438.0 | 85679.2 | 3.7% | ok |
| followup_real_reduce_sum_axis1_pytorch | #2034 | pytorch | pytorch-cpu | f64 | case_id=real.reduce_sum_axis1, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [2048, 2048], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 4 | 745519.8 | 18225.6 | 2.0% | ok |
| followup_real_reduce_max_axis0_direct-shared | #2034 | direct-shared | tenferro-rs | f64 | case_id=real.reduce_max_axis0, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [2048, 2048], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 2 | 1491590.0 | 12949.2 | 0.6% | ok |
| followup_real_reduce_max_axis0_julia-base | #2034 | julia-base | julia-base | f64 | case_id=real.reduce_max_axis0, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [2048, 2048], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 2 | 930320.5 | 18359.8 | 1.9% | ok |
| followup_real_reduce_min_axis1_direct-shared | #2034 | direct-shared | tenferro-rs | f64 | case_id=real.reduce_min_axis1, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [4096, 4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 8987743.0 | 109101.0 | 0.9% | ok |
| followup_real_reduce_min_axis1_julia-base | #2034 | julia-base | julia-base | f64 | case_id=real.reduce_min_axis1, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [4096, 4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 7389496.0 | 65467.0 | 1.6% | ok |
| followup_real_reduce_min_axis0_direct-shared | #2034 | direct-shared | tenferro-rs | f64 | case_id=real.reduce_min_axis0, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [2048, 2048], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 2 | 1430785.5 | 7278.5 | 0.4% | ok |
| followup_real_reduce_min_axis0_julia-base | #2034 | julia-base | julia-base | f64 | case_id=real.reduce_min_axis0, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [2048, 2048], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 4 | 937190.8 | 30916.9 | 2.2% | ok |
| followup_real_reduce_max_all_direct-shared | #2034 | direct-shared | tenferro-rs | f64 | case_id=real.reduce_max_all, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [8192, 4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 11541136.0 | 72026.5 | 0.8% | ok |
| followup_real_reduce_max_all_julia-base | #2034 | julia-base | julia-base | f64 | case_id=real.reduce_max_all, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [8192, 4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 8065367.0 | 422690.0 | 2.9% | ok |
| followup_real_reduce_min_all_direct-shared | #2034 | direct-shared | tenferro-rs | f64 | case_id=real.reduce_min_all, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [8192, 4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 11057139.0 | 15398.5 | 0.3% | ok |
| followup_real_reduce_min_all_julia-base | #2034 | julia-base | julia-base | f64 | case_id=real.reduce_min_all, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [8192, 4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 7787705.0 | 276599.5 | 2.6% | ok |
| followup_complex_norm_fro_direct-shared | #2035 | direct-shared | tenferro-rs | c64 | case_id=complex.norm_fro, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [2048, 1536], 'dtype': 'c64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 2 | 1196644.5 | 36917.0 | 1.8% | ok |
| followup_complex_norm_fro_jax | #2035 | jax | jax-cpu | c64 | case_id=complex.norm_fro, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [2048, 1536], 'dtype': 'c64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 4479674.0 | 17216.0 | 0.5% | ok |
| followup_index_gather_direct-shared | #2036 #1719 | direct-shared | tenferro-rs | f64 | case_id=index.gather, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [262144], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 4 | 274839.0 | 2035.6 | 0.7% | ok |
| followup_index_gather_jax | #2036 #1719 | jax | jax-cpu | f64 | case_id=index.gather, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [262144], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 4 | 826452.0 | 13597.5 | 3.6% | ok |
| followup_index_scatter_direct-shared | #2036 #1719 | direct-shared | tenferro-rs | f64 | case_id=index.scatter, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [262144], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 4 | 632194.8 | 9252.4 | 1.0% | ok |
| followup_index_scatter_pytorch | #2036 #1719 | pytorch | pytorch-cpu | f64 | case_id=index.scatter, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [262144], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 4 | 990477.2 | 39107.9 | 5.2% | ok |
| followup_index_dynamic_update_slice_direct-shared | #2036 #1719 | direct-shared | tenferro-rs | f64 | case_id=index.dynamic_update_slice, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [2097152], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 2 | 1090077.0 | 10976.2 | 2.0% | ok |
| followup_index_dynamic_update_slice_jax | #2036 #1719 | jax | jax-cpu | f64 | case_id=index.dynamic_update_slice, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [2097152], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 2 | 7097389.5 | 496626.5 | 4.1% | ok |
| followup_perm_reverse_15d_3_materialize | #2037 #2228 | materialize | tenferro-rs | f64 | case_id=perm.reverse_15d_3, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 102543264.0 | 2624681.0 | 7.5% | ok |
| followup_perm_reverse_15d_3_strided-rs | #2037 #2228 | strided-rs | strided-rs | f64 | case_id=perm.reverse_15d_3, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 92483599.0 | 3102748.0 | 3.2% | ok |
| followup_structural_transpose_direct-shared | #2037 #2228 | direct-shared | tenferro-rs | f64 | case_id=structural.transpose, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [4096, 4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 79139220.0 | 2367606.0 | 8.9% | ok |
| followup_structural_transpose_pytorch | #2037 #2228 | pytorch | pytorch-cpu | f64 | case_id=structural.transpose, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [4096, 4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 70247537.0 | 535040.5 | 0.6% | ok |
| followup_perm_reverse_23d_2_materialize | #2037 #2228 | materialize | tenferro-rs | f64 | case_id=perm.reverse_23d_2, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 26903480.0 | 190135.5 | 4.3% | ok |
| followup_perm_reverse_23d_2_strided-rs | #2037 #2228 | strided-rs | strided-rs | f64 | case_id=perm.reverse_23d_2, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 50019688.0 | 186290.5 | 0.4% | ok |
| followup_perm_transpose_3d_256_201_materialize | #2037 #2228 | materialize | tenferro-rs | f64 | case_id=perm.transpose_3d_256_201, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [256, 256, 256], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 78708007.0 | 2592277.5 | 2.7% | ok |
| followup_perm_transpose_3d_256_201_strided-rs | #2037 #2228 | strided-rs | strided-rs | f64 | case_id=perm.transpose_3d_256_201, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [256, 256, 256], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 79680169.0 | 3195946.0 | 2.9% | ok |
| followup_perm_cyclic_15d_3_materialize | #2037 #2228 | materialize | tenferro-rs | f64 | case_id=perm.cyclic_15d_3, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 54640471.0 | 132881.5 | 0.2% | ok |
| followup_perm_cyclic_15d_3_strided-rs | #2037 #2228 | strided-rs | strided-rs | f64 | case_id=perm.cyclic_15d_3, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 55377691.0 | 251366.5 | 0.4% | ok |
| followup_perm_transpose_3d_256_102_materialize | #2037 #2228 | materialize | tenferro-rs | f64 | case_id=perm.transpose_3d_256_102, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [256, 256, 256], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 73983287.0 | 907141.5 | 0.8% | ok |
| followup_perm_transpose_3d_256_102_strided-rs | #2037 #2228 | strided-rs | strided-rs | f64 | case_id=perm.transpose_3d_256_102, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [256, 256, 256], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 73403198.0 | 2182842.5 | 7.2% | ok |
| followup_perm_tn_light_415_24d_scattered_to_colmajor_materialize | #2037 #2228 | materialize | tenferro-rs | f64 | case_id=perm.tn_light_415_24d_scattered_to_colmajor, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 67840333.0 | 311476.0 | 1.0% | ok |
| followup_perm_tn_light_415_24d_scattered_to_colmajor_strided-rs | #2037 #2228 | strided-rs | strided-rs | f64 | case_id=perm.tn_light_415_24d_scattered_to_colmajor, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 67316829.0 | 1573050.5 | 6.9% | ok |
| followup_perm_tn_light_415_24d_contiguous_same_perm_materialize | #2037 #2228 | materialize | tenferro-rs | f64 | case_id=perm.tn_light_415_24d_contiguous_same_perm, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 63732894.0 | 1161703.5 | 1.3% | ok |
| followup_perm_tn_light_415_24d_contiguous_same_perm_strided-rs | #2037 #2228 | strided-rs | strided-rs | f64 | case_id=perm.tn_light_415_24d_contiguous_same_perm, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 63478106.0 | 2867470.5 | 2.4% | ok |
| followup_perm_rotation_6d_32_32_32_32_16_16_materialize | #2037 #2228 | materialize | tenferro-rs | f64 | case_id=perm.rotation_6d_32_32_32_32_16_16, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [32, 32, 32, 32, 16, 16], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 1272346578.0 | 6941729.0 | 1.4% | ok |
| followup_perm_rotation_6d_32_32_32_32_16_16_strided-rs | #2037 #2228 | strided-rs | strided-rs | f64 | case_id=perm.rotation_6d_32_32_32_32_16_16, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [32, 32, 32, 32, 16, 16], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 1285110748.0 | 17323121.5 | 1.8% | ok |
| followup_structural_extract_diagonal_direct-shared | #2038 #2228 | direct-shared | tenferro-rs | f64 | case_id=structural.extract_diagonal, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [8388608, 2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 53732071.0 | 3731637.5 | 3.7% | ok |
| followup_structural_extract_diagonal_jax | #2038 #2228 | jax | jax-cpu | f64 | case_id=structural.extract_diagonal, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [8388608, 2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 62929831.0 | 1074954.0 | 3.0% | ok |
| followup_fft_irfft_direct-shared | #2039 #2230 #2228 | direct-shared | tenferro-rs | c32 | case_id=fft.irfft, rust_path=direct-shared, reference_path=mkl-dfti, fixture={'input_shape': [524289], 'dtype': 'c32', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 3046194.0 | 47490.5 | 1.3% | ok |
| followup_fft_irfft_mkl-dfti | #2039 #2230 #2228 | mkl-dfti | mkl-dfti | c32 | case_id=fft.irfft, rust_path=direct-shared, reference_path=mkl-dfti, fixture={'input_shape': [524289], 'dtype': 'c32', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 2035169.0 | 50629.5 | 5.0% | ok |
| followup_fft_rfft_direct-shared | #2039 #2230 #2228 | direct-shared | tenferro-rs | f32 | case_id=fft.rfft, rust_path=direct-shared, reference_path=mkl-dfti, fixture={'input_shape': [1048576], 'dtype': 'f32', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 2895011.0 | 33495.0 | 1.0% | ok |
| followup_fft_rfft_mkl-dfti | #2039 #2230 #2228 | mkl-dfti | mkl-dfti | f32 | case_id=fft.rfft, rust_path=direct-shared, reference_path=mkl-dfti, fixture={'input_shape': [1048576], 'dtype': 'f32', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 2 | 1988115.0 | 32398.0 | 5.3% | ok |
| followup_fft_fft_direct-shared | #2039 #2228 | direct-shared | tenferro-rs | c64 | case_id=fft.fft, rust_path=direct-shared, reference_path=mkl-dfti, fixture={'input_shape': [1048576], 'dtype': 'c64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 11657122.0 | 620662.5 | 4.8% | ok |
| followup_fft_fft_mkl-dfti | #2039 #2228 | mkl-dfti | mkl-dfti | c64 | case_id=fft.fft, rust_path=direct-shared, reference_path=mkl-dfti, fixture={'input_shape': [1048576], 'dtype': 'c64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 11619171.0 | 454302.0 | 3.0% | ok |
| followup_fft_ifft_direct-shared | #2039 #2228 | direct-shared | tenferro-rs | c32 | case_id=fft.ifft, rust_path=direct-shared, reference_path=mkl-dfti, fixture={'input_shape': [1048576], 'dtype': 'c32', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 5837071.0 | 157037.0 | 4.0% | ok |
| followup_fft_ifft_mkl-dfti | #2039 #2228 | mkl-dfti | mkl-dfti | c32 | case_id=fft.ifft, rust_path=direct-shared, reference_path=mkl-dfti, fixture={'input_shape': [1048576], 'dtype': 'c32', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 1 | 1 | 5002968.0 | 326843.0 | 4.5% | ok |
| followup_metadata_transpose_view_metadata | #2040 #1719 | metadata | tenferro-rs | f64 | case_id=metadata.transpose_view, rust_path=metadata, reference_path=julia-base, fixture={'input_shape': [2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 1 | 8192 | 290.4 | 6.7 | 1.8% | ok |
| followup_metadata_transpose_view_julia-base | #2040 #1719 | julia-base | julia-base | f64 | case_id=metadata.transpose_view, rust_path=metadata, reference_path=julia-base, fixture={'input_shape': [2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 1 | 1864135 | 0.6 | 0.0 | 1.3% | ok |
| followup_metadata_transpose_view_metadata-host-dyn | #2040 #1719 | metadata-host-dyn | tenferro-rs | f64 | case_id=metadata.transpose_view, rust_path=metadata-host-dyn, reference_path=julia-base, fixture={'input_shape': [2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 1 | 8192 | 368.9 | 6.7 | 1.8% | ok |
| followup_metadata_transpose_view_metadata-host-static | #2040 #1719 | metadata-host-static | tenferro-rs | f64 | case_id=metadata.transpose_view, rust_path=metadata-host-static, reference_path=julia-base, fixture={'input_shape': [2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 1 | 8192 | 307.8 | 2.3 | 1.3% | ok |
| followup_metadata_slice_view_metadata | #2040 #1719 | metadata | tenferro-rs | f64 | case_id=metadata.slice_view, rust_path=metadata, reference_path=julia-base, fixture={'input_shape': [4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 1 | 8192 | 345.4 | 1.2 | 0.6% | ok |
| followup_metadata_slice_view_julia-base | #2040 #1719 | julia-base | julia-base | f64 | case_id=metadata.slice_view, rust_path=metadata, reference_path=julia-base, fixture={'input_shape': [4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 1 | 16256 | 2.3 | 1.2 | 41.9% | NOISY |
| followup_metadata_slice_view_metadata-host-dyn | #2040 #1719 | metadata-host-dyn | tenferro-rs | f64 | case_id=metadata.slice_view, rust_path=metadata-host-dyn, reference_path=julia-base, fixture={'input_shape': [4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 1 | 8192 | 459.7 | 5.9 | 1.4% | ok |
| followup_metadata_slice_view_metadata-host-static | #2040 #1719 | metadata-host-static | tenferro-rs | f64 | case_id=metadata.slice_view, rust_path=metadata-host-static, reference_path=julia-base, fixture={'input_shape': [4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 1 | 8192 | 338.2 | 12.1 | 43.1% | NOISY |
| followup_metadata_reshape_view_metadata | #2040 #1719 | metadata | tenferro-rs | f64 | case_id=metadata.reshape_view, rust_path=metadata, reference_path=julia-base, fixture={'input_shape': [1024], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 1 | 8192 | 445.8 | 1.9 | 0.3% | ok |
| followup_metadata_reshape_view_julia-base | #2040 #1719 | julia-base | julia-base | f64 | case_id=metadata.reshape_view, rust_path=metadata, reference_path=julia-base, fixture={'input_shape': [1024], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 1 | 63550 | 9.7 | 1.7 | 16.8% | NOISY |
| followup_metadata_reshape_view_metadata-host-dyn | #2040 #1719 | metadata-host-dyn | tenferro-rs | f64 | case_id=metadata.reshape_view, rust_path=metadata-host-dyn, reference_path=julia-base, fixture={'input_shape': [1024], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 1 | 8192 | 514.2 | 9.3 | 1.4% | ok |
| followup_metadata_reshape_view_metadata-host-static | #2040 #1719 | metadata-host-static | tenferro-rs | f64 | case_id=metadata.reshape_view, rust_path=metadata-host-static, reference_path=julia-base, fixture={'input_shape': [1024], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 1 | 8192 | 468.1 | 9.7 | 35.1% | NOISY |
| typed_elemwise_f64_add_held_len16 | #2139 | tenferro | tenferro-rs | f64 | n=16, route=held | 1 | 2048 | 1019.8 | 6.3 | 3.0% | ok |
| typed_elemwise_f64_add_held_len16_pytorch | #2139 | pytorch | pytorch-cpu | f64 | n=16, route=held | 1 | 2048 | 1228.3 | 5.4 | 4.2% | ok |
| typed_elemwise_f64_add_into_held_len16 | #2139 | tenferro | tenferro-rs | f64 | n=16, route=into_held | 1 | 4096 | 771.6 | 4.2 | 0.4% | ok |
| typed_elemwise_f64_add_into_held_len16_pytorch | #2139 | pytorch | pytorch-cpu | f64 | n=16, route=into_held | 1 | 2048 | 1061.7 | 10.0 | 1.0% | ok |
| typed_elemwise_f64_add_held_len256 | #2139 | tenferro | tenferro-rs | f64 | n=256, route=held | 1 | 2048 | 1078.3 | 12.8 | 3.6% | ok |
| typed_elemwise_f64_add_held_len256_pytorch | #2139 | pytorch | pytorch-cpu | f64 | n=256, route=held | 1 | 2048 | 1313.4 | 19.7 | 1.6% | ok |
| typed_elemwise_f64_add_into_held_len256 | #2139 | tenferro | tenferro-rs | f64 | n=256, route=into_held | 1 | 4096 | 796.3 | 2.8 | 0.3% | ok |
| typed_elemwise_f64_add_into_held_len256_pytorch | #2139 | pytorch | pytorch-cpu | f64 | n=256, route=into_held | 1 | 2048 | 1100.5 | 7.6 | 0.7% | ok |
| typed_elemwise_f64_add_held_len4096 | #2139 | tenferro | tenferro-rs | f64 | n=4096, route=held | 1 | 256 | 2144.7 | 32.1 | 3.2% | ok |
| typed_elemwise_f64_add_held_len4096_pytorch | #2139 | pytorch | pytorch-cpu | f64 | n=4096, route=held | 1 | 1024 | 3535.8 | 111.7 | 2.9% | ok |
| typed_elemwise_f64_add_into_held_len4096 | #2139 | tenferro | tenferro-rs | f64 | n=4096, route=into_held | 1 | 2048 | 1716.5 | 5.2 | 0.9% | ok |
| typed_elemwise_f64_add_into_held_len4096_pytorch | #2139 | pytorch | pytorch-cpu | f64 | n=4096, route=into_held | 1 | 1024 | 1955.3 | 16.0 | 0.5% | ok |
| tiny_matmul_2x2_typed_held | #2139 | tenferro | tenferro-rs | f64 | n=2 | 1 | 2048 | 1352.1 | 4.3 | 0.3% | ok |
| tiny_matmul_2x2_typed_held_pytorch | #2139 | pytorch | pytorch-cpu | f64 | n=2 | 1 | 2048 | 1265.5 | 5.5 | 0.7% | ok |
| analytic_neg_f64_sweep_8MiB_t1 | #2139 | tenferro | tenferro-rs | f64 | n=1048576 | 1 | 2 | 243180.0 | 9253.8 | 4.5% | ok |
| analytic_neg_f64_sweep_8MiB_t1_pytorch | #2139 | pytorch | pytorch-cpu | f64 | n=1048576 | 1 | 8 | 1878263.8 | 39705.5 | 2.0% | ok |
| analytic_neg_f64_sweep_32MiB_t1 | #2139 | tenferro | tenferro-rs | f64 | n=4194304 | 1 | 2 | 1777033.0 | 94247.0 | 4.0% | ok |
| analytic_neg_f64_sweep_32MiB_t1_pytorch | #2139 | pytorch | pytorch-cpu | f64 | n=4194304 | 1 | 1 | 12777314.0 | 221191.5 | 1.0% | ok |
| gemm_bound_f64_t1_t8_scaling | #2141 | tenferro | tenferro-rs | f64 | n=1536 | 1 | 1 | 137833397.0 | 1877637.0 | 1.1% | ok |
| gemm_bound_f64_t1_t8_scaling_pytorch | #2141 | pytorch | pytorch-cpu | f64 | n=1536 | 1 | 1 | 138444476.0 | 777689.5 | 5.0% | ok |
| lu_batched_f64_factor_n8_b1024 | #2142 | tenferro | tenferro-rs | f64 | n=8, batch=1024, op=factor | 1 | 8 | 271597.9 | 3717.9 | 1.0% | ok |
| lu_batched_f64_factor_n8_b1024_host | #2142 | host | rust-loop | f64 | n=8, batch=1024, op=factor | 1 | 8 | 458173.8 | 3560.3 | 0.7% | ok |
| lu_batched_f64_solve_n8_b1024 | #2142 | tenferro | tenferro-rs | f64 | n=8, batch=1024, op=solve | 1 | 16 | 164269.8 | 1155.2 | 0.9% | ok |
| lu_batched_f64_solve_n8_b1024_pytorch | #2142 | pytorch | pytorch-cpu | f64 | n=8, batch=1024, op=solve | 1 | 16 | 139251.6 | 972.7 | 0.7% | ok |
| lu_batched_f64_factor_n16_b1024 | #2142 | tenferro | tenferro-rs | f64 | n=16, batch=1024, op=factor | 1 | 2 | 815796.0 | 4459.2 | 0.5% | ok |
| lu_batched_f64_factor_n16_b1024_host | #2142 | host | rust-loop | f64 | n=16, batch=1024, op=factor | 1 | 2 | 1830131.0 | 12150.2 | 1.6% | ok |
| lu_batched_f64_solve_n16_b1024 | #2142 | tenferro | tenferro-rs | f64 | n=16, batch=1024, op=solve | 1 | 8 | 247246.9 | 487.0 | 0.6% | ok |
| lu_batched_f64_solve_n16_b1024_pytorch | #2142 | pytorch | pytorch-cpu | f64 | n=16, batch=1024, op=solve | 1 | 8 | 299474.1 | 1144.9 | 0.5% | ok |
| axpby_accum_strided_dest_f64_block | #2143 | tenferro | tenferro-rs | f64 | n=64 | 1 | 512 | 5946.8 | 188.0 | 2.4% | ok |
| axpby_accum_strided_dest_f64_block_host | #2143 | host | rust-loop | f64 | n=64 | 1 | 512 | 4676.8 | 144.2 | 2.2% | ok |
| small_grad_qr_f64_trace-prepared_n4 | #2186 | trace-prepared | tenferro-rs | f64 | op=qr, n=4 | 1 | 64 | 37627.7 | 626.5 | 1.4% | ok |
| small_grad_qr_f64_pytorch_n4 | #2186 | pytorch | pytorch-cpu | f64 | op=qr, n=4 | 1 | 32 | 82277.0 | 805.7 | 0.8% | ok |
| small_grad_qr_f64_trace-prepared_n16 | #2186 | trace-prepared | tenferro-rs | f64 | op=qr, n=16 | 1 | 32 | 48388.1 | 644.2 | 1.0% | ok |
| small_grad_qr_f64_pytorch_n16 | #2186 | pytorch | pytorch-cpu | f64 | op=qr, n=16 | 1 | 32 | 92308.7 | 349.4 | 1.4% | ok |
| small_grad_eigh_f64_trace-prepared_n4 | #2186 | trace-prepared | tenferro-rs | f64 | op=eigh, n=4 | 1 | 64 | 47759.5 | 213.1 | 1.0% | ok |
| small_grad_eigh_f64_pytorch_n4 | #2186 | pytorch | pytorch-cpu | f64 | op=eigh, n=4 | 1 | 32 | 84957.2 | 540.2 | 1.1% | ok |
| small_grad_eigh_f64_trace-prepared_n16 | #2186 | trace-prepared | tenferro-rs | f64 | op=eigh, n=16 | 1 | 32 | 80619.5 | 765.3 | 1.7% | ok |
| small_grad_eigh_f64_pytorch_n16 | #2186 | pytorch | pytorch-cpu | f64 | op=eigh, n=16 | 1 | 32 | 106831.7 | 1841.2 | 1.8% | ok |
| small_grad_lu_f64_trace-prepared_n4 | #2186 | trace-prepared | tenferro-rs | f64 | op=lu, n=4 | 1 | 128 | 23894.7 | 76.8 | 0.4% | ok |
| small_grad_lu_f64_pytorch_n4 | #2186 | pytorch | pytorch-cpu | f64 | op=lu, n=4 | 1 | 32 | 89691.1 | 697.6 | 1.2% | ok |
| small_grad_lu_f64_trace-prepared_n16 | #2186 | trace-prepared | tenferro-rs | f64 | op=lu, n=16 | 1 | 64 | 29893.3 | 95.8 | 0.8% | ok |
| small_grad_lu_f64_pytorch_n16 | #2186 | pytorch | pytorch-cpu | f64 | op=lu, n=16 | 1 | 32 | 97474.4 | 1337.4 | 1.1% | ok |
| small_grad_svd_f64_trace-prepared_n4 | #2186 | trace-prepared | tenferro-rs | f64 | op=svd, n=4 | 1 | 32 | 78062.0 | 399.5 | 0.6% | ok |
| small_grad_svd_f64_pytorch_n4 | #2186 | pytorch | pytorch-cpu | f64 | op=svd, n=4 | 1 | 32 | 121689.7 | 834.2 | 1.0% | ok |
| small_grad_svd_f64_trace-prepared_n16 | #2186 | trace-prepared | tenferro-rs | f64 | op=svd, n=16 | 1 | 16 | 146853.4 | 4625.6 | 2.0% | ok |
| small_grad_svd_f64_pytorch_n16 | #2186 | pytorch | pytorch-cpu | f64 | op=svd, n=16 | 1 | 16 | 161117.6 | 2660.5 | 1.7% | ok |
| small_grad_solve_f64_trace-prepared_n4 | #2186 | trace-prepared | tenferro-rs | f64 | op=solve, n=4 | 1 | 128 | 19800.5 | 121.8 | 1.8% | ok |
| small_grad_solve_f64_pytorch_n4 | #2186 | pytorch | pytorch-cpu | f64 | op=solve, n=4 | 1 | 32 | 62968.7 | 1445.9 | 1.8% | ok |
| small_grad_solve_f64_trace-prepared_n16 | #2186 | trace-prepared | tenferro-rs | f64 | op=solve, n=16 | 1 | 128 | 27871.0 | 119.3 | 0.6% | ok |
| small_grad_solve_f64_pytorch_n16 | #2186 | pytorch | pytorch-cpu | f64 | op=solve, n=16 | 1 | 32 | 73501.2 | 1964.8 | 1.9% | ok |
| fft_axis0_c64_fft-executor_n128 | #1765 | fft-executor | tenferro-rs | c64 | n=128 | 1 | 1 | 5803121.0 | 51361.0 | 0.7% | ok |
| fft_axis0_c64_pytorch_n128 | #1765 | pytorch | pytorch-cpu | c64 | n=128 | 1 | 1 | 10979806.0 | 340208.0 | 3.5% | ok |
| degenerate_dot_hadamard_f64_dot-general-shared_n1024 | #2111 | dot-general-shared | tenferro-rs | f64 | form=hadamard, n=1024 | 1 | 2 | 492784.0 | 39536.0 | 5.3% | ok |
| degenerate_dot_hadamard_f64_pytorch_n1024 | #2111 | pytorch | pytorch-cpu | f64 | form=hadamard, n=1024 | 1 | 8 | 2290912.5 | 108476.1 | 2.9% | ok |
| degenerate_dot_outer_f64_dot-general-shared_n1024 | #2111 | dot-general-shared | tenferro-rs | f64 | form=outer, n=1024 | 1 | 2 | 168283.5 | 8111.8 | 4.3% | ok |
| degenerate_dot_outer_f64_pytorch_n1024 | #2111 | pytorch | pytorch-cpu | f64 | form=outer, n=1024 | 1 | 8 | 2006017.5 | 66371.2 | 2.1% | ok |
| degenerate_dot_rank0_f64_dot-general-shared_n1024 | #2111 | dot-general-shared | tenferro-rs | f64 | form=rank0, n=1024 | 1 | 2 | 847182.5 | 13389.5 | 1.2% | ok |
| degenerate_dot_rank0_f64_pytorch_n1024 | #2111 | pytorch | pytorch-cpu | f64 | form=rank0, n=1024 | 1 | 4 | 331485.2 | 5566.9 | 2.6% | ok |
| erased_add_f64_session-add_n256 | #1766 | session-add | tenferro-rs | f64 | n=256 | 1 | 2048 | 847.0 | 37.2 | 7.4% | ok |
| erased_add_f64_pytorch_n256 | #1766 | pytorch | pytorch-cpu | f64 | n=256 | 1 | 2048 | 1290.4 | 6.4 | 3.0% | ok |
| erased_add_f64_session-add_n4096 | #1766 | session-add | tenferro-rs | f64 | n=4096 | 1 | 256 | 1880.0 | 59.2 | 4.1% | ok |
| erased_add_f64_pytorch_n4096 | #1766 | pytorch | pytorch-cpu | f64 | n=4096 | 1 | 1024 | 3295.8 | 82.7 | 1.6% | ok |
| erased_add_f64_session-add_n1048576 | #1766 | session-add | tenferro-rs | f64 | n=1048576 | 1 | 2 | 476414.5 | 12881.8 | 4.0% | ok |
| erased_add_f64_pytorch_n1048576 | #1766 | pytorch | pytorch-cpu | f64 | n=1048576 | 1 | 4 | 445666.5 | 7632.5 | 1.6% | ok |
| axpby_strided_source_f64_1024x1024_t1 | #2220 #2214 | strided-source | tenferro-rs | f64 | n=1024 | 1 | 2 | 2937313.5 | 202997.5 | 5.6% | ok |
| axpby_contiguous_source_f64_1024x1024_t1 | #2220 #2214 | contiguous-source | tenferro-rs | f64 | n=1024 | 1 | 8 | 417683.9 | 4287.6 | 2.3% | ok |
| axpby_strided_source_f64_1024x1024_host_t1 | #2220 #2214 | host | rust-loop | f64 | n=1024 | 1 | 2 | 1613991.5 | 89072.0 | 3.6% | ok |
| erased_construct_f64_col_n8 | #2177 | col-major | tenferro-rs | f64 | n=8, layout=col | 1 | 16384 | 98.9 | 1.8 | 27.6% | NOISY |
| erased_construct_f64_row_n8 | #2177 | row-major | tenferro-rs | f64 | n=8, layout=row | 1 | 8192 | 210.7 | 2.9 | 12.8% | NOISY |
| erased_construct_f64_col_n4096 | #2177 | col-major | tenferro-rs | f64 | n=4096, layout=col | 1 | 16384 | 222.4 | 6.2 | 1.9% | ok |
| erased_construct_f64_row_n4096 | #2177 | row-major | tenferro-rs | f64 | n=4096, layout=row | 1 | 256 | 13107.5 | 188.2 | 2.3% | ok |
| erased_construct_f64_col_n65536 | #2177 | col-major | tenferro-rs | f64 | n=65536, layout=col | 1 | 1024 | 112.6 | 2.0 | 2.3% | ok |
| erased_construct_f64_row_n65536 | #2177 | row-major | tenferro-rs | f64 | n=65536, layout=row | 1 | 16 | 211313.8 | 3138.4 | 2.9% | ok |
| eager_small_leaf_f64_eager-held-session | #1704 | eager-held-session | tenferro-rs | f64 | op=leaf | 1 | 1024 | 2347.9 | 52.7 | 2.0% | ok |
| eager_small_add_f64_eager-held-session | #1667 | eager-held-session | tenferro-rs | f64 | op=add | 1 | 1024 | 1706.1 | 12.1 | 0.6% | ok |
| eager_small_add_f64_pytorch | #1667 | pytorch | pytorch-cpu | f64 | op=add | 1 | 2048 | 1217.0 | 8.5 | 0.5% | ok |
| eager_small_gemm_f64_eager-held-session | #1904 | eager-held-session | tenferro-rs | f64 | op=gemm | 1 | 1024 | 2619.0 | 4.8 | 0.4% | ok |
| eager_small_gemm_f64_pytorch | #1904 | pytorch | pytorch-cpu | f64 | op=gemm | 1 | 2048 | 1349.0 | 13.0 | 0.7% | ok |
| trace_metadata_trace-fold_f32_1024x64_k50 | #2226 #2238 | trace-fold | tenferro-rs | f32 | rows=1024, cols=64, steps=50 | 1 | 1 | 4543899.0 | 14250.0 | 0.2% | ok |
| trace_metadata_trace-outputs_f32_1024x64_k50 | #2226 #2238 | trace-outputs | tenferro-rs | f32 | rows=1024, cols=64, steps=50 | 1 | 1 | 2006167.0 | 36870.0 | 1.8% | ok |
| trace_metadata_eager_f32_1024x64_k50 | #2226 #2238 | eager | tenferro-rs | f32 | rows=1024, cols=64, steps=50 | 1 | 1 | 5047554.0 | 31722.0 | 0.7% | ok |
| trace_metadata_pytorch_f32_1024x64_k50 | #2226 #2238 | pytorch | pytorch-cpu | f32 | rows=1024, cols=64, steps=50 | 1 | 1 | 2092905.0 | 6509.0 | 2.2% | ok |
| trace_metadata_fixed_neg_f32_8 | #2226 #2238 | trace-fold | tenferro-rs | f32 | rows=8, cols=1, steps=0 | 1 | 1024 | 1932.4 | 3.9 | 0.5% | ok |
| two_operand_einsum_f64_2x2_string | #2325 #2324 | string | tenferro-rs | f64 | m=2, k=2, n=2, format=ij,jk->ik | 4 | 1024 | 2970.1 | 15.1 | 0.8% | ok |
| two_operand_einsum_f64_2x2_subscripts | #2325 #2324 | subscripts | tenferro-rs | f64 | m=2, k=2, n=2, format=ij,jk->ik | 4 | 1024 | 2313.3 | 181.4 | 15.6% | NOISY |
| two_operand_einsum_f64_2x2_prepared | #2325 #2324 | prepared | tenferro-rs | f64 | m=2, k=2, n=2, format=ij,jk->ik | 4 | 1024 | 1986.4 | 6.9 | 0.5% | ok |
| two_operand_einsum_f64_2x2_dot-general | #2325 #2324 | dot-general | tenferro-rs | f64 | m=2, k=2, n=2, format=ij,jk->ik | 4 | 1024 | 1927.9 | 11.3 | 0.5% | ok |
| two_operand_einsum_f64_2x2_pytorch | #2325 #2324 | pytorch | pytorch-cpu | f64 | m=2, k=2, n=2, format=ij,jk->ik | 4 | 256 | 8799.5 | 93.6 | 0.7% | ok |
| two_operand_einsum_f64_16x16_string | #2325 #2324 | string | tenferro-rs | f64 | m=16, k=16, n=16, format=ij,jk->ik | 4 | 512 | 3665.2 | 17.2 | 0.4% | ok |
| two_operand_einsum_f64_16x16_subscripts | #2325 #2324 | subscripts | tenferro-rs | f64 | m=16, k=16, n=16, format=ij,jk->ik | 4 | 1024 | 2920.1 | 18.6 | 0.8% | ok |
| two_operand_einsum_f64_16x16_prepared | #2325 #2324 | prepared | tenferro-rs | f64 | m=16, k=16, n=16, format=ij,jk->ik | 4 | 1024 | 2700.9 | 15.6 | 0.7% | ok |
| two_operand_einsum_f64_16x16_dot-general | #2325 #2324 | dot-general | tenferro-rs | f64 | m=16, k=16, n=16, format=ij,jk->ik | 4 | 1024 | 2620.3 | 25.7 | 3.5% | ok |
| two_operand_einsum_f64_16x16_pytorch | #2325 #2324 | pytorch | pytorch-cpu | f64 | m=16, k=16, n=16, format=ij,jk->ik | 4 | 256 | 9784.7 | 70.9 | 0.9% | ok |
| two_operand_einsum_c64_2x2_string | #2325 #2324 | string | tenferro-rs | c64 | m=2, k=2, n=2, format=ij,jk->ik | 4 | 512 | 2982.2 | 33.2 | 10.1% | NOISY |
| two_operand_einsum_c64_2x2_subscripts | #2325 #2324 | subscripts | tenferro-rs | c64 | m=2, k=2, n=2, format=ij,jk->ik | 4 | 1024 | 2231.5 | 10.3 | 3.9% | ok |
| two_operand_einsum_c64_2x2_prepared | #2325 #2324 | prepared | tenferro-rs | c64 | m=2, k=2, n=2, format=ij,jk->ik | 4 | 1024 | 1929.0 | 238.6 | 8.0% | ok |
| two_operand_einsum_c64_2x2_dot-general | #2325 #2324 | dot-general | tenferro-rs | c64 | m=2, k=2, n=2, format=ij,jk->ik | 4 | 1024 | 1841.3 | 13.6 | 9.5% | ok |
| two_operand_einsum_c64_2x2_pytorch | #2325 #2324 | pytorch | pytorch-cpu | c64 | m=2, k=2, n=2, format=ij,jk->ik | 4 | 256 | 8782.5 | 103.0 | 0.9% | ok |
| two_operand_einsum_c64_16x16_string | #2325 #2324 | string | tenferro-rs | c64 | m=16, k=16, n=16, format=ij,jk->ik | 4 | 512 | 3961.0 | 27.4 | 0.8% | ok |
| two_operand_einsum_c64_16x16_subscripts | #2325 #2324 | subscripts | tenferro-rs | c64 | m=16, k=16, n=16, format=ij,jk->ik | 4 | 512 | 3335.2 | 60.1 | 2.4% | ok |
| two_operand_einsum_c64_16x16_prepared | #2325 #2324 | prepared | tenferro-rs | c64 | m=16, k=16, n=16, format=ij,jk->ik | 4 | 512 | 3056.4 | 34.7 | 1.4% | ok |
| two_operand_einsum_c64_16x16_dot-general | #2325 #2324 | dot-general | tenferro-rs | c64 | m=16, k=16, n=16, format=ij,jk->ik | 4 | 512 | 2952.5 | 52.3 | 1.8% | ok |
| two_operand_einsum_c64_16x16_pytorch | #2325 #2324 | pytorch | pytorch-cpu | c64 | m=16, k=16, n=16, format=ij,jk->ik | 4 | 256 | 10087.8 | 84.3 | 0.9% | ok |
| two_operand_einsum_f64_abc_cef_abef_string | #2325 #2324 | string | tenferro-rs | f64 | m=8, k=2, n=16, format=abc,cef->abef | 4 | 1024 | 3501.2 | 16.2 | 0.6% | ok |
| two_operand_einsum_f64_abc_cef_abef_subscripts | #2325 #2324 | subscripts | tenferro-rs | f64 | m=8, k=2, n=16, format=abc,cef->abef | 4 | 1024 | 2558.7 | 16.8 | 0.6% | ok |
| two_operand_einsum_f64_abc_cef_abef_prepared | #2325 #2324 | prepared | tenferro-rs | f64 | m=8, k=2, n=16, format=abc,cef->abef | 4 | 1024 | 2189.1 | 21.2 | 0.9% | ok |
| two_operand_einsum_f64_abc_cef_abef_dot-general | #2325 #2324 | dot-general | tenferro-rs | f64 | m=8, k=2, n=16, format=abc,cef->abef | 4 | 1024 | 2130.1 | 12.5 | 0.8% | ok |
| two_operand_einsum_f64_abc_cef_abef_pytorch | #2325 #2324 | pytorch | pytorch-cpu | f64 | m=8, k=2, n=16, format=abc,cef->abef | 4 | 256 | 10759.8 | 134.9 | 0.9% | ok |
| decode_proj_f32_eager-shared_in1024_out1024_len8 | #1992 #2003 #1995 #2229 | eager-shared | tenferro-rs | f32 | in=1024, out=1024, len=8 | 4 | 4 | 887542.5 | 4571.8 | 0.6% | ok |
| decode_proj_f32_faer-direct_in1024_out1024_len8 | #1992 #2003 #1995 #2229 | faer-direct | faer-direct | f32 | in=1024, out=1024, len=8 | 4 | 16 | 175440.7 | 628.1 | 3.1% | ok |
| decode_proj_f32_host-sgemm_in1024_out1024_len8 | #1992 #2003 #1995 #2229 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=1024, len=8 | 4 | 2 | 1156520.0 | 4063.5 | 0.3% | ok |
| decode_proj_f32_pytorch_in1024_out1024_len8 | #1992 #2003 #1995 #2229 | pytorch | pytorch-cpu | f32 | in=1024, out=1024, len=8 | 4 | 8 | 277759.5 | 3384.4 | 1.2% | ok |
| decode_proj_f32_eager-shared_in1024_out1024_len64 | #1992 #2003 #1995 #2229 | eager-shared | tenferro-rs | f32 | in=1024, out=1024, len=64 | 4 | 2 | 1581333.0 | 12543.2 | 0.8% | ok |
| decode_proj_f32_faer-direct_in1024_out1024_len64 | #1992 #2003 #1995 #2229 | faer-direct | faer-direct | f32 | in=1024, out=1024, len=64 | 4 | 4 | 659770.5 | 3185.9 | 0.5% | ok |
| decode_proj_f32_host-sgemm_in1024_out1024_len64 | #1992 #2003 #1995 #2229 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=1024, len=64 | 4 | 2 | 1642569.0 | 8592.8 | 0.6% | ok |
| decode_proj_f32_pytorch_in1024_out1024_len64 | #1992 #2003 #1995 #2229 | pytorch | pytorch-cpu | f32 | in=1024, out=1024, len=64 | 4 | 4 | 986696.8 | 4969.9 | 0.4% | ok |
| decode_proj_f32_eager-shared_in1024_out1024_len512 | #1992 #2003 #1995 #2229 | eager-shared | tenferro-rs | f32 | in=1024, out=1024, len=512 | 4 | 1 | 11188555.0 | 906160.5 | 4.7% | ok |
| decode_proj_f32_faer-direct_in1024_out1024_len512 | #1992 #2003 #1995 #2229 | faer-direct | faer-direct | f32 | in=1024, out=1024, len=512 | 4 | 1 | 4823773.0 | 25107.0 | 0.4% | ok |
| decode_proj_f32_host-sgemm_in1024_out1024_len512 | #1992 #2003 #1995 #2229 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=1024, len=512 | 4 | 1 | 6129684.0 | 18156.5 | 0.3% | ok |
| decode_proj_f32_pytorch_in1024_out1024_len512 | #1992 #2003 #1995 #2229 | pytorch | pytorch-cpu | f32 | in=1024, out=1024, len=512 | 4 | 1 | 6949537.0 | 74770.5 | 2.5% | ok |
| decode_proj_f32_eager-shared_in1024_out4096_len8 | #1992 #2003 #1995 #2229 | eager-shared | tenferro-rs | f32 | in=1024, out=4096, len=8 | 4 | 1 | 4191739.0 | 122662.0 | 2.7% | ok |
| decode_proj_f32_faer-direct_in1024_out4096_len8 | #1992 #2003 #1995 #2229 | faer-direct | faer-direct | f32 | in=1024, out=4096, len=8 | 4 | 4 | 631333.8 | 4859.6 | 1.3% | ok |
| decode_proj_f32_host-sgemm_in1024_out4096_len8 | #1992 #2003 #1995 #2229 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=4096, len=8 | 4 | 1 | 4715163.0 | 40940.0 | 0.8% | ok |
| decode_proj_f32_pytorch_in1024_out4096_len8 | #1992 #2003 #1995 #2229 | pytorch | pytorch-cpu | f32 | in=1024, out=4096, len=8 | 4 | 2 | 1228762.5 | 25968.5 | 1.4% | ok |
| decode_proj_f32_eager-shared_in1024_out4096_len64 | #1992 #2003 #1995 #2229 | eager-shared | tenferro-rs | f32 | in=1024, out=4096, len=64 | 4 | 1 | 7072018.0 | 125125.5 | 1.9% | ok |
| decode_proj_f32_faer-direct_in1024_out4096_len64 | #1992 #2003 #1995 #2229 | faer-direct | faer-direct | f32 | in=1024, out=4096, len=64 | 4 | 1 | 2421671.0 | 20665.0 | 4.4% | ok |
| decode_proj_f32_host-sgemm_in1024_out4096_len64 | #1992 #2003 #1995 #2229 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=4096, len=64 | 4 | 1 | 6639298.0 | 34951.5 | 0.4% | ok |
| decode_proj_f32_pytorch_in1024_out4096_len64 | #1992 #2003 #1995 #2229 | pytorch | pytorch-cpu | f32 | in=1024, out=4096, len=64 | 4 | 1 | 4264567.0 | 59327.5 | 1.1% | ok |
| decode_proj_f32_eager-shared_in1024_out4096_len512 | #1992 #2003 #1995 #2229 | eager-shared | tenferro-rs | f32 | in=1024, out=4096, len=512 | 4 | 1 | 52884774.0 | 10233337.0 | 11.1% | NOISY |
| decode_proj_f32_faer-direct_in1024_out4096_len512 | #1992 #2003 #1995 #2229 | faer-direct | faer-direct | f32 | in=1024, out=4096, len=512 | 4 | 1 | 19252640.0 | 44736.0 | 0.3% | ok |
| decode_proj_f32_host-sgemm_in1024_out4096_len512 | #1992 #2003 #1995 #2229 | host-sgemm | matrixmultiply-sgemm | f32 | in=1024, out=4096, len=512 | 4 | 1 | 24732594.0 | 124269.0 | 0.8% | ok |
| decode_proj_f32_pytorch_in1024_out4096_len512 | #1992 #2003 #1995 #2229 | pytorch | pytorch-cpu | f32 | in=1024, out=4096, len=512 | 4 | 1 | 27331917.0 | 496815.0 | 14.7% | NOISY |
| decode_proj_view_operand_f32_prepared_in1024_out1024_len8_t4 | #2216 | prepared-view | tenferro-rs | f32 | in=1024, out=1024, len=8 | 4 | 1 | 2899207.0 | 11857.5 | 0.4% | ok |
| decode_proj_contiguous_operand_f32_prepared_in1024_out1024_len8_t4 | #2216 | prepared-contiguous | tenferro-rs | f32 | in=1024, out=1024, len=8 | 4 | 4 | 886040.2 | 1579.9 | 0.3% | ok |
| decode_proj_view_operand_f32_pytorch_in1024_out1024_len8_t4 | #2216 | pytorch | pytorch-cpu | f32 | in=1024, out=1024, len=8 | 4 | 8 | 284794.9 | 3932.8 | 0.9% | ok |
| gemm_c64_mm256_t1 | #1900 | dot-general-shared | tenferro-rs | c64 | n=256 | 4 | 2 | 1652653.5 | 72940.2 | 2.5% | ok |
| gemm_f64_mm256_t1 | #1900 | dot-general-shared | tenferro-rs | f64 | n=256 | 4 | 8 | 395531.1 | 5041.2 | 7.6% | ok |
| conj_dot_f64_dot-general-rank1_len100 | #1615 | dot-general-rank1 | tenferro-rs | f64 | len=100 | 4 | 1024 | 1715.5 | 26.3 | 7.8% | ok |
| conj_dot_f64_portable-loop_len100 | #1615 | portable-loop | rust-loop | f64 | len=100 | 4 | 32768 | 58.7 | 0.3 | 1.0% | ok |
| conj_dot_f64_pytorch-vdot_len100 | #1615 | pytorch-vdot | pytorch-cpu | f64 | len=100 | 4 | 2048 | 1103.1 | 11.8 | 6.4% | ok |
| conj_dot_f64_dot-general-rank1_len10000 | #1615 | dot-general-rank1 | tenferro-rs | f64 | len=10000 | 4 | 512 | 5928.3 | 61.9 | 3.0% | ok |
| conj_dot_f64_portable-loop_len10000 | #1615 | portable-loop | rust-loop | f64 | len=10000 | 4 | 256 | 8182.4 | 40.9 | 0.3% | ok |
| conj_dot_f64_pytorch-vdot_len10000 | #1615 | pytorch-vdot | pytorch-cpu | f64 | len=10000 | 4 | 1024 | 3828.5 | 18.8 | 1.4% | ok |
| conj_dot_f64_dot-general-rank1_len1000000 | #1615 | dot-general-rank1 | tenferro-rs | f64 | len=1000000 | 4 | 8 | 419484.1 | 1561.6 | 0.3% | ok |
| conj_dot_f64_portable-loop_len1000000 | #1615 | portable-loop | rust-loop | f64 | len=1000000 | 4 | 4 | 823705.5 | 3110.8 | 1.3% | ok |
| conj_dot_f64_pytorch-vdot_len1000000 | #1615 | pytorch-vdot | pytorch-cpu | f64 | len=1000000 | 4 | 32 | 87442.4 | 657.5 | 0.6% | ok |
| conj_dot_c64_dot-general-rank1_len100 | #1615 | dot-general-rank1 | tenferro-rs | c64 | len=100 | 4 | 1024 | 1725.4 | 7.7 | 8.5% | ok |
| conj_dot_c64_portable-loop_len100 | #1615 | portable-loop | rust-loop | c64 | len=100 | 4 | 32768 | 85.6 | 0.3 | 1.5% | ok |
| conj_dot_c64_pytorch-vdot_len100 | #1615 | pytorch-vdot | pytorch-cpu | c64 | len=100 | 4 | 2048 | 978.5 | 16.0 | 285.9% | NOISY |
| conj_dot_c64_dot-general-rank1_len10000 | #1615 | dot-general-rank1 | tenferro-rs | c64 | len=10000 | 4 | 256 | 10008.2 | 20.6 | 1.1% | ok |
| conj_dot_c64_portable-loop_len10000 | #1615 | portable-loop | rust-loop | c64 | len=10000 | 4 | 256 | 8255.4 | 21.0 | 0.3% | ok |
| conj_dot_c64_pytorch-vdot_len10000 | #1615 | pytorch-vdot | pytorch-cpu | c64 | len=10000 | 4 | 512 | 5257.6 | 20.6 | 0.8% | ok |
| conj_dot_c64_dot-general-rank1_len1000000 | #1615 | dot-general-rank1 | tenferro-rs | c64 | len=1000000 | 4 | 4 | 897534.8 | 6323.2 | 0.5% | ok |
| conj_dot_c64_portable-loop_len1000000 | #1615 | portable-loop | rust-loop | c64 | len=1000000 | 4 | 4 | 909472.2 | 2886.8 | 0.6% | ok |
| conj_dot_c64_pytorch-vdot_len1000000 | #1615 | pytorch-vdot | pytorch-cpu | c64 | len=1000000 | 4 | 8 | 305163.8 | 9448.9 | 2.1% | ok |
| small_solve_f64_solve-loop_k16_b64 | #2007 #1878 | solve-loop | tenferro-rs | f64 | k=16, batch=64 | 4 | 8 | 264335.9 | 2528.1 | 0.9% | ok |
| small_solve_f64_triangular-solve-loop_k16_b64 | #2007 #1878 | triangular-solve-loop | tenferro-rs | f64 | k=16, batch=64 | 4 | 16 | 209451.6 | 1304.4 | 0.7% | ok |
| small_solve_f64_solve-batched_k16_b64 | #2007 #1878 | solve-batched | tenferro-rs | f64 | k=16, batch=64 | 4 | 16 | 203227.4 | 1184.9 | 0.5% | ok |
| small_solve_f64_pytorch-solve-batched_k16_b64 | #2007 #1878 | pytorch-solve-batched | pytorch-cpu | f64 | k=16, batch=64 | 4 | 16 | 196941.2 | 923.5 | 1.3% | ok |
| small_solve_f64_pytorch-solve-triangular-batched_k16_b64 | #2007 #1878 | pytorch-solve-triangular-batched | pytorch-cpu | f64 | k=16, batch=64 | 4 | 32 | 58812.9 | 1719.7 | 7.0% | ok |
| small_solve_f64_solve-loop_k64_b64 | #2007 #1878 | solve-loop | tenferro-rs | f64 | k=64, batch=64 | 4 | 1 | 5606991.0 | 26093.5 | 0.6% | ok |
| small_solve_f64_triangular-solve-loop_k64_b64 | #2007 #1878 | triangular-solve-loop | tenferro-rs | f64 | k=64, batch=64 | 4 | 1 | 2008700.0 | 34240.0 | 10.9% | NOISY |
| small_solve_f64_solve-batched_k64_b64 | #2007 #1878 | solve-batched | tenferro-rs | f64 | k=64, batch=64 | 4 | 1 | 5431332.0 | 51727.0 | 3.6% | ok |
| small_solve_f64_pytorch-solve-batched_k64_b64 | #2007 #1878 | pytorch-solve-batched | pytorch-cpu | f64 | k=64, batch=64 | 4 | 1 | 2739041.0 | 28476.5 | 1.1% | ok |
| small_solve_f64_pytorch-solve-triangular-batched_k64_b64 | #2007 #1878 | pytorch-solve-triangular-batched | pytorch-cpu | f64 | k=64, batch=64 | 4 | 4 | 752067.0 | 4337.6 | 0.7% | ok |
| tanh_chain_f32_compiled-prepared_k10_1024x64 | #1990 | compiled-prepared | tenferro-rs | f32 | rows=1024, cols=64, chain=10 | 4 | 8 | 439492.5 | 1012.8 | 2.4% | ok |
| tanh_chain_f32_eager-shared_k10_1024x64 | #1990 | eager-shared | tenferro-rs | f32 | rows=1024, cols=64, chain=10 | 4 | 8 | 474735.8 | 1263.8 | 0.6% | ok |
| tanh_chain_f32_compiled-prepared_k100_1024x64 | #1990 | compiled-prepared | tenferro-rs | f32 | rows=1024, cols=64, chain=100 | 4 | 1 | 4318075.0 | 16163.5 | 2.3% | ok |
| tanh_chain_f32_eager-shared_k100_1024x64 | #1990 | eager-shared | tenferro-rs | f32 | rows=1024, cols=64, chain=100 | 4 | 1 | 4729674.0 | 14796.5 | 3.7% | ok |
| tanh_chain_f32_compiled-prepared_k200_1024x64 | #1990 | compiled-prepared | tenferro-rs | f32 | rows=1024, cols=64, chain=200 | 4 | 1 | 8673165.0 | 26530.0 | 0.4% | ok |
| tanh_chain_f32_eager-shared_k200_1024x64 | #1990 | eager-shared | tenferro-rs | f32 | rows=1024, cols=64, chain=200 | 4 | 1 | 9448331.0 | 35518.0 | 2.0% | ok |
| layer_norm_f32_eager-composed_d1024_len8_b1 | #2006 | eager-composed | tenferro-rs | f32 | norm=layer_norm, d=1024, len=8, batch=1 | 4 | 32 | 51430.8 | 582.7 | 8.3% | ok |
| layer_norm_f32_pytorch-fused_d1024_len8_b1 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=layer_norm, d=1024, len=8, batch=1 | 4 | 512 | 7170.7 | 250.7 | 3.8% | ok |
| layer_norm_f32_eager-single-call_d1024_len8_b1 | #2006 | eager-single-call | tenferro-rs | f32 | norm=layer_norm, d=1024, len=8, batch=1 | 4 | 32 | 53899.4 | 274.9 | 7.6% | ok |
| layer_norm_f32_eager-composed_d1024_len8_b8 | #2006 | eager-composed | tenferro-rs | f32 | norm=layer_norm, d=1024, len=8, batch=8 | 4 | 16 | 139981.4 | 2422.3 | 1.2% | ok |
| layer_norm_f32_pytorch-fused_d1024_len8_b8 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=layer_norm, d=1024, len=8, batch=8 | 4 | 128 | 15681.5 | 205.6 | 2.2% | ok |
| layer_norm_f32_eager-single-call_d1024_len8_b8 | #2006 | eager-single-call | tenferro-rs | f32 | norm=layer_norm, d=1024, len=8, batch=8 | 4 | 16 | 151380.1 | 4321.4 | 2.0% | ok |
| layer_norm_f32_eager-composed_d1024_len64_b1 | #2006 | eager-composed | tenferro-rs | f32 | norm=layer_norm, d=1024, len=64, batch=1 | 4 | 16 | 140525.9 | 3761.5 | 1.5% | ok |
| layer_norm_f32_pytorch-fused_d1024_len64_b1 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=layer_norm, d=1024, len=64, batch=1 | 4 | 128 | 16086.7 | 509.7 | 3.1% | ok |
| layer_norm_f32_eager-single-call_d1024_len64_b1 | #2006 | eager-single-call | tenferro-rs | f32 | norm=layer_norm, d=1024, len=64, batch=1 | 4 | 16 | 152459.8 | 1750.4 | 3.3% | ok |
| layer_norm_f32_eager-composed_d1024_len64_b8 | #2006 | eager-composed | tenferro-rs | f32 | norm=layer_norm, d=1024, len=64, batch=8 | 4 | 4 | 572792.0 | 28942.8 | 3.4% | ok |
| layer_norm_f32_pytorch-fused_d1024_len64_b8 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=layer_norm, d=1024, len=64, batch=8 | 4 | 16 | 90368.7 | 1066.6 | 1.6% | ok |
| layer_norm_f32_eager-single-call_d1024_len64_b8 | #2006 | eager-single-call | tenferro-rs | f32 | norm=layer_norm, d=1024, len=64, batch=8 | 4 | 4 | 527970.8 | 14943.4 | 2.1% | ok |
| rms_norm_f32_eager-composed_d1024_len8_b1 | #2006 | eager-composed | tenferro-rs | f32 | norm=rms_norm, d=1024, len=8, batch=1 | 4 | 64 | 27782.8 | 796.0 | 8.6% | ok |
| rms_norm_f32_pytorch-fused_d1024_len8_b1 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=rms_norm, d=1024, len=8, batch=1 | 4 | 256 | 13328.5 | 611.0 | 4.7% | ok |
| rms_norm_f32_eager-single-call_d1024_len8_b1 | #2006 | eager-single-call | tenferro-rs | f32 | norm=rms_norm, d=1024, len=8, batch=1 | 4 | 64 | 32252.4 | 449.7 | 5.7% | ok |
| rms_norm_f32_eager-composed_d1024_len8_b8 | #2006 | eager-composed | tenferro-rs | f32 | norm=rms_norm, d=1024, len=8, batch=8 | 4 | 16 | 69678.1 | 2910.6 | 3.0% | ok |
| rms_norm_f32_pytorch-fused_d1024_len8_b8 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=rms_norm, d=1024, len=8, batch=8 | 4 | 64 | 51604.8 | 733.6 | 2.8% | ok |
| rms_norm_f32_eager-single-call_d1024_len8_b8 | #2006 | eager-single-call | tenferro-rs | f32 | norm=rms_norm, d=1024, len=8, batch=8 | 4 | 16 | 73970.1 | 1708.4 | 1.7% | ok |
| rms_norm_f32_eager-composed_d1024_len64_b1 | #2006 | eager-composed | tenferro-rs | f32 | norm=rms_norm, d=1024, len=64, batch=1 | 4 | 16 | 69650.5 | 3300.9 | 3.3% | ok |
| rms_norm_f32_pytorch-fused_d1024_len64_b1 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=rms_norm, d=1024, len=64, batch=1 | 4 | 64 | 50558.2 | 549.3 | 1.2% | ok |
| rms_norm_f32_eager-single-call_d1024_len64_b1 | #2006 | eager-single-call | tenferro-rs | f32 | norm=rms_norm, d=1024, len=64, batch=1 | 4 | 16 | 73984.4 | 1115.7 | 0.9% | ok |
| rms_norm_f32_eager-composed_d1024_len64_b8 | #2006 | eager-composed | tenferro-rs | f32 | norm=rms_norm, d=1024, len=64, batch=8 | 4 | 4 | 251301.8 | 5555.8 | 2.1% | ok |
| rms_norm_f32_pytorch-fused_d1024_len64_b8 | #2006 | pytorch-fused | pytorch-cpu | f32 | norm=rms_norm, d=1024, len=64, batch=8 | 4 | 16 | 151895.9 | 6367.9 | 4.8% | ok |
| rms_norm_f32_eager-single-call_d1024_len64_b8 | #2006 | eager-single-call | tenferro-rs | f32 | norm=rms_norm, d=1024, len=64, batch=8 | 4 | 4 | 270522.5 | 6726.2 | 2.6% | ok |
| small_contraction_abcd-dbef-acef_f64_shared-session-default-pool_d4 | #1885 #1897 #1901 | shared-session-default-pool | tenferro-rs | f64 | extent=4 | 4 | 512 | 3549.8 | 19.6 | 0.8% | ok |
| activation_erf_f32_eager-single-call_1024x64 | #1975 #2069 #2032 | eager-single-call | tenferro-rs | f32 | op=erf, rows=1024, cols=64 | 4 | 32 | 72809.5 | 387.6 | 0.6% | ok |
| activation_erf_f32_session-single-call_1024x64 | #1975 #2069 #2032 | session-single-call | tenferro-rs | f32 | op=erf, rows=1024, cols=64 | 4 | 32 | 71910.7 | 599.0 | 1.0% | ok |
| activation_erf_f32_pytorch_1024x64 | #1975 #2069 #2032 | pytorch | pytorch-cpu | f32 | op=erf, rows=1024, cols=64 | 4 | 2 | 1989377.0 | 1862689.2 | 80.0% | NOISY |
| activation_sigmoid_f32_eager-single-call_1024x64 | #1975 #2069 #2032 | eager-single-call | tenferro-rs | f32 | op=sigmoid, rows=1024, cols=64 | 4 | 64 | 14455.4 | 132.1 | 0.9% | ok |
| activation_sigmoid_f32_session-single-call_1024x64 | #1975 #2069 #2032 | session-single-call | tenferro-rs | f32 | op=sigmoid, rows=1024, cols=64 | 4 | 64 | 14063.9 | 122.0 | 1.7% | ok |
| activation_sigmoid_f32_eager-composed_1024x64 | #1975 #2069 #2032 | eager-composed | tenferro-rs | f32 | op=sigmoid, rows=1024, cols=64 | 4 | 32 | 43334.7 | 623.9 | 2.6% | ok |
| activation_sigmoid_f32_pytorch_1024x64 | #1975 #2069 #2032 | pytorch | pytorch-cpu | f32 | op=sigmoid, rows=1024, cols=64 | 4 | 8 | 28385.8 | 255.3 | 192.6% | NOISY |
| activation_silu_f32_eager-single-call_1024x64 | #1975 #2069 #2032 | eager-single-call | tenferro-rs | f32 | op=silu, rows=1024, cols=64 | 4 | 64 | 14936.0 | 144.0 | 6.3% | ok |
| activation_silu_f32_session-single-call_1024x64 | #1975 #2069 #2032 | session-single-call | tenferro-rs | f32 | op=silu, rows=1024, cols=64 | 4 | 64 | 14383.3 | 99.5 | 0.8% | ok |
| activation_silu_f32_eager-composed_1024x64 | #1975 #2069 #2032 | eager-composed | tenferro-rs | f32 | op=silu, rows=1024, cols=64 | 4 | 32 | 53158.6 | 391.0 | 0.8% | ok |
| activation_silu_f32_pytorch_1024x64 | #1975 #2069 #2032 | pytorch | pytorch-cpu | f32 | op=silu, rows=1024, cols=64 | 4 | 8 | 30537.8 | 1397.8 | 168.0% | NOISY |
| activation_softplus_f32_eager-single-call_1024x64 | #1975 #2069 #2032 | eager-single-call | tenferro-rs | f32 | op=softplus, rows=1024, cols=64 | 4 | 64 | 40715.8 | 318.8 | 2.2% | ok |
| activation_softplus_f32_session-single-call_1024x64 | #1975 #2069 #2032 | session-single-call | tenferro-rs | f32 | op=softplus, rows=1024, cols=64 | 4 | 64 | 40057.2 | 161.6 | 2.3% | ok |
| activation_softplus_f32_eager-composed_1024x64 | #1975 #2069 #2032 | eager-composed | tenferro-rs | f32 | op=softplus, rows=1024, cols=64 | 4 | 32 | 79498.8 | 726.4 | 1.9% | ok |
| activation_softplus_f32_pytorch_1024x64 | #1975 #2069 #2032 | pytorch | pytorch-cpu | f32 | op=softplus, rows=1024, cols=64 | 4 | 1 | 85817.0 | 516.0 | 4.7% | ok |
| activation_gelu_f32_eager-single-call_1024x64 | #1975 #2030 #2069 #2228 | eager-single-call | tenferro-rs | f32 | op=gelu, rows=1024, cols=64 | 4 | 64 | 19932.3 | 147.6 | 0.6% | ok |
| activation_gelu_f32_session-single-call_1024x64 | #1975 #2030 #2069 #2228 | session-single-call | tenferro-rs | f32 | op=gelu, rows=1024, cols=64 | 4 | 64 | 19428.7 | 131.4 | 1.6% | ok |
| activation_gelu_f32_pytorch_1024x64 | #1975 #2030 #2069 #2228 | pytorch | pytorch-cpu | f32 | op=gelu, rows=1024, cols=64 | 4 | 128 | 24163.8 | 141.0 | 0.8% | ok |
| activation_gelu_tanh_f32_eager-single-call_1024x64 | #1975 #2069 #2032 | eager-single-call | tenferro-rs | f32 | op=gelu_tanh, rows=1024, cols=64 | 4 | 32 | 50057.8 | 198.6 | 1.9% | ok |
| activation_gelu_tanh_f32_session-single-call_1024x64 | #1975 #2069 #2032 | session-single-call | tenferro-rs | f32 | op=gelu_tanh, rows=1024, cols=64 | 4 | 32 | 49773.6 | 483.8 | 2.4% | ok |
| activation_gelu_tanh_f32_eager-composed_1024x64 | #1975 #2069 #2032 | eager-composed | tenferro-rs | f32 | op=gelu_tanh, rows=1024, cols=64 | 4 | 16 | 149255.1 | 1135.5 | 4.8% | ok |
| activation_gelu_tanh_f32_pytorch_1024x64 | #1975 #2069 #2032 | pytorch | pytorch-cpu | f32 | op=gelu_tanh, rows=1024, cols=64 | 4 | 1 | 51707.0 | 250.5 | 2.2% | ok |
| softmax_f32_eager-single-call_len64_b8 | #1976 #2031 #2231 | eager-single-call | tenferro-rs | f32 | op=softmax, len=64, batch=8, axis=0 | 4 | 64 | 13410.1 | 67.3 | 0.7% | ok |
| softmax_f32_session-single-call_len64_b8 | #1976 #2031 #2231 | session-single-call | tenferro-rs | f32 | op=softmax, len=64, batch=8, axis=0 | 4 | 64 | 13051.6 | 122.1 | 1.0% | ok |
| softmax_f32_eager-composed_len64_b8 | #1976 #2031 #2231 | eager-composed | tenferro-rs | f32 | op=softmax, len=64, batch=8, axis=0 | 4 | 32 | 65703.4 | 415.1 | 2.6% | ok |
| softmax_f32_pytorch_len64_b8 | #1976 #2031 #2231 | pytorch | pytorch-cpu | f32 | op=softmax, len=64, batch=8, axis=0 | 4 | 256 | 13397.2 | 284.7 | 3.7% | ok |
| softmax_f32_eager-single-call_len512_b8 | #1976 #2231 | eager-single-call | tenferro-rs | f32 | op=softmax, len=512, batch=8, axis=0 | 4 | 2 | 508321.5 | 5676.2 | 0.6% | ok |
| softmax_f32_session-single-call_len512_b8 | #1976 #2231 | session-single-call | tenferro-rs | f32 | op=softmax, len=512, batch=8, axis=0 | 4 | 2 | 503715.0 | 7646.5 | 1.7% | ok |
| softmax_f32_eager-composed_len512_b8 | #1976 #2231 | eager-composed | tenferro-rs | f32 | op=softmax, len=512, batch=8, axis=0 | 4 | 2 | 1722176.5 | 31704.2 | 2.4% | ok |
| softmax_f32_pytorch_len512_b8 | #1976 #2231 | pytorch | pytorch-cpu | f32 | op=softmax, len=512, batch=8, axis=0 | 4 | 2 | 542327.0 | 4920.0 | 1.3% | ok |
| log_softmax_f32_eager-single-call_len64_b8 | #1976 | eager-single-call | tenferro-rs | f32 | op=log_softmax, len=64, batch=8, axis=0 | 4 | 32 | 91279.0 | 175.6 | 0.3% | ok |
| log_softmax_f32_session-single-call_len64_b8 | #1976 | session-single-call | tenferro-rs | f32 | op=log_softmax, len=64, batch=8, axis=0 | 4 | 32 | 66832.3 | 878.5 | 1.9% | ok |
| log_softmax_f32_eager-composed_len64_b8 | #1976 | eager-composed | tenferro-rs | f32 | op=log_softmax, len=64, batch=8, axis=0 | 4 | 32 | 67240.4 | 961.8 | 1.0% | ok |
| log_softmax_f32_pytorch_len64_b8 | #1976 | pytorch | pytorch-cpu | f32 | op=log_softmax, len=64, batch=8, axis=0 | 4 | 256 | 14217.0 | 377.9 | 4.0% | ok |
| log_softmax_f32_eager-single-call_len512_b8 | #1976 | eager-single-call | tenferro-rs | f32 | op=log_softmax, len=512, batch=8, axis=0 | 4 | 2 | 1467259.5 | 48884.8 | 2.8% | ok |
| log_softmax_f32_session-single-call_len512_b8 | #1976 | session-single-call | tenferro-rs | f32 | op=log_softmax, len=512, batch=8, axis=0 | 4 | 2 | 1764529.5 | 35555.0 | 1.7% | ok |
| log_softmax_f32_eager-composed_len512_b8 | #1976 | eager-composed | tenferro-rs | f32 | op=log_softmax, len=512, batch=8, axis=0 | 4 | 2 | 1649314.0 | 56413.0 | 8.3% | ok |
| log_softmax_f32_pytorch_len512_b8 | #1976 | pytorch | pytorch-cpu | f32 | op=log_softmax, len=512, batch=8, axis=0 | 4 | 2 | 569442.0 | 11379.5 | 2.4% | ok |
| masked_softmax_f32_eager-single-call_len64_b8 | #1976 | eager-single-call | tenferro-rs | f32 | op=masked_softmax, len=64, batch=8, axis=0 | 4 | 16 | 134885.1 | 1412.7 | 1.3% | ok |
| masked_softmax_f32_session-single-call_len64_b8 | #1976 | session-single-call | tenferro-rs | f32 | op=masked_softmax, len=64, batch=8, axis=0 | 4 | 16 | 117605.2 | 716.4 | 1.2% | ok |
| masked_softmax_f32_eager-composed_len64_b8 | #1976 | eager-composed | tenferro-rs | f32 | op=masked_softmax, len=64, batch=8, axis=0 | 4 | 32 | 83629.9 | 689.0 | 0.6% | ok |
| masked_softmax_f32_pytorch_len64_b8 | #1976 | pytorch | pytorch-cpu | f32 | op=masked_softmax, len=64, batch=8, axis=0 | 4 | 32 | 62227.3 | 169.6 | 0.8% | ok |
| masked_softmax_f32_eager-single-call_len512_b8 | #1976 | eager-single-call | tenferro-rs | f32 | op=masked_softmax, len=512, batch=8, axis=0 | 4 | 1 | 3116118.0 | 31581.5 | 2.1% | ok |
| masked_softmax_f32_session-single-call_len512_b8 | #1976 | session-single-call | tenferro-rs | f32 | op=masked_softmax, len=512, batch=8, axis=0 | 4 | 1 | 3208004.0 | 75702.5 | 2.6% | ok |
| masked_softmax_f32_eager-composed_len512_b8 | #1976 | eager-composed | tenferro-rs | f32 | op=masked_softmax, len=512, batch=8, axis=0 | 4 | 1 | 2447239.0 | 63794.5 | 5.6% | ok |
| masked_softmax_f32_pytorch_len512_b8 | #1976 | pytorch | pytorch-cpu | f32 | op=masked_softmax, len=512, batch=8, axis=0 | 4 | 2 | 1323817.5 | 24556.2 | 1.5% | ok |
| softmax_f32_eager-single-call_len64_b8_t4 | #2233 #2235 | eager-single-call | tenferro-rs | f32 | op=softmax, len=64, batch=8, axis=0 | 4 | 64 | 13373.4 | 146.3 | 1.0% | ok |
| softmax_f32_eager-composed_len64_b8_t4 | #2233 #2235 | eager-composed | tenferro-rs | f32 | op=softmax, len=64, batch=8, axis=0 | 4 | 32 | 66692.7 | 1013.0 | 1.0% | ok |
| softmax_f32_pytorch_len64_b8_t4 | #2233 #2235 | pytorch | pytorch-cpu | f32 | op=softmax, len=64, batch=8, axis=0 | 4 | 256 | 13568.2 | 84.6 | 0.9% | ok |
| reduce_mean_f32_eager-single-call_1024x64_axis0 | #1976 | eager-single-call | tenferro-rs | f32 | rows=1024, cols=64, axis=0 | 4 | 256 | 14326.3 | 92.1 | 0.4% | ok |
| reduce_mean_f32_session-single-call_1024x64_axis0 | #1976 | session-single-call | tenferro-rs | f32 | rows=1024, cols=64, axis=0 | 4 | 256 | 8000.9 | 51.9 | 1.7% | ok |
| reduce_mean_f32_eager-composed_1024x64_axis0 | #1976 | eager-composed | tenferro-rs | f32 | rows=1024, cols=64, axis=0 | 4 | 256 | 14096.6 | 220.8 | 1.7% | ok |
| reduce_mean_f32_pytorch_1024x64_axis0 | #1976 | pytorch | pytorch-cpu | f32 | rows=1024, cols=64, axis=0 | 4 | 256 | 8644.7 | 67.7 | 1.3% | ok |
| take_along_axis_rows_f64_eager-single-call_n64_b256 | #2008 | eager-single-call | tenferro-rs | f64 | n=64, batch=256, axis=0 | 4 | 1 | 3992925.0 | 30214.5 | 0.7% | ok |
| take_along_axis_rows_f64_session-single-call_n64_b256 | #2008 | session-single-call | tenferro-rs | f64 | n=64, batch=256, axis=0 | 4 | 1 | 3920087.0 | 9470.0 | 2.8% | ok |
| take_along_axis_rows_f64_eager-gather-prebuilt_n64_b256 | #2008 | eager-gather-prebuilt | tenferro-rs | f64 | n=64, batch=256, axis=0 | 4 | 1 | 3774252.0 | 29168.0 | 2.1% | ok |
| take_along_axis_rows_f64_pytorch_n64_b256 | #2008 | pytorch | pytorch-cpu | f64 | n=64, batch=256, axis=0 | 4 | 2 | 1382574.5 | 15483.0 | 0.7% | ok |
| lstsq_f64_eager-shared_m768_n384_rhs16 | #2027 | eager-shared | tenferro-rs | f64 | operation=lstsq | 4 | 1 | 2283497985.0 | 10793403.5 | 0.4% | ok |
| lstsq_f64_pytorch_m768_n384_rhs16 | #2027 | pytorch | pytorch-cpu | f64 | operation=lstsq | 4 | 1 | 5118289.0 | 16704.5 | 0.5% | ok |
| triangular_f64_concrete-shared_n4096_rhs64 | #2028 | concrete-shared | tenferro-rs | f64 | operation=triangular | 4 | 1 | 51578590.0 | 1924306.5 | 3.0% | ok |
| triangular_f64_pytorch_n4096_rhs64 | #2028 | pytorch | pytorch-cpu | f64 | operation=triangular | 4 | 1 | 21828552.0 | 2845693.5 | 10.2% | NOISY |
| reshape_f64_concrete-shared_n33554432_to8192x4096 | #2029 | concrete-shared | tenferro-rs | f64 | operation=reshape | 4 | 1 | 37108009.0 | 2944089.0 | 4.0% | ok |
| reshape_f64_pytorch_n33554432_to8192x4096 | #2029 | pytorch | pytorch-cpu | f64 | operation=reshape | 4 | 1 | 37405399.0 | 1404091.5 | 3.2% | ok |
| metadata_transpose_view_f64_rank8_t4 | #2180 | metadata-clone | tenferro-rs | f64 | shape=[2, 2, 2, 2, 2, 2, 2, 2] | 4 | 8192 | 58.4 | 1.2 | 82.7% | NOISY |
| metadata_transpose_view_f64_rank8_t4_julia-base | #2180 | julia-base | julia-base | f64 | shape=[2, 2, 2, 2, 2, 2, 2, 2] | 4 | 1048576 | 0.2 | 0.0 | 9.6% | ok |
| followup_structural_broadcast_in_dim_direct-shared | #2021 | direct-shared | tenferro-rs | f64 | case_id=structural.broadcast_in_dim, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [8192, 1], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 39546943.0 | 254161.0 | 0.7% | ok |
| followup_structural_broadcast_in_dim_pytorch | #2021 | pytorch | pytorch-cpu | f64 | case_id=structural.broadcast_in_dim, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [8192, 1], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 29288433.0 | 1197795.5 | 3.3% | ok |
| followup_real_tanh_direct-shared | #2033 | direct-shared | tenferro-rs | f64 | case_id=real.tanh, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [8388608], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 11826199.0 | 39839.0 | 1.7% | ok |
| followup_real_tanh_jax | #2033 | jax | jax-cpu | f64 | case_id=real.tanh, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [8388608], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 16375175.0 | 391639.0 | 2.3% | ok |
| followup_real_log1p_direct-shared | #2033 | direct-shared | tenferro-rs | f64 | case_id=real.log1p, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [4194304], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 3409932.0 | 17931.5 | 0.3% | ok |
| followup_real_log1p_pytorch | #2033 | pytorch | pytorch-cpu | f64 | case_id=real.log1p, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [4194304], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 7620018.0 | 253690.0 | 2.3% | ok |
| followup_real_cos_direct-shared | #2033 | direct-shared | tenferro-rs | f64 | case_id=real.cos, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [8388608], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 5873495.0 | 7095.5 | 0.2% | ok |
| followup_real_cos_pytorch | #2033 | pytorch | pytorch-cpu | f64 | case_id=real.cos, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [8388608], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 18934089.0 | 427707.5 | 1.7% | ok |
| followup_real_sin_direct-shared | #2033 | direct-shared | tenferro-rs | f64 | case_id=real.sin, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [8388608], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 4992513.0 | 90098.0 | 2.6% | ok |
| followup_real_sin_pytorch | #2033 | pytorch | pytorch-cpu | f64 | case_id=real.sin, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [8388608], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 19108497.0 | 348085.0 | 4.7% | ok |
| followup_real_exp_direct-shared | #2033 | direct-shared | tenferro-rs | f64 | case_id=real.exp, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [8388608], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 3835952.0 | 9008.5 | 1.6% | ok |
| followup_real_exp_jax | #2033 | jax | jax-cpu | f64 | case_id=real.exp, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [8388608], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 15836467.0 | 477190.0 | 1.7% | ok |
| followup_real_expm1_direct-shared | #2033 | direct-shared | tenferro-rs | f64 | case_id=real.expm1, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [4194304], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 4626802.0 | 7310.5 | 1.0% | ok |
| followup_real_expm1_jax | #2033 | jax | jax-cpu | f64 | case_id=real.expm1, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [4194304], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 10364416.0 | 200557.5 | 2.0% | ok |
| followup_real_pow_direct-shared | #2033 | direct-shared | tenferro-rs | f64 | case_id=real.pow, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [4194304], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 19983118.0 | 201428.0 | 3.7% | ok |
| followup_real_pow_pytorch | #2033 | pytorch | pytorch-cpu | f64 | case_id=real.pow, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [4194304], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 12996956.0 | 203616.5 | 1.4% | ok |
| followup_complex_exp_direct-shared | #2033 | direct-shared | tenferro-rs | c64 | case_id=complex.exp, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [4194304], 'dtype': 'c64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 25230615.0 | 88165.0 | 0.3% | ok |
| followup_complex_exp_julia-base | #2033 | julia-base | julia-base | c64 | case_id=complex.exp, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [4194304], 'dtype': 'c64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 61840638.0 | 2098383.0 | 1.7% | ok |
| followup_real_log_direct-shared | #2033 | direct-shared | tenferro-rs | f64 | case_id=real.log, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [8388608], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 6859043.0 | 27511.0 | 3.8% | ok |
| followup_real_log_pytorch | #2033 | pytorch | pytorch-cpu | f64 | case_id=real.log, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [8388608], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 16755970.0 | 635304.0 | 15.5% | NOISY |
| followup_real_reduce_max_axis1_direct-shared | #2034 | direct-shared | tenferro-rs | f64 | case_id=real.reduce_max_axis1, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [2048, 2048], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 4 | 962407.8 | 45583.5 | 5.0% | ok |
| followup_real_reduce_max_axis1_julia-base | #2034 | julia-base | julia-base | f64 | case_id=real.reduce_max_axis1, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [2048, 2048], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 2 | 1053517.5 | 77907.5 | 4.5% | ok |
| followup_real_reduce_prod_axis1_direct-shared | #2034 | direct-shared | tenferro-rs | f64 | case_id=real.reduce_prod_axis1, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [2048, 2048], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 4 | 743306.8 | 49048.8 | 9.1% | ok |
| followup_real_reduce_prod_axis1_pytorch | #2034 | pytorch | pytorch-cpu | f64 | case_id=real.reduce_prod_axis1, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [2048, 2048], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 8 | 296676.2 | 4358.9 | 0.9% | ok |
| followup_real_reduce_sum_axis1_direct-shared | #2034 | direct-shared | tenferro-rs | f64 | case_id=real.reduce_sum_axis1, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [2048, 2048], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 4 | 709333.8 | 27352.9 | 2.3% | ok |
| followup_real_reduce_sum_axis1_pytorch | #2034 | pytorch | pytorch-cpu | f64 | case_id=real.reduce_sum_axis1, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [2048, 2048], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 8 | 318994.2 | 15188.8 | 2.5% | ok |
| followup_real_reduce_max_axis0_direct-shared | #2034 | direct-shared | tenferro-rs | f64 | case_id=real.reduce_max_axis0, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [2048, 2048], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 4 | 688891.2 | 3217.2 | 0.9% | ok |
| followup_real_reduce_max_axis0_julia-base | #2034 | julia-base | julia-base | f64 | case_id=real.reduce_max_axis0, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [2048, 2048], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 2 | 941627.5 | 50292.0 | 4.9% | ok |
| followup_real_reduce_min_axis1_direct-shared | #2034 | direct-shared | tenferro-rs | f64 | case_id=real.reduce_min_axis1, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [4096, 4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 4657157.0 | 167343.0 | 2.3% | ok |
| followup_real_reduce_min_axis1_julia-base | #2034 | julia-base | julia-base | f64 | case_id=real.reduce_min_axis1, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [4096, 4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 7642640.0 | 137292.5 | 1.7% | ok |
| followup_real_reduce_min_axis0_direct-shared | #2034 | direct-shared | tenferro-rs | f64 | case_id=real.reduce_min_axis0, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [2048, 2048], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 4 | 677118.5 | 6593.2 | 8.0% | ok |
| followup_real_reduce_min_axis0_julia-base | #2034 | julia-base | julia-base | f64 | case_id=real.reduce_min_axis0, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [2048, 2048], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 4 | 926846.5 | 6988.9 | 1.6% | ok |
| followup_real_reduce_max_all_direct-shared | #2034 | direct-shared | tenferro-rs | f64 | case_id=real.reduce_max_all, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [8192, 4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 7474192.0 | 181624.0 | 1.9% | ok |
| followup_real_reduce_max_all_julia-base | #2034 | julia-base | julia-base | f64 | case_id=real.reduce_max_all, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [8192, 4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 8159274.0 | 203430.5 | 2.3% | ok |
| followup_real_reduce_min_all_direct-shared | #2034 | direct-shared | tenferro-rs | f64 | case_id=real.reduce_min_all, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [8192, 4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 6931649.0 | 623787.5 | 4.8% | ok |
| followup_real_reduce_min_all_julia-base | #2034 | julia-base | julia-base | f64 | case_id=real.reduce_min_all, rust_path=direct-shared, reference_path=julia-base, fixture={'input_shape': [8192, 4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 8725272.0 | 1170666.5 | 27.2% | NOISY |
| followup_complex_norm_fro_direct-shared | #2035 | direct-shared | tenferro-rs | c64 | case_id=complex.norm_fro, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [2048, 1536], 'dtype': 'c64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 2 | 779823.0 | 107231.2 | 9.0% | ok |
| followup_complex_norm_fro_jax | #2035 | jax | jax-cpu | c64 | case_id=complex.norm_fro, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [2048, 1536], 'dtype': 'c64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 2814212.0 | 76002.5 | 5.8% | ok |
| followup_index_gather_direct-shared | #2036 #1719 | direct-shared | tenferro-rs | f64 | case_id=index.gather, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [262144], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 8 | 279883.9 | 4299.4 | 1.2% | ok |
| followup_index_gather_jax | #2036 #1719 | jax | jax-cpu | f64 | case_id=index.gather, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [262144], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 8 | 432467.0 | 3853.8 | 1.0% | ok |
| followup_index_scatter_direct-shared | #2036 #1719 | direct-shared | tenferro-rs | f64 | case_id=index.scatter, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [262144], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 4 | 825122.5 | 6336.9 | 0.8% | ok |
| followup_index_scatter_pytorch | #2036 #1719 | pytorch | pytorch-cpu | f64 | case_id=index.scatter, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [262144], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 4 | 774402.5 | 35585.0 | 4.5% | ok |
| followup_index_dynamic_update_slice_direct-shared | #2036 #1719 | direct-shared | tenferro-rs | f64 | case_id=index.dynamic_update_slice, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [2097152], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 2 | 781165.0 | 33259.0 | 5.0% | ok |
| followup_index_dynamic_update_slice_jax | #2036 #1719 | jax | jax-cpu | f64 | case_id=index.dynamic_update_slice, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [2097152], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 2 | 3024231.5 | 105926.8 | 8.5% | ok |
| followup_perm_reverse_15d_3_materialize | #2037 #2228 | materialize | tenferro-rs | f64 | case_id=perm.reverse_15d_3, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 55708512.0 | 1251580.0 | 13.0% | NOISY |
| followup_perm_reverse_15d_3_strided-rs | #2037 #2228 | strided-rs | strided-rs | f64 | case_id=perm.reverse_15d_3, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 47611320.0 | 1581574.0 | 2.1% | ok |
| followup_structural_transpose_direct-shared | #2037 #2228 | direct-shared | tenferro-rs | f64 | case_id=structural.transpose, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [4096, 4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 29994441.0 | 739894.0 | 2.5% | ok |
| followup_structural_transpose_pytorch | #2037 #2228 | pytorch | pytorch-cpu | f64 | case_id=structural.transpose, rust_path=direct-shared, reference_path=pytorch, fixture={'input_shape': [4096, 4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 75197250.0 | 2468415.0 | 7.6% | ok |
| followup_perm_reverse_23d_2_materialize | #2037 #2228 | materialize | tenferro-rs | f64 | case_id=perm.reverse_23d_2, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 19512733.0 | 807579.5 | 3.0% | ok |
| followup_perm_reverse_23d_2_strided-rs | #2037 #2228 | strided-rs | strided-rs | f64 | case_id=perm.reverse_23d_2, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 36116754.0 | 1125434.0 | 4.2% | ok |
| followup_perm_transpose_3d_256_201_materialize | #2037 #2228 | materialize | tenferro-rs | f64 | case_id=perm.transpose_3d_256_201, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [256, 256, 256], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 28497439.0 | 385044.5 | 1.7% | ok |
| followup_perm_transpose_3d_256_201_strided-rs | #2037 #2228 | strided-rs | strided-rs | f64 | case_id=perm.transpose_3d_256_201, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [256, 256, 256], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 28260979.0 | 696375.5 | 2.2% | ok |
| followup_perm_cyclic_15d_3_materialize | #2037 #2228 | materialize | tenferro-rs | f64 | case_id=perm.cyclic_15d_3, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 19248943.0 | 234110.5 | 2.6% | ok |
| followup_perm_cyclic_15d_3_strided-rs | #2037 #2228 | strided-rs | strided-rs | f64 | case_id=perm.cyclic_15d_3, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 19050079.0 | 343828.5 | 1.4% | ok |
| followup_perm_transpose_3d_256_102_materialize | #2037 #2228 | materialize | tenferro-rs | f64 | case_id=perm.transpose_3d_256_102, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [256, 256, 256], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 28190685.0 | 377278.0 | 1.2% | ok |
| followup_perm_transpose_3d_256_102_strided-rs | #2037 #2228 | strided-rs | strided-rs | f64 | case_id=perm.transpose_3d_256_102, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [256, 256, 256], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 27086362.0 | 661720.0 | 1.5% | ok |
| followup_perm_tn_light_415_24d_scattered_to_colmajor_materialize | #2037 #2228 | materialize | tenferro-rs | f64 | case_id=perm.tn_light_415_24d_scattered_to_colmajor, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 23495571.0 | 401398.5 | 1.2% | ok |
| followup_perm_tn_light_415_24d_scattered_to_colmajor_strided-rs | #2037 #2228 | strided-rs | strided-rs | f64 | case_id=perm.tn_light_415_24d_scattered_to_colmajor, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 23879208.0 | 823142.5 | 2.0% | ok |
| followup_perm_tn_light_415_24d_contiguous_same_perm_materialize | #2037 #2228 | materialize | tenferro-rs | f64 | case_id=perm.tn_light_415_24d_contiguous_same_perm, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 21660918.0 | 369075.5 | 4.6% | ok |
| followup_perm_tn_light_415_24d_contiguous_same_perm_strided-rs | #2037 #2228 | strided-rs | strided-rs | f64 | case_id=perm.tn_light_415_24d_contiguous_same_perm, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 20703922.0 | 723981.0 | 2.3% | ok |
| followup_perm_rotation_6d_32_32_32_32_16_16_materialize | #2037 #2228 | materialize | tenferro-rs | f64 | case_id=perm.rotation_6d_32_32_32_32_16_16, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [32, 32, 32, 32, 16, 16], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 468799753.0 | 2804422.0 | 0.5% | ok |
| followup_perm_rotation_6d_32_32_32_32_16_16_strided-rs | #2037 #2228 | strided-rs | strided-rs | f64 | case_id=perm.rotation_6d_32_32_32_32_16_16, rust_path=materialize, reference_path=strided-rs, fixture={'input_shape': [32, 32, 32, 32, 16, 16], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 484198570.0 | 8510147.0 | 3.8% | ok |
| followup_structural_extract_diagonal_direct-shared | #2038 #2228 | direct-shared | tenferro-rs | f64 | case_id=structural.extract_diagonal, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [8388608, 2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 36445889.0 | 955192.5 | 1.7% | ok |
| followup_structural_extract_diagonal_jax | #2038 #2228 | jax | jax-cpu | f64 | case_id=structural.extract_diagonal, rust_path=direct-shared, reference_path=jax, fixture={'input_shape': [8388608, 2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 34816188.0 | 845130.0 | 1.7% | ok |
| followup_fft_irfft_direct-shared | #2039 #2230 #2228 | direct-shared | tenferro-rs | c32 | case_id=fft.irfft, rust_path=direct-shared, reference_path=mkl-dfti, fixture={'input_shape': [524289], 'dtype': 'c32', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 3217539.0 | 40720.0 | 1.2% | ok |
| followup_fft_irfft_mkl-dfti | #2039 #2230 #2228 | mkl-dfti | mkl-dfti | c32 | case_id=fft.irfft, rust_path=direct-shared, reference_path=mkl-dfti, fixture={'input_shape': [524289], 'dtype': 'c32', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 2048540.0 | 10231.0 | 0.7% | ok |
| followup_fft_rfft_direct-shared | #2039 #2230 #2228 | direct-shared | tenferro-rs | f32 | case_id=fft.rfft, rust_path=direct-shared, reference_path=mkl-dfti, fixture={'input_shape': [1048576], 'dtype': 'f32', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 2966076.0 | 65056.0 | 2.1% | ok |
| followup_fft_rfft_mkl-dfti | #2039 #2230 #2228 | mkl-dfti | mkl-dfti | f32 | case_id=fft.rfft, rust_path=direct-shared, reference_path=mkl-dfti, fixture={'input_shape': [1048576], 'dtype': 'f32', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 2012035.0 | 73914.0 | 3.7% | ok |
| followup_fft_fft_direct-shared | #2039 #2228 | direct-shared | tenferro-rs | c64 | case_id=fft.fft, rust_path=direct-shared, reference_path=mkl-dfti, fixture={'input_shape': [1048576], 'dtype': 'c64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 11238609.0 | 295688.0 | 2.3% | ok |
| followup_fft_fft_mkl-dfti | #2039 #2228 | mkl-dfti | mkl-dfti | c64 | case_id=fft.fft, rust_path=direct-shared, reference_path=mkl-dfti, fixture={'input_shape': [1048576], 'dtype': 'c64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 11043601.0 | 381429.0 | 3.1% | ok |
| followup_fft_ifft_direct-shared | #2039 #2228 | direct-shared | tenferro-rs | c32 | case_id=fft.ifft, rust_path=direct-shared, reference_path=mkl-dfti, fixture={'input_shape': [1048576], 'dtype': 'c32', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 5617116.0 | 58802.0 | 1.1% | ok |
| followup_fft_ifft_mkl-dfti | #2039 #2228 | mkl-dfti | mkl-dfti | c32 | case_id=fft.ifft, rust_path=direct-shared, reference_path=mkl-dfti, fixture={'input_shape': [1048576], 'dtype': 'c32', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout'} | 4 | 1 | 3493495.0 | 135950.5 | 7.8% | ok |
| followup_metadata_transpose_view_metadata | #2040 #1719 | metadata | tenferro-rs | f64 | case_id=metadata.transpose_view, rust_path=metadata, reference_path=julia-base, fixture={'input_shape': [2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 4 | 8192 | 305.6 | 10.4 | 4.0% | ok |
| followup_metadata_transpose_view_julia-base | #2040 #1719 | julia-base | julia-base | f64 | case_id=metadata.transpose_view, rust_path=metadata, reference_path=julia-base, fixture={'input_shape': [2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 4 | 1864135 | 0.6 | 0.0 | 4.4% | ok |
| followup_metadata_transpose_view_metadata-host-dyn | #2040 #1719 | metadata-host-dyn | tenferro-rs | f64 | case_id=metadata.transpose_view, rust_path=metadata-host-dyn, reference_path=julia-base, fixture={'input_shape': [2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 4 | 8192 | 374.0 | 10.6 | 2.4% | ok |
| followup_metadata_transpose_view_metadata-host-static | #2040 #1719 | metadata-host-static | tenferro-rs | f64 | case_id=metadata.transpose_view, rust_path=metadata-host-static, reference_path=julia-base, fixture={'input_shape': [2, 2], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 4 | 8192 | 308.9 | 5.7 | 2.1% | ok |
| followup_metadata_slice_view_metadata | #2040 #1719 | metadata | tenferro-rs | f64 | case_id=metadata.slice_view, rust_path=metadata, reference_path=julia-base, fixture={'input_shape': [4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 4 | 8192 | 334.4 | 2.8 | 3.1% | ok |
| followup_metadata_slice_view_julia-base | #2040 #1719 | julia-base | julia-base | f64 | case_id=metadata.slice_view, rust_path=metadata, reference_path=julia-base, fixture={'input_shape': [4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 4 | 16256 | 3.7 | 1.3 | 28.3% | NOISY |
| followup_metadata_slice_view_metadata-host-dyn | #2040 #1719 | metadata-host-dyn | tenferro-rs | f64 | case_id=metadata.slice_view, rust_path=metadata-host-dyn, reference_path=julia-base, fixture={'input_shape': [4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 4 | 8192 | 442.1 | 14.8 | 3.5% | ok |
| followup_metadata_slice_view_metadata-host-static | #2040 #1719 | metadata-host-static | tenferro-rs | f64 | case_id=metadata.slice_view, rust_path=metadata-host-static, reference_path=julia-base, fixture={'input_shape': [4096], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 4 | 8192 | 338.5 | 6.1 | 3.6% | ok |
| followup_metadata_reshape_view_metadata | #2040 #1719 | metadata | tenferro-rs | f64 | case_id=metadata.reshape_view, rust_path=metadata, reference_path=julia-base, fixture={'input_shape': [1024], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 4 | 8192 | 446.3 | 1.7 | 0.4% | ok |
| followup_metadata_reshape_view_julia-base | #2040 #1719 | julia-base | julia-base | f64 | case_id=metadata.reshape_view, rust_path=metadata, reference_path=julia-base, fixture={'input_shape': [1024], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 4 | 63550 | 9.9 | 1.2 | 12.3% | NOISY |
| followup_metadata_reshape_view_metadata-host-dyn | #2040 #1719 | metadata-host-dyn | tenferro-rs | f64 | case_id=metadata.reshape_view, rust_path=metadata-host-dyn, reference_path=julia-base, fixture={'input_shape': [1024], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 4 | 8192 | 484.1 | 15.2 | 8.3% | ok |
| followup_metadata_reshape_view_metadata-host-static | #2040 #1719 | metadata-host-static | tenferro-rs | f64 | case_id=metadata.reshape_view, rust_path=metadata-host-static, reference_path=julia-base, fixture={'input_shape': [1024], 'dtype': 'f64', 'layout': 'native column-major for Rust/Julia; equivalent logical values in Python native layout', 'extent_policy': 'smaller pure-metadata repro; does not confirm original large-input timings'} | 4 | 8192 | 456.5 | 10.0 | 1.9% | ok |
| lu_batched_f64_factor_n8_b1024 | #2142 | tenferro | tenferro-rs | f64 | n=8, batch=1024, op=factor | 4 | 8 | 272969.9 | 1947.9 | 0.7% | ok |
| lu_batched_f64_factor_n8_b1024_host | #2142 | host | rust-loop | f64 | n=8, batch=1024, op=factor | 4 | 8 | 487914.0 | 283743.9 | 29.2% | NOISY |
| lu_batched_f64_solve_n8_b1024 | #2142 | tenferro | tenferro-rs | f64 | n=8, batch=1024, op=solve | 4 | 16 | 165798.2 | 693.2 | 0.4% | ok |
| lu_batched_f64_solve_n8_b1024_pytorch | #2142 | pytorch | pytorch-cpu | f64 | n=8, batch=1024, op=solve | 4 | 16 | 139354.8 | 608.0 | 3.9% | ok |
| lu_batched_f64_factor_n16_b1024 | #2142 | tenferro | tenferro-rs | f64 | n=16, batch=1024, op=factor | 4 | 2 | 817403.5 | 3668.2 | 5.0% | ok |
| lu_batched_f64_factor_n16_b1024_host | #2142 | host | rust-loop | f64 | n=16, batch=1024, op=factor | 4 | 2 | 1800317.0 | 25931.0 | 1.3% | ok |
| lu_batched_f64_solve_n16_b1024 | #2142 | tenferro | tenferro-rs | f64 | n=16, batch=1024, op=solve | 4 | 8 | 249612.9 | 1336.4 | 1.8% | ok |
| lu_batched_f64_solve_n16_b1024_pytorch | #2142 | pytorch | pytorch-cpu | f64 | n=16, batch=1024, op=solve | 4 | 8 | 298874.5 | 1083.4 | 0.3% | ok |
| small_grad_qr_f64_trace-prepared_n4 | #2186 | trace-prepared | tenferro-rs | f64 | op=qr, n=4 | 4 | 64 | 52603.3 | 357.6 | 0.6% | ok |
| small_grad_qr_f64_pytorch_n4 | #2186 | pytorch | pytorch-cpu | f64 | op=qr, n=4 | 4 | 32 | 109557.0 | 964.2 | 9.2% | ok |
| small_grad_qr_f64_trace-prepared_n16 | #2186 | trace-prepared | tenferro-rs | f64 | op=qr, n=16 | 4 | 32 | 61250.6 | 335.2 | 1.0% | ok |
| small_grad_qr_f64_pytorch_n16 | #2186 | pytorch | pytorch-cpu | f64 | op=qr, n=16 | 4 | 32 | 119046.0 | 1517.7 | 1.8% | ok |
| small_grad_eigh_f64_trace-prepared_n4 | #2186 | trace-prepared | tenferro-rs | f64 | op=eigh, n=4 | 4 | 32 | 68019.7 | 640.8 | 0.8% | ok |
| small_grad_eigh_f64_pytorch_n4 | #2186 | pytorch | pytorch-cpu | f64 | op=eigh, n=4 | 4 | 32 | 85504.2 | 1946.5 | 2.1% | ok |
| small_grad_eigh_f64_trace-prepared_n16 | #2186 | trace-prepared | tenferro-rs | f64 | op=eigh, n=16 | 4 | 32 | 100003.2 | 844.8 | 0.6% | ok |
| small_grad_eigh_f64_pytorch_n16 | #2186 | pytorch | pytorch-cpu | f64 | op=eigh, n=16 | 4 | 32 | 106931.6 | 903.2 | 0.6% | ok |
| small_grad_lu_f64_trace-prepared_n4 | #2186 | trace-prepared | tenferro-rs | f64 | op=lu, n=4 | 4 | 64 | 32293.9 | 116.2 | 0.6% | ok |
| small_grad_lu_f64_pytorch_n4 | #2186 | pytorch | pytorch-cpu | f64 | op=lu, n=4 | 4 | 32 | 122113.7 | 989.1 | 0.9% | ok |
| small_grad_lu_f64_trace-prepared_n16 | #2186 | trace-prepared | tenferro-rs | f64 | op=lu, n=16 | 4 | 64 | 39851.6 | 298.8 | 0.5% | ok |
| small_grad_lu_f64_pytorch_n16 | #2186 | pytorch | pytorch-cpu | f64 | op=lu, n=16 | 4 | 16 | 131209.7 | 1332.9 | 1.2% | ok |
| small_grad_svd_f64_trace-prepared_n4 | #2186 | trace-prepared | tenferro-rs | f64 | op=svd, n=4 | 4 | 32 | 108844.4 | 1742.4 | 1.7% | ok |
| small_grad_svd_f64_pytorch_n4 | #2186 | pytorch | pytorch-cpu | f64 | op=svd, n=4 | 4 | 16 | 122150.7 | 3036.4 | 1.3% | ok |
| small_grad_svd_f64_trace-prepared_n16 | #2186 | trace-prepared | tenferro-rs | f64 | op=svd, n=16 | 4 | 16 | 174281.5 | 1872.8 | 0.7% | ok |
| small_grad_svd_f64_pytorch_n16 | #2186 | pytorch | pytorch-cpu | f64 | op=svd, n=16 | 4 | 16 | 160477.9 | 1548.2 | 0.9% | ok |
| small_grad_solve_f64_trace-prepared_n4 | #2186 | trace-prepared | tenferro-rs | f64 | op=solve, n=4 | 4 | 128 | 26496.5 | 81.4 | 0.3% | ok |
| small_grad_solve_f64_pytorch_n4 | #2186 | pytorch | pytorch-cpu | f64 | op=solve, n=4 | 4 | 32 | 61876.2 | 1678.6 | 1.5% | ok |
| small_grad_solve_f64_trace-prepared_n16 | #2186 | trace-prepared | tenferro-rs | f64 | op=solve, n=16 | 4 | 64 | 35095.8 | 344.7 | 2.0% | ok |
| small_grad_solve_f64_pytorch_n16 | #2186 | pytorch | pytorch-cpu | f64 | op=solve, n=16 | 4 | 32 | 72357.6 | 1218.8 | 1.3% | ok |
| fft_axis0_c64_fft-executor_n128 | #1765 | fft-executor | tenferro-rs | c64 | n=128 | 4 | 1 | 6790361.0 | 109983.0 | 2.5% | ok |
| fft_axis0_c64_pytorch_n128 | #1765 | pytorch | pytorch-cpu | c64 | n=128 | 4 | 1 | 8753104.0 | 156868.0 | 2.0% | ok |
| degenerate_dot_hadamard_f64_dot-general-shared_n1024 | #2111 | dot-general-shared | tenferro-rs | f64 | form=hadamard, n=1024 | 4 | 4 | 251066.2 | 26312.6 | 7.3% | ok |
| degenerate_dot_hadamard_f64_pytorch_n1024 | #2111 | pytorch | pytorch-cpu | f64 | form=hadamard, n=1024 | 4 | 4 | 253209.5 | 38120.1 | 9.4% | ok |
| degenerate_dot_outer_f64_dot-general-shared_n1024 | #2111 | dot-general-shared | tenferro-rs | f64 | form=outer, n=1024 | 4 | 2 | 374073.0 | 110097.8 | 20.7% | NOISY |
| degenerate_dot_outer_f64_pytorch_n1024 | #2111 | pytorch | pytorch-cpu | f64 | form=outer, n=1024 | 4 | 2 | 59122.5 | 8512.8 | 19.3% | NOISY |
| degenerate_dot_rank0_f64_dot-general-shared_n1024 | #2111 | dot-general-shared | tenferro-rs | f64 | form=rank0, n=1024 | 4 | 2 | 906751.0 | 16744.8 | 1.6% | ok |
| degenerate_dot_rank0_f64_pytorch_n1024 | #2111 | pytorch | pytorch-cpu | f64 | form=rank0, n=1024 | 4 | 4 | 166135.8 | 20013.2 | 16.3% | NOISY |
| erased_add_f64_session-add_n256 | #1766 | session-add | tenferro-rs | f64 | n=256 | 4 | 1024 | 1435.9 | 33.8 | 12.6% | NOISY |
| erased_add_f64_pytorch_n256 | #1766 | pytorch | pytorch-cpu | f64 | n=256 | 4 | 2048 | 1253.4 | 37.0 | 5.0% | ok |
| erased_add_f64_session-add_n4096 | #1766 | session-add | tenferro-rs | f64 | n=4096 | 4 | 256 | 2534.8 | 304.3 | 11.3% | NOISY |
| erased_add_f64_pytorch_n4096 | #1766 | pytorch | pytorch-cpu | f64 | n=4096 | 4 | 1024 | 3211.2 | 62.5 | 1.2% | ok |
| erased_add_f64_session-add_n1048576 | #1766 | session-add | tenferro-rs | f64 | n=1048576 | 4 | 4 | 276115.8 | 14700.4 | 5.4% | ok |
| erased_add_f64_pytorch_n1048576 | #1766 | pytorch | pytorch-cpu | f64 | n=1048576 | 4 | 4 | 232924.5 | 22192.9 | 9.8% | ok |
| axpby_strided_source_f64_1024x1024_t4 | #2220 #2214 | strided-source | tenferro-rs | f64 | n=1024 | 4 | 4 | 1235972.8 | 104464.5 | 5.7% | ok |
| axpby_contiguous_source_f64_1024x1024_t4 | #2220 #2214 | contiguous-source | tenferro-rs | f64 | n=1024 | 4 | 8 | 313869.1 | 5780.4 | 1.8% | ok |
| axpby_strided_source_f64_1024x1024_host_t4 | #2220 #2214 | host | rust-loop | f64 | n=1024 | 4 | 2 | 1538895.5 | 38592.0 | 2.6% | ok |
| erased_construct_f64_col_n8 | #2177 | col-major | tenferro-rs | f64 | n=8, layout=col | 4 | 16384 | 110.8 | 1.3 | 25.3% | NOISY |
| erased_construct_f64_row_n8 | #2177 | row-major | tenferro-rs | f64 | n=8, layout=row | 4 | 8192 | 229.3 | 4.7 | 15.3% | NOISY |
| erased_construct_f64_col_n4096 | #2177 | col-major | tenferro-rs | f64 | n=4096, layout=col | 4 | 16384 | 223.3 | 5.8 | 1.9% | ok |
| erased_construct_f64_row_n4096 | #2177 | row-major | tenferro-rs | f64 | n=4096, layout=row | 4 | 256 | 13054.7 | 109.9 | 0.9% | ok |
| erased_construct_f64_col_n65536 | #2177 | col-major | tenferro-rs | f64 | n=65536, layout=col | 4 | 1024 | 115.9 | 2.4 | 3.3% | ok |
| erased_construct_f64_row_n65536 | #2177 | row-major | tenferro-rs | f64 | n=65536, layout=row | 4 | 16 | 210654.1 | 11472.6 | 4.2% | ok |
| eager_small_leaf_f64_eager-held-session | #1704 | eager-held-session | tenferro-rs | f64 | op=leaf | 4 | 1024 | 2198.0 | 45.2 | 3.0% | ok |
| eager_small_add_f64_eager-held-session | #1667 | eager-held-session | tenferro-rs | f64 | op=add | 4 | 1024 | 2469.2 | 31.2 | 3.1% | ok |
| eager_small_add_f64_pytorch | #1667 | pytorch | pytorch-cpu | f64 | op=add | 4 | 2048 | 1182.0 | 9.3 | 0.7% | ok |
| eager_small_gemm_f64_eager-held-session | #1904 | eager-held-session | tenferro-rs | f64 | op=gemm | 4 | 1024 | 3362.1 | 7.2 | 0.3% | ok |
| eager_small_gemm_f64_pytorch | #1904 | pytorch | pytorch-cpu | f64 | op=gemm | 4 | 2048 | 1348.6 | 11.8 | 0.9% | ok |
| trace_metadata_trace-fold_f32_1024x64_k50 | #2226 #2238 | trace-fold | tenferro-rs | f32 | rows=1024, cols=64, steps=50 | 4 | 2 | 1994444.5 | 2105.8 | 0.1% | ok |
| trace_metadata_trace-outputs_f32_1024x64_k50 | #2226 #2238 | trace-outputs | tenferro-rs | f32 | rows=1024, cols=64, steps=50 | 4 | 2 | 2016567.0 | 47902.8 | 1.8% | ok |
| trace_metadata_eager_f32_1024x64_k50 | #2226 #2238 | eager | tenferro-rs | f32 | rows=1024, cols=64, steps=50 | 4 | 1 | 3139102.0 | 11927.5 | 1.3% | ok |
| trace_metadata_pytorch_f32_1024x64_k50 | #2226 #2238 | pytorch | pytorch-cpu | f32 | rows=1024, cols=64, steps=50 | 4 | 2 | 1699868.5 | 6805.0 | 1.7% | ok |
| trace_metadata_fixed_neg_f32_8 | #2226 #2238 | trace-fold | tenferro-rs | f32 | rows=8, cols=1, steps=0 | 4 | 1024 | 2650.9 | 11.2 | 2.4% | ok |


## Timing boundaries

- **activation / eager-composed** — inside: hand-written formula from pre-B1 eager primitives, intermediate and output allocation; outside: backend/runtime construction, input, constant, mask and index construction, eager session entry, correctness_check.
- **activation / eager-single-call** — inside: declared API call, intrinsic output allocation, native completion; outside: fixtures, runtime/backend/session creation, input wrappers/descriptors, graph construction/compilation/preparation, FFT plans/scratch priming, validation/signatures, output retention allocation, input restoration, output destruction.
- **activation / eager-single-call** — inside: one activation call, intermediate and output allocation; outside: backend/runtime construction, input, constant, mask and index construction, eager session entry, correctness_check.
- **activation / pytorch** — inside: API execution, intrinsic output allocation, native completion; outside: fixtures, conversion, JIT compilation, initialization, validation, retention allocation, output destruction.
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
- **cpu_followup_mwe / direct-shared** — inside: declared API call, intrinsic output allocation, native completion; outside: fixtures, runtime/backend/session creation, input wrappers/descriptors, graph construction/compilation/preparation, FFT plans/scratch priming, validation/signatures, output retention allocation, input restoration, output destruction.
- **cpu_followup_mwe / jax** — inside: API execution, intrinsic output allocation, native completion; outside: fixtures, conversion, JIT compilation, initialization, validation, retention allocation, output destruction.
- **cpu_followup_mwe / materialize** — inside: declared API call, intrinsic output allocation, native completion; outside: fixtures, runtime/backend/session creation, input wrappers/descriptors, graph construction/compilation/preparation, FFT plans/scratch priming, validation/signatures, output retention allocation, input restoration, output destruction.
- **cpu_followup_mwe / metadata** — inside: declared API call, intrinsic output allocation, native completion; outside: fixtures, runtime/backend/session creation, input wrappers/descriptors, graph construction/compilation/preparation, FFT plans/scratch priming, validation/signatures, output retention allocation, input restoration, output destruction.
- **cpu_followup_mwe / metadata-host-dyn** — inside: declared API call, intrinsic output allocation, native completion; outside: fixtures, runtime/backend/session creation, input wrappers/descriptors, graph construction/compilation/preparation, FFT plans/scratch priming, validation/signatures, output retention allocation, input restoration, output destruction.
- **cpu_followup_mwe / metadata-host-static** — inside: declared API call, intrinsic output allocation, native completion; outside: fixtures, runtime/backend/session creation, input wrappers/descriptors, graph construction/compilation/preparation, FFT plans/scratch priming, validation/signatures, output retention allocation, input restoration, output destruction.
- **cpu_followup_mwe / mkl-dfti** — inside: API execution, intrinsic output allocation, native completion; outside: fixtures, conversion, JIT compilation, initialization, validation, retention allocation, output destruction.
- **cpu_followup_mwe / pytorch** — inside: API execution, intrinsic output allocation, native completion; outside: fixtures, conversion, JIT compilation, initialization, validation, retention allocation, output destruction.
- **cpu_followup_mwe / strided-rs** — inside: declared API call, intrinsic output allocation, native completion; outside: fixtures, runtime/backend/session creation, input wrappers/descriptors, graph construction/compilation/preparation, FFT plans/scratch priming, validation/signatures, output retention allocation, input restoration, output destruction.
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
