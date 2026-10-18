# CPU Overhead Smoke (ns/op and allocs/op)

- Suite: `cpu/overhead_smoke`
- Target profile: `amd-cpu`
- Raw run: `data/results/amd-cpu/cpu/overhead_smoke/20261017_075853`
- tenferro-rs commit: `5e53f9bff263b5557c185bc2b3400cfa978a26a8`
- Collection command: `BENCHMARK_TARGET_PROFILE=amd-cpu TENFERRO_CPU_FEATURES=blas-mkl ./scripts/run_overhead_smoke.sh `

Trend view for tenferro-rs #2178 (tenferro-benchmark #197); nothing here gates. Every row runs on one run thread; the prepared-runtime diagnostics set the CPU backend thread budget explicitly. **allocs/op** counts process-wide Rust allocator events over 64 calls after pool warmup, dropping outputs between calls. These observed counts include Rust workers and exclude foreign BLAS/C allocations; compare matching fixtures, thread budgets and execution scopes. **ns/op** is the median of retained-output batches, with destruction after the clock stops. Shared nightly runner timings are reference indicators only.

## Steady-state operations

Inputs, entered reusable sessions, initialization, tracing and preparation are outside timing. Only the declared operation and intrinsic output allocation or pooled checkout are timed.

| Case | Workload | Budget | Median ns/op | IQR ns | CoV | allocs/op | bytes/op | Refs |
|---|---|---:|---:|---:|---:|---:|---:|---|
| typed_elemwise_f64_add_bcast_n16 | typed add, rank-0 scalar rhs (stride-0 operand) | 1 | 2804.0 | 13.8 | 1.5% | 16.00 | 328 | tenferro-rs#2178 |
| typed_elemwise_f64_add_bcast_8x8x8 | typed add, rank-0 scalar rhs (stride-0 operand) | 1 | 3289.1 | 11.7 | 1.8% | 16.00 | 392 | tenferro-rs#2178 |
| typed_elemwise_f64_add_permuted_view_8x8x8 | add_read, lhs a permuted view (perm [2,0,1]) | 1 | 3894.6 | 25.3 | 1.5% | 16.00 | 440 | tenferro-rs#2178 |
| permute_materialize_f64_r4_4x4x4x4 | materializing rank-4 transpose (perm [3,1,0,2]) | 1 | 3974.7 | 1064.9 | 15.3% NOISY | 19.03 | 1440 | tenferro-rs#2178 |
| reduce_axis_f64_8x8x8 | typed reduce_sum over axis 1 | 1 | 4134.8 | 1022.1 | 15.1% NOISY | 15.00 | 545 | tenferro-rs#2178 |
| concatenate_axis1_f64_8x8x8 | concatenate two 8x8x8 tensors along axis 1 | 1 | 2744.7 | 121.9 | 5.8% | 7.00 | 1232 | tenferro-rs#2178 |
| einsum_prepared_f64_n2 | prepared ij,jk->ik execute_into a caller output (overlaps cpu/small_work) | 1 | 2496.2 | 640.6 | 15.4% NOISY | 0.00 | 0 | tenferro-rs#2178 tenferro-rs#1899 |
| einsum_prepared_f64_n4 | prepared ij,jk->ik execute_into a caller output (overlaps cpu/small_work) | 1 | 1775.8 | 127.2 | 4.6% | 0.00 | 0 | tenferro-rs#2178 tenferro-rs#1899 |

## Runtime session-entry diagnostics

These rows time public `Runtime::run_prepared`, including its internal session entry and exit. Runtime construction, inputs, tracing and preparation remain outside timing. They are **diagnostic public-path latency**, not shared-session operation-performance evidence. Their allocation counters also include internal session entry. They are intentionally separate from the steady-state table.

This raw run predates explicit diagnostic case IDs and per-case timing metadata. Its `run_prepared` rows are reclassified here from their recorded fixtures; the original raw IDs, samples and metadata are unchanged. The old blanket claim that session entry was outside timing does not apply to these rows.

| Case | Workload | Budget | Median ns/op | IQR ns | CoV | allocs/op | bytes/op | Refs |
|---|---|---:|---:|---:|---:|---:|---:|---|
| fused_elemwise_f64_budget1_32x32x32 (session-entry diagnostic; legacy raw ID) | traced a*b+c via run_prepared, backend thread budget 1 | 1 | 110995.9 | 278.8 | 0.2% | 29.00 | 27352 | tenferro-rs#2178 strided-rs#305 |
| fused_elemwise_f64_budget2_32x32x32 (session-entry diagnostic; legacy raw ID) | traced a*b+c via run_prepared, backend thread budget 2 | 2 | 124595.6 | 2594.9 | 1.4% | 29.03 | 27400 | tenferro-rs#2178 strided-rs#305 |

Numerical checks: **10/10 rows passed** against independent host loops before counting and sampling.

## Prepared-runtime region execution (diagnostics)

Whether the runtime executed the traced region as one fused command or declined to per-op dispatch (cumulative over validation, the allocator pass and timed batches). `run_prepared` enters a runtime-internal session per call, and its entry is included in both diagnostic ns/op and allocs/op.

| Case | Planned regions | Fused executions | Fallbacks |
|---|---:|---:|---:|
| fused_elemwise_f64_budget1_32x32x32 | 1 | 0 | 344 |
| fused_elemwise_f64_budget2_32x32x32 | 1 | 0 | 344 |

## CPU Information

- Model: `Intel(R) Xeon(R) CPU E5-2699 v3 @ 2.30GHz`
- Vendor: `GenuineIntel`
- Logical CPUs: `36`
- Sockets: `1`
- Cores per socket: `18`
- Threads per core: `2`
- NUMA nodes: `1`
- Python platform: `Linux-6.8.0-146-generic-x86_64-with-glibc2.39`

## Run metadata

Full metadata: `data/results/amd-cpu/cpu/overhead_smoke/20261017_075853/run.yaml`

```yaml
timestamp: '2026-10-17T07:58:53Z'
tenferro_rs:
  path: /workspaces/tenferro-benchmark/extern/tenferro-rs
  commit: 5e53f9bff263b5557c185bc2b3400cfa978a26a8
  dirty: false
  features:
  - blas-mkl
  resolved_path: /workspaces/tenferro-benchmark/extern/tenferro-rs
blas:
  implementation: mkl
  version: 2026.0.1
  root: /opt/intel/oneapi/mkl/latest
  library: /opt/intel/oneapi/mkl/latest/lib/libmkl_rt.so
environment:
  hostname: 8c4fc82a23a6
  os: Linux-6.8.0-146-generic-x86_64-with-glibc2.39
  arch: x86_64
  cpu: x86_64
  env:
    OMP_NUM_THREADS: '1'
    OMP_THREAD_LIMIT: '1'
    OMP_DYNAMIC: 'FALSE'
    RAYON_NUM_THREADS: '1'
    BENCHMARK_TARGET_PROFILE: amd-cpu
    BENCHMARK_COMMIT: fdf7f60bf9c1f20303b66a81115a53dbf4d77960
    OPENBLAS_ROOT: /opt/openblas
    MKLROOT: /opt/intel/oneapi/mkl/latest
    OPENBLAS_NUM_THREADS: '1'
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
    USE_CUDA: '0'
    BENCHMARK_BUILD_PROFILE: release
    RUSTC_VERSION: rustc 1.99.0 (b940084d7 2026-09-28)
    BENCH_INSTANCE: null
collection:
  command: 'BENCHMARK_TARGET_PROFILE=amd-cpu TENFERRO_CPU_FEATURES=blas-mkl ./scripts/run_overhead_smoke.sh '
  timing_scope: steady_state_host_api; setup, session entry, tracing, compilation
    and preparation outside timing; outputs retained until the clock stops
  allocation_scope: 'separate pooled pass: 64 calls after warmup, outputs dropped
    between calls'
  gating: none (trend view)
```
