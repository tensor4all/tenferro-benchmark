# CPU Overhead Smoke (ns/op and allocs/op)

- Suite: `cpu/overhead_smoke`
- Target profile: `mac-cpu`
- Raw run: `data/results/mac-cpu/cpu/overhead_smoke/20261022_033042`
- tenferro-rs commit: `d5ef90538a4c05810fe63bc1b5713d871ac98279`
- Collection command: `BENCHMARK_TARGET_PROFILE=mac-cpu TENFERRO_CPU_FEATURES=blas-accelerate /Users/runner/work/tenferro-benchmark/tenferro-benchmark/scripts/run_overhead_smoke.sh `

Trend view for tenferro-rs #2178 (tenferro-benchmark #197); nothing here gates. Every row runs on one run thread; the prepared-runtime diagnostics set the CPU backend thread budget explicitly. **allocs/op** counts process-wide Rust allocator events over 64 calls after pool warmup, dropping outputs between calls. These observed counts include Rust workers and exclude foreign BLAS/C allocations; compare matching fixtures, thread budgets and execution scopes. **ns/op** is the median of retained-output batches, with destruction after the clock stops. Shared nightly runner timings are reference indicators only.

## Steady-state operations

Inputs, entered reusable sessions, initialization, tracing and preparation are outside timing. Only the declared operation and intrinsic output allocation or pooled checkout are timed.

| Case | Workload | Budget | Median ns/op | IQR ns | CoV | allocs/op | bytes/op | Refs |
|---|---|---:|---:|---:|---:|---:|---:|---|
| typed_elemwise_f64_add_bcast_n16 | typed add, rank-0 scalar rhs (stride-0 operand) | 1 | 1118.3 | 134.6 | 9.1% | 0.00 | 0 | tenferro-rs#2178 |
| typed_elemwise_f64_add_bcast_8x8x8 | typed add, rank-0 scalar rhs (stride-0 operand) | 1 | 1331.3 | 81.8 | 4.4% | 0.00 | 0 | tenferro-rs#2178 |
| typed_elemwise_f64_add_permuted_view_8x8x8 | add_read, lhs a permuted view (perm [2,0,1]) | 1 | 1442.5 | 89.8 | 4.2% | 0.00 | 0 | tenferro-rs#2178 |
| permute_materialize_f64_r4_4x4x4x4 | materializing rank-4 transpose (perm [3,1,0,2]) | 1 | 1441.8 | 76.2 | 4.7% | 0.00 | 0 | tenferro-rs#2178 |
| reduce_axis_f64_8x8x8 | typed reduce_sum over axis 1 | 1 | 1770.3 | 77.0 | 3.5% | 0.00 | 0 | tenferro-rs#2178 |
| concatenate_axis1_f64_8x8x8 | concatenate two 8x8x8 tensors along axis 1 | 1 | 1677.3 | 156.6 | 5.5% | 0.00 | 0 | tenferro-rs#2178 |
| einsum_prepared_f64_n2 | prepared ij,jk->ik execute_into a caller output (overlaps cpu/small_work) | 1 | 928.5 | 81.5 | 5.1% | 0.00 | 0 | tenferro-rs#2178 tenferro-rs#1899 |
| einsum_prepared_f64_n4 | prepared ij,jk->ik execute_into a caller output (overlaps cpu/small_work) | 1 | 918.1 | 51.4 | 3.8% | 0.00 | 0 | tenferro-rs#2178 tenferro-rs#1899 |

## Runtime session-entry diagnostics

These rows time public `Runtime::run_prepared`, including its internal session entry and exit. Runtime construction, inputs, tracing and preparation remain outside timing. They are **diagnostic public-path latency**, not shared-session operation-performance evidence. Their allocation counters also include internal session entry. They are intentionally separate from the steady-state table.

| Case | Workload | Budget | Median ns/op | IQR ns | CoV | allocs/op | bytes/op | Refs |
|---|---|---:|---:|---:|---:|---:|---:|---|
| prepared_mul_add_f64_budget1_32x32x32_session_entry_diagnostic | run_prepared a*b+c including internal session entry/exit, backend thread budget 1 (diagnostic) | 1 | 29434.9 | 4176.1 | 10.8% NOISY | 36.00 | 25988 | tenferro-rs#2178 strided-rs#305 |
| prepared_mul_add_f64_budget2_32x32x32_session_entry_diagnostic | run_prepared a*b+c including internal session entry/exit, backend thread budget 2 (diagnostic) | 2 | 33030.6 | 2380.2 | 10.2% NOISY | 36.00 | 25988 | tenferro-rs#2178 strided-rs#305 |

Numerical checks: **10/10 rows passed** against independent host loops before counting and sampling.

## Prepared-runtime region execution (diagnostics)

Whether the runtime executed the traced region as one fused command or declined to per-op dispatch (cumulative over validation, the allocator pass and timed batches). `run_prepared` enters a runtime-internal session per call, and its entry is included in both diagnostic ns/op and allocs/op.

| Case | Planned regions | Fused executions | Fallbacks |
|---|---:|---:|---:|
| prepared_mul_add_f64_budget1_32x32x32_session_entry_diagnostic | 1 | 1160 | 0 |
| prepared_mul_add_f64_budget2_32x32x32_session_entry_diagnostic | 1 | 1160 | 0 |

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
- Python platform: `macOS-26.6.2-arm64-arm-64bit-Mach-O`

## Run metadata

Full metadata: `data/results/mac-cpu/cpu/overhead_smoke/20261022_033042/run.yaml`

```yaml
timestamp: '2026-10-22T03:30:42Z'
tenferro_rs:
  path: /Users/runner/work/tenferro-benchmark/tenferro-benchmark/extern/tenferro-rs
  commit: d5ef90538a4c05810fe63bc1b5713d871ac98279
  dirty: false
  features:
  - blas-accelerate
  resolved_path: /Users/runner/work/tenferro-benchmark/tenferro-benchmark/extern/tenferro-rs
blas:
  implementation: accelerate
  version: unknown
  library: /System/Library/Frameworks/Accelerate.framework
environment:
  hostname: sat12-bq154-d7ab3148-c059-43aa-8396-42f7e64423bc-3E3EEB82DECD.local
  os: macOS-26.6.2-arm64-arm-64bit-Mach-O
  arch: arm64
  cpu: arm
  env:
    OMP_NUM_THREADS: '1'
    OMP_THREAD_LIMIT: '1'
    OMP_DYNAMIC: 'FALSE'
    RAYON_NUM_THREADS: '1'
    BENCHMARK_TARGET_PROFILE: mac-cpu
    BENCHMARK_COMMIT: 1711cc92c7d04a38b85df1170873ddd4cf029b5b
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
    PUBLICATION_GATE_PROFILE: quick
    BENCHMARK_BUILD_PROFILE: release
    RUSTC_VERSION: rustc 1.99.1 (58cf8f9c3 2026-10-13)
    BENCH_INSTANCE: null
collection:
  command: 'BENCHMARK_TARGET_PROFILE=mac-cpu TENFERRO_CPU_FEATURES=blas-accelerate
    /Users/runner/work/tenferro-benchmark/tenferro-benchmark/scripts/run_overhead_smoke.sh '
  timing_scope: mixed; see timing_scopes_by_case
  timing_scopes_by_case:
    typed_elemwise_f64_add_bcast_n16: steady_state_host_api
    typed_elemwise_f64_add_bcast_8x8x8: steady_state_host_api
    typed_elemwise_f64_add_permuted_view_8x8x8: steady_state_host_api
    permute_materialize_f64_r4_4x4x4x4: steady_state_host_api
    prepared_mul_add_f64_budget1_32x32x32_session_entry_diagnostic: public_path_latency_including_internal_session_entry
    prepared_mul_add_f64_budget2_32x32x32_session_entry_diagnostic: public_path_latency_including_internal_session_entry
    reduce_axis_f64_8x8x8: steady_state_host_api
    concatenate_axis1_f64_8x8x8: steady_state_host_api
    einsum_prepared_f64_n2: steady_state_host_api
    einsum_prepared_f64_n4: steady_state_host_api
  setup_policy:
    steady_state_host_api: inputs, entered session, initialization, tracing and preparation
      outside timing
    public_path_latency_including_internal_session_entry: runtime/input/tracing/preparation
      setup outside timing; run_prepared internal session entry/exit inside timing;
      diagnostic only
  output_policy: outputs retained until the clock stops; destruction outside timing
  allocation_scope: 'separate pooled pass: 64 calls after warmup, outputs dropped
    between calls'
  gating: none (trend view)
```
