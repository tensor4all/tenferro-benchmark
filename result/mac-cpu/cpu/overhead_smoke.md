# CPU Overhead Smoke (ns/op and allocs/op)

- Suite: `cpu/overhead_smoke`
- Target profile: `mac-cpu`
- Raw run: `data/results/mac-cpu/cpu/overhead_smoke/20261104_182149`
- tenferro-rs commit: `a000e5c4929d5bebb91e783c329b27106baed12e`
- Collection command: `BENCHMARK_TARGET_PROFILE=mac-cpu TENFERRO_CPU_FEATURES=blas-accelerate /Users/runner/work/tenferro-benchmark/tenferro-benchmark/scripts/run_overhead_smoke.sh `

Trend view for tenferro-rs #2178 (tenferro-benchmark #197); nothing here gates. Every row runs on one run thread; the prepared-runtime diagnostics set the CPU backend thread budget explicitly. **allocs/op** counts process-wide Rust allocator events over 64 calls after pool warmup, dropping outputs between calls. These observed counts include Rust workers and exclude foreign BLAS/C allocations; compare matching fixtures, thread budgets and execution scopes. **ns/op** is the median of retained-output batches, with destruction after the clock stops. Shared nightly runner timings are reference indicators only.

## Steady-state operations

Inputs, entered reusable sessions, initialization, tracing and preparation are outside timing. Only the declared operation and intrinsic output allocation or pooled checkout are timed.

| Case | Workload | Budget | Median ns/op | IQR ns | CoV | allocs/op | bytes/op | Refs |
|---|---|---:|---:|---:|---:|---:|---:|---|
| typed_elemwise_f64_add_bcast_n16 | typed add, rank-0 scalar rhs (stride-0 operand) | 1 | 867.9 | 15.9 | 1.0% | 0.00 | 0 | tenferro-rs#2178 |
| typed_elemwise_f64_add_bcast_8x8x8 | typed add, rank-0 scalar rhs (stride-0 operand) | 1 | 1161.0 | 40.8 | 15.6% NOISY | 0.00 | 0 | tenferro-rs#2178 |
| typed_elemwise_f64_add_permuted_view_8x8x8 | add_read, lhs a permuted view (perm [2,0,1]) | 1 | 1333.9 | 48.9 | 5.3% | 0.00 | 0 | tenferro-rs#2178 |
| permute_materialize_f64_r4_4x4x4x4 | materializing rank-4 transpose (perm [3,1,0,2]) | 1 | 1260.5 | 6.6 | 0.4% | 0.00 | 0 | tenferro-rs#2178 |
| reduce_axis_f64_8x8x8 | typed reduce_sum over axis 1 | 1 | 1558.4 | 33.9 | 2.2% | 0.00 | 0 | tenferro-rs#2178 |
| concatenate_axis1_f64_8x8x8 | concatenate two 8x8x8 tensors along axis 1 | 1 | 1384.3 | 27.3 | 1.4% | 0.00 | 0 | tenferro-rs#2178 |
| einsum_prepared_f64_n2 | prepared ij,jk->ik execute_into a caller output (overlaps cpu/small_work) | 1 | 795.8 | 32.3 | 2.2% | 0.00 | 0 | tenferro-rs#2178 tenferro-rs#1899 |
| einsum_prepared_f64_n4 | prepared ij,jk->ik execute_into a caller output (overlaps cpu/small_work) | 1 | 844.3 | 65.1 | 8.8% | 0.00 | 0 | tenferro-rs#2178 tenferro-rs#1899 |

## Runtime session-entry diagnostics

These rows time public `Runtime::run_prepared`, including its internal session entry and exit. Runtime construction, inputs, tracing and preparation remain outside timing. They are **diagnostic public-path latency**, not shared-session operation-performance evidence. Their allocation counters also include internal session entry. They are intentionally separate from the steady-state table.

| Case | Workload | Budget | Median ns/op | IQR ns | CoV | allocs/op | bytes/op | Refs |
|---|---|---:|---:|---:|---:|---:|---:|---|
| prepared_mul_add_f64_budget1_32x32x32_session_entry_diagnostic | run_prepared a*b+c including internal session entry/exit, backend thread budget 1 (diagnostic) | 1 | 24004.6 | 1225.3 | 3.3% | 26.00 | 12656 | tenferro-rs#2178 strided-rs#305 |
| prepared_mul_add_f64_budget2_32x32x32_session_entry_diagnostic | run_prepared a*b+c including internal session entry/exit, backend thread budget 2 (diagnostic) | 2 | 31129.6 | 1594.1 | 3.6% | 26.00 | 12656 | tenferro-rs#2178 strided-rs#305 |

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
- Python platform: `macOS-26.7.2-arm64-arm-64bit-Mach-O`
- Container: `none`

## Run metadata

Full metadata: `data/results/mac-cpu/cpu/overhead_smoke/20261104_182149/run.yaml`

```yaml
timestamp: '2026-11-04T18:21:49Z'
tenferro_rs:
  path: /Users/runner/work/tenferro-benchmark/tenferro-benchmark/extern/tenferro-rs
  commit: a000e5c4929d5bebb91e783c329b27106baed12e
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
  hostname: sat12-bq165-60724c85-8f95-4f8f-a31a-4ca8fff3c199-C62EC05580DD.local
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
    BENCHMARK_COMMIT: ef930a2e0f075f6b42bbbf9608afa9ea69ed412b
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
