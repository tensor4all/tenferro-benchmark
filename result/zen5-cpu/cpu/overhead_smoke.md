# CPU Overhead Smoke (ns/op and allocs/op)

- Suite: `cpu/overhead_smoke`
- Target profile: `zen5-cpu`
- Raw run: `data/results/zen5-cpu/cpu/overhead_smoke/20261112_160628`
- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`
- Collection command: `BENCHMARK_TARGET_PROFILE=zen5-cpu TENFERRO_CPU_FEATURES=native scripts/run_overhead_smoke.sh `

Trend view for tenferro-rs #2178 (tenferro-benchmark #197); nothing here gates. Every row runs on one run thread; the prepared-runtime diagnostics set the CPU backend thread budget explicitly. **allocs/op** counts process-wide Rust allocator events over 64 calls after pool warmup, dropping outputs between calls. These observed counts include Rust workers and exclude foreign BLAS/C allocations; compare matching fixtures, thread budgets and execution scopes. **ns/op** is the median of retained-output batches, with destruction after the clock stops. Shared nightly runner timings are reference indicators only.

## Steady-state operations

Inputs, entered reusable sessions, initialization, tracing and preparation are outside timing. Only the declared operation and intrinsic output allocation or pooled checkout are timed.

| Case | Workload | Budget | Median ns/op | IQR ns | CoV | allocs/op | bytes/op | Refs |
|---|---|---:|---:|---:|---:|---:|---:|---|
| typed_elemwise_f64_add_bcast_n16 | typed add, rank-0 scalar rhs (stride-0 operand) | 1 | 2062.9 | 10.4 | 0.9% | 0.00 | 0 | tenferro-rs#2178 |
| typed_elemwise_f64_add_bcast_8x8x8 | typed add, rank-0 scalar rhs (stride-0 operand) | 1 | 2472.3 | 8.4 | 1.0% | 0.00 | 0 | tenferro-rs#2178 |
| typed_elemwise_f64_add_permuted_view_8x8x8 | add_read, lhs a permuted view (perm [2,0,1]) | 1 | 2820.1 | 233.3 | 11.8% NOISY | 0.00 | 0 | tenferro-rs#2178 |
| permute_materialize_f64_r4_4x4x4x4 | materializing rank-4 transpose (perm [3,1,0,2]) | 1 | 2892.3 | 27.1 | 1.0% | 0.00 | 0 | tenferro-rs#2178 |
| reduce_axis_f64_8x8x8 | typed reduce_sum over axis 1 | 1 | 3049.7 | 22.1 | 0.6% | 0.06 | 102 | tenferro-rs#2178 |
| concatenate_axis1_f64_8x8x8 | concatenate two 8x8x8 tensors along axis 1 | 1 | 2923.2 | 17.2 | 16.7% NOISY | 0.00 | 0 | tenferro-rs#2178 |
| einsum_prepared_f64_n2 | prepared ij,jk->ik execute_into a caller output (overlaps cpu/small_work) | 1 | 1937.7 | 6.9 | 0.4% | 0.00 | 0 | tenferro-rs#2178 tenferro-rs#1899 |
| einsum_prepared_f64_n4 | prepared ij,jk->ik execute_into a caller output (overlaps cpu/small_work) | 1 | 2072.6 | 6.3 | 0.3% | 0.00 | 0 | tenferro-rs#2178 tenferro-rs#1899 |

## Runtime session-entry diagnostics

These rows time public `Runtime::run_prepared`, including its internal session entry and exit. Runtime construction, inputs, tracing and preparation remain outside timing. They are **diagnostic public-path latency**, not shared-session operation-performance evidence. Their allocation counters also include internal session entry. They are intentionally separate from the steady-state table.

| Case | Workload | Budget | Median ns/op | IQR ns | CoV | allocs/op | bytes/op | Refs |
|---|---|---:|---:|---:|---:|---:|---:|---|
| prepared_mul_add_f64_budget1_32x32x32_session_entry_diagnostic | run_prepared a*b+c including internal session entry/exit, backend thread budget 1 (diagnostic) | 1 | 34587.0 | 201.2 | 2.3% | 26.00 | 11968 | tenferro-rs#2178 strided-rs#305 |
| prepared_mul_add_f64_budget2_32x32x32_session_entry_diagnostic | run_prepared a*b+c including internal session entry/exit, backend thread budget 2 (diagnostic) | 2 | 45476.0 | 610.7 | 2.2% | 26.00 | 11968 | tenferro-rs#2178 strided-rs#305 |

Numerical checks: **10/10 rows passed** against independent host loops before counting and sampling.

## Prepared-runtime region execution (diagnostics)

Whether the runtime executed the traced region as one fused command or declined to per-op dispatch (cumulative over validation, the allocator pass and timed batches). `run_prepared` enters a runtime-internal session per call, and its entry is included in both diagnostic ns/op and allocs/op.

| Case | Planned regions | Fused executions | Fallbacks |
|---|---:|---:|---:|
| prepared_mul_add_f64_budget1_32x32x32_session_entry_diagnostic | 1 | 616 | 0 |
| prepared_mul_add_f64_budget2_32x32x32_session_entry_diagnostic | 1 | 616 | 0 |

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

## Run metadata

Full metadata: `data/results/zen5-cpu/cpu/overhead_smoke/20261112_160628/run.yaml`

```yaml
timestamp: '2026-11-12T16:06:29Z'
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
collection:
  command: 'BENCHMARK_TARGET_PROFILE=zen5-cpu TENFERRO_CPU_FEATURES=native scripts/run_overhead_smoke.sh '
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
