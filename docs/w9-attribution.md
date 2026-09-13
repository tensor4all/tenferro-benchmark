# W9 CPU Attribution Record

This record captures the measurement-only work requested by
`tensor4all/tenferro-rs#1490`. The instrumentation does not change a kernel or
the existing steady-state timing region.

## Pinned context

- tenferro-rs: `87df2567c38a70b8adea263b41afd9ec94ef5ca7`
- tenferro-benchmark: `46788588e951c9555e85ae987ed7e66f49611d4e`
- strided-rs pin used by tenferro-rs:
  `95a607c061e95cc3a698a083dd279283a905f99c`
- target public rows: `compare_lt`, `eigvals`, `inv`, and
  `broadcast_in_dim`

In the historical CSV discussed here, `tenferro-eager` was the direct
`Tensor + CpuBackend` API. It does not construct an `EagerTensor`. Reports
rename the column to `tenferro-rs direct API`, but attribution must not treat
this path as eager-AD conversion overhead.

## Source audit

### `compare_lt`

The direct path is:

```text
CpuBackend::compare
  -> compare_read_with_pool
  -> typed_same_shape_binary_view_with_pool
  -> zip_map2_into
```

The traced single comparison is not eligible for CPU elementwise fusion,
because `ElementwiseFusionOp` has no comparison variant. It reaches the same
CPU compare implementation through runtime dispatch.

Neither path uses an erased Bool destination. The strided erased API only calls
`ErasedRawStridedMut::validate_data_if_needed` after mutable bytes have escaped
through `data_mut`; a newly constructed descriptor starts with
`needs_data_revalidation = false`. Therefore an erased Bool validity scan is
not a candidate cause for the current `compare_lt` row. Profiling should still
verify that the measured samples land in the typed comparison kernel and not
in allocation, dispatch, or shape broadcasting.

### Direct versus trace

The public runner already excludes trace graph construction, compilation, and
runtime construction from its steady-state timing. The optional stage CSV now
records these excluded layers plus first and steady execution:

- direct: `steady_execute`
- trace: `graph_build`, `compile`, `runtime_build`, `first_execute`,
  `steady_execute`

`steady_execute` includes output construction, shape/dtype consumption, and
output drop, matching the existing public row. Sampling profiles are required
to split backend work from runtime dispatch, validation, allocator, and output
lifetime costs without adding timers inside hot library code.

Op-specific hypotheses remain measurements, not conclusions:

- `eigvals`: compare the eigensolver/provider frames with temporary
  eigenvector allocation/drop and runtime slot-workspace frames.
- `inv`: compare identity construction and solve/provider frames with runtime
  preparation and extension dispatch.
- `broadcast_in_dim`: compare validation/allocation frames with
  `copy_into`; direct and trace should reach the same materialization kernel.

## Results

The paired campaign ran twice on an AMD EPYC 7713P. CPUs 60-63 were idle before
collection. Each cell used 3 warmups and 15 samples; t1 and t4 cells ran
sequentially.

| workload | threads | run 1 direct / trace | run 2 direct / trace |
|---|---:|---:|---:|
| `compare_lt` | 1 | 61.461 / 61.350 ms | 61.464 / 60.693 ms |
| `compare_lt` | 4 | 20.851 / 20.975 ms | 20.882 / 20.911 ms |
| `eigvals` | 1 | 9.774 / 7.761 ms | 9.737 / 7.505 ms |
| `eigvals` | 4 | 12.004 / 8.547 ms | 11.424 / 8.560 ms |
| `inv` | 1 | 38.784 / 43.430 ms | 39.051 / 43.932 ms |
| `inv` | 4 | 13.502 / 22.292 ms | 12.458 / 18.865 ms |
| `broadcast_in_dim` | 1 | 143.279 / 142.538 ms | 143.637 / 143.057 ms |
| `broadcast_in_dim` | 4 | 64.724 / 64.705 ms | 66.478 / 61.925 ms |

Attribution outcomes:

- `compare_lt` reaches the same typed strided comparison kernel from direct and
  trace. There is no erased Bool validity scan on this path. The remaining
  kernel-throughput work is tracked in `tensor4all/strided-rs#179`.
- Direct `eigvals` computes and discards eigenvectors, while trace live-output
  pruning selects the values-only primitive. The fix is tracked in
  `tensor4all/tenferro-rs#1524`.
- A paired t1 `strace` run found 418 direct versus 1,362 trace futex calls and
  718 versus 1,067 `sched_yield` calls across 18 executions of `inv`. The
  executor-admission work is tracked in `tensor4all/tenferro-rs#1525`.
- The reported `broadcast_in_dim` asymmetry did not reproduce. No speculative
  optimization issue was filed.

Hardware `perf` sampling was unavailable under the host security policy. The
campaign did not weaken that policy.

## Execution protocol

Do not run these commands while another timing or profiling job is active.
Build once without CPU affinity, then run one process at a time. The `taskset`
binding below is diagnostic isolation, not the final publication campaign.
Use the same provider, thread count, build, and CPU set for each direct/trace
pair.

```bash
export WT=/home/shinaoka/tensor4all/tenferro-benchmark/.worktrees/w9-measurement-instrumentation
export TARGET=/home/shinaoka/tensor4all/.target/w9-attribution
cd "$WT"

CARGO_TARGET_DIR="$TARGET" \
  cargo build --release --no-default-features --features cpu-faer \
  --bin benchmark_cpu_public_api
```

Run stage attribution for one row and one path:

The publication wrapper rejects both diagnostic variables so a filtered or
multi-invocation run cannot overwrite the latest report or truncate one shared
attribution file. Invoke the Rust binary directly as shown below.

```bash
PUBLIC_API_SUITE_FILTER=cpu/linalg_uncovered \
PUBLIC_API_BENCHMARK_FILTER=eigvals \
PUBLIC_API_EXECUTION_FILTER=trace \
PUBLIC_API_ATTRIBUTION_OUTPUT=/tmp/w9-eigvals-trace-t1-stages.csv \
BENCH_WARMUPS=3 BENCH_RUNS=15 RAYON_NUM_THREADS=1 \
taskset -c 60 "$TARGET/release/benchmark_cpu_public_api" \
  --num-threads 1 --output /tmp/w9-eigvals-trace-t1.csv
```

Use these suite/benchmark pairs:

```text
cpu/elementwise_reduction compare_lt
cpu/linalg_uncovered       eigvals
cpu/linalg_uncovered       inv
cpu/structural_shape       broadcast_in_dim
```

Repeat each row with `PUBLIC_API_EXECUTION_FILTER=direct` and `trace`, first at
thread 1 on CPU 60, then at thread 4 on CPUs 60-63. Use distinct output files.

When host policy permits hardware profiling, collect call-graph attribution
with the same filters. The current host may set `perf_event_paranoid` high
enough to reject this operation; that does not invalidate the unprofiled stage
attribution and must not be bypassed by weakening host security policy.
Example for `compare_lt` direct, thread 1:

```bash
PUBLIC_API_SUITE_FILTER=cpu/elementwise_reduction \
PUBLIC_API_BENCHMARK_FILTER=compare_lt \
PUBLIC_API_EXECUTION_FILTER=direct \
PUBLIC_API_ATTRIBUTION_OUTPUT=/tmp/w9-compare-lt-direct-t1-stages.csv \
BENCH_WARMUPS=3 BENCH_RUNS=15 RAYON_NUM_THREADS=1 \
perf record -F 499 --call-graph dwarf \
  -o /tmp/w9-compare-lt-direct-t1.perf.data -- \
  taskset -c 60 "$TARGET/release/benchmark_cpu_public_api" \
  --num-threads 1 --output /tmp/w9-compare-lt-direct-t1.csv

perf report --stdio -i /tmp/w9-compare-lt-direct-t1.perf.data \
  > /tmp/w9-compare-lt-direct-t1.perf.txt
```

Collect counters separately so call-graph sampling does not contaminate the
timing samples:

```bash
PUBLIC_API_SUITE_FILTER=cpu/elementwise_reduction \
PUBLIC_API_BENCHMARK_FILTER=compare_lt \
PUBLIC_API_EXECUTION_FILTER=direct \
BENCH_WARMUPS=3 BENCH_RUNS=15 RAYON_NUM_THREADS=1 \
perf stat -x, -e cycles,instructions,cache-misses,page-faults \
  -o /tmp/w9-compare-lt-direct-t1.stat -- \
  taskset -c 60 "$TARGET/release/benchmark_cpu_public_api" \
  --num-threads 1 --output /tmp/w9-compare-lt-direct-t1-stat.csv
```

## Decision rule

For each direct/trace pair, report median/IQR from the unprofiled stage run.
Include sample percentages when `perf` is permitted. File a focused issue only
when stage timing, source-path audit, or an available profile explains a
reproducible material share of the row. If symbols are inlined beyond useful
attribution, rebuild the same SHA with debuginfo and DWARF call graphs; do not
add timers to library hot loops as the first step.

Current runners label direct operations `tenferro-direct`. See [timing policy revision 2](timing-policy.md) before interpreting the historical timings above.
