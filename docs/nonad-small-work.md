# Non-AD eager extension comparisons

Issue #98 extends the ordinary `cpu/small_work` suite with 120 F64 cases:
binary `ij,jk->ik`, three-operand `ij,jk,kl->il`, QR (Q/R), thin SVD
(U/S/Vt), and solve (all n RHS), at n=2,4,8,16,64. All five operations
also have transposed metadata-view inputs at n=16. Logical fixture values
match compact cases; both APIs receive the non-contiguous view. Default
QR/SVD options and default Auto einsum strategy are identical.

Each workload has direct/eager shared-session and direct/eager per-call
session routes. The per-call routes are explicitly session-entry diagnostics.
Eager values are untracked, under no_grad, in a fresh process that never enters
capture_trace. Validation checks input/output tracks_grad and checks each
route independently. The runner does not query private capture state.
Einsum ordinary public calls include planning/cache lookup on both routes;
their gap must not be described as pure wrapper cost. Existing prepared-repeat
cases remain separate preparation-excluded baselines.

All output tensors are retained until the batch clock stops. Extra owned output
materialization is used only for untimed validation. Memory-bounded batches
target 2 ms. Sub-1-ms measured batches and CoV above 10% remain inconclusive.
Initial public calls inside an entered session are separately recorded as
single-call initialization diagnostics, never used for performance ratios.

Use the repository devcontainer image and provider policy. Collect sequential
1T and 4T runs of the full suite:

```bash
devcontainer exec --workspace-folder . bash -lc '
  export TENFERRO_CPU_FEATURES=blas-mkl BENCHMARK_TARGET_PROFILE=amd-cpu
  export BENCH_COVERAGE=full BENCH_EFFORT=standard
  ./scripts/run_small_work.sh 1 4'
```

To collect just these comparisons, set BENCH_INSTANCE to a comma-separated
list of the `nonad_` case IDs in `data/instances/small_work.json`, with full
coverage. Quick adds only the twenty n=2 routes to the existing steady-state
selection. Full coverage and measurement effort remain independent.

After latency collection the same runner builds a separate executable under
`${CARGO_TARGET_DIR}-allocations` with `allocation-diagnostics` and runs seven
allocation observations per matching case/thread setting. That executable
refuses latency mode. Calling-thread Rust allocator requests include result
packaging and realloc's requested new size; they exclude foreign BLAS and
worker-thread allocation, deallocation, validation, and result destruction.
Counts are not total process allocation or retained memory.

Raw latency and allocation JSONL files, initial-call diagnostics, numerical
residuals, provider/revision/thread metadata, and case accounting live under
`data/results/<target_profile>/cpu/small_work/<timestamp>/`. The generated
`result/<target_profile>/cpu/small_work.md` shows matched session-policy pairs,
absolute eager-minus-direct differences, ratios, variability, and separate
allocation observations with the corresponding case IDs. These observations
are coverage, not an optimization or a performance threshold.
