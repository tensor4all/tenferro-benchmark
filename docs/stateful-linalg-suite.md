# Stateful incremental QR

`cpu/stateful_linalg` covers benchmark Issue #94 and tenferro-rs #1750 through
public compact Householder QR APIs. It uses F64 deterministic full-rank
column-major inputs with 64,128,256 rows, initial rank 5 and nine width-3
appends, reaching rank 32. Each of the nine steps appends a block, extracts R
and requests the three newly appended Q columns.

Paths are concrete shared session, concrete per-operation session, eager
shared session and eager per-operation session (both with a reused runtime),
separately attributed concrete append/R/Q, explicit-Q BCGS2, and separately
attributed BCGS2 append. BCGS2 uses two public projection/residual corrections,
QR of the residual block, and public concatenation to update explicit Q/R.
It is a diagnostic comparator, rather than full accumulated QR.

Setup, initial factors, input blocks, shared session admission, restoration,
output-retention vectors and numerical validation are untimed. All output
states, snapshots and BCGS2 intermediates are retained until the batch clock
stops. An explicit initialization pass runs even with zero requested warmups.
Attribution runs separately, so per-call clocks cannot contaminate ordinary
complete-workflow samples. Fresh session paths explicitly measure workflow
entry/exit costs, as requested by this suite.

The standard protocol uses 64 workflows per interval, three warmups and three
intervals in each of seven fresh processes, alternating the entire path/case
order each round. Reconstruction is checked at every observed prefix, then
final reconstruction and Q orthogonality. Eager fixtures/results are untracked
under no_grad; the process never enters capture_trace. Initial factorization
does not contribute to the operation table. Binary initialization and phase
wall times are recorded separately from process startup/total collection.

Collect sequentially in the repository's default devcontainer:

```bash
devcontainer exec --workspace-folder . bash -lc '
  export TENFERRO_CPU_FEATURES=blas-mkl BENCHMARK_TARGET_PROFILE=amd-cpu
  ./scripts/run_cpu_stateful_linalg.sh 1 4'
```

One thread is the primary diagnostic; the ordinary 4T setting remains usable.
The native macOS path uses blas-accelerate and mac-cpu. No affinity pinning is
applied. Rows with process-median CoV above 10%, missing observations/replicates
or sub-1-ms process-median batches remain inconclusive. The report presents
complete/(append+R+Q), fresh/shared, eager/shared and compact/BCGS2 append
ratios; these diagnostics are not historical speedup claims or an exact
overhead decomposition. Component clocks and differing session policies are
explicit in the raw rows.

Raw samples and metadata go to
`data/results/<target_profile>/cpu/stateful_linalg/<timestamp>/`; generated
tables go to `result/<target_profile>/cpu/stateful_linalg.md`. This is a
standalone runner, not an extra cost in run_all. Future estimator-maintenance
workloads can use this suite once a suitable public solve/update API exists.
