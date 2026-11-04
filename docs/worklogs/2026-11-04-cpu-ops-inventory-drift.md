# CPU ingestion inventory drift (issue #329 B)

Date: 2026-11-04
Runs used: [41687773887](https://github.com/tensor4all/tenferro-benchmark/actions/runs/41687773887)
(linux CPU, `main` @ `0baddde`, 2026-11-03) and
[41751198993](https://github.com/tensor4all/tenferro-benchmark/actions/runs/41751198993)
(mac CPU, `fix/thread-env-empty-array` @ `a25e625`, 2026-11-04), both from their
`*-benchmark-diagnostics-<id>` artifacts.

This worklog records how the numbers were obtained and why they were changed.
The code change is the regeneration script and the regenerated file.

## Method

Packaging aborted before writing `ingestion.json`, so the bundle document the
issue asks for does not exist. Both diagnostics artifacts still carry
`data/results/<profile>/cpu/**` and `result/<profile>/`, so the packaging step
was replayed locally over the extracted artifact:

```bash
uv run python scripts/ci/prepare_result_bundle.py --profile amd-cpu \
  --harness-commit 0badddea0f684f968c31d7df5335c90109c2abd4 \
  --started-at 2026-11-03T17:48:25Z \
  --root <diag artifact> --bundle-dir /tmp/bundle
```

This reproduces `ValueError: CPU workload selection does not match declared
nightly inventory: cpu/cpu_ops` exactly. The identity sets it computes come from
the recorded `cpu_ops_t*.csv` pairs of the `cpu/einsum` run directories
(`20261103_180309` + `20261103_183913` for linux, `20261104_010357` +
`20261104_013754` for mac), because `cpu_ops` and `linalg_jvp_vjp` are derived
from that CSV.

## Result

| suite | declared | collected (linux) | collected (mac) |
| --- | --- | --- | --- |
| `cpu/cpu_ops` | 892 / `ccde3d12…` | 992 / `fe3a724a…` | 992 / `fe3a724a…` |
| `cpu/linalg_jvp_vjp` | 300 / `a37200d0…` | 400 / `826379c4…` | 400 / `826379c4…` |

Seven of the nine declared suites appear in both run windows. Five of them
reproduce their declaration exactly (`cpu/einsum` 480, `cpu/fft` 96,
`cpu/public_api` 3024, `cpu/small_work` 144, `cpu/session_matrix` 114); the two
in the table above do not. `cpu/permutation` and `cpu/overhead_smoke` have no
run in either window, so both artifacts report them as `missing_suites`, and
their declared 92 and 10 identities were carried over unchanged rather than
re-measured here.

Note the direction: the ledger declares **892** and the collection produces
**992**, not the reverse. `cpu_ops` sorts first, which is why the adapter names
it while `linalg_jvp_vjp` is also stale.

The declared fingerprints are reproducible from the 2026-10-06 runs
(`data/results/amd-cpu/cpu/einsum/20261006_172037` + `…_175113`): 446 rows per
thread, 892 identities, `ccde3d12…`. The ledger is a faithful snapshot of that
collection, not a hand-typed guess.

## The 100 added identities

All 100 are `backend=tenferro-eager`, 10 case ids × 5 shapes × 2 thread counts,
with no removals:

| case id | shapes |
| --- | --- |
| `grad_sum_eigh_jvp`, `grad_sum_eigh_vjp` | 2x2, 4x4, 8x8, 256x256, 512x512 |
| `grad_sum_lu_jvp`, `grad_sum_lu_vjp` | 2x2, 4x4, 8x8, 256x256, 512x512 |
| `grad_sum_qr_jvp`, `grad_sum_qr_vjp` | 2x2, 4x4, 8x8, 256x256, 512x512 |
| `grad_sum_svd_s_jvp`, `grad_sum_svd_s_vjp` | 2x2, 4x4, 8x8, 256x256, 512x512 |
| `grad_sum_solve_jvp`, `grad_sum_solve_vjp` | 2x2/rhs=1, 4x4/rhs=1, 8x8/rhs=1, 256x256/rhs=1, 512x512/rhs=1 |

`cpu/linalg_jvp_vjp` gains the same 10 case ids × 5 shapes × 2 threads = 100
identities: the derived AD page is built from the CPU-ops rows named `_jvp` /
`_vjp`.

Per-thread CSV row counts show the same thing: in 2026-10-06 `tenferro-eager`
had 21 of the 71 linalg-AD rows, in 2026-11-03 it had all 71. `pytorch-cpu`,
`jax-cpu` and `tenferro-trace` were already complete.

## Which change moved the row set

`e479872` — *cpu/publication_gate: measure linalg JVP/VJP on the eager arm
(#311)*, 2026-11-02. Its message states the ten case ids "were registered only
through `push_linalg_jvp_vjp_trace_benches`", registers them on the eager arm, and
adds a test that fails if either arm loses one. Its `src/bin/publication_gate.rs`
diff adds `push_linalg_jvp_vjp_eager_benches` calls into `run_small_latency`,
`run_large_throughput` and `run_batched_small` for the five losses, which is the
10 case ids × 5 shapes × 2 threads the CSV counts gained. The ledger was not
regenerated in that commit, and nothing in CI could have noticed.

## Decision

The added rows are intended (a missing eager measurement arm, not a regression),
so the declaration was regenerated rather than the rows dropped. No timing value,
report or raw record changed; only the two fingerprints and the file's
`description` line.

The declaration is now produced by
`scripts/ci/generate_cpu_ingestion_selection.py`, which packages a collection
through the same code path that gates ingestion (`enforce_inventory=False` on
`package_cpu`/`rebuild_cpu` only for the generator) and has a `--check` mode that
exits nonzero and prints declared/collected counts plus digests per stale suite.
The gate itself is unchanged: a row-set change still fails collection and
ingestion until the ledger is regenerated and reviewed. The generator also
passes the collection window to `build_source`, so a report naming an older run
is rejected instead of fingerprinted.

`--check` only covers suites the collection produced. `cpu/permutation` and
`cpu/overhead_smoke` have no run in either artifact, so this change does not
re-verify their declarations:
"matches all 7 collected suites" is not "all nine declarations are current".

A consequence worth stating: re-ingesting an old bundle (for example the 10-29
green run, whose manifest declares 892 identities) now fails the ledger check.
That is intended — an old bundle cannot match today's inventory — and the
diagnostics-recovery path is manual, not scheduled.
