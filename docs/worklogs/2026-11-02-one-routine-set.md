# One routine set

**Decision (shinaoka, 2026-11-02):** "full と quick も分けたくない．常に全部走らせる前提で 1 セットの方がシンプル" — the routine GPU suite has one curated set; no mode switch survives in code, manifests, run metadata, reports, tests or docs.

- Set contents: every routine suite once, one representative per operation family; size sweeps (solve sizes, AD n=2/8 and n=512) are characterization, not weekly. 106 rows (`docs/gpu-routine-coverage.md` has the cost table).
- Vendor arms are opt-in (`BENCH_ROUTINE_VENDOR_ARMS=1`), so they stay visible in the manifest without joining the weekly bundle (133 rows with them).
- `gpu/linalg_jvp_vjp` and `gpu/linalg_ad_latency` joined the set through their existing collector; `run_gpu_suite.sh` delegates to it per suite.
- The routine control lane keeps the ten timed and eight state mechanisms it already ran weekly; the wider `gpu/perf_issues` variants stay in the instance data and in that suite's own quick/full matrix, which is not the routine selection.
- Every suite now records its own start, end and wall time in run metadata, so per-suite cost appears in the routine workflow report on any device.
- First real-DEVICE validation of the single set is pending on primerose; no GPU run was made in this change.

## Follow-up: no suite is silently outside the set

Reviewed every `benchmarks/gpu/*.yaml`: 11 files were outside the first version of
the set. Eight are now collected (`permutation`, `lstsq`, `linalg_public_api`, `fft`,
`nn`, `structural`, `isolated_ops`, `reuse_contraction`) through two existing
collectors, which meant teaching the routine selection to curate instance-driven
suites (`problems: {source, include}`) and adding one fold
(`scripts/normalize_gpu_records.py`) so the accounting, the bundle row gate and the
dashboard summary keep one record contract. Two are excluded by design (`*_throughput`:
deferred-batch regime with its own driver) and one is the diagnostic matrix
(`perf_issues`, whose curated control set already runs weekly). The set is 714 rows;
the documented fallback if one A100 night cannot hold it is to move
`isolated_ops` + `reuse_contraction` to monthly.
