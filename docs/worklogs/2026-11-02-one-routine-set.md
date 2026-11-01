# One routine set

**Decision (shinaoka, 2026-11-02):** "full と quick も分けたくない．常に全部走らせる前提で 1 セットの方がシンプル" — the routine GPU suite has one curated set; no mode switch survives in code, manifests, run metadata, reports, tests or docs.

- Set contents: every routine suite once, one representative per operation family; size sweeps (solve sizes, AD n=2/8 and n=512) are characterization, not weekly. 106 rows (`docs/gpu-routine-coverage.md` has the cost table).
- Vendor arms are opt-in (`BENCH_ROUTINE_VENDOR_ARMS=1`), so they stay visible in the manifest without joining the weekly bundle (133 rows with them).
- `gpu/linalg_jvp_vjp` and `gpu/linalg_ad_latency` joined the set through their existing collector; `run_gpu_suite.sh` delegates to it per suite.
- The routine control lane keeps the ten timed and eight state mechanisms it already ran weekly; the wider `gpu/perf_issues` variants stay in the instance data and in that suite's own quick/full matrix, which is not the routine selection.
- Every suite now records its own start, end and wall time in run metadata, so per-suite cost appears in the routine workflow report on any device.
- First real-DEVICE validation of the single set is pending on primerose; no GPU run was made in this change.
