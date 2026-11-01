# Retired: `linux-cpu` target profile

Retired on 2026-11-02. The `linux-cpu` target profile is no longer a
maintained, collectable or publishable profile. Its reports below are kept
byte-for-byte as history; nothing here was deleted.

## Why it was retired

- The tracked reports are stale. `cpu/einsum`, `cpu/cpu_ops`,
  `cpu/linalg_jvp_vjp` and `cpu/permutation` were last regenerated 2026-07-28,
  97 days before retirement. `cpu/linalg_jvp_jvp` was last rewritten
  2026-10-08 (25 days), also not superseded by a later run.
- The tree mixes tenferro-rs revisions: `cpu/permutation` was measured at
  `80ebcc38ce11fb93385e8b6a1a49b613bc17452f`, `cpu/linalg_jvp_jvp` at
  `5cf78c7ec0ad9516dd78bab546d5e7bd42fa3102`. A single profile column is not
  supposed to hold two revisions, so these rows cannot be read as one
  comparable set.
- It was never a real hardware profile: since the Linux nightly moved to
  `BENCHMARK_TARGET_PROFILE=amd-cpu`, `result/linux-cpu/` only held report
  aliases rewritten from `amd-cpu` runs (`scripts/mirror_linux_cpu_cpu_reports.sh`).
- Its coverage is superseded. `zen5-cpu` (AMD Ryzen AI 9 HX 470, pinned
  1/4/8-thread rows, 2026-10-27/28) and `amd-cpu` (EPYC 7713P Linux
  devcontainer, nightly) are the maintained Linux CPU profiles and are strict
  supersets of what this profile measured.

## What was removed (user-facing and write paths)

- Removed from `README.md`, `AGENTS.md`, `docs/results.md`,
  `docs/result-ingestion.md` profile listings, and from the dashboard profile
  maps / report discovery in `scripts/generate_dashboard.py`, so no page can
  be generated for it (including `--all-profiles`).
- Removed from the accepted profiles of `scripts/run_cpu_fft.sh`,
  `scripts/run_cpu_public_api.sh` and `scripts/run_paired_timing.sh`, and from
  `TARGET_PROFILE_RE` (already absent before retirement), so a collection run
  naming `linux-cpu` fails instead of writing a report.
- The remaining writers and linkers refuse it too:
  `scripts/regenerate_report_summaries.py --publish` rejects a recorded run
  whose `run.yaml` names `linux-cpu`, and
  `scripts/generate_dashboard.py --profile linux-cpu` fails instead of
  generating a page or a history link.
- `scripts/ci/result_ingestion.py` rejects a bundle whose manifest declares
  `linux-cpu` (`Retired target profile`), so a stale bundle cannot publish.
- `scripts/mirror_linux_cpu_cpu_reports.sh` and
  `scripts/reproduce_linux_cpu_linalg_jvp_jvp.sh` were deleted: their only
  product was `result/linux-cpu/` reports.

The raw runs these reports were built from remain under
`data/results/linux-cpu/`, and the reports themselves remain under
`result/linux-cpu/cpu/`. Git history has everything.

## Reviving the profile

The data is intact, so revival is a code change plus fresh measurements. Every
gate below excludes `linux-cpu` today; a revival must lift all of them:

1. Profile allowlists: `TARGET_PROFILE_RE` (`scripts/benchmark_layout.py`), the
   `target_profile` enum in `schemas/benchmark-run.schema.json`, and the
   `case` statements of `scripts/run_all.sh`, `scripts/run_cpu_fft.sh`,
   `scripts/run_cpu_public_api.sh`, `scripts/run_permutation.sh`,
   `scripts/run_elementwise.sh` and `scripts/run_paired_timing.sh`.
2. Publication: delete `linux-cpu` from `RETIRED_PROFILES` in
   `scripts/generate_dashboard.py` and `scripts/ci/result_ingestion.py`, and
   re-add it to the dashboard maps (`PROFILE_ORDER`, `PROFILE_FAMILY`,
   `LOCAL_PROFILE_LABEL`, `GITHUB_HOSTED_PROFILES`, `MAINTAINED`). Add it to
   `benchmarks/profiles.yaml` (`maintained` to publish a page, `optional` for a
   collection-only profile).
3. Ingestion: add it to the packaging choices in
   `scripts/ci/bundle_cpu_results.py` and `scripts/ci/prepare_result_bundle.py`;
   to publish through ingestion, map the producing canonical workflow to the
   profile in `WORKFLOWS`, add its diagnostics artifact name, extend
   `scripts/ci/ingest_benchmark_results.py`'s diagnostics map, and make that
   workflow set `BENCHMARK_TARGET_PROFILE=linux-cpu`.
4. Re-measure. These reports are not revival candidates: they carry two
   tenferro revisions and predate the current timing policy, so a revived
   `linux-cpu` needs fresh `run_all.sh` collections at the same revision as its
   siblings before any page is published.
