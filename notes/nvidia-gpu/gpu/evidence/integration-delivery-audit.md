# Integration delivery audit — 2026-09-19

## Dependency order and submitted state

1. CubeCL PR #19 merged as `a2adda17`; final head `7697cf4` had all six CI
   jobs successful. Startup registration, publication-before-registration,
   check-to-park notification, full/partial buffers, concurrent publication and
   retained task ownership were checked. Native common tests: 46; actual Miri: 43.
   See [CubeCL evidence](cubecl-pr-validation.md), including its separately
   reproduced baseline f16 failure and unexecuted Loom coverage.
2. Cubek PR #13 aligned the existing 0.2 release line, merged as `739dbfb4`.
   tenferro PR #1820 then merged as `d7a8c60c` at 14:08:34Z after required checks
   passed. Its final head `ccf54787` and merge share tree
   `7aa48784764fa7f051e4a38a40bdb0a2e6d67213`.
   [Integration validation](tenferro-integration-validation.md) records local
   tests/lint/sample checks, actual hosted CUDA/PJRT execution, and the late
   closed-PR duplicate-run failure display. That post-merge safety-guard failure
   was not relabeled successful; actual GPU tests passed twice and no status
   check or approval was bypassed.
3. Collection began at 14:27Z, after the dependency merges. Every selected run
   records clean tenferro `d7a8c60c`; per-run `provenance.json` and the
   [manifest](integration-refresh-summary.json) identify benchmark/CubeCL/Cubek
   revisions and canonical raw-file hashes. The resolved Cargo lock and binary
   hashes are retained alongside the command snapshots.

## Measurement and correctness

- All eight original suites and their backend/problem sets were retained.
  Latest canonical output: **267 rows: 201 successful, 61 unsupported,
  five not configured**; keys and status sets match the historical collection.
- The successful rows comprise 101 existing numerical comparisons (including
  44 exact permutation checks) and 100 newly checked AD directions. Both AD
  suites originally mislabeled execution-only rows as verified; those first
  records remain under `20260919_142600`, with explicit provenance warnings.
  Corrected AD runs at `20260919_151433` all pass fixed h=1e-5 central differences
  using unchanged YAML tolerances. Wrong/non-finite derivative rejection tests
  and 20 actual small CUDA smoke cases passed. This is directional validation,
  not a full-gradient or cross-backend gauge proof.
- CPU provider configuration is 1T; OpenBLAS's runtime API reported 1. The three
  Python CLIs explicitly set/log intra-op and inter-op counts, with subprocess
  tests checking both equal 1. Rayon/OpenMP/MKL limits are recorded in run YAML.
  CUDA service threads are distinct from provider workers. Affinity was not
  pinned (allowed CPUs 0–63).
- Suite/backend execution was sequential. Existing prepared-runtime timing,
  completion synchronization and output retention were preserved. AD reference
  evaluations/downloads occur after timed samples. VJP forward work remains
  inside the measured VJP call. Small rows remain single-call diagnostics.
- The initial shell monitor timed out, but the container process completed all
  outputs; the permutation run was not restarted or replaced. Console capture
  was incomplete, so canonical output/schema/hash/report audits were used.
- Original failed notification gates, placement confounds and final small-case
  increases remain in item notes. No non-small tenferro comparison exceeded a
  10% increase (largest ratio 1.0674); these historical comparisons are not
  randomized causal proof. No post-hoc latency retry was selected.

## Artifact checks and preservation

Standard run/result schema checks, GPU AD formatter/runner contracts, GPU
permutation contracts, layout/docs-output tests, focused Rust/Python tests and
changed Rust-binary Clippy passed. Dependency warnings were not described as
warning-free; unrelated Julia fixture checks were unavailable. All eight reports
regenerated identically twice; hand-maintained notes were unchanged. All seven
published item-note links resolve to existing files and anchors.

Implementation and notes/report diffs were self-reviewed. Main user checkouts,
separate shared-rule edits, original experiments, local worklogs and large
build/profile artifacts were preserved. Only selected canonical raw data,
provenance, notes/evidence and generated latest reports belong in this delivery.
The delivery branch is `agent/refresh-nvidia-gpu-20260919`; its final commit/push
is identified in the session handoff rather than a self-referential commit hash.
