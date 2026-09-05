# Small-work public boundary measurement

Status: DeepSeek V4 Flash design round 1 Correct-to-merge; implementation pending.
The four non-blocking review pins are incorporated below.
Issue: https://github.com/tensor4all/tenferro-benchmark/issues/95
Parent: https://github.com/tensor4all/tenferro-rs/issues/1758
Benchmark starting revision: `ce729ecfa040b4c825850e0bcf0907142bdac58f`.
Library starting revision: `0457a2ed0aeea21b14f4297f7f4731e09b3a0507`.

## Outcome

Add `cpu/small_work` without changing the purpose or timing boundaries of existing
`public_api`, `view_metadata` or `output_reuse` comparisons. Measure complete ordinary
calls and workflows as well as explicit setup stages; do not treat numerical aliases
as equivalent wrapper overhead. Component probes remain in tenferro-rs. This design
covers the benchmark-owned runner/schema/contracts/report layer and public API cases;
any owning-library component instrumentation requires its own approved design/diff.
The complete issue remains open until integrated component probes and published raw
evidence exist. No optimization or new library public fast API is authorized here.

## Reuse and provenance

Reuse suite/run/result conventions, `scripts/validate_benchmark_suite.py`, existing
fixture/correctness/timing-scope/plausibility checks and target-profile paths. Existing
`benchmark-result.schema.json` is currently GPU-oriented; extend conventions with a
small-work schema and validator kind rather than weakening old result contracts.
Do not modify existing measured rows just to make the new suite fit.

The unpushed prototype commits `30f446761d08bd085da356b21f6a509651776b10` and `c5dba6bd`
are absent from this host's tenferro object database. Record this provenance limitation;
do not fabricate their contents or present them as remotely fetchable baselines.
Implement equivalent probes from current public APIs and document that distinction.
Resolve/pin all tenferro workspace dependencies consistently to the selected library
revision. Use `TENFERRO_REF` and a detached checkout in this worktree's own
`extern/tenferro-rs`, not another user's checkout. The library revision is locally
resolved and equals fetched origin/main; verify its fetchability during setup and
record the resolved SHA. Never change other worktrees' extern pins. If migration breaks existing benchmark binaries, fix and validate their
public API usage together or propose a focused design amendment before implementing
an incompatible dependency split. Preserve all unrelated existing user worktrees.

## Cases and API tiers

A bounded explicit manifest selects representative cases, not a Cartesian product:
elementwise add, sum reduction, matmul/einsum, values-only linalg, indexing, and
metadata views; small and medium sizes; selected rank/input-count scaling; contiguous,
non-contiguous and broadcast layouts; fixed and changing metadata where applicable.
Use deterministic nontrivial values with known results/residuals. Integrate #1759
case IDs and phase/setup contracts once its export lands. Reject contract mismatches;
report non-applicable routes explicitly. Stateful QR stays in existing #94.

Measure distinct supported tiers: direct concrete fresh session, direct concrete
shared session, actual EagerTensor no-AD and active-AD calls, compatible prepared
repetition and compiled repetition. Backend/provider is separate from API tier.
Never label Tensor + CpuBackend calls as EagerTensor. Small-work tier keys use a
separate unambiguous namespace such as `concrete-fresh`, `concrete-shared`,
`eager-no-ad`, `eager-ad`, `prepared-repeat`, and `compiled-repeat`; do not inherit
the legacy `tenferro-eager` key's concrete meaning. Active AD records graph/rule
work inclusions explicitly; compilation/preparation is outside repeated execution
but measured separately. Shared-session entry/exit remains excluded only in that
named tier; fresh-session complete call includes both. Every workflow declares
calls per workflow, including ten-call chains. Results expose both workflow and
per-call units without treating a chain as one operation.

## Timing and correctness

One driver runs child processes sequentially, with calibrated aggregate duration,
independent repetitions and warmups. Time with a monotonic clock. Black-box inputs
and value-dependent outputs; no rank/shape-only consumption. Validate actual output
values outside timing for each case/tier and changing-metadata variant. Output
construction/destruction and final value inspection inclusions are explicit per
contract. Allocation counting is a separate diagnostic mode, never enabled during
latency collection. Record counts for metadata stages independently from validation.

Records include suite/problem/contract IDs, phase, setup inclusions/exclusions, API
tier, backend/provider, dtype/layout/shape, calls/workflow, aggregates, process/sample
indices, integer raw elapsed ns, iterations, normalized ns/call and ns/workflow,
correctness status and independent timing-validity status/reasons. Reports render
ns/us/ms adaptively without rounding nonzero costs to zero. Summaries include median,
IQR, process-median dispersion and declared uncertainty statistics.

Before collection freeze cases, source revisions, lockfiles, profile/provider,
thread policy, repetition/calibration protocol, validity/noise thresholds and output
location in run metadata. Preserve all raw samples, errors and failed attempts.
Declare valid/INCONCLUSIVE independently of correct/incorrect. Correctness failures
never yield successful performance cells. Comparison rejects unequal operation work,
setup boundaries, dtype/layout or resource policy. Do not selectively retry cells.

## Resources and execution command

Document one selected-suite command using existing Linux devcontainer conventions.
Use existing container tooling rather than install dependencies globally or reset
another worktree's environment. Linux collection must use allowed low-load physical
CPUs: observe per-CPU load including SMT siblings, cpuset and ancestor quota limits,
NUMA placement; select and verify affinity; record effective threads. Keep explicit
1/4-thread and default-thread policy visible; if a default policy needs more idle
resources than available, retain INCONCLUSIVE rather than quietly changing default.
User authorization to use idle CPUs and #1758 affinity requirements supersede the
old no-affinity convention for this new suite only. Other suites stay unchanged.
Never stop unrelated jobs. If resource selection cannot produce a valid set, write
a run-level INCONCLUSIVE record and do not collect timings. Benchmark #96 will reuse
this resource/provenance seam for paired change-aware gates, not duplicate it.

Baseline/candidate acceptance runs use identical resources and run sequentially.
No performance improvement is claimed by the infrastructure PR itself. Fresh runs
must satisfy frozen correctness and noise checks before use as a regression baseline.

## Outputs and tests

Raw source/provenance/run metadata, samples and failures belong under
`data/results/<target_profile>/cpu/small_work/<timestamp>/`; generated latest reports
under `result/<target_profile>/cpu/small_work.md`. Do not copy tables into overview
docs. Run metadata names benchmark/library revisions, dirty state, resolved
lock/dependencies, rustc/profile, hardware, container/provider and effective affinity
and thread settings. Extend the strict `schemas/benchmark-run.schema.json` and
`scripts/collect_run_metadata.py` together for new metadata, keeping old valid runs
compatible. Extend `tests/test_cpu_thread_metadata.sh` and result-layout tests for
these fields. Publish tracked source and complete evidence before calling any
baseline remotely reproducible.

Add focused standard-library unit tests and existing schema validation for phase/API
mislabels, ns normalization and ten-call chains, invalid comparisons, missing output
verification, malformed/empty samples, calibration failure, process dispersion,
INCONCLUSIVE propagation and retained errors. Test idle-core/cpuset/quota/SMT selection
with fixtures and affinity verification without timing tests. Reuse existing fixture
and result-layout tests; run `tests/test_suite_result_layout.sh`,
`tests/test_run_all_docs_outputs.sh`, `tests/test_tenferro_timing_scope.sh` and relevant
public-API validity tests. Add equivalent timing-boundary guards for the new runner
(the existing timing-scope test targets `src/main.rs`, not new binaries), ensuring
input creation/setup and correctness validation stay outside execution timing.
Compile/test changed Rust bins and meaningful small case
smokes before release timing. Run formatting, suite/schema validators and all relevant
repository-local gates; builds/tests use short observable watchdogs. Never lower
numerical tolerances to make cases pass. Do not treat synthetic timing fixtures as
actual performance evidence.

## Review and completion

Luna implements only after Flash approves this design. Keep the change independently
reviewable; record design and full-diff verdicts and actual commands in docs/worklogs.
Final Flash review includes complete diff and all generated artifacts. Required checks
must pass before PR creation/merge. #95 is complete only with a runnable documented
command, the full required representative surface/phase coverage, source-compatible
component integration, real provenance/raw evidence, and passing contract checks.
If dependency migration or missing hardware prevents that, keep #95 unresolved and
record the exact blocker rather than silently narrowing acceptance.
