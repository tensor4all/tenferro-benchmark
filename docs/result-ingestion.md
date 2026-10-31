# Benchmark result collection, ingestion, and publication

This contract implements #216 and is the acceptance policy for #220. It does
not change operation timing boundaries or make hosted-runner measurements into
controlled performance evidence.

## Three stages

1. Collection runs the existing suites and uploads raw samples, metadata,
   generated reports and summaries. It has read-only repository permissions.
2. Ingestion selects exactly one `success` or `failure` run, validates its bundle, and
   commits accepted reports, summaries and provenance to `main`.
3. Pages renders committed `result/` with the checkout's git history. It never
   discovers benchmark runs or downloads benchmark artifacts.

The published data is a function of the checkout and its history. Rendering
must not depend on checkout mtimes, artifact availability or the current date.
Installation of dependencies and deployment still need network access.

## Trusted source and supported profiles

Only runs of the canonical repository `tensor4all/tenferro-benchmark`, on
`main`, with conclusion `success` or `failure` and one of these workflow paths qualify.
`cancelled`, `skipped`, `timed_out` and all other conclusions are rejected.
Forks and `RunPod readback (manual report)` are excluded, including manual ingestion dispatches:

| Workflow | Artifact name | Target profile |
| --- | --- | --- |
| `.github/workflows/benchmark-nightly-mac-cpu.yml` | `mac-cpu-benchmark-results-<run-id>-attempt<attempt>` | `mac-cpu` |
| `.github/workflows/benchmark-nightly-linux-cpu.yml` | `linux-cpu-benchmark-results-<run-id>-attempt<attempt>` | `amd-cpu` |
| `.github/workflows/benchmark-runpod-gpu.yml` | `runpod-gpu-benchmark-<run-id>-attempt<attempt>` | `nvidia-gpu` |

New result artifact names include the run attempt, preserving earlier raw
artifacts and their provenance links on reruns. Legacy run-ID-only names are
accepted only when created within the selected attempt; old-attempt artifacts
are not substituted.

The ingestion workflow reads run, attempt and artifact identity from GitHub's
API, not an untrusted artifact. It checks the source head repository as well
as the workflow path and branch. A rerun of an old run keeps its original
source commit. No artifact is executed, imported as Python, or sourced as shell.
Reject traversal, absolute paths and symbolic links in bundles. Do not check
out or execute the source run's code in the privileged ingestion job.

Keep `amd-cpu` as the Linux nightly's identity; do not duplicate it as
`linux-cpu`. Existing independently maintained `linux-cpu` reports remain
unchanged. CPU summaries must contain hardware identity (at least host/CPU)
and a tenferro commit. Hosted CPU results remain reference indicators. GPU
publication requires an NVIDIA A100 device name (`NVIDIA A100 80GB PCIe`,
`NVIDIA A100-SXM4-80GB`, `NVIDIA A100-SXM4-40GB`; case-insensitive whitespace
normalization is allowed). The exact model is kept in the summary and
provenance, so reports show which A100 measured them. Diagnostic hardware cannot replace standard reports.

## Bundle manifest, version 1

New collection bundles carry `ingestion.json`, with `schema_version: 1`,
`target_profile`, `harness_commit`, and a nonempty `suites` array. Each entry
contains:

- `suite_id`: `cpu/<name>` or `gpu/<name>`, using lowercase letters, digits,
  underscore, dot and hyphen, with no path traversal components;
- `timestamp`: timezone-aware ISO collection timestamp, matching summary and
  run metadata; ordering uses UTC instants with full precision;
- `tenferro_commit` and `hardware`, matching summary and run metadata;
- `expected_rows`: unique `[case, group, backend]` summary identities (empty
  group is explicit), derived from that collection's selected workload and
  thread/backend policy, not merely copied from whatever successful rows exist;
- `files`: relative bundle paths mapped to lowercase SHA-256 digests, covering
  the canonical `result/<profile>/<suite>.md` and `.summary.json`, the per-run
  report and summary, metadata, raw records and declared workload inputs.

The manifest identifies each suite's `run_directory` under
`data/results/<profile>/<suite>/<timestamp-directory>/`. Reports and summaries
must be byte-identical to their per-run copies. All digests are checked before
any repository writes. Summary schema 1 or 2 is required. Rows must match declared
identities exactly with no duplicates; `ok` rows have finite nonnegative medians
and the existing runner/formatter verification result. This does not add a
numerical parity guarantee beyond that runner. `unsupported` rows and declared
policy `skipped` rows have a raw-data explanation and no median. Legacy
`unsupported: <reason>` statuses remain readable. Skips never count as coverage.
Explicit `failed`, `compile_failed`, `runtime_failed`, `verification_failed`,
`oom` and `not_configured` rows are accepted as failures, never as measurements:
they must have no median or IQR. Missing/unexpected rows and unknown statuses
still reject the bundle. Summary documents must pass the existing JSON schema.
A suite with a complete set of failed rows can publish its failure annotations
without any timing values; it does not suppress other suites' valid results.
CPU suite-specific selectors/thread counts must
be checked against collection declarations, and raw-to-summary identities and
values checked using the appropriate formatter/record adapter.

Existing GPU `publication.json` schema 2 remains supported through a strict
adapter: revalidate its digests and exact expected raw identities in the ingestion adapter,
preserving verification and measured-sample checks on `ok` rows and rejecting
numeric measurement metrics on failed raw rows before summary normalization,
require summary identity/value consistency, matching run metadata and exact
standard hardware, and construct the same internal entries. A legacy GPU
bundle without a summary is rejected explicitly. A failed run without a result artifact can recover from its exact named diagnostic
artifact as described below. Backfill never starts a fresh timing run.

## Durable accepted-result provenance

Each accepted suite has `result/<profile>/<suite>.provenance.json`, schema 1.
It stores `target_profile`, `suite_id`, `collected_at`, `harness_commit`,
`tenferro_commit`, `hardware`, `source` and `files`. `source` contains:

- `repository`, canonical `workflow_path`, `head_branch`, `head_sha`;
- integer `run_id`, `run_attempt`, `artifact_id`;
- `run_url`, `artifact_url`, artifact `created_at` and `expires_at` from the API;
- artifact archive digest when provided by GitHub, otherwise an explicit null.

`files` retains the validated bundle-relative SHA-256 map, including raw data.
Do not store credentials, signed temporary download URLs, the entire raw
bundle, or acceptance wall-clock time. Source and measurement provenance are
separate: the harness commit is the verified source run's head SHA; the
measured tenferro commit comes from collection metadata.

New receipts add `status` (`complete` / `partial`), `case_outcomes` and
`missing_suites` without changing schema 1 or rewriting older receipts.
`case_outcomes` has `failed`, `skipped` and `unsupported` objects, each with
`count` and sorted `cases`. Counts are distinct case names, deduplicated across
backends and thread groups within that suite; a case may occur in multiple
categories. The original rows preserve the backend/group/status details.
`partial` means at least one non-ok case, an unproduced suite, or a source
workflow with conclusion `failure`. Source also records the API `conclusion`,
`artifact_name`, `artifact_kind` (`results` / `diagnostics`) and `event`.
A workflow failure outside case collection does not invent failed cases.

Pages shows nonzero outcome counts and collapsible case names, and explicitly
labels source-workflow failure and unproduced suites. Zero-count notes are
omitted. An old report retained for an unproduced suite is labelled against the
latest accepted collection's coverage. The `updated` line keeps its original
revision/time meaning; timing values, ratios and raw reports are unchanged.

Illustrative accepted CPU and GPU receipts are in
`docs/examples/result-provenance/`. Their IDs and digests are placeholders,
not measurement evidence.

## Ordering, partial bundles and atomicity

The publication key is `(target_profile, suite_id)`. Compare the tuple
`(collected_at UTC instant, run_id, run_attempt, artifact_id)` lexicographically.
Collection time is primary; completion/ingestion time and artifact search
order never determine freshness. Break identical timestamps deterministically
with source IDs. A newer attempt does not make older measurements newer.

Validate every entry in a bundle before changing any file. Then decide each
suite independently: a valid newer subset can update its suites without
removing other suites. Partial suites are allowed only when every declared case
has an explicit result row (including failures/skips/unsupported); absent case
rows still fail. Duplicate suite entries fail. No case is dropped, interpolated
or zero-filled. A few failed cases do not block publication.

An identical source key and digest map is a duplicate no-op. The same source
key with different content is an error. Older valid entries are stale no-ops.
A legacy committed summary without provenance supplies a timestamp floor:
a candidate must be strictly newer; equal timestamps preserve the legacy
report. A malformed legacy summary fails explicitly rather than allowing an
unchecked overwrite. Reports without summaries remain until a validated
candidate replaces them.

Serialize ingestion writes using a single workflow concurrency group, with
cancellation disabled and `queue: max` (up to 100 pending runs). The default
single-pending queue would replace earlier pending ingestion requests; it is
not suitable here. Queue overflow is visible as a canceled run and can be
recovered by manual run-ID dispatch. Before committing, fetch current `main`, reevaluate
ordering against that tree and replay only the accepted report, summary and
provenance paths. `main` requires status checks, so the result commit is
pushed to a driver-owned `ingest/run-<id>-attempt-<n>` branch and merged
through a pull request once the required checks pass, pinned to that exact
commit (squash, so `main` gets one result commit). If `main` moved so the PR
cannot merge, fetch and revalidate/retry with a bounded limit; never push or
force-push `main` and never drop unrelated changes. Zero accepted
entries means no commit. Each result commit names the source run and attempt.
A failed bundle never produces a partial commit.

## Publication triggers and permissions

Ingestion has `actions: read`, `checks: read` and `contents: write`. Its
result commits are restricted to the three accepted file types. The
`ingest/` branch push and the pull request use a token of the
tensor4all-ai-bot GitHub App (`INGEST_APP_ID`/`INGEST_APP_PRIVATE_KEY`,
contents and pull-requests write), because a pull request opened with
`GITHUB_TOKEN` starts no workflows and its required checks would never
report. No arbitrary artifact path may
be staged. After ingestion explicitly send a repository-dispatch event with
an ingestion event type to trigger Pages; ordinary `GITHUB_TOKEN` pushes do
not trigger another push workflow. Retry ingestion after a dispatch failure:
duplicate acceptance still requests publication, allowing recovery after the
commit succeeded. Repository dispatch uses the existing contents-write
permission; Pages itself needs only contents-read plus Pages deployment
permissions. Renderer changes and relevant result/dependency changes also
trigger Pages directly; manual dispatch is available for recovery.

## Retention and reproducibility limits

New CPU/GPU result artifacts use 90-day retention. Legacy GPU artifacts may
have the earlier 30-day retention; always use the recorded API expiry. Provenance links point to the
exact run/artifact, require GitHub access, and may cease to download after the
recorded expiry. Show that retention limit without consulting the network or
clock at render time. Never fall back to a different run when a raw artifact
expires. Digests and committed reports remain available; raw reanalysis after
expiry is not promised. Durable raw archival is separate work.

Keep existing committed reports as the initial state, including reports
without summaries or provenance. Label missing provenance as legacy rather
than inventing metadata. Git history retains earlier accepted summaries for
regression baseline selection; no historical report copies are maintained.

## Rollout and recovery

Land and test the contract, ingestion implementation, then repository-only
Pages implementation. Keep current publishing enabled until ingestion is
available. Verify a controlled ingestion from an existing valid artifact and
its Pages publication before declaring #219 complete. If no valid artifact is
available, report that explicitly and keep migration verification pending.
No paid GPU or new benchmark collection is needed solely for this migration.

Operations must expose accepted/stale/duplicate counts and exact rejection
reasons, with nonzero exit on invalid/missing/expired artifacts. Manual recovery
selects an exact run ID; it must use the same validation path. Revert an
incorrect result commit (all three files together) and republish the committed
tree. Never hand-edit generated timing values. A permanent exclusion of a bad
source run must be explicit before ingestion is retried.

## CPU normalized bundle details

The nightly publication inventory is the ten suites explicitly declared
in `scripts/ci/cpu_ingestion_adapter.py`: einsum, cpu_ops, linalg_jvp_vjp, fft,
public_api, small_work, session_matrix, permutation, overhead_smoke and perf_issues. Existing other reports
remain maintained in git. CPU packaging is non-timing work after collection.

Combined core reports name two original runs. Packaging copies these into
`<run_directory>/inputs/<index>/`, preserves all original raw bytes and reports,
and adds an aggregate `run.yaml` plus regenerated `report.summary.json`.
`entry.sources` lists original metadata and raw input paths; `entry.threads`
declares 1 and 4. The aggregate timestamp is the latest selected source's
collection timestamp. Ingestion replays trusted formatters against every
original source; it never treats the aggregate metadata as measurement proof.
Original suite IDs for cpu_ops/linalg_jvp_vjp historically say cpu/einsum;
the adapter checks that original spelling and emits the correct summary ID.
Unsupported explanations lost by old summaries are restored from raw notes.
The overhead_smoke suite is collected once at 1T, with its own declared thread
selection in the inventory; the other nightly suites use 1T and 4T. Its raw
samples rebuild both ns/op and allocation counters. Eight steady-state rows
and two explicitly scoped session-entry diagnostics stay separate in Pages;
this trend never gates. Filtered, legacy-scope or incomplete smoke runs cannot
replace the standard nightly report.

Reports and timing values remain unchanged. New small_work/session_matrix
summaries use ns per declared operation/workflow and retain per-case scopes;
FFT retains its report's timing audit and must not be cited as compliant
operation-performance evidence.

`benchmarks/cpu/ingestion-selection.json` is an explicit versioned inventory
of the standard quick nightly's 1T/4T workload identities. It records only
row counts and SHA-256 of sorted `[case, group, backend]` identities encoded
with JSON separators `(',', ':')`; no timings/statuses influence it. Collection
and ingestion check it independently of artifact-supplied expected rows, so
removing both raw and summary rows cannot silently redefine completeness.
Update this inventory deliberately when the suite's selected workloads change,
with fixture evidence for the intended selection. Custom CPU filters or thread
counts are diagnostic and cannot replace these standard nightly reports.
Selected complete suites may be ingested independently; a missing workload
inside a produced suite is rejected.

Only explicit source paths named by a report are selected. Permutation's ISO
timestamp identifies exactly one metadata-matching run when its report does
not name a path. No newest-directory fallback is used. Results artifacts without
`ingestion.json` are rejected. The explicit same-run diagnostic recovery below
uses the original API-selected raw artifact and preserves its source revision;
an unrelated locally prepared bundle is never substituted.
Failed collection diagnostics use a separate artifact name.

## Packaging failed collections and recovering old runs

CPU and GPU workflows run packaging after collection even when collection exits
nonzero. They upload a result artifact only after a validated manifest exists;
`if-no-files-found: error` remains enforced. A collection with no outputs or an
attempted suite missing required files fails explicitly. GPU diagnostics and
prepared-runtime artifacts keep their existing names, contents and retention.
Collection failure remains a workflow failure; packaging does not turn it green.

`scripts/ci/prepare_result_bundle.py` is shared by live packaging and recovery.
It selects current-attempt outputs by recorded time, never stale checked-in
reports. An attempted suite must have all its metadata/raw/report inputs and
exact expected case identities. Truly unproduced suites are listed in
`missing_suites`; their older published reports remain, explicitly labelled as
not produced by this collection. CPU perf_issues is replayed once over its
explicit per-thread inputs, with every thread's identity and the trusted quick
inventory checked. No collector or case selection is changed.

Manual `run_id` dispatch uses the same validation as automatic ingestion. Prefer
the exact results artifact; if absent on a `failure` run, select only that run's
`linux-cpu-benchmark-diagnostics-<id>`, `mac-cpu-benchmark-diagnostics-<id>` or
`runpod-gpu-diagnostics-<id>`, created within the selected attempt. Expired,
ambiguous, missing or older-attempt artifacts are rejected. Runtime/build
artifacts are never substituted. GPU diagnostic recovery is restricted to
scheduled default-suite collections: custom selections cannot be inferred from
successful rows. Recovery executes only the trusted checkout's packager and
formatters, not any artifact code, and validates the resulting schema, digests,
raw-to-summary values, expected rows, hardware, revision and time window before
any commit. A malformed result artifact is rejected, not replaced by diagnostics.
Missing required raw outputs remain a rejection, not permission to fabricate them.
Existing profile visibility and `pub-` label policy are unchanged.

Source `run_started_at` is also retained. Artifacts predating the latest run
attempt and collection timestamps outside the selected attempt are rejected.
A repository-dispatch listener must be present before final Pages cutover.

GitHub concurrency queue semantics: [official documentation](https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/control-workflow-concurrency).

## Operational commands

Validate a specific existing run without committing or publishing:

```bash
gh workflow run ingest-results.yml -f run_id=RUN_ID -f validate_only=true
```

Ingest that same run through the identical validation path:

```bash
gh workflow run ingest-results.yml -f run_id=RUN_ID -f validate_only=false
```

Inspect the ingestion run's step summary for the source ID, result commit,
and per-suite accepted/stale/duplicate decisions. A rejection exits nonzero
and states the failed check. A successful ingestion requests Pages even when
all suites are duplicate/stale, so repeating the command recovers a dispatch
failure after a successful commit. To republish an already committed tree
without downloading any benchmark data:

```bash
gh workflow run publish-results.yml
```

For local rendering from a full-history checkout:

```bash
uv run --no-project --python 3.12 --with-requirements scripts/dashboard-requirements.txt \
  python scripts/generate_dashboard.py --result-dir result --output-dir _site --git-repo .
```

Render-only dependencies are pinned separately to the versions in `uv.lock`.
The generated suite pages include raw artifact expiry and link provenance JSON.
Legacy pages do not infer raw URLs. Corruption of a report or summary with
accepted provenance fails rendering rather than quietly substituting an older
report. The renderer has no artifact API permissions and never creates a
`_site/raw` overlay.

For rollback, revert the accepted result commit (report, summary and provenance
together) and dispatch Pages. Record and resolve the source-run problem before
retrying ingestion; merely reverting a result is not a permanent exclusion.
Use the ingestion concurrency queue and normal pull-request retry path. Do not bypass
validation, force-push, or manually change timing values.
