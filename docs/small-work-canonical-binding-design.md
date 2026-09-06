# Canonical binding for `cpu/small_work`

## Current changed-path selection

`--changed-path` delegates to the library's existing inventory checker's selector;
no mapping table is copied into the benchmark repository. It expands each selected
canonical ID into every matching suite case, retaining layout/dtype/tier variants.
The source identity is checked around selection and again before collection.
Normal binding validation reuses the successful inventory check only while that
identity remains unchanged. Source/checker errors never fall back to manual cases.

Dry-run exposes missing contract IDs. Normal collection refuses missing or empty
selections before any benchmark child, rather than silently timing the available
subset. Manual exact-ID filtering and changed-path selection are mutually
exclusive. The generated exact-ID filter and `change_selection.json` preserve the
selection in the run archive and prevent replacing a full-suite latest report.
This checks declared contract availability, not completeness of all required
family/dtype/layout variants. Paired revision orchestration and selection across
both revisions are still outstanding under #96.

## Historical four-case migration design

The remainder records the original migration: its four-case counts, pin and model
review/implementation assignments are historical, not current coverage or standing
review gates. The current suite is v2 with 148 executed correctness cases; see
`small-work-einsum-design.md` and the worklog for current verification.

**Historical review:** DeepSeek V4 Flash, pre-implementation review: **Correct-to-merge**.
The review was recorded before implementation; the four minor clarifications below
are incorporated. Luna is the selected implementer. This document is the bounded
binding slice for #95, not a redesign of provenance.

## Boundary and current state

The current suite has `contract_version: 1`, result/schema version `1`, and an
invented `small-work.v1/<case_id>` contract ID. The Rust child dispatches exactly
four cases. The library PR1767 revision is
`aa68a0d7703b8f38700f9f3234e8f06ed757fb05`; it changes no Rust/Cargo source
relative to the old `0457a2ed0aeea21b14f4297f7f4731e09b3a0507` benchmark checkout.
The generated canonical export contains 181 route contracts. Its
`baseline_revision` is historical survey metadata, not checkout identity.

Only the owned, clean `extern/tenferro-rs` checkout may be pinned, and only to the
merged SHA above. Use the existing setup convention with
`TENFERRO_REF=<sha>`. Do not change other externs, `build.rs`, Cargo/library code,
legacy schemas, numerical paths, timing protocol, or provenance semantics. Reuse
the existing collector and identity/provenance helpers.

## Contract and wire change

Advance only the dedicated small-work suite contract, result schema, campaign
JSON envelope, and Rust-child wire to **version 2**. The common `run.yaml` envelope
retains benchmark-run schema version1, with its `small_work.contract_version`2
and canonical-binding metadata recorded inside the extension. Preparation receipts,
canonical library v1 exports, and legacy/resource-only schemas retain their versions.

Every manifest case and result must require and preserve:

- `contract_id`: the canonical route ID (never `small-work.v1/...`);
- `family`: canonical family descriptor;
- `surface`: canonical public surface descriptor;
- existing `id`/`case_id`, operation, phase, tier, setup, scope, and workflow fields.

Validation requires exact manifest/result equality for all three new fields and
rejects unknown, missing, or invented contract IDs, duplicate exported definitions,
and duplicate scenario IDs. Multiple scenarios may share a canonical contract ID.
Shapes, dimensions, calls,
instance IDs, and setup boundaries remain benchmark-owned and are not canonical
route identity. No fallback alias or rewrite of old raw files is allowed.

The four existing cases map as follows:

| Rust case | canonical `contract_id` | family | surface |
|---|---|---|---|
| `add_f64_concrete_fresh` | `core.add.ordinary.concrete` | `core` | `concrete` |
| `add_f64_concrete_shared` | `core.add.ordinary.concrete` | `core` | `concrete` |
| `add_f64_eager_no_ad` | `core.add.ordinary.eager` | `core` | `eager` |
| `add_f64_eager_active_ad` | `core.add.ordinary.eager` | `core` | `eager` |

The Rust dispatcher must emit `family` and `surface` from its case-selection
descriptor (`descriptor_source: rust_case_selection`), alongside operation,
phase, tier, and the other descriptor fields. The Python driver verifies those
fields against the manifest and never treats a caller-supplied contract string
as measured identity. Different shapes, setup boundaries, or ten-call repetition
remain explicit and are not a balanced comparison.

## Binding verification and publication gate

For the selected library checkout, read the existing files at their fixed paths:

- `docs/internals/public-boundary-benchmarks.json` (canonical route export);
- `docs/internals/public-boundary-overhead-inventory.json` (maintained overhead
  inventory).

Require the existing export schema, unique case IDs, and one matching selected
contract. Match family, operation, surface, and phase. In the overhead inventory,
require one unique selector for that family/operation and reject `unsupported` for
the exact `(family, operation, surface)`; a different surface is not evidence.
`follow-up` means optimization follow-up, not absent implementation. Pending is
a route declaration, never performance evidence. Do not create aliases for
prepared, compiled, or private probes.

Before any child launch, invoke the library's existing inventory checker in
read-only mode (never `--generate`) with a hard **30-second** timeout and captured
stdout/stderr. Missing data/tool, timeout, checker failure, schema failure,
ambiguity, or mismatch makes binding unverified. Do not copy discovery/rendering
logic into this repository.

Freeze in run metadata/result provenance: both canonical JSON paths and SHA-256
hashes, selected canonical IDs, and the actual library checkout identity from
the shared collector (HEAD, dirty state, and source path). Do not use the export's
survey baseline as identity. Recheck both hashes, selected IDs, and actual
checkout identity immediately before child launch and again after measurement and
affinity restoration. A changed input cannot replace `latest`.

Binding failure persists an actionable `INCONCLUSIVE` result, launches no measured
children, and publishes no latest report. Invalid correctness-only setup exits
nonzero without calling it numerical failure or running mismatched cases.
Synthetic fixtures remain test-only and never performance evidence. Existing
provenance collection is imported/reused; no second provenance protocol is added.

## Test scenarios and exact verification

Focused tests must cover: genuine current four-case binding; unknown and duplicate
IDs; operation, family, surface, and phase mismatch (including eager-to-concrete);
unsupported exact routes; missing, stale, or changed export/inventory; checker
failure and 30-second timeout; changed inputs before publication preserving the
previous latest; Rust descriptor mismatch; correctness-only no-child behavior;
and version-2 rejection of version-1 suite/result/child payloads. Also validate
that the two concrete cases and two eager cases retain their independent setup
boundaries and that `core.add.ordinary.eager` is not treated as concrete.

Record and run, sequentially, with no timing or cold release build:

```text
uv run python -m unittest discover -s tests -p 'test_small_work*.py'
cargo test --locked --no-default-features --features cpu-faer --bin small_work_case
cargo check --locked --no-default-features --features cpu-faer --bin small_work_case
cargo build --locked --no-default-features --features cpu-faer --bin small_work_case
uv run python scripts/run_small_work.py --suite benchmarks/cpu/small_work.yaml --case-binary "$PWD/target/debug/small_work_case" --correctness-only --results-root /tmp/95-canonical-check --output /tmp/95-canonical-check.json
bash tests/test_suite_result_layout.sh
bash tests/test_cpu_thread_metadata.sh
bash tests/test_tenferro_timing_scope.sh
```

The correctness binary is rebuilt with the exact `cargo build` command above
before the four-case correctness-only run. Any failure remains a blocker; no
performance collection is authorized in this slice.

## Worklog / review record

Flash's pre-implementation **Correct-to-merge** verdict is recorded here before
Luna's implementation. The implementation diff must receive a separate full-diff
review before merge. The commands actually run for this docs-only checkpoint were
`codegraph init`, `git status --short`, `git diff --check --
docs/small-work-canonical-binding-design.md`, and read-only inspection of the
pinned revision's export/inventory paths; the diff check passed. The verification
commands above are recorded for the implementation worklog and are not claimed
passed by this checkpoint. No commit, PR, push, or unrelated worktree change is
part of this slice.
