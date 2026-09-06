# Issue 95: small-work suite

## Public timing uses the shared passive observer

Public-suite timing now enables run_sequential's existing bounded, read-only
sensor observer, already used by component timing. commands[].machine_observations
is retained in run.json; operation samples/statistics remain unchanged.
Correctness-only and dry-runs do not observe. No new polling implementation,
thread, service, privileged requirement or host setting is introduced.

Python99 ran:98 passed,1 existing live-resource smoke skipped. Tests verify public
timing opt-in and archive forwarding, correctness-only opt-out, existing sensor
parsing/limits/failure cleanup and dry-run no-launch behavior. These are contract
checks, not a new release campaign or isolation proof. The124-case numerical
producer is unchanged. Do not pool future observed runs with old unobserved runs;
#96 paired orchestration, selection union and competing-load validation remain.

## Public campaign noise is not structural failure

Before adding #96 paired orchestration, corrected the existing public runner's
CoV handling. Complete but noisy campaigns retain their samples/statistics and
return INCONCLUSIVE without replacing latest; the formatter hides unusable
values. Structural, child or correctness errors remain FAILED, including mixed
noise/failure cases. The frozen thresholds and statistics implementation are
unchanged. Exit0 for INCONCLUSIVE follows the existing resource policy and is
not acceptance.

Regression tests reproduced the previous FAILED classification and exception.
Focused tests now cover raw/results.jsonl retention, suppressed report values,
unchanged latest, valid publication, malformed/missing/non-finite metadata and
mixed failures. Python99 ran with98 passed and one existing live-resource smoke
skipped. No Rust source or producer changed; prior124 numerical cases remain
applicable to the unchanged producer, not new performance evidence. No paired
CLI, selection union or during-run contention claim is made by this correction.

## Representative indexing gather

Added six concrete fresh/shared gather cases for4/16/256 F64 data elements with
I64 index tensors. Public TensorIndexing::gather on BackendSession is the actual
API, not Tensor::gather or an eager alias. Non-monotonic repeated indices retain
output order, checked by a host indexed lookup. Tests preserve both inputs and
reject floating indices. Per-call GatherConfig construction follows public_api's
existing workflow and is explicitly timed as index_config_construction; this
caller cost is not attributed to library-only validation.

Rust19, strict Clippy/rustfmt and all124 actual correctness/descriptor/schema
checks passed; previous118 definitions are unchanged. Python ran99 tests, with
one live-resource smoke skipped (98 passed). No timings or acceptance claimed.
Reduction and standalone metadata/view coverage, other missing scenarios, #96
workflow, required #1760–1763 evidence and final merges remain incomplete.

## #1761 required dimensions audit

Re-read current #1760–1763 issue bodies. #1761 explicitly requests n2/8/32;
the previous n2/4/16 suite alone did not meet that matrix. Added16 F64/C64 n8/n32
concrete fresh/shared and prepared setup/repeat cases. Repetition is retained as
a non-regression control, and none of the old102 cases or noisy trials is replaced.
Other operation selectors retain their previous supported sizes.

Rust18, Python98, strict Clippy and actual all118 correctness/descriptor/schema
checks passed. New real/complex tests exercise direct sessions and prepared
execution against the independent product oracle. Performance at these sizes
is still unmeasured. The audit also confirmed that the optimizer behavioral seam,
orientation/rank/N-ary/larger-work controls, #1762 normal/default-setting work and
#1763 consumer guidance/reference publication remain separate obligations.

## Real stride-zero broadcast inputs

Added six F64 n2/4/16 borrowed fresh/shared broadcast cases. TypedTensorView has
shape[n,n], strides[1,0] and only n physical elements per operand. Correctness
uses independently expanded dense matrices and the host product oracle. Positive
B column sums avoid a degenerate all-zero n4 product. No owned tensor is relabeled
as broadcast, and input/view construction remains outside timing.

Rust17, Python97, strict Clippy and all102 actual correctness/descriptor/schema
checks passed; old96 cases are unchanged. Tests verify physical length, strides
and values in fresh/shared sessions. This is no performance claim; C64 borrowed,
other families and rank/input-count/changing-metadata/default-thread scenarios,
#96 workflow and final merge obligations remain incomplete.

## Representative linalg solve

Added six F64 n2/4/16 concrete fresh/shared solve cases with the existing
TensorLinalgExt API and linalg.solve.ordinary.concrete contract. No eager alias
or new dependency was introduced. A is dense off-diagonal, nonsymmetric and
strictly diagonally dominant; B is built from a known nonuniform X by the
independent host matrix-product oracle. Correctness checks full X and A X=B.
Tests check both session paths, unchanged A/B, dominance, and a wrong RHS that
passes the solution-reference check but fails the residual check.

Rust16, Python97, strict Clippy and actual all96 correctness/descriptor/result
schema checks passed against library38fbc7a1. All previous90 case definitions
are unchanged. Linalg is now represented, but this is not full family/layout/AD
coverage or performance acceptance. Reduction, indexing, metadata/view and other
missing scenarios, #96 completion and final PR merges remain outstanding.

## Traced setup producer integration

Split trace_compile_einsum from Runtime construction and added six F64/C64
n2/4/16 compiled-setup cases. Setup includes parameter specs, trace/finish,
compilation and program lifetime, but excludes runtime/engine construction,
payloads, binding and execution. It is not compiler-only time. Correctness
executes the same produced program with rebound inputs. Repeat scopes are retained.

The clean managed extern checkout was moved from 181cbad to explicit library
38fbc7a1dc89ba5fe6c167e24556e800d20d891c, which provides the new canonical contract
and the existing planner candidate. A short-hash local fetch was rejected; the
named local branch fetch succeeded and the exact detached HEAD was confirmed.
Frozen measurement clones and unrelated user work remain untouched.

Verification: Rust15, Python96, strict Clippy, actual all90 correctness/descriptor
and result-schema checks passed against this library source. Previous84 suite
case definitions are preserved. The old owned-F64 test initially included the
new compiled setup cases; its selector now excludes both compiled tiers, covered
separately, and the full suite passed after correction. These are not timings,
release readiness or performance acceptance. Remote publication, C64 borrowed,
other families, paired orchestration and remaining #96 requirements still remain.

## C64 AD recording and numerical gradients

Added n2/4/16 C64 eager-ad cases using the existing public recording path.
Backward remains outside timing. Correctness checks complex gradients against
conjugated analytic sums under the documented Hermitian convention and clears
grad slots. A separate n2 central-difference test perturbs every real/imaginary
component of both operands against the independent matrix-product oracle.

Verification: 15 Rust tests, 95 Python tests, strict Clippy and all 84 actual
correctness/descriptor/schema records passed; previous 81 cases are unchanged.
An initial compile error identified that grad() returns an EagerTensor, corrected
by explicit to_tensor() outside timing. No library AD rule was changed. C64
borrowed layouts, other families, compile costs, accepted performance and final
PR merges remain outstanding. This is not a backward performance measurement.

## Passive machine observations for component diagnostics

Component timing now opts into bounded read-only machine observation in the
existing serial runner. Available selected-CPU frequencies and thermal/hwmon
readings are archived with monotonic read bounds, missing-value errors, explicit
boundary-only coverage for short children, and a 1024-sample cap. Polling does not
reset the original child timeout; unexpected observer failures terminate/reap the
owned child and preserve output. No new thread, service or host setting is used.
Ordinary public timing remains unchanged; during-run competing-load validation
and full #96 acceptance are not implemented by this slice.

Verification: all 94 Python tests passed, followed by seven focused observation
tests after strengthening complete-output assertions. Tests cover read units,
missing/invalid sensors, timeout non-reset, bounded history, short-child coverage,
real stdout/stderr preservation and owned-child cleanup on observer failure.
A real read-only sleeper smoke captured interior observations, 64 readable CPU
frequency values and 11 thermal/hwmon paths. Those all-CPU snapshots took about
105ms each, so the effective cadence is coarser than the 100ms wait interval.
It was not a numerical benchmark or performance evidence. Sensor-read overhead
and lack of per-sample synchronization
are documented. Frozen measurement clones and all timing thresholds are unchanged.

## C64 eager and compiled execution

Added six n2/4/16 C64 eager-no-ad/compiled-repeat cases. Compiled input specs use
the actual input dtype, without literals or dtype conversion. The same program
is checked on original/swapped/original complex inputs and rejects a real input
in a complex slot. Eager inputs are constructed as complex tensors rather than
silently taking the old F64 fixture branch. Public execution APIs are unchanged.

Verification: 14 Rust tests, 87 Python tests, strict producer Clippy and actual
correctness/descriptor checks for all 81 cases passed. Every record validates
against the result schema. C64 AD/borrowed, other families, separate compile/setup
cost measurement and accepted performance comparisons remain outstanding. No
host settings or timing thresholds were changed; these are correctness checks.

## C64 prepared setup and execution

Added six n2/4/16 C64 prepared-setup/repeat cases using the existing producer
branches and ConcreteEinsumPlan API. Setup correctness executes the plan, while
repeat keeps preparation outside its timer. Tests check original/swapped/original
complex inputs on one plan and reject wrong dtype/rank. No new execution adapter,
canonical alias, or library API was needed.

Verification: 13 Rust tests, strict producer Clippy, 86 Python tests, and actual
correctness/descriptor checks for all 75 suite cases passed. Every record validates
against the result schema, including setup's `not-applicable` provider. The first
suite assembly copied duplicate YAML anchors and was rejected before execution;
the six new cases were regenerated without aliases, preserving the old 69 cases.
These remain correctness-only checks, not setup timing or performance acceptance.

## Public C64 concrete einsum

Added six C64 fresh/shared concrete cases at n2/4/16. Nonuniform real and imaginary
input components exercise ordinary complex multiplication, not conjugation or a
real surrogate. The independent triple-loop oracle now serves F64 and C64 with
standard scalar arithmetic; a known n2 complex answer anchors it. Correctness
checks both components, exact dtype/shape and finiteness. Reference Tensor
construction stays outside timing; the numerical public execution path is reused.
C64 eager/AD/borrowed/prepared/compiled tiers remain explicit remaining coverage.

All 69 actual cases passed correctness/descriptor checks and result-schema
validation. Rust: 12 tests; strict producer Clippy passed. Python: 85 tests, one
skip. The full Python run found stale mock targets after the prior snapshot-import
fix; tests now patch the reference actually used by the runner. Campaign failure
tests also use executable-shaped fixtures with a verified mock receipt, assert
child collection is reached, and include a valid control so unrelated earlier
failures cannot make the negative cases pass. No production gate was relaxed.
No timing, release readiness or baseline/candidate acceptance is claimed.

## Public compiled einsum repetition

Added three F64 compiled-repeat cases at n2/4/16 via existing TraceContext input
slots, ordinary einsum tracing, default GraphCompiler and Runtime::run_compiled.
The same compiled graph is checked with original, swapped, then original inputs
against independent triple-loop references, excluding constant/stale-result
benchmarks. Trace/compile/runtime setup and the binding array are outside the
repeat timer; runtime admission, execution and output lifetime are inside.
Separate compile/setup-cost measurement and C64 compiled coverage remain required.

Verification: 11 Rust tests, strict producer Clippy, and 83 Python tests passed;
all 63 actual producer cases passed correctness/descriptor matching. The now
complete concrete.rs canonical selection exposed a leftover local import that
shadowed verify_canonical_snapshot before execution. Removed it and added a
regression; five focused tests passed. The real selected path then executed all
39 einsum cases successfully. Both 63/39 collections validate against the result
schema. No timing or release-readiness claim is made. The unchanged numerical
paths retain the all-case evidence; the selector correction was checked through
its real newly reachable successful execution path.

## Canonical changed-path selection (#96)

Added repeatable `--changed-path` for ordinary-suite dry-run and collection. It
calls the existing library inventory selector, retains every suite variant of
selected canonical IDs, and archives the source identity and missing IDs. It
does not implement another mapping registry. Missing/empty selections are
INCONCLUSIVE and normal collection launches no partial benchmark matrix. An
explicit case filter cannot silently narrow changed-path coverage. This is not
yet a paired-revision command or proof of the complete required variant matrix.

Verification: the full small-work Python suite passed 81 tests; following path
normalization and a source-race regression, eight binding and four dry-run tests
passed. Real `concrete.rs` selection identifies 36 available einsum cases and the
missing `einsum.einsum.prepared.traced` contract. Dry-run reports this; a normal
correctness-only request refuses execution with zero commands. The unmapped
`planning/tree.rs` path conservatively selects all 184 library contracts, exposing
178 missing suite contracts instead of silently selecting the available subset.
No measurements or host configuration changes were made. Current numerical and
layout evidence is unchanged because no producer/Rust code changed in this slice.

## Borrowed public einsum layouts

Added 18 actual F64 borrowed-view cases (fresh/shared sessions, n2/4/16,
column-major control/row-major/padded-strided) to the existing producer. Owned
Tensor remains column-major; no relabeling, materialization before the public
read call, new library API or replacement registry is used. Padding is NaN and
Rust tests assert the physical view strides/offset and all output values.

Verification: 10 Rust producer tests, strict producer Clippy, and 78 Python tests
(one skipped). The actual producer passed correctness and descriptor matching for
all 60 cases. After the stride-construction correction, all 18 affected borrowed
cases passed again; the old 42 cases are unaffected. Both output collections
validate against the result schema. These runs are debug correctness checks,
not release readiness, timing evidence, or full family/dtype/compiled coverage.
The filter accepts exact IDs, not globs; the rejected glob attempt did not run
any benchmark and was replaced with the explicit 18-ID selection.

## Ordinary-suite dry-run (#96)

The existing CLI now supports `--suite ... --dry-run`, reusing suite validation,
case filtering and live resource selection without building, validating a binary,
changing affinity or running correctness/allocation/timing commands. It reports
unchecked readiness requirements explicitly and preserves INCONCLUSIVE resources.
Selection remains manual; automatic shared-path mapping and a maintained paired
revision command are still outstanding.

The small-work Python suite ran 76 tests with one skip; after adding the invalid
filter test, all three focused dry-run tests passed. A real Linux preview listed
42 case IDs, valid resources and zero commands; those IDs are not executed tests
or measurements. No latest report was published. Issue #96 explicitly prohibits
requiring sudo: a governor change is not a prerequisite, and no host setting was
changed. During-run passive contention/frequency observation is still missing.

## Component timing integration and newly available resources

Latest correction: the verified devcontainer release at library `e120bbf` consumes
full parser output through `black_box`; its 181 tests passed. The first 1T pilot
stopped after 11 stages because a C64 parse sample lasted 998947ns, below the
unchanged 1000000ns minimum. No partial timing is accepted. Calibration now seeks
twice the minimum for every stage; under-duration aborts are INCONCLUSIVE. A real
Cargo regression test also exposed needless cached lib-test rejection after a
runner-only commit; the cross-receipt comparison now distinguishes Rust artifact
inputs from runner Git identity, while retaining current-run identity and all
library/config/dependency/binary checks. All 73 small-work Python tests pass.

- Connected component timing to the existing suite protocol, release receipt,
  idle-core selector, child affinity check, calibration, independent-process
  statistics, and raw archives. Allocation and timing are mutually exclusive.
  Source/receipt and resource eligibility are rechecked after collection; parent
  affinity restoration is covered on success and failure.
- Python small-work tests: 72 run, one skipped, no failures. This verifies the
  runner, not a measured baseline/candidate performance claim.
- After the user stopped background workloads, fresh two-second observations
  selected valid 1-thread `[0]` and 4-thread `[0,1,2,3]` configurations. The earlier
  resource-unavailable observations remain historical evidence, not current state.
- Integrated library `1fa2cbb9` release lib-test built successfully in 6m15s.
  The first command failed because the new worktree lacked its ignored lockfile;
  manifests were identical to the old probe worktree, whose lockfile was copied
  before retrying with `--locked`. No profile or timing threshold was relaxed.
- Optimized lib-test: 181 passed, one ignored. Real component smoke: 30 correctness
  records and 30 allocation diagnostics. The smoke deliberately has unverified
  preparation provenance and is NOT performance acceptance.
- Full matrix completion, clean measurement-source preparation, paired baseline/
  candidate acceptance, shared-owner optimization, final gates and both PR merges
  remain outstanding. No final PR has been opened.

## Direct-work resumption

Publication decision: one final PR per repository (two total), no intermediate
PRs. The closed tenferro #1769 changes and private probe commit are now combined
locally at tenferro candidate `1fa2cbb9`; no new PR was opened. Old mandatory
cross-model reviews in the historical notes below are superseded.

The last CLI/cache-helper delta passed the existing uv suite (64 tests, one
telemetry skip). Source inspection then reproduced correctness-only runs accepting
wrong-operation payloads or failed children when their payload claimed correctness.
The aggregate now also requires no contract/execution errors. Regression tests fail
before the fix; the runner's 34 tests and the existing debug producer's 24 actual
cases passed after it. Debug correctness evidence is not timing acceptance.

Added a small adapter in the existing CLI for the crate-owned lib-test probe:
export contracts, invoke correctness once per case, strictly match stage records,
and retain tagged output and child status. It reuses sequential execution and
optional preparation verification, does not copy a registry, and rejects timing
requests. Existing debug probe execution yielded 30 contracts and 30 successful
stage records in nine processes. Its provenance is explicitly unverified; no
fresh release build or accepted performance measurement is claimed.

Checks: full small-work suite 67 tests/one skip before the final receipt-type
check; the affected three component tests passed after it. Suite schema validation
(`--kind small-work-suite`), result-layout shell check and diff whitespace passed.
Using the generic suite schema was an incorrect command, not a schema defect.
Evidence: `/tmp/1758-correctness-status-tests.log`,
`/tmp/1758-correctness-status-real.json`, `/tmp/1758-component-tests.log`,
`/tmp/1758-component-smoke.json`, `/tmp/1758-result-layout.log`.

Three fresh one-second resource observations, using the existing 20% busy
threshold and all-SMT-siblings policy, all returned INCONCLUSIVE: no physical
core had idle telemetry for every sibling. Evidence:
`/tmp/1758-resume-resource-observation.json`. Do not relax this threshold or retry
unchanged timed/release work on the contended host. Timing adapter, release readiness, remaining coverage, valid comparisons and
final gates/merges remain unfinished.

The component adapter now also supports separate allocation diagnostics via
`--allocation-iterations`: correctness first, then one child per exported
case/stage. Four focused tests passed, including partial counts, invalid/negative
counters, duplicate/wrong records and failed children. The existing debug probe
produced 30 correctness and 30 allocation records successfully with two iterations.
`/tmp/1758-component-allocation.json` and its copy under
`.artifacts/issue-1758/direct-resume/` retain the result. Its status is
ALLOCATION_DIAGNOSTIC with unverified provenance, not performance acceptance;
caller-only allocation calls/requested traffic exclude setup and workers/native
allocations. No timed run or new PR was started.

## Public einsum correctness expansion

Added the 18-case F64 square-matrix matrix described in
[small-work-einsum-design](../small-work-einsum-design.md), preserving the 24 add
cases. Ordinary concrete/eager calls, plan setup and prepared execution have
separate scopes and canonical IDs. Independent dyadic reference values and full
analytic eager gradients passed at dimensions 2/4/16. Setup does not claim a
backend provider or session admission; repeat retains public revalidation.

Fresh checks: nine Rust unit tests, debug producer rebuild, all 42 real cases,
selected 18 einsum cases, selected raw metadata/result schema validation,
strict producer Clippy and rustfmt passed. Python suite: 70 passed. Schema tests
now generate their own synthetic metadata rather than loading historical files
from `/tmp`; this removes a checkout-reproducibility defect without treating
synthetic data as benchmark evidence. Logs: `/tmp/1758-einsum42-rust.log`,
`/tmp/1758-einsum42-python.log`, `/tmp/1758-einsum42-clippy.log`,
`/tmp/1758-einsum42-correctness.json`, `/tmp/1758-einsum18-filter.json`.
The first schema-validator import attempt omitted its script path; invoking its
supported CLI validated the actual records successfully. No timings or new PRs.

## Recovery and current acceptance status

A worker reported overwriting earlier uncommitted worklog prose. The surviving
checkpoint below was preserved before recovery. This section reconstructs verified
milestones from the parent evidence ledger/logs; it is not a verbatim restoration.
No complete suite, performance readiness, final full-diff approval or PR is claimed.

- Initial design: `../small-work-design.md`, Flash Correct-to-merge before Luna
  implementation. Starting benchmark commit6005fe6 records the initial design;
  subsequent infrastructure remains uncommitted work under that task.
- Provenance refinement: `../small-work-provenance-design.md`, Flash approval
  before implementation. Parent real-Cargo contract fixtures verified preparation,
  debug timing rejection, cache reuse with prior proof and substituted-cache
  rejection. Parent discovered and then verified fixes for mutable foreign path
  dependencies and ancestor Cargo configuration. These fixtures executed no
  benchmark binary and produced no performance samples.
- Source inspection and explicit binary rebuild resolved stale diagnostic APIs
  and task-vanishing handling. Parent four-case correctness-only execution passed;
  active AD timing is forward recording only, with backward checked outside it.
- Canonical binding: `../small-work-canonical-binding-design.md`, Flash design
  Correct-to-merge before code. Library contract refinement PR1767 merged as
  aa68a0d7703b8f38700f9f3234e8f06ed757fb05, with eight required checks verified.
  The181 route declarations are not181 implemented benchmarks.
- Earlier45-test success did NOT establish binding safety: existing campaign tests
  mocked binding. Parent found missing negative tests, placeholder canonical IDs,
  incomplete identity/freshness checks, failure archival/status defects and missing
  frozen metadata. Rework is being verified against these findings.
- Generic run.yaml metadata retains envelope version1, with small_work contract2;
  dedicated small-work campaign/result/child formats use2. The earlier checkpoint's
  blanket metadata-version2 statement below is historical and superseded.

Evidence is retained in the workspace `.artifacts/issue-1758/progress.md`, including
PR/CI snapshots and parent logs `/tmp/95-gen10-{positive,foreign,cache}.log` and
`/tmp/95-gen10-correctness.json`. Source/lock/protocol freeze, balanced workflows,
remaining families/dtypes/layouts, private probe integration, actual low-load
performance evidence, final review and publication remain required. No historical
raw records are rewritten or promoted by this recovery.

## Earlier binding v2 implementation checkpoint (worker-reported)

Migrated small-work test fixtures to contract version 2 with canonical contract IDs,
family, and surface descriptors; repeated canonical references are allowed while case IDs
remain unique. Child payload fixtures now include schema_version and canonical fields.
The runner and frozen metadata emit schema version/contract version 2. Canonical snapshots
also freeze the tenferro checkout revision alongside export/inventory hashes, so stale
checkout mutation fails closed before publication.

Verification commands (all passed):
- `uv run python -m unittest discover -s tests -p 'test_small_work*.py'` (45 tests, 1 telemetry skip)
- `cargo check --locked --no-default-features --features cpu-faer --bin small_work_case`
- `cargo test --locked --no-default-features --features cpu-faer --bin small_work_case`
- `cargo build --locked --no-default-features --features cpu-faer --bin small_work_case`
- `uv run python scripts/run_small_work.py --suite benchmarks/cpu/small_work.yaml --case-binary target/debug/small_work_case --correctness-only --output /tmp/small-work-correctness.json --target-profile amd-cpu --results-root /tmp/small-results --result-root /tmp/small-result --command-timeout 30`
  Result: `CORRECTNESS_ONLY`, 4 records, no errors.

## Binding runner rework B checkpoint

Implemented timestamp-root archiving for every suite-run exit (including explicit
`--output` copies), fail-closed correctness binding failures with `not_run` records,
and restored schema version 1 for common/fixture/legacy envelopes. The validated
canonical binding remains contract version 2 in frozen metadata.

Verification commands (all passed):
- `uv run python -m unittest discover -s tests -p 'test_small_work*.py'` (50 tests, 1 telemetry skip)
- `cargo build --locked --no-default-features --features cpu-faer --bin small_work_case`
- `uv run python -m py_compile scripts/run_small_work.py`
- `bash tests/test_suite_result_layout.sh`
- Real correctness campaign command above (`CORRECTNESS_ONLY`, 4 records, no errors); archive `run.json` was created under the timestamp root.

## Binding provenance receipt integration checkpoint

Frozen `run.yaml` tenferro identity now comes from the verified canonical binding checkout; claimed `--tenferro-commit` mismatches fail before child launch. Parsed Cargo receipts are archived verbatim under each timestamp root with SHA-256/file references in `run.json` and `run.yaml`; receipt build features drive metadata rather than caller flags.

Verification: `uv run python -m unittest discover -s tests -p 'test_small_work*.py'` (50 passed), `cargo build --locked --no-default-features --features cpu-faer --bin small_work_case`, and real four-case correctness-only campaign (4 records, no errors).

## Balanced add/dependent workflow implementation

Implemented the approved bounded matrix: 24 F64 contiguous cases across four API tiers, sizes 4/16/256, and single/dependent10 workflows. Concrete execution now constructs one backend per process, with fresh admission per ADD and shared admission outside all timing; all tiers use the generic dependent-chain helper. Eager runtime uses the captured `CpuBackend::kind()` provider and active AD checks gradients 1/1 or 1/10 outside timing. Scenario IDs are opaque and selectors are explicit child arguments. `SMALL_WORK_CASE_FILTER` resolves exact IDs before binding/launch and freezes selection metadata; the documented filter runs the eight size-4 cases.

Verification commands (all passed):
- `uv run python -m unittest discover -s tests -p 'test_small_work*.py'` (58 passed, 1 telemetry skip)
- `cargo check --locked --no-default-features --features cpu-faer --bin small_work_case`
- `cargo test --locked --no-default-features --features cpu-faer --bin small_work_case` (3 passed)
- `cargo fmt --all -- --check`
- `bash tests/test_suite_result_layout.sh`
- `bash tests/test_cpu_thread_metadata.sh`
- Real rebuilt binary correctness-only matrix: 24/24 passed; exact filtered n4 command: 8/8 passed.

## Lib-test preparation final review fixes

`--preparation-receipt` is also the prior proof supplied to `--prepare` for a cached lib-test artifact; without it, cached preparation fails closed. The tiny real-Cargo CLI regression covers first prepare, proven cached reuse, and unproved-cache rejection without running the test executable. Shared package/target/source selection is kept in one helper for preparation and receipt verification.
