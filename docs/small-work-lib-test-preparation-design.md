# Explicit Cargo lib-test preparation receipts

Build receipts bind build-project/package/target/kind in input identity
and previous-receipt checks, enforce kind-specific test flags and workspace
membership, preserve exact selected-manifest checks, and discover config/toolchain
from the actual build cwd. These are requirements, not optional follow-ups.
This is the first, bounded prerequisite for private-probe integration, not the
component runner, component manifest, or measurement acceptance.

## Cached library artifacts after runner changes

A real Cargo regression test reproduced unnecessary cache rejection after a
Python-only benchmark commit, despite unchanged library sources and executable.
For previous-receipt reuse of a lib-test built at the library root, compare actual
artifact inputs: omit only the benchmark Git HEAD/tree hash/count from that
cross-build comparison. Keep benchmark path/cleanliness, both projects' Cargo and
config hashes, complete dependency identities, library Git identity, toolchain,
selectors, profile/features, and executable hash. An existing verified receipt is
still required for a cached artifact. Full benchmark identity is still recorded
and checked before/after the current preparation and before/after timing; this is
not permission to change runner source during a run or reuse a changed library.

## Existing seams

small_work_provenance.py already owns Cargo metadata, complete resolved dependency
identity, config/toolchain observation, compiler events, artifact/profile validation,
cached-artifact prior proof and re-verification. Reuse it; do not add a second
collector. Currently collect_identity runs Cargo/config/compiler discovery from the
benchmark root. prepare_cargo derives binary.stem and builds --bin; profile.test
is classified as debug. These assumptions are correct for the supported ordinary
binary but cannot identify a library unit-test harness.

The existing owning probe is tenferro-einsum's ignored lib-test entrypoint
concrete::probes::component_probe_entrypoint. It requires an explicit case in
correctness mode. Parent debug execution of all8 cases produced30 successful
stage JSON records, not timing samples. Its output is tagged
TENFERRO_EINSUM_PROBE_JSON and includes ordinary Rust test-harness stdout.

## Proposed minimal extension

Add explicit artifact selection to existing preparation/verification functions:
build_project, package, target_name and artifact_kind (bin or lib-test). Ordinary
CLI behavior remains bin at the benchmark root. The lib-test path selects the
library workspace and exact package/target. Avoid a new object hierarchy or generic
build plugin framework. Do not infer target identity from a hashed test filename.

Keep benchmark-harness identity separate from Cargo build-project identity. All
Cargo metadata, rustc/toolchain and inherited Cargo configuration discovery must
use the actual build cwd. Both source repositories remain identified and clean;
foreign path-dependency checks and the narrow benchmark-root exemption stay intact.
Handle the real virtual-workspace metadata root: select the exact package/target
from metadata rather than assuming resolve.root is a package or taking the first
workspace member. Retain the full resolved graph/features.

Build lib-test with cargo test --locked --no-run --lib -p <package> and the existing
explicit profile/features/target flags. Select exactly one successful executable
compiler-artifact matching package ID, lib target name/kind and source target.
Discover its actual path from Cargo JSON; do not guess target/<profile>/deps hashes,
rename a harness to a --bin target or accept unrelated test executables.

Separate harness kind from optimization. Bin requires test=false; lib-test requires
test=true. Both require known supported profile fields. Release timing requires
optimized code, disabled debug assertions and the existing debug-info policy;
unknown/debug/opt0 configurations remain ineligible. test=true alone must not make
an optimized harness a debug build. Cross-kind receipt substitution must fail.

Reuse successful-build, binary-byte identity, matching resolved features, timeout
logs/process-group cleanup and cached-fresh prior-proof checks. A cached lib-test
artifact requires a prior verified receipt for that same kind/package/target/build
project and input identity. No automatic clean/rebuild loop or relaxed cache proof.
This task never executes a produced fixture or real probe and never times code.

## Approved format decision

Preparation receipt v2 makes build-project and artifact identity mandatory. Reject
old v1 for new preparation/promotion; preserve archived v1 evidence without rewriting
or treating it as v2. Common run envelope1, small-work contract2 and library route
export1 remain unchanged. Document the narrow receipt migration explicitly; no
compatibility fallback that guesses old receipt kind. Reviewer must approve this
before edits (completed: Flash Correct-to-merge). No accepted performance baseline
currently depends on receipt v1.

## Required verification

Use existing uv unittest/jsonschema tooling, no new dependencies. Tiny REAL Cargo
fixtures (no numerical execution) must cover:
- Existing bin debug/release preparation and re-verification still work.
- Actual lib-test JSON artifact discovery for a virtual workspace/package.
- Release lib-test is eligible by observed profile; debug harness is not.
- Wrong package/kind/target, unknown profile, opt0/debug assertions, failed build,
  ambiguous artifact, changed binary and unproved cached output fail.
- Matching cached receipt succeeds; substitutions between bin and lib-test fail.
- Changed build-cwd ancestor configuration/toolchain/dependency source is detected;
  benchmark source identity is not replaced with the library's identity.
- Timeout retains logs and cleans only owned descendants; prior archives preserved.

Run existing full small-work tests, preparation real-Cargo smoke checks and relevant
local CI-helper gates. Review full diff with Flash after implementation. Actual
large release lib-test build remains a separate readiness gate; do not repeat its
unchanged cold build merely to satisfy this fixture-level task.

## Direct integration steps

Reuse `run_sequential` for a correctness-only component adapter in the existing
`run_small_work.py` CLI. Invoke the exact ignored lib-test entrypoint with one test
thread and `--nocapture`; extract the probe's tagged JSON from test-harness stdout.
Export contracts first, derive case/stage identities from that export, then invoke
correctness once per exported case. Require successful child exits and exactly one
successful record per exported case/stage. Archive contracts and all child output
under the existing raw-result layout, without publishing latest timing tables.
No copied case registry, public Rust API, new dependency, or timing claim is needed.
The existing preparation receipt can verify the selected lib-test artifact; without
one, explicitly label the correctness smoke's provenance unverified.

Allocation diagnostics reuse the same invocation/archive path after all exported
correctness stages pass: one separate child per case/stage, with an explicit positive
iteration count. Require matching case/stage, complete iteration counts, successful
child exit and nonnegative integer counters. Label caller-thread calls/requested-byte
traffic explicitly; do not call it retained memory, worker accounting or latency.
Timing mode will reuse the public suite's protocol through `--suite`, not its
public-operation case registry. Require a verified release lib-test receipt and
valid existing idle-core selection before launching the probe. Export the component
contracts and pass correctness first. For each exported stage, calibrate a separate
child with a one-nanosecond minimum, doubling iterations up to a fixed bound until
its aggregate reaches twice the protocol target (calibration headroom). Archive calibration separately. Run
independent measured children with warmup samples discarded only as predeclared;
require every recorded aggregate to meet the target, match case/stage/sample IDs,
and pass existing process-median CoV statistics. No favorable-case retry or latest
publication. A high-CoV stage stays in the report as invalid and makes the whole
run INCONCLUSIVE, but does not omit remaining stages; collection still completes
the matrix and performs final provenance/resource checks. Structural or child
execution failures still abort. Allocation and timing remain separate modes. Recheck source receipt
and resource availability after collection and restore parent affinity on all exits.
Return timing diagnostics, never claim baseline/candidate performance acceptance.

## Explicit later work

The tagged-output extractor, component contract source (distinct from public route
IDs), allocation/latency modes, calibration/sample aggregation, verified affinity,
complete evidence archive and publication remain separate reviewed work. Successful
preparation alone does not make a component or public performance campaign READY.
