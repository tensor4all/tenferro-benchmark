# Next bounded #95 slice: balanced add and dependent workflows

Flash pre-implementation design review: **Correct-to-merge**, no blocking findings.
The minor filter-naming clarification is incorporated below. Do not implement until
the canonical-binding continuation is verified and this design is placed in the
benchmark worktree. This is not the full-family matrix or a frozen timing protocol.

## Source-backed problems

`small_work_case.rs::concrete_fresh` constructs CpuBackend for every call, while
shared execution retains its backend/session. Fresh/shared also currently use
n4/one call versus n16/ten independent calls. `execute_eager` and `execute_shared`
repeat additions of the same lhs/rhs; neither is a dependent chain.
`check_active_ad` correctly checks gradients1/1 only for the single-add case.
The current shared setup.includes also still mentions backend construction/session
entry although scope.outside_timer excludes them, and the eager AD workflow name
mentions backward despite backward being outside timing. Eliminate this ambiguity
before any baseline freeze, rather than reinterpret old unmeasured records.

## Proposed bounded matrix

F64, contiguous vectors, sizes4/16/256, four existing API tiers, two workflows:
- single add: one logical operation, output x+y, leaf gradients1/1;
- dependent ten-add chain: ten operations, output x+10*y, gradients1/10.

Exactly24 cases; n4 forms an eight-case local subset. Following the public_api
runner's comma-separated ID-filter convention, introduce SMALL_WORK_CASE_FILTER
in run_small_work.py's suite-loading path before binding/launch. It is new to the
small-work runner, not an already implemented option. Its value is a list of exact
scenario IDs, unset means all24; reject unknown, duplicate or empty selections.
Document the eight n4 IDs in the local command, not a second maintained registry.
Resolve selection before child launch and freeze its IDs/filter in run metadata;
validate full manifest syntax but bind/execute the explicitly selected cases.
No automatic favorable-row selection. All tiers have matching
shape/layout/dtype/operation counts for a given workflow. No independent batch is
called a chain. No ratios/timing claims in this implementation slice. C64, layouts,
other families and prepared/compiled tiers remain required subsequent coverage.

## Execution and identity

- Construct one backend per process outside timing for concrete cases. Fresh means
  session admission per ADD, including each chain step; shared means one admission
  outside warmups/calibration/all samples. Do not add a public fast path.
- Prefer one small generic chain helper accepting the add closure, shared by all
  tiers, so tests can count real closure invocations without a fake backend or
  duplicate loops. This is reuse of three existing loops, not a new framework.
- For the chain, compute the first result directly from borrowed x/y, then each next
  result from the previous result and y. Do not clone x to seed the loop. Intermediates
  feed the final result; use the existing output black-box/drop discipline. Do not
  add a full-output checksum, materialization or new transfer inside the timer.
- Eager cases use actual EagerTensor calls and existing tracked/untracked leaves
  prepared outside timing. Backward and complete output/gradient checks stay outside
  timing; drop intermediates normally so the test exercises ordinary lifetimes.
- Validate all values, exact shape/dtype, and both gradients for each size/workflow.
  Clear gradient slots after checks. Each invocation restarts from x and y, not the
  previous invocation's output; no unbounded across-sample numerical chain.
- Scenario IDs must no longer double as Rust dispatch keys. Prefer passing existing
  operation/dtype/api_tier/workflow fields explicitly to the binary, with --case as
  opaque instance identity. Require selector arguments rather than silently infer
  an implementation from an ID prefix. One existing Rust dispatch definition owns
  implemented cases; this is not a new source-operation registry/framework.
- Preserve the canonical route links introduced by the previous slice. Different
  shapes/workflows can share a canonical route while keeping explicit scopes.
- setup.includes/excludes must consistently describe timed setup, agree with scope,
  and distinguish backend creation, session entry/exit, input/leaf creation, result
  lifetime and backward. Use truthful common workflow names across tiers.
- The in-development v2 schema from canonical binding already carries these axes;
  avoid another format/framework or compatibility aliases. Update CLI docs/tests
  together, retaining earlier raw evidence as historical, not silently relabeling it.

## Verification required

Source tests must assert one versus ten operation execution and analytic outcomes,
not merely echoed --calls. Parent correctness-only matrix should run all24 actual
cases; tests reject unsupported selector tuples and workflow/count contradictions.
Verify per-tier scopes and matrix pairing, current42+binding regression tests,
Rust check/unit tests, layout/thread-policy guards and relevant lint/docs gates.
Use uv unittest, not pytest. Rebuild the real binary before end-to-end tests.
Fresh/shared backend provider/configuration must be identical and actual provider
identity must be recorded rather than guessed from a caller label. Verified existing
APIs: CpuBackend::kind() (backend.rs:2119) yields Faer/Blas; construct the same normal
CpuBackend configuration for all tiers and inspect its kind outside timing. For eager,
move that backend into EagerRuntime::with_cpu_backend (eager.rs:1200), which is the
same construction path EagerRuntime::new() already uses (1178). No new access API or
provider guessing is necessary; preserve explicit versus environment/default thread
configuration rather than forcing a different provider/thread policy per tier.

No source optimization or performance acceptance occurs until the broader matrix,
protocol/threshold/source freeze, clean verified preparation, idle physical resources
and reviewed full deliverable are ready. No new dependency or library API is needed.
