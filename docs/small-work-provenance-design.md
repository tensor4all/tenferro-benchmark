# #95 provenance implementation boundary

Continuation of the approved small-work design, not a new optimization.
DeepSeek V4 Flash design verdict: **Correct-to-merge**, before implementation.
Parent examined collect_run_metadata.py and build.rs. The first review attempt
was incomplete and did not approve anything; the subsequent complete single-document
review issued the approval recorded here.

## Reuse / avoid

Reuse collect_run_metadata.run_git_rev_parse, git_dirty and collect_environment
where suitable. Its resolve_tenferro_commit explicitly trusts caller overrides;
the new timing gate must additionally resolve/compare the actual checkout. Do not
change old suites' accepted metadata semantics. Do not call build_metadata blindly:
it probes Python/Julia backends unnecessary for this Rust-only preparation path.
Keep build.rs unchanged; it owns BLAS linkage and has early platform branches.
Do not infer a release binary from its filename or an asserted --features flag.

## Minimal preparation evidence

A benchmark-owned preparation operation should own the Cargo invocation and write
one versioned build receipt, using Cargo's JSON compiler-artifact event rather than
inventing embedded Rust git/version macros or a new generic build framework.
Preparation/correctness precede resource observation and all timings. It may be a
small helper in the existing small-work code and a documented preparation mode.

Capture before and after preparation:
- full benchmark and library HEADs, actual dirty/untracked state and source paths;
- tracked source membership, Cargo manifest/lock/config hashes;
- resolved dependency manifests/features from Cargo metadata, confirming all
  tenferro path dependencies resolve under the claimed library checkout;
- actual cargo/rustc identity and allowlisted build/profile/thread environment;
- exact argv, cwd, selected target/features/profile and build stdout/stderr;
- executable path, SHA-256 and the matching compiler-artifact package/target/profile
  and enabled features. Require successful build-finished and unchanged inputs.

Use an existing resolved lockfile and --locked; capture its hash, never silently
regenerate dependencies during timing. Do not store credentials or the whole
process environment. Cargo config can be hashed without copying credential data.
Tests may inject fake Cargo events but must remain explicitly synthetic artifacts.

The runner verifies the receipt against CURRENT source/lock/binary identities
before launching any measured child, records it with the frozen manifest and
protocol, and rechecks mutable inputs before promotion. Dirty/unpublished source,
missing/contradictory evidence, profile mismatch, debug-only artifacts, substituted
binaries, changed lockfiles or a caller SHA inconsistent with the checkout cannot
produce READY/latest. Record unknown explicitly, never invent `0000000` or borrow
the benchmark repository HEAD as the library revision. Correctness-only mode may
still run with unavailable provenance, retaining that limitation and no timings.

Use one shared identity collector for preparation, pre-launch verification and
pre-promotion verification, and one receipt writer. Do not independently estimate
identity in different paths. Partially unknown required state and unrecognized
potentially relevant build/profile/runtime overrides cannot produce READY. This
does not mean collecting secret values or rejecting irrelevant environment variables;
document the relevant allowlist and limitations explicitly.

## Independence from unfinished work

A verified receipt is necessary, not sufficient: complete campaign receipts,
resource/affinity/noise validity, canonical case linkage and actual semantic
contracts remain separately required. Source checksum equality does not establish
compiler correctness or numerical validity. Preserve failed/inconclusive raw
artifacts without replacing a previous latest report.

## Deterministic regression cases

Missing/non-Git checkout; contradictory explicit SHA; dirty source; missing tracked
producer; changed lock/config; wrong dependency checkout; absent/failed Cargo event;
wrong package/target; debug artifact asserted release; binary replaced after receipt;
input changed during build; stale receipt at promotion. Verify actual existing debug
correctness binary remains usable only for correctness, not valid timing evidence.

Do not start this implementation concurrently with the current campaign-receipt
worker in the same worktree. Next parent decision should keep it one bounded slice
and reuse existing collectors rather than introduce multiple overlapping metadata
schemas or registries. No performance run on the contended host is authorized.
