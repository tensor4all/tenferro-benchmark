# Merged tenferro integration validation

- CubeCL [PR #19](https://github.com/tensor4all/cubecl/pull/19):
  `a2adda17affd40494393a1f40d90980e1235617c`.
- Cubek [PR #13](https://github.com/tensor4all/cubek/pull/13), on the existing
  `release/t4a-0.2.0` line: `739dbfb414dffb27d44af3ec369f1e8bead9cccf`.
  This aligns runtime types, without adopting unrelated 0.3 development.
  Six local library tests/check/lint passed; its hosted WGPU job is compile-only.
- tenferro [PR #1820](https://github.com/tensor4all/tenferro-rs/pull/1820):
  head `ccf54787dce0ab2879b12daec3ef33222ac5cefc`, merged without squash at
  2026-09-19T14:08:34Z as `d7a8c60caedac9aa3e630eea793d606df02dfc09`.

## Verification

Integrated local A100 checks passed: 160 CUDA device unit tests, 99 non-device
GPU unit tests, 73 GPU integration contracts, 29 CUDA linalg tests and five
CUDA storage identity/binding tests. The standalone CubeCL sample compile-check,
fast PR gate (including CPU extension tests), CUDA/WebGPU all-target Clippy,
and committed-head deterministic rules review also passed. No external AI
review was requested or used. Cargo resolves one CubeCL common revision.

All eight required hosted checks passed before merge: rustfmt, coverage, docs-site, BLAS
injection, PR workspace tests, GPU gate, repository-rules gate and macOS tests.
The branch required no separate human approval. CI configuration and sample
checks also passed; no check was bypassed.

[Successful RunPod run 35447106690](https://github.com/tensor4all/tenferro-rs/actions/runs/35447106690)
actually executed CUDA host 899/899 and device 209/209 tests, plus PJRT host
51/51 and device 3/3 tests. All three new CUDA read-path regressions executed.
The existing two-GPU/compile-only/benchmark/external-tool partitions are not
claimed as executed by these counts. GPU provisioning and cleanup succeeded.
A second actual run [35447119679](https://github.com/tensor4all/tenferro-rs/actions/runs/35447119679)
also completed successfully.

**Post-merge check-display caveat:** queued duplicate run
[35447622072](https://github.com/tensor4all/tenferro-rs/actions/runs/35447622072)
started provisioning validation at 14:18Z, after the 14:08Z merge. Its open-PR
revalidation correctly refused the now-closed PR; GPU tests were skipped and
cleanup succeeded. Its publisher then emitted a failing `CI GPU gate` at
14:19:01Z, superseding the displayed successful check. This is not an actual
GPU test failure. The required passing evidence at merge remains the runs above;
we did not overwrite check status or bypass the closed-PR safety guard. The
late-run check-publishing behavior is an existing CI bookkeeping limitation.

Earlier CI failures exposed stale sample/test revision pins, a stale generated
boundary digest and five unclassified new tests. These were corrected, with
three audited device tests and two source-only tests assigned to their respective
lanes; existing classifications and exhaustive inventory enforcement remain.
A local CI-config attempt additionally lacked jq/actionlint; after installing
the tools it passed (local actionlint 1.7.12; hosted CI retained 1.7.7).

The new CubeCL pin also contains earlier OOM-reclaim changes. Retained-address
and fixed-stream-slot invariants were rechecked; no ownership or synchronization
checks were relaxed. See [CubeCL validation](cubecl-pr-validation.md) for its
separately documented baseline f16 reinterpretation failure and Miri coverage.

Historical materialization and wakeup performance experiments predate these
final pins. They are not measurements of this integrated revision. The full
post-merge benchmark refresh follows [the declared protocol](integration-refresh-protocol.md).
