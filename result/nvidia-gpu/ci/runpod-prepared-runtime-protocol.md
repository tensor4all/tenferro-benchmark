# Prepared-runtime cloud comparison protocol

The new baseline is merged tenferro-rs #2058, commit `356a2011`. This experiment
measures another 10% reduction in whole paid Pod lifetime, with the same full
285 CUDA tests, 3 PJRT tests and tutorial. It does not retroactively validate
the previous inconclusive bootstrap campaigns.

Before any paid run, bind the published candidate image digest and exact harness
commit in the [full protocol](runpod-prepared-runtime-protocol.json). The
workflow defaults to a free artifact-only preflight, and refuses either paid
arm until the prepared image is anonymously available by immutable digest.

Use two A/A baselines, then eight balanced randomized A/B time blocks (18 Pods
maximum, one complete attempt, $3 aggregate GPU ceiling, A40 at <=$0.60/h).
Every test identity, failure, host observation, cold pull and deletion is kept.
The candidate is never warmed with an unreported Pod. Missing evidence,
allocation/compatibility/correctness/deletion failures and an A/A spread above
10% stop further allocation. A provider observation timeout never restarts a
live run.

RunPod cannot select a driver patch. This distinct candidate therefore uses a
predeclared randomized cloud-pool comparison rather than pretending all hosts
are identical. GPU, region and selected runtime are fixed; full driver and CPU
identity are recorded. Require driver-patch distribution total variation at
most 0.25 between arms. No host subgroup may replace the complete result.

Acceptance requires both mean and median paid lifetime to decrease by at least
10%, a one-sided exact balanced-block randomization p-value <=0.05, candidate
maximum lifetime <=1.25 times baseline maximum, and complete numerical/cleanup
success. The exact test enumerates all 70 balanced assignments of eight blocks.
Cold image-pull costs remain in these statistics. The goal is unproven until
this complete protocol passes; no automatic repeat is allowed.
