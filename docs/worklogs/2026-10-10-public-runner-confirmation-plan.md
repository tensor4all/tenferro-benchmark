# Public runner confirmation: draft decision

The whole paid Pod lifetime must fall by at least 10%; setup-only time and
payload bytes do not complete that goal. Main already contains the official
runner and direct NVRTC installation, but the previous bootstrap comparison
is INCONCLUSIVE and its two-attempt ceiling is exhausted. Its verdict and all
samples remain unchanged.

## Concrete comparison proposed

Compare the historical startup at tenferro-rs `ff94aeded9cc98d6c49889c8ef6fca365763d93a`
with startup at current main `8dafe4202c33078f3be5c4821fc748e79b13c5ce`.
Both arms execute the same current artifacts from successful run 38051889375:
all 292 CUDA and three PJRT cases, plus the tutorial. Their exact identities
are frozen in the [draft protocol](../../result/nvidia-gpu/ci/public-runner-confirmation-draft.json).
Use the current pre-expanded runtime payload in both arms so library contents,
test code, profile and numerical checks are identical. The declared treatment
is the image/bootstrap/package preparation/readiness-polling mechanism.

This compares the startup improvement against its original baseline. It does
not claim another 10% beyond current main, nor a speedup caused by Ubuntu alone.

One independent attempt: two A/A runs and eight randomized, balanced A/B time
blocks, at most 18 fresh A40 Pods in CA-MTL-1, CUDA 12.8, $0.60/hour price ceiling
and $3 aggregate GPU ceiling. Keep cold pulls, all failures, observations and
cleanup. Require mean and median paid time to improve by at least 10%, the
one-sided exact balanced randomization p-value to be <=0.05, and candidate
maximum time to stay within 1.25 times baseline maximum. A/A spread remains
<=10%; any numerical, identity, evidence or cleanup failure stops allocation.

The proposed model records driver patches and requires between-arm driver
distribution total variation <=0.25, instead of demanding the same patch on
all 18 fresh allocations. This is a proposed change to the exhausted comparison
design, not a reinterpretation of old samples. It needs a maintainer decision
before paid dispatch. The harness still needs implementation and free preflight;
this document is a reviewable plan, not an executable or accepted experiment.

## Alternatives checked

The complete zstd-10 runtime byte screen failed its declared threshold:
[retained result](https://github.com/tensor4all/tenferro-benchmark/blob/0780d923801449073a11dd59a072b65b6262efe3/result/nvidia-gpu/ci/runtime-compression-screen.md).
No level substitution, production promotion or paid sampling follows it.

A shared Runpod network volume is not a direct replacement for the immutable
runtime artifacts. The repository requires Pod consumers to have no shared-cache
write capability. The documented Pod mount has no read-only selection, and the
[S3 API](https://docs.runpod.io/storage/s3-api) lists presigned URLs, bucket
policies and ACLs as unsupported. It also needs a separate S3 credential and
does not list CA-MTL-1 among its supported datacenters. This route would require
an additional storage/access/placement design, not just a CI cache-key change.
No volume or credential was created. The public official runner constraint and
the existing CI cache trust boundary remain intact.
