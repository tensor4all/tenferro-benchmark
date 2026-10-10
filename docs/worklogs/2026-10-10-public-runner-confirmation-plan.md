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
before paid dispatch. The implementation and free preflight are complete below;
the design remains a draft and paid dispatch stays disabled. After a maintainer
decision, record that decision and bind the final harness commit in the external
campaign protocol before starting. No old samples are reused.

## Executable preparation

The generator reads the two exact tenferro-rs revisions. Current main owns the
common test execution, runtime payload handling, setup watchdog and mandatory
deletion; only the declared startup and execution-tool preparation differ.
All 15 transfer artifact IDs and hosted digests are frozen, preventing a later
artifact replacement from changing inputs between arms. The GPU consumer keeps
read-only shared-cache access. Every source change remains on this experimental
branch; production CI is unchanged.

Six estimator tests cover improvement, no improvement, unfavorable cold pulls,
driver imbalance, missing samples and A/A noise. Shell/workflow checks verify
unchanged numerical execution, mandatory deletion, the unapproved-design guard,
and refusal to overwrite an existing campaign state. An observation failure
continues polling its existing run rather than allocating a replacement.

[Hosted free preflight 38054769719](https://github.com/tensor4all/tenferro-benchmark/actions/runs/38054769719)
passed artifact access and inventory checks; every GPU lifecycle job was skipped.
Both declared images passed native checks in disposable local containers with
the exact current runtime payload: runner, Cargo, nextest, the real NVRTC
installer/package version, CubeCL header compilation, cuTENSOR loading and PJRT
components. No runner registered and no GPU was allocated. This is functional
preflight evidence, not GPU numerical validation or a timing comparison.
[Full preflight evidence](../../result/nvidia-gpu/ci/public-runner-preflight.json.gz)
retains logs and the source run/artifact inventory. The final artifact binding
was additionally compared with the live API after it was added to the protocol.

Reproduce the local harness checks from the branch root:

```bash
python3 scripts/runpod-public-confirmation/generate.py --tenferro-checkout /path/to/tenferro-rs
python3 scripts/runpod-public-confirmation/check.py --tenferro-checkout /path/to/tenferro-rs
python3 scripts/runpod-public-confirmation/test_compare.py
actionlint .github/workflows/benchmark-runpod-gpu.yml
```

The workflow defaults to `prepare_only=true`; a draft protocol also rejects
`prepare_only=false`. The campaign runner consumes an external `protocol.json`
under `RUNPOD_COMPARISON_DIR` and refuses draft, unbound or pre-existing state.
Do not delete a state file to recover an observation: inspect the recorded
GitHub run first. Source regeneration and statistical thresholds are fixed
before any approved paid experiment.

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
