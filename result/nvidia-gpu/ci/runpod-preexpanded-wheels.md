# Incremental wheel preparation on the public official runner

The maintainer selected the public official runner and incremental work rather
than waiting for custom-image package administration. This is a smaller new
candidate, not another attempt at a previously failed paired time experiment.
The additional 10% whole-paid-Pod goal remains unproven.

## Scope and validation declared before GPU execution

- Production baseline: tenferro-rs `356a2011ef0f9d2d345ed31201fbe838995329a2`.
- Producer/consumer candidate: `64e0692e614eac94562882e02273b647cb91bfcc`.
- Frozen workload: all 285 CUDA tests, three PJRT tests and CUDA tutorial from
  run 38034449617, Rust source `853fabf55fdfe7d55960c0427c850d34db21dcb0`.
- Official runner image is unchanged, including digest. Both CUDA tiers,
  numerical library versions, zstd level 3 and all file contents are retained.
- Incremental acceptance: at least 5% fewer common artifact bytes, identical
  complete file/link inventory, executable NVCC tools, successful old/new
  format tests, and complete GPU workload with confirmed deletion.
- The existing offline screen established the 5% byte criterion before this
  implementation. Common/SDK preparation accounts for 27–28 paid seconds,
  above the 5% need criterion. Byte savings are not elapsed-time savings.
- One paid correctness validation, no automatic replacement/retry. A40,
  CA-MTL-1, selected CUDA 12.8, nextest concurrency one, profile ci, one
  allocation attempt, hourly ceiling $0.60 and one-hour lifecycle ceiling.
  Retain host identity, all test cases, stages, cost and deletion evidence.
- No timing comparison or 10% paid-lifetime claim is made from one run. The
  independent custom-image campaign has no paid samples and is abandoned.

The producer opts in through the trusted controller, preserving the prior
ZIP payload for old controllers during PR validation. New consumers accept
both layouts for historical-ref recovery. The exact transformed payload is
verified and re-decoded on a hosted runner before any GPU allocation.

## Local content verification

The actual candidate's extraction block was executed on the frozen original
payload. All 79 retained files/links matched, including every member of all
three wheels. The decoded NVCC ptxas starts and PJRT exports GetPjrtApi.
Common bytes: 1,196,496,508 -> 1,084,483,318 (9.36176% reduction). No timing
claim is attached to these local checks.

[Complete file inventory and checksums](runpod-preexpanded-wheels-local.json.gz)

Reproduction:

```bash
python3 scripts/runpod-preexpanded-wheels/prepare.py \
  --artifacts input --producer source/scripts/ci/prepare_gpu_execution_payload.sh \
  --output prepared
```

Here `input` contains the five merged common artifact parts and checksum from
run 38034449617; `source` is the candidate checkout. The workflow embeds the
candidate consumer and runs unchanged CUDA and tutorial commands.

## Hosted and full GPU validation

[Run 38046438056](https://github.com/tensor4all/tenferro-benchmark/actions/runs/38046438056)
succeeded at harness `b8a6e9fd587229e781ca54544431b61a7663d5f9`. Hosted
preparation preserved the same complete 79-file/link inventory and produced
1,080,710,232 common bytes, 9.67711% below the frozen original. The local and
hosted archive byte counts differ; both use unchanged `tar --zstd` settings
and independently pass the 5% byte gate and exact decoded-content checks.
Cargo, nextest and NVCC started in the unchanged public official runner.

The fresh A40 in CA-MTL-1 used driver 570.195.03 and selected CUDA 12.8. All
288 exact frozen CUDA/PJRT case identities and the CUDA tutorial passed.
DELETE returned HTTP 204. Paid lifetime was 352.480 seconds at $0.59/hour,
estimated GPU cost $0.0577676. Runtime setup was 26 seconds and PJRT including
its setup was four seconds. These are one-run diagnostics, not a paired
improvement estimate; the 10% whole-paid-lifetime goal remains unproven.

[Complete GPU evidence](runpod-preexpanded-wheels-gpu.json.gz) retains the log,
all case identities, hosted byte/content verification, placement, job steps,
stage costs and confirmed deletion. No extra candidate allocation was made.

## Post-merge integration validation

PR [#2060](https://github.com/tensor4all/tenferro-rs/pull/2060) merged as
`8fbd73d5d08e5a91e8f418728711f13a8dbb9043`.
[Run 38047804467](https://github.com/tensor4all/tenferro-rs/actions/runs/38047804467)
succeeded using the trusted main workflows, including the producer's
`--unpack-pjrt-wheels` opt-in and the prepared wheel directory on the Pod.
All 292 CUDA tests, three PJRT tests and the CUDA tutorial passed. Main also
contains unrelated CUDA changes from PR #2059, so this workload is distinct
from the frozen 288-case validation above.

The normal scheduler selected an NVIDIA RTX A4500 with driver 570.195.03.
Paid lifetime was 396.521 seconds at $0.25/hour, estimated GPU cost
$0.0275362. Runtime setup was 79 seconds. These are integration diagnostics
from a different device and workload, not a comparable performance sample.
Pod deletion returned HTTP 204. The whole-paid-lifetime 10% goal remains
unproven; the verified improvement is the common-payload byte reduction.

[Complete main integration evidence](runpod-preexpanded-wheels-main.json.gz)
retains all 295 passed case lines, the run log, job steps and stage costs.
