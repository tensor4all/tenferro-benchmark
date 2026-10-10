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
