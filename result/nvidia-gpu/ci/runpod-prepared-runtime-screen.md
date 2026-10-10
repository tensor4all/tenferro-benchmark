# RunPod prepared-runtime offline screen

This is a preparation/transfer screen, not GPU operation latency or acceptance
of the additional 10% paid-lifetime goal. Baseline: tenferro-rs
`356a2011ef0f9d2d345ed31201fbe838995329a2` (PR #2058).
The frozen common payload comes from run 38034449617 at
`853fabf55fdfe7d55960c0427c850d34db21dcb0`, whose tree equals the merged baseline.
Its original SHA-256 was verified before extraction.

| Payload | Compressed bytes | Reduction |
| --- | ---: | ---: |
| Original wheel containers | 1,196,496,508 | — |
| Complete wheel contents, zstd 3 | 1,084,483,330 | 9.36% |
| Complete wheel contents, zstd 10 | 961,491,571 | 19.64% |

Every one of the 79 retained files/symlinks matches the original runtime and
complete unpacked wheel contents. The wheel ZIP containers are redundant after
extracting all members; no runtime library or test is removed.
Both candidates pass the predeclared 5% byte-screen gate. That is not evidence
of a 10% reduction in whole paid Pod lifetime. Original preparation took 6.288 s
and zstd-3 preparation 1.617 s locally, but local timing is diagnostic only;
CPU contention affected the later compression/decode observations. All host
observations and timings, including the busy observations, are retained.

The existing identical-bootstrap runs spend 27–28 s in common/SDK preparation
(7.60–9.03% of paid lifetime), above the 5% need threshold. Pre-extraction alone
is unlikely to save the approximately 33 s required for the next goal. A
prepared image containing both runtime tiers, existing execution tools and
complete PJRT dependencies is the next candidate to prepare. Its larger image
pull is a potential regression and requires a new full cloud experiment.
Registry publication remains human-only. No image has been published or paid
campaign started for this candidate.

[Full protocol, measurements and file inventory](runpod-prepared-runtime-screen.json.gz)


## Prepared image native validation

[Hosted validation run 38035993659](https://github.com/tensor4all/tenferro-benchmark/actions/runs/38035993659)
succeeded at source `8388132a874d2df42a8482e26a452d1a1735ee5d`. It reconstructed
all three original runtime bundles with their checksums, built the official
runner image with the execution tools and full prepared payload, compiled
CubeCL-style JIT headers with both CUDA 12.6 and 12.8, checked each loaded NVRTC
version and cuTENSOR load, and verified runner/Node/Cargo/nextest and all PJRT
dependencies. The compressed Docker image is retained as the
`runpod-prepared-runner-image` artifact for a human publication handoff; the
build workflow did not log into a registry or publish any image.

The first native check (38035832552) failed because its cuTENSOR load omitted
the selected SDK from `LD_LIBRARY_PATH`. The validation script now uses the same
selected-library setting as the actual GPU job, for each CUDA tier separately.
This CPU validation does not establish GPU numerical correctness, cold image
pull time or the additional 10% paid-lifetime goal. A new complete cloud
comparison remains required after the image is made available to RunPod.
