# Complete runtime compression screen

**Decision: FAIL. Do not promote or start a paid comparison for this candidate.**

The predeclared screen required at least 15% fewer common-plus-selected-SDK
bytes for **both** CUDA tiers. zstd level 10 does not meet that threshold.
The additional 10% whole paid Pod lifetime goal remains unproven.

The input artifacts are from [successful main run 38051889375](https://github.com/tensor4all/tenferro-rs/actions/runs/38051889375)
at `8dafe4202c33078f3be5c4821fc748e79b13c5ce`. Screen and protocol were
committed as `1ea15d4` before measurement. All three original hosted checksums
passed; all six decoded control/candidate tar streams matched the original tar
SHA-256 exactly, preserving files, links, modes and metadata.

| Payload | Original hosted bytes | Recompressed level 3 | Level 10 | Reduction vs control | Reduction vs hosted |
| --- | ---: | ---: | ---: | ---: | ---: |
| cuda12.6 + common | 1,737,276,747 | 1,743,167,731 | 1,541,838,401 | 11.55% | 11.25% |
| cuda12.8 + common | 2,155,157,121 | 2,162,174,595 | 1,929,012,152 | 10.78% | 10.49% |

Both controlled arms use four zstd threads and the same complete uncompressed
tar bytes. The production archive uses tar's default streaming compression;
frame/thread differences make the level-3 control slightly larger than the
original hosted archive. Both comparisons are shown, and neither reaches 15%.
There was one complete screen; no level selection or retries followed the result.
Local elapsed times are retained as diagnostics only, not performance evidence.

The latest paid workload lasted 320.178 seconds. Its common/SDK transfers took
17 seconds and total runtime setup took 26 seconds. A simple constant-throughput
estimate would save only about two seconds from an 11% byte reduction. This is
an estimate, not a measured saving, and cannot justify a 32-second whole-Pod
improvement. Earlier production run 38047804467 took 79 seconds for runtime
setup on different hardware; it is need evidence, not a paired baseline.

No production code, image, library version, numerical workload or cache policy
changed. No GPU was allocated for this screen. An independent optimization with
more headroom is needed. Prior bootstrap campaigns retain their INCONCLUSIVE
verdicts and exhausted retry limit; these results do not reopen them.

[Predeclared protocol](runtime-compression-protocol.json) and
[all measurements, checksums, logs and need evidence](runtime-compression-screen.json.gz).

Reproduce with the committed `scripts/screen_runtime_compression.py`: download
all `gpu-runtime-38051889375-*` artifacts from the linked run into separate
artifact-name directories, then run the script from a clean worktree with
`--downloads <directory> --output <new-directory>`. Outputs stay outside the
checkout. Tiny real-tar fixtures exercised all three cases and full byte
comparison; a corrupted fixture was rejected before decompression.
