# Hosted cuTENSOR runtime cache footprint

The installer cached static and independent multi-GPU/MPI libraries that
GPU payload preparation already discards. Only the existing single-GPU shared
ABI is retained by the candidate. This changes hosted cache storage; GPU
payload contents and numerical workload remain identical.

Baseline: tenferro-rs `8fbd73d5d08e5a91e8f418728711f13a8dbb9043`.
Candidate: `615e6af97014aa613f55d3de00bd31b1e2f344a5`.
Both installers execute against one cuTENSOR 2.6.0.4 NVIDIA archive, substituting
only network and privilege commands. Actual tar extraction and validation run.
No elapsed-time, cache-hit-rate or paid-Pod improvement is claimed.

## Need and predeclared acceptance

[Publisher run 38047788138](https://github.com/tensor4all/tenferro-rs/actions/runs/38047788138)
saved 608,291,096 compressed cuTENSOR bytes. A subsequent cache API snapshot
reported 10,564,514,068 active bytes, with that entry absent. Its size was
about 5.7% of the observed total. This does not establish the reason for its
removal. GPU run 38047804467 missed all three hosted runtime caches.

Before measurement the [protocol](cutensor-runtime-cache-protocol.json)
required at least 50% fewer unpacked bytes and 40% fewer compressed bytes,
with every retained file hash and symlink target matching the baseline.
One complete offline comparison used identical `tar --zstd` settings
(zstd 1.5.5). Repetition and host-noise timing gates do not apply to byte
counts. Cold vendor download bytes are unchanged.

## Result

| Metric | Baseline | Candidate | Reduction |
| --- | ---: | ---: | ---: |
| Unpacked regular-file bytes | 1,497,546,900 | 478,410,136 | 68.05% |
| Compressed bytes | 610,823,965 | 238,093,023 | 61.02% |

PASS: both predeclared byte thresholds and exact retained-content comparison.
The candidate keeps one shared library and both soname links. The hosted cache
service archive size can differ from this controlled local tar comparison;
its original 608,291,096-byte observation is not mixed into the paired ratio.

[Full inventories, hashes, protocol and results](cutensor-runtime-cache.json.gz).
The external vendor archive SHA-256 is
`28dbda315a95b3edc58af986ff1b886dcf2bd1c9651094539011bbce234cb9ae`.

## Reproduction

Download `libcutensor-linux-x86_64-2.6.0.4_cuda12-archive.tar.xz` from NVIDIA's
cuTENSOR redistributable directory, then run against a checkout containing
both source commits:

```bash
python3 scripts/screen_cutensor_runtime_cache.py \
  --repository ../tenferro-rs \
  --protocol result/nvidia-gpu/ci/cutensor-runtime-cache-protocol.json \
  --candidate 615e6af97014aa613f55d3de00bd31b1e2f344a5 \
  --archive /tmp/libcutensor.tar.xz \
  --output /tmp/cutensor-runtime-cache-comparison
```

All 360 production CI helper tests and the local fast gate passed. Tests use
real tar archives for selective installation, stale destination replacement,
and missing/empty/broken runtime ABI rejection. Production shared-cache
publication requires the trusted main workflow after merge.
