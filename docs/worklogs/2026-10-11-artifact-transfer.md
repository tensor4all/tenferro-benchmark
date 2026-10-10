# Diagnose GPU setup transfers and screen raw artifact transport

The serial CUDA comparison was invalidated by preparation variability: the same
common and CUDA SDK bundles downloaded in 126/122 seconds on one Pod and 9/9
seconds on another, while CUDA execution was 240 seconds in both. This warrants
investigating transfer behavior before allocating more Pods for test scheduling.

The official download log already identifies each artifact's name, bytes,
digest and ID-specific request start. Joining the digest-completion log by
expected digest reveals individual slow parts without adding work to the paid
path. It measures request-to-digest wall time, including network, ZIP processing
and hashing. It does not identify first-byte latency or prove a ZIP bottleneck.
Missing completions remain missing; ambiguous shared digests are not guessed.

Screen `archive: false` with the pinned official upload/download actions on a
GitHub-hosted Ubuntu 24.04 CPU runner. Preserve the exact common and CUDA 12.8
compressed parts from source run 38056837195. ZIP remains compression-level 0;
raw transport uses the required filename-derived artifact names. Preserve the
full-archive checksums in a hosted preparation manifest, verify each part and
both reconstructed byte-stream identities after every download, and retain the
ZIP checksum file check. No numerical code or production transfer format changes.

The protocol is committed before the first candidate measurement: two ZIP A/A
samples followed by four balanced pairs, all on one VM, no replacements. Require
A/A <=1.15 separately per bundle, mean and median combined download reductions
>=20%, and no bundle median regression >10%. Keep all observations and failures.
This is an explicitly scoped CI transport wall-time experiment using official
Actions, not the standard CPU numerical benchmark/devcontainer suite. Default
runner affinity is retained; host/image/load are recorded. A pass is only grounds
for a later RunPod validation, not a paid-lifetime or production speedup claim.

[Protocol](../../result/linux-cpu/ci/artifact-transfer-protocol.json).
Zero GPU Pods are allocated by this workflow; it has no RunPod secrets or API calls.
