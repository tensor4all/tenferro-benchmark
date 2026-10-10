# CUDA nextest concurrency experiment

The previous bootstrap campaign is closed and remains INCONCLUSIVE. This is a
new candidate: two CUDA nextest processes instead of one, with unchanged current
numerical test archives and serial PJRT execution. The user requested trying
this optimization together with the metadata reliability fix. Both arms use
the metadata fix, the same public official runner and Ubuntu 24.04; no prepared
custom image or additional administrator permission is needed.

The [frozen protocol](../../result/nvidia-gpu/ci/runpod-concurrency-protocol.json)
records 304 CUDA and three PJRT case identities, the tutorial, source/artifact
revisions and all 15 transfer artifact identities. The source run's crate code,
lockfile and test partitions match current main; only the trusted CI helpers
come from the candidate branch. Every test assertion, inventory check and timeout
remains active. Nextest isolates tests in processes; the inspected session/cache
cases use process-owned state. Observe memory rather than infer safety from that
isolation: one-second nvidia-smi samples must remain <=12 GiB, leaving headroom
relative to the production 16 GiB minimum. This sampled bound does not capture
sub-second peaks and cannot prove all supported GPUs have identical behavior.

One attempt, at most ten fresh A40 Pods in CA-MTL-1, CUDA 12.8, at most $0.60/hour
and $1.50 aggregate estimated GPU cost. Two serial A/A runs must have max/min
<=1.15; then four balanced, randomized adjacent pairs (seed 304) compare serial
and two-process CUDA execution. Record full driver, CPU, host load, placement,
peak sampled VRAM, every test identity and confirmed deletion. Driver distribution
total variation between arms must be <=0.5. Keep all cold pulls and failures;
missing/invalid evidence stops spending, with no sample replacement or retry.

Promotion requires >=10% lower mean AND median entire paid lifetime, >=15% lower
median CUDA stage, candidate maximum <=1.2 times baseline maximum, the memory
bound, and all numerical/lifecycle/identity gates. Report the paired log-ratio
Student t interval (four pairs, df=3) descriptively; it is not a large-sample or
population-wide guarantee. No formal significance claim from the six possible
balanced permutations. The purpose is a bounded CI scheduling trial, not a
numerical-kernel benchmark. CPU affinity remains production CI's unpinned policy.

The observer binds the committed harness, refuses to overwrite existing state,
and polls the same run after observation errors. Free preflight validates artifact
access before paid dispatch. The metadata fallback is covered locally with a DNS
failure and preserves the provider start time and price; it never invents them.
Production CUDA concurrency stays serial until the declared experiment passes.
