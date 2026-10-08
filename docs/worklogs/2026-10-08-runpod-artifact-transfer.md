# RunPod artifact transfer experiment for tenferro-rs #2002

## Decisions

The user authorized a paid experiment of actual Actions artifacts on RunPod.
This experiment-only branch repurposes the existing manually dispatched
benchmark workflow. Production main and tensor operation benchmarks stay intact.
Provisioning uses the existing benchmark scripts pinned to main commit
`2eea3c04cede56419ce12a892d6998f97bea14cb`; the measured transport harness is
identified by the workflow run's branch commit.

The fixed source is tenferro-rs run `37740724065`, artifact `11534740118`
(213,322,326 ZIP bytes; original files are zstd-compressed test archives).
The original files are re-uploaded as one artifact and as five equal partitions
of their concatenation, with compression level zero on both arms. The source
SHA-256 values and file lengths are verified after each transfer; checksums
cover this immutable Actions artifact boundary. Neither arm runs the binaries.

Before measurement, the complete protocol is fixed in `PROTOCOL` in the
harness: three pairs in single/split, split/single, single/split order, no
selective retries, median improvement of at least 20%. The primary wall-clock
includes action startup, download, ZIP extraction, reconstruction where needed,
and whole-file verification. Archive preparation and upload happen on a hosted
runner before renting a pod. The experiment uses one pod and the same original
files and action versions across all six samples; a fresh directory is used
for each sample, without Actions cache restoration.

Validity requires complete correct results, repeated-arm max/min <= 1.5,
monitored-core CPU busy <= 20%, idle non-loopback receive traffic <= 5 MiB/s,
and one-minute load <= reported CPU count. Observations are sampled for one
second before and after each sample. Affinity and host identity are recorded;
transfer workers retain normal runner affinity because this is network/CI setup
measurement, not a single-core numerical benchmark. These observations cannot
exclude RunPod-host or remote CDN contention; three pairs are limited evidence.
Any failed gate makes the complete experiment INCONCLUSIVE.

A live-priced reviewed L4 at <= $0.50/hour is selected, with one create
attempt and no candidate escalation. Downloads have three-minute step bounds,
the paid job has a 20-minute bound, and the existing cleanup always deletes the
pod. Setup/provision time and failure costs are separate from the primary metric.

## Verification conclusions and constraints

Local reconstruction tests cover original file boundaries, five equal parts,
missing/corrupted parts, changed originals, and a wrong transfer manifest.
The reporting tests cover incomplete samples and the predeclared noise gates.
Actual paid results and confirmed deletion will be linked here after execution.
No production tenferro-rs change is promoted on the strength of the earlier
HTTP Range probe.

The first two allocation attempts selected A5000 from live pricing but received
HTTP 500 (no available instances) before any pod was created. No measurements
or paid time were produced. Selection now targets L4, which successfully
registered in benchmark run `37779219119`, under the same price limit.
