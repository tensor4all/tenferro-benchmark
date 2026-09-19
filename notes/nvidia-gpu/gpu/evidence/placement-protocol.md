# Placement diagnostic, declared after the first small-case study

The six unbound pairs reproduce fast (~48 us) and slow (~68–75 us) batch=1 modes in both binaries. Separate CUDA profiles keep GEMM duration at 6.912 us while host/inter-kernel gaps differ ~17 us. Thus the observed bimodality is host-side in these traces; a persistent +20 us notification cost is not supported. CPU is EPYC 7713P, 64 physical cores, one NUMA node, eight independent 8-core L3 domains.

Intervention: restrict only child-process CPU affinity to one L3 domain (CPUs 0–7) versus the existing unrestricted 0–63 mask. Provider count remains explicitly one; this is not an 8-thread provider benchmark. No frequency/governor/shared-service changes. Record effective per-thread affinity, last CPU and sysfs scaling frequency with a lightweight external 10 ms sampler in both arms.

Both arms use 30 warmups/300 samples of the same two small eager cases to expose steady host placement over a longer window. Four fresh-process pairs per affinity, alternating baseline/candidate order, rotating affinity order by round. All runs sequential, same executables. Same idle GPU/load preflight gates as previous diagnostic. Save every result and trajectory. This is a causal placement/steady-state diagnostic, not a rerun that replaces the failed original acceptance experiment. No promotion claim from pinned-only improvements.

Interpretation: if the two modes collapse under single-L3 placement while remaining under unrestricted placement, placement/cache locality is supported as a contributor. If not, retain frequency/other scheduling effects as unresolved. Correlation with sampled CPU frequency alone is not proof of a frequency cause.
