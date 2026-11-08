# CPU Permutation Benchmark Results

- Target profile: `amd-cpu`
- Suite: `cpu/permutation`
- Suite file: `benchmarks/cpu/permutation.yaml`
- Timestamp: `2026-11-08T20:22:28.729750+00:00`
- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`

## CPU Information

- Model: `AMD EPYC 7763 64-Core Processor`
- Vendor: `AuthenticAMD`
- Logical CPUs: `4`
- Sockets: `1`
- Cores per socket: `2`
- Threads per core: `2`
- NUMA nodes: `1`
- Transparent huge pages: `madvise (always [madvise] never)`
- Transparent huge page defrag: `madvise (always defer defer+madvise [madvise] never)`
- Python platform: `Linux-7.0.0-1019-azure-x86_64-with-glibc2.39`
- L3 domains (shared CPUs): `0-3`
- Container: `devcontainer`

Thread counts are controlled via `RAYON_NUM_THREADS` / `OMP_NUM_THREADS` / `JULIA_NUM_THREADS`; no CPU-affinity pinning (`taskset` / `numactl`) was applied. The controlled thread environment is recorded per thread count in the run's `run_t<N>.yaml`.

`tenferro-rs` measures `TypedTensorView::transpose_view` followed by `BackendSession::to_contiguous_read`; one shared session is entered before timing, and input view descriptors and metadata-only `transpose_view` are built outside the timed region. Materialization accepts arbitrary source strides. Every backend allocates a fresh destination inside each timed call, so the table compares allocation-inclusive end-to-end materialization rather than destination-reuse copy kernels. `hptt` only participates in patterns with a contiguous source and destination. Correctness is verified against an internal, untimed odometer reference before any timing; a `FAILED` cell means that backend's output did not match the reference for that pattern.

## Threads: 1

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | **57.071 (56.687 / 57.901)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 61.288 (61.052 / 61.618) | 110.124 (81.675 / 115.076) | 84.823 (57.709 / 87.952) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **67.808 (66.645 / 68.265)** | - | - | 68.665 (67.542 / 69.305) |
| `reverse_15d_3` | 15D 3^15 reverse | 107.604 (106.480 / 109.068) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **97.964 (97.193 / 99.599)** | 303.626 (289.740 / 323.310) | 138.173 (118.609 / 144.308) | - |
| `reverse_23d_2` | 23D 2^23 reverse | **32.355 (32.099 / 32.896)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 59.869 (59.779 / 60.441) | 274.616 (258.277 / 285.617) | 72.473 (67.264 / 87.772) | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | **1315.160 (1300.846 / 1325.802)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 1384.037 (1367.436 / 1414.183) | 1980.553 (1969.373 / 1983.052) | 1617.255 (1611.258 / 1617.798) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | **68.541 (67.745 / 68.909)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 68.595 (67.862 / 69.305) | 139.272 (116.744 / 144.910) | 98.153 (68.696 / 103.202) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | 71.703 (70.848 / 72.258) | - | **71.170 (70.578 / 71.488)** | 345.533 (341.295 / 352.980) | 92.117 (87.898 / 109.613) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | **8.386 (8.236 / 8.664)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 20.897 (20.814 / 21.080) | 57.154 (47.919 / 62.021) | 34.391 (29.765 / 39.700) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | 82.362 (82.068 / 84.726) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **82.330 (81.547 / 85.242)** | 113.144 (111.839 / 119.697) | 108.739 (106.421 / 116.599) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 87.499 (86.338 / 89.509) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **85.079 (83.262 / 86.574)** | 154.237 (151.007 / 160.458) | 143.723 (141.737 / 152.381) | - |

## Threads: 4

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | 20.128 (19.886 / 20.389) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **20.059 (19.511 / 20.293)** | 97.788 (90.463 / 98.530) | 42.846 (20.003 / 44.415) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | 67.553 (66.050 / 68.111) | - | - | **67.531 (66.347 / 67.846)** |
| `reverse_15d_3` | 15D 3^15 reverse | 59.149 (58.235 / 59.819) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **55.897 (54.754 / 57.047)** | 314.613 (305.937 / 321.341) | 66.655 (62.506 / 67.276) | - |
| `reverse_23d_2` | 23D 2^23 reverse | **26.119 (25.719 / 26.721)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 43.253 (42.982 / 43.824) | 302.770 (295.190 / 319.358) | 44.008 (28.521 / 45.320) | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | **507.427 (500.303 / 515.049)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 520.605 (517.884 / 526.053) | 2005.929 (2000.527 / 2012.900) | 676.145 (672.510 / 678.730) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | 24.296 (23.553 / 24.458) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **23.205 (22.761 / 23.338)** | 132.024 (126.476 / 132.687) | 45.405 (40.459 / 49.142) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | 26.186 (25.654 / 26.543) | - | **25.187 (24.851 / 25.318)** | 340.755 (338.137 / 343.241) | 45.657 (43.882 / 49.423) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | **3.357 (3.266 / 3.614)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 7.736 (7.607 / 8.042) | 54.486 (52.101 / 55.307) | 18.118 (10.451 / 19.842) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | 30.003 (29.274 / 30.842) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **29.398 (29.096 / 29.772)** | 110.713 (108.630 / 114.351) | 46.237 (41.073 / 48.309) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 28.565 (28.052 / 29.267) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **28.118 (27.723 / 29.489)** | 152.913 (150.298 / 153.598) | 60.131 (58.644 / 63.360) | - |
