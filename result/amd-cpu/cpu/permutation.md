# CPU Permutation Benchmark Results

- Target profile: `amd-cpu`
- Suite: `cpu/permutation`
- Suite file: `benchmarks/cpu/permutation.yaml`
- Timestamp: `2026-11-15T20:22:57.856723+00:00`
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
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | **57.495 (57.017 / 57.701)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 61.230 (61.010 / 61.421) | 81.053 (79.127 / 106.727) | 57.942 (56.591 / 82.929) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **67.698 (66.067 / 68.194)** | - | - | 67.863 (66.091 / 68.198) |
| `reverse_15d_3` | 15D 3^15 reverse | 108.800 (107.549 / 109.812) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **98.868 (98.428 / 99.919)** | 310.196 (299.549 / 325.646) | 115.764 (112.653 / 134.406) | - |
| `reverse_23d_2` | 23D 2^23 reverse | **32.635 (32.457 / 32.720)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 59.983 (59.882 / 60.552) | 290.538 (277.596 / 303.304) | 68.388 (67.628 / 75.228) | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | **1331.778 (1320.656 / 1339.163)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 1386.119 (1377.444 / 1420.040) | 2026.087 (2010.736 / 2035.218) | 1639.759 (1619.149 / 1652.236) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | **67.765 (67.410 / 68.314)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 68.559 (67.783 / 69.323) | 120.255 (119.753 / 122.275) | 80.096 (76.851 / 80.321) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | 74.344 (74.196 / 74.742) | - | **72.883 (71.868 / 73.318)** | 333.971 (331.550 / 336.036) | 82.075 (81.046 / 82.484) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | **7.930 (7.841 / 8.117)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 20.800 (20.693 / 21.034) | 50.070 (50.012 / 50.178) | 32.823 (32.736 / 33.011) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | 84.176 (83.822 / 84.570) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **81.473 (79.947 / 82.240)** | 109.028 (105.930 / 110.672) | 103.543 (101.742 / 105.851) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 86.247 (85.285 / 87.490) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **84.479 (82.863 / 85.929)** | 148.087 (147.009 / 148.927) | 137.694 (136.207 / 137.946) | - |

## Threads: 4

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | 20.296 (20.165 / 20.374) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **19.584 (19.461 / 20.236)** | 80.068 (78.114 / 93.559) | 20.047 (18.980 / 39.018) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **67.716 (67.398 / 68.013)** | - | - | 68.111 (66.882 / 68.400) |
| `reverse_15d_3` | 15D 3^15 reverse | 59.566 (59.283 / 60.380) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 55.079 (54.341 / 55.983) | 296.351 (288.741 / 313.850) | **49.852 (45.806 / 77.552)** | - |
| `reverse_23d_2` | 23D 2^23 reverse | **27.454 (26.203 / 28.177)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 43.876 (43.784 / 48.424) | 281.490 (268.459 / 287.336) | 27.596 (27.210 / 37.068) | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | 574.070 (507.023 / 594.303) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **518.495 (505.977 / 573.181)** | 2019.686 (2008.318 / 2031.574) | 713.533 (671.424 / 723.677) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | **24.029 (23.720 / 24.573)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 32.296 (23.663 / 34.516) | 123.362 (122.420 / 123.931) | 38.330 (38.197 / 38.582) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | 26.449 (26.216 / 26.520) | - | **26.081 (25.578 / 26.478)** | 332.762 (330.989 / 333.179) | 40.519 (38.832 / 41.409) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | **3.948 (3.843 / 4.125)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 7.717 (7.611 / 7.877) | 50.316 (50.243 / 50.404) | 14.347 (14.006 / 14.575) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | **29.447 (29.195 / 29.935)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 29.471 (29.148 / 29.774) | 110.007 (106.175 / 110.233) | 42.999 (40.531 / 43.235) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 28.646 (28.252 / 29.108) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **28.128 (27.962 / 28.577)** | 147.679 (146.706 / 148.219) | 57.869 (56.058 / 58.661) | - |
