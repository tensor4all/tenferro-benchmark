# CPU Permutation Benchmark Results

- Target profile: `amd-cpu`
- Suite: `cpu/permutation`
- Suite file: `benchmarks/cpu/permutation.yaml`
- Timestamp: `2026-11-14T20:22:32.734690+00:00`
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
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | **57.776 (57.586 / 58.395)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 61.804 (61.285 / 63.188) | 82.706 (79.973 / 101.437) | 60.346 (57.315 / 85.142) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **67.540 (66.645 / 68.245)** | - | - | 68.022 (67.104 / 69.078) |
| `reverse_15d_3` | 15D 3^15 reverse | 109.233 (106.496 / 111.039) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **98.168 (97.789 / 100.237)** | 300.050 (284.861 / 305.279) | 112.410 (110.658 / 133.561) | - |
| `reverse_23d_2` | 23D 2^23 reverse | **32.433 (32.226 / 32.901)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 59.714 (59.537 / 60.103) | 283.946 (270.571 / 291.327) | 67.039 (66.346 / 68.508) | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | **1319.376 (1313.048 / 1333.084)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 1401.559 (1392.065 / 1436.378) | 2022.278 (2007.626 / 2031.179) | 1614.668 (1599.619 / 1619.368) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | **68.321 (67.947 / 69.449)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 68.739 (68.004 / 69.554) | 120.987 (119.564 / 121.201) | 78.662 (76.871 / 79.632) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | **72.502 (71.900 / 73.489)** | - | 72.562 (71.329 / 73.118) | 331.109 (329.666 / 332.854) | 82.419 (81.863 / 82.675) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | **7.884 (7.784 / 8.115)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 20.986 (20.713 / 21.718) | 50.228 (50.035 / 50.395) | 32.879 (32.760 / 33.196) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | 84.349 (83.220 / 85.186) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **81.852 (80.729 / 83.357)** | 109.639 (107.410 / 110.504) | 106.299 (105.364 / 107.045) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 87.206 (86.152 / 88.292) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **85.292 (84.792 / 86.253)** | 146.802 (146.071 / 148.017) | 135.579 (133.306 / 136.094) | - |

## Threads: 4

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | 20.040 (19.743 / 20.150) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **19.842 (19.682 / 19.994)** | 84.444 (81.824 / 99.372) | 19.967 (19.277 / 44.248) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | 68.455 (68.142 / 69.289) | - | - | **68.454 (66.925 / 69.395)** |
| `reverse_15d_3` | 15D 3^15 reverse | 59.841 (59.447 / 60.542) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 56.109 (55.422 / 57.320) | 297.337 (287.373 / 304.396) | **49.012 (45.273 / 63.876)** | - |
| `reverse_23d_2` | 23D 2^23 reverse | **26.047 (25.314 / 26.537)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 43.711 (43.575 / 44.531) | 278.092 (261.863 / 290.536) | 28.790 (27.871 / 29.826) | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | 548.325 (500.159 / 574.972) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **522.729 (518.713 / 579.116)** | 2018.922 (2009.571 / 2031.029) | 697.982 (676.824 / 743.243) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | 23.827 (23.233 / 24.018) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **23.220 (22.813 / 23.879)** | 124.753 (122.983 / 127.706) | 38.938 (38.189 / 39.247) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | **25.581 (25.297 / 26.151)** | - | 25.950 (25.635 / 26.169) | 330.274 (327.381 / 337.377) | 39.857 (36.978 / 41.165) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | **3.138 (3.083 / 3.347)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 7.536 (7.448 / 7.742) | 49.964 (49.713 / 50.100) | 14.487 (14.302 / 14.810) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | **29.016 (28.826 / 29.219)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 29.538 (28.650 / 30.055) | 107.230 (105.881 / 107.390) | 42.270 (40.450 / 43.058) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 29.166 (28.911 / 29.553) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **27.502 (27.339 / 27.692)** | 147.088 (146.177 / 147.428) | 58.213 (57.042 / 58.360) | - |
