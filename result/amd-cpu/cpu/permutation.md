# CPU Permutation Benchmark Results

- Target profile: `amd-cpu`
- Suite: `cpu/permutation`
- Suite file: `benchmarks/cpu/permutation.yaml`
- Timestamp: `2026-11-12T20:21:55.977432+00:00`
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
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | **56.662 (56.422 / 56.758)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 60.353 (60.013 / 60.540) | 106.087 (79.948 / 111.827) | 85.599 (57.376 / 86.241) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **66.309 (65.407 / 66.829)** | - | - | 66.625 (65.654 / 67.270) |
| `reverse_15d_3` | 15D 3^15 reverse | 109.105 (107.578 / 111.594) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **96.254 (94.924 / 96.858)** | 299.738 (278.172 / 305.645) | 133.505 (118.758 / 140.110) | - |
| `reverse_23d_2` | 23D 2^23 reverse | **31.963 (31.874 / 32.026)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 59.783 (59.521 / 60.095) | 269.558 (251.644 / 272.300) | 74.739 (68.696 / 92.586) | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | **1271.303 (1263.149 / 1290.230)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 1365.758 (1351.791 / 1389.965) | 2000.476 (1992.418 / 2067.908) | 1629.246 (1623.948 / 1635.802) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | 69.345 (68.848 / 69.588) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **67.936 (67.566 / 68.636)** | 132.459 (128.440 / 144.895) | 96.930 (68.055 / 100.282) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | 72.766 (71.585 / 73.242) | - | **72.472 (72.069 / 72.700)** | 340.738 (339.719 / 352.806) | 92.300 (88.368 / 105.238) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | **7.063 (7.029 / 7.144)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 20.068 (20.010 / 20.279) | 56.667 (47.604 / 61.904) | 34.391 (29.825 / 39.119) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | 80.637 (79.950 / 82.177) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **78.648 (76.994 / 78.934)** | 112.028 (107.517 / 118.835) | 106.702 (104.442 / 116.337) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 80.698 (80.299 / 81.464) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **79.981 (79.660 / 80.283)** | 151.744 (150.410 / 158.739) | 141.970 (137.984 / 150.457) | - |

## Threads: 4

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | 20.096 (19.913 / 20.216) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **19.937 (19.756 / 20.357)** | 99.706 (82.080 / 101.318) | 42.970 (19.596 / 44.883) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | 67.235 (66.041 / 67.576) | - | - | **66.993 (66.362 / 67.484)** |
| `reverse_15d_3` | 15D 3^15 reverse | 57.478 (56.882 / 58.030) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **54.576 (53.832 / 55.227)** | 293.092 (285.752 / 307.410) | 65.623 (61.577 / 67.332) | - |
| `reverse_23d_2` | 23D 2^23 reverse | **26.857 (26.458 / 27.061)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 42.938 (42.806 / 43.116) | 272.885 (257.330 / 275.250) | 43.113 (27.816 / 44.867) | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | 536.131 (504.399 / 574.310) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **515.738 (507.622 / 579.056)** | 2006.726 (1991.508 / 2018.991) | 723.169 (685.735 / 750.672) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | 23.518 (23.217 / 23.909) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **22.650 (22.444 / 23.014)** | 130.245 (126.174 / 133.419) | 44.995 (38.449 / 50.781) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | 25.421 (25.352 / 25.549) | - | **24.845 (24.593 / 25.087)** | 338.633 (336.951 / 343.044) | 44.287 (42.678 / 50.390) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | **3.034 (3.003 / 3.082)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 7.914 (7.823 / 8.016) | 54.369 (51.628 / 55.574) | 18.956 (10.500 / 19.689) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | **28.349 (28.157 / 28.664)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 28.865 (28.555 / 29.455) | 110.037 (108.862 / 112.712) | 45.166 (43.470 / 49.111) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 28.141 (27.897 / 28.222) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **27.967 (27.860 / 28.322)** | 150.721 (148.315 / 152.769) | 59.678 (57.145 / 62.185) | - |
