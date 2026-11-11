# CPU Permutation Benchmark Results

- Target profile: `amd-cpu`
- Suite: `cpu/permutation`
- Suite file: `benchmarks/cpu/permutation.yaml`
- Timestamp: `2026-11-11T19:59:22.643547+00:00`
- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`

## CPU Information

- Model: `AMD EPYC 9V74 80-Core Processor`
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
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | 54.209 (54.042 / 54.305) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **52.481 (52.430 / 52.723)** | 97.880 (71.862 / 102.388) | 87.957 (62.873 / 88.886) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | 56.543 (54.154 / 56.621) | - | - | **56.394 (54.500 / 56.583)** |
| `reverse_15d_3` | 15D 3^15 reverse | 99.776 (99.049 / 100.820) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **89.725 (88.885 / 92.208)** | 245.501 (220.095 / 249.400) | 134.619 (107.972 / 137.395) | - |
| `reverse_23d_2` | 23D 2^23 reverse | **26.351 (26.274 / 26.402)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 48.321 (48.249 / 48.393) | 262.891 (251.451 / 274.282) | 79.194 (65.357 / 86.313) | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | **1232.139 (1229.277 / 1257.747)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 1332.157 (1321.414 / 1343.769) | 1397.712 (1384.722 / 1400.166) | 1682.962 (1668.621 / 1693.573) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | 63.561 (63.502 / 63.764) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **62.364 (62.308 / 62.441)** | 110.773 (84.497 / 117.714) | 96.507 (71.434 / 102.494) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | 67.088 (67.055 / 67.306) | - | **65.084 (64.224 / 65.530)** | 274.060 (272.587 / 285.634) | 92.176 (81.520 / 109.526) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | **6.500 (6.483 / 6.593)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 17.261 (17.231 / 17.429) | 45.710 (36.710 / 50.721) | 30.655 (26.880 / 36.344) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | 75.140 (74.683 / 75.543) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **71.524 (69.617 / 71.830)** | 92.997 (80.117 / 99.996) | 81.014 (68.205 / 88.216) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 80.657 (77.859 / 83.406) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **76.670 (75.634 / 76.959)** | 139.226 (136.387 / 145.403) | 110.247 (107.288 / 120.685) | - |

## Threads: 4

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | **18.557 (18.503 / 18.589)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 18.562 (18.387 / 18.653) | 87.163 (73.561 / 90.897) | 44.951 (22.360 / 45.381) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | 56.959 (54.387 / 57.112) | - | - | **56.789 (54.995 / 56.913)** |
| `reverse_15d_3` | 15D 3^15 reverse | 54.480 (54.008 / 54.778) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **47.177 (46.651 / 48.210)** | 242.684 (239.583 / 243.596) | 60.901 (58.294 / 62.154) | - |
| `reverse_23d_2` | 23D 2^23 reverse | **23.139 (22.073 / 23.787)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 35.365 (35.059 / 35.578) | 257.923 (252.577 / 270.531) | 40.181 (26.323 / 41.485) | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | **465.491 (460.829 / 527.228)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 527.803 (487.286 / 544.574) | 1383.799 (1373.282 / 1393.501) | 672.734 (663.540 / 714.564) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | 20.908 (20.850 / 21.197) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **20.522 (20.269 / 20.658)** | 104.042 (97.631 / 104.691) | 44.559 (25.341 / 47.063) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | 23.488 (23.109 / 23.564) | - | **22.738 (22.513 / 22.898)** | 273.041 (268.799 / 274.676) | 45.732 (42.347 / 51.377) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | **4.130 (4.115 / 4.152)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 6.840 (6.773 / 7.057) | 42.767 (40.496 / 44.274) | 15.746 (8.889 / 17.524) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | 26.887 (26.769 / 27.015) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **26.687 (26.496 / 26.843)** | 92.752 (92.121 / 94.639) | 38.701 (34.977 / 40.189) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 27.472 (27.263 / 27.661) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **26.817 (26.681 / 27.089)** | 136.615 (135.356 / 140.560) | 56.283 (54.438 / 58.508) | - |
