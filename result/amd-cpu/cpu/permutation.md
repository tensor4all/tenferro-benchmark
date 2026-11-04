# CPU Permutation Benchmark Results

- Target profile: `amd-cpu`
- Suite: `cpu/permutation`
- Suite file: `benchmarks/cpu/permutation.yaml`
- Timestamp: `2026-11-04T22:54:27.020562+00:00`
- tenferro-rs commit: `a000e5c4929d5bebb91e783c329b27106baed12e`

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
- Python platform: `Linux-7.0.0-1017-azure-x86_64-with-glibc2.39`
- L3 domains (shared CPUs): `0-3`
- Container: `devcontainer`

Thread counts are controlled via `RAYON_NUM_THREADS` / `OMP_NUM_THREADS` / `JULIA_NUM_THREADS`; no CPU-affinity pinning (`taskset` / `numactl`) was applied. The controlled thread environment is recorded per thread count in the run's `run_t<N>.yaml`.

`tenferro-rs` measures `TypedTensorView::transpose_view` followed by `BackendSession::to_contiguous_read`; one shared session is entered before timing, and input view descriptors and metadata-only `transpose_view` are built outside the timed region. Materialization accepts arbitrary source strides. Every backend allocates a fresh destination inside each timed call, so the table compares allocation-inclusive end-to-end materialization rather than destination-reuse copy kernels. `hptt` only participates in patterns with a contiguous source and destination. Correctness is verified against an internal, untimed odometer reference before any timing; a `FAILED` cell means that backend's output did not match the reference for that pattern.

## Threads: 1

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | **62.484 (62.274 / 63.168)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 67.670 (67.447 / 67.941) | 117.122 (88.869 / 118.894) | 94.874 (65.344 / 98.407) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **72.139 (71.576 / 74.474)** | - | - | 74.494 (71.991 / 75.626) |
| `reverse_15d_3` | 15D 3^15 reverse | 115.899 (114.895 / 117.114) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **104.654 (103.369 / 105.494)** | 338.741 (319.748 / 347.534) | 142.965 (122.206 / 153.790) | - |
| `reverse_23d_2` | 23D 2^23 reverse | **32.930 (32.763 / 33.189)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 62.791 (62.646 / 63.363) | 283.134 (274.893 / 301.068) | 82.695 (77.540 / 102.227) | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | **1459.113 (1430.278 / 1467.440)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 1498.521 (1484.934 / 1508.113) | 2109.654 (2104.739 / 2111.759) | 1711.004 (1706.953 / 1715.178) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | **73.031 (72.759 / 74.143)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 73.238 (70.364 / 74.060) | 143.079 (117.999 / 155.110) | 107.602 (75.002 / 111.473) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | 79.902 (78.928 / 80.815) | - | **79.039 (77.505 / 80.119)** | 353.712 (350.880 / 365.446) | 99.237 (97.028 / 118.416) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | **8.210 (8.071 / 8.282)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 22.354 (22.210 / 22.579) | 57.575 (48.101 / 63.304) | 37.728 (31.108 / 42.757) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | 89.457 (87.638 / 90.284) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **86.810 (85.923 / 87.772)** | 116.421 (109.680 / 123.485) | 113.008 (110.658 / 120.708) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 93.469 (92.306 / 94.198) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **88.369 (86.714 / 89.384)** | 158.261 (155.066 / 166.045) | 149.719 (146.044 / 157.728) | - |

## Threads: 4

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | 21.864 (21.702 / 22.154) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **21.444 (21.014 / 22.400)** | 110.759 (94.406 / 112.386) | 48.112 (21.836 / 50.997) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **74.956 (72.803 / 75.594)** | - | - | 75.857 (73.713 / 77.463) |
| `reverse_15d_3` | 15D 3^15 reverse | 63.528 (62.650 / 64.067) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **61.114 (60.671 / 61.669)** | 340.541 (332.332 / 345.229) | 72.378 (68.818 / 74.178) | - |
| `reverse_23d_2` | 23D 2^23 reverse | **17.342 (17.186 / 17.554)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 47.740 (47.129 / 48.148) | 318.864 (306.817 / 329.463) | 47.887 (29.539 / 50.068) | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | **541.759 (536.960 / 547.195)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 564.493 (560.486 / 569.471) | 2094.705 (2075.374 / 2099.512) | 710.272 (705.382 / 719.057) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | 25.603 (25.095 / 25.741) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **25.166 (24.201 / 25.692)** | 139.172 (134.798 / 146.032) | 48.663 (42.225 / 55.950) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | 28.933 (28.806 / 29.378) | - | **27.908 (27.655 / 28.213)** | 347.379 (345.436 / 349.501) | 50.323 (46.499 / 56.040) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | **4.084 (3.787 / 4.393)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 8.095 (7.985 / 8.252) | 56.762 (53.651 / 57.730) | 20.215 (10.969 / 20.842) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | 31.492 (30.997 / 31.890) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **30.528 (30.358 / 31.045)** | 118.162 (116.283 / 119.781) | 47.107 (45.621 / 48.642) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 30.332 (29.455 / 30.924) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **29.050 (28.876 / 29.242)** | 158.419 (156.802 / 160.618) | 63.543 (61.139 / 67.194) | - |
