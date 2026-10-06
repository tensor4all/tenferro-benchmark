# CPU Permutation Benchmark Results

- Target profile: `amd-cpu`
- Suite: `cpu/permutation`
- Suite file: `benchmarks/cpu/permutation.yaml`
- Timestamp: `2026-10-06T19:23:18.384957+00:00`
- tenferro-rs commit: `5cf78c7ec0ad9516dd78bab546d5e7bd42fa3102`

## CPU Information

- Model: `AMD EPYC 7713P 64-Core Processor`
- Vendor: `AuthenticAMD`
- Logical CPUs: `64`
- Sockets: `1`
- Cores per socket: `64`
- Threads per core: `1`
- NUMA nodes: `1`
- Python platform: `Linux-6.8.0-101-generic-x86_64-with-glibc2.39`

On the Linux CPU devcontainer, thread counts are controlled via `RAYON_NUM_THREADS` / `OMP_NUM_THREADS` / `JULIA_NUM_THREADS`; no CPU-affinity pinning (`taskset` / `numactl`) is applied, matching the repository devcontainer convention. The controlled thread environment is recorded per thread count in the run's `run_t<N>.yaml`.

`tenferro-rs` measures `TypedTensorView::transpose_view` followed by `BackendSession::to_contiguous_read`; one shared session is entered before timing, and input view descriptors and metadata-only `transpose_view` are built outside the timed region. Materialization accepts arbitrary source strides. Every backend allocates a fresh destination inside each timed call, so the table compares allocation-inclusive end-to-end materialization rather than destination-reuse copy kernels. `hptt` only participates in patterns with a contiguous source and destination. Correctness is verified against an internal, untimed odometer reference before any timing; a `FAILED` cell means that backend's output did not match the reference for that pattern.

## Threads: 1

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | 78.083 (77.627 / 78.348) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **65.767 (65.649 / 66.018)** | 117.813 (91.206 / 121.469) | 87.092 (62.017 / 90.767) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | 72.406 (71.173 / 76.227) | - | - | **70.209 (69.027 / 71.029)** |
| `reverse_15d_3` | 15D 3^15 reverse | 187.301 (183.528 / 192.048) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **118.162 (117.798 / 118.392)** | 298.743 (274.414 / 301.665) | 145.581 (127.784 / 154.342) | - |
| `reverse_23d_2` | 23D 2^23 reverse | **64.421 (64.048 / 64.920)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 67.522 (67.243 / 67.775) | 300.856 (272.898 / 309.595) | 74.752 (69.624 / 87.217) | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | 1788.364 (1736.315 / 1941.825) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **1594.731 (1559.954 / 1640.572)** | 2288.015 (2248.180 / 2293.287) | 1701.797 (1668.663 / 1719.175) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | 85.768 (85.562 / 85.966) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **80.711 (80.414 / 80.869)** | 141.667 (118.696 / 151.705) | 107.736 (77.053 / 111.563) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | 94.378 (94.207 / 94.544) | - | **87.472 (87.384 / 87.680)** | 334.414 (326.556 / 343.212) | 101.820 (96.959 / 112.137) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | **14.561 (14.273 / 15.079)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 22.694 (21.723 / 23.596) | 61.971 (52.069 / 66.182) | 38.954 (33.940 / 44.427) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | 99.113 (96.368 / 101.125) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **85.545 (84.608 / 90.668)** | 131.240 (130.166 / 139.193) | 128.625 (127.508 / 136.929) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 138.174 (130.871 / 143.464) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **87.976 (86.655 / 90.061)** | 177.217 (172.724 / 181.764) | 164.129 (161.661 / 168.915) | - |

## Threads: 4

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | **25.074 (23.185 / 29.483)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 61.693 (59.816 / 67.470) | 117.863 (94.801 / 119.331) | 48.882 (20.373 / 51.304) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **75.303 (74.798 / 79.738)** | - | - | 79.159 (78.045 / 81.532) |
| `reverse_15d_3` | 15D 3^15 reverse | 82.273 (64.528 / 84.374) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 81.568 (81.024 / 82.765) | 329.516 (318.387 / 335.565) | **61.673 (59.810 / 68.854)** | - |
| `reverse_23d_2` | 23D 2^23 reverse | **16.926 (16.787 / 17.117)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 54.317 (53.501 / 54.634) | 299.481 (275.885 / 310.058) | 41.867 (21.517 / 42.611) | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | 544.971 (508.094 / 555.855) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **448.376 (445.979 / 451.049)** | 2308.232 (2282.699 / 2311.910) | 681.303 (674.239 / 690.855) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | 25.938 (25.768 / 26.265) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **23.625 (23.378 / 23.958)** | 145.246 (142.924 / 155.136) | 48.132 (27.895 / 56.619) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | 26.986 (25.811 / 27.394) | - | **26.250 (26.064 / 26.588)** | 369.156 (368.090 / 379.529) | 45.683 (44.413 / 55.829) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | **4.303 (4.201 / 4.379)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 6.921 (6.723 / 7.113) | 60.904 (55.455 / 64.210) | 16.965 (9.659 / 23.502) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | 29.688 (29.230 / 29.924) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **26.346 (25.961 / 26.597)** | 130.186 (129.291 / 136.716) | 48.649 (47.256 / 55.015) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 38.678 (38.330 / 44.316) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **26.983 (26.774 / 27.295)** | 170.197 (169.113 / 176.367) | 56.763 (55.523 / 63.703) | - |
