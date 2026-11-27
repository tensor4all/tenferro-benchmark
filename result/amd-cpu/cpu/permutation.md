# CPU Permutation Benchmark Results

- Target profile: `amd-cpu`
- Suite: `cpu/permutation`
- Suite file: `benchmarks/cpu/permutation.yaml`
- Timestamp: `2026-11-27T19:34:40.270991+00:00`
- tenferro-rs commit: `1ba9f33521691170c763dfce9b6c9bb56ca523c2`

## CPU Information

- Model: `INTEL(R) XEON(R) PLATINUM 8573C`
- Vendor: `GenuineIntel`
- Logical CPUs: `4`
- Sockets: `1`
- Cores per socket: `2`
- Threads per core: `2`
- NUMA nodes: `1`
- Transparent huge pages: `madvise (always [madvise] never)`
- Transparent huge page defrag: `madvise (always defer defer+madvise [madvise] never)`
- Python platform: `Linux-7.0.0-1020-azure-x86_64-with-glibc2.39`
- L3 domains (shared CPUs): `0-3`
- Container: `devcontainer`

Thread counts are controlled via `RAYON_NUM_THREADS` / `OMP_NUM_THREADS` / `JULIA_NUM_THREADS`; no CPU-affinity pinning (`taskset` / `numactl`) was applied. The controlled thread environment is recorded per thread count in the run's `run_t<N>.yaml`.

`tenferro-rs` measures `TypedTensorView::transpose_view` followed by `BackendSession::to_contiguous_read`; one shared session is entered before timing, and input view descriptors and metadata-only `transpose_view` are built outside the timed region. Materialization accepts arbitrary source strides. Every backend allocates a fresh destination inside each timed call, so the table compares allocation-inclusive end-to-end materialization rather than destination-reuse copy kernels. `hptt` only participates in patterns with a contiguous source and destination. Correctness is verified against an internal, untimed odometer reference before any timing; a `FAILED` cell means that backend's output did not match the reference for that pattern.

## Threads: 1

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | **57.297 (57.073 / 57.419)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 58.469 (58.400 / 58.593) | 84.855 (82.513 / 94.400) | 59.294 (58.265 / 67.628) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | 66.319 (66.245 / 66.637) | - | - | **66.161 (66.032 / 66.517)** |
| `reverse_15d_3` | 15D 3^15 reverse | 97.307 (96.850 / 98.013) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **90.666 (90.501 / 90.777)** | 210.960 (208.653 / 224.365) | 107.858 (106.731 / 121.798) | - |
| `reverse_23d_2` | 23D 2^23 reverse | **26.097 (25.956 / 26.281)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 49.672 (48.923 / 49.998) | 181.546 (179.883 / 183.264) | 83.628 (80.669 / 93.918) | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | **1082.272 (1066.635 / 1098.709)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 1146.787 (1139.744 / 1155.501) | 1491.112 (1457.904 / 1495.258) | 1172.253 (1158.219 / 1180.265) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | 57.677 (56.377 / 57.929) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **57.002 (56.508 / 57.226)** | 96.127 (95.896 / 99.682) | 73.502 (69.769 / 73.801) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | 59.677 (58.889 / 60.018) | - | **58.977 (58.620 / 59.422)** | 246.578 (243.290 / 250.614) | 78.347 (77.297 / 79.041) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | **10.138 (9.184 / 10.574)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 25.437 (24.668 / 26.420) | 34.859 (34.662 / 35.111) | 22.308 (22.196 / 22.486) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | **69.128 (68.673 / 69.938)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 69.444 (69.138 / 69.749) | 85.375 (83.748 / 85.845) | 80.219 (78.612 / 80.697) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 185.928 (185.280 / 187.337) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 170.986 (170.434 / 172.650) | 86.645 (85.189 / 87.095) | **79.706 (79.052 / 80.008)** | - |

## Threads: 4

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | 21.111 (20.943 / 21.605) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **19.233 (19.187 / 19.485)** | 82.080 (81.762 / 88.771) | 21.162 (20.701 / 36.672) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **66.784 (66.407 / 67.065)** | - | - | 68.403 (67.407 / 68.865) |
| `reverse_15d_3` | 15D 3^15 reverse | 56.121 (56.016 / 56.524) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 54.387 (54.124 / 54.798) | 207.403 (205.380 / 216.037) | **53.076 (49.676 / 70.338)** | - |
| `reverse_23d_2` | 23D 2^23 reverse | 50.911 (50.486 / 51.192) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 36.515 (36.402 / 36.693) | 182.255 (179.727 / 191.071) | **30.101 (29.921 / 30.687)** | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | **423.024 (419.705 / 459.389)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 440.894 (430.287 / 457.045) | 1411.020 (1404.285 / 1419.182) | 508.440 (504.672 / 525.604) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | 20.980 (20.861 / 21.163) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **20.608 (20.516 / 20.823)** | 104.859 (104.475 / 104.955) | 34.008 (33.693 / 34.166) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | 24.702 (24.561 / 25.188) | - | **21.580 (21.446 / 21.911)** | 233.455 (232.203 / 234.253) | 36.176 (35.370 / 36.291) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | **3.939 (3.778 / 4.384)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 7.109 (6.976 / 7.252) | 33.039 (32.704 / 33.310) | 9.779 (9.421 / 9.902) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | **27.385 (27.188 / 27.666)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 27.574 (27.406 / 27.729) | 84.546 (82.102 / 85.108) | 36.106 (35.784 / 37.237) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | **26.829 (26.588 / 26.999)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 27.224 (27.066 / 27.539) | 84.910 (83.906 / 85.367) | 35.393 (35.070 / 36.026) | - |
