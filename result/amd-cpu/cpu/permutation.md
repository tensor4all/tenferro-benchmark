# CPU Permutation Benchmark Results

- Target profile: `amd-cpu`
- Suite: `cpu/permutation`
- Suite file: `benchmarks/cpu/permutation.yaml`
- Timestamp: `2026-10-21T19:18:37.658393+00:00`
- tenferro-rs commit: `f1d4d8d961ff6b937768111888ad2a493314892a`

## CPU Information

- Model: `AMD EPYC 7763 64-Core Processor`
- Vendor: `AuthenticAMD`
- Logical CPUs: `4`
- Sockets: `1`
- Cores per socket: `2`
- Threads per core: `2`
- NUMA nodes: `1`
- Python platform: `Linux-6.17.0-1022-azure-x86_64-with-glibc2.39`

On the Linux CPU devcontainer, thread counts are controlled via `RAYON_NUM_THREADS` / `OMP_NUM_THREADS` / `JULIA_NUM_THREADS`; no CPU-affinity pinning (`taskset` / `numactl`) is applied, matching the repository devcontainer convention. The controlled thread environment is recorded per thread count in the run's `run_t<N>.yaml`.

`tenferro-rs` measures `TypedTensorView::transpose_view` followed by `BackendSession::to_contiguous_read`; one shared session is entered before timing, and input view descriptors and metadata-only `transpose_view` are built outside the timed region. Materialization accepts arbitrary source strides. Every backend allocates a fresh destination inside each timed call, so the table compares allocation-inclusive end-to-end materialization rather than destination-reuse copy kernels. `hptt` only participates in patterns with a contiguous source and destination. Correctness is verified against an internal, untimed odometer reference before any timing; a `FAILED` cell means that backend's output did not match the reference for that pattern.

## Threads: 1

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | 22.382 (22.356 / 22.536) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **14.200 (13.925 / 14.304)** | 45.769 (33.471 / 46.694) | 26.895 (15.476 / 27.207) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **13.226 (13.147 / 13.308)** | - | - | 13.689 (13.459 / 13.798) |
| `reverse_15d_3` | 15D 3^15 reverse | 54.751 (52.238 / 55.362) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **52.349 (51.595 / 53.977)** | 214.179 (202.318 / 220.870) | 75.086 (64.929 / 86.466) | - |
| `reverse_23d_2` | 23D 2^23 reverse | **31.494 (31.363 / 31.729)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 35.570 (35.498 / 35.830) | 262.043 (257.112 / 268.957) | 37.404 (35.693 / 47.312) | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | 737.442 (727.318 / 749.554) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 855.572 (848.723 / 858.813) | 1045.685 (1039.048 / 1356.739) | **709.198 (705.862 / 717.797)** | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | **14.931 (14.877 / 15.031)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 14.975 (14.904 / 15.158) | 74.659 (63.243 / 87.445) | 30.782 (18.888 / 31.244) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | **17.263 (17.018 / 17.484)** | - | 17.556 (17.407 / 17.742) | 281.102 (280.132 / 294.188) | 33.718 (33.018 / 47.581) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | 15.544 (15.518 / 15.566) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **10.028 (9.910 / 10.249)** | 76.341 (73.520 / 84.377) | 70.913 (69.144 / 73.732) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | 37.185 (37.089 / 37.502) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **35.512 (35.412 / 35.782)** | 58.358 (54.332 / 60.425) | 50.231 (44.896 / 50.957) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 42.026 (41.606 / 42.293) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **38.473 (38.065 / 39.067)** | 78.270 (77.019 / 85.721) | 68.395 (66.658 / 76.585) | - |

## Threads: 4

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | 11.330 (11.218 / 11.410) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **9.107 (9.038 / 9.194)** | 39.799 (32.909 / 40.447) | 16.378 (9.958 / 16.774) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **13.314 (13.272 / 13.512)** | - | - | 13.407 (13.375 / 13.568) |
| `reverse_15d_3` | 15D 3^15 reverse | 23.104 (22.223 / 23.652) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **22.078 (21.058 / 23.455)** | 202.184 (197.419 / 202.900) | 37.812 (37.236 / 42.523) | - |
| `reverse_23d_2` | 23D 2^23 reverse | 21.259 (16.166 / 21.504) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **20.119 (20.057 / 20.267)** | 255.702 (250.506 / 257.508) | 22.101 (16.298 / 23.056) | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | 314.036 (307.323 / 331.579) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 362.061 (360.338 / 364.289) | 1031.138 (1029.255 / 1056.169) | **305.428 (297.360 / 306.739)** | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | 9.195 (8.932 / 9.495) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **8.845 (8.727 / 9.374)** | 69.308 (68.861 / 74.240) | 16.543 (10.497 / 20.673) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | 9.194 (9.068 / 9.420) | - | **9.052 (8.824 / 9.188)** | 277.358 (275.501 / 281.387) | 17.307 (16.848 / 22.261) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | 5.250 (5.215 / 5.346) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **3.425 (3.381 / 3.584)** | 72.543 (70.922 / 75.194) | 14.915 (13.171 / 18.502) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | 15.670 (15.349 / 15.888) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **15.436 (15.333 / 15.692)** | 56.402 (54.325 / 59.883) | 18.825 (16.064 / 22.275) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 15.882 (15.634 / 16.033) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **12.969 (12.679 / 13.075)** | 72.086 (70.471 / 74.908) | 29.429 (26.361 / 31.079) | - |
