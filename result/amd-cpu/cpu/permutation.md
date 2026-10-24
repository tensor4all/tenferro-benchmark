# CPU Permutation Benchmark Results

- Target profile: `amd-cpu`
- Suite: `cpu/permutation`
- Suite file: `benchmarks/cpu/permutation.yaml`
- Timestamp: `2026-10-24T19:05:15.193314+00:00`
- tenferro-rs commit: `fa39e8d78d9c58e36575084b67448a8112ec1d6c`

## CPU Information

- Model: `AMD EPYC 9V74 80-Core Processor`
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
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | **12.462 (12.370 / 12.505)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 12.902 (12.870 / 12.947) | 35.911 (23.957 / 36.236) | 31.858 (21.503 / 33.238) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | 11.985 (11.953 / 12.049) | - | - | **11.978 (11.931 / 12.086)** |
| `reverse_15d_3` | 15D 3^15 reverse | 62.353 (58.800 / 84.497) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **58.670 (57.650 / 61.559)** | 185.217 (174.543 / 190.220) | 83.724 (70.695 / 84.097) | - |
| `reverse_23d_2` | 23D 2^23 reverse | **25.739 (25.578 / 25.806)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 28.794 (28.745 / 28.851) | 225.068 (223.766 / 232.686) | 38.358 (36.882 / 50.670) | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | 946.499 (910.589 / 975.623) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 830.446 (826.885 / 834.490) | **692.912 (686.657 / 1154.429)** | 916.752 (910.009 / 920.909) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | 11.239 (11.218 / 11.269) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **11.125 (11.101 / 11.173)** | 55.644 (42.391 / 56.855) | 36.558 (25.477 / 36.803) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | 12.317 (12.232 / 12.349) | - | **12.312 (12.282 / 12.355)** | 225.648 (214.879 / 240.970) | 54.560 (42.879 / 68.387) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | **6.640 (6.525 / 6.742)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 8.218 (8.117 / 8.314) | 35.724 (31.693 / 43.579) | 17.634 (16.996 / 21.005) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | 41.398 (41.186 / 41.574) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 41.379 (41.254 / 41.527) | 46.549 (42.141 / 46.679) | **32.759 (28.556 / 33.292)** | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | **40.873 (40.699 / 40.888)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 41.304 (41.192 / 41.502) | 64.184 (60.451 / 73.008) | 52.395 (48.434 / 60.703) | - |

## Threads: 4

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | 8.523 (8.407 / 8.670) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **8.444 (8.376 / 8.599)** | 29.923 (24.246 / 32.007) | 16.614 (10.289 / 18.069) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | 12.829 (12.741 / 13.032) | - | - | **12.708 (12.599 / 12.740)** |
| `reverse_15d_3` | 15D 3^15 reverse | 25.467 (24.452 / 26.168) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **22.679 (22.043 / 23.410)** | 183.484 (179.830 / 187.791) | 43.910 (40.356 / 45.290) | - |
| `reverse_23d_2` | 23D 2^23 reverse | **13.418 (13.351 / 19.590)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 16.721 (16.626 / 16.807) | 216.523 (214.038 / 223.427) | 17.564 (13.519 / 20.664) | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | **323.215 (319.524 / 328.859)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 333.708 (328.487 / 339.115) | 683.419 (677.162 / 775.768) | 364.239 (354.010 / 367.975) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | 9.036 (8.652 / 9.544) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **8.315 (8.220 / 8.363)** | 49.051 (43.811 / 54.031) | 20.229 (13.325 / 20.678) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | 8.783 (8.709 / 8.899) | - | **8.489 (8.345 / 8.617)** | 220.541 (220.019 / 226.033) | 27.079 (23.790 / 29.596) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | 4.058 (4.017 / 4.124) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **4.039 (3.965 / 4.171)** | 33.602 (32.831 / 36.790) | 7.814 (6.008 / 9.438) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | 17.554 (17.469 / 17.780) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 17.236 (17.131 / 17.371) | 44.575 (42.277 / 48.197) | **15.842 (13.128 / 19.169)** | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 14.779 (14.461 / 15.046) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **14.517 (14.383 / 14.685)** | 57.757 (56.810 / 61.155) | 22.177 (20.443 / 25.069) | - |
