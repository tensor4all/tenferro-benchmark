# CPU Permutation Benchmark Results

- Target profile: `amd-cpu`
- Suite: `cpu/permutation`
- Suite file: `benchmarks/cpu/permutation.yaml`
- Timestamp: `2026-11-16T20:55:03.110319+00:00`
- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`

## CPU Information

- Model: `Intel(R) Xeon(R) 6975P-C`
- Vendor: `GenuineIntel`
- Logical CPUs: `4`
- Sockets: `1`
- Cores per socket: `2`
- Threads per core: `2`
- NUMA nodes: `1`
- Transparent huge pages: `madvise (always [madvise] never)`
- Transparent huge page defrag: `madvise (always defer defer+madvise [madvise] never)`
- Python platform: `Linux-7.0.0-1016-aws-x86_64-with-glibc2.39`
- L3 domains (shared CPUs): `0-3`
- Container: `devcontainer`

Thread counts are controlled via `RAYON_NUM_THREADS` / `OMP_NUM_THREADS` / `JULIA_NUM_THREADS`; no CPU-affinity pinning (`taskset` / `numactl`) was applied. The controlled thread environment is recorded per thread count in the run's `run_t<N>.yaml`.

`tenferro-rs` measures `TypedTensorView::transpose_view` followed by `BackendSession::to_contiguous_read`; one shared session is entered before timing, and input view descriptors and metadata-only `transpose_view` are built outside the timed region. Materialization accepts arbitrary source strides. Every backend allocates a fresh destination inside each timed call, so the table compares allocation-inclusive end-to-end materialization rather than destination-reuse copy kernels. `hptt` only participates in patterns with a contiguous source and destination. Correctness is verified against an internal, untimed odometer reference before any timing; a `FAILED` cell means that backend's output did not match the reference for that pattern.

## Threads: 1

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | **43.354 (43.270 / 43.726)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 47.788 (47.522 / 47.953) | 68.968 (65.068 / 72.013) | 48.082 (45.245 / 51.255) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | 51.453 (51.375 / 51.564) | - | - | **51.427 (51.360 / 51.608)** |
| `reverse_15d_3` | 15D 3^15 reverse | 66.043 (65.415 / 67.091) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **64.756 (64.198 / 66.105)** | 171.509 (163.543 / 175.081) | 93.759 (92.968 / 104.711) | - |
| `reverse_23d_2` | 23D 2^23 reverse | **16.816 (16.393 / 17.458)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 32.655 (32.403 / 33.023) | 81.621 (72.433 / 96.413) | 67.554 (66.461 / 68.032) | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | **838.943 (835.606 / 841.092)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 860.720 (858.778 / 864.282) | 1270.740 (1264.123 / 1282.173) | 935.557 (932.974 / 940.962) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | 42.935 (42.738 / 43.097) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **42.590 (42.513 / 42.638)** | 89.714 (85.965 / 92.692) | 56.383 (55.973 / 57.010) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | 44.421 (44.252 / 44.498) | - | **43.441 (43.373 / 43.650)** | 185.136 (183.656 / 185.478) | 58.197 (56.812 / 59.324) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | **7.694 (7.686 / 7.701)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 13.911 (13.862 / 14.017) | 36.373 (35.987 / 37.371) | 16.928 (16.613 / 17.141) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | 51.022 (50.120 / 51.379) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **49.501 (49.354 / 49.615)** | 65.552 (63.482 / 66.050) | 63.593 (62.387 / 66.340) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 62.809 (60.497 / 63.997) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **60.148 (58.050 / 63.913)** | 68.464 (67.951 / 69.729) | 62.231 (61.004 / 62.875) | - |

## Threads: 4

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | 16.041 (15.972 / 16.212) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **15.775 (15.718 / 16.116)** | 77.026 (67.982 / 87.710) | 18.087 (17.245 / 29.485) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **51.551 (51.473 / 51.757)** | - | - | 52.503 (52.132 / 54.132) |
| `reverse_15d_3` | 15D 3^15 reverse | 44.791 (43.449 / 44.991) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 42.052 (41.712 / 42.378) | 184.259 (167.989 / 188.362) | **38.209 (36.491 / 53.818)** | - |
| `reverse_23d_2` | 23D 2^23 reverse | **13.658 (13.622 / 13.730)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 28.813 (28.551 / 28.985) | 74.571 (68.036 / 80.237) | 20.429 (20.302 / 21.281) | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | **325.665 (324.889 / 361.718)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 352.889 (350.874 / 376.770) | 1234.566 (1225.275 / 1238.070) | 410.264 (405.634 / 412.820) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | 16.392 (16.163 / 16.738) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **15.921 (15.744 / 16.211)** | 90.082 (81.034 / 98.268) | 25.634 (24.705 / 25.950) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | 19.576 (19.138 / 19.945) | - | **16.710 (16.587 / 16.849)** | 185.905 (180.899 / 190.645) | 28.086 (27.604 / 28.429) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | **2.494 (2.482 / 2.535)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 5.157 (5.017 / 5.383) | 27.948 (27.766 / 28.235) | 7.296 (7.162 / 7.531) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | 20.693 (20.601 / 20.750) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **20.456 (20.345 / 20.804)** | 62.364 (61.881 / 63.071) | 28.464 (27.952 / 28.998) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 22.582 (22.035 / 23.053) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **21.641 (21.478 / 21.937)** | 64.077 (63.609 / 64.794) | 29.731 (27.582 / 29.930) | - |
