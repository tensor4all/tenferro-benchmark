# CPU Permutation Benchmark Results

- Target profile: `amd-cpu`
- Suite: `cpu/permutation`
- Suite file: `benchmarks/cpu/permutation.yaml`
- Timestamp: `2026-11-09T20:03:51.355769+00:00`
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
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | **55.174 (54.913 / 55.265)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 55.436 (55.326 / 55.603) | 100.971 (74.716 / 109.349) | 91.516 (64.520 / 92.345) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **57.434 (55.129 / 57.919)** | - | - | 57.694 (55.160 / 59.703) |
| `reverse_15d_3` | 15D 3^15 reverse | 102.442 (101.325 / 103.523) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **92.178 (91.335 / 93.411)** | 253.418 (230.226 / 260.835) | 140.499 (113.997 / 144.012) | - |
| `reverse_23d_2` | 23D 2^23 reverse | **27.285 (27.190 / 27.427)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 49.048 (48.805 / 49.220) | 280.286 (271.218 / 291.775) | 82.993 (69.531 / 88.275) | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | **1267.531 (1256.376 / 1278.499)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 1364.871 (1353.628 / 1386.018) | 1422.286 (1414.538 / 1429.816) | 1694.297 (1685.439 / 1696.016) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | **63.950 (62.840 / 64.039)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 64.130 (62.487 / 65.533) | 108.420 (82.640 / 110.587) | 101.663 (72.573 / 108.991) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | 66.041 (64.836 / 66.328) | - | **66.020 (64.447 / 66.371)** | 277.150 (265.949 / 292.665) | 94.811 (81.820 / 111.426) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | **7.941 (6.930 / 9.311)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 17.759 (17.609 / 17.901) | 45.277 (37.159 / 51.325) | 33.491 (27.417 / 36.325) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | 74.951 (74.270 / 75.389) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **72.952 (72.020 / 73.947)** | 94.766 (81.498 / 101.152) | 81.657 (67.786 / 88.108) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 78.415 (77.613 / 79.261) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **77.879 (76.626 / 78.595)** | 140.448 (139.008 / 146.293) | 112.673 (108.071 / 122.249) | - |

## Threads: 4

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | 19.373 (19.220 / 19.473) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **18.968 (18.505 / 19.113)** | 88.564 (73.026 / 91.539) | 41.028 (22.213 / 44.473) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | 57.560 (55.533 / 57.838) | - | - | **57.368 (55.888 / 57.541)** |
| `reverse_15d_3` | 15D 3^15 reverse | 55.306 (54.807 / 55.683) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **48.623 (48.294 / 49.898)** | 241.903 (239.574 / 245.324) | 62.522 (57.824 / 66.170) | - |
| `reverse_23d_2` | 23D 2^23 reverse | **21.267 (20.324 / 22.022)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 35.951 (35.720 / 36.063) | 279.877 (272.032 / 285.553) | 42.316 (26.638 / 42.935) | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | **470.491 (469.991 / 472.144)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 493.706 (489.368 / 498.895) | 1390.953 (1383.282 / 1397.595) | 668.987 (665.927 / 673.773) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | 21.846 (21.249 / 22.671) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **20.999 (20.847 / 21.364)** | 99.277 (98.729 / 102.151) | 45.370 (25.727 / 51.383) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | 23.771 (23.648 / 24.233) | - | **22.761 (22.632 / 23.221)** | 273.471 (271.201 / 277.314) | 47.623 (44.192 / 52.350) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | **4.372 (4.315 / 4.493)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 7.064 (6.850 / 9.211) | 43.253 (41.322 / 45.079) | 15.098 (8.918 / 17.485) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | 27.262 (27.002 / 27.458) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **27.069 (26.917 / 27.183)** | 92.482 (92.030 / 94.810) | 37.822 (35.729 / 39.853) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 27.805 (27.754 / 28.250) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **27.228 (27.177 / 27.468)** | 139.453 (138.654 / 141.231) | 54.077 (53.447 / 58.012) | - |
