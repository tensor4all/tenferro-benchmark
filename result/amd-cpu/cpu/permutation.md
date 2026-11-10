# CPU Permutation Benchmark Results

- Target profile: `amd-cpu`
- Suite: `cpu/permutation`
- Suite file: `benchmarks/cpu/permutation.yaml`
- Timestamp: `2026-11-10T20:02:35.885677+00:00`
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
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | 54.475 (54.322 / 54.598) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **53.449 (53.347 / 53.575)** | 98.341 (72.886 / 105.332) | 90.359 (63.842 / 91.957) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **56.932 (54.487 / 57.208)** | - | - | 57.120 (56.567 / 60.108) |
| `reverse_15d_3` | 15D 3^15 reverse | 100.857 (100.253 / 101.786) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **89.979 (89.577 / 91.441)** | 249.555 (225.044 / 254.258) | 138.073 (110.944 / 141.281) | - |
| `reverse_23d_2` | 23D 2^23 reverse | **27.722 (26.694 / 28.065)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 48.228 (48.128 / 48.340) | 268.207 (255.947 / 273.454) | 77.633 (65.817 / 83.076) | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | **1259.947 (1237.045 / 1277.613)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 1303.485 (1298.340 / 1306.994) | 1383.752 (1376.020 / 1388.923) | 1688.489 (1682.949 / 1693.629) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | **62.962 (61.877 / 63.207)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 63.166 (62.165 / 63.465) | 111.787 (83.108 / 113.687) | 99.619 (71.565 / 105.268) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | 65.297 (64.257 / 65.667) | - | **65.268 (64.064 / 65.463)** | 276.747 (275.645 / 286.776) | 94.296 (93.972 / 108.883) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | **6.808 (6.759 / 6.864)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 17.427 (17.375 / 17.525) | 45.407 (37.146 / 50.353) | 31.283 (26.737 / 35.113) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | 74.858 (73.848 / 75.320) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **72.097 (71.290 / 72.438)** | 96.747 (91.001 / 101.891) | 81.588 (69.686 / 88.374) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 79.643 (78.473 / 80.240) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **77.358 (76.501 / 78.550)** | 139.068 (137.375 / 146.046) | 114.435 (113.310 / 125.196) | - |

## Threads: 4

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | 18.790 (18.699 / 19.179) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **18.333 (17.836 / 18.523)** | 87.217 (73.299 / 89.738) | 41.007 (22.270 / 45.121) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **57.190 (56.028 / 57.450)** | - | - | 57.231 (56.875 / 57.659) |
| `reverse_15d_3` | 15D 3^15 reverse | 54.285 (53.578 / 54.682) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **48.020 (47.472 / 48.933)** | 236.128 (234.271 / 238.777) | 60.005 (58.275 / 64.462) | - |
| `reverse_23d_2` | 23D 2^23 reverse | **21.427 (20.641 / 22.758)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 35.463 (35.412 / 35.874) | 272.351 (261.174 / 275.633) | 42.046 (26.370 / 42.383) | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | **462.364 (461.399 / 465.921)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 467.397 (464.424 / 478.341) | 1375.570 (1368.261 / 1394.446) | 672.762 (664.672 / 676.933) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | 21.751 (21.534 / 21.886) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **21.166 (20.851 / 21.273)** | 100.102 (97.584 / 101.812) | 43.167 (25.407 / 50.222) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | 23.644 (23.565 / 23.801) | - | **22.566 (22.390 / 22.895)** | 270.841 (268.007 / 275.028) | 52.537 (45.593 / 60.157) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | **3.492 (3.432 / 3.574)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 6.904 (6.835 / 7.083) | 43.264 (39.323 / 44.483) | 16.181 (8.823 / 17.340) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | 27.229 (26.817 / 29.345) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **26.403 (26.228 / 26.733)** | 92.174 (91.268 / 96.712) | 38.950 (35.227 / 41.386) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 26.699 (26.579 / 26.976) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **26.514 (26.421 / 26.714)** | 137.794 (135.928 / 139.672) | 61.122 (54.149 / 62.774) | - |
