# CPU Permutation Benchmark Results

- Target profile: `amd-cpu`
- Suite: `cpu/permutation`
- Suite file: `benchmarks/cpu/permutation.yaml`
- Timestamp: `2026-11-25T19:43:58.178450+00:00`
- tenferro-rs commit: `1ba9f33521691170c763dfce9b6c9bb56ca523c2`

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
- Python platform: `Linux-7.0.0-1020-azure-x86_64-with-glibc2.39`
- L3 domains (shared CPUs): `0-3`
- Container: `devcontainer`

Thread counts are controlled via `RAYON_NUM_THREADS` / `OMP_NUM_THREADS` / `JULIA_NUM_THREADS`; no CPU-affinity pinning (`taskset` / `numactl`) was applied. The controlled thread environment is recorded per thread count in the run's `run_t<N>.yaml`.

`tenferro-rs` measures `TypedTensorView::transpose_view` followed by `BackendSession::to_contiguous_read`; one shared session is entered before timing, and input view descriptors and metadata-only `transpose_view` are built outside the timed region. Materialization accepts arbitrary source strides. Every backend allocates a fresh destination inside each timed call, so the table compares allocation-inclusive end-to-end materialization rather than destination-reuse copy kernels. `hptt` only participates in patterns with a contiguous source and destination. Correctness is verified against an internal, untimed odometer reference before any timing; a `FAILED` cell means that backend's output did not match the reference for that pattern.

## Threads: 1

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | 61.110 (60.951 / 61.630) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 63.599 (63.299 / 63.916) | 81.468 (77.481 / 98.841) | **59.948 (55.902 / 83.290)** | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **70.482 (68.161 / 70.777)** | - | - | 70.536 (67.744 / 70.847) |
| `reverse_15d_3` | 15D 3^15 reverse | 108.527 (107.184 / 110.079) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **96.710 (95.597 / 97.282)** | 275.822 (272.837 / 290.562) | 113.538 (110.880 / 134.635) | - |
| `reverse_23d_2` | 23D 2^23 reverse | **31.057 (30.976 / 31.247)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 59.391 (59.181 / 59.598) | 257.764 (251.618 / 265.441) | 73.774 (70.211 / 74.814) | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | **1331.262 (1325.563 / 1340.956)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 1372.738 (1360.602 / 1396.036) | 2021.972 (2014.271 / 2026.540) | 1632.355 (1609.079 / 1634.804) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | **68.783 (67.070 / 69.349)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 70.891 (69.483 / 71.863) | 123.858 (122.923 / 124.249) | 82.775 (81.561 / 82.984) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | **73.951 (72.344 / 74.783)** | - | 75.028 (73.129 / 75.897) | 334.017 (326.976 / 336.712) | 82.914 (82.286 / 83.899) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | **7.940 (7.861 / 8.004)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 21.065 (20.900 / 21.395) | 50.737 (50.609 / 50.826) | 32.362 (32.259 / 32.475) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | 83.286 (81.318 / 84.072) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **81.719 (80.590 / 83.776)** | 106.092 (104.719 / 106.938) | 104.286 (101.781 / 104.637) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 85.047 (82.647 / 85.491) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **84.229 (82.239 / 84.853)** | 146.577 (143.837 / 147.235) | 138.418 (135.563 / 139.173) | - |

## Threads: 4

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | 21.859 (21.777 / 21.980) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **19.817 (19.454 / 20.191)** | 81.932 (78.776 / 92.943) | 20.869 (18.906 / 48.665) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **70.332 (68.271 / 70.961)** | - | - | 70.678 (67.435 / 71.142) |
| `reverse_15d_3` | 15D 3^15 reverse | 60.165 (58.623 / 60.953) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 56.111 (55.641 / 56.729) | 277.132 (272.350 / 291.232) | **44.961 (41.693 / 65.136)** | - |
| `reverse_23d_2` | 23D 2^23 reverse | **26.495 (25.157 / 27.355)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 43.595 (43.368 / 44.281) | 258.199 (252.702 / 278.079) | 27.465 (27.105 / 30.456) | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | **519.038 (481.012 / 568.471)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 519.076 (508.329 / 562.087) | 2034.229 (2022.348 / 2047.660) | 681.195 (669.233 / 728.360) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | **23.419 (23.178 / 23.587)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 33.996 (26.436 / 34.592) | 125.051 (123.750 / 125.298) | 38.685 (38.080 / 38.763) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | 25.584 (25.295 / 25.954) | - | **24.886 (24.756 / 25.841)** | 334.593 (325.753 / 338.110) | 39.142 (36.927 / 39.907) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | **3.518 (3.402 / 3.663)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 7.433 (7.346 / 7.726) | 51.109 (50.986 / 51.270) | 14.423 (13.973 / 15.364) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | **28.130 (28.008 / 28.437)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 28.552 (28.207 / 28.727) | 110.117 (106.948 / 110.220) | 41.262 (38.776 / 41.846) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 27.666 (27.226 / 28.187) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **27.598 (27.322 / 27.947)** | 150.009 (148.417 / 150.118) | 56.609 (55.294 / 57.156) | - |
