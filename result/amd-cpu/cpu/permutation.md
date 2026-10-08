# CPU Permutation Benchmark Results

- Target profile: `amd-cpu`
- Suite: `cpu/permutation`
- Suite file: `benchmarks/cpu/permutation.yaml`
- Timestamp: `2026-10-08T06:29:10.793239+00:00`
- tenferro-rs commit: `5cf78c7ec0ad9516dd78bab546d5e7bd42fa3102`

## CPU Information

- Model: `AMD Ryzen 9 9955HX 16-Core Processor`
- Vendor: `AuthenticAMD`
- Logical CPUs: `32`
- Sockets: `1`
- Cores per socket: `16`
- Threads per core: `2`
- NUMA nodes: `1`
- Python platform: `Linux-7.0.0-38-generic-x86_64-with-glibc2.39`

On the Linux CPU devcontainer, thread counts are controlled via `RAYON_NUM_THREADS` / `OMP_NUM_THREADS` / `JULIA_NUM_THREADS`; no CPU-affinity pinning (`taskset` / `numactl`) is applied, matching the repository devcontainer convention. The controlled thread environment is recorded per thread count in the run's `run_t<N>.yaml`.

`tenferro-rs` measures `TypedTensorView::transpose_view` followed by `BackendSession::to_contiguous_read`; one shared session is entered before timing, and input view descriptors and metadata-only `transpose_view` are built outside the timed region. Materialization accepts arbitrary source strides. Every backend allocates a fresh destination inside each timed call, so the table compares allocation-inclusive end-to-end materialization rather than destination-reuse copy kernels. `hptt` only participates in patterns with a contiguous source and destination. Correctness is verified against an internal, untimed odometer reference before any timing; a `FAILED` cell means that backend's output did not match the reference for that pattern.

## Threads: 1

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | 43.428 (43.293 / 43.641) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **30.912 (30.832 / 31.054)** | 70.271 (48.917 / 72.250) | 70.970 (48.568 / 74.026) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **33.059 (32.716 / 34.073)** | - | - | 33.694 (33.060 / 34.125) |
| `reverse_15d_3` | 15D 3^15 reverse | 174.645 (172.916 / 175.021) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **68.357 (68.276 / 68.489)** | 228.213 (207.518 / 229.146) | 115.139 (93.134 / 116.559) | - |
| `reverse_23d_2` | 23D 2^23 reverse | 68.456 (67.816 / 70.443) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **39.152 (39.024 / 39.266)** | 189.556 (187.501 / 204.933) | 63.183 (59.431 / 77.662) | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | 801.105 (800.025 / 802.949) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **645.027 (642.611 / 649.145)** | 930.205 (922.960 / 936.123) | 1027.967 (1007.088 / 1036.917) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | 45.538 (45.400 / 45.759) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **36.591 (36.403 / 36.766)** | 100.711 (77.936 / 104.983) | 85.408 (61.491 / 86.067) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | 51.610 (51.391 / 51.780) | - | **38.616 (38.226 / 39.803)** | 278.056 (277.639 / 289.547) | 81.224 (80.720 / 92.976) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | **12.605 (12.597 / 12.622)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 12.778 (12.759 / 12.863) | 49.170 (43.830 / 55.919) | 29.847 (22.699 / 30.596) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | 54.298 (54.228 / 54.369) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **41.991 (41.721 / 42.368)** | 70.279 (59.114 / 75.546) | 70.098 (61.264 / 76.621) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 79.303 (77.644 / 80.023) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **52.152 (52.071 / 52.401)** | 114.556 (114.044 / 120.723) | 97.905 (96.180 / 106.642) | - |

## Threads: 4

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | **13.297 (13.168 / 16.297)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 30.771 (30.671 / 30.842) | 61.158 (49.089 / 62.557) | 27.945 (14.200 / 30.961) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **34.258 (34.140 / 34.376)** | - | - | 34.281 (33.834 / 34.647) |
| `reverse_15d_3` | 15D 3^15 reverse | 58.290 (57.717 / 58.665) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **37.132 (36.886 / 37.189)** | 219.260 (218.332 / 220.289) | 46.466 (45.565 / 47.701) | - |
| `reverse_23d_2` | 23D 2^23 reverse | **18.049 (17.910 / 18.425)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 27.438 (27.326 / 27.665) | 189.504 (185.550 / 201.339) | 27.292 (15.843 / 29.083) | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | **217.972 (217.632 / 218.641)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 244.851 (244.446 / 245.176) | 933.426 (925.739 / 942.930) | 350.841 (332.753 / 353.981) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | 14.066 (13.605 / 14.276) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **12.252 (12.126 / 12.618)** | 89.811 (88.965 / 90.397) | 33.849 (26.592 / 37.445) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | 15.198 (14.921 / 15.513) | - | **12.858 (12.711 / 13.049)** | 273.029 (272.827 / 273.267) | 34.109 (31.919 / 36.338) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | **3.384 (3.085 / 3.434)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 3.989 (3.910 / 4.150) | 47.123 (46.988 / 49.644) | 11.705 (9.212 / 14.046) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | 16.072 (15.829 / 16.437) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **13.677 (13.503 / 13.850)** | 67.727 (67.404 / 69.981) | 26.950 (23.139 / 30.670) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 22.180 (21.809 / 23.173) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **15.667 (15.545 / 15.978)** | 114.239 (112.580 / 114.510) | 34.011 (31.975 / 38.154) | - |
