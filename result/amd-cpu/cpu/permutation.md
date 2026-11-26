# CPU Permutation Benchmark Results

- Target profile: `amd-cpu`
- Suite: `cpu/permutation`
- Suite file: `benchmarks/cpu/permutation.yaml`
- Timestamp: `2026-11-26T19:42:52.574939+00:00`
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
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | 59.944 (59.771 / 60.111) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 61.523 (61.377 / 61.886) | 82.574 (80.210 / 107.764) | **59.319 (57.801 / 84.402)** | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **67.982 (67.838 / 68.576)** | - | - | 69.008 (68.606 / 69.494) |
| `reverse_15d_3` | 15D 3^15 reverse | 108.292 (107.269 / 109.419) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **96.419 (95.079 / 97.812)** | 278.245 (277.360 / 293.934) | 112.542 (111.593 / 137.686) | - |
| `reverse_23d_2` | 23D 2^23 reverse | **31.131 (30.990 / 31.222)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 59.011 (58.789 / 59.169) | 272.658 (266.399 / 274.602) | 67.997 (66.790 / 88.792) | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | **1261.356 (1254.738 / 1266.478)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 1387.296 (1376.962 / 1390.479) | 2054.319 (2037.349 / 2058.111) | 1622.853 (1615.131 / 1629.935) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | **67.392 (66.854 / 67.769)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 69.321 (69.121 / 69.990) | 122.740 (122.084 / 123.183) | 79.657 (79.034 / 80.168) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | **71.314 (70.489 / 71.635)** | - | 72.105 (71.566 / 72.749) | 330.476 (320.587 / 332.860) | 82.368 (80.405 / 83.190) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | **7.736 (7.683 / 7.849)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 20.754 (20.600 / 21.008) | 50.503 (50.352 / 50.606) | 33.090 (33.022 / 33.175) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | 83.315 (82.824 / 83.598) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **81.380 (81.024 / 81.594)** | 109.574 (106.291 / 110.061) | 104.886 (103.225 / 107.218) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 83.927 (83.568 / 84.305) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **82.840 (82.546 / 82.975)** | 148.333 (146.501 / 149.199) | 137.066 (135.720 / 137.321) | - |

## Threads: 4

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | 22.147 (22.021 / 22.415) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 20.053 (19.454 / 20.118) | 79.723 (77.448 / 101.659) | **19.917 (19.010 / 36.896)** | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **67.860 (67.116 / 68.439)** | - | - | 68.185 (67.316 / 68.674) |
| `reverse_15d_3` | 15D 3^15 reverse | 59.844 (58.943 / 60.077) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 55.536 (55.129 / 56.250) | 277.872 (275.487 / 299.729) | **46.297 (43.762 / 75.345)** | - |
| `reverse_23d_2` | 23D 2^23 reverse | 27.013 (26.920 / 27.449) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 43.807 (43.227 / 44.240) | 284.275 (279.725 / 288.368) | **26.791 (26.390 / 26.819)** | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | **494.490 (477.546 / 527.700)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 566.215 (509.550 / 600.584) | 2011.241 (1991.780 / 2026.856) | 723.117 (665.262 / 730.996) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | 23.763 (23.541 / 23.883) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **23.035 (22.877 / 23.685)** | 122.222 (121.610 / 122.757) | 37.810 (35.062 / 48.679) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | 26.363 (26.224 / 26.709) | - | **26.158 (26.009 / 26.307)** | 333.611 (330.874 / 339.697) | 38.091 (36.256 / 39.010) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | **3.476 (3.443 / 3.519)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 8.031 (7.958 / 8.117) | 50.488 (50.320 / 50.678) | 14.235 (14.046 / 14.514) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | 28.831 (28.539 / 29.171) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **28.465 (28.303 / 28.801)** | 108.571 (106.511 / 109.337) | 42.188 (41.639 / 42.322) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 28.743 (28.563 / 29.303) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **28.335 (28.149 / 28.493)** | 149.334 (146.987 / 149.763) | 56.146 (53.730 / 56.455) | - |
