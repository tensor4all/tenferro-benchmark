# CPU Permutation Benchmark Results

- Target profile: `amd-cpu`
- Suite: `cpu/permutation`
- Suite file: `benchmarks/cpu/permutation.yaml`
- Timestamp: `2026-11-05T20:00:35.399828+00:00`
- tenferro-rs commit: `a000e5c4929d5bebb91e783c329b27106baed12e`

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
- Python platform: `Linux-7.0.0-1017-azure-x86_64-with-glibc2.39`
- L3 domains (shared CPUs): `0-3`
- Container: `devcontainer`

Thread counts are controlled via `RAYON_NUM_THREADS` / `OMP_NUM_THREADS` / `JULIA_NUM_THREADS`; no CPU-affinity pinning (`taskset` / `numactl`) was applied. The controlled thread environment is recorded per thread count in the run's `run_t<N>.yaml`.

`tenferro-rs` measures `TypedTensorView::transpose_view` followed by `BackendSession::to_contiguous_read`; one shared session is entered before timing, and input view descriptors and metadata-only `transpose_view` are built outside the timed region. Materialization accepts arbitrary source strides. Every backend allocates a fresh destination inside each timed call, so the table compares allocation-inclusive end-to-end materialization rather than destination-reuse copy kernels. `hptt` only participates in patterns with a contiguous source and destination. Correctness is verified against an internal, untimed odometer reference before any timing; a `FAILED` cell means that backend's output did not match the reference for that pattern.

## Threads: 1

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | **61.011 (60.935 / 61.328)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 65.553 (65.269 / 65.873) | 111.188 (85.005 / 114.364) | 90.703 (62.537 / 92.137) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **72.297 (71.910 / 73.785)** | - | - | 73.117 (70.016 / 73.930) |
| `reverse_15d_3` | 15D 3^15 reverse | 113.662 (112.471 / 115.433) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **103.298 (102.445 / 104.250)** | 321.284 (294.394 / 323.754) | 136.599 (117.584 / 148.616) | - |
| `reverse_23d_2` | 23D 2^23 reverse | **32.367 (32.276 / 32.469)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 61.289 (60.777 / 61.351) | 284.246 (274.586 / 296.008) | 80.476 (73.398 / 95.781) | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | 1451.403 (1427.112 / 1460.547) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **1449.136 (1440.376 / 1471.865)** | 2074.845 (2059.428 / 2086.151) | 1689.258 (1682.475 / 1691.853) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | 73.238 (71.474 / 74.346) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **72.205 (71.593 / 72.557)** | 139.567 (118.229 / 150.361) | 105.296 (74.651 / 107.066) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | 78.623 (75.873 / 79.683) | - | **77.907 (75.021 / 78.697)** | 345.962 (343.884 / 356.488) | 96.716 (94.697 / 112.213) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | **7.937 (7.847 / 8.056)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 21.483 (21.380 / 21.677) | 56.694 (47.782 / 62.097) | 35.479 (30.585 / 40.155) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | 87.278 (85.260 / 87.887) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **83.535 (83.109 / 84.242)** | 115.035 (112.696 / 121.526) | 110.969 (108.492 / 121.355) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 89.142 (88.931 / 89.429) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **86.984 (85.686 / 87.798)** | 157.199 (153.542 / 164.148) | 146.716 (144.553 / 153.841) | - |

## Threads: 4

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | 21.053 (20.875 / 21.165) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **21.025 (20.758 / 21.436)** | 102.020 (95.960 / 105.403) | 43.405 (20.938 / 44.541) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **73.699 (71.234 / 75.302)** | - | - | 75.139 (72.457 / 75.759) |
| `reverse_15d_3` | 15D 3^15 reverse | 60.991 (60.468 / 61.456) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **58.971 (58.165 / 59.554)** | 316.110 (310.819 / 317.893) | 65.621 (62.081 / 68.198) | - |
| `reverse_23d_2` | 23D 2^23 reverse | **16.825 (16.727 / 16.938)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 45.520 (45.232 / 46.043) | 335.661 (323.497 / 340.108) | 45.551 (28.474 / 47.773) | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | **523.560 (517.592 / 526.812)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 544.505 (539.760 / 548.700) | 2069.843 (2062.145 / 2070.989) | 702.314 (696.204 / 703.226) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | 24.206 (23.844 / 24.601) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **23.685 (23.558 / 24.276)** | 132.429 (130.900 / 136.265) | 44.322 (38.781 / 51.119) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | **26.442 (26.092 / 26.555)** | - | 27.104 (26.217 / 27.358) | 341.218 (339.272 / 343.214) | 47.190 (46.084 / 53.844) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | **3.653 (3.528 / 3.885)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 7.934 (7.819 / 8.146) | 55.189 (52.630 / 56.507) | 18.917 (10.786 / 20.299) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | 30.797 (30.472 / 31.202) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **29.547 (29.394 / 29.941)** | 115.553 (114.335 / 117.859) | 47.707 (44.282 / 50.370) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 29.045 (28.877 / 29.326) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **28.894 (28.738 / 29.218)** | 156.177 (154.788 / 158.168) | 62.232 (60.585 / 65.740) | - |
