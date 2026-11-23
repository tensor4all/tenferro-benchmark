# CPU Permutation Benchmark Results

- Target profile: `amd-cpu`
- Suite: `cpu/permutation`
- Suite file: `benchmarks/cpu/permutation.yaml`
- Timestamp: `2026-11-23T20:11:15.651982+00:00`
- tenferro-rs commit: `ce9125d0cff4ee40841120759b5df45252c27f22`

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
- Python platform: `Linux-7.0.0-1020-azure-x86_64-with-glibc2.39`
- L3 domains (shared CPUs): `0-3`
- Container: `devcontainer`

Thread counts are controlled via `RAYON_NUM_THREADS` / `OMP_NUM_THREADS` / `JULIA_NUM_THREADS`; no CPU-affinity pinning (`taskset` / `numactl`) was applied. The controlled thread environment is recorded per thread count in the run's `run_t<N>.yaml`.

`tenferro-rs` measures `TypedTensorView::transpose_view` followed by `BackendSession::to_contiguous_read`; one shared session is entered before timing, and input view descriptors and metadata-only `transpose_view` are built outside the timed region. Materialization accepts arbitrary source strides. Every backend allocates a fresh destination inside each timed call, so the table compares allocation-inclusive end-to-end materialization rather than destination-reuse copy kernels. `hptt` only participates in patterns with a contiguous source and destination. Correctness is verified against an internal, untimed odometer reference before any timing; a `FAILED` cell means that backend's output did not match the reference for that pattern.

## Threads: 1

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | 68.019 (67.940 / 68.387) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 67.558 (67.338 / 68.089) | 96.643 (95.404 / 112.822) | **63.877 (63.236 / 90.001)** | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **71.377 (71.155 / 71.705)** | - | - | 71.655 (71.324 / 72.416) |
| `reverse_15d_3` | 15D 3^15 reverse | 120.682 (120.240 / 122.672) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **108.860 (108.003 / 112.278)** | 312.811 (306.234 / 332.420) | 126.039 (124.814 / 142.784) | - |
| `reverse_23d_2` | 23D 2^23 reverse | **33.804 (33.696 / 33.943)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 61.653 (61.523 / 62.004) | 323.619 (319.479 / 332.490) | 77.520 (76.510 / 78.915) | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | **1526.912 (1522.344 / 1553.710)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 1585.768 (1568.457 / 1602.777) | 2024.903 (2012.712 / 2040.117) | 1701.734 (1675.722 / 1727.227) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | 81.123 (79.957 / 81.458) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **80.271 (80.003 / 80.661)** | 129.022 (127.703 / 129.222) | 87.811 (87.083 / 88.020) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | **84.092 (81.815 / 84.443)** | - | 84.446 (82.782 / 85.004) | 340.970 (339.701 / 344.894) | 87.638 (87.201 / 87.945) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | **8.032 (7.894 / 8.102)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 22.368 (22.190 / 22.495) | 49.721 (49.593 / 49.806) | 35.809 (35.561 / 36.034) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | 91.764 (90.933 / 92.355) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **88.978 (88.119 / 89.391)** | 118.657 (115.385 / 119.389) | 112.213 (109.948 / 113.185) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 97.270 (95.603 / 97.607) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **95.099 (93.406 / 95.521)** | 158.251 (156.627 / 159.405) | 147.438 (145.950 / 148.512) | - |

## Threads: 4

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | 23.407 (23.184 / 23.844) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 23.198 (22.686 / 23.668) | 95.397 (92.428 / 114.014) | **21.903 (20.610 / 40.748)** | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **70.416 (69.806 / 71.846)** | - | - | 71.751 (70.936 / 73.603) |
| `reverse_15d_3` | 15D 3^15 reverse | 67.056 (66.428 / 67.915) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 58.621 (57.614 / 59.434) | 315.022 (306.628 / 320.509) | **57.150 (50.054 / 88.169)** | - |
| `reverse_23d_2` | 23D 2^23 reverse | **27.409 (26.000 / 28.733)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 45.337 (45.293 / 45.691) | 327.772 (318.804 / 337.977) | 29.437 (28.785 / 30.198) | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | **557.626 (553.472 / 625.636)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 607.176 (567.410 / 635.960) | 2036.715 (2009.408 / 2046.326) | 750.781 (709.822 / 774.801) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | 27.099 (26.553 / 28.179) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **26.415 (26.280 / 26.561)** | 128.198 (125.931 / 130.238) | 40.446 (39.039 / 41.012) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | 29.603 (29.300 / 30.270) | - | **28.444 (28.246 / 29.187)** | 343.726 (342.482 / 344.374) | 42.653 (41.823 / 43.834) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | **5.012 (4.963 / 5.035)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 8.414 (8.321 / 8.606) | 49.771 (49.608 / 49.870) | 15.930 (15.638 / 16.538) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | 32.911 (32.637 / 33.214) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **32.488 (32.148 / 33.227)** | 119.188 (115.965 / 120.044) | 46.332 (45.553 / 47.288) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 32.962 (32.596 / 35.379) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **32.493 (32.274 / 33.147)** | 159.559 (157.570 / 159.790) | 61.725 (61.156 / 62.163) | - |
