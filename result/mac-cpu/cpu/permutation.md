# CPU Permutation Benchmark Results

- Target profile: `mac-cpu`
- Suite: `cpu/permutation`
- Suite file: `benchmarks/cpu/permutation.yaml`
- Timestamp: `2026-11-15T18:45:56.591639+00:00`
- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`

## CPU Information

- Model: `Apple M1 (Virtual)`
- Vendor: `Apple`
- Logical CPUs: `3`
- Physical CPUs: `3`
- Sockets: `1`
- Cores per socket: `3`
- Threads per core: `1`
- NUMA nodes: `1`
- Performance levels: `Standard: 3 physical / 3 logical (L1i 192 KiB, L1d 128 KiB, L2 12 MiB, 3 CPUs/L2)`
- Python platform: `macOS-26.7.2-arm64-arm-64bit-Mach-O`
- Container: `none`

Thread counts are controlled via `RAYON_NUM_THREADS` / `OMP_NUM_THREADS` / `JULIA_NUM_THREADS`; no CPU-affinity pinning (`taskset` / `numactl`) was applied. The controlled thread environment is recorded per thread count in the run's `run_t<N>.yaml`.

`tenferro-rs` measures `TypedTensorView::transpose_view` followed by `BackendSession::to_contiguous_read`; one shared session is entered before timing, and input view descriptors and metadata-only `transpose_view` are built outside the timed region. Materialization accepts arbitrary source strides. Every backend allocates a fresh destination inside each timed call, so the table compares allocation-inclusive end-to-end materialization rather than destination-reuse copy kernels. `hptt` only participates in patterns with a contiguous source and destination. Correctness is verified against an internal, untimed odometer reference before any timing; a `FAILED` cell means that backend's output did not match the reference for that pattern.

## Threads: 1

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | **8.879 (8.450 / 9.050)** | skipped (rebuild with --features hptt) | 11.649 (11.165 / 12.894) | 11.928 (11.166 / 12.500) | 12.017 (11.389 / 13.456) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **8.864 (8.212 / 9.596)** | - | - | 9.564 (8.997 / 9.801) |
| `reverse_15d_3` | 15D 3^15 reverse | 47.206 (42.034 / 62.388) | skipped (rebuild with --features hptt) | 36.013 (34.743 / 44.137) | 118.046 (111.652 / 125.639) | **21.251 (20.645 / 21.833)** | - |
| `reverse_23d_2` | 23D 2^23 reverse | 33.169 (30.199 / 39.219) | skipped (rebuild with --features hptt) | **28.248 (26.114 / 31.092)** | 74.677 (68.248 / 101.314) | 31.547 (25.888 / 38.404) | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | **3306.351 (3139.521 / 3797.887)** | skipped (rebuild with --features hptt) | 5327.184 (4560.224 / 6487.654) | 7420.520 (6844.370 / 8120.729) | 4442.426 (3709.580 / 4890.290) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | 10.934 (10.456 / 12.213) | skipped (rebuild with --features hptt) | 12.694 (10.818 / 14.133) | 15.986 (15.657 / 16.497) | **10.048 (9.805 / 10.199)** | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | 33.326 (26.466 / 38.289) | - | 18.564 (17.984 / 20.615) | 193.883 (189.460 / 202.924) | **16.737 (15.768 / 17.072)** | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | 11.523 (10.573 / 13.971) | skipped (rebuild with --features hptt) | 13.200 (12.018 / 15.515) | 20.641 (16.117 / 22.363) | **10.344 (9.447 / 11.532)** | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | 21.728 (20.577 / 27.495) | skipped (rebuild with --features hptt) | 29.866 (27.793 / 31.519) | **20.964 (19.988 / 22.213)** | 23.657 (22.300 / 24.311) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 40.934 (36.577 / 42.538) | skipped (rebuild with --features hptt) | **37.223 (35.473 / 44.055)** | 284.239 (264.854 / 300.934) | 43.096 (34.728 / 48.033) | - |

## Threads: 4

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | **8.237 (7.973 / 8.427)** | skipped (rebuild with --features hptt) | 10.161 (10.101 / 11.586) | 12.364 (11.501 / 14.868) | 8.625 (8.377 / 8.919) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **6.840 (6.814 / 7.351)** | - | - | 6.918 (6.836 / 7.932) |
| `reverse_15d_3` | 15D 3^15 reverse | 25.046 (22.505 / 30.163) | skipped (rebuild with --features hptt) | 33.029 (30.881 / 35.348) | 118.868 (113.467 / 120.812) | **15.205 (14.355 / 15.818)** | - |
| `reverse_23d_2` | 23D 2^23 reverse | 31.843 (26.923 / 48.678) | skipped (rebuild with --features hptt) | 24.137 (22.003 / 27.634) | 75.343 (67.071 / 85.775) | **12.209 (11.101 / 12.676)** | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | **2787.943 (2404.536 / 3660.063)** | skipped (rebuild with --features hptt) | 3228.411 (3043.998 / 3903.464) | 3458.783 (3031.476 / 5216.111) | 3026.587 (2729.014 / 3382.155) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | **8.254 (7.954 / 8.720)** | skipped (rebuild with --features hptt) | 10.947 (10.580 / 12.388) | 23.533 (20.290 / 25.333) | 12.680 (10.672 / 13.275) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | 13.888 (12.416 / 14.276) | - | **12.481 (12.366 / 13.121)** | 193.641 (188.155 / 200.425) | 13.481 (12.157 / 14.455) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | **4.118 (3.879 / 4.485)** | skipped (rebuild with --features hptt) | 4.875 (4.722 / 5.031) | 15.360 (15.081 / 15.892) | 6.196 (5.787 / 6.411) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | 13.349 (11.076 / 13.876) | skipped (rebuild with --features hptt) | 17.018 (16.437 / 19.605) | 18.330 (18.228 / 18.527) | **10.716 (10.232 / 10.846)** | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | **13.637 (13.412 / 14.132)** | skipped (rebuild with --features hptt) | 18.150 (17.596 / 19.306) | 257.087 (249.273 / 263.437) | 16.999 (16.051 / 17.384) | - |
