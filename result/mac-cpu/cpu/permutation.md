# CPU Permutation Benchmark Results

- Target profile: `mac-cpu`
- Suite: `cpu/permutation`
- Suite file: `benchmarks/cpu/permutation.yaml`
- Timestamp: `2026-09-13T09:46:03.007520+00:00`
- tenferro-rs commit: `28dfc7e383a653e1461d38b73ced3d563b4efdad`

## CPU Information

- Model: `Apple M5 Max`
- Vendor: `Apple`
- Logical CPUs: `18`
- Physical CPUs: `18`
- Sockets: `1`
- Cores per socket: `18`
- Threads per core: `1`
- NUMA nodes: `1`
- Performance levels: `Super: 6 physical / 6 logical (L1i 192 KiB, L1d 128 KiB, L2 16 MiB, 6 CPUs/L2); Performance: 12 physical / 12 logical (L1i 128 KiB, L1d 64 KiB, L2 8 MiB, 6 CPUs/L2)`
- Python platform: `macOS-26.5.1-arm64-arm-64bit-Mach-O`

CPU pinning is unavailable on macOS; thread counts are controlled only via `RAYON_NUM_THREADS` / `OMP_NUM_THREADS` / `JULIA_NUM_THREADS`.

`tenferro-rs` measures `TypedTensorView::transpose_view` followed by `BackendSession::to_contiguous_read`; one shared session is entered before timing, and input view descriptors and metadata-only `transpose_view` are built outside the timed region. Materialization accepts arbitrary source strides. Every backend allocates a fresh destination inside each timed call, so the table compares allocation-inclusive end-to-end materialization rather than destination-reuse copy kernels. `hptt` only participates in patterns with a contiguous source and destination. Correctness is verified against an internal, untimed odometer reference before any timing; a `FAILED` cell means that backend's output did not match the reference for that pattern.

## Threads: 1

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | 7.103 (7.075 / 7.159) | skipped (rebuild with --features hptt) | **3.702 (3.688 / 3.724)** | 10.818 (10.718 / 17.559) | 4.531 (4.376 / 10.895) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **2.375 (2.354 / 2.409)** | - | - | 2.459 (2.442 / 2.503) |
| `reverse_15d_3` | 15D 3^15 reverse | 30.585 (30.494 / 30.757) | skipped (rebuild with --features hptt) | 20.897 (20.877 / 20.981) | 56.234 (52.974 / 62.967) | **9.245 (9.066 / 15.750)** | - |
| `reverse_23d_2` | 23D 2^23 reverse | 37.797 (37.690 / 38.095) | skipped (rebuild with --features hptt) | 11.650 (11.532 / 11.950) | 40.775 (38.969 / 43.853) | **8.980 (8.760 / 15.303)** | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | 276.664 (275.355 / 277.437) | skipped (rebuild with --features hptt) | 197.052 (197.019 / 197.700) | 211.569 (185.869 / 217.507) | **183.925 (182.283 / 214.486)** | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | 5.345 (5.336 / 5.355) | skipped (rebuild with --features hptt) | **3.714 (3.697 / 3.742)** | 16.414 (16.341 / 23.579) | 5.967 (5.717 / 13.212) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | 7.765 (7.696 / 7.841) | - | **6.870 (6.815 / 6.948)** | 114.464 (111.210 / 126.397) | 9.663 (7.756 / 15.195) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | 5.724 (5.672 / 5.773) | skipped (rebuild with --features hptt) | **4.685 (4.649 / 4.729)** | 10.805 (10.028 / 15.934) | 5.679 (5.629 / 9.247) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | 11.987 (11.943 / 12.097) | skipped (rebuild with --features hptt) | 11.274 (11.031 / 11.582) | 15.069 (14.918 / 18.881) | **11.126 (10.769 / 14.696)** | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 124.399 (124.146 / 124.743) | skipped (rebuild with --features hptt) | **15.497 (15.410 / 15.802)** | 108.049 (100.791 / 121.508) | 92.775 (88.330 / 100.136) | - |

## Threads: 4

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | **3.029 (3.010 / 3.374)** | skipped (rebuild with --features hptt) | 3.679 (3.668 / 3.688) | 10.856 (10.793 / 15.068) | 5.587 (1.476 / 5.682) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | 2.539 (2.519 / 2.556) | - | - | **2.526 (2.459 / 2.553)** |
| `reverse_15d_3` | 15D 3^15 reverse | 20.892 (20.662 / 25.000) | skipped (rebuild with --features hptt) | 7.888 (7.871 / 7.951) | 56.331 (53.277 / 60.583) | **7.562 (3.493 / 8.478)** | - |
| `reverse_23d_2` | 23D 2^23 reverse | 21.319 (21.247 / 21.346) | skipped (rebuild with --features hptt) | 6.077 (6.031 / 6.129) | 42.142 (41.400 / 43.290) | **5.538 (5.382 / 5.600)** | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | 113.204 (112.770 / 113.623) | skipped (rebuild with --features hptt) | **76.363 (76.240 / 76.544)** | 193.992 (186.318 / 195.896) | 77.173 (70.758 / 78.469) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | 2.631 (2.628 / 2.646) | skipped (rebuild with --features hptt) | **1.974 (1.961 / 1.998)** | 16.925 (16.521 / 22.989) | 2.098 (1.608 / 5.824) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | 5.193 (5.070 / 5.396) | - | **3.227 (3.172 / 3.325)** | 113.378 (110.951 / 115.850) | 7.483 (3.148 / 14.419) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | 2.023 (1.990 / 2.057) | skipped (rebuild with --features hptt) | **1.406 (1.395 / 1.423)** | 12.836 (10.170 / 20.510) | 1.880 (1.766 / 4.481) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | **3.984 (3.959 / 4.005)** | skipped (rebuild with --features hptt) | 4.095 (4.048 / 4.182) | 17.653 (15.120 / 20.891) | 5.951 (3.355 / 13.169) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 43.518 (42.158 / 43.990) | skipped (rebuild with --features hptt) | **8.144 (8.118 / 8.220)** | 120.273 (105.378 / 147.506) | 34.238 (32.438 / 39.618) | - |
