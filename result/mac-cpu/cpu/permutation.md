# CPU Permutation Benchmark Results

- Target profile: `mac-cpu`
- Suite: `cpu/permutation`
- Suite file: `benchmarks/cpu/permutation.yaml`
- Timestamp: `2026-08-22T00:15:36.342997+00:00`
- tenferro-rs commit: `a21a4c602fc6700b9bc0c3f1b14ebd19b9d7ec45`

## CPU Information

- Model: `Apple M4`
- Vendor: `Apple`
- Logical CPUs: `10`
- Physical CPUs: `10`
- Sockets: `1`
- Cores per socket: `10`
- Threads per core: `1`
- NUMA nodes: `1`
- Performance levels: `Performance: 4 physical / 4 logical (L1i 192 KiB, L1d 128 KiB, L2 16 MiB, 4 CPUs/L2); Efficiency: 6 physical / 6 logical (L1i 128 KiB, L1d 64 KiB, L2 4 MiB, 6 CPUs/L2)`
- Python platform: `macOS-26.5.2-arm64-arm-64bit`

CPU pinning is unavailable on macOS; thread counts are controlled only via `RAYON_NUM_THREADS` / `OMP_NUM_THREADS` / `JULIA_NUM_THREADS`.

`tenferro-rs` measures `TypedTensorView::transpose_view` followed by `CpuBackend::to_contiguous`; the metadata-only `transpose_view` is built outside the timed region, and `to_contiguous` accepts arbitrary source strides. Every backend allocates a fresh destination inside each timed call, so the table compares allocation-inclusive end-to-end materialization rather than destination-reuse copy kernels. `hptt` only participates in patterns with a contiguous source and destination. Correctness is verified against an internal, untimed odometer reference before any timing; a `FAILED` cell means that backend's output did not match the reference for that pattern.

## Threads: 1

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | 7.918 (7.897 / 8.035) | skipped (rebuild with --features hptt) | **4.813 (4.773 / 4.937)** | 11.840 (11.715 / 12.323) | 7.364 (7.173 / 7.869) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **4.097 (4.060 / 4.181)** | - | - | 4.123 (4.096 / 4.156) |
| `reverse_15d_3` | 15D 3^15 reverse | 34.151 (33.209 / 34.951) | skipped (rebuild with --features hptt) | 27.077 (26.859 / 27.379) | 73.208 (72.722 / 73.511) | **11.632 (11.243 / 11.870)** | - |
| `reverse_23d_2` | 23D 2^23 reverse | 42.682 (42.107 / 43.607) | skipped (rebuild with --features hptt) | **12.924 (12.854 / 12.993)** | 74.013 (60.933 / 94.021) | 13.155 (12.786 / 13.377) | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | 339.271 (336.788 / 341.960) | skipped (rebuild with --features hptt) | 237.016 (236.086 / 237.585) | 243.455 (241.105 / 246.850) | **223.872 (212.138 / 228.280)** | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | 6.052 (6.022 / 6.094) | skipped (rebuild with --features hptt) | **5.299 (5.275 / 5.386)** | 17.693 (17.611 / 17.737) | 6.368 (6.291 / 6.434) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | 8.321 (8.248 / 8.457) | - | **6.816 (6.780 / 6.862)** | 128.215 (126.699 / 129.325) | 8.832 (8.522 / 8.999) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | 5.179 (5.165 / 5.228) | skipped (rebuild with --features hptt) | 5.848 (5.836 / 5.878) | 9.291 (9.161 / 9.358) | **5.125 (5.056 / 5.179)** | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | 13.457 (13.435 / 13.506) | skipped (rebuild with --features hptt) | 13.015 (12.925 / 13.073) | 13.215 (13.125 / 13.257) | **12.416 (12.197 / 12.637)** | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 22.515 (22.388 / 22.680) | skipped (rebuild with --features hptt) | **13.944 (13.917 / 13.989)** | 252.706 (117.355 / 268.247) | 19.987 (19.244 / 20.408) | - |

## Threads: 4

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | **3.859 (3.666 / 6.740)** | skipped (rebuild with --features hptt) | 4.869 (4.785 / 4.945) | 12.254 (11.802 / 12.590) | 4.521 (3.905 / 11.611) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **4.032 (4.011 / 4.197)** | - | - | 4.164 (4.080 / 4.312) |
| `reverse_15d_3` | 15D 3^15 reverse | 25.411 (24.796 / 25.617) | skipped (rebuild with --features hptt) | 12.507 (12.459 / 12.645) | 72.080 (71.461 / 72.677) | **9.353 (7.662 / 14.065)** | - |
| `reverse_23d_2` | 23D 2^23 reverse | 23.620 (23.462 / 24.178) | skipped (rebuild with --features hptt) | 7.224 (7.208 / 7.312) | 67.632 (55.121 / 87.231) | **5.730 (5.100 / 5.992)** | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | 150.410 (147.992 / 152.658) | skipped (rebuild with --features hptt) | 132.693 (131.760 / 133.749) | 240.999 (239.449 / 241.762) | **110.485 (100.767 / 113.211)** | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | **4.450 (4.328 / 4.864)** | skipped (rebuild with --features hptt) | 5.180 (5.139 / 5.243) | 17.685 (17.631 / 17.727) | 4.580 (4.106 / 10.944) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | 7.251 (5.680 / 9.385) | - | **5.362 (5.344 / 5.438)** | 128.971 (127.100 / 129.717) | 11.218 (4.465 / 11.474) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | 2.704 (2.645 / 2.868) | skipped (rebuild with --features hptt) | **1.830 (1.815 / 1.895)** | 9.440 (9.301 / 9.535) | 2.738 (2.596 / 6.543) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | 6.434 (5.661 / 7.011) | skipped (rebuild with --features hptt) | **6.267 (6.149 / 6.552)** | 13.279 (13.188 / 13.294) | 8.780 (4.894 / 10.242) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | **9.471 (8.988 / 10.351)** | skipped (rebuild with --features hptt) | 9.979 (9.875 / 10.486) | 209.323 (110.160 / 271.363) | 18.619 (13.990 / 25.020) | - |
