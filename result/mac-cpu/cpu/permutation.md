# CPU Permutation Benchmark Results

- Target profile: `mac-cpu`
- Suite: `cpu/permutation`
- Suite file: `benchmarks/cpu/permutation.yaml`
- Timestamp: `2026-09-23T12:05:30.508140+00:00`
- tenferro-rs commit: `8a1839febeb3c868a502e26397ca10761bbc568d`

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
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | 6.950 (6.940 / 6.961) | skipped (rebuild with --features hptt) | **3.692 (3.673 / 3.707)** | 10.955 (10.827 / 11.137) | 4.481 (4.373 / 10.653) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **2.412 (2.350 / 2.426)** | - | - | 2.445 (2.385 / 2.496) |
| `reverse_15d_3` | 15D 3^15 reverse | 24.696 (24.582 / 24.935) | skipped (rebuild with --features hptt) | 20.432 (20.271 / 20.519) | 56.140 (53.963 / 61.237) | **9.798 (9.438 / 16.922)** | - |
| `reverse_23d_2` | 23D 2^23 reverse | 38.879 (38.699 / 39.081) | skipped (rebuild with --features hptt) | **11.212 (11.170 / 11.522)** | 41.634 (40.821 / 45.320) | 14.200 (13.753 / 20.677) | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | 299.642 (299.476 / 300.224) | skipped (rebuild with --features hptt) | 192.280 (191.953 / 193.032) | 211.134 (190.874 / 223.133) | **191.096 (189.551 / 221.078)** | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | 5.364 (5.335 / 5.389) | skipped (rebuild with --features hptt) | **3.686 (3.674 / 3.697)** | 16.589 (16.458 / 23.964) | 5.921 (5.688 / 12.373) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | 7.459 (7.438 / 7.500) | - | **7.218 (7.162 / 7.315)** | 115.880 (112.943 / 126.349) | 8.811 (8.011 / 15.520) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | 5.287 (5.204 / 5.525) | skipped (rebuild with --features hptt) | **4.532 (4.494 / 4.586)** | 11.509 (10.025 / 15.584) | 5.782 (5.649 / 9.168) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | 11.861 (11.819 / 11.896) | skipped (rebuild with --features hptt) | **10.602 (10.379 / 11.320)** | 15.716 (15.068 / 19.097) | 11.120 (10.922 / 14.679) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 144.735 (144.569 / 145.150) | skipped (rebuild with --features hptt) | **14.001 (13.877 / 15.154)** | 154.736 (125.710 / 160.596) | 113.637 (107.516 / 119.084) | - |

## Threads: 4

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | **2.891 (2.876 / 2.900)** | skipped (rebuild with --features hptt) | 3.674 (3.661 / 3.697) | 11.058 (10.813 / 14.890) | 5.431 (1.506 / 6.630) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **2.378 (2.358 / 2.393)** | - | - | 2.425 (2.398 / 2.468) |
| `reverse_15d_3` | 15D 3^15 reverse | 16.619 (16.564 / 16.757) | skipped (rebuild with --features hptt) | 7.774 (7.739 / 8.992) | 56.062 (52.040 / 58.362) | **7.405 (3.502 / 8.355)** | - |
| `reverse_23d_2` | 23D 2^23 reverse | 16.375 (16.349 / 16.431) | skipped (rebuild with --features hptt) | 5.977 (5.969 / 6.007) | 43.412 (38.560 / 44.056) | **4.123 (3.979 / 4.266)** | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | 127.313 (124.858 / 131.531) | skipped (rebuild with --features hptt) | **75.170 (74.910 / 75.489)** | 192.188 (185.110 / 195.168) | 78.595 (70.792 / 79.209) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | 2.474 (2.452 / 2.481) | skipped (rebuild with --features hptt) | **1.958 (1.942 / 1.999)** | 17.516 (16.530 / 20.699) | 5.652 (1.644 / 5.913) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | 4.048 (3.941 / 4.202) | - | **3.084 (3.056 / 3.129)** | 114.965 (111.585 / 117.859) | 7.385 (3.184 / 14.745) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | 1.822 (1.807 / 1.832) | skipped (rebuild with --features hptt) | **1.412 (1.386 / 1.433)** | 12.567 (10.056 / 20.077) | 1.820 (1.754 / 4.220) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | **4.097 (4.054 / 4.265)** | skipped (rebuild with --features hptt) | 4.104 (4.041 / 4.203) | 17.636 (15.256 / 20.371) | 5.872 (3.375 / 5.898) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 32.418 (32.351 / 32.715) | skipped (rebuild with --features hptt) | **7.926 (7.872 / 8.025)** | 104.072 (99.617 / 115.494) | 35.102 (29.681 / 39.593) | - |
