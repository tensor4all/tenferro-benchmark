# CPU Permutation Benchmark Results

- Target profile: `mac-cpu`
- Suite: `cpu/permutation`
- Suite file: `benchmarks/cpu/permutation.yaml`
- Timestamp: `2026-09-23T09:18:30.124026+00:00`
- tenferro-rs commit: `412852a8cfb13b14246443fc09b1c0c34f5ce1cd`

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
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | 6.906 (6.887 / 6.984) | skipped (rebuild with --features hptt) | **3.705 (3.689 / 3.715)** | 10.797 (10.700 / 17.376) | 4.432 (4.334 / 10.440) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **2.396 (2.362 / 2.421)** | - | - | 2.421 (2.386 / 2.518) |
| `reverse_15d_3` | 15D 3^15 reverse | 24.906 (24.719 / 25.000) | skipped (rebuild with --features hptt) | 20.932 (20.724 / 21.420) | 55.639 (52.570 / 63.528) | **8.950 (8.881 / 15.672)** | - |
| `reverse_23d_2` | 23D 2^23 reverse | 37.558 (36.690 / 38.196) | skipped (rebuild with --features hptt) | **11.458 (11.308 / 11.545)** | 49.322 (45.026 / 51.193) | 14.708 (14.480 / 14.960) | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | 296.085 (295.824 / 296.654) | skipped (rebuild with --features hptt) | 200.754 (200.439 / 201.129) | 210.828 (184.741 / 215.116) | **189.141 (186.000 / 217.082)** | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | 5.352 (5.310 / 5.471) | skipped (rebuild with --features hptt) | **3.694 (3.684 / 3.707)** | 16.502 (16.402 / 23.546) | 5.870 (5.739 / 12.145) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | 7.437 (7.406 / 7.510) | - | **6.341 (6.311 / 6.532)** | 114.653 (111.351 / 120.247) | 7.818 (7.491 / 14.337) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | 5.956 (5.766 / 6.138) | skipped (rebuild with --features hptt) | **4.538 (4.496 / 4.603)** | 12.777 (10.862 / 18.042) | 5.955 (5.812 / 9.490) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | 12.421 (12.031 / 12.720) | skipped (rebuild with --features hptt) | **11.148 (10.955 / 11.422)** | 15.309 (14.981 / 18.825) | 12.984 (11.435 / 15.628) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 84.467 (78.847 / 86.478) | skipped (rebuild with --features hptt) | **14.410 (14.170 / 14.897)** | 124.028 (106.502 / 149.337) | 110.945 (104.325 / 112.353) | - |

## Threads: 4

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | 4.060 (3.584 / 4.176) | skipped (rebuild with --features hptt) | 3.698 (3.688 / 3.717) | 11.124 (10.840 / 14.845) | **1.988 (1.585 / 6.138)** | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **2.392 (2.354 / 2.410)** | - | - | 2.394 (2.367 / 2.407) |
| `reverse_15d_3` | 15D 3^15 reverse | 15.853 (14.092 / 16.126) | skipped (rebuild with --features hptt) | 9.502 (9.317 / 9.529) | 56.108 (53.528 / 61.011) | **4.815 (4.085 / 7.676)** | - |
| `reverse_23d_2` | 23D 2^23 reverse | 14.976 (14.739 / 15.124) | skipped (rebuild with --features hptt) | 7.408 (7.372 / 7.553) | 44.596 (43.607 / 49.526) | **5.857 (5.742 / 6.012)** | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | 113.129 (112.919 / 113.511) | skipped (rebuild with --features hptt) | 80.315 (80.061 / 80.930) | 198.533 (192.092 / 204.009) | **79.000 (71.476 / 79.238)** | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | 2.557 (2.543 / 2.594) | skipped (rebuild with --features hptt) | **1.778 (1.742 / 1.814)** | 17.708 (16.957 / 21.291) | 5.957 (1.673 / 7.558) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | 4.621 (4.326 / 4.686) | - | **2.799 (2.779 / 2.852)** | 116.818 (112.523 / 117.862) | 7.729 (3.184 / 14.855) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | 1.772 (1.751 / 1.813) | skipped (rebuild with --features hptt) | **1.407 (1.392 / 1.424)** | 14.702 (12.196 / 21.935) | 2.053 (1.934 / 5.044) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | **4.030 (4.008 / 4.319)** | skipped (rebuild with --features hptt) | 4.113 (4.069 / 4.209) | 18.067 (15.596 / 19.047) | 6.354 (3.666 / 8.245) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 34.458 (34.235 / 34.893) | skipped (rebuild with --features hptt) | **8.004 (7.891 / 8.055)** | 99.339 (92.118 / 133.003) | 29.833 (27.320 / 37.206) | - |
