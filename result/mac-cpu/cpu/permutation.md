# CPU Permutation Benchmark Results

- Target profile: `mac-cpu`
- Suite: `cpu/permutation`
- Suite file: `benchmarks/cpu/permutation.yaml`
- Timestamp: `2026-09-13T03:51:30.405231+00:00`
- tenferro-rs commit: `a48866b1a0bb52e6f9712c485925105b14ea30b9`

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

`tenferro-rs` measures `TypedTensorView::transpose_view` followed by `CpuBackend::to_contiguous`; the metadata-only `transpose_view` is built outside the timed region, and `to_contiguous` accepts arbitrary source strides. Every backend allocates a fresh destination inside each timed call, so the table compares allocation-inclusive end-to-end materialization rather than destination-reuse copy kernels. `hptt` only participates in patterns with a contiguous source and destination. Correctness is verified against an internal, untimed odometer reference before any timing; a `FAILED` cell means that backend's output did not match the reference for that pattern.

## Threads: 1

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | 7.335 (7.150 / 7.418) | skipped (rebuild with --features hptt) | **3.779 (3.751 / 3.795)** | 11.160 (11.119 / 11.189) | 4.839 (4.552 / 4.882) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **2.478 (2.461 / 2.708)** | - | - | 2.669 (2.600 / 2.733) |
| `reverse_15d_3` | 15D 3^15 reverse | 31.356 (31.007 / 31.665) | skipped (rebuild with --features hptt) | 21.431 (21.288 / 21.606) | 56.661 (55.120 / 57.408) | **9.889 (9.664 / 10.062)** | - |
| `reverse_23d_2` | 23D 2^23 reverse | 47.517 (47.102 / 48.491) | skipped (rebuild with --features hptt) | 12.263 (12.144 / 12.542) | 50.898 (48.746 / 52.112) | **9.542 (9.331 / 9.843)** | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | 307.864 (304.096 / 312.386) | skipped (rebuild with --features hptt) | 212.254 (210.385 / 217.675) | 198.762 (196.029 / 201.311) | **188.910 (186.488 / 190.139)** | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | 5.574 (5.532 / 5.601) | skipped (rebuild with --features hptt) | **3.731 (3.701 / 3.776)** | 16.960 (16.830 / 17.403) | 6.213 (6.085 / 6.233) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | 7.638 (7.623 / 7.664) | - | **6.594 (6.467 / 6.683)** | 115.229 (114.168 / 115.774) | 7.801 (7.671 / 7.841) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | 6.671 (6.569 / 6.804) | skipped (rebuild with --features hptt) | **4.976 (4.946 / 5.029)** | 12.615 (12.484 / 12.883) | 5.923 (5.852 / 5.987) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | 13.241 (13.094 / 13.477) | skipped (rebuild with --features hptt) | 12.124 (11.946 / 12.294) | 15.203 (15.149 / 15.432) | **11.162 (11.063 / 11.191)** | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 105.478 (104.935 / 115.940) | skipped (rebuild with --features hptt) | **16.292 (16.197 / 16.422)** | 132.829 (106.970 / 151.740) | 106.054 (104.967 / 107.531) | - |

## Threads: 4

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | 3.087 (2.974 / 3.172) | skipped (rebuild with --features hptt) | 3.730 (3.722 / 3.757) | 11.437 (11.293 / 11.479) | **1.878 (1.530 / 5.848)** | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **2.432 (2.415 / 2.478)** | - | - | 2.546 (2.477 / 2.858) |
| `reverse_15d_3` | 15D 3^15 reverse | 24.935 (24.777 / 25.022) | skipped (rebuild with --features hptt) | 9.769 (9.583 / 9.805) | 57.409 (56.539 / 58.252) | **8.481 (4.258 / 9.390)** | - |
| `reverse_23d_2` | 23D 2^23 reverse | 21.885 (21.847 / 22.096) | skipped (rebuild with --features hptt) | 7.873 (7.852 / 7.959) | 66.689 (62.319 / 67.124) | **6.497 (6.073 / 6.773)** | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | 262.204 (260.198 / 262.715) | skipped (rebuild with --features hptt) | **70.075 (69.820 / 70.335)** | 199.363 (186.541 / 209.553) | 80.992 (72.928 / 81.257) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | 6.331 (6.300 / 6.383) | skipped (rebuild with --features hptt) | **1.682 (1.666 / 1.751)** | 17.048 (16.985 / 17.074) | 6.311 (1.791 / 6.591) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | 4.252 (4.129 / 4.348) | - | **2.864 (2.817 / 2.884)** | 114.130 (113.506 / 114.497) | 7.994 (3.059 / 15.371) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | 1.943 (1.868 / 1.991) | skipped (rebuild with --features hptt) | **1.434 (1.416 / 1.464)** | 12.441 (12.308 / 12.660) | 2.006 (1.812 / 4.455) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | 4.669 (4.427 / 4.867) | skipped (rebuild with --features hptt) | **4.235 (4.205 / 4.278)** | 15.706 (15.652 / 15.761) | 6.126 (3.708 / 6.373) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 28.868 (28.317 / 29.274) | skipped (rebuild with --features hptt) | **8.543 (8.450 / 8.682)** | 101.381 (95.149 / 103.666) | 34.840 (32.017 / 40.908) | - |
