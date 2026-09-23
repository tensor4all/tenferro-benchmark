# CPU Permutation Benchmark Results

- Target profile: `mac-cpu`
- Suite: `cpu/permutation`
- Suite file: `benchmarks/cpu/permutation.yaml`
- Timestamp: `2026-09-23T02:19:51.533901+00:00`
- tenferro-rs commit: `f6279624e964c253ad21dec6521d82df6b00f7db`

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
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | 7.072 (6.988 / 7.104) | skipped (rebuild with --features hptt) | **3.701 (3.688 / 3.732)** | 10.917 (10.750 / 11.194) | 4.376 (4.276 / 11.056) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **2.379 (2.351 / 2.410)** | - | - | 2.470 (2.417 / 2.527) |
| `reverse_15d_3` | 15D 3^15 reverse | 30.517 (30.431 / 30.760) | skipped (rebuild with --features hptt) | 20.779 (20.682 / 20.867) | 55.337 (52.710 / 61.220) | **9.370 (8.909 / 15.632)** | - |
| `reverse_23d_2` | 23D 2^23 reverse | 37.954 (37.521 / 38.060) | skipped (rebuild with --features hptt) | 11.675 (11.511 / 11.842) | 40.755 (39.474 / 44.380) | **9.264 (8.955 / 9.656)** | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | 299.919 (299.023 / 301.022) | skipped (rebuild with --features hptt) | 207.789 (207.286 / 207.999) | **187.146 (179.665 / 210.816)** | 187.341 (185.344 / 216.245) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | 5.334 (5.322 / 5.338) | skipped (rebuild with --features hptt) | **3.704 (3.698 / 3.712)** | 16.605 (16.441 / 23.116) | 5.629 (5.474 / 12.946) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | 7.451 (7.421 / 7.486) | - | **7.093 (7.062 / 7.204)** | 114.369 (110.848 / 125.592) | 8.810 (7.640 / 14.585) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | 5.785 (5.666 / 5.916) | skipped (rebuild with --features hptt) | **4.681 (4.632 / 4.735)** | 10.555 (9.890 / 15.962) | 5.727 (5.662 / 9.294) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | 12.162 (12.105 / 12.215) | skipped (rebuild with --features hptt) | 11.154 (10.922 / 11.669) | 15.158 (14.875 / 18.545) | **10.974 (10.901 / 14.646)** | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 134.355 (134.029 / 135.067) | skipped (rebuild with --features hptt) | **15.344 (15.104 / 15.675)** | 128.800 (79.098 / 152.844) | 85.085 (81.382 / 87.047) | - |

## Threads: 4

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | **2.905 (2.896 / 2.907)** | skipped (rebuild with --features hptt) | 3.683 (3.675 / 3.695) | 11.149 (10.809 / 15.083) | 5.573 (1.473 / 5.691) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **2.401 (2.385 / 2.435)** | - | - | 2.431 (2.408 / 2.480) |
| `reverse_15d_3` | 15D 3^15 reverse | 20.745 (20.646 / 24.982) | skipped (rebuild with --features hptt) | 7.815 (7.777 / 7.902) | 55.873 (52.378 / 60.318) | **7.623 (3.508 / 8.465)** | - |
| `reverse_23d_2` | 23D 2^23 reverse | 21.315 (21.276 / 21.430) | skipped (rebuild with --features hptt) | 6.140 (6.113 / 6.190) | 42.509 (40.279 / 44.399) | **4.598 (4.479 / 4.753)** | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | 113.844 (113.341 / 114.066) | skipped (rebuild with --features hptt) | **76.889 (76.797 / 76.985)** | 179.138 (170.466 / 183.358) | 78.659 (71.026 / 78.893) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | 2.630 (2.610 / 2.647) | skipped (rebuild with --features hptt) | **1.979 (1.965 / 1.991)** | 17.419 (16.536 / 20.697) | 5.969 (1.660 / 6.862) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | 4.179 (4.101 / 4.325) | - | **3.208 (3.156 / 3.249)** | 115.171 (111.788 / 117.420) | 7.506 (3.145 / 14.338) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | 1.796 (1.767 / 1.880) | skipped (rebuild with --features hptt) | **1.420 (1.405 / 1.460)** | 12.765 (10.091 / 20.262) | 1.872 (1.777 / 4.253) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | **3.970 (3.945 / 3.980)** | skipped (rebuild with --features hptt) | 4.081 (4.073 / 4.192) | 17.513 (15.270 / 18.534) | 5.968 (3.437 / 13.061) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 38.790 (38.546 / 39.022) | skipped (rebuild with --features hptt) | **8.144 (8.071 / 8.188)** | 127.477 (102.649 / 284.627) | 36.980 (33.570 / 41.546) | - |
