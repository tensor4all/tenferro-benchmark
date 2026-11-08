# CPU Permutation Benchmark Results

- Target profile: `mac-cpu`
- Suite: `cpu/permutation`
- Suite file: `benchmarks/cpu/permutation.yaml`
- Timestamp: `2026-11-08T18:47:05.557137+00:00`
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
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | **9.906 (9.408 / 10.297)** | skipped (rebuild with --features hptt) | 11.632 (10.657 / 12.787) | 26.793 (15.395 / 37.038) | 23.997 (13.054 / 25.352) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **10.367 (9.394 / 11.700)** | - | - | 10.526 (10.105 / 11.548) |
| `reverse_15d_3` | 15D 3^15 reverse | 43.386 (38.791 / 47.743) | skipped (rebuild with --features hptt) | 38.856 (35.196 / 43.254) | 139.220 (130.502 / 145.233) | **33.155 (26.393 / 36.975)** | - |
| `reverse_23d_2` | 23D 2^23 reverse | 48.349 (42.528 / 52.461) | skipped (rebuild with --features hptt) | **27.657 (26.798 / 34.668)** | 75.258 (64.435 / 81.159) | 36.679 (27.255 / 38.905) | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | **4108.261 (3563.956 / 4883.711)** | skipped (rebuild with --features hptt) | 6707.515 (4414.329 / 8177.701) | 8858.290 (6513.686 / 9693.500) | 7206.426 (5098.284 / 11965.845) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | **11.376 (11.130 / 13.927)** | skipped (rebuild with --features hptt) | 13.102 (12.603 / 13.339) | 29.179 (16.356 / 30.682) | 20.149 (10.293 / 21.913) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | 19.582 (17.491 / 21.480) | - | **19.130 (18.457 / 20.461)** | 206.725 (193.328 / 252.138) | 28.579 (18.255 / 36.733) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | 11.750 (11.134 / 12.688) | skipped (rebuild with --features hptt) | **11.084 (10.887 / 11.418)** | 28.247 (22.336 / 32.501) | 11.293 (8.828 / 15.359) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | **21.523 (21.012 / 22.217)** | skipped (rebuild with --features hptt) | 24.096 (23.781 / 26.279) | 25.785 (21.010 / 38.197) | 29.597 (23.455 / 34.990) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 31.465 (29.440 / 35.650) | skipped (rebuild with --features hptt) | **30.729 (29.209 / 33.959)** | 280.611 (272.322 / 284.752) | 56.861 (48.646 / 60.745) | - |

## Threads: 4

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | **8.358 (8.139 / 8.540)** | skipped (rebuild with --features hptt) | 10.848 (10.564 / 11.175) | 23.674 (15.611 / 28.976) | 18.571 (9.349 / 20.320) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **7.492 (7.383 / 7.920)** | - | - | 7.721 (7.200 / 8.267) |
| `reverse_15d_3` | 15D 3^15 reverse | 26.877 (19.432 / 27.722) | skipped (rebuild with --features hptt) | **24.544 (23.224 / 26.537)** | 165.400 (152.602 / 176.512) | 31.522 (27.369 / 35.162) | - |
| `reverse_23d_2` | 23D 2^23 reverse | 24.729 (19.709 / 26.656) | skipped (rebuild with --features hptt) | **16.138 (15.422 / 17.140)** | 71.942 (67.471 / 74.816) | 18.639 (16.911 / 20.145) | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | 7282.916 (4512.230 / 8832.187) | skipped (rebuild with --features hptt) | **2513.283 (2359.840 / 3702.384)** | 7146.209 (4145.800 / 7655.049) | 7288.483 (6423.692 / 8118.890) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | **9.709 (9.375 / 10.144)** | skipped (rebuild with --features hptt) | 12.298 (11.192 / 14.117) | 40.021 (28.019 / 53.574) | 25.945 (16.296 / 28.402) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | 17.090 (14.568 / 20.655) | - | **14.523 (14.252 / 15.560)** | 283.046 (261.627 / 293.338) | 26.456 (19.972 / 30.326) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | **5.403 (4.630 / 6.282)** | skipped (rebuild with --features hptt) | 6.040 (5.775 / 7.055) | 21.770 (20.301 / 26.873) | 7.701 (5.181 / 9.820) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | **9.807 (9.571 / 10.215)** | skipped (rebuild with --features hptt) | 15.825 (15.234 / 17.615) | 25.034 (20.499 / 25.987) | 15.696 (13.321 / 16.743) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | **14.387 (12.809 / 16.587)** | skipped (rebuild with --features hptt) | 14.509 (14.028 / 15.522) | 278.848 (254.752 / 286.661) | 25.001 (23.991 / 28.166) | - |
