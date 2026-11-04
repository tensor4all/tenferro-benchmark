# CPU Permutation Benchmark Results

- Target profile: `mac-cpu`
- Suite: `cpu/permutation`
- Suite file: `benchmarks/cpu/permutation.yaml`
- Timestamp: `2026-11-04T18:21:53.484278+00:00`
- tenferro-rs commit: `a000e5c4929d5bebb91e783c329b27106baed12e`

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
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | 11.180 (10.479 / 11.243) | skipped (rebuild with --features hptt) | **9.772 (8.780 / 10.328)** | 21.652 (11.083 / 22.837) | 12.836 (11.967 / 23.894) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **6.477 (6.447 / 6.522)** | - | - | 7.635 (6.501 / 8.257) |
| `reverse_15d_3` | 15D 3^15 reverse | 33.202 (31.057 / 36.438) | skipped (rebuild with --features hptt) | 32.405 (29.372 / 34.445) | 122.607 (111.521 / 126.026) | **26.065 (20.811 / 31.803)** | - |
| `reverse_23d_2` | 23D 2^23 reverse | 38.639 (28.721 / 44.387) | skipped (rebuild with --features hptt) | 29.305 (25.979 / 33.389) | 63.594 (57.348 / 71.811) | **27.402 (21.287 / 31.404)** | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | 2419.379 (2227.964 / 2831.678) | skipped (rebuild with --features hptt) | **2356.027 (2104.147 / 2473.335)** | 3081.836 (2690.706 / 3522.167) | 2859.785 (2487.369 / 3073.060) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | **10.002 (8.590 / 10.242)** | skipped (rebuild with --features hptt) | 11.321 (9.796 / 11.912) | 25.172 (15.809 / 27.198) | 17.164 (8.481 / 20.436) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | **12.837 (12.381 / 14.976)** | - | 15.135 (14.295 / 17.180) | 182.503 (174.175 / 190.947) | 26.925 (16.974 / 30.357) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | 8.194 (8.065 / 8.891) | skipped (rebuild with --features hptt) | 8.729 (8.443 / 9.357) | 17.915 (15.589 / 20.078) | **7.606 (6.192 / 10.668)** | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | **17.034 (15.428 / 17.463)** | skipped (rebuild with --features hptt) | 17.890 (17.602 / 20.059) | 21.003 (16.829 / 25.545) | 21.150 (18.842 / 22.962) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | **18.952 (18.691 / 24.726)** | skipped (rebuild with --features hptt) | 20.837 (20.690 / 27.085) | 346.125 (262.949 / 370.766) | 42.167 (37.062 / 44.636) | - |

## Threads: 4

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | **7.297 (7.247 / 7.631)** | skipped (rebuild with --features hptt) | 9.917 (9.734 / 10.081) | 18.152 (11.807 / 20.272) | 14.267 (8.207 / 14.502) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **6.470 (6.447 / 7.967)** | - | - | 6.648 (6.464 / 7.519) |
| `reverse_15d_3` | 15D 3^15 reverse | 19.267 (18.649 / 20.248) | skipped (rebuild with --features hptt) | **17.545 (17.234 / 19.941)** | 118.317 (113.018 / 123.948) | 21.519 (20.712 / 22.288) | - |
| `reverse_23d_2` | 23D 2^23 reverse | 15.607 (14.850 / 16.203) | skipped (rebuild with --features hptt) | **14.723 (13.220 / 14.997)** | 69.179 (57.613 / 71.794) | 18.825 (13.747 / 23.538) | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | 3185.630 (3094.265 / 3298.803) | skipped (rebuild with --features hptt) | **2584.148 (2015.653 / 2725.186)** | 3084.444 (2822.951 / 3816.851) | 2866.688 (2375.020 / 3102.174) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | **8.550 (8.023 / 8.859)** | skipped (rebuild with --features hptt) | 11.302 (11.058 / 11.639) | 21.943 (16.464 / 22.505) | 15.409 (8.506 / 15.999) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | **11.729 (11.103 / 11.949)** | - | 13.289 (13.064 / 13.525) | 181.733 (180.198 / 183.438) | 17.118 (16.364 / 19.060) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | **3.864 (3.706 / 5.374)** | skipped (rebuild with --features hptt) | 4.477 (4.263 / 4.528) | 16.620 (15.503 / 18.455) | 6.422 (4.666 / 7.397) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | **10.414 (10.336 / 11.161)** | skipped (rebuild with --features hptt) | 12.590 (12.516 / 12.703) | 20.414 (18.778 / 21.328) | 12.960 (11.090 / 15.698) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | **12.354 (12.256 / 12.509)** | skipped (rebuild with --features hptt) | 14.855 (14.363 / 15.275) | 375.902 (276.965 / 384.745) | 20.455 (17.833 / 21.739) | - |
