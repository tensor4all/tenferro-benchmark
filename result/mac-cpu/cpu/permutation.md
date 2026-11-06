# CPU Permutation Benchmark Results

- Target profile: `mac-cpu`
- Suite: `cpu/permutation`
- Suite file: `benchmarks/cpu/permutation.yaml`
- Timestamp: `2026-11-06T18:46:44.019043+00:00`
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
- Python platform: `macOS-26.6.2-arm64-arm-64bit-Mach-O`
- Container: `none`

Thread counts are controlled via `RAYON_NUM_THREADS` / `OMP_NUM_THREADS` / `JULIA_NUM_THREADS`; no CPU-affinity pinning (`taskset` / `numactl`) was applied. The controlled thread environment is recorded per thread count in the run's `run_t<N>.yaml`.

`tenferro-rs` measures `TypedTensorView::transpose_view` followed by `BackendSession::to_contiguous_read`; one shared session is entered before timing, and input view descriptors and metadata-only `transpose_view` are built outside the timed region. Materialization accepts arbitrary source strides. Every backend allocates a fresh destination inside each timed call, so the table compares allocation-inclusive end-to-end materialization rather than destination-reuse copy kernels. `hptt` only participates in patterns with a contiguous source and destination. Correctness is verified against an internal, untimed odometer reference before any timing; a `FAILED` cell means that backend's output did not match the reference for that pattern.

## Threads: 1

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | 15.623 (13.173 / 17.347) | skipped (rebuild with --features hptt) | **11.764 (10.430 / 13.132)** | 30.800 (14.974 / 38.662) | 26.528 (13.638 / 33.568) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **7.430 (7.401 / 7.878)** | - | - | 7.446 (7.392 / 7.518) |
| `reverse_15d_3` | 15D 3^15 reverse | 38.638 (36.291 / 41.371) | skipped (rebuild with --features hptt) | 38.161 (35.059 / 43.800) | 149.204 (138.998 / 151.419) | **36.912 (31.300 / 42.361)** | - |
| `reverse_23d_2` | 23D 2^23 reverse | 43.883 (32.705 / 47.097) | skipped (rebuild with --features hptt) | **28.198 (27.112 / 28.799)** | 84.439 (74.460 / 100.077) | 32.451 (25.390 / 44.322) | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | **2776.520 (2619.661 / 2855.707)** | skipped (rebuild with --features hptt) | 2925.393 (2743.048 / 3624.866) | 7303.460 (5374.666 / 7757.918) | 7768.771 (4376.126 / 8632.173) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | **13.080 (12.237 / 15.187)** | skipped (rebuild with --features hptt) | 17.019 (14.391 / 18.956) | 31.014 (18.558 / 39.040) | 21.507 (12.325 / 26.020) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | **20.341 (17.817 / 22.327)** | - | 25.516 (22.037 / 28.009) | 214.357 (200.222 / 217.167) | 25.759 (24.667 / 35.656) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | **10.347 (9.837 / 10.791)** | skipped (rebuild with --features hptt) | 11.243 (11.123 / 11.610) | 24.797 (21.136 / 29.667) | 11.081 (9.432 / 16.357) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | **19.717 (19.335 / 20.050)** | skipped (rebuild with --features hptt) | 21.822 (21.660 / 22.244) | 40.362 (24.676 / 47.144) | 31.172 (22.983 / 34.761) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | **27.103 (26.548 / 27.634)** | skipped (rebuild with --features hptt) | 30.108 (29.901 / 30.590) | 277.609 (255.799 / 282.003) | 50.027 (41.861 / 57.188) | - |

## Threads: 4

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | **7.331 (6.965 / 7.953)** | skipped (rebuild with --features hptt) | 9.488 (9.068 / 9.906) | 16.628 (10.398 / 17.678) | 14.796 (7.808 / 15.460) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **8.257 (7.535 / 11.172)** | - | - | 10.501 (8.778 / 11.492) |
| `reverse_15d_3` | 15D 3^15 reverse | **18.079 (17.235 / 18.908)** | skipped (rebuild with --features hptt) | 19.260 (18.732 / 19.934) | 128.534 (120.078 / 144.528) | 20.554 (18.281 / 23.488) | - |
| `reverse_23d_2` | 23D 2^23 reverse | 27.111 (17.639 / 35.926) | skipped (rebuild with --features hptt) | **14.459 (14.164 / 14.682)** | 68.694 (52.077 / 71.970) | 17.234 (11.071 / 17.539) | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | 9522.449 (8024.273 / 10440.363) | skipped (rebuild with --features hptt) | 3721.547 (3500.898 / 4037.467) | 3240.284 (2841.657 / 3757.487) | **2753.092 (2408.593 / 3106.111)** | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | **8.419 (8.334 / 8.463)** | skipped (rebuild with --features hptt) | 10.798 (10.706 / 11.113) | 22.502 (15.494 / 24.452) | 14.796 (8.223 / 16.002) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | **10.487 (10.237 / 11.065)** | - | 12.958 (12.606 / 13.392) | 188.239 (183.389 / 188.895) | 17.826 (15.463 / 18.072) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | **5.223 (5.089 / 5.366)** | skipped (rebuild with --features hptt) | 6.038 (5.864 / 6.183) | 18.073 (16.913 / 20.762) | 6.712 (4.895 / 8.981) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | **11.119 (10.391 / 11.430)** | skipped (rebuild with --features hptt) | 16.111 (15.541 / 16.510) | 22.919 (19.779 / 25.987) | 15.791 (12.287 / 19.024) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | **15.550 (14.927 / 18.982)** | skipped (rebuild with --features hptt) | 16.797 (15.734 / 18.721) | 301.309 (259.720 / 310.733) | 24.003 (21.121 / 25.231) | - |
