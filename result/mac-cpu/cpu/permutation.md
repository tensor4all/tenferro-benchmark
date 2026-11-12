# CPU Permutation Benchmark Results

- Target profile: `mac-cpu`
- Suite: `cpu/permutation`
- Suite file: `benchmarks/cpu/permutation.yaml`
- Timestamp: `2026-11-12T18:42:20.994628+00:00`
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
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | **6.936 (6.725 / 7.554)** | skipped (rebuild with --features hptt) | 10.185 (10.001 / 10.622) | 19.886 (10.137 / 25.150) | 18.338 (10.494 / 22.824) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **7.135 (6.915 / 7.467)** | - | - | 7.704 (7.472 / 8.121) |
| `reverse_15d_3` | 15D 3^15 reverse | 34.565 (34.392 / 36.722) | skipped (rebuild with --features hptt) | 32.181 (31.459 / 32.795) | 134.375 (117.023 / 144.568) | **25.783 (19.045 / 32.456)** | - |
| `reverse_23d_2` | 23D 2^23 reverse | 41.599 (31.826 / 54.038) | skipped (rebuild with --features hptt) | 28.853 (28.519 / 30.010) | 65.650 (56.824 / 76.786) | **28.563 (20.173 / 31.951)** | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | **2374.515 (2292.578 / 2816.700)** | skipped (rebuild with --features hptt) | 2383.878 (2301.990 / 2482.090) | 2831.351 (2427.112 / 2985.209) | 3206.569 (2550.804 / 3225.690) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | **7.411 (7.385 / 7.492)** | skipped (rebuild with --features hptt) | 9.449 (9.403 / 9.722) | 23.174 (15.066 / 30.606) | 17.339 (8.494 / 19.511) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | **13.421 (13.350 / 14.509)** | - | 16.544 (16.459 / 16.874) | 210.370 (199.323 / 211.215) | 28.308 (19.204 / 34.405) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | 8.537 (8.432 / 8.781) | skipped (rebuild with --features hptt) | 8.991 (8.855 / 9.377) | 17.286 (13.915 / 21.285) | **8.284 (6.079 / 10.806)** | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | **16.290 (16.067 / 17.100)** | skipped (rebuild with --features hptt) | 18.653 (18.358 / 18.853) | 20.417 (17.440 / 25.369) | 20.357 (17.256 / 21.786) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | **19.904 (19.351 / 21.020)** | skipped (rebuild with --features hptt) | 22.636 (22.065 / 23.731) | 367.697 (264.660 / 391.282) | 44.950 (38.896 / 46.367) | - |

## Threads: 4

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | **7.620 (7.332 / 7.820)** | skipped (rebuild with --features hptt) | 8.702 (8.479 / 9.101) | 15.674 (9.664 / 17.534) | 14.053 (7.866 / 14.916) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | 6.493 (6.457 / 6.663) | - | - | **6.486 (6.452 / 6.593)** |
| `reverse_15d_3` | 15D 3^15 reverse | **16.143 (14.967 / 17.449)** | skipped (rebuild with --features hptt) | 16.734 (15.314 / 17.875) | 118.707 (116.586 / 120.686) | 17.168 (15.827 / 17.723) | - |
| `reverse_23d_2` | 23D 2^23 reverse | 28.598 (18.664 / 37.398) | skipped (rebuild with --features hptt) | **13.959 (13.794 / 14.097)** | 63.140 (54.466 / 66.059) | 15.784 (9.936 / 16.724) | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | 3079.104 (2342.048 / 3392.673) | skipped (rebuild with --features hptt) | **2793.092 (2753.636 / 2845.557)** | 3649.746 (2797.527 / 4051.606) | 3026.223 (2684.611 / 3306.240) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | **7.586 (7.530 / 7.881)** | skipped (rebuild with --features hptt) | 9.470 (9.421 / 11.318) | 24.308 (15.921 / 27.911) | 16.210 (9.013 / 19.215) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | **10.285 (9.736 / 10.545)** | - | 12.291 (11.826 / 13.034) | 185.799 (182.795 / 186.363) | 17.736 (15.020 / 19.619) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | **3.791 (3.752 / 3.860)** | skipped (rebuild with --features hptt) | 4.217 (4.201 / 4.272) | 16.680 (15.166 / 17.000) | 6.327 (4.324 / 6.659) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | **9.559 (9.241 / 9.985)** | skipped (rebuild with --features hptt) | 11.370 (11.317 / 12.003) | 21.879 (18.790 / 24.831) | 16.810 (14.017 / 18.581) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | **12.828 (12.545 / 12.974)** | skipped (rebuild with --features hptt) | 14.782 (14.623 / 15.387) | 330.605 (251.959 / 367.754) | 21.613 (18.090 / 22.809) | - |
