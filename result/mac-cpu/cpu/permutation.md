# CPU Permutation Benchmark Results

- Target profile: `mac-cpu`
- Suite: `cpu/permutation`
- Suite file: `benchmarks/cpu/permutation.yaml`
- Timestamp: `2026-10-24T18:31:35.396589+00:00`
- tenferro-rs commit: `fa39e8d78d9c58e36575084b67448a8112ec1d6c`

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

CPU pinning is unavailable on macOS; thread counts are controlled only via `RAYON_NUM_THREADS` / `OMP_NUM_THREADS` / `JULIA_NUM_THREADS`.

`tenferro-rs` measures `TypedTensorView::transpose_view` followed by `BackendSession::to_contiguous_read`; one shared session is entered before timing, and input view descriptors and metadata-only `transpose_view` are built outside the timed region. Materialization accepts arbitrary source strides. Every backend allocates a fresh destination inside each timed call, so the table compares allocation-inclusive end-to-end materialization rather than destination-reuse copy kernels. `hptt` only participates in patterns with a contiguous source and destination. Correctness is verified against an internal, untimed odometer reference before any timing; a `FAILED` cell means that backend's output did not match the reference for that pattern.

## Threads: 1

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | **7.999 (7.804 / 8.389)** | skipped (rebuild with --features hptt) | 9.547 (9.382 / 9.827) | 22.299 (12.621 / 30.920) | 20.041 (11.703 / 27.512) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **6.454 (6.440 / 6.506)** | - | - | 6.465 (6.440 / 6.944) |
| `reverse_15d_3` | 15D 3^15 reverse | 37.882 (34.870 / 46.568) | skipped (rebuild with --features hptt) | 33.999 (33.342 / 34.403) | 119.009 (112.049 / 124.425) | **31.209 (24.617 / 34.616)** | - |
| `reverse_23d_2` | 23D 2^23 reverse | 34.428 (32.005 / 43.615) | skipped (rebuild with --features hptt) | 29.009 (28.331 / 29.960) | 63.898 (58.811 / 76.250) | **27.795 (19.239 / 30.887)** | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | 2298.921 (2199.262 / 2456.223) | skipped (rebuild with --features hptt) | **2037.826 (1934.540 / 2416.492)** | 8644.277 (6032.116 / 10233.809) | 4017.860 (2580.763 / 5522.936) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | **11.517 (11.073 / 12.822)** | skipped (rebuild with --features hptt) | 15.262 (14.721 / 17.062) | 24.817 (15.879 / 30.143) | 16.761 (8.043 / 18.720) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | **16.344 (16.140 / 16.560)** | - | 18.397 (17.700 / 19.678) | 179.276 (178.201 / 186.250) | 22.787 (15.538 / 29.595) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | **8.185 (7.942 / 8.790)** | skipped (rebuild with --features hptt) | 8.527 (8.321 / 8.601) | 23.335 (19.558 / 27.917) | 11.915 (8.554 / 15.746) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | 17.814 (17.467 / 18.545) | skipped (rebuild with --features hptt) | **17.444 (17.382 / 17.671)** | 25.888 (18.494 / 31.036) | 26.603 (22.154 / 34.504) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | **19.708 (19.347 / 20.582)** | skipped (rebuild with --features hptt) | 21.249 (20.868 / 21.822) | 253.143 (249.604 / 255.822) | 43.102 (40.391 / 48.433) | - |

## Threads: 4

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | **7.947 (7.779 / 8.290)** | skipped (rebuild with --features hptt) | 10.179 (9.881 / 10.699) | 18.884 (11.461 / 23.864) | 16.400 (8.330 / 17.462) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **8.441 (7.984 / 9.630)** | - | - | 8.588 (8.207 / 9.371) |
| `reverse_15d_3` | 15D 3^15 reverse | 25.816 (24.206 / 27.908) | skipped (rebuild with --features hptt) | 26.690 (25.207 / 29.124) | 127.128 (121.029 / 128.172) | **24.257 (21.161 / 27.263)** | - |
| `reverse_23d_2` | 23D 2^23 reverse | 50.256 (38.991 / 113.665) | skipped (rebuild with --features hptt) | **17.691 (16.552 / 19.322)** | 75.294 (66.730 / 81.737) | 23.700 (18.292 / 28.821) | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | **3531.685 (2665.386 / 3727.295)** | skipped (rebuild with --features hptt) | 5221.428 (4220.685 / 6400.593) | 7315.967 (6395.090 / 8437.708) | 4088.610 (3294.181 / 5049.668) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | **9.472 (9.274 / 10.228)** | skipped (rebuild with --features hptt) | 11.998 (11.796 / 12.461) | 26.104 (16.692 / 31.084) | 18.244 (9.780 / 20.476) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | **11.963 (11.385 / 16.416)** | - | 14.315 (13.937 / 15.771) | 200.363 (192.189 / 206.622) | 21.925 (18.953 / 24.926) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | 7.206 (6.767 / 8.368) | skipped (rebuild with --features hptt) | **5.884 (5.343 / 6.468)** | 19.810 (18.120 / 22.692) | 7.510 (5.284 / 10.804) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | **12.206 (10.683 / 12.954)** | skipped (rebuild with --features hptt) | 14.498 (13.352 / 15.822) | 21.583 (18.327 / 25.074) | 17.822 (13.072 / 20.576) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | **14.580 (13.815 / 15.183)** | skipped (rebuild with --features hptt) | 17.577 (16.969 / 18.079) | 304.264 (269.201 / 311.868) | 26.488 (25.412 / 28.762) | - |
