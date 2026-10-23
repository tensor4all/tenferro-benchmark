# CPU Permutation Benchmark Results

- Target profile: `mac-cpu`
- Suite: `cpu/permutation`
- Suite file: `benchmarks/cpu/permutation.yaml`
- Timestamp: `2026-10-23T18:35:22.311466+00:00`
- tenferro-rs commit: `74d138135bc227c5a1ff41db8863f853b647d863`

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
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | 11.440 (9.071 / 13.005) | skipped (rebuild with --features hptt) | **10.446 (10.121 / 11.217)** | 19.843 (11.020 / 26.238) | 22.696 (11.659 / 25.207) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **7.166 (7.128 / 7.282)** | - | - | 7.238 (7.176 / 7.655) |
| `reverse_15d_3` | 15D 3^15 reverse | 39.223 (36.255 / 40.713) | skipped (rebuild with --features hptt) | 39.836 (37.864 / 42.974) | 142.976 (118.920 / 146.428) | **32.722 (24.148 / 36.212)** | - |
| `reverse_23d_2` | 23D 2^23 reverse | 32.400 (29.958 / 34.180) | skipped (rebuild with --features hptt) | **30.853 (27.198 / 32.700)** | 74.995 (65.503 / 98.625) | 34.562 (28.704 / 44.832) | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | 5847.266 (3921.081 / 9741.788) | skipped (rebuild with --features hptt) | 6129.920 (5819.011 / 8023.493) | 7654.658 (6603.722 / 8168.905) | **3896.136 (2983.231 / 6902.748)** | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | **11.409 (10.457 / 12.375)** | skipped (rebuild with --features hptt) | 15.511 (13.216 / 17.553) | 25.716 (15.883 / 35.504) | 21.108 (10.883 / 23.266) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | **15.490 (15.085 / 18.303)** | - | 18.848 (18.015 / 19.778) | 218.896 (193.060 / 228.604) | 27.032 (24.018 / 34.756) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | **8.428 (8.359 / 8.793)** | skipped (rebuild with --features hptt) | 8.899 (8.870 / 8.973) | 29.402 (25.560 / 34.797) | 14.104 (11.581 / 20.654) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | **16.461 (16.355 / 16.985)** | skipped (rebuild with --features hptt) | 18.699 (18.659 / 19.143) | 29.377 (20.463 / 38.307) | 33.457 (25.773 / 39.343) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 26.274 (25.039 / 28.208) | skipped (rebuild with --features hptt) | **24.035 (23.610 / 24.602)** | 258.829 (244.671 / 262.230) | 89.008 (73.860 / 97.484) | - |

## Threads: 4

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | **8.675 (8.476 / 8.830)** | skipped (rebuild with --features hptt) | 11.074 (10.705 / 11.904) | 18.001 (11.221 / 19.249) | 14.256 (8.155 / 16.417) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **7.679 (7.597 / 7.794)** | - | - | 7.704 (7.578 / 7.846) |
| `reverse_15d_3` | 15D 3^15 reverse | 30.333 (26.783 / 36.111) | skipped (rebuild with --features hptt) | 32.442 (28.726 / 37.112) | 123.998 (120.396 / 127.135) | **19.533 (16.741 / 21.435)** | - |
| `reverse_23d_2` | 23D 2^23 reverse | 32.108 (28.225 / 57.702) | skipped (rebuild with --features hptt) | 24.525 (20.320 / 28.968) | 83.243 (68.650 / 88.986) | **21.779 (18.970 / 22.802)** | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | 8364.410 (6526.748 / 11681.967) | skipped (rebuild with --features hptt) | **5057.964 (4313.287 / 6862.336)** | 9130.615 (7971.767 / 9748.605) | 7290.928 (6249.343 / 8284.878) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | **10.227 (9.567 / 11.484)** | skipped (rebuild with --features hptt) | 13.355 (12.129 / 14.316) | 23.407 (16.730 / 25.025) | 15.726 (8.659 / 17.112) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | **14.601 (13.442 / 16.887)** | - | 15.583 (14.864 / 16.791) | 190.320 (187.313 / 194.505) | 18.319 (16.269 / 18.886) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | **5.196 (4.656 / 5.407)** | skipped (rebuild with --features hptt) | 7.713 (7.178 / 8.837) | 29.615 (21.569 / 36.807) | 8.224 (5.120 / 10.588) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | **11.540 (10.588 / 12.347)** | skipped (rebuild with --features hptt) | 15.026 (14.243 / 15.566) | 31.599 (25.626 / 42.351) | 22.613 (18.998 / 25.935) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | **14.629 (13.715 / 15.916)** | skipped (rebuild with --features hptt) | 17.581 (17.064 / 17.983) | 268.996 (254.319 / 274.448) | 31.870 (25.187 / 38.277) | - |
