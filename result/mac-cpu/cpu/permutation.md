# CPU Permutation Benchmark Results

- Target profile: `mac-cpu`
- Suite: `cpu/permutation`
- Suite file: `benchmarks/cpu/permutation.yaml`
- Timestamp: `2026-11-11T18:45:30.583073+00:00`
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
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | **10.528 (10.084 / 11.047)** | skipped (rebuild with --features hptt) | 12.554 (11.961 / 13.103) | 27.998 (14.452 / 42.339) | 24.791 (14.253 / 29.282) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **8.906 (8.496 / 9.411)** | - | - | 12.204 (10.555 / 13.395) |
| `reverse_15d_3` | 15D 3^15 reverse | 69.290 (61.896 / 72.872) | skipped (rebuild with --features hptt) | 46.340 (41.625 / 48.051) | 147.524 (131.678 / 157.814) | **36.103 (29.419 / 41.529)** | - |
| `reverse_23d_2` | 23D 2^23 reverse | 53.409 (38.846 / 55.621) | skipped (rebuild with --features hptt) | **27.413 (27.160 / 27.949)** | 90.870 (75.429 / 102.626) | 36.648 (23.064 / 41.224) | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | **7995.041 (4789.973 / 8817.356)** | skipped (rebuild with --features hptt) | 10017.804 (5618.134 / 10645.304) | 9287.427 (8218.565 / 10342.764) | 8995.938 (7818.271 / 9568.448) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | **9.858 (9.471 / 10.207)** | skipped (rebuild with --features hptt) | 13.296 (12.878 / 13.865) | 26.695 (16.621 / 33.004) | 18.422 (9.471 / 23.180) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | **16.120 (15.724 / 16.928)** | - | 18.013 (16.339 / 19.294) | 236.429 (191.485 / 246.543) | 44.875 (31.910 / 52.319) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | **9.078 (8.685 / 9.465)** | skipped (rebuild with --features hptt) | 9.859 (9.484 / 11.397) | 23.542 (20.383 / 28.839) | 12.603 (9.180 / 15.999) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | **17.684 (17.148 / 19.112)** | skipped (rebuild with --features hptt) | 20.166 (19.795 / 20.727) | 24.661 (20.711 / 32.279) | 24.329 (20.628 / 29.812) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | **24.013 (23.403 / 27.429)** | skipped (rebuild with --features hptt) | 26.833 (25.181 / 31.077) | 267.492 (251.976 / 282.510) | 52.023 (47.719 / 54.356) | - |

## Threads: 4

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | **7.986 (7.718 / 8.055)** | skipped (rebuild with --features hptt) | 10.620 (10.063 / 11.021) | 24.632 (13.105 / 30.093) | 17.743 (9.266 / 20.612) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **8.016 (7.876 / 8.947)** | - | - | 10.891 (8.957 / 13.027) |
| `reverse_15d_3` | 15D 3^15 reverse | **23.356 (21.988 / 25.185)** | skipped (rebuild with --features hptt) | 23.849 (21.830 / 26.610) | 139.199 (132.143 / 158.139) | 26.465 (22.187 / 27.482) | - |
| `reverse_23d_2` | 23D 2^23 reverse | 24.258 (19.190 / 29.453) | skipped (rebuild with --features hptt) | **15.972 (15.764 / 16.239)** | 80.029 (72.761 / 85.216) | 18.908 (14.965 / 21.818) | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | 8674.320 (7224.862 / 10921.057) | skipped (rebuild with --features hptt) | 6058.046 (4302.996 / 7937.811) | 6326.840 (3899.272 / 7533.750) | **3800.137 (3122.362 / 4431.405)** | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | **10.126 (9.798 / 10.645)** | skipped (rebuild with --features hptt) | 12.923 (12.450 / 13.295) | 25.863 (17.188 / 29.675) | 18.316 (10.199 / 18.665) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | **12.308 (11.747 / 13.033)** | - | 13.686 (13.182 / 17.824) | 227.555 (217.975 / 232.964) | 19.877 (16.974 / 21.751) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | **4.000 (3.871 / 4.693)** | skipped (rebuild with --features hptt) | 5.718 (4.431 / 7.054) | 24.450 (20.189 / 27.243) | 8.135 (5.338 / 11.007) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | **10.294 (10.126 / 12.348)** | skipped (rebuild with --features hptt) | 12.447 (12.039 / 17.441) | 22.879 (18.790 / 28.726) | 16.534 (15.208 / 19.355) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | **17.258 (14.543 / 20.203)** | skipped (rebuild with --features hptt) | 19.222 (18.102 / 20.254) | 263.156 (258.144 / 273.432) | 26.681 (21.046 / 33.480) | - |
