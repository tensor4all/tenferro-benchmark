# CPU Permutation Benchmark Results

- Target profile: `mac-cpu`
- Suite: `cpu/permutation`
- Suite file: `benchmarks/cpu/permutation.yaml`
- Timestamp: `2026-11-14T18:45:36.382651+00:00`
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
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | **7.521 (7.360 / 8.122)** | skipped (rebuild with --features hptt) | 9.117 (9.062 / 9.569) | 12.734 (11.919 / 13.275) | 12.495 (12.186 / 13.597) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **6.787 (6.691 / 7.051)** | - | - | 7.513 (7.258 / 8.347) |
| `reverse_15d_3` | 15D 3^15 reverse | 35.083 (33.901 / 36.365) | skipped (rebuild with --features hptt) | 33.098 (31.421 / 34.960) | 115.040 (110.621 / 117.628) | **24.144 (23.365 / 24.615)** | - |
| `reverse_23d_2` | 23D 2^23 reverse | 33.326 (31.907 / 38.728) | skipped (rebuild with --features hptt) | 28.659 (28.229 / 30.069) | 80.956 (69.133 / 90.840) | **28.546 (23.699 / 29.409)** | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | 4001.907 (3502.064 / 4807.767) | skipped (rebuild with --features hptt) | **2347.708 (2217.484 / 2489.339)** | 3240.197 (2916.786 / 3741.997) | 3989.006 (3073.482 / 5864.624) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | **9.264 (9.109 / 9.998)** | skipped (rebuild with --features hptt) | 10.101 (9.941 / 10.500) | 21.721 (18.603 / 22.921) | 13.699 (11.466 / 15.567) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | **14.034 (13.898 / 14.791)** | - | 15.981 (15.782 / 16.316) | 207.450 (194.532 / 214.882) | 17.569 (17.188 / 18.190) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | 8.998 (8.649 / 9.246) | skipped (rebuild with --features hptt) | 9.660 (9.517 / 9.942) | 14.832 (14.309 / 15.706) | **7.753 (7.658 / 8.116)** | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | **17.052 (16.943 / 17.352)** | skipped (rebuild with --features hptt) | 18.390 (18.289 / 18.508) | 18.402 (18.274 / 18.470) | 20.061 (19.458 / 20.417) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | **20.622 (20.410 / 21.636)** | skipped (rebuild with --features hptt) | 23.748 (23.568 / 24.281) | 280.864 (258.925 / 301.846) | 38.953 (38.347 / 39.280) | - |

## Threads: 4

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | **7.976 (7.690 / 8.196)** | skipped (rebuild with --features hptt) | 11.055 (10.794 / 12.220) | 12.887 (11.851 / 16.568) | 8.157 (7.696 / 8.386) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **8.269 (7.928 / 8.448)** | - | - | 8.412 (8.290 / 8.684) |
| `reverse_15d_3` | 15D 3^15 reverse | 33.213 (29.631 / 36.048) | skipped (rebuild with --features hptt) | 26.674 (22.847 / 28.909) | 115.315 (110.186 / 118.014) | **21.409 (19.446 / 22.787)** | - |
| `reverse_23d_2` | 23D 2^23 reverse | 39.373 (35.212 / 45.037) | skipped (rebuild with --features hptt) | 23.646 (21.993 / 25.948) | 68.566 (62.332 / 73.258) | **10.910 (9.944 / 11.326)** | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | **1346.417 (970.560 / 1859.137)** | skipped (rebuild with --features hptt) | 3864.468 (3681.277 / 4321.497) | 3590.242 (3061.316 / 4467.440) | 2674.042 (2269.273 / 3012.589) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | 10.066 (9.611 / 10.303) | skipped (rebuild with --features hptt) | 12.349 (11.967 / 13.241) | 16.367 (16.033 / 17.010) | **9.970 (9.902 / 10.077)** | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | 11.949 (11.399 / 12.881) | - | 16.035 (15.316 / 19.189) | 188.830 (186.224 / 199.280) | **11.163 (10.440 / 11.320)** | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | **4.179 (3.890 / 4.564)** | skipped (rebuild with --features hptt) | 4.739 (4.689 / 4.860) | 17.976 (16.526 / 18.968) | 5.425 (5.241 / 6.068) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | **11.070 (10.663 / 12.500)** | skipped (rebuild with --features hptt) | 11.989 (11.579 / 12.740) | 17.913 (17.833 / 18.767) | 14.467 (12.030 / 15.816) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | **13.787 (13.214 / 14.121)** | skipped (rebuild with --features hptt) | 16.291 (16.053 / 16.655) | 251.451 (245.061 / 254.086) | 15.387 (14.686 / 17.356) | - |
