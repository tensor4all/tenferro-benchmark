# CPU Permutation Benchmark Results

- Target profile: `mac-cpu`
- Suite: `cpu/permutation`
- Suite file: `benchmarks/cpu/permutation.yaml`
- Timestamp: `2026-11-13T18:54:45.485491+00:00`
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
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | **8.468 (8.291 / 8.644)** | skipped (rebuild with --features hptt) | 12.552 (11.998 / 13.146) | 24.284 (13.159 / 35.319) | 24.162 (13.522 / 25.258) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **9.134 (8.658 / 9.705)** | - | - | 9.762 (9.428 / 10.067) |
| `reverse_15d_3` | 15D 3^15 reverse | 36.548 (36.050 / 37.139) | skipped (rebuild with --features hptt) | **33.813 (33.013 / 34.912)** | 146.981 (134.108 / 162.017) | 46.562 (28.317 / 53.164) | - |
| `reverse_23d_2` | 23D 2^23 reverse | 37.896 (32.588 / 44.767) | skipped (rebuild with --features hptt) | **28.128 (27.142 / 35.605)** | 87.450 (74.934 / 92.166) | 36.512 (24.182 / 43.050) | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | **4492.657 (4133.473 / 5302.056)** | skipped (rebuild with --features hptt) | 4957.547 (4380.152 / 5143.226) | 7429.351 (6270.116 / 8094.576) | 8625.896 (7910.042 / 11037.238) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | **11.263 (10.727 / 14.357)** | skipped (rebuild with --features hptt) | 14.018 (13.293 / 16.182) | 29.099 (18.408 / 39.493) | 21.171 (10.480 / 32.007) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | **15.215 (15.122 / 15.427)** | - | 18.198 (17.990 / 18.648) | 236.737 (209.360 / 251.301) | 35.014 (24.032 / 41.296) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | 10.556 (10.211 / 11.614) | skipped (rebuild with --features hptt) | **10.251 (9.507 / 11.017)** | 32.249 (28.095 / 38.402) | 11.728 (9.914 / 15.115) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | **20.116 (19.776 / 21.526)** | skipped (rebuild with --features hptt) | 22.032 (20.570 / 22.665) | 43.986 (27.419 / 47.144) | 32.413 (25.386 / 43.665) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | **26.819 (25.434 / 33.739)** | skipped (rebuild with --features hptt) | 32.564 (31.765 / 35.007) | 292.990 (257.386 / 320.064) | 84.947 (65.315 / 111.603) | - |

## Threads: 4

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | **10.282 (9.366 / 11.599)** | skipped (rebuild with --features hptt) | 11.362 (10.808 / 12.189) | 27.602 (22.458 / 37.033) | 21.726 (12.123 / 32.242) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | 9.314 (8.864 / 9.878) | - | - | **9.134 (8.780 / 9.307)** |
| `reverse_15d_3` | 15D 3^15 reverse | 29.951 (23.494 / 34.697) | skipped (rebuild with --features hptt) | 30.683 (27.622 / 31.982) | 133.893 (128.246 / 143.578) | **28.108 (24.032 / 30.374)** | - |
| `reverse_23d_2` | 23D 2^23 reverse | 50.941 (43.481 / 56.098) | skipped (rebuild with --features hptt) | **22.493 (21.711 / 27.161)** | 91.804 (71.236 / 110.986) | 25.560 (21.883 / 28.570) | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | 9097.372 (5237.521 / 10190.656) | skipped (rebuild with --features hptt) | 8187.505 (5133.857 / 10246.213) | 7284.272 (5840.975 / 8658.065) | **6340.286 (5195.279 / 7552.762)** | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | **12.073 (11.600 / 12.670)** | skipped (rebuild with --features hptt) | 13.747 (12.979 / 14.798) | 27.851 (19.804 / 34.341) | 18.653 (9.997 / 20.573) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | **17.742 (14.561 / 19.960)** | - | 17.902 (16.189 / 20.372) | 209.209 (199.878 / 215.628) | 21.070 (20.062 / 26.736) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | **5.080 (4.766 / 5.614)** | skipped (rebuild with --features hptt) | 6.047 (5.625 / 6.381) | 21.847 (19.287 / 24.231) | 7.937 (6.013 / 11.045) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | **14.031 (12.821 / 17.524)** | skipped (rebuild with --features hptt) | 15.983 (14.798 / 16.451) | 27.449 (22.851 / 32.653) | 14.728 (12.605 / 16.895) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | **16.186 (15.506 / 20.225)** | skipped (rebuild with --features hptt) | 18.650 (18.381 / 18.828) | 280.892 (271.160 / 293.232) | 30.425 (18.765 / 34.062) | - |
