# CPU Permutation Benchmark Results

- Target profile: `mac-cpu`
- Suite: `cpu/permutation`
- Suite file: `benchmarks/cpu/permutation.yaml`
- Timestamp: `2026-11-26T18:57:07.841561+00:00`
- tenferro-rs commit: `1ba9f33521691170c763dfce9b6c9bb56ca523c2`

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
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | 11.373 (10.672 / 12.971) | skipped (rebuild with --features hptt) | **10.869 (10.672 / 11.452)** | 16.043 (12.539 / 20.439) | 14.479 (12.015 / 15.423) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **6.803 (6.708 / 7.157)** | - | - | 9.271 (8.068 / 10.076) |
| `reverse_15d_3` | 15D 3^15 reverse | 36.184 (35.065 / 38.062) | skipped (rebuild with --features hptt) | 35.858 (34.879 / 41.469) | 146.038 (121.239 / 149.645) | **24.783 (20.720 / 28.844)** | - |
| `reverse_23d_2` | 23D 2^23 reverse | 58.526 (36.528 / 65.222) | skipped (rebuild with --features hptt) | **28.241 (27.622 / 30.683)** | 128.958 (73.656 / 144.513) | 34.554 (24.316 / 39.038) | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | **2575.422 (2374.462 / 3462.755)** | skipped (rebuild with --features hptt) | 2836.765 (2535.797 / 10510.942) | 7058.850 (4390.410 / 9673.403) | 8078.323 (6029.667 / 8235.933) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | 12.352 (10.288 / 14.457) | skipped (rebuild with --features hptt) | 13.518 (13.146 / 14.325) | 17.875 (16.009 / 20.982) | **11.588 (10.202 / 14.346)** | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | 20.551 (17.483 / 24.091) | - | 19.893 (19.490 / 20.490) | 210.601 (191.630 / 226.485) | **18.515 (16.783 / 20.931)** | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | **6.715 (6.569 / 7.071)** | skipped (rebuild with --features hptt) | 9.123 (9.064 / 9.270) | 22.681 (18.794 / 25.590) | 9.710 (8.768 / 11.484) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | **16.070 (16.017 / 16.385)** | skipped (rebuild with --features hptt) | 18.198 (18.167 / 18.336) | 18.878 (18.660 / 19.004) | 20.387 (19.672 / 20.764) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | **21.193 (19.864 / 21.533)** | skipped (rebuild with --features hptt) | 22.202 (21.973 / 22.600) | 269.587 (254.535 / 278.896) | 41.699 (40.594 / 42.341) | - |

## Threads: 4

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | 13.422 (11.139 / 23.890) | skipped (rebuild with --features hptt) | 10.437 (9.816 / 12.630) | 15.434 (13.515 / 17.456) | **8.791 (8.456 / 9.334)** | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **9.259 (7.826 / 12.484)** | - | - | 9.997 (9.049 / 11.753) |
| `reverse_15d_3` | 15D 3^15 reverse | 48.998 (37.719 / 57.148) | skipped (rebuild with --features hptt) | 25.696 (23.509 / 28.641) | 115.781 (110.395 / 124.339) | **16.518 (13.507 / 17.142)** | - |
| `reverse_23d_2` | 23D 2^23 reverse | 22.685 (19.211 / 28.912) | skipped (rebuild with --features hptt) | 21.409 (19.818 / 24.630) | 62.253 (56.533 / 67.392) | **14.040 (12.545 / 15.358)** | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | **4470.061 (3903.515 / 4687.309)** | skipped (rebuild with --features hptt) | 4924.661 (4250.383 / 5788.657) | 8972.979 (6230.342 / 9947.239) | 6799.872 (5537.689 / 8562.663) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | **9.812 (9.117 / 10.317)** | skipped (rebuild with --features hptt) | 12.025 (11.522 / 12.955) | 19.909 (17.923 / 21.609) | 12.043 (10.626 / 13.076) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | **11.937 (11.273 / 13.344)** | - | 14.247 (13.485 / 15.291) | 213.740 (192.827 / 219.127) | 12.183 (11.191 / 12.624) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | **5.270 (4.708 / 6.124)** | skipped (rebuild with --features hptt) | 5.771 (5.308 / 6.377) | 16.763 (15.953 / 18.177) | 6.040 (5.458 / 6.505) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | 17.380 (15.830 / 21.496) | skipped (rebuild with --features hptt) | 15.951 (15.219 / 17.571) | 18.515 (18.283 / 18.963) | **13.798 (12.505 / 14.308)** | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | **17.074 (15.910 / 17.748)** | skipped (rebuild with --features hptt) | 22.051 (18.760 / 26.185) | 271.832 (260.281 / 278.337) | 27.589 (20.523 / 27.995) | - |
