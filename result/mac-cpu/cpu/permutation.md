# CPU Permutation Benchmark Results

- Target profile: `mac-cpu`
- Suite: `cpu/permutation`
- Suite file: `benchmarks/cpu/permutation.yaml`
- Timestamp: `2026-11-10T18:01:36.193420+00:00`
- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`

## CPU Information

- Model: `Apple M2 Pro (Virtual)`
- Vendor: `Apple`
- Logical CPUs: `5`
- Physical CPUs: `5`
- Sockets: `1`
- Cores per socket: `5`
- Threads per core: `1`
- NUMA nodes: `1`
- Performance levels: `Standard: 5 physical / 5 logical (L1i 128 KiB, L1d 64 KiB, L2 4 MiB, 5 CPUs/L2)`
- Python platform: `macOS-26.7.2-arm64-arm-64bit-Mach-O`
- Container: `none`

Thread counts are controlled via `RAYON_NUM_THREADS` / `OMP_NUM_THREADS` / `JULIA_NUM_THREADS`; no CPU-affinity pinning (`taskset` / `numactl`) was applied. The controlled thread environment is recorded per thread count in the run's `run_t<N>.yaml`.

`tenferro-rs` measures `TypedTensorView::transpose_view` followed by `BackendSession::to_contiguous_read`; one shared session is entered before timing, and input view descriptors and metadata-only `transpose_view` are built outside the timed region. Materialization accepts arbitrary source strides. Every backend allocates a fresh destination inside each timed call, so the table compares allocation-inclusive end-to-end materialization rather than destination-reuse copy kernels. `hptt` only participates in patterns with a contiguous source and destination. Correctness is verified against an internal, untimed odometer reference before any timing; a `FAILED` cell means that backend's output did not match the reference for that pattern.

## Threads: 1

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | **5.538 (5.043 / 5.767)** | skipped (rebuild with --features hptt) | 8.217 (8.182 / 8.635) | 17.257 (10.275 / 17.797) | 19.845 (11.609 / 21.088) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | 3.919 (3.816 / 3.962) | - | - | **3.880 (3.846 / 3.962)** |
| `reverse_15d_3` | 15D 3^15 reverse | 38.112 (34.430 / 39.562) | skipped (rebuild with --features hptt) | 29.866 (29.723 / 30.353) | 78.334 (70.701 / 82.827) | **26.630 (19.729 / 30.261)** | - |
| `reverse_23d_2` | 23D 2^23 reverse | 25.220 (24.258 / 25.358) | skipped (rebuild with --features hptt) | 25.074 (25.008 / 25.546) | 45.844 (38.134 / 48.703) | **24.327 (21.307 / 35.249)** | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | **346.899 (338.228 / 351.197)** | skipped (rebuild with --features hptt) | 362.764 (356.797 / 377.732) | 2243.669 (979.940 / 2335.447) | 1593.368 (1479.258 / 1687.511) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | **7.101 (7.012 / 7.321)** | skipped (rebuild with --features hptt) | 7.739 (7.317 / 7.819) | 22.856 (14.217 / 26.436) | 17.405 (9.466 / 17.950) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | 13.895 (13.508 / 14.406) | - | **13.306 (12.633 / 14.934)** | 184.859 (179.848 / 193.033) | 23.307 (15.501 / 32.637) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | 8.349 (8.218 / 8.489) | skipped (rebuild with --features hptt) | 8.646 (8.452 / 9.129) | 18.418 (16.407 / 22.887) | **8.126 (7.719 / 11.803)** | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | **15.935 (15.787 / 16.120)** | skipped (rebuild with --features hptt) | 16.543 (16.450 / 16.666) | 19.768 (17.614 / 25.663) | 21.896 (19.426 / 24.203) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | **17.833 (17.569 / 17.880)** | skipped (rebuild with --features hptt) | 17.993 (17.901 / 18.080) | 101.355 (99.018 / 104.267) | 29.330 (25.736 / 31.945) | - |

## Threads: 4

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | **2.417 (2.371 / 2.559)** | skipped (rebuild with --features hptt) | 3.959 (3.877 / 4.143) | 14.925 (10.715 / 16.123) | 8.120 (4.065 / 9.057) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **3.937 (3.894 / 4.045)** | - | - | 4.284 (4.101 / 4.397) |
| `reverse_15d_3` | 15D 3^15 reverse | 15.714 (15.600 / 15.854) | skipped (rebuild with --features hptt) | **14.836 (14.760 / 15.058)** | 81.024 (74.974 / 82.053) | 17.771 (15.068 / 19.037) | - |
| `reverse_23d_2` | 23D 2^23 reverse | 13.510 (13.327 / 13.556) | skipped (rebuild with --features hptt) | 14.102 (14.044 / 14.352) | 64.493 (43.701 / 67.242) | **12.320 (8.393 / 15.554)** | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | **116.669 (113.533 / 121.032)** | skipped (rebuild with --features hptt) | 138.552 (134.852 / 141.392) | 1946.611 (1279.428 / 2040.198) | 1118.680 (992.333 / 1220.951) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | **3.369 (3.268 / 3.530)** | skipped (rebuild with --features hptt) | 5.109 (4.793 / 5.508) | 19.656 (15.206 / 23.174) | 9.189 (4.586 / 9.460) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | **4.889 (4.688 / 5.566)** | - | 8.162 (7.917 / 8.368) | 182.739 (178.899 / 185.927) | 11.295 (10.390 / 13.907) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | **2.313 (2.255 / 2.338)** | skipped (rebuild with --features hptt) | 2.849 (2.707 / 3.306) | 18.114 (16.645 / 20.043) | 4.773 (3.056 / 7.453) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | **6.362 (4.996 / 7.749)** | skipped (rebuild with --features hptt) | 8.400 (7.112 / 8.589) | 19.692 (18.112 / 23.148) | 9.328 (7.448 / 11.763) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | **10.961 (10.654 / 11.099)** | skipped (rebuild with --features hptt) | 13.447 (13.138 / 13.903) | 92.953 (89.473 / 95.219) | 13.431 (12.156 / 14.817) | - |
