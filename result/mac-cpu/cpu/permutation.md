# CPU Permutation Benchmark Results

- Target profile: `mac-cpu`
- Suite: `cpu/permutation`
- Suite file: `benchmarks/cpu/permutation.yaml`
- Timestamp: `2026-11-05T18:02:15.620588+00:00`
- tenferro-rs commit: `a000e5c4929d5bebb91e783c329b27106baed12e`

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
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | 11.637 (11.225 / 11.962) | skipped (rebuild with --features hptt) | **7.855 (7.806 / 8.312)** | 15.150 (8.693 / 15.361) | 17.840 (10.514 / 18.405) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **3.672 (3.613 / 3.713)** | - | - | 3.684 (3.649 / 3.722) |
| `reverse_15d_3` | 15D 3^15 reverse | 41.953 (40.373 / 42.642) | skipped (rebuild with --features hptt) | 35.050 (33.101 / 38.458) | 70.382 (64.125 / 77.696) | **23.184 (15.893 / 25.174)** | - |
| `reverse_23d_2` | 23D 2^23 reverse | 30.851 (29.850 / 31.585) | skipped (rebuild with --features hptt) | 26.606 (25.887 / 29.191) | 46.209 (38.570 / 52.865) | **18.582 (18.273 / 27.419)** | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | **376.810 (366.395 / 380.376)** | skipped (rebuild with --features hptt) | 399.803 (376.516 / 415.332) | 1952.671 (1372.446 / 2047.842) | 1488.457 (1342.212 / 1579.407) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | 7.913 (7.641 / 8.220) | skipped (rebuild with --features hptt) | **7.770 (7.508 / 8.173)** | 19.304 (12.756 / 19.503) | 15.095 (7.548 / 15.320) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | **14.243 (12.378 / 15.271)** | - | 14.822 (14.587 / 15.171) | 171.238 (162.727 / 179.554) | 21.134 (13.188 / 28.705) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | **8.462 (8.214 / 8.781)** | skipped (rebuild with --features hptt) | 8.586 (8.515 / 8.699) | 21.634 (17.840 / 25.533) | 9.116 (8.192 / 12.215) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | **15.288 (15.007 / 15.503)** | skipped (rebuild with --features hptt) | 17.028 (16.892 / 17.962) | 21.377 (18.496 / 29.369) | 23.363 (20.071 / 23.482) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | **17.555 (17.445 / 17.969)** | skipped (rebuild with --features hptt) | 19.050 (18.941 / 19.146) | 100.315 (97.113 / 103.697) | 36.264 (32.888 / 44.076) | - |

## Threads: 4

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | **2.399 (2.249 / 2.473)** | skipped (rebuild with --features hptt) | 3.720 (3.655 / 3.782) | 14.667 (10.284 / 16.231) | 8.213 (4.079 / 8.649) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **3.745 (3.708 / 3.837)** | - | - | 4.090 (3.976 / 4.248) |
| `reverse_15d_3` | 15D 3^15 reverse | 15.243 (14.919 / 15.452) | skipped (rebuild with --features hptt) | **14.648 (14.616 / 14.788)** | 80.473 (73.126 / 84.570) | 18.828 (17.197 / 21.878) | - |
| `reverse_23d_2` | 23D 2^23 reverse | **12.749 (12.542 / 13.214)** | skipped (rebuild with --features hptt) | 13.414 (13.264 / 13.693) | 49.264 (46.700 / 54.673) | 13.472 (9.834 / 15.491) | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | **109.501 (107.187 / 112.791)** | skipped (rebuild with --features hptt) | 132.786 (129.320 / 134.189) | 1702.970 (616.706 / 1923.215) | 1057.741 (878.599 / 1217.667) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | **3.703 (3.550 / 3.846)** | skipped (rebuild with --features hptt) | 4.619 (4.107 / 4.797) | 18.741 (14.369 / 21.320) | 8.440 (4.182 / 8.774) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | **4.544 (4.418 / 4.743)** | - | 5.830 (5.710 / 6.202) | 170.783 (167.198 / 174.681) | 10.752 (9.915 / 13.967) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | **2.175 (2.167 / 2.204)** | skipped (rebuild with --features hptt) | 2.827 (2.748 / 3.007) | 16.613 (15.063 / 18.117) | 4.099 (3.137 / 6.164) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | **4.957 (4.859 / 5.060)** | skipped (rebuild with --features hptt) | 6.215 (6.189 / 6.339) | 16.576 (15.316 / 19.919) | 7.872 (6.323 / 8.081) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | **9.292 (9.183 / 9.587)** | skipped (rebuild with --features hptt) | 10.800 (10.555 / 11.127) | 85.928 (81.207 / 90.916) | 13.816 (11.699 / 14.994) | - |
