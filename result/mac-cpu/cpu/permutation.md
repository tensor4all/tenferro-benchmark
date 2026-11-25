# CPU Permutation Benchmark Results

- Target profile: `mac-cpu`
- Suite: `cpu/permutation`
- Suite file: `benchmarks/cpu/permutation.yaml`
- Timestamp: `2026-11-25T19:00:16.103754+00:00`
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
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | **11.529 (10.463 / 13.852)** | skipped (rebuild with --features hptt) | 13.076 (11.983 / 16.429) | 27.047 (19.075 / 31.911) | 15.130 (14.321 / 16.396) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | 9.129 (8.863 / 9.573) | - | - | **9.055 (8.582 / 9.302)** |
| `reverse_15d_3` | 15D 3^15 reverse | 75.739 (64.944 / 83.322) | skipped (rebuild with --features hptt) | 52.613 (45.819 / 57.306) | 136.830 (121.946 / 152.297) | **48.653 (40.301 / 58.951)** | - |
| `reverse_23d_2` | 23D 2^23 reverse | 44.433 (41.816 / 46.994) | skipped (rebuild with --features hptt) | 46.302 (41.611 / 49.672) | 68.833 (62.049 / 71.074) | **24.412 (21.830 / 25.517)** | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | 9697.478 (7975.406 / 10741.671) | skipped (rebuild with --features hptt) | **9137.651 (8269.679 / 10415.687)** | 9341.034 (7440.057 / 10194.823) | 11683.704 (7148.039 / 15220.439) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | **12.335 (11.163 / 13.408)** | skipped (rebuild with --features hptt) | 13.782 (13.389 / 17.971) | 24.906 (20.587 / 25.348) | 14.977 (12.371 / 15.853) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | **18.793 (17.533 / 20.558)** | - | 23.150 (20.262 / 25.249) | 324.804 (268.574 / 377.803) | 25.001 (21.539 / 29.107) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | **7.564 (7.212 / 8.595)** | skipped (rebuild with --features hptt) | 11.021 (10.205 / 11.748) | 22.143 (17.095 / 24.788) | 10.037 (8.576 / 11.654) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | **17.345 (16.297 / 20.587)** | skipped (rebuild with --features hptt) | 22.794 (20.518 / 26.101) | 21.507 (19.721 / 23.066) | 28.005 (25.756 / 29.868) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 31.188 (29.735 / 37.203) | skipped (rebuild with --features hptt) | **23.586 (22.281 / 27.185)** | 248.942 (241.399 / 253.291) | 39.928 (38.529 / 42.438) | - |

## Threads: 4

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | **8.235 (7.855 / 8.549)** | skipped (rebuild with --features hptt) | 10.850 (10.747 / 11.126) | 20.522 (12.354 / 24.532) | 11.991 (10.108 / 14.311) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **10.547 (8.715 / 12.825)** | - | - | 11.081 (10.141 / 12.891) |
| `reverse_15d_3` | 15D 3^15 reverse | 46.871 (39.763 / 50.805) | skipped (rebuild with --features hptt) | **26.562 (24.021 / 31.340)** | 208.759 (146.591 / 236.930) | 27.478 (21.207 / 30.623) | - |
| `reverse_23d_2` | 23D 2^23 reverse | 36.456 (29.635 / 39.798) | skipped (rebuild with --features hptt) | 23.872 (20.293 / 25.214) | 110.806 (92.207 / 118.703) | **18.218 (13.893 / 21.640)** | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | 6864.001 (6274.947 / 8472.877) | skipped (rebuild with --features hptt) | **5167.715 (4865.691 / 6459.421)** | 9548.139 (5218.481 / 10655.515) | 7374.558 (6464.003 / 8035.838) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | **9.947 (9.521 / 10.258)** | skipped (rebuild with --features hptt) | 12.607 (12.129 / 12.801) | 23.728 (20.587 / 26.306) | 11.580 (11.066 / 12.981) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | **12.446 (11.994 / 13.775)** | - | 16.268 (15.843 / 17.703) | 309.283 (256.648 / 343.909) | 14.655 (12.856 / 15.479) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | 6.171 (5.431 / 7.086) | skipped (rebuild with --features hptt) | 6.473 (5.909 / 7.280) | 19.149 (16.919 / 20.577) | **5.722 (5.469 / 5.979)** | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | 13.294 (12.890 / 16.796) | skipped (rebuild with --features hptt) | 13.364 (12.771 / 15.627) | 18.603 (18.369 / 18.967) | **11.710 (11.163 / 12.657)** | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | **15.716 (15.135 / 16.630)** | skipped (rebuild with --features hptt) | 17.530 (16.175 / 18.586) | 272.079 (263.049 / 285.878) | 18.990 (15.426 / 21.961) | - |
