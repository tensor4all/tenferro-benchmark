# CPU Permutation Benchmark Results

- Target profile: `mac-cpu`
- Suite: `cpu/permutation`
- Suite file: `benchmarks/cpu/permutation.yaml`
- Timestamp: `2026-11-16T18:05:48.049789+00:00`
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
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | **5.008 (4.876 / 5.053)** | skipped (rebuild with --features hptt) | 8.507 (8.225 / 8.780) | 12.370 (10.960 / 17.595) | 11.577 (10.890 / 11.725) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **5.229 (5.025 / 5.681)** | - | - | 5.304 (5.164 / 5.752) |
| `reverse_15d_3` | 15D 3^15 reverse | 36.536 (34.419 / 39.325) | skipped (rebuild with --features hptt) | 35.094 (31.650 / 36.574) | 78.534 (70.971 / 81.781) | **18.309 (16.760 / 19.914)** | - |
| `reverse_23d_2` | 23D 2^23 reverse | 27.597 (26.989 / 28.608) | skipped (rebuild with --features hptt) | 28.941 (26.426 / 30.262) | 46.376 (40.640 / 47.417) | **23.810 (20.174 / 24.619)** | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | **354.406 (332.732 / 390.765)** | skipped (rebuild with --features hptt) | 354.697 (348.099 / 358.640) | 1896.595 (1637.014 / 1973.471) | 1544.796 (1087.083 / 1594.208) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | **7.283 (6.884 / 7.486)** | skipped (rebuild with --features hptt) | 7.995 (7.803 / 8.363) | 14.766 (14.002 / 14.912) | 8.886 (8.487 / 9.347) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | **13.002 (12.212 / 13.334)** | - | 14.007 (13.720 / 14.203) | 181.541 (175.585 / 191.427) | 14.501 (14.106 / 15.014) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | 9.881 (9.508 / 10.243) | skipped (rebuild with --features hptt) | 9.472 (9.335 / 9.610) | 18.448 (17.666 / 18.941) | **7.230 (6.852 / 7.599)** | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | **16.839 (16.575 / 17.044)** | skipped (rebuild with --features hptt) | 18.840 (18.620 / 18.947) | 18.448 (18.001 / 19.292) | 19.787 (19.417 / 19.970) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | **20.034 (19.744 / 20.308)** | skipped (rebuild with --features hptt) | 20.205 (19.988 / 20.743) | 99.119 (97.386 / 99.940) | 34.638 (32.191 / 35.697) | - |

## Threads: 4

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | **2.618 (2.560 / 2.673)** | skipped (rebuild with --features hptt) | 4.384 (4.187 / 4.553) | 9.080 (8.913 / 9.402) | 3.519 (3.382 / 3.988) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | 3.768 (3.745 / 4.053) | - | - | **3.688 (3.673 / 4.282)** |
| `reverse_15d_3` | 15D 3^15 reverse | 15.195 (15.019 / 16.629) | skipped (rebuild with --features hptt) | 14.504 (14.463 / 14.774) | 66.506 (65.923 / 67.940) | **12.385 (10.659 / 12.490)** | - |
| `reverse_23d_2` | 23D 2^23 reverse | 12.390 (12.312 / 12.899) | skipped (rebuild with --features hptt) | 13.993 (13.317 / 14.330) | 40.937 (39.555 / 41.816) | **6.926 (6.722 / 7.122)** | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | **111.855 (110.366 / 117.599)** | skipped (rebuild with --features hptt) | 148.722 (143.601 / 149.592) | 1387.420 (1030.978 / 1696.990) | 361.297 (164.158 / 500.356) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | 4.251 (3.607 / 4.383) | skipped (rebuild with --features hptt) | 4.942 (4.489 / 5.212) | 15.875 (15.046 / 16.188) | **3.723 (3.610 / 3.746)** | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | **4.793 (4.420 / 4.962)** | - | 7.491 (6.576 / 7.870) | 172.110 (164.995 / 181.254) | 6.991 (6.672 / 7.383) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | **2.270 (2.227 / 2.443)** | skipped (rebuild with --features hptt) | 2.503 (2.483 / 2.557) | 16.319 (15.524 / 16.811) | 3.445 (3.182 / 3.779) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | **5.000 (4.818 / 5.017)** | skipped (rebuild with --features hptt) | 6.351 (6.247 / 6.401) | 19.130 (18.482 / 19.501) | 10.293 (7.374 / 12.200) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | **9.237 (8.844 / 9.798)** | skipped (rebuild with --features hptt) | 11.120 (10.886 / 11.588) | 97.062 (93.950 / 99.866) | 13.141 (12.365 / 13.680) | - |
