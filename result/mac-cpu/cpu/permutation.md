# CPU Permutation Benchmark Results

- Target profile: `mac-cpu`
- Suite: `cpu/permutation`
- Suite file: `benchmarks/cpu/permutation.yaml`
- Timestamp: `2026-10-22T03:30:47.505981+00:00`
- tenferro-rs commit: `d5ef90538a4c05810fe63bc1b5713d871ac98279`

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
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | **6.692 (6.544 / 6.978)** | skipped (rebuild with --features hptt) | 8.360 (7.923 / 8.609) | 21.082 (10.894 / 27.597) | 17.521 (11.050 / 20.349) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **7.233 (7.073 / 8.073)** | - | - | 9.886 (8.694 / 12.318) |
| `reverse_15d_3` | 15D 3^15 reverse | **30.373 (30.012 / 30.769)** | skipped (rebuild with --features hptt) | 30.918 (30.521 / 31.806) | 118.709 (112.071 / 128.548) | 31.666 (21.155 / 36.765) | - |
| `reverse_23d_2` | 23D 2^23 reverse | 33.936 (32.396 / 37.792) | skipped (rebuild with --features hptt) | **28.561 (24.879 / 29.244)** | 88.877 (68.773 / 93.332) | 33.441 (22.967 / 38.732) | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | 4603.153 (3842.466 / 5665.012) | skipped (rebuild with --features hptt) | 4243.297 (3839.161 / 4382.803) | 4177.979 (2973.979 / 5435.758) | **3585.105 (2953.764 / 4402.870)** | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | **9.692 (9.257 / 10.030)** | skipped (rebuild with --features hptt) | 11.372 (10.637 / 12.760) | 24.298 (14.926 / 31.334) | 17.480 (9.595 / 21.647) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | **13.785 (13.236 / 14.112)** | - | 15.289 (15.165 / 15.490) | 196.854 (181.851 / 204.965) | 24.386 (16.467 / 29.853) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | 9.873 (9.642 / 10.529) | skipped (rebuild with --features hptt) | **9.024 (8.937 / 9.719)** | 16.027 (13.209 / 19.988) | 9.395 (8.476 / 13.125) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | **15.992 (15.627 / 16.943)** | skipped (rebuild with --features hptt) | 19.504 (18.459 / 23.442) | 22.837 (18.520 / 30.563) | 22.302 (18.592 / 25.178) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 24.059 (21.963 / 26.995) | skipped (rebuild with --features hptt) | **21.279 (20.849 / 22.335)** | 282.821 (268.388 / 288.216) | 43.984 (38.218 / 48.315) | - |

## Threads: 4

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | **8.015 (7.859 / 8.135)** | skipped (rebuild with --features hptt) | 11.009 (10.512 / 11.730) | 15.487 (9.765 / 16.495) | 12.504 (6.940 / 13.132) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **7.184 (6.767 / 7.641)** | - | - | 8.495 (7.042 / 10.174) |
| `reverse_15d_3` | 15D 3^15 reverse | 22.182 (18.045 / 24.787) | skipped (rebuild with --features hptt) | 25.086 (20.210 / 28.364) | 114.591 (111.252 / 116.643) | **16.342 (15.670 / 16.994)** | - |
| `reverse_23d_2` | 23D 2^23 reverse | 29.132 (22.933 / 32.546) | skipped (rebuild with --features hptt) | **15.055 (14.613 / 20.155)** | 62.607 (55.852 / 69.182) | 15.175 (9.567 / 16.187) | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | **3197.758 (2870.827 / 4409.416)** | skipped (rebuild with --features hptt) | 5163.607 (4336.944 / 5793.459) | 5121.001 (3782.932 / 5834.139) | 3348.870 (2670.954 / 3888.493) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | **10.673 (10.295 / 12.058)** | skipped (rebuild with --features hptt) | 12.681 (11.930 / 13.111) | 21.669 (14.356 / 22.192) | 14.340 (8.030 / 15.441) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | **10.743 (10.414 / 11.421)** | - | 13.581 (12.692 / 15.015) | 179.766 (175.724 / 195.451) | 16.778 (14.380 / 16.930) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | 4.432 (4.033 / 6.211) | skipped (rebuild with --features hptt) | **4.357 (4.273 / 4.511)** | 22.124 (19.440 / 24.570) | 6.893 (5.056 / 8.551) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | **12.238 (11.343 / 13.910)** | skipped (rebuild with --features hptt) | 13.565 (12.724 / 14.808) | 21.158 (19.058 / 22.280) | 14.965 (11.406 / 16.303) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | **13.870 (13.400 / 15.526)** | skipped (rebuild with --features hptt) | 15.477 (15.319 / 15.974) | 267.661 (252.729 / 272.232) | 19.477 (16.951 / 20.609) | - |
