# CPU Permutation Benchmark Results

- Target profile: `mac-cpu`
- Suite: `cpu/permutation`
- Suite file: `benchmarks/cpu/permutation.yaml`
- Timestamp: `2026-10-22T18:45:47.239252+00:00`
- tenferro-rs commit: `2007d101d23b5baf7deba4afb8c7b92da347dbba`

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
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | **12.681 (11.949 / 15.103)** | skipped (rebuild with --features hptt) | 12.690 (11.423 / 13.170) | 20.961 (11.417 / 22.325) | 21.016 (11.756 / 23.782) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **10.245 (9.989 / 10.995)** | - | - | 10.345 (10.216 / 11.729) |
| `reverse_15d_3` | 15D 3^15 reverse | 47.301 (38.923 / 60.150) | skipped (rebuild with --features hptt) | 45.597 (38.501 / 47.330) | 158.673 (134.294 / 192.734) | **39.218 (32.184 / 46.113)** | - |
| `reverse_23d_2` | 23D 2^23 reverse | 38.675 (34.738 / 44.602) | skipped (rebuild with --features hptt) | **32.594 (28.771 / 38.933)** | 89.944 (81.665 / 122.967) | 41.921 (27.418 / 44.557) | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | **5545.410 (4269.975 / 6595.852)** | skipped (rebuild with --features hptt) | 6799.465 (5787.435 / 10460.922) | 8255.100 (7266.145 / 9440.736) | 7971.599 (6516.895 / 8670.578) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | **13.708 (11.907 / 15.371)** | skipped (rebuild with --features hptt) | 16.043 (14.835 / 16.594) | 27.293 (16.198 / 36.154) | 14.743 (10.692 / 23.602) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | **19.328 (15.883 / 21.967)** | - | 22.185 (20.283 / 25.891) | 211.333 (199.107 / 211.916) | 28.167 (25.092 / 33.347) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | 10.942 (10.610 / 11.724) | skipped (rebuild with --features hptt) | **9.092 (8.895 / 9.541)** | 23.680 (20.851 / 28.917) | 11.658 (8.680 / 16.491) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | **19.753 (19.046 / 21.484)** | skipped (rebuild with --features hptt) | 26.578 (23.293 / 31.190) | 25.189 (19.999 / 29.662) | 26.000 (22.104 / 31.392) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 32.413 (27.176 / 37.767) | skipped (rebuild with --features hptt) | **30.620 (29.297 / 34.542)** | 266.794 (257.256 / 271.966) | 62.345 (46.649 / 71.544) | - |

## Threads: 4

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | **9.573 (9.034 / 10.059)** | skipped (rebuild with --features hptt) | 11.256 (11.019 / 11.642) | 18.503 (11.707 / 20.169) | 13.839 (7.820 / 15.024) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **7.220 (6.929 / 7.493)** | - | - | 7.463 (7.256 / 7.855) |
| `reverse_15d_3` | 15D 3^15 reverse | 30.819 (25.360 / 35.405) | skipped (rebuild with --features hptt) | 26.736 (25.236 / 28.678) | 129.416 (120.250 / 141.245) | **20.543 (17.870 / 21.421)** | - |
| `reverse_23d_2` | 23D 2^23 reverse | 30.337 (25.287 / 35.335) | skipped (rebuild with --features hptt) | 31.703 (30.642 / 32.579) | 84.452 (70.408 / 97.248) | **17.946 (11.241 / 18.214)** | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | 8693.450 (5359.259 / 10562.888) | skipped (rebuild with --features hptt) | 6361.831 (4889.030 / 7931.137) | 8365.226 (7499.142 / 9181.292) | **4127.039 (3530.076 / 5850.681)** | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | **9.392 (8.939 / 9.844)** | skipped (rebuild with --features hptt) | 13.581 (13.407 / 13.704) | 22.895 (16.471 / 24.937) | 15.447 (8.681 / 17.789) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | **12.008 (10.816 / 13.141)** | - | 14.250 (14.100 / 15.327) | 190.475 (184.594 / 193.509) | 24.260 (16.136 / 30.142) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | 4.528 (4.357 / 4.888) | skipped (rebuild with --features hptt) | **4.465 (4.365 / 4.549)** | 21.472 (19.521 / 22.986) | 6.897 (4.872 / 8.624) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | **11.480 (10.805 / 13.808)** | skipped (rebuild with --features hptt) | 15.271 (14.285 / 16.429) | 24.836 (19.017 / 26.541) | 15.664 (12.256 / 17.558) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | **13.506 (13.057 / 14.277)** | skipped (rebuild with --features hptt) | 17.805 (16.651 / 18.381) | 256.970 (248.686 / 264.466) | 26.083 (22.694 / 28.162) | - |
