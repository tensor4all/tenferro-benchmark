# CPU Permutation Benchmark Results

- Target profile: `mac-cpu`
- Suite: `cpu/permutation`
- Suite file: `benchmarks/cpu/permutation.yaml`
- Timestamp: `2026-09-16T01:50:15.633027+00:00`
- tenferro-rs commit: `bca2d54a5884586b1aeae8006caa9b8c0ba44a26`

## CPU Information

- Model: `Apple M5 Max`
- Vendor: `Apple`
- Logical CPUs: `18`
- Physical CPUs: `18`
- Sockets: `1`
- Cores per socket: `18`
- Threads per core: `1`
- NUMA nodes: `1`
- Performance levels: `Super: 6 physical / 6 logical (L1i 192 KiB, L1d 128 KiB, L2 16 MiB, 6 CPUs/L2); Performance: 12 physical / 12 logical (L1i 128 KiB, L1d 64 KiB, L2 8 MiB, 6 CPUs/L2)`
- Python platform: `macOS-26.5.1-arm64-arm-64bit-Mach-O`

CPU pinning is unavailable on macOS; thread counts are controlled only via `RAYON_NUM_THREADS` / `OMP_NUM_THREADS` / `JULIA_NUM_THREADS`.

`tenferro-rs` measures `TypedTensorView::transpose_view` followed by `BackendSession::to_contiguous_read`; one shared session is entered before timing, and input view descriptors and metadata-only `transpose_view` are built outside the timed region. Materialization accepts arbitrary source strides. Every backend allocates a fresh destination inside each timed call, so the table compares allocation-inclusive end-to-end materialization rather than destination-reuse copy kernels. `hptt` only participates in patterns with a contiguous source and destination. Correctness is verified against an internal, untimed odometer reference before any timing; a `FAILED` cell means that backend's output did not match the reference for that pattern.

## Threads: 1

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | 7.020 (6.980 / 7.051) | skipped (rebuild with --features hptt) | **3.743 (3.728 / 3.770)** | 10.890 (10.831 / 11.152) | 4.508 (4.425 / 11.303) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | 2.458 (2.441 / 2.511) | - | - | **2.454 (2.416 / 2.475)** |
| `reverse_15d_3` | 15D 3^15 reverse | 31.385 (31.272 / 31.599) | skipped (rebuild with --features hptt) | 21.296 (21.242 / 21.350) | 59.429 (55.742 / 63.426) | **9.585 (9.411 / 16.096)** | - |
| `reverse_23d_2` | 23D 2^23 reverse | 37.968 (36.079 / 39.072) | skipped (rebuild with --features hptt) | 11.961 (11.729 / 12.075) | 44.831 (43.807 / 49.913) | **9.178 (8.764 / 15.314)** | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | 292.984 (292.627 / 294.183) | skipped (rebuild with --features hptt) | 190.424 (190.252 / 190.861) | 191.394 (183.314 / 216.264) | **189.762 (185.878 / 217.916)** | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | 5.393 (5.364 / 5.410) | skipped (rebuild with --features hptt) | **3.712 (3.684 / 3.737)** | 16.662 (16.501 / 23.705) | 6.003 (5.677 / 13.602) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | 7.948 (7.924 / 7.971) | - | **7.340 (7.312 / 7.403)** | 115.090 (112.093 / 127.215) | 9.296 (7.901 / 14.964) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | 6.219 (6.176 / 6.354) | skipped (rebuild with --features hptt) | **4.820 (4.783 / 4.858)** | 11.299 (10.577 / 16.255) | 5.804 (5.747 / 9.381) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | 12.369 (12.282 / 12.458) | skipped (rebuild with --features hptt) | 11.517 (11.412 / 11.855) | 18.569 (15.033 / 18.846) | **11.188 (11.014 / 14.655)** | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 177.049 (176.556 / 178.020) | skipped (rebuild with --features hptt) | **16.210 (16.030 / 16.434)** | 191.970 (134.540 / 198.399) | 127.159 (123.650 / 134.917) | - |

## Threads: 4

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | **2.899 (2.890 / 2.926)** | skipped (rebuild with --features hptt) | 3.687 (3.677 / 3.710) | 10.833 (10.801 / 15.181) | 5.578 (1.498 / 5.789) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **2.470 (2.438 / 2.520)** | - | - | 2.480 (2.475 / 2.491) |
| `reverse_15d_3` | 15D 3^15 reverse | 22.414 (22.045 / 26.447) | skipped (rebuild with --features hptt) | 9.864 (9.721 / 10.039) | 69.760 (58.746 / 75.085) | **7.740 (4.151 / 10.042)** | - |
| `reverse_23d_2` | 23D 2^23 reverse | 21.788 (21.079 / 22.139) | skipped (rebuild with --features hptt) | 7.401 (7.197 / 8.355) | 55.489 (49.651 / 56.645) | **4.032 (3.845 / 4.115)** | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | 121.016 (119.635 / 121.489) | skipped (rebuild with --features hptt) | **70.976 (69.490 / 72.803)** | 205.193 (190.246 / 210.899) | 82.727 (72.867 / 83.417) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | 2.774 (2.739 / 2.801) | skipped (rebuild with --features hptt) | **1.807 (1.774 / 1.965)** | 17.712 (16.957 / 21.327) | 3.032 (1.768 / 8.243) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | 4.250 (4.052 / 4.454) | - | **2.968 (2.892 / 3.051)** | 117.318 (114.125 / 118.598) | 7.143 (3.089 / 14.079) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | 1.961 (1.936 / 2.046) | skipped (rebuild with --features hptt) | **1.439 (1.397 / 1.473)** | 15.290 (13.092 / 21.713) | 2.190 (1.977 / 4.755) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | **4.122 (4.005 / 4.146)** | skipped (rebuild with --features hptt) | 4.172 (4.109 / 4.303) | 17.949 (15.484 / 19.358) | 6.468 (3.782 / 6.870) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 45.067 (44.559 / 45.523) | skipped (rebuild with --features hptt) | **8.275 (8.205 / 8.327)** | 109.733 (91.225 / 114.929) | 29.605 (27.710 / 36.132) | - |
