# CPU Permutation Benchmark Results

- Target profile: `mac-cpu`
- Suite: `cpu/permutation`
- Suite file: `benchmarks/cpu/permutation.yaml`
- Timestamp: `2026-09-16T08:02:59.175361+00:00`
- tenferro-rs commit: `d8759f4320a337d2399f4a87dfec55af51d2ebf1`

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
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | 7.288 (7.147 / 7.336) | skipped (rebuild with --features hptt) | **3.794 (3.772 / 3.856)** | 11.108 (11.019 / 11.384) | 4.620 (4.487 / 11.599) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **2.449 (2.429 / 2.511)** | - | - | 2.476 (2.438 / 2.541) |
| `reverse_15d_3` | 15D 3^15 reverse | 31.502 (31.175 / 31.879) | skipped (rebuild with --features hptt) | 21.789 (21.553 / 22.768) | 57.148 (54.973 / 62.184) | **10.056 (9.457 / 16.771)** | - |
| `reverse_23d_2` | 23D 2^23 reverse | 47.327 (47.145 / 47.902) | skipped (rebuild with --features hptt) | 12.376 (12.101 / 12.786) | 45.780 (44.254 / 46.933) | **8.733 (8.649 / 9.017)** | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | 284.661 (280.986 / 291.009) | skipped (rebuild with --features hptt) | 191.010 (190.889 / 192.133) | 198.461 (171.463 / 224.062) | **189.784 (183.236 / 215.632)** | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | 5.439 (5.417 / 5.479) | skipped (rebuild with --features hptt) | **3.810 (3.788 / 3.849)** | 17.001 (16.763 / 23.663) | 6.064 (5.844 / 13.273) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | 8.056 (7.931 / 8.142) | - | **5.889 (5.860 / 6.007)** | 121.494 (114.751 / 129.834) | 8.308 (7.357 / 14.568) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | 6.361 (6.297 / 6.421) | skipped (rebuild with --features hptt) | **4.881 (4.834 / 4.932)** | 12.537 (11.920 / 17.464) | 5.952 (5.830 / 9.482) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | 12.656 (12.587 / 12.742) | skipped (rebuild with --features hptt) | 11.808 (11.696 / 12.117) | 15.816 (15.521 / 19.498) | **11.210 (11.109 / 14.801)** | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 18.875 (18.781 / 19.119) | skipped (rebuild with --features hptt) | **12.713 (12.577 / 12.807)** | 49.538 (37.639 / 54.518) | 19.885 (19.530 / 23.374) | - |

## Threads: 4

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | 6.087 (2.859 / 18.709) | skipped (rebuild with --features hptt) | 3.733 (3.710 / 3.762) | 11.252 (10.955 / 15.132) | **1.832 (1.686 / 6.026)** | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **2.440 (2.431 / 2.489)** | - | - | 2.475 (2.446 / 2.499) |
| `reverse_15d_3` | 15D 3^15 reverse | 47.293 (30.327 / 49.207) | skipped (rebuild with --features hptt) | 9.678 (9.650 / 9.913) | 61.984 (58.972 / 63.991) | **8.034 (3.675 / 8.871)** | - |
| `reverse_23d_2` | 23D 2^23 reverse | 38.287 (24.489 / 76.153) | skipped (rebuild with --features hptt) | 7.842 (7.734 / 7.977) | 52.009 (50.436 / 53.703) | **3.982 (3.821 / 4.060)** | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | 121.598 (117.967 / 123.283) | skipped (rebuild with --features hptt) | **68.746 (68.507 / 68.905)** | 179.265 (171.778 / 181.349) | 76.208 (68.005 / 76.822) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | 2.607 (2.599 / 2.710) | skipped (rebuild with --features hptt) | **1.725 (1.680 / 1.785)** | 17.199 (16.709 / 21.028) | 2.292 (2.181 / 6.706) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | 4.593 (4.315 / 4.746) | - | **2.801 (2.698 / 2.865)** | 121.221 (118.464 / 123.439) | 8.266 (3.632 / 15.872) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | 2.100 (2.023 / 2.184) | skipped (rebuild with --features hptt) | **1.423 (1.402 / 1.471)** | 13.809 (11.235 / 21.455) | 1.890 (1.766 / 4.403) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | 5.704 (4.249 / 6.558) | skipped (rebuild with --features hptt) | **4.258 (4.217 / 4.313)** | 18.403 (15.979 / 18.726) | 5.988 (3.552 / 6.031) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 36.366 (36.046 / 37.262) | skipped (rebuild with --features hptt) | **8.256 (8.195 / 8.317)** | 105.492 (92.802 / 112.473) | 32.489 (30.575 / 38.579) | - |
