# CPU Permutation Benchmark Results

- Target profile: `mac-cpu`
- Suite: `cpu/permutation`
- Suite file: `benchmarks/cpu/permutation.yaml`
- Timestamp: `2026-11-27T18:53:24.955573+00:00`
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
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | 14.886 (13.275 / 18.507) | skipped (rebuild with --features hptt) | 13.265 (12.606 / 14.991) | 14.179 (12.345 / 17.541) | **12.257 (12.165 / 12.803)** | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **7.328 (7.278 / 7.421)** | - | - | 7.589 (7.437 / 8.149) |
| `reverse_15d_3` | 15D 3^15 reverse | 54.659 (53.456 / 69.755) | skipped (rebuild with --features hptt) | 34.970 (33.904 / 36.505) | 124.297 (115.952 / 127.965) | **27.142 (25.053 / 29.715)** | - |
| `reverse_23d_2` | 23D 2^23 reverse | 29.611 (27.415 / 30.691) | skipped (rebuild with --features hptt) | 28.482 (28.194 / 33.903) | 70.191 (59.774 / 89.472) | **25.726 (23.151 / 26.087)** | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | 8955.930 (8511.289 / 9355.702) | skipped (rebuild with --features hptt) | **5818.889 (4497.673 / 8188.882)** | 8329.562 (7518.243 / 8913.580) | 8504.528 (6451.121 / 9788.600) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | 12.748 (10.473 / 14.537) | skipped (rebuild with --features hptt) | 11.517 (10.815 / 12.678) | 17.239 (16.148 / 17.824) | **10.431 (10.239 / 10.714)** | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | **16.715 (16.235 / 18.217)** | - | 17.302 (16.919 / 18.323) | 194.173 (188.511 / 209.087) | 16.940 (16.638 / 17.297) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | 6.746 (6.653 / 6.855) | skipped (rebuild with --features hptt) | 9.646 (9.579 / 9.707) | 14.210 (13.329 / 16.790) | **6.543 (6.498 / 7.110)** | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | **17.871 (17.729 / 18.178)** | skipped (rebuild with --features hptt) | 20.772 (20.655 / 22.533) | 18.263 (18.021 / 18.550) | 20.316 (17.813 / 21.162) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | **24.075 (23.285 / 25.316)** | skipped (rebuild with --features hptt) | 26.135 (25.874 / 26.916) | 305.136 (294.390 / 310.043) | 36.902 (35.238 / 38.175) | - |

## Threads: 4

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | **9.092 (8.454 / 9.536)** | skipped (rebuild with --features hptt) | 10.077 (9.925 / 10.322) | 17.126 (15.484 / 19.558) | 9.738 (8.417 / 9.891) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **7.526 (7.199 / 8.024)** | - | - | 9.595 (8.787 / 10.786) |
| `reverse_15d_3` | 15D 3^15 reverse | 55.002 (48.180 / 64.486) | skipped (rebuild with --features hptt) | 28.016 (24.582 / 33.840) | 135.803 (130.131 / 145.674) | **13.125 (11.401 / 16.242)** | - |
| `reverse_23d_2` | 23D 2^23 reverse | 16.644 (14.845 / 24.227) | skipped (rebuild with --features hptt) | 18.437 (16.595 / 19.665) | 59.889 (55.887 / 76.658) | **10.403 (9.394 / 11.699)** | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | 7995.872 (6719.229 / 9136.409) | skipped (rebuild with --features hptt) | **5124.095 (4178.204 / 6519.037)** | 8695.933 (7159.350 / 9509.929) | 5649.785 (4384.346 / 6318.001) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | **9.006 (8.778 / 9.487)** | skipped (rebuild with --features hptt) | 11.297 (11.167 / 11.667) | 17.846 (17.457 / 18.092) | 12.879 (10.961 / 13.436) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | **12.072 (11.125 / 13.397)** | - | 14.010 (13.720 / 15.292) | 220.890 (195.996 / 231.824) | 13.470 (11.297 / 13.613) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | 5.229 (4.806 / 5.786) | skipped (rebuild with --features hptt) | 7.319 (6.928 / 7.933) | 12.890 (12.596 / 13.339) | **4.755 (4.494 / 5.390)** | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | 13.148 (12.856 / 13.930) | skipped (rebuild with --features hptt) | 13.315 (12.390 / 15.019) | 19.196 (19.038 / 19.277) | **10.549 (10.320 / 10.650)** | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | **15.025 (14.478 / 15.324)** | skipped (rebuild with --features hptt) | 17.526 (17.173 / 17.935) | 263.271 (244.910 / 283.193) | 39.178 (33.996 / 42.233) | - |
