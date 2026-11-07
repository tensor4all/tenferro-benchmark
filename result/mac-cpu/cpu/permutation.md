# CPU Permutation Benchmark Results

- Target profile: `mac-cpu`
- Suite: `cpu/permutation`
- Suite file: `benchmarks/cpu/permutation.yaml`
- Timestamp: `2026-11-07T18:44:09.192651+00:00`
- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`

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
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | **9.907 (7.235 / 10.675)** | skipped (rebuild with --features hptt) | 11.134 (10.715 / 11.849) | 24.694 (13.466 / 32.852) | 21.306 (12.962 / 22.820) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **7.580 (7.044 / 7.914)** | - | - | 8.484 (8.169 / 8.728) |
| `reverse_15d_3` | 15D 3^15 reverse | 54.792 (50.777 / 60.307) | skipped (rebuild with --features hptt) | **40.378 (37.420 / 45.951)** | 128.658 (120.829 / 144.978) | 41.299 (36.009 / 51.589) | - |
| `reverse_23d_2` | 23D 2^23 reverse | 36.414 (32.595 / 40.031) | skipped (rebuild with --features hptt) | **26.372 (26.094 / 30.014)** | 73.259 (64.000 / 79.678) | 35.211 (24.611 / 37.543) | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | **2491.731 (2390.010 / 2657.447)** | skipped (rebuild with --features hptt) | 7003.501 (6336.451 / 10140.562) | 6462.015 (5385.600 / 7407.846) | 10080.430 (4370.103 / 11919.085) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | **12.938 (11.754 / 16.535)** | skipped (rebuild with --features hptt) | 14.325 (12.784 / 15.033) | 26.745 (15.928 / 34.567) | 17.989 (9.868 / 22.059) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | **15.890 (15.602 / 16.181)** | - | 16.894 (16.165 / 17.226) | 210.483 (194.017 / 231.792) | 30.766 (19.044 / 42.783) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | 9.478 (8.988 / 10.125) | skipped (rebuild with --features hptt) | **8.898 (8.849 / 9.040)** | 19.932 (17.955 / 24.494) | 11.581 (8.758 / 16.685) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | **15.731 (15.565 / 16.224)** | skipped (rebuild with --features hptt) | 18.352 (18.181 / 18.744) | 24.119 (20.715 / 32.299) | 22.824 (19.296 / 23.731) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | **21.535 (20.147 / 22.950)** | skipped (rebuild with --features hptt) | 23.431 (23.155 / 24.755) | 277.377 (259.566 / 293.311) | 52.615 (45.924 / 56.906) | - |

## Threads: 4

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | **8.427 (8.070 / 8.663)** | skipped (rebuild with --features hptt) | 10.970 (10.574 / 11.307) | 21.738 (14.247 / 25.700) | 14.050 (8.394 / 15.032) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **7.240 (6.894 / 7.911)** | - | - | 10.014 (9.511 / 10.373) |
| `reverse_15d_3` | 15D 3^15 reverse | **25.485 (24.710 / 28.886)** | skipped (rebuild with --features hptt) | 27.410 (27.160 / 31.461) | 139.053 (123.808 / 140.248) | 28.444 (24.417 / 35.650) | - |
| `reverse_23d_2` | 23D 2^23 reverse | **16.699 (15.012 / 18.771)** | skipped (rebuild with --features hptt) | 21.264 (16.904 / 22.634) | 88.072 (70.367 / 118.805) | 24.883 (19.535 / 28.529) | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | 9698.320 (9400.927 / 11366.954) | skipped (rebuild with --features hptt) | **4279.149 (3569.411 / 5298.150)** | 10946.155 (8212.666 / 11213.247) | 5369.581 (4281.754 / 6661.147) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | **11.992 (11.213 / 12.378)** | skipped (rebuild with --features hptt) | 13.599 (13.192 / 14.224) | 24.018 (16.088 / 25.556) | 16.211 (9.281 / 17.694) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | **14.660 (13.752 / 16.609)** | - | 18.567 (16.561 / 20.861) | 198.418 (195.750 / 211.726) | 22.270 (16.121 / 23.991) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | **3.911 (3.783 / 4.310)** | skipped (rebuild with --features hptt) | 4.538 (4.339 / 5.380) | 34.163 (30.912 / 45.268) | 8.245 (6.222 / 13.725) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | 13.562 (12.875 / 14.565) | skipped (rebuild with --features hptt) | **11.911 (11.503 / 13.347)** | 32.198 (27.969 / 33.275) | 22.992 (15.652 / 26.535) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 20.023 (18.507 / 22.303) | skipped (rebuild with --features hptt) | **16.695 (16.503 / 16.840)** | 263.671 (248.380 / 275.846) | 32.829 (27.587 / 35.437) | - |
