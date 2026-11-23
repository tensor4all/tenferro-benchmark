# CPU Permutation Benchmark Results

- Target profile: `mac-cpu`
- Suite: `cpu/permutation`
- Suite file: `benchmarks/cpu/permutation.yaml`
- Timestamp: `2026-11-23T18:23:20.637434+00:00`
- tenferro-rs commit: `ce9125d0cff4ee40841120759b5df45252c27f22`

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
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | **6.940 (6.875 / 7.148)** | skipped (rebuild with --features hptt) | 8.991 (8.961 / 9.127) | 11.016 (9.695 / 11.471) | 10.944 (10.685 / 11.306) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **6.744 (6.723 / 6.767)** | - | - | 6.758 (6.724 / 6.873) |
| `reverse_15d_3` | 15D 3^15 reverse | 33.198 (30.945 / 33.346) | skipped (rebuild with --features hptt) | 32.133 (31.412 / 33.297) | 123.046 (118.703 / 123.845) | **19.154 (18.549 / 20.248)** | - |
| `reverse_23d_2` | 23D 2^23 reverse | 28.344 (26.811 / 28.594) | skipped (rebuild with --features hptt) | 24.179 (23.973 / 25.011) | 67.108 (61.649 / 70.981) | **21.823 (21.485 / 22.050)** | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | 2326.014 (2282.927 / 2452.880) | skipped (rebuild with --features hptt) | **2208.050 (2197.357 / 2290.306)** | 2617.755 (2110.226 / 2910.968) | 2477.278 (2321.852 / 2526.275) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | **8.351 (8.273 / 8.369)** | skipped (rebuild with --features hptt) | 10.465 (10.425 / 11.020) | 16.552 (16.214 / 16.786) | 9.325 (8.818 / 9.462) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | **12.583 (12.533 / 12.715)** | - | 14.468 (14.387 / 14.547) | 190.592 (184.317 / 197.617) | 15.684 (15.514 / 15.942) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | 8.968 (8.753 / 9.110) | skipped (rebuild with --features hptt) | 8.863 (8.854 / 8.911) | 13.225 (13.044 / 13.973) | **6.412 (6.372 / 6.453)** | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | **16.367 (16.336 / 16.590)** | skipped (rebuild with --features hptt) | 17.721 (17.704 / 17.798) | 18.005 (17.848 / 18.891) | 19.064 (18.322 / 19.250) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | **20.724 (20.574 / 20.885)** | skipped (rebuild with --features hptt) | 22.807 (22.693 / 23.052) | 329.727 (284.529 / 351.251) | 38.676 (38.077 / 38.965) | - |

## Threads: 4

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | **6.239 (6.186 / 6.532)** | skipped (rebuild with --features hptt) | 9.057 (9.026 / 9.254) | 10.869 (10.246 / 11.589) | 7.517 (7.155 / 7.854) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **6.611 (6.573 / 6.766)** | - | - | 6.670 (6.568 / 6.708) |
| `reverse_15d_3` | 15D 3^15 reverse | 19.383 (18.824 / 19.748) | skipped (rebuild with --features hptt) | 17.417 (17.284 / 17.623) | 115.684 (112.893 / 117.729) | **12.306 (11.916 / 12.718)** | - |
| `reverse_23d_2` | 23D 2^23 reverse | 16.980 (14.761 / 18.146) | skipped (rebuild with --features hptt) | 14.254 (14.146 / 14.467) | 59.098 (56.800 / 61.719) | **13.881 (12.815 / 14.154)** | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | 3127.210 (2997.870 / 3252.045) | skipped (rebuild with --features hptt) | 2735.811 (2640.267 / 2801.229) | 3277.399 (2847.954 / 3879.200) | **2576.386 (2290.413 / 2678.136)** | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | **7.983 (7.956 / 8.082)** | skipped (rebuild with --features hptt) | 10.001 (9.928 / 10.362) | 16.858 (15.777 / 17.260) | 8.043 (7.980 / 8.193) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | 10.658 (10.473 / 10.795) | - | 12.052 (12.006 / 12.452) | 184.414 (178.907 / 189.726) | **9.769 (9.725 / 9.818)** | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | **3.765 (3.724 / 3.879)** | skipped (rebuild with --features hptt) | 4.319 (4.260 / 4.406) | 13.290 (13.132 / 13.449) | 4.993 (4.933 / 5.020) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | 10.910 (10.065 / 11.214) | skipped (rebuild with --features hptt) | 12.337 (12.086 / 12.636) | 17.325 (17.300 / 17.414) | **10.268 (10.202 / 10.326)** | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | **11.959 (11.771 / 12.203)** | skipped (rebuild with --features hptt) | 14.656 (14.519 / 15.139) | 296.077 (270.305 / 305.348) | 15.637 (15.013 / 15.967) | - |
