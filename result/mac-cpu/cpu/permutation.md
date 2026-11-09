# CPU Permutation Benchmark Results

- Target profile: `mac-cpu`
- Suite: `cpu/permutation`
- Suite file: `benchmarks/cpu/permutation.yaml`
- Timestamp: `2026-11-09T18:50:17.553251+00:00`
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
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | **8.132 (7.732 / 8.594)** | skipped (rebuild with --features hptt) | 10.301 (9.929 / 10.842) | 22.257 (12.775 / 30.376) | 20.094 (11.627 / 21.326) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | 7.838 (7.683 / 8.049) | - | - | **7.205 (7.010 / 7.481)** |
| `reverse_15d_3` | 15D 3^15 reverse | 35.376 (34.848 / 37.383) | skipped (rebuild with --features hptt) | **32.633 (32.353 / 34.989)** | 140.171 (122.896 / 144.859) | 33.080 (30.532 / 40.853) | - |
| `reverse_23d_2` | 23D 2^23 reverse | 37.976 (33.620 / 54.044) | skipped (rebuild with --features hptt) | **27.228 (25.915 / 31.460)** | 80.464 (65.225 / 85.118) | 38.361 (27.583 / 41.064) | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | 7430.662 (5969.083 / 8919.162) | skipped (rebuild with --features hptt) | 8529.810 (4707.609 / 10243.530) | 7519.475 (6169.823 / 8253.971) | **7093.364 (4868.383 / 8084.503)** | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | **8.564 (8.313 / 8.799)** | skipped (rebuild with --features hptt) | 11.126 (10.245 / 11.790) | 28.546 (17.391 / 31.082) | 20.874 (10.746 / 23.396) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | **17.013 (16.423 / 17.408)** | - | 17.047 (16.890 / 18.046) | 217.938 (202.682 / 229.569) | 27.706 (18.907 / 35.638) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | **9.656 (9.150 / 10.612)** | skipped (rebuild with --features hptt) | 10.294 (9.985 / 10.947) | 27.825 (20.503 / 37.461) | 12.259 (9.771 / 18.730) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | **20.528 (19.838 / 23.727)** | skipped (rebuild with --features hptt) | 20.573 (20.229 / 21.645) | 32.360 (25.574 / 38.502) | 31.857 (27.282 / 35.956) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | **24.978 (23.131 / 26.719)** | skipped (rebuild with --features hptt) | 28.689 (27.785 / 31.325) | 276.071 (255.446 / 279.114) | 67.036 (48.327 / 72.285) | - |

## Threads: 4

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | **8.419 (8.116 / 8.613)** | skipped (rebuild with --features hptt) | 10.817 (10.686 / 11.129) | 27.581 (19.347 / 40.155) | 20.272 (11.536 / 26.348) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **7.927 (6.944 / 9.408)** | - | - | 11.565 (10.294 / 12.325) |
| `reverse_15d_3` | 15D 3^15 reverse | 62.862 (45.812 / 69.220) | skipped (rebuild with --features hptt) | 30.175 (25.810 / 33.611) | 143.716 (135.776 / 148.036) | **29.729 (25.500 / 32.004)** | - |
| `reverse_23d_2` | 23D 2^23 reverse | **21.457 (15.619 / 23.064)** | skipped (rebuild with --features hptt) | 36.257 (31.833 / 42.081) | 106.853 (78.482 / 123.287) | 30.675 (23.878 / 32.852) | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | **3678.992 (2965.858 / 4616.177)** | skipped (rebuild with --features hptt) | 8874.278 (8435.667 / 9340.774) | 14867.169 (9624.252 / 15742.882) | 8313.359 (6049.337 / 10285.130) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | **8.605 (8.335 / 9.085)** | skipped (rebuild with --features hptt) | 12.362 (12.007 / 12.885) | 32.077 (18.800 / 35.850) | 18.916 (10.387 / 20.355) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | **13.217 (11.948 / 15.107)** | - | 14.943 (13.137 / 17.883) | 245.427 (225.952 / 280.269) | 25.308 (20.082 / 29.676) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | **5.311 (4.787 / 5.922)** | skipped (rebuild with --features hptt) | 5.555 (5.097 / 5.854) | 27.594 (24.159 / 32.178) | 8.174 (5.650 / 9.020) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | **10.857 (10.438 / 11.805)** | skipped (rebuild with --features hptt) | 13.433 (12.501 / 17.123) | 26.697 (20.966 / 31.533) | 19.490 (16.286 / 22.344) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 16.974 (12.383 / 22.193) | skipped (rebuild with --features hptt) | **15.650 (15.209 / 17.135)** | 272.745 (266.110 / 274.334) | 34.195 (26.252 / 37.456) | - |
