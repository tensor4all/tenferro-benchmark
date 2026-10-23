# CPU Permutation Benchmark Results

- Target profile: `amd-cpu`
- Suite: `cpu/permutation`
- Suite file: `benchmarks/cpu/permutation.yaml`
- Timestamp: `2026-10-23T18:51:48.997231+00:00`
- tenferro-rs commit: `a24fccea2dc7cee09df442104201b7c52f9b943b`

## CPU Information

- Model: `AMD EPYC 9V45 96-Core Processor`
- Vendor: `AuthenticAMD`
- Logical CPUs: `4`
- Sockets: `1`
- Cores per socket: `2`
- Threads per core: `2`
- NUMA nodes: `1`
- Python platform: `Linux-6.17.0-1022-azure-x86_64-with-glibc2.39`

On the Linux CPU devcontainer, thread counts are controlled via `RAYON_NUM_THREADS` / `OMP_NUM_THREADS` / `JULIA_NUM_THREADS`; no CPU-affinity pinning (`taskset` / `numactl`) is applied, matching the repository devcontainer convention. The controlled thread environment is recorded per thread count in the run's `run_t<N>.yaml`.

`tenferro-rs` measures `TypedTensorView::transpose_view` followed by `BackendSession::to_contiguous_read`; one shared session is entered before timing, and input view descriptors and metadata-only `transpose_view` are built outside the timed region. Materialization accepts arbitrary source strides. Every backend allocates a fresh destination inside each timed call, so the table compares allocation-inclusive end-to-end materialization rather than destination-reuse copy kernels. `hptt` only participates in patterns with a contiguous source and destination. Correctness is verified against an internal, untimed odometer reference before any timing; a `FAILED` cell means that backend's output did not match the reference for that pattern.

## Threads: 1

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | 12.977 (12.931 / 13.046) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **12.482 (12.461 / 12.522)** | 28.965 (19.930 / 29.762) | 20.182 (18.818 / 30.014) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | 11.172 (11.141 / 11.191) | - | - | **11.159 (11.123 / 11.211)** |
| `reverse_15d_3` | 15D 3^15 reverse | 45.145 (43.542 / 46.439) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **44.583 (43.414 / 46.430)** | 206.751 (196.883 / 208.019) | 85.512 (74.510 / 87.279) | - |
| `reverse_23d_2` | 23D 2^23 reverse | **18.008 (17.983 / 18.048)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 19.609 (19.565 / 19.676) | 205.462 (197.987 / 209.785) | 51.004 (49.198 / 59.543) | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | 789.334 (785.549 / 795.335) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 746.831 (742.002 / 758.677) | 488.629 (487.834 / 842.421) | **389.733 (384.952 / 398.662)** | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | 10.321 (10.183 / 10.452) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **10.274 (10.147 / 10.445)** | 43.873 (33.839 / 47.118) | 33.722 (22.766 / 34.751) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | **10.239 (10.223 / 10.308)** | - | 10.506 (10.490 / 10.554) | 152.803 (143.877 / 154.265) | 37.690 (26.661 / 37.984) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | **5.228 (5.213 / 5.275)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 6.792 (6.776 / 6.837) | 41.172 (37.527 / 49.925) | 13.525 (13.202 / 17.400) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | 19.659 (19.530 / 19.713) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **19.159 (19.038 / 19.286)** | 29.392 (24.954 / 29.638) | 27.984 (23.423 / 28.056) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | **28.908 (28.693 / 29.155)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 28.948 (28.737 / 29.137) | 40.611 (36.233 / 49.831) | 38.198 (33.925 / 38.653) | - |

## Threads: 4

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | **4.821 (4.704 / 4.942)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 4.934 (4.832 / 5.118) | 26.769 (20.244 / 26.947) | 13.726 (7.749 / 14.190) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | 11.738 (11.669 / 11.940) | - | - | **11.733 (11.670 / 11.809)** |
| `reverse_15d_3` | 15D 3^15 reverse | **26.067 (25.282 / 27.306)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 26.310 (24.974 / 27.928) | 206.766 (200.594 / 208.837) | 45.282 (37.593 / 50.779) | - |
| `reverse_23d_2` | 23D 2^23 reverse | **9.449 (8.473 / 14.542)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 12.174 (11.945 / 12.578) | 203.532 (196.819 / 211.584) | 20.847 (19.723 / 26.335) | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | 243.322 (242.197 / 248.244) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 259.399 (256.658 / 261.634) | 520.984 (511.111 / 765.505) | **188.554 (180.354 / 196.661)** | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | 4.763 (4.668 / 4.875) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **4.692 (4.599 / 4.829)** | 41.934 (35.256 / 42.796) | 17.299 (10.384 / 17.846) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | **5.448 (5.383 / 5.585)** | - | 5.729 (5.592 / 5.791) | 153.845 (150.512 / 163.242) | 24.509 (22.269 / 31.925) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | **2.646 (2.580 / 2.985)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 3.591 (3.492 / 3.686) | 48.843 (47.050 / 52.181) | 6.916 (4.915 / 9.324) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | 10.286 (10.029 / 10.442) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **9.641 (9.616 / 9.734)** | 28.366 (25.439 / 28.832) | 12.579 (10.212 / 13.418) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | **10.864 (10.819 / 11.133)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 11.871 (11.699 / 12.146) | 40.722 (38.974 / 46.690) | 16.736 (13.967 / 23.434) | - |
