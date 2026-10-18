# CPU Permutation Benchmark Results

- Target profile: `amd-cpu`
- Suite: `cpu/permutation`
- Suite file: `benchmarks/cpu/permutation.yaml`
- Timestamp: `2026-10-18T19:17:50.465020+00:00`
- tenferro-rs commit: `c2ddccb14b89f8e9fa8a8fe3957367e32fb4e677`

## CPU Information

- Model: `AMD EPYC 7763 64-Core Processor`
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
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | 21.976 (21.906 / 22.126) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **13.840 (13.722 / 14.235)** | 43.761 (32.859 / 44.756) | 25.848 (15.334 / 27.207) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **13.243 (13.171 / 13.317)** | - | - | 13.468 (13.224 / 13.702) |
| `reverse_15d_3` | 15D 3^15 reverse | 52.367 (51.066 / 53.623) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **50.389 (49.523 / 52.329)** | 208.367 (199.782 / 215.466) | 73.899 (63.506 / 83.666) | - |
| `reverse_23d_2` | 23D 2^23 reverse | **30.944 (30.638 / 31.122)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 35.028 (34.824 / 35.433) | 271.554 (264.592 / 280.548) | 43.549 (37.547 / 49.252) | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | 917.840 (913.799 / 925.167) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 766.867 (764.094 / 772.827) | 1031.475 (1029.409 / 1379.445) | **710.874 (705.128 / 718.304)** | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | 14.946 (14.811 / 15.025) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **14.824 (14.696 / 15.209)** | 73.359 (62.463 / 85.059) | 29.744 (18.550 / 30.596) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | **16.986 (16.814 / 17.096)** | - | 17.065 (16.873 / 17.250) | 280.346 (279.506 / 292.476) | 32.560 (31.872 / 45.244) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | 15.611 (15.537 / 15.763) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **9.675 (9.561 / 9.885)** | 75.479 (71.549 / 81.446) | 70.561 (68.462 / 74.886) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | 36.709 (36.596 / 37.107) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **35.280 (35.177 / 36.015)** | 58.287 (54.099 / 59.313) | 50.431 (45.216 / 50.652) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 39.754 (39.434 / 39.919) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **36.903 (36.585 / 38.635)** | 77.054 (74.647 / 84.244) | 68.197 (66.444 / 75.903) | - |

## Threads: 4

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | 10.284 (9.992 / 10.344) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **8.915 (8.853 / 9.381)** | 39.531 (32.546 / 40.069) | 17.261 (10.549 / 17.880) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **12.767 (12.692 / 12.863)** | - | - | 12.911 (12.844 / 12.995) |
| `reverse_15d_3` | 15D 3^15 reverse | 23.018 (22.583 / 23.402) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **22.766 (21.837 / 23.286)** | 208.575 (205.178 / 210.586) | 38.557 (37.371 / 44.825) | - |
| `reverse_23d_2` | 23D 2^23 reverse | **19.107 (15.885 / 23.161)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 19.891 (19.758 / 20.057) | 266.597 (260.301 / 272.572) | 22.096 (16.288 / 22.754) | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | 313.055 (309.068 / 316.502) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 347.111 (343.510 / 348.401) | 1029.567 (1026.272 / 1258.218) | **301.010 (294.265 / 305.803)** | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | **8.837 (8.704 / 9.056)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 8.955 (8.847 / 9.207) | 69.101 (68.946 / 74.274) | 17.281 (11.046 / 22.160) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | 9.576 (9.233 / 10.009) | - | **9.365 (9.134 / 9.703)** | 276.247 (275.359 / 281.259) | 18.442 (17.978 / 24.207) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | 5.186 (5.113 / 5.294) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **3.351 (3.317 / 3.414)** | 73.102 (71.048 / 75.623) | 16.586 (14.627 / 20.418) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | 15.336 (15.305 / 15.505) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **14.704 (14.526 / 14.872)** | 56.624 (54.364 / 59.354) | 19.494 (15.810 / 21.874) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 15.025 (14.852 / 15.288) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **12.911 (12.722 / 13.145)** | 74.943 (73.819 / 77.793) | 30.134 (27.955 / 33.297) | - |
