# CPU Permutation Benchmark Results

- Target profile: `amd-cpu`
- Suite: `cpu/permutation`
- Suite file: `benchmarks/cpu/permutation.yaml`
- Timestamp: `2026-10-19T18:53:28.252283+00:00`
- tenferro-rs commit: `14bc1ab933e3c7a9c966e08355670331ab9d993c`

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
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | 13.940 (13.706 / 14.161) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **12.624 (12.573 / 12.755)** | 28.509 (20.000 / 29.185) | 19.078 (18.110 / 28.794) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **11.009 (10.962 / 11.056)** | - | - | 11.055 (10.955 / 11.150) |
| `reverse_15d_3` | 15D 3^15 reverse | **44.317 (43.190 / 45.677)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 45.037 (43.458 / 45.908) | 205.003 (195.410 / 206.445) | 80.410 (70.692 / 95.749) | - |
| `reverse_23d_2` | 23D 2^23 reverse | **16.877 (16.819 / 17.013)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 19.285 (19.161 / 19.465) | 193.060 (191.743 / 200.865) | 45.798 (45.389 / 56.109) | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | 774.119 (767.168 / 779.910) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 566.937 (561.644 / 570.503) | 489.111 (472.980 / 852.655) | **376.304 (374.195 / 386.276)** | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | **9.337 (9.235 / 9.408)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 9.661 (9.572 / 9.788) | 42.409 (32.344 / 43.433) | 32.883 (22.164 / 33.768) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | 10.413 (10.353 / 10.530) | - | **10.364 (10.236 / 10.419)** | 144.197 (134.059 / 144.799) | 36.328 (26.075 / 36.844) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | 10.514 (10.453 / 10.730) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **6.559 (6.526 / 6.673)** | 39.973 (36.333 / 47.925) | 12.667 (12.146 / 16.310) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | 23.178 (22.959 / 23.653) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **18.276 (18.254 / 18.413)** | 27.479 (23.472 / 28.272) | 26.336 (22.398 / 27.400) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 33.378 (32.626 / 33.727) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **27.640 (27.282 / 27.924)** | 43.458 (39.091 / 52.510) | 39.149 (34.976 / 40.141) | - |

## Threads: 4

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | **7.034 (6.951 / 7.146)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 7.044 (6.919 / 7.100) | 25.324 (20.025 / 26.365) | 14.541 (9.082 / 15.533) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **11.212 (11.166 / 11.244)** | - | - | 11.615 (11.549 / 11.687) |
| `reverse_15d_3` | 15D 3^15 reverse | **20.217 (19.889 / 21.354)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 20.653 (19.871 / 21.349) | 197.673 (192.472 / 199.319) | 42.752 (38.539 / 44.054) | - |
| `reverse_23d_2` | 23D 2^23 reverse | **8.720 (8.033 / 13.896)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 12.246 (11.223 / 12.380) | 194.462 (187.859 / 199.619) | 21.846 (17.914 / 23.810) | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | 261.981 (260.040 / 266.328) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 242.082 (238.943 / 259.533) | 593.806 (479.615 / 642.371) | **189.745 (184.754 / 193.582)** | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | 7.464 (7.238 / 7.594) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **7.344 (7.241 / 7.628)** | 37.635 (32.526 / 43.316) | 17.267 (10.975 / 17.594) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | **8.487 (8.397 / 8.665)** | - | 8.834 (8.784 / 8.991) | 141.572 (139.953 / 146.452) | 23.074 (21.689 / 27.831) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | 6.598 (6.185 / 7.739) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **3.347 (3.286 / 3.408)** | 40.962 (39.967 / 43.983) | 7.093 (4.996 / 8.110) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | 11.478 (11.385 / 11.658) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **9.372 (9.271 / 9.438)** | 26.231 (23.522 / 27.077) | 13.083 (10.209 / 13.436) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 17.382 (17.202 / 17.543) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **11.289 (11.181 / 11.580)** | 36.961 (36.125 / 40.359) | 16.363 (14.107 / 19.929) | - |
