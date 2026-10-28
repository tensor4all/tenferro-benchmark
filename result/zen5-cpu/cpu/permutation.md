# CPU Permutation Benchmark Results

- Target profile: `zen5-cpu`
- Suite: `cpu/permutation`
- Suite file: `benchmarks/cpu/permutation.yaml`
- Timestamp: `2026-10-28T01:14:23.435274+00:00`
- tenferro-rs commit: `8fec1ca70852b930c092e7177c30bb8af7594a57`

## CPU Information

- Model: `AMD Ryzen AI 9 HX 470 w/ Radeon 890M`
- Vendor: `AuthenticAMD`
- Logical CPUs: `24`
- Sockets: `1`
- Cores per socket: `12`
- Threads per core: `2`
- NUMA nodes: `1`
- Python platform: `Linux-7.0.0-38-generic-x86_64-with-glibc2.39`
- L3 domains (shared CPUs): `0-3,12-15; 4-11,16-23`
- CPU boost: `enabled`

On the Linux CPU devcontainer, thread counts are controlled via `RAYON_NUM_THREADS` / `OMP_NUM_THREADS` / `JULIA_NUM_THREADS`; no CPU-affinity pinning (`taskset` / `numactl`) is applied, matching the repository devcontainer convention. The controlled thread environment is recorded per thread count in the run's `run_t<N>.yaml`.

`tenferro-rs` measures `TypedTensorView::transpose_view` followed by `BackendSession::to_contiguous_read`; one shared session is entered before timing, and input view descriptors and metadata-only `transpose_view` are built outside the timed region. Materialization accepts arbitrary source strides. Every backend allocates a fresh destination inside each timed call, so the table compares allocation-inclusive end-to-end materialization rather than destination-reuse copy kernels. `hptt` only participates in patterns with a contiguous source and destination. Correctness is verified against an internal, untimed odometer reference before any timing; a `FAILED` cell means that backend's output did not match the reference for that pattern.

## Threads: 1

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | **23.353 (23.298 / 23.532)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 26.773 (26.702 / 26.829) | 56.022 (36.738 / 56.507) | 53.681 (35.659 / 55.072) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | 24.092 (24.015 / 24.209) | - | - | **23.889 (23.618 / 24.065)** |
| `reverse_15d_3` | 15D 3^15 reverse | **47.437 (46.900 / 48.859)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 47.575 (46.979 / 48.816) | 201.032 (181.421 / 202.038) | 100.057 (80.743 / 100.978) | - |
| `reverse_23d_2` | 23D 2^23 reverse | **17.333 (17.313 / 17.380)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 29.096 (29.058 / 29.117) | 175.393 (173.558 / 188.906) | 46.826 (46.424 / 61.352) | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | **530.417 (528.564 / 532.553)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 550.466 (550.134 / 552.926) | 907.605 (899.829 / 914.239) | 682.466 (674.342 / 684.357) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | **26.322 (26.207 / 26.392)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 28.836 (28.773 / 28.985) | 67.239 (47.095 / 68.515) | 66.003 (44.076 / 66.644) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | **26.294 (26.202 / 26.574)** | - | 26.419 (26.402 / 26.628) | 154.294 (134.099 / 156.392) | 64.943 (49.823 / 76.306) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | **5.196 (5.186 / 5.214)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 9.648 (9.632 / 9.672) | 29.653 (23.035 / 34.900) | 15.069 (13.943 / 21.794) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | **32.327 (32.251 / 32.465)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 32.716 (32.631 / 33.008) | 54.321 (41.108 / 56.498) | 52.217 (39.332 / 53.282) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 51.355 (51.324 / 51.409) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **51.173 (51.113 / 51.319)** | 88.407 (88.254 / 95.491) | 65.306 (60.579 / 72.930) | - |

## Threads: 4

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | **9.008 (8.932 / 9.065)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 9.059 (8.909 / 9.487) | 50.240 (37.573 / 52.024) | 22.914 (10.939 / 26.361) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **24.198 (24.118 / 24.301)** | - | - | 25.528 (25.503 / 25.576) |
| `reverse_15d_3` | 15D 3^15 reverse | 24.914 (24.701 / 25.104) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **24.792 (24.615 / 25.143)** | 191.802 (189.575 / 192.260) | 45.386 (44.662 / 48.099) | - |
| `reverse_23d_2` | 23D 2^23 reverse | **12.688 (12.517 / 12.758)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 21.878 (21.760 / 21.948) | 179.001 (175.032 / 185.913) | 25.863 (15.158 / 29.587) | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | **216.095 (215.903 / 216.182)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 217.389 (217.121 / 217.500) | 921.240 (908.189 / 927.503) | 254.964 (237.442 / 255.947) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | 10.907 (10.834 / 10.969) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **9.868 (9.718 / 10.346)** | 59.900 (59.151 / 60.029) | 27.169 (13.708 / 31.244) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | 10.560 (10.505 / 10.591) | - | **10.541 (10.231 / 10.759)** | 145.558 (145.314 / 147.580) | 28.214 (26.435 / 31.748) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | **3.024 (3.007 / 3.078)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 3.608 (3.574 / 3.670) | 28.356 (26.795 / 29.115) | 9.642 (4.851 / 10.987) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | 12.371 (12.174 / 12.537) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **11.215 (11.149 / 11.668)** | 49.375 (48.966 / 49.980) | 23.870 (20.237 / 24.570) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 14.894 (14.771 / 15.030) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **14.375 (14.147 / 14.714)** | 87.760 (86.166 / 88.744) | 26.953 (25.138 / 28.641) | - |

## Threads: 8

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | **7.705 (7.535 / 7.859)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 7.849 (7.626 / 7.896) | 60.210 (54.298 / 61.530) | 24.627 (8.717 / 27.502) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | 32.796 (32.213 / 32.886) | - | - | **32.537 (32.150 / 32.612)** |
| `reverse_15d_3` | 15D 3^15 reverse | 30.773 (30.432 / 31.163) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **30.715 (30.514 / 31.087)** | 216.372 (214.980 / 217.893) | 42.450 (39.950 / 44.811) | - |
| `reverse_23d_2` | 23D 2^23 reverse | **12.782 (12.610 / 12.881)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 26.251 (26.207 / 26.440) | 196.518 (193.408 / 203.619) | 22.663 (10.607 / 24.372) | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | 209.380 (209.182 / 209.718) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **201.310 (201.215 / 206.096)** | 1030.315 (1009.038 / 1037.005) | 245.357 (227.913 / 246.195) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | 8.854 (8.711 / 9.025) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **8.415 (8.316 / 8.546)** | 79.651 (78.840 / 83.257) | 27.067 (22.598 / 29.473) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | 9.683 (9.523 / 9.703) | - | **9.550 (9.384 / 9.640)** | 216.703 (215.886 / 217.616) | 28.088 (24.323 / 28.440) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | **2.498 (2.475 / 2.512)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 2.887 (2.830 / 2.944) | 39.579 (38.281 / 40.740) | 9.534 (3.997 / 11.266) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | 9.444 (9.331 / 9.589) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **9.242 (9.027 / 9.339)** | 61.644 (61.431 / 64.140) | 24.718 (19.734 / 26.591) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 11.053 (10.974 / 11.122) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **10.936 (10.783 / 11.146)** | 104.484 (103.505 / 105.902) | 26.280 (23.242 / 29.144) | - |
