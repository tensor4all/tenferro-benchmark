# CPU Permutation Benchmark Results

- Target profile: `amd-cpu`
- Suite: `cpu/permutation`
- Suite file: `benchmarks/cpu/permutation.yaml`
- Timestamp: `2026-11-06T19:45:35.280888+00:00`
- tenferro-rs commit: `a000e5c4929d5bebb91e783c329b27106baed12e`

## CPU Information

- Model: `INTEL(R) XEON(R) PLATINUM 8573C`
- Vendor: `GenuineIntel`
- Logical CPUs: `4`
- Sockets: `1`
- Cores per socket: `2`
- Threads per core: `2`
- NUMA nodes: `1`
- Transparent huge pages: `madvise (always [madvise] never)`
- Transparent huge page defrag: `madvise (always defer defer+madvise [madvise] never)`
- Python platform: `Linux-7.0.0-1019-azure-x86_64-with-glibc2.39`
- L3 domains (shared CPUs): `0-3`
- Container: `devcontainer`

Thread counts are controlled via `RAYON_NUM_THREADS` / `OMP_NUM_THREADS` / `JULIA_NUM_THREADS`; no CPU-affinity pinning (`taskset` / `numactl`) was applied. The controlled thread environment is recorded per thread count in the run's `run_t<N>.yaml`.

`tenferro-rs` measures `TypedTensorView::transpose_view` followed by `BackendSession::to_contiguous_read`; one shared session is entered before timing, and input view descriptors and metadata-only `transpose_view` are built outside the timed region. Materialization accepts arbitrary source strides. Every backend allocates a fresh destination inside each timed call, so the table compares allocation-inclusive end-to-end materialization rather than destination-reuse copy kernels. `hptt` only participates in patterns with a contiguous source and destination. Correctness is verified against an internal, untimed odometer reference before any timing; a `FAILED` cell means that backend's output did not match the reference for that pattern.

## Threads: 1

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | **52.175 (52.071 / 52.318)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 56.312 (56.157 / 56.508) | 109.066 (85.003 / 111.323) | 81.509 (58.196 / 82.182) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **64.569 (64.420 / 65.111)** | - | - | 64.806 (64.631 / 65.055) |
| `reverse_15d_3` | 15D 3^15 reverse | 94.653 (94.498 / 95.201) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **88.835 (88.413 / 89.531)** | 210.673 (184.289 / 215.671) | 131.448 (105.652 / 131.714) | - |
| `reverse_23d_2` | 23D 2^23 reverse | **31.023 (30.653 / 31.240)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 44.995 (44.757 / 45.465) | 129.159 (119.923 / 146.285) | 85.916 (83.263 / 104.752) | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | **1032.038 (1024.537 / 1039.828)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 1061.053 (1060.089 / 1064.529) | 1391.197 (1379.230 / 1401.121) | 1158.310 (1149.822 / 1162.105) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | 54.363 (54.035 / 54.463) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **54.174 (53.819 / 54.504)** | 116.979 (91.940 / 117.706) | 91.344 (66.039 / 93.835) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | 56.847 (56.660 / 57.138) | - | **56.460 (56.245 / 56.783)** | 250.141 (232.071 / 260.195) | 89.012 (86.987 / 99.865) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | **10.905 (10.466 / 11.204)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 19.011 (18.829 / 19.428) | 38.854 (30.754 / 43.848) | 20.988 (20.472 / 29.178) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | 67.076 (66.789 / 67.342) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **66.236 (66.000 / 66.670)** | 94.726 (78.999 / 96.871) | 84.674 (73.930 / 92.224) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 176.983 (172.585 / 180.338) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 156.498 (153.252 / 161.461) | 91.400 (82.560 / 97.546) | **83.827 (76.369 / 92.136)** | - |

## Threads: 4

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | 19.346 (19.261 / 19.512) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **19.269 (19.175 / 19.400)** | 103.142 (86.440 / 104.405) | 38.868 (21.028 / 41.377) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **64.510 (64.249 / 64.639)** | - | - | 65.787 (65.399 / 66.301) |
| `reverse_15d_3` | 15D 3^15 reverse | 54.187 (53.699 / 54.950) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **52.737 (52.395 / 53.690)** | 197.116 (188.392 / 203.162) | 66.700 (59.884 / 71.103) | - |
| `reverse_23d_2` | 23D 2^23 reverse | **15.950 (15.889 / 16.278)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 36.184 (36.065 / 36.317) | 142.073 (126.778 / 147.230) | 34.786 (28.943 / 44.174) | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | **395.438 (393.816 / 400.498)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 427.386 (424.137 / 430.272) | 1279.149 (1264.566 / 1284.123) | 493.671 (491.672 / 496.943) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | 20.326 (20.027 / 20.535) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **20.054 (19.926 / 20.106)** | 104.799 (104.281 / 112.720) | 45.873 (24.495 / 46.359) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | 21.848 (21.692 / 22.171) | - | **21.264 (21.190 / 21.303)** | 238.356 (237.470 / 245.455) | 41.365 (39.563 / 47.741) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | **3.926 (3.856 / 4.101)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 6.492 (6.429 / 6.651) | 35.537 (31.667 / 39.352) | 12.029 (7.485 / 14.648) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | **26.902 (26.780 / 27.081)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 26.914 (26.839 / 27.018) | 84.524 (79.110 / 90.284) | 38.697 (27.688 / 44.377) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 25.996 (25.653 / 26.121) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **25.943 (25.735 / 26.182)** | 86.687 (85.507 / 91.633) | 38.493 (36.720 / 43.867) | - |
