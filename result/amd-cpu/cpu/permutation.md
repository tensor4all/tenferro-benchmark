# CPU Permutation Benchmark Results

- Target profile: `amd-cpu`
- Suite: `cpu/permutation`
- Suite file: `benchmarks/cpu/permutation.yaml`
- Timestamp: `2026-11-24T20:16:01.308422+00:00`
- tenferro-rs commit: `ce9125d0cff4ee40841120759b5df45252c27f22`

## CPU Information

- Model: `AMD EPYC 9V74 80-Core Processor`
- Vendor: `AuthenticAMD`
- Logical CPUs: `4`
- Sockets: `1`
- Cores per socket: `2`
- Threads per core: `2`
- NUMA nodes: `1`
- Transparent huge pages: `madvise (always [madvise] never)`
- Transparent huge page defrag: `madvise (always defer defer+madvise [madvise] never)`
- Python platform: `Linux-7.0.0-1020-azure-x86_64-with-glibc2.39`
- L3 domains (shared CPUs): `0-3`
- Container: `devcontainer`

Thread counts are controlled via `RAYON_NUM_THREADS` / `OMP_NUM_THREADS` / `JULIA_NUM_THREADS`; no CPU-affinity pinning (`taskset` / `numactl`) was applied. The controlled thread environment is recorded per thread count in the run's `run_t<N>.yaml`.

`tenferro-rs` measures `TypedTensorView::transpose_view` followed by `BackendSession::to_contiguous_read`; one shared session is entered before timing, and input view descriptors and metadata-only `transpose_view` are built outside the timed region. Materialization accepts arbitrary source strides. Every backend allocates a fresh destination inside each timed call, so the table compares allocation-inclusive end-to-end materialization rather than destination-reuse copy kernels. `hptt` only participates in patterns with a contiguous source and destination. Correctness is verified against an internal, untimed odometer reference before any timing; a `FAILED` cell means that backend's output did not match the reference for that pattern.

## Threads: 1

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | **58.792 (58.326 / 59.553)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 58.874 (58.718 / 59.745) | 81.540 (78.207 / 105.731) | 71.561 (70.152 / 95.880) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **61.950 (61.711 / 64.039)** | - | - | 61.989 (60.776 / 64.568) |
| `reverse_15d_3` | 15D 3^15 reverse | 109.069 (107.750 / 110.463) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **96.812 (95.822 / 98.255)** | 249.480 (241.568 / 258.902) | 122.653 (118.535 / 144.637) | - |
| `reverse_23d_2` | 23D 2^23 reverse | **28.530 (28.008 / 29.319)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 52.547 (51.514 / 53.142) | 284.724 (283.003 / 299.156) | 78.105 (73.250 / 83.212) | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | **1354.429 (1345.627 / 1373.936)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 1405.699 (1389.571 / 1411.143) | 1568.552 (1523.670 / 1577.936) | 1856.334 (1839.998 / 1865.602) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | **68.732 (67.769 / 69.662)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 69.927 (69.509 / 70.876) | 102.673 (101.066 / 103.771) | 89.033 (87.353 / 90.637) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | 71.585 (70.673 / 72.440) | - | **70.765 (69.565 / 72.090)** | 298.708 (283.625 / 307.395) | 91.832 (87.207 / 92.348) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | **7.364 (7.302 / 7.482)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 19.259 (19.101 / 19.774) | 41.952 (40.709 / 42.363) | 31.138 (30.596 / 31.587) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | 83.595 (82.756 / 84.332) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **79.079 (78.144 / 80.962)** | 98.063 (96.876 / 98.534) | 84.676 (83.711 / 84.757) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 86.083 (84.612 / 86.869) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **84.326 (82.463 / 85.015)** | 141.259 (139.902 / 142.368) | 125.648 (124.136 / 126.825) | - |

## Threads: 4

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | 21.177 (20.912 / 21.413) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **20.912 (20.177 / 21.165)** | 80.673 (79.401 / 98.675) | 25.157 (23.728 / 52.153) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **63.055 (62.419 / 64.257)** | - | - | 63.572 (63.006 / 64.750) |
| `reverse_15d_3` | 15D 3^15 reverse | 60.444 (59.861 / 62.066) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 53.422 (53.154 / 55.541) | 242.743 (239.490 / 259.067) | **50.110 (46.431 / 79.051)** | - |
| `reverse_23d_2` | 23D 2^23 reverse | **21.786 (21.069 / 22.309)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 39.853 (39.694 / 39.992) | 288.481 (285.934 / 292.307) | 28.433 (27.520 / 47.336) | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | 526.743 (516.818 / 581.752) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **526.306 (510.693 / 578.450)** | 1547.859 (1523.829 / 1599.898) | 785.842 (748.983 / 795.867) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | 24.943 (24.519 / 25.360) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **23.055 (22.763 / 23.370)** | 105.960 (103.711 / 106.595) | 40.521 (38.970 / 42.223) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | **25.311 (24.704 / 25.923)** | - | 25.400 (25.091 / 26.935) | 301.255 (291.372 / 305.151) | 47.303 (46.584 / 47.498) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | **4.530 (4.435 / 4.609)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 7.853 (7.726 / 8.059) | 43.890 (43.141 / 44.129) | 14.014 (13.702 / 14.337) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | **28.414 (28.125 / 29.228)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 29.261 (28.986 / 30.099) | 101.178 (97.949 / 102.729) | 42.158 (38.801 / 43.334) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 29.726 (29.163 / 30.085) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **29.196 (28.989 / 30.113)** | 150.314 (148.591 / 151.583) | 58.095 (56.895 / 59.386) | - |
