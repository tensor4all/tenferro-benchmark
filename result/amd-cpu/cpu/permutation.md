# CPU Permutation Benchmark Results

- Target profile: `amd-cpu`
- Suite: `cpu/permutation`
- Suite file: `benchmarks/cpu/permutation.yaml`
- Timestamp: `2026-11-07T20:19:06.741674+00:00`
- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`

## CPU Information

- Model: `AMD EPYC 7763 64-Core Processor`
- Vendor: `AuthenticAMD`
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
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | **55.232 (55.101 / 55.355)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 61.145 (60.791 / 62.613) | 104.699 (79.457 / 107.668) | 84.690 (57.002 / 86.828) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **68.308 (67.301 / 69.181)** | - | - | 68.462 (67.887 / 69.172) |
| `reverse_15d_3` | 15D 3^15 reverse | 103.346 (102.023 / 104.586) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **92.705 (92.073 / 93.144)** | 287.995 (263.752 / 295.718) | 127.378 (117.463 / 138.456) | - |
| `reverse_23d_2` | 23D 2^23 reverse | **31.200 (31.174 / 31.288)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 58.017 (57.875 / 58.093) | 264.219 (255.130 / 278.114) | 71.832 (67.659 / 92.557) | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | **1269.565 (1250.252 / 1281.753)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 1345.633 (1328.432 / 1363.791) | 1987.074 (1979.782 / 1990.969) | 1605.440 (1601.540 / 1606.735) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | 68.450 (68.307 / 69.184) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **68.272 (67.724 / 68.550)** | 130.065 (110.800 / 144.179) | 96.333 (67.899 / 97.054) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | 71.593 (70.980 / 72.246) | - | **71.235 (70.095 / 71.453)** | 337.328 (335.109 / 350.942) | 88.871 (87.949 / 105.877) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | **7.486 (7.436 / 7.601)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 21.128 (20.734 / 21.451) | 55.527 (46.844 / 60.972) | 34.136 (29.331 / 38.513) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | 80.576 (80.049 / 81.550) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **78.829 (78.505 / 79.822)** | 109.592 (104.991 / 116.429) | 105.256 (103.511 / 116.718) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 82.507 (82.392 / 83.109) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **80.699 (80.139 / 81.709)** | 150.238 (148.844 / 157.891) | 140.446 (138.821 / 146.201) | - |

## Threads: 4

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | 19.952 (19.799 / 20.101) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **19.623 (19.027 / 19.774)** | 94.362 (80.379 / 95.867) | 38.339 (19.329 / 41.599) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **66.600 (65.762 / 66.782)** | - | - | 66.694 (65.884 / 66.833) |
| `reverse_15d_3` | 15D 3^15 reverse | 57.711 (57.206 / 57.885) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **54.440 (54.106 / 54.581)** | 278.470 (275.220 / 280.313) | 62.837 (59.919 / 64.210) | - |
| `reverse_23d_2` | 23D 2^23 reverse | **25.856 (25.588 / 26.094)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 42.966 (42.634 / 45.421) | 250.188 (234.740 / 251.787) | 43.521 (27.528 / 45.517) | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | **486.097 (485.423 / 487.279)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 502.584 (495.650 / 506.504) | 1982.696 (1978.082 / 1987.265) | 673.387 (671.123 / 675.907) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | 23.168 (22.630 / 23.596) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **22.859 (22.687 / 23.027)** | 127.154 (126.378 / 131.436) | 42.734 (37.095 / 47.827) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | 25.969 (25.862 / 26.074) | - | **25.554 (25.082 / 25.689)** | 337.322 (336.253 / 342.361) | 44.709 (40.835 / 49.472) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | **2.973 (2.952 / 3.008)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 7.409 (7.332 / 7.653) | 53.899 (50.809 / 55.067) | 18.617 (10.424 / 19.640) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | 29.050 (28.886 / 29.260) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **28.360 (28.165 / 28.711)** | 108.627 (107.316 / 113.096) | 43.769 (43.063 / 47.551) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 28.423 (28.048 / 28.474) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **27.817 (27.591 / 28.140)** | 148.979 (148.441 / 150.884) | 60.573 (57.369 / 61.541) | - |
