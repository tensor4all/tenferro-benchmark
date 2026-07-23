# CPU Permutation Benchmark Results

- Target profile: `amd-cpu`
- Suite: `cpu/permutation`
- Suite file: `benchmarks/cpu/permutation.yaml`
- Timestamp: `2026-07-23T05:17:03.038688+00:00`
- tenferro-rs commit: `68855c2b65b5adc42dccca9bac04fd136a8f14c8`

## CPU Information

- Model: `Intel(R) Xeon(R) CPU E5-2699 v3 @ 2.30GHz`
- Vendor: `GenuineIntel`
- Logical CPUs: `36`
- Sockets: `1`
- Cores per socket: `18`
- Threads per core: `2`
- NUMA nodes: `1`
- Python platform: `Linux-6.8.0-136-generic-x86_64-with-glibc2.39`

On the Linux CPU devcontainer, thread counts are controlled via `RAYON_NUM_THREADS` / `OMP_NUM_THREADS` / `JULIA_NUM_THREADS`; no CPU-affinity pinning (`taskset` / `numactl`) is applied, matching the repository devcontainer convention. The controlled thread environment is recorded per thread count in the run's `run_t<N>.yaml`.

`tenferro-rs` measures `TypedTensorView::transpose_view` followed by `CpuBackend::to_contiguous`; the metadata-only `transpose_view` is built outside the timed region, and `to_contiguous` accepts arbitrary source strides. Every backend allocates a fresh destination inside each timed call, so the table compares allocation-inclusive end-to-end materialization rather than destination-reuse copy kernels. `hptt` only participates in patterns with a contiguous source and destination. Correctness is verified against an internal, untimed odometer reference before any timing; a `FAILED` cell means that backend's output did not match the reference for that pattern.

## Threads: 1

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | 91.425 (91.012 / 92.924) | 84.530 (83.683 / 85.105) | **67.168 (66.333 / 67.939)** | 101.116 (100.828 / 101.630) | 70.259 (69.550 / 70.465) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **83.769 (83.518 / 86.055)** | - | - | 84.863 (83.533 / 85.576) |
| `reverse_15d_3` | 15D 3^15 reverse | 341.525 (340.126 / 343.598) | **121.577 (119.598 / 123.177)** | 166.660 (166.095 / 169.543) | 623.188 (608.472 / 626.373) | 157.524 (152.996 / 167.649) | - |
| `reverse_23d_2` | 23D 2^23 reverse | 183.166 (181.076 / 185.453) | 141.329 (137.295 / 143.686) | **103.400 (101.658 / 105.000)** | 353.508 (344.888 / 361.535) | 110.340 (107.585 / 118.322) | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | 2231.766 (2205.805 / 2251.516) | 2110.830 (2090.700 / 2138.637) | **1603.324 (1588.968 / 1638.379)** | 1970.715 (1959.330 / 1981.889) | 1812.010 (1795.805 / 1821.965) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | 101.255 (98.051 / 101.949) | 109.944 (109.778 / 111.235) | **78.224 (77.760 / 79.257)** | 125.836 (125.153 / 126.787) | 86.372 (84.346 / 89.890) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | 113.331 (111.283 / 115.101) | - | 91.614 (90.028 / 92.390) | 516.385 (512.643 / 517.863) | **91.099 (90.530 / 91.169)** | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | 35.944 (35.412 / 36.672) | 37.368 (36.614 / 37.694) | 29.347 (28.832 / 29.878) | 51.505 (51.131 / 52.732) | **27.540 (27.216 / 27.865)** | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | 115.683 (112.857 / 117.247) | 136.535 (135.050 / 138.108) | 119.610 (118.854 / 120.187) | 121.301 (119.248 / 122.167) | **92.747 (91.928 / 93.945)** | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 265.361 (252.484 / 279.352) | 165.972 (164.922 / 167.501) | **144.535 (144.267 / 145.508)** | 400.513 (397.992 / 401.775) | 225.941 (224.352 / 236.216) | - |

## Threads: 4

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | 41.178 (38.269 / 42.436) | 37.708 (37.633 / 37.983) | 68.537 (68.269 / 69.095) | 116.378 (116.000 / 116.913) | **28.613 (28.088 / 71.694)** | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **81.215 (80.774 / 81.746)** | - | - | 90.783 (82.703 / 91.625) |
| `reverse_15d_3` | 15D 3^15 reverse | 129.847 (129.158 / 130.339) | **71.387 (71.091 / 71.967)** | 100.602 (100.053 / 101.218) | 592.401 (590.536 / 594.414) | 83.638 (56.622 / 116.154) | - |
| `reverse_23d_2` | 23D 2^23 reverse | 61.625 (59.979 / 117.997) | 70.086 (69.905 / 70.437) | 71.513 (71.418 / 71.869) | 328.782 (327.092 / 331.105) | **34.915 (34.672 / 36.178)** | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | 769.929 (749.298 / 773.971) | 751.736 (733.483 / 757.257) | **606.017 (582.958 / 616.629)** | 1965.699 (1943.358 / 2003.793) | 717.295 (490.960 / 882.223) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | 45.059 (43.014 / 48.141) | 47.242 (44.489 / 47.290) | **44.644 (41.956 / 46.586)** | 148.531 (147.815 / 148.689) | 61.949 (34.845 / 84.350) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | 47.273 (45.111 / 50.402) | - | **45.111 (43.117 / 46.933)** | 563.009 (512.574 / 564.320) | 85.586 (38.811 / 117.037) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | **12.671 (12.091 / 13.902)** | 14.318 (14.218 / 15.005) | 16.326 (16.177 / 16.594) | 52.885 (52.603 / 53.169) | 15.379 (15.245 / 38.646) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | **46.764 (44.650 / 49.823)** | 48.339 (46.899 / 52.693) | 55.515 (52.259 / 57.350) | 128.765 (118.045 / 130.857) | 71.863 (35.917 / 84.552) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 86.338 (86.100 / 87.720) | **57.053 (56.244 / 64.657)** | 61.713 (60.230 / 62.420) | 340.453 (338.120 / 341.854) | 105.408 (78.715 / 121.732) | - |
