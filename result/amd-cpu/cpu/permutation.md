# CPU Permutation Benchmark Results

- Target profile: `amd-cpu`
- Suite: `cpu/permutation`
- Suite file: `benchmarks/cpu/permutation.yaml`
- Timestamp: `2026-07-16T10:24:55.994400+00:00`
- tenferro-rs commit: `3abc9108e4f0500f4e75519711cdb4e21b9625df`

## CPU Information

- Model: `Intel(R) Xeon(R) CPU E5-2699 v3 @ 2.30GHz`
- Vendor: `GenuineIntel`
- Logical CPUs: `36`
- Sockets: `1`
- Cores per socket: `18`
- Threads per core: `2`
- NUMA nodes: `1`
- Python platform: `Linux-6.8.0-134-generic-x86_64-with-glibc2.39`

On the Linux CPU devcontainer, thread counts are controlled via `RAYON_NUM_THREADS` / `OMP_NUM_THREADS` / `JULIA_NUM_THREADS`; no CPU-affinity pinning (`taskset` / `numactl`) is applied, matching the repository devcontainer convention. The controlled thread environment is recorded per thread count in the run's `run_t<N>.yaml`.

`tenferro-transpose` is the eager `CpuBackend::transpose` structural op (compact col-major input only); `tenferro-to-contiguous` is `TypedTensorView::transpose_view` + `to_contiguous()` (accepts arbitrary source strides). Every backend allocates a fresh destination inside each timed call, so the table compares allocation-inclusive end-to-end materialization rather than destination-reuse copy kernels. `hptt` only participates in patterns with a contiguous source and destination. Correctness is verified against the `naive` odometer reference before any timing; a `FAILED` cell means that backend's output did not match the reference for that pattern.

## Threads: 1

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | naive odometer (ms) | tenferro-rs transpose (ms) | tenferro-rs to_contiguous (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `cyclic_13d_2` | 13D 2^13 cyclic [1,2,...,0] | 0.029 (0.028 / 0.035) | 0.053 (0.051 / 0.055) | 0.045 (0.043 / 0.047) | - | **0.010 (0.010 / 0.010)** | 0.029 (0.027 / 0.033) | 0.016 (0.015 / 0.018) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | - | - | **74.542 (74.330 / 74.701)** | - | - | 76.635 (76.588 / 76.708) |
| `reverse_13d_2` | 13D 2^13 reverse | 0.032 (0.031 / 0.036) | 0.139 (0.137 / 0.144) | 0.043 (0.043 / 0.048) | - | **0.013 (0.013 / 0.013)** | 0.030 (0.029 / 0.030) | 0.037 (0.036 / 0.039) | - |
| `reverse_20d_2` | 20D 2^20 reverse | 8.411 (8.408 / 8.417) | 12.036 (11.947 / 12.102) | 10.199 (10.196 / 10.205) | 8.809 (8.789 / 8.852) | **4.260 (4.175 / 4.346)** | 11.364 (7.882 / 11.523) | 4.729 (4.698 / 4.765) | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | 3687.371 (3685.564 / 3690.055) | 2166.281 (2160.833 / 2173.841) | 3756.963 (3754.698 / 3759.024) | 1972.714 (1970.275 / 1976.872) | **1447.279 (1446.033 / 1447.994)** | 1894.930 (1893.054 / 1895.346) | 1629.632 (1624.314 / 1630.414) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | 112.615 (112.525 / 113.348) | 90.340 (89.916 / 90.660) | 143.157 (142.923 / 143.758) | 97.141 (97.071 / 97.271) | **71.295 (71.225 / 71.373)** | 111.368 (110.879 / 111.646) | 76.738 (76.655 / 76.907) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | 118.815 (118.632 / 119.268) | - | 144.711 (144.543 / 145.270) | - | 82.863 (82.726 / 83.098) | 425.932 (424.756 / 426.031) | **81.823 (80.989 / 82.951)** | - |
| `transpose_2d_1024` | 2D 1024^2 transpose [1,0] | 6.834 (6.832 / 6.912) | 2.871 (2.844 / 3.154) | 7.047 (7.043 / 7.132) | 3.600 (3.490 / 3.666) | **1.944 (1.943 / 1.944)** | 10.209 (7.249 / 10.783) | 5.545 (3.494 / 5.604) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | 50.983 (50.655 / 51.192) | 34.482 (34.366 / 34.663) | 54.194 (54.043 / 54.452) | 32.333 (32.295 / 32.402) | **25.453 (25.418 / 25.506)** | 48.780 (48.667 / 48.889) | 26.571 (26.466 / 26.925) | - |
| `transpose_2d_256` | 2D 256^2 transpose [1,0] | 0.238 (0.237 / 0.240) | 0.151 (0.150 / 0.155) | 0.349 (0.333 / 0.354) | **0.132 (0.131 / 0.140)** | 0.192 (0.192 / 0.193) | 0.339 (0.333 / 0.356) | 0.284 (0.281 / 0.286) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | 142.618 (142.437 / 143.090) | 102.055 (101.812 / 102.373) | 176.591 (176.438 / 177.125) | 117.626 (117.475 / 118.019) | 104.360 (104.208 / 104.438) | 109.517 (109.403 / 109.588) | **84.529 (84.442 / 84.676)** | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 349.550 (349.370 / 350.217) | 234.621 (234.181 / 234.816) | 360.190 (359.520 / 360.633) | 145.305 (144.915 / 145.504) | **129.318 (129.200 / 129.766)** | 360.011 (359.349 / 360.572) | 218.672 (218.481 / 219.214) | - |

## Threads: 4

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | naive odometer (ms) | tenferro-rs transpose (ms) | tenferro-rs to_contiguous (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `cyclic_13d_2` | 13D 2^13 cyclic [1,2,...,0] | 0.084 (0.083 / 0.084) | 0.081 (0.078 / 0.089) | 0.067 (0.067 / 0.068) | - | **0.016 (0.015 / 0.016)** | 0.029 (0.026 / 0.031) | 0.017 (0.016 / 0.019) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | - | - | **75.485 (75.445 / 75.712)** | - | - | 75.862 (75.648 / 76.762) |
| `reverse_13d_2` | 13D 2^13 reverse | 0.093 (0.093 / 0.093) | 0.180 (0.176 / 0.185) | 0.131 (0.131 / 0.131) | - | 0.042 (0.042 / 0.042) | **0.027 (0.026 / 0.027)** | 0.036 (0.035 / 0.036) | - |
| `reverse_20d_2` | 20D 2^20 reverse | 8.805 (8.799 / 8.834) | 3.597 (3.523 / 3.826) | 11.103 (11.101 / 11.148) | 4.302 (4.287 / 4.324) | 3.974 (3.929 / 4.026) | 8.356 (7.995 / 11.515) | **3.000 (2.976 / 3.041)** | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | 3750.451 (3748.665 / 3751.942) | 701.945 (699.678 / 703.180) | 3827.454 (3826.019 / 3830.466) | 632.284 (629.298 / 636.899) | **510.951 (509.009 / 516.058)** | 1908.927 (1906.510 / 1910.120) | 622.294 (473.887 / 691.768) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | 116.970 (116.535 / 133.331) | **30.663 (30.489 / 31.085)** | 143.942 (143.798 / 144.464) | 40.602 (40.566 / 40.789) | 46.043 (41.383 / 48.348) | 112.672 (112.481 / 113.455) | 57.540 (32.511 / 79.735) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | 119.342 (119.135 / 120.039) | - | 144.555 (144.350 / 144.798) | - | **47.422 (47.177 / 47.664)** | 419.354 (411.153 / 420.693) | 63.195 (35.032 / 81.320) | - |
| `transpose_2d_1024` | 2D 1024^2 transpose [1,0] | 6.927 (6.926 / 6.930) | **0.792 (0.768 / 0.822)** | 7.338 (7.323 / 7.910) | 2.007 (1.939 / 2.041) | 1.624 (1.603 / 1.673) | 7.752 (7.719 / 8.400) | 3.468 (1.422 / 3.548) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | 52.199 (52.124 / 52.500) | 15.911 (15.289 / 17.040) | 57.481 (57.383 / 57.789) | **13.734 (13.047 / 14.626)** | 16.729 (16.547 / 17.037) | 48.224 (48.139 / 48.345) | 14.737 (14.404 / 15.129) | - |
| `transpose_2d_256` | 2D 256^2 transpose [1,0] | 0.255 (0.255 / 0.258) | 0.111 (0.106 / 0.126) | 0.357 (0.355 / 0.383) | **0.092 (0.091 / 0.102)** | 0.186 (0.185 / 0.188) | 0.326 (0.324 / 0.329) | 0.414 (0.394 / 0.422) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | 150.161 (150.010 / 150.618) | 48.465 (44.941 / 50.223) | 186.023 (185.755 / 186.349) | **42.308 (42.161 / 42.441)** | 51.278 (48.711 / 51.545) | 106.488 (106.360 / 106.771) | 53.375 (33.247 / 83.774) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 395.320 (394.878 / 396.024) | 80.181 (79.590 / 80.672) | 416.519 (415.501 / 416.880) | **51.936 (51.534 / 52.196)** | 58.760 (57.616 / 59.406) | 360.547 (359.718 / 361.460) | 102.361 (85.683 / 119.832) | - |
