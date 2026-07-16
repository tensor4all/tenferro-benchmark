# CPU Permutation Benchmark Results

- Target profile: `amd-cpu`
- Suite: `cpu/permutation`
- Suite file: `benchmarks/cpu/permutation.yaml`
- Timestamp: `2026-07-16T11:17:56.833883+00:00`
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

`tenferro-transpose` is the eager `CpuBackend::transpose` structural op (compact col-major input only); `tenferro-to-contiguous` is `TypedTensorView::transpose_view` + `CpuBackend::to_contiguous` (accepts arbitrary source strides). Every backend allocates a fresh destination inside each timed call, so the table compares allocation-inclusive end-to-end materialization rather than destination-reuse copy kernels. `hptt` only participates in patterns with a contiguous source and destination. Correctness is verified against the `naive` odometer reference before any timing; a `FAILED` cell means that backend's output did not match the reference for that pattern.

## Threads: 1

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | naive odometer (ms) | tenferro-rs transpose (ms) | tenferro-rs to_contiguous (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `cyclic_13d_2` | 13D 2^13 cyclic [1,2,...,0] | 0.030 (0.028 / 0.035) | 0.052 (0.051 / 0.053) | 0.044 (0.043 / 0.046) | 0.012 (0.012 / 0.012) | **0.010 (0.010 / 0.010)** | 0.029 (0.027 / 0.034) | 0.017 (0.016 / 0.019) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | - | - | **77.214 (76.839 / 78.058)** | - | - | 77.571 (77.129 / 79.287) |
| `reverse_13d_2` | 13D 2^13 reverse | 0.034 (0.031 / 0.039) | 0.143 (0.140 / 0.148) | 0.043 (0.043 / 0.046) | 0.063 (0.063 / 0.071) | **0.014 (0.014 / 0.014)** | 0.030 (0.029 / 0.031) | 0.038 (0.038 / 0.039) | - |
| `reverse_20d_2` | 20D 2^20 reverse | 9.050 (9.045 / 9.076) | 11.944 (11.895 / 12.005) | 11.063 (11.058 / 11.172) | 9.172 (9.160 / 9.200) | **4.273 (4.179 / 4.311)** | 11.893 (8.376 / 12.051) | 4.935 (4.919 / 4.958) | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | 3533.213 (3529.031 / 3538.549) | 2138.777 (2135.469 / 2141.407) | 3734.577 (3731.278 / 3738.503) | 2049.931 (2048.866 / 2053.258) | **1523.065 (1522.178 / 1525.044)** | 1958.158 (1956.601 / 1959.005) | 1685.362 (1683.900 / 1687.225) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | 114.045 (113.918 / 114.642) | 92.156 (91.832 / 92.367) | 145.420 (145.284 / 145.834) | 100.502 (100.350 / 100.727) | **74.114 (74.052 / 74.345)** | 112.685 (112.452 / 113.118) | 78.874 (78.584 / 79.396) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | 119.944 (119.634 / 120.522) | - | 145.497 (145.110 / 146.136) | - | **82.688 (82.607 / 82.921)** | 432.650 (431.186 / 433.424) | 85.423 (83.762 / 85.754) | - |
| `transpose_2d_1024` | 2D 1024^2 transpose [1,0] | 6.933 (6.930 / 7.039) | 3.072 (2.867 / 3.687) | 7.911 (7.426 / 8.330) | 3.686 (3.656 / 3.763) | **1.954 (1.953 / 1.956)** | 10.587 (7.252 / 10.944) | 5.128 (3.035 / 5.205) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | 50.585 (50.419 / 50.885) | 33.518 (33.239 / 33.769) | 56.167 (55.033 / 58.529) | 32.502 (32.368 / 32.947) | **26.280 (26.183 / 26.536)** | 49.607 (49.496 / 49.946) | 26.382 (26.293 / 26.610) | - |
| `transpose_2d_256` | 2D 256^2 transpose [1,0] | 0.260 (0.260 / 0.263) | **0.152 (0.151 / 0.153)** | 0.369 (0.369 / 0.372) | 0.175 (0.174 / 0.178) | 0.200 (0.199 / 0.203) | 0.346 (0.344 / 0.349) | 0.290 (0.288 / 0.294) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | 144.519 (143.996 / 144.806) | 104.831 (104.087 / 105.247) | 180.540 (180.086 / 180.790) | 118.876 (118.581 / 119.199) | 109.916 (109.781 / 110.239) | 115.425 (115.140 / 115.651) | **87.997 (87.728 / 88.755)** | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 358.857 (356.790 / 363.310) | 234.050 (233.275 / 234.718) | 362.576 (361.834 / 363.779) | 149.137 (149.009 / 149.731) | **130.200 (129.910 / 130.702)** | 392.063 (391.432 / 392.975) | 246.286 (245.968 / 246.536) | - |

## Threads: 4

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | naive odometer (ms) | tenferro-rs transpose (ms) | tenferro-rs to_contiguous (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `cyclic_13d_2` | 13D 2^13 cyclic [1,2,...,0] | 0.083 (0.083 / 0.084) | 0.087 (0.083 / 0.096) | 0.130 (0.130 / 0.131) | 0.031 (0.030 / 0.031) | 0.027 (0.027 / 0.027) | 0.028 (0.028 / 0.033) | **0.017 (0.016 / 0.019)** | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | - | - | **74.490 (74.395 / 74.701)** | - | - | 75.150 (74.912 / 75.432) |
| `reverse_13d_2` | 13D 2^13 reverse | 0.093 (0.092 / 0.093) | 0.182 (0.177 / 0.187) | 0.130 (0.130 / 0.131) | 0.092 (0.091 / 0.094) | 0.039 (0.039 / 0.040) | **0.029 (0.029 / 0.030)** | 0.039 (0.038 / 0.048) | - |
| `reverse_20d_2` | 20D 2^20 reverse | 8.967 (8.887 / 9.103) | 3.653 (3.566 / 3.845) | 10.425 (10.290 / 10.534) | 4.365 (4.344 / 4.432) | 4.132 (4.091 / 4.150) | 31.014 (12.554 / 40.672) | **3.051 (2.855 / 3.099)** | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | 3533.908 (3532.375 / 3536.953) | 645.934 (641.033 / 654.557) | 3729.289 (3727.499 / 3730.707) | 619.448 (618.334 / 620.326) | **511.116 (509.097 / 513.241)** | 1955.437 (1943.400 / 1963.382) | 623.481 (567.766 / 639.614) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | 118.982 (115.197 / 120.075) | 48.757 (47.432 / 51.350) | 147.810 (145.659 / 149.885) | **42.600 (42.247 / 43.021)** | 47.099 (43.802 / 47.689) | 127.469 (125.968 / 129.834) | 62.226 (33.635 / 89.939) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | 124.711 (124.537 / 125.027) | - | 151.325 (150.894 / 151.631) | - | **34.100 (33.521 / 34.887)** | 519.645 (467.997 / 529.621) | 76.609 (35.640 / 104.307) | - |
| `transpose_2d_1024` | 2D 1024^2 transpose [1,0] | 6.929 (6.928 / 6.971) | **0.762 (0.742 / 0.782)** | 6.999 (6.998 / 7.001) | 1.981 (1.923 / 1.998) | 1.642 (1.603 / 1.773) | 11.467 (8.939 / 11.973) | 1.448 (1.381 / 3.592) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | 50.222 (50.151 / 50.407) | 16.253 (14.759 / 16.624) | 54.514 (54.355 / 54.883) | **12.086 (11.958 / 12.189)** | 16.426 (16.248 / 16.639) | 53.114 (52.728 / 54.122) | 15.157 (14.259 / 15.401) | - |
| `transpose_2d_256` | 2D 256^2 transpose [1,0] | 0.237 (0.237 / 0.238) | **0.098 (0.097 / 0.101)** | 0.363 (0.333 / 0.383) | 0.107 (0.094 / 0.132) | 0.179 (0.179 / 0.182) | 0.362 (0.343 / 0.385) | 0.423 (0.403 / 0.429) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | 145.202 (145.060 / 145.386) | 52.728 (50.852 / 54.097) | 175.375 (174.904 / 175.983) | **42.744 (42.203 / 43.320)** | 52.043 (51.662 / 52.482) | 122.988 (113.254 / 124.658) | 68.257 (37.963 / 89.748) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 355.207 (354.753 / 356.718) | 81.413 (81.114 / 81.689) | 349.385 (348.216 / 353.662) | **52.219 (51.858 / 52.530)** | 59.403 (58.633 / 59.642) | 373.706 (366.028 / 376.165) | 107.039 (78.802 / 132.584) | - |
