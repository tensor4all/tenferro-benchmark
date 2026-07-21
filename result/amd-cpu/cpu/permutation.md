# CPU Permutation Benchmark Results

- Target profile: `amd-cpu`
- Suite: `cpu/permutation`
- Suite file: `benchmarks/cpu/permutation.yaml`
- Timestamp: `2026-07-21T07:31:12.953506+00:00`
- tenferro-rs commit: `85855e272b1495611deb601a9ee06f3546772c3c`

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

`tenferro-transpose` is the eager `CpuBackend::transpose` structural op (compact col-major input only); `tenferro-to-contiguous` is `TypedTensorView::transpose_view` + `CpuBackend::to_contiguous` (accepts arbitrary source strides). Every backend allocates a fresh destination inside each timed call, so the table compares allocation-inclusive end-to-end materialization rather than destination-reuse copy kernels. `hptt` only participates in patterns with a contiguous source and destination. Correctness is verified against the `naive` odometer reference before any timing; a `FAILED` cell means that backend's output did not match the reference for that pattern.

## Threads: 1

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | naive odometer (ms) | tenferro-rs transpose (ms) | tenferro-rs to_contiguous (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `cyclic_13d_2` | 13D 2^13 cyclic [1,2,...,0] | 0.028 (0.028 / 0.042) | 0.044 (0.044 / 0.044) | 0.041 (0.041 / 0.042) | 0.012 (0.012 / 0.012) | **0.010 (0.010 / 0.010)** | 0.029 (0.026 / 0.035) | 0.018 (0.017 / 0.019) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | - | - | **77.621 (77.496 / 78.008)** | - | - | 77.926 (77.628 / 78.241) |
| `reverse_13d_2` | 13D 2^13 reverse | 0.034 (0.033 / 0.040) | 0.137 (0.133 / 0.147) | 0.127 (0.126 / 0.135) | 0.062 (0.062 / 0.073) | **0.015 (0.015 / 0.015)** | 0.030 (0.030 / 0.030) | 0.037 (0.037 / 0.038) | - |
| `reverse_20d_2` | 20D 2^20 reverse | 8.856 (8.787 / 9.115) | 12.604 (12.506 / 12.847) | 12.095 (11.929 / 12.365) | 13.227 (12.720 / 13.240) | **4.378 (4.138 / 4.452)** | 10.229 (8.122 / 11.637) | 4.932 (4.853 / 5.055) | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | 3413.874 (3402.348 / 3419.906) | 2146.896 (2144.491 / 2154.826) | 2063.669 (2060.181 / 2066.350) | 2112.914 (2110.304 / 2122.944) | **1525.423 (1524.565 / 1527.087)** | 1883.793 (1881.947 / 1885.213) | 1683.667 (1681.803 / 1684.146) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | 116.473 (115.736 / 116.865) | 93.627 (90.872 / 94.181) | 94.599 (94.170 / 95.286) | 102.253 (101.822 / 102.570) | **74.875 (74.693 / 75.367)** | 112.705 (112.465 / 113.048) | 77.955 (77.380 / 78.606) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | 122.680 (122.533 / 123.325) | - | 106.648 (103.216 / 107.331) | - | 89.397 (89.228 / 89.866) | 437.756 (436.160 / 454.974) | **82.751 (82.658 / 83.244)** | - |
| `transpose_2d_1024` | 2D 1024^2 transpose [1,0] | 6.915 (6.911 / 7.153) | 2.768 (2.766 / 2.854) | 2.738 (2.728 / 2.832) | 2.830 (2.827 / 2.879) | **2.229 (2.227 / 2.261)** | 10.552 (7.589 / 10.937) | 5.660 (3.696 / 5.789) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | 53.871 (53.660 / 54.088) | 31.977 (31.920 / 32.116) | 32.004 (31.847 / 32.325) | 34.751 (34.638 / 34.923) | **27.663 (27.592 / 27.976)** | 49.777 (49.303 / 50.017) | 28.256 (27.900 / 28.484) | - |
| `transpose_2d_256` | 2D 256^2 transpose [1,0] | 0.254 (0.254 / 0.257) | 0.110 (0.110 / 0.117) | 0.108 (0.107 / 0.114) | 0.172 (0.169 / 0.184) | **0.094 (0.094 / 0.094)** | 0.369 (0.357 / 0.381) | 0.289 (0.285 / 0.321) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | 146.697 (145.993 / 147.241) | 106.819 (103.672 / 107.499) | 106.925 (106.697 / 107.100) | 118.182 (118.059 / 118.312) | 109.802 (109.627 / 110.061) | 112.551 (112.231 / 112.864) | **86.804 (86.517 / 86.935)** | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 378.244 (377.564 / 379.094) | 240.914 (240.093 / 241.933) | 241.571 (241.038 / 241.944) | 155.973 (155.786 / 156.840) | **135.449 (135.149 / 136.109)** | 337.303 (335.796 / 339.675) | 216.540 (216.026 / 216.797) | - |

## Threads: 4

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | naive odometer (ms) | tenferro-rs transpose (ms) | tenferro-rs to_contiguous (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `cyclic_13d_2` | 13D 2^13 cyclic [1,2,...,0] | 0.085 (0.084 / 0.085) | 0.058 (0.058 / 0.059) | 0.056 (0.055 / 0.057) | 0.030 (0.030 / 0.031) | 0.028 (0.028 / 0.028) | 0.030 (0.027 / 0.038) | **0.019 (0.018 / 0.022)** | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | - | - | **76.787 (76.677 / 76.919)** | - | - | 76.896 (76.765 / 77.549) |
| `reverse_13d_2` | 13D 2^13 reverse | 0.093 (0.093 / 0.094) | 0.145 (0.142 / 0.149) | 0.148 (0.142 / 0.158) | 0.091 (0.090 / 0.092) | 0.039 (0.039 / 0.039) | **0.029 (0.029 / 0.030)** | 0.039 (0.039 / 0.040) | - |
| `reverse_20d_2` | 20D 2^20 reverse | 8.905 (8.857 / 9.030) | 3.622 (3.582 / 3.816) | 3.431 (3.411 / 3.631) | 4.303 (4.250 / 4.495) | 4.247 (4.208 / 4.282) | 8.810 (8.396 / 11.927) | **2.952 (2.890 / 3.072)** | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | 3400.129 (3395.617 / 3403.698) | 684.875 (682.087 / 686.038) | 701.476 (697.584 / 705.844) | 692.039 (682.449 / 694.263) | **560.923 (558.204 / 568.018)** | 1909.860 (1907.508 / 1911.241) | 655.163 (468.800 / 741.892) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | 118.961 (118.782 / 119.868) | **39.327 (37.166 / 43.023)** | 45.774 (43.303 / 47.502) | 46.425 (40.984 / 47.040) | 43.142 (41.909 / 45.902) | 121.390 (120.841 / 121.926) | 45.958 (33.297 / 77.722) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | 119.750 (119.447 / 119.858) | - | **43.184 (40.864 / 46.130)** | - | 45.553 (43.726 / 48.687) | 436.364 (419.781 / 436.702) | 76.762 (37.376 / 81.999) | - |
| `transpose_2d_1024` | 2D 1024^2 transpose [1,0] | 6.942 (6.915 / 7.144) | 0.965 (0.956 / 0.977) | **0.803 (0.800 / 0.807)** | 2.049 (1.938 / 2.191) | 1.372 (1.338 / 1.410) | 8.971 (7.333 / 10.992) | 2.171 (1.634 / 3.666) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | 53.150 (52.926 / 53.480) | 14.931 (13.307 / 15.190) | 16.632 (16.406 / 16.828) | 14.030 (13.830 / 14.128) | **12.000 (11.873 / 12.263)** | 50.154 (50.000 / 50.294) | 14.115 (11.749 / 15.305) | - |
| `transpose_2d_256` | 2D 256^2 transpose [1,0] | 0.257 (0.257 / 0.260) | 0.089 (0.087 / 0.092) | **0.088 (0.086 / 0.090)** | 0.094 (0.093 / 0.102) | 0.177 (0.172 / 0.188) | 0.339 (0.337 / 0.344) | 0.145 (0.144 / 0.150) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | 145.962 (145.680 / 146.173) | **42.651 (40.045 / 44.825)** | 48.286 (46.977 / 49.386) | 50.476 (42.260 / 50.761) | 48.644 (43.759 / 52.372) | 115.707 (113.849 / 117.064) | 56.831 (35.342 / 79.642) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 394.403 (393.892 / 395.920) | 81.605 (78.452 / 82.056) | 82.001 (81.685 / 82.496) | 53.772 (53.642 / 54.114) | **46.442 (46.024 / 46.792)** | 319.259 (318.731 / 320.358) | 104.511 (78.346 / 139.986) | - |
