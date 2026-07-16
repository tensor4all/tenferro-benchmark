# CPU Permutation Benchmark Results

- Target profile: `amd-cpu`
- Suite: `cpu/permutation`
- Suite file: `benchmarks/cpu/permutation.yaml`
- Timestamp: `2026-07-16T08:32:51.892115+00:00`
- tenferro-rs commit: `8c267c636537d3d407925b61a2000f9bbe07e6a1`

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

`tenferro-transpose` is the eager `tenferro_cpu::transpose` structural op (compact col-major input only); `tenferro-to-contiguous` is `TypedTensorView::transpose_view` + `to_contiguous()` (accepts arbitrary source strides). Both allocate a fresh destination on every call; every other backend reuses a destination buffer allocated once per pattern. `hptt` only participates in patterns with a contiguous source and destination. Correctness is verified against the `naive` odometer reference before any timing; a `FAILED` cell means that backend's output did not match the reference for that pattern.

## Threads: 1

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | naive odometer (ms) | tenferro-rs transpose (ms) | tenferro-rs to_contiguous (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `cyclic_13d_2` | 13D 2^13 cyclic [1,2,...,0] | 0.026 (0.026 / 0.028) | 0.024 (0.024 / 0.030) | 0.039 (0.038 / 0.040) | - | **0.008 (0.008 / 0.008)** | 0.029 (0.025 / 0.029) | 0.009 (0.008 / 0.009) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | - | - | 12.522 (12.511 / 12.530) | - | - | **12.290 (12.274 / 12.315)** |
| `reverse_13d_2` | 13D 2^13 reverse | 0.029 (0.029 / 0.034) | 0.080 (0.080 / 0.080) | 0.040 (0.040 / 0.045) | - | **0.012 (0.011 / 0.012)** | 0.028 (0.027 / 0.028) | 0.028 (0.027 / 0.028) | - |
| `reverse_20d_2` | 20D 2^20 reverse | 8.551 (8.520 / 8.576) | 11.716 (11.688 / 11.772) | 10.836 (10.835 / 10.842) | 8.601 (8.588 / 8.648) | **3.933 (3.827 / 3.949)** | 8.238 (8.232 / 8.250) | 4.925 (4.919 / 4.933) | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | 2924.278 (2923.108 / 2926.364) | 2132.227 (2130.238 / 2133.426) | 3812.359 (3809.975 / 3814.161) | 1049.185 (1048.475 / 1049.454) | **712.672 (712.492 / 713.225)** | 1134.402 (1134.103 / 1134.946) | 829.075 (828.043 / 829.686) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | 59.647 (59.554 / 59.844) | 73.630 (73.602 / 74.041) | 135.319 (135.056 / 135.823) | 51.404 (51.359 / 51.518) | 31.768 (31.740 / 31.807) | 58.420 (58.337 / 58.553) | **31.642 (31.587 / 31.661)** | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | 61.445 (61.400 / 61.623) | - | 138.810 (138.596 / 139.343) | - | 34.275 (34.225 / 34.290) | 329.313 (328.354 / 329.743) | **29.217 (29.189 / 29.282)** | - |
| `transpose_2d_1024` | 2D 1024^2 transpose [1,0] | 6.561 (6.557 / 6.565) | 2.877 (2.876 / 2.881) | 7.215 (7.204 / 7.268) | 2.863 (2.862 / 2.865) | **1.543 (1.540 / 1.544)** | 7.179 (7.175 / 7.190) | 2.770 (2.768 / 2.772) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | 36.737 (36.572 / 37.167) | 29.058 (28.997 / 29.167) | 53.437 (53.183 / 53.723) | 20.583 (20.542 / 20.673) | **14.212 (14.197 / 14.252)** | 34.416 (34.366 / 34.493) | 16.636 (16.564 / 16.670) | - |
| `transpose_2d_256` | 2D 256^2 transpose [1,0] | 0.235 (0.235 / 0.239) | 0.094 (0.094 / 0.098) | 0.361 (0.361 / 0.364) | 0.123 (0.123 / 0.123) | 0.171 (0.170 / 0.172) | 0.160 (0.160 / 0.171) | **0.086 (0.086 / 0.086)** | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | 90.195 (90.073 / 90.408) | 84.414 (84.222 / 84.887) | 173.382 (173.076 / 173.691) | 61.404 (61.320 / 61.499) | 59.178 (59.130 / 59.233) | 72.951 (72.870 / 72.995) | **35.917 (35.877 / 35.951)** | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 272.436 (271.019 / 272.880) | 219.543 (218.945 / 221.243) | 361.412 (361.061 / 362.053) | 101.776 (101.703 / 102.156) | **81.781 (81.701 / 82.012)** | 308.848 (308.687 / 309.205) | 158.342 (158.217 / 158.525) | - |

## Threads: 4

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | naive odometer (ms) | tenferro-rs transpose (ms) | tenferro-rs to_contiguous (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `cyclic_13d_2` | 13D 2^13 cyclic [1,2,...,0] | 0.079 (0.078 / 0.079) | 0.074 (0.073 / 0.076) | 0.107 (0.106 / 0.113) | - | 0.020 (0.020 / 0.020) | 0.026 (0.025 / 0.027) | **0.008 (0.007 / 0.009)** | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | - | - | 12.520 (12.511 / 12.585) | - | - | **12.417 (12.401 / 12.441)** |
| `reverse_13d_2` | 13D 2^13 reverse | 0.086 (0.085 / 0.086) | 0.244 (0.242 / 0.249) | 0.121 (0.121 / 0.122) | - | 0.034 (0.034 / 0.034) | **0.028 (0.028 / 0.028)** | 0.031 (0.031 / 0.032) | - |
| `reverse_20d_2` | 20D 2^20 reverse | 8.755 (8.749 / 8.786) | 5.079 (4.873 / 5.244) | 11.248 (11.243 / 11.308) | 3.516 (3.502 / 3.592) | 3.893 (3.868 / 3.897) | 7.695 (7.687 / 7.747) | **2.934 (2.921 / 2.957)** | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | 2918.964 (2918.185 / 2920.526) | 703.433 (701.844 / 705.647) | 3804.591 (3802.353 / 3806.926) | 310.513 (309.686 / 311.320) | **255.359 (254.666 / 256.216)** | 1092.516 (1092.146 / 1093.409) | 269.303 (267.309 / 270.253) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | 59.421 (59.354 / 59.488) | 36.717 (33.422 / 40.395) | 136.039 (135.751 / 136.534) | 20.320 (20.195 / 20.518) | **14.717 (14.687 / 14.750)** | 57.313 (57.282 / 57.486) | 14.886 (14.519 / 15.592) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | 61.755 (61.668 / 61.807) | - | 138.779 (138.633 / 139.144) | - | 15.435 (15.398 / 15.480) | 330.949 (329.778 / 331.499) | **14.601 (14.535 / 14.799)** | - |
| `transpose_2d_1024` | 2D 1024^2 transpose [1,0] | 6.559 (6.555 / 6.561) | 1.522 (1.498 / 1.561) | 7.336 (7.325 / 7.397) | **0.959 (0.949 / 0.960)** | 1.113 (1.104 / 1.125) | 7.161 (7.159 / 7.196) | 1.403 (1.384 / 1.419) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | 37.535 (37.447 / 37.771) | 16.058 (15.808 / 16.367) | 56.581 (56.407 / 56.993) | 6.481 (6.450 / 6.532) | **5.269 (5.085 / 5.356)** | 34.718 (34.639 / 34.875) | 5.600 (5.433 / 6.931) | - |
| `transpose_2d_256` | 2D 256^2 transpose [1,0] | 0.238 (0.237 / 0.240) | 0.142 (0.136 / 0.149) | 0.372 (0.369 / 0.392) | **0.052 (0.044 / 0.074)** | 0.110 (0.107 / 0.115) | 0.149 (0.149 / 0.150) | 0.127 (0.127 / 0.130) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | 90.412 (90.332 / 90.567) | 40.499 (36.351 / 41.802) | 175.990 (175.722 / 176.947) | 19.277 (19.261 / 19.337) | 18.882 (18.335 / 20.554) | 71.807 (71.739 / 71.841) | **14.794 (14.715 / 14.814)** | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 337.847 (337.500 / 338.410) | 77.504 (76.113 / 77.921) | 403.162 (402.273 / 404.073) | 29.186 (29.067 / 29.375) | **23.672 (22.873 / 24.277)** | 280.219 (279.948 / 280.871) | 59.010 (58.874 / 59.138) | - |
