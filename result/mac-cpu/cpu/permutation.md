# CPU Permutation Benchmark Results

- Target profile: `mac-cpu`
- Suite: `cpu/permutation`
- Suite file: `benchmarks/cpu/permutation.yaml`
- Timestamp: `2026-07-16T08:08:09.080614+00:00`
- tenferro-rs commit: `3abc9108e4f0500f4e75519711cdb4e21b9625df`

## CPU Information

- Model: `Apple M4`
- Vendor: `Apple`
- Logical CPUs: `10`
- Physical CPUs: `10`
- Sockets: `1`
- Cores per socket: `10`
- Threads per core: `1`
- NUMA nodes: `1`
- Performance levels: `Performance: 4 physical / 4 logical (L1i 192 KiB, L1d 128 KiB, L2 16 MiB, 4 CPUs/L2); Efficiency: 6 physical / 6 logical (L1i 128 KiB, L1d 64 KiB, L2 4 MiB, 6 CPUs/L2)`
- Python platform: `macOS-26.5.2-arm64-arm-64bit`

CPU pinning is unavailable on macOS; thread counts are controlled only via `RAYON_NUM_THREADS` / `OMP_NUM_THREADS` / `JULIA_NUM_THREADS`.

`tenferro-transpose` is the eager `CpuBackend::transpose` structural op (compact col-major input only); `tenferro-to-contiguous` is `TypedTensorView::transpose_view` + `to_contiguous()` (accepts arbitrary source strides). Both allocate a fresh destination on every call; every other backend reuses a destination buffer allocated once per pattern. `hptt` only participates in patterns with a contiguous source and destination. Correctness is verified against the `naive` odometer reference before any timing; a `FAILED` cell means that backend's output did not match the reference for that pattern.

## Threads: 1

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | naive odometer (ms) | tenferro-rs transpose (ms) | tenferro-rs to_contiguous (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `cyclic_13d_2` | 13D 2^13 cyclic [1,2,...,0] | 0.010 (0.010 / 0.010) | 0.010 (0.010 / 0.010) | 0.017 (0.017 / 0.018) | - | **0.002 (0.002 / 0.002)** | 0.009 (0.009 / 0.010) | 0.004 (0.003 / 0.004) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | - | - | **3.028 (2.955 / 3.272)** | - | - | 3.694 (3.529 / 4.168) |
| `reverse_13d_2` | 13D 2^13 reverse | 0.010 (0.010 / 0.010) | 0.033 (0.032 / 0.036) | 0.018 (0.018 / 0.019) | - | **0.005 (0.005 / 0.005)** | 0.009 (0.009 / 0.011) | 0.009 (0.008 / 0.012) | - |
| `reverse_20d_2` | 20D 2^20 reverse | 1.638 (1.618 / 1.691) | 4.323 (4.232 / 4.489) | 2.432 (2.396 / 2.520) | 2.131 (2.117 / 2.169) | **0.796 (0.780 / 0.843)** | 1.250 (1.242 / 1.260) | 1.385 (1.373 / 1.400) | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | 495.701 (494.837 / 514.946) | 342.118 (341.107 / 344.665) | 671.052 (664.584 / 705.514) | **208.876 (204.429 / 224.648)** | 226.857 (226.177 / 228.860) | 243.247 (241.504 / 245.639) | 223.543 (214.031 / 232.442) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | 20.140 (20.041 / 20.296) | 6.298 (6.136 / 9.956) | 36.744 (36.468 / 37.028) | 5.849 (5.673 / 6.036) | **4.401 (4.354 / 4.600)** | 17.506 (17.485 / 17.561) | 6.453 (6.251 / 6.788) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | 23.072 (22.801 / 23.362) | - | 38.111 (37.773 / 38.344) | - | **5.791 (5.740 / 5.857)** | 130.479 (129.728 / 130.572) | 8.789 (8.632 / 8.850) | - |
| `transpose_2d_1024` | 2D 1024^2 transpose [1,0] | 1.991 (1.961 / 2.074) | 0.970 (0.959 / 0.999) | 2.227 (2.215 / 2.300) | **0.448 (0.434 / 0.460)** | 0.512 (0.502 / 0.530) | 0.987 (0.921 / 1.090) | 1.009 (0.947 / 1.150) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | 11.159 (10.877 / 11.406) | 5.370 (5.315 / 5.433) | 12.684 (11.967 / 13.363) | **4.589 (4.486 / 4.660)** | 5.705 (5.591 / 5.996) | 12.229 (11.179 / 14.266) | 5.618 (5.257 / 6.679) | - |
| `transpose_2d_256` | 2D 256^2 transpose [1,0] | 0.099 (0.051 / 0.119) | 0.035 (0.034 / 0.035) | 0.133 (0.132 / 0.138) | 0.019 (0.019 / 0.019) | **0.015 (0.015 / 0.015)** | 0.039 (0.039 / 0.040) | 0.028 (0.026 / 0.087) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | 30.896 (30.635 / 31.143) | 13.711 (13.656 / 13.734) | 35.915 (35.846 / 35.941) | **10.983 (10.335 / 11.039)** | 12.024 (11.926 / 12.110) | 13.247 (13.153 / 13.297) | 12.367 (12.192 / 12.406) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 98.063 (95.100 / 102.177) | 20.896 (20.775 / 21.088) | 44.304 (44.238 / 44.721) | 13.469 (13.209 / 13.611) | **13.054 (12.990 / 13.145)** | 94.190 (82.014 / 98.231) | 19.923 (19.408 / 20.151) | - |

## Threads: 4

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | naive odometer (ms) | tenferro-rs transpose (ms) | tenferro-rs to_contiguous (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `cyclic_13d_2` | 13D 2^13 cyclic [1,2,...,0] | 0.011 (0.011 / 0.011) | 0.046 (0.036 / 0.067) | 0.021 (0.021 / 0.021) | - | **0.003 (0.003 / 0.003)** | 0.008 (0.008 / 0.008) | 0.003 (0.003 / 0.004) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | - | - | **2.994 (2.954 / 3.128)** | - | - | 3.076 (2.997 / 3.165) |
| `reverse_13d_2` | 13D 2^13 reverse | 0.011 (0.011 / 0.011) | 0.066 (0.063 / 0.073) | 0.021 (0.021 / 0.021) | - | **0.005 (0.005 / 0.005)** | 0.008 (0.008 / 0.008) | 0.007 (0.007 / 0.008) | - |
| `reverse_20d_2` | 20D 2^20 reverse | 2.939 (2.595 / 4.176) | 3.536 (3.397 / 3.831) | 3.241 (3.034 / 3.488) | 0.871 (0.861 / 0.908) | 0.459 (0.445 / 0.485) | 1.249 (1.242 / 1.257) | **0.400 (0.386 / 0.443)** | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | 494.031 (493.033 / 501.403) | 150.137 (148.381 / 151.549) | 664.911 (660.752 / 671.990) | 103.576 (98.854 / 107.207) | 118.448 (115.999 / 125.168) | 255.306 (248.477 / 260.112) | **103.099 (98.390 / 103.765)** | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | 23.420 (22.396 / 24.523) | 4.506 (4.444 / 4.704) | 42.398 (41.185 / 43.540) | 4.702 (4.589 / 4.836) | **4.120 (3.988 / 4.261)** | 17.728 (17.605 / 17.811) | 4.263 (4.201 / 4.391) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | 30.266 (29.581 / 33.663) | - | 42.381 (41.788 / 43.190) | - | **4.389 (4.321 / 4.507)** | 132.324 (130.831 / 135.994) | 4.521 (4.461 / 4.713) | - |
| `transpose_2d_1024` | 2D 1024^2 transpose [1,0] | 1.944 (1.923 / 1.984) | 0.534 (0.512 / 0.577) | 2.233 (2.212 / 2.297) | 0.214 (0.195 / 0.248) | **0.159 (0.153 / 0.172)** | 0.854 (0.847 / 0.874) | 0.594 (0.529 / 1.722) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | 11.377 (11.127 / 11.583) | 2.775 (2.540 / 2.936) | 13.798 (13.509 / 13.963) | 1.920 (1.891 / 1.959) | **1.622 (1.607 / 1.697)** | 9.857 (9.627 / 10.124) | 2.512 (2.460 / 2.596) | - |
| `transpose_2d_256` | 2D 256^2 transpose [1,0] | 0.059 (0.051 / 0.119) | 0.082 (0.058 / 0.091) | 0.132 (0.132 / 0.132) | 0.018 (0.018 / 0.019) | **0.011 (0.011 / 0.015)** | 0.039 (0.039 / 0.039) | 0.044 (0.042 / 0.045) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | 31.494 (31.100 / 33.558) | 5.620 (5.447 / 5.880) | 35.861 (35.790 / 35.978) | **4.950 (4.693 / 5.100)** | 5.018 (4.972 / 5.322) | 13.214 (13.102 / 13.252) | 5.127 (4.923 / 5.355) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 44.501 (43.565 / 45.498) | 9.092 (8.841 / 9.601) | 45.900 (45.796 / 46.260) | **6.608 (6.497 / 7.058)** | 9.296 (8.986 / 9.799) | 99.680 (95.323 / 106.460) | 12.694 (10.570 / 13.956) | - |
