# CPU Permutation Benchmark Results

- Target profile: `mac-cpu`
- Suite: `cpu/permutation`
- Suite file: `benchmarks/cpu/permutation.yaml`
- Timestamp: `2026-07-16T02:08:54.528282+00:00`
- tenferro-rs commit: `50c6623dd57f41e3caacf98c5f3b94cf91561e1b`

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

`tenferro-transpose` is the eager `tenferro_cpu::transpose` structural op (compact col-major input only); `tenferro-to-contiguous` is `TypedTensorView::transpose_view` + `to_contiguous()` (accepts arbitrary source strides). Both allocate a fresh destination on every call; every other backend reuses a destination buffer allocated once per pattern. `hptt` only participates in patterns with a contiguous source and destination. Correctness is verified against the `naive` odometer reference before any timing; a `FAILED` cell means that backend's output did not match the reference for that pattern.

## Threads: 1

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | naive odometer (ms) | tenferro-rs transpose (ms) | tenferro-rs to_contiguous (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `cyclic_13d_2` | 13D 2^13 cyclic [1,2,...,0] | 0.010 (0.010 / 0.010) | 0.010 (0.010 / 0.010) | 0.018 (0.017 / 0.018) | - | **0.002 (0.002 / 0.002)** | 0.008 (0.008 / 0.008) | 0.003 (0.003 / 0.004) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | - | - | **2.892 (2.869 / 2.958)** | - | - | 3.032 (2.878 / 3.375) |
| `reverse_13d_2` | 13D 2^13 reverse | 0.010 (0.010 / 0.010) | 0.032 (0.032 / 0.032) | 0.019 (0.018 / 0.019) | - | **0.005 (0.005 / 0.005)** | 0.009 (0.009 / 0.009) | 0.008 (0.008 / 0.008) | - |
| `reverse_20d_2` | 20D 2^20 reverse | 1.635 (1.626 / 1.670) | 4.090 (4.052 / 4.127) | 2.525 (2.513 / 2.546) | 2.120 (2.103 / 2.137) | **0.729 (0.727 / 0.735)** | 1.250 (1.238 / 1.268) | 1.395 (1.385 / 1.402) | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | 487.631 (483.846 / 541.604) | 337.615 (335.396 / 348.598) | 652.807 (651.620 / 657.274) | **202.195 (199.895 / 216.094)** | 218.989 (218.138 / 220.177) | 241.774 (239.454 / 243.093) | 206.012 (204.641 / 207.868) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | 19.845 (19.806 / 19.911) | 6.072 (6.053 / 6.126) | 36.247 (36.182 / 36.517) | 5.611 (5.603 / 5.643) | **4.262 (4.258 / 4.271)** | 17.369 (17.329 / 17.412) | 6.552 (6.463 / 6.648) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | 22.878 (22.605 / 22.926) | - | 37.807 (37.192 / 38.089) | - | **5.725 (5.699 / 5.746)** | 128.508 (127.672 / 128.930) | 8.300 (8.267 / 8.376) | - |
| `transpose_2d_1024` | 2D 1024^2 transpose [1,0] | 1.911 (1.884 / 1.939) | 0.968 (0.941 / 1.008) | 2.216 (2.211 / 2.239) | **0.458 (0.447 / 0.491)** | 0.489 (0.484 / 0.504) | 0.849 (0.847 / 0.855) | 0.846 (0.838 / 0.853) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | 9.858 (9.594 / 10.029) | 5.296 (5.213 / 5.432) | 11.934 (11.511 / 13.162) | **4.389 (4.363 / 4.454)** | 5.228 (5.196 / 5.288) | 8.905 (8.865 / 8.948) | 4.986 (4.969 / 5.003) | - |
| `transpose_2d_256` | 2D 256^2 transpose [1,0] | 0.052 (0.051 / 0.089) | 0.033 (0.033 / 0.034) | 0.132 (0.132 / 0.133) | 0.020 (0.019 / 0.020) | **0.015 (0.015 / 0.015)** | 0.040 (0.040 / 0.040) | 0.027 (0.025 / 0.027) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | 30.799 (30.335 / 31.123) | 13.193 (13.168 / 13.401) | 35.470 (35.422 / 35.570) | **11.210 (10.942 / 11.388)** | 11.257 (11.238 / 11.316) | 12.871 (12.853 / 12.888) | 12.126 (12.003 / 12.210) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 53.549 (44.245 / 56.770) | 21.409 (20.682 / 22.309) | 44.446 (44.041 / 44.586) | **12.261 (12.129 / 12.547)** | 12.361 (12.298 / 12.548) | 118.901 (113.022 / 120.250) | 18.882 (18.719 / 18.955) | - |

## Threads: 4

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | naive odometer (ms) | tenferro-rs transpose (ms) | tenferro-rs to_contiguous (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `cyclic_13d_2` | 13D 2^13 cyclic [1,2,...,0] | 0.010 (0.010 / 0.010) | 0.010 (0.010 / 0.010) | 0.020 (0.020 / 0.021) | - | **0.002 (0.002 / 0.002)** | 0.009 (0.009 / 0.009) | 0.004 (0.003 / 0.004) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | - | - | **2.852 (2.842 / 2.871)** | - | - | 2.854 (2.844 / 2.868) |
| `reverse_13d_2` | 13D 2^13 reverse | 0.011 (0.010 / 0.011) | 0.035 (0.035 / 0.035) | 0.020 (0.020 / 0.021) | - | **0.005 (0.005 / 0.005)** | 0.008 (0.008 / 0.008) | 0.008 (0.008 / 0.008) | - |
| `reverse_20d_2` | 20D 2^20 reverse | 1.930 (1.804 / 2.086) | 2.541 (2.420 / 2.668) | 2.923 (2.698 / 3.209) | 0.862 (0.843 / 0.882) | 0.436 (0.432 / 0.470) | 1.250 (1.239 / 1.270) | **0.374 (0.367 / 0.407)** | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | 488.199 (484.422 / 490.359) | 141.823 (141.006 / 144.084) | 661.485 (658.609 / 663.042) | 98.376 (98.208 / 99.146) | 115.124 (114.969 / 115.527) | 242.906 (240.961 / 243.604) | **98.356 (97.799 / 98.852)** | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | 21.352 (21.314 / 21.494) | 4.154 (4.147 / 4.205) | 38.096 (37.903 / 39.383) | 4.185 (4.141 / 4.531) | **4.105 (4.052 / 4.122)** | 17.464 (17.380 / 17.501) | 4.162 (4.120 / 4.200) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | 24.627 (23.034 / 25.076) | - | 39.154 (38.873 / 39.580) | - | **4.178 (4.159 / 4.208)** | 128.806 (127.832 / 129.236) | 4.435 (4.358 / 4.508) | - |
| `transpose_2d_1024` | 2D 1024^2 transpose [1,0] | 1.899 (1.855 / 1.939) | 0.500 (0.484 / 0.526) | 2.222 (2.213 / 2.248) | 0.177 (0.160 / 0.198) | **0.163 (0.154 / 0.192)** | 0.848 (0.845 / 0.860) | 0.441 (0.413 / 0.452) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | 9.423 (9.382 / 9.525) | 2.792 (2.713 / 2.862) | 10.677 (10.587 / 10.886) | 1.807 (1.778 / 1.873) | **1.696 (1.643 / 1.764)** | 9.049 (8.985 / 9.116) | 2.686 (2.573 / 2.751) | - |
| `transpose_2d_256` | 2D 256^2 transpose [1,0] | 0.117 (0.053 / 0.119) | 0.032 (0.029 / 0.042) | 0.134 (0.132 / 0.134) | 0.018 (0.018 / 0.019) | **0.010 (0.010 / 0.013)** | 0.040 (0.040 / 0.040) | 0.044 (0.015 / 0.044) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | 30.568 (29.966 / 30.905) | 5.353 (5.297 / 5.415) | 35.294 (35.266 / 35.358) | **4.564 (4.535 / 4.579)** | 4.943 (4.909 / 4.980) | 13.004 (12.920 / 13.098) | 5.442 (4.939 / 5.717) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 97.005 (81.187 / 102.046) | 8.360 (8.225 / 8.467) | 43.409 (43.131 / 44.107) | **6.452 (6.401 / 6.474)** | 8.678 (8.610 / 8.731) | 48.891 (43.081 / 52.052) | 10.182 (8.934 / 11.740) | - |
