# CPU Permutation Benchmark Results

- Target profile: `mac-cpu`
- Suite: `cpu/permutation`
- Suite file: `benchmarks/cpu/permutation.yaml`
- Timestamp: `2026-07-23T03:26:23.496824+00:00`
- tenferro-rs commit: `68855c2b65b5adc42dccca9bac04fd136a8f14c8`

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

`tenferro-rs` measures `TypedTensorView::transpose_view` followed by `CpuBackend::to_contiguous`; the metadata-only `transpose_view` is built outside the timed region, and `to_contiguous` accepts arbitrary source strides. Every backend allocates a fresh destination inside each timed call, so the table compares allocation-inclusive end-to-end materialization rather than destination-reuse copy kernels. `hptt` only participates in patterns with a contiguous source and destination. Correctness is verified against an internal, untimed odometer reference before any timing; a `FAILED` cell means that backend's output did not match the reference for that pattern.

## Threads: 1

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `cyclic_13d_2` | 13D 2^13 cyclic [1,2,...,0] | 0.024 (0.024 / 0.024) | 0.004 (0.004 / 0.004) | **0.003 (0.003 / 0.003)** | 0.011 (0.011 / 0.011) | 0.008 (0.008 / 0.009) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **4.535 (4.381 / 4.686)** | - | - | 4.692 (4.320 / 5.127) |
| `reverse_13d_2` | 13D 2^13 reverse | 0.033 (0.031 / 0.033) | 0.020 (0.020 / 0.020) | **0.006 (0.006 / 0.006)** | 0.009 (0.009 / 0.009) | 0.010 (0.010 / 0.010) | - |
| `reverse_20d_2` | 20D 2^20 reverse | 4.316 (4.256 / 4.421) | 2.179 (2.167 / 2.202) | **0.770 (0.757 / 0.793)** | 1.566 (1.520 / 1.760) | 1.405 (1.384 / 1.431) | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | 341.527 (339.925 / 346.466) | 220.236 (219.181 / 223.863) | 241.181 (240.790 / 242.198) | 242.405 (240.680 / 245.282) | **206.410 (205.740 / 209.090)** | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | 6.049 (6.005 / 6.113) | 6.850 (6.728 / 6.942) | **5.420 (5.364 / 5.532)** | 17.853 (17.732 / 17.936) | 6.811 (6.618 / 6.931) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | 8.011 (7.971 / 8.068) | - | **6.861 (6.793 / 6.955)** | 132.032 (129.142 / 134.614) | 8.798 (8.537 / 8.857) | - |
| `transpose_2d_1024` | 2D 1024^2 transpose [1,0] | 1.254 (1.112 / 1.442) | **0.509 (0.468 / 0.590)** | 0.585 (0.538 / 0.739) | 0.941 (0.936 / 0.958) | 1.075 (0.995 / 1.105) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | 5.392 (5.323 / 5.534) | **4.793 (4.774 / 4.837)** | 6.091 (5.995 / 6.186) | 9.440 (9.273 / 9.637) | 5.207 (5.156 / 5.280) | - |
| `transpose_2d_256` | 2D 256^2 transpose [1,0] | 0.037 (0.035 / 0.041) | 0.024 (0.023 / 0.024) | **0.018 (0.018 / 0.018)** | 0.063 (0.059 / 0.065) | 0.043 (0.042 / 0.043) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | 13.883 (13.851 / 14.038) | **12.147 (10.823 / 12.348)** | 13.058 (13.042 / 13.143) | 13.183 (13.138 / 13.233) | 12.293 (12.075 / 12.315) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 23.544 (23.079 / 24.048) | **13.790 (13.747 / 13.848)** | 14.020 (13.949 / 14.054) | 149.483 (136.995 / 163.896) | 19.519 (19.349 / 20.789) | - |

## Threads: 4

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `cyclic_13d_2` | 13D 2^13 cyclic [1,2,...,0] | 0.015 (0.014 / 0.015) | 0.022 (0.021 / 0.024) | **0.003 (0.003 / 0.003)** | 0.011 (0.011 / 0.013) | 0.008 (0.008 / 0.009) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **4.138 (4.099 / 4.199)** | - | - | 4.239 (4.132 / 4.476) |
| `reverse_13d_2` | 13D 2^13 reverse | 0.040 (0.039 / 0.041) | 0.036 (0.032 / 0.042) | **0.006 (0.006 / 0.006)** | 0.009 (0.009 / 0.009) | 0.010 (0.009 / 0.010) | - |
| `reverse_20d_2` | 20D 2^20 reverse | 2.653 (2.527 / 2.897) | 1.051 (1.040 / 1.217) | **0.439 (0.432 / 0.462)** | 1.781 (1.623 / 1.943) | 0.560 (0.498 / 0.670) | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | 147.566 (142.079 / 149.393) | 125.752 (119.088 / 129.267) | 132.827 (131.967 / 133.782) | 243.096 (241.706 / 246.864) | **111.847 (105.916 / 131.191)** | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | **4.648 (4.456 / 4.903)** | 5.717 (5.370 / 5.960) | 5.238 (5.192 / 5.401) | 17.828 (17.735 / 17.957) | 4.697 (4.136 / 11.807) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | **5.005 (4.968 / 5.179)** | - | 5.413 (5.398 / 5.534) | 130.235 (128.932 / 135.662) | 12.043 (4.561 / 12.611) | - |
| `transpose_2d_1024` | 2D 1024^2 transpose [1,0] | 0.534 (0.500 / 0.588) | 0.221 (0.212 / 0.268) | **0.206 (0.199 / 0.223)** | 0.998 (0.965 / 1.050) | 0.542 (0.502 / 0.628) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | 2.643 (2.474 / 2.725) | 2.111 (2.081 / 2.157) | **1.848 (1.820 / 1.894)** | 9.773 (9.600 / 9.999) | 2.573 (2.429 / 2.710) | - |
| `transpose_2d_256` | 2D 256^2 transpose [1,0] | 0.030 (0.029 / 0.031) | 0.030 (0.029 / 0.032) | **0.018 (0.018 / 0.018)** | 0.059 (0.041 / 0.062) | 0.070 (0.051 / 0.075) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | 6.826 (6.688 / 7.205) | 6.638 (6.403 / 7.265) | 6.133 (6.100 / 6.231) | 13.359 (13.240 / 13.406) | **5.707 (4.996 / 9.844)** | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 9.369 (8.869 / 9.598) | **7.802 (7.626 / 7.952)** | 11.197 (10.768 / 11.547) | 209.837 (190.060 / 213.721) | 18.037 (15.032 / 23.501) | - |
