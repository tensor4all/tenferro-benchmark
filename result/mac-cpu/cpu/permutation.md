# CPU Permutation Benchmark Results

- Target profile: `mac-cpu`
- Suite: `cpu/permutation`
- Suite file: `benchmarks/cpu/permutation.yaml`
- Timestamp: `2026-07-16T09:04:21.718224+00:00`
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

`tenferro-transpose` is the eager `CpuBackend::transpose` structural op (compact col-major input only); `tenferro-to-contiguous` is `TypedTensorView::transpose_view` + `to_contiguous()` (accepts arbitrary source strides). Every backend allocates a fresh destination inside each timed call, so the table compares allocation-inclusive end-to-end materialization rather than destination-reuse copy kernels. `hptt` only participates in patterns with a contiguous source and destination. Correctness is verified against the `naive` odometer reference before any timing; a `FAILED` cell means that backend's output did not match the reference for that pattern.

## Threads: 1

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | naive odometer (ms) | tenferro-rs transpose (ms) | tenferro-rs to_contiguous (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `cyclic_13d_2` | 13D 2^13 cyclic [1,2,...,0] | 0.011 (0.011 / 0.011) | 0.009 (0.009 / 0.010) | 0.019 (0.019 / 0.019) | - | **0.003 (0.003 / 0.003)** | 0.011 (0.011 / 0.011) | 0.009 (0.008 / 0.010) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | - | - | **4.378 (4.186 / 4.494)** | - | - | 4.800 (4.414 / 4.935) |
| `reverse_13d_2` | 13D 2^13 reverse | 0.011 (0.011 / 0.011) | 0.032 (0.032 / 0.034) | 0.020 (0.020 / 0.020) | - | **0.006 (0.006 / 0.006)** | 0.009 (0.009 / 0.009) | 0.009 (0.009 / 0.009) | - |
| `reverse_20d_2` | 20D 2^20 reverse | 1.670 (1.647 / 1.693) | 4.439 (4.331 / 4.718) | 2.837 (2.594 / 3.448) | 2.351 (2.279 / 2.440) | **0.880 (0.808 / 1.100)** | 1.638 (1.331 / 1.703) | 1.414 (1.379 / 1.430) | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | 512.620 (511.860 / 515.624) | 343.441 (342.007 / 392.473) | 672.125 (668.182 / 676.174) | 223.122 (222.782 / 223.334) | 246.351 (245.952 / 246.776) | 241.946 (241.285 / 254.426) | **207.868 (205.899 / 208.051)** | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | 21.290 (21.218 / 21.366) | 6.312 (6.155 / 6.487) | 38.171 (38.069 / 38.315) | 7.015 (6.892 / 7.050) | **5.538 (5.381 / 5.638)** | 17.979 (17.859 / 18.111) | 6.628 (6.529 / 6.708) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | 24.939 (24.537 / 26.896) | - | 40.001 (39.613 / 40.341) | - | **6.959 (6.943 / 7.057)** | 130.086 (129.451 / 130.867) | 9.181 (8.901 / 10.648) | - |
| `transpose_2d_1024` | 2D 1024^2 transpose [1,0] | 2.016 (1.995 / 2.218) | 1.178 (1.073 / 1.352) | 2.273 (2.231 / 2.498) | **0.518 (0.471 / 0.910)** | 0.532 (0.525 / 0.553) | 0.958 (0.944 / 1.014) | 1.095 (1.015 / 1.161) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | 11.745 (11.560 / 12.177) | 5.431 (5.340 / 5.554) | 13.782 (13.302 / 14.036) | **4.930 (4.866 / 5.121)** | 6.160 (5.957 / 6.456) | 10.252 (9.824 / 10.464) | 5.359 (5.266 / 5.455) | - |
| `transpose_2d_256` | 2D 256^2 transpose [1,0] | 0.125 (0.056 / 0.131) | 0.035 (0.034 / 0.037) | 0.267 (0.157 / 0.323) | 0.023 (0.023 / 0.024) | **0.018 (0.018 / 0.018)** | 0.062 (0.059 / 0.065) | 0.041 (0.040 / 0.042) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | 32.160 (31.928 / 32.246) | 13.607 (13.507 / 13.731) | 35.771 (35.726 / 35.869) | **12.300 (12.152 / 12.373)** | 13.243 (13.156 / 13.300) | 13.425 (13.266 / 13.486) | 12.424 (12.259 / 12.465) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 41.883 (41.326 / 43.011) | 23.294 (22.986 / 23.617) | 45.167 (44.458 / 47.080) | **14.208 (14.077 / 14.329)** | 14.363 (14.303 / 14.479) | 53.760 (48.420 / 66.022) | 20.198 (19.594 / 20.927) | - |

## Threads: 4

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | naive odometer (ms) | tenferro-rs transpose (ms) | tenferro-rs to_contiguous (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `cyclic_13d_2` | 13D 2^13 cyclic [1,2,...,0] | 0.012 (0.012 / 0.012) | 0.050 (0.039 / 0.064) | 0.021 (0.021 / 0.022) | - | **0.004 (0.004 / 0.004)** | 0.009 (0.009 / 0.011) | 0.008 (0.008 / 0.009) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | - | - | **4.291 (4.173 / 4.385)** | - | - | 4.301 (4.142 / 4.397) |
| `reverse_13d_2` | 13D 2^13 reverse | 0.012 (0.012 / 0.012) | 0.071 (0.066 / 0.077) | 0.021 (0.021 / 0.021) | - | **0.007 (0.007 / 0.007)** | 0.009 (0.009 / 0.009) | 0.009 (0.009 / 0.009) | - |
| `reverse_20d_2` | 20D 2^20 reverse | 1.721 (1.675 / 1.758) | 2.625 (2.518 / 2.745) | 2.733 (2.645 / 2.833) | 0.908 (0.881 / 1.059) | **0.477 (0.464 / 0.503)** | 1.585 (1.551 / 1.706) | 1.134 (0.893 / 1.493) | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | 515.423 (512.447 / 519.695) | 143.451 (142.247 / 144.682) | 668.888 (666.077 / 671.847) | 120.557 (119.194 / 121.449) | 133.778 (133.034 / 134.880) | 242.429 (241.355 / 243.402) | **112.123 (102.561 / 115.601)** | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | 22.920 (22.454 / 23.407) | **4.514 (4.369 / 4.573)** | 37.434 (37.285 / 39.088) | 5.505 (5.462 / 5.632) | 5.262 (5.225 / 5.376) | 18.028 (17.935 / 18.121) | 4.676 (4.172 / 11.711) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | 25.781 (25.495 / 26.626) | - | 38.696 (38.361 / 39.209) | - | **5.466 (5.338 / 5.595)** | 133.416 (131.042 / 136.368) | 13.142 (4.884 / 14.200) | - |
| `transpose_2d_1024` | 2D 1024^2 transpose [1,0] | 1.974 (1.928 / 1.996) | 0.510 (0.492 / 0.540) | 2.239 (2.217 / 2.329) | 0.212 (0.187 / 0.232) | **0.201 (0.193 / 0.215)** | 0.963 (0.946 / 0.977) | 0.545 (0.498 / 0.602) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | 11.428 (11.290 / 11.734) | 2.799 (2.703 / 2.923) | 12.970 (12.811 / 13.202) | 2.108 (2.086 / 2.147) | **1.845 (1.829 / 1.936)** | 9.621 (9.474 / 9.962) | 2.611 (2.474 / 4.150) | - |
| `transpose_2d_256` | 2D 256^2 transpose [1,0] | 0.076 (0.054 / 0.124) | 0.077 (0.059 / 0.098) | 0.132 (0.131 / 0.132) | 0.028 (0.028 / 0.031) | **0.018 (0.018 / 0.019)** | 0.059 (0.040 / 0.062) | 0.055 (0.046 / 0.075) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | 32.155 (31.969 / 32.414) | **5.681 (5.366 / 5.883)** | 35.796 (35.765 / 35.935) | 5.874 (5.804 / 5.991) | 6.195 (6.115 / 6.314) | 13.396 (13.210 / 13.470) | 9.102 (4.891 / 9.492) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 43.907 (43.396 / 44.845) | 10.045 (9.204 / 10.859) | 45.334 (45.052 / 45.975) | **7.883 (7.635 / 8.016)** | 10.045 (9.883 / 10.111) | 87.435 (68.698 / 100.660) | 15.226 (13.097 / 22.922) | - |
