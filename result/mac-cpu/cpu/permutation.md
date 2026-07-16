# CPU Permutation Benchmark Results

- Target profile: `mac-cpu`
- Suite: `cpu/permutation`
- Suite file: `benchmarks/cpu/permutation.yaml`
- Timestamp: `2026-07-16T09:28:58.590645+00:00`
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
| `cyclic_13d_2` | 13D 2^13 cyclic [1,2,...,0] | 0.011 (0.011 / 0.011) | 0.010 (0.010 / 0.010) | 0.019 (0.019 / 0.020) | 0.005 (0.005 / 0.005) | **0.004 (0.004 / 0.004)** | 0.009 (0.009 / 0.011) | 0.008 (0.008 / 0.009) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | - | - | **4.209 (4.127 / 4.299)** | - | - | 4.401 (4.221 / 4.678) |
| `reverse_13d_2` | 13D 2^13 reverse | 0.011 (0.011 / 0.011) | 0.032 (0.032 / 0.034) | 0.019 (0.019 / 0.019) | 0.020 (0.020 / 0.020) | **0.006 (0.006 / 0.006)** | 0.009 (0.009 / 0.009) | 0.009 (0.009 / 0.009) | - |
| `reverse_20d_2` | 20D 2^20 reverse | 1.702 (1.669 / 1.747) | 4.665 (4.543 / 4.967) | 4.689 (4.651 / 4.956) | 2.241 (2.203 / 2.328) | **0.806 (0.784 / 0.880)** | 1.880 (1.587 / 1.941) | 1.429 (1.408 / 1.460) | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | 513.341 (511.371 / 517.161) | 336.746 (335.717 / 337.561) | 665.315 (663.159 / 713.472) | 222.847 (222.489 / 224.438) | 243.732 (242.822 / 258.044) | 241.345 (239.350 / 242.510) | **219.784 (206.358 / 250.427)** | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | 21.767 (21.404 / 22.569) | 6.234 (6.122 / 6.423) | 38.263 (37.903 / 38.398) | 7.055 (7.012 / 7.185) | **5.517 (5.470 / 5.668)** | 17.937 (17.860 / 18.060) | 6.669 (6.374 / 6.860) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | 25.295 (24.896 / 26.370) | - | 40.102 (39.442 / 40.670) | - | **7.051 (7.018 / 7.164)** | 130.339 (128.749 / 131.077) | 8.904 (8.729 / 8.954) | - |
| `transpose_2d_1024` | 2D 1024^2 transpose [1,0] | 1.878 (1.861 / 1.938) | 0.932 (0.919 / 0.968) | 2.230 (2.221 / 2.319) | **0.470 (0.454 / 0.478)** | 0.496 (0.491 / 0.510) | 0.947 (0.929 / 0.971) | 1.095 (1.004 / 1.168) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | 11.227 (11.027 / 11.495) | 5.232 (5.137 / 5.374) | 12.935 (12.648 / 13.450) | **4.722 (4.606 / 4.964)** | 5.772 (5.493 / 5.891) | 10.362 (10.035 / 10.718) | 5.341 (5.276 / 5.433) | - |
| `transpose_2d_256` | 2D 256^2 transpose [1,0] | 0.123 (0.122 / 0.123) | 0.035 (0.034 / 0.036) | 0.132 (0.132 / 0.133) | 0.022 (0.021 / 0.023) | **0.016 (0.016 / 0.017)** | 0.063 (0.061 / 0.069) | 0.045 (0.042 / 0.047) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | 31.647 (31.553 / 31.860) | 13.310 (13.282 / 13.392) | 35.822 (35.752 / 35.921) | **11.968 (10.754 / 12.096)** | 12.631 (12.561 / 12.715) | 13.182 (13.109 / 13.211) | 12.114 (11.981 / 12.210) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 42.357 (41.270 / 43.275) | 22.656 (21.594 / 23.162) | 46.498 (45.930 / 46.915) | **13.765 (13.636 / 13.836)** | 13.929 (13.814 / 14.057) | 177.771 (167.647 / 183.509) | 19.235 (19.051 / 21.031) | - |

## Threads: 4

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | naive odometer (ms) | tenferro-rs transpose (ms) | tenferro-rs to_contiguous (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `cyclic_13d_2` | 13D 2^13 cyclic [1,2,...,0] | 0.011 (0.011 / 0.011) | 0.036 (0.034 / 0.046) | 0.019 (0.019 / 0.019) | 0.022 (0.021 / 0.023) | **0.003 (0.003 / 0.003)** | 0.011 (0.009 / 0.012) | 0.010 (0.010 / 0.011) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | - | - | **4.194 (4.155 / 4.309)** | - | - | 4.338 (4.142 / 4.455) |
| `reverse_13d_2` | 13D 2^13 reverse | 0.011 (0.011 / 0.011) | 0.068 (0.064 / 0.080) | 0.019 (0.019 / 0.019) | 0.030 (0.030 / 0.031) | **0.006 (0.006 / 0.006)** | 0.009 (0.009 / 0.009) | 0.010 (0.010 / 0.010) | - |
| `reverse_20d_2` | 20D 2^20 reverse | 1.855 (1.846 / 1.904) | 2.401 (2.269 / 3.352) | 2.897 (2.708 / 2.987) | 1.346 (1.319 / 1.383) | **0.433 (0.427 / 0.454)** | 1.847 (1.618 / 2.077) | 0.502 (0.466 / 0.523) | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | 512.819 (510.412 / 514.771) | 142.543 (141.567 / 144.924) | 679.767 (662.014 / 739.176) | 121.409 (118.383 / 142.038) | 134.497 (133.378 / 136.936) | 243.014 (240.171 / 243.740) | **111.203 (100.367 / 121.797)** | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | 21.457 (21.391 / 21.562) | 5.619 (4.762 / 6.449) | 38.447 (37.941 / 39.292) | 5.359 (5.296 / 5.724) | **5.217 (5.191 / 5.417)** | 18.653 (18.006 / 18.875) | 6.989 (4.787 / 13.057) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | 24.644 (24.156 / 25.353) | - | 39.497 (39.275 / 40.061) | - | **5.474 (5.344 / 5.526)** | 130.608 (129.056 / 131.312) | 12.383 (4.871 / 14.302) | - |
| `transpose_2d_1024` | 2D 1024^2 transpose [1,0] | 1.976 (1.939 / 2.050) | 0.522 (0.489 / 0.574) | 2.232 (2.222 / 2.307) | 0.254 (0.218 / 0.291) | **0.198 (0.189 / 0.205)** | 0.951 (0.939 / 0.969) | 0.535 (0.503 / 0.611) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | 11.653 (11.436 / 11.991) | 2.692 (2.578 / 2.810) | 13.757 (13.620 / 13.906) | 2.123 (2.094 / 2.179) | **1.856 (1.817 / 1.896)** | 9.986 (9.678 / 10.296) | 2.719 (2.533 / 3.719) | - |
| `transpose_2d_256` | 2D 256^2 transpose [1,0] | 0.123 (0.118 / 0.123) | 0.078 (0.058 / 0.094) | 0.132 (0.132 / 0.133) | 0.022 (0.022 / 0.022) | **0.018 (0.018 / 0.019)** | 0.059 (0.041 / 0.063) | 0.071 (0.051 / 0.075) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | 31.990 (31.661 / 32.423) | 5.867 (5.449 / 6.041) | 35.992 (35.896 / 36.552) | 6.021 (5.764 / 6.387) | 6.406 (6.328 / 6.611) | 13.407 (13.224 / 13.453) | **5.597 (5.018 / 9.382)** | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 46.926 (44.849 / 47.563) | 9.888 (9.042 / 10.196) | 50.930 (48.057 / 53.052) | **7.920 (7.687 / 8.429)** | 10.252 (10.083 / 10.332) | 82.411 (50.303 / 92.048) | 15.217 (11.818 / 21.628) | - |
