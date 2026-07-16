# CPU Permutation Benchmark Results

- Target profile: `amd-cpu`
- Suite: `cpu/permutation`
- Suite file: `benchmarks/cpu/permutation.yaml`
- Timestamp: `2026-07-16T02:24:54.833031+00:00`
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
| `cyclic_13d_2` | 13D 2^13 cyclic [1,2,...,0] | 0.028 (0.028 / 0.028) | 0.026 (0.026 / 0.026) | 0.039 (0.038 / 0.039) | - | 0.009 (0.009 / 0.009) | 0.025 (0.025 / 0.026) | **0.008 (0.007 / 0.008)** | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | - | - | **12.864 (12.849 / 12.942)** | - | - | 13.533 (13.423 / 13.629) |
| `reverse_13d_2` | 13D 2^13 reverse | 0.029 (0.029 / 0.030) | 0.083 (0.082 / 0.087) | 0.044 (0.044 / 0.044) | - | **0.011 (0.011 / 0.011)** | 0.028 (0.028 / 0.029) | 0.027 (0.027 / 0.028) | - |
| `reverse_20d_2` | 20D 2^20 reverse | 8.590 (8.584 / 8.643) | 11.664 (11.636 / 11.692) | 11.264 (11.260 / 11.312) | 8.851 (8.838 / 8.894) | **3.864 (3.832 / 3.868)** | 8.127 (8.125 / 8.131) | 4.967 (4.964 / 4.972) | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | 2628.094 (2590.413 / 2653.022) | 2160.879 (2159.569 / 2161.833) | 3684.429 (3677.324 / 3686.528) | 996.804 (996.430 / 996.952) | **714.520 (713.857 / 715.357)** | 979.295 (978.853 / 979.579) | 753.518 (753.194 / 753.772) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | 59.667 (59.559 / 59.734) | 74.550 (74.332 / 74.878) | 137.014 (136.908 / 137.413) | 50.018 (49.999 / 50.037) | **31.031 (30.998 / 31.067)** | 58.654 (58.560 / 58.776) | 31.933 (31.892 / 31.946) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | 61.297 (61.230 / 61.422) | - | 139.625 (139.417 / 140.305) | - | 32.648 (32.607 / 32.797) | 330.100 (329.151 / 330.386) | **30.298 (30.222 / 30.325)** | - |
| `transpose_2d_1024` | 2D 1024^2 transpose [1,0] | 6.618 (6.578 / 7.108) | 2.001 (1.995 / 2.090) | 7.204 (7.192 / 7.424) | 3.236 (3.216 / 3.577) | **1.604 (1.600 / 1.604)** | 7.216 (7.215 / 7.219) | 1.972 (1.971 / 1.973) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | 40.183 (39.776 / 41.872) | 26.796 (26.476 / 29.067) | 58.833 (57.919 / 61.637) | 22.326 (21.642 / 22.964) | 16.215 (15.926 / 16.770) | 35.193 (35.149 / 35.317) | **15.906 (15.891 / 15.932)** | - |
| `transpose_2d_256` | 2D 256^2 transpose [1,0] | 0.235 (0.235 / 0.254) | 0.093 (0.093 / 0.097) | 0.365 (0.362 / 0.377) | 0.138 (0.133 / 0.147) | **0.066 (0.066 / 0.066)** | 0.163 (0.163 / 0.163) | 0.087 (0.086 / 0.087) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | 99.665 (98.343 / 100.547) | 87.549 (87.402 / 90.303) | 187.430 (186.021 / 189.173) | 68.195 (67.880 / 69.014) | 60.501 (60.389 / 61.749) | 77.093 (77.039 / 77.130) | **37.289 (37.246 / 37.326)** | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 269.530 (267.886 / 270.962) | 226.072 (224.294 / 231.566) | 352.530 (344.959 / 357.679) | 95.296 (94.285 / 96.053) | **77.374 (77.201 / 78.683)** | 302.785 (302.298 / 303.184) | 162.535 (162.485 / 162.651) | - |

## Threads: 4

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | naive odometer (ms) | tenferro-rs transpose (ms) | tenferro-rs to_contiguous (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `cyclic_13d_2` | 13D 2^13 cyclic [1,2,...,0] | 0.079 (0.078 / 0.080) | 0.076 (0.076 / 0.076) | 0.107 (0.107 / 0.109) | - | 0.020 (0.019 / 0.020) | 0.027 (0.027 / 0.027) | **0.007 (0.007 / 0.007)** | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | - | - | 13.091 (13.074 / 13.101) | - | - | **13.078 (13.056 / 13.142)** |
| `reverse_13d_2` | 13D 2^13 reverse | 0.085 (0.085 / 0.086) | 0.248 (0.246 / 0.253) | 0.122 (0.120 / 0.122) | - | 0.033 (0.033 / 0.033) | **0.025 (0.025 / 0.025)** | 0.031 (0.031 / 0.031) | - |
| `reverse_20d_2` | 20D 2^20 reverse | 8.470 (8.466 / 8.473) | 4.620 (4.572 / 4.906) | 10.927 (10.919 / 10.974) | 3.405 (3.397 / 3.478) | 4.003 (3.887 / 4.073) | 7.995 (7.993 / 7.997) | **2.988 (2.941 / 3.019)** | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | 2600.476 (2598.624 / 2603.162) | 687.338 (686.374 / 690.427) | 3695.162 (3687.323 / 3708.728) | 295.065 (289.612 / 296.673) | 256.284 (255.129 / 257.263) | 981.431 (980.971 / 981.858) | **255.916 (254.694 / 256.455)** | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | 59.079 (59.045 / 60.775) | 40.228 (38.085 / 43.043) | 137.281 (137.122 / 137.756) | 19.760 (19.641 / 19.815) | **14.819 (14.668 / 14.901)** | 63.677 (58.397 / 63.734) | 14.867 (14.611 / 15.764) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | 64.278 (64.247 / 64.439) | - | 140.730 (140.504 / 142.784) | - | **15.261 (15.237 / 15.369)** | 328.820 (328.317 / 329.165) | 16.043 (14.979 / 16.296) | - |
| `transpose_2d_1024` | 2D 1024^2 transpose [1,0] | 6.500 (6.497 / 6.504) | 1.289 (1.254 / 1.310) | 7.006 (7.004 / 7.072) | **0.747 (0.745 / 0.760)** | 1.150 (1.101 / 1.261) | 7.211 (7.209 / 7.217) | 1.141 (1.062 / 1.149) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | 38.395 (38.288 / 38.522) | 17.004 (16.542 / 17.262) | 55.460 (55.346 / 55.724) | 7.061 (7.036 / 7.099) | **4.959 (4.825 / 5.245)** | 35.143 (35.040 / 35.299) | 5.487 (4.846 / 6.655) | - |
| `transpose_2d_256` | 2D 256^2 transpose [1,0] | 0.219 (0.219 / 0.219) | 0.141 (0.137 / 0.147) | 0.336 (0.333 / 0.356) | **0.045 (0.045 / 0.045)** | 0.104 (0.100 / 0.109) | 0.159 (0.159 / 0.159) | 0.128 (0.128 / 0.131) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | 99.298 (99.226 / 99.400) | 41.567 (39.925 / 44.032) | 186.656 (186.413 / 187.642) | 20.321 (19.647 / 20.430) | 20.686 (20.617 / 21.234) | 77.001 (76.937 / 77.044) | **16.158 (16.007 / 16.201)** | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 281.866 (281.462 / 282.455) | 80.058 (77.752 / 80.827) | 359.434 (359.122 / 360.519) | 28.728 (28.630 / 28.953) | **22.302 (22.267 / 22.361)** | 276.850 (276.475 / 277.585) | 45.075 (44.271 / 59.293) | - |
