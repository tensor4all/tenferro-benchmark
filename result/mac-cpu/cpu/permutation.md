# CPU Permutation Benchmark Results

- Target profile: `mac-cpu`
- Suite: `cpu/permutation`
- Suite file: `benchmarks/cpu/permutation.yaml`
- Timestamp: `2026-08-06T06:40:03.304127+00:00`
- tenferro-rs commit: `11a5b5a3c30b6258919557b4b69b429b6e686d75`

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
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | 8.046 (8.010 / 8.125) | 8.011 (7.672 / 8.463) | **4.776 (4.725 / 4.873)** | 11.746 (11.707 / 11.758) | 7.331 (7.254 / 7.442) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **4.662 (4.537 / 5.012)** | - | - | 4.986 (4.526 / 5.513) |
| `reverse_15d_3` | 15D 3^15 reverse | 32.879 (32.678 / 33.212) | 18.259 (18.146 / 18.532) | 26.200 (26.025 / 26.538) | 79.408 (73.401 / 81.197) | **11.692 (11.577 / 11.801)** | - |
| `reverse_23d_2` | 23D 2^23 reverse | 45.276 (43.279 / 46.422) | 23.259 (22.579 / 26.902) | **13.166 (13.098 / 13.435)** | 64.145 (56.470 / 69.159) | 13.423 (12.664 / 14.074) | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | 339.853 (336.434 / 344.720) | 217.848 (217.355 / 220.102) | 242.199 (240.963 / 243.714) | 245.800 (244.078 / 257.617) | **206.587 (206.134 / 206.781)** | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | 6.211 (6.143 / 6.232) | 6.841 (6.783 / 6.895) | **5.464 (5.428 / 5.589)** | 17.572 (17.505 / 17.586) | 6.676 (6.392 / 6.714) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | 8.099 (8.050 / 8.227) | - | **6.801 (6.754 / 6.853)** | 128.780 (126.670 / 129.428) | 8.534 (8.443 / 8.586) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | 5.938 (5.662 / 6.084) | **5.062 (4.920 / 5.352)** | 5.998 (5.936 / 6.278) | 9.217 (9.186 / 9.239) | 5.142 (5.104 / 5.173) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | 13.633 (13.532 / 13.673) | **12.223 (12.159 / 12.285)** | 13.370 (13.157 / 13.930) | 13.383 (13.287 / 13.548) | 13.112 (12.473 / 13.538) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 25.293 (22.377 / 27.509) | 15.331 (14.867 / 15.576) | **14.558 (14.425 / 14.682)** | 169.203 (87.225 / 252.117) | 21.873 (19.933 / 23.347) | - |

## Threads: 4

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | **3.720 (3.671 / 3.783)** | 5.976 (5.942 / 6.167) | 4.747 (4.674 / 4.835) | 11.827 (11.775 / 11.918) | 4.139 (3.744 / 10.622) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **4.111 (4.061 / 4.188)** | - | - | 4.145 (4.078 / 4.218) |
| `reverse_15d_3` | 15D 3^15 reverse | 20.196 (19.830 / 23.671) | 11.322 (11.234 / 13.343) | 12.570 (12.397 / 12.793) | 75.831 (72.302 / 77.867) | **9.066 (7.512 / 13.657)** | - |
| `reverse_23d_2` | 23D 2^23 reverse | 21.777 (21.663 / 21.902) | 9.650 (9.581 / 9.745) | 7.316 (7.281 / 7.386) | 68.254 (59.727 / 78.844) | **5.535 (5.531 / 5.690)** | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | 253.926 (247.157 / 254.624) | **117.239 (116.267 / 117.667)** | 135.236 (133.228 / 136.188) | 242.878 (240.126 / 245.755) | 134.319 (102.846 / 221.956) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | **4.243 (4.216 / 4.282)** | 5.430 (5.344 / 5.490) | 5.370 (5.241 / 5.516) | 17.669 (17.618 / 17.706) | 6.168 (4.173 / 11.270) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | 7.983 (7.963 / 8.093) | - | **5.454 (5.413 / 5.516)** | 130.399 (128.853 / 132.030) | 12.037 (4.466 / 19.149) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | 2.743 (2.697 / 2.832) | 2.116 (2.103 / 2.144) | **1.836 (1.812 / 1.851)** | 9.230 (9.196 / 9.252) | 2.748 (2.632 / 6.583) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | **5.255 (5.240 / 5.287)** | 5.754 (5.676 / 5.788) | 6.077 (6.025 / 6.219) | 13.231 (13.149 / 13.239) | 8.746 (4.854 / 9.312) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 8.601 (8.467 / 8.825) | **7.572 (7.505 / 7.652)** | 9.822 (9.788 / 9.897) | 226.049 (208.256 / 231.939) | 17.671 (11.508 / 23.662) | - |
