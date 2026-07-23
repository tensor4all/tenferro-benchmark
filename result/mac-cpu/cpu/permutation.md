# CPU Permutation Benchmark Results

- Target profile: `mac-cpu`
- Suite: `cpu/permutation`
- Suite file: `benchmarks/cpu/permutation.yaml`
- Timestamp: `2026-07-23T04:04:50.556302+00:00`
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
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | 7.973 (7.907 / 8.049) | 7.273 (7.200 / 7.645) | **4.896 (4.824 / 4.999)** | 11.867 (11.761 / 12.063) | 7.391 (7.002 / 7.993) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **4.112 (4.042 / 4.177)** | - | - | 4.166 (4.129 / 4.381) |
| `reverse_15d_3` | 15D 3^15 reverse | 33.480 (33.002 / 33.877) | 18.735 (18.601 / 18.917) | 27.463 (26.924 / 27.863) | 73.230 (72.831 / 73.506) | **11.515 (11.359 / 11.770)** | - |
| `reverse_23d_2` | 23D 2^23 reverse | 43.465 (42.010 / 45.282) | 21.688 (21.372 / 21.870) | **13.041 (12.960 / 13.581)** | 79.091 (61.495 / 93.238) | 13.226 (12.911 / 13.281) | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | 338.963 (338.083 / 341.114) | 225.021 (223.742 / 226.775) | 244.521 (243.459 / 246.250) | 246.002 (243.346 / 247.661) | **213.711 (212.240 / 214.653)** | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | 6.021 (5.958 / 6.535) | 6.887 (6.784 / 7.618) | **5.422 (5.386 / 5.548)** | 17.813 (17.760 / 17.892) | 6.860 (6.579 / 6.950) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | 7.986 (7.885 / 8.375) | - | **6.885 (6.847 / 6.977)** | 131.121 (129.449 / 131.184) | 8.809 (8.702 / 9.179) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | 5.208 (5.158 / 5.263) | **4.884 (4.833 / 4.945)** | 5.585 (5.534 / 5.773) | 9.426 (9.323 / 9.750) | 5.301 (5.253 / 5.392) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | 13.711 (13.480 / 14.085) | **10.496 (10.347 / 10.930)** | 12.686 (12.601 / 13.295) | 13.325 (13.143 / 13.401) | 12.216 (12.114 / 12.763) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 21.198 (20.982 / 21.330) | 13.821 (13.694 / 14.023) | **13.807 (13.709 / 14.491)** | 241.489 (220.055 / 246.070) | 20.141 (19.440 / 20.540) | - |

## Threads: 4

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | **3.665 (3.607 / 4.206)** | 6.131 (6.025 / 6.191) | 4.883 (4.818 / 5.008) | 12.058 (11.811 / 12.373) | 3.694 (3.642 / 10.449) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **4.117 (4.084 / 4.154)** | - | - | 4.203 (4.078 / 4.564) |
| `reverse_15d_3` | 15D 3^15 reverse | 22.773 (20.959 / 29.423) | 15.966 (14.276 / 18.558) | 12.564 (12.455 / 13.240) | 71.410 (70.711 / 72.306) | **9.644 (7.442 / 15.737)** | - |
| `reverse_23d_2` | 23D 2^23 reverse | 22.522 (22.270 / 22.645) | 14.781 (12.794 / 15.307) | 7.634 (7.483 / 7.688) | 58.210 (53.396 / 93.277) | **5.847 (5.471 / 6.243)** | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | 142.455 (141.471 / 143.297) | 124.593 (122.110 / 145.149) | 134.064 (132.252 / 135.151) | 244.393 (242.997 / 245.685) | **113.019 (103.440 / 117.488)** | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | **4.231 (4.193 / 4.409)** | 5.287 (5.241 / 5.566) | 5.313 (5.233 / 5.574) | 17.751 (17.704 / 17.800) | 4.564 (4.211 / 11.176) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | **4.884 (4.817 / 5.138)** | - | 5.639 (5.524 / 5.866) | 130.685 (129.356 / 131.498) | 12.286 (4.506 / 19.352) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | 2.722 (2.648 / 2.822) | 2.132 (2.103 / 2.174) | **1.886 (1.846 / 1.930)** | 9.491 (9.281 / 9.740) | 2.738 (2.547 / 6.657) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | **5.846 (5.459 / 6.060)** | 6.151 (5.785 / 6.409) | 6.201 (6.106 / 6.663) | 13.197 (13.058 / 13.415) | 8.918 (4.867 / 9.627) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 10.602 (9.373 / 11.174) | **8.201 (7.603 / 8.914)** | 10.147 (9.978 / 10.529) | 224.832 (189.759 / 240.971) | 16.258 (13.818 / 25.192) | - |
