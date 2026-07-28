# CPU Permutation Benchmark Results

- Target profile: `mac-cpu`
- Suite: `cpu/permutation`
- Suite file: `benchmarks/cpu/permutation.yaml`
- Timestamp: `2026-07-28T01:10:44.510012+00:00`
- tenferro-rs commit: `80ebcc38ce11fb93385e8b6a1a49b613bc17452f`

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
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | 7.887 (7.831 / 7.936) | 7.296 (7.099 / 7.426) | **4.721 (4.691 / 4.860)** | 11.876 (11.793 / 11.927) | 7.267 (7.158 / 7.364) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **4.225 (4.145 / 4.294)** | - | - | 4.314 (4.225 / 4.879) |
| `reverse_15d_3` | 15D 3^15 reverse | 35.685 (35.511 / 35.990) | 18.280 (18.180 / 18.407) | 26.727 (26.393 / 26.876) | 72.311 (71.425 / 72.704) | **11.425 (11.195 / 11.599)** | - |
| `reverse_23d_2` | 23D 2^23 reverse | 44.522 (43.768 / 45.082) | 21.931 (21.573 / 22.114) | 13.143 (13.112 / 13.235) | 62.995 (54.965 / 77.844) | **13.095 (12.903 / 13.163)** | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | 339.402 (338.775 / 339.935) | 221.416 (221.098 / 221.593) | 241.665 (241.201 / 242.068) | 240.232 (239.140 / 240.721) | **206.643 (206.249 / 206.943)** | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | 6.215 (6.151 / 6.465) | 7.175 (7.106 / 7.683) | **5.570 (5.436 / 5.654)** | 17.732 (17.681 / 17.764) | 6.650 (6.293 / 6.846) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | 8.090 (7.967 / 8.278) | - | **6.897 (6.838 / 7.021)** | 129.674 (127.884 / 130.201) | 8.806 (8.646 / 8.880) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | 5.347 (5.284 / 5.474) | **4.869 (4.819 / 4.973)** | 5.876 (5.826 / 5.982) | 9.535 (9.393 / 9.736) | 5.290 (5.234 / 5.360) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | 13.646 (13.507 / 13.947) | **11.879 (10.592 / 12.166)** | 13.241 (13.098 / 13.896) | 13.255 (13.170 / 13.298) | 12.367 (12.213 / 12.430) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 21.770 (21.705 / 22.099) | **13.739 (13.604 / 14.002)** | 14.172 (14.107 / 14.239) | 114.763 (104.518 / 122.495) | 20.706 (19.281 / 21.768) | - |

## Threads: 4

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | 4.258 (4.145 / 4.592) | 6.158 (6.078 / 6.256) | 4.861 (4.807 / 4.996) | 11.899 (11.842 / 11.946) | **3.794 (3.673 / 4.282)** | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **4.104 (4.074 / 4.187)** | - | - | 4.219 (4.058 / 4.282) |
| `reverse_15d_3` | 15D 3^15 reverse | 21.100 (20.453 / 23.595) | 12.398 (11.532 / 14.348) | 12.676 (12.582 / 12.794) | 72.304 (71.646 / 72.518) | **7.585 (7.237 / 8.793)** | - |
| `reverse_23d_2` | 23D 2^23 reverse | 23.593 (21.929 / 25.060) | 11.815 (11.060 / 12.553) | 7.310 (7.251 / 7.374) | 72.420 (56.412 / 100.304) | **5.715 (5.583 / 5.995)** | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | 141.769 (139.840 / 142.822) | **117.952 (117.710 / 118.453)** | 132.867 (132.137 / 133.844) | 241.479 (241.129 / 241.766) | 152.807 (103.138 / 509.062) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | **4.311 (4.163 / 4.359)** | 5.336 (5.315 / 5.610) | 5.233 (5.204 / 5.347) | 17.847 (17.789 / 17.856) | 4.780 (4.249 / 11.560) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | **4.812 (4.739 / 4.868)** | - | 5.349 (5.305 / 5.526) | 130.745 (129.829 / 131.188) | 11.302 (4.526 / 11.743) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | 2.635 (2.567 / 2.729) | 2.120 (2.103 / 2.142) | **1.847 (1.824 / 1.887)** | 9.520 (9.305 / 9.826) | 2.647 (2.538 / 6.526) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | **5.593 (5.480 / 6.083)** | 5.821 (5.725 / 5.999) | 6.183 (6.102 / 6.337) | 13.266 (13.185 / 13.327) | 8.832 (4.886 / 9.439) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 9.155 (8.444 / 9.522) | **7.627 (7.580 / 8.293)** | 10.022 (9.855 / 10.175) | 116.898 (74.887 / 124.678) | 14.296 (12.138 / 22.847) | - |
