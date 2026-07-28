# Linux CPU Permutation Benchmark Results

- Suite: `cpu/permutation`
- Target profile: `linux-cpu`
- Source timestamp: `20260728_021651`

Mirrored from `data/results/amd-cpu/cpu/permutation/20260728_021651`;
the raw-run alias is `data/results/linux-cpu/cpu/permutation/20260728_021651`.
Regenerate with `./scripts/mirror_linux_cpu_cpu_reports.sh 1:20260728_013419 4:20260728_015909 permutation:20260728_021651`.

### CPU Permutation Benchmark Results

- Target profile: `linux-cpu`
- Suite: `cpu/permutation`
- Suite file: `benchmarks/cpu/permutation.yaml`
- Timestamp: `2026-07-28T02:16:51.615166+00:00`
- tenferro-rs commit: `80ebcc38ce11fb93385e8b6a1a49b613bc17452f`

#### CPU Information

- Model: `AMD EPYC 7713P 64-Core Processor`
- Vendor: `AuthenticAMD`
- Logical CPUs: `64`
- Sockets: `1`
- Cores per socket: `64`
- Threads per core: `1`
- NUMA nodes: `1`
- Python platform: `Linux-6.8.0-101-generic-x86_64-with-glibc2.39`

On the Linux CPU devcontainer, thread counts are controlled via `RAYON_NUM_THREADS` / `OMP_NUM_THREADS` / `JULIA_NUM_THREADS`; no CPU-affinity pinning (`taskset` / `numactl`) is applied, matching the repository devcontainer convention. The controlled thread environment is recorded per thread count in the run's `run_t<N>.yaml`.

`tenferro-rs` measures `TypedTensorView::transpose_view` followed by `CpuBackend::to_contiguous`; the metadata-only `transpose_view` is built outside the timed region, and `to_contiguous` accepts arbitrary source strides. Every backend allocates a fresh destination inside each timed call, so the table compares allocation-inclusive end-to-end materialization rather than destination-reuse copy kernels. `hptt` only participates in patterns with a contiguous source and destination. Correctness is verified against an internal, untimed odometer reference before any timing; a `FAILED` cell means that backend's output did not match the reference for that pattern.

#### Threads: 1

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | 96.014 (94.465 / 100.590) | 81.425 (80.402 / 81.802) | **64.586 (64.450 / 64.928)** | 83.853 (83.247 / 83.964) | 66.570 (66.254 / 67.157) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **74.975 (74.865 / 75.942)** | - | - | 75.315 (74.579 / 76.362) |
| `reverse_15d_3` | 15D 3^15 reverse | 200.011 (193.198 / 227.123) | 137.961 (137.226 / 138.857) | 118.256 (117.922 / 119.848) | 234.169 (230.801 / 248.098) | **114.779 (109.872 / 128.624)** | - |
| `reverse_23d_2` | 23D 2^23 reverse | 108.579 (107.129 / 110.207) | 77.862 (77.633 / 78.136) | **66.620 (66.426 / 66.758)** | 179.662 (177.808 / 190.027) | 68.105 (64.097 / 74.387) | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | 1781.900 (1779.067 / 1792.102) | 1695.720 (1691.869 / 1707.511) | 1576.681 (1557.255 / 1582.546) | **1549.537 (1537.846 / 1555.183)** | 1574.848 (1557.555 / 1582.935) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | 91.068 (90.543 / 91.488) | 95.266 (91.592 / 96.647) | **74.808 (74.610 / 75.280)** | 103.407 (102.845 / 103.829) | 80.807 (79.986 / 81.114) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | 98.474 (89.970 / 100.428) | - | **77.503 (77.122 / 78.019)** | 280.761 (278.947 / 285.610) | 81.556 (80.643 / 82.666) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | 35.431 (35.186 / 36.272) | 27.681 (27.405 / 28.111) | **23.410 (23.188 / 23.635)** | 45.519 (45.438 / 46.039) | 32.447 (32.195 / 32.676) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | 112.651 (111.481 / 113.870) | 94.614 (94.013 / 97.012) | **90.071 (89.543 / 90.682)** | 106.126 (105.131 / 106.873) | 102.874 (102.101 / 104.145) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 144.007 (142.921 / 145.740) | 93.553 (93.120 / 94.070) | **90.179 (89.894 / 90.687)** | 138.463 (137.891 / 138.718) | 130.527 (129.911 / 131.051) | - |

#### Threads: 4

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | 33.884 (33.357 / 35.763) | **27.908 (26.587 / 28.515)** | 65.115 (65.021 / 65.307) | 85.423 (84.982 / 85.977) | 33.671 (18.339 / 56.698) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **76.637 (76.393 / 77.020)** | - | - | 77.004 (76.477 / 78.549) |
| `reverse_15d_3` | 15D 3^15 reverse | 113.107 (111.767 / 116.907) | 76.739 (76.197 / 80.945) | 93.821 (91.658 / 94.919) | 238.809 (234.538 / 242.671) | **41.037 (39.346 / 43.028)** | - |
| `reverse_23d_2` | 23D 2^23 reverse | 65.961 (60.015 / 66.782) | 45.669 (45.371 / 46.116) | 61.943 (60.975 / 64.448) | 190.774 (185.726 / 192.314) | **20.499 (18.682 / 21.181)** | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | 662.114 (650.374 / 671.927) | 614.670 (610.750 / 622.900) | **584.739 (577.929 / 587.191)** | 1538.633 (1531.553 / 1540.863) | 594.333 (559.047 / 604.040) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | 36.159 (35.330 / 36.861) | 39.282 (37.935 / 39.770) | **33.560 (33.267 / 34.510)** | 109.307 (108.436 / 109.538) | 48.624 (22.335 / 62.623) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | **35.151 (34.560 / 36.401)** | - | 36.869 (36.080 / 38.069) | 280.916 (279.231 / 283.382) | 52.985 (21.795 / 75.557) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | 13.904 (13.238 / 14.415) | 9.117 (8.881 / 9.979) | 10.425 (10.279 / 10.936) | 46.944 (46.671 / 47.446) | **9.068 (8.297 / 22.069)** | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | 42.011 (40.495 / 43.293) | 35.928 (35.472 / 38.703) | 40.574 (39.903 / 41.790) | 108.559 (108.086 / 109.722) | **29.266 (27.742 / 58.805)** | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 49.622 (48.862 / 51.263) | **36.790 (35.838 / 38.124)** | 39.002 (38.001 / 40.300) | 139.559 (138.752 / 141.253) | 67.569 (33.767 / 77.377) | - |
