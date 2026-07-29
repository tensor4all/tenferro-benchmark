# CPU Permutation Benchmark Results

- Target profile: `mac-cpu`
- Suite: `cpu/permutation`
- Suite file: `benchmarks/cpu/permutation.yaml`
- Timestamp: `2026-07-29T11:04:17.043405+00:00`
- tenferro-rs commit: `57c41c218356e871532e4dea4465a6bcf2d6c8fb`

## CPU Information

- Model: `Apple M5 Max`
- Vendor: `Apple`
- Logical CPUs: `18`
- Physical CPUs: `18`
- Sockets: `1`
- Cores per socket: `18`
- Threads per core: `1`
- NUMA nodes: `1`
- Performance levels: `Super: 6 physical / 6 logical (L1i 192 KiB, L1d 128 KiB, L2 16 MiB, 6 CPUs/L2); Performance: 12 physical / 12 logical (L1i 128 KiB, L1d 64 KiB, L2 8 MiB, 6 CPUs/L2)`
- Python platform: `macOS-26.5.1-arm64-arm-64bit`

CPU pinning is unavailable on macOS; thread counts are controlled only via `RAYON_NUM_THREADS` / `OMP_NUM_THREADS` / `JULIA_NUM_THREADS`.

`tenferro-rs` measures `TypedTensorView::transpose_view` followed by `CpuBackend::to_contiguous`; the metadata-only `transpose_view` is built outside the timed region, and `to_contiguous` accepts arbitrary source strides. Every backend allocates a fresh destination inside each timed call, so the table compares allocation-inclusive end-to-end materialization rather than destination-reuse copy kernels. `hptt` only participates in patterns with a contiguous source and destination. Correctness is verified against an internal, untimed odometer reference before any timing; a `FAILED` cell means that backend's output did not match the reference for that pattern.

## Threads: 1

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | 7.163 (7.050 / 7.244) | 5.034 (4.962 / 5.152) | **3.725 (3.716 / 3.737)** | 11.117 (11.086 / 11.139) | 4.672 (4.507 / 4.690) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **2.441 (2.425 / 2.455)** | - | - | 2.459 (2.433 / 2.513) |
| `reverse_15d_3` | 15D 3^15 reverse | 30.489 (30.321 / 30.607) | 14.498 (14.386 / 14.757) | 21.088 (21.030 / 21.129) | 54.161 (53.734 / 54.636) | **9.639 (9.525 / 9.744)** | - |
| `reverse_23d_2` | 23D 2^23 reverse | 35.852 (35.704 / 36.178) | 17.151 (17.104 / 17.307) | **11.804 (11.678 / 11.888)** | 47.051 (45.982 / 47.549) | 13.508 (13.136 / 13.869) | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | 308.184 (305.149 / 312.535) | 192.603 (191.776 / 193.456) | 194.006 (193.026 / 195.174) | 186.917 (184.346 / 188.069) | **186.020 (184.777 / 187.561)** | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | 6.509 (5.541 / 7.345) | 4.870 (4.805 / 4.951) | **3.726 (3.702 / 3.744)** | 16.752 (16.663 / 16.768) | 6.092 (6.021 / 6.160) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | 7.485 (7.476 / 7.722) | - | **6.528 (6.500 / 6.651)** | 112.063 (111.911 / 112.254) | 7.860 (7.808 / 7.938) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | 6.041 (6.011 / 6.082) | **3.861 (3.839 / 3.887)** | 4.876 (4.823 / 4.942) | 10.449 (10.324 / 10.716) | 5.642 (5.573 / 5.737) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | 12.474 (12.364 / 12.509) | 12.177 (11.960 / 12.258) | 11.477 (11.102 / 12.090) | 15.529 (15.439 / 15.575) | **11.207 (11.090 / 11.262)** | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 133.472 (133.082 / 134.016) | 20.592 (20.562 / 20.622) | **15.771 (15.531 / 15.850)** | 127.245 (116.753 / 147.054) | 101.335 (101.022 / 101.990) | - |

## Threads: 4

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | 2.921 (2.899 / 2.995) | 2.090 (1.969 / 2.236) | 3.687 (3.678 / 3.699) | 11.171 (11.103 / 11.177) | **1.592 (1.504 / 5.750)** | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **2.430 (2.417 / 2.456)** | - | - | 2.437 (2.425 / 2.451) |
| `reverse_15d_3` | 15D 3^15 reverse | 20.911 (20.590 / 25.082) | 8.641 (8.408 / 8.988) | 10.606 (10.203 / 10.976) | 53.869 (53.610 / 54.533) | **7.745 (3.776 / 11.805)** | - |
| `reverse_23d_2` | 23D 2^23 reverse | 19.791 (19.730 / 19.934) | 9.292 (6.718 / 9.515) | 7.693 (7.662 / 7.876) | 43.445 (42.903 / 43.759) | **4.148 (4.003 / 4.217)** | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | 128.126 (127.848 / 128.896) | 72.873 (72.526 / 73.541) | **70.544 (70.496 / 70.842)** | 175.808 (175.085 / 176.946) | 77.703 (70.190 / 77.876) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | 2.700 (2.643 / 2.778) | 2.052 (2.043 / 2.158) | **1.675 (1.644 / 1.713)** | 16.784 (16.740 / 16.810) | 6.187 (1.672 / 6.316) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | 4.670 (4.247 / 4.788) | - | **2.719 (2.677 / 2.892)** | 112.736 (112.289 / 112.837) | 7.928 (3.571 / 15.300) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | 1.934 (1.898 / 1.988) | **1.400 (1.343 / 1.449)** | 1.425 (1.412 / 1.451) | 10.504 (10.288 / 10.774) | 1.826 (1.747 / 4.278) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | 4.132 (4.065 / 4.193) | **3.978 (3.901 / 4.470)** | 4.123 (4.077 / 4.221) | 15.768 (15.524 / 15.790) | 6.007 (3.426 / 6.609) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 46.974 (46.143 / 47.276) | **6.826 (6.738 / 6.886)** | 8.194 (8.176 / 8.280) | 105.507 (93.158 / 108.216) | 34.876 (32.712 / 40.323) | - |
