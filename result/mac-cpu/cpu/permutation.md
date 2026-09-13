# CPU Permutation Benchmark Results

- Target profile: `mac-cpu`
- Suite: `cpu/permutation`
- Suite file: `benchmarks/cpu/permutation.yaml`
- Timestamp: `2026-09-13T05:31:37.455464+00:00`
- tenferro-rs commit: `a793c2e95693f053722fbff8d0db25722336c21f`

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
- Python platform: `macOS-26.5.1-arm64-arm-64bit-Mach-O`

CPU pinning is unavailable on macOS; thread counts are controlled only via `RAYON_NUM_THREADS` / `OMP_NUM_THREADS` / `JULIA_NUM_THREADS`.

`tenferro-rs` measures `TypedTensorView::transpose_view` followed by `CpuBackend::to_contiguous`; the metadata-only `transpose_view` is built outside the timed region, and `to_contiguous` accepts arbitrary source strides. Every backend allocates a fresh destination inside each timed call, so the table compares allocation-inclusive end-to-end materialization rather than destination-reuse copy kernels. `hptt` only participates in patterns with a contiguous source and destination. Correctness is verified against an internal, untimed odometer reference before any timing; a `FAILED` cell means that backend's output did not match the reference for that pattern.

## Threads: 1

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | 7.189 (7.139 / 7.277) | skipped (rebuild with --features hptt) | **3.769 (3.759 / 3.785)** | 11.105 (11.037 / 11.282) | 4.718 (4.539 / 11.916) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **2.425 (2.417 / 2.505)** | - | - | 2.445 (2.426 / 2.537) |
| `reverse_15d_3` | 15D 3^15 reverse | 31.521 (31.375 / 31.633) | skipped (rebuild with --features hptt) | 22.277 (22.226 / 22.396) | 62.544 (59.086 / 67.029) | **10.223 (10.055 / 16.948)** | - |
| `reverse_23d_2` | 23D 2^23 reverse | 45.664 (45.476 / 45.930) | skipped (rebuild with --features hptt) | 12.938 (12.858 / 13.052) | 53.210 (52.731 / 57.584) | **9.941 (9.724 / 16.375)** | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | 315.839 (315.716 / 316.898) | skipped (rebuild with --features hptt) | 212.443 (212.072 / 213.018) | 211.118 (190.420 / 223.956) | **194.491 (192.975 / 226.963)** | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | 5.444 (5.400 / 5.487) | skipped (rebuild with --features hptt) | **3.802 (3.776 / 3.857)** | 16.921 (16.818 / 24.373) | 5.929 (5.822 / 12.889) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | 8.313 (8.236 / 8.588) | - | **7.792 (7.687 / 7.917)** | 120.135 (117.357 / 127.751) | 9.641 (8.387 / 16.062) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | 6.378 (6.315 / 6.456) | skipped (rebuild with --features hptt) | **5.056 (5.023 / 5.115)** | 12.590 (11.976 / 17.716) | 6.132 (6.058 / 9.757) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | 12.880 (12.814 / 12.940) | skipped (rebuild with --features hptt) | 12.241 (12.073 / 12.461) | 15.677 (15.520 / 19.425) | **11.658 (11.596 / 15.446)** | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 127.063 (125.437 / 131.357) | skipped (rebuild with --features hptt) | **16.676 (16.580 / 16.823)** | 120.151 (110.897 / 129.003) | 108.285 (103.645 / 110.601) | - |

## Threads: 4

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | 2.752 (2.703 / 2.782) | skipped (rebuild with --features hptt) | 3.804 (3.760 / 3.820) | 11.126 (11.053 / 15.236) | **1.590 (1.479 / 6.004)** | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **2.500 (2.460 / 2.544)** | - | - | 2.548 (2.491 / 2.609) |
| `reverse_15d_3` | 15D 3^15 reverse | 26.541 (26.445 / 26.695) | skipped (rebuild with --features hptt) | 9.883 (9.812 / 9.994) | 59.530 (59.250 / 60.154) | **8.602 (4.282 / 8.937)** | - |
| `reverse_23d_2` | 23D 2^23 reverse | 23.774 (23.683 / 23.900) | skipped (rebuild with --features hptt) | 8.202 (8.186 / 8.238) | 53.159 (52.716 / 53.249) | **4.045 (3.869 / 4.100)** | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | 316.890 (270.036 / 321.984) | skipped (rebuild with --features hptt) | **71.560 (71.418 / 71.742)** | 207.076 (199.437 / 209.452) | 81.226 (72.443 / 81.468) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | 6.450 (6.377 / 6.523) | skipped (rebuild with --features hptt) | **1.777 (1.756 / 1.838)** | 17.283 (16.910 / 21.338) | 6.122 (1.699 / 6.599) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | 4.145 (4.059 / 4.222) | - | **2.924 (2.822 / 2.954)** | 114.771 (114.092 / 118.721) | 8.277 (3.389 / 15.802) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | 2.102 (2.085 / 2.171) | skipped (rebuild with --features hptt) | **1.476 (1.442 / 1.509)** | 14.743 (12.203 / 22.342) | 1.939 (1.814 / 4.616) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | **4.411 (4.342 / 4.511)** | skipped (rebuild with --features hptt) | 4.443 (4.367 / 4.517) | 18.133 (15.649 / 18.944) | 6.169 (3.597 / 6.436) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 36.991 (36.066 / 37.737) | skipped (rebuild with --features hptt) | **8.549 (8.530 / 8.610)** | 113.574 (104.211 / 114.921) | 36.308 (32.523 / 41.566) | - |
