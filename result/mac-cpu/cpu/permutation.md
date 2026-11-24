# CPU Permutation Benchmark Results

- Target profile: `mac-cpu`
- Suite: `cpu/permutation`
- Suite file: `benchmarks/cpu/permutation.yaml`
- Timestamp: `2026-11-24T18:52:43.890471+00:00`
- tenferro-rs commit: `ce9125d0cff4ee40841120759b5df45252c27f22`

## CPU Information

- Model: `Apple M1 (Virtual)`
- Vendor: `Apple`
- Logical CPUs: `3`
- Physical CPUs: `3`
- Sockets: `1`
- Cores per socket: `3`
- Threads per core: `1`
- NUMA nodes: `1`
- Performance levels: `Standard: 3 physical / 3 logical (L1i 192 KiB, L1d 128 KiB, L2 12 MiB, 3 CPUs/L2)`
- Python platform: `macOS-26.7.2-arm64-arm-64bit-Mach-O`
- Container: `none`

Thread counts are controlled via `RAYON_NUM_THREADS` / `OMP_NUM_THREADS` / `JULIA_NUM_THREADS`; no CPU-affinity pinning (`taskset` / `numactl`) was applied. The controlled thread environment is recorded per thread count in the run's `run_t<N>.yaml`.

`tenferro-rs` measures `TypedTensorView::transpose_view` followed by `BackendSession::to_contiguous_read`; one shared session is entered before timing, and input view descriptors and metadata-only `transpose_view` are built outside the timed region. Materialization accepts arbitrary source strides. Every backend allocates a fresh destination inside each timed call, so the table compares allocation-inclusive end-to-end materialization rather than destination-reuse copy kernels. `hptt` only participates in patterns with a contiguous source and destination. Correctness is verified against an internal, untimed odometer reference before any timing; a `FAILED` cell means that backend's output did not match the reference for that pattern.

## Threads: 1

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | **8.609 (8.321 / 9.343)** | skipped (rebuild with --features hptt) | 13.555 (12.797 / 14.481) | 15.200 (13.593 / 17.043) | 14.214 (13.310 / 15.009) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **7.661 (7.140 / 7.972)** | - | - | 9.352 (8.608 / 10.215) |
| `reverse_15d_3` | 15D 3^15 reverse | 72.251 (63.240 / 80.775) | skipped (rebuild with --features hptt) | 50.341 (48.528 / 54.069) | 131.512 (110.168 / 134.748) | **33.195 (25.641 / 38.914)** | - |
| `reverse_23d_2` | 23D 2^23 reverse | 36.159 (33.922 / 37.458) | skipped (rebuild with --features hptt) | 27.240 (26.248 / 29.208) | 93.988 (72.724 / 105.294) | **24.030 (22.732 / 27.131)** | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | 8536.116 (7683.397 / 9181.607) | skipped (rebuild with --features hptt) | 5470.147 (4502.564 / 8135.464) | **4342.222 (3342.733 / 5794.263)** | 5283.015 (3620.610 / 6273.545) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | 14.495 (13.433 / 16.338) | skipped (rebuild with --features hptt) | 14.723 (13.391 / 15.776) | 16.618 (15.917 / 17.009) | **11.380 (11.176 / 11.899)** | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | 22.617 (19.013 / 32.666) | - | 27.698 (22.702 / 35.280) | 201.430 (190.991 / 213.188) | **17.886 (17.663 / 18.256)** | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | 9.448 (8.496 / 9.759) | skipped (rebuild with --features hptt) | 9.713 (9.273 / 10.339) | 27.801 (24.273 / 32.192) | **9.244 (8.774 / 10.210)** | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | **20.201 (18.551 / 23.292)** | skipped (rebuild with --features hptt) | 26.759 (24.483 / 29.429) | 24.034 (20.879 / 25.646) | 21.061 (20.158 / 22.732) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | **22.350 (20.956 / 23.133)** | skipped (rebuild with --features hptt) | 24.415 (23.734 / 26.037) | 254.513 (246.464 / 267.644) | 48.766 (43.462 / 54.336) | - |

## Threads: 4

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | **8.186 (7.785 / 9.031)** | skipped (rebuild with --features hptt) | 10.547 (10.428 / 10.964) | 20.038 (14.372 / 27.477) | 9.247 (7.917 / 9.867) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **8.436 (8.144 / 8.738)** | - | - | 9.591 (8.054 / 10.040) |
| `reverse_15d_3` | 15D 3^15 reverse | 25.408 (23.046 / 29.053) | skipped (rebuild with --features hptt) | 26.988 (25.316 / 30.057) | 142.025 (131.729 / 176.315) | **20.133 (14.519 / 22.641)** | - |
| `reverse_23d_2` | 23D 2^23 reverse | 29.372 (24.577 / 42.100) | skipped (rebuild with --features hptt) | **22.393 (21.689 / 25.909)** | 85.913 (76.010 / 91.879) | 24.605 (18.389 / 28.074) | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | **5374.657 (4499.667 / 8889.601)** | skipped (rebuild with --features hptt) | 7569.227 (7103.346 / 9748.239) | 8466.324 (7752.501 / 9741.329) | 7490.750 (3226.871 / 8654.373) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | **8.677 (8.454 / 8.944)** | skipped (rebuild with --features hptt) | 11.961 (10.567 / 12.340) | 18.951 (18.255 / 20.483) | 9.993 (8.536 / 10.337) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | 13.409 (12.307 / 14.569) | - | 14.794 (14.132 / 15.957) | 213.004 (189.550 / 230.695) | **10.856 (10.194 / 11.879)** | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | **5.623 (5.290 / 6.076)** | skipped (rebuild with --features hptt) | 6.190 (6.034 / 6.563) | 25.735 (20.086 / 30.599) | 5.908 (5.687 / 6.432) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | 14.289 (13.532 / 14.885) | skipped (rebuild with --features hptt) | 16.536 (16.068 / 17.587) | 21.331 (20.638 / 22.129) | **11.807 (10.051 / 12.377)** | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | **16.929 (15.882 / 17.351)** | skipped (rebuild with --features hptt) | 18.151 (17.492 / 18.450) | 274.777 (259.339 / 298.710) | 21.770 (20.551 / 26.362) | - |
