# CPU Permutation Benchmark Results

- Target profile: `amd-cpu`
- Suite: `cpu/permutation`
- Suite file: `benchmarks/cpu/permutation.yaml`
- Timestamp: `2026-10-22T19:19:44.656088+00:00`
- tenferro-rs commit: `2007d101d23b5baf7deba4afb8c7b92da347dbba`

## CPU Information

- Model: `AMD EPYC 7763 64-Core Processor`
- Vendor: `AuthenticAMD`
- Logical CPUs: `4`
- Sockets: `1`
- Cores per socket: `2`
- Threads per core: `2`
- NUMA nodes: `1`
- Python platform: `Linux-6.17.0-1022-azure-x86_64-with-glibc2.39`

On the Linux CPU devcontainer, thread counts are controlled via `RAYON_NUM_THREADS` / `OMP_NUM_THREADS` / `JULIA_NUM_THREADS`; no CPU-affinity pinning (`taskset` / `numactl`) is applied, matching the repository devcontainer convention. The controlled thread environment is recorded per thread count in the run's `run_t<N>.yaml`.

`tenferro-rs` measures `TypedTensorView::transpose_view` followed by `BackendSession::to_contiguous_read`; one shared session is entered before timing, and input view descriptors and metadata-only `transpose_view` are built outside the timed region. Materialization accepts arbitrary source strides. Every backend allocates a fresh destination inside each timed call, so the table compares allocation-inclusive end-to-end materialization rather than destination-reuse copy kernels. `hptt` only participates in patterns with a contiguous source and destination. Correctness is verified against an internal, untimed odometer reference before any timing; a `FAILED` cell means that backend's output did not match the reference for that pattern.

## Threads: 1

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | 14.070 (14.034 / 14.258) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **13.288 (13.179 / 13.591)** | 43.841 (34.151 / 44.299) | 25.579 (15.853 / 27.053) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **13.348 (13.315 / 13.428)** | - | - | 13.541 (13.406 / 13.653) |
| `reverse_15d_3` | 15D 3^15 reverse | 53.638 (51.504 / 55.045) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **51.593 (50.813 / 53.146)** | 207.699 (196.781 / 210.876) | 76.759 (66.908 / 78.143) | - |
| `reverse_23d_2` | 23D 2^23 reverse | **30.609 (30.524 / 30.765)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 34.901 (34.824 / 35.051) | 256.934 (251.312 / 266.691) | 37.931 (35.503 / 46.562) | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | 1210.221 (1207.561 / 1219.383) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 934.247 (895.011 / 974.661) | 1055.808 (1035.039 / 1519.482) | **707.974 (706.746 / 717.234)** | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | 14.564 (14.504 / 14.663) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **14.412 (14.390 / 14.489)** | 72.930 (62.353 / 86.272) | 29.793 (18.677 / 30.139) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | **16.792 (16.699 / 16.989)** | - | 17.095 (17.002 / 17.142) | 281.202 (280.053 / 293.149) | 34.656 (25.429 / 49.053) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | **7.643 (7.603 / 7.678)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 9.632 (9.569 / 9.760) | 77.549 (74.566 / 83.868) | 73.043 (68.988 / 79.544) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | 36.218 (36.137 / 36.496) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **35.717 (35.654 / 35.971)** | 58.289 (53.703 / 58.746) | 49.660 (44.433 / 49.804) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | **37.155 (37.038 / 37.327)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 37.166 (37.021 / 37.359) | 80.000 (77.696 / 86.738) | 66.112 (65.006 / 74.094) | - |

## Threads: 4

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | 9.686 (9.449 / 9.851) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **9.190 (9.073 / 9.290)** | 40.329 (33.346 / 43.522) | 16.759 (9.405 / 17.221) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **13.315 (13.293 / 13.406)** | - | - | 13.477 (13.419 / 13.634) |
| `reverse_15d_3` | 15D 3^15 reverse | **23.628 (23.123 / 24.206)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 24.029 (23.552 / 24.498) | 207.141 (204.648 / 209.643) | 39.794 (39.357 / 44.503) | - |
| `reverse_23d_2` | 23D 2^23 reverse | **16.147 (16.105 / 16.214)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 20.402 (20.243 / 20.611) | 260.725 (255.597 / 264.217) | 22.280 (15.993 / 22.779) | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | 331.669 (321.796 / 359.832) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 365.314 (362.956 / 366.156) | 1063.085 (1049.387 / 1212.345) | **298.880 (296.181 / 300.647)** | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | **9.125 (8.997 / 9.300)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 9.299 (9.025 / 9.409) | 70.275 (69.441 / 74.000) | 16.976 (10.686 / 21.066) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | 9.376 (9.187 / 9.514) | - | **9.264 (9.243 / 9.471)** | 278.673 (276.250 / 281.544) | 18.629 (18.373 / 22.988) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | **3.156 (3.117 / 3.207)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 3.497 (3.434 / 3.597) | 74.657 (72.778 / 75.931) | 15.316 (13.125 / 18.506) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | **15.134 (14.825 / 15.361)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 15.238 (15.168 / 15.389) | 56.696 (54.395 / 59.918) | 18.455 (16.007 / 21.755) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | **12.374 (12.236 / 12.488)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 12.777 (12.661 / 12.982) | 72.078 (70.972 / 74.844) | 28.204 (26.071 / 29.533) | - |
