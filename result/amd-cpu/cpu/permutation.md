# CPU Permutation Benchmark Results

- Target profile: `amd-cpu`
- Suite: `cpu/permutation`
- Suite file: `benchmarks/cpu/permutation.yaml`
- Timestamp: `2026-11-13T20:19:34.460272+00:00`
- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`

## CPU Information

- Model: `AMD EPYC 7763 64-Core Processor`
- Vendor: `AuthenticAMD`
- Logical CPUs: `4`
- Sockets: `1`
- Cores per socket: `2`
- Threads per core: `2`
- NUMA nodes: `1`
- Transparent huge pages: `madvise (always [madvise] never)`
- Transparent huge page defrag: `madvise (always defer defer+madvise [madvise] never)`
- Python platform: `Linux-7.0.0-1019-azure-x86_64-with-glibc2.39`
- L3 domains (shared CPUs): `0-3`
- Container: `devcontainer`

Thread counts are controlled via `RAYON_NUM_THREADS` / `OMP_NUM_THREADS` / `JULIA_NUM_THREADS`; no CPU-affinity pinning (`taskset` / `numactl`) was applied. The controlled thread environment is recorded per thread count in the run's `run_t<N>.yaml`.

`tenferro-rs` measures `TypedTensorView::transpose_view` followed by `BackendSession::to_contiguous_read`; one shared session is entered before timing, and input view descriptors and metadata-only `transpose_view` are built outside the timed region. Materialization accepts arbitrary source strides. Every backend allocates a fresh destination inside each timed call, so the table compares allocation-inclusive end-to-end materialization rather than destination-reuse copy kernels. `hptt` only participates in patterns with a contiguous source and destination. Correctness is verified against an internal, untimed odometer reference before any timing; a `FAILED` cell means that backend's output did not match the reference for that pattern.

## Threads: 1

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | **55.770 (55.442 / 56.040)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 59.567 (59.378 / 59.805) | 104.271 (78.600 / 106.372) | 84.309 (56.799 / 85.639) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | 66.457 (66.182 / 66.722) | - | - | **66.050 (65.887 / 66.497)** |
| `reverse_15d_3` | 15D 3^15 reverse | 103.305 (101.953 / 103.934) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **92.425 (91.963 / 93.737)** | 297.846 (274.947 / 303.383) | 132.671 (121.180 / 141.877) | - |
| `reverse_23d_2` | 23D 2^23 reverse | **31.043 (31.025 / 31.204)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 58.066 (57.880 / 58.554) | 264.013 (255.158 / 279.852) | 74.822 (68.468 / 91.560) | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | **1268.517 (1251.849 / 1272.574)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 1369.229 (1361.271 / 1376.545) | 2003.963 (1985.553 / 2013.506) | 1605.466 (1592.175 / 1632.414) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | **67.914 (67.013 / 68.018)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 68.809 (68.063 / 69.397) | 130.995 (121.577 / 144.570) | 97.239 (67.331 / 100.681) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | **70.490 (70.025 / 70.698)** | - | 71.665 (70.897 / 72.209) | 339.410 (336.874 / 350.129) | 88.246 (86.995 / 105.447) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | **7.259 (7.176 / 7.422)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 19.964 (19.895 / 20.183) | 55.846 (47.186 / 61.483) | 34.049 (29.631 / 40.470) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | 79.825 (79.331 / 81.118) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **78.992 (77.737 / 79.784)** | 118.979 (97.822 / 120.842) | 108.900 (107.133 / 118.728) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 82.298 (81.729 / 82.611) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **80.590 (80.141 / 81.152)** | 152.904 (151.199 / 161.462) | 139.769 (137.955 / 147.486) | - |

## Threads: 4

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs (ms) | HPTT (ms) | strided-rs (ms) | Julia Base (ms) | Strided.jl (ms) | memcpy (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | 19.843 (19.639 / 20.035) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **19.496 (19.109 / 19.946)** | 93.031 (86.459 / 97.436) | 41.301 (19.230 / 42.466) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | **66.810 (66.218 / 67.298)** | - | - | 67.316 (66.165 / 67.820) |
| `reverse_15d_3` | 15D 3^15 reverse | 57.067 (56.148 / 57.594) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **54.125 (53.342 / 54.697)** | 284.886 (278.075 / 286.070) | 63.563 (59.763 / 65.541) | - |
| `reverse_23d_2` | 23D 2^23 reverse | **27.150 (26.805 / 27.570)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 43.047 (42.958 / 43.491) | 255.514 (242.179 / 258.729) | 42.177 (26.806 / 44.433) | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | 566.076 (499.797 / 590.930) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **553.923 (509.495 / 575.250)** | 1974.438 (1967.519 / 1979.427) | 687.868 (674.554 / 734.299) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | 23.411 (23.024 / 23.655) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **23.008 (22.755 / 23.453)** | 125.194 (124.536 / 132.361) | 43.056 (37.397 / 47.297) | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | 25.792 (25.644 / 26.001) | - | **25.408 (24.981 / 25.569)** | 340.582 (336.382 / 346.013) | 43.519 (40.353 / 48.263) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | **3.687 (3.623 / 3.784)** | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | 8.089 (7.983 / 8.280) | 53.896 (51.308 / 55.158) | 17.732 (10.513 / 19.369) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | 28.954 (28.657 / 29.160) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **28.446 (28.201 / 28.717)** | 108.390 (106.878 / 111.339) | 43.984 (41.497 / 47.082) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 28.655 (28.271 / 28.965) | skipped (hptt crate exposes only one-shot transpose with per-call plan construction; no prepared-plan API for operation-only timing) | **28.495 (28.168 / 29.554)** | 147.394 (144.941 / 150.173) | 61.979 (56.106 / 63.615) | - |
