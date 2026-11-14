# GPU Permutation Benchmark Results

- Target profile: `nvidia-gpu`
- Suite: `gpu/permutation`
- Suite file: `benchmarks/gpu/permutation.yaml`
- Timestamp: `2026-11-14T08:28:52+00:00`
- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`

## GPU Information

- Device: `cuda:0`
- Name: `NVIDIA A100 80GB PCIe`
- UUID: `GPU-c70851ec-1874-5bb5-8a0a-85e4eed17212`
- Memory: `80 GiB`
- Driver version: `595.91.07`
- CUDA version: `13.2`
- CUDA runtime: `12.8`
- cuDNN version: `91900`
- Container: `devcontainer`
- CUDA toolkit: `unknown`
- CUDA toolkit root: `/opt/tenferro-benchmark-ci/cuda`
- cuTENSOR version: `2.2.0`
- cuSOLVER version: `11.7.3`

`tenferro-cuda-transpose` is the eager `TensorStructural::transpose` op on `CudaBackend` (compact col-major input only); `tenferro-cuda-to-contiguous` is `TypedTensor::backend_region_view` (source layout) + `TypedTensorView::transpose_view(perm)` + `TensorViewCanonicalization::to_contiguous` (accepts arbitrary source strides). For framework comparisons, `tenferro-cuda-to-contiguous` is the primary like-for-like column for PyTorch's view/permute-then-materialize semantics; allocation and reuse timings are shown separately. `tenferro-cuda-transpose` is the direct structural-permutation comparison for primitive/kernel-oriented backends such as cuTENSOR. Both public tenferro columns allocate a fresh device tensor on every call. `tenferro-cuda-destination-reuse` calls the public `CudaBackend::copy_read_into` override from a compact source into an inverse-permuted caller-owned destination view allocated once outside timing. The production path uses the native CUDA copy kernel, so it is distinct from the raw cuTENSOR control. Its per-call destination-view metadata construction is included in the timed public API dispatch. `cutensor`, `pytorch-cuda`, and `memcpy-d2d` also reuse a destination buffer allocated once per pattern. `memcpy-d2d` only participates in the contiguous identity-permutation baseline pattern. Correctness is verified against a host-computed naive reference, downloaded outside any timed region, before any timing; a `FAILED` cell means that backend's output did not match the reference for that pattern. A `skipped` cell means the backend does not participate in that pattern's semantics (or, for `cutensor`, that the installed cuTENSOR library rejected the pattern's rank at runtime) -- both are reported as `skipped` rather than a failure.

The GPU-only pattern set is intentionally larger than the CPU permutation set: most rows contain 2^29 f64 elements (4 GiB per tensor) so A100-class measurements exercise steady-state device throughput at roughly the 10 ms scale instead of launch/synchronization overhead.

Device-copy baseline: `memcpy-d2d` median 5.189 ms, 1655.52 GB/s for `memcpy_24d_64x2`. It is a bandwidth reference, not a permutation participant, so it is not shown as a table column.

## Allocating output

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs CUDA transpose (ms) | tenferro-rs CUDA to_contiguous (ms) |
|---|---|---:|---:|
| `cyclic_18d_3` | 18D 3^18 cyclic [1,2,...,0] | **21.109 (21.104 / 21.113)** | 21.153 (21.110 / 21.684) |
| `reverse_18d_3` | 18D 3^18 reverse | 30.294 (30.261 / 30.451) | **30.251 (30.239 / 30.448)** |
| `reverse_23d_128x2` | 23D 128x2^22 reverse | **42.264 (42.218 / 42.434)** | 42.486 (42.186 / 43.308) |
| `rotation_6d_64_32_32_32_16_16` | 6D 64x32^3x16^2 rotation [5,0,4,1,3,2] | 9.479 (9.464 / 9.551) | **9.456 (9.444 / 9.552)** |
| `tn_light_415_24d_contiguous_same_perm_gpu` | 24D 2^29 contiguous source, TN light 415 permutation | 37.229 (37.134 / 38.501) | **37.209 (37.093 / 37.645)** |
| `tn_light_415_24d_scattered_to_colmajor_gpu` | 24D 2^29 scattered -> col-major | - | **37.119 (37.076 / 37.513)** |
| `transpose_2d_32768_16384` | 2D 32768x16384 transpose [1,0] | 23.125 (23.117 / 23.147) | **23.114 (23.087 / 23.239)** |
| `transpose_3d_1024_1024_512_102` | 3D 1024x1024x512 transpose [1,0,2] | **10.125 (10.086 / 10.329)** | 10.149 (10.063 / 10.562) |
| `transpose_3d_1024_1024_512_201` | 3D 1024x1024x512 transpose [2,0,1] | **29.861 (29.847 / 29.908)** | 29.954 (29.859 / 30.124) |

## Reusing output

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs CUDA copy_read_into (ms) | cuTENSOR (ms) | PyTorch CUDA (ms) |
|---|---|---:|---:|---:|
| `cyclic_18d_3` | 18D 3^18 cyclic [1,2,...,0] | 38.117 (38.000 / 39.529) | **6.181 (6.054 / 6.199)** | 7.491 (7.456 / 8.081) |
| `reverse_18d_3` | 18D 3^18 reverse | 61.457 (61.344 / 62.232) | **8.529 (8.462 / 8.559)** | 33.299 (33.288 / 33.374) |
| `reverse_23d_128x2` | 23D 128x2^22 reverse | 85.906 (85.288 / 87.335) | **7.597 (7.591 / 7.645)** | 45.953 (45.803 / 47.025) |
| `rotation_6d_64_32_32_32_16_16` | 6D 64x32^3x16^2 rotation [5,0,4,1,3,2] | 64.757 (64.500 / 65.110) | **5.449 (5.389 / 5.813)** | 5.977 (5.970 / 5.997) |
| `tn_light_415_24d_contiguous_same_perm_gpu` | 24D 2^29 contiguous source, TN light 415 permutation | 71.268 (69.653 / 74.400) | 5.479 (5.431 / 5.509) | **5.311 (5.305 / 5.317)** |
| `tn_light_415_24d_scattered_to_colmajor_gpu` | 24D 2^29 scattered -> col-major | - | 5.514 (5.440 / 5.528) | **5.452 (5.436 / 5.510)** |
| `transpose_2d_32768_16384` | 2D 32768x16384 transpose [1,0] | 25.648 (25.632 / 25.900) | **5.544 (5.535 / 5.615)** | 21.231 (20.984 / 21.616) |
| `transpose_3d_1024_1024_512_102` | 3D 1024x1024x512 transpose [1,0,2] | 16.743 (16.733 / 16.766) | **5.264 (5.257 / 5.271)** | 7.472 (7.442 / 7.484) |
| `transpose_3d_1024_1024_512_201` | 3D 1024x1024x512 transpose [2,0,1] | 85.180 (85.028 / 85.334) | **5.362 (5.349 / 5.426)** | 27.519 (27.510 / 27.552) |
