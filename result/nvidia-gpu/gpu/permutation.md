# GPU Permutation Benchmark Results

- Target profile: `nvidia-gpu`
- Suite: `gpu/permutation`
- Suite file: `benchmarks/gpu/permutation.yaml`
- Timestamp: `2026-10-06T19:49:20.873768+00:00`
- tenferro-rs commit: `5cf78c7ec0ad9516dd78bab546d5e7bd42fa3102`

## GPU Information

- Device: `cuda:0`
- Name: `NVIDIA A100 80GB PCIe`
- UUID: `GPU-530977e1-4968-9283-4129-9fbec3e66542`
- Memory: `80 GiB`
- Driver version: `580.126.09`
- CUDA version: `13.0`
- CUDA runtime: `12.9`
- cuDNN version: `92700`

`tenferro-cuda-transpose` is the eager `TensorStructural::transpose` op on `CudaBackend` (compact col-major input only); `tenferro-cuda-to-contiguous` is `TypedTensor::backend_region_view` (source layout) + `TypedTensorView::transpose_view(perm)` + `TensorViewCanonicalization::to_contiguous` (accepts arbitrary source strides). For framework comparisons, `tenferro-cuda-to-contiguous` is the primary like-for-like column for PyTorch's view/permute-then-materialize semantics; allocation and reuse timings are shown separately. `tenferro-cuda-transpose` is the direct structural-permutation comparison for primitive/kernel-oriented backends such as cuTENSOR. Both public tenferro columns allocate a fresh device tensor on every call. `tenferro-cuda-destination-reuse` calls the public `CudaBackend::copy_read_into` override from a compact source into an inverse-permuted caller-owned destination view allocated once outside timing. The production path uses the native CUDA copy kernel, so it is distinct from the raw cuTENSOR control. Its per-call destination-view metadata construction is included in the timed public API dispatch. `cutensor`, `pytorch-cuda`, and `memcpy-d2d` also reuse a destination buffer allocated once per pattern. `memcpy-d2d` only participates in the contiguous identity-permutation baseline pattern. Correctness is verified against a host-computed naive reference, downloaded outside any timed region, before any timing; a `FAILED` cell means that backend's output did not match the reference for that pattern. A `skipped` cell means the backend does not participate in that pattern's semantics (or, for `cutensor`, that the installed cuTENSOR library rejected the pattern's rank at runtime) -- both are reported as `skipped` rather than a failure.

The GPU-only pattern set is intentionally larger than the CPU permutation set: most rows contain 2^29 f64 elements (4 GiB per tensor) so A100-class measurements exercise steady-state device throughput at roughly the 10 ms scale instead of launch/synchronization overhead.

Device-copy baseline: `memcpy-d2d` median 5.132 ms, 1673.74 GB/s for `memcpy_24d_64x2`. It is a bandwidth reference, not a permutation participant, so it is not shown as a table column.

## Allocating output

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs CUDA transpose (ms) | tenferro-rs CUDA to_contiguous (ms) |
|---|---|---:|---:|
| `cyclic_18d_3` | 18D 3^18 cyclic [1,2,...,0] | **6.340 (6.165 / 6.358)** | 6.359 (6.130 / 6.362) |
| `reverse_18d_3` | 18D 3^18 reverse | 8.727 (8.606 / 8.731) | **8.726 (8.594 / 8.735)** |
| `reverse_23d_128x2` | 23D 128x2^22 reverse | **7.674 (7.673 / 7.676)** | 8.323 (8.320 / 8.325) |
| `rotation_6d_64_32_32_32_16_16` | 6D 64x32^3x16^2 rotation [5,0,4,1,3,2] | **5.413 (5.409 / 5.415)** | 5.571 (5.561 / 5.578) |
| `tn_light_415_24d_contiguous_same_perm_gpu` | 24D 2^29 contiguous source, TN light 415 permutation | **5.562 (5.455 / 5.605)** | 5.589 (5.502 / 5.653) |
| `tn_light_415_24d_scattered_to_colmajor_gpu` | 24D 2^29 scattered -> col-major | - | **5.647 (5.495 / 5.682)** |
| `transpose_2d_32768_16384` | 2D 32768x16384 transpose [1,0] | **5.578 (5.573 / 5.610)** | 7.017 (7.013 / 7.023) |
| `transpose_3d_1024_1024_512_102` | 3D 1024x1024x512 transpose [1,0,2] | **5.370 (5.367 / 5.377)** | 5.466 (5.461 / 5.473) |
| `transpose_3d_1024_1024_512_201` | 3D 1024x1024x512 transpose [2,0,1] | **5.443 (5.415 / 5.453)** | 6.899 (6.897 / 6.902) |

## Reusing output

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs CUDA copy_read_into (ms) | cuTENSOR (ms) | PyTorch CUDA (ms) |
|---|---|---:|---:|---:|
| `cyclic_18d_3` | 18D 3^18 cyclic [1,2,...,0] | 6.320 (6.118 / 6.321) | **6.284 (6.080 / 6.285)** | 7.357 (7.351 / 7.361) |
| `reverse_18d_3` | 18D 3^18 reverse | 8.761 (8.616 / 8.764) | **8.688 (8.676 / 8.690)** | 33.187 (33.185 / 33.191) |
| `reverse_23d_128x2` | 23D 128x2^22 reverse | 7.658 (7.657 / 7.660) | **7.586 (7.586 / 7.587)** | 45.790 (45.788 / 45.793) |
| `rotation_6d_64_32_32_32_16_16` | 6D 64x32^3x16^2 rotation [5,0,4,1,3,2] | 5.377 (5.360 / 5.384) | **5.336 (5.334 / 5.343)** | 5.937 (5.926 / 5.946) |
| `tn_light_415_24d_contiguous_same_perm_gpu` | 24D 2^29 contiguous source, TN light 415 permutation | 5.584 (5.512 / 5.614) | 5.490 (5.402 / 5.568) | **5.259 (5.258 / 5.262)** |
| `tn_light_415_24d_scattered_to_colmajor_gpu` | 24D 2^29 scattered -> col-major | - | 5.562 (5.442 / 5.565) | **5.404 (5.398 / 5.407)** |
| `transpose_2d_32768_16384` | 2D 32768x16384 transpose [1,0] | 5.574 (5.568 / 5.581) | **5.511 (5.507 / 5.516)** | 20.761 (20.760 / 20.763) |
| `transpose_3d_1024_1024_512_102` | 3D 1024x1024x512 transpose [1,0,2] | 5.317 (5.314 / 5.324) | **5.261 (5.253 / 5.264)** | 7.496 (7.444 / 7.501) |
| `transpose_3d_1024_1024_512_201` | 3D 1024x1024x512 transpose [2,0,1] | 5.389 (5.382 / 5.395) | **5.320 (5.312 / 5.323)** | 27.400 (27.397 / 27.404) |
