# GPU Permutation Benchmark Results

- Target profile: `nvidia-gpu`
- Suite: `gpu/permutation`
- Suite file: `benchmarks/gpu/permutation.yaml`
- Timestamp: `2026-09-13T08:36:27.578973+00:00`
- tenferro-rs commit: `28dfc7e383a653e1461d38b73ced3d563b4efdad`

## GPU Information

- Device: `cuda:0`
- Name: `NVIDIA A100 80GB PCIe`
- UUID: `GPU-530977e1-4968-9283-4129-9fbec3e66542`
- Memory: `80 GiB`
- Driver version: `580.126.09`
- CUDA version: `13.0`
- CUDA runtime: `12.9`
- cuDNN version: `92000`

`tenferro-cuda-transpose` is the eager `TensorStructural::transpose` op on `CudaBackend` (compact col-major input only); `tenferro-cuda-to-contiguous` is `TypedTensor::backend_region_view` (source layout) + `TypedTensorView::transpose_view(perm)` + `TensorViewCanonicalization::to_contiguous` (accepts arbitrary source strides). For framework comparisons, `tenferro-cuda-to-contiguous` is the primary like-for-like column for PyTorch's view/permute-then-materialize semantics; allocation and reuse timings are shown separately. `tenferro-cuda-transpose` is the direct structural-permutation comparison for primitive/kernel-oriented backends such as cuTENSOR. Both public tenferro columns allocate a fresh device tensor on every call. `tenferro-cuda-destination-reuse` calls the public `CudaBackend::copy_read_into` override from a compact source into an inverse-permuted caller-owned destination view allocated once outside timing. The production path uses the native CUDA copy kernel, so it is distinct from the raw cuTENSOR control. Its per-call destination-view metadata construction is included in the timed public API dispatch. `cutensor`, `pytorch-cuda`, and `memcpy-d2d` also reuse a destination buffer allocated once per pattern. `memcpy-d2d` only participates in the contiguous identity-permutation baseline pattern. Correctness is verified against a host-computed naive reference, downloaded outside any timed region, before any timing; a `FAILED` cell means that backend's output did not match the reference for that pattern. A `skipped` cell means the backend does not participate in that pattern's semantics (or, for `cutensor`, that the installed cuTENSOR library rejected the pattern's rank at runtime) -- both are reported as `skipped` rather than a failure.

The GPU-only pattern set is intentionally larger than the CPU permutation set: most rows contain 2^29 f64 elements (4 GiB per tensor) so A100-class measurements exercise steady-state device throughput at roughly the 10 ms scale instead of launch/synchronization overhead.

Device-copy baseline: `memcpy-d2d` median 5.134 ms, 1673.22 GB/s for `memcpy_24d_64x2`. It is a bandwidth reference, not a permutation participant, so it is not shown as a table column.

## Allocating output

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs CUDA transpose (ms) | tenferro-rs CUDA to_contiguous (ms) |
|---|---|---:|---:|
| `cyclic_18d_3` | 18D 3^18 cyclic [1,2,...,0] | **6.237 (6.014 / 6.404)** | 6.383 (6.191 / 6.405) |
| `reverse_18d_3` | 18D 3^18 reverse | 8.708 (8.613 / 8.796) | **8.651 (8.614 / 8.797)** |
| `reverse_23d_128x2` | 23D 128x2^22 reverse | **7.788 (7.624 / 7.795)** | 8.400 (8.385 / 8.410) |
| `rotation_6d_64_32_32_32_16_16` | 6D 64x32^3x16^2 rotation [5,0,4,1,3,2] | **5.417 (5.405 / 5.434)** | 5.599 (5.584 / 5.620) |
| `tn_light_415_24d_contiguous_same_perm_gpu` | 24D 2^29 contiguous source, TN light 415 permutation | **5.612 (5.594 / 5.623)** | 5.626 (5.589 / 5.783) |
| `tn_light_415_24d_scattered_to_colmajor_gpu` | 24D 2^29 scattered -> col-major | - | **5.626 (5.595 / 5.686)** |
| `transpose_2d_32768_16384` | 2D 32768x16384 transpose [1,0] | **5.600 (5.580 / 5.616)** | 7.060 (7.001 / 7.118) |
| `transpose_3d_1024_1024_512_102` | 3D 1024x1024x512 transpose [1,0,2] | **5.393 (5.378 / 5.409)** | 5.553 (5.455 / 5.568) |
| `transpose_3d_1024_1024_512_201` | 3D 1024x1024x512 transpose [2,0,1] | **5.415 (5.406 / 5.447)** | 6.999 (6.996 / 7.004) |

## Reusing output

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs CUDA copy_read_into (ms) | cuTENSOR (ms) | PyTorch CUDA (ms) |
|---|---|---:|---:|---:|
| `cyclic_18d_3` | 18D 3^18 cyclic [1,2,...,0] | 6.385 (6.052 / 6.411) | **6.239 (5.949 / 6.280)** | 7.355 (7.353 / 7.357) |
| `reverse_18d_3` | 18D 3^18 reverse | 8.723 (8.614 / 8.793) | **8.685 (8.582 / 8.686)** | 33.190 (33.185 / 33.192) |
| `reverse_23d_128x2` | 23D 128x2^22 reverse | 7.794 (7.618 / 7.795) | **7.583 (7.583 / 7.584)** | 45.814 (45.812 / 45.820) |
| `rotation_6d_64_32_32_32_16_16` | 6D 64x32^3x16^2 rotation [5,0,4,1,3,2] | 5.403 (5.386 / 5.410) | **5.344 (5.338 / 5.348)** | 5.934 (5.927 / 5.942) |
| `tn_light_415_24d_contiguous_same_perm_gpu` | 24D 2^29 contiguous source, TN light 415 permutation | 5.595 (5.580 / 5.633) | 5.486 (5.396 / 5.560) | **5.260 (5.259 / 5.262)** |
| `tn_light_415_24d_scattered_to_colmajor_gpu` | 24D 2^29 scattered -> col-major | - | 5.484 (5.399 / 5.562) | **5.398 (5.396 / 5.401)** |
| `transpose_2d_32768_16384` | 2D 32768x16384 transpose [1,0] | 5.596 (5.582 / 5.615) | **5.512 (5.504 / 5.521)** | 20.760 (20.759 / 20.762) |
| `transpose_3d_1024_1024_512_102` | 3D 1024x1024x512 transpose [1,0,2] | 5.394 (5.379 / 5.427) | **5.263 (5.259 / 5.267)** | 7.496 (7.443 / 7.509) |
| `transpose_3d_1024_1024_512_201` | 3D 1024x1024x512 transpose [2,0,1] | 5.392 (5.367 / 5.434) | **5.330 (5.324 / 5.334)** | 27.394 (27.390 / 27.397) |
