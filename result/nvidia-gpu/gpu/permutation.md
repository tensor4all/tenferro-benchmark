# GPU Permutation Benchmark Results

- Target profile: `nvidia-gpu`
- Suite: `gpu/permutation`
- Suite file: `benchmarks/gpu/permutation.yaml`
- Timestamp: `2026-09-17T21:11:01.160938+00:00`
- tenferro-rs commit: `292cdffef8d910abb88dca44cd3c928bc6051005`

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

Device-copy baseline: `memcpy-d2d` median 5.140 ms, 1671.12 GB/s for `memcpy_24d_64x2`. It is a bandwidth reference, not a permutation participant, so it is not shown as a table column.

## Allocating output

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs CUDA transpose (ms) | tenferro-rs CUDA to_contiguous (ms) |
|---|---|---:|---:|
| `cyclic_18d_3` | 18D 3^18 cyclic [1,2,...,0] | 6.376 (6.104 / 6.408) | **6.215 (6.197 / 6.380)** |
| `reverse_18d_3` | 18D 3^18 reverse | 8.793 (8.693 / 8.814) | **8.783 (8.640 / 8.808)** |
| `reverse_23d_128x2` | 23D 128x2^22 reverse | **7.724 (7.667 / 7.752)** | 8.404 (8.387 / 8.409) |
| `rotation_6d_64_32_32_32_16_16` | 6D 64x32^3x16^2 rotation [5,0,4,1,3,2] | **5.441 (5.419 / 5.563)** | 5.614 (5.591 / 5.661) |
| `tn_light_415_24d_contiguous_same_perm_gpu` | 24D 2^29 contiguous source, TN light 415 permutation | **5.619 (5.593 / 5.678)** | 5.620 (5.602 / 5.791) |
| `tn_light_415_24d_scattered_to_colmajor_gpu` | 24D 2^29 scattered -> col-major | - | **5.672 (5.612 / 5.793)** |
| `transpose_2d_32768_16384` | 2D 32768x16384 transpose [1,0] | **5.624 (5.607 / 5.675)** | 7.161 (7.109 / 7.201) |
| `transpose_3d_1024_1024_512_102` | 3D 1024x1024x512 transpose [1,0,2] | **5.438 (5.378 / 5.508)** | 5.567 (5.426 / 5.584) |
| `transpose_3d_1024_1024_512_201` | 3D 1024x1024x512 transpose [2,0,1] | **5.409 (5.396 / 5.561)** | 7.009 (6.997 / 7.104) |

## Reusing output

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs CUDA copy_read_into (ms) | cuTENSOR (ms) | PyTorch CUDA (ms) |
|---|---|---:|---:|---:|
| `cyclic_18d_3` | 18D 3^18 cyclic [1,2,...,0] | 6.379 (6.116 / 6.412) | **6.284 (6.080 / 6.284)** | 7.354 (7.351 / 7.357) |
| `reverse_18d_3` | 18D 3^18 reverse | 8.795 (8.767 / 8.821) | **8.687 (8.494 / 8.689)** | 33.200 (33.192 / 33.206) |
| `reverse_23d_128x2` | 23D 128x2^22 reverse | 7.706 (7.682 / 7.734) | **7.587 (7.586 / 7.587)** | 45.697 (45.694 / 45.702) |
| `rotation_6d_64_32_32_32_16_16` | 6D 64x32^3x16^2 rotation [5,0,4,1,3,2] | 5.426 (5.383 / 5.498) | **5.347 (5.344 / 5.353)** | 5.951 (5.946 / 5.958) |
| `tn_light_415_24d_contiguous_same_perm_gpu` | 24D 2^29 contiguous source, TN light 415 permutation | 5.609 (5.594 / 5.631) | 5.517 (5.424 / 5.567) | **5.260 (5.258 / 5.261)** |
| `tn_light_415_24d_scattered_to_colmajor_gpu` | 24D 2^29 scattered -> col-major | - | 5.491 (5.407 / 5.565) | **5.412 (5.408 / 5.413)** |
| `transpose_2d_32768_16384` | 2D 32768x16384 transpose [1,0] | 5.675 (5.602 / 5.688) | **5.517 (5.514 / 5.525)** | 20.761 (20.760 / 20.762) |
| `transpose_3d_1024_1024_512_102` | 3D 1024x1024x512 transpose [1,0,2] | 5.398 (5.388 / 5.408) | **5.270 (5.264 / 5.275)** | 7.525 (7.471 / 7.528) |
| `transpose_3d_1024_1024_512_201` | 3D 1024x1024x512 transpose [2,0,1] | 5.417 (5.376 / 5.429) | **5.332 (5.328 / 5.340)** | 27.396 (27.393 / 27.398) |
