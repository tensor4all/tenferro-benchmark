# GPU Permutation Benchmark Results

- Target profile: `nvidia-gpu`
- Suite: `gpu/permutation`
- Suite file: `benchmarks/gpu/permutation.yaml`
- Timestamp: `2026-09-19T08:27:09.982757+00:00`
- tenferro-rs commit: `40e24634f9aa814db82efd9faae5e99a8fcee8c4`

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

Device-copy baseline: `memcpy-d2d` median 5.133 ms, 1673.41 GB/s for `memcpy_24d_64x2`. It is a bandwidth reference, not a permutation participant, so it is not shown as a table column.

## Allocating output

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs CUDA transpose (ms) | tenferro-rs CUDA to_contiguous (ms) |
|---|---|---:|---:|
| `cyclic_18d_3` | 18D 3^18 cyclic [1,2,...,0] | 6.381 (6.197 / 6.401) | **6.377 (6.130 / 6.407)** |
| `reverse_18d_3` | 18D 3^18 reverse | 8.792 (8.787 / 8.820) | **8.788 (8.644 / 8.828)** |
| `reverse_23d_128x2` | 23D 128x2^22 reverse | **7.737 (7.685 / 7.784)** | 8.371 (8.353 / 8.427) |
| `rotation_6d_64_32_32_32_16_16` | 6D 64x32^3x16^2 rotation [5,0,4,1,3,2] | **5.518 (5.470 / 5.533)** | 5.646 (5.603 / 5.713) |
| `tn_light_415_24d_contiguous_same_perm_gpu` | 24D 2^29 contiguous source, TN light 415 permutation | **5.608 (5.594 / 5.635)** | 5.629 (5.589 / 5.796) |
| `tn_light_415_24d_scattered_to_colmajor_gpu` | 24D 2^29 scattered -> col-major | - | **5.629 (5.604 / 5.700)** |
| `transpose_2d_32768_16384` | 2D 32768x16384 transpose [1,0] | **5.620 (5.603 / 5.715)** | 7.188 (7.022 / 7.202) |
| `transpose_3d_1024_1024_512_102` | 3D 1024x1024x512 transpose [1,0,2] | **5.436 (5.414 / 5.469)** | 5.530 (5.472 / 5.580) |
| `transpose_3d_1024_1024_512_201` | 3D 1024x1024x512 transpose [2,0,1] | **5.405 (5.395 / 5.433)** | 6.999 (6.996 / 7.001) |

## Reusing output

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs CUDA copy_read_into (ms) | cuTENSOR (ms) | PyTorch CUDA (ms) |
|---|---|---:|---:|---:|
| `cyclic_18d_3` | 18D 3^18 cyclic [1,2,...,0] | 6.381 (6.206 / 6.412) | **6.218 (6.059 / 6.285)** | 7.354 (7.351 / 7.357) |
| `reverse_18d_3` | 18D 3^18 reverse | 8.790 (8.641 / 8.800) | **8.688 (8.588 / 8.691)** | 33.190 (33.182 / 33.197) |
| `reverse_23d_128x2` | 23D 128x2^22 reverse | 7.733 (7.679 / 7.786) | **7.586 (7.586 / 7.587)** | 45.770 (45.764 / 45.774) |
| `rotation_6d_64_32_32_32_16_16` | 6D 64x32^3x16^2 rotation [5,0,4,1,3,2] | 5.414 (5.393 / 5.427) | **5.346 (5.344 / 5.349)** | 5.944 (5.938 / 5.949) |
| `tn_light_415_24d_contiguous_same_perm_gpu` | 24D 2^29 contiguous source, TN light 415 permutation | 5.595 (5.539 / 5.638) | 5.525 (5.404 / 5.570) | **5.259 (5.258 / 5.261)** |
| `tn_light_415_24d_scattered_to_colmajor_gpu` | 24D 2^29 scattered -> col-major | - | 5.521 (5.401 / 5.564) | **5.400 (5.397 / 5.402)** |
| `transpose_2d_32768_16384` | 2D 32768x16384 transpose [1,0] | 5.593 (5.579 / 5.610) | **5.519 (5.508 / 5.524)** | 20.761 (20.759 / 20.762) |
| `transpose_3d_1024_1024_512_102` | 3D 1024x1024x512 transpose [1,0,2] | 5.401 (5.370 / 5.422) | **5.269 (5.265 / 5.272)** | 7.492 (7.441 / 7.501) |
| `transpose_3d_1024_1024_512_201` | 3D 1024x1024x512 transpose [2,0,1] | 5.411 (5.386 / 5.433) | **5.337 (5.333 / 5.342)** | 27.393 (27.392 / 27.396) |
