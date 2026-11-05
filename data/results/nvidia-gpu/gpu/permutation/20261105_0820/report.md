# GPU Permutation Benchmark Results

- Target profile: `nvidia-gpu`
- Suite: `gpu/permutation`
- Suite file: `benchmarks/gpu/permutation.yaml`
- Timestamp: `2026-11-05T08:02:00+00:00`
- tenferro-rs commit: `a000e5c4929d5bebb91e783c329b27106baed12e`

## GPU Information

- Device: `cuda:0`
- Name: `NVIDIA A100 80GB PCIe`
- UUID: `GPU-530977e1-4968-9283-4129-9fbec3e66542`
- Memory: `80 GiB`
- Driver version: `580.178.04`
- CUDA version: `13.0`
- CUDA runtime: `12.6`
- cuDNN version: `92700`
- Container: `none`
- CUDA toolkit: `12.6`
- CUDA toolkit root: `/usr/local/cuda-12.6`
- cuTENSOR version: `2.8.0`
- cuSOLVER version: `11.7.1`

`tenferro-cuda-transpose` is the eager `TensorStructural::transpose` op on `CudaBackend` (compact col-major input only); `tenferro-cuda-to-contiguous` is `TypedTensor::backend_region_view` (source layout) + `TypedTensorView::transpose_view(perm)` + `TensorViewCanonicalization::to_contiguous` (accepts arbitrary source strides). For framework comparisons, `tenferro-cuda-to-contiguous` is the primary like-for-like column for PyTorch's view/permute-then-materialize semantics; allocation and reuse timings are shown separately. `tenferro-cuda-transpose` is the direct structural-permutation comparison for primitive/kernel-oriented backends such as cuTENSOR. Both public tenferro columns allocate a fresh device tensor on every call. `tenferro-cuda-destination-reuse` calls the public `CudaBackend::copy_read_into` override from a compact source into an inverse-permuted caller-owned destination view allocated once outside timing. The production path uses the native CUDA copy kernel, so it is distinct from the raw cuTENSOR control. Its per-call destination-view metadata construction is included in the timed public API dispatch. `cutensor`, `pytorch-cuda`, and `memcpy-d2d` also reuse a destination buffer allocated once per pattern. `memcpy-d2d` only participates in the contiguous identity-permutation baseline pattern. Correctness is verified against a host-computed naive reference, downloaded outside any timed region, before any timing; a `FAILED` cell means that backend's output did not match the reference for that pattern. A `skipped` cell means the backend does not participate in that pattern's semantics (or, for `cutensor`, that the installed cuTENSOR library rejected the pattern's rank at runtime) -- both are reported as `skipped` rather than a failure.

The GPU-only pattern set is intentionally larger than the CPU permutation set: most rows contain 2^29 f64 elements (4 GiB per tensor) so A100-class measurements exercise steady-state device throughput at roughly the 10 ms scale instead of launch/synchronization overhead.

Device-copy baseline: `memcpy-d2d` median 5.135 ms, 1672.92 GB/s for `memcpy_24d_64x2`. It is a bandwidth reference, not a permutation participant, so it is not shown as a table column.

## Allocating output

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs CUDA transpose (ms) | tenferro-rs CUDA to_contiguous (ms) |
|---|---|---:|---:|
| `cyclic_18d_3` | 18D 3^18 cyclic [1,2,...,0] | 22.182 (22.176 / 22.185) | **22.172 (22.169 / 22.175)** |
| `reverse_18d_3` | 18D 3^18 reverse | 30.180 (30.175 / 30.187) | **30.156 (30.149 / 30.166)** |
| `reverse_23d_128x2` | 23D 128x2^22 reverse | 41.960 (41.957 / 41.967) | **41.941 (41.932 / 41.952)** |
| `rotation_6d_64_32_32_32_16_16` | 6D 64x32^3x16^2 rotation [5,0,4,1,3,2] | **9.447 (9.432 / 9.513)** | 9.533 (9.488 / 9.583) |
| `tn_light_415_24d_contiguous_same_perm_gpu` | 24D 2^29 contiguous source, TN light 415 permutation | 38.906 (38.901 / 38.914) | **38.886 (38.885 / 38.888)** |
| `tn_light_415_24d_scattered_to_colmajor_gpu` | 24D 2^29 scattered -> col-major | - | **38.878 (38.871 / 38.882)** |
| `transpose_2d_32768_16384` | 2D 32768x16384 transpose [1,0] | 23.034 (23.031 / 23.039) | **23.006 (23.003 / 23.011)** |
| `transpose_3d_1024_1024_512_102` | 3D 1024x1024x512 transpose [1,0,2] | 10.109 (10.101 / 10.115) | **10.074 (10.065 / 10.277)** |
| `transpose_3d_1024_1024_512_201` | 3D 1024x1024x512 transpose [2,0,1] | 29.772 (29.768 / 29.775) | **29.749 (29.746 / 29.753)** |

## Reusing output

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs CUDA copy_read_into (ms) | cuTENSOR (ms) | PyTorch CUDA (ms) |
|---|---|---:|---:|---:|
| `cyclic_18d_3` | 18D 3^18 cyclic [1,2,...,0] | 38.548 (38.547 / 38.549) | **4.778 (4.633 / 4.852)** | 7.357 (7.353 / 7.359) |
| `reverse_18d_3` | 18D 3^18 reverse | 61.120 (61.015 / 61.165) | **5.028 (5.021 / 5.034)** | 33.195 (33.191 / 33.203) |
| `reverse_23d_128x2` | 23D 128x2^22 reverse | 84.839 (84.777 / 84.918) | **5.647 (5.646 / 5.653)** | 45.751 (45.748 / 45.754) |
| `rotation_6d_64_32_32_32_16_16` | 6D 64x32^3x16^2 rotation [5,0,4,1,3,2] | 64.686 (64.656 / 65.211) | **5.401 (5.396 / 5.405)** | 5.937 (5.929 / 5.945) |
| `tn_light_415_24d_contiguous_same_perm_gpu` | 24D 2^29 contiguous source, TN light 415 permutation | 70.021 (70.020 / 70.026) | **5.206 (5.202 / 5.210)** | 5.259 (5.257 / 5.260) |
| `tn_light_415_24d_scattered_to_colmajor_gpu` | 24D 2^29 scattered -> col-major | - | 5.475 (5.472 / 5.481) | **5.398 (5.397 / 5.403)** |
| `transpose_2d_32768_16384` | 2D 32768x16384 transpose [1,0] | 25.584 (25.568 / 25.598) | **5.398 (5.395 / 5.400)** | 20.759 (20.758 / 20.762) |
| `transpose_3d_1024_1024_512_102` | 3D 1024x1024x512 transpose [1,0,2] | 17.768 (17.764 / 17.782) | **5.311 (5.306 / 5.313)** | 7.436 (7.430 / 7.437) |
| `transpose_3d_1024_1024_512_201` | 3D 1024x1024x512 transpose [2,0,1] | 84.987 (84.870 / 85.099) | **5.337 (5.333 / 5.341)** | 27.395 (27.393 / 27.397) |
