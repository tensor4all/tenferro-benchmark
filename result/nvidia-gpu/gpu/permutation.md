# GPU Permutation Benchmark Results

- Target profile: `nvidia-gpu`
- Suite: `gpu/permutation`
- Suite file: `benchmarks/gpu/permutation.yaml`
- Timestamp: `2026-09-16T02:20:19.647943+00:00`
- tenferro-rs commit: `bca2d54a5884586b1aeae8006caa9b8c0ba44a26`

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

Device-copy baseline: `memcpy-d2d` median 5.140 ms, 1671.25 GB/s for `memcpy_24d_64x2`. It is a bandwidth reference, not a permutation participant, so it is not shown as a table column.

## Allocating output

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs CUDA transpose (ms) | tenferro-rs CUDA to_contiguous (ms) |
|---|---|---:|---:|
| `cyclic_18d_3` | 18D 3^18 cyclic [1,2,...,0] | 6.373 (6.030 / 6.412) | **6.299 (6.201 / 6.420)** |
| `reverse_18d_3` | 18D 3^18 reverse | **8.707 (8.630 / 8.789)** | 8.793 (8.638 / 8.799) |
| `reverse_23d_128x2` | 23D 128x2^22 reverse | **7.795 (7.792 / 7.810)** | 8.410 (8.374 / 8.418) |
| `rotation_6d_64_32_32_32_16_16` | 6D 64x32^3x16^2 rotation [5,0,4,1,3,2] | **5.435 (5.411 / 5.565)** | 5.660 (5.600 / 5.713) |
| `tn_light_415_24d_contiguous_same_perm_gpu` | 24D 2^29 contiguous source, TN light 415 permutation | **5.598 (5.592 / 5.642)** | 5.616 (5.589 / 5.709) |
| `tn_light_415_24d_scattered_to_colmajor_gpu` | 24D 2^29 scattered -> col-major | - | **5.637 (5.608 / 5.741)** |
| `transpose_2d_32768_16384` | 2D 32768x16384 transpose [1,0] | **5.594 (5.588 / 5.603)** | 7.014 (7.002 / 7.111) |
| `transpose_3d_1024_1024_512_102` | 3D 1024x1024x512 transpose [1,0,2] | **5.401 (5.380 / 5.412)** | 5.456 (5.433 / 5.560) |
| `transpose_3d_1024_1024_512_201` | 3D 1024x1024x512 transpose [2,0,1] | **5.488 (5.463 / 5.532)** | 7.001 (6.993 / 7.005) |

## Reusing output

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs CUDA copy_read_into (ms) | cuTENSOR (ms) | PyTorch CUDA (ms) |
|---|---|---:|---:|---:|
| `cyclic_18d_3` | 18D 3^18 cyclic [1,2,...,0] | 6.369 (6.145 / 6.409) | **6.249 (6.011 / 6.285)** | 7.355 (7.349 / 7.356) |
| `reverse_18d_3` | 18D 3^18 reverse | 8.693 (8.638 / 8.781) | **8.689 (8.496 / 8.691)** | 33.189 (33.186 / 33.195) |
| `reverse_23d_128x2` | 23D 128x2^22 reverse | 7.793 (7.739 / 7.796) | **7.588 (7.588 / 7.589)** | 45.814 (45.809 / 45.820) |
| `rotation_6d_64_32_32_32_16_16` | 6D 64x32^3x16^2 rotation [5,0,4,1,3,2] | 5.437 (5.420 / 5.511) | **5.351 (5.346 / 5.356)** | 5.934 (5.930 / 5.939) |
| `tn_light_415_24d_contiguous_same_perm_gpu` | 24D 2^29 contiguous source, TN light 415 permutation | 5.594 (5.588 / 5.722) | 5.506 (5.413 / 5.566) | **5.260 (5.258 / 5.263)** |
| `tn_light_415_24d_scattered_to_colmajor_gpu` | 24D 2^29 scattered -> col-major | - | 5.521 (5.403 / 5.553) | **5.402 (5.398 / 5.404)** |
| `transpose_2d_32768_16384` | 2D 32768x16384 transpose [1,0] | 5.597 (5.582 / 5.624) | **5.513 (5.511 / 5.519)** | 20.760 (20.759 / 20.761) |
| `transpose_3d_1024_1024_512_102` | 3D 1024x1024x512 transpose [1,0,2] | 5.402 (5.393 / 5.410) | **5.266 (5.262 / 5.269)** | 7.499 (7.444 / 7.506) |
| `transpose_3d_1024_1024_512_201` | 3D 1024x1024x512 transpose [2,0,1] | 5.398 (5.385 / 5.474) | **5.333 (5.330 / 5.339)** | 27.395 (27.393 / 27.397) |
