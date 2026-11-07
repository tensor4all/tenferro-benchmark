# GPU Permutation Benchmark Results

- Target profile: `nvidia-gpu`
- Suite: `gpu/permutation`
- Suite file: `benchmarks/gpu/permutation.yaml`
- Timestamp: `2026-11-07T20:17:28+00:00`
- tenferro-rs commit: `ffd352bdb9f98537be747101bc278db8a50a1595`

## GPU Information

- Device: `cuda:0`
- Name: `NVIDIA A100-SXM4-80GB`
- UUID: `GPU-3f509591-db26-6474-253e-fcb29d2b374b`
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

Device-copy baseline: `memcpy-d2d` median 4.905 ms, 1751.27 GB/s for `memcpy_24d_64x2`. It is a bandwidth reference, not a permutation participant, so it is not shown as a table column.

## Allocating output

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs CUDA transpose (ms) | tenferro-rs CUDA to_contiguous (ms) |
|---|---|---:|---:|
| `cyclic_18d_3` | 18D 3^18 cyclic [1,2,...,0] | 20.967 (20.965 / 20.975) | **20.957 (20.951 / 20.959)** |
| `reverse_18d_3` | 18D 3^18 reverse | 29.802 (29.795 / 29.817) | **29.784 (29.782 / 29.787)** |
| `reverse_23d_128x2` | 23D 128x2^22 reverse | 41.480 (41.472 / 42.074) | **41.467 (41.461 / 41.915)** |
| `rotation_6d_64_32_32_32_16_16` | 6D 64x32^3x16^2 rotation [5,0,4,1,3,2] | 9.213 (9.211 / 9.219) | **9.198 (9.189 / 9.264)** |
| `tn_light_415_24d_contiguous_same_perm_gpu` | 24D 2^29 contiguous source, TN light 415 permutation | 36.976 (36.853 / 37.533) | **36.862 (36.817 / 37.497)** |
| `tn_light_415_24d_scattered_to_colmajor_gpu` | 24D 2^29 scattered -> col-major | - | **36.845 (36.831 / 37.115)** |
| `transpose_2d_32768_16384` | 2D 32768x16384 transpose [1,0] | 22.940 (22.936 / 22.949) | **22.914 (22.911 / 22.918)** |
| `transpose_3d_1024_1024_512_102` | 3D 1024x1024x512 transpose [1,0,2] | 9.746 (9.742 / 9.750) | **9.712 (9.701 / 9.878)** |
| `transpose_3d_1024_1024_512_201` | 3D 1024x1024x512 transpose [2,0,1] | 29.686 (29.679 / 29.691) | **29.662 (29.657 / 29.665)** |

## Reusing output

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs CUDA copy_read_into (ms) | cuTENSOR (ms) | PyTorch CUDA (ms) |
|---|---|---:|---:|---:|
| `cyclic_18d_3` | 18D 3^18 cyclic [1,2,...,0] | 37.941 (37.940 / 37.947) | **5.954 (5.952 / 5.954)** | 7.009 (7.008 / 7.012) |
| `reverse_18d_3` | 18D 3^18 reverse | 61.458 (61.390 / 61.544) | **8.422 (8.421 / 8.423)** | 32.468 (32.462 / 32.470) |
| `reverse_23d_128x2` | 23D 128x2^22 reverse | 85.144 (85.108 / 85.947) | **7.561 (7.560 / 7.561)** | 44.651 (44.645 / 44.656) |
| `rotation_6d_64_32_32_32_16_16` | 6D 64x32^3x16^2 rotation [5,0,4,1,3,2] | 64.042 (64.024 / 64.084) | **5.098 (5.093 / 5.106)** | 5.686 (5.677 / 5.689) |
| `tn_light_415_24d_contiguous_same_perm_gpu` | 24D 2^29 contiguous source, TN light 415 permutation | 69.041 (68.898 / 69.585) | 5.279 (5.278 / 5.281) | **5.017 (5.016 / 5.018)** |
| `tn_light_415_24d_scattered_to_colmajor_gpu` | 24D 2^29 scattered -> col-major | - | 5.280 (5.277 / 5.281) | **5.153 (5.149 / 5.154)** |
| `transpose_2d_32768_16384` | 2D 32768x16384 transpose [1,0] | 25.172 (25.169 / 25.178) | **5.270 (5.265 / 5.274)** | 20.767 (20.765 / 20.771) |
| `transpose_3d_1024_1024_512_102` | 3D 1024x1024x512 transpose [1,0,2] | 16.459 (16.442 / 16.469) | **5.009 (5.002 / 5.015)** | 7.324 (7.322 / 7.325) |
| `transpose_3d_1024_1024_512_201` | 3D 1024x1024x512 transpose [2,0,1] | 85.167 (85.138 / 85.256) | **5.092 (5.088 / 5.093)** | 27.406 (27.403 / 27.408) |
