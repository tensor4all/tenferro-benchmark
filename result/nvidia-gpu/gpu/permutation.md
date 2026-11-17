# GPU Permutation Benchmark Results

- Target profile: `nvidia-gpu`
- Suite: `gpu/permutation`
- Suite file: `benchmarks/gpu/permutation.yaml`
- Timestamp: `2026-11-17T09:15:52+00:00`
- tenferro-rs commit: `ce9125d0cff4ee40841120759b5df45252c27f22`

## GPU Information

- Device: `cuda:0`
- Name: `NVIDIA A100-SXM4-80GB`
- UUID: `GPU-704abe6c-29a2-539e-7ccd-01f9de4eed3b`
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

Device-copy baseline: `memcpy-d2d` median 4.902 ms, 1752.17 GB/s for `memcpy_24d_64x2`. It is a bandwidth reference, not a permutation participant, so it is not shown as a table column.

## Allocating output

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs CUDA transpose (ms) | tenferro-rs CUDA to_contiguous (ms) |
|---|---|---:|---:|
| `cyclic_18d_3` | 18D 3^18 cyclic [1,2,...,0] | 9.809 (9.808 / 9.811) | **9.809 (9.807 / 9.809)** |
| `reverse_18d_3` | 18D 3^18 reverse | 29.814 (29.804 / 29.856) | **29.779 (29.777 / 29.787)** |
| `reverse_23d_128x2` | 23D 128x2^22 reverse | **41.479 (41.457 / 42.005)** | 41.482 (41.444 / 42.400) |
| `rotation_6d_64_32_32_32_16_16` | 6D 64x32^3x16^2 rotation [5,0,4,1,3,2] | **6.919 (6.919 / 6.922)** | 6.947 (6.919 / 6.948) |
| `tn_light_415_24d_contiguous_same_perm_gpu` | 24D 2^29 contiguous source, TN light 415 permutation | **5.935 (5.932 / 5.935)** | 5.976 (5.957 / 5.978) |
| `tn_light_415_24d_scattered_to_colmajor_gpu` | 24D 2^29 scattered -> col-major | - | **8.280 (8.279 / 8.282)** |
| `transpose_2d_32768_16384` | 2D 32768x16384 transpose [1,0] | **5.331 (5.329 / 5.338)** | 5.470 (5.464 / 5.474) |
| `transpose_3d_1024_1024_512_102` | 3D 1024x1024x512 transpose [1,0,2] | **5.061 (5.059 / 5.068)** | 5.174 (5.165 / 5.177) |
| `transpose_3d_1024_1024_512_201` | 3D 1024x1024x512 transpose [2,0,1] | **4.976 (4.976 / 4.980)** | 5.069 (4.954 / 5.077) |

## Reusing output

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs CUDA copy_read_into (ms) | cuTENSOR (ms) | PyTorch CUDA (ms) |
|---|---|---:|---:|---:|
| `cyclic_18d_3` | 18D 3^18 cyclic [1,2,...,0] | 9.805 (9.804 / 9.805) | **5.961 (5.961 / 5.961)** | 7.004 (7.002 / 7.008) |
| `reverse_18d_3` | 18D 3^18 reverse | 61.405 (61.327 / 61.563) | **8.455 (8.454 / 8.456)** | 32.470 (32.466 / 32.476) |
| `reverse_23d_128x2` | 23D 128x2^22 reverse | 85.151 (85.118 / 85.833) | **7.569 (7.568 / 7.569)** | 44.644 (44.634 / 44.647) |
| `rotation_6d_64_32_32_32_16_16` | 6D 64x32^3x16^2 rotation [5,0,4,1,3,2] | 6.913 (6.913 / 6.914) | **5.099 (5.097 / 5.100)** | 5.682 (5.677 / 5.687) |
| `tn_light_415_24d_contiguous_same_perm_gpu` | 24D 2^29 contiguous source, TN light 415 permutation | 5.929 (5.925 / 5.932) | 5.285 (5.283 / 5.287) | **5.012 (5.010 / 5.014)** |
| `tn_light_415_24d_scattered_to_colmajor_gpu` | 24D 2^29 scattered -> col-major | - | 5.282 (5.281 / 5.283) | **5.150 (5.148 / 5.151)** |
| `transpose_2d_32768_16384` | 2D 32768x16384 transpose [1,0] | 5.325 (5.323 / 5.332) | **5.264 (5.255 / 5.268)** | 20.788 (20.786 / 20.802) |
| `transpose_3d_1024_1024_512_102` | 3D 1024x1024x512 transpose [1,0,2] | 5.046 (5.041 / 5.048) | **5.010 (5.003 / 5.013)** | 7.317 (7.316 / 7.321) |
| `transpose_3d_1024_1024_512_201` | 3D 1024x1024x512 transpose [2,0,1] | **4.946 (4.943 / 4.953)** | 5.095 (5.092 / 5.100) | 27.433 (27.431 / 27.440) |
