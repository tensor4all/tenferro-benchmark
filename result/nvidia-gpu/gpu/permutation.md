# GPU Permutation Benchmark Results

- Target profile: `nvidia-gpu`
- Suite: `gpu/permutation`
- Suite file: `benchmarks/gpu/permutation.yaml`
- Timestamp: `2026-07-28T04:27:56.075285+00:00`
- tenferro-rs commit: `80ebcc38ce11fb93385e8b6a1a49b613bc17452f`

## GPU Information

- Device: `cuda:0`
- Name: `NVIDIA A100 80GB PCIe`
- UUID: `GPU-530977e1-4968-9283-4129-9fbec3e66542`
- Memory: `80 GiB`
- Driver version: `580.126.09`
- CUDA version: `13.0`
- CUDA runtime: `12.9`
- cuDNN version: `92400`

`tenferro-cuda-transpose` is the eager `TensorStructural::transpose` op on `CudaBackend` (compact col-major input only); `tenferro-cuda-to-contiguous` is `TypedTensor::backend_region_view` (source layout) + `TypedTensorView::transpose_view(perm)` + `TensorViewCanonicalization::to_contiguous` (accepts arbitrary source strides). For framework comparisons, `tenferro-cuda-to-contiguous` is the primary like-for-like column for PyTorch's view/permute-then-materialize path. `tenferro-cuda-transpose` is the direct structural-permutation comparison for primitive/kernel-oriented backends such as cuTENSOR. Both allocate a fresh device tensor on every call; `cutensor`, `pytorch-cuda`, and `memcpy-d2d` reuse a destination buffer allocated once per pattern. `memcpy-d2d` only participates in the contiguous identity-permutation baseline pattern. Correctness is verified against a host-computed naive reference, downloaded outside any timed region, before any timing; a `FAILED` cell means that backend's output did not match the reference for that pattern. A `skipped` cell means the backend does not participate in that pattern's semantics (or, for `cutensor`, that the installed cuTENSOR library rejected the pattern's rank at runtime) -- both are reported as `skipped` rather than a failure.

The GPU-only pattern set is intentionally larger than the CPU permutation set: most rows contain 2^29 f64 elements (4 GiB per tensor) so A100-class measurements exercise steady-state device throughput at roughly the 10 ms scale instead of launch/synchronization overhead.

Device-copy baseline: `memcpy-d2d` median 5.134 ms, 1673.04 GB/s for `memcpy_24d_64x2`. It is a bandwidth reference, not a permutation participant, so it is not shown as a table column.

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs CUDA transpose (ms) | tenferro-rs CUDA to_contiguous (ms) | cuTENSOR (ms) | PyTorch CUDA (ms) |
|---|---|---:|---:|---:|---:|
| `cyclic_18d_3` | 18D 3^18 cyclic [1,2,...,0] | 24.999 (24.993 / 25.006) | 12.951 (12.850 / 12.999) | **6.280 (6.012 / 6.281)** | 7.357 (7.350 / 7.358) |
| `reverse_18d_3` | 18D 3^18 reverse | 28.406 (28.385 / 28.412) | 28.585 (28.561 / 28.621) | **8.710 (8.612 / 8.712)** | 33.194 (33.193 / 33.196) |
| `reverse_23d_128x2` | 23D 128x2^22 reverse | 46.432 (46.363 / 46.529) | 39.591 (39.572 / 39.625) | **7.562 (7.562 / 7.563)** | 45.753 (45.749 / 45.756) |
| `rotation_6d_64_32_32_32_16_16` | 6D 64x32^3x16^2 rotation [5,0,4,1,3,2] | 12.001 (11.997 / 12.011) | 7.379 (7.213 / 7.409) | **5.349 (5.346 / 5.352)** | 5.935 (5.925 / 5.939) |
| `tn_light_415_24d_contiguous_same_perm_gpu` | 24D 2^29 contiguous source, TN light 415 permutation | 45.206 (45.203 / 45.208) | 24.404 (24.384 / 24.413) | 5.487 (5.397 / 5.563) | **5.259 (5.258 / 5.260)** |
| `tn_light_415_24d_scattered_to_colmajor_gpu` | 24D 2^29 scattered -> col-major | - | 24.369 (24.362 / 24.471) | 5.521 (5.399 / 5.561) | **5.402 (5.400 / 5.403)** |
| `transpose_2d_32768_16384` | 2D 32768x16384 transpose [1,0] | 21.204 (21.200 / 21.206) | 20.998 (20.996 / 21.002) | **5.527 (5.522 / 5.533)** | 20.761 (20.760 / 20.763) |
| `transpose_3d_1024_1024_512_102` | 3D 1024x1024x512 transpose [1,0,2] | 8.377 (8.209 / 8.480) | 7.847 (7.731 / 7.983) | **5.265 (5.264 / 5.273)** | 7.323 (7.276 / 7.336) |
| `transpose_3d_1024_1024_512_201` | 3D 1024x1024x512 transpose [2,0,1] | 27.998 (27.927 / 28.011) | 27.604 (27.590 / 27.615) | **5.340 (5.336 / 5.345)** | 27.392 (27.388 / 27.393) |
