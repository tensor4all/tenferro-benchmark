# GPU Permutation Benchmark Results

- Target profile: `nvidia-gpu`
- Suite: `gpu/permutation`
- Suite file: `benchmarks/gpu/permutation.yaml`
- Timestamp: `2026-09-19T14:36:30.889278+00:00`
- tenferro-rs commit: `d7a8c60caedac9aa3e630eea793d606df02dfc09`

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

Device-copy baseline: `memcpy-d2d` median 5.135 ms, 1672.98 GB/s for `memcpy_24d_64x2`. It is a bandwidth reference, not a permutation participant, so it is not shown as a table column.

## Allocating output

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs CUDA transpose (ms) | tenferro-rs CUDA to_contiguous (ms) |
|---|---|---:|---:|
| `cyclic_18d_3` | 18D 3^18 cyclic [1,2,...,0] | 6.316 (5.985 / 6.316) | **6.247 (6.049 / 6.316)** |
| `reverse_18d_3` | 18D 3^18 reverse | 8.741 (8.639 / 8.747) | **8.720 (8.641 / 8.739)** |
| `reverse_23d_128x2` | 23D 128x2^22 reverse | **7.621 (7.620 / 7.622)** | 8.280 (8.279 / 8.281) |
| `rotation_6d_64_32_32_32_16_16` | 6D 64x32^3x16^2 rotation [5,0,4,1,3,2] | **5.396 (5.390 / 5.400)** | 5.556 (5.551 / 5.563) |
| `tn_light_415_24d_contiguous_same_perm_gpu` | 24D 2^29 contiguous source, TN light 415 permutation | 5.562 (5.433 / 5.601) | **5.558 (5.429 / 5.630)** |
| `tn_light_415_24d_scattered_to_colmajor_gpu` | 24D 2^29 scattered -> col-major | - | **5.569 (5.447 / 5.666)** |
| `transpose_2d_32768_16384` | 2D 32768x16384 transpose [1,0] | **5.566 (5.561 / 5.571)** | 7.023 (7.011 / 7.043) |
| `transpose_3d_1024_1024_512_102` | 3D 1024x1024x512 transpose [1,0,2] | **5.322 (5.312 / 5.327)** | 5.426 (5.423 / 5.431) |
| `transpose_3d_1024_1024_512_201` | 3D 1024x1024x512 transpose [2,0,1] | **5.401 (5.395 / 5.406)** | 6.889 (6.885 / 6.890) |

## Reusing output

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs CUDA copy_read_into (ms) | cuTENSOR (ms) | PyTorch CUDA (ms) |
|---|---|---:|---:|---:|
| `cyclic_18d_3` | 18D 3^18 cyclic [1,2,...,0] | 6.310 (6.129 / 6.316) | **6.215 (5.953 / 6.284)** | 7.355 (7.352 / 7.360) |
| `reverse_18d_3` | 18D 3^18 reverse | 8.724 (8.644 / 8.732) | **8.687 (8.580 / 8.688)** | 33.189 (33.186 / 33.196) |
| `reverse_23d_128x2` | 23D 128x2^22 reverse | 7.615 (7.613 / 7.620) | **7.587 (7.587 / 7.588)** | 45.772 (45.768 / 45.774) |
| `rotation_6d_64_32_32_32_16_16` | 6D 64x32^3x16^2 rotation [5,0,4,1,3,2] | 5.360 (5.357 / 5.364) | **5.336 (5.333 / 5.341)** | 5.942 (5.940 / 5.945) |
| `tn_light_415_24d_contiguous_same_perm_gpu` | 24D 2^29 contiguous source, TN light 415 permutation | 5.553 (5.470 / 5.591) | 5.523 (5.399 / 5.567) | **5.261 (5.257 / 5.262)** |
| `tn_light_415_24d_scattered_to_colmajor_gpu` | 24D 2^29 scattered -> col-major | - | 5.511 (5.402 / 5.566) | **5.400 (5.398 / 5.401)** |
| `transpose_2d_32768_16384` | 2D 32768x16384 transpose [1,0] | 5.535 (5.528 / 5.544) | **5.512 (5.509 / 5.516)** | 20.761 (20.758 / 20.762) |
| `transpose_3d_1024_1024_512_102` | 3D 1024x1024x512 transpose [1,0,2] | 5.280 (5.274 / 5.283) | **5.252 (5.251 / 5.258)** | 7.489 (7.440 / 7.497) |
| `transpose_3d_1024_1024_512_201` | 3D 1024x1024x512 transpose [2,0,1] | 5.341 (5.337 / 5.353) | **5.317 (5.309 / 5.322)** | 27.394 (27.390 / 27.395) |
