# GPU Permutation Benchmark Results

- Target profile: `nvidia-gpu`
- Suite: `gpu/permutation`
- Suite file: `benchmarks/gpu/permutation.yaml`
- Timestamp: `2026-10-17T00:07:42.343886+00:00`
- tenferro-rs commit: `2754c8c85dcc35c1b63cb7bbdd94ddfd7ea5746e`

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

Device-copy baseline: `memcpy-d2d` median 5.136 ms, 1672.43 GB/s for `memcpy_24d_64x2`. It is a bandwidth reference, not a permutation participant, so it is not shown as a table column.

## Allocating output

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs CUDA transpose (ms) | tenferro-rs CUDA to_contiguous (ms) |
|---|---|---:|---:|
| `cyclic_18d_3` | 18D 3^18 cyclic [1,2,...,0] | 6.320 (6.076 / 6.325) | **6.254 (5.989 / 6.316)** |
| `reverse_18d_3` | 18D 3^18 reverse | 8.723 (8.588 / 8.726) | **8.718 (8.622 / 8.721)** |
| `reverse_23d_128x2` | 23D 128x2^22 reverse | **7.612 (7.611 / 7.613)** | 8.258 (8.257 / 8.260) |
| `rotation_6d_64_32_32_32_16_16` | 6D 64x32^3x16^2 rotation [5,0,4,1,3,2] | **5.392 (5.388 / 5.397)** | 5.569 (5.564 / 5.575) |
| `tn_light_415_24d_contiguous_same_perm_gpu` | 24D 2^29 contiguous source, TN light 415 permutation | **5.603 (5.440 / 5.607)** | 5.653 (5.505 / 5.656) |
| `tn_light_415_24d_scattered_to_colmajor_gpu` | 24D 2^29 scattered -> col-major | - | **5.612 (5.481 / 5.663)** |
| `transpose_2d_32768_16384` | 2D 32768x16384 transpose [1,0] | **5.560 (5.559 / 5.567)** | 6.969 (6.966 / 6.970) |
| `transpose_3d_1024_1024_512_102` | 3D 1024x1024x512 transpose [1,0,2] | **5.328 (5.325 / 5.339)** | 5.428 (5.425 / 5.432) |
| `transpose_3d_1024_1024_512_201` | 3D 1024x1024x512 transpose [2,0,1] | **5.398 (5.391 / 5.404)** | 6.909 (6.908 / 6.912) |

## Reusing output

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs CUDA copy_read_into (ms) | cuTENSOR (ms) | PyTorch CUDA (ms) |
|---|---|---:|---:|---:|
| `cyclic_18d_3` | 18D 3^18 cyclic [1,2,...,0] | 6.281 (5.975 / 6.306) | **6.279 (6.076 / 6.280)** | 7.354 (7.352 / 7.358) |
| `reverse_18d_3` | 18D 3^18 reverse | 8.717 (8.544 / 8.719) | **8.687 (8.533 / 8.745)** | 33.189 (33.188 / 33.193) |
| `reverse_23d_128x2` | 23D 128x2^22 reverse | 7.614 (7.613 / 7.615) | **7.583 (7.582 / 7.583)** | 45.667 (45.663 / 45.671) |
| `rotation_6d_64_32_32_32_16_16` | 6D 64x32^3x16^2 rotation [5,0,4,1,3,2] | 5.360 (5.359 / 5.365) | **5.334 (5.333 / 5.341)** | 5.943 (5.941 / 5.952) |
| `tn_light_415_24d_contiguous_same_perm_gpu` | 24D 2^29 contiguous source, TN light 415 permutation | 5.551 (5.428 / 5.594) | 5.533 (5.402 / 5.569) | **5.266 (5.260 / 5.268)** |
| `tn_light_415_24d_scattered_to_colmajor_gpu` | 24D 2^29 scattered -> col-major | - | 5.517 (5.399 / 5.558) | **5.408 (5.405 / 5.412)** |
| `transpose_2d_32768_16384` | 2D 32768x16384 transpose [1,0] | 5.535 (5.532 / 5.537) | **5.506 (5.504 / 5.512)** | 20.761 (20.760 / 20.762) |
| `transpose_3d_1024_1024_512_102` | 3D 1024x1024x512 transpose [1,0,2] | 5.267 (5.264 / 5.272) | **5.247 (5.243 / 5.254)** | 7.592 (7.530 / 7.594) |
| `transpose_3d_1024_1024_512_201` | 3D 1024x1024x512 transpose [2,0,1] | 5.337 (5.333 / 5.343) | **5.315 (5.310 / 5.322)** | 27.394 (27.393 / 27.397) |
