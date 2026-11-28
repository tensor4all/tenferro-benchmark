# GPU Permutation Benchmark Results

- Target profile: `nvidia-gpu`
- Suite: `gpu/permutation`
- Suite file: `benchmarks/gpu/permutation.yaml`
- Timestamp: `2026-11-28T08:23:46+00:00`
- tenferro-rs commit: `1ba9f33521691170c763dfce9b6c9bb56ca523c2`

## GPU Information

- Device: `cuda:0`
- Name: `NVIDIA A100 80GB PCIe`
- UUID: `GPU-12b45076-79d5-7b84-cb5c-7c6300074c8a`
- Memory: `80 GiB`
- Driver version: `595.112.07`
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

Device-copy baseline: `memcpy-d2d` median 5.152 ms, 1667.35 GB/s for `memcpy_24d_64x2`. It is a bandwidth reference, not a permutation participant, so it is not shown as a table column.

## Allocating output

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs CUDA transpose (ms) | tenferro-rs CUDA to_contiguous (ms) |
|---|---|---:|---:|
| `cyclic_18d_3` | 18D 3^18 cyclic [1,2,...,0] | **9.896 (9.890 / 9.899)** | 9.901 (9.899 / 9.903) |
| `reverse_18d_3` | 18D 3^18 reverse | 10.448 (10.357 / 10.461) | **10.442 (10.423 / 10.448)** |
| `reverse_23d_128x2` | 23D 128x2^22 reverse | 9.962 (9.958 / 9.965) | **9.961 (9.955 / 10.017)** |
| `rotation_6d_64_32_32_32_16_16` | 6D 64x32^3x16^2 rotation [5,0,4,1,3,2] | **7.306 (7.004 / 7.312)** | 7.319 (7.196 / 7.343) |
| `tn_light_415_24d_contiguous_same_perm_gpu` | 24D 2^29 contiguous source, TN light 415 permutation | **6.137 (6.026 / 6.160)** | 6.196 (6.127 / 6.217) |
| `tn_light_415_24d_scattered_to_colmajor_gpu` | 24D 2^29 scattered -> col-major | - | **8.354 (8.280 / 8.377)** |
| `transpose_2d_32768_16384` | 2D 32768x16384 transpose [1,0] | **5.585 (5.580 / 5.586)** | 5.725 (5.578 / 5.730) |
| `transpose_3d_1024_1024_512_102` | 3D 1024x1024x512 transpose [1,0,2] | 5.284 (5.283 / 5.287) | **5.274 (5.269 / 5.278)** |
| `transpose_3d_1024_1024_512_201` | 3D 1024x1024x512 transpose [2,0,1] | **5.213 (5.209 / 5.217)** | 5.311 (5.187 / 5.316) |

## Reusing output

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs CUDA copy_read_into (ms) | cuTENSOR (ms) | PyTorch CUDA (ms) |
|---|---|---:|---:|---:|
| `cyclic_18d_3` | 18D 3^18 cyclic [1,2,...,0] | 9.905 (9.799 / 9.910) | **6.314 (6.030 / 6.357)** | 7.368 (7.364 / 7.369) |
| `reverse_18d_3` | 18D 3^18 reverse | **8.724 (8.623 / 8.826)** | 8.830 (8.726 / 8.833) | 33.259 (33.240 / 33.260) |
| `reverse_23d_128x2` | 23D 128x2^22 reverse | 7.572 (7.569 / 7.574) | **7.563 (7.562 / 7.564)** | 45.750 (45.743 / 45.757) |
| `rotation_6d_64_32_32_32_16_16` | 6D 64x32^3x16^2 rotation [5,0,4,1,3,2] | 7.307 (7.025 / 7.315) | **5.346 (5.340 / 5.348)** | 5.950 (5.943 / 5.957) |
| `tn_light_415_24d_contiguous_same_perm_gpu` | 24D 2^29 contiguous source, TN light 415 permutation | 6.160 (6.041 / 6.165) | 5.540 (5.401 / 5.614) | **5.265 (5.263 / 5.268)** |
| `tn_light_415_24d_scattered_to_colmajor_gpu` | 24D 2^29 scattered -> col-major | - | 5.571 (5.434 / 5.621) | **5.406 (5.403 / 5.408)** |
| `transpose_2d_32768_16384` | 2D 32768x16384 transpose [1,0] | 5.576 (5.573 / 5.583) | **5.524 (5.514 / 5.528)** | 20.818 (20.799 / 20.820) |
| `transpose_3d_1024_1024_512_102` | 3D 1024x1024x512 transpose [1,0,2] | 5.266 (5.265 / 5.269) | **5.247 (5.239 / 5.250)** | 7.543 (7.475 / 7.551) |
| `transpose_3d_1024_1024_512_201` | 3D 1024x1024x512 transpose [2,0,1] | **5.183 (5.180 / 5.184)** | 5.329 (5.323 / 5.336) | 27.448 (27.437 / 27.460) |
