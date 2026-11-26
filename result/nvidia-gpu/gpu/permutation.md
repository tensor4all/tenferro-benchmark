# GPU Permutation Benchmark Results

- Target profile: `nvidia-gpu`
- Suite: `gpu/permutation`
- Suite file: `benchmarks/gpu/permutation.yaml`
- Timestamp: `2026-11-26T06:29:50+00:00`
- tenferro-rs commit: `1ba9f33521691170c763dfce9b6c9bb56ca523c2`

## GPU Information

- Device: `cuda:0`
- Name: `NVIDIA A100 80GB PCIe`
- UUID: `GPU-e0c82f3d-3a11-a28c-b42e-1c92d26527fd`
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

Device-copy baseline: `memcpy-d2d` median 5.017 ms, 1712.03 GB/s for `memcpy_24d_64x2`. It is a bandwidth reference, not a permutation participant, so it is not shown as a table column.

## Allocating output

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs CUDA transpose (ms) | tenferro-rs CUDA to_contiguous (ms) |
|---|---|---:|---:|
| `cyclic_18d_3` | 18D 3^18 cyclic [1,2,...,0] | **10.101 (9.995 / 10.101)** | 10.101 (10.100 / 10.101) |
| `reverse_18d_3` | 18D 3^18 reverse | 10.541 (10.477 / 10.590) | **10.474 (10.464 / 10.532)** |
| `reverse_23d_128x2` | 23D 128x2^22 reverse | 9.856 (9.855 / 9.861) | **9.844 (9.838 / 9.892)** |
| `rotation_6d_64_32_32_32_16_16` | 6D 64x32^3x16^2 rotation [5,0,4,1,3,2] | 7.420 (7.056 / 7.422) | **7.022 (6.996 / 7.025)** |
| `tn_light_415_24d_contiguous_same_perm_gpu` | 24D 2^29 contiguous source, TN light 415 permutation | 6.255 (6.035 / 6.317) | **6.220 (6.016 / 6.261)** |
| `tn_light_415_24d_scattered_to_colmajor_gpu` | 24D 2^29 scattered -> col-major | - | **8.526 (8.282 / 8.527)** |
| `transpose_2d_32768_16384` | 2D 32768x16384 transpose [1,0] | **5.372 (5.370 / 5.376)** | 5.516 (5.513 / 5.517) |
| `transpose_3d_1024_1024_512_102` | 3D 1024x1024x512 transpose [1,0,2] | **5.033 (5.032 / 5.039)** | 5.157 (5.149 / 5.159) |
| `transpose_3d_1024_1024_512_201` | 3D 1024x1024x512 transpose [2,0,1] | **5.023 (5.018 / 5.026)** | 5.136 (5.132 / 5.139) |

## Reusing output

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs CUDA copy_read_into (ms) | cuTENSOR (ms) | PyTorch CUDA (ms) |
|---|---|---:|---:|---:|
| `cyclic_18d_3` | 18D 3^18 cyclic [1,2,...,0] | 10.100 (9.977 / 10.102) | **6.498 (6.423 / 6.502)** | 7.146 (7.143 / 7.148) |
| `reverse_18d_3` | 18D 3^18 reverse | 9.023 (8.789 / 9.040) | **8.917 (8.915 / 8.918)** | 29.402 (29.400 / 29.406) |
| `reverse_23d_128x2` | 23D 128x2^22 reverse | 7.564 (7.563 / 7.565) | **7.555 (7.554 / 7.555)** | 40.415 (40.407 / 40.420) |
| `rotation_6d_64_32_32_32_16_16` | 6D 64x32^3x16^2 rotation [5,0,4,1,3,2] | 7.399 (7.093 / 7.400) | **5.141 (5.139 / 5.144)** | 5.539 (5.536 / 5.547) |
| `tn_light_415_24d_contiguous_same_perm_gpu` | 24D 2^29 contiguous source, TN light 415 permutation | 6.193 (5.976 / 6.256) | 5.722 (5.337 / 5.820) | **5.093 (5.091 / 5.095)** |
| `tn_light_415_24d_scattered_to_colmajor_gpu` | 24D 2^29 scattered -> col-major | - | 5.625 (5.286 / 5.753) | **5.139 (5.136 / 5.140)** |
| `transpose_2d_32768_16384` | 2D 32768x16384 transpose [1,0] | 5.370 (5.368 / 5.373) | **5.316 (5.315 / 5.322)** | 20.761 (20.759 / 20.763) |
| `transpose_3d_1024_1024_512_102` | 3D 1024x1024x512 transpose [1,0,2] | **5.035 (5.028 / 5.038)** | 5.150 (5.145 / 5.154) | 7.627 (7.527 / 7.642) |
| `transpose_3d_1024_1024_512_201` | 3D 1024x1024x512 transpose [2,0,1] | **5.012 (5.011 / 5.014)** | 5.141 (5.137 / 5.144) | 27.411 (27.409 / 27.416) |
