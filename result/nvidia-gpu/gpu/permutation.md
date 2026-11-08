# GPU Permutation Benchmark Results

- Target profile: `nvidia-gpu`
- Suite: `gpu/permutation`
- Suite file: `benchmarks/gpu/permutation.yaml`
- Timestamp: `2026-11-08T11:52:37+00:00`
- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`

## GPU Information

- Device: `cuda:0`
- Name: `NVIDIA A100 80GB PCIe`
- UUID: `GPU-e8de7039-b03e-2d5e-3147-bf92141dbc74`
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

Device-copy baseline: `memcpy-d2d` median 5.017 ms, 1712.19 GB/s for `memcpy_24d_64x2`. It is a bandwidth reference, not a permutation participant, so it is not shown as a table column.

## Allocating output

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs CUDA transpose (ms) | tenferro-rs CUDA to_contiguous (ms) |
|---|---|---:|---:|
| `cyclic_18d_3` | 18D 3^18 cyclic [1,2,...,0] | 21.218 (21.207 / 21.329) | **21.205 (21.200 / 21.208)** |
| `reverse_18d_3` | 18D 3^18 reverse | 26.696 (26.690 / 26.698) | **26.635 (26.629 / 26.651)** |
| `reverse_23d_128x2` | 23D 128x2^22 reverse | **40.184 (39.635 / 40.354)** | 40.202 (40.188 / 40.840) |
| `rotation_6d_64_32_32_32_16_16` | 6D 64x32^3x16^2 rotation [5,0,4,1,3,2] | 9.545 (9.538 / 9.622) | **9.437 (9.424 / 9.498)** |
| `tn_light_415_24d_contiguous_same_perm_gpu` | 24D 2^29 contiguous source, TN light 415 permutation | **36.921 (36.906 / 37.530)** | 36.930 (36.897 / 37.615) |
| `tn_light_415_24d_scattered_to_colmajor_gpu` | 24D 2^29 scattered -> col-major | - | **36.917 (36.897 / 37.561)** |
| `transpose_2d_32768_16384` | 2D 32768x16384 transpose [1,0] | 23.067 (23.065 / 23.070) | **23.045 (23.043 / 23.050)** |
| `transpose_3d_1024_1024_512_102` | 3D 1024x1024x512 transpose [1,0,2] | 10.049 (9.980 / 10.053) | **9.916 (9.884 / 9.937)** |
| `transpose_3d_1024_1024_512_201` | 3D 1024x1024x512 transpose [2,0,1] | 29.855 (29.852 / 29.856) | **29.837 (29.834 / 29.839)** |

## Reusing output

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs CUDA copy_read_into (ms) | cuTENSOR (ms) | PyTorch CUDA (ms) |
|---|---|---:|---:|---:|
| `cyclic_18d_3` | 18D 3^18 cyclic [1,2,...,0] | 38.039 (38.034 / 38.042) | **6.352 (6.027 / 6.426)** | 7.153 (7.153 / 7.155) |
| `reverse_18d_3` | 18D 3^18 reverse | 58.178 (58.137 / 58.208) | **8.981 (8.728 / 8.983)** | 29.414 (29.408 / 29.420) |
| `reverse_23d_128x2` | 23D 128x2^22 reverse | 81.764 (81.641 / 82.305) | **7.602 (7.563 / 7.604)** | 40.419 (40.411 / 40.423) |
| `rotation_6d_64_32_32_32_16_16` | 6D 64x32^3x16^2 rotation [5,0,4,1,3,2] | 61.778 (61.664 / 61.819) | **5.138 (5.137 / 5.140)** | 5.552 (5.547 / 5.557) |
| `tn_light_415_24d_contiguous_same_perm_gpu` | 24D 2^29 contiguous source, TN light 415 permutation | 69.138 (69.094 / 69.722) | 5.627 (5.395 / 5.692) | **5.096 (5.093 / 5.098)** |
| `tn_light_415_24d_scattered_to_colmajor_gpu` | 24D 2^29 scattered -> col-major | - | 5.626 (5.343 / 5.689) | **5.145 (5.142 / 5.147)** |
| `transpose_2d_32768_16384` | 2D 32768x16384 transpose [1,0] | 25.387 (25.371 / 25.398) | **5.309 (5.307 / 5.314)** | 20.897 (20.891 / 20.904) |
| `transpose_3d_1024_1024_512_102` | 3D 1024x1024x512 transpose [1,0,2] | 16.673 (16.655 / 16.680) | **5.159 (5.153 / 5.163)** | 7.641 (7.407 / 7.645) |
| `transpose_3d_1024_1024_512_201` | 3D 1024x1024x512 transpose [2,0,1] | 81.095 (81.056 / 81.140) | **5.147 (5.142 / 5.151)** | 27.582 (27.575 / 27.587) |
