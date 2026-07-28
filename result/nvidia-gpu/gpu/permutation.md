# GPU Permutation Benchmark Results

- Target profile: `nvidia-gpu`
- Suite: `gpu/permutation`
- Suite file: `benchmarks/gpu/permutation.yaml`
- Timestamp: `2026-07-28T03:56:14.663809+00:00`
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

Device-copy baseline: `memcpy-d2d` median 0.172 ms, 1556.19 GB/s for `memcpy_24d_contiguous`. It is a bandwidth reference, not a permutation participant, so it is not shown as a table column.

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs CUDA transpose (ms) | tenferro-rs CUDA to_contiguous (ms) | cuTENSOR (ms) | PyTorch CUDA (ms) |
|---|---|---:|---:|---:|---:|
| `cyclic_15d_3` | 15D 3^15 cyclic [1,2,...,0] | 1.000 (1.000 / 1.000) | 0.611 (0.584 / 0.613) | **0.201 (0.201 / 0.201)** | 0.285 (0.284 / 0.286) |
| `reverse_15d_3` | 15D 3^15 reverse | 1.025 (1.023 / 1.203) | 1.026 (1.024 / 1.203) | **0.305 (0.304 / 0.306)** | 1.002 (1.001 / 1.003) |
| `reverse_23d_2` | 23D 2^23 reverse | 0.804 (0.797 / 0.866) | 0.611 (0.611 / 0.613) | **0.099 (0.098 / 0.103)** | 0.573 (0.573 / 0.574) |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | 6.002 (5.995 / 6.052) | 3.622 (3.583 / 3.648) | **2.681 (2.679 / 2.683)** | 2.968 (2.961 / 2.972) |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | 1.592 (1.591 / 1.613) | 0.858 (0.855 / 0.860) | **0.181 (0.180 / 0.184)** | 0.187 (0.183 / 0.192) |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | - | 0.819 (0.796 / 0.859) | **0.187 (0.182 / 0.190)** | 0.192 (0.177 / 0.196) |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | 0.087 (0.086 / 0.087) | 0.087 (0.086 / 0.087) | **0.052 (0.051 / 0.054)** | 0.073 (0.073 / 0.074) |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | 0.259 (0.258 / 0.261) | 0.407 (0.406 / 0.408) | **0.175 (0.175 / 0.176)** | 0.232 (0.231 / 0.233) |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 0.817 (0.795 / 0.863) | 0.796 (0.795 / 0.816) | **0.180 (0.175 / 0.182)** | 0.805 (0.804 / 0.805) |
