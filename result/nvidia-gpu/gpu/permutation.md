# GPU Permutation Benchmark Results

- Target profile: `nvidia-gpu`
- Suite: `gpu/permutation`
- Suite file: `benchmarks/gpu/permutation.yaml`
- Timestamp: `2026-07-16T04:27:39.584306+00:00`
- tenferro-rs commit: `8c267c636537d3d407925b61a2000f9bbe07e6a1`

## GPU Information

- Device: `cuda:0`
- Name: `NVIDIA GeForce RTX 3060`
- UUID: `GPU-a78d5217-eba3-72c2-3d5b-8ae496ebbc2e`
- Memory: `12 GiB`
- Driver version: `580.159.03`
- CUDA version: `13.0`
- CUDA runtime: `12.9`
- cuDNN version: `92400`

`tenferro-cuda-transpose` is the eager `TensorStructural::transpose` op on `CudaBackend` (compact col-major input only); `tenferro-cuda-to-contiguous` is `TypedTensor::backend_region_view` (source layout) + `TypedTensorView::transpose_view(perm)` + `TensorViewCanonicalization::to_contiguous` (accepts arbitrary source strides). Both allocate a fresh device tensor on every call; `cutensor`, `pytorch-cuda`, and `memcpy-d2d` reuse a destination buffer allocated once per pattern. `jax-cuda` allocates a fresh output array per call (functional `jnp.transpose`), cannot express arbitrary source strides (so it does not participate in the explicit-stride pattern), and materializes its output in XLA's default row-major layout -- the public JAX API cannot request a col-major output, so at the byte level `jax-cuda` performs the reversal-conjugate permutation task (same shape multiset and mirrored stride structure, equivalent difficulty class) rather than the identical col-major write the other columns perform (see docs/gpu-permutation-suite.md). `jax-cuda` is additionally excluded from the rank-24 contiguous pattern (`tn_light_415_24d_contiguous_same_perm`) because XLA's jit compilation of the rank-24 transpose does not complete in practical time (>40 min observed). `memcpy-d2d` only participates in the contiguous identity-permutation baseline pattern. Correctness is verified against a host-computed naive reference, downloaded outside any timed region, before any timing; a `FAILED` cell means that backend's output did not match the reference for that pattern. A `skipped` cell means the backend does not participate in that pattern's semantics (or, for `cutensor`, that the installed cuTENSOR library rejected the pattern's rank at runtime) -- both are reported as `skipped` rather than a failure.

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs CUDA transpose (ms) | tenferro-rs CUDA to_contiguous (ms) | cuTENSOR (ms) | PyTorch CUDA (ms) | JAX CUDA (ms) | memcpy D2D (ms) |
|---|---|---:|---:|---:|---:|---:|---:|
| `cyclic_13d_2` | 13D 2^13 cyclic [1,2,...,0] | 0.036 (0.035 / 0.037) | 0.035 (0.035 / 0.037) | **0.012 (0.012 / 0.013)** | 0.024 (0.023 / 0.025) | 0.121 (0.118 / 0.123) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | - | - | - | **0.824 (0.823 / 0.825)** |
| `reverse_13d_2` | 13D 2^13 reverse | 0.040 (0.040 / 0.043) | 0.039 (0.035 / 0.040) | **0.013 (0.012 / 0.013)** | 0.024 (0.024 / 0.025) | 0.137 (0.125 / 0.162) | - |
| `reverse_20d_2` | 20D 2^20 reverse | 0.424 (0.420 / 0.436) | 0.362 (0.360 / 0.377) | **0.062 (0.062 / 0.063)** | 0.245 (0.244 / 0.247) | 0.165 (0.161 / 0.175) | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | 16.799 (16.795 / 16.811) | 13.243 (13.207 / 13.380) | **12.913 (12.912 / 12.913)** | 13.161 (13.158 / 13.162) | 13.378 (13.326 / 13.418) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | 4.403 (4.376 / 4.423) | 2.397 (2.396 / 2.411) | **0.817 (0.816 / 0.817)** | 0.826 (0.825 / 0.827) | - | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | - | 2.331 (2.316 / 2.335) | **0.818 (0.818 / 0.818)** | 0.826 (0.825 / 0.827) | - | - |
| `transpose_2d_1024` | 2D 1024^2 transpose [1,0] | 0.082 (0.081 / 0.083) | 0.085 (0.084 / 0.087) | **0.061 (0.061 / 0.061)** | 0.074 (0.074 / 0.075) | 0.219 (0.215 / 0.225) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | 0.506 (0.491 / 0.512) | 0.564 (0.558 / 0.575) | **0.218 (0.217 / 0.218)** | 0.330 (0.330 / 0.332) | 0.559 (0.474 / 0.599) | - |
| `transpose_2d_256` | 2D 256^2 transpose [1,0] | 0.019 (0.019 / 0.020) | 0.021 (0.020 / 0.021) | **0.010 (0.010 / 0.011)** | 0.023 (0.023 / 0.024) | 0.133 (0.121 / 0.154) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | 0.968 (0.966 / 0.983) | 0.969 (0.966 / 0.974) | **0.815 (0.814 / 0.815)** | 0.844 (0.843 / 0.845) | 1.265 (1.220 / 1.281) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 2.806 (2.801 / 2.811) | 2.751 (2.716 / 2.760) | **0.816 (0.816 / 0.817)** | 2.580 (2.575 / 2.581) | 1.201 (1.169 / 1.259) | - |
