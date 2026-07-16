# GPU Permutation Benchmark Results

- Target profile: `nvidia-gpu`
- Suite: `gpu/permutation`
- Suite file: `benchmarks/gpu/permutation.yaml`
- Timestamp: `2026-07-16T11:10:51.122278+00:00`
- tenferro-rs commit: `3abc9108e4f0500f4e75519711cdb4e21b9625df`

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
| `cyclic_13d_2` | 13D 2^13 cyclic [1,2,...,0] | 0.039 (0.038 / 0.040) | 0.035 (0.034 / 0.035) | **0.013 (0.012 / 0.013)** | 0.024 (0.024 / 0.024) | 0.130 (0.126 / 0.160) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | - | - | - | **0.822 (0.822 / 0.824)** |
| `reverse_13d_2` | 13D 2^13 reverse | 0.039 (0.038 / 0.044) | 0.037 (0.036 / 0.039) | **0.011 (0.011 / 0.012)** | 0.024 (0.024 / 0.024) | 0.135 (0.128 / 0.170) | - |
| `reverse_20d_2` | 20D 2^20 reverse | 0.408 (0.406 / 0.418) | 0.358 (0.357 / 0.370) | **0.062 (0.062 / 0.063)** | 0.245 (0.244 / 0.246) | 0.165 (0.163 / 0.168) | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | 16.796 (16.609 / 16.809) | 13.206 (13.203 / 13.271) | **12.913 (12.912 / 12.914)** | 13.153 (13.147 / 13.162) | 13.359 (13.306 / 13.393) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | 4.402 (4.392 / 4.407) | 2.343 (2.339 / 2.344) | **0.816 (0.816 / 0.817)** | 0.827 (0.826 / 0.828) | - | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | - | 2.387 (2.386 / 2.396) | **0.818 (0.818 / 0.818)** | 0.828 (0.827 / 0.829) | - | - |
| `transpose_2d_1024` | 2D 1024^2 transpose [1,0] | 0.080 (0.079 / 0.081) | 0.083 (0.082 / 0.084) | **0.060 (0.060 / 0.061)** | 0.074 (0.074 / 0.075) | 0.215 (0.212 / 0.220) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | 0.499 (0.494 / 0.507) | 0.504 (0.499 / 0.511) | **0.218 (0.217 / 0.219)** | 0.331 (0.330 / 0.332) | 0.574 (0.542 / 0.603) | - |
| `transpose_2d_256` | 2D 256^2 transpose [1,0] | 0.019 (0.018 / 0.020) | 0.020 (0.019 / 0.021) | **0.010 (0.009 / 0.010)** | 0.024 (0.023 / 0.026) | 0.130 (0.129 / 0.141) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | 0.916 (0.899 / 0.927) | 0.909 (0.897 / 0.917) | **0.815 (0.814 / 0.819)** | 0.846 (0.844 / 0.847) | 1.232 (1.186 / 1.236) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 2.801 (2.795 / 2.804) | 2.735 (2.684 / 2.748) | **0.816 (0.816 / 0.816)** | 2.551 (2.549 / 2.553) | 1.267 (1.231 / 1.278) | - |
