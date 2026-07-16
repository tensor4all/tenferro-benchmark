# GPU Permutation Benchmark Results

- Target profile: `nvidia-gpu`
- Suite: `gpu/permutation`
- Suite file: `benchmarks/gpu/permutation.yaml`
- Timestamp: `2026-07-16T09:16:45.210597+00:00`
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
| `cyclic_13d_2` | 13D 2^13 cyclic [1,2,...,0] | 0.035 (0.034 / 0.035) | 0.032 (0.032 / 0.034) | **0.012 (0.011 / 0.012)** | 0.025 (0.025 / 0.026) | 0.098 (0.097 / 0.099) | - |
| `memcpy_24d_contiguous` | memcpy baseline (24D 2^24) | - | - | - | - | - | **0.823 (0.822 / 0.823)** |
| `reverse_13d_2` | 13D 2^13 reverse | 0.034 (0.034 / 0.039) | 0.035 (0.034 / 0.041) | **0.012 (0.011 / 0.013)** | 0.026 (0.026 / 0.028) | 0.096 (0.095 / 0.108) | - |
| `reverse_20d_2` | 20D 2^20 reverse | 0.305 (0.303 / 0.310) | 0.306 (0.305 / 0.314) | **0.061 (0.061 / 0.062)** | 0.245 (0.245 / 0.248) | 0.165 (0.162 / 0.170) | - |
| `rotation_6d_32_32_32_32_16_16` | 6D 32^4x16x16 rotation [5,0,4,1,3,2] | 16.625 (16.599 / 16.792) | 13.392 (13.374 / 13.415) | **12.912 (12.911 / 12.914)** | 13.163 (13.160 / 13.173) | 13.385 (13.315 / 13.416) | - |
| `tn_light_415_24d_contiguous_same_perm` | 24D contiguous source, TN light 415 late-step permutation | 4.364 (4.357 / 4.374) | 2.338 (2.330 / 2.457) | **0.816 (0.816 / 0.817)** | 0.826 (0.825 / 0.827) | - | - |
| `tn_light_415_24d_scattered_to_colmajor` | 24D scattered -> col-major | - | 2.189 (2.149 / 2.316) | **0.819 (0.818 / 0.822)** | 0.827 (0.827 / 0.828) | - | - |
| `transpose_2d_1024` | 2D 1024^2 transpose [1,0] | 0.090 (0.088 / 0.091) | 0.089 (0.088 / 0.090) | **0.061 (0.060 / 0.061)** | 0.075 (0.075 / 0.076) | 0.214 (0.210 / 0.225) | - |
| `transpose_2d_2048` | 2D 2048^2 transpose [1,0] | 0.559 (0.557 / 0.563) | 0.560 (0.556 / 0.564) | **0.218 (0.217 / 0.218)** | 0.331 (0.330 / 0.332) | 0.566 (0.472 / 0.599) | - |
| `transpose_2d_256` | 2D 256^2 transpose [1,0] | 0.034 (0.033 / 0.035) | 0.029 (0.026 / 0.035) | **0.012 (0.011 / 0.012)** | 0.023 (0.023 / 0.024) | 0.171 (0.167 / 0.177) | - |
| `transpose_3d_256_102` | 3D 256^3 transpose [1,0,2] | 0.911 (0.902 / 0.920) | 0.910 (0.887 / 0.924) | **0.814 (0.814 / 0.815)** | 0.845 (0.845 / 0.845) | 1.241 (1.222 / 1.261) | - |
| `transpose_3d_256_201` | 3D 256^3 transpose [2,0,1] | 2.739 (2.713 / 2.764) | 2.734 (2.705 / 2.752) | **0.816 (0.816 / 0.816)** | 2.552 (2.549 / 2.553) | 1.250 (1.225 / 1.267) | - |
