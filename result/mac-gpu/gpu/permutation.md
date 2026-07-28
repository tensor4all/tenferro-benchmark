# GPU Permutation Benchmark Results

- Target profile: `mac-gpu`
- Suite: `gpu/permutation`
- Suite file: `benchmarks/gpu/permutation-mac.yaml`
- Timestamp: `2026-07-28T23:54:42Z`
- tenferro-rs commit: `4111d452375bd5169892c8792f23d0c8da8ffdfe`

- Metal device: `Apple M5 Max`
- Runtime: `wgpu/Metal`

`tenferro-webgpu-transpose-baseline` is the unoptimized native CubeCL structural path captured before kernel work. `tenferro-webgpu-to-contiguous` measures the public strided-view materialization path. PyTorch MPS and JAX Metal are logical framework comparisons; a backend without a Metal runtime is reported as `not_configured` and is never allowed to fall back to CPU. Every timed iteration ends with explicit device synchronization; correctness downloads and JIT compilation are outside timing.

The mac-gpu patterns contain 3^15 or roughly 15 million f32 elements (about 60 MiB per tensor). They remain below the baseline kernel's one-dimensional 65,535-workgroup dispatch ceiling. This M5 collection is the entry-gate baseline; the final tile sweep must still be run on the issue's target M4 machine.

Device-copy baseline: `memcpy-metal-d2d` median 0.469 ms, 268.48 GB/s for `mac_memcpy_1d`. It is a bandwidth reference, not a permutation participant, so it is not shown as a table column.

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs wgpu baseline (ms) | tenferro-rs wgpu to_contiguous (ms) | PyTorch MPS (ms) | JAX Metal (ms) |
|---|---|---:|---:|---:|---:|
| `mac_cyclic_15d_3` | 15D 3^15 cyclic | **1.467 (1.464 / 2.633)** | 1.472 (1.469 / 1.489) | 3.654 (3.599 / 3.942) | not_configured (no JAX Metal device; CPU fallback is forbidden for the mac-gpu profile) |
| `mac_reverse_15d_3` | 15D 3^15 reverse | **2.629 (2.628 / 2.646)** | 2.656 (2.645 / 2.724) | 2.861 (2.324 / 2.976) | not_configured (no JAX Metal device; CPU fallback is forbidden for the mac-gpu profile) |
| `mac_reverse_21d` | 21D 15x2^20 reverse | **2.715 (2.686 / 3.696)** | 2.740 (2.669 / 3.767) | skipped (PyTorch MPS supports at most 16 dimensions) | not_configured (no JAX Metal device; CPU fallback is forbidden for the mac-gpu profile) |
| `mac_rotation_6d` | 64x32x32x15x4x4 rotation | 1.468 (1.466 / 1.473) | **1.466 (1.461 / 1.486)** | 2.677 (2.645 / 3.144) | not_configured (no JAX Metal device; CPU fallback is forbidden for the mac-gpu profile) |
| `mac_tn_contiguous` | 21D contiguous source, TN permutation | **2.616 (2.615 / 2.633)** | 2.617 (2.598 / 2.631) | skipped (PyTorch MPS supports at most 16 dimensions) | not_configured (no JAX Metal device; CPU fallback is forbidden for the mac-gpu profile) |
| `mac_tn_scattered` | 21D scattered source -> col-major | - | **2.648 (2.636 / 3.975)** | skipped (PyTorch MPS supports at most 16 dimensions) | not_configured (no JAX Metal device; CPU fallback is forbidden for the mac-gpu profile) |
| `mac_transpose_2d` | 4096x3840 transpose [1,0] | 1.472 (1.467 / 2.759) | 1.481 (1.477 / 1.490) | **0.664 (0.611 / 1.417)** | not_configured (no JAX Metal device; CPU fallback is forbidden for the mac-gpu profile) |
| `mac_transpose_3d_102` | 256x256x240 transpose [1,0,2] | 1.478 (1.467 / 1.481) | 1.464 (1.463 / 1.502) | **0.837 (0.817 / 1.113)** | not_configured (no JAX Metal device; CPU fallback is forbidden for the mac-gpu profile) |
| `mac_transpose_3d_201` | 256x256x240 transpose [2,0,1] | 1.472 (1.472 / 1.491) | 1.466 (1.463 / 1.472) | **0.998 (0.956 / 1.393)** | not_configured (no JAX Metal device; CPU fallback is forbidden for the mac-gpu profile) |
