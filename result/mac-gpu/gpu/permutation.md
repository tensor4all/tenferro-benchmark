# GPU Permutation Benchmark Results

- Target profile: `mac-gpu`
- Suite: `gpu/permutation`
- Suite file: `benchmarks/gpu/permutation-mac.yaml`
- Timestamp: `2026-07-29T03:42:53Z`
- tenferro-rs commit: `70ef3cb67f03f589e69cd3119562e4753ca09eb1`

- Metal device: `Apple M5 Max`
- Runtime: `wgpu/Metal`

`tenferro-webgpu-transpose-baseline` is the historical profile identifier for the native CubeCL structural path; the recorded tenferro revision and `TENFERRO_NATIVE_TRANSPOSE_TILE` value distinguish baseline and optimized runs. `tenferro-webgpu-to-contiguous` measures the public strided-view materialization path. PyTorch MPS and JAX Metal are logical framework comparisons; a backend without a Metal runtime is reported as `not_configured` and is never allowed to fall back to CPU. Every timed iteration ends with explicit device synchronization; correctness downloads and JIT compilation are outside timing. Both tenferro and PyTorch MPS allocate a fresh destination per timed call; the medians remain end-to-end host-API measurements rather than isolated kernel timings.

The mac-gpu patterns contain 3^15 or roughly 15 million f32 elements (about 60 MiB per tensor). The entry-gate baseline and development tile sweep were collected on M5. This is an accepted development substitute, but the final tile sweep must still be run on the issue's target M4 machine.

Device-copy baseline: `memcpy-metal-d2d` median 0.415 ms, 302.99 GB/s for `mac_memcpy_1d`. It is a bandwidth reference, not a permutation participant, so it is not shown as a table column.

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs wgpu native (ms) | tenferro-rs wgpu to_contiguous (ms) | PyTorch MPS (ms) | JAX Metal (ms) |
|---|---|---:|---:|---:|---:|
| `mac_cyclic_15d_3` | 15D 3^15 cyclic | **1.715 (1.703 / 1.722)** | 1.725 (1.722 / 1.729) | 3.750 (3.349 / 3.833) | not_configured (no JAX Metal device; CPU fallback is forbidden for the mac-gpu profile) |
| `mac_reverse_15d_3` | 15D 3^15 reverse | 3.104 (2.915 / 3.279) | 3.112 (3.102 / 3.123) | **2.322 (2.282 / 2.369)** | not_configured (no JAX Metal device; CPU fallback is forbidden for the mac-gpu profile) |
| `mac_reverse_21d` | 21D 15x2^20 reverse | **3.089 (2.807 / 3.101)** | 3.092 (3.087 / 3.108) | skipped (PyTorch MPS supports at most 16 dimensions) | not_configured (no JAX Metal device; CPU fallback is forbidden for the mac-gpu profile) |
| `mac_rotation_6d` | 64x32x32x15x4x4 rotation | **1.710 (1.706 / 1.716)** | 1.723 (1.710 / 1.728) | 2.684 (2.523 / 2.719) | not_configured (no JAX Metal device; CPU fallback is forbidden for the mac-gpu profile) |
| `mac_tn_contiguous` | 21D contiguous source, TN permutation | 1.717 (1.704 / 1.729) | **1.712 (1.708 / 1.724)** | skipped (PyTorch MPS supports at most 16 dimensions) | not_configured (no JAX Metal device; CPU fallback is forbidden for the mac-gpu profile) |
| `mac_tn_scattered` | 21D scattered source -> col-major | - | **1.716 (1.708 / 1.725)** | skipped (PyTorch MPS supports at most 16 dimensions) | not_configured (no JAX Metal device; CPU fallback is forbidden for the mac-gpu profile) |
| `mac_transpose_2d` | 4096x3840 transpose [1,0] | 1.731 (1.723 / 1.739) | 1.737 (1.730 / 1.744) | **0.529 (0.518 / 0.545)** | not_configured (no JAX Metal device; CPU fallback is forbidden for the mac-gpu profile) |
| `mac_transpose_3d_102` | 256x256x240 transpose [1,0,2] | 1.715 (1.707 / 1.723) | 1.727 (1.724 / 1.730) | **0.859 (0.834 / 0.887)** | not_configured (no JAX Metal device; CPU fallback is forbidden for the mac-gpu profile) |
| `mac_transpose_3d_201` | 256x256x240 transpose [2,0,1] | 1.713 (1.708 / 1.725) | 1.721 (1.715 / 1.733) | **0.872 (0.840 / 0.896)** | not_configured (no JAX Metal device; CPU fallback is forbidden for the mac-gpu profile) |
