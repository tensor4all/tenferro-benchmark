# GPU Permutation Benchmark Results

- Target profile: `mac-gpu`
- Suite: `gpu/permutation`
- Suite file: `benchmarks/gpu/permutation-mac.yaml`
- Timestamp: `2026-07-29T03:14:38Z`
- tenferro-rs commit: `0a9e283d828b208c3c2e19fdfa7ee457ca41a594`

- Metal device: `Apple M5 Max`
- Runtime: `wgpu/Metal`

`tenferro-webgpu-transpose-baseline` is the historical profile identifier for the native CubeCL structural path; the recorded tenferro revision and `TENFERRO_NATIVE_TRANSPOSE_TILE` value distinguish baseline and optimized runs. `tenferro-webgpu-to-contiguous` measures the public strided-view materialization path. PyTorch MPS and JAX Metal are logical framework comparisons; a backend without a Metal runtime is reported as `not_configured` and is never allowed to fall back to CPU. Every timed iteration ends with explicit device synchronization; correctness downloads and JIT compilation are outside timing. tenferro allocates a fresh destination per call, while PyTorch MPS reuses one destination allocation per pattern, so their host-API medians are not a kernel-only comparison.

The mac-gpu patterns contain 3^15 or roughly 15 million f32 elements (about 60 MiB per tensor). The entry-gate baseline and development tile sweep were collected on M5. This is an accepted development substitute, but the final tile sweep must still be run on the issue's target M4 machine.

Device-copy baseline: `memcpy-metal-d2d` median 0.774 ms, 162.57 GB/s for `mac_memcpy_1d`. It is a bandwidth reference, not a permutation participant, so it is not shown as a table column.

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs wgpu native (ms) | tenferro-rs wgpu to_contiguous (ms) | PyTorch MPS (ms) | JAX Metal (ms) |
|---|---|---:|---:|---:|---:|
| `mac_cyclic_15d_3` | 15D 3^15 cyclic | 1.754 (1.614 / 3.227) | **1.734 (1.732 / 1.756)** | 3.413 (3.357 / 3.720) | not_configured (no JAX Metal device; CPU fallback is forbidden for the mac-gpu profile) |
| `mac_reverse_15d_3` | 15D 3^15 reverse | 3.106 (2.972 / 3.158) | 3.103 (2.883 / 3.301) | **2.307 (2.185 / 3.229)** | not_configured (no JAX Metal device; CPU fallback is forbidden for the mac-gpu profile) |
| `mac_reverse_21d` | 21D 15x2^20 reverse | 3.644 (3.295 / 4.891) | **3.359 (3.300 / 3.638)** | skipped (PyTorch MPS supports at most 16 dimensions) | not_configured (no JAX Metal device; CPU fallback is forbidden for the mac-gpu profile) |
| `mac_rotation_6d` | 64x32x32x15x4x4 rotation | 1.738 (1.731 / 1.743) | **1.730 (1.728 / 1.733)** | 2.539 (2.513 / 2.763) | not_configured (no JAX Metal device; CPU fallback is forbidden for the mac-gpu profile) |
| `mac_tn_contiguous` | 21D contiguous source, TN permutation | 1.752 (1.735 / 1.776) | **1.730 (1.729 / 1.745)** | skipped (PyTorch MPS supports at most 16 dimensions) | not_configured (no JAX Metal device; CPU fallback is forbidden for the mac-gpu profile) |
| `mac_tn_scattered` | 21D scattered source -> col-major | - | **1.757 (1.754 / 1.771)** | skipped (PyTorch MPS supports at most 16 dimensions) | not_configured (no JAX Metal device; CPU fallback is forbidden for the mac-gpu profile) |
| `mac_transpose_2d` | 4096x3840 transpose [1,0] | 1.773 (1.769 / 1.820) | 1.728 (1.675 / 1.731) | **0.936 (0.912 / 1.085)** | not_configured (no JAX Metal device; CPU fallback is forbidden for the mac-gpu profile) |
| `mac_transpose_3d_102` | 256x256x240 transpose [1,0,2] | 1.739 (1.721 / 1.780) | 1.731 (1.728 / 1.732) | **0.876 (0.836 / 0.902)** | not_configured (no JAX Metal device; CPU fallback is forbidden for the mac-gpu profile) |
| `mac_transpose_3d_201` | 256x256x240 transpose [2,0,1] | 1.750 (1.740 / 1.801) | 1.735 (1.730 / 1.750) | **0.856 (0.821 / 0.879)** | not_configured (no JAX Metal device; CPU fallback is forbidden for the mac-gpu profile) |
