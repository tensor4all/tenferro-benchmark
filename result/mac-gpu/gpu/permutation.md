# GPU Permutation Benchmark Results

- Target profile: `mac-gpu`
- Suite: `gpu/permutation`
- Suite file: `benchmarks/gpu/permutation-mac.yaml`
- Timestamp: `2026-07-29T11:06:52Z`
- tenferro-rs commit: `57c41c218356e871532e4dea4465a6bcf2d6c8fb`

- Metal device: `Apple M5 Max`
- Runtime: `wgpu/Metal`

`tenferro-webgpu-transpose-baseline` is the historical profile identifier for the native CubeCL structural path; the recorded tenferro revision and `TENFERRO_NATIVE_TRANSPOSE_TILE` value distinguish baseline and optimized runs. `tenferro-webgpu-to-contiguous` measures the public strided-view materialization path. PyTorch MPS and JAX Metal are logical framework comparisons; a backend without a Metal runtime is reported as `not_configured` and is never allowed to fall back to CPU. Every timed iteration ends with explicit device synchronization; correctness downloads and JIT compilation are outside timing. Both tenferro and PyTorch MPS allocate a fresh destination per timed call; the medians remain end-to-end host-API measurements rather than isolated kernel timings.

The mac-gpu patterns contain 3^15 or roughly 15 million f32 elements (about 60 MiB per tensor). The entry-gate baseline and development tile sweep were collected on M5. This is an accepted development substitute, but the final tile sweep must still be run on the issue's target M4 machine.

Device-copy baseline: `memcpy-metal-d2d` median 0.491 ms, 256.14 GB/s for `mac_memcpy_1d`. It is a bandwidth reference, not a permutation participant, so it is not shown as a table column.

Median (p25 / p75) in ms. Missing backends are shown as `-`; the fastest backend per pattern is **bolded**.

| pattern | label | tenferro-rs wgpu native (ms) | tenferro-rs wgpu to_contiguous (ms) | PyTorch MPS (ms) | JAX Metal (ms) |
|---|---|---:|---:|---:|---:|
| `mac_cyclic_15d_3` | 15D 3^15 cyclic | 1.462 (1.453 / 1.468) | **1.460 (1.459 / 1.465)** | 3.457 (3.422 / 3.752) | not_configured (no JAX Metal device; CPU fallback is forbidden for the mac-gpu profile) |
| `mac_reverse_15d_3` | 15D 3^15 reverse | 2.628 (2.623 / 3.885) | 2.619 (2.616 / 2.628) | **2.454 (2.391 / 2.472)** | not_configured (no JAX Metal device; CPU fallback is forbidden for the mac-gpu profile) |
| `mac_reverse_21d` | 21D 15x2^20 reverse | 2.756 (2.729 / 3.878) | **2.620 (2.616 / 2.631)** | skipped (PyTorch MPS supports at most 16 dimensions) | not_configured (no JAX Metal device; CPU fallback is forbidden for the mac-gpu profile) |
| `mac_rotation_6d` | 64x32x32x15x4x4 rotation | **1.462 (1.459 / 1.473)** | 1.465 (1.462 / 1.473) | 2.664 (2.588 / 2.770) | not_configured (no JAX Metal device; CPU fallback is forbidden for the mac-gpu profile) |
| `mac_tn_contiguous` | 21D contiguous source, TN permutation | 1.462 (1.462 / 1.465) | **1.458 (1.457 / 1.465)** | skipped (PyTorch MPS supports at most 16 dimensions) | not_configured (no JAX Metal device; CPU fallback is forbidden for the mac-gpu profile) |
| `mac_tn_scattered` | 21D scattered source -> col-major | - | **1.464 (1.462 / 1.474)** | skipped (PyTorch MPS supports at most 16 dimensions) | not_configured (no JAX Metal device; CPU fallback is forbidden for the mac-gpu profile) |
| `mac_transpose_2d` | 4096x3840 transpose [1,0] | **1.467 (1.464 / 1.505)** | 1.583 (1.571 / 1.682) | 1.910 (1.776 / 2.000) | not_configured (no JAX Metal device; CPU fallback is forbidden for the mac-gpu profile) |
| `mac_transpose_3d_102` | 256x256x240 transpose [1,0,2] | 1.475 (1.472 / 1.495) | 1.460 (1.459 / 1.466) | **0.879 (0.855 / 0.968)** | not_configured (no JAX Metal device; CPU fallback is forbidden for the mac-gpu profile) |
| `mac_transpose_3d_201` | 256x256x240 transpose [2,0,1] | 1.470 (1.468 / 1.488) | 1.460 (1.457 / 1.466) | **1.092 (1.074 / 1.137)** | not_configured (no JAX Metal device; CPU fallback is forbidden for the mac-gpu profile) |
