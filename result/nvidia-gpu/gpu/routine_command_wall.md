# GPU command-wall observations

Wall time includes preparation, validation, sampling and reporting. These are end-to-end workflow costs; operation-performance ratios are not inferred. Missing phase instrumentation is displayed as unavailable, not zero. Filtered or unsuccessful runs do not establish full suite coverage.

## gpu/dense / 20261013_232208

Raw run: `data/results/nvidia-gpu/gpu/dense/20261013_232208`.

tenferro-rs `9655b4e0daced17e0aa35a30d35710b069e10644`; harness `020332f02732fe193a6ccce5d1e771046b1e511a`. GPU: NVIDIA GeForce RTX 3060 (`GPU-a78d5217-eba3-72c2-3d5b-8ae496ebbc2e`).

Command through this suite: **64.995 s**; collection including build/release delays: 56.940 s; report generation/copy: 0.141 s.

Scope: script clock-read through this suite report copy; devcontainer CLI startup and final wall-record write excluded.

| Backend | Build s | Process s (startup/setup/checks/samples) |
|---|---:|---:|
| tenferro-cuda-eager | 0.368 | 16.002 |
| pytorch-cuda | 0.000 | 14.199 |
| jax-cuda | 0.000 | 20.069 |

Case phases below exclude process startup/JSONL writing. Preparation includes input generation, transfers and reference preparation; lazy initialization can occur in that phase. Sampler walls include untimed batch preparation/cleanup. Command totals are cumulative across suites in one invocation, so do not sum those totals.

| Problem | Backend | Status | Backend init s | Prepare s | Compile s | Probe/init s | Warmup s | Calibration s | Sampling s | Validation s |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dense_matmul_f64_3072 | tenferro-cuda-eager | ok | 0.000 | 2.314 | — | 0.341 | 0.940 | 0.313 | 2.193 | 2.847 |
| dense_batched_matmul_f64_b16_32 | tenferro-cuda-eager | ok | 0.000 | 0.008 | — | 0.000 | 0.000 | 0.006 | 0.019 | 0.000 |
| dense_qr_f64_1536 | tenferro-cuda-eager | ok | 0.000 | 0.042 | — | 0.231 | 0.194 | 0.095 | 0.476 | 0.649 |
| dense_solve_f64_512_rhs16 | tenferro-cuda-eager | ok | 0.000 | 0.254 | — | 0.181 | 0.008 | 0.004 | 0.020 | 0.002 |
| dense_solve_f64_2048_rhs16 | tenferro-cuda-eager | ok | 0.000 | 0.103 | — | 0.051 | 0.097 | 0.048 | 0.241 | 0.057 |
| dense_svd_f64_256 | tenferro-cuda-eager | ok | 0.000 | 0.008 | — | 0.107 | 0.164 | 0.082 | 0.409 | 0.006 |
| dense_eigh_f64_1024 | tenferro-cuda-eager | ok | 0.000 | 2.015 | — | 0.085 | 0.154 | 0.074 | 0.372 | 0.195 |
| dense_matmul_f64_3072 | pytorch-cuda | ok | 1.711 | 1.795 | 0.000 | 0.349 | 0.939 | 0.313 | 2.192 | 1.044 |
| dense_batched_matmul_f64_b16_32 | pytorch-cuda | ok | 0.000 | 0.002 | 0.000 | 0.000 | 0.000 | 0.008 | 0.020 | 0.001 |
| dense_qr_f64_1536 | pytorch-cuda | ok | 0.000 | 0.076 | 0.000 | 0.127 | 0.190 | 0.095 | 0.475 | 0.851 |
| dense_solve_f64_512_rhs16 | pytorch-cuda | ok | 0.000 | 0.014 | 0.000 | 0.019 | 0.006 | 0.003 | 0.015 | 0.005 |
| dense_solve_f64_2048_rhs16 | pytorch-cuda | ok | 0.000 | 0.152 | 0.000 | 0.048 | 0.096 | 0.048 | 0.239 | 0.066 |
| dense_svd_f64_256 | pytorch-cuda | ok | 0.000 | 0.003 | 0.000 | 0.087 | 0.167 | 0.084 | 0.419 | 0.096 |
| dense_eigh_f64_1024 | pytorch-cuda | ok | 0.000 | 0.066 | 0.000 | 0.084 | 0.149 | 0.075 | 0.374 | 0.404 |
| dense_matmul_f64_3072 | jax-cuda | ok | 0.967 | 0.876 | 5.849 | 0.275 | 0.815 | 0.272 | 1.902 | 1.428 |
| dense_batched_matmul_f64_b16_32 | jax-cuda | ok | 0.000 | 0.025 | 0.368 | 0.001 | 0.000 | 0.006 | 0.019 | 0.002 |
| dense_qr_f64_1536 | jax-cuda | ok | 0.000 | 0.090 | 0.199 | 0.103 | 0.197 | 0.097 | 0.485 | 1.089 |
| dense_solve_f64_512_rhs16 | jax-cuda | ok | 0.000 | 0.018 | 0.222 | 0.008 | 0.007 | 0.003 | 0.017 | 0.008 |
| dense_solve_f64_2048_rhs16 | jax-cuda | ok | 0.000 | 0.192 | 0.207 | 0.050 | 0.098 | 0.049 | 0.244 | 0.128 |
| dense_svd_f64_256 | jax-cuda | ok | 0.000 | 0.007 | 0.253 | 0.088 | 0.165 | 0.082 | 0.412 | 0.101 |
| dense_eigh_f64_1024 | jax-cuda | ok | 0.000 | 0.075 | 0.234 | 0.082 | 0.151 | 0.075 | 0.377 | 0.432 |
## gpu/einsum / 20261013_232208

Raw run: `data/results/nvidia-gpu/gpu/einsum/20261013_232208`.

tenferro-rs `9655b4e0daced17e0aa35a30d35710b069e10644`; harness `020332f02732fe193a6ccce5d1e771046b1e511a`. GPU: NVIDIA GeForce RTX 3060 (`GPU-a78d5217-eba3-72c2-3d5b-8ae496ebbc2e`).

Command through this suite: **111.483 s**; collection including build/release delays: 39.417 s; report generation/copy: 0.153 s.

Scope: script clock-read through this suite report copy; devcontainer CLI startup and final wall-record write excluded.

| Backend | Build s | Process s (startup/setup/checks/samples) |
|---|---:|---:|
| tenferro-cuda-eager | 0.000 | 9.446 |
| pytorch-cuda | 0.000 | 10.322 |
| jax-cuda | 0.000 | 13.343 |

Case phases below exclude process startup/JSONL writing. Preparation includes input generation, transfers and reference preparation; lazy initialization can occur in that phase. Sampler walls include untimed batch preparation/cleanup. Command totals are cumulative across suites in one invocation, so do not sum those totals.

| Problem | Backend | Status | Backend init s | Prepare s | Compile s | Probe/init s | Warmup s | Calibration s | Sampling s | Validation s |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| einsum_bin_matmul_3072_f64 | tenferro-cuda-eager | ok | 0.000 | 2.337 | — | 0.344 | 0.946 | 0.315 | 2.208 | 2.768 |
| einsum_bin_matmul_3072_f64 | pytorch-cuda | ok | 1.766 | 1.760 | 0.000 | 0.354 | 0.947 | 0.316 | 2.208 | 1.980 |
| einsum_bin_matmul_3072_f64 | jax-cuda | ok | 0.997 | 0.874 | 5.887 | 0.274 | 0.815 | 0.272 | 1.902 | 1.447 |
## gpu/tensornetwork / 20261013_232208

Raw run: `data/results/nvidia-gpu/gpu/tensornetwork/20261013_232208`.

tenferro-rs `9655b4e0daced17e0aa35a30d35710b069e10644`; harness `020332f02732fe193a6ccce5d1e771046b1e511a`. GPU: NVIDIA GeForce RTX 3060 (`GPU-a78d5217-eba3-72c2-3d5b-8ae496ebbc2e`).

Command through this suite: **148.916 s**; collection including build/release delays: 30.404 s; report generation/copy: 0.148 s.

Scope: script clock-read through this suite report copy; devcontainer CLI startup and final wall-record write excluded.

| Backend | Build s | Process s (startup/setup/checks/samples) |
|---|---:|---:|
| tenferro-cuda-eager | 0.000 | 5.591 |
| pytorch-cuda | 0.000 | 8.257 |
| jax-cuda | 0.000 | 10.251 |

Case phases below exclude process startup/JSONL writing. Preparation includes input generation, transfers and reference preparation; lazy initialization can occur in that phase. Sampler walls include untimed batch preparation/cleanup. Command totals are cumulative across suites in one invocation, so do not sum those totals.

| Problem | Backend | Status | Backend init s | Prepare s | Compile s | Probe/init s | Warmup s | Calibration s | Sampling s | Validation s |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tensornetwork_permutation_optimized_f32 | tenferro-cuda-eager | ok | 0.000 | 2.136 | — | 1.319 | 0.372 | 0.118 | 1.146 | 0.002 |
| tensornetwork_permutation_optimized_f32 | pytorch-cuda | ok | 1.706 | 0.047 | 0.000 | 0.435 | 1.012 | 0.337 | 3.378 | 0.339 |
| tensornetwork_permutation_optimized_f32 | jax-cuda | ok | 1.019 | 0.095 | 3.284 | 0.447 | 0.858 | 0.286 | 2.861 | 0.386 |
