# GPU command-wall observations

Wall time includes preparation, validation, sampling and reporting. These are end-to-end workflow costs; operation-performance ratios are not inferred. Missing phase instrumentation is displayed as unavailable, not zero. Filtered or unsuccessful runs do not establish full suite coverage.

## gpu/dense / 20261013_234113

Raw run: `data/results/nvidia-gpu/gpu/dense/20261013_234113`.

tenferro-rs `9655b4e0daced17e0aa35a30d35710b069e10644`; harness `9e89674e200526e294a01595152425befafd53f5`. GPU: NVIDIA GeForce RTX 3060 (`GPU-a78d5217-eba3-72c2-3d5b-8ae496ebbc2e`).

Command through this suite: **69.863 s**; collection including build/release delays: 61.754 s; report generation/copy: 0.139 s.

Scope: script clock-read through this suite report copy; devcontainer CLI startup and final wall-record write excluded.

| Backend | Build s | Process s (startup/setup/checks/samples) |
|---|---:|---:|
| tenferro-cuda-eager | 5.016 | 16.183 |
| pytorch-cuda | 0.000 | 14.185 |
| jax-cuda | 0.000 | 20.066 |

Case phases below exclude process startup/JSONL writing. Preparation includes input generation, transfers and reference preparation; lazy initialization can occur in that phase. Sampler walls include untimed batch preparation/cleanup. Command totals are cumulative across suites in one invocation, so do not sum those totals.

| Problem | Backend | Status | Backend init s | Prepare s | Compile s | Probe/init s | Warmup s | Calibration s | Sampling s | Validation s |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dense_matmul_f64_3072 | tenferro-cuda-eager | ok | 0.000 | 2.376 | — | 0.343 | 0.941 | 0.314 | 2.196 | 2.587 |
| dense_batched_matmul_f64_b16_32 | tenferro-cuda-eager | ok | 0.000 | 0.009 | — | 0.000 | 0.000 | 0.006 | 0.018 | 0.000 |
| dense_qr_f64_1536 | tenferro-cuda-eager | ok | 0.000 | 0.041 | — | 0.344 | 0.191 | 0.095 | 0.477 | 0.655 |
| dense_solve_f64_512_rhs16 | tenferro-cuda-eager | ok | 0.000 | 0.252 | — | 0.401 | 0.008 | 0.004 | 0.020 | 0.002 |
| dense_solve_f64_2048_rhs16 | tenferro-cuda-eager | ok | 0.000 | 0.102 | — | 0.050 | 0.097 | 0.048 | 0.242 | 0.057 |
| dense_svd_f64_256 | tenferro-cuda-eager | ok | 0.000 | 0.007 | — | 0.133 | 0.164 | 0.082 | 0.409 | 0.006 |
| dense_eigh_f64_1024 | tenferro-cuda-eager | ok | 0.000 | 2.018 | — | 0.085 | 0.158 | 0.075 | 0.372 | 0.189 |
| dense_matmul_f64_3072 | pytorch-cuda | ok | 1.686 | 1.761 | 0.000 | 0.357 | 0.941 | 0.314 | 2.196 | 1.059 |
| dense_batched_matmul_f64_b16_32 | pytorch-cuda | ok | 0.000 | 0.002 | 0.000 | 0.000 | 0.000 | 0.007 | 0.021 | 0.001 |
| dense_qr_f64_1536 | pytorch-cuda | ok | 0.000 | 0.080 | 0.000 | 0.131 | 0.190 | 0.095 | 0.475 | 0.844 |
| dense_solve_f64_512_rhs16 | pytorch-cuda | ok | 0.000 | 0.014 | 0.000 | 0.019 | 0.006 | 0.003 | 0.015 | 0.005 |
| dense_solve_f64_2048_rhs16 | pytorch-cuda | ok | 0.000 | 0.148 | 0.000 | 0.048 | 0.096 | 0.048 | 0.240 | 0.068 |
| dense_svd_f64_256 | pytorch-cuda | ok | 0.000 | 0.003 | 0.000 | 0.087 | 0.168 | 0.084 | 0.419 | 0.096 |
| dense_eigh_f64_1024 | pytorch-cuda | ok | 0.000 | 0.066 | 0.000 | 0.084 | 0.149 | 0.075 | 0.373 | 0.399 |
| dense_matmul_f64_3072 | jax-cuda | ok | 1.020 | 0.891 | 5.751 | 0.275 | 0.815 | 0.272 | 1.902 | 1.455 |
| dense_batched_matmul_f64_b16_32 | jax-cuda | ok | 0.000 | 0.026 | 0.365 | 0.001 | 0.000 | 0.006 | 0.019 | 0.002 |
| dense_qr_f64_1536 | jax-cuda | ok | 0.000 | 0.092 | 0.213 | 0.103 | 0.198 | 0.097 | 0.486 | 1.091 |
| dense_solve_f64_512_rhs16 | jax-cuda | ok | 0.000 | 0.018 | 0.187 | 0.008 | 0.007 | 0.003 | 0.017 | 0.008 |
| dense_solve_f64_2048_rhs16 | jax-cuda | ok | 0.000 | 0.203 | 0.198 | 0.049 | 0.098 | 0.049 | 0.245 | 0.125 |
| dense_svd_f64_256 | jax-cuda | ok | 0.000 | 0.006 | 0.234 | 0.088 | 0.165 | 0.083 | 0.413 | 0.101 |
| dense_eigh_f64_1024 | jax-cuda | ok | 0.000 | 0.076 | 0.199 | 0.082 | 0.151 | 0.076 | 0.378 | 0.449 |
## gpu/einsum / 20261013_234113

Raw run: `data/results/nvidia-gpu/gpu/einsum/20261013_234113`.

tenferro-rs `9655b4e0daced17e0aa35a30d35710b069e10644`; harness `9e89674e200526e294a01595152425befafd53f5`. GPU: NVIDIA GeForce RTX 3060 (`GPU-a78d5217-eba3-72c2-3d5b-8ae496ebbc2e`).

Command through this suite: **116.029 s**; collection including build/release delays: 39.178 s; report generation/copy: 0.137 s.

Scope: script clock-read through this suite report copy; devcontainer CLI startup and final wall-record write excluded.

| Backend | Build s | Process s (startup/setup/checks/samples) |
|---|---:|---:|
| tenferro-cuda-eager | 0.000 | 9.321 |
| pytorch-cuda | 0.000 | 10.278 |
| jax-cuda | 0.000 | 13.284 |

Case phases below exclude process startup/JSONL writing. Preparation includes input generation, transfers and reference preparation; lazy initialization can occur in that phase. Sampler walls include untimed batch preparation/cleanup. Command totals are cumulative across suites in one invocation, so do not sum those totals.

| Problem | Backend | Status | Backend init s | Prepare s | Compile s | Probe/init s | Warmup s | Calibration s | Sampling s | Validation s |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| einsum_bin_matmul_3072_f64 | tenferro-cuda-eager | ok | 0.000 | 2.329 | — | 0.343 | 0.946 | 0.315 | 2.208 | 2.692 |
| einsum_bin_matmul_3072_f64 | pytorch-cuda | ok | 1.722 | 1.768 | 0.000 | 0.350 | 0.946 | 0.316 | 2.208 | 2.011 |
| einsum_bin_matmul_3072_f64 | jax-cuda | ok | 1.002 | 0.847 | 5.913 | 0.274 | 0.815 | 0.272 | 1.902 | 1.345 |
## gpu/tensornetwork / 20261013_234113

Raw run: `data/results/nvidia-gpu/gpu/tensornetwork/20261013_234113`.

tenferro-rs `9655b4e0daced17e0aa35a30d35710b069e10644`; harness `9e89674e200526e294a01595152425befafd53f5`. GPU: NVIDIA GeForce RTX 3060 (`GPU-a78d5217-eba3-72c2-3d5b-8ae496ebbc2e`).

Command through this suite: **155.349 s**; collection including build/release delays: 32.441 s; report generation/copy: 0.137 s.

Scope: script clock-read through this suite report copy; devcontainer CLI startup and final wall-record write excluded.

| Backend | Build s | Process s (startup/setup/checks/samples) |
|---|---:|---:|
| tenferro-cuda-eager | 0.000 | 7.695 |
| pytorch-cuda | 0.000 | 8.246 |
| jax-cuda | 0.000 | 10.196 |

Case phases below exclude process startup/JSONL writing. Preparation includes input generation, transfers and reference preparation; lazy initialization can occur in that phase. Sampler walls include untimed batch preparation/cleanup. Command totals are cumulative across suites in one invocation, so do not sum those totals.

| Problem | Backend | Status | Backend init s | Prepare s | Compile s | Probe/init s | Warmup s | Calibration s | Sampling s | Validation s |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tensornetwork_permutation_optimized_f32 | tenferro-cuda-eager | ok | 0.000 | 2.102 | — | 3.448 | 0.373 | 0.118 | 1.144 | 0.002 |
| tensornetwork_permutation_optimized_f32 | pytorch-cuda | ok | 1.701 | 0.048 | 0.000 | 0.438 | 1.011 | 0.336 | 3.372 | 0.339 |
| tensornetwork_permutation_optimized_f32 | jax-cuda | ok | 0.970 | 0.095 | 3.327 | 0.452 | 0.859 | 0.286 | 2.860 | 0.386 |
