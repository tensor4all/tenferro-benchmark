# GPU command-wall observations

Wall time includes preparation, validation, sampling and reporting. These are end-to-end workflow costs; operation-performance ratios are not inferred. Missing phase instrumentation is displayed as unavailable, not zero. Filtered or unsuccessful runs do not establish full suite coverage.

## gpu/dense / 20261014_005047

Raw run: `data/results/nvidia-gpu/gpu/dense/20261014_005047`.

tenferro-rs `9655b4e0daced17e0aa35a30d35710b069e10644`; harness `25f190048045c01cb363384ba45b2a0b0b640db8`. GPU: NVIDIA GeForce RTX 3060 (`GPU-a78d5217-eba3-72c2-3d5b-8ae496ebbc2e`).

Command through this suite: **238.737 s**; collection including build/release delays: 231.161 s; report generation/copy: 0.162 s.

Scope: script clock-read through this suite report copy; devcontainer CLI startup and final wall-record write excluded.

| Backend | Build s | Process s (startup/setup/checks/samples) |
|---|---:|---:|
| tenferro-cuda-trace | 67.081 | 0.423 |
| tenferro-cuda-eager | 0.000 | 24.742 |
| pytorch-cuda | 0.000 | 29.626 |
| jax-cuda | 0.000 | 42.167 |
| cublaslt | 0.000 | 21.659 |
| cutlass | 0.000 | 20.882 |
| cusolver | 0.000 | 9.977 |

Case phases below exclude process startup/JSONL writing. Preparation includes input generation, transfers and reference preparation; lazy initialization can occur in that phase. Sampler walls include untimed batch preparation/cleanup. Command totals are cumulative across suites in one invocation, so do not sum those totals.

| Problem | Backend | Status | Backend init s | Prepare s | Compile s | Probe/init s | Warmup s | Calibration s | Sampling s | Validation s |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dense_matmul_f64_3072 | tenferro-cuda-trace | unsupported | — | — | — | — | — | — | — | — |
| dense_batched_matmul_f64_b1024_256 | tenferro-cuda-trace | unsupported | — | — | — | — | — | — | — | — |
| dense_qr_f64_1536 | tenferro-cuda-trace | unsupported | — | — | — | — | — | — | — | — |
| dense_solve_f64_512_rhs16 | tenferro-cuda-trace | unsupported | — | — | — | — | — | — | — | — |
| dense_solve_f64_1024_rhs16 | tenferro-cuda-trace | unsupported | — | — | — | — | — | — | — | — |
| dense_solve_f64_2048_rhs16 | tenferro-cuda-trace | unsupported | — | — | — | — | — | — | — | — |
| dense_solve_f64_4096_rhs16 | tenferro-cuda-trace | unsupported | — | — | — | — | — | — | — | — |
| dense_solve_f64_2048_rhs128 | tenferro-cuda-trace | unsupported | — | — | — | — | — | — | — | — |
| dense_svd_f64_256 | tenferro-cuda-trace | unsupported | — | — | — | — | — | — | — | — |
| dense_eigh_f64_1024 | tenferro-cuda-trace | unsupported | — | — | — | — | — | — | — | — |
| dense_batched_matmul_f64_b16_32 | tenferro-cuda-trace | unsupported | — | — | — | — | — | — | — | — |
| dense_matmul_f64_3072 | tenferro-cuda-eager | ok | 0.000 | 2.336 | — | 0.342 | 0.314 | 0.314 | 0.941 | 2.825 |
| dense_batched_matmul_f64_b1024_256 | tenferro-cuda-eager | ok | 0.000 | 1.982 | — | 0.200 | 0.186 | 0.186 | 0.559 | 5.048 |
| dense_qr_f64_1536 | tenferro-cuda-eager | ok | 0.000 | 0.043 | — | 0.340 | 0.096 | 0.096 | 0.286 | 0.624 |
| dense_solve_f64_512_rhs16 | tenferro-cuda-eager | ok | 0.000 | 0.255 | — | 0.400 | 0.004 | 0.004 | 0.012 | 0.002 |
| dense_solve_f64_1024_rhs16 | tenferro-cuda-eager | ok | 0.000 | 0.051 | — | 0.015 | 0.011 | 0.011 | 0.033 | 0.007 |
| dense_solve_f64_2048_rhs16 | tenferro-cuda-eager | ok | 0.000 | 0.094 | — | 0.051 | 0.048 | 0.048 | 0.145 | 0.073 |
| dense_solve_f64_4096_rhs16 | tenferro-cuda-eager | ok | 0.000 | 0.310 | — | 0.290 | 0.288 | 0.288 | 0.864 | 0.251 |
| dense_solve_f64_2048_rhs128 | tenferro-cuda-eager | ok | 0.000 | 0.074 | — | 0.057 | 0.055 | 0.055 | 0.164 | 0.092 |
| dense_svd_f64_256 | tenferro-cuda-eager | ok | 0.000 | 0.007 | — | 0.133 | 0.082 | 0.082 | 0.245 | 0.006 |
| dense_eigh_f64_1024 | tenferro-cuda-eager | ok | 0.000 | 2.023 | — | 0.085 | 0.079 | 0.079 | 0.225 | 0.201 |
| dense_batched_matmul_f64_b16_32 | tenferro-cuda-eager | ok | 0.000 | 0.007 | — | 0.000 | 0.000 | 0.006 | 0.009 | 0.000 |
| dense_matmul_f64_3072 | pytorch-cuda | ok | 1.759 | 1.763 | 0.000 | 0.353 | 0.314 | 0.314 | 0.941 | 1.055 |
| dense_batched_matmul_f64_b1024_256 | pytorch-cuda | ok | 0.000 | 7.388 | 0.000 | 0.194 | 0.186 | 0.187 | 0.563 | 5.369 |
| dense_qr_f64_1536 | pytorch-cuda | ok | 0.000 | 0.078 | 0.000 | 0.132 | 0.101 | 0.096 | 0.287 | 0.847 |
| dense_solve_f64_512_rhs16 | pytorch-cuda | ok | 0.000 | 0.015 | 0.000 | 0.019 | 0.003 | 0.003 | 0.009 | 0.005 |
| dense_solve_f64_1024_rhs16 | pytorch-cuda | ok | 0.000 | 0.032 | 0.000 | 0.011 | 0.011 | 0.011 | 0.032 | 0.015 |
| dense_solve_f64_2048_rhs16 | pytorch-cuda | ok | 0.000 | 0.156 | 0.000 | 0.048 | 0.048 | 0.048 | 0.145 | 0.064 |
| dense_solve_f64_4096_rhs16 | pytorch-cuda | ok | 0.000 | 0.838 | 0.000 | 0.289 | 0.289 | 0.289 | 0.868 | 0.368 |
| dense_solve_f64_2048_rhs128 | pytorch-cuda | ok | 0.000 | 0.163 | 0.000 | 0.054 | 0.054 | 0.054 | 0.161 | 0.118 |
| dense_svd_f64_256 | pytorch-cuda | ok | 0.000 | 0.003 | 0.000 | 0.087 | 0.084 | 0.084 | 0.253 | 0.096 |
| dense_eigh_f64_1024 | pytorch-cuda | ok | 0.000 | 0.069 | 0.000 | 0.085 | 0.075 | 0.075 | 0.225 | 0.404 |
| dense_batched_matmul_f64_b16_32 | pytorch-cuda | ok | 0.000 | 0.001 | 0.000 | 0.000 | 0.000 | 0.008 | 0.010 | 0.001 |
| dense_matmul_f64_3072 | jax-cuda | ok | 0.948 | 0.883 | 5.769 | 0.274 | 0.272 | 0.272 | 0.815 | 1.436 |
| dense_batched_matmul_f64_b1024_256 | jax-cuda | ok | 0.000 | 7.628 | 4.296 | 0.165 | 0.167 | 0.161 | 0.484 | 5.943 |
| dense_qr_f64_1536 | jax-cuda | ok | 0.000 | 0.085 | 0.183 | 0.109 | 0.098 | 0.097 | 0.291 | 1.057 |
| dense_solve_f64_512_rhs16 | jax-cuda | ok | 0.000 | 0.018 | 0.211 | 0.009 | 0.003 | 0.003 | 0.010 | 0.008 |
| dense_solve_f64_1024_rhs16 | jax-cuda | ok | 0.000 | 0.063 | 0.241 | 0.012 | 0.011 | 0.011 | 0.033 | 0.025 |
| dense_solve_f64_2048_rhs16 | jax-cuda | ok | 0.000 | 0.260 | 0.235 | 0.050 | 0.049 | 0.049 | 0.147 | 0.129 |
| dense_solve_f64_4096_rhs16 | jax-cuda | ok | 0.000 | 1.067 | 0.238 | 0.291 | 0.290 | 0.290 | 0.870 | 0.539 |
| dense_solve_f64_2048_rhs128 | jax-cuda | ok | 0.000 | 0.250 | 0.213 | 0.055 | 0.054 | 0.054 | 0.163 | 0.215 |
| dense_svd_f64_256 | jax-cuda | ok | 0.000 | 0.004 | 0.289 | 0.088 | 0.083 | 0.083 | 0.248 | 0.102 |
| dense_eigh_f64_1024 | jax-cuda | ok | 0.000 | 0.107 | 0.209 | 0.085 | 0.075 | 0.076 | 0.226 | 0.458 |
| dense_batched_matmul_f64_b16_32 | jax-cuda | ok | 0.000 | 0.018 | 0.390 | 0.001 | 0.000 | 0.006 | 0.008 | 0.002 |
| dense_matmul_f64_3072 | cublaslt | ok | 0.273 | 1.754 | 0.000 | 0.356 | 0.314 | 0.314 | 0.941 | 1.062 |
| dense_batched_matmul_f64_b1024_256 | cublaslt | ok | 0.000 | 7.374 | 0.000 | 0.198 | 0.186 | 0.186 | 0.559 | 5.398 |
| dense_qr_f64_1536 | cublaslt | unsupported | — | — | — | — | — | — | — | — |
| dense_solve_f64_512_rhs16 | cublaslt | unsupported | — | — | — | — | — | — | — | — |
| dense_solve_f64_1024_rhs16 | cublaslt | unsupported | — | — | — | — | — | — | — | — |
| dense_solve_f64_2048_rhs16 | cublaslt | unsupported | — | — | — | — | — | — | — | — |
| dense_solve_f64_4096_rhs16 | cublaslt | unsupported | — | — | — | — | — | — | — | — |
| dense_solve_f64_2048_rhs128 | cublaslt | unsupported | — | — | — | — | — | — | — | — |
| dense_svd_f64_256 | cublaslt | unsupported | — | — | — | — | — | — | — | — |
| dense_eigh_f64_1024 | cublaslt | unsupported | — | — | — | — | — | — | — | — |
| dense_batched_matmul_f64_b16_32 | cublaslt | ok | 0.000 | 0.002 | 0.000 | 0.000 | 0.000 | 0.008 | 0.009 | 0.001 |
| dense_matmul_f64_3072 | cutlass | ok | 1.929 | 1.753 | — | 0.334 | 0.320 | 0.319 | 0.959 | 0.939 |
| dense_batched_matmul_f64_b1024_256 | cutlass | ok | 0.000 | 7.402 | — | 0.200 | 0.188 | 0.188 | 0.563 | 4.586 |
| dense_qr_f64_1536 | cutlass | unsupported | — | — | — | — | — | — | — | — |
| dense_solve_f64_512_rhs16 | cutlass | unsupported | — | — | — | — | — | — | — | — |
| dense_solve_f64_1024_rhs16 | cutlass | unsupported | — | — | — | — | — | — | — | — |
| dense_solve_f64_2048_rhs16 | cutlass | unsupported | — | — | — | — | — | — | — | — |
| dense_solve_f64_4096_rhs16 | cutlass | unsupported | — | — | — | — | — | — | — | — |
| dense_solve_f64_2048_rhs128 | cutlass | unsupported | — | — | — | — | — | — | — | — |
| dense_svd_f64_256 | cutlass | unsupported | — | — | — | — | — | — | — | — |
| dense_eigh_f64_1024 | cutlass | unsupported | — | — | — | — | — | — | — | — |
| dense_batched_matmul_f64_b16_32 | cutlass | ok | 0.000 | 0.002 | — | 0.000 | 0.000 | 0.006 | 0.009 | 0.001 |
| dense_matmul_f64_3072 | cusolver | unsupported | — | — | — | — | — | — | — | — |
| dense_batched_matmul_f64_b1024_256 | cusolver | unsupported | — | — | — | — | — | — | — | — |
| dense_qr_f64_1536 | cusolver | ok | 0.268 | 0.084 | 0.000 | 0.160 | 0.095 | 0.095 | 0.284 | 0.869 |
| dense_solve_f64_512_rhs16 | cusolver | ok | 0.000 | 0.015 | 0.000 | 0.011 | 0.003 | 0.003 | 0.009 | 0.004 |
| dense_solve_f64_1024_rhs16 | cusolver | ok | 0.000 | 0.037 | 0.000 | 0.011 | 0.011 | 0.011 | 0.032 | 0.015 |
| dense_solve_f64_2048_rhs16 | cusolver | ok | 0.000 | 0.158 | 0.000 | 0.048 | 0.048 | 0.048 | 0.143 | 0.064 |
| dense_solve_f64_4096_rhs16 | cusolver | ok | 0.000 | 0.850 | 0.000 | 0.287 | 0.286 | 0.286 | 0.858 | 0.361 |
| dense_solve_f64_2048_rhs128 | cusolver | ok | 0.000 | 0.163 | 0.000 | 0.053 | 0.053 | 0.053 | 0.160 | 0.117 |
| dense_svd_f64_256 | cusolver | ok | 0.000 | 0.003 | 0.000 | 0.036 | 0.027 | 0.027 | 0.082 | 0.039 |
| dense_eigh_f64_1024 | cusolver | ok | 0.000 | 0.075 | 0.000 | 0.084 | 0.074 | 0.074 | 0.223 | 0.416 |
| dense_batched_matmul_f64_b16_32 | cusolver | unsupported | — | — | — | — | — | — | — | — |
## gpu/einsum / 20261014_005047

Raw run: `data/results/nvidia-gpu/gpu/einsum/20261014_005047`.

tenferro-rs `9655b4e0daced17e0aa35a30d35710b069e10644`; harness `25f190048045c01cb363384ba45b2a0b0b640db8`. GPU: NVIDIA GeForce RTX 3060 (`GPU-a78d5217-eba3-72c2-3d5b-8ae496ebbc2e`).

Command through this suite: **302.395 s**; collection including build/release delays: 56.715 s; report generation/copy: 0.138 s.

Scope: script clock-read through this suite report copy; devcontainer CLI startup and final wall-record write excluded.

| Backend | Build s | Process s (startup/setup/checks/samples) |
|---|---:|---:|
| tenferro-cuda-trace | 0.000 | 0.370 |
| tenferro-cuda-eager | 0.000 | 7.460 |
| pytorch-cuda | 0.000 | 8.355 |
| jax-cuda | 0.000 | 11.648 |
| cublaslt | 0.000 | 8.330 |
| cutlass | 0.000 | 8.026 |

Case phases below exclude process startup/JSONL writing. Preparation includes input generation, transfers and reference preparation; lazy initialization can occur in that phase. Sampler walls include untimed batch preparation/cleanup. Command totals are cumulative across suites in one invocation, so do not sum those totals.

| Problem | Backend | Status | Backend init s | Prepare s | Compile s | Probe/init s | Warmup s | Calibration s | Sampling s | Validation s |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| einsum_bin_matmul_3072_f64 | tenferro-cuda-trace | unsupported | — | — | — | — | — | — | — | — |
| einsum_bin_matmul_3072_f64 | tenferro-cuda-eager | ok | 0.000 | 2.329 | — | 0.340 | 0.312 | 0.312 | 0.936 | 2.761 |
| einsum_bin_matmul_3072_f64 | pytorch-cuda | ok | 1.747 | 1.768 | 0.000 | 0.355 | 0.312 | 0.312 | 0.936 | 1.971 |
| einsum_bin_matmul_3072_f64 | jax-cuda | ok | 1.005 | 0.886 | 5.820 | 0.271 | 0.269 | 0.269 | 0.806 | 1.421 |
| einsum_bin_matmul_3072_f64 | cublaslt | ok | 0.238 | 1.769 | 0.000 | 0.349 | 0.312 | 0.312 | 0.937 | 1.999 |
| einsum_bin_matmul_3072_f64 | cutlass | ok | 1.916 | 1.766 | — | 0.332 | 0.318 | 0.318 | 0.954 | 1.534 |
## gpu/sparse / 20261014_005047

Raw run: `data/results/nvidia-gpu/gpu/sparse/20261014_005047`.

tenferro-rs `9655b4e0daced17e0aa35a30d35710b069e10644`; harness `25f190048045c01cb363384ba45b2a0b0b640db8`. GPU: NVIDIA GeForce RTX 3060 (`GPU-a78d5217-eba3-72c2-3d5b-8ae496ebbc2e`).

Command through this suite: **548.099 s**; collection including build/release delays: 238.680 s; report generation/copy: 0.152 s.

Scope: script clock-read through this suite report copy; devcontainer CLI startup and final wall-record write excluded.

| Backend | Build s | Process s (startup/setup/checks/samples) |
|---|---:|---:|
| pytorch-cuda | 0.000 | 85.345 |
| cusparse | 0.000 | 85.919 |
| ginkgo | 0.000 | 61.116 |

Case phases below exclude process startup/JSONL writing. Preparation includes input generation, transfers and reference preparation; lazy initialization can occur in that phase. Sampler walls include untimed batch preparation/cleanup. Command totals are cumulative across suites in one invocation, so do not sum those totals.

| Problem | Backend | Status | Backend init s | Prepare s | Compile s | Probe/init s | Warmup s | Calibration s | Sampling s | Validation s |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sparse_synthetic_4m_64m_spmv_f64 | pytorch-cuda | ok | 1.769 | 63.425 | 0.000 | 0.041 | 0.020 | 0.020 | 0.059 | 0.351 |
| sparse_synthetic_64k_4m_spmm_f64_rhs1024 | pytorch-cuda | ok | 0.000 | 12.779 | 0.000 | 0.122 | 0.108 | 0.108 | 0.325 | 5.404 |
| sparse_synthetic_4m_64m_spmv_f64 | cusparse | ok | 0.268 | 63.983 | 0.000 | 0.042 | 0.020 | 0.020 | 0.059 | 0.351 |
| sparse_synthetic_64k_4m_spmm_f64_rhs1024 | cusparse | ok | 0.000 | 12.826 | 0.000 | 0.123 | 0.108 | 0.108 | 0.325 | 5.390 |
| sparse_synthetic_4m_64m_spmv_f64 | ginkgo | ok | 1.897 | 33.419 | — | 0.038 | 0.021 | 0.021 | 0.063 | 0.250 |
| sparse_synthetic_64k_4m_spmm_f64_rhs1024 | ginkgo | ok | 0.000 | 11.421 | — | 1.176 | 1.175 | 1.175 | 3.526 | 4.934 |
## gpu/tensornetwork / 20261014_005047

Raw run: `data/results/nvidia-gpu/gpu/tensornetwork/20261014_005047`.

tenferro-rs `9655b4e0daced17e0aa35a30d35710b069e10644`; harness `25f190048045c01cb363384ba45b2a0b0b640db8`. GPU: NVIDIA GeForce RTX 3060 (`GPU-a78d5217-eba3-72c2-3d5b-8ae496ebbc2e`).

Command through this suite: **865.537 s**; collection including build/release delays: 310.621 s; report generation/copy: 0.140 s.

Scope: script clock-read through this suite report copy; devcontainer CLI startup and final wall-record write excluded.

| Backend | Build s | Process s (startup/setup/checks/samples) |
|---|---:|---:|
| tenferro-cuda-trace | 0.000 | 0.371 |
| tenferro-cuda-eager | 0.000 | 6.736 |
| pytorch-cuda | 0.000 | 5.210 |
| jax-cuda | 0.000 | 7.723 |
| jax-cuda-tree-jit | 0.000 | 51.554 |
| jax-cuda-eager | 0.000 | 226.511 |

Case phases below exclude process startup/JSONL writing. Preparation includes input generation, transfers and reference preparation; lazy initialization can occur in that phase. Sampler walls include untimed batch preparation/cleanup. Command totals are cumulative across suites in one invocation, so do not sum those totals.

| Problem | Backend | Status | Backend init s | Prepare s | Compile s | Probe/init s | Warmup s | Calibration s | Sampling s | Validation s |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tensornetwork_permutation_optimized_f32 | tenferro-cuda-trace | unsupported | — | — | — | — | — | — | — | — |
| tensornetwork_permutation_optimized_f32 | tenferro-cuda-eager | ok | 0.000 | 2.121 | — | 3.500 | 0.136 | 0.122 | 0.353 | 0.002 |
| tensornetwork_permutation_optimized_f32 | pytorch-cuda | ok | 1.706 | 0.045 | 0.000 | 0.436 | 0.337 | 0.336 | 1.010 | 0.341 |
| tensornetwork_permutation_optimized_f32 | jax-cuda | ok | 0.984 | 0.092 | 3.344 | 0.457 | 0.286 | 0.286 | 0.858 | 0.385 |
| tensornetwork_permutation_optimized_f32 | jax-cuda-tree-jit | ok | 0.952 | 0.090 | 46.512 | 0.380 | 0.360 | 0.362 | 1.088 | 0.459 |
| tensornetwork_permutation_optimized_f32 | jax-cuda-eager | ok | 0.945 | 0.106 | 0.000 | 217.946 | 0.820 | 0.831 | 2.517 | 0.946 |
