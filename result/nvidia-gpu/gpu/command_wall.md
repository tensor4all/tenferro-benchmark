# GPU command-wall observations

Wall time includes preparation, validation, sampling and reporting. These are end-to-end workflow costs; operation-performance ratios are not inferred. Missing phase instrumentation is displayed as unavailable, not zero. Filtered or unsuccessful runs do not establish full suite coverage.

## gpu/dense / 20261013_224354

Raw run: `data/results/nvidia-gpu/gpu/dense/20261013_224354`.

tenferro-rs `39d71d4037d9eb31fd3301f1e9ae31eed8fcaa9b`; harness `94c3057235650219019e3d9d9424e98f2b7aad37`. GPU: NVIDIA GeForce RTX 3060 (`GPU-a78d5217-eba3-72c2-3d5b-8ae496ebbc2e`).

Command through this suite: **22.852 s**; collection including build/release delays: 14.816 s; report generation/copy: 0.135 s.

Scope: script clock-read through this suite report copy; devcontainer CLI startup and final wall-record write excluded.

| Backend | Build s | Process s (startup/setup/checks/samples) |
|---|---:|---:|
| tenferro-cuda-eager | 0.354 | 3.577 |
| pytorch-cuda | 0.000 | 2.499 |
| jax-cuda | 0.000 | 2.064 |

Case phases below exclude process startup/JSONL writing. Preparation includes input generation, transfers and reference preparation; lazy initialization can occur in that phase. Sampler walls include untimed batch preparation/cleanup. Command totals are cumulative across suites in one invocation, so do not sum those totals.

| Problem | Backend | Status | Backend init s | Prepare s | Compile s | Probe/init s | Warmup s | Calibration s | Sampling s | Validation s |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dense_solve_f64_512_rhs16 | tenferro-cuda-eager | ok | 0.000 | 2.325 | — | 0.673 | 0.009 | 0.004 | 0.020 | 0.024 |
| dense_solve_f64_512_rhs16 | pytorch-cuda | ok | 1.740 | 0.025 | 0.000 | 0.060 | 0.006 | 0.003 | 0.016 | 0.005 |
| dense_solve_f64_512_rhs16 | jax-cuda | ok | 0.998 | 0.025 | 0.242 | 0.043 | 0.007 | 0.004 | 0.018 | 0.093 |
## gpu/dense / 20261013_224635

Raw run: `data/results/nvidia-gpu/gpu/dense/20261013_224635`.

tenferro-rs `39d71d4037d9eb31fd3301f1e9ae31eed8fcaa9b`; harness `94c3057235650219019e3d9d9424e98f2b7aad37`. GPU: NVIDIA GeForce RTX 3060 (`GPU-a78d5217-eba3-72c2-3d5b-8ae496ebbc2e`).

Command through this suite: **283.900 s**; collection including build/release delays: 275.937 s; report generation/copy: 0.140 s.

Scope: script clock-read through this suite report copy; devcontainer CLI startup and final wall-record write excluded.

| Backend | Build s | Process s (startup/setup/checks/samples) |
|---|---:|---:|
| tenferro-cuda-trace | 0.359 | 0.444 |
| tenferro-cuda-eager | 0.000 | 70.910 |
| pytorch-cuda | 0.000 | 34.751 |
| jax-cuda | 0.000 | 45.898 |
| cublaslt | 0.000 | 24.618 |
| cutlass | 0.000 | 72.556 |
| cusolver | 0.000 | 11.785 |

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
| dense_matmul_f64_3072 | tenferro-cuda-eager | ok | 0.000 | 2.314 | — | 0.342 | 0.943 | 0.314 | 2.200 | 2.745 |
| dense_batched_matmul_f64_b1024_256 | tenferro-cuda-eager | ok | 0.000 | 1.933 | — | 0.194 | 0.560 | 0.187 | 1.307 | 4.957 |
| dense_qr_f64_1536 | tenferro-cuda-eager | ok | 0.000 | 0.044 | — | 0.358 | 0.191 | 0.096 | 0.478 | 26.874 |
| dense_solve_f64_512_rhs16 | tenferro-cuda-eager | ok | 0.000 | 0.258 | — | 0.180 | 0.009 | 0.004 | 0.021 | 0.020 |
| dense_solve_f64_1024_rhs16 | tenferro-cuda-eager | ok | 0.000 | 0.025 | — | 0.015 | 0.024 | 0.012 | 0.059 | 0.109 |
| dense_solve_f64_2048_rhs16 | tenferro-cuda-eager | ok | 0.000 | 0.028 | — | 0.050 | 0.097 | 0.048 | 0.242 | 0.555 |
| dense_solve_f64_4096_rhs16 | tenferro-cuda-eager | ok | 0.000 | 0.281 | — | 0.290 | 0.577 | 0.289 | 1.442 | 3.832 |
| dense_solve_f64_2048_rhs128 | tenferro-cuda-eager | ok | 0.000 | 0.032 | — | 0.060 | 0.117 | 0.056 | 0.275 | 4.211 |
| dense_svd_f64_256 | tenferro-cuda-eager | ok | 0.000 | 0.006 | — | 0.187 | 0.169 | 0.082 | 0.410 | 0.070 |
| dense_eigh_f64_1024 | tenferro-cuda-eager | ok | 0.000 | 2.017 | — | 0.085 | 0.151 | 0.075 | 0.374 | 7.366 |
| dense_matmul_f64_3072 | pytorch-cuda | ok | 1.769 | 1.756 | 0.000 | 0.358 | 0.946 | 0.315 | 2.208 | 1.057 |
| dense_batched_matmul_f64_b1024_256 | pytorch-cuda | ok | 0.000 | 7.460 | 0.000 | 0.191 | 0.565 | 0.188 | 1.312 | 5.352 |
| dense_qr_f64_1536 | pytorch-cuda | ok | 0.000 | 0.084 | 0.000 | 0.134 | 0.196 | 0.096 | 0.479 | 0.847 |
| dense_solve_f64_512_rhs16 | pytorch-cuda | ok | 0.000 | 0.014 | 0.000 | 0.019 | 0.006 | 0.003 | 0.015 | 0.005 |
| dense_solve_f64_1024_rhs16 | pytorch-cuda | ok | 0.000 | 0.031 | 0.000 | 0.011 | 0.021 | 0.011 | 0.053 | 0.015 |
| dense_solve_f64_2048_rhs16 | pytorch-cuda | ok | 0.000 | 0.155 | 0.000 | 0.048 | 0.097 | 0.048 | 0.241 | 0.064 |
| dense_solve_f64_4096_rhs16 | pytorch-cuda | ok | 0.000 | 0.856 | 0.000 | 0.289 | 0.578 | 0.289 | 1.446 | 0.367 |
| dense_solve_f64_2048_rhs128 | pytorch-cuda | ok | 0.000 | 0.162 | 0.000 | 0.054 | 0.108 | 0.054 | 0.269 | 0.116 |
| dense_svd_f64_256 | pytorch-cuda | ok | 0.000 | 0.002 | 0.000 | 0.087 | 0.169 | 0.084 | 0.422 | 0.096 |
| dense_eigh_f64_1024 | pytorch-cuda | ok | 0.000 | 0.064 | 0.000 | 0.084 | 0.150 | 0.075 | 0.375 | 0.431 |
| dense_matmul_f64_3072 | jax-cuda | ok | 0.974 | 0.891 | 5.804 | 0.275 | 0.815 | 0.272 | 1.902 | 1.290 |
| dense_batched_matmul_f64_b1024_256 | jax-cuda | ok | 0.000 | 7.667 | 4.314 | 0.166 | 0.484 | 0.161 | 1.130 | 5.712 |
| dense_qr_f64_1536 | jax-cuda | ok | 0.000 | 0.083 | 0.209 | 0.108 | 0.195 | 0.097 | 0.483 | 1.043 |
| dense_solve_f64_512_rhs16 | jax-cuda | ok | 0.000 | 0.017 | 0.199 | 0.008 | 0.007 | 0.003 | 0.017 | 0.007 |
| dense_solve_f64_1024_rhs16 | jax-cuda | ok | 0.000 | 0.064 | 0.229 | 0.012 | 0.022 | 0.011 | 0.056 | 0.026 |
| dense_solve_f64_2048_rhs16 | jax-cuda | ok | 0.000 | 0.264 | 0.225 | 0.051 | 0.102 | 0.049 | 0.243 | 0.134 |
| dense_solve_f64_4096_rhs16 | jax-cuda | ok | 0.000 | 1.078 | 0.203 | 0.290 | 0.578 | 0.289 | 1.445 | 0.528 |
| dense_solve_f64_2048_rhs128 | jax-cuda | ok | 0.000 | 0.239 | 0.219 | 0.055 | 0.109 | 0.054 | 0.272 | 0.212 |
| dense_svd_f64_256 | jax-cuda | ok | 0.000 | 0.004 | 0.244 | 0.087 | 0.165 | 0.082 | 0.411 | 0.098 |
| dense_eigh_f64_1024 | jax-cuda | ok | 0.000 | 0.071 | 0.230 | 0.082 | 0.151 | 0.075 | 0.377 | 0.440 |
| dense_matmul_f64_3072 | cublaslt | ok | — | — | — | — | — | — | — | — |
| dense_batched_matmul_f64_b1024_256 | cublaslt | ok | — | — | — | — | — | — | — | — |
| dense_qr_f64_1536 | cublaslt | unsupported | — | — | — | — | — | — | — | — |
| dense_solve_f64_512_rhs16 | cublaslt | unsupported | — | — | — | — | — | — | — | — |
| dense_solve_f64_1024_rhs16 | cublaslt | unsupported | — | — | — | — | — | — | — | — |
| dense_solve_f64_2048_rhs16 | cublaslt | unsupported | — | — | — | — | — | — | — | — |
| dense_solve_f64_4096_rhs16 | cublaslt | unsupported | — | — | — | — | — | — | — | — |
| dense_solve_f64_2048_rhs128 | cublaslt | unsupported | — | — | — | — | — | — | — | — |
| dense_svd_f64_256 | cublaslt | unsupported | — | — | — | — | — | — | — | — |
| dense_eigh_f64_1024 | cublaslt | unsupported | — | — | — | — | — | — | — | — |
| dense_matmul_f64_3072 | cutlass | ok | — | — | — | — | — | — | — | — |
| dense_batched_matmul_f64_b1024_256 | cutlass | ok | — | — | — | — | — | — | — | — |
| dense_qr_f64_1536 | cutlass | unsupported | — | — | — | — | — | — | — | — |
| dense_solve_f64_512_rhs16 | cutlass | unsupported | — | — | — | — | — | — | — | — |
| dense_solve_f64_1024_rhs16 | cutlass | unsupported | — | — | — | — | — | — | — | — |
| dense_solve_f64_2048_rhs16 | cutlass | unsupported | — | — | — | — | — | — | — | — |
| dense_solve_f64_4096_rhs16 | cutlass | unsupported | — | — | — | — | — | — | — | — |
| dense_solve_f64_2048_rhs128 | cutlass | unsupported | — | — | — | — | — | — | — | — |
| dense_svd_f64_256 | cutlass | unsupported | — | — | — | — | — | — | — | — |
| dense_eigh_f64_1024 | cutlass | unsupported | — | — | — | — | — | — | — | — |
| dense_matmul_f64_3072 | cusolver | unsupported | — | — | — | — | — | — | — | — |
| dense_batched_matmul_f64_b1024_256 | cusolver | unsupported | — | — | — | — | — | — | — | — |
| dense_qr_f64_1536 | cusolver | ok | — | — | — | — | — | — | — | — |
| dense_solve_f64_512_rhs16 | cusolver | ok | — | — | — | — | — | — | — | — |
| dense_solve_f64_1024_rhs16 | cusolver | ok | — | — | — | — | — | — | — | — |
| dense_solve_f64_2048_rhs16 | cusolver | ok | — | — | — | — | — | — | — | — |
| dense_solve_f64_4096_rhs16 | cusolver | ok | — | — | — | — | — | — | — | — |
| dense_solve_f64_2048_rhs128 | cusolver | ok | — | — | — | — | — | — | — | — |
| dense_svd_f64_256 | cusolver | ok | — | — | — | — | — | — | — | — |
| dense_eigh_f64_1024 | cusolver | ok | — | — | — | — | — | — | — | — |
## gpu/einsum / 20261013_224635

Raw run: `data/results/nvidia-gpu/gpu/einsum/20261013_224635`.

tenferro-rs `39d71d4037d9eb31fd3301f1e9ae31eed8fcaa9b`; harness `94c3057235650219019e3d9d9424e98f2b7aad37`. GPU: NVIDIA GeForce RTX 3060 (`GPU-a78d5217-eba3-72c2-3d5b-8ae496ebbc2e`).

Command through this suite: **354.452 s**; collection including build/release delays: 63.685 s; report generation/copy: 0.137 s.

Scope: script clock-read through this suite report copy; devcontainer CLI startup and final wall-record write excluded.

| Backend | Build s | Process s (startup/setup/checks/samples) |
|---|---:|---:|
| tenferro-cuda-trace | 0.000 | 0.375 |
| tenferro-cuda-eager | 0.000 | 9.115 |
| pytorch-cuda | 0.000 | 10.259 |
| jax-cuda | 0.000 | 13.069 |
| cublaslt | 0.000 | 10.220 |
| cutlass | 0.000 | 8.124 |

Case phases below exclude process startup/JSONL writing. Preparation includes input generation, transfers and reference preparation; lazy initialization can occur in that phase. Sampler walls include untimed batch preparation/cleanup. Command totals are cumulative across suites in one invocation, so do not sum those totals.

| Problem | Backend | Status | Backend init s | Prepare s | Compile s | Probe/init s | Warmup s | Calibration s | Sampling s | Validation s |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| einsum_bin_matmul_3072_f64 | tenferro-cuda-trace | unsupported | — | — | — | — | — | — | — | — |
| einsum_bin_matmul_3072_f64 | tenferro-cuda-eager | ok | 0.000 | 2.304 | — | 0.343 | 0.948 | 0.316 | 2.211 | 2.525 |
| einsum_bin_matmul_3072_f64 | pytorch-cuda | ok | 1.730 | 1.759 | 0.000 | 0.359 | 0.948 | 0.316 | 2.211 | 1.973 |
| einsum_bin_matmul_3072_f64 | jax-cuda | ok | 0.995 | 0.861 | 5.812 | 0.271 | 0.806 | 0.269 | 1.882 | 1.317 |
| einsum_bin_matmul_3072_f64 | cublaslt | ok | — | — | — | — | — | — | — | — |
| einsum_bin_matmul_3072_f64 | cutlass | ok | — | — | — | — | — | — | — | — |
## gpu/sparse / 20261013_224635

Raw run: `data/results/nvidia-gpu/gpu/sparse/20261013_224635`.

tenferro-rs `39d71d4037d9eb31fd3301f1e9ae31eed8fcaa9b`; harness `94c3057235650219019e3d9d9424e98f2b7aad37`. GPU: NVIDIA GeForce RTX 3060 (`GPU-a78d5217-eba3-72c2-3d5b-8ae496ebbc2e`).

Command through this suite: **666.151 s**; collection including build/release delays: 304.819 s; report generation/copy: 0.136 s.

Scope: script clock-read through this suite report copy; devcontainer CLI startup and final wall-record write excluded.

| Backend | Build s | Process s (startup/setup/checks/samples) |
|---|---:|---:|
| pytorch-cuda | 0.000 | 86.932 |
| cusparse | 0.000 | 88.688 |
| ginkgo | 0.000 | 122.881 |

Case phases below exclude process startup/JSONL writing. Preparation includes input generation, transfers and reference preparation; lazy initialization can occur in that phase. Sampler walls include untimed batch preparation/cleanup. Command totals are cumulative across suites in one invocation, so do not sum those totals.

| Problem | Backend | Status | Backend init s | Prepare s | Compile s | Probe/init s | Warmup s | Calibration s | Sampling s | Validation s |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sparse_synthetic_4m_64m_spmv_f64 | pytorch-cuda | ok | 1.735 | 63.883 | 0.000 | 0.059 | 0.060 | 0.020 | 0.139 | 0.347 |
| sparse_synthetic_64k_4m_spmm_f64_rhs1024 | pytorch-cuda | ok | 0.000 | 13.245 | 0.000 | 0.125 | 0.325 | 0.108 | 0.758 | 5.321 |
| sparse_synthetic_4m_64m_spmv_f64 | cusparse | ok | — | — | — | — | — | — | — | — |
| sparse_synthetic_64k_4m_spmm_f64_rhs1024 | cusparse | ok | — | — | — | — | — | — | — | — |
| sparse_synthetic_4m_64m_spmv_f64 | ginkgo | ok | — | — | — | — | — | — | — | — |
| sparse_synthetic_64k_4m_spmm_f64_rhs1024 | ginkgo | ok | — | — | — | — | — | — | — | — |
## gpu/tensornetwork / 20261013_224635

Raw run: `data/results/nvidia-gpu/gpu/tensornetwork/20261013_224635`.

tenferro-rs `39d71d4037d9eb31fd3301f1e9ae31eed8fcaa9b`; harness `94c3057235650219019e3d9d9424e98f2b7aad37`. GPU: NVIDIA GeForce RTX 3060 (`GPU-a78d5217-eba3-72c2-3d5b-8ae496ebbc2e`).

Command through this suite: **1001.972 s**; collection including build/release delays: 328.998 s; report generation/copy: 0.132 s.

Scope: script clock-read through this suite report copy; devcontainer CLI startup and final wall-record write excluded.

| Backend | Build s | Process s (startup/setup/checks/samples) |
|---|---:|---:|
| tenferro-cuda-trace | 0.000 | 0.405 |
| tenferro-cuda-eager | 0.000 | 7.820 |
| pytorch-cuda | 0.000 | 8.240 |
| jax-cuda | 0.000 | 10.400 |
| jax-cuda-tree-jit | 0.000 | 54.602 |
| jax-cuda-eager | 0.000 | 235.010 |

Case phases below exclude process startup/JSONL writing. Preparation includes input generation, transfers and reference preparation; lazy initialization can occur in that phase. Sampler walls include untimed batch preparation/cleanup. Command totals are cumulative across suites in one invocation, so do not sum those totals.

| Problem | Backend | Status | Backend init s | Prepare s | Compile s | Probe/init s | Warmup s | Calibration s | Sampling s | Validation s |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tensornetwork_permutation_optimized_f32 | tenferro-cuda-trace | unsupported | — | — | — | — | — | — | — | — |
| tensornetwork_permutation_optimized_f32 | tenferro-cuda-eager | ok | 0.000 | 2.112 | — | 3.555 | 0.375 | 0.118 | 1.143 | 0.002 |
| tensornetwork_permutation_optimized_f32 | pytorch-cuda | ok | 1.716 | 0.046 | 0.000 | 0.442 | 1.008 | 0.336 | 3.360 | 0.338 |
| tensornetwork_permutation_optimized_f32 | jax-cuda | ok | 0.973 | 0.092 | 3.446 | 0.457 | 0.855 | 0.286 | 2.861 | 0.386 |
| tensornetwork_permutation_optimized_f32 | jax-cuda-tree-jit | ok | 0.967 | 0.091 | 46.366 | 0.375 | 1.081 | 0.359 | 3.605 | 0.460 |
| tensornetwork_permutation_optimized_f32 | jax-cuda-eager | ok | 0.992 | 0.113 | 0.000 | 218.143 | 2.673 | 0.843 | 8.881 | 0.974 |
