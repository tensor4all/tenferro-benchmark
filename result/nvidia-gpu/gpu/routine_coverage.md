# GPU routine coverage

Source accounting: `/home/shinaoka/tensor4all/tenferro-benchmark/.worktrees/gpu-refresh-20261017/data/results/nvidia-gpu/gpu/routine_selection/20261016_234450/case_status.json`.

Manifest version: 1; coverage: quick; effort: standard.

Measured coverage complete: yes.

Verification-only, unsupported, missing and unselected rows are not measured coverage. Noise flags are descriptive screens, not regression conclusions.

| Classification | Count |
|---|---:|
| executed | 27 |
| unsupported | 0 |
| failed | 0 |
| missing | 0 |
| noisy | 0 |
| verification_only | 0 |
| raw_samples_unavailable | 0 |
| expected | 27 |
| selected | 27 |
| unselected | 68 |
| duplicates | 0 |
| unexpected | 0 |

## expected

- `gpu/dense:dense_matmul_f64_3072:tenferro-cuda-eager`
- `gpu/dense:dense_matmul_f64_3072:pytorch-cuda`
- `gpu/dense:dense_matmul_f64_3072:jax-cuda`
- `gpu/dense:dense_batched_matmul_f64_b16_32:tenferro-cuda-eager`
- `gpu/dense:dense_batched_matmul_f64_b16_32:pytorch-cuda`
- `gpu/dense:dense_batched_matmul_f64_b16_32:jax-cuda`
- `gpu/dense:dense_qr_f64_1536:tenferro-cuda-eager`
- `gpu/dense:dense_qr_f64_1536:pytorch-cuda`
- `gpu/dense:dense_qr_f64_1536:jax-cuda`
- `gpu/dense:dense_solve_f64_512_rhs16:tenferro-cuda-eager`
- `gpu/dense:dense_solve_f64_512_rhs16:pytorch-cuda`
- `gpu/dense:dense_solve_f64_512_rhs16:jax-cuda`
- `gpu/dense:dense_solve_f64_2048_rhs16:tenferro-cuda-eager`
- `gpu/dense:dense_solve_f64_2048_rhs16:pytorch-cuda`
- `gpu/dense:dense_solve_f64_2048_rhs16:jax-cuda`
- `gpu/dense:dense_svd_f64_256:tenferro-cuda-eager`
- `gpu/dense:dense_svd_f64_256:pytorch-cuda`
- `gpu/dense:dense_svd_f64_256:jax-cuda`
- `gpu/dense:dense_eigh_f64_1024:tenferro-cuda-eager`
- `gpu/dense:dense_eigh_f64_1024:pytorch-cuda`
- `gpu/dense:dense_eigh_f64_1024:jax-cuda`
- `gpu/einsum:einsum_bin_matmul_3072_f64:tenferro-cuda-eager`
- `gpu/einsum:einsum_bin_matmul_3072_f64:pytorch-cuda`
- `gpu/einsum:einsum_bin_matmul_3072_f64:jax-cuda`
- `gpu/tensornetwork:tensornetwork_permutation_optimized_f32:tenferro-cuda-eager`
- `gpu/tensornetwork:tensornetwork_permutation_optimized_f32:pytorch-cuda`
- `gpu/tensornetwork:tensornetwork_permutation_optimized_f32:jax-cuda`

## selected

- `gpu/dense:dense_matmul_f64_3072:tenferro-cuda-eager`
- `gpu/dense:dense_matmul_f64_3072:pytorch-cuda`
- `gpu/dense:dense_matmul_f64_3072:jax-cuda`
- `gpu/dense:dense_batched_matmul_f64_b16_32:tenferro-cuda-eager`
- `gpu/dense:dense_batched_matmul_f64_b16_32:pytorch-cuda`
- `gpu/dense:dense_batched_matmul_f64_b16_32:jax-cuda`
- `gpu/dense:dense_qr_f64_1536:tenferro-cuda-eager`
- `gpu/dense:dense_qr_f64_1536:pytorch-cuda`
- `gpu/dense:dense_qr_f64_1536:jax-cuda`
- `gpu/dense:dense_solve_f64_512_rhs16:tenferro-cuda-eager`
- `gpu/dense:dense_solve_f64_512_rhs16:pytorch-cuda`
- `gpu/dense:dense_solve_f64_512_rhs16:jax-cuda`
- `gpu/dense:dense_solve_f64_2048_rhs16:tenferro-cuda-eager`
- `gpu/dense:dense_solve_f64_2048_rhs16:pytorch-cuda`
- `gpu/dense:dense_solve_f64_2048_rhs16:jax-cuda`
- `gpu/dense:dense_svd_f64_256:tenferro-cuda-eager`
- `gpu/dense:dense_svd_f64_256:pytorch-cuda`
- `gpu/dense:dense_svd_f64_256:jax-cuda`
- `gpu/dense:dense_eigh_f64_1024:tenferro-cuda-eager`
- `gpu/dense:dense_eigh_f64_1024:pytorch-cuda`
- `gpu/dense:dense_eigh_f64_1024:jax-cuda`
- `gpu/einsum:einsum_bin_matmul_3072_f64:tenferro-cuda-eager`
- `gpu/einsum:einsum_bin_matmul_3072_f64:pytorch-cuda`
- `gpu/einsum:einsum_bin_matmul_3072_f64:jax-cuda`
- `gpu/tensornetwork:tensornetwork_permutation_optimized_f32:tenferro-cuda-eager`
- `gpu/tensornetwork:tensornetwork_permutation_optimized_f32:pytorch-cuda`
- `gpu/tensornetwork:tensornetwork_permutation_optimized_f32:jax-cuda`

## executed

- `gpu/dense:dense_matmul_f64_3072:tenferro-cuda-eager`
- `gpu/dense:dense_matmul_f64_3072:pytorch-cuda`
- `gpu/dense:dense_matmul_f64_3072:jax-cuda`
- `gpu/dense:dense_batched_matmul_f64_b16_32:tenferro-cuda-eager`
- `gpu/dense:dense_batched_matmul_f64_b16_32:pytorch-cuda`
- `gpu/dense:dense_batched_matmul_f64_b16_32:jax-cuda`
- `gpu/dense:dense_qr_f64_1536:tenferro-cuda-eager`
- `gpu/dense:dense_qr_f64_1536:pytorch-cuda`
- `gpu/dense:dense_qr_f64_1536:jax-cuda`
- `gpu/dense:dense_solve_f64_512_rhs16:tenferro-cuda-eager`
- `gpu/dense:dense_solve_f64_512_rhs16:pytorch-cuda`
- `gpu/dense:dense_solve_f64_512_rhs16:jax-cuda`
- `gpu/dense:dense_solve_f64_2048_rhs16:tenferro-cuda-eager`
- `gpu/dense:dense_solve_f64_2048_rhs16:pytorch-cuda`
- `gpu/dense:dense_solve_f64_2048_rhs16:jax-cuda`
- `gpu/dense:dense_svd_f64_256:tenferro-cuda-eager`
- `gpu/dense:dense_svd_f64_256:pytorch-cuda`
- `gpu/dense:dense_svd_f64_256:jax-cuda`
- `gpu/dense:dense_eigh_f64_1024:tenferro-cuda-eager`
- `gpu/dense:dense_eigh_f64_1024:pytorch-cuda`
- `gpu/dense:dense_eigh_f64_1024:jax-cuda`
- `gpu/einsum:einsum_bin_matmul_3072_f64:tenferro-cuda-eager`
- `gpu/einsum:einsum_bin_matmul_3072_f64:pytorch-cuda`
- `gpu/einsum:einsum_bin_matmul_3072_f64:jax-cuda`
- `gpu/tensornetwork:tensornetwork_permutation_optimized_f32:tenferro-cuda-eager`
- `gpu/tensornetwork:tensornetwork_permutation_optimized_f32:pytorch-cuda`
- `gpu/tensornetwork:tensornetwork_permutation_optimized_f32:jax-cuda`

## unsupported

None.

## failed

None.

## missing

None.

## verification_only

None.

## raw_samples_unavailable

None.

## noisy

None.

## unselected

- `gpu/dense:dense_matmul_f64_3072:tenferro-cuda-trace`
- `gpu/dense:dense_matmul_f64_3072:cublaslt`
- `gpu/dense:dense_matmul_f64_3072:cutlass`
- `gpu/dense:dense_matmul_f64_3072:cusolver`
- `gpu/dense:dense_batched_matmul_f64_b1024_256:tenferro-cuda-trace`
- `gpu/dense:dense_batched_matmul_f64_b1024_256:tenferro-cuda-eager`
- `gpu/dense:dense_batched_matmul_f64_b1024_256:pytorch-cuda`
- `gpu/dense:dense_batched_matmul_f64_b1024_256:jax-cuda`
- `gpu/dense:dense_batched_matmul_f64_b1024_256:cublaslt`
- `gpu/dense:dense_batched_matmul_f64_b1024_256:cutlass`
- `gpu/dense:dense_batched_matmul_f64_b1024_256:cusolver`
- `gpu/dense:dense_qr_f64_1536:tenferro-cuda-trace`
- `gpu/dense:dense_qr_f64_1536:cublaslt`
- `gpu/dense:dense_qr_f64_1536:cutlass`
- `gpu/dense:dense_qr_f64_1536:cusolver`
- `gpu/dense:dense_solve_f64_512_rhs16:tenferro-cuda-trace`
- `gpu/dense:dense_solve_f64_512_rhs16:cublaslt`
- `gpu/dense:dense_solve_f64_512_rhs16:cutlass`
- `gpu/dense:dense_solve_f64_512_rhs16:cusolver`
- `gpu/dense:dense_solve_f64_1024_rhs16:tenferro-cuda-trace`
- `gpu/dense:dense_solve_f64_1024_rhs16:tenferro-cuda-eager`
- `gpu/dense:dense_solve_f64_1024_rhs16:pytorch-cuda`
- `gpu/dense:dense_solve_f64_1024_rhs16:jax-cuda`
- `gpu/dense:dense_solve_f64_1024_rhs16:cublaslt`
- `gpu/dense:dense_solve_f64_1024_rhs16:cutlass`
- `gpu/dense:dense_solve_f64_1024_rhs16:cusolver`
- `gpu/dense:dense_solve_f64_2048_rhs16:tenferro-cuda-trace`
- `gpu/dense:dense_solve_f64_2048_rhs16:cublaslt`
- `gpu/dense:dense_solve_f64_2048_rhs16:cutlass`
- `gpu/dense:dense_solve_f64_2048_rhs16:cusolver`
- `gpu/dense:dense_solve_f64_4096_rhs16:tenferro-cuda-trace`
- `gpu/dense:dense_solve_f64_4096_rhs16:tenferro-cuda-eager`
- `gpu/dense:dense_solve_f64_4096_rhs16:pytorch-cuda`
- `gpu/dense:dense_solve_f64_4096_rhs16:jax-cuda`
- `gpu/dense:dense_solve_f64_4096_rhs16:cublaslt`
- `gpu/dense:dense_solve_f64_4096_rhs16:cutlass`
- `gpu/dense:dense_solve_f64_4096_rhs16:cusolver`
- `gpu/dense:dense_solve_f64_2048_rhs128:tenferro-cuda-trace`
- `gpu/dense:dense_solve_f64_2048_rhs128:tenferro-cuda-eager`
- `gpu/dense:dense_solve_f64_2048_rhs128:pytorch-cuda`
- `gpu/dense:dense_solve_f64_2048_rhs128:jax-cuda`
- `gpu/dense:dense_solve_f64_2048_rhs128:cublaslt`
- `gpu/dense:dense_solve_f64_2048_rhs128:cutlass`
- `gpu/dense:dense_solve_f64_2048_rhs128:cusolver`
- `gpu/dense:dense_svd_f64_256:tenferro-cuda-trace`
- `gpu/dense:dense_svd_f64_256:cublaslt`
- `gpu/dense:dense_svd_f64_256:cutlass`
- `gpu/dense:dense_svd_f64_256:cusolver`
- `gpu/dense:dense_eigh_f64_1024:tenferro-cuda-trace`
- `gpu/dense:dense_eigh_f64_1024:cublaslt`
- `gpu/dense:dense_eigh_f64_1024:cutlass`
- `gpu/dense:dense_eigh_f64_1024:cusolver`
- `gpu/dense:dense_batched_matmul_f64_b16_32:tenferro-cuda-trace`
- `gpu/dense:dense_batched_matmul_f64_b16_32:cublaslt`
- `gpu/dense:dense_batched_matmul_f64_b16_32:cutlass`
- `gpu/dense:dense_batched_matmul_f64_b16_32:cusolver`
- `gpu/einsum:einsum_bin_matmul_3072_f64:tenferro-cuda-trace`
- `gpu/einsum:einsum_bin_matmul_3072_f64:cublaslt`
- `gpu/einsum:einsum_bin_matmul_3072_f64:cutlass`
- `gpu/sparse:sparse_synthetic_4m_64m_spmv_f64:pytorch-cuda`
- `gpu/sparse:sparse_synthetic_4m_64m_spmv_f64:cusparse`
- `gpu/sparse:sparse_synthetic_4m_64m_spmv_f64:ginkgo`
- `gpu/sparse:sparse_synthetic_64k_4m_spmm_f64_rhs1024:pytorch-cuda`
- `gpu/sparse:sparse_synthetic_64k_4m_spmm_f64_rhs1024:cusparse`
- `gpu/sparse:sparse_synthetic_64k_4m_spmm_f64_rhs1024:ginkgo`
- `gpu/tensornetwork:tensornetwork_permutation_optimized_f32:tenferro-cuda-trace`
- `gpu/tensornetwork:tensornetwork_permutation_optimized_f32:jax-cuda-tree-jit`
- `gpu/tensornetwork:tensornetwork_permutation_optimized_f32:jax-cuda-eager`

## duplicates

None.

## unexpected

None.
