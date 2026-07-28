# CPU Public API Benchmark Results

- Suite: `cpu/public_api`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/public_api.yaml`
- Public API coverage manifest: `benchmarks/cpu/public_api_coverage.yaml`
- Run metadata: `data/results/mac-cpu/cpu/public_api/20260728_201507/run.yaml`
- Timestamp: `20260728_201507`

Latest run: `PUBLICATION_GATE_PROFILE=full ./scripts/run_cpu_public_api.sh 1 4`.

- Sampling: `full` profile, 15 measured runs, 3 warmups per row

This file is generated from sequential CPU public API runs under `data/results/mac-cpu/cpu/public_api/20260728_201507`.

- tenferro-rs commit: `4ff6b8e1d56fc8d3f6b772f6053c09c745429585`

## CPU Information

- Model: `Apple M4`
- Vendor: `Apple`
- Logical CPUs: `10`
- Physical CPUs: `10`
- Sockets: `1`
- Cores per socket: `10`
- Threads per core: `1`
- NUMA nodes: `1`
- Performance levels: `Performance: 4 physical / 4 logical (L1i 192 KiB, L1d 128 KiB, L2 16 MiB, 4 CPUs/L2); Efficiency: 6 physical / 6 logical (L1i 128 KiB, L1d 64 KiB, L2 4 MiB, 6 CPUs/L2)`
- Python platform: `macOS-26.5.2-arm64-arm-64bit`

## Thread Environments

### Threads: 1

- Run metadata: `data/results/mac-cpu/cpu/public_api/20260728_201507/run_t1.yaml`
- OMP_NUM_THREADS: `1`
- OMP_THREAD_LIMIT: `1`
- OMP_DYNAMIC: `FALSE`
- RAYON_NUM_THREADS: `1`
- OPENBLAS_NUM_THREADS: `1`
- GOTO_NUM_THREADS: `1`
- MKL_NUM_THREADS: `1`
- VECLIB_MAXIMUM_THREADS: `1`
- VECLIB_NUM_THREADS: `1`
- NUMEXPR_NUM_THREADS: `1`
- BLIS_NUM_THREADS: `1`
- XLA_FLAGS: `--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1`

### Threads: 4

- Run metadata: `data/results/mac-cpu/cpu/public_api/20260728_201507/run_t4.yaml`
- OMP_NUM_THREADS: `4`
- OMP_THREAD_LIMIT: `4`
- OMP_DYNAMIC: `FALSE`
- RAYON_NUM_THREADS: `4`
- OPENBLAS_NUM_THREADS: `4`
- GOTO_NUM_THREADS: `4`
- MKL_NUM_THREADS: `4`
- VECLIB_MAXIMUM_THREADS: `4`
- VECLIB_NUM_THREADS: `4`
- NUMEXPR_NUM_THREADS: `4`
- BLIS_NUM_THREADS: `4`
- XLA_FLAGS: `--xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=4`

## Timing Discipline

- Input fixture tensors are created during warmup and outside the measured region for tenferro-rs, PyTorch, and JAX.
- The tenferro-rs direct column measures immediate public operations (normally concrete `Tensor + CpuBackend`; `lstsq`/`svd_full` use `EagerTensor` because no concrete spelling exists). It is not labeled as the `EagerTensor` call layer.
- tenferro-rs trace graphs are constructed and compiled outside the measured region; each compiled graph is reused for every warmup and timed run.
- JAX functions are compiled with `jax.jit` during warmup, outside the measured region; timed calls include dispatch through `jax.block_until_ready`.
- Allocation-returning API rows create their output tensor inside each timed call.
- `cpu/output_reuse` rows allocate the destination during warmup and reuse it; tenferro-rs `*_into` is compared with PyTorch `out=`/`copy_`.
- Trace mode is `unsupported` for caller-output rows because compiled tenferro-rs graphs own their output tensors; JAX is shown as missing because it has no equivalent mutable `out=` API.
- PyTorch view-producing indexing operations are cloned inside the timed region to match tenferro-rs owned, materialized outputs.
- `reshape` compares materialized outputs: PyTorch clones its reshape view inside timing to match tenferro-rs' output-sized write; JAX has value semantics and no public strided-view contract.
- `cpu/view_metadata` separately compares concrete tenferro-rs and PyTorch view creation without an output-sized copy; trace mode is unsupported and JAX is missing because neither exposes the same concrete strided-view contract.
- Rust, PyTorch, and JAX fixtures contain identical logical values. Python/JAX reconstruct the Rust column-major fixture in each framework's native layout before timing.
- PyTorch complex conjugation uses `torch.conj_physical` to match tenferro-rs physical output rather than the lazy conjugate view from `torch.conj`.
- `dot_general_with_conj` instead uses PyTorch's lazy conjugate view so conjugation can be handled by the contraction, matching tenferro-rs' conjugation flags; trace is unsupported because there is no equivalent public traced API.
- `dynamic_update_slice` reports trace mode as `unsupported` because tenferro-rs does not currently expose a corresponding `TracedTensor` API.
- `full_piv_lu` and `full_piv_lu_solve` are excluded because PyTorch has no direct public full-pivot equivalent; substituting `torch.linalg.solve` would compare different algorithms.
- `svd_full` remains in the table even when the selected tenferro-rs provider reports it as unsupported.

## Threads: 1 4

- CSV: `data/results/mac-cpu/cpu/public_api/20260728_201507/cpu_public_api_t1_20260728_201507.csv`
- CSV: `data/results/mac-cpu/cpu/public_api/20260728_201507/cpu_public_api_t4_20260728_201507.csv`
- Source table: `data/results/mac-cpu/cpu/public_api/20260728_201507/cpu_public_api_20260728_201507.md`

## CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|
| cpu/complex | `cholesky` | c64 | 1 | `448x448` | 2.720 ± 0.018 | 2.693 ± 0.051 | 3.106 ± 0.084 | 0.800 ± 0.011 |
| cpu/complex | `cholesky` | c64 | 4 | `448x448` | 2.732 ± 0.044 | 2.665 ± 0.065 | 2.766 ± 0.163 | 0.802 ± 0.041 |
| cpu/complex | `conj` | c64 | 1 | `16777216` | 7.345 ± 0.733 | 6.874 ± 0.970 | 6.690 ± 0.598 | 5.497 ± 0.331 |
| cpu/complex | `conj` | c64 | 4 | `16777216` | 5.748 ± 0.573 | 5.657 ± 0.366 | 5.855 ± 0.472 | 6.149 ± 0.446 |
| cpu/complex | `div` | c64 | 1 | `8388608` | 7.555 ± 0.039 | 7.568 ± 0.031 | 9.527 ± 0.436 | 6.531 ± 0.181 |
| cpu/complex | `div` | c64 | 4 | `8388608` | 4.292 ± 0.271 | 4.289 ± 0.109 | 4.312 ± 0.343 | 7.689 ± 1.357 |
| cpu/complex | `dot_general` | c64 | 1 | `640x640` | 6.453 ± 0.232 | 6.446 ± 0.224 | 6.336 ± 0.170 | 7.237 ± 0.427 |
| cpu/complex | `dot_general` | c64 | 4 | `640x640` | 5.936 ± 0.345 | 5.772 ± 0.073 | 5.735 ± 0.095 | 7.837 ± 0.546 |
| cpu/complex | `dot_general_with_conj` | c64 | 1 | `640x640` | 6.972 ± 0.177 | unsupported | 6.468 ± 0.173 | 7.407 ± 0.559 |
| cpu/complex | `dot_general_with_conj` | c64 | 4 | `640x640` | 6.284 ± 0.185 | unsupported | 6.024 ± 0.117 | 8.495 ± 0.865 |
| cpu/complex | `eig` | c64 | 1 | `112x112` | 6.099 ± 0.030 | 6.156 ± 0.041 | 6.147 ± 0.038 | 10.809 ± 0.035 |
| cpu/complex | `eig` | c64 | 4 | `112x112` | 6.187 ± 0.037 | 6.184 ± 0.036 | 6.198 ± 0.278 | 12.033 ± 0.784 |
| cpu/complex | `exp` | c64 | 1 | `4194304` | 23.445 ± 0.691 | 24.477 ± 0.433 | 24.168 ± 0.374 | 7.748 ± 0.328 |
| cpu/complex | `exp` | c64 | 4 | `4194304` | 7.293 ± 1.075 | 7.207 ± 0.944 | 8.247 ± 1.540 | 8.845 ± 1.482 |
| cpu/complex | `log` | c64 | 1 | `4194304` | 27.378 ± 0.083 | 26.382 ± 0.517 | 28.287 ± 0.061 | 12.680 ± 0.298 |
| cpu/complex | `log` | c64 | 4 | `4194304` | 8.606 ± 1.569 | 8.579 ± 1.161 | 10.226 ± 1.628 | 14.162 ± 0.797 |
| cpu/complex | `mul` | c64 | 1 | `8388608` | 5.201 ± 0.378 | 4.610 ± 0.419 | 4.622 ± 0.257 | 4.377 ± 0.277 |
| cpu/complex | `mul` | c64 | 4 | `8388608` | 4.268 ± 0.096 | 4.308 ± 0.167 | 4.330 ± 0.145 | 4.382 ± 0.181 |
| cpu/complex | `norm_fro` | c64 | 1 | `2048x1536` | 20.852 ± 0.383 | 21.597 ± 0.265 | 4.058 ± 0.097 | 1.446 ± 0.303 |
| cpu/complex | `norm_fro` | c64 | 4 | `2048x1536` | 6.841 ± 1.027 | 6.838 ± 0.908 | 1.737 ± 0.236 | 1.346 ± 0.270 |
| cpu/complex | `qr` | c64 | 1 | `256x256` | 4.540 ± 0.029 | 4.593 ± 0.051 | 4.823 ± 0.037 | 3.770 ± 0.021 |
| cpu/complex | `qr` | c64 | 4 | `256x256` | 4.602 ± 0.022 | 4.610 ± 0.085 | 4.684 ± 0.082 | 3.789 ± 0.034 |
| cpu/complex | `solve` | c64 | 1 | `384x384,rhs=8` | 3.129 ± 0.037 | 3.356 ± 0.053 | 3.168 ± 0.030 | 1.302 ± 0.051 |
| cpu/complex | `solve` | c64 | 4 | `384x384,rhs=8` | 3.288 ± 0.043 | 3.666 ± 0.089 | 3.630 ± 1.354 | 1.320 ± 0.079 |
| cpu/complex | `svd` | c64 | 1 | `160x160` | 3.789 ± 0.031 | 3.874 ± 0.068 | 3.773 ± 0.041 | 2.778 ± 0.030 |
| cpu/complex | `svd` | c64 | 4 | `160x160` | 3.880 ± 0.050 | 3.868 ± 0.120 | 3.770 ± 0.090 | 2.783 ± 0.042 |
| cpu/complex | `tensordot` | c64 | 1 | `640x640` | 6.353 ± 0.167 | 6.453 ± 0.119 | 6.238 ± 0.226 | 7.420 ± 0.436 |
| cpu/complex | `tensordot` | c64 | 4 | `640x640` | 5.791 ± 0.094 | 5.760 ± 0.197 | 5.970 ± 0.884 | 8.365 ± 0.613 |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 1 | `1024x1024` | 5.195 ± 0.400 | unsupported | 5.053 ± 0.162 | 7.866 ± 1.017 |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 4 | `1024x1024` | 4.977 ± 0.308 | unsupported | 4.659 ± 0.244 | 8.404 ± 1.001 |
| cpu/elementwise_reduction | `abs` | f64 | 1 | `33554432` | 6.321 ± 0.178 | 6.308 ± 0.234 | 6.486 ± 0.509 | 5.908 ± 0.886 |
| cpu/elementwise_reduction | `abs` | f64 | 4 | `33554432` | 5.456 ± 0.162 | 5.462 ± 0.138 | 5.576 ± 0.350 | 6.404 ± 0.497 |
| cpu/elementwise_reduction | `add` | f64 | 1 | `33554432` | 8.758 ± 0.671 | 9.377 ± 0.792 | 8.734 ± 0.479 | 11.836 ± 0.320 |
| cpu/elementwise_reduction | `add` | f64 | 4 | `33554432` | 8.692 ± 0.367 | 8.581 ± 0.359 | 8.602 ± 0.272 | 12.662 ± 1.457 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 1 | `4194304` | 18.579 ± 0.325 | 18.370 ± 0.162 | 22.219 ± 0.050 | 5.339 ± 0.106 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 4 | `4194304` | 7.077 ± 0.686 | 7.296 ± 0.680 | 10.100 ± 2.653 | 7.778 ± 1.208 |
| cpu/elementwise_reduction | `clamp` | f64 | 1 | `8388608` | 3.169 ± 0.227 | 3.518 ± 0.820 | 2.975 ± 0.110 | 3.748 ± 0.059 |
| cpu/elementwise_reduction | `clamp` | f64 | 4 | `8388608` | 2.928 ± 0.048 | 2.931 ± 0.047 | 3.295 ± 0.257 | 4.112 ± 0.543 |
| cpu/elementwise_reduction | `compare_lt` | f64 | 1 | `33554432` | 38.706 ± 0.129 | 39.594 ± 0.110 | 8.658 ± 0.309 | 7.829 ± 0.547 |
| cpu/elementwise_reduction | `compare_lt` | f64 | 4 | `33554432` | 10.339 ± 0.126 | 10.264 ± 0.132 | 7.258 ± 0.481 | 7.438 ± 0.553 |
| cpu/elementwise_reduction | `cos` | f64 | 1 | `8388608` | 17.464 ± 0.364 | 17.527 ± 0.126 | 21.172 ± 0.517 | 4.235 ± 0.067 |
| cpu/elementwise_reduction | `cos` | f64 | 4 | `8388608` | 5.953 ± 0.892 | 6.026 ± 0.554 | 10.090 ± 2.225 | 5.837 ± 0.667 |
| cpu/elementwise_reduction | `div` | f64 | 1 | `33554432` | 9.232 ± 0.540 | 9.786 ± 0.526 | 8.905 ± 0.520 | 11.883 ± 0.405 |
| cpu/elementwise_reduction | `div` | f64 | 4 | `33554432` | 8.607 ± 0.445 | 8.558 ± 0.158 | 8.588 ± 0.480 | 11.910 ± 0.597 |
| cpu/elementwise_reduction | `exp` | f64 | 1 | `8388608` | 17.812 ± 0.186 | 17.719 ± 0.273 | 14.912 ± 0.084 | 3.170 ± 0.369 |
| cpu/elementwise_reduction | `exp` | f64 | 4 | `8388608` | 5.601 ± 0.574 | 5.949 ± 0.977 | 7.742 ± 0.790 | 4.622 ± 0.667 |
| cpu/elementwise_reduction | `expm1` | f64 | 1 | `4194304` | 8.964 ± 0.158 | 8.844 ± 0.032 | 17.735 ± 0.081 | 3.459 ± 0.092 |
| cpu/elementwise_reduction | `expm1` | f64 | 4 | `4194304` | 2.972 ± 0.651 | 2.974 ± 0.178 | 8.954 ± 1.085 | 4.505 ± 0.989 |
| cpu/elementwise_reduction | `log` | f64 | 1 | `8388608` | 16.474 ± 0.709 | 17.038 ± 0.194 | 26.628 ± 0.569 | 5.026 ± 0.140 |
| cpu/elementwise_reduction | `log` | f64 | 4 | `8388608` | 5.945 ± 2.092 | 5.832 ± 0.782 | 10.675 ± 3.283 | 7.070 ± 1.538 |
| cpu/elementwise_reduction | `log1p` | f64 | 1 | `4194304` | 11.444 ± 2.176 | 11.455 ± 2.044 | 13.623 ± 0.029 | 3.325 ± 0.132 |
| cpu/elementwise_reduction | `log1p` | f64 | 4 | `4194304` | 3.694 ± 0.850 | 3.698 ± 0.260 | 6.106 ± 3.106 | 4.310 ± 1.275 |
| cpu/elementwise_reduction | `maximum` | f64 | 1 | `33554432` | 9.489 ± 0.254 | 9.141 ± 0.763 | 9.109 ± 0.443 | 11.437 ± 0.209 |
| cpu/elementwise_reduction | `maximum` | f64 | 4 | `33554432` | 8.488 ± 0.177 | 8.549 ± 0.420 | 8.646 ± 0.751 | 11.685 ± 0.677 |
| cpu/elementwise_reduction | `minimum` | f64 | 1 | `33554432` | 9.136 ± 1.148 | 10.046 ± 1.057 | 9.056 ± 0.444 | 11.617 ± 0.365 |
| cpu/elementwise_reduction | `minimum` | f64 | 4 | `33554432` | 8.552 ± 0.067 | 8.497 ± 0.074 | 8.529 ± 0.418 | 11.862 ± 0.677 |
| cpu/elementwise_reduction | `mul` | f64 | 1 | `33554432` | 9.228 ± 0.968 | 9.693 ± 0.766 | 9.094 ± 0.718 | 11.929 ± 0.221 |
| cpu/elementwise_reduction | `mul` | f64 | 4 | `33554432` | 8.632 ± 0.519 | 8.611 ± 0.235 | 8.663 ± 0.621 | 12.702 ± 1.650 |
| cpu/elementwise_reduction | `neg` | f64 | 1 | `33554432` | 6.837 ± 0.941 | 6.393 ± 0.340 | 6.303 ± 0.339 | 6.531 ± 0.767 |
| cpu/elementwise_reduction | `neg` | f64 | 4 | `33554432` | 5.508 ± 0.346 | 5.470 ± 0.075 | 5.473 ± 0.315 | 6.159 ± 0.424 |
| cpu/elementwise_reduction | `pow` | f64 | 1 | `4194304` | 20.170 ± 0.064 | 20.765 ± 0.731 | 45.234 ± 0.287 | 6.034 ± 0.086 |
| cpu/elementwise_reduction | `pow` | f64 | 4 | `4194304` | 6.131 ± 1.251 | 6.542 ± 0.664 | 21.507 ± 14.782 | 7.628 ± 2.010 |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 1 | `2048x2048` | 6.291 ± 0.062 | 6.219 ± 0.078 | 9.134 ± 0.129 | 0.613 ± 0.135 |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 4 | `2048x2048` | 6.135 ± 0.023 | 6.247 ± 0.393 | 6.857 ± 0.622 | 0.611 ± 0.176 |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 1 | `4096x4096` | 3.848 ± 0.298 | 3.766 ± 0.034 | 4.481 ± 0.116 | 1.958 ± 0.074 |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 4 | `4096x4096` | 3.782 ± 0.097 | 3.777 ± 0.039 | 1.800 ± 0.030 | 1.889 ± 0.326 |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 1 | `8192x4096` | 29.904 ± 1.100 | 30.327 ± 3.320 | 4.113 ± 0.375 | 5.208 ± 0.029 |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 4 | `8192x4096` | 8.206 ± 1.385 | 7.500 ± 0.025 | 3.379 ± 0.424 | 6.044 ± 1.739 |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 1 | `8192x4096` | 19.819 ± 1.508 | 18.518 ± 0.415 | 3.910 ± 0.052 | 4.165 ± 0.256 |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 4 | `8192x4096` | 5.432 ± 0.534 | 4.971 ± 0.613 | 3.745 ± 0.273 | 7.006 ± 1.937 |
| cpu/elementwise_reduction | `rem` | f64 | 1 | `8388608` | 19.313 ± 0.089 | 19.575 ± 1.506 | 22.535 ± 0.073 | 6.719 ± 0.235 |
| cpu/elementwise_reduction | `rem` | f64 | 4 | `8388608` | 6.718 ± 2.395 | 6.103 ± 1.312 | 6.685 ± 1.008 | 8.142 ± 1.474 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 1 | `33554432` | 12.861 ± 0.246 | 12.775 ± 0.097 | 12.811 ± 0.053 | 6.778 ± 0.326 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 4 | `33554432` | 5.678 ± 0.658 | 5.601 ± 0.275 | 6.533 ± 0.466 | 7.292 ± 0.420 |
| cpu/elementwise_reduction | `select` | f64 | 1 | `33554432` | 12.187 ± 0.602 | 12.106 ± 1.177 | 9.544 ± 0.422 | 12.139 ± 0.260 |
| cpu/elementwise_reduction | `select` | f64 | 4 | `33554432` | 11.751 ± 0.121 | 11.713 ± 0.138 | 10.249 ± 1.028 | 11.460 ± 0.540 |
| cpu/elementwise_reduction | `sign` | f64 | 1 | `33554432` | 6.872 ± 0.678 | 7.453 ± 1.200 | 6.824 ± 0.677 | 6.086 ± 0.484 |
| cpu/elementwise_reduction | `sign` | f64 | 4 | `33554432` | 5.706 ± 0.120 | 5.729 ± 0.087 | 5.499 ± 0.377 | 6.435 ± 0.624 |
| cpu/elementwise_reduction | `sin` | f64 | 1 | `8388608` | 16.836 ± 0.210 | 16.382 ± 0.060 | 18.173 ± 0.081 | 4.183 ± 0.119 |
| cpu/elementwise_reduction | `sin` | f64 | 4 | `8388608` | 5.702 ± 0.644 | 5.498 ± 0.421 | 10.231 ± 2.754 | 5.842 ± 1.549 |
| cpu/elementwise_reduction | `sqrt` | f64 | 1 | `33554432` | 8.514 ± 0.332 | 8.616 ± 0.610 | 8.484 ± 0.131 | 5.683 ± 0.413 |
| cpu/elementwise_reduction | `sqrt` | f64 | 4 | `33554432` | 5.567 ± 0.255 | 5.571 ± 0.618 | 5.471 ± 0.474 | 6.083 ± 0.677 |
| cpu/elementwise_reduction | `sub` | f64 | 1 | `33554432` | 8.848 ± 0.414 | 9.554 ± 0.869 | 8.802 ± 0.642 | 11.931 ± 0.367 |
| cpu/elementwise_reduction | `sub` | f64 | 4 | `33554432` | 8.683 ± 0.208 | 8.644 ± 0.180 | 8.674 ± 0.451 | 11.848 ± 0.553 |
| cpu/elementwise_reduction | `tanh` | f64 | 1 | `8388608` | 23.017 ± 0.245 | 22.830 ± 0.127 | 43.339 ± 0.765 | 2.835 ± 0.147 |
| cpu/elementwise_reduction | `tanh` | f64 | 4 | `8388608` | 7.646 ± 1.494 | 7.606 ± 0.252 | 15.544 ± 4.084 | 3.253 ± 0.510 |
| cpu/indexing_layout | `concatenate` | f64 | 1 | `1048576+1048576` | 10.433 ± 0.013 | 10.454 ± 0.083 | 0.346 ± 0.026 | 0.367 ± 0.035 |
| cpu/indexing_layout | `concatenate` | f64 | 4 | `1048576+1048576` | 10.537 ± 1.016 | 10.980 ± 0.745 | 0.277 ± 0.013 | 0.343 ± 0.085 |
| cpu/indexing_layout | `dynamic_slice` | f64 | 1 | `4194304` | 10.542 ± 0.084 | 10.247 ± 0.446 | 0.306 ± 0.025 | 0.268 ± 0.016 |
| cpu/indexing_layout | `dynamic_slice` | f64 | 4 | `4194304` | 1.662 ± 0.685 | 1.809 ± 0.386 | 0.274 ± 0.012 | 0.275 ± 0.009 |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 1 | `2097152` | 5.373 ± 0.096 | unsupported | 0.529 ± 0.031 | 0.531 ± 0.038 |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 4 | `2097152` | 1.346 ± 0.383 | unsupported | 0.494 ± 0.065 | 0.497 ± 0.042 |
| cpu/indexing_layout | `gather` | f64 | 1 | `262144` | 3.240 ± 0.998 | 2.774 ± 0.022 | 0.179 ± 0.000 | 0.070 ± 0.009 |
| cpu/indexing_layout | `gather` | f64 | 4 | `262144` | 1.302 ± 0.198 | 0.978 ± 0.391 | 0.183 ± 0.004 | 0.081 ± 0.016 |
| cpu/indexing_layout | `pad` | f64 | 1 | `2097152` | 7.584 ± 0.052 | 7.599 ± 0.022 | 0.544 ± 0.017 | 0.255 ± 0.015 |
| cpu/indexing_layout | `pad` | f64 | 4 | `2097152` | 7.537 ± 0.083 | 7.541 ± 0.032 | 0.498 ± 0.122 | 0.258 ± 0.023 |
| cpu/indexing_layout | `reverse` | f64 | 1 | `2097152` | 5.264 ± 0.053 | 5.326 ± 0.092 | 0.433 ± 0.122 | 0.262 ± 0.013 |
| cpu/indexing_layout | `reverse` | f64 | 4 | `2097152` | 5.279 ± 0.059 | 5.270 ± 0.080 | 0.252 ± 0.011 | 0.298 ± 0.025 |
| cpu/indexing_layout | `scatter` | f64 | 1 | `262144` | 4.104 ± 0.007 | 4.076 ± 0.066 | 0.280 ± 0.009 | 0.233 ± 0.011 |
| cpu/indexing_layout | `scatter` | f64 | 4 | `262144` | 4.201 ± 0.166 | 4.275 ± 0.192 | 0.281 ± 0.008 | 0.258 ± 0.017 |
| cpu/indexing_layout | `slice` | f64 | 1 | `4194304` | 7.386 ± 0.040 | 7.445 ± 0.174 | 0.608 ± 0.096 | 0.512 ± 0.062 |
| cpu/indexing_layout | `slice` | f64 | 4 | `4194304` | 7.451 ± 0.111 | 7.361 ± 0.103 | 0.494 ± 0.042 | 0.501 ± 0.077 |
| cpu/linalg_uncovered | `cholesky` | f64 | 1 | `1536x1536` | 5.838 ± 0.545 | 5.107 ± 0.086 | 14.205 ± 0.145 | 7.090 ± 0.368 |
| cpu/linalg_uncovered | `cholesky` | f64 | 4 | `1536x1536` | 5.373 ± 0.701 | 4.961 ± 0.179 | 7.636 ± 0.616 | 7.028 ± 0.651 |
| cpu/linalg_uncovered | `det` | f64 | 1 | `1024x1024` | 5.386 ± 0.193 | 5.737 ± 0.409 | 4.355 ± 0.082 | 4.183 ± 0.087 |
| cpu/linalg_uncovered | `det` | f64 | 4 | `1024x1024` | 5.172 ± 0.073 | 5.525 ± 0.272 | 4.440 ± 0.327 | 4.393 ± 0.258 |
| cpu/linalg_uncovered | `eig` | f64 | 1 | `160x160` | 4.691 ± 0.034 | 4.710 ± 0.033 | 4.730 ± 0.020 | 5.554 ± 0.225 |
| cpu/linalg_uncovered | `eig` | f64 | 4 | `160x160` | 4.736 ± 0.138 | 4.727 ± 0.046 | 4.711 ± 0.051 | 5.434 ± 0.129 |
| cpu/linalg_uncovered | `eigvals` | f64 | 1 | `192x192` | 7.079 ± 0.019 | 4.902 ± 0.018 | 4.923 ± 0.020 | 4.459 ± 0.034 |
| cpu/linalg_uncovered | `eigvals` | f64 | 4 | `192x192` | 7.110 ± 0.066 | 4.934 ± 0.033 | 5.088 ± 0.146 | 4.586 ± 0.051 |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 1 | `512x512` | 7.824 ± 0.039 | 0.228 ± 0.004 | 0.282 ± 0.005 | 8.050 ± 0.041 |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 4 | `512x512` | 7.904 ± 0.098 | 0.243 ± 0.008 | 0.282 ± 0.011 | 8.241 ± 0.630 |
| cpu/linalg_uncovered | `inv` | f64 | 1 | `768x768` | 5.319 ± 0.318 | 6.667 ± 0.102 | 5.242 ± 0.113 | 4.961 ± 0.032 |
| cpu/linalg_uncovered | `inv` | f64 | 4 | `768x768` | 4.868 ± 0.116 | 6.206 ± 0.094 | 4.673 ± 0.089 | 4.663 ± 0.172 |
| cpu/linalg_uncovered | `lstsq` | f64 | 1 | `768x384,rhs=16` | 5.242 ± 0.058 | 5.425 ± 0.132 | 3.075 ± 0.023 | 12.781 ± 0.093 |
| cpu/linalg_uncovered | `lstsq` | f64 | 4 | `768x384,rhs=16` | 5.344 ± 0.100 | 5.428 ± 0.108 | 2.973 ± 0.028 | 12.933 ± 0.246 |
| cpu/linalg_uncovered | `lu` | f64 | 1 | `1024x1024` | 5.132 ± 0.108 | 5.388 ± 0.054 | 6.809 ± 0.091 | 5.053 ± 0.091 |
| cpu/linalg_uncovered | `lu` | f64 | 4 | `1024x1024` | 4.984 ± 0.146 | 5.238 ± 0.220 | 6.281 ± 0.162 | 5.120 ± 0.250 |
| cpu/linalg_uncovered | `norm_fro` | f64 | 1 | `2048x2048` | 23.358 ± 0.315 | 24.403 ± 0.137 | 1.361 ± 0.015 | 1.187 ± 0.035 |
| cpu/linalg_uncovered | `norm_fro` | f64 | 4 | `2048x2048` | 8.447 ± 1.373 | 8.153 ± 1.214 | 1.379 ± 0.028 | 2.059 ± 1.196 |
| cpu/linalg_uncovered | `pinv` | f64 | 1 | `512x256` | 6.427 ± 0.092 | 6.449 ± 0.076 | 6.294 ± 0.052 | 6.310 ± 0.049 |
| cpu/linalg_uncovered | `pinv` | f64 | 4 | `512x256` | 6.570 ± 0.109 | 6.612 ± 0.142 | 6.351 ± 0.262 | 6.452 ± 0.227 |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 1 | `512x256` | 6.415 ± 0.147 | 6.401 ± 0.106 | 6.309 ± 0.055 | 6.335 ± 0.080 |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 4 | `512x256` | 6.568 ± 0.102 | 6.625 ± 0.127 | 6.372 ± 0.063 | 6.461 ± 0.074 |
| cpu/linalg_uncovered | `slogdet` | f64 | 1 | `1024x1024` | 5.577 ± 0.702 | 6.020 ± 0.523 | 4.346 ± 0.088 | 4.179 ± 0.080 |
| cpu/linalg_uncovered | `slogdet` | f64 | 4 | `1024x1024` | 5.206 ± 0.104 | 5.398 ± 0.267 | 4.393 ± 0.301 | 4.566 ± 0.694 |
| cpu/linalg_uncovered | `svd_full` | f64 | 1 | `768x384` | unsupported | unsupported | 18.077 ± 0.080 | 17.736 ± 0.082 |
| cpu/linalg_uncovered | `svd_full` | f64 | 4 | `768x384` | unsupported | unsupported | 18.885 ± 0.427 | 21.565 ± 4.327 |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 1 | `4096x4096,rhs=64` | 6.016 ± 0.131 | 6.103 ± 0.365 | 4.513 ± 0.109 | 15.351 ± 0.383 |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 4 | `4096x4096,rhs=64` | 6.900 ± 0.380 | 6.843 ± 0.192 | 5.114 ± 0.359 | 17.739 ± 2.069 |
| cpu/output_reuse | `add_into` | f64 | 1 | `33554432` | 17.608 ± 0.492 | unsupported | 8.920 ± 0.402 | - |
| cpu/output_reuse | `add_into` | f64 | 4 | `33554432` | 18.820 ± 4.086 | unsupported | 9.889 ± 1.325 | - |
| cpu/output_reuse | `conj_into` | c64 | 1 | `16777216` | 13.798 ± 0.854 | unsupported | 6.454 ± 0.484 | - |
| cpu/output_reuse | `conj_into` | c64 | 4 | `16777216` | 12.424 ± 1.925 | unsupported | 6.258 ± 1.151 | - |
| cpu/output_reuse | `copy_read_into` | f64 | 1 | `33554432` | 6.643 ± 0.432 | unsupported | 6.258 ± 0.322 | - |
| cpu/output_reuse | `copy_read_into` | f64 | 4 | `33554432` | 7.007 ± 1.094 | unsupported | 5.927 ± 1.119 | - |
| cpu/output_reuse | `div_into` | f64 | 1 | `33554432` | 16.741 ± 0.853 | unsupported | 9.061 ± 0.290 | - |
| cpu/output_reuse | `div_into` | f64 | 4 | `33554432` | 15.660 ± 1.544 | unsupported | 9.890 ± 0.716 | - |
| cpu/output_reuse | `dot_general_read_into` | f64 | 1 | `1024x1024` | 5.148 ± 0.217 | unsupported | 4.981 ± 0.109 | - |
| cpu/output_reuse | `dot_general_read_into` | f64 | 4 | `1024x1024` | 4.889 ± 0.349 | unsupported | 4.640 ± 0.248 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 1 | `1024x1024` | 6.779 ± 2.216 | unsupported | 5.914 ± 0.117 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 4 | `1024x1024` | 6.132 ± 0.730 | unsupported | 5.067 ± 0.377 | - |
| cpu/output_reuse | `mul_into` | f64 | 1 | `33554432` | 17.458 ± 2.657 | unsupported | 9.331 ± 0.868 | - |
| cpu/output_reuse | `mul_into` | f64 | 4 | `33554432` | 17.945 ± 1.231 | unsupported | 8.665 ± 0.493 | - |
| cpu/output_reuse | `neg_into` | f64 | 1 | `33554432` | 13.389 ± 0.631 | unsupported | 6.339 ± 0.398 | - |
| cpu/output_reuse | `neg_into` | f64 | 4 | `33554432` | 13.504 ± 0.931 | unsupported | 6.148 ± 1.412 | - |
| cpu/output_reuse | `sub_into` | f64 | 1 | `33554432` | 15.896 ± 0.335 | unsupported | 9.011 ± 0.309 | - |
| cpu/output_reuse | `sub_into` | f64 | 4 | `33554432` | 15.234 ± 1.125 | unsupported | 9.549 ± 0.672 | - |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 1 | `8192x1 -> 8192x4096` | 5.424 ± 0.302 | 3.594 ± 0.018 | 3.671 ± 0.028 | 3.370 ± 0.320 |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 4 | `8192x1 -> 8192x4096` | 3.980 ± 0.073 | 3.856 ± 0.065 | 3.686 ± 0.243 | 3.656 ± 0.257 |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 1 | `33554432` | 4.992 ± 0.104 | 5.002 ± 0.120 | 5.037 ± 0.420 | 4.307 ± 0.102 |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 4 | `33554432` | 4.211 ± 0.067 | 4.242 ± 0.060 | 4.768 ± 0.914 | 4.406 ± 0.246 |
| cpu/structural_shape | `embed_diagonal` | f64 | 1 | `8192 -> 8192x8192` | 5.189 ± 0.530 | 5.209 ± 0.506 | 7.837 ± 0.749 | 7.828 ± 0.630 |
| cpu/structural_shape | `embed_diagonal` | f64 | 4 | `8192 -> 8192x8192` | 4.719 ± 0.332 | 4.695 ± 0.138 | 8.058 ± 0.945 | 7.274 ± 0.183 |
| cpu/structural_shape | `extract_diagonal` | f64 | 1 | `8388608x2x2 -> 8388608x2` | 3.089 ± 0.093 | 3.132 ± 0.088 | 6.336 ± 0.195 | 4.403 ± 0.269 |
| cpu/structural_shape | `extract_diagonal` | f64 | 4 | `8388608x2x2 -> 8388608x2` | 3.070 ± 0.046 | 3.154 ± 0.480 | 5.519 ± 0.792 | 4.327 ± 0.111 |
| cpu/structural_shape | `reshape` | f64 | 1 | `33554432 -> 8192x4096` | 6.157 ± 0.135 | 6.144 ± 0.124 | 6.336 ± 0.335 | 5.219 ± 0.134 |
| cpu/structural_shape | `reshape` | f64 | 4 | `33554432 -> 8192x4096` | 5.960 ± 0.062 | 6.061 ± 0.275 | 6.450 ± 1.278 | 5.176 ± 0.192 |
| cpu/structural_shape | `transpose` | f64 | 1 | `4096x4096` | 21.603 ± 0.129 | 21.611 ± 0.153 | 19.571 ± 0.382 | 7.285 ± 0.504 |
| cpu/structural_shape | `transpose` | f64 | 4 | `4096x4096` | 11.333 ± 0.446 | 10.024 ± 0.656 | 20.046 ± 0.708 | 7.183 ± 0.321 |
| cpu/structural_shape | `tril` | f64 | 1 | `4096x4096` | 21.868 ± 0.405 | 22.320 ± 1.424 | 4.979 ± 0.176 | 2.784 ± 0.084 |
| cpu/structural_shape | `tril` | f64 | 4 | `4096x4096` | 21.777 ± 0.163 | 21.781 ± 0.086 | 4.803 ± 1.224 | 2.773 ± 0.230 |
| cpu/structural_shape | `triu` | f64 | 1 | `4096x4096` | 21.864 ± 0.892 | 22.960 ± 2.735 | 4.655 ± 0.098 | 2.803 ± 0.169 |
| cpu/structural_shape | `triu` | f64 | 4 | `4096x4096` | 22.029 ± 1.318 | 22.379 ± 1.119 | 4.777 ± 1.523 | 3.011 ± 0.170 |
| cpu/view_metadata | `broadcast_in_dim_view` | f64 | 1 | `8192x1 -> 8192x4096` | 0.000 ± 0.000 | unsupported | 0.001 ± 0.000 | - |
| cpu/view_metadata | `broadcast_in_dim_view` | f64 | 4 | `8192x1 -> 8192x4096` | 0.000 ± 0.000 | unsupported | 0.001 ± 0.000 | - |
| cpu/view_metadata | `reshape_view` | f64 | 1 | `33554432 -> 8192x4096` | 0.000 ± 0.000 | unsupported | 0.001 ± 0.000 | - |
| cpu/view_metadata | `reshape_view` | f64 | 4 | `33554432 -> 8192x4096` | 0.000 ± 0.000 | unsupported | 0.001 ± 0.000 | - |
| cpu/view_metadata | `slice_view` | f64 | 1 | `4194304 -> 2096128` | 0.000 ± 0.000 | unsupported | 0.001 ± 0.000 | - |
| cpu/view_metadata | `slice_view` | f64 | 4 | `4194304 -> 2096128` | 0.000 ± 0.000 | unsupported | 0.001 ± 0.000 | - |
| cpu/view_metadata | `transpose_view` | f64 | 1 | `4096x4096` | 0.000 ± 0.000 | unsupported | 0.001 ± 0.000 | - |
| cpu/view_metadata | `transpose_view` | f64 | 4 | `4096x4096` | 0.000 ± 0.000 | unsupported | 0.001 ± 0.000 | - |
