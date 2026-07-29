# CPU Public API Benchmark Results

- Suite: `cpu/public_api`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/public_api.yaml`
- Public API coverage manifest: `benchmarks/cpu/public_api_coverage.yaml`
- Run metadata: `data/results/mac-cpu/cpu/public_api/20260729_200153/run.yaml`
- Timestamp: `20260729_200153`

Latest run: `PUBLICATION_GATE_PROFILE=full ./scripts/run_cpu_public_api.sh 1 4`.

- Sampling: `full` profile, 15 measured runs, 3 warmups per row

This file is generated from sequential CPU public API runs under `data/results/mac-cpu/cpu/public_api/20260729_200153`.

- tenferro-rs commit: `57c41c218356e871532e4dea4465a6bcf2d6c8fb`

## CPU Information

- Model: `Apple M5 Max`
- Vendor: `Apple`
- Logical CPUs: `18`
- Physical CPUs: `18`
- Sockets: `1`
- Cores per socket: `18`
- Threads per core: `1`
- NUMA nodes: `1`
- Performance levels: `Super: 6 physical / 6 logical (L1i 192 KiB, L1d 128 KiB, L2 16 MiB, 6 CPUs/L2); Performance: 12 physical / 12 logical (L1i 128 KiB, L1d 64 KiB, L2 8 MiB, 6 CPUs/L2)`
- Python platform: `macOS-26.5.1-arm64-arm-64bit`

## Thread Environments

### Threads: 1

- Run metadata: `data/results/mac-cpu/cpu/public_api/20260729_200153/run_t1.yaml`
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

- Run metadata: `data/results/mac-cpu/cpu/public_api/20260729_200153/run_t4.yaml`
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

- CSV: `data/results/mac-cpu/cpu/public_api/20260729_200153/cpu_public_api_t1_20260729_200153.csv`
- CSV: `data/results/mac-cpu/cpu/public_api/20260729_200153/cpu_public_api_t4_20260729_200153.csv`
- Source table: `data/results/mac-cpu/cpu/public_api/20260729_200153/cpu_public_api_20260729_200153.md`

## CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|
| cpu/complex | `cholesky` | c64 | 1 | `448x448` | 2.443 ± 0.062 | 2.417 ± 0.065 | 2.991 ± 0.210 | 0.769 ± 0.041 |
| cpu/complex | `cholesky` | c64 | 4 | `448x448` | 2.533 ± 0.079 | 2.476 ± 0.054 | 2.550 ± 0.126 | 0.829 ± 0.039 |
| cpu/complex | `conj` | c64 | 1 | `16777216` | 4.698 ± 0.130 | 4.530 ± 0.086 | 4.539 ± 0.046 | 1.832 ± 0.124 |
| cpu/complex | `conj` | c64 | 4 | `16777216` | 1.630 ± 0.121 | 1.619 ± 0.062 | 1.721 ± 0.106 | 2.337 ± 0.189 |
| cpu/complex | `div` | c64 | 1 | `8388608` | 4.610 ± 0.121 | 4.915 ± 0.705 | 7.942 ± 0.143 | 6.270 ± 0.126 |
| cpu/complex | `div` | c64 | 4 | `8388608` | 2.335 ± 0.278 | 2.237 ± 0.296 | 2.845 ± 0.052 | 5.732 ± 0.182 |
| cpu/complex | `dot_general` | c64 | 1 | `640x640` | 5.803 ± 0.691 | 5.745 ± 0.207 | 5.735 ± 0.223 | 3.294 ± 0.495 |
| cpu/complex | `dot_general` | c64 | 4 | `640x640` | 6.033 ± 0.317 | 5.959 ± 0.237 | 6.391 ± 0.479 | 3.609 ± 0.295 |
| cpu/complex | `dot_general_with_conj` | c64 | 1 | `640x640` | 6.011 ± 0.179 | unsupported | 5.908 ± 0.255 | 3.252 ± 0.134 |
| cpu/complex | `dot_general_with_conj` | c64 | 4 | `640x640` | 6.509 ± 0.302 | unsupported | 6.498 ± 0.292 | 3.532 ± 0.136 |
| cpu/complex | `eig` | c64 | 1 | `112x112` | 5.808 ± 0.066 | 5.845 ± 0.040 | 5.969 ± 0.064 | 10.116 ± 0.113 |
| cpu/complex | `eig` | c64 | 4 | `112x112` | 5.872 ± 0.057 | 5.821 ± 0.185 | 5.868 ± 0.042 | 10.283 ± 0.441 |
| cpu/complex | `exp` | c64 | 1 | `4194304` | 20.704 ± 0.411 | 20.863 ± 0.172 | 20.191 ± 0.100 | 5.503 ± 0.236 |
| cpu/complex | `exp` | c64 | 4 | `4194304` | 7.058 ± 1.189 | 6.972 ± 0.550 | 6.008 ± 0.279 | 6.294 ± 0.186 |
| cpu/complex | `log` | c64 | 1 | `4194304` | 23.742 ± 0.275 | 23.683 ± 0.197 | 25.688 ± 0.064 | 5.496 ± 1.578 |
| cpu/complex | `log` | c64 | 4 | `4194304` | 8.789 ± 1.647 | 8.677 ± 0.934 | 8.283 ± 0.650 | 6.552 ± 0.390 |
| cpu/complex | `mul` | c64 | 1 | `8388608` | 3.951 ± 0.212 | 3.974 ± 0.041 | 4.249 ± 0.242 | 1.532 ± 0.101 |
| cpu/complex | `mul` | c64 | 4 | `8388608` | 2.376 ± 0.069 | 2.388 ± 0.098 | 1.475 ± 0.025 | 1.994 ± 0.060 |
| cpu/complex | `norm_fro` | c64 | 1 | `2048x1536` | 6.243 ± 0.041 | 5.885 ± 0.029 | 3.765 ± 0.152 | 0.691 ± 0.039 |
| cpu/complex | `norm_fro` | c64 | 4 | `2048x1536` | 2.243 ± 0.108 | 1.748 ± 0.094 | 0.956 ± 0.030 | 0.713 ± 0.060 |
| cpu/complex | `qr` | c64 | 1 | `256x256` | 4.768 ± 0.035 | 4.724 ± 0.034 | 5.072 ± 0.078 | 3.650 ± 0.063 |
| cpu/complex | `qr` | c64 | 4 | `256x256` | 4.626 ± 0.026 | 4.675 ± 0.032 | 4.742 ± 0.218 | 3.559 ± 0.078 |
| cpu/complex | `solve` | c64 | 1 | `384x384,rhs=8` | 3.043 ± 0.042 | 3.310 ± 0.060 | 3.170 ± 0.076 | 1.337 ± 0.195 |
| cpu/complex | `solve` | c64 | 4 | `384x384,rhs=8` | 3.277 ± 0.063 | 3.579 ± 0.082 | 3.347 ± 0.149 | 1.371 ± 0.082 |
| cpu/complex | `svd` | c64 | 1 | `160x160` | 3.880 ± 0.026 | 3.921 ± 0.030 | 4.040 ± 0.145 | 2.799 ± 0.047 |
| cpu/complex | `svd` | c64 | 4 | `160x160` | 3.864 ± 0.031 | 3.898 ± 0.086 | 3.825 ± 0.051 | 2.917 ± 0.042 |
| cpu/complex | `tensordot` | c64 | 1 | `640x640` | 5.724 ± 0.160 | 5.725 ± 0.244 | 5.983 ± 0.125 | 3.064 ± 0.165 |
| cpu/complex | `tensordot` | c64 | 4 | `640x640` | 6.576 ± 0.479 | 6.209 ± 0.335 | 6.125 ± 0.487 | 3.477 ± 0.240 |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 1 | `1024x1024` | 5.804 ± 1.668 | unsupported | 4.937 ± 0.417 | 2.865 ± 0.351 |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 4 | `1024x1024` | 5.200 ± 0.864 | unsupported | 6.061 ± 0.501 | 3.161 ± 0.398 |
| cpu/elementwise_reduction | `abs` | f64 | 1 | `33554432` | 4.362 ± 0.079 | 4.377 ± 0.047 | 4.421 ± 0.045 | 2.129 ± 0.057 |
| cpu/elementwise_reduction | `abs` | f64 | 4 | `33554432` | 1.565 ± 0.025 | 1.587 ± 0.049 | 1.790 ± 0.224 | 2.097 ± 0.108 |
| cpu/elementwise_reduction | `add` | f64 | 1 | `33554432` | 6.464 ± 0.081 | 6.487 ± 0.118 | 6.669 ± 0.594 | 3.663 ± 0.197 |
| cpu/elementwise_reduction | `add` | f64 | 4 | `33554432` | 2.593 ± 0.181 | 2.605 ± 0.110 | 2.587 ± 0.052 | 3.724 ± 0.268 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 1 | `4194304` | 18.159 ± 3.259 | 16.953 ± 0.177 | 17.944 ± 0.055 | 3.967 ± 0.124 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 4 | `4194304` | 5.013 ± 0.121 | 5.007 ± 0.131 | 5.740 ± 0.347 | 4.693 ± 0.194 |
| cpu/elementwise_reduction | `clamp` | f64 | 1 | `8388608` | 2.796 ± 0.030 | 2.824 ± 0.026 | 2.242 ± 0.031 | 1.880 ± 0.084 |
| cpu/elementwise_reduction | `clamp` | f64 | 4 | `8388608` | 1.461 ± 0.239 | 1.627 ± 0.260 | 1.004 ± 0.096 | 1.869 ± 0.116 |
| cpu/elementwise_reduction | `compare_lt` | f64 | 1 | `33554432` | 30.057 ± 0.212 | 30.045 ± 0.148 | 8.593 ± 0.081 | 2.803 ± 0.212 |
| cpu/elementwise_reduction | `compare_lt` | f64 | 4 | `33554432` | 10.765 ± 0.162 | 7.771 ± 0.054 | 2.956 ± 0.089 | 2.429 ± 0.415 |
| cpu/elementwise_reduction | `cos` | f64 | 1 | `8388608` | 16.211 ± 0.268 | 14.735 ± 2.140 | 17.000 ± 0.041 | 2.754 ± 0.102 |
| cpu/elementwise_reduction | `cos` | f64 | 4 | `8388608` | 4.356 ± 0.125 | 4.443 ± 0.286 | 5.265 ± 0.434 | 3.360 ± 0.260 |
| cpu/elementwise_reduction | `div` | f64 | 1 | `33554432` | 6.443 ± 0.085 | 6.519 ± 0.050 | 6.532 ± 0.047 | 3.762 ± 0.337 |
| cpu/elementwise_reduction | `div` | f64 | 4 | `33554432` | 2.552 ± 0.062 | 2.549 ± 0.097 | 2.577 ± 0.116 | 4.018 ± 0.309 |
| cpu/elementwise_reduction | `exp` | f64 | 1 | `8388608` | 17.026 ± 0.316 | 17.053 ± 0.311 | 12.206 ± 0.059 | 2.321 ± 0.395 |
| cpu/elementwise_reduction | `exp` | f64 | 4 | `8388608` | 4.675 ± 0.315 | 4.627 ± 0.175 | 4.472 ± 0.219 | 2.785 ± 0.251 |
| cpu/elementwise_reduction | `expm1` | f64 | 1 | `4194304` | 8.414 ± 0.557 | 8.513 ± 0.544 | 14.694 ± 0.063 | 2.422 ± 0.105 |
| cpu/elementwise_reduction | `expm1` | f64 | 4 | `4194304` | 2.323 ± 0.044 | 2.334 ± 0.085 | 4.361 ± 0.150 | 2.779 ± 0.147 |
| cpu/elementwise_reduction | `log` | f64 | 1 | `8388608` | 15.304 ± 0.284 | 15.367 ± 0.230 | 20.791 ± 0.046 | 3.277 ± 0.316 |
| cpu/elementwise_reduction | `log` | f64 | 4 | `8388608` | 4.283 ± 0.126 | 4.371 ± 0.214 | 7.041 ± 0.198 | 3.629 ± 0.350 |
| cpu/elementwise_reduction | `log1p` | f64 | 1 | `4194304` | 10.183 ± 0.045 | 10.634 ± 0.441 | 11.003 ± 0.038 | 2.399 ± 0.291 |
| cpu/elementwise_reduction | `log1p` | f64 | 4 | `4194304` | 3.000 ± 0.286 | 2.936 ± 0.165 | 3.299 ± 0.206 | 3.015 ± 0.181 |
| cpu/elementwise_reduction | `maximum` | f64 | 1 | `33554432` | 6.560 ± 0.042 | 6.594 ± 0.077 | 6.561 ± 0.048 | 3.703 ± 0.280 |
| cpu/elementwise_reduction | `maximum` | f64 | 4 | `33554432` | 2.628 ± 0.170 | 2.587 ± 0.098 | 2.584 ± 0.080 | 3.805 ± 0.107 |
| cpu/elementwise_reduction | `minimum` | f64 | 1 | `33554432` | 6.645 ± 0.090 | 6.648 ± 0.225 | 6.541 ± 0.039 | 4.099 ± 0.236 |
| cpu/elementwise_reduction | `minimum` | f64 | 4 | `33554432` | 2.540 ± 0.041 | 2.551 ± 0.070 | 2.639 ± 0.071 | 3.817 ± 0.213 |
| cpu/elementwise_reduction | `mul` | f64 | 1 | `33554432` | 6.667 ± 0.195 | 6.661 ± 0.064 | 6.691 ± 0.412 | 3.853 ± 0.213 |
| cpu/elementwise_reduction | `mul` | f64 | 4 | `33554432` | 2.760 ± 0.030 | 2.787 ± 0.067 | 2.595 ± 0.071 | 4.006 ± 0.424 |
| cpu/elementwise_reduction | `neg` | f64 | 1 | `33554432` | 4.379 ± 0.034 | 4.387 ± 0.044 | 4.431 ± 0.064 | 2.139 ± 0.309 |
| cpu/elementwise_reduction | `neg` | f64 | 4 | `33554432` | 1.583 ± 0.044 | 1.584 ± 0.073 | 1.634 ± 0.028 | 2.083 ± 0.098 |
| cpu/elementwise_reduction | `pow` | f64 | 1 | `4194304` | 17.900 ± 0.694 | 18.421 ± 0.851 | 38.280 ± 0.168 | 4.238 ± 0.271 |
| cpu/elementwise_reduction | `pow` | f64 | 4 | `4194304` | 5.248 ± 0.717 | 5.005 ± 0.275 | 10.862 ± 0.410 | 5.154 ± 0.580 |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 1 | `2048x2048` | 5.780 ± 0.065 | 5.706 ± 0.117 | 10.286 ± 0.499 | 0.434 ± 0.030 |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 4 | `2048x2048` | 5.769 ± 0.030 | 5.759 ± 0.045 | 5.644 ± 0.599 | 0.443 ± 0.032 |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 1 | `4096x4096` | 3.102 ± 0.018 | 3.136 ± 0.061 | 4.016 ± 0.153 | 0.767 ± 0.029 |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 4 | `4096x4096` | 3.116 ± 0.025 | 3.140 ± 0.040 | 1.105 ± 0.033 | 0.753 ± 0.091 |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 1 | `8192x4096` | 24.314 ± 1.673 | 23.042 ± 0.661 | 3.253 ± 0.078 | 4.390 ± 0.126 |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 4 | `8192x4096` | 5.983 ± 0.056 | 5.991 ± 0.017 | 1.350 ± 0.081 | 4.216 ± 0.257 |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 1 | `8192x4096` | 17.195 ± 0.368 | 17.002 ± 0.267 | 3.024 ± 0.074 | 3.770 ± 0.059 |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 4 | `8192x4096` | 4.609 ± 0.291 | 4.473 ± 0.083 | 1.663 ± 0.088 | 3.097 ± 0.089 |
| cpu/elementwise_reduction | `rem` | f64 | 1 | `8388608` | 15.979 ± 0.418 | 15.964 ± 0.240 | 19.327 ± 0.045 | 3.896 ± 0.076 |
| cpu/elementwise_reduction | `rem` | f64 | 4 | `8388608` | 4.896 ± 0.799 | 4.488 ± 0.446 | 6.404 ± 0.080 | 4.286 ± 0.455 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 1 | `33554432` | 11.506 ± 0.188 | 11.597 ± 0.482 | 11.698 ± 0.243 | 5.216 ± 0.321 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 4 | `33554432` | 3.084 ± 0.038 | 3.197 ± 0.112 | 3.502 ± 0.188 | 5.053 ± 0.113 |
| cpu/elementwise_reduction | `select` | f64 | 1 | `33554432` | 7.813 ± 0.023 | 7.845 ± 0.087 | 9.127 ± 0.170 | 4.452 ± 0.429 |
| cpu/elementwise_reduction | `select` | f64 | 4 | `33554432` | 3.336 ± 0.079 | 3.458 ± 0.081 | 3.707 ± 0.070 | 4.212 ± 0.140 |
| cpu/elementwise_reduction | `sign` | f64 | 1 | `33554432` | 5.081 ± 0.041 | 5.102 ± 0.044 | 5.082 ± 0.035 | 2.750 ± 0.065 |
| cpu/elementwise_reduction | `sign` | f64 | 4 | `33554432` | 3.258 ± 0.116 | 3.305 ± 0.103 | 1.761 ± 0.075 | 2.773 ± 0.093 |
| cpu/elementwise_reduction | `sin` | f64 | 1 | `8388608` | 29.677 ± 0.239 | 14.692 ± 0.050 | 14.189 ± 0.047 | 2.738 ± 0.225 |
| cpu/elementwise_reduction | `sin` | f64 | 4 | `8388608` | 3.951 ± 0.177 | 3.999 ± 0.128 | 4.609 ± 0.427 | 2.966 ± 0.535 |
| cpu/elementwise_reduction | `sqrt` | f64 | 1 | `33554432` | 7.622 ± 0.244 | 7.757 ± 0.419 | 7.829 ± 0.167 | 2.060 ± 0.149 |
| cpu/elementwise_reduction | `sqrt` | f64 | 4 | `33554432` | 2.047 ± 0.035 | 2.023 ± 0.033 | 2.833 ± 0.130 | 2.055 ± 0.057 |
| cpu/elementwise_reduction | `sub` | f64 | 1 | `33554432` | 6.483 ± 0.069 | 6.477 ± 0.093 | 6.603 ± 0.051 | 3.655 ± 0.202 |
| cpu/elementwise_reduction | `sub` | f64 | 4 | `33554432` | 2.631 ± 0.176 | 2.717 ± 0.147 | 2.570 ± 0.080 | 3.827 ± 0.198 |
| cpu/elementwise_reduction | `tanh` | f64 | 1 | `8388608` | 21.250 ± 0.077 | 21.933 ± 0.975 | 35.604 ± 0.091 | 1.875 ± 0.088 |
| cpu/elementwise_reduction | `tanh` | f64 | 4 | `8388608` | 5.664 ± 0.175 | 5.637 ± 0.180 | 10.700 ± 1.096 | 2.280 ± 0.202 |
| cpu/indexing_layout | `concatenate` | f64 | 1 | `1048576+1048576` | 9.411 ± 0.322 | 9.130 ± 0.508 | 0.221 ± 0.013 | 0.233 ± 0.011 |
| cpu/indexing_layout | `concatenate` | f64 | 4 | `1048576+1048576` | 9.338 ± 0.579 | 9.395 ± 0.361 | 0.130 ± 0.005 | 0.240 ± 0.011 |
| cpu/indexing_layout | `dynamic_slice` | f64 | 1 | `4194304` | 8.621 ± 0.093 | 8.587 ± 0.055 | 0.275 ± 0.003 | 0.121 ± 0.013 |
| cpu/indexing_layout | `dynamic_slice` | f64 | 4 | `4194304` | 1.154 ± 0.021 | 1.208 ± 0.189 | 0.124 ± 0.002 | 0.124 ± 0.027 |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 1 | `2097152` | 5.190 ± 0.516 | unsupported | 0.410 ± 0.017 | 0.275 ± 0.013 |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 4 | `2097152` | 0.863 ± 0.014 | unsupported | 0.222 ± 0.007 | 0.277 ± 0.008 |
| cpu/indexing_layout | `gather` | f64 | 1 | `262144` | 3.106 ± 0.471 | 2.418 ± 0.165 | 0.170 ± 0.011 | 0.080 ± 0.006 |
| cpu/indexing_layout | `gather` | f64 | 4 | `262144` | 0.515 ± 0.088 | 0.542 ± 0.092 | 0.169 ± 0.001 | 0.104 ± 0.026 |
| cpu/indexing_layout | `pad` | f64 | 1 | `2097152` | 7.511 ± 0.286 | 7.475 ± 0.094 | 0.404 ± 0.022 | 0.125 ± 0.008 |
| cpu/indexing_layout | `pad` | f64 | 4 | `2097152` | 7.524 ± 0.055 | 7.560 ± 0.065 | 0.205 ± 0.004 | 0.134 ± 0.011 |
| cpu/indexing_layout | `reverse` | f64 | 1 | `2097152` | 4.439 ± 0.113 | 4.414 ± 0.036 | 0.277 ± 0.052 | 0.122 ± 0.007 |
| cpu/indexing_layout | `reverse` | f64 | 4 | `2097152` | 4.465 ± 0.033 | 4.467 ± 0.053 | 0.128 ± 0.005 | 0.124 ± 0.029 |
| cpu/indexing_layout | `scatter` | f64 | 1 | `262144` | 3.223 ± 0.124 | 3.172 ± 0.284 | 0.248 ± 0.009 | 0.220 ± 0.014 |
| cpu/indexing_layout | `scatter` | f64 | 4 | `262144` | 3.702 ± 0.156 | 3.929 ± 0.512 | 0.234 ± 0.010 | 0.272 ± 0.043 |
| cpu/indexing_layout | `slice` | f64 | 1 | `4194304` | 6.808 ± 0.458 | 7.082 ± 0.232 | 0.447 ± 0.029 | 0.297 ± 0.017 |
| cpu/indexing_layout | `slice` | f64 | 4 | `4194304` | 7.242 ± 0.372 | 7.199 ± 0.501 | 0.249 ± 0.015 | 0.192 ± 0.040 |
| cpu/linalg_uncovered | `cholesky` | f64 | 1 | `1536x1536` | 4.803 ± 1.060 | 4.192 ± 0.144 | 15.323 ± 0.371 | 4.992 ± 0.375 |
| cpu/linalg_uncovered | `cholesky` | f64 | 4 | `1536x1536` | 4.508 ± 0.895 | 4.206 ± 0.266 | 8.069 ± 0.413 | 5.059 ± 0.312 |
| cpu/linalg_uncovered | `det` | f64 | 1 | `1024x1024` | 4.354 ± 0.125 | 4.692 ± 0.279 | 4.089 ± 0.057 | 3.967 ± 0.222 |
| cpu/linalg_uncovered | `det` | f64 | 4 | `1024x1024` | 4.421 ± 0.152 | 4.846 ± 0.403 | 4.520 ± 0.835 | 4.445 ± 0.334 |
| cpu/linalg_uncovered | `eig` | f64 | 1 | `160x160` | 4.428 ± 0.073 | 4.410 ± 0.058 | 4.440 ± 0.043 | 5.394 ± 0.133 |
| cpu/linalg_uncovered | `eig` | f64 | 4 | `160x160` | 4.442 ± 0.065 | 4.434 ± 0.038 | 4.431 ± 0.059 | 5.362 ± 0.043 |
| cpu/linalg_uncovered | `eigvals` | f64 | 1 | `192x192` | 6.915 ± 0.103 | 4.657 ± 0.044 | 4.654 ± 0.060 | 4.169 ± 0.122 |
| cpu/linalg_uncovered | `eigvals` | f64 | 4 | `192x192` | 6.821 ± 0.038 | 4.654 ± 0.021 | 4.607 ± 0.257 | 4.435 ± 0.413 |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 1 | `512x512` | 7.374 ± 0.138 | 0.208 ± 0.009 | 0.267 ± 0.008 | 7.698 ± 0.034 |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 4 | `512x512` | 7.390 ± 0.076 | 0.225 ± 0.011 | 0.273 ± 0.002 | 7.919 ± 0.500 |
| cpu/linalg_uncovered | `inv` | f64 | 1 | `768x768` | 5.799 ± 1.455 | 7.023 ± 0.207 | 5.641 ± 0.058 | 5.176 ± 0.073 |
| cpu/linalg_uncovered | `inv` | f64 | 4 | `768x768` | 5.605 ± 0.285 | 7.092 ± 0.432 | 5.615 ± 0.597 | 5.399 ± 0.261 |
| cpu/linalg_uncovered | `lstsq` | f64 | 1 | `768x384,rhs=16` | 5.810 ± 0.278 | 5.683 ± 0.078 | 3.264 ± 0.026 | 13.341 ± 0.228 |
| cpu/linalg_uncovered | `lstsq` | f64 | 4 | `768x384,rhs=16` | 5.970 ± 0.425 | 5.829 ± 0.123 | 3.263 ± 0.113 | 13.326 ± 0.053 |
| cpu/linalg_uncovered | `lu` | f64 | 1 | `1024x1024` | 4.440 ± 0.212 | 4.837 ± 0.305 | 6.236 ± 0.066 | 4.476 ± 0.066 |
| cpu/linalg_uncovered | `lu` | f64 | 4 | `1024x1024` | 4.504 ± 0.300 | 4.771 ± 0.237 | 6.320 ± 0.568 | 4.624 ± 0.189 |
| cpu/linalg_uncovered | `norm_fro` | f64 | 1 | `2048x2048` | 2.947 ± 0.039 | 2.966 ± 0.226 | 1.275 ± 0.028 | 0.706 ± 0.028 |
| cpu/linalg_uncovered | `norm_fro` | f64 | 4 | `2048x2048` | 0.935 ± 0.132 | 0.945 ± 0.124 | 1.295 ± 0.035 | 0.782 ± 0.104 |
| cpu/linalg_uncovered | `pinv` | f64 | 1 | `512x256` | 6.698 ± 0.074 | 6.675 ± 0.092 | 6.462 ± 0.054 | 6.472 ± 0.118 |
| cpu/linalg_uncovered | `pinv` | f64 | 4 | `512x256` | 6.802 ± 0.117 | 7.020 ± 0.142 | 6.561 ± 0.200 | 6.499 ± 0.095 |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 1 | `512x256` | 6.854 ± 0.521 | 6.733 ± 0.202 | 6.486 ± 0.053 | 6.447 ± 0.116 |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 4 | `512x256` | 6.723 ± 0.065 | 6.764 ± 0.071 | 6.674 ± 0.235 | 6.906 ± 0.428 |
| cpu/linalg_uncovered | `slogdet` | f64 | 1 | `1024x1024` | 6.028 ± 3.448 | 5.241 ± 0.444 | 4.074 ± 0.082 | 3.953 ± 0.195 |
| cpu/linalg_uncovered | `slogdet` | f64 | 4 | `1024x1024` | 4.316 ± 0.104 | 4.970 ± 0.673 | 4.283 ± 0.315 | 4.226 ± 0.309 |
| cpu/linalg_uncovered | `svd_full` | f64 | 1 | `768x384` | unsupported | unsupported | 18.582 ± 0.136 | 18.343 ± 0.172 |
| cpu/linalg_uncovered | `svd_full` | f64 | 4 | `768x384` | unsupported | unsupported | 19.067 ± 0.317 | 18.767 ± 0.147 |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 1 | `4096x4096,rhs=64` | 5.406 ± 0.182 | 5.552 ± 0.933 | 3.949 ± 0.055 | 10.033 ± 1.382 |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 4 | `4096x4096,rhs=64` | 5.584 ± 0.108 | 5.630 ± 0.176 | 4.190 ± 0.385 | 10.755 ± 1.038 |
| cpu/output_reuse | `add_into` | f64 | 1 | `33554432` | 10.788 ± 0.295 | unsupported | 6.627 ± 0.077 | - |
| cpu/output_reuse | `add_into` | f64 | 4 | `33554432` | 4.439 ± 0.324 | unsupported | 2.929 ± 0.240 | - |
| cpu/output_reuse | `conj_into` | c64 | 1 | `16777216` | 8.429 ± 0.201 | unsupported | 4.547 ± 0.030 | - |
| cpu/output_reuse | `conj_into` | c64 | 4 | `16777216` | 3.303 ± 0.073 | unsupported | 1.836 ± 0.147 | - |
| cpu/output_reuse | `copy_read_into` | f64 | 1 | `33554432` | 3.879 ± 0.027 | unsupported | 4.385 ± 0.057 | - |
| cpu/output_reuse | `copy_read_into` | f64 | 4 | `33554432` | 1.679 ± 0.132 | unsupported | 1.691 ± 0.120 | - |
| cpu/output_reuse | `div_into` | f64 | 1 | `33554432` | 10.618 ± 0.113 | unsupported | 6.544 ± 0.050 | - |
| cpu/output_reuse | `div_into` | f64 | 4 | `33554432` | 4.226 ± 0.112 | unsupported | 3.019 ± 0.163 | - |
| cpu/output_reuse | `dot_general_read_into` | f64 | 1 | `1024x1024` | 4.542 ± 0.100 | unsupported | 4.729 ± 0.429 | - |
| cpu/output_reuse | `dot_general_read_into` | f64 | 4 | `1024x1024` | 4.972 ± 0.994 | unsupported | 6.357 ± 0.491 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 1 | `1024x1024` | 5.159 ± 0.764 | unsupported | 5.210 ± 0.123 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 4 | `1024x1024` | 5.326 ± 0.563 | unsupported | 6.316 ± 0.501 | - |
| cpu/output_reuse | `mul_into` | f64 | 1 | `33554432` | 10.638 ± 0.139 | unsupported | 6.543 ± 0.029 | - |
| cpu/output_reuse | `mul_into` | f64 | 4 | `33554432` | 4.431 ± 0.072 | unsupported | 3.020 ± 0.218 | - |
| cpu/output_reuse | `neg_into` | f64 | 1 | `33554432` | 8.414 ± 0.114 | unsupported | 4.430 ± 0.090 | - |
| cpu/output_reuse | `neg_into` | f64 | 4 | `33554432` | 3.300 ± 0.135 | unsupported | 1.893 ± 0.181 | - |
| cpu/output_reuse | `sub_into` | f64 | 1 | `33554432` | 10.747 ± 0.295 | unsupported | 6.621 ± 0.057 | - |
| cpu/output_reuse | `sub_into` | f64 | 4 | `33554432` | 4.239 ± 0.061 | unsupported | 2.825 ± 0.148 | - |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 1 | `8192x1 -> 8192x4096` | 3.551 ± 0.166 | 2.120 ± 0.063 | 2.074 ± 0.021 | 1.121 ± 0.028 |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 4 | `8192x1 -> 8192x4096` | 2.013 ± 0.149 | 1.160 ± 0.033 | 1.080 ± 0.164 | 1.146 ± 0.061 |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 1 | `33554432` | 3.598 ± 0.087 | 3.596 ± 0.054 | 3.682 ± 0.151 | 1.504 ± 0.088 |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 4 | `33554432` | 1.421 ± 0.057 | 1.864 ± 0.083 | 1.363 ± 0.043 | 2.074 ± 0.257 |
| cpu/structural_shape | `embed_diagonal` | f64 | 1 | `8192 -> 8192x8192` | 2.277 ± 0.100 | 2.167 ± 0.076 | 4.048 ± 0.276 | 2.342 ± 0.315 |
| cpu/structural_shape | `embed_diagonal` | f64 | 4 | `8192 -> 8192x8192` | 2.194 ± 0.074 | 2.146 ± 0.039 | 2.046 ± 0.347 | 2.274 ± 0.050 |
| cpu/structural_shape | `extract_diagonal` | f64 | 1 | `8388608x2x2 -> 8388608x2` | 2.188 ± 0.073 | 2.236 ± 0.072 | 4.433 ± 0.049 | 1.495 ± 0.072 |
| cpu/structural_shape | `extract_diagonal` | f64 | 4 | `8388608x2x2 -> 8388608x2` | 2.317 ± 0.059 | 2.304 ± 0.091 | 1.770 ± 0.044 | 2.264 ± 0.200 |
| cpu/structural_shape | `reshape` | f64 | 1 | `33554432 -> 8192x4096` | 3.872 ± 0.043 | 3.952 ± 0.300 | 4.413 ± 0.035 | 1.579 ± 0.099 |
| cpu/structural_shape | `reshape` | f64 | 4 | `33554432 -> 8192x4096` | 3.940 ± 0.026 | 4.014 ± 0.050 | 1.571 ± 0.063 | 1.774 ± 0.290 |
| cpu/structural_shape | `transpose` | f64 | 1 | `4096x4096` | 28.769 ± 0.232 | 24.113 ± 0.475 | 19.588 ± 0.591 | 3.448 ± 0.306 |
| cpu/structural_shape | `transpose` | f64 | 4 | `4096x4096` | 11.058 ± 1.445 | 7.657 ± 0.280 | 19.900 ± 0.520 | 3.882 ± 0.189 |
| cpu/structural_shape | `tril` | f64 | 1 | `4096x4096` | 19.424 ± 0.335 | 19.303 ± 0.198 | 3.521 ± 0.148 | 0.906 ± 0.066 |
| cpu/structural_shape | `tril` | f64 | 4 | `4096x4096` | 19.373 ± 0.316 | 19.428 ± 0.100 | 1.332 ± 0.037 | 1.434 ± 0.142 |
| cpu/structural_shape | `triu` | f64 | 1 | `4096x4096` | 19.124 ± 0.146 | 19.295 ± 0.234 | 3.287 ± 0.083 | 0.862 ± 0.051 |
| cpu/structural_shape | `triu` | f64 | 4 | `4096x4096` | 19.444 ± 0.391 | 19.394 ± 0.314 | 1.205 ± 0.044 | 1.358 ± 0.218 |
| cpu/view_metadata | `broadcast_in_dim_view` | f64 | 1 | `8192x1 -> 8192x4096` | 0.000 ± 0.000 | unsupported | 0.001 ± 0.000 | - |
| cpu/view_metadata | `broadcast_in_dim_view` | f64 | 4 | `8192x1 -> 8192x4096` | 0.000 ± 0.000 | unsupported | 0.001 ± 0.000 | - |
| cpu/view_metadata | `reshape_view` | f64 | 1 | `33554432 -> 8192x4096` | 0.000 ± 0.000 | unsupported | 0.001 ± 0.000 | - |
| cpu/view_metadata | `reshape_view` | f64 | 4 | `33554432 -> 8192x4096` | 0.000 ± 0.000 | unsupported | 0.001 ± 0.000 | - |
| cpu/view_metadata | `slice_view` | f64 | 1 | `4194304 -> 2096128` | 0.000 ± 0.000 | unsupported | 0.000 ± 0.000 | - |
| cpu/view_metadata | `slice_view` | f64 | 4 | `4194304 -> 2096128` | 0.000 ± 0.000 | unsupported | 0.001 ± 0.000 | - |
| cpu/view_metadata | `transpose_view` | f64 | 1 | `4096x4096` | 0.000 ± 0.000 | unsupported | 0.001 ± 0.000 | - |
| cpu/view_metadata | `transpose_view` | f64 | 4 | `4096x4096` | 0.000 ± 0.000 | unsupported | 0.001 ± 0.000 | - |
