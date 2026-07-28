# CPU Public API Benchmark Results

- Suite: `cpu/public_api`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/public_api.yaml`
- Run metadata: `data/results/mac-cpu/cpu/public_api/20260728_170003/run.yaml`
- Timestamp: `20260728_170003`

Latest run: `./scripts/run_cpu_public_api.sh 4`.

This file is generated from one CPU public API run under `data/results/mac-cpu/cpu/public_api/20260728_170003`.

- tenferro-rs commit: `80ebcc38ce11fb93385e8b6a1a49b613bc17452f`

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

## Thread Environment

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

- This coverage suite is allocation-inclusive: fixture tensor construction is inside the measured public API case for tenferro-rs and outside for PyTorch where setup can be expressed directly.
- Treat the rows as a first-pass API coverage signal, not as the final steady-state kernel-only comparison used by the focused suites.
- Each timed call creates the output tensor.
- `full_piv_lu` is intentionally excluded from this speed table because PyTorch has no direct public equivalent selected for this suite.
- PyTorch `full_piv_lu_solve` uses `torch.linalg.solve` as the closest solve-level comparison.

## Threads: 4

- CSV: `data/results/mac-cpu/cpu/public_api/20260728_170003/cpu_public_api_t4_20260728_170003.csv`
- Source table: `data/results/mac-cpu/cpu/public_api/20260728_170003/cpu_public_api_t4_20260728_170003.md`

## CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|
| cpu/complex | `cholesky` | c64 | 4 | `32x32` | 0.018 ± 0.005 | - | 0.027 ± 0.004 | - |
| cpu/complex | `conj` | c64 | 4 | `65536` | 0.100 ± 0.019 | - | 0.001 ± 0.000 | - |
| cpu/complex | `div` | c64 | 4 | `65536` | 0.131 ± 0.019 | - | 0.061 ± 0.034 | - |
| cpu/complex | `dot_general_conj` | c64 | 4 | `128x128` | 0.172 ± 0.011 | - | 0.129 ± 0.016 | - |
| cpu/complex | `eig` | c64 | 4 | `32x32` | 0.378 ± 0.021 | - | 0.373 ± 0.002 | - |
| cpu/complex | `exp` | c64 | 4 | `65536` | 0.357 ± 0.025 | - | 0.142 ± 0.002 | - |
| cpu/complex | `log` | c64 | 4 | `65536` | 0.329 ± 0.036 | - | 0.164 ± 0.001 | - |
| cpu/complex | `mul` | c64 | 4 | `65536` | 0.161 ± 0.003 | - | 0.034 ± 0.001 | - |
| cpu/complex | `norm_fro` | c64 | 4 | `64x64` | 0.081 ± 0.013 | - | 0.009 ± 0.000 | - |
| cpu/complex | `qr` | c64 | 4 | `32x32` | 0.063 ± 0.002 | - | 0.069 ± 0.007 | - |
| cpu/complex | `solve` | c64 | 4 | `32x32,rhs=4` | 0.024 ± 0.005 | - | 0.021 ± 0.000 | - |
| cpu/complex | `svd` | c64 | 4 | `32x32` | 0.223 ± 0.017 | - | 0.201 ± 0.001 | - |
| cpu/elementwise_reduction | `abs` | f64 | 4 | `262144` | 0.104 ± 0.008 | - | 0.034 ± 0.001 | - |
| cpu/elementwise_reduction | `add` | f64 | 4 | `262144` | 0.281 ± 0.127 | - | 0.046 ± 0.002 | - |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 4 | `262144` | 0.925 ± 0.041 | - | 0.813 ± 0.011 | - |
| cpu/elementwise_reduction | `clamp` | f64 | 4 | `262144` | 0.209 ± 0.042 | - | 0.080 ± 0.006 | - |
| cpu/elementwise_reduction | `compare_lt` | f64 | 4 | `262144` | 0.319 ± 0.028 | - | 0.055 ± 0.002 | - |
| cpu/elementwise_reduction | `cos` | f64 | 4 | `262144` | 0.367 ± 0.032 | - | 0.381 ± 0.008 | - |
| cpu/elementwise_reduction | `div` | f64 | 4 | `262144` | 0.302 ± 0.039 | - | 0.042 ± 0.001 | - |
| cpu/elementwise_reduction | `exp` | f64 | 4 | `262144` | 0.365 ± 0.044 | - | 0.280 ± 0.022 | - |
| cpu/elementwise_reduction | `expm1` | f64 | 4 | `262144` | 0.386 ± 0.034 | - | 0.606 ± 0.006 | - |
| cpu/elementwise_reduction | `log` | f64 | 4 | `262144` | 0.459 ± 0.027 | - | 0.444 ± 0.033 | - |
| cpu/elementwise_reduction | `log1p` | f64 | 4 | `262144` | 0.571 ± 0.046 | - | 0.489 ± 0.004 | - |
| cpu/elementwise_reduction | `maximum` | f64 | 4 | `262144` | 0.168 ± 0.001 | - | 0.042 ± 0.000 | - |
| cpu/elementwise_reduction | `minimum` | f64 | 4 | `262144` | 0.210 ± 0.061 | - | 0.044 ± 0.002 | - |
| cpu/elementwise_reduction | `mul` | f64 | 4 | `262144` | 0.199 ± 0.013 | - | 0.045 ± 0.002 | - |
| cpu/elementwise_reduction | `neg` | f64 | 4 | `262144` | 0.103 ± 0.003 | - | 0.033 ± 0.001 | - |
| cpu/elementwise_reduction | `pow` | f64 | 4 | `262144` | 0.906 ± 0.056 | - | 1.473 ± 0.164 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 4 | `256x1024` | 0.502 ± 0.012 | - | 0.159 ± 0.005 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 4 | `256x1024` | 0.146 ± 0.004 | - | 0.051 ± 0.001 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 4 | `256x1024` | 0.178 ± 0.029 | - | 0.032 ± 0.001 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 4 | `256x1024` | 0.183 ± 0.052 | - | 0.030 ± 0.001 | - |
| cpu/elementwise_reduction | `rem` | f64 | 4 | `262144` | 0.529 ± 0.038 | - | 0.317 ± 0.095 | - |
| cpu/elementwise_reduction | `rsqrt` | f64 | 4 | `262144` | 0.292 ± 0.071 | - | 0.044 ± 0.000 | - |
| cpu/elementwise_reduction | `select` | f64 | 4 | `262144` | 0.276 ± 0.041 | - | 0.084 ± 0.018 | - |
| cpu/elementwise_reduction | `sign` | f64 | 4 | `262144` | 0.147 ± 0.043 | - | 0.039 ± 0.001 | - |
| cpu/elementwise_reduction | `sin` | f64 | 4 | `262144` | 0.349 ± 0.023 | - | 0.340 ± 0.011 | - |
| cpu/elementwise_reduction | `sqrt` | f64 | 4 | `262144` | 0.259 ± 0.027 | - | 0.037 ± 0.002 | - |
| cpu/elementwise_reduction | `sub` | f64 | 4 | `262144` | 0.202 ± 0.005 | - | 0.049 ± 0.003 | - |
| cpu/elementwise_reduction | `tanh` | f64 | 4 | `262144` | 0.518 ± 0.044 | - | 0.513 ± 0.048 | - |
| cpu/indexing_layout | `concatenate` | f64 | 4 | `32768+32768` | 0.488 ± 0.012 | - | 0.014 ± 0.001 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 4 | `65536` | 0.263 ± 0.022 | - | 0.001 ± 0.000 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 4 | `65536` | 0.277 ± 0.003 | - | 6.183 ± 0.067 | - |
| cpu/indexing_layout | `gather` | f64 | 4 | `65536` | 0.989 ± 0.027 | - | 0.064 ± 0.001 | - |
| cpu/indexing_layout | `pad` | f64 | 4 | `65536` | 0.356 ± 0.005 | - | 0.047 ± 0.022 | - |
| cpu/indexing_layout | `reverse` | f64 | 4 | `65536` | 0.256 ± 0.004 | - | 0.020 ± 0.001 | - |
| cpu/indexing_layout | `scatter` | f64 | 4 | `65536` | 1.457 ± 0.189 | - | 0.111 ± 0.007 | - |
| cpu/indexing_layout | `slice` | f64 | 4 | `65536` | 0.187 ± 0.006 | - | 0.001 ± 0.000 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 4 | `128x128` | 0.036 ± 0.001 | - | 0.062 ± 0.012 | - |
| cpu/linalg_uncovered | `det` | f64 | 4 | `128x128` | 0.162 ± 0.040 | - | 0.054 ± 0.001 | - |
| cpu/linalg_uncovered | `eig` | f64 | 4 | `64x64` | 0.764 ± 0.026 | - | 0.747 ± 0.003 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 4 | `64x64` | 0.745 ± 0.012 | - | 0.429 ± 0.002 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 4 | `128x128` | 0.937 ± 0.017 | - | 0.758 ± 0.006 | - |
| cpu/linalg_uncovered | `full_piv_lu_solve` | f64 | 4 | `64x64,rhs=8` | 0.188 ± 0.015 | - | 0.027 ± 0.001 | - |
| cpu/linalg_uncovered | `inv` | f64 | 4 | `128x128` | 0.121 ± 0.004 | - | 0.102 ± 0.004 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 4 | `256x256` | 0.352 ± 0.030 | - | 0.032 ± 0.000 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 4 | `128x64` | 0.705 ± 0.066 | - | 0.479 ± 0.033 | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 4 | `128x128` | 0.138 ± 0.012 | - | 0.055 ± 0.001 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 4 | `128x128,rhs=16` | 0.032 ± 0.005 | - | 0.009 ± 0.001 | - |
