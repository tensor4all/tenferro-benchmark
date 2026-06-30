# CPU Linalg JVP/VJP Benchmark Results

- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `mac-cpu`
- Timestamp: `20260630_183200`

Latest run: `./scripts/run_all.sh 4`.

Derived from the CPU ops CSV under `data/results/mac-cpu/cpu/einsum/20260630_183200`.

- tenferro-rs commit: `38b1c5f2a0f0229336dda751d1033cae3cfc106a`

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
- Python platform: `macOS-26.5.1-arm64-arm-64bit-Mach-O`

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

## Python Backend Providers

- PyTorch: BLAS provider `accelerate`, version `2.12.0`, BLAS_INFO `accelerate`, LAPACK_INFO `accelerate`
  - linked BLAS/LAPACK libs: `/System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate`; `@rpath/libomp.dylib`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `internal_lapack`
  - linked LAPACK libs: `bazel-out/darwin_arm64-opt/bin/jaxlib/cpu/_lapack.so`

## Threads: 4

- CSV: `data/results/mac-cpu/cpu/einsum/20260630_183200/cpu_ops_t4_20260630_183200.csv`
- Source table: `data/results/mac-cpu/cpu/einsum/20260630_183200/linalg_jvp_vjp_t4_20260630_183200.md`

## Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext`; PyTorch uses `torch.func.jvp` /
`vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | 3.209 ± 0.038 | 3.267 ± 0.084 | 3.793 ± 0.040 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | 14.829 ± 0.230 | 15.406 ± 0.213 | 17.876 ± 0.437 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | 3.184 ± 0.073 | 2.927 ± 0.054 | 3.570 ± 0.072 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | 14.939 ± 0.831 | 13.883 ± 0.237 | 16.742 ± 0.178 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | 1.283 ± 0.025 | 1.336 ± 0.068 | 1.106 ± 0.095 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | 6.589 ± 0.222 | 8.505 ± 0.257 | 5.047 ± 0.190 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | 2.125 ± 0.249 | 0.857 ± 0.022 | 1.335 ± 0.098 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | 11.199 ± 0.535 | 4.736 ± 0.349 | 5.549 ± 0.078 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | 1.875 ± 0.068 | 1.880 ± 0.023 | 2.364 ± 0.258 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | 9.106 ± 0.238 | 8.971 ± 0.084 | 9.523 ± 0.071 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | 2.663 ± 0.143 | 1.820 ± 0.127 | 2.280 ± 0.044 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | 12.559 ± 0.555 | 8.776 ± 0.271 | 10.270 ± 0.156 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | 0.262 ± 0.019 | 0.413 ± 0.036 | 0.455 ± 0.085 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | 1.402 ± 0.035 | 1.335 ± 0.038 | 1.233 ± 0.125 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | 0.310 ± 0.010 | 0.454 ± 0.004 | 0.585 ± 0.103 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | 1.417 ± 0.099 | 2.066 ± 0.048 | 1.296 ± 0.054 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | 64.036 ± 0.175 | 5.752 ± 0.139 | 6.182 ± 0.160 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | 617.695 ± 1.424 | 26.915 ± 0.368 | 28.095 ± 1.708 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | 63.950 ± 0.763 | 5.163 ± 0.137 | 5.836 ± 0.037 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | 624.223 ± 8.006 | 24.878 ± 7.273 | 34.196 ± 0.271 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | 0.028 ± 0.010 | 0.047 ± 0.004 | 0.215 ± 0.154 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | 0.029 ± 0.013 | 0.049 ± 0.005 | 0.162 ± 0.012 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | 0.080 ± 0.075 | 0.067 ± 0.009 | 0.173 ± 0.008 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | 0.036 ± 0.003 | 0.040 ± 0.001 | 0.380 ± 0.026 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | 0.032 ± 0.014 | 0.042 ± 0.002 | 0.386 ± 0.019 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | 0.029 ± 0.041 | 0.050 ± 0.007 | 0.386 ± 0.018 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | 0.045 ± 0.014 | 0.121 ± 0.015 | 0.240 ± 0.116 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | 0.043 ± 0.001 | 0.110 ± 0.007 | 0.238 ± 0.010 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | 0.049 ± 0.029 | 0.148 ± 0.018 | 0.234 ± 0.004 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | 0.060 ± 0.008 | 0.090 ± 0.032 | 0.667 ± 0.109 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | 0.056 ± 0.004 | 0.090 ± 0.013 | 0.633 ± 0.027 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | 0.065 ± 0.007 | 0.117 ± 0.008 | 0.633 ± 0.021 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | 0.072 ± 0.010 | 0.085 ± 0.025 | 0.238 ± 0.007 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | 0.069 ± 0.021 | 0.072 ± 0.007 | 0.240 ± 0.011 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | 0.067 ± 0.008 | 0.086 ± 0.009 | 0.234 ± 0.005 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | 0.128 ± 0.033 | 0.084 ± 0.011 | 0.564 ± 0.025 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | 0.120 ± 0.006 | 0.077 ± 0.008 | 0.683 ± 0.169 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | 0.118 ± 0.019 | 0.092 ± 0.023 | 0.579 ± 0.025 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | 0.021 ± 0.005 | 0.117 ± 0.010 | 0.188 ± 0.017 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | 0.037 ± 0.027 | 0.120 ± 0.003 | 0.190 ± 0.012 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | 0.021 ± 0.004 | 0.118 ± 0.003 | 0.197 ± 0.008 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | 0.060 ± 0.029 | 0.057 ± 0.003 | 0.408 ± 0.022 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | 0.073 ± 0.039 | 0.060 ± 0.004 | 0.438 ± 0.080 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | 0.040 ± 0.007 | 0.058 ± 0.002 | 0.456 ± 0.159 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | 0.013 ± 0.006 | 0.065 ± 0.010 | 0.168 ± 0.012 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | 0.017 ± 0.002 | 0.065 ± 0.001 | 0.175 ± 0.017 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | 0.025 ± 0.001 | 0.075 ± 0.002 | 0.172 ± 0.007 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | 0.045 ± 0.044 | 0.048 ± 0.006 | 0.415 ± 0.048 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | 0.028 ± 0.004 | 0.043 ± 0.001 | 0.442 ± 0.178 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | 0.035 ± 0.003 | 0.050 ± 0.002 | 0.418 ± 0.030 |

## Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A
