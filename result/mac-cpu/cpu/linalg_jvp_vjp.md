# CPU Linalg JVP/VJP Benchmark Results

- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `mac-cpu`
- Timestamp: `20260728_100627`

Latest run: `./scripts/run_all.sh 4`.

Derived from the CPU ops CSV under `data/results/mac-cpu/cpu/einsum/20260728_100627`.

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

## Tenferro CPU BLAS Backend

- tenferro-rs features: `system-accelerate`
- TENFERRO_CPU_BACKEND_KIND: `blas`
- BLAS implementation: `accelerate`
- BLAS version: `unknown`
- BLAS library: `/System/Library/Frameworks/Accelerate.framework`

## Python Backend Providers

- PyTorch: BLAS provider `accelerate`, version `2.12.0`, BLAS_INFO `accelerate`, LAPACK_INFO `accelerate`
  - linked BLAS/LAPACK libs: `/System/Library/Frameworks/Accelerate.framework/Versions/A/Accelerate`; `@rpath/libomp.dylib`
- JAX: dot backend `xla_cpu`, version `0.10.1`, jaxlib `0.10.1`, default backend `cpu`, LAPACK provider `internal_lapack`
  - linked LAPACK libs: `bazel-out/darwin_arm64-opt/bin/jaxlib/cpu/_lapack.so`

## Threads: 4

- CSV: `data/results/mac-cpu/cpu/einsum/20260728_100627/cpu_ops_t4_20260728_100627.csv`
- Source table: `data/results/mac-cpu/cpu/einsum/20260728_100627/linalg_jvp_vjp_t4_20260728_100627.md`

## Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext`; PyTorch uses `torch.func.jvp` /
`vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | 3.214 ± 0.029 | 3.196 ± 0.052 | 3.766 ± 0.052 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | 14.734 ± 0.159 | 15.203 ± 0.149 | 17.752 ± 0.458 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | 3.191 ± 0.037 | 2.972 ± 0.079 | 3.608 ± 0.163 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | 14.832 ± 0.209 | 13.600 ± 0.437 | 16.772 ± 0.242 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | 1.000 ± 0.011 | 1.386 ± 0.138 | 1.162 ± 0.175 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | 5.467 ± 0.308 | 8.397 ± 0.305 | 5.107 ± 0.331 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | 0.917 ± 0.017 | 0.839 ± 0.046 | 1.343 ± 0.153 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | 5.268 ± 0.191 | 4.756 ± 0.252 | 5.686 ± 0.443 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | 2.085 ± 0.098 | 1.873 ± 0.142 | 2.217 ± 0.184 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | 9.275 ± 0.249 | 8.968 ± 0.129 | 9.573 ± 0.393 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | 1.939 ± 0.021 | 1.783 ± 0.037 | 2.280 ± 0.090 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | 9.092 ± 0.260 | 8.596 ± 0.119 | 10.517 ± 0.314 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | 0.354 ± 0.027 | 0.398 ± 0.041 | 0.435 ± 0.053 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | 1.443 ± 0.030 | 1.324 ± 0.077 | 1.295 ± 0.084 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | 0.381 ± 0.002 | 0.451 ± 0.009 | 0.522 ± 0.023 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | 1.438 ± 0.064 | 2.074 ± 0.119 | 1.287 ± 0.073 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | 5.275 ± 0.059 | 5.728 ± 0.066 | 6.177 ± 0.138 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | 24.211 ± 0.063 | 26.467 ± 0.380 | 27.974 ± 0.364 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | 5.216 ± 0.144 | 5.167 ± 0.086 | 5.878 ± 0.120 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | 31.655 ± 7.491 | 31.385 ± 0.177 | 34.409 ± 0.470 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `2x2` | 0.111 ± 0.010 | 0.044 ± 0.003 | 0.158 ± 0.013 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `4x4` | 0.100 ± 0.017 | 0.044 ± 0.002 | 0.150 ± 0.009 |
| small | `grad_sum_eigh_jvp` | f64 | 4 | `8x8` | 0.098 ± 0.017 | 0.052 ± 0.003 | 0.151 ± 0.010 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `2x2` | 0.093 ± 0.007 | 0.042 ± 0.003 | 0.354 ± 0.036 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `4x4` | 0.095 ± 0.016 | 0.039 ± 0.006 | 0.388 ± 0.162 |
| small | `grad_sum_eigh_vjp` | f64 | 4 | `8x8` | 0.100 ± 0.013 | 0.048 ± 0.004 | 0.378 ± 0.026 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `2x2` | 0.086 ± 0.005 | 0.123 ± 0.010 | 0.216 ± 0.009 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `4x4` | 0.094 ± 0.014 | 0.135 ± 0.009 | 0.217 ± 0.005 |
| small | `grad_sum_lu_jvp` | f64 | 4 | `8x8` | 0.083 ± 0.004 | 0.145 ± 0.009 | 0.218 ± 0.012 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `2x2` | 0.098 ± 0.002 | 0.089 ± 0.018 | 0.574 ± 0.035 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `4x4` | 0.089 ± 0.002 | 0.103 ± 0.007 | 0.559 ± 0.028 |
| small | `grad_sum_lu_vjp` | f64 | 4 | `8x8` | 0.101 ± 0.009 | 0.112 ± 0.038 | 0.575 ± 0.014 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `2x2` | 0.175 ± 0.015 | 0.082 ± 0.018 | 0.223 ± 0.010 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `4x4` | 0.171 ± 0.090 | 0.085 ± 0.004 | 0.211 ± 0.009 |
| small | `grad_sum_qr_jvp` | f64 | 4 | `8x8` | 0.156 ± 0.006 | 0.081 ± 0.003 | 0.220 ± 0.019 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `2x2` | 0.156 ± 0.006 | 0.074 ± 0.016 | 0.529 ± 0.032 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `4x4` | 0.168 ± 0.011 | 0.085 ± 0.018 | 0.560 ± 0.042 |
| small | `grad_sum_qr_vjp` | f64 | 4 | `8x8` | 0.159 ± 0.022 | 0.083 ± 0.007 | 0.545 ± 0.044 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `2x2,rhs=1` | 0.122 ± 0.016 | 0.110 ± 0.013 | 0.174 ± 0.008 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `4x4,rhs=1` | 0.103 ± 0.004 | 0.113 ± 0.005 | 0.177 ± 0.014 |
| small | `grad_sum_solve_jvp` | f64 | 4 | `8x8,rhs=1` | 0.105 ± 0.005 | 0.113 ± 0.003 | 0.184 ± 0.007 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `2x2,rhs=1` | 0.114 ± 0.005 | 0.059 ± 0.004 | 0.379 ± 0.047 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `4x4,rhs=1` | 0.114 ± 0.005 | 0.056 ± 0.003 | 0.387 ± 0.011 |
| small | `grad_sum_solve_vjp` | f64 | 4 | `8x8,rhs=1` | 0.117 ± 0.007 | 0.057 ± 0.013 | 0.385 ± 0.025 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `2x2` | 0.055 ± 0.008 | 0.061 ± 0.005 | 0.155 ± 0.010 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `4x4` | 0.052 ± 0.007 | 0.062 ± 0.003 | 0.159 ± 0.011 |
| small | `grad_sum_svd_s_jvp` | f64 | 4 | `8x8` | 0.060 ± 0.008 | 0.070 ± 0.001 | 0.167 ± 0.016 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `2x2` | 0.058 ± 0.014 | 0.051 ± 0.007 | 0.401 ± 0.013 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `4x4` | 0.053 ± 0.005 | 0.043 ± 0.004 | 0.391 ± 0.029 |
| small | `grad_sum_svd_s_vjp` | f64 | 4 | `8x8` | 0.061 ± 0.004 | 0.045 ± 0.002 | 0.413 ± 0.022 |

## Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A
