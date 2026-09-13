# CPU Linalg JVP/VJP Benchmark Results

- Suite: `cpu/linalg_jvp_vjp`
- Target profile: `mac-cpu`
- Timestamp: `20260913_153332`

Latest run: `./scripts/run_all.sh 4`.

Derived from the CPU ops CSV under `data/results/mac-cpu/cpu/einsum/20260913_153332`.

- tenferro-rs commit: `a793c2e95693f053722fbff8d0db25722336c21f`

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

- CSV: `data/results/mac-cpu/cpu/einsum/20260913_153332/cpu_ops_t4_20260913_153332.csv`
- Source table: `data/results/mac-cpu/cpu/einsum/20260913_153332/linalg_jvp_vjp_t4_20260913_153332.md`

## Linalg JVP/VJP Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

tenferro-rs JVP/VJP use trace-mode `AdContext`; PyTorch uses `torch.func.jvp` /
`vjp`; JAX uses `jax.jvp` / `jax.vjp`.

| suite | benchmark | dtype | threads | shape | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) |
|---|---|---:|---:|---|---:|---:|---:|
| large | `grad_sum_eigh_jvp` | f64 | 4 | `1024x1024` | 73.590 ± 0.219 | 83.368 ± 1.314 | 86.838 ± 0.538 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `256x256` | 3.057 ± 0.022 | 3.188 ± 0.101 | 3.212 ± 0.082 |
| large | `grad_sum_eigh_jvp` | f64 | 4 | `512x512` | 13.592 ± 0.085 | 14.652 ± 0.189 | 15.515 ± 0.174 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `1024x1024` | 74.085 ± 0.178 | 72.067 ± 1.547 | 89.216 ± 9.486 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `256x256` | 3.035 ± 0.049 | 2.942 ± 0.118 | 3.153 ± 0.121 |
| large | `grad_sum_eigh_vjp` | f64 | 4 | `512x512` | 13.568 ± 0.065 | 13.965 ± 1.299 | 15.534 ± 0.140 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `1024x1024` | 27.850 ± 1.022 | 60.333 ± 4.310 | 36.831 ± 0.624 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `256x256` | 0.902 ± 0.043 | 1.434 ± 0.067 | 0.826 ± 0.045 |
| large | `grad_sum_lu_jvp` | f64 | 4 | `512x512` | 4.367 ± 0.177 | 9.384 ± 0.794 | 5.570 ± 0.147 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `1024x1024` | 28.442 ± 0.881 | 31.705 ± 2.578 | 37.488 ± 0.535 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `256x256` | 0.850 ± 0.023 | 0.869 ± 0.026 | 0.851 ± 0.082 |
| large | `grad_sum_lu_vjp` | f64 | 4 | `512x512` | 4.287 ± 0.061 | 5.301 ± 0.140 | 5.593 ± 0.263 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `1024x1024` | 47.447 ± 6.380 | 57.834 ± 1.792 | 59.783 ± 1.612 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `256x256` | 2.026 ± 0.056 | 1.930 ± 0.125 | 1.920 ± 0.133 |
| large | `grad_sum_qr_jvp` | f64 | 4 | `512x512` | 8.460 ± 0.058 | 8.973 ± 0.112 | 10.367 ± 0.190 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `1024x1024` | 46.627 ± 0.379 | 53.242 ± 7.321 | 65.626 ± 0.405 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `256x256` | 2.003 ± 0.013 | 1.902 ± 0.119 | 1.964 ± 0.064 |
| large | `grad_sum_qr_vjp` | f64 | 4 | `512x512` | 8.314 ± 0.058 | 8.711 ± 0.158 | 10.722 ± 0.207 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `1024x1024,rhs=1` | 5.794 ± 0.305 | 5.838 ± 0.360 | 5.801 ± 0.316 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `256x256,rhs=1` | 0.415 ± 0.015 | 0.323 ± 0.084 | 0.245 ± 0.009 |
| large | `grad_sum_solve_jvp` | f64 | 4 | `512x512,rhs=1` | 1.286 ± 0.028 | 1.329 ± 0.315 | 1.216 ± 0.061 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `1024x1024,rhs=1` | 5.460 ± 0.165 | 9.748 ± 0.295 | 5.515 ± 0.206 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `256x256,rhs=1` | 0.404 ± 0.007 | 0.423 ± 0.039 | 0.230 ± 0.012 |
| large | `grad_sum_solve_vjp` | f64 | 4 | `512x512,rhs=1` | 1.234 ± 0.015 | 2.020 ± 0.342 | 1.099 ± 0.047 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `1024x1024` | 117.026 ± 0.273 | 145.150 ± 5.372 | 133.247 ± 0.965 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `256x256` | 5.101 ± 0.030 | 5.795 ± 0.104 | 5.532 ± 0.086 |
| large | `grad_sum_svd_s_jvp` | f64 | 4 | `512x512` | 23.141 ± 0.101 | 26.187 ± 0.178 | 25.173 ± 0.193 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `1024x1024` | 116.038 ± 0.745 | 117.445 ± 1.779 | 132.491 ± 0.209 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `256x256` | 5.053 ± 0.014 | 5.296 ± 0.078 | 5.455 ± 0.100 |
| large | `grad_sum_svd_s_vjp` | f64 | 4 | `512x512` | 22.955 ± 0.032 | 23.626 ± 0.097 | 25.059 ± 0.104 |

## Loss Definitions

- `grad_sum_eigh`: loss = sum(eigenvalues); w.r.t. SPD input matrix A
- `grad_sum_lu`: loss = sum(L) + sum(U); w.r.t. input matrix A
- `grad_sum_qr`: loss = sum(Q) + sum(R); w.r.t. input matrix A
- `grad_sum_solve`: loss = sum(solve(A, b)); w.r.t. input matrix A (rhs fixed)
- `grad_sum_svd_s`: loss = sum(singular values); w.r.t. input matrix A
