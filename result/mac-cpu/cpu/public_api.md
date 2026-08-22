# CPU Public API Benchmark Results

- Suite: `cpu/public_api`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/public_api.yaml`
- Public API coverage manifest: `benchmarks/cpu/public_api_coverage.yaml`
- Run metadata: `data/results/mac-cpu/cpu/public_api/20260822_091106/run.yaml`
- Timestamp: `20260822_091106`

Latest run: `PUBLICATION_GATE_PROFILE=full ./scripts/run_cpu_public_api.sh 1 4`.

- Sampling: `full` profile, 15 measured runs, 3 warmups per row

This file is generated from sequential CPU public API runs under `data/results/mac-cpu/cpu/public_api/20260822_091106`.

- tenferro-rs commit: `a21a4c602fc6700b9bc0c3f1b14ebd19b9d7ec45`

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

- Run metadata: `data/results/mac-cpu/cpu/public_api/20260822_091106/run_t1.yaml`
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
- JULIA_NUM_THREADS: `1`

### Threads: 4

- Run metadata: `data/results/mac-cpu/cpu/public_api/20260822_091106/run_t4.yaml`
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
- JULIA_NUM_THREADS: `4`

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
- `cpu/view_metadata` separately compares concrete tenferro-rs, PyTorch, and Julia (`julia-base`) view creation without an output-sized copy; trace mode is unsupported and JAX is missing because neither exposes the same concrete strided-view contract.
- Julia `cpu/view_metadata` rows are `reshape`/`transpose`/`@view` lazy wrappers with no output-sized copy; `broadcast_in_dim_view` has no natural Base spelling and stays missing for Julia.
- Julia `cpu/output_reuse` rows use broadcast-into (`.=`), `mul!`, and `copyto!` with destinations allocated during warmup and reused, the same reuse discipline as the tenferro-rs `_into`/PyTorch `out=` rows above.
- Rust, PyTorch, and JAX fixtures contain identical logical values. Python/JAX reconstruct the Rust column-major fixture in each framework's native layout before timing.
- PyTorch complex conjugation uses `torch.conj_physical` to match tenferro-rs physical output rather than the lazy conjugate view from `torch.conj`.
- `dot_general_with_conj` instead uses PyTorch's lazy conjugate view so conjugation can be handled by the contraction, matching tenferro-rs' conjugation flags; trace is unsupported because there is no equivalent public traced API. Julia's `dot_general_with_conj` row materializes the conjugate before the GEMM instead, since Julia has no lazy conj-without-transpose spelling that BLAS can fuse.
- `pad` has no natural Base spelling and stays missing for Julia.
- `dynamic_update_slice` reports trace mode as `unsupported` because tenferro-rs does not currently expose a corresponding `TracedTensor` API.
- `full_piv_lu` and `full_piv_lu_solve` are excluded because PyTorch has no direct public full-pivot equivalent; substituting `torch.linalg.solve` would compare different algorithms.
- `svd_full` remains in the table even when the selected tenferro-rs provider reports it as unsupported.
- Julia is column-major, like tenferro-rs, so the `julia-base`/`strided-jl` columns need no PyTorch/JAX-style layout reconstruction to keep the same logical fixture values.
- Julia warmup runs move JIT compilation outside the measured region, the same way PyTorch/JAX warmups do.
- Julia factorization rows materialize their factors inside the timed call (`cholesky` returns the factor matrix, `lu` returns P/L/U, `qr` returns Q/R). A Julia `Factorization` keeps its factors packed in LAPACK's working storage, so timing the compact object would compare strictly less work than the tenferro-rs and PyTorch columns, which return separate materialized tensors.
- Julia dense linalg rows run through Julia's own BLAS/LAPACK (libblastrampoline, by default OpenBLAS), recorded as `julia.blas_provider` in the run metadata. When that differs from the provider tenferro-rs and PyTorch link against (Accelerate on macOS), those rows partly compare BLAS implementations rather than framework overhead; read them together with the recorded providers.
- The Julia `lstsq` row uses `qr(a) \ rhs` rather than `a \ rhs`: the bare backslash runs a column-pivoted, rank-revealing QR (the LAPACK `gelsy` algorithm), while the PyTorch row selects the `gels` driver, so the unpivoted spelling is the like-for-like comparison.
- `julia-base` uses the natural Base/LinearAlgebra spelling and `strided-jl` the natural Strided.jl (`@strided`) spelling; each is populated only for rows where that spelling naturally applies (reductions and dense linalg have no natural Strided.jl spelling, so `strided-jl` covers elementwise/chain/transpose rows, the elementwise `cpu/output_reuse` `_into` rows, and the elementwise `cpu/complex` rows conj/mul/div/exp/log).
- Strided.jl (https://github.com/Jutho/Strided.jl) is prior art for tenferro-rs' strided-rs kernel layer; the `strided-jl` column credits that lineage directly in the report.

## Threads: 1 4

- CSV: `data/results/mac-cpu/cpu/public_api/20260822_091106/cpu_public_api_t1_20260822_091106.csv`
- CSV: `data/results/mac-cpu/cpu/public_api/20260822_091106/cpu_public_api_t4_20260822_091106.csv`
- Source table: `data/results/mac-cpu/cpu/public_api/20260822_091106/cpu_public_api_20260822_091106.md`

## CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| cpu/complex | `cholesky` | c64 | 1 | `448x448` | 2.675 ± 0.045 | 2.855 ± 0.067 | 3.133 ± 0.097 | 0.805 ± 0.029 | 3.085 ± 0.244 | - |
| cpu/complex | `cholesky` | c64 | 4 | `448x448` | 2.747 ± 0.036 | 2.860 ± 0.043 | 2.735 ± 0.041 | 0.791 ± 0.018 | 1.438 ± 0.255 | - |
| cpu/complex | `conj` | c64 | 1 | `16777216` | 7.737 ± 1.286 | 6.176 ± 0.284 | 6.186 ± 0.129 | 6.074 ± 0.866 | 6.250 ± 0.090 | 6.281 ± 0.128 |
| cpu/complex | `conj` | c64 | 4 | `16777216` | 5.526 ± 0.192 | 5.509 ± 0.284 | 5.331 ± 0.083 | 5.390 ± 0.297 | 6.245 ± 0.390 | 16.648 ± 8.506 |
| cpu/complex | `div` | c64 | 1 | `8388608` | 7.572 ± 0.076 | 7.604 ± 0.159 | 11.364 ± 0.531 | 6.598 ± 0.699 | 23.262 ± 0.062 | 23.209 ± 0.089 |
| cpu/complex | `div` | c64 | 4 | `8388608` | 4.207 ± 0.042 | 4.239 ± 0.068 | 4.223 ± 0.061 | 7.094 ± 0.785 | 25.985 ± 0.478 | 7.816 ± 5.384 |
| cpu/complex | `dot_general` | c64 | 1 | `640x640` | 6.414 ± 0.294 | 6.714 ± 0.384 | 6.146 ± 0.233 | 7.181 ± 0.587 | 42.792 ± 0.140 | - |
| cpu/complex | `dot_general` | c64 | 4 | `640x640` | 5.684 ± 0.700 | 6.154 ± 0.075 | 5.535 ± 0.163 | 7.884 ± 1.136 | 13.087 ± 1.582 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 1 | `640x640` | 6.710 ± 0.201 | unsupported | 6.337 ± 0.286 | 7.714 ± 0.895 | 42.826 ± 0.142 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 4 | `640x640` | 6.004 ± 0.154 | unsupported | 5.802 ± 0.146 | 8.474 ± 0.709 | 13.820 ± 1.750 | - |
| cpu/complex | `eig` | c64 | 1 | `112x112` | 6.150 ± 0.032 | 6.143 ± 0.048 | 6.176 ± 0.021 | 11.452 ± 0.880 | 5.520 ± 0.030 | - |
| cpu/complex | `eig` | c64 | 4 | `112x112` | 6.178 ± 0.054 | 6.201 ± 0.016 | 6.179 ± 0.083 | 12.206 ± 0.687 | 5.944 ± 0.409 | - |
| cpu/complex | `exp` | c64 | 1 | `4194304` | 23.218 ± 0.635 | 24.127 ± 0.341 | 24.086 ± 0.714 | 8.299 ± 1.154 | 27.376 ± 0.092 | 27.292 ± 0.098 |
| cpu/complex | `exp` | c64 | 4 | `4194304` | 6.983 ± 0.668 | 7.055 ± 0.769 | 7.735 ± 0.371 | 9.341 ± 1.107 | 30.245 ± 3.517 | 9.198 ± 3.539 |
| cpu/complex | `log` | c64 | 1 | `4194304` | 26.165 ± 0.128 | 26.191 ± 0.141 | 28.313 ± 0.099 | 12.664 ± 0.672 | 40.975 ± 2.860 | 38.333 ± 1.375 |
| cpu/complex | `log` | c64 | 4 | `4194304` | 8.041 ± 1.229 | 7.993 ± 0.366 | 9.256 ± 0.213 | 13.273 ± 0.640 | 43.983 ± 3.101 | 14.962 ± 3.118 |
| cpu/complex | `mul` | c64 | 1 | `8388608` | 4.877 ± 0.776 | 4.458 ± 0.586 | 4.570 ± 0.041 | 4.522 ± 0.225 | 4.396 ± 0.077 | 4.400 ± 0.068 |
| cpu/complex | `mul` | c64 | 4 | `8388608` | 4.330 ± 0.197 | 4.285 ± 0.085 | 4.189 ± 0.025 | 4.335 ± 0.374 | 4.450 ± 0.085 | 7.390 ± 10.734 |
| cpu/complex | `norm_fro` | c64 | 1 | `2048x1536` | 4.818 ± 0.063 | 4.341 ± 0.036 | 4.027 ± 0.056 | 1.302 ± 0.122 | 6.238 ± 0.023 | - |
| cpu/complex | `norm_fro` | c64 | 4 | `2048x1536` | 2.165 ± 0.030 | 1.670 ± 0.183 | 1.117 ± 0.022 | 1.254 ± 0.097 | 6.715 ± 0.268 | - |
| cpu/complex | `qr` | c64 | 1 | `256x256` | 4.539 ± 0.051 | 4.641 ± 0.036 | 4.816 ± 0.064 | 3.773 ± 0.026 | 6.382 ± 0.078 | - |
| cpu/complex | `qr` | c64 | 4 | `256x256` | 4.531 ± 0.044 | 4.570 ± 0.019 | 4.624 ± 0.064 | 3.776 ± 0.037 | 3.492 ± 1.424 | - |
| cpu/complex | `solve` | c64 | 1 | `384x384,rhs=8` | 3.427 ± 0.046 | 3.470 ± 0.092 | 3.171 ± 0.052 | 1.318 ± 0.054 | 3.840 ± 0.023 | - |
| cpu/complex | `solve` | c64 | 4 | `384x384,rhs=8` | 3.592 ± 0.026 | 3.589 ± 0.045 | 3.275 ± 0.033 | 1.292 ± 0.014 | 1.683 ± 0.162 | - |
| cpu/complex | `svd` | c64 | 1 | `160x160` | 3.781 ± 0.082 | 3.871 ± 0.053 | 3.851 ± 0.057 | 2.715 ± 0.098 | 4.737 ± 0.048 | - |
| cpu/complex | `svd` | c64 | 4 | `160x160` | 3.888 ± 0.032 | 3.892 ± 0.018 | 3.852 ± 0.018 | 2.730 ± 0.072 | 3.349 ± 0.291 | - |
| cpu/complex | `tensordot` | c64 | 1 | `640x640` | 6.343 ± 0.587 | 6.686 ± 0.182 | 6.161 ± 0.332 | 8.727 ± 1.000 | 42.582 ± 0.157 | - |
| cpu/complex | `tensordot` | c64 | 4 | `640x640` | 5.608 ± 0.211 | 6.127 ± 0.093 | 5.636 ± 0.149 | 8.320 ± 1.265 | 13.533 ± 2.338 | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 1 | `1024x1024` | 6.423 ± 1.175 | unsupported | 5.151 ± 0.420 | 7.328 ± 0.574 | - | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 4 | `1024x1024` | 4.723 ± 0.590 | unsupported | 4.613 ± 0.030 | 8.072 ± 0.328 | - | - |
| cpu/elementwise_reduction | `abs` | f64 | 1 | `33554432` | 6.068 ± 0.594 | 6.164 ± 0.220 | 6.102 ± 0.130 | 5.455 ± 0.404 | 6.091 ± 0.147 | 8.467 ± 0.133 |
| cpu/elementwise_reduction | `abs` | f64 | 4 | `33554432` | 5.309 ± 0.105 | 5.358 ± 0.065 | 5.382 ± 0.200 | 5.441 ± 0.181 | 6.147 ± 0.118 | 5.664 ± 3.569 |
| cpu/elementwise_reduction | `add` | f64 | 1 | `33554432` | 9.562 ± 0.966 | 9.606 ± 0.980 | 8.721 ± 0.579 | 11.325 ± 0.387 | 9.938 ± 1.197 | 10.049 ± 1.102 |
| cpu/elementwise_reduction | `add` | f64 | 4 | `33554432` | 8.427 ± 0.181 | 8.429 ± 0.068 | 8.367 ± 0.073 | 11.303 ± 0.031 | 8.847 ± 0.107 | 12.438 ± 10.374 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 1 | `4194304` | 19.137 ± 1.702 | 19.172 ± 0.732 | 22.284 ± 0.114 | 5.572 ± 0.476 | 23.884 ± 0.692 | 24.920 ± 0.756 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 4 | `4194304` | 6.590 ± 0.693 | 6.786 ± 0.503 | 8.105 ± 1.587 | 5.930 ± 0.640 | 26.546 ± 1.582 | 7.750 ± 0.867 |
| cpu/elementwise_reduction | `clamp` | f64 | 1 | `8388608` | 3.137 ± 0.072 | 3.221 ± 0.254 | 2.954 ± 0.039 | 3.723 ± 0.077 | 2.985 ± 0.131 | 3.305 ± 0.097 |
| cpu/elementwise_reduction | `clamp` | f64 | 4 | `8388608` | 2.909 ± 0.042 | 2.955 ± 0.601 | 2.892 ± 0.016 | 3.699 ± 0.242 | 3.200 ± 0.330 | 3.179 ± 1.288 |
| cpu/elementwise_reduction | `compare_lt` | f64 | 1 | `33554432` | 7.517 ± 0.190 | 7.557 ± 0.606 | 8.750 ± 0.395 | 7.789 ± 0.268 | 7.699 ± 0.096 | - |
| cpu/elementwise_reduction | `compare_lt` | f64 | 4 | `33554432` | 7.349 ± 0.107 | 7.378 ± 0.077 | 7.336 ± 0.034 | 7.752 ± 0.099 | 7.797 ± 0.149 | - |
| cpu/elementwise_reduction | `cos` | f64 | 1 | `8388608` | 17.412 ± 0.204 | 17.420 ± 0.185 | 21.027 ± 0.048 | 4.170 ± 0.191 | 22.419 ± 0.107 | 23.576 ± 1.009 |
| cpu/elementwise_reduction | `cos` | f64 | 4 | `8388608` | 5.725 ± 0.963 | 5.609 ± 0.298 | 6.583 ± 0.232 | 4.790 ± 0.973 | 24.564 ± 1.452 | 7.116 ± 0.791 |
| cpu/elementwise_reduction | `div` | f64 | 1 | `33554432` | 8.593 ± 0.199 | 8.643 ± 0.260 | 8.679 ± 0.300 | 11.322 ± 0.465 | 8.651 ± 0.091 | 9.264 ± 0.523 |
| cpu/elementwise_reduction | `div` | f64 | 4 | `33554432` | 8.376 ± 0.091 | 8.408 ± 0.079 | 8.407 ± 0.076 | 11.157 ± 0.401 | 8.720 ± 0.075 | 9.195 ± 3.665 |
| cpu/elementwise_reduction | `exp` | f64 | 1 | `8388608` | 18.667 ± 2.164 | 17.990 ± 0.675 | 14.944 ± 0.042 | 3.182 ± 0.497 | 20.860 ± 0.095 | 20.840 ± 0.079 |
| cpu/elementwise_reduction | `exp` | f64 | 4 | `8388608` | 6.382 ± 1.269 | 5.648 ± 0.404 | 4.830 ± 0.473 | 3.515 ± 0.562 | 23.282 ± 2.506 | 6.156 ± 3.035 |
| cpu/elementwise_reduction | `expm1` | f64 | 1 | `4194304` | 8.957 ± 0.123 | 9.466 ± 1.817 | 17.826 ± 0.103 | 3.383 ± 0.131 | 13.914 ± 0.108 | 14.078 ± 0.092 |
| cpu/elementwise_reduction | `expm1` | f64 | 4 | `4194304` | 3.551 ± 0.976 | 2.770 ± 0.349 | 5.536 ± 0.426 | 3.712 ± 0.417 | 15.570 ± 0.584 | 4.393 ± 2.054 |
| cpu/elementwise_reduction | `log` | f64 | 1 | `8388608` | 17.948 ± 0.714 | 17.227 ± 0.669 | 26.333 ± 0.114 | 4.702 ± 0.209 | 24.429 ± 0.766 | 25.795 ± 1.327 |
| cpu/elementwise_reduction | `log` | f64 | 4 | `8388608` | 5.420 ± 0.996 | 5.456 ± 0.202 | 7.934 ± 0.166 | 5.382 ± 0.658 | 26.742 ± 1.477 | 8.549 ± 2.824 |
| cpu/elementwise_reduction | `log1p` | f64 | 1 | `4194304` | 11.562 ± 2.655 | 12.072 ± 1.063 | 13.660 ± 0.073 | 3.392 ± 0.526 | 13.464 ± 0.889 | 13.293 ± 0.057 |
| cpu/elementwise_reduction | `log1p` | f64 | 4 | `4194304` | 4.170 ± 2.603 | 3.435 ± 0.221 | 4.180 ± 0.677 | 3.844 ± 1.475 | 14.731 ± 0.512 | 3.948 ± 0.216 |
| cpu/elementwise_reduction | `maximum` | f64 | 1 | `33554432` | 8.653 ± 0.163 | 8.717 ± 0.226 | 8.793 ± 0.548 | 11.304 ± 0.454 | 8.677 ± 0.057 | 9.738 ± 1.161 |
| cpu/elementwise_reduction | `maximum` | f64 | 4 | `33554432` | 8.405 ± 0.183 | 8.407 ± 0.058 | 8.356 ± 0.087 | 11.304 ± 0.065 | 8.827 ± 0.466 | 8.696 ± 3.209 |
| cpu/elementwise_reduction | `minimum` | f64 | 1 | `33554432` | 8.717 ± 0.425 | 8.742 ± 0.149 | 8.640 ± 0.667 | 11.339 ± 0.496 | 8.693 ± 0.095 | 8.979 ± 0.239 |
| cpu/elementwise_reduction | `minimum` | f64 | 4 | `33554432` | 8.368 ± 0.085 | 8.392 ± 0.094 | 8.392 ± 0.083 | 11.314 ± 0.192 | 8.750 ± 0.312 | 9.177 ± 3.555 |
| cpu/elementwise_reduction | `mul` | f64 | 1 | `33554432` | 8.973 ± 0.878 | 8.763 ± 0.874 | 8.610 ± 0.493 | 11.329 ± 0.468 | 8.603 ± 0.108 | 9.304 ± 0.484 |
| cpu/elementwise_reduction | `mul` | f64 | 4 | `33554432` | 8.358 ± 0.064 | 8.402 ± 0.068 | 8.376 ± 0.049 | 11.346 ± 0.026 | 8.865 ± 0.205 | 8.821 ± 3.173 |
| cpu/elementwise_reduction | `neg` | f64 | 1 | `33554432` | 6.093 ± 0.135 | 6.350 ± 0.242 | 6.093 ± 0.110 | 5.442 ± 0.184 | 6.176 ± 0.340 | 8.466 ± 0.099 |
| cpu/elementwise_reduction | `neg` | f64 | 4 | `33554432` | 5.328 ± 0.091 | 5.364 ± 0.085 | 5.365 ± 0.090 | 6.099 ± 0.168 | 6.167 ± 0.148 | 5.502 ± 3.161 |
| cpu/elementwise_reduction | `pow` | f64 | 1 | `4194304` | 20.210 ± 0.143 | 20.298 ± 0.133 | 45.141 ± 0.180 | 6.134 ± 0.990 | 37.810 ± 1.275 | 37.930 ± 0.417 |
| cpu/elementwise_reduction | `pow` | f64 | 4 | `4194304` | 6.355 ± 1.104 | 6.028 ± 0.227 | 13.197 ± 0.629 | 6.829 ± 2.173 | 37.697 ± 3.417 | 11.608 ± 0.376 |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 1 | `2048x2048` | 6.150 ± 0.068 | 6.158 ± 0.032 | 9.162 ± 0.404 | 0.591 ± 0.051 | 0.405 ± 0.006 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 4 | `2048x2048` | 6.151 ± 0.036 | 6.188 ± 0.053 | 5.993 ± 0.716 | 0.600 ± 0.038 | 0.413 ± 0.013 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 1 | `4096x4096` | 3.754 ± 0.030 | 3.821 ± 0.141 | 4.301 ± 0.083 | 1.952 ± 0.080 | 2.077 ± 0.013 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 4 | `4096x4096` | 3.755 ± 0.042 | 3.776 ± 0.033 | 1.752 ± 0.012 | 1.967 ± 0.039 | 2.075 ± 0.019 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 1 | `8192x4096` | 4.070 ± 0.404 | 4.042 ± 0.216 | 3.906 ± 0.035 | 5.304 ± 0.399 | 3.832 ± 0.038 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 4 | `8192x4096` | 8.218 ± 0.495 | 7.555 ± 0.108 | 3.696 ± 0.096 | 5.300 ± 1.733 | 3.877 ± 0.027 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 1 | `8192x4096` | 4.217 ± 0.101 | 3.986 ± 0.303 | 3.857 ± 0.072 | 4.190 ± 0.375 | 3.806 ± 0.040 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 4 | `8192x4096` | 5.653 ± 0.601 | 4.804 ± 0.101 | 3.699 ± 0.008 | 4.071 ± 0.205 | 3.825 ± 0.051 | - |
| cpu/elementwise_reduction | `rem` | f64 | 1 | `8388608` | 19.375 ± 0.137 | 19.527 ± 0.132 | 22.574 ± 0.187 | 6.552 ± 0.665 | 15.706 ± 2.953 | 16.588 ± 0.840 |
| cpu/elementwise_reduction | `rem` | f64 | 4 | `8388608` | 5.032 ± 0.877 | 5.750 ± 0.488 | 6.024 ± 0.378 | 6.709 ± 1.839 | 16.435 ± 1.357 | 4.857 ± 1.617 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 1 | `33554432` | 12.774 ± 0.234 | 12.845 ± 0.342 | 12.795 ± 0.237 | 6.838 ± 0.572 | 30.151 ± 0.119 | 29.960 ± 0.302 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 4 | `33554432` | 5.349 ± 0.184 | 5.730 ± 0.536 | 5.328 ± 0.033 | 6.789 ± 1.001 | 30.105 ± 0.342 | 8.768 ± 4.162 |
| cpu/elementwise_reduction | `select` | f64 | 1 | `33554432` | 12.151 ± 0.524 | 12.088 ± 0.497 | 9.940 ± 0.956 | 11.828 ± 0.464 | 12.046 ± 0.088 | - |
| cpu/elementwise_reduction | `select` | f64 | 4 | `33554432` | 12.034 ± 1.300 | 11.727 ± 0.074 | 8.854 ± 1.114 | 11.782 ± 0.164 | 12.778 ± 1.218 | - |
| cpu/elementwise_reduction | `sign` | f64 | 1 | `33554432` | 6.223 ± 0.229 | 6.309 ± 0.243 | 6.404 ± 0.563 | 5.423 ± 0.116 | 6.145 ± 0.065 | 12.677 ± 0.075 |
| cpu/elementwise_reduction | `sign` | f64 | 4 | `33554432` | 5.615 ± 0.072 | 5.691 ± 0.096 | 5.319 ± 0.099 | 5.503 ± 0.277 | 6.483 ± 0.674 | 5.820 ± 3.209 |
| cpu/elementwise_reduction | `sin` | f64 | 1 | `8388608` | 16.210 ± 0.203 | 16.860 ± 0.838 | 18.067 ± 0.082 | 4.160 ± 0.238 | 20.515 ± 0.075 | 20.464 ± 0.150 |
| cpu/elementwise_reduction | `sin` | f64 | 4 | `8388608` | 5.142 ± 0.935 | 5.504 ± 0.582 | 5.665 ± 0.489 | 4.204 ± 0.191 | 23.472 ± 2.380 | 6.037 ± 0.523 |
| cpu/elementwise_reduction | `sqrt` | f64 | 1 | `33554432` | 8.371 ± 0.310 | 8.451 ± 0.422 | 8.389 ± 0.290 | 5.441 ± 0.411 | 17.848 ± 0.618 | 17.560 ± 0.213 |
| cpu/elementwise_reduction | `sqrt` | f64 | 4 | `33554432` | 5.386 ± 0.133 | 5.389 ± 0.081 | 5.340 ± 0.139 | 5.426 ± 0.254 | 18.088 ± 0.496 | 6.982 ± 3.983 |
| cpu/elementwise_reduction | `sub` | f64 | 1 | `33554432` | 9.175 ± 0.485 | 9.175 ± 0.417 | 8.659 ± 0.194 | 11.323 ± 0.444 | 9.765 ± 0.567 | 9.064 ± 0.421 |
| cpu/elementwise_reduction | `sub` | f64 | 4 | `33554432` | 8.359 ± 0.072 | 8.424 ± 0.085 | 8.370 ± 0.060 | 11.289 ± 0.115 | 8.981 ± 0.103 | 9.093 ± 11.765 |
| cpu/elementwise_reduction | `tanh` | f64 | 1 | `8388608` | 22.840 ± 0.186 | 22.951 ± 0.137 | 42.878 ± 0.143 | 2.784 ± 0.376 | 27.140 ± 0.236 | 27.175 ± 0.173 |
| cpu/elementwise_reduction | `tanh` | f64 | 4 | `8388608` | 7.158 ± 1.198 | 7.291 ± 0.562 | 12.925 ± 0.597 | 2.888 ± 0.606 | 29.803 ± 1.174 | 8.629 ± 1.560 |
| cpu/indexing_layout | `concatenate` | f64 | 1 | `1048576+1048576` | 0.295 ± 0.004 | 0.682 ± 0.048 | 0.335 ± 0.036 | 0.336 ± 0.044 | 0.378 ± 0.020 | - |
| cpu/indexing_layout | `concatenate` | f64 | 4 | `1048576+1048576` | 0.311 ± 0.007 | 0.704 ± 0.032 | 0.300 ± 0.029 | 0.329 ± 0.026 | 0.375 ± 0.029 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 1 | `4194304 -> 2097152` | 0.315 ± 0.022 | 1.136 ± 0.111 | 0.303 ± 0.012 | 0.280 ± 0.011 | 0.376 ± 0.029 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 4 | `4194304 -> 2097152` | 0.308 ± 0.027 | 1.084 ± 0.029 | 0.264 ± 0.013 | 0.256 ± 0.007 | 0.369 ± 0.026 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 1 | `2097152` | 0.505 ± 0.041 | unsupported | 0.521 ± 0.044 | 0.463 ± 0.033 | 0.538 ± 0.024 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 4 | `2097152` | 0.496 ± 0.014 | unsupported | 0.449 ± 0.005 | 0.457 ± 0.016 | 0.549 ± 0.033 | - |
| cpu/indexing_layout | `gather` | f64 | 1 | `262144` | 3.346 ± 0.840 | 2.998 ± 0.042 | 0.180 ± 0.005 | 0.113 ± 0.011 | 0.282 ± 0.007 | - |
| cpu/indexing_layout | `gather` | f64 | 4 | `262144` | 1.101 ± 0.254 | 0.923 ± 0.079 | 0.179 ± 0.001 | 0.079 ± 0.009 | 0.281 ± 0.091 | - |
| cpu/indexing_layout | `pad` | f64 | 1 | `2097152` | 0.507 ± 0.013 | 0.929 ± 0.027 | 0.581 ± 0.101 | 0.255 ± 0.013 | - | - |
| cpu/indexing_layout | `pad` | f64 | 4 | `2097152` | 0.515 ± 0.036 | 0.868 ± 0.016 | 0.499 ± 0.006 | 0.275 ± 0.017 | - | - |
| cpu/indexing_layout | `reverse` | f64 | 1 | `2097152` | 0.791 ± 0.016 | 1.113 ± 0.013 | 0.358 ± 0.145 | 0.268 ± 0.030 | 0.837 ± 0.012 | - |
| cpu/indexing_layout | `reverse` | f64 | 4 | `2097152` | 0.793 ± 0.016 | 1.078 ± 0.030 | 0.253 ± 0.005 | 0.265 ± 0.009 | 0.839 ± 0.005 | - |
| cpu/indexing_layout | `scatter` | f64 | 1 | `262144` | 2.552 ± 0.019 | 2.699 ± 0.087 | 0.278 ± 0.006 | 0.247 ± 0.009 | 0.284 ± 0.012 | - |
| cpu/indexing_layout | `scatter` | f64 | 4 | `262144` | 2.568 ± 0.260 | 2.701 ± 0.061 | 0.276 ± 0.006 | 0.247 ± 0.018 | 0.285 ± 0.073 | - |
| cpu/indexing_layout | `slice` | f64 | 1 | `4194304 -> 2096128` | 0.787 ± 0.007 | 1.517 ± 0.035 | 0.569 ± 0.038 | 0.477 ± 0.028 | 0.667 ± 0.028 | - |
| cpu/indexing_layout | `slice` | f64 | 4 | `4194304 -> 2096128` | 0.805 ± 0.016 | 1.494 ± 0.024 | 0.488 ± 0.013 | 0.495 ± 0.018 | 0.676 ± 0.020 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 1 | `1536x1536` | 5.395 ± 0.357 | 5.551 ± 0.243 | 14.033 ± 0.218 | 6.907 ± 0.263 | 26.989 ± 34.313 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 4 | `1536x1536` | 5.008 ± 0.684 | 5.574 ± 0.089 | 7.429 ± 0.123 | 7.156 ± 0.396 | 13.376 ± 12.618 | - |
| cpu/linalg_uncovered | `det` | f64 | 1 | `1024x1024` | 5.146 ± 0.074 | 5.312 ± 0.082 | 4.271 ± 0.080 | 4.074 ± 0.106 | 14.595 ± 0.162 | - |
| cpu/linalg_uncovered | `det` | f64 | 4 | `1024x1024` | 5.153 ± 0.083 | 5.341 ± 0.056 | 4.227 ± 0.057 | 4.255 ± 0.350 | 6.877 ± 0.583 | - |
| cpu/linalg_uncovered | `eig` | f64 | 1 | `160x160` | 4.717 ± 0.085 | 4.726 ± 0.070 | 4.739 ± 0.082 | 5.437 ± 0.114 | 4.302 ± 0.043 | - |
| cpu/linalg_uncovered | `eig` | f64 | 4 | `160x160` | 4.741 ± 0.055 | 4.764 ± 0.030 | 4.738 ± 0.021 | 5.451 ± 0.037 | 4.508 ± 0.249 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 1 | `192x192` | 4.947 ± 0.031 | 4.934 ± 0.022 | 5.030 ± 0.448 | 4.463 ± 0.062 | 4.798 ± 0.055 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 4 | `192x192` | 4.969 ± 0.209 | 5.002 ± 0.037 | 4.953 ± 0.112 | 4.569 ± 0.061 | 5.384 ± 0.234 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 1 | `512x512` | 13.296 ± 0.081 | 13.143 ± 0.188 | 13.314 ± 0.056 | 13.050 ± 0.198 | 11.883 ± 0.080 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 4 | `512x512` | 13.622 ± 0.305 | 13.387 ± 0.094 | 13.264 ± 0.369 | 12.832 ± 0.197 | 8.177 ± 0.641 | - |
| cpu/linalg_uncovered | `inv` | f64 | 1 | `768x768` | 7.499 ± 0.113 | 6.629 ± 0.087 | 5.206 ± 0.082 | 4.949 ± 0.127 | 20.522 ± 0.183 | - |
| cpu/linalg_uncovered | `inv` | f64 | 4 | `768x768` | 7.076 ± 0.047 | 6.309 ± 0.080 | 4.509 ± 0.055 | 4.577 ± 0.203 | 8.770 ± 2.927 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 1 | `768x384,rhs=16` | 5.332 ± 0.064 | 5.151 ± 0.084 | 3.067 ± 0.021 | 12.638 ± 0.323 | 5.472 ± 0.035 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 4 | `768x384,rhs=16` | 5.408 ± 0.048 | 5.264 ± 0.044 | 2.979 ± 0.016 | 12.870 ± 0.171 | 3.858 ± 0.193 | - |
| cpu/linalg_uncovered | `lu` | f64 | 1 | `1024x1024` | 4.807 ± 0.214 | 5.256 ± 0.106 | 6.674 ± 0.134 | 5.053 ± 0.090 | 16.198 ± 4.589 | - |
| cpu/linalg_uncovered | `lu` | f64 | 4 | `1024x1024` | 4.758 ± 0.082 | 5.284 ± 0.054 | 6.071 ± 0.083 | 5.141 ± 0.253 | 9.915 ± 3.246 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 1 | `2048x2048` | 0.428 ± 0.023 | 0.469 ± 0.040 | 1.363 ± 0.012 | 1.243 ± 0.168 | 4.158 ± 0.019 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 4 | `2048x2048` | 0.621 ± 0.020 | 0.645 ± 0.022 | 1.375 ± 0.023 | 1.213 ± 0.157 | 4.558 ± 0.104 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 1 | `512x256` | 6.442 ± 0.112 | 6.483 ± 0.051 | 6.296 ± 0.051 | 6.328 ± 0.081 | 14.152 ± 0.115 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 4 | `512x256` | 6.583 ± 0.042 | 6.696 ± 0.071 | 6.365 ± 0.052 | 6.438 ± 0.089 | 12.440 ± 3.296 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 1 | `512x256` | 6.464 ± 0.075 | 6.508 ± 0.049 | 6.295 ± 0.052 | 6.331 ± 0.123 | 14.058 ± 0.122 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 4 | `512x256` | 6.594 ± 0.038 | 6.644 ± 0.065 | 6.367 ± 0.062 | 6.404 ± 0.069 | 11.773 ± 1.895 | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 1 | `1024x1024` | 5.128 ± 0.130 | 5.546 ± 0.077 | 4.277 ± 0.214 | 4.123 ± 0.156 | 14.558 ± 0.160 | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 4 | `1024x1024` | 5.122 ± 0.046 | 5.634 ± 0.058 | 4.183 ± 0.049 | 4.107 ± 0.047 | 6.681 ± 0.574 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 1 | `768x384` | unsupported | unsupported | 18.084 ± 0.256 | 17.615 ± 0.312 | 47.598 ± 0.242 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 4 | `768x384` | unsupported | unsupported | 18.674 ± 0.216 | 18.584 ± 0.264 | 36.554 ± 1.218 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 1 | `4096x4096,rhs=64` | 5.814 ± 0.108 | 9.144 ± 0.425 | 4.487 ± 0.053 | 14.828 ± 0.296 | 21.749 ± 0.135 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 4 | `4096x4096,rhs=64` | 6.685 ± 0.041 | 9.505 ± 0.158 | 4.846 ± 0.023 | 16.181 ± 0.924 | 9.555 ± 1.642 | - |
| cpu/output_reuse | `add_into` | f64 | 1 | `33554432` | 25.190 ± 0.204 | unsupported | 8.579 ± 0.258 | - | 8.465 ± 0.089 | 8.952 ± 0.561 |
| cpu/output_reuse | `add_into` | f64 | 4 | `33554432` | 8.583 ± 0.384 | unsupported | 8.553 ± 0.115 | - | 8.510 ± 0.060 | 8.336 ± 0.051 |
| cpu/output_reuse | `conj_into` | c64 | 1 | `16777216` | 10.572 ± 0.322 | unsupported | 6.184 ± 0.133 | - | 6.029 ± 0.066 | 6.077 ± 0.059 |
| cpu/output_reuse | `conj_into` | c64 | 4 | `16777216` | 5.461 ± 0.145 | unsupported | 6.148 ± 1.068 | - | 6.079 ± 0.066 | 5.372 ± 0.094 |
| cpu/output_reuse | `copy_read_into` | f64 | 1 | `33554432` | 5.876 ± 0.424 | unsupported | 6.051 ± 0.677 | - | 5.826 ± 0.079 | - |
| cpu/output_reuse | `copy_read_into` | f64 | 4 | `33554432` | 5.345 ± 0.223 | unsupported | 5.421 ± 0.401 | - | 5.852 ± 0.039 | - |
| cpu/output_reuse | `div_into` | f64 | 1 | `33554432` | 29.458 ± 0.123 | unsupported | 8.626 ± 0.539 | - | 8.507 ± 0.249 | 8.916 ± 0.603 |
| cpu/output_reuse | `div_into` | f64 | 4 | `33554432` | 8.534 ± 0.350 | unsupported | 8.386 ± 0.052 | - | 8.815 ± 0.350 | 8.346 ± 0.091 |
| cpu/output_reuse | `dot_general_read_into` | f64 | 1 | `1024x1024` | 5.017 ± 0.172 | unsupported | 4.868 ± 0.183 | - | 38.245 ± 0.061 | - |
| cpu/output_reuse | `dot_general_read_into` | f64 | 4 | `1024x1024` | 4.550 ± 0.022 | unsupported | 4.571 ± 0.026 | - | 12.104 ± 0.542 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 1 | `1024x1024` | 5.651 ± 0.228 | unsupported | 5.606 ± 0.062 | - | 38.188 ± 0.116 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 4 | `1024x1024` | 4.868 ± 0.037 | unsupported | 5.002 ± 0.057 | - | 12.471 ± 0.391 | - |
| cpu/output_reuse | `mul_into` | f64 | 1 | `33554432` | 25.211 ± 0.155 | unsupported | 8.599 ± 0.531 | - | 8.518 ± 0.080 | 9.156 ± 0.498 |
| cpu/output_reuse | `mul_into` | f64 | 4 | `33554432` | 8.527 ± 0.372 | unsupported | 8.889 ± 0.676 | - | 8.508 ± 0.062 | 8.329 ± 0.052 |
| cpu/output_reuse | `neg_into` | f64 | 1 | `33554432` | 16.755 ± 0.216 | unsupported | 6.093 ± 0.296 | - | 5.991 ± 0.048 | 8.378 ± 0.108 |
| cpu/output_reuse | `neg_into` | f64 | 4 | `33554432` | 5.360 ± 0.115 | unsupported | 5.366 ± 0.063 | - | 6.075 ± 0.133 | 5.389 ± 0.141 |
| cpu/output_reuse | `sub_into` | f64 | 1 | `33554432` | 25.191 ± 0.150 | unsupported | 8.683 ± 0.492 | - | 8.477 ± 0.035 | 9.066 ± 0.260 |
| cpu/output_reuse | `sub_into` | f64 | 4 | `33554432` | 8.624 ± 0.214 | unsupported | 8.396 ± 0.097 | - | 8.462 ± 0.087 | 8.342 ± 0.055 |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 1 | `8192x1 -> 8192x4096` | 5.294 ± 0.434 | 3.660 ± 0.024 | 3.697 ± 0.060 | 3.656 ± 0.025 | 9.078 ± 0.032 | - |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 4 | `8192x1 -> 8192x4096` | 3.893 ± 0.101 | 3.605 ± 0.263 | 3.649 ± 0.016 | 3.663 ± 0.267 | 9.195 ± 0.109 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 1 | `33554432` | 4.915 ± 0.141 | 11.011 ± 0.726 | 4.932 ± 0.200 | 4.274 ± 0.115 | 4.958 ± 0.051 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 4 | `33554432` | 4.200 ± 0.084 | 9.625 ± 0.064 | 4.163 ± 0.039 | 4.282 ± 0.153 | 4.992 ± 0.039 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 1 | `8192 -> 8192x8192` | 4.692 ± 0.243 | 4.710 ± 0.122 | 7.400 ± 0.216 | 7.292 ± 0.173 | 4.631 ± 0.103 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 4 | `8192 -> 8192x8192` | 4.613 ± 0.022 | 4.643 ± 0.038 | 7.369 ± 0.294 | 7.072 ± 0.666 | 4.643 ± 0.195 | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 1 | `8388608x2x2 -> 8388608x2` | 3.003 ± 0.067 | 9.367 ± 0.552 | 6.347 ± 0.092 | 4.255 ± 0.153 | - | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 4 | `8388608x2x2 -> 8388608x2` | 3.061 ± 0.053 | 8.503 ± 0.050 | 4.225 ± 0.120 | 4.312 ± 0.135 | - | - |
| cpu/structural_shape | `reshape` | f64 | 1 | `33554432 -> 8192x4096` | 5.889 ± 0.192 | 6.125 ± 0.250 | 6.184 ± 0.506 | 5.184 ± 0.462 | 5.930 ± 0.123 | - |
| cpu/structural_shape | `reshape` | f64 | 4 | `33554432 -> 8192x4096` | 5.842 ± 0.090 | 5.400 ± 0.056 | 5.354 ± 0.160 | 5.115 ± 0.075 | 5.940 ± 0.086 | - |
| cpu/structural_shape | `transpose` | f64 | 1 | `4096x4096` | 21.728 ± 0.508 | 21.690 ± 0.485 | 19.812 ± 0.609 | 7.097 ± 0.436 | 54.927 ± 2.322 | 20.953 ± 0.114 |
| cpu/structural_shape | `transpose` | f64 | 4 | `4096x4096` | 10.433 ± 1.187 | 11.066 ± 1.027 | 19.939 ± 0.828 | 7.104 ± 0.325 | 52.841 ± 0.515 | 20.699 ± 8.112 |
| cpu/structural_shape | `tril` | f64 | 1 | `4096x4096` | 4.520 ± 0.080 | 7.715 ± 0.442 | 5.217 ± 0.390 | 2.769 ± 0.161 | 5.427 ± 0.048 | - |
| cpu/structural_shape | `tril` | f64 | 4 | `4096x4096` | 4.548 ± 0.146 | 7.335 ± 0.050 | 3.689 ± 1.172 | 2.788 ± 0.157 | 5.548 ± 0.165 | - |
| cpu/structural_shape | `triu` | f64 | 1 | `4096x4096` | 5.035 ± 0.593 | 9.277 ± 0.934 | 4.676 ± 0.112 | 2.750 ± 0.163 | 5.295 ± 0.058 | - |
| cpu/structural_shape | `triu` | f64 | 4 | `4096x4096` | 4.849 ± 0.079 | 7.880 ± 0.101 | 3.174 ± 0.348 | 2.927 ± 0.180 | 5.363 ± 0.130 | - |
| cpu/view_metadata | `broadcast_in_dim_view` | f64 | 1 | `8192x1 -> 8192x4096` | 0.002 ± 0.000 | unsupported | 0.001 ± 0.000 | - | - | - |
| cpu/view_metadata | `broadcast_in_dim_view` | f64 | 4 | `8192x1 -> 8192x4096` | 0.002 ± 0.000 | unsupported | 0.001 ± 0.000 | - | - | - |
| cpu/view_metadata | `reshape_view` | f64 | 1 | `33554432 -> 8192x4096` | 6.678 ± 0.394 | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `reshape_view` | f64 | 4 | `33554432 -> 8192x4096` | 7.739 ± 0.140 | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `slice_view` | f64 | 1 | `4194304 -> 2096128` | 0.681 ± 0.010 | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `slice_view` | f64 | 4 | `4194304 -> 2096128` | 0.718 ± 0.030 | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `transpose_view` | f64 | 1 | `4096x4096` | 2.878 ± 0.049 | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `transpose_view` | f64 | 4 | `4096x4096` | 2.958 ± 0.142 | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |

## Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `jax-cpu` is 15.5x faster than the slowest successful cell (`pytorch-cpu`, 9.162 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 22.6x faster than the slowest successful cell (`pytorch-cpu`, 9.162 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 10.3x faster than the slowest successful cell (`tenferro-trace`, 6.188 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 15.0x faster than the slowest successful cell (`tenferro-trace`, 6.188 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/tanh` (f64, threads=1, shape=`8388608`): `jax-cpu` is 15.4x faster than the slowest successful cell (`pytorch-cpu`, 42.878 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/tanh` (f64, threads=4, shape=`8388608`): `jax-cpu` is 10.3x faster than the slowest successful cell (`julia-base`, 29.803 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/gather` (f64, threads=1, shape=`262144`): `jax-cpu` is 29.5x faster than the slowest successful cell (`tenferro-eager`, 3.346 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/gather` (f64, threads=1, shape=`262144`): `julia-base` is 11.9x faster than the slowest successful cell (`tenferro-eager`, 3.346 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/gather` (f64, threads=1, shape=`262144`): `pytorch-cpu` is 18.6x faster than the slowest successful cell (`tenferro-eager`, 3.346 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/gather` (f64, threads=4, shape=`262144`): `jax-cpu` is 13.9x faster than the slowest successful cell (`tenferro-eager`, 1.101 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/scatter` (f64, threads=1, shape=`262144`): `jax-cpu` is 10.9x faster than the slowest successful cell (`tenferro-trace`, 2.699 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/scatter` (f64, threads=4, shape=`262144`): `jax-cpu` is 11.0x faster than the slowest successful cell (`tenferro-trace`, 2.701 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=1, shape=`33554432 -> 8192x4096`): `pytorch-cpu` is 10684.0x faster than the slowest successful cell (`tenferro-eager`, 6.678 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=4, shape=`33554432 -> 8192x4096`): `pytorch-cpu` is 12382.2x faster than the slowest successful cell (`tenferro-eager`, 7.739 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=1, shape=`4194304 -> 2096128`): `pytorch-cpu` is 1257.1x faster than the slowest successful cell (`tenferro-eager`, 0.681 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=4, shape=`4194304 -> 2096128`): `pytorch-cpu` is 1229.4x faster than the slowest successful cell (`tenferro-eager`, 0.718 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=1, shape=`4096x4096`): `pytorch-cpu` is 4315.3x faster than the slowest successful cell (`tenferro-eager`, 2.878 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=4, shape=`4096x4096`): `pytorch-cpu` is 4177.6x faster than the slowest successful cell (`tenferro-eager`, 2.958 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

## Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.
