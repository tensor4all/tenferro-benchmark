# CPU Public API Benchmark Results

- Suite: `cpu/public_api`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/public_api.yaml`
- Public API coverage manifest: `benchmarks/cpu/public_api_coverage.yaml`
- Run metadata: `data/results/mac-cpu/cpu/public_api/20260916_165807/run.yaml`
- Timestamp: `20260916_165807`

Latest run: `PUBLICATION_GATE_PROFILE=full ./scripts/run_cpu_public_api.sh 1 4`.

- Sampling: `full` profile, 15 measured runs, 3 warmups per row

This file is generated from sequential CPU public API runs under `data/results/mac-cpu/cpu/public_api/20260916_165807`.

- tenferro-rs commit: `d8759f4320a337d2399f4a87dfec55af51d2ebf1`

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

## Thread Environments

### Threads: 1

- Run metadata: `data/results/mac-cpu/cpu/public_api/20260916_165807/run_t1.yaml`
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

- Run metadata: `data/results/mac-cpu/cpu/public_api/20260916_165807/run_t4.yaml`
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

- CSV: `data/results/mac-cpu/cpu/public_api/20260916_165807/cpu_public_api_t1_20260916_165807.csv`
- CSV: `data/results/mac-cpu/cpu/public_api/20260916_165807/cpu_public_api_t4_20260916_165807.csv`
- Source table: `data/results/mac-cpu/cpu/public_api/20260916_165807/cpu_public_api_20260916_165807.md`

## CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| cpu/complex | `cholesky` | c64 | 1 | `448x448` | 2.522 ± 0.094 | - | 2.500 ± 0.075 | 2.851 ± 0.097 | 0.979 ± 0.035 | 3.113 ± 0.291 | - |
| cpu/complex | `cholesky` | c64 | 4 | `448x448` | 2.477 ± 0.041 | - | 2.538 ± 0.062 | 2.502 ± 0.029 | 0.775 ± 0.033 | 1.314 ± 0.373 | - |
| cpu/complex | `conj` | c64 | 1 | `16777216` | 11.805 ± 0.122 | - | 11.834 ± 0.470 | 4.656 ± 0.073 | 4.523 ± 0.079 | 4.632 ± 0.540 | 4.562 ± 0.068 |
| cpu/complex | `conj` | c64 | 4 | `16777216` | 3.014 ± 0.105 | - | 2.968 ± 0.108 | 1.632 ± 0.089 | 1.682 ± 0.040 | 4.588 ± 0.166 | 5.604 ± 7.617 |
| cpu/complex | `div` | c64 | 1 | `8388608` | 4.539 ± 0.052 | - | 4.599 ± 0.053 | 8.109 ± 0.074 | 23.542 ± 0.113 | 18.797 ± 0.080 | 18.323 ± 0.078 |
| cpu/complex | `div` | c64 | 4 | `8388608` | 1.596 ± 0.052 | - | 1.621 ± 0.064 | 2.857 ± 0.059 | 7.838 ± 0.031 | 19.975 ± 1.115 | 7.008 ± 5.607 |
| cpu/complex | `dot_general` | c64 | 1 | `640x640` | 6.040 ± 1.016 | - | 6.041 ± 0.188 | 5.934 ± 0.273 | 37.094 ± 0.095 | 42.425 ± 1.311 | - |
| cpu/complex | `dot_general` | c64 | 4 | `640x640` | 5.916 ± 0.728 | - | 6.016 ± 0.223 | 5.724 ± 0.223 | 9.774 ± 0.255 | 11.042 ± 0.518 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 1 | `640x640` | 6.145 ± 0.126 | - | unsupported | 6.100 ± 0.196 | 37.761 ± 0.603 | 41.122 ± 1.957 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 4 | `640x640` | 5.969 ± 0.153 | - | unsupported | 5.937 ± 0.228 | 10.157 ± 0.147 | 11.277 ± 0.174 | - |
| cpu/complex | `eig` | c64 | 1 | `112x112` | 5.961 ± 0.034 | - | 5.943 ± 0.054 | 5.907 ± 0.050 | 10.281 ± 0.889 | 5.333 ± 0.131 | - |
| cpu/complex | `eig` | c64 | 4 | `112x112` | 5.940 ± 0.083 | - | 5.986 ± 0.055 | 5.917 ± 0.082 | 10.084 ± 0.086 | 4.998 ± 0.118 | - |
| cpu/complex | `exp` | c64 | 1 | `4194304` | 21.046 ± 0.201 | - | 21.053 ± 0.086 | 20.432 ± 0.311 | 24.210 ± 0.293 | 25.825 ± 2.637 | 24.814 ± 0.060 |
| cpu/complex | `exp` | c64 | 4 | `4194304` | 5.549 ± 0.693 | - | 5.405 ± 0.117 | 5.992 ± 0.044 | 6.656 ± 0.087 | 26.347 ± 1.544 | 6.872 ± 2.342 |
| cpu/complex | `log` | c64 | 1 | `4194304` | 24.154 ± 1.235 | - | 23.750 ± 0.086 | 25.866 ± 0.154 | 50.182 ± 0.292 | 35.656 ± 0.094 | 34.702 ± 0.124 |
| cpu/complex | `log` | c64 | 4 | `4194304` | 6.830 ± 0.128 | - | 6.786 ± 0.169 | 7.269 ± 0.201 | 14.079 ± 0.322 | 36.000 ± 0.766 | 10.070 ± 1.857 |
| cpu/complex | `mul` | c64 | 1 | `8388608` | 4.169 ± 0.362 | - | 4.041 ± 0.152 | 4.324 ± 0.085 | 3.705 ± 0.212 | 3.744 ± 0.094 | 3.533 ± 0.107 |
| cpu/complex | `mul` | c64 | 4 | `8388608` | 1.462 ± 0.111 | - | 1.863 ± 0.069 | 1.495 ± 0.054 | 1.388 ± 0.123 | 3.754 ± 0.143 | 3.877 ± 8.702 |
| cpu/complex | `norm_fro` | c64 | 1 | `2048x1536` | 4.787 ± 0.390 | - | 4.009 ± 0.253 | 3.697 ± 0.040 | 2.102 ± 0.118 | 5.896 ± 0.075 | - |
| cpu/complex | `norm_fro` | c64 | 4 | `2048x1536` | 2.265 ± 0.069 | - | 1.435 ± 0.046 | 0.937 ± 0.024 | 0.729 ± 0.031 | 5.946 ± 0.082 | - |
| cpu/complex | `qr` | c64 | 1 | `256x256` | 4.732 ± 0.062 | - | 4.875 ± 0.097 | 5.057 ± 0.046 | 3.695 ± 0.050 | 6.170 ± 0.122 | - |
| cpu/complex | `qr` | c64 | 4 | `256x256` | 4.805 ± 0.061 | - | 4.844 ± 0.041 | 4.890 ± 0.035 | 3.632 ± 0.071 | 2.842 ± 0.270 | - |
| cpu/complex | `solve` | c64 | 1 | `384x384,rhs=8` | 3.348 ± 0.085 | - | 3.381 ± 0.083 | 3.220 ± 0.104 | 1.457 ± 0.081 | 3.776 ± 0.050 | - |
| cpu/complex | `solve` | c64 | 4 | `384x384,rhs=8` | 3.541 ± 0.117 | - | 3.586 ± 0.078 | 3.379 ± 0.098 | 1.336 ± 0.049 | 1.530 ± 0.249 | - |
| cpu/complex | `svd` | c64 | 1 | `160x160` | 3.909 ± 0.060 | - | 3.908 ± 0.147 | 3.884 ± 0.039 | 2.791 ± 0.046 | 4.460 ± 0.103 | - |
| cpu/complex | `svd` | c64 | 4 | `160x160` | 3.900 ± 0.065 | - | 4.042 ± 0.060 | 3.905 ± 0.065 | 2.773 ± 0.053 | 2.908 ± 0.287 | - |
| cpu/complex | `tensordot` | c64 | 1 | `640x640` | 5.769 ± 0.294 | - | 6.017 ± 0.171 | 5.815 ± 0.151 | 37.151 ± 0.399 | 40.996 ± 0.362 | - |
| cpu/complex | `tensordot` | c64 | 4 | `640x640` | 5.703 ± 0.139 | - | 6.045 ± 0.178 | 5.763 ± 0.213 | 9.949 ± 0.238 | 10.999 ± 0.240 | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 1 | `1024x1024` | 5.847 ± 0.713 | - | unsupported | 4.790 ± 0.360 | 33.232 ± 0.200 | - | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 4 | `1024x1024` | 5.238 ± 0.784 | - | unsupported | 4.842 ± 0.306 | 9.229 ± 0.325 | - | - |
| cpu/elementwise_reduction | `abs` | f64 | 1 | `33554432` | 4.403 ± 0.061 | - | 4.403 ± 0.070 | 4.451 ± 0.107 | 9.333 ± 0.278 | 4.520 ± 0.103 | 8.005 ± 0.084 |
| cpu/elementwise_reduction | `abs` | f64 | 4 | `33554432` | 1.731 ± 0.131 | - | 1.853 ± 0.967 | 1.669 ± 0.065 | 2.546 ± 0.041 | 4.510 ± 0.121 | 2.857 ± 4.176 |
| cpu/elementwise_reduction | `add` | f64 | 1 | `33554432` | 6.657 ± 0.258 | - | 6.633 ± 0.130 | 6.687 ± 0.108 | 11.922 ± 0.641 | 7.202 ± 0.529 | 8.724 ± 0.302 |
| cpu/elementwise_reduction | `add` | f64 | 4 | `33554432` | 2.763 ± 0.733 | - | 2.712 ± 0.253 | 2.654 ± 0.097 | 3.570 ± 0.199 | 6.913 ± 0.315 | 6.548 ± 8.162 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 1 | `4194304` | 16.806 ± 0.078 | - | 16.874 ± 0.114 | 18.047 ± 0.096 | 18.287 ± 0.170 | 21.499 ± 0.287 | 21.661 ± 0.394 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 4 | `4194304` | 5.115 ± 0.283 | - | 5.233 ± 0.226 | 5.351 ± 0.262 | 4.965 ± 0.285 | 21.360 ± 0.404 | 5.790 ± 0.479 |
| cpu/elementwise_reduction | `clamp` | f64 | 1 | `8388608` | 2.817 ± 0.032 | - | 2.869 ± 0.076 | 2.271 ± 0.081 | 3.816 ± 0.040 | 2.254 ± 0.118 | 3.115 ± 0.107 |
| cpu/elementwise_reduction | `clamp` | f64 | 4 | `8388608` | 1.620 ± 0.094 | - | 1.520 ± 0.183 | 0.907 ± 0.070 | 1.806 ± 0.087 | 2.303 ± 0.226 | 1.712 ± 1.666 |
| cpu/elementwise_reduction | `compare_lt` | f64 | 1 | `33554432` | 5.213 ± 0.072 | - | 5.211 ± 0.040 | 8.827 ± 0.507 | 8.798 ± 0.041 | 5.960 ± 0.131 | - |
| cpu/elementwise_reduction | `compare_lt` | f64 | 4 | `33554432` | 2.225 ± 0.179 | - | 2.207 ± 0.106 | 2.947 ± 0.082 | 2.738 ± 0.111 | 5.797 ± 0.109 | - |
| cpu/elementwise_reduction | `cos` | f64 | 1 | `8388608` | 16.502 ± 0.493 | - | 17.012 ± 0.572 | 17.173 ± 0.069 | 12.471 ± 0.326 | 20.928 ± 0.313 | 21.040 ± 0.223 |
| cpu/elementwise_reduction | `cos` | f64 | 4 | `8388608` | 4.494 ± 0.224 | - | 4.398 ± 0.123 | 4.773 ± 0.402 | 3.395 ± 0.142 | 20.989 ± 1.988 | 5.562 ± 0.323 |
| cpu/elementwise_reduction | `div` | f64 | 1 | `33554432` | 6.623 ± 0.078 | - | 6.577 ± 0.111 | 6.728 ± 0.119 | 11.849 ± 0.105 | 6.772 ± 0.148 | 8.855 ± 0.255 |
| cpu/elementwise_reduction | `div` | f64 | 4 | `33554432` | 2.719 ± 0.131 | - | 2.788 ± 0.272 | 2.634 ± 0.078 | 3.582 ± 0.228 | 6.720 ± 0.155 | 3.500 ± 2.420 |
| cpu/elementwise_reduction | `exp` | f64 | 1 | `8388608` | 16.973 ± 0.152 | - | 16.945 ± 0.105 | 12.333 ± 0.095 | 10.468 ± 0.071 | 17.027 ± 0.184 | 16.843 ± 0.928 |
| cpu/elementwise_reduction | `exp` | f64 | 4 | `8388608` | 4.423 ± 0.242 | - | 4.524 ± 0.184 | 4.424 ± 0.054 | 2.873 ± 0.215 | 16.910 ± 0.458 | 6.012 ± 2.864 |
| cpu/elementwise_reduction | `expm1` | f64 | 1 | `4194304` | 8.569 ± 0.136 | - | 8.468 ± 0.518 | 14.779 ± 0.046 | 11.318 ± 0.183 | 13.474 ± 0.540 | 12.953 ± 0.197 |
| cpu/elementwise_reduction | `expm1` | f64 | 4 | `4194304` | 2.182 ± 0.050 | - | 2.221 ± 0.064 | 4.075 ± 0.345 | 3.113 ± 0.408 | 12.936 ± 0.565 | 3.510 ± 0.954 |
| cpu/elementwise_reduction | `log` | f64 | 1 | `8388608` | 15.311 ± 0.071 | - | 15.345 ± 0.259 | 21.225 ± 0.122 | 14.315 ± 0.150 | 22.679 ± 0.160 | 22.705 ± 0.170 |
| cpu/elementwise_reduction | `log` | f64 | 4 | `8388608` | 4.549 ± 0.184 | - | 4.598 ± 0.239 | 5.741 ± 1.185 | 3.948 ± 0.359 | 22.734 ± 0.224 | 6.354 ± 0.578 |
| cpu/elementwise_reduction | `log1p` | f64 | 1 | `4194304` | 10.319 ± 0.055 | - | 10.316 ± 0.050 | 11.052 ± 0.060 | 10.179 ± 0.116 | 12.553 ± 0.450 | 12.353 ± 0.120 |
| cpu/elementwise_reduction | `log1p` | f64 | 4 | `4194304` | 2.777 ± 0.178 | - | 2.833 ± 0.124 | 3.045 ± 0.210 | 2.798 ± 0.320 | 12.305 ± 0.171 | 3.366 ± 0.457 |
| cpu/elementwise_reduction | `maximum` | f64 | 1 | `33554432` | 6.606 ± 0.045 | - | 6.602 ± 0.074 | 6.576 ± 0.050 | 11.615 ± 0.109 | 6.631 ± 0.154 | 8.602 ± 0.279 |
| cpu/elementwise_reduction | `maximum` | f64 | 4 | `33554432` | 2.778 ± 0.180 | - | 2.800 ± 0.205 | 2.773 ± 0.143 | 3.476 ± 0.093 | 6.641 ± 0.132 | 3.133 ± 2.735 |
| cpu/elementwise_reduction | `minimum` | f64 | 1 | `33554432` | 6.623 ± 0.035 | - | 6.641 ± 0.060 | 6.556 ± 0.070 | 11.548 ± 0.147 | 6.634 ± 0.124 | 8.839 ± 0.274 |
| cpu/elementwise_reduction | `minimum` | f64 | 4 | `33554432` | 2.787 ± 0.181 | - | 2.940 ± 0.339 | 2.698 ± 0.161 | 3.460 ± 0.055 | 6.658 ± 0.080 | 3.217 ± 2.093 |
| cpu/elementwise_reduction | `mul` | f64 | 1 | `33554432` | 6.695 ± 0.046 | - | 6.876 ± 0.080 | 6.559 ± 0.035 | 11.983 ± 0.766 | 6.696 ± 0.089 | 8.627 ± 0.263 |
| cpu/elementwise_reduction | `mul` | f64 | 4 | `33554432` | 2.843 ± 0.095 | - | 2.845 ± 0.079 | 2.649 ± 0.066 | 3.648 ± 0.221 | 6.772 ± 0.070 | 3.510 ± 2.251 |
| cpu/elementwise_reduction | `neg` | f64 | 1 | `33554432` | 4.413 ± 0.074 | - | 4.394 ± 0.079 | 4.466 ± 0.085 | 8.757 ± 0.960 | 4.468 ± 0.142 | 8.168 ± 0.143 |
| cpu/elementwise_reduction | `neg` | f64 | 4 | `33554432` | 1.683 ± 0.094 | - | 1.649 ± 0.129 | 1.655 ± 0.089 | 2.501 ± 0.114 | 4.554 ± 0.162 | 3.383 ± 2.241 |
| cpu/elementwise_reduction | `pow` | f64 | 1 | `4194304` | 17.860 ± 0.134 | - | 18.049 ± 0.178 | 38.473 ± 0.382 | 17.776 ± 0.057 | 32.173 ± 0.965 | 32.433 ± 0.815 |
| cpu/elementwise_reduction | `pow` | f64 | 4 | `4194304` | 5.045 ± 0.238 | - | 5.101 ± 0.192 | 10.249 ± 0.316 | 5.095 ± 0.230 | 32.438 ± 1.447 | 8.857 ± 0.352 |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 1 | `2048x2048` | 5.972 ± 0.457 | - | 5.867 ± 0.131 | 10.551 ± 0.376 | 1.230 ± 0.162 | 0.375 ± 0.014 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 4 | `2048x2048` | 5.886 ± 0.297 | - | 5.847 ± 0.063 | 4.818 ± 0.390 | 0.451 ± 0.054 | 0.373 ± 0.008 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 1 | `4096x4096` | 3.199 ± 0.135 | - | 3.185 ± 0.132 | 4.002 ± 0.157 | 1.728 ± 0.107 | 1.949 ± 0.051 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 4 | `4096x4096` | 3.170 ± 0.054 | - | 3.253 ± 0.055 | 1.067 ± 0.038 | 0.679 ± 0.043 | 1.933 ± 0.042 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 1 | `8192x4096` | 3.756 ± 0.323 | - | 3.937 ± 0.354 | 3.211 ± 0.044 | 18.006 ± 0.329 | 3.089 ± 0.088 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 4 | `8192x4096` | 6.349 ± 0.121 | - | 6.175 ± 0.353 | 1.026 ± 0.039 | 5.484 ± 0.092 | 3.118 ± 0.092 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 1 | `8192x4096` | 3.257 ± 0.099 | - | 3.512 ± 0.296 | 2.933 ± 0.089 | 13.913 ± 0.404 | 3.064 ± 0.116 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 4 | `8192x4096` | 4.732 ± 0.328 | - | 4.972 ± 0.208 | 1.025 ± 0.062 | 4.592 ± 0.060 | 3.132 ± 0.133 | - |
| cpu/elementwise_reduction | `rem` | f64 | 1 | `8388608` | 15.388 ± 1.826 | - | 16.033 ± 0.508 | 19.654 ± 0.213 | 15.265 ± 0.118 | 14.505 ± 2.879 | 13.398 ± 0.102 |
| cpu/elementwise_reduction | `rem` | f64 | 4 | `8388608` | 4.621 ± 0.417 | - | 5.357 ± 1.096 | 6.442 ± 0.131 | 4.417 ± 0.126 | 13.502 ± 0.835 | 3.739 ± 1.640 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 1 | `33554432` | 11.568 ± 0.088 | - | 11.644 ± 0.626 | 11.871 ± 0.561 | 23.942 ± 0.268 | 23.572 ± 0.355 | 23.924 ± 0.665 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 4 | `33554432` | 3.146 ± 0.037 | - | 3.341 ± 0.090 | 3.259 ± 0.177 | 6.176 ± 0.064 | 23.844 ± 0.354 | 6.312 ± 2.582 |
| cpu/elementwise_reduction | `select` | f64 | 1 | `33554432` | 7.882 ± 0.034 | - | 7.895 ± 0.055 | 9.211 ± 0.250 | 14.488 ± 0.182 | 7.943 ± 0.071 | - |
| cpu/elementwise_reduction | `select` | f64 | 4 | `33554432` | 3.612 ± 0.172 | - | 3.699 ± 0.391 | 3.631 ± 0.137 | 4.971 ± 1.418 | 7.961 ± 0.082 | - |
| cpu/elementwise_reduction | `sign` | f64 | 1 | `33554432` | 5.128 ± 0.069 | - | 5.117 ± 0.042 | 5.107 ± 0.097 | 10.039 ± 0.124 | 4.523 ± 0.072 | 12.042 ± 0.247 |
| cpu/elementwise_reduction | `sign` | f64 | 4 | `33554432` | 3.324 ± 0.057 | - | 3.259 ± 0.156 | 1.753 ± 0.073 | 3.347 ± 0.065 | 4.522 ± 0.407 | 3.459 ± 2.802 |
| cpu/elementwise_reduction | `sin` | f64 | 1 | `8388608` | 14.740 ± 0.290 | - | 14.768 ± 0.198 | 14.368 ± 0.055 | 11.275 ± 0.097 | 17.488 ± 0.133 | 17.554 ± 0.140 |
| cpu/elementwise_reduction | `sin` | f64 | 4 | `8388608` | 4.334 ± 0.349 | - | 4.401 ± 0.364 | 4.925 ± 1.306 | 3.394 ± 0.423 | 17.219 ± 0.243 | 4.579 ± 0.289 |
| cpu/elementwise_reduction | `sqrt` | f64 | 1 | `33554432` | 7.646 ± 0.211 | - | 7.619 ± 0.222 | 7.774 ± 0.193 | 8.751 ± 0.894 | 15.956 ± 0.230 | 15.838 ± 0.152 |
| cpu/elementwise_reduction | `sqrt` | f64 | 4 | `33554432` | 2.125 ± 0.117 | - | 2.093 ± 0.149 | 2.821 ± 0.117 | 2.493 ± 0.100 | 15.839 ± 0.188 | 4.191 ± 2.700 |
| cpu/elementwise_reduction | `sub` | f64 | 1 | `33554432` | 6.515 ± 0.100 | - | 6.631 ± 0.140 | 6.604 ± 0.047 | 12.187 ± 1.262 | 6.615 ± 0.091 | 8.915 ± 0.326 |
| cpu/elementwise_reduction | `sub` | f64 | 4 | `33554432` | 2.664 ± 0.190 | - | 2.826 ± 0.243 | 2.614 ± 0.114 | 3.544 ± 0.158 | 6.884 ± 0.200 | 3.370 ± 2.235 |
| cpu/elementwise_reduction | `tanh` | f64 | 1 | `8388608` | 21.612 ± 0.444 | - | 21.629 ± 0.372 | 36.506 ± 0.628 | 8.504 ± 0.044 | 23.561 ± 0.170 | 23.570 ± 0.199 |
| cpu/elementwise_reduction | `tanh` | f64 | 4 | `8388608` | 5.617 ± 0.210 | - | 5.580 ± 0.120 | 9.880 ± 0.288 | 2.335 ± 0.097 | 23.837 ± 0.456 | 6.522 ± 0.741 |
| cpu/indexing_layout | `concatenate` | f64 | 1 | `1048576+1048576` | 0.278 ± 0.012 | - | 0.586 ± 0.049 | 0.234 ± 0.004 | 0.228 ± 0.009 | 0.281 ± 0.009 | - |
| cpu/indexing_layout | `concatenate` | f64 | 4 | `1048576+1048576` | 0.294 ± 0.018 | - | 0.470 ± 0.050 | 0.116 ± 0.009 | 0.234 ± 0.017 | 0.291 ± 0.006 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 1 | `4194304 -> 2097152` | 0.250 ± 0.020 | - | 0.803 ± 0.042 | 0.277 ± 0.005 | 0.279 ± 0.018 | 0.287 ± 0.044 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 4 | `4194304 -> 2097152` | 0.253 ± 0.011 | - | 0.616 ± 0.025 | 0.093 ± 0.024 | 0.114 ± 0.014 | 0.293 ± 0.017 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 1 | `2097152` | 0.403 ± 0.055 | - | unsupported | 0.417 ± 0.010 | 0.374 ± 0.044 | 0.422 ± 0.149 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 4 | `2097152` | 0.401 ± 0.013 | - | unsupported | 0.160 ± 0.037 | 0.309 ± 0.012 | 0.410 ± 0.059 | - |
| cpu/indexing_layout | `gather` | f64 | 1 | `262144` | 0.279 ± 0.018 | - | 0.374 ± 0.031 | 0.168 ± 0.006 | 0.186 ± 0.022 | 0.294 ± 0.015 | - |
| cpu/indexing_layout | `gather` | f64 | 4 | `262144` | 0.242 ± 0.109 | - | 0.291 ± 0.062 | 0.167 ± 0.005 | 0.080 ± 0.013 | 0.273 ± 0.118 | - |
| cpu/indexing_layout | `pad` | f64 | 1 | `2097152` | 0.361 ± 0.020 | - | 0.644 ± 0.036 | 0.407 ± 0.015 | 0.292 ± 0.014 | - | - |
| cpu/indexing_layout | `pad` | f64 | 4 | `2097152` | 0.377 ± 0.023 | - | 0.520 ± 0.032 | 0.169 ± 0.002 | 0.131 ± 0.012 | - | - |
| cpu/indexing_layout | `reverse` | f64 | 1 | `2097152` | 0.511 ± 0.020 | - | 0.788 ± 0.046 | 0.298 ± 0.026 | 0.282 ± 0.010 | 0.574 ± 0.077 | - |
| cpu/indexing_layout | `reverse` | f64 | 4 | `2097152` | 0.511 ± 0.010 | - | 0.689 ± 0.219 | 0.108 ± 0.021 | 0.111 ± 0.012 | 0.653 ± 0.316 | - |
| cpu/indexing_layout | `scatter` | f64 | 1 | `262144` | 0.408 ± 0.037 | - | 0.560 ± 0.122 | 0.237 ± 0.021 | 0.243 ± 0.025 | 0.225 ± 0.037 | - |
| cpu/indexing_layout | `scatter` | f64 | 4 | `262144` | 0.483 ± 0.041 | - | 0.542 ± 0.019 | 0.307 ± 0.028 | 0.216 ± 0.006 | 0.320 ± 0.154 | - |
| cpu/indexing_layout | `slice` | f64 | 1 | `4194304 -> 2096128` | 0.647 ± 0.020 | - | 1.234 ± 0.100 | 0.445 ± 0.021 | 0.428 ± 0.017 | 0.747 ± 0.124 | - |
| cpu/indexing_layout | `slice` | f64 | 4 | `4194304 -> 2096128` | 0.651 ± 0.272 | - | 0.951 ± 0.071 | 0.177 ± 0.035 | 0.287 ± 0.021 | 0.727 ± 0.049 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 1 | `1536x1536` | 5.070 ± 0.565 | - | 5.047 ± 0.233 | 15.698 ± 0.756 | 9.383 ± 0.235 | 26.348 ± 3.506 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 4 | `1536x1536` | 4.767 ± 0.621 | - | 4.745 ± 0.141 | 7.961 ± 0.299 | 6.142 ± 0.276 | 11.186 ± 12.903 | - |
| cpu/linalg_uncovered | `det` | f64 | 1 | `1024x1024` | 4.572 ± 0.379 | - | 4.992 ± 0.216 | 4.198 ± 0.208 | 4.672 ± 0.300 | 13.937 ± 0.442 | - |
| cpu/linalg_uncovered | `det` | f64 | 4 | `1024x1024` | 4.412 ± 0.278 | - | 4.620 ± 0.168 | 4.194 ± 0.272 | 4.357 ± 4.625 | 5.387 ± 0.633 | - |
| cpu/linalg_uncovered | `eig` | f64 | 1 | `160x160` | 4.449 ± 0.089 | - | 4.417 ± 0.098 | 4.507 ± 0.052 | 5.458 ± 0.081 | 4.083 ± 0.093 | - |
| cpu/linalg_uncovered | `eig` | f64 | 4 | `160x160` | 4.425 ± 0.058 | - | 4.461 ± 0.050 | 4.490 ± 0.091 | 5.470 ± 0.068 | 4.568 ± 0.243 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 1 | `192x192` | 4.672 ± 0.048 | - | 4.681 ± 0.065 | 4.755 ± 0.148 | 4.298 ± 0.056 | 4.483 ± 0.126 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 4 | `192x192` | 4.623 ± 0.032 | - | 4.693 ± 0.036 | 4.736 ± 0.165 | 4.444 ± 0.052 | 4.675 ± 0.183 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 1 | `512x512` | 14.858 ± 0.112 | - | 14.806 ± 0.250 | 14.502 ± 0.531 | 12.865 ± 0.182 | 11.124 ± 0.500 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 4 | `512x512` | 14.765 ± 0.115 | - | 15.085 ± 0.124 | 14.326 ± 0.122 | 12.640 ± 0.160 | 7.203 ± 0.746 | - |
| cpu/linalg_uncovered | `inv` | f64 | 1 | `768x768` | 7.368 ± 0.158 | - | 6.956 ± 0.167 | 5.771 ± 0.121 | 5.920 ± 0.180 | 19.475 ± 0.172 | - |
| cpu/linalg_uncovered | `inv` | f64 | 4 | `768x768` | 7.259 ± 0.120 | - | 6.835 ± 0.101 | 5.497 ± 0.427 | 5.265 ± 0.115 | 6.943 ± 0.544 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 1 | `768x384,rhs=16` | - | 5.895 ± 0.095 | 5.619 ± 0.048 | 3.265 ± 0.036 | 13.969 ± 0.093 | 5.284 ± 0.065 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 4 | `768x384,rhs=16` | - | 5.833 ± 0.064 | 5.688 ± 0.099 | 3.193 ± 0.032 | 13.369 ± 0.077 | 3.236 ± 0.234 | - |
| cpu/linalg_uncovered | `lu` | f64 | 1 | `1024x1024` | 4.434 ± 0.162 | - | 4.623 ± 0.107 | 6.310 ± 0.111 | 7.156 ± 0.251 | 16.157 ± 3.000 | - |
| cpu/linalg_uncovered | `lu` | f64 | 4 | `1024x1024` | 4.279 ± 0.167 | - | 4.553 ± 0.143 | 5.640 ± 0.250 | 5.274 ± 0.253 | 7.514 ± 3.187 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 1 | `2048x2048` | 0.393 ± 0.020 | - | 0.393 ± 0.018 | 1.301 ± 0.061 | 2.936 ± 0.247 | 3.914 ± 0.095 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 4 | `2048x2048` | 0.573 ± 0.037 | - | 0.598 ± 0.064 | 1.285 ± 0.021 | 0.827 ± 0.046 | 3.997 ± 0.103 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 1 | `512x256` | 6.815 ± 0.137 | - | 6.769 ± 0.126 | 6.516 ± 0.078 | 7.496 ± 0.092 | 13.030 ± 0.220 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 4 | `512x256` | 6.666 ± 0.044 | - | 6.896 ± 0.096 | 6.593 ± 0.115 | 6.624 ± 0.075 | 9.902 ± 0.231 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 1 | `512x256` | 6.754 ± 0.072 | - | 6.757 ± 0.095 | 6.497 ± 0.072 | 7.509 ± 0.135 | 13.109 ± 0.167 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 4 | `512x256` | 6.755 ± 0.128 | - | 6.865 ± 0.062 | 6.560 ± 0.120 | 6.607 ± 0.088 | 9.969 ± 0.419 | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 1 | `1024x1024` | 4.446 ± 0.278 | - | 4.813 ± 0.207 | 4.255 ± 0.219 | 4.854 ± 0.285 | 13.807 ± 0.218 | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 4 | `1024x1024` | 4.283 ± 0.126 | - | 4.641 ± 0.143 | 4.247 ± 0.480 | 3.898 ± 0.154 | 5.411 ± 0.324 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 1 | `768x384` | - | unsupported | unsupported | 19.162 ± 0.300 | 19.346 ± 0.093 | 45.759 ± 1.246 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 4 | `768x384` | - | unsupported | unsupported | 19.112 ± 0.202 | 19.097 ± 0.172 | 30.913 ± 0.862 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 1 | `4096x4096,rhs=64` | 5.383 ± 0.224 | - | 8.220 ± 0.471 | 3.996 ± 0.099 | 26.016 ± 0.847 | 20.811 ± 0.403 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 4 | `4096x4096,rhs=64` | 5.440 ± 0.171 | - | 6.961 ± 0.281 | 3.893 ± 0.150 | 11.327 ± 1.072 | 7.968 ± 0.609 | - |
| cpu/output_reuse | `add_into` | f64 | 1 | `33554432` | 17.013 ± 0.191 | - | unsupported | 6.628 ± 0.044 | - | 6.622 ± 0.172 | 8.590 ± 0.234 |
| cpu/output_reuse | `add_into` | f64 | 4 | `33554432` | 6.010 ± 0.478 | - | unsupported | 2.683 ± 0.053 | - | 6.689 ± 0.077 | 3.158 ± 0.300 |
| cpu/output_reuse | `conj_into` | c64 | 1 | `16777216` | 8.044 ± 0.038 | - | unsupported | 4.571 ± 0.066 | - | 4.523 ± 0.065 | 4.447 ± 0.104 |
| cpu/output_reuse | `conj_into` | c64 | 4 | `16777216` | 1.880 ± 0.090 | - | unsupported | 1.632 ± 0.069 | - | 4.497 ± 0.037 | 1.746 ± 0.129 |
| cpu/output_reuse | `copy_read_into` | f64 | 1 | `33554432` | 3.873 ± 0.059 | - | unsupported | 4.390 ± 0.111 | - | 3.840 ± 0.081 | - |
| cpu/output_reuse | `copy_read_into` | f64 | 4 | `33554432` | 1.647 ± 0.098 | - | unsupported | 1.614 ± 0.056 | - | 3.946 ± 0.053 | - |
| cpu/output_reuse | `div_into` | f64 | 1 | `33554432` | 23.567 ± 0.196 | - | unsupported | 6.633 ± 0.030 | - | 6.509 ± 0.118 | 8.723 ± 0.279 |
| cpu/output_reuse | `div_into` | f64 | 4 | `33554432` | 5.037 ± 0.412 | - | unsupported | 2.603 ± 0.074 | - | 6.752 ± 0.049 | 3.141 ± 0.429 |
| cpu/output_reuse | `dot_general_read_into` | f64 | 1 | `1024x1024` | 4.952 ± 0.321 | - | unsupported | 4.581 ± 0.219 | - | 36.725 ± 0.528 | - |
| cpu/output_reuse | `dot_general_read_into` | f64 | 4 | `1024x1024` | 4.762 ± 0.321 | - | unsupported | 4.718 ± 0.454 | - | 10.340 ± 0.311 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 1 | `1024x1024` | 5.431 ± 0.281 | - | unsupported | 5.221 ± 0.217 | - | 36.664 ± 0.256 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 4 | `1024x1024` | 6.189 ± 1.347 | - | unsupported | 7.002 ± 1.637 | - | 10.037 ± 0.216 | - |
| cpu/output_reuse | `mul_into` | f64 | 1 | `33554432` | 16.936 ± 0.099 | - | unsupported | 6.709 ± 0.093 | - | 6.765 ± 0.286 | 8.477 ± 0.430 |
| cpu/output_reuse | `mul_into` | f64 | 4 | `33554432` | 5.043 ± 0.441 | - | unsupported | 2.626 ± 0.118 | - | 6.559 ± 0.040 | 3.109 ± 0.224 |
| cpu/output_reuse | `neg_into` | f64 | 1 | `33554432` | 11.473 ± 2.684 | - | unsupported | 4.472 ± 0.102 | - | 4.422 ± 0.072 | 7.941 ± 0.102 |
| cpu/output_reuse | `neg_into` | f64 | 4 | `33554432` | 2.887 ± 0.171 | - | unsupported | 1.610 ± 0.084 | - | 4.491 ± 0.071 | 2.863 ± 0.308 |
| cpu/output_reuse | `sub_into` | f64 | 1 | `33554432` | 17.027 ± 0.400 | - | unsupported | 6.622 ± 0.104 | - | 6.524 ± 0.073 | 8.619 ± 0.201 |
| cpu/output_reuse | `sub_into` | f64 | 4 | `33554432` | 5.628 ± 0.694 | - | unsupported | 2.663 ± 0.131 | - | 6.735 ± 0.058 | 3.087 ± 0.275 |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 1 | `8192x1 -> 8192x4096` | 2.036 ± 0.046 | - | 2.044 ± 0.022 | 2.043 ± 0.030 | 1.958 ± 0.107 | 8.597 ± 0.339 | - |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 4 | `8192x1 -> 8192x4096` | 1.949 ± 0.126 | - | 1.165 ± 0.040 | 1.090 ± 0.021 | 1.007 ± 0.145 | 8.652 ± 0.128 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 1 | `33554432` | 3.561 ± 0.102 | - | 8.183 ± 0.061 | 3.594 ± 0.080 | 4.708 ± 0.476 | 3.864 ± 0.334 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 4 | `33554432` | 1.363 ± 0.103 | - | 3.065 ± 0.053 | 1.332 ± 0.092 | 1.364 ± 0.088 | 3.698 ± 0.088 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 1 | `8192 -> 8192x8192` | 2.068 ± 0.148 | - | 2.034 ± 0.110 | 4.085 ± 0.071 | 6.217 ± 0.226 | 2.280 ± 0.150 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 4 | `8192 -> 8192x8192` | 2.178 ± 0.064 | - | 2.163 ± 0.055 | 1.891 ± 0.081 | 2.086 ± 0.050 | 2.250 ± 0.136 | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 1 | `8388608x2x2 -> 8388608x2` | 2.243 ± 0.068 | - | 6.766 ± 0.112 | 4.565 ± 1.054 | 3.595 ± 0.166 | - | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 4 | `8388608x2x2 -> 8388608x2` | 2.243 ± 0.061 | - | 4.034 ± 0.267 | 1.749 ± 0.080 | 1.804 ± 0.058 | - | - |
| cpu/structural_shape | `reshape` | f64 | 1 | `33554432 -> 8192x4096` | 4.015 ± 0.191 | - | 4.387 ± 0.103 | 4.387 ± 0.082 | 3.645 ± 0.123 | 4.059 ± 0.139 | - |
| cpu/structural_shape | `reshape` | f64 | 4 | `33554432 -> 8192x4096` | 4.057 ± 0.165 | - | 2.531 ± 0.257 | 1.608 ± 0.105 | 1.740 ± 0.058 | 4.070 ± 0.113 | - |
| cpu/structural_shape | `transpose` | f64 | 1 | `4096x4096` | 24.303 ± 0.489 | - | 24.213 ± 0.281 | 18.949 ± 0.212 | 13.640 ± 0.468 | 52.026 ± 5.601 | 30.520 ± 2.094 |
| cpu/structural_shape | `transpose` | f64 | 4 | `4096x4096` | 11.255 ± 1.388 | - | 7.808 ± 0.289 | 18.990 ± 0.138 | 5.317 ± 0.465 | 54.338 ± 1.248 | 18.401 ± 6.579 |
| cpu/structural_shape | `tril` | f64 | 1 | `4096x4096` | 3.373 ± 0.165 | - | 5.942 ± 0.348 | 3.572 ± 0.128 | 2.249 ± 0.020 | 4.320 ± 0.150 | - |
| cpu/structural_shape | `tril` | f64 | 4 | `4096x4096` | 3.351 ± 0.235 | - | 4.876 ± 0.090 | 1.376 ± 0.061 | 0.843 ± 0.045 | 4.378 ± 0.089 | - |
| cpu/structural_shape | `triu` | f64 | 1 | `4096x4096` | 3.533 ± 0.147 | - | 5.766 ± 0.054 | 3.225 ± 0.097 | 2.251 ± 0.028 | 4.152 ± 0.189 | - |
| cpu/structural_shape | `triu` | f64 | 4 | `4096x4096` | 3.456 ± 0.310 | - | 4.934 ± 0.202 | 1.205 ± 0.042 | 0.818 ± 0.048 | 4.176 ± 0.098 | - |
| cpu/view_metadata | `broadcast_in_dim_view` | f64 | 1 | `8192x1 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.000 ± 0.000 | - | - | - |
| cpu/view_metadata | `broadcast_in_dim_view` | f64 | 4 | `8192x1 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.000 ± 0.000 | - | - | - |
| cpu/view_metadata | `reshape_view` | f64 | 1 | `33554432 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.000 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `reshape_view` | f64 | 4 | `33554432 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.000 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `slice_view` | f64 | 1 | `4194304 -> 2096128` | 0.000 ± 0.000 | - | unsupported | 0.000 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `slice_view` | f64 | 4 | `4194304 -> 2096128` | 0.000 ± 0.000 | - | unsupported | 0.000 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `transpose_view` | f64 | 1 | `4096x4096` | 0.000 ± 0.000 | - | unsupported | 0.000 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `transpose_view` | f64 | 4 | `4096x4096` | 0.000 ± 0.000 | - | unsupported | 0.000 ± 0.000 | - | 0.000 ± 0.000 | - |

## Short-operation batches

Total batch duration and normalized ns/op are shown without rounding sub-microsecond calls to zero. Metadata views require no execution session.

| Operation | Threads | Backend | Operations/batch | Median batch ms | Median ns/op |
|---|---:|---|---:|---:|---:|
| broadcast_in_dim_view | 1 | PyTorch Python (ms) | 16 | 0.006167 | 385.44 |
| broadcast_in_dim_view | 1 | tenferro-rs direct API (ms) | 16 | 0.002000 | 125.00 |
| broadcast_in_dim_view | 4 | PyTorch Python (ms) | 16 | 0.005958 | 372.38 |
| broadcast_in_dim_view | 4 | tenferro-rs direct API (ms) | 16 | 0.001958 | 122.38 |
| reshape_view | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000042 | 2.62 |
| reshape_view | 1 | PyTorch Python (ms) | 16 | 0.005541 | 346.31 |
| reshape_view | 1 | tenferro-rs direct API (ms) | 16 | 0.003708 | 231.75 |
| reshape_view | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000042 | 2.62 |
| reshape_view | 4 | PyTorch Python (ms) | 16 | 0.006833 | 427.06 |
| reshape_view | 4 | tenferro-rs direct API (ms) | 16 | 0.003625 | 226.56 |
| slice_view | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000042 | 2.62 |
| slice_view | 1 | PyTorch Python (ms) | 16 | 0.004833 | 302.06 |
| slice_view | 1 | tenferro-rs direct API (ms) | 16 | 0.003416 | 213.50 |
| slice_view | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000042 | 2.62 |
| slice_view | 4 | PyTorch Python (ms) | 16 | 0.005167 | 322.94 |
| slice_view | 4 | tenferro-rs direct API (ms) | 16 | 0.003291 | 205.69 |
| transpose_view | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000042 | 2.62 |
| transpose_view | 1 | PyTorch Python (ms) | 16 | 0.004959 | 309.94 |
| transpose_view | 1 | tenferro-rs direct API (ms) | 16 | 0.002750 | 171.88 |
| transpose_view | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000042 | 2.62 |
| transpose_view | 4 | PyTorch Python (ms) | 16 | 0.005167 | 322.94 |
| transpose_view | 4 | tenferro-rs direct API (ms) | 16 | 0.002917 | 182.31 |

## Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `cpu/complex/div` (c64, threads=4, shape=`8388608`): `tenferro-direct` is 12.5x faster than the slowest successful cell (`julia-base`, 19.975 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/div` (c64, threads=4, shape=`8388608`): `tenferro-trace` is 12.3x faster than the slowest successful cell (`julia-base`, 19.975 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 28.2x faster than the slowest successful cell (`pytorch-cpu`, 10.551 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 13.1x faster than the slowest successful cell (`tenferro-direct`, 5.886 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 15.8x faster than the slowest successful cell (`tenferro-direct`, 5.886 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/tanh` (f64, threads=4, shape=`8388608`): `jax-cpu` is 10.2x faster than the slowest successful cell (`julia-base`, 23.837 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`): `jax-cpu` is 10.2x faster than the slowest successful cell (`julia-base`, 54.338 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=1, shape=`33554432 -> 8192x4096`): `julia-base` is 115.3x faster than the slowest successful cell (`pytorch-cpu`, 0.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=4, shape=`33554432 -> 8192x4096`): `julia-base` is 142.3x faster than the slowest successful cell (`pytorch-cpu`, 0.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=1, shape=`4194304 -> 2096128`): `julia-base` is 100.7x faster than the slowest successful cell (`pytorch-cpu`, 0.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=4, shape=`4194304 -> 2096128`): `julia-base` is 107.7x faster than the slowest successful cell (`pytorch-cpu`, 0.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=1, shape=`4096x4096`): `julia-base` is 103.3x faster than the slowest successful cell (`pytorch-cpu`, 0.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=4, shape=`4096x4096`): `julia-base` is 107.7x faster than the slowest successful cell (`pytorch-cpu`, 0.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

## Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.
