# CPU Public API Benchmark Results

- Suite: `cpu/public_api`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/public_api.yaml`
- Run metadata: `data/results/mac-cpu/cpu/public_api/20261113_182652/run.yaml`
- Timestamp: `2026-11-13T18:26:52.437993+00:00`

Latest run: `./scripts/run_cpu_public_api.sh 1 4`.

- Public API coverage manifest: `benchmarks/cpu/public_api_coverage.yaml`

This file is generated from sequential CPU public API runs under `data/results/mac-cpu/cpu/public_api/20261113_182652`.

- tenferro-rs commit: `9a850b06155405acc99ecd0648829e9e6d3866ba`

## CPU Information

- Model: `Apple M1 (Virtual)`
- Vendor: `Apple`
- Logical CPUs: `3`
- Physical CPUs: `3`
- Sockets: `1`
- Cores per socket: `3`
- Threads per core: `1`
- NUMA nodes: `1`
- Performance levels: `Standard: 3 physical / 3 logical (L1i 192 KiB, L1d 128 KiB, L2 12 MiB, 3 CPUs/L2)`
- Python platform: `macOS-26.7.2-arm64-arm-64bit-Mach-O`
- Container: `none`

## Thread Environments

### Threads: 1

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
- XLA_FLAGS: `--xla_cpu_multi_thread_eigen=false --xla_cpu_experimental_ynn_fusion_type= intra_op_parallelism_threads=1`
- JULIA_NUM_THREADS: `1`

### Threads: 4

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
- XLA_FLAGS: `--xla_cpu_multi_thread_eigen=true --xla_cpu_experimental_ynn_fusion_type= intra_op_parallelism_threads=4`
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
- Compact Householder QR and RRQR use explicit fixture/session setup; QR extraction and reconstruction/orthogonality checks are untimed. RRQR retains Q, R, permutation and numerical rank; JAX's three-output pivoted QR and PyTorch's unpivoted QR are not equivalent and are unsupported. Compact trace state and caller-output trace solve are unsupported.
- The gap cases use explicit setup and numerical validation outside sampling. Axis-wise sum-of-squares and PyTorch/Julia masked log-softmax compositions are excluded from direct operation comparisons.
- PyTorch complex conjugation uses `torch.conj_physical` to match tenferro-rs physical output rather than the lazy conjugate view from `torch.conj`.
- `dot_general_with_conj` instead uses PyTorch's lazy conjugate view so conjugation can be handled by the contraction, matching tenferro-rs' conjugation flags; trace is unsupported because there is no equivalent public traced API. Julia's `dot_general_with_conj` row materializes the conjugate before the GEMM instead, since Julia has no lazy conj-without-transpose spelling that BLAS can fuse.
- `pad` has no natural Base spelling and stays missing for Julia.
- `dynamic_update_slice` reports trace mode as `unsupported` because tenferro-rs does not currently expose a corresponding `TracedTensor` API.
- `full_piv_lu` and `full_piv_lu_solve` are excluded because PyTorch has no direct public full-pivot equivalent; substituting `torch.linalg.solve` would compare different algorithms.
- `cpu/linalg_batched` rows use a batch of 1024 matrices with one rhs column. `batched_lu_factor` times the packed LU factorization alone. `batched_lu_solve` reuses LU factors prepared during warmup, outside the measured region, and times only the solve (tenferro-rs `LinalgBackend::lu_solve_prepared`, PyTorch `torch.linalg.lu_solve`, JAX `jax.scipy.linalg.lu_solve`, Julia `LAPACK.getrs!`). `batched_triangular_solve` times a lower-triangular solve.
- Trace mode is `unsupported` for `batched_lu_factor` and `batched_lu_solve` because `LinalgOp` is not public, so a trace cannot hold a bare LU factor or prepared solve. The traced solve (LU factor plus prepared solve) and its backward are measured by the `cpu/cpu_ops` `batched_solve` and `grad_sum_batched_solve_backward` rows.
- Julia has no batched LAPACK entry point, so the Julia `cpu/linalg_batched` rows loop the per-matrix LAPACK call over the batch axis and allocate their outputs inside the timed call.
- The tenferro-rs runner builds its direct `CpuBackend`, the trace runtime, and the eager runtime with exactly `--num-threads` workers, and it fails at startup when a thread environment variable disagrees with that count or when the backend execution scope reports a different Rayon thread count.
- `svd_full` remains in the table even when the selected tenferro-rs provider reports it as unsupported.
- Julia is column-major, like tenferro-rs, so the `julia-base`/`strided-jl` columns need no PyTorch/JAX-style layout reconstruction to keep the same logical fixture values.
- Julia warmup runs move JIT compilation outside the measured region, the same way PyTorch/JAX warmups do.
- Julia factorization rows materialize their factors inside the timed call (`cholesky` returns the factor matrix, `lu` returns P/L/U, `qr` returns Q/R). A Julia `Factorization` keeps its factors packed in LAPACK's working storage, so timing the compact object would compare strictly less work than the tenferro-rs and PyTorch columns, which return separate materialized tensors.
- Julia dense linalg rows run through Julia's own BLAS/LAPACK (libblastrampoline, by default OpenBLAS), recorded as `julia.blas_provider` in the run metadata. When that differs from the provider tenferro-rs and PyTorch link against (Accelerate on macOS), those rows partly compare BLAS implementations rather than framework overhead; read them together with the recorded providers.
- The Julia `lstsq` row uses `qr(a) \ rhs` rather than `a \ rhs`: the bare backslash runs a column-pivoted, rank-revealing QR (the LAPACK `gelsy` algorithm), while the PyTorch row selects the `gels` driver, so the unpivoted spelling is the like-for-like comparison.
- `julia-base` uses the natural Base/LinearAlgebra spelling and `strided-jl` the natural Strided.jl (`@strided`) spelling; each is populated only for rows where that spelling naturally applies (reductions and dense linalg have no natural Strided.jl spelling, so `strided-jl` covers elementwise/chain/transpose rows, the elementwise `cpu/output_reuse` `_into` rows, and the elementwise `cpu/complex` rows conj/mul/div/exp/log).
- Strided.jl (https://github.com/Jutho/Strided.jl) is prior art for tenferro-rs' strided-rs kernel layer; the `strided-jl` column credits that lineage directly in the report.

## Threads: 1 4

- CSV: `data/results/mac-cpu/cpu/public_api/20261113_182652/cpu_public_api_t1_20261113_182652.csv`
- CSV: `data/results/mac-cpu/cpu/public_api/20261113_182652/cpu_public_api_t4_20261113_182652.csv`

## CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| cpu/complex | `cholesky` | c64 | 1 | `448x448` | 4.654 ± 0.282 | - | 4.671 ± 0.674 | 4.737 ± 0.198 | 1.746 ± 0.080 | 7.581 ± 1.455 | - |
| cpu/complex | `cholesky` | c64 | 4 | `448x448` | 5.030 ± 3.277 | - | 3.982 ± 0.669 | 3.569 ± 0.225 | 1.959 ± 0.533 | 83.951 ± 22.588 | - |
| cpu/complex | `conj` | c64 | 1 | `16777216` | 15.834 ± 0.853 | - | 13.070 ± 3.836 | 13.675 ± 0.841 | 15.400 ± 0.575 | 17.919 ± 3.073 | 16.819 ± 0.861 |
| cpu/complex | `conj` | c64 | 4 | `16777216` | 13.400 ± 0.934 | - | 13.223 ± 0.775 | 13.359 ± 2.809 | 12.507 ± 0.678 | 18.365 ± 7.181 | 14.319 ± 0.577 |
| cpu/complex | `div` | c64 | 1 | `8388608` | 53.338 ± 4.724 | - | 49.982 ± 3.061 | 22.534 ± 1.914 | 38.182 ± 6.092 | 58.586 ± 3.994 | 60.500 ± 2.539 |
| cpu/complex | `div` | c64 | 4 | `8388608` | 27.310 ± 10.424 | - | 24.411 ± 2.906 | 13.552 ± 3.263 | 26.987 ± 3.190 | 65.073 ± 13.482 | 34.143 ± 2.133 |
| cpu/complex | `dot_general` | c64 | 1 | `640x640` | 14.065 ± 4.106 | - | 16.438 ± 5.415 | 12.959 ± 1.363 | 58.183 ± 4.506 | 66.021 ± 7.720 | - |
| cpu/complex | `dot_general` | c64 | 4 | `640x640` | 11.027 ± 3.315 | - | 11.012 ± 0.313 | 11.485 ± 2.088 | 25.656 ± 3.778 | 49.911 ± 10.891 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 1 | `640x640` | 26.712 ± 15.488 | - | unsupported | 13.267 ± 1.325 | 59.500 ± 3.554 | 75.233 ± 4.582 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 4 | `640x640` | 11.631 ± 1.389 | - | unsupported | 12.432 ± 7.161 | 27.095 ± 3.511 | 51.885 ± 7.065 | - |
| cpu/complex | `eig` | c64 | 1 | `112x112` | 13.137 ± 1.290 | - | 12.359 ± 1.007 | 13.014 ± 0.913 | 21.461 ± 3.668 | 9.887 ± 1.254 | - |
| cpu/complex | `eig` | c64 | 4 | `112x112` | 12.486 ± 0.319 | - | 10.284 ± 1.219 | 11.820 ± 0.540 | 24.698 ± 1.345 | 489.877 ± 274.224 | - |
| cpu/complex | `exp` | c64 | 1 | `4194304` | 43.823 ± 1.994 | - | 40.081 ± 3.718 | 36.642 ± 0.780 | 61.548 ± 31.517 | 50.577 ± 4.800 | 48.097 ± 2.612 |
| cpu/complex | `exp` | c64 | 4 | `4194304` | 23.487 ± 3.798 | - | 22.264 ± 6.135 | 18.759 ± 0.549 | 43.960 ± 12.940 | 65.698 ± 5.179 | 46.107 ± 24.211 |
| cpu/complex | `log` | c64 | 1 | `4194304` | 52.154 ± 0.984 | - | 51.936 ± 4.021 | 54.809 ± 4.593 | 106.710 ± 17.126 | 72.487 ± 3.921 | 79.252 ± 8.948 |
| cpu/complex | `log` | c64 | 4 | `4194304` | 28.133 ± 1.397 | - | 30.646 ± 4.120 | 17.486 ± 3.164 | 59.554 ± 11.852 | 111.114 ± 13.464 | 41.728 ± 6.625 |
| cpu/complex | `mul` | c64 | 1 | `8388608` | 11.459 ± 2.259 | - | 10.180 ± 1.157 | 14.833 ± 1.183 | 10.836 ± 2.297 | 15.345 ± 4.755 | 11.735 ± 0.648 |
| cpu/complex | `mul` | c64 | 4 | `8388608` | 10.300 ± 1.505 | - | 11.271 ± 1.710 | 10.122 ± 1.272 | 10.803 ± 0.775 | 11.702 ± 3.011 | 11.831 ± 0.845 |
| cpu/complex | `norm_fro` | c64 | 1 | `2048x1536` | 1.583 ± 0.139 | - | 8.589 ± 1.457 | 7.025 ± 0.883 | 5.104 ± 0.592 | 10.007 ± 1.036 | - |
| cpu/complex | `norm_fro` | c64 | 4 | `2048x1536` | 1.576 ± 0.370 | - | 6.116 ± 0.909 | 3.038 ± 0.400 | 4.878 ± 0.538 | 10.505 ± 1.327 | - |
| cpu/complex | `qr` | c64 | 1 | `256x256` | 13.980 ± 2.062 | - | 8.542 ± 0.519 | 8.755 ± 0.802 | 7.334 ± 4.267 | 13.746 ± 3.950 | - |
| cpu/complex | `qr` | c64 | 4 | `256x256` | 8.577 ± 2.053 | - | 8.639 ± 0.205 | 11.421 ± 6.814 | 8.205 ± 0.715 | 163.188 ± 55.251 | - |
| cpu/complex | `solve` | c64 | 1 | `384x384,rhs=8` | 5.444 ± 0.196 | - | 5.734 ± 0.582 | 6.234 ± 0.485 | 2.744 ± 0.229 | 8.007 ± 0.785 | - |
| cpu/complex | `solve` | c64 | 4 | `384x384,rhs=8` | 6.981 ± 2.607 | - | 5.922 ± 1.684 | 5.674 ± 2.184 | 3.251 ± 0.543 | 143.168 ± 52.186 | - |
| cpu/complex | `svd` | c64 | 1 | `160x160` | 10.580 ± 0.944 | - | 12.065 ± 1.758 | 9.051 ± 1.008 | 7.226 ± 0.729 | 8.749 ± 1.658 | - |
| cpu/complex | `svd` | c64 | 4 | `160x160` | 7.123 ± 0.061 | - | 8.592 ± 0.843 | 7.797 ± 2.860 | 8.154 ± 0.806 | 644.050 ± 402.014 | - |
| cpu/complex | `tensordot` | c64 | 1 | `640x640` | 21.964 ± 4.322 | - | 21.763 ± 2.613 | 11.899 ± 1.154 | 57.779 ± 4.406 | 71.326 ± 7.025 | - |
| cpu/complex | `tensordot` | c64 | 4 | `640x640` | 9.551 ± 0.308 | - | 9.529 ± 1.561 | 15.455 ± 7.382 | 25.118 ± 1.568 | 45.113 ± 7.780 | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 1 | `1024x1024` | 9.153 ± 1.234 | - | unsupported | 13.062 ± 1.823 | 64.961 ± 5.361 | - | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 4 | `1024x1024` | 13.332 ± 1.443 | - | unsupported | 16.747 ± 3.136 | 28.941 ± 2.658 | - | - |
| cpu/elementwise_reduction | `abs` | f64 | 1 | `33554432` | 13.179 ± 3.391 | - | 15.270 ± 6.208 | 16.759 ± 5.888 | 18.675 ± 2.983 | 14.681 ± 7.939 | 15.202 ± 2.490 |
| cpu/elementwise_reduction | `abs` | f64 | 4 | `33554432` | 15.768 ± 3.089 | - | 13.344 ± 2.919 | 13.697 ± 0.810 | 11.263 ± 0.587 | 19.276 ± 7.294 | 15.152 ± 1.764 |
| cpu/elementwise_reduction | `add` | f64 | 1 | `33554432` | 17.144 ± 1.294 | - | 19.810 ± 1.765 | 23.239 ± 2.864 | 27.749 ± 4.588 | 28.204 ± 23.730 | 25.090 ± 6.056 |
| cpu/elementwise_reduction | `add` | f64 | 4 | `33554432` | 23.466 ± 2.780 | - | 22.990 ± 4.234 | 19.117 ± 1.445 | 27.884 ± 3.719 | 20.941 ± 15.214 | 21.417 ± 2.430 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 1 | `4194304` | 26.945 ± 4.788 | - | 33.686 ± 1.591 | 42.999 ± 1.171 | 52.344 ± 19.697 | 60.045 ± 6.945 | 62.913 ± 3.252 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 4 | `4194304` | 20.548 ± 2.295 | - | 21.398 ± 2.353 | 35.416 ± 8.816 | 17.112 ± 3.289 | 65.581 ± 8.452 | 32.594 ± 8.722 |
| cpu/elementwise_reduction | `clamp` | f64 | 1 | `8388608` | 7.268 ± 0.690 | - | 6.992 ± 0.589 | 9.975 ± 3.945 | 9.147 ± 1.508 | 7.886 ± 0.420 | 9.808 ± 7.588 |
| cpu/elementwise_reduction | `clamp` | f64 | 4 | `8388608` | 7.610 ± 1.012 | - | 8.396 ± 2.158 | 9.091 ± 0.732 | 8.783 ± 0.432 | 8.414 ± 2.575 | 9.136 ± 19.289 |
| cpu/elementwise_reduction | `compare_lt` | f64 | 1 | `33554432` | 24.699 ± 3.662 | - | 23.413 ± 1.808 | 27.636 ± 6.736 | 21.169 ± 2.170 | 18.703 ± 10.627 | - |
| cpu/elementwise_reduction | `compare_lt` | f64 | 4 | `33554432` | 15.454 ± 1.058 | - | 15.245 ± 2.220 | 14.751 ± 1.440 | 15.479 ± 0.677 | 20.662 ± 6.351 | - |
| cpu/elementwise_reduction | `cos` | f64 | 1 | `8388608` | 31.067 ± 0.561 | - | 30.655 ± 0.341 | 37.171 ± 2.071 | 28.499 ± 1.761 | 40.934 ± 0.848 | 41.989 ± 1.135 |
| cpu/elementwise_reduction | `cos` | f64 | 4 | `8388608` | 18.438 ± 4.889 | - | 24.929 ± 4.843 | 19.243 ± 5.112 | 15.080 ± 3.140 | 44.178 ± 10.164 | 22.997 ± 7.317 |
| cpu/elementwise_reduction | `div` | f64 | 1 | `33554432` | 25.086 ± 6.140 | - | 23.681 ± 1.945 | 21.105 ± 2.010 | 31.547 ± 3.220 | 20.910 ± 5.442 | 25.834 ± 4.996 |
| cpu/elementwise_reduction | `div` | f64 | 4 | `33554432` | 21.834 ± 4.798 | - | 22.679 ± 2.543 | 16.363 ± 1.446 | 24.573 ± 1.930 | 36.229 ± 26.473 | 24.078 ± 0.729 |
| cpu/elementwise_reduction | `exp` | f64 | 1 | `8388608` | 24.319 ± 10.864 | - | 20.753 ± 5.853 | 31.079 ± 4.202 | 17.358 ± 0.875 | 42.898 ± 7.043 | 42.655 ± 8.149 |
| cpu/elementwise_reduction | `exp` | f64 | 4 | `8388608` | 11.492 ± 1.236 | - | 12.029 ± 1.378 | 15.192 ± 2.197 | 10.281 ± 3.464 | 45.283 ± 13.895 | 21.416 ± 2.888 |
| cpu/elementwise_reduction | `expm1` | f64 | 1 | `4194304` | 26.786 ± 2.260 | - | 26.039 ± 0.662 | 29.029 ± 3.516 | 23.276 ± 24.147 | 30.180 ± 7.426 | 33.523 ± 7.997 |
| cpu/elementwise_reduction | `expm1` | f64 | 4 | `4194304` | 15.590 ± 2.308 | - | 16.889 ± 3.706 | 20.205 ± 5.746 | 19.490 ± 2.965 | 26.933 ± 3.528 | 16.627 ± 6.410 |
| cpu/elementwise_reduction | `log` | f64 | 1 | `8388608` | 33.683 ± 5.838 | - | 28.911 ± 3.882 | 47.437 ± 3.684 | 27.803 ± 0.671 | 66.460 ± 24.657 | 57.265 ± 3.888 |
| cpu/elementwise_reduction | `log` | f64 | 4 | `8388608` | 19.381 ± 7.723 | - | 23.775 ± 8.040 | 30.219 ± 7.370 | 16.297 ± 2.869 | 41.562 ± 3.939 | 26.651 ± 14.175 |
| cpu/elementwise_reduction | `log1p` | f64 | 1 | `4194304` | 15.299 ± 0.397 | - | 15.296 ± 1.496 | 25.875 ± 2.079 | 38.945 ± 10.204 | 29.997 ± 3.286 | 28.966 ± 1.787 |
| cpu/elementwise_reduction | `log1p` | f64 | 4 | `4194304` | 9.430 ± 3.391 | - | 13.119 ± 4.221 | 18.870 ± 7.366 | 16.182 ± 5.266 | 31.404 ± 5.218 | 13.775 ± 6.978 |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 1 | `1024x64` | 0.258 ± 0.004 | 0.294 ± 0.044 | unsupported | unsupported | 0.302 ± 0.061 | - | - |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 4 | `1024x64` | 0.354 ± 0.084 | 0.394 ± 0.101 | unsupported | unsupported | 0.178 ± 0.155 | - | - |
| cpu/elementwise_reduction | `maximum` | f64 | 1 | `33554432` | 20.684 ± 3.838 | - | 18.688 ± 1.853 | 27.118 ± 4.447 | 33.690 ± 2.175 | 27.111 ± 7.241 | 26.288 ± 8.665 |
| cpu/elementwise_reduction | `maximum` | f64 | 4 | `33554432` | 22.508 ± 0.991 | - | 21.635 ± 0.951 | 19.007 ± 0.930 | 25.061 ± 2.020 | 36.822 ± 134.617 | 50.550 ± 34.383 |
| cpu/elementwise_reduction | `minimum` | f64 | 1 | `33554432` | 18.702 ± 1.231 | - | 37.306 ± 24.272 | 25.686 ± 9.128 | 36.540 ± 9.952 | 132.775 ± 228.502 | 28.977 ± 6.540 |
| cpu/elementwise_reduction | `minimum` | f64 | 4 | `33554432` | 21.632 ± 1.303 | - | 24.032 ± 2.699 | 18.594 ± 2.327 | 26.858 ± 5.046 | 27.565 ± 15.452 | 26.120 ± 6.109 |
| cpu/elementwise_reduction | `mul` | f64 | 1 | `33554432` | 19.769 ± 1.148 | - | 24.433 ± 2.565 | 23.174 ± 3.295 | 34.375 ± 7.296 | 22.618 ± 2.490 | 25.410 ± 3.495 |
| cpu/elementwise_reduction | `mul` | f64 | 4 | `33554432` | 22.093 ± 1.678 | - | 22.419 ± 2.623 | 19.553 ± 1.511 | 24.863 ± 0.773 | 27.181 ± 7.609 | 19.762 ± 1.629 |
| cpu/elementwise_reduction | `neg` | f64 | 1 | `33554432` | 14.607 ± 4.795 | - | 15.284 ± 2.603 | 14.545 ± 1.476 | 17.390 ± 2.125 | 12.788 ± 0.625 | 16.829 ± 2.604 |
| cpu/elementwise_reduction | `neg` | f64 | 4 | `33554432` | 16.317 ± 2.813 | - | 14.436 ± 1.949 | 12.921 ± 0.559 | 10.177 ± 1.483 | 14.782 ± 5.500 | 15.434 ± 4.130 |
| cpu/elementwise_reduction | `pow` | f64 | 1 | `4194304` | 42.137 ± 3.312 | - | 39.390 ± 13.656 | 76.331 ± 2.003 | 65.494 ± 17.240 | 102.680 ± 33.174 | 95.994 ± 10.345 |
| cpu/elementwise_reduction | `pow` | f64 | 4 | `4194304` | 25.141 ± 3.748 | - | 21.830 ± 9.775 | 36.754 ± 19.781 | 41.066 ± 10.965 | 92.506 ± 24.160 | 38.853 ± 17.185 |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 1 | `8192x4096` | 10.685 ± 5.172 | - | 13.861 ± 2.372 | 13.395 ± 1.222 | 23.371 ± 3.881 | 6.339 ± 0.427 | - |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 4 | `8192x4096` | 7.115 ± 0.834 | - | 7.673 ± 1.032 | 15.685 ± 4.101 | 11.932 ± 1.079 | 8.432 ± 2.045 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 1 | `2048x2048` | 1.503 ± 0.702 | - | 1.496 ± 1.308 | 21.174 ± 5.928 | 3.967 ± 1.489 | 0.784 ± 0.020 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 4 | `2048x2048` | 1.592 ± 0.622 | - | 0.988 ± 0.575 | 14.062 ± 6.533 | 1.633 ± 0.244 | 0.869 ± 0.050 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 1 | `2048x2048` | 0.877 ± 0.139 | - | 1.061 ± 0.060 | 1.593 ± 0.468 | 1.678 ± 0.379 | 1.135 ± 0.750 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 4 | `2048x2048` | 2.106 ± 0.257 | - | 2.040 ± 0.247 | 1.838 ± 1.257 | 1.460 ± 0.705 | 0.880 ± 0.054 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 1 | `8192x4096` | 11.425 ± 6.922 | - | 10.290 ± 7.622 | 12.956 ± 1.030 | 23.827 ± 1.422 | 6.357 ± 0.408 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 4 | `8192x4096` | 8.467 ± 2.735 | - | 7.230 ± 0.802 | 16.482 ± 4.794 | 11.395 ± 4.272 | 7.611 ± 1.591 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 1 | `2048x2048` | 1.082 ± 0.223 | - | 0.862 ± 0.032 | 19.354 ± 0.892 | 2.755 ± 0.287 | 2.413 ± 0.985 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 4 | `2048x2048` | 0.835 ± 0.073 | - | 0.972 ± 0.029 | 15.830 ± 3.430 | 3.326 ± 0.404 | 0.828 ± 0.071 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 1 | `4096x4096` | 3.678 ± 0.221 | - | 3.062 ± 0.097 | 6.658 ± 0.505 | 5.418 ± 0.991 | 3.939 ± 2.309 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 4 | `4096x4096` | 6.824 ± 0.521 | - | 7.368 ± 0.459 | 7.738 ± 2.460 | 5.542 ± 1.377 | 3.216 ± 0.292 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 1 | `8192x4096` | 7.088 ± 0.775 | - | 9.219 ± 1.681 | 9.943 ± 0.965 | 41.430 ± 4.500 | 6.469 ± 0.294 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 4 | `8192x4096` | 7.038 ± 1.304 | - | 7.008 ± 0.188 | 9.911 ± 1.826 | 20.773 ± 1.184 | 7.798 ± 0.643 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 1 | `2048x2048` | 0.752 ± 0.021 | - | 0.757 ± 0.019 | 5.231 ± 0.909 | 2.990 ± 0.360 | 1.766 ± 0.510 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 4 | `2048x2048` | 0.901 ± 0.070 | - | 0.953 ± 0.247 | 3.134 ± 0.608 | 2.640 ± 0.250 | 0.880 ± 0.183 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 1 | `2048x2048` | 0.742 ± 0.003 | - | 0.821 ± 0.083 | 1.290 ± 0.454 | 1.709 ± 0.418 | 2.389 ± 0.840 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 4 | `2048x2048` | 1.993 ± 0.147 | - | 1.963 ± 0.251 | 1.070 ± 0.232 | 0.930 ± 0.861 | 0.790 ± 0.016 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 1 | `8192x4096` | 4.958 ± 1.491 | - | 6.203 ± 0.302 | 9.007 ± 2.559 | 35.399 ± 3.905 | 7.370 ± 0.589 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 4 | `8192x4096` | 7.618 ± 1.514 | - | 8.901 ± 2.601 | 10.982 ± 6.896 | 15.533 ± 0.503 | 8.467 ± 1.392 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 1 | `2048x2048` | 0.628 ± 0.011 | - | 0.669 ± 0.035 | 4.853 ± 0.298 | 2.923 ± 0.523 | 0.870 ± 0.087 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 4 | `2048x2048` | 0.912 ± 0.093 | - | 0.854 ± 0.024 | 4.274 ± 0.831 | 2.658 ± 0.987 | 0.656 ± 0.040 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 1 | `2048x2048` | 0.849 ± 0.106 | - | 0.730 ± 0.008 | 1.087 ± 0.182 | 1.233 ± 0.416 | 0.987 ± 0.155 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 4 | `2048x2048` | 2.000 ± 0.255 | - | 2.115 ± 0.283 | 1.008 ± 0.111 | 1.227 ± 0.305 | 0.795 ± 0.126 | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 1 | `2048x2048` | 0.808 ± 0.098 | 0.792 ± 0.017 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 4 | `2048x2048` | 0.893 ± 0.048 | 0.942 ± 0.191 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `rem` | f64 | 1 | `8388608` | 33.466 ± 1.974 | - | 36.263 ± 7.703 | 39.369 ± 4.086 | 40.840 ± 2.644 | 36.250 ± 53.017 | 38.107 ± 16.003 |
| cpu/elementwise_reduction | `rem` | f64 | 4 | `8388608` | 15.409 ± 1.234 | - | 15.642 ± 2.332 | 20.198 ± 3.073 | 32.684 ± 6.765 | 27.127 ± 3.014 | 24.198 ± 7.492 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 1 | `33554432` | 19.143 ± 0.840 | - | 19.272 ± 2.577 | 20.480 ± 1.040 | 42.982 ± 1.020 | 47.967 ± 2.058 | 47.800 ± 3.902 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 4 | `33554432` | 14.864 ± 2.640 | - | 16.853 ± 2.807 | 12.743 ± 4.200 | 22.140 ± 4.250 | 46.915 ± 9.157 | 21.006 ± 1.342 |
| cpu/elementwise_reduction | `select` | f64 | 1 | `33554432` | 34.292 ± 9.393 | - | 28.774 ± 5.536 | 32.419 ± 4.269 | 49.231 ± 5.067 | 34.770 ± 27.869 | - |
| cpu/elementwise_reduction | `select` | f64 | 4 | `33554432` | 29.012 ± 2.988 | - | 30.447 ± 3.872 | 21.928 ± 3.630 | 26.993 ± 2.589 | 35.844 ± 8.428 | - |
| cpu/elementwise_reduction | `sign` | f64 | 1 | `33554432` | 13.058 ± 1.450 | - | 12.931 ± 6.211 | 15.555 ± 2.158 | 18.160 ± 2.595 | 17.585 ± 41.358 | 22.231 ± 2.406 |
| cpu/elementwise_reduction | `sign` | f64 | 4 | `33554432` | 14.104 ± 2.736 | - | 13.961 ± 0.872 | 14.258 ± 2.690 | 12.495 ± 3.255 | 19.700 ± 3.603 | 18.162 ± 5.566 |
| cpu/elementwise_reduction | `sin` | f64 | 1 | `8388608` | 24.427 ± 2.581 | - | 23.927 ± 0.296 | 32.739 ± 1.212 | 26.257 ± 0.594 | 38.532 ± 3.271 | 37.975 ± 6.017 |
| cpu/elementwise_reduction | `sin` | f64 | 4 | `8388608` | 18.220 ± 7.584 | - | 17.667 ± 3.960 | 18.254 ± 4.675 | 13.800 ± 2.328 | 37.664 ± 3.824 | 18.477 ± 5.018 |
| cpu/elementwise_reduction | `sqrt` | f64 | 1 | `33554432` | 14.309 ± 2.435 | - | 13.134 ± 0.227 | 25.582 ± 4.858 | 24.678 ± 2.159 | 33.390 ± 14.350 | 31.243 ± 1.623 |
| cpu/elementwise_reduction | `sqrt` | f64 | 4 | `33554432` | 15.193 ± 1.405 | - | 16.622 ± 1.655 | 12.873 ± 1.009 | 18.646 ± 2.779 | 29.535 ± 33.885 | 17.209 ± 2.380 |
| cpu/elementwise_reduction | `sub` | f64 | 1 | `33554432` | 26.689 ± 6.200 | - | 19.529 ± 1.689 | 21.628 ± 5.317 | 32.108 ± 2.067 | 25.094 ± 3.696 | 25.651 ± 7.489 |
| cpu/elementwise_reduction | `sub` | f64 | 4 | `33554432` | 22.039 ± 3.770 | - | 21.042 ± 2.542 | 19.279 ± 0.968 | 31.665 ± 4.228 | 24.720 ± 6.104 | 21.269 ± 1.233 |
| cpu/elementwise_reduction | `tanh` | f64 | 1 | `8388608` | 84.724 ± 3.770 | - | 65.942 ± 0.801 | 73.977 ± 6.685 | 14.412 ± 0.973 | 51.818 ± 5.202 | 59.853 ± 18.425 |
| cpu/elementwise_reduction | `tanh` | f64 | 4 | `8388608` | 45.903 ± 5.035 | - | 48.681 ± 7.408 | 42.977 ± 10.612 | 8.117 ± 1.477 | 58.922 ± 17.962 | 22.688 ± 7.352 |
| cpu/elementwise_reduction | `where_select` | f64 | 1 | `33554432` | 27.736 ± 1.815 | 31.597 ± 2.563 | unsupported | 33.591 ± 7.268 | 34.025 ± 3.243 | - | - |
| cpu/elementwise_reduction | `where_select` | f64 | 4 | `33554432` | 30.000 ± 4.516 | 29.132 ± 0.581 | unsupported | 25.152 ± 6.688 | 29.011 ± 3.796 | - | - |
| cpu/indexing_layout | `concatenate` | f64 | 1 | `1048576+1048576` | 0.703 ± 0.200 | - | 0.656 ± 0.024 | 0.767 ± 0.050 | 0.725 ± 0.145 | 0.968 ± 0.156 | - |
| cpu/indexing_layout | `concatenate` | f64 | 4 | `1048576+1048576` | 0.828 ± 0.271 | - | 0.978 ± 0.173 | 0.958 ± 0.275 | 1.255 ± 0.828 | 1.483 ± 0.382 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 1 | `4194304 -> 2097152` | 0.773 ± 0.045 | - | 0.738 ± 0.101 | 0.963 ± 0.163 | 1.112 ± 0.431 | 0.992 ± 0.161 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 4 | `4194304 -> 2097152` | 0.673 ± 0.032 | - | 0.789 ± 0.110 | 0.911 ± 0.059 | 1.092 ± 0.127 | 1.931 ± 0.318 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 1 | `2097152` | 1.100 ± 0.083 | - | unsupported | 1.664 ± 1.063 | 1.742 ± 0.995 | 1.458 ± 0.393 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 4 | `2097152` | 1.363 ± 0.149 | - | unsupported | 1.533 ± 0.323 | 1.117 ± 0.261 | 2.274 ± 1.231 | - |
| cpu/indexing_layout | `gather` | f64 | 1 | `262144` | 0.329 ± 0.001 | - | 0.412 ± 0.124 | 0.435 ± 0.423 | 0.492 ± 0.486 | 1.349 ± 0.380 | - |
| cpu/indexing_layout | `gather` | f64 | 4 | `262144` | 0.615 ± 0.277 | - | 1.036 ± 0.184 | 0.356 ± 0.056 | 0.547 ± 0.291 | 2.893 ± 0.940 | - |
| cpu/indexing_layout | `pad` | f64 | 1 | `2097152` | 0.853 ± 0.204 | - | 0.674 ± 0.010 | 1.278 ± 0.230 | 1.080 ± 0.196 | - | - |
| cpu/indexing_layout | `pad` | f64 | 4 | `2097152` | 0.724 ± 0.073 | - | 0.766 ± 0.108 | 1.683 ± 0.241 | 1.468 ± 0.451 | - | - |
| cpu/indexing_layout | `reverse` | f64 | 1 | `2097152` | 0.851 ± 0.336 | - | 0.858 ± 0.120 | 1.195 ± 0.289 | 1.104 ± 0.334 | 1.564 ± 0.376 | - |
| cpu/indexing_layout | `reverse` | f64 | 4 | `2097152` | 1.074 ± 0.362 | - | 0.903 ± 0.294 | 0.943 ± 0.312 | 1.448 ± 0.625 | 1.444 ± 1.101 | - |
| cpu/indexing_layout | `scatter` | f64 | 1 | `262144` | 0.568 ± 0.025 | - | 0.663 ± 0.034 | 0.610 ± 0.143 | 0.597 ± 0.207 | 1.313 ± 2.368 | - |
| cpu/indexing_layout | `scatter` | f64 | 4 | `262144` | 1.164 ± 0.382 | - | 1.158 ± 0.420 | 0.961 ± 0.638 | 0.535 ± 0.277 | 2.030 ± 0.908 | - |
| cpu/indexing_layout | `slice` | f64 | 1 | `4194304 -> 2096128` | 1.269 ± 0.078 | - | 0.004 ± 0.000 | 1.505 ± 0.654 | 1.464 ± 0.295 | 1.632 ± 0.476 | - |
| cpu/indexing_layout | `slice` | f64 | 4 | `4194304 -> 2096128` | 1.315 ± 0.227 | - | 0.002 ± 0.000 | 1.638 ± 0.383 | 1.551 ± 0.312 | 3.175 ± 0.944 | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x2x2` | 0.055 ± 0.000 | - | 0.056 ± 0.000 | 0.118 ± 0.011 | 0.118 ± 0.122 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x4x4` | 0.087 ± 0.000 | - | 0.089 ± 0.000 | 0.162 ± 0.008 | 0.150 ± 0.081 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x8x8` | 0.188 ± 0.002 | - | 0.193 ± 0.005 | 0.270 ± 0.012 | 0.349 ± 0.089 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x128x128` | 0.026 ± 0.001 | - | 0.028 ± 0.002 | 0.038 ± 0.004 | 0.097 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x32x32` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.050 ± 0.115 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x4x4` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.004 ± 0.000 | 0.010 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x8x8` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x2x2` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x4x4` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.004 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x8x8` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x2x2` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.004 ± 0.000 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x4x4` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x8x8` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.061 ± 0.057 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x128x128` | 0.176 ± 0.004 | - | 0.176 ± 0.010 | 0.260 ± 0.422 | 0.326 ± 0.102 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x32x32` | 0.010 ± 0.000 | - | 0.012 ± 0.000 | 0.017 ± 0.002 | 0.058 ± 0.028 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x4x4` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x8x8` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.070 ± 0.028 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x2x2` | 0.060 ± 0.001 | - | 0.063 ± 0.005 | 0.217 ± 0.153 | 0.182 ± 0.039 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x4x4` | 0.097 ± 0.002 | - | 0.100 ± 0.002 | 0.216 ± 0.202 | 0.154 ± 0.060 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x8x8` | 0.200 ± 0.035 | - | 0.207 ± 0.026 | 0.721 ± 0.695 | 0.257 ± 0.069 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x128x128` | 0.026 ± 0.001 | - | 0.029 ± 0.002 | 0.226 ± 0.328 | 0.073 ± 0.064 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.019 ± 0.002 | 0.010 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x32x32` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.021 ± 0.007 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.020 ± 0.001 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x8x8` | 0.002 ± 0.000 | - | 0.009 ± 0.001 | 0.020 ± 0.133 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x2x2` | 0.004 ± 0.000 | - | 0.004 ± 0.000 | 0.019 ± 0.003 | 0.010 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.001 | 0.019 ± 0.002 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x8x8` | 0.002 ± 0.000 | - | 0.010 ± 0.004 | 0.020 ± 0.000 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.001 | 0.019 ± 0.003 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x4x4` | 0.002 ± 0.000 | - | 0.010 ± 0.025 | 0.019 ± 0.001 | 0.088 ± 0.083 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x8x8` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.021 ± 0.006 | 0.021 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x128x128` | 0.181 ± 0.013 | - | 0.182 ± 0.035 | 0.451 ± 0.235 | 0.191 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x2x2` | 0.005 ± 0.000 | - | 0.004 ± 0.003 | 0.020 ± 0.002 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x32x32` | 0.012 ± 0.001 | - | 0.013 ± 0.001 | 0.170 ± 0.191 | 0.071 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.007 | 0.020 ± 0.000 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x8x8` | 0.004 ± 0.000 | - | 0.005 ± 0.007 | 0.021 ± 0.003 | 0.008 ± 0.031 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x2x2` | 0.272 ± 0.105 | - | 0.269 ± 0.035 | 0.499 ± 0.110 | 1.200 ± 0.303 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x4x4` | 1.812 ± 1.129 | - | 1.540 ± 0.482 | 1.911 ± 0.119 | 2.479 ± 0.175 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x8x8` | 7.853 ± 1.522 | - | 6.991 ± 1.570 | 8.096 ± 1.293 | 8.031 ± 1.440 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x128x128` | 5.024 ± 1.326 | - | 4.011 ± 0.046 | 4.022 ± 0.828 | 4.094 ± 0.140 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.001 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x32x32` | 0.097 ± 0.012 | - | 0.106 ± 0.049 | 0.094 ± 0.023 | 0.336 ± 0.154 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x8x8` | 0.007 ± 0.000 | - | 0.008 ± 0.000 | 0.010 ± 0.001 | 0.028 ± 0.146 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.001 | 0.006 ± 0.001 | 0.053 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x4x4` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.009 ± 0.003 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x8x8` | 0.018 ± 0.000 | - | 0.019 ± 0.000 | 0.024 ± 0.004 | 0.074 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.008 ± 0.000 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x4x4` | 0.007 ± 0.000 | - | 0.008 ± 0.000 | 0.011 ± 0.000 | 0.019 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x8x8` | 0.023 ± 0.001 | - | 0.028 ± 0.003 | 0.029 ± 0.000 | 0.272 ± 0.383 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x128x128` | 35.539 ± 2.544 | - | 33.326 ± 0.437 | 29.089 ± 1.630 | 31.409 ± 0.876 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.001 | 0.008 ± 0.000 | 0.019 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x32x32` | 0.835 ± 0.558 | - | 0.703 ± 0.028 | 0.840 ± 0.116 | 0.953 ± 0.064 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x4x4` | 0.013 ± 0.000 | - | 0.014 ± 0.000 | 0.017 ± 0.000 | 0.031 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x8x8` | 0.054 ± 0.009 | - | 0.054 ± 0.006 | 0.054 ± 0.010 | 0.257 ± 0.269 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x2x2` | 0.309 ± 0.049 | - | 0.324 ± 0.035 | 0.582 ± 0.105 | 1.102 ± 0.136 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x4x4` | 1.873 ± 0.088 | - | 1.822 ± 0.088 | 1.833 ± 0.059 | 1.641 ± 0.114 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x8x8` | 7.614 ± 0.192 | - | 8.040 ± 0.674 | 7.341 ± 0.865 | 4.935 ± 0.348 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x128x128` | 5.371 ± 0.965 | - | 5.219 ± 0.182 | 3.440 ± 0.083 | 5.888 ± 1.785 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.001 | 0.006 ± 0.001 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x32x32` | 0.092 ± 0.012 | - | 0.095 ± 0.009 | 0.099 ± 0.018 | 0.181 ± 0.261 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x4x4` | 0.004 ± 0.001 | - | 0.006 ± 0.001 | 0.008 ± 0.009 | 0.015 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x8x8` | 0.009 ± 0.017 | - | 0.011 ± 0.002 | 0.011 ± 0.000 | 0.022 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x2x2` | 0.004 ± 0.000 | - | 0.005 ± 0.001 | 0.007 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x4x4` | 0.008 ± 0.002 | - | 0.010 ± 0.001 | 0.026 ± 0.022 | 0.019 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x8x8` | 0.023 ± 0.002 | - | 0.026 ± 0.003 | 0.023 ± 0.000 | 0.105 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x2x2` | 0.004 ± 0.001 | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.016 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x4x4` | 0.010 ± 0.001 | - | 0.011 ± 0.003 | 0.012 ± 0.000 | 0.021 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x8x8` | 0.031 ± 0.001 | - | 0.034 ± 0.004 | 0.029 ± 0.000 | 0.121 ± 0.101 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x128x128` | 44.067 ± 2.441 | - | 44.386 ± 8.053 | 27.745 ± 2.263 | 22.278 ± 4.096 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x2x2` | 0.004 ± 0.001 | - | 0.007 ± 0.001 | 0.009 ± 0.001 | 0.020 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x32x32` | 0.879 ± 0.321 | - | 0.890 ± 0.043 | 0.777 ± 0.165 | 0.809 ± 0.827 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x4x4` | 0.017 ± 0.005 | - | 0.019 ± 0.004 | 0.022 ± 0.003 | 0.032 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x8x8` | 0.061 ± 0.006 | - | 0.062 ± 0.002 | 0.054 ± 0.001 | 0.126 ± 0.081 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x2x2` | 0.228 ± 0.000 | - | 0.230 ± 0.000 | 0.261 ± 0.013 | 0.649 ± 0.034 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x4x4` | 1.703 ± 0.764 | - | 1.441 ± 0.236 | 1.444 ± 0.179 | 1.698 ± 2.478 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x8x8` | 5.646 ± 1.261 | - | 5.912 ± 1.847 | 5.819 ± 0.531 | 5.436 ± 0.113 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x128x128` | 1.001 ± 0.300 | - | 1.081 ± 0.102 | 1.200 ± 0.284 | 1.096 ± 0.097 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.002 ± 0.000 | 0.004 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x32x32` | 0.070 ± 0.003 | - | 0.071 ± 0.004 | 0.070 ± 0.003 | 0.116 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.012 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x8x8` | 0.007 ± 0.000 | - | 0.008 ± 0.001 | 0.008 ± 0.003 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x4x4` | 0.005 ± 0.000 | - | 0.006 ± 0.000 | 0.009 ± 0.002 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x8x8` | 0.015 ± 0.000 | - | 0.017 ± 0.000 | 0.019 ± 0.000 | 0.026 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.006 ± 0.001 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x4x4` | 0.006 ± 0.000 | - | 0.007 ± 0.000 | 0.010 ± 0.002 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x8x8` | 0.020 ± 0.000 | - | 0.025 ± 0.005 | 0.026 ± 0.006 | 0.031 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x128x128` | 9.519 ± 3.286 | - | 8.961 ± 1.829 | 9.104 ± 0.607 | 8.709 ± 1.075 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.001 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x32x32` | 0.774 ± 0.371 | - | 0.599 ± 0.059 | 0.564 ± 0.124 | 0.602 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x4x4` | 0.010 ± 0.000 | - | 0.011 ± 0.000 | 0.014 ± 0.003 | 0.021 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x8x8` | 0.038 ± 0.000 | - | 0.040 ± 0.007 | 0.044 ± 0.049 | 0.120 ± 0.031 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x2x2` | 0.392 ± 0.031 | - | 0.398 ± 0.029 | 0.708 ± 0.090 | 1.373 ± 0.465 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x4x4` | 1.567 ± 0.052 | - | 1.589 ± 0.482 | 2.300 ± 0.744 | 1.023 ± 0.077 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x8x8` | 6.506 ± 0.650 | - | 6.019 ± 0.829 | 10.015 ± 2.483 | 3.430 ± 0.845 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x128x128` | 1.111 ± 0.038 | - | 1.115 ± 0.073 | 1.795 ± 0.614 | 1.199 ± 0.071 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.014 ± 0.001 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x32x32` | 0.068 ± 0.010 | - | 0.072 ± 0.004 | 0.073 ± 0.002 | 0.140 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x4x4` | 0.005 ± 0.000 | - | 0.006 ± 0.001 | 0.006 ± 0.001 | 0.039 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x8x8` | 0.008 ± 0.002 | - | 0.026 ± 0.050 | 0.011 ± 0.004 | 0.017 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x2x2` | 0.007 ± 0.000 | - | 0.005 ± 0.000 | 0.016 ± 0.001 | 0.032 ± 0.041 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x4x4` | 0.006 ± 0.000 | - | 0.009 ± 0.000 | 0.011 ± 0.001 | 0.046 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x8x8` | 0.020 ± 0.002 | - | 0.021 ± 0.002 | 0.039 ± 0.061 | 0.029 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x2x2` | 0.004 ± 0.000 | - | 0.007 ± 0.002 | 0.017 ± 0.000 | 0.014 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x4x4` | 0.008 ± 0.000 | - | 0.011 ± 0.001 | 0.013 ± 0.001 | 0.017 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x8x8` | 0.025 ± 0.006 | - | 0.028 ± 0.001 | 0.138 ± 0.089 | 0.033 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x128x128` | 9.993 ± 1.467 | - | 9.463 ± 0.669 | 9.328 ± 0.396 | 4.318 ± 0.253 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x2x2` | 0.006 ± 0.000 | - | 0.009 ± 0.001 | 0.020 ± 0.000 | 0.013 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x32x32` | 0.566 ± 0.080 | - | 0.539 ± 0.046 | 0.795 ± 0.345 | 0.363 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x4x4` | 0.014 ± 0.001 | - | 0.016 ± 0.001 | 0.017 ± 0.003 | 0.067 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x8x8` | 0.044 ± 0.007 | - | 0.050 ± 0.003 | 0.103 ± 0.106 | 0.113 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x2x2` | 0.109 ± 0.001 | - | 0.112 ± 0.021 | 0.126 ± 0.003 | 0.810 ± 0.098 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x4x4` | 0.728 ± 0.020 | - | 0.731 ± 0.004 | 0.792 ± 0.398 | 1.928 ± 0.088 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x8x8` | 2.358 ± 0.040 | - | 2.352 ± 0.010 | 3.031 ± 0.482 | 6.883 ± 0.787 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x128x128` | 0.460 ± 0.018 | - | 0.450 ± 0.019 | 1.057 ± 0.220 | 1.435 ± 0.470 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.006 ± 0.012 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x32x32` | 0.025 ± 0.001 | - | 0.026 ± 0.000 | 0.030 ± 0.001 | 0.123 ± 0.049 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x8x8` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.013 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x2x2` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.005 ± 0.000 | 0.029 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x8x8` | 0.008 ± 0.000 | - | 0.009 ± 0.000 | 0.012 ± 0.000 | 0.025 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x2x2` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.005 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x4x4` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x8x8` | 0.010 ± 0.002 | - | 0.011 ± 0.000 | 0.014 ± 0.000 | 0.030 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x128x128` | 3.605 ± 0.393 | - | 3.575 ± 0.034 | 3.798 ± 0.515 | 9.185 ± 1.077 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x32x32` | 0.193 ± 0.007 | - | 0.200 ± 0.007 | 0.199 ± 0.006 | 0.665 ± 0.133 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x4x4` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.010 ± 0.000 | 0.021 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x8x8` | 0.019 ± 0.000 | - | 0.020 ± 0.000 | 0.023 ± 0.000 | 0.130 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x2x2` | 0.120 ± 0.010 | - | 0.113 ± 0.006 | 0.129 ± 0.019 | 0.599 ± 0.042 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x4x4` | 0.774 ± 0.032 | - | 0.815 ± 0.013 | 0.865 ± 0.460 | 1.836 ± 1.103 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x8x8` | 2.662 ± 0.113 | - | 2.826 ± 0.847 | 2.549 ± 0.053 | 4.209 ± 0.911 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x128x128` | 0.505 ± 0.019 | - | 0.472 ± 0.029 | 0.466 ± 0.042 | 1.415 ± 0.288 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.009 ± 0.000 | 0.005 ± 0.001 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x32x32` | 0.051 ± 0.001 | - | 0.029 ± 0.006 | 0.030 ± 0.001 | 0.171 ± 0.055 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x4x4` | 0.002 ± 0.001 | - | 0.012 ± 0.000 | 0.006 ± 0.002 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x8x8` | 0.004 ± 0.000 | - | 0.005 ± 0.001 | 0.008 ± 0.001 | 0.058 ± 0.051 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x2x2` | 0.005 ± 0.000 | - | 0.010 ± 0.000 | 0.006 ± 0.000 | 0.030 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x4x4` | 0.004 ± 0.001 | - | 0.005 ± 0.003 | 0.009 ± 0.002 | 0.016 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x8x8` | 0.011 ± 0.002 | - | 0.012 ± 0.001 | 0.012 ± 0.000 | 0.026 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x2x2` | 0.006 ± 0.001 | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x4x4` | 0.006 ± 0.001 | - | 0.006 ± 0.000 | 0.011 ± 0.004 | 0.017 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x8x8` | 0.012 ± 0.003 | - | 0.013 ± 0.003 | 0.015 ± 0.000 | 0.029 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x128x128` | 3.862 ± 0.224 | - | 3.931 ± 1.132 | 3.950 ± 0.143 | 5.430 ± 1.493 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.014 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x32x32` | 0.221 ± 0.018 | - | 0.215 ± 0.011 | 0.206 ± 0.011 | 0.530 ± 0.064 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x4x4` | 0.015 ± 0.001 | - | 0.009 ± 0.002 | 0.011 ± 0.001 | 0.024 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x8x8` | 0.024 ± 0.002 | - | 0.022 ± 0.003 | 0.024 ± 0.000 | 0.260 ± 0.225 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x2x2` | 0.092 ± 0.002 | - | 0.093 ± 0.010 | 0.159 ± 0.069 | 0.346 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x4x4` | 0.194 ± 0.002 | - | 0.196 ± 0.003 | 0.256 ± 0.098 | 0.232 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x8x8` | 0.432 ± 0.013 | - | 0.582 ± 0.085 | 1.317 ± 0.370 | 0.510 ± 0.036 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x128x128` | 0.064 ± 0.012 | - | 0.060 ± 0.005 | 0.070 ± 0.002 | 0.145 ± 0.048 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.007 ± 0.001 | 0.010 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x32x32` | 0.007 ± 0.000 | - | 0.008 ± 0.002 | 0.014 ± 0.038 | 0.049 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x64x24` | 0.009 ± 0.001 | - | 0.009 ± 0.001 | 0.013 ± 0.000 | 0.090 ± 0.200 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x8x8` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.008 ± 0.002 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x4x4` | 0.003 ± 0.000 | - | 0.003 ± 0.000 | 0.007 ± 0.000 | 0.036 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x8x8` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.010 ± 0.001 | 0.069 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.007 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x4x4` | 0.003 ± 0.000 | - | 0.003 ± 0.000 | 0.008 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x8x8` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.011 ± 0.002 | 0.070 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x128x128` | 0.458 ± 0.011 | - | 0.464 ± 0.018 | 0.895 ± 0.866 | 1.966 ± 1.213 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | 0.003 ± 0.000 | 0.008 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x32x32` | 0.038 ± 0.002 | - | 0.040 ± 0.003 | 0.045 ± 0.008 | 0.124 ± 0.182 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.001 | 0.009 ± 0.000 | 0.074 ± 0.051 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x8x8` | 0.005 ± 0.000 | - | 0.006 ± 0.000 | 0.010 ± 0.000 | 0.113 ± 0.073 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x2x2` | 0.104 ± 0.105 | - | 0.209 ± 0.168 | 0.422 ± 0.075 | 0.366 ± 0.061 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x4x4` | 0.233 ± 0.041 | - | 0.216 ± 0.012 | 0.484 ± 0.053 | 0.285 ± 0.239 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x8x8` | 0.537 ± 0.163 | - | 0.618 ± 0.111 | 0.797 ± 0.134 | 0.772 ± 0.331 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x128x128` | 0.063 ± 0.011 | - | 0.128 ± 0.168 | 0.334 ± 0.090 | 0.284 ± 0.072 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x2x2` | 0.003 ± 0.000 | - | 0.012 ± 0.000 | 0.111 ± 0.071 | 0.012 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x32x32` | 0.010 ± 0.001 | - | 0.007 ± 0.000 | 0.142 ± 0.028 | 0.091 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x4x4` | 0.004 ± 0.000 | - | 0.012 ± 0.000 | 0.083 ± 0.062 | 0.068 ± 0.066 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x64x24` | 0.025 ± 0.000 | - | 0.031 ± 0.001 | 0.103 ± 0.040 | 0.098 ± 0.068 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x8x8` | 0.004 ± 0.000 | - | 0.005 ± 0.001 | 0.113 ± 0.035 | 0.043 ± 0.089 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x2x2` | 0.008 ± 0.000 | - | 0.012 ± 0.000 | 0.145 ± 0.038 | 0.016 ± 0.029 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x4x4` | 0.008 ± 0.000 | - | 0.005 ± 0.000 | 0.088 ± 0.050 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x8x8` | 0.005 ± 0.000 | - | 0.006 ± 0.000 | 0.124 ± 0.045 | 0.052 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x2x2` | 0.008 ± 0.000 | - | 0.005 ± 0.000 | 0.101 ± 0.051 | 0.011 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x4x4` | 0.004 ± 0.000 | - | 0.015 ± 0.013 | 0.100 ± 0.038 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x8x8` | 0.004 ± 0.000 | - | 0.016 ± 0.000 | 0.146 ± 0.029 | 0.010 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x128x128` | 0.730 ± 0.637 | - | 0.848 ± 0.632 | 0.916 ± 0.108 | 1.266 ± 0.280 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x2x2` | 0.008 ± 0.000 | - | 0.013 ± 0.000 | 0.178 ± 0.061 | 0.016 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x32x32` | 0.036 ± 0.005 | - | 0.092 ± 0.001 | 0.217 ± 0.023 | 0.146 ± 0.047 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x4x4` | 0.010 ± 0.006 | - | 0.017 ± 0.007 | 0.160 ± 0.030 | 0.085 ± 0.112 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x8x8` | 0.007 ± 0.000 | - | 0.010 ± 0.002 | 0.144 ± 0.112 | 0.082 ± 0.042 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x2x2` | 0.230 ± 0.013 | - | 0.233 ± 0.013 | 0.375 ± 0.247 | 0.592 ± 0.083 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x4x4` | 0.586 ± 0.015 | - | 0.590 ± 0.013 | 0.636 ± 0.143 | 1.032 ± 0.333 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x8x8` | 1.304 ± 0.130 | - | 1.399 ± 0.139 | 1.366 ± 0.081 | 2.119 ± 0.512 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x128x128` | 0.410 ± 0.037 | - | 0.323 ± 0.040 | 0.388 ± 0.048 | 0.363 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.010 ± 0.001 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x32x32` | 0.012 ± 0.000 | - | 0.013 ± 0.000 | 0.014 ± 0.001 | 0.107 ± 0.033 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.003 ± 0.000 | 0.013 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x64x24` | 0.014 ± 0.000 | - | 0.016 ± 0.004 | 0.017 ± 0.001 | 0.067 ± 0.034 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x8x8` | 0.003 ± 0.001 | - | 0.004 ± 0.001 | 0.004 ± 0.001 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.004 ± 0.000 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x8x8` | 0.006 ± 0.001 | - | 0.008 ± 0.001 | 0.006 ± 0.000 | 0.050 ± 0.088 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.012 ± 0.010 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x4x4` | 0.004 ± 0.001 | - | 0.006 ± 0.000 | 0.006 ± 0.001 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x8x8` | 0.007 ± 0.001 | - | 0.008 ± 0.001 | 0.008 ± 0.002 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x128x128` | 2.810 ± 0.719 | - | 2.844 ± 0.182 | 3.174 ± 0.119 | 3.954 ± 5.610 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.014 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x32x32` | 0.091 ± 0.007 | - | 0.091 ± 0.009 | 0.102 ± 0.011 | 0.236 ± 0.156 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x4x4` | 0.006 ± 0.002 | - | 0.008 ± 0.000 | 0.007 ± 0.000 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x8x8` | 0.014 ± 0.000 | - | 0.015 ± 0.002 | 0.012 ± 0.000 | 0.025 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x2x2` | 0.240 ± 0.040 | - | 0.236 ± 0.070 | 0.412 ± 0.282 | 0.355 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x4x4` | 0.601 ± 0.041 | - | 0.600 ± 0.044 | 1.053 ± 0.272 | 0.783 ± 0.076 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x8x8` | 1.583 ± 0.133 | - | 1.581 ± 0.807 | 3.094 ± 1.034 | 1.227 ± 0.504 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x128x128` | 0.371 ± 0.168 | - | 0.418 ± 0.063 | 0.584 ± 0.188 | 0.425 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.037 ± 0.037 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x32x32` | 0.016 ± 0.021 | - | 0.017 ± 0.004 | 0.061 ± 0.013 | 0.110 ± 0.042 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x4x4` | 0.003 ± 0.000 | - | 0.012 ± 0.001 | 0.050 ± 0.098 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x64x24` | 0.019 ± 0.003 | - | 0.020 ± 0.001 | 0.066 ± 0.014 | 0.084 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x8x8` | 0.004 ± 0.000 | - | 0.006 ± 0.001 | 0.058 ± 0.026 | 0.014 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.001 | 0.040 ± 0.056 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x4x4` | 0.010 ± 0.000 | - | 0.006 ± 0.001 | 0.033 ± 0.016 | 0.013 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x8x8` | 0.007 ± 0.001 | - | 0.013 ± 0.045 | 0.040 ± 0.025 | 0.017 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.001 | 0.022 ± 0.001 | 0.024 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x4x4` | 0.005 ± 0.000 | - | 0.007 ± 0.001 | 0.085 ± 0.090 | 0.015 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x8x8` | 0.009 ± 0.000 | - | 0.011 ± 0.001 | 0.049 ± 0.071 | 0.019 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x128x128` | 2.984 ± 0.329 | - | 2.906 ± 0.215 | 5.119 ± 2.423 | 2.759 ± 0.189 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x2x2` | 0.004 ± 0.000 | - | 0.005 ± 0.001 | 0.025 ± 0.015 | 0.015 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x32x32` | 0.101 ± 0.016 | - | 0.280 ± 0.174 | 0.335 ± 0.103 | 0.316 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x4x4` | 0.007 ± 0.000 | - | 0.009 ± 0.001 | 0.081 ± 0.091 | 0.018 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x8x8` | 0.014 ± 0.001 | - | 0.016 ± 0.001 | 0.064 ± 0.080 | 0.025 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.134 ± 0.001 | - | 0.121 ± 0.001 | 0.105 ± 0.010 | 0.171 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.222 ± 0.001 | - | 0.212 ± 0.002 | 0.179 ± 0.083 | 0.370 ± 0.136 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.370 ± 0.001 | - | 0.383 ± 0.000 | 0.370 ± 0.018 | 0.529 ± 0.065 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x128x128,rhs=1` | 0.047 ± 0.001 | - | 0.054 ± 0.001 | 0.068 ± 0.005 | 0.164 ± 0.322 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.005 ± 0.010 | 0.010 ± 0.002 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x32x32,rhs=1` | 0.004 ± 0.000 | - | 0.008 ± 0.000 | 0.015 ± 0.076 | 0.069 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x4x4,rhs=1` | 0.001 ± 0.000 | - | 0.004 ± 0.001 | 0.009 ± 0.003 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x8x8,rhs=1` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.009 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.001 | 0.009 ± 0.000 | 0.033 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x8x8,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.039 ± 0.075 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.011 ± 0.002 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x8x8,rhs=1` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.010 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x128x128,rhs=1` | 0.362 ± 0.001 | - | 0.387 ± 0.001 | 0.451 ± 0.131 | 0.920 ± 0.228 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x32x32,rhs=1` | 0.027 ± 0.000 | - | 0.033 ± 0.000 | 0.037 ± 0.004 | 0.308 ± 0.134 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.010 ± 0.002 | 0.011 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x8x8,rhs=1` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.013 ± 0.003 | 0.016 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.174 ± 0.015 | - | 0.162 ± 0.018 | 0.157 ± 0.102 | 0.194 ± 0.031 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.212 ± 0.040 | - | 0.197 ± 0.037 | 0.314 ± 0.474 | 0.431 ± 0.298 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.459 ± 0.147 | - | 0.485 ± 0.114 | 0.450 ± 0.231 | 0.949 ± 0.215 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x128x128,rhs=1` | 0.055 ± 0.017 | - | 0.064 ± 0.003 | 0.069 ± 0.004 | 0.393 ± 0.071 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x2x2,rhs=1` | 0.006 ± 0.007 | - | 0.008 ± 0.011 | 0.013 ± 0.002 | 0.040 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x32x32,rhs=1` | 0.006 ± 0.000 | - | 0.012 ± 0.000 | 0.041 ± 0.002 | 0.060 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.008 ± 0.001 | 0.012 ± 0.002 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x8x8,rhs=1` | 0.003 ± 0.001 | - | 0.008 ± 0.002 | 0.011 ± 0.001 | 0.030 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x2x2,rhs=1` | 0.006 ± 0.000 | - | 0.007 ± 0.005 | 0.011 ± 0.002 | 0.033 ± 0.046 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x4x4,rhs=1` | 0.003 ± 0.001 | - | 0.007 ± 0.002 | 0.011 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x8x8,rhs=1` | 0.003 ± 0.000 | - | 0.023 ± 0.046 | 0.013 ± 0.003 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x2x2,rhs=1` | 0.002 ± 0.001 | - | 0.021 ± 0.000 | 0.013 ± 0.001 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.008 ± 0.001 | 0.012 ± 0.003 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x8x8,rhs=1` | 0.009 ± 0.000 | - | 0.009 ± 0.001 | 0.012 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x128x128,rhs=1` | 0.443 ± 0.039 | - | 0.513 ± 0.075 | 0.506 ± 0.271 | 0.617 ± 0.287 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x2x2,rhs=1` | 0.004 ± 0.004 | - | 0.009 ± 0.001 | 0.019 ± 0.004 | 0.012 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x32x32,rhs=1` | 0.034 ± 0.004 | - | 0.042 ± 0.006 | 0.056 ± 0.003 | 0.238 ± 0.176 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.009 ± 0.000 | 0.029 ± 0.008 | 0.015 ± 0.041 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x8x8,rhs=1` | 0.005 ± 0.000 | - | 0.011 ± 0.002 | 0.059 ± 0.056 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x2x2` | 1.434 ± 0.208 | - | 1.400 ± 0.353 | 1.205 ± 0.046 | 6.037 ± 1.302 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x4x4` | 5.345 ± 0.704 | - | 5.097 ± 2.947 | 4.371 ± 0.083 | 7.948 ± 2.177 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x8x8` | 17.882 ± 2.590 | - | 18.918 ± 3.926 | 19.132 ± 0.865 | 20.926 ± 1.801 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x128x128` | 5.955 ± 0.143 | - | 5.858 ± 0.412 | 5.314 ± 0.331 | 6.309 ± 0.316 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x2x2` | 0.009 ± 0.004 | - | 0.014 ± 0.000 | 0.007 ± 0.000 | 0.024 ± 0.039 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x32x32` | 0.183 ± 0.009 | - | 0.178 ± 0.007 | 0.236 ± 0.076 | 0.345 ± 0.133 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x4x4` | 0.007 ± 0.001 | - | 0.007 ± 0.001 | 0.033 ± 0.001 | 0.028 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x64x24` | 0.158 ± 0.001 | - | 0.159 ± 0.001 | 0.325 ± 0.092 | 0.243 ± 0.046 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x8x8` | 0.021 ± 0.001 | - | 0.022 ± 0.001 | 0.065 ± 0.048 | 0.038 ± 0.031 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x2x2` | 0.014 ± 0.000 | - | 0.020 ± 0.001 | 0.009 ± 0.000 | 0.067 ± 0.065 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x4x4` | 0.017 ± 0.003 | - | 0.018 ± 0.001 | 0.057 ± 0.001 | 0.031 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x8x8` | 0.052 ± 0.007 | - | 0.056 ± 0.005 | 0.137 ± 0.001 | 0.065 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x2x2` | 0.018 ± 0.000 | - | 0.023 ± 0.000 | 0.010 ± 0.000 | 0.049 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x4x4` | 0.022 ± 0.003 | - | 0.020 ± 0.003 | 0.068 ± 0.001 | 0.038 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x8x8` | 0.074 ± 0.003 | - | 0.073 ± 0.005 | 0.177 ± 0.001 | 0.083 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x128x128` | 45.257 ± 1.246 | - | 46.451 ± 4.829 | 53.310 ± 8.779 | 48.899 ± 2.920 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x2x2` | 0.029 ± 0.001 | - | 0.021 ± 0.016 | 0.015 ± 0.000 | 0.041 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x32x32` | 1.406 ± 0.088 | - | 1.411 ± 0.066 | 1.682 ± 0.256 | 1.898 ± 0.232 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x4x4` | 0.041 ± 0.003 | - | 0.044 ± 0.001 | 0.120 ± 0.003 | 0.075 ± 0.049 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x8x8` | 0.168 ± 0.007 | - | 0.172 ± 0.032 | 0.326 ± 0.009 | 0.192 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x2x2` | 2.034 ± 0.762 | - | 2.050 ± 1.335 | 1.280 ± 0.065 | 3.701 ± 1.133 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x4x4` | 7.719 ± 2.015 | - | 6.989 ± 0.923 | 5.194 ± 1.359 | 5.072 ± 2.916 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x8x8` | 19.457 ± 8.152 | - | 16.827 ± 0.774 | 18.795 ± 1.369 | 12.431 ± 0.550 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x128x128` | 6.203 ± 0.505 | - | 6.245 ± 0.632 | 5.423 ± 1.829 | 5.717 ± 0.107 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x2x2` | 0.012 ± 0.001 | - | 0.019 ± 0.001 | 0.008 ± 0.000 | 0.021 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x32x32` | 0.192 ± 0.009 | - | 0.207 ± 0.003 | 0.203 ± 0.011 | 0.564 ± 0.326 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x4x4` | 0.023 ± 0.006 | - | 0.029 ± 0.001 | 0.011 ± 0.000 | 0.023 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x64x24` | 0.201 ± 0.020 | - | 0.209 ± 0.121 | 0.194 ± 0.007 | 0.352 ± 0.078 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x8x8` | 0.022 ± 0.003 | - | 0.025 ± 0.007 | 0.028 ± 0.006 | 0.035 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x2x2` | 0.018 ± 0.000 | - | 0.009 ± 0.009 | 0.010 ± 0.001 | 0.022 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x4x4` | 0.044 ± 0.011 | - | 0.035 ± 0.021 | 0.021 ± 0.005 | 0.038 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x8x8` | 0.127 ± 0.037 | - | 0.050 ± 0.010 | 0.057 ± 0.006 | 0.069 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x2x2` | 0.009 ± 0.002 | - | 0.011 ± 0.012 | 0.011 ± 0.000 | 0.025 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x4x4` | 0.024 ± 0.008 | - | 0.063 ± 0.003 | 0.024 ± 0.002 | 0.043 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x8x8` | 0.076 ± 0.043 | - | 0.077 ± 0.015 | 0.076 ± 0.005 | 0.095 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x128x128` | 62.166 ± 8.559 | - | 63.494 ± 4.495 | 42.070 ± 6.197 | 24.995 ± 4.828 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x2x2` | 0.015 ± 0.007 | - | 0.014 ± 0.008 | 0.016 ± 0.000 | 0.046 ± 0.135 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x32x32` | 1.590 ± 0.040 | - | 1.612 ± 0.646 | 1.650 ± 0.303 | 1.261 ± 0.083 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x4x4` | 0.036 ± 0.002 | - | 0.048 ± 0.008 | 0.040 ± 0.002 | 0.233 ± 0.063 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x8x8` | 0.165 ± 0.030 | - | 0.132 ± 0.006 | 0.144 ± 0.005 | 0.256 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x2x2` | 0.589 ± 0.027 | - | 0.576 ± 0.021 | 0.607 ± 0.094 | 2.549 ± 0.868 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x4x4` | 2.682 ± 0.027 | - | 2.694 ± 0.037 | 3.672 ± 0.949 | 5.106 ± 0.775 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x8x8` | 10.541 ± 0.794 | - | 11.224 ± 0.928 | 12.532 ± 1.132 | 18.492 ± 4.176 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x128x128` | 1.466 ± 0.027 | - | 1.544 ± 0.097 | 1.887 ± 0.544 | 2.688 ± 1.368 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x2x2` | 0.004 ± 0.004 | - | 0.003 ± 0.000 | 0.006 ± 0.001 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x32x32` | 0.119 ± 0.000 | - | 0.120 ± 0.001 | 0.126 ± 0.013 | 0.213 ± 0.063 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.008 ± 0.001 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x64x24` | 0.105 ± 0.003 | - | 0.105 ± 0.022 | 0.108 ± 0.001 | 0.220 ± 0.198 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x8x8` | 0.012 ± 0.000 | - | 0.013 ± 0.000 | 0.020 ± 0.001 | 0.028 ± 0.070 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x2x2` | 0.004 ± 0.002 | - | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.018 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x4x4` | 0.009 ± 0.000 | - | 0.010 ± 0.000 | 0.013 ± 0.000 | 0.024 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x8x8` | 0.029 ± 0.001 | - | 0.030 ± 0.001 | 0.035 ± 0.008 | 0.046 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.001 | 0.008 ± 0.000 | 0.020 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x4x4` | 0.012 ± 0.000 | - | 0.014 ± 0.001 | 0.016 ± 0.000 | 0.029 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x8x8` | 0.039 ± 0.000 | - | 0.040 ± 0.000 | 0.051 ± 0.007 | 0.206 ± 0.213 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x128x128` | 12.547 ± 0.688 | - | 13.483 ± 4.139 | 13.051 ± 2.955 | 16.173 ± 0.996 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x2x2` | 0.008 ± 0.001 | - | 0.007 ± 0.002 | 0.010 ± 0.000 | 0.027 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x32x32` | 0.941 ± 0.017 | - | 0.962 ± 0.044 | 1.081 ± 0.100 | 1.252 ± 0.336 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x4x4` | 0.023 ± 0.000 | - | 0.024 ± 0.000 | 0.027 ± 0.002 | 0.053 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x8x8` | 0.078 ± 0.001 | - | 0.079 ± 0.001 | 0.089 ± 0.016 | 0.174 ± 0.091 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x2x2` | 0.635 ± 0.050 | - | 1.417 ± 0.932 | 0.605 ± 0.025 | 2.559 ± 0.382 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x4x4` | 3.107 ± 1.633 | - | 2.882 ± 0.189 | 2.853 ± 0.365 | 3.033 ± 1.308 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x8x8` | 16.863 ± 6.344 | - | 16.775 ± 2.222 | 11.360 ± 0.463 | 8.430 ± 0.741 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x128x128` | 2.537 ± 1.484 | - | 2.447 ± 0.491 | 1.600 ± 0.143 | 1.837 ± 2.500 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x2x2` | 0.003 ± 0.001 | - | 0.005 ± 0.000 | 0.007 ± 0.001 | 0.016 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x32x32` | 0.135 ± 0.029 | - | 0.145 ± 0.039 | 0.178 ± 0.042 | 0.241 ± 0.100 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x4x4` | 0.007 ± 0.000 | - | 0.008 ± 0.000 | 0.009 ± 0.001 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x64x24` | 0.271 ± 0.032 | - | 0.294 ± 0.025 | 0.100 ± 0.004 | 0.150 ± 0.051 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x8x8` | 0.013 ± 0.000 | - | 0.043 ± 0.001 | 0.017 ± 0.005 | 0.025 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x2x2` | 0.005 ± 0.000 | - | 0.006 ± 0.001 | 0.008 ± 0.002 | 0.018 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x4x4` | 0.012 ± 0.002 | - | 0.036 ± 0.000 | 0.014 ± 0.003 | 0.029 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x8x8` | 0.030 ± 0.002 | - | 0.033 ± 0.006 | 0.036 ± 0.002 | 0.053 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x2x2` | 0.005 ± 0.000 | - | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.021 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x4x4` | 0.038 ± 0.000 | - | 0.044 ± 0.000 | 0.017 ± 0.003 | 0.083 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x8x8` | 0.040 ± 0.001 | - | 0.119 ± 0.017 | 0.045 ± 0.001 | 0.067 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x128x128` | 20.010 ± 3.975 | - | 18.912 ± 7.170 | 12.774 ± 0.518 | 9.090 ± 0.453 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x2x2` | 0.009 ± 0.000 | - | 0.011 ± 0.003 | 0.011 ± 0.001 | 0.027 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x32x32` | 1.507 ± 0.547 | - | 1.624 ± 0.893 | 0.999 ± 0.065 | 0.760 ± 0.046 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x4x4` | 0.068 ± 0.001 | - | 0.075 ± 0.002 | 0.033 ± 0.006 | 0.050 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x8x8` | 0.091 ± 0.008 | - | 0.223 ± 0.004 | 0.087 ± 0.002 | 0.190 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x2x2` | 0.202 ± 0.000 | - | unsupported | 0.682 ± 0.362 | 1.390 ± 0.105 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x4x4` | 1.294 ± 0.015 | - | unsupported | 2.743 ± 0.839 | 3.030 ± 1.782 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x8x8` | 3.635 ± 0.083 | - | unsupported | 6.571 ± 1.553 | 5.491 ± 0.388 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x128x128` | 0.761 ± 0.016 | - | unsupported | 1.394 ± 0.776 | 0.986 ± 0.234 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | unsupported | 0.019 ± 0.013 | 0.031 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x32x32` | 0.036 ± 0.001 | - | unsupported | 0.076 ± 0.098 | 0.047 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | unsupported | 0.019 ± 0.001 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x64x24` | 0.030 ± 0.000 | - | unsupported | 0.036 ± 0.009 | 0.042 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x8x8` | 0.004 ± 0.000 | - | unsupported | 0.009 ± 0.001 | 0.015 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | unsupported | 0.018 ± 0.000 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x4x4` | 0.005 ± 0.000 | - | unsupported | 0.027 ± 0.001 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x8x8` | 0.011 ± 0.000 | - | unsupported | 0.019 ± 0.003 | 0.024 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | unsupported | 0.019 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x4x4` | 0.006 ± 0.000 | - | unsupported | 0.030 ± 0.000 | 0.020 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x8x8` | 0.015 ± 0.000 | - | unsupported | 0.024 ± 0.017 | 0.029 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x128x128` | 6.086 ± 0.062 | - | unsupported | 8.455 ± 3.524 | 6.730 ± 0.397 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | unsupported | 0.021 ± 0.000 | 0.019 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x32x32` | 0.290 ± 0.007 | - | unsupported | 0.365 ± 0.122 | 0.324 ± 0.030 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x4x4` | 0.011 ± 0.000 | - | unsupported | 0.043 ± 0.019 | 0.030 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x8x8` | 0.029 ± 0.000 | - | unsupported | 0.043 ± 0.165 | 0.049 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x2x2` | 0.246 ± 0.020 | - | unsupported | 0.265 ± 0.004 | 1.561 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x4x4` | 2.370 ± 0.586 | - | unsupported | 1.634 ± 0.743 | 2.085 ± 0.459 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x8x8` | 5.986 ± 1.826 | - | unsupported | 5.114 ± 0.689 | 3.323 ± 0.484 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x128x128` | 0.853 ± 0.185 | - | unsupported | 1.259 ± 0.640 | 0.963 ± 0.038 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | unsupported | 0.007 ± 0.000 | 0.012 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x32x32` | 0.039 ± 0.005 | - | unsupported | 0.048 ± 0.008 | 0.055 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x4x4` | 0.004 ± 0.000 | - | unsupported | 0.008 ± 0.001 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x64x24` | 0.038 ± 0.003 | - | unsupported | 0.042 ± 0.004 | 0.048 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x8x8` | 0.005 ± 0.000 | - | unsupported | 0.011 ± 0.000 | 0.016 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x2x2` | 0.002 ± 0.000 | - | unsupported | 0.007 ± 0.001 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x4x4` | 0.006 ± 0.000 | - | unsupported | 0.011 ± 0.001 | 0.017 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x8x8` | 0.012 ± 0.000 | - | unsupported | 0.017 ± 0.000 | 0.025 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x2x2` | 0.003 ± 0.000 | - | unsupported | 0.008 ± 0.001 | 0.023 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x4x4` | 0.009 ± 0.000 | - | unsupported | 0.013 ± 0.001 | 0.021 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x8x8` | 0.018 ± 0.005 | - | unsupported | 0.025 ± 0.003 | 0.031 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x128x128` | 10.027 ± 3.119 | - | unsupported | 8.081 ± 0.959 | 4.609 ± 0.803 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x2x2` | 0.003 ± 0.000 | - | unsupported | 0.009 ± 0.001 | 0.023 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x32x32` | 0.611 ± 0.185 | - | unsupported | 0.386 ± 0.167 | 0.370 ± 0.044 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x4x4` | 0.014 ± 0.002 | - | unsupported | 0.019 ± 0.001 | 0.032 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x8x8` | 0.044 ± 0.082 | - | unsupported | 0.040 ± 0.007 | 0.049 ± 0.005 | - | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x16x16,rhs=1` | 0.759 ± 0.002 | - | unsupported | 1.250 ± 0.203 | 1.209 ± 0.104 | 3.232 ± 0.319 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x2x2,rhs=1` | 0.055 ± 0.001 | - | unsupported | 0.066 ± 0.012 | 0.233 ± 0.021 | 0.106 ± 0.019 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x4x4,rhs=1` | 0.109 ± 0.002 | - | unsupported | 0.130 ± 0.023 | 0.204 ± 0.012 | 0.249 ± 0.033 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x8x8,rhs=1` | 0.252 ± 0.002 | - | unsupported | 0.284 ± 0.015 | 0.391 ± 0.014 | 0.786 ± 0.049 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x16x16,rhs=1` | 1.057 ± 0.212 | - | unsupported | 1.000 ± 0.101 | 0.661 ± 0.093 | 6.458 ± 3.676 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x2x2,rhs=1` | 0.071 ± 0.002 | - | unsupported | 0.280 ± 0.070 | 0.147 ± 0.036 | 0.099 ± 0.024 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x4x4,rhs=1` | 0.125 ± 0.008 | - | unsupported | 0.229 ± 0.041 | 0.267 ± 0.076 | 0.232 ± 0.050 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x8x8,rhs=1` | 0.320 ± 0.008 | - | unsupported | 0.402 ± 0.041 | 0.452 ± 0.157 | 0.913 ± 1.037 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.258 ± 0.006 | - | unsupported | 0.212 ± 0.027 | 0.623 ± 0.397 | 0.487 ± 0.111 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.068 ± 0.005 | - | unsupported | 0.062 ± 0.002 | 0.079 ± 0.001 | 0.068 ± 0.004 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.069 ± 0.010 | - | unsupported | 0.067 ± 0.014 | 0.203 ± 0.249 | 0.094 ± 0.001 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.101 ± 0.006 | - | unsupported | 0.088 ± 0.009 | 0.254 ± 0.227 | 0.230 ± 0.022 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.256 ± 0.023 | - | unsupported | 0.209 ± 0.116 | 0.536 ± 0.185 | 0.552 ± 0.054 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.060 ± 0.001 | - | unsupported | 0.141 ± 0.016 | 0.105 ± 0.026 | 0.078 ± 0.003 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.068 ± 0.015 | - | unsupported | 0.058 ± 0.006 | 0.132 ± 0.102 | 0.114 ± 0.008 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.117 ± 0.041 | - | unsupported | 0.091 ± 0.018 | 0.221 ± 0.171 | 0.291 ± 0.038 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.130 ± 0.008 | - | 0.131 ± 0.011 | 0.091 ± 0.007 | 0.349 ± 0.169 | 0.305 ± 0.027 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.053 ± 0.002 | - | 0.050 ± 0.008 | 0.024 ± 0.004 | 0.085 ± 0.129 | 0.040 ± 0.003 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.051 ± 0.005 | - | 0.056 ± 0.006 | 0.027 ± 0.007 | 0.034 ± 0.010 | 0.057 ± 0.010 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.064 ± 0.006 | - | 0.067 ± 0.003 | 0.039 ± 0.029 | 0.059 ± 0.003 | 0.139 ± 0.020 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.136 ± 0.045 | - | 0.134 ± 0.002 | 0.102 ± 0.011 | 0.351 ± 0.085 | 0.291 ± 0.050 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.059 ± 0.001 | - | 0.060 ± 0.011 | 0.031 ± 0.016 | 0.029 ± 0.005 | 0.046 ± 0.008 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.054 ± 0.001 | - | 0.052 ± 0.010 | 0.032 ± 0.006 | 0.032 ± 0.001 | 0.059 ± 0.010 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.069 ± 0.060 | - | 0.079 ± 0.089 | 0.037 ± 0.005 | 0.091 ± 0.042 | 0.126 ± 0.005 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 1 | `1536x1536` | 8.508 ± 0.903 | - | 11.677 ± 1.299 | 30.536 ± 3.769 | 21.475 ± 3.868 | 49.339 ± 6.367 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 4 | `1536x1536` | 11.614 ± 2.533 | - | 12.005 ± 0.213 | 21.586 ± 2.641 | 18.559 ± 7.194 | 426.530 ± 437.016 | - |
| cpu/linalg_uncovered | `det` | f64 | 1 | `1024x1024` | 14.620 ± 4.927 | - | 18.650 ± 4.308 | 8.920 ± 0.762 | 9.064 ± 0.402 | 24.680 ± 4.944 | - |
| cpu/linalg_uncovered | `det` | f64 | 4 | `1024x1024` | 12.121 ± 3.065 | - | 12.242 ± 4.877 | 8.890 ± 2.978 | 9.342 ± 1.233 | 266.062 ± 42.422 | - |
| cpu/linalg_uncovered | `eig` | f64 | 1 | `160x160` | 6.506 ± 0.145 | - | 6.830 ± 0.199 | 9.359 ± 2.300 | 9.829 ± 1.760 | 8.772 ± 0.582 | - |
| cpu/linalg_uncovered | `eig` | f64 | 4 | `160x160` | 7.674 ± 1.301 | - | 7.892 ± 0.785 | 7.447 ± 2.802 | 10.399 ± 0.933 | 324.209 ± 32.507 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 1 | `192x192` | 7.459 ± 0.131 | - | 7.358 ± 0.083 | 9.968 ± 1.419 | 8.682 ± 0.770 | 9.492 ± 0.326 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 4 | `192x192` | 9.046 ± 1.228 | - | 9.331 ± 2.041 | 9.082 ± 1.980 | 8.270 ± 0.693 | 272.618 ± 13.780 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 1 | `512x512` | 21.637 ± 2.021 | - | 22.473 ± 1.773 | 23.535 ± 2.043 | 36.677 ± 3.753 | 16.388 ± 0.718 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 4 | `512x512` | 23.526 ± 1.226 | - | 23.818 ± 0.986 | 24.753 ± 1.056 | 29.158 ± 0.479 | 639.460 ± 28.486 | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `256x128` | 0.486 ± 0.098 | 0.476 ± 0.166 | unsupported | 0.348 ± 0.279 | 0.368 ± 0.008 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `768x384` | 3.587 ± 0.199 | 3.563 ± 0.155 | unsupported | 4.468 ± 0.439 | 4.984 ± 0.313 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `256x128` | 0.308 ± 0.012 | 0.334 ± 0.030 | unsupported | 0.356 ± 0.040 | 0.541 ± 0.026 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `768x384` | 3.939 ± 0.629 | 7.078 ± 2.086 | unsupported | 5.431 ± 0.777 | 6.290 ± 0.539 | - | - |
| cpu/linalg_uncovered | `inv` | f64 | 1 | `768x768` | 22.347 ± 18.477 | - | 13.797 ± 3.898 | 11.480 ± 3.491 | 12.369 ± 2.162 | 33.537 ± 2.283 | - |
| cpu/linalg_uncovered | `inv` | f64 | 4 | `768x768` | 17.326 ± 3.239 | - | 14.199 ± 11.597 | 12.360 ± 4.252 | 16.345 ± 8.392 | 380.957 ± 56.958 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 1 | `768x384,rhs=16` | - | 9.124 ± 0.798 | 9.293 ± 0.355 | 5.395 ± 0.154 | 41.132 ± 4.917 | 9.533 ± 1.222 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 4 | `768x384,rhs=16` | - | 16.487 ± 4.274 | 16.364 ± 6.264 | 6.704 ± 5.298 | 44.555 ± 3.652 | 149.636 ± 38.174 | - |
| cpu/linalg_uncovered | `lu` | f64 | 1 | `1024x1024` | 9.052 ± 5.244 | - | 9.404 ± 0.280 | 14.722 ± 6.625 | 16.601 ± 2.429 | 40.092 ± 19.648 | - |
| cpu/linalg_uncovered | `lu` | f64 | 4 | `1024x1024` | 14.653 ± 6.170 | - | 10.690 ± 4.218 | 12.750 ± 2.631 | 14.071 ± 0.756 | 249.198 ± 65.929 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 1 | `2048x2048` | 2.600 ± 0.683 | - | 2.021 ± 1.212 | 2.238 ± 0.095 | 7.047 ± 0.496 | 6.558 ± 0.478 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 4 | `2048x2048` | 1.112 ± 0.207 | - | 0.934 ± 0.084 | 2.421 ± 0.383 | 6.402 ± 1.758 | 6.635 ± 3.187 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 1 | `512x256` | 16.528 ± 1.572 | - | 17.028 ± 6.333 | 11.308 ± 0.667 | 14.129 ± 4.118 | 21.989 ± 1.305 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 4 | `512x256` | 17.717 ± 6.391 | - | 16.376 ± 3.590 | 13.639 ± 1.913 | 19.498 ± 6.870 | 1038.381 ± 330.692 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 1 | `512x256` | 17.925 ± 2.442 | - | 13.168 ± 3.492 | 11.288 ± 0.388 | 14.345 ± 0.881 | 21.779 ± 1.728 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 4 | `512x256` | 16.294 ± 4.306 | - | 19.778 ± 6.985 | 13.264 ± 1.793 | 18.635 ± 3.847 | 1011.824 ± 612.099 | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `256x128` | 1.468 ± 0.127 | 1.467 ± 0.114 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `768x384` | 14.104 ± 2.094 | 14.104 ± 1.522 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `256x128` | 1.511 ± 0.105 | 1.508 ± 0.167 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `768x384` | 20.859 ± 2.253 | 19.938 ± 2.674 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `256x128` | 1.455 ± 0.058 | 1.410 ± 0.068 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `768x384` | 12.732 ± 1.146 | 13.754 ± 2.459 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `256x128` | 1.487 ± 0.012 | 1.518 ± 0.045 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `768x384` | 18.146 ± 5.897 | 15.681 ± 4.046 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 1 | `1024x1024` | 14.143 ± 7.184 | - | 10.454 ± 2.005 | 11.162 ± 3.460 | 9.244 ± 0.719 | 24.650 ± 2.347 | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 4 | `1024x1024` | 11.847 ± 2.199 | - | 14.878 ± 2.766 | 11.718 ± 3.564 | 16.143 ± 5.484 | 455.775 ± 267.130 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 1 | `768x384` | - | 58.922 ± 7.542 | 60.706 ± 5.680 | 38.820 ± 2.769 | 43.030 ± 3.170 | 88.800 ± 9.411 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 4 | `768x384` | - | 48.882 ± 10.967 | 47.649 ± 3.138 | 54.002 ± 11.897 | 50.208 ± 11.041 | 1780.063 ± 44.191 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 1 | `4096x4096,rhs=64` | 16.535 ± 0.583 | - | 17.152 ± 0.406 | 18.990 ± 3.699 | 77.887 ± 1.432 | 37.191 ± 3.503 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 4 | `4096x4096,rhs=64` | 23.080 ± 11.078 | - | 25.744 ± 5.184 | 22.208 ± 7.370 | 59.858 ± 19.322 | 27.620 ± 5.797 | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `1024x64` | 0.182 ± 0.029 | 0.175 ± 0.010 | unsupported | 0.102 ± 0.073 | 0.046 ± 0.039 | - | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `4096x256` | 3.875 ± 0.908 | 2.816 ± 0.402 | unsupported | 0.967 ± 0.936 | 0.517 ± 0.028 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `1024x64` | 0.227 ± 0.027 | 0.235 ± 0.035 | unsupported | 0.068 ± 0.034 | 0.084 ± 0.034 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `4096x256` | 2.714 ± 1.312 | 2.598 ± 0.278 | unsupported | 0.773 ± 0.054 | 0.629 ± 0.217 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `1024x64` | 0.096 ± 0.038 | 0.101 ± 0.002 | unsupported | 0.118 ± 0.101 | 0.242 ± 0.023 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `4096x256` | 1.661 ± 0.486 | 2.012 ± 0.220 | unsupported | 1.714 ± 0.128 | 2.515 ± 0.129 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `1024x64` | 0.131 ± 0.011 | 0.143 ± 0.009 | unsupported | 0.088 ± 0.015 | 0.310 ± 0.072 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `4096x256` | 1.234 ± 0.109 | 1.226 ± 0.493 | unsupported | 1.156 ± 0.865 | 2.391 ± 0.287 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `1024x64` | 0.266 ± 0.006 | 0.270 ± 0.008 | unsupported | 0.359 ± 0.146 | 0.100 ± 0.006 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `4096x256` | 6.527 ± 1.982 | 5.169 ± 0.789 | unsupported | 5.957 ± 0.368 | 1.059 ± 0.144 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `1024x64` | 0.288 ± 0.023 | 0.295 ± 0.033 | unsupported | 0.200 ± 0.040 | 0.245 ± 0.098 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `4096x256` | 3.914 ± 1.341 | 4.033 ± 1.698 | unsupported | 2.487 ± 0.340 | 0.983 ± 0.273 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `1024x64` | 0.046 ± 0.001 | 0.064 ± 0.001 | unsupported | 0.060 ± 0.018 | 0.090 ± 0.004 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `4096x256` | 0.844 ± 0.185 | 0.999 ± 0.670 | unsupported | 0.682 ± 0.054 | 0.913 ± 0.404 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `1024x64` | 0.069 ± 0.010 | 0.093 ± 0.014 | unsupported | 0.051 ± 0.003 | 0.182 ± 0.022 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `4096x256` | 1.402 ± 0.238 | 1.472 ± 0.391 | unsupported | 0.425 ± 0.040 | 0.989 ± 0.302 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `1024x64` | 0.090 ± 0.002 | 0.104 ± 0.001 | unsupported | 0.137 ± 0.049 | 0.124 ± 0.009 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `4096x256` | 1.299 ± 0.328 | 1.461 ± 0.384 | unsupported | 1.589 ± 0.391 | 1.618 ± 0.269 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `1024x64` | 0.118 ± 0.014 | 0.136 ± 0.020 | unsupported | 0.090 ± 0.099 | 0.257 ± 0.061 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `4096x256` | 2.624 ± 0.356 | 2.738 ± 0.127 | unsupported | 0.772 ± 0.017 | 1.668 ± 0.412 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `1024x64` | 0.129 ± 0.009 | 0.144 ± 0.002 | unsupported | unsupported | 0.139 ± 0.035 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `4096x256` | 1.907 ± 0.437 | 2.170 ± 0.113 | unsupported | unsupported | 2.245 ± 0.530 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `1024x64` | 0.169 ± 0.010 | 0.181 ± 0.021 | unsupported | unsupported | 0.306 ± 0.023 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `4096x256` | 3.377 ± 0.713 | 2.984 ± 0.142 | unsupported | unsupported | 2.267 ± 0.801 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `1024x64` | 0.012 ± 0.001 | 0.016 ± 0.000 | unsupported | 0.009 ± 0.003 | 0.099 ± 0.118 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `4096x256` | 0.112 ± 0.006 | 0.114 ± 0.004 | unsupported | 0.072 ± 0.013 | 0.253 ± 0.070 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `1024x64` | 0.016 ± 0.001 | 0.021 ± 0.001 | unsupported | 0.041 ± 0.001 | 0.086 ± 0.014 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `4096x256` | 0.235 ± 0.035 | 0.298 ± 0.050 | unsupported | 0.059 ± 0.014 | 0.442 ± 0.202 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `1024x64` | 0.024 ± 0.001 | 0.036 ± 0.002 | unsupported | 0.035 ± 0.007 | 0.075 ± 0.010 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `4096x256` | 0.393 ± 0.284 | 0.496 ± 0.086 | unsupported | 0.636 ± 0.377 | 1.070 ± 0.367 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `1024x64` | 0.038 ± 0.008 | 0.072 ± 0.040 | unsupported | 0.137 ± 0.027 | 0.217 ± 0.106 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `4096x256` | 0.675 ± 0.360 | 0.641 ± 0.169 | unsupported | 0.392 ± 0.017 | 0.807 ± 0.161 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `1024x64` | 0.048 ± 0.000 | 0.048 ± 0.000 | unsupported | 0.095 ± 0.008 | 0.125 ± 0.033 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `4096x256` | 0.935 ± 0.212 | 0.781 ± 0.086 | unsupported | 1.565 ± 0.078 | 1.238 ± 0.367 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `1024x64` | 0.082 ± 0.014 | 0.087 ± 0.008 | unsupported | 0.083 ± 0.014 | 0.172 ± 0.022 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `4096x256` | 0.931 ± 0.316 | 0.716 ± 0.256 | unsupported | 1.401 ± 0.365 | 0.805 ± 0.182 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `1024x64` | 0.050 ± 0.000 | 0.050 ± 0.000 | unsupported | 0.099 ± 0.015 | 0.111 ± 0.017 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `4096x256` | 0.937 ± 0.093 | 0.793 ± 0.037 | unsupported | 1.550 ± 0.071 | 1.004 ± 0.041 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `1024x64` | 0.093 ± 0.018 | 0.032 ± 0.043 | unsupported | 0.080 ± 0.006 | 0.190 ± 0.030 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `4096x256` | 0.888 ± 0.334 | 0.875 ± 0.503 | unsupported | 1.432 ± 0.628 | 0.909 ± 0.203 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `1024x64` | 0.088 ± 0.001 | 0.110 ± 0.012 | unsupported | 0.120 ± 0.028 | 0.118 ± 0.003 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `4096x256` | 1.289 ± 0.237 | 1.292 ± 0.101 | unsupported | 2.036 ± 0.156 | 1.576 ± 0.336 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `1024x64` | 0.128 ± 0.017 | 0.138 ± 0.019 | unsupported | 0.082 ± 0.002 | 0.270 ± 0.035 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `4096x256` | 2.821 ± 0.855 | 2.287 ± 0.393 | unsupported | 0.884 ± 0.001 | 1.469 ± 0.482 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `1024x64` | 0.220 ± 0.001 | 0.221 ± 0.002 | unsupported | 0.365 ± 0.139 | 0.419 ± 0.192 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `4096x256` | 3.477 ± 0.289 | 3.812 ± 0.503 | unsupported | 5.552 ± 0.368 | 5.010 ± 0.695 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `1024x64` | 0.303 ± 0.052 | 0.284 ± 0.024 | unsupported | 0.208 ± 0.060 | 0.442 ± 0.079 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `4096x256` | 4.178 ± 0.278 | 3.418 ± 0.582 | unsupported | 3.850 ± 0.688 | 3.827 ± 0.450 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `1024x64` | 0.537 ± 0.022 | 0.698 ± 0.118 | unsupported | 0.151 ± 0.007 | 0.112 ± 0.060 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `4096x256` | 15.685 ± 1.717 | 16.874 ± 6.375 | unsupported | 2.518 ± 0.144 | 1.665 ± 0.362 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `1024x64` | 1.035 ± 0.161 | 1.258 ± 0.251 | unsupported | 0.248 ± 0.027 | 0.191 ± 0.066 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `4096x256` | 11.413 ± 0.864 | 11.788 ± 0.745 | unsupported | 1.289 ± 0.048 | 1.141 ± 0.271 | - | - |
| cpu/output_reuse | `add_into` | f64 | 1 | `33554432` | 22.376 ± 5.344 | - | unsupported | 29.261 ± 6.569 | - | 21.230 ± 5.172 | 26.353 ± 12.201 |
| cpu/output_reuse | `add_into` | f64 | 4 | `33554432` | 22.647 ± 1.334 | - | unsupported | 24.901 ± 1.471 | - | 18.956 ± 0.623 | 22.833 ± 4.435 |
| cpu/output_reuse | `conj_into` | c64 | 1 | `16777216` | 10.887 ± 0.412 | - | unsupported | 19.584 ± 6.951 | - | 16.095 ± 6.092 | 15.364 ± 4.688 |
| cpu/output_reuse | `conj_into` | c64 | 4 | `16777216` | 13.865 ± 4.971 | - | unsupported | 13.973 ± 0.835 | - | 13.927 ± 0.638 | 13.847 ± 1.246 |
| cpu/output_reuse | `copy_read_into` | f64 | 1 | `33554432` | 11.525 ± 0.394 | - | unsupported | 18.408 ± 5.484 | - | 14.101 ± 3.777 | - |
| cpu/output_reuse | `copy_read_into` | f64 | 4 | `33554432` | 16.129 ± 2.909 | - | unsupported | 13.365 ± 0.830 | - | 14.867 ± 1.279 | - |
| cpu/output_reuse | `div_into` | f64 | 1 | `33554432` | 18.898 ± 0.304 | - | unsupported | 34.966 ± 13.813 | - | 30.497 ± 11.915 | 30.340 ± 8.816 |
| cpu/output_reuse | `div_into` | f64 | 4 | `33554432` | 22.878 ± 3.089 | - | unsupported | 23.029 ± 5.762 | - | 21.850 ± 1.142 | 21.397 ± 1.800 |
| cpu/output_reuse | `dot_general_read_into` | f64 | 1 | `1024x1024` | 11.594 ± 0.418 | - | unsupported | 14.102 ± 1.513 | - | 56.950 ± 3.171 | - |
| cpu/output_reuse | `dot_general_read_into` | f64 | 4 | `1024x1024` | 12.927 ± 4.806 | - | unsupported | 11.847 ± 0.523 | - | 37.626 ± 4.457 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 1 | `1024x1024` | 13.777 ± 1.296 | - | unsupported | 13.779 ± 1.505 | - | 69.729 ± 9.199 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 4 | `1024x1024` | 20.863 ± 4.621 | - | unsupported | 13.199 ± 9.384 | - | 37.228 ± 4.192 | - |
| cpu/output_reuse | `mul_into` | f64 | 1 | `33554432` | 17.703 ± 1.548 | - | unsupported | 27.407 ± 6.411 | - | 31.392 ± 9.033 | 30.139 ± 2.709 |
| cpu/output_reuse | `mul_into` | f64 | 4 | `33554432` | 21.468 ± 3.452 | - | unsupported | 21.663 ± 1.705 | - | 20.902 ± 3.084 | 20.588 ± 1.392 |
| cpu/output_reuse | `neg_into` | f64 | 1 | `33554432` | 11.568 ± 1.295 | - | unsupported | 20.965 ± 6.779 | - | 17.371 ± 4.503 | 16.819 ± 3.331 |
| cpu/output_reuse | `neg_into` | f64 | 4 | `33554432` | 15.477 ± 1.134 | - | unsupported | 14.572 ± 0.770 | - | 13.690 ± 1.191 | 14.898 ± 0.637 |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `256x256,rhs=16` | 0.228 ± 0.031 | unsupported | unsupported | 0.347 ± 0.066 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `768x768,rhs=16` | 2.892 ± 0.117 | unsupported | unsupported | 4.193 ± 0.530 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `256x256,rhs=16` | 0.396 ± 0.240 | unsupported | unsupported | 0.455 ± 0.094 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `768x768,rhs=16` | 3.393 ± 0.403 | unsupported | unsupported | 3.837 ± 0.490 | unsupported | - | - |
| cpu/output_reuse | `sub_into` | f64 | 1 | `33554432` | 19.482 ± 6.961 | - | unsupported | 32.097 ± 8.948 | - | 28.814 ± 10.588 | 26.828 ± 10.357 |
| cpu/output_reuse | `sub_into` | f64 | 4 | `33554432` | 22.085 ± 1.414 | - | unsupported | 21.323 ± 1.420 | - | 20.424 ± 1.507 | 21.484 ± 1.189 |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 1 | `8192x1 -> 8192x4096` | 9.919 ± 1.669 | - | 0.002 ± 0.000 | 6.936 ± 0.874 | 9.298 ± 1.192 | 16.452 ± 1.088 | - |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 4 | `8192x1 -> 8192x4096` | 10.634 ± 3.916 | - | 0.001 ± 0.000 | 4.813 ± 0.221 | 6.936 ± 5.936 | 14.366 ± 0.354 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 1 | `33554432` | 11.274 ± 0.891 | - | 24.898 ± 1.696 | 11.739 ± 2.229 | 19.477 ± 2.586 | 10.848 ± 0.870 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 4 | `33554432` | 10.287 ± 0.860 | - | 24.118 ± 4.807 | 10.701 ± 2.445 | 11.411 ± 1.953 | 10.370 ± 1.661 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 1 | `8192 -> 8192x8192` | 12.285 ± 2.491 | - | 12.640 ± 1.674 | 12.619 ± 1.431 | 24.270 ± 1.503 | 13.130 ± 0.849 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 4 | `8192 -> 8192x8192` | 13.792 ± 2.587 | - | 14.985 ± 1.618 | 14.371 ± 2.056 | 16.990 ± 3.431 | 17.437 ± 11.207 | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 1 | `8388608x2x2 -> 8388608x2` | 6.998 ± 0.355 | - | 6.661 ± 0.662 | 12.365 ± 4.279 | 16.388 ± 3.294 | - | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 4 | `8388608x2x2 -> 8388608x2` | 5.879 ± 0.280 | - | 5.843 ± 0.298 | 10.656 ± 1.031 | 12.748 ± 1.990 | - | - |
| cpu/structural_shape | `reshape` | f64 | 1 | `33554432 -> 8192x4096` | 16.854 ± 4.356 | - | 0.002 ± 0.001 | 15.689 ± 2.260 | 16.906 ± 6.436 | 12.635 ± 1.998 | - |
| cpu/structural_shape | `reshape` | f64 | 4 | `33554432 -> 8192x4096` | 13.454 ± 0.231 | - | 0.001 ± 0.000 | 11.266 ± 0.987 | 15.632 ± 2.769 | 18.988 ± 1.967 | - |
| cpu/structural_shape | `transpose` | f64 | 1 | `4096x4096` | 45.322 ± 5.641 | - | 0.002 ± 0.009 | 55.109 ± 6.880 | 60.601 ± 7.032 | 225.113 ± 16.524 | 63.144 ± 7.104 |
| cpu/structural_shape | `transpose` | f64 | 4 | `4096x4096` | 23.887 ± 4.416 | - | 0.002 ± 0.004 | 46.319 ± 3.755 | 35.301 ± 3.500 | 217.198 ± 9.263 | 24.096 ± 7.969 |
| cpu/structural_shape | `tril` | f64 | 1 | `4096x4096` | 8.430 ± 0.586 | - | 9.036 ± 1.023 | 8.960 ± 0.712 | 13.230 ± 2.542 | 10.236 ± 1.357 | - |
| cpu/structural_shape | `tril` | f64 | 4 | `4096x4096` | 6.890 ± 1.086 | - | 7.798 ± 1.205 | 9.279 ± 2.243 | 9.733 ± 2.463 | 10.537 ± 2.060 | - |
| cpu/structural_shape | `triu` | f64 | 1 | `4096x4096` | 7.875 ± 0.878 | - | 9.223 ± 1.171 | 9.620 ± 2.204 | 14.009 ± 4.426 | 9.664 ± 1.754 | - |
| cpu/structural_shape | `triu` | f64 | 4 | `4096x4096` | 7.878 ± 1.232 | - | 6.407 ± 0.266 | 7.199 ± 0.466 | 8.346 ± 2.146 | 13.298 ± 0.770 | - |
| cpu/view_metadata | `broadcast_in_dim_view` | f64 | 1 | `8192x1 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | - | - |
| cpu/view_metadata | `broadcast_in_dim_view` | f64 | 4 | `8192x1 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | - | - |
| cpu/view_metadata | `reshape_view` | f64 | 1 | `33554432 -> 8192x4096` | 0.002 ± 0.003 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `reshape_view` | f64 | 4 | `33554432 -> 8192x4096` | 0.003 ± 0.003 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `slice_view` | f64 | 1 | `4194304 -> 2096128` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `slice_view` | f64 | 4 | `4194304 -> 2096128` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `transpose_view` | f64 | 1 | `4096x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `transpose_view` | f64 | 4 | `4096x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |

## Short-operation batches

Total batch duration and normalized ns/op are shown without rounding sub-microsecond calls to zero. Metadata views require no execution session.

| Operation | Dtype | Shape | Threads | Backend | Operations/batch | Median batch ms | Median ns/op |
|---|---|---|---:|---|---:|---:|---:|
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 16 | 4.830042 | 301877.62 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs direct API (ms) | 16 | 4.131958 | 258247.38 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 16 | 4.697875 | 293617.19 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 16 | 2.844208 | 177763.00 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs direct API (ms) | 16 | 5.656791 | 353549.44 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 16 | 6.299334 | 393708.38 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs direct API (ms) | 16 | 12.935583 | 808473.94 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs eager mode (ms) | 16 | 12.670833 | 791927.06 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs direct API (ms) | 16 | 14.293916 | 893369.75 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs eager mode (ms) | 16 | 15.064375 | 941523.44 |
| householder_qr | f64 | `256x128` | 1 | JAX Python (XLA CPU) (ms) | 16 | 5.882792 | 367674.50 |
| householder_qr | f64 | `256x128` | 1 | PyTorch Python (ms) | 16 | 5.570291 | 348143.19 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 7.770959 | 485684.94 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 7.613459 | 475841.19 |
| householder_qr | f64 | `768x384` | 1 | JAX Python (XLA CPU) (ms) | 16 | 79.750958 | 4984434.88 |
| householder_qr | f64 | `768x384` | 1 | PyTorch Python (ms) | 16 | 71.480875 | 4467554.69 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 57.391625 | 3586976.56 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 57.008833 | 3563052.06 |
| householder_qr | f64 | `256x128` | 4 | JAX Python (XLA CPU) (ms) | 16 | 8.648500 | 540531.25 |
| householder_qr | f64 | `256x128` | 4 | PyTorch Python (ms) | 16 | 5.699833 | 356239.56 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 4.932167 | 308260.44 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 5.342708 | 333919.25 |
| householder_qr | f64 | `768x384` | 4 | JAX Python (XLA CPU) (ms) | 16 | 100.636792 | 6289799.50 |
| householder_qr | f64 | `768x384` | 4 | PyTorch Python (ms) | 16 | 86.901125 | 5431320.31 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 63.023250 | 3938953.12 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 113.241625 | 7077601.56 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 23.484000 | 1467750.00 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 23.464209 | 1466513.06 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 225.667417 | 14104213.56 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 225.671084 | 14104442.75 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 24.176000 | 1511000.00 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 24.134625 | 1508414.06 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 333.740208 | 20858763.00 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 319.006709 | 19937919.31 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 23.284708 | 1455294.25 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 22.560125 | 1410007.81 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 203.705333 | 12731583.31 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 220.066250 | 13754140.62 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 23.788000 | 1486750.00 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 24.281875 | 1517617.19 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 290.333167 | 18145822.94 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 250.891042 | 15680690.12 |
| erf | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 2.913250 | 45519.53 |
| erf | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 6.550292 | 102348.31 |
| erf | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 11.651167 | 182049.48 |
| erf | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 11.206709 | 175104.83 |
| erf | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 4.139666 | 517458.25 |
| erf | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 7.733333 | 966666.62 |
| erf | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 31.001583 | 3875197.88 |
| erf | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 22.529958 | 2816244.75 |
| erf | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 5.405375 | 84458.98 |
| erf | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 4.362959 | 68171.23 |
| erf | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 14.546542 | 227289.72 |
| erf | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 15.036792 | 234949.88 |
| erf | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 5.029959 | 628744.88 |
| erf | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 6.185583 | 773197.88 |
| erf | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 21.712917 | 2714114.62 |
| erf | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 20.780792 | 2597599.00 |
| gelu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 15.503292 | 242238.94 |
| gelu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 7.563250 | 118175.78 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 6.128542 | 95758.47 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 6.470959 | 101108.73 |
| gelu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 20.120250 | 2515031.25 |
| gelu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 13.711583 | 1713947.88 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 13.289000 | 1661125.00 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 16.093958 | 2011744.75 |
| gelu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 19.866791 | 310418.61 |
| gelu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 5.633417 | 88022.14 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 8.399292 | 131238.94 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 9.129500 | 142648.44 |
| gelu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 19.125667 | 2390708.38 |
| gelu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 9.251459 | 1156432.38 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 9.870625 | 1233828.12 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 9.807125 | 1225890.62 |
| gelu_tanh | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 6.424916 | 100389.31 |
| gelu_tanh | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 22.982792 | 359106.12 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 17.052666 | 266447.91 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 17.286041 | 270094.39 |
| gelu_tanh | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 8.473166 | 1059145.75 |
| gelu_tanh | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 47.658417 | 5957302.12 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 52.212292 | 6526536.50 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 41.348625 | 5168578.12 |
| gelu_tanh | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 15.662875 | 244732.42 |
| gelu_tanh | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 12.818125 | 200283.20 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 18.451917 | 288311.20 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 18.857959 | 294655.61 |
| gelu_tanh | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 7.867292 | 983411.50 |
| gelu_tanh | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 19.896500 | 2487062.50 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 31.314542 | 3914317.75 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 32.267208 | 4033401.00 |
| layer_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 5.751334 | 89864.59 |
| layer_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 3.813500 | 59585.94 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 2.965167 | 46330.73 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 4.068625 | 63572.27 |
| layer_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 7.300875 | 912609.38 |
| layer_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 5.455375 | 681921.88 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 6.755125 | 844390.62 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 7.992792 | 999099.00 |
| layer_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 11.661917 | 182217.45 |
| layer_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 3.238000 | 50593.75 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 4.403416 | 68803.38 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 5.968458 | 93257.16 |
| layer_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 7.915167 | 989395.88 |
| layer_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 3.399875 | 424984.38 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 11.214166 | 1401770.75 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 11.777583 | 1472197.88 |
| log_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 7.908834 | 123575.53 |
| log_softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 8.758083 | 136845.05 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 5.784333 | 90380.20 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 6.673375 | 104271.48 |
| log_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 12.943500 | 1617937.50 |
| log_softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 12.708041 | 1588505.12 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 10.392916 | 1299114.50 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 11.691958 | 1461494.75 |
| log_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 16.440667 | 256885.42 |
| log_softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 5.750750 | 89855.47 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 7.553000 | 118015.62 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 8.706542 | 136039.72 |
| log_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 13.347292 | 1668411.50 |
| log_softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 6.177375 | 772171.88 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 20.995167 | 2624395.88 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 21.903000 | 2737875.00 |
| masked_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 8.893250 | 138957.03 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 8.236500 | 128695.31 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 9.218666 | 144041.66 |
| masked_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 17.961500 | 2245187.50 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 15.259708 | 1907463.50 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 17.362625 | 2170328.12 |
| masked_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 19.554959 | 305546.23 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 10.845750 | 169464.84 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 11.564291 | 180692.05 |
| masked_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 18.136042 | 2267005.25 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 27.013167 | 3376645.88 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 23.871875 | 2983984.38 |
| reduce_mean | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 6.308375 | 98568.36 |
| reduce_mean | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 0.573375 | 8958.98 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 0.741542 | 11586.59 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 1.015375 | 15865.23 |
| reduce_mean | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 2.023167 | 252895.88 |
| reduce_mean | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 0.573459 | 71682.38 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 0.892083 | 111510.38 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 0.912458 | 114057.25 |
| reduce_mean | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 5.480292 | 85629.56 |
| reduce_mean | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 2.640625 | 41259.77 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 1.016000 | 15875.00 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 1.371500 | 21429.69 |
| reduce_mean | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 3.538542 | 442317.75 |
| reduce_mean | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 0.470333 | 58791.62 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 1.878375 | 234796.88 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 2.385625 | 298203.12 |
| rms_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 4.827292 | 75426.44 |
| rms_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 2.265875 | 35404.30 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 1.556583 | 24321.61 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 2.286417 | 35725.27 |
| rms_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 8.558500 | 1069812.50 |
| rms_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 5.089041 | 636130.12 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 3.145625 | 393203.12 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 3.964958 | 495619.75 |
| rms_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 13.918917 | 217483.08 |
| rms_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 8.769208 | 137018.88 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 2.402333 | 37536.45 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 4.625000 | 72265.62 |
| rms_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 6.452542 | 806567.75 |
| rms_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 3.135375 | 391921.88 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 5.403083 | 675385.38 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 5.128959 | 641119.88 |
| sigmoid | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 8.007208 | 125112.62 |
| sigmoid | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 6.097625 | 95275.39 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 3.096750 | 48386.72 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 3.099209 | 48425.14 |
| sigmoid | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 9.900750 | 1237593.75 |
| sigmoid | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 12.522250 | 1565281.25 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 7.477375 | 934671.88 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 6.244958 | 780619.75 |
| sigmoid | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 11.026833 | 172294.27 |
| sigmoid | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 5.316792 | 83074.88 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 5.270292 | 82348.31 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 5.595750 | 87433.59 |
| sigmoid | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 6.440417 | 805052.12 |
| sigmoid | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 11.211083 | 1401385.38 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 7.449916 | 931239.50 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 5.724625 | 715578.12 |
| silu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 7.128291 | 111379.55 |
| silu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 6.358334 | 99348.97 |
| silu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 3.214292 | 50223.31 |
| silu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 3.216334 | 50255.22 |
| silu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 8.029500 | 1003687.50 |
| silu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 12.400625 | 1550078.12 |
| silu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 7.492875 | 936609.38 |
| silu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 6.344417 | 793052.12 |
| silu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 12.138500 | 189664.06 |
| silu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 5.118834 | 79981.78 |
| silu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 5.951709 | 92995.45 |
| silu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 2.045125 | 31955.08 |
| silu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 7.268833 | 908604.12 |
| silu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 11.459125 | 1432390.62 |
| silu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 7.103583 | 887947.88 |
| silu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 7.003458 | 875432.25 |
| softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 7.555542 | 118055.34 |
| softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 7.701584 | 120337.25 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 5.652958 | 88327.47 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 7.009791 | 109527.98 |
| softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 12.611750 | 1576468.75 |
| softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 16.288291 | 2036036.38 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 10.309375 | 1288671.88 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 10.332333 | 1291541.62 |
| softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 17.274792 | 269918.62 |
| softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 5.217500 | 81523.44 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 8.183333 | 127864.58 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 8.834417 | 138037.77 |
| softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 11.749083 | 1468635.38 |
| softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 7.071000 | 883875.00 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 22.568833 | 2821104.12 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 18.293125 | 2286640.62 |
| softplus | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 26.843875 | 419435.55 |
| softplus | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 23.336208 | 364628.25 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 14.052292 | 219567.06 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 14.124375 | 220693.36 |
| softplus | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 40.079292 | 5009911.50 |
| softplus | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 44.412958 | 5551619.75 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 27.812833 | 3476604.12 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 30.499416 | 3812427.00 |
| softplus | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 28.306667 | 442291.67 |
| softplus | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 13.288167 | 207627.61 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 19.416667 | 303385.42 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 18.147334 | 283552.09 |
| softplus | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 30.616000 | 3827000.00 |
| softplus | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 30.802458 | 3850307.25 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 33.421125 | 4177640.63 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 27.343417 | 3417927.12 |
| take_along_axis | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 7.150333 | 111723.95 |
| take_along_axis | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 9.690709 | 151417.33 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 34.377167 | 537143.23 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 44.689417 | 698272.14 |
| take_along_axis | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 13.320542 | 1665067.75 |
| take_along_axis | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 20.145125 | 2518140.62 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 125.476583 | 15684572.88 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 134.989709 | 16873713.62 |
| take_along_axis | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 12.213708 | 190839.19 |
| take_along_axis | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 15.888625 | 248259.77 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 66.220542 | 1034695.97 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 80.526541 | 1258227.20 |
| take_along_axis | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 9.129375 | 1141171.88 |
| take_along_axis | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 10.314125 | 1289265.62 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 91.306875 | 11413359.38 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 94.300416 | 11787552.00 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | PyTorch Python (ms) | 16 | 5.554042 | 347127.62 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 3.649125 | 228070.31 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | PyTorch Python (ms) | 16 | 67.084375 | 4192773.44 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 46.272167 | 2892010.44 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | PyTorch Python (ms) | 16 | 7.283333 | 455208.31 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 6.339584 | 396224.00 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | PyTorch Python (ms) | 16 | 61.394917 | 3837182.31 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 54.290958 | 3393184.88 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.013083 | 817.69 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.003709 | 231.81 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.017750 | 1109.38 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.004917 | 307.31 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000084 | 5.25 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.011667 | 729.19 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.036167 | 2260.44 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000166 | 10.38 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.014875 | 929.69 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.048125 | 3007.81 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000167 | 10.44 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | PyTorch Python (ms) | 16 | 0.010083 | 630.19 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | tenferro-rs direct API (ms) | 16 | 0.003750 | 234.38 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000667 | 41.69 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | PyTorch Python (ms) | 16 | 0.011500 | 718.75 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | tenferro-rs direct API (ms) | 16 | 0.003084 | 192.75 |
| transpose_view | f64 | `4096x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000209 | 13.06 |
| transpose_view | f64 | `4096x4096` | 1 | PyTorch Python (ms) | 16 | 0.010666 | 666.62 |
| transpose_view | f64 | `4096x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.003125 | 195.31 |
| transpose_view | f64 | `4096x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000084 | 5.25 |
| transpose_view | f64 | `4096x4096` | 4 | PyTorch Python (ms) | 16 | 0.015250 | 953.12 |
| transpose_view | f64 | `4096x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.003417 | 213.56 |

## Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `cpu/complex/cholesky` (c64, threads=4, shape=`448x448`): `jax-cpu` is 42.9x faster than the slowest successful cell (`julia-base`, 83.951 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/cholesky` (c64, threads=4, shape=`448x448`): `pytorch-cpu` is 23.5x faster than the slowest successful cell (`julia-base`, 83.951 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/cholesky` (c64, threads=4, shape=`448x448`): `tenferro-direct` is 16.7x faster than the slowest successful cell (`julia-base`, 83.951 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/cholesky` (c64, threads=4, shape=`448x448`): `tenferro-trace` is 21.1x faster than the slowest successful cell (`julia-base`, 83.951 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/eig` (c64, threads=4, shape=`112x112`): `jax-cpu` is 19.8x faster than the slowest successful cell (`julia-base`, 489.877 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/eig` (c64, threads=4, shape=`112x112`): `pytorch-cpu` is 41.4x faster than the slowest successful cell (`julia-base`, 489.877 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/eig` (c64, threads=4, shape=`112x112`): `tenferro-direct` is 39.2x faster than the slowest successful cell (`julia-base`, 489.877 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/eig` (c64, threads=4, shape=`112x112`): `tenferro-trace` is 47.6x faster than the slowest successful cell (`julia-base`, 489.877 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `jax-cpu` is 19.9x faster than the slowest successful cell (`julia-base`, 163.188 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `pytorch-cpu` is 14.3x faster than the slowest successful cell (`julia-base`, 163.188 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `tenferro-direct` is 19.0x faster than the slowest successful cell (`julia-base`, 163.188 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `tenferro-trace` is 18.9x faster than the slowest successful cell (`julia-base`, 163.188 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `jax-cpu` is 44.0x faster than the slowest successful cell (`julia-base`, 143.168 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `pytorch-cpu` is 25.2x faster than the slowest successful cell (`julia-base`, 143.168 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `tenferro-direct` is 20.5x faster than the slowest successful cell (`julia-base`, 143.168 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `tenferro-trace` is 24.2x faster than the slowest successful cell (`julia-base`, 143.168 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/svd` (c64, threads=4, shape=`160x160`): `jax-cpu` is 79.0x faster than the slowest successful cell (`julia-base`, 644.050 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/svd` (c64, threads=4, shape=`160x160`): `pytorch-cpu` is 82.6x faster than the slowest successful cell (`julia-base`, 644.050 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/svd` (c64, threads=4, shape=`160x160`): `tenferro-direct` is 90.4x faster than the slowest successful cell (`julia-base`, 644.050 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/svd` (c64, threads=4, shape=`160x160`): `tenferro-trace` is 75.0x faster than the slowest successful cell (`julia-base`, 644.050 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 27.0x faster than the slowest successful cell (`pytorch-cpu`, 21.174 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 14.1x faster than the slowest successful cell (`pytorch-cpu`, 21.174 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 14.2x faster than the slowest successful cell (`pytorch-cpu`, 21.174 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 16.2x faster than the slowest successful cell (`pytorch-cpu`, 14.062 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-trace` is 14.2x faster than the slowest successful cell (`pytorch-cpu`, 14.062 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 17.9x faster than the slowest successful cell (`pytorch-cpu`, 19.354 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 22.4x faster than the slowest successful cell (`pytorch-cpu`, 19.354 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 19.1x faster than the slowest successful cell (`pytorch-cpu`, 15.830 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 18.9x faster than the slowest successful cell (`pytorch-cpu`, 15.830 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-trace` is 16.3x faster than the slowest successful cell (`pytorch-cpu`, 15.830 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=1, shape=`4194304 -> 2096128`): `tenferro-trace` is 412.3x faster than the slowest successful cell (`julia-base`, 1.632 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`): `tenferro-trace` is 1859.1x faster than the slowest successful cell (`julia-base`, 3.175 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x2x2`): `tenferro-direct` is 10.3x faster than the slowest successful cell (`jax-cpu`, 0.008 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x32x32`): `tenferro-direct` is 27.3x faster than the slowest successful cell (`jax-cpu`, 0.050 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x32x32`): `tenferro-trace` is 15.6x faster than the slowest successful cell (`jax-cpu`, 0.050 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x4x4`): `tenferro-direct` is 11.2x faster than the slowest successful cell (`jax-cpu`, 0.010 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x8x8`): `pytorch-cpu` is 13.8x faster than the slowest successful cell (`jax-cpu`, 0.061 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x8x8`): `tenferro-direct` is 42.1x faster than the slowest successful cell (`jax-cpu`, 0.061 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x8x8`): `tenferro-trace` is 21.4x faster than the slowest successful cell (`jax-cpu`, 0.061 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`8x8x8`): `pytorch-cpu` is 12.5x faster than the slowest successful cell (`jax-cpu`, 0.070 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`8x8x8`): `tenferro-direct` is 31.7x faster than the slowest successful cell (`jax-cpu`, 0.070 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`8x8x8`): `tenferro-trace` is 18.1x faster than the slowest successful cell (`jax-cpu`, 0.070 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x2x2`): `tenferro-direct` is 13.7x faster than the slowest successful cell (`pytorch-cpu`, 0.019 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x4x4`): `tenferro-direct` is 10.7x faster than the slowest successful cell (`pytorch-cpu`, 0.020 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x4x4`): `tenferro-direct` is 10.2x faster than the slowest successful cell (`pytorch-cpu`, 0.019 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x2x2`): `tenferro-direct` is 10.3x faster than the slowest successful cell (`pytorch-cpu`, 0.019 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x4x4`): `tenferro-direct` is 45.9x faster than the slowest successful cell (`jax-cpu`, 0.088 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x32x32`): `tenferro-direct` is 14.7x faster than the slowest successful cell (`pytorch-cpu`, 0.170 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x32x32`): `tenferro-trace` is 12.7x faster than the slowest successful cell (`pytorch-cpu`, 0.170 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigh` (c64, threads=1, shape=`3x2x2`): `tenferro-direct` is 26.5x faster than the slowest successful cell (`jax-cpu`, 0.053 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigh` (c64, threads=1, shape=`3x2x2`): `tenferro-trace` is 15.5x faster than the slowest successful cell (`jax-cpu`, 0.053 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigh` (c64, threads=1, shape=`4x8x8`): `tenferro-direct` is 11.6x faster than the slowest successful cell (`jax-cpu`, 0.272 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigvalsh` (f64, threads=1, shape=`3x2x2`): `tenferro-direct` is 21.8x faster than the slowest successful cell (`jax-cpu`, 0.029 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigvalsh` (f64, threads=1, shape=`3x2x2`): `tenferro-trace` is 12.7x faster than the slowest successful cell (`jax-cpu`, 0.029 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigvalsh` (f64, threads=4, shape=`1x8x8`): `tenferro-direct` is 14.5x faster than the slowest successful cell (`jax-cpu`, 0.058 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigvalsh` (f64, threads=4, shape=`1x8x8`): `tenferro-trace` is 11.6x faster than the slowest successful cell (`jax-cpu`, 0.058 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigvalsh` (f64, threads=4, shape=`8x8x8`): `pytorch-cpu` is 10.9x faster than the slowest successful cell (`jax-cpu`, 0.260 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigvalsh` (f64, threads=4, shape=`8x8x8`): `tenferro-direct` is 10.9x faster than the slowest successful cell (`jax-cpu`, 0.260 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigvalsh` (f64, threads=4, shape=`8x8x8`): `tenferro-trace` is 11.8x faster than the slowest successful cell (`jax-cpu`, 0.260 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`1x64x24`): `tenferro-direct` is 10.1x faster than the slowest successful cell (`jax-cpu`, 0.090 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`3x4x4`): `tenferro-direct` is 14.6x faster than the slowest successful cell (`jax-cpu`, 0.036 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`3x4x4`): `tenferro-trace` is 11.2x faster than the slowest successful cell (`jax-cpu`, 0.036 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`3x8x8`): `tenferro-direct` is 23.4x faster than the slowest successful cell (`jax-cpu`, 0.069 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`3x8x8`): `tenferro-trace` is 17.7x faster than the slowest successful cell (`jax-cpu`, 0.069 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`4x8x8`): `tenferro-direct` is 20.8x faster than the slowest successful cell (`jax-cpu`, 0.070 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`4x8x8`): `tenferro-trace` is 16.4x faster than the slowest successful cell (`jax-cpu`, 0.070 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`8x4x4`): `tenferro-direct` is 22.8x faster than the slowest successful cell (`jax-cpu`, 0.074 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`8x4x4`): `tenferro-trace` is 14.1x faster than the slowest successful cell (`jax-cpu`, 0.074 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`8x8x8`): `pytorch-cpu` is 10.8x faster than the slowest successful cell (`jax-cpu`, 0.113 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`8x8x8`): `tenferro-direct` is 22.7x faster than the slowest successful cell (`jax-cpu`, 0.113 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`8x8x8`): `tenferro-trace` is 18.8x faster than the slowest successful cell (`jax-cpu`, 0.113 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x2x2`): `tenferro-direct` is 33.2x faster than the slowest successful cell (`pytorch-cpu`, 0.111 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x32x32`): `tenferro-direct` is 14.9x faster than the slowest successful cell (`pytorch-cpu`, 0.142 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x32x32`): `tenferro-trace` is 19.9x faster than the slowest successful cell (`pytorch-cpu`, 0.142 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x4x4`): `tenferro-direct` is 20.7x faster than the slowest successful cell (`pytorch-cpu`, 0.083 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x8x8`): `tenferro-direct` is 27.3x faster than the slowest successful cell (`pytorch-cpu`, 0.113 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x8x8`): `tenferro-trace` is 23.1x faster than the slowest successful cell (`pytorch-cpu`, 0.113 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x2x2`): `tenferro-direct` is 19.0x faster than the slowest successful cell (`pytorch-cpu`, 0.145 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x2x2`): `tenferro-trace` is 12.0x faster than the slowest successful cell (`pytorch-cpu`, 0.145 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x4x4`): `tenferro-direct` is 11.1x faster than the slowest successful cell (`pytorch-cpu`, 0.088 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x4x4`): `tenferro-trace` is 16.9x faster than the slowest successful cell (`pytorch-cpu`, 0.088 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x8x8`): `tenferro-direct` is 26.2x faster than the slowest successful cell (`pytorch-cpu`, 0.124 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x8x8`): `tenferro-trace` is 21.5x faster than the slowest successful cell (`pytorch-cpu`, 0.124 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x2x2`): `tenferro-direct` is 13.1x faster than the slowest successful cell (`pytorch-cpu`, 0.101 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x2x2`): `tenferro-trace` is 21.0x faster than the slowest successful cell (`pytorch-cpu`, 0.101 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x4x4`): `tenferro-direct` is 24.5x faster than the slowest successful cell (`pytorch-cpu`, 0.100 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x8x8`): `jax-cpu` is 14.0x faster than the slowest successful cell (`pytorch-cpu`, 0.146 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x8x8`): `tenferro-direct` is 35.8x faster than the slowest successful cell (`pytorch-cpu`, 0.146 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x2x2`): `jax-cpu` is 11.1x faster than the slowest successful cell (`pytorch-cpu`, 0.178 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x2x2`): `tenferro-direct` is 21.1x faster than the slowest successful cell (`pytorch-cpu`, 0.178 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x2x2`): `tenferro-trace` is 13.4x faster than the slowest successful cell (`pytorch-cpu`, 0.178 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x4x4`): `tenferro-direct` is 15.8x faster than the slowest successful cell (`pytorch-cpu`, 0.160 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x8x8`): `tenferro-direct` is 20.4x faster than the slowest successful cell (`pytorch-cpu`, 0.144 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x8x8`): `tenferro-trace` is 15.1x faster than the slowest successful cell (`pytorch-cpu`, 0.144 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`1x2x2`): `tenferro-direct` is 11.0x faster than the slowest successful cell (`pytorch-cpu`, 0.037 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`1x4x4`): `tenferro-direct` is 15.2x faster than the slowest successful cell (`pytorch-cpu`, 0.050 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`1x8x8`): `tenferro-direct` is 14.4x faster than the slowest successful cell (`pytorch-cpu`, 0.058 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`3x2x2`): `tenferro-direct` is 11.9x faster than the slowest successful cell (`pytorch-cpu`, 0.040 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`4x4x4`): `tenferro-direct` is 16.7x faster than the slowest successful cell (`pytorch-cpu`, 0.085 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`4x4x4`): `tenferro-trace` is 12.8x faster than the slowest successful cell (`pytorch-cpu`, 0.085 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`8x4x4`): `tenferro-direct` is 11.1x faster than the slowest successful cell (`pytorch-cpu`, 0.081 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=1, shape=`1x32x32,rhs=1`): `tenferro-direct` is 15.9x faster than the slowest successful cell (`jax-cpu`, 0.069 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=1, shape=`3x2x2,rhs=1`): `tenferro-direct` is 20.0x faster than the slowest successful cell (`jax-cpu`, 0.033 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=1, shape=`3x8x8,rhs=1`): `tenferro-direct` is 17.1x faster than the slowest successful cell (`jax-cpu`, 0.039 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=1, shape=`8x32x32,rhs=1`): `tenferro-direct` is 11.3x faster than the slowest successful cell (`jax-cpu`, 0.308 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1x8x8,rhs=1`): `tenferro-direct` is 11.8x faster than the slowest successful cell (`jax-cpu`, 0.030 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x8x8,rhs=1`): `tenferro-direct` is 12.0x faster than the slowest successful cell (`pytorch-cpu`, 0.059 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_svdvals` (f64, threads=1, shape=`1x2x2`): `tenferro-direct` is 23.3x faster than the slowest successful cell (`jax-cpu`, 0.031 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_svdvals` (f64, threads=1, shape=`3x2x2`): `tenferro-direct` is 10.5x faster than the slowest successful cell (`pytorch-cpu`, 0.018 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/cholesky` (f64, threads=4, shape=`1536x1536`): `jax-cpu` is 23.0x faster than the slowest successful cell (`julia-base`, 426.530 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/cholesky` (f64, threads=4, shape=`1536x1536`): `pytorch-cpu` is 19.8x faster than the slowest successful cell (`julia-base`, 426.530 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/cholesky` (f64, threads=4, shape=`1536x1536`): `tenferro-direct` is 36.7x faster than the slowest successful cell (`julia-base`, 426.530 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/cholesky` (f64, threads=4, shape=`1536x1536`): `tenferro-trace` is 35.5x faster than the slowest successful cell (`julia-base`, 426.530 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `jax-cpu` is 28.5x faster than the slowest successful cell (`julia-base`, 266.062 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `pytorch-cpu` is 29.9x faster than the slowest successful cell (`julia-base`, 266.062 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `tenferro-direct` is 21.9x faster than the slowest successful cell (`julia-base`, 266.062 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `tenferro-trace` is 21.7x faster than the slowest successful cell (`julia-base`, 266.062 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eig` (f64, threads=4, shape=`160x160`): `jax-cpu` is 31.2x faster than the slowest successful cell (`julia-base`, 324.209 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eig` (f64, threads=4, shape=`160x160`): `pytorch-cpu` is 43.5x faster than the slowest successful cell (`julia-base`, 324.209 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eig` (f64, threads=4, shape=`160x160`): `tenferro-direct` is 42.3x faster than the slowest successful cell (`julia-base`, 324.209 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eig` (f64, threads=4, shape=`160x160`): `tenferro-trace` is 41.1x faster than the slowest successful cell (`julia-base`, 324.209 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvals` (f64, threads=4, shape=`192x192`): `jax-cpu` is 33.0x faster than the slowest successful cell (`julia-base`, 272.618 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvals` (f64, threads=4, shape=`192x192`): `pytorch-cpu` is 30.0x faster than the slowest successful cell (`julia-base`, 272.618 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvals` (f64, threads=4, shape=`192x192`): `tenferro-direct` is 30.1x faster than the slowest successful cell (`julia-base`, 272.618 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvals` (f64, threads=4, shape=`192x192`): `tenferro-trace` is 29.2x faster than the slowest successful cell (`julia-base`, 272.618 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvalsh` (f64, threads=4, shape=`512x512`): `jax-cpu` is 21.9x faster than the slowest successful cell (`julia-base`, 639.460 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvalsh` (f64, threads=4, shape=`512x512`): `pytorch-cpu` is 25.8x faster than the slowest successful cell (`julia-base`, 639.460 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvalsh` (f64, threads=4, shape=`512x512`): `tenferro-direct` is 27.2x faster than the slowest successful cell (`julia-base`, 639.460 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvalsh` (f64, threads=4, shape=`512x512`): `tenferro-trace` is 26.8x faster than the slowest successful cell (`julia-base`, 639.460 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `jax-cpu` is 23.3x faster than the slowest successful cell (`julia-base`, 380.957 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `pytorch-cpu` is 30.8x faster than the slowest successful cell (`julia-base`, 380.957 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `tenferro-direct` is 22.0x faster than the slowest successful cell (`julia-base`, 380.957 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `tenferro-trace` is 26.8x faster than the slowest successful cell (`julia-base`, 380.957 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lstsq` (f64, threads=4, shape=`768x384,rhs=16`): `pytorch-cpu` is 22.3x faster than the slowest successful cell (`julia-base`, 149.636 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `jax-cpu` is 17.7x faster than the slowest successful cell (`julia-base`, 249.198 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `pytorch-cpu` is 19.5x faster than the slowest successful cell (`julia-base`, 249.198 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `tenferro-direct` is 17.0x faster than the slowest successful cell (`julia-base`, 249.198 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `tenferro-trace` is 23.3x faster than the slowest successful cell (`julia-base`, 249.198 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `jax-cpu` is 53.3x faster than the slowest successful cell (`julia-base`, 1038.381 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `pytorch-cpu` is 76.1x faster than the slowest successful cell (`julia-base`, 1038.381 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `tenferro-direct` is 58.6x faster than the slowest successful cell (`julia-base`, 1038.381 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `tenferro-trace` is 63.4x faster than the slowest successful cell (`julia-base`, 1038.381 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `jax-cpu` is 54.3x faster than the slowest successful cell (`julia-base`, 1011.824 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `pytorch-cpu` is 76.3x faster than the slowest successful cell (`julia-base`, 1011.824 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `tenferro-direct` is 62.1x faster than the slowest successful cell (`julia-base`, 1011.824 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `tenferro-trace` is 51.2x faster than the slowest successful cell (`julia-base`, 1011.824 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `jax-cpu` is 28.2x faster than the slowest successful cell (`julia-base`, 455.775 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `pytorch-cpu` is 38.9x faster than the slowest successful cell (`julia-base`, 455.775 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `tenferro-direct` is 38.5x faster than the slowest successful cell (`julia-base`, 455.775 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `tenferro-trace` is 30.6x faster than the slowest successful cell (`julia-base`, 455.775 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `jax-cpu` is 35.5x faster than the slowest successful cell (`julia-base`, 1780.063 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `pytorch-cpu` is 33.0x faster than the slowest successful cell (`julia-base`, 1780.063 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `tenferro-eager` is 36.4x faster than the slowest successful cell (`julia-base`, 1780.063 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `tenferro-trace` is 37.4x faster than the slowest successful cell (`julia-base`, 1780.063 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/reduce_mean` (f32, threads=1, shape=`1024x64`): `pytorch-cpu` is 11.0x faster than the slowest successful cell (`jax-cpu`, 0.099 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=1, shape=`4096x256`): `jax-cpu` is 10.1x faster than the slowest successful cell (`tenferro-eager`, 16.874 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 10.3x faster than the slowest successful cell (`tenferro-eager`, 11.788 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=1, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 9875.3x faster than the slowest successful cell (`julia-base`, 16.452 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=4, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 10447.8x faster than the slowest successful cell (`julia-base`, 14.366 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=1, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 10141.6x faster than the slowest successful cell (`jax-cpu`, 16.906 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=4, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 14244.5x faster than the slowest successful cell (`julia-base`, 18.988 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`): `tenferro-trace` is 131798.9x faster than the slowest successful cell (`julia-base`, 225.113 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`): `tenferro-trace` is 115839.0x faster than the slowest successful cell (`julia-base`, 217.198 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=1, shape=`33554432 -> 8192x4096`): `julia-base` is 452.0x faster than the slowest successful cell (`tenferro-direct`, 0.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=4, shape=`33554432 -> 8192x4096`): `julia-base` is 300.8x faster than the slowest successful cell (`tenferro-direct`, 0.003 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=1, shape=`4194304 -> 2096128`): `julia-base` is 63.0x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=4, shape=`4194304 -> 2096128`): `julia-base` is 17.1x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=1, shape=`4096x4096`): `julia-base` is 51.3x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=4, shape=`4096x4096`): `julia-base` is 190.6x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

## Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

- `cpu/indexing_layout/slice` (f64, threads=1, shape=`4194304 -> 2096128`), `tenferro-trace`: 16769024 minimum logical bytes imply 4236.7 GB/s at 0.004 ms. The conservative physical lower bound is 0.168 ms, so the cell is 42.4x faster than the bound.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`), `tenferro-trace`: 16769024 minimum logical bytes imply 9817.9 GB/s at 0.002 ms. The conservative physical lower bound is 0.042 ms, so the cell is 24.5x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=1, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 161125.7 GB/s at 0.002 ms. The conservative physical lower bound is 2.684 ms, so the cell is 1611.3x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=4, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 195225.8 GB/s at 0.001 ms. The conservative physical lower bound is 0.671 ms, so the cell is 488.1x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=1, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 161029.1 GB/s at 0.002 ms. The conservative physical lower bound is 2.684 ms, so the cell is 1610.3x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=4, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 201376.9 GB/s at 0.001 ms. The conservative physical lower bound is 0.671 ms, so the cell is 503.4x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 78581.8 GB/s at 0.002 ms. The conservative physical lower bound is 1.342 ms, so the cell is 785.8x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 71582.8 GB/s at 0.002 ms. The conservative physical lower bound is 0.336 ms, so the cell is 179.0x faster than the bound.
