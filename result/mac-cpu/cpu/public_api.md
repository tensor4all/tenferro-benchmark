# CPU Public API Benchmark Results

- Suite: `cpu/public_api`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/public_api.yaml`
- Run metadata: `data/results/mac-cpu/cpu/public_api/20261109_182559/run.yaml`
- Timestamp: `2026-11-09T18:25:59.618560+00:00`

Latest run: `./scripts/run_cpu_public_api.sh 1 4`.

- Public API coverage manifest: `benchmarks/cpu/public_api_coverage.yaml`

This file is generated from sequential CPU public API runs under `data/results/mac-cpu/cpu/public_api/20261109_182559`.

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

- CSV: `data/results/mac-cpu/cpu/public_api/20261109_182559/cpu_public_api_t1_20261109_182559.csv`
- CSV: `data/results/mac-cpu/cpu/public_api/20261109_182559/cpu_public_api_t4_20261109_182559.csv`

## CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| cpu/complex | `cholesky` | c64 | 1 | `448x448` | 3.389 ± 0.035 | - | 3.357 ± 0.034 | 3.841 ± 0.178 | 1.693 ± 0.130 | 4.932 ± 2.792 | - |
| cpu/complex | `cholesky` | c64 | 4 | `448x448` | 3.713 ± 0.084 | - | 3.822 ± 0.515 | 4.974 ± 0.554 | 1.671 ± 0.191 | 174.110 ± 37.494 | - |
| cpu/complex | `conj` | c64 | 1 | `16777216` | 10.999 ± 1.968 | - | 9.211 ± 0.287 | 10.519 ± 0.169 | 10.631 ± 1.248 | 12.527 ± 1.903 | 12.032 ± 5.100 |
| cpu/complex | `conj` | c64 | 4 | `16777216` | 16.043 ± 2.787 | - | 12.426 ± 1.592 | 14.039 ± 2.010 | 10.965 ± 1.464 | 14.730 ± 4.460 | 14.463 ± 3.837 |
| cpu/complex | `div` | c64 | 1 | `8388608` | 44.493 ± 0.697 | - | 44.381 ± 0.967 | 18.651 ± 0.170 | 33.413 ± 0.281 | 47.361 ± 0.577 | 54.010 ± 2.009 |
| cpu/complex | `div` | c64 | 4 | `8388608` | 25.295 ± 4.937 | - | 23.822 ± 4.103 | 14.546 ± 4.994 | 17.555 ± 5.768 | 55.474 ± 1.147 | 25.781 ± 3.792 |
| cpu/complex | `dot_general` | c64 | 1 | `640x640` | 11.928 ± 1.066 | - | 11.880 ± 0.318 | 9.511 ± 0.267 | 47.942 ± 1.252 | 97.507 ± 5.724 | - |
| cpu/complex | `dot_general` | c64 | 4 | `640x640` | 11.308 ± 1.614 | - | 11.737 ± 1.779 | 12.050 ± 4.476 | 17.216 ± 2.301 | 33.318 ± 9.031 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 1 | `640x640` | 12.894 ± 0.732 | - | unsupported | 9.906 ± 0.583 | 47.561 ± 0.536 | 69.756 ± 3.388 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 4 | `640x640` | 14.287 ± 4.045 | - | unsupported | 15.002 ± 7.730 | 17.688 ± 0.669 | 34.399 ± 10.248 | - |
| cpu/complex | `eig` | c64 | 1 | `112x112` | 10.595 ± 0.109 | - | 9.129 ± 0.045 | 10.920 ± 0.131 | 21.270 ± 1.174 | 9.092 ± 0.165 | - |
| cpu/complex | `eig` | c64 | 4 | `112x112` | 11.893 ± 0.690 | - | 10.933 ± 1.073 | 14.292 ± 1.539 | 20.754 ± 1.319 | 453.745 ± 31.383 | - |
| cpu/complex | `exp` | c64 | 1 | `4194304` | 32.523 ± 0.288 | - | 32.354 ± 0.210 | 34.436 ± 1.281 | 43.168 ± 2.422 | 48.904 ± 1.023 | 46.779 ± 3.710 |
| cpu/complex | `exp` | c64 | 4 | `4194304` | 22.522 ± 5.429 | - | 16.410 ± 4.626 | 24.360 ± 14.452 | 22.073 ± 0.622 | 58.486 ± 2.078 | 26.300 ± 7.939 |
| cpu/complex | `log` | c64 | 1 | `4194304` | 45.055 ± 0.699 | - | 48.221 ± 0.554 | 43.415 ± 0.247 | 85.275 ± 2.801 | 77.259 ± 7.603 | 84.829 ± 4.480 |
| cpu/complex | `log` | c64 | 4 | `4194304` | 21.929 ± 7.570 | - | 15.463 ± 10.808 | 32.782 ± 8.827 | 34.219 ± 1.545 | 92.217 ± 5.564 | 42.013 ± 10.255 |
| cpu/complex | `mul` | c64 | 1 | `8388608` | 8.173 ± 0.318 | - | 8.303 ± 0.497 | 8.520 ± 0.172 | 8.095 ± 1.085 | 9.841 ± 1.007 | 8.592 ± 2.332 |
| cpu/complex | `mul` | c64 | 4 | `8388608` | 10.292 ± 0.698 | - | 11.117 ± 1.523 | 11.117 ± 2.163 | 9.476 ± 0.434 | 10.168 ± 1.144 | 15.762 ± 4.069 |
| cpu/complex | `norm_fro` | c64 | 1 | `2048x1536` | 0.883 ± 0.014 | - | 6.780 ± 0.430 | 6.466 ± 1.196 | 5.183 ± 0.162 | 8.559 ± 0.250 | - |
| cpu/complex | `norm_fro` | c64 | 4 | `2048x1536` | 1.196 ± 0.100 | - | 4.227 ± 0.312 | 4.646 ± 0.626 | 3.531 ± 0.653 | 12.121 ± 1.539 | - |
| cpu/complex | `qr` | c64 | 1 | `256x256` | 7.740 ± 0.318 | - | 6.173 ± 0.075 | 6.913 ± 0.097 | 7.150 ± 0.370 | 11.757 ± 2.207 | - |
| cpu/complex | `qr` | c64 | 4 | `256x256` | 8.756 ± 0.400 | - | 7.225 ± 0.631 | 8.195 ± 1.724 | 7.029 ± 0.615 | 149.153 ± 37.865 | - |
| cpu/complex | `solve` | c64 | 1 | `384x384,rhs=8` | 4.462 ± 0.089 | - | 4.557 ± 0.094 | 4.953 ± 0.138 | 2.373 ± 0.020 | 6.545 ± 1.084 | - |
| cpu/complex | `solve` | c64 | 4 | `384x384,rhs=8` | 5.799 ± 0.545 | - | 5.718 ± 0.135 | 7.135 ± 0.869 | 2.748 ± 0.185 | 123.103 ± 31.153 | - |
| cpu/complex | `svd` | c64 | 1 | `160x160` | 6.887 ± 0.368 | - | 8.292 ± 0.680 | 6.972 ± 0.113 | 6.131 ± 0.046 | 8.889 ± 0.355 | - |
| cpu/complex | `svd` | c64 | 4 | `160x160` | 6.862 ± 0.407 | - | 8.118 ± 0.391 | 9.145 ± 1.756 | 6.384 ± 0.314 | 640.036 ± 561.310 | - |
| cpu/complex | `tensordot` | c64 | 1 | `640x640` | 11.217 ± 0.259 | - | 11.522 ± 0.590 | 9.953 ± 0.766 | 48.964 ± 10.574 | 68.922 ± 1.602 | - |
| cpu/complex | `tensordot` | c64 | 4 | `640x640` | 10.205 ± 0.858 | - | 10.765 ± 0.412 | 11.199 ± 0.530 | 18.608 ± 2.230 | 32.035 ± 3.772 | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 1 | `1024x1024` | 9.679 ± 0.363 | - | unsupported | 11.440 ± 1.496 | 48.577 ± 2.287 | - | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 4 | `1024x1024` | 9.893 ± 0.501 | - | unsupported | 15.741 ± 1.595 | 29.787 ± 6.831 | - | - |
| cpu/elementwise_reduction | `abs` | f64 | 1 | `33554432` | 9.907 ± 1.147 | - | 9.595 ± 0.631 | 11.352 ± 1.149 | 13.045 ± 0.940 | 12.623 ± 4.206 | 14.511 ± 4.879 |
| cpu/elementwise_reduction | `abs` | f64 | 4 | `33554432` | 13.671 ± 1.204 | - | 13.521 ± 2.085 | 14.408 ± 2.176 | 15.510 ± 2.845 | 14.246 ± 2.149 | 12.534 ± 1.485 |
| cpu/elementwise_reduction | `add` | f64 | 1 | `33554432` | 18.683 ± 0.533 | - | 16.750 ± 1.288 | 19.292 ± 1.361 | 22.159 ± 1.936 | 20.497 ± 13.132 | 22.665 ± 1.382 |
| cpu/elementwise_reduction | `add` | f64 | 4 | `33554432` | 21.136 ± 2.844 | - | 21.989 ± 3.331 | 22.961 ± 6.590 | 27.172 ± 1.589 | 16.170 ± 2.930 | 19.429 ± 2.535 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 1 | `4194304` | 24.215 ± 0.186 | - | 33.217 ± 0.085 | 39.624 ± 2.319 | 28.493 ± 0.474 | 48.914 ± 0.666 | 50.687 ± 1.164 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 4 | `4194304` | 17.114 ± 6.009 | - | 21.798 ± 4.088 | 31.167 ± 5.873 | 22.547 ± 9.845 | 60.483 ± 10.287 | 32.978 ± 10.359 |
| cpu/elementwise_reduction | `clamp` | f64 | 1 | `8388608` | 5.685 ± 0.547 | - | 5.303 ± 0.317 | 6.449 ± 1.102 | 7.199 ± 0.085 | 7.000 ± 1.153 | 8.134 ± 4.095 |
| cpu/elementwise_reduction | `clamp` | f64 | 4 | `8388608` | 8.899 ± 1.723 | - | 9.214 ± 3.122 | 6.437 ± 0.453 | 9.922 ± 1.887 | 7.025 ± 3.838 | 7.097 ± 1.280 |
| cpu/elementwise_reduction | `compare_lt` | f64 | 1 | `33554432` | 13.484 ± 3.344 | - | 12.608 ± 2.356 | 17.851 ± 1.836 | 12.835 ± 0.364 | 16.013 ± 3.662 | - |
| cpu/elementwise_reduction | `compare_lt` | f64 | 4 | `33554432` | 13.255 ± 0.222 | - | 13.559 ± 0.200 | 15.639 ± 4.239 | 15.503 ± 1.004 | 13.711 ± 1.370 | - |
| cpu/elementwise_reduction | `cos` | f64 | 1 | `8388608` | 30.648 ± 0.176 | - | 30.561 ± 0.163 | 30.876 ± 0.609 | 27.844 ± 0.555 | 40.663 ± 45.538 | 36.537 ± 2.662 |
| cpu/elementwise_reduction | `cos` | f64 | 4 | `8388608` | 26.888 ± 5.671 | - | 21.995 ± 3.906 | 15.892 ± 5.855 | 17.901 ± 5.344 | 40.018 ± 2.545 | 21.128 ± 1.552 |
| cpu/elementwise_reduction | `div` | f64 | 1 | `33554432` | 15.568 ± 0.188 | - | 14.715 ± 1.076 | 23.330 ± 6.243 | 22.152 ± 1.364 | 23.212 ± 32.591 | 26.902 ± 6.721 |
| cpu/elementwise_reduction | `div` | f64 | 4 | `33554432` | 20.297 ± 0.505 | - | 20.828 ± 1.251 | 23.111 ± 2.493 | 26.597 ± 1.643 | 27.649 ± 21.802 | 22.260 ± 4.443 |
| cpu/elementwise_reduction | `exp` | f64 | 1 | `8388608` | 15.938 ± 0.082 | - | 15.966 ± 0.058 | 28.085 ± 0.529 | 16.612 ± 0.288 | 36.227 ± 13.764 | 35.181 ± 8.321 |
| cpu/elementwise_reduction | `exp` | f64 | 4 | `8388608` | 18.428 ± 4.802 | - | 13.182 ± 2.644 | 12.769 ± 4.947 | 9.967 ± 1.402 | 36.282 ± 9.342 | 13.636 ± 2.370 |
| cpu/elementwise_reduction | `expm1` | f64 | 1 | `4194304` | 25.805 ± 0.134 | - | 25.549 ± 2.206 | 28.933 ± 1.340 | 16.297 ± 0.131 | 24.212 ± 0.763 | 25.791 ± 0.922 |
| cpu/elementwise_reduction | `expm1` | f64 | 4 | `4194304` | 15.851 ± 2.821 | - | 15.527 ± 2.682 | 17.054 ± 2.496 | 13.794 ± 11.354 | 37.326 ± 9.014 | 22.108 ± 2.648 |
| cpu/elementwise_reduction | `log` | f64 | 1 | `8388608` | 28.540 ± 0.358 | - | 29.665 ± 0.092 | 40.573 ± 0.990 | 25.878 ± 0.312 | 48.948 ± 12.396 | 58.858 ± 19.403 |
| cpu/elementwise_reduction | `log` | f64 | 4 | `8388608` | 28.554 ± 1.805 | - | 23.127 ± 10.708 | 25.637 ± 0.948 | 17.797 ± 9.538 | 43.758 ± 4.834 | 18.963 ± 4.240 |
| cpu/elementwise_reduction | `log1p` | f64 | 1 | `4194304` | 14.422 ± 0.112 | - | 14.435 ± 0.388 | 22.956 ± 0.806 | 17.483 ± 0.149 | 24.144 ± 1.988 | 23.589 ± 2.793 |
| cpu/elementwise_reduction | `log1p` | f64 | 4 | `4194304` | 9.775 ± 2.450 | - | 11.412 ± 1.604 | 17.652 ± 2.442 | 10.212 ± 5.495 | 36.412 ± 18.619 | 18.253 ± 11.408 |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 1 | `1024x64` | 0.242 ± 0.007 | 0.254 ± 0.010 | unsupported | unsupported | 0.278 ± 0.018 | - | - |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 4 | `1024x64` | 0.331 ± 0.064 | 0.420 ± 0.430 | unsupported | unsupported | 0.273 ± 0.083 | - | - |
| cpu/elementwise_reduction | `maximum` | f64 | 1 | `33554432` | 16.299 ± 0.870 | - | 17.173 ± 2.488 | 19.897 ± 1.231 | 22.177 ± 1.699 | 15.777 ± 10.056 | 16.580 ± 2.257 |
| cpu/elementwise_reduction | `maximum` | f64 | 4 | `33554432` | 22.008 ± 2.024 | - | 20.695 ± 1.694 | 20.752 ± 1.512 | 25.674 ± 2.045 | 18.491 ± 5.841 | 16.823 ± 3.013 |
| cpu/elementwise_reduction | `minimum` | f64 | 1 | `33554432` | 16.240 ± 1.844 | - | 18.292 ± 4.452 | 20.188 ± 2.775 | 19.194 ± 0.227 | 23.088 ± 8.391 | 31.409 ± 5.011 |
| cpu/elementwise_reduction | `minimum` | f64 | 4 | `33554432` | 20.971 ± 1.054 | - | 19.297 ± 1.214 | 23.291 ± 1.493 | 25.105 ± 1.846 | 18.771 ± 3.886 | 16.612 ± 1.355 |
| cpu/elementwise_reduction | `mul` | f64 | 1 | `33554432` | 16.255 ± 0.929 | - | 16.160 ± 1.729 | 18.140 ± 6.742 | 22.221 ± 0.734 | 18.847 ± 2.890 | 22.699 ± 6.272 |
| cpu/elementwise_reduction | `mul` | f64 | 4 | `33554432` | 19.171 ± 0.447 | - | 19.791 ± 0.530 | 26.229 ± 5.462 | 27.446 ± 4.655 | 21.125 ± 0.460 | 20.423 ± 2.254 |
| cpu/elementwise_reduction | `neg` | f64 | 1 | `33554432` | 9.425 ± 0.179 | - | 9.343 ± 1.227 | 12.296 ± 2.189 | 13.291 ± 1.278 | 13.799 ± 18.875 | 20.775 ± 7.449 |
| cpu/elementwise_reduction | `neg` | f64 | 4 | `33554432` | 13.472 ± 0.777 | - | 13.440 ± 1.109 | 13.915 ± 0.660 | 17.826 ± 3.553 | 15.988 ± 3.929 | 14.460 ± 3.027 |
| cpu/elementwise_reduction | `pow` | f64 | 1 | `4194304` | 35.540 ± 0.917 | - | 35.283 ± 1.428 | 66.981 ± 4.852 | 44.134 ± 0.521 | 81.588 ± 11.992 | 80.381 ± 6.923 |
| cpu/elementwise_reduction | `pow` | f64 | 4 | `4194304` | 21.261 ± 2.733 | - | 21.047 ± 1.092 | 53.912 ± 7.367 | 26.224 ± 5.905 | 88.774 ± 6.378 | 52.750 ± 8.664 |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 1 | `8192x4096` | 6.916 ± 0.631 | - | 6.336 ± 0.040 | 12.302 ± 0.580 | 21.177 ± 0.437 | 6.886 ± 0.824 | - |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 4 | `8192x4096` | 6.381 ± 0.858 | - | 7.265 ± 1.673 | 7.343 ± 0.868 | 18.508 ± 3.467 | 10.143 ± 1.418 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 1 | `2048x2048` | 0.854 ± 0.050 | - | 0.853 ± 0.037 | 16.060 ± 7.961 | 2.042 ± 0.139 | 0.779 ± 0.271 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 4 | `2048x2048` | 1.125 ± 0.375 | - | 1.059 ± 0.544 | 18.811 ± 1.732 | 2.599 ± 0.183 | 1.294 ± 1.368 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 1 | `2048x2048` | 0.790 ± 0.018 | - | 0.900 ± 0.078 | 1.653 ± 0.076 | 0.761 ± 0.008 | 0.852 ± 0.057 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 4 | `2048x2048` | 2.505 ± 0.612 | - | 2.250 ± 0.543 | 0.951 ± 0.061 | 1.064 ± 0.031 | 0.952 ± 0.209 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 1 | `8192x4096` | 7.095 ± 0.702 | - | 6.681 ± 0.598 | 12.216 ± 0.543 | 21.197 ± 0.247 | 7.382 ± 1.544 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 4 | `8192x4096` | 6.873 ± 2.410 | - | 8.464 ± 1.721 | 7.024 ± 0.543 | 17.744 ± 1.321 | 8.694 ± 3.263 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 1 | `2048x2048` | 0.855 ± 0.011 | - | 0.870 ± 0.021 | 15.862 ± 0.616 | 1.843 ± 0.011 | 0.747 ± 0.135 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 4 | `2048x2048` | 1.017 ± 0.277 | - | 1.098 ± 0.177 | 12.803 ± 2.149 | 2.378 ± 0.681 | 1.325 ± 0.684 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 1 | `4096x4096` | 3.187 ± 0.088 | - | 3.178 ± 0.318 | 7.167 ± 0.505 | 2.734 ± 0.684 | 3.450 ± 0.296 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 4 | `4096x4096` | 7.146 ± 0.464 | - | 7.505 ± 0.598 | 4.226 ± 0.771 | 4.015 ± 0.168 | 8.682 ± 1.745 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 1 | `8192x4096` | 6.373 ± 0.959 | - | 5.976 ± 0.201 | 6.553 ± 1.718 | 40.290 ± 0.485 | 7.685 ± 0.516 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 4 | `8192x4096` | 7.174 ± 0.589 | - | 6.717 ± 1.596 | 7.101 ± 0.601 | 28.980 ± 8.246 | 9.503 ± 1.211 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 1 | `2048x2048` | 0.770 ± 0.022 | - | 0.825 ± 0.057 | 4.801 ± 0.646 | 1.883 ± 0.038 | 0.819 ± 0.052 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 4 | `2048x2048` | 0.913 ± 0.380 | - | 0.936 ± 0.179 | 2.470 ± 0.354 | 2.308 ± 0.297 | 0.978 ± 0.124 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 1 | `2048x2048` | 0.784 ± 0.028 | - | 0.747 ± 0.039 | 0.959 ± 0.198 | 1.118 ± 0.018 | 0.889 ± 0.165 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 4 | `2048x2048` | 1.940 ± 0.054 | - | 2.140 ± 0.176 | 0.648 ± 0.002 | 0.929 ± 0.116 | 1.021 ± 0.195 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 1 | `8192x4096` | 5.445 ± 0.142 | - | 5.852 ± 0.634 | 5.470 ± 1.211 | 29.982 ± 1.448 | 7.944 ± 1.107 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 4 | `8192x4096` | 8.112 ± 1.743 | - | 8.441 ± 1.914 | 6.729 ± 1.115 | 19.071 ± 1.986 | 9.697 ± 2.421 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 1 | `2048x2048` | 0.650 ± 0.023 | - | 0.678 ± 0.034 | 4.589 ± 0.531 | 1.843 ± 0.012 | 0.670 ± 0.038 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 4 | `2048x2048` | 0.846 ± 0.266 | - | 0.917 ± 0.113 | 3.279 ± 2.209 | 3.417 ± 1.312 | 1.155 ± 0.471 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 1 | `2048x2048` | 0.737 ± 0.007 | - | 0.770 ± 0.027 | 0.849 ± 0.091 | 0.977 ± 0.077 | 0.827 ± 0.067 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 4 | `2048x2048` | 2.078 ± 0.172 | - | 2.128 ± 0.230 | 0.686 ± 0.049 | 1.230 ± 0.122 | 1.289 ± 0.133 | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 1 | `2048x2048` | 0.696 ± 0.025 | 0.704 ± 0.037 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 4 | `2048x2048` | 0.918 ± 0.050 | 0.897 ± 0.108 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `rem` | f64 | 1 | `8388608` | 27.272 ± 0.106 | - | 26.892 ± 0.341 | 33.231 ± 8.586 | 35.505 ± 0.675 | 38.017 ± 47.687 | 31.686 ± 6.244 |
| cpu/elementwise_reduction | `rem` | f64 | 4 | `8388608` | 15.200 ± 2.384 | - | 19.220 ± 7.249 | 21.727 ± 6.551 | 28.135 ± 3.850 | 25.134 ± 2.242 | 17.774 ± 3.143 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 1 | `33554432` | 18.816 ± 0.336 | - | 18.887 ± 0.081 | 17.985 ± 0.434 | 35.533 ± 0.291 | 45.543 ± 1.341 | 45.331 ± 7.498 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 4 | `33554432` | 13.409 ± 1.216 | - | 13.804 ± 0.743 | 18.532 ± 7.974 | 25.288 ± 5.170 | 51.539 ± 7.064 | 17.783 ± 7.965 |
| cpu/elementwise_reduction | `select` | f64 | 1 | `33554432` | 25.044 ± 2.471 | - | 25.064 ± 2.227 | 17.959 ± 2.359 | 25.581 ± 2.104 | 33.019 ± 9.559 | - |
| cpu/elementwise_reduction | `select` | f64 | 4 | `33554432` | 27.173 ± 2.417 | - | 27.835 ± 2.259 | 22.542 ± 1.745 | 27.901 ± 3.906 | 28.102 ± 4.637 | - |
| cpu/elementwise_reduction | `sign` | f64 | 1 | `33554432` | 10.578 ± 0.702 | - | 11.247 ± 1.260 | 12.249 ± 2.466 | 14.758 ± 0.291 | 11.578 ± 0.798 | 18.127 ± 2.167 |
| cpu/elementwise_reduction | `sign` | f64 | 4 | `33554432` | 12.891 ± 1.292 | - | 14.024 ± 1.188 | 16.273 ± 3.408 | 17.246 ± 2.493 | 11.560 ± 5.394 | 10.240 ± 0.605 |
| cpu/elementwise_reduction | `sin` | f64 | 1 | `8388608` | 23.540 ± 0.086 | - | 23.641 ± 0.087 | 27.171 ± 0.431 | 25.251 ± 0.114 | 36.973 ± 5.436 | 39.808 ± 11.741 |
| cpu/elementwise_reduction | `sin` | f64 | 4 | `8388608` | 20.329 ± 3.912 | - | 19.686 ± 3.712 | 13.509 ± 3.730 | 17.044 ± 6.445 | 30.992 ± 1.127 | 14.432 ± 3.026 |
| cpu/elementwise_reduction | `sqrt` | f64 | 1 | `33554432` | 12.758 ± 0.320 | - | 12.641 ± 0.314 | 12.071 ± 0.838 | 20.983 ± 3.553 | 26.990 ± 2.240 | 32.640 ± 7.747 |
| cpu/elementwise_reduction | `sqrt` | f64 | 4 | `33554432` | 13.119 ± 0.181 | - | 14.023 ± 1.838 | 13.845 ± 1.916 | 19.803 ± 8.729 | 37.872 ± 13.922 | 14.181 ± 2.790 |
| cpu/elementwise_reduction | `sub` | f64 | 1 | `33554432` | 17.779 ± 3.556 | - | 16.404 ± 0.731 | 19.782 ± 2.036 | 24.286 ± 3.746 | 21.247 ± 6.925 | 20.049 ± 3.532 |
| cpu/elementwise_reduction | `sub` | f64 | 4 | `33554432` | 20.620 ± 1.660 | - | 19.679 ± 1.500 | 24.561 ± 3.698 | 26.439 ± 1.471 | 19.596 ± 2.445 | 21.527 ± 1.508 |
| cpu/elementwise_reduction | `tanh` | f64 | 1 | `8388608` | 63.952 ± 1.760 | - | 74.113 ± 10.805 | 65.527 ± 0.645 | 13.253 ± 0.238 | 43.425 ± 1.843 | 49.618 ± 7.547 |
| cpu/elementwise_reduction | `tanh` | f64 | 4 | `8388608` | 56.575 ± 12.139 | - | 38.702 ± 7.105 | 32.702 ± 3.887 | 7.806 ± 0.436 | 48.173 ± 0.893 | 22.979 ± 7.214 |
| cpu/elementwise_reduction | `where_select` | f64 | 1 | `33554432` | 23.265 ± 0.381 | 28.858 ± 3.879 | unsupported | 18.317 ± 0.953 | 31.212 ± 1.604 | - | - |
| cpu/elementwise_reduction | `where_select` | f64 | 4 | `33554432` | 28.455 ± 1.375 | 31.502 ± 1.520 | unsupported | 19.574 ± 3.044 | 37.170 ± 13.032 | - | - |
| cpu/indexing_layout | `concatenate` | f64 | 1 | `1048576+1048576` | 0.742 ± 0.156 | - | 0.548 ± 0.020 | 0.746 ± 0.089 | 0.690 ± 0.022 | 0.649 ± 0.018 | - |
| cpu/indexing_layout | `concatenate` | f64 | 4 | `1048576+1048576` | 1.016 ± 0.193 | - | 1.171 ± 0.548 | 0.731 ± 0.084 | 0.924 ± 0.174 | 1.015 ± 0.181 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 1 | `4194304 -> 2097152` | 0.556 ± 0.001 | - | 0.558 ± 0.079 | 0.716 ± 0.035 | 0.810 ± 0.051 | 0.762 ± 0.072 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 4 | `4194304 -> 2097152` | 1.022 ± 0.340 | - | 0.856 ± 0.128 | 0.673 ± 0.032 | 1.488 ± 0.657 | 0.888 ± 0.119 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 1 | `2097152` | 0.799 ± 0.196 | - | unsupported | 1.152 ± 0.162 | 1.096 ± 0.107 | 0.935 ± 0.090 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 4 | `2097152` | 1.256 ± 0.113 | - | unsupported | 1.350 ± 0.352 | 1.491 ± 0.374 | 1.572 ± 1.186 | - |
| cpu/indexing_layout | `gather` | f64 | 1 | `262144` | 0.329 ± 0.001 | - | 0.377 ± 0.009 | 0.331 ± 0.069 | 0.419 ± 0.080 | 3.640 ± 5.597 | - |
| cpu/indexing_layout | `gather` | f64 | 4 | `262144` | 0.588 ± 0.181 | - | 0.907 ± 0.766 | 0.303 ± 0.034 | 0.246 ± 0.012 | 1.030 ± 0.150 | - |
| cpu/indexing_layout | `pad` | f64 | 1 | `2097152` | 0.504 ± 0.005 | - | 0.620 ± 0.094 | 1.081 ± 0.047 | 0.735 ± 0.052 | - | - |
| cpu/indexing_layout | `pad` | f64 | 4 | `2097152` | 0.789 ± 0.150 | - | 0.798 ± 0.212 | 1.189 ± 0.149 | 1.099 ± 0.399 | - | - |
| cpu/indexing_layout | `reverse` | f64 | 1 | `2097152` | 0.577 ± 0.158 | - | 0.543 ± 0.034 | 0.824 ± 0.154 | 0.718 ± 0.074 | 1.091 ± 0.280 | - |
| cpu/indexing_layout | `reverse` | f64 | 4 | `2097152` | 0.973 ± 0.212 | - | 1.014 ± 0.392 | 0.725 ± 0.041 | 2.065 ± 2.417 | 1.626 ± 0.635 | - |
| cpu/indexing_layout | `scatter` | f64 | 1 | `262144` | 0.513 ± 0.001 | - | 0.572 ± 0.019 | 0.396 ± 0.023 | 0.440 ± 0.060 | 0.930 ± 0.071 | - |
| cpu/indexing_layout | `scatter` | f64 | 4 | `262144` | 1.256 ± 1.525 | - | 1.317 ± 0.348 | 0.474 ± 0.060 | 0.468 ± 0.025 | 1.340 ± 1.799 | - |
| cpu/indexing_layout | `slice` | f64 | 1 | `4194304 -> 2096128` | 1.004 ± 0.094 | - | 0.001 ± 0.000 | 1.219 ± 0.075 | 1.490 ± 0.066 | 1.454 ± 0.186 | - |
| cpu/indexing_layout | `slice` | f64 | 4 | `4194304 -> 2096128` | 1.514 ± 0.157 | - | 0.004 ± 0.004 | 1.057 ± 0.110 | 1.378 ± 0.234 | 1.482 ± 0.073 | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x2x2` | 0.055 ± 0.001 | - | 0.078 ± 0.003 | 0.132 ± 0.068 | 0.093 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x4x4` | 0.091 ± 0.003 | - | 0.120 ± 0.000 | 0.174 ± 0.017 | 0.143 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x8x8` | 0.203 ± 0.004 | - | 0.213 ± 0.005 | 0.287 ± 0.013 | 0.208 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x128x128` | 0.027 ± 0.002 | - | 0.029 ± 0.003 | 0.040 ± 0.003 | 0.077 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.004 ± 0.000 | 0.024 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x32x32` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.001 | 0.048 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x4x4` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x8x8` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x4x4` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x8x8` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.007 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x4x4` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x8x8` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.051 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x128x128` | 0.210 ± 0.005 | - | 0.177 ± 0.002 | 0.257 ± 0.013 | 0.242 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x32x32` | 0.011 ± 0.000 | - | 0.013 ± 0.001 | 0.017 ± 0.001 | 0.046 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x4x4` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x8x8` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.050 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x2x2` | 0.059 ± 0.000 | - | 0.082 ± 0.002 | 0.288 ± 0.082 | 0.111 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x4x4` | 0.095 ± 0.005 | - | 0.121 ± 0.001 | 0.385 ± 0.053 | 0.168 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x8x8` | 0.202 ± 0.010 | - | 0.218 ± 0.012 | 0.454 ± 0.227 | 0.271 ± 0.032 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x128x128` | 0.025 ± 0.001 | - | 0.028 ± 0.002 | 0.159 ± 0.025 | 0.134 ± 0.056 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x2x2` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.100 ± 0.028 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x32x32` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.110 ± 0.014 | 0.009 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x4x4` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.092 ± 0.071 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x8x8` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.070 ± 0.023 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.097 ± 0.044 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.078 ± 0.087 | 0.009 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x8x8` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.082 ± 0.020 | 0.024 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.049 ± 0.082 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.098 ± 0.023 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x8x8` | 0.002 ± 0.002 | - | 0.004 ± 0.000 | 0.106 ± 0.044 | 0.009 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x128x128` | 0.174 ± 0.001 | - | 0.182 ± 0.011 | 0.532 ± 0.143 | 0.499 ± 0.340 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x2x2` | 0.002 ± 0.006 | - | 0.004 ± 0.000 | 0.092 ± 0.037 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x32x32` | 0.012 ± 0.000 | - | 0.014 ± 0.000 | 0.122 ± 0.013 | 0.054 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.091 ± 0.010 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x8x8` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.094 ± 0.069 | 0.045 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x2x2` | 0.240 ± 0.001 | - | 0.241 ± 0.001 | 0.458 ± 0.032 | 0.809 ± 0.030 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x4x4` | 1.408 ± 0.011 | - | 1.422 ± 0.012 | 1.785 ± 0.179 | 2.040 ± 0.048 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x8x8` | 5.824 ± 0.039 | - | 5.831 ± 0.023 | 6.779 ± 0.142 | 11.217 ± 0.851 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x128x128` | 4.227 ± 0.012 | - | 5.959 ± 0.046 | 3.264 ± 0.059 | 3.777 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x2x2` | 0.001 ± 0.001 | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x32x32` | 0.087 ± 0.001 | - | 0.088 ± 0.001 | 0.093 ± 0.001 | 0.154 ± 0.035 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x8x8` | 0.007 ± 0.000 | - | 0.008 ± 0.000 | 0.011 ± 0.003 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.006 ± 0.001 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x4x4` | 0.005 ± 0.000 | - | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x8x8` | 0.018 ± 0.000 | - | 0.019 ± 0.001 | 0.023 ± 0.001 | 0.066 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x4x4` | 0.007 ± 0.000 | - | 0.008 ± 0.000 | 0.011 ± 0.000 | 0.018 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x8x8` | 0.024 ± 0.001 | - | 0.025 ± 0.000 | 0.033 ± 0.007 | 0.061 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x128x128` | 35.472 ± 0.183 | - | 48.671 ± 2.146 | 25.478 ± 0.890 | 30.147 ± 0.426 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x32x32` | 0.670 ± 0.013 | - | 0.671 ± 0.002 | 0.737 ± 0.036 | 0.845 ± 0.036 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x4x4` | 0.012 ± 0.000 | - | 0.013 ± 0.000 | 0.017 ± 0.000 | 0.027 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x8x8` | 0.048 ± 0.000 | - | 0.049 ± 0.000 | 0.054 ± 0.001 | 0.115 ± 0.028 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x2x2` | 0.263 ± 0.012 | - | 0.276 ± 0.033 | 0.558 ± 0.217 | 0.825 ± 0.033 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x4x4` | 1.524 ± 0.025 | - | 1.639 ± 0.158 | 1.992 ± 0.390 | 1.155 ± 0.264 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x8x8` | 6.400 ± 0.163 | - | 6.326 ± 0.979 | 7.591 ± 0.926 | 3.958 ± 0.144 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x128x128` | 4.781 ± 0.212 | - | 5.209 ± 0.180 | 3.585 ± 0.725 | 4.311 ± 0.374 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.001 | 0.016 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x32x32` | 0.100 ± 0.019 | - | 0.093 ± 0.001 | 0.105 ± 0.016 | 0.177 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x4x4` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x8x8` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.010 ± 0.003 | 0.018 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.007 ± 0.001 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x4x4` | 0.006 ± 0.000 | - | 0.008 ± 0.002 | 0.011 ± 0.002 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x8x8` | 0.019 ± 0.002 | - | 0.021 ± 0.004 | 0.023 ± 0.006 | 0.086 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.006 ± 0.001 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x4x4` | 0.010 ± 0.002 | - | 0.010 ± 0.000 | 0.012 ± 0.003 | 0.035 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x8x8` | 0.030 ± 0.001 | - | 0.029 ± 0.005 | 0.032 ± 0.007 | 0.081 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x128x128` | 42.471 ± 5.699 | - | 52.394 ± 4.935 | 31.155 ± 3.065 | 17.479 ± 0.940 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x2x2` | 0.005 ± 0.001 | - | 0.006 ± 0.001 | 0.008 ± 0.002 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x32x32` | 0.752 ± 0.057 | - | 0.740 ± 0.038 | 0.750 ± 0.042 | 0.508 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x4x4` | 0.013 ± 0.000 | - | 0.015 ± 0.001 | 0.021 ± 0.001 | 0.026 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x8x8` | 0.059 ± 0.003 | - | 0.051 ± 0.006 | 0.053 ± 0.003 | 0.093 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x2x2` | 0.229 ± 0.005 | - | 0.297 ± 0.098 | 0.322 ± 0.048 | 0.592 ± 0.063 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x4x4` | 1.201 ± 0.026 | - | 1.190 ± 0.008 | 1.395 ± 0.037 | 1.542 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x8x8` | 4.630 ± 0.049 | - | 4.681 ± 0.432 | 5.426 ± 0.943 | 5.271 ± 0.040 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x128x128` | 0.915 ± 0.006 | - | 0.945 ± 0.046 | 1.035 ± 0.058 | 1.041 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.001 | 0.044 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x32x32` | 0.062 ± 0.002 | - | 0.064 ± 0.002 | 0.074 ± 0.009 | 0.140 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x4x4` | 0.003 ± 0.001 | - | 0.005 ± 0.003 | 0.005 ± 0.001 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x8x8` | 0.005 ± 0.000 | - | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x4x4` | 0.006 ± 0.003 | - | 0.019 ± 0.005 | 0.017 ± 0.013 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x8x8` | 0.015 ± 0.000 | - | 0.016 ± 0.000 | 0.019 ± 0.001 | 0.026 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.014 ± 0.023 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x4x4` | 0.009 ± 0.011 | - | 0.022 ± 0.010 | 0.009 ± 0.001 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x8x8` | 0.019 ± 0.001 | - | 0.020 ± 0.001 | 0.030 ± 0.009 | 0.030 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x128x128` | 7.271 ± 0.046 | - | 7.257 ± 0.282 | 8.104 ± 0.152 | 7.916 ± 0.113 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x32x32` | 0.487 ± 0.002 | - | 0.491 ± 0.070 | 0.531 ± 0.014 | 0.577 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x4x4` | 0.016 ± 0.008 | - | 0.077 ± 0.103 | 0.023 ± 0.014 | 0.022 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x8x8` | 0.037 ± 0.001 | - | 0.038 ± 0.002 | 0.045 ± 0.005 | 0.112 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x2x2` | 0.251 ± 0.019 | - | 0.239 ± 0.010 | 0.337 ± 0.057 | 0.570 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x4x4` | 1.313 ± 0.030 | - | 1.300 ± 0.032 | 1.509 ± 0.304 | 1.020 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x8x8` | 4.921 ± 0.211 | - | 5.114 ± 0.230 | 5.696 ± 1.780 | 2.783 ± 0.036 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x128x128` | 1.006 ± 0.036 | - | 1.004 ± 0.024 | 1.041 ± 0.047 | 1.116 ± 0.034 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x2x2` | 0.003 ± 0.001 | - | 0.004 ± 0.001 | 0.004 ± 0.001 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x32x32` | 0.069 ± 0.006 | - | 0.068 ± 0.004 | 0.074 ± 0.005 | 0.133 ± 0.057 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x4x4` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.015 ± 0.016 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x8x8` | 0.007 ± 0.002 | - | 0.008 ± 0.000 | 0.009 ± 0.002 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x2x2` | 0.003 ± 0.004 | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x4x4` | 0.006 ± 0.001 | - | 0.008 ± 0.001 | 0.027 ± 0.037 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x8x8` | 0.016 ± 0.001 | - | 0.018 ± 0.000 | 0.019 ± 0.004 | 0.027 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x2x2` | 0.003 ± 0.001 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x4x4` | 0.007 ± 0.000 | - | 0.009 ± 0.002 | 0.029 ± 0.004 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x8x8` | 0.020 ± 0.004 | - | 0.022 ± 0.000 | 0.028 ± 0.006 | 0.031 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x128x128` | 8.043 ± 0.439 | - | 7.990 ± 0.466 | 9.527 ± 0.710 | 4.683 ± 0.162 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x2x2` | 0.009 ± 0.007 | - | 0.006 ± 0.000 | 0.007 ± 0.017 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x32x32` | 0.545 ± 0.027 | - | 0.531 ± 0.017 | 0.532 ± 0.026 | 0.417 ± 0.090 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x4x4` | 0.011 ± 0.000 | - | 0.013 ± 0.000 | 0.042 ± 0.000 | 0.022 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x8x8` | 0.039 ± 0.001 | - | 0.046 ± 0.013 | 0.047 ± 0.003 | 0.101 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x2x2` | 0.116 ± 0.000 | - | 0.118 ± 0.001 | 0.126 ± 0.001 | 0.511 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x4x4` | 0.731 ± 0.020 | - | 0.732 ± 0.025 | 0.738 ± 0.017 | 1.432 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x8x8` | 2.341 ± 0.029 | - | 2.345 ± 0.012 | 2.360 ± 0.008 | 5.024 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x128x128` | 0.445 ± 0.008 | - | 0.445 ± 0.002 | 0.453 ± 0.020 | 0.976 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.005 ± 0.000 | 0.015 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x32x32` | 0.025 ± 0.001 | - | 0.026 ± 0.000 | 0.029 ± 0.000 | 0.099 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.006 ± 0.001 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x8x8` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.013 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.007 ± 0.010 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x8x8` | 0.008 ± 0.000 | - | 0.009 ± 0.000 | 0.012 ± 0.000 | 0.023 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x4x4` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x8x8` | 0.010 ± 0.000 | - | 0.011 ± 0.000 | 0.014 ± 0.000 | 0.027 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x128x128` | 3.600 ± 0.042 | - | 3.683 ± 0.416 | 3.606 ± 0.073 | 7.562 ± 0.037 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.001 | 0.005 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x32x32` | 0.191 ± 0.003 | - | 0.192 ± 0.001 | 0.196 ± 0.001 | 0.540 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x4x4` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.012 ± 0.003 | 0.018 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x8x8` | 0.019 ± 0.000 | - | 0.021 ± 0.000 | 0.023 ± 0.003 | 0.077 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x2x2` | 0.162 ± 0.013 | - | 0.165 ± 0.002 | 0.146 ± 0.063 | 0.601 ± 0.068 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x4x4` | 0.832 ± 0.052 | - | 0.858 ± 0.052 | 0.802 ± 0.173 | 1.951 ± 0.829 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x8x8` | 2.739 ± 0.168 | - | 2.496 ± 0.334 | 2.980 ± 0.243 | 3.126 ± 0.877 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x128x128` | 0.446 ± 0.009 | - | 0.574 ± 0.101 | 0.758 ± 0.716 | 1.097 ± 0.032 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.001 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x32x32` | 0.027 ± 0.003 | - | 0.029 ± 0.000 | 0.037 ± 0.003 | 0.153 ± 0.035 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x4x4` | 0.006 ± 0.000 | - | 0.005 ± 0.001 | 0.007 ± 0.001 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x8x8` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.016 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x2x2` | 0.005 ± 0.000 | - | 0.006 ± 0.004 | 0.007 ± 0.002 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x4x4` | 0.009 ± 0.009 | - | 0.005 ± 0.000 | 0.007 ± 0.002 | 0.017 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x8x8` | 0.009 ± 0.000 | - | 0.012 ± 0.002 | 0.030 ± 0.023 | 0.034 ± 0.096 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.001 | 0.006 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x4x4` | 0.013 ± 0.015 | - | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x8x8` | 0.011 ± 0.000 | - | 0.013 ± 0.000 | 0.015 ± 0.002 | 0.029 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x128x128` | 4.126 ± 0.136 | - | 4.256 ± 0.265 | 4.794 ± 0.610 | 4.446 ± 0.145 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.001 | 0.005 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x32x32` | 0.193 ± 0.030 | - | 0.199 ± 0.000 | 0.226 ± 0.022 | 0.385 ± 0.059 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x4x4` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.010 ± 0.000 | 0.023 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x8x8` | 0.025 ± 0.005 | - | 0.043 ± 0.033 | 0.023 ± 0.002 | 0.094 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x2x2` | 0.089 ± 0.001 | - | 0.093 ± 0.005 | 0.148 ± 0.001 | 0.151 ± 0.040 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x4x4` | 0.195 ± 0.017 | - | 0.194 ± 0.001 | 0.234 ± 0.002 | 0.242 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x8x8` | 0.432 ± 0.005 | - | 0.431 ± 0.002 | 0.476 ± 0.009 | 0.401 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x128x128` | 0.059 ± 0.004 | - | 0.059 ± 0.004 | 0.067 ± 0.002 | 0.116 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.007 ± 0.001 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x32x32` | 0.005 ± 0.000 | - | 0.006 ± 0.000 | 0.010 ± 0.000 | 0.043 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x64x24` | 0.008 ± 0.000 | - | 0.009 ± 0.000 | 0.013 ± 0.001 | 0.047 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x8x8` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.006 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.007 ± 0.000 | 0.012 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x4x4` | 0.003 ± 0.000 | - | 0.003 ± 0.000 | 0.007 ± 0.000 | 0.012 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x8x8` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.068 ± 0.050 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.007 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x8x8` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.009 ± 0.000 | 0.054 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x128x128` | 0.453 ± 0.005 | - | 0.459 ± 0.007 | 0.462 ± 0.007 | 0.548 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.011 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x32x32` | 0.032 ± 0.000 | - | 0.033 ± 0.000 | 0.038 ± 0.000 | 0.089 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.009 ± 0.000 | 0.065 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x8x8` | 0.005 ± 0.000 | - | 0.006 ± 0.000 | 0.010 ± 0.000 | 0.044 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x2x2` | 0.090 ± 0.000 | - | 0.091 ± 0.001 | 0.474 ± 0.393 | 0.264 ± 0.109 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x4x4` | 0.237 ± 0.028 | - | 0.195 ± 0.017 | 0.271 ± 0.040 | 0.605 ± 0.079 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x8x8` | 0.440 ± 0.043 | - | 0.449 ± 0.028 | 1.592 ± 1.446 | 0.523 ± 0.359 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x128x128` | 0.062 ± 0.002 | - | 0.064 ± 0.001 | 0.255 ± 0.155 | 0.142 ± 0.029 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.033 ± 0.030 | 0.010 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x32x32` | 0.006 ± 0.000 | - | 0.007 ± 0.000 | 0.080 ± 0.100 | 0.035 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.020 ± 0.000 | 0.012 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x64x24` | 0.010 ± 0.003 | - | 0.010 ± 0.000 | 0.048 ± 0.036 | 0.085 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x8x8` | 0.003 ± 0.012 | - | 0.005 ± 0.001 | 0.024 ± 0.004 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.039 ± 0.067 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x4x4` | 0.003 ± 0.001 | - | 0.004 ± 0.000 | 0.026 ± 0.037 | 0.013 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x8x8` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.025 ± 0.005 | 0.057 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.061 ± 0.010 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.025 ± 0.005 | 0.011 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x8x8` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.024 ± 0.034 | 0.063 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x128x128` | 0.468 ± 0.010 | - | 0.489 ± 0.011 | 2.369 ± 1.476 | 0.698 ± 0.507 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x2x2` | 0.003 ± 0.001 | - | 0.005 ± 0.000 | 0.078 ± 0.023 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x32x32` | 0.032 ± 0.000 | - | 0.033 ± 0.000 | 0.058 ± 0.005 | 0.146 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x4x4` | 0.004 ± 0.001 | - | 0.006 ± 0.001 | 0.032 ± 0.005 | 0.072 ± 0.056 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x8x8` | 0.005 ± 0.000 | - | 0.008 ± 0.035 | 0.097 ± 0.327 | 0.056 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x2x2` | 0.260 ± 0.000 | - | 0.262 ± 0.001 | 0.325 ± 0.023 | 0.313 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x4x4` | 0.598 ± 0.002 | - | 0.519 ± 0.124 | 0.716 ± 0.044 | 0.633 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x8x8` | 1.249 ± 0.095 | - | 1.217 ± 0.033 | 1.374 ± 0.123 | 1.466 ± 0.185 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x128x128` | 0.312 ± 0.018 | - | 0.308 ± 0.019 | 0.361 ± 0.028 | 0.378 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.001 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x32x32` | 0.012 ± 0.000 | - | 0.013 ± 0.001 | 0.018 ± 0.005 | 0.075 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x64x24` | 0.014 ± 0.000 | - | 0.015 ± 0.000 | 0.018 ± 0.001 | 0.081 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x8x8` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.004 ± 0.001 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.006 ± 0.001 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x8x8` | 0.005 ± 0.000 | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.005 ± 0.001 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x8x8` | 0.006 ± 0.000 | - | 0.007 ± 0.000 | 0.009 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x128x128` | 2.809 ± 0.460 | - | 2.391 ± 0.082 | 2.996 ± 0.078 | 2.671 ± 0.101 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.006 ± 0.002 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x32x32` | 0.086 ± 0.002 | - | 0.088 ± 0.014 | 0.093 ± 0.009 | 0.141 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x4x4` | 0.006 ± 0.000 | - | 0.009 ± 0.000 | 0.010 ± 0.001 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x8x8` | 0.011 ± 0.000 | - | 0.012 ± 0.000 | 0.015 ± 0.000 | 0.021 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x2x2` | 0.233 ± 0.010 | - | 0.239 ± 0.009 | 0.675 ± 0.178 | 0.362 ± 0.048 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x4x4` | 0.522 ± 0.078 | - | 0.599 ± 0.253 | 1.062 ± 0.456 | 0.713 ± 0.771 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x8x8` | 1.237 ± 0.014 | - | 1.348 ± 0.209 | 2.415 ± 1.013 | 0.930 ± 0.438 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x128x128` | 0.319 ± 0.034 | - | 0.323 ± 0.015 | 0.687 ± 0.313 | 0.417 ± 0.028 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x2x2` | 0.002 ± 0.001 | - | 0.004 ± 0.002 | 0.042 ± 0.024 | 0.011 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x32x32` | 0.014 ± 0.002 | - | 0.016 ± 0.003 | 0.052 ± 0.010 | 0.086 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.050 ± 0.026 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x64x24` | 0.015 ± 0.000 | - | 0.016 ± 0.000 | 0.116 ± 0.075 | 0.085 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x8x8` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.036 ± 0.036 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.001 | 0.064 ± 0.049 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x4x4` | 0.003 ± 0.001 | - | 0.006 ± 0.000 | 0.066 ± 0.013 | 0.039 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x8x8` | 0.006 ± 0.000 | - | 0.009 ± 0.001 | 0.033 ± 0.030 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.001 | 0.063 ± 0.008 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x4x4` | 0.004 ± 0.001 | - | 0.007 ± 0.000 | 0.076 ± 0.028 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x8x8` | 0.007 ± 0.002 | - | 0.009 ± 0.002 | 0.069 ± 0.012 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x128x128` | 2.673 ± 0.161 | - | 2.581 ± 0.099 | 3.824 ± 1.599 | 1.631 ± 0.097 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x2x2` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.074 ± 0.045 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x32x32` | 0.099 ± 0.013 | - | 0.089 ± 0.009 | 0.321 ± 0.063 | 0.169 ± 0.057 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x4x4` | 0.007 ± 0.000 | - | 0.009 ± 0.001 | 0.064 ± 0.050 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x8x8` | 0.012 ± 0.003 | - | 0.013 ± 0.003 | 0.096 ± 0.012 | 0.021 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.142 ± 0.008 | - | 0.123 ± 0.008 | 0.118 ± 0.016 | 0.175 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.198 ± 0.002 | - | 0.180 ± 0.007 | 0.181 ± 0.019 | 0.255 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.359 ± 0.010 | - | 0.360 ± 0.026 | 0.384 ± 0.042 | 0.484 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x128x128,rhs=1` | 0.048 ± 0.002 | - | 0.061 ± 0.003 | 0.066 ± 0.003 | 0.105 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x2x2,rhs=1` | 0.001 ± 0.000 | - | 0.006 ± 0.011 | 0.012 ± 0.002 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x32x32,rhs=1` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.014 ± 0.003 | 0.045 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x4x4,rhs=1` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.009 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x8x8,rhs=1` | 0.002 ± 0.000 | - | 0.007 ± 0.010 | 0.010 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.003 | 0.011 ± 0.003 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.004 | 0.010 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x8x8,rhs=1` | 0.007 ± 0.003 | - | 0.005 ± 0.000 | 0.012 ± 0.002 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.012 ± 0.002 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.003 | 0.011 ± 0.002 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x8x8,rhs=1` | 0.003 ± 0.000 | - | 0.007 ± 0.002 | 0.010 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x128x128,rhs=1` | 0.386 ± 0.060 | - | 0.400 ± 0.032 | 0.435 ± 0.019 | 0.585 ± 0.047 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.007 ± 0.003 | 0.010 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x32x32,rhs=1` | 0.026 ± 0.001 | - | 0.031 ± 0.000 | 0.037 ± 0.001 | 0.082 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.006 ± 0.005 | 0.010 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x8x8,rhs=1` | 0.005 ± 0.000 | - | 0.007 ± 0.001 | 0.012 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.138 ± 0.010 | - | 0.126 ± 0.006 | 0.212 ± 0.067 | 0.196 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.203 ± 0.009 | - | 0.148 ± 0.019 | 0.244 ± 0.054 | 0.282 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.331 ± 0.005 | - | 0.544 ± 0.159 | 0.475 ± 0.088 | 0.606 ± 0.264 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x128x128,rhs=1` | 0.052 ± 0.001 | - | 0.061 ± 0.004 | 0.065 ± 0.001 | 0.145 ± 0.052 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.008 ± 0.007 | 0.010 ± 0.000 | 0.030 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x32x32,rhs=1` | 0.006 ± 0.002 | - | 0.010 ± 0.002 | 0.013 ± 0.000 | 0.178 ± 0.116 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x4x4,rhs=1` | 0.002 ± 0.001 | - | 0.006 ± 0.000 | 0.012 ± 0.001 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x8x8,rhs=1` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.010 ± 0.001 | 0.010 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.010 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x4x4,rhs=1` | 0.003 ± 0.001 | - | 0.007 ± 0.001 | 0.012 ± 0.002 | 0.034 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x8x8,rhs=1` | 0.004 ± 0.000 | - | 0.009 ± 0.002 | 0.012 ± 0.003 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.010 ± 0.002 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x4x4,rhs=1` | 0.003 ± 0.001 | - | 0.007 ± 0.000 | 0.012 ± 0.003 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x8x8,rhs=1` | 0.003 ± 0.000 | - | 0.008 ± 0.002 | 0.011 ± 0.003 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x128x128,rhs=1` | 0.408 ± 0.022 | - | 0.505 ± 0.009 | 0.564 ± 0.093 | 1.327 ± 0.464 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.008 ± 0.002 | 0.065 ± 0.034 | 0.011 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x32x32,rhs=1` | 0.033 ± 0.003 | - | 0.034 ± 0.005 | 0.177 ± 0.032 | 0.245 ± 0.094 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x4x4,rhs=1` | 0.003 ± 0.001 | - | 0.008 ± 0.002 | 0.064 ± 0.077 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x8x8,rhs=1` | 0.005 ± 0.000 | - | 0.009 ± 0.002 | 0.043 ± 0.014 | 0.017 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x2x2` | 1.137 ± 0.017 | - | 1.192 ± 0.049 | 1.159 ± 0.036 | 2.881 ± 0.043 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x4x4` | 4.212 ± 0.016 | - | 4.169 ± 0.146 | 4.215 ± 0.079 | 6.076 ± 0.461 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x8x8` | 14.292 ± 0.268 | - | 15.554 ± 1.850 | 14.557 ± 0.132 | 16.663 ± 0.071 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x128x128` | 5.798 ± 0.108 | - | 5.852 ± 0.178 | 4.309 ± 0.050 | 5.001 ± 0.090 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.018 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x32x32` | 0.181 ± 0.005 | - | 0.188 ± 0.017 | 0.179 ± 0.005 | 0.236 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x4x4` | 0.006 ± 0.001 | - | 0.008 ± 0.000 | 0.011 ± 0.000 | 0.019 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x64x24` | 0.175 ± 0.022 | - | 0.169 ± 0.020 | 0.170 ± 0.008 | 0.220 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x8x8` | 0.017 ± 0.001 | - | 0.018 ± 0.001 | 0.021 ± 0.000 | 0.030 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x2x2` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.009 ± 0.000 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x4x4` | 0.014 ± 0.000 | - | 0.015 ± 0.001 | 0.018 ± 0.000 | 0.031 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x8x8` | 0.044 ± 0.001 | - | 0.046 ± 0.000 | 0.049 ± 0.001 | 0.063 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x2x2` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.010 ± 0.001 | 0.024 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x4x4` | 0.019 ± 0.002 | - | 0.019 ± 0.000 | 0.022 ± 0.000 | 0.036 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x8x8` | 0.060 ± 0.001 | - | 0.061 ± 0.001 | 0.065 ± 0.001 | 0.080 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x128x128` | 48.693 ± 1.157 | - | 48.998 ± 0.542 | 34.510 ± 0.687 | 38.392 ± 0.566 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x2x2` | 0.011 ± 0.000 | - | 0.012 ± 0.000 | 0.016 ± 0.001 | 0.035 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x32x32` | 1.436 ± 0.065 | - | 1.427 ± 0.033 | 1.394 ± 0.028 | 1.570 ± 0.086 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x4x4` | 0.034 ± 0.000 | - | 0.037 ± 0.009 | 0.039 ± 0.008 | 0.060 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x8x8` | 0.115 ± 0.001 | - | 0.116 ± 0.002 | 0.121 ± 0.001 | 0.231 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x2x2` | 2.109 ± 1.960 | - | 1.523 ± 2.071 | 1.711 ± 0.096 | 2.995 ± 0.048 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x4x4` | 4.903 ± 1.394 | - | 5.328 ± 0.966 | 5.466 ± 0.716 | 4.457 ± 1.500 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x8x8` | 16.051 ± 2.172 | - | 16.458 ± 2.095 | 18.530 ± 1.328 | 8.993 ± 1.702 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x128x128` | 6.949 ± 0.481 | - | 6.514 ± 0.543 | 5.382 ± 0.284 | 5.822 ± 0.789 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x2x2` | 0.012 ± 0.001 | - | 0.019 ± 0.000 | 0.008 ± 0.001 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x32x32` | 0.181 ± 0.005 | - | 0.187 ± 0.012 | 0.201 ± 0.034 | 0.262 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x4x4` | 0.007 ± 0.001 | - | 0.029 ± 0.001 | 0.011 ± 0.000 | 0.019 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x64x24` | 0.237 ± 0.087 | - | 0.175 ± 0.011 | 0.193 ± 0.072 | 0.248 ± 0.039 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x8x8` | 0.022 ± 0.000 | - | 0.024 ± 0.000 | 0.022 ± 0.003 | 0.040 ± 0.069 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x2x2` | 0.018 ± 0.001 | - | 0.026 ± 0.001 | 0.013 ± 0.002 | 0.022 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x4x4` | 0.014 ± 0.000 | - | 0.018 ± 0.004 | 0.022 ± 0.002 | 0.031 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x8x8` | 0.061 ± 0.008 | - | 0.057 ± 0.002 | 0.058 ± 0.010 | 0.067 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x2x2` | 0.020 ± 0.000 | - | 0.027 ± 0.000 | 0.015 ± 0.001 | 0.026 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x4x4` | 0.023 ± 0.002 | - | 0.026 ± 0.001 | 0.026 ± 0.004 | 0.037 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x8x8` | 0.077 ± 0.011 | - | 0.076 ± 0.000 | 0.073 ± 0.006 | 0.080 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x128x128` | 51.463 ± 1.464 | - | 51.721 ± 2.763 | 43.294 ± 3.395 | 24.079 ± 2.749 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x2x2` | 0.032 ± 0.001 | - | 0.041 ± 0.002 | 0.022 ± 0.001 | 0.037 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x32x32` | 1.513 ± 0.079 | - | 1.594 ± 0.124 | 1.879 ± 0.538 | 1.196 ± 0.152 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x4x4` | 0.040 ± 0.006 | - | 0.115 ± 0.009 | 0.052 ± 0.009 | 0.103 ± 0.043 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x8x8` | 0.153 ± 0.008 | - | 0.158 ± 0.015 | 0.137 ± 0.007 | 0.203 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x2x2` | 0.559 ± 0.002 | - | 0.561 ± 0.002 | 0.567 ± 0.005 | 1.963 ± 0.051 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x4x4` | 2.670 ± 0.026 | - | 2.688 ± 0.034 | 2.683 ± 0.025 | 4.100 ± 0.104 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x8x8` | 10.413 ± 0.054 | - | 10.410 ± 0.044 | 10.460 ± 0.119 | 12.073 ± 0.394 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x128x128` | 1.440 ± 0.026 | - | 1.430 ± 0.049 | 1.444 ± 0.019 | 1.919 ± 0.333 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x2x2` | 0.003 ± 0.001 | - | 0.004 ± 0.000 | 0.006 ± 0.001 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x32x32` | 0.155 ± 0.013 | - | 0.122 ± 0.043 | 0.120 ± 0.009 | 0.194 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x4x4` | 0.005 ± 0.000 | - | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x64x24` | 0.090 ± 0.002 | - | 0.088 ± 0.008 | 0.095 ± 0.007 | 0.193 ± 0.048 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x8x8` | 0.012 ± 0.000 | - | 0.013 ± 0.000 | 0.016 ± 0.001 | 0.024 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x2x2` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.017 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x4x4` | 0.010 ± 0.000 | - | 0.011 ± 0.000 | 0.013 ± 0.000 | 0.024 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x8x8` | 0.029 ± 0.001 | - | 0.030 ± 0.000 | 0.033 ± 0.001 | 0.045 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x2x2` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.019 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x4x4` | 0.013 ± 0.000 | - | 0.014 ± 0.000 | 0.016 ± 0.000 | 0.029 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x8x8` | 0.040 ± 0.000 | - | 0.041 ± 0.000 | 0.043 ± 0.001 | 0.057 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x128x128` | 11.445 ± 0.148 | - | 11.613 ± 0.340 | 11.542 ± 0.285 | 12.690 ± 1.622 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x2x2` | 0.006 ± 0.000 | - | 0.007 ± 0.000 | 0.010 ± 0.000 | 0.027 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x32x32` | 0.916 ± 0.021 | - | 0.899 ± 0.017 | 0.921 ± 0.010 | 1.083 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x4x4` | 0.023 ± 0.001 | - | 0.024 ± 0.000 | 0.027 ± 0.000 | 0.051 ± 0.032 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x8x8` | 0.079 ± 0.001 | - | 0.080 ± 0.000 | 0.083 ± 0.001 | 0.148 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x2x2` | 0.743 ± 0.016 | - | 0.789 ± 0.321 | 0.642 ± 0.086 | 1.962 ± 0.049 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x4x4` | 3.406 ± 0.708 | - | 3.276 ± 0.301 | 3.539 ± 0.430 | 1.815 ± 0.049 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x8x8` | 12.203 ± 0.846 | - | 12.718 ± 1.540 | 13.803 ± 1.845 | 6.432 ± 0.058 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x128x128` | 1.636 ± 0.176 | - | 1.637 ± 0.122 | 1.796 ± 0.360 | 1.688 ± 0.191 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x2x2` | 0.005 ± 0.001 | - | 0.006 ± 0.002 | 0.021 ± 0.001 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x32x32` | 0.120 ± 0.002 | - | 0.312 ± 0.071 | 0.190 ± 0.135 | 0.247 ± 0.341 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x4x4` | 0.005 ± 0.000 | - | 0.008 ± 0.000 | 0.010 ± 0.001 | 0.017 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x64x24` | 0.264 ± 0.009 | - | 0.272 ± 0.005 | 0.102 ± 0.005 | 0.172 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x8x8` | 0.014 ± 0.000 | - | 0.017 ± 0.000 | 0.020 ± 0.004 | 0.025 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x2x2` | 0.004 ± 0.000 | - | 0.007 ± 0.001 | 0.009 ± 0.005 | 0.019 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x4x4` | 0.012 ± 0.003 | - | 0.014 ± 0.000 | 0.016 ± 0.002 | 0.025 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x8x8` | 0.033 ± 0.005 | - | 0.036 ± 0.004 | 0.042 ± 0.008 | 0.046 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x2x2` | 0.005 ± 0.000 | - | 0.007 ± 0.002 | 0.026 ± 0.001 | 0.020 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x4x4` | 0.015 ± 0.001 | - | 0.045 ± 0.029 | 0.017 ± 0.001 | 0.028 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x8x8` | 0.044 ± 0.011 | - | 0.047 ± 0.000 | 0.045 ± 0.007 | 0.057 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x128x128` | 13.575 ± 1.307 | - | 15.224 ± 5.375 | 16.647 ± 4.715 | 7.201 ± 0.700 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x2x2` | 0.011 ± 0.002 | - | 0.013 ± 0.003 | 0.013 ± 0.003 | 0.027 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x32x32` | 0.975 ± 0.293 | - | 1.018 ± 0.184 | 1.197 ± 0.164 | 0.668 ± 0.044 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x4x4` | 0.029 ± 0.003 | - | 0.030 ± 0.002 | 0.032 ± 0.003 | 0.044 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x8x8` | 0.086 ± 0.006 | - | 0.089 ± 0.055 | 0.100 ± 0.011 | 0.163 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x2x2` | 0.233 ± 0.025 | - | unsupported | 0.219 ± 0.006 | 1.326 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x4x4` | 1.462 ± 0.035 | - | unsupported | 1.358 ± 0.021 | 2.530 ± 0.034 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x8x8` | 4.345 ± 0.874 | - | unsupported | 3.824 ± 0.014 | 5.099 ± 0.059 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x128x128` | 0.828 ± 0.027 | - | unsupported | 0.767 ± 0.012 | 0.773 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | unsupported | 0.005 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x32x32` | 0.037 ± 0.007 | - | unsupported | 0.041 ± 0.007 | 0.049 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x4x4` | 0.003 ± 0.001 | - | unsupported | 0.006 ± 0.001 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x64x24` | 0.030 ± 0.000 | - | unsupported | 0.034 ± 0.001 | 0.043 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x8x8` | 0.005 ± 0.000 | - | unsupported | 0.009 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | unsupported | 0.005 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x4x4` | 0.005 ± 0.000 | - | unsupported | 0.009 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x8x8` | 0.012 ± 0.000 | - | unsupported | 0.016 ± 0.000 | 0.024 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | unsupported | 0.006 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x4x4` | 0.006 ± 0.000 | - | unsupported | 0.010 ± 0.000 | 0.019 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x8x8` | 0.016 ± 0.001 | - | unsupported | 0.020 ± 0.000 | 0.029 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x128x128` | 6.105 ± 0.120 | - | unsupported | 6.092 ± 0.026 | 6.789 ± 0.408 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | unsupported | 0.006 ± 0.000 | 0.019 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x32x32` | 0.317 ± 0.034 | - | unsupported | 0.294 ± 0.003 | 0.317 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x4x4` | 0.012 ± 0.011 | - | unsupported | 0.015 ± 0.000 | 0.028 ± 0.000 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x8x8` | 0.030 ± 0.000 | - | unsupported | 0.034 ± 0.001 | 0.050 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x2x2` | 0.233 ± 0.024 | - | unsupported | 0.270 ± 0.140 | 1.438 ± 0.071 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x4x4` | 1.557 ± 0.165 | - | unsupported | 1.772 ± 0.068 | 1.947 ± 1.585 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x8x8` | 4.351 ± 0.210 | - | unsupported | 5.201 ± 1.942 | 3.705 ± 1.945 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x128x128` | 1.151 ± 0.202 | - | unsupported | 2.139 ± 1.341 | 0.866 ± 0.087 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | unsupported | 0.006 ± 0.000 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x32x32` | 0.053 ± 0.012 | - | unsupported | 0.043 ± 0.004 | 0.052 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x4x4` | 0.003 ± 0.000 | - | unsupported | 0.006 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x64x24` | 0.031 ± 0.006 | - | unsupported | 0.034 ± 0.001 | 0.103 ± 0.160 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x8x8` | 0.006 ± 0.000 | - | unsupported | 0.011 ± 0.001 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | unsupported | 0.005 ± 0.001 | 0.014 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x4x4` | 0.007 ± 0.002 | - | unsupported | 0.010 ± 0.002 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x8x8` | 0.015 ± 0.003 | - | unsupported | 0.016 ± 0.000 | 0.025 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x2x2` | 0.003 ± 0.000 | - | unsupported | 0.007 ± 0.001 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x4x4` | 0.007 ± 0.000 | - | unsupported | 0.010 ± 0.003 | 0.019 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x8x8` | 0.018 ± 0.000 | - | unsupported | 0.020 ± 0.001 | 0.029 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x128x128` | 6.745 ± 1.458 | - | unsupported | 9.383 ± 2.345 | 4.528 ± 0.590 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x2x2` | 0.003 ± 0.000 | - | unsupported | 0.008 ± 0.001 | 0.019 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x32x32` | 0.411 ± 0.124 | - | unsupported | 0.326 ± 0.169 | 0.210 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x4x4` | 0.013 ± 0.003 | - | unsupported | 0.019 ± 0.004 | 0.029 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x8x8` | 0.035 ± 0.007 | - | unsupported | 0.035 ± 0.004 | 0.050 ± 0.003 | - | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x16x16,rhs=1` | 0.804 ± 0.033 | - | unsupported | 0.917 ± 0.071 | 1.061 ± 0.019 | 3.477 ± 1.045 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x2x2,rhs=1` | 0.062 ± 0.003 | - | unsupported | 0.094 ± 0.014 | 0.130 ± 0.003 | 0.085 ± 0.012 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x4x4,rhs=1` | 0.122 ± 0.003 | - | unsupported | 0.138 ± 0.021 | 0.178 ± 0.003 | 0.157 ± 0.015 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x8x8,rhs=1` | 0.289 ± 0.023 | - | unsupported | 0.311 ± 0.021 | 0.369 ± 0.015 | 0.734 ± 0.054 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x16x16,rhs=1` | 0.824 ± 0.206 | - | unsupported | 1.125 ± 0.553 | 1.745 ± 0.249 | 3.333 ± 0.754 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x2x2,rhs=1` | 0.058 ± 0.000 | - | unsupported | 0.157 ± 0.047 | 0.382 ± 0.248 | 0.109 ± 0.007 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x4x4,rhs=1` | 0.113 ± 0.001 | - | unsupported | 0.388 ± 0.093 | 0.208 ± 0.043 | 0.228 ± 0.021 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x8x8,rhs=1` | 0.261 ± 0.005 | - | unsupported | 0.545 ± 0.180 | 0.448 ± 0.119 | 1.454 ± 1.167 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.244 ± 0.017 | - | unsupported | 0.212 ± 0.028 | 0.453 ± 0.028 | 0.515 ± 0.085 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.073 ± 0.005 | - | unsupported | 0.054 ± 0.007 | 0.104 ± 0.025 | 0.075 ± 0.012 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.099 ± 0.004 | - | unsupported | 0.057 ± 0.006 | 0.115 ± 0.018 | 0.094 ± 0.000 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.105 ± 0.004 | - | unsupported | 0.100 ± 0.019 | 0.181 ± 0.027 | 0.226 ± 0.011 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.220 ± 0.001 | - | unsupported | 0.280 ± 0.284 | 1.095 ± 0.276 | 0.494 ± 0.009 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.067 ± 0.001 | - | unsupported | 0.043 ± 0.001 | 0.100 ± 0.006 | 0.066 ± 0.001 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.098 ± 0.001 | - | unsupported | 0.050 ± 0.010 | 0.119 ± 0.021 | 0.079 ± 0.000 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.102 ± 0.006 | - | unsupported | 0.089 ± 0.011 | 0.216 ± 0.053 | 0.210 ± 0.001 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.108 ± 0.001 | - | 0.111 ± 0.001 | 0.102 ± 0.021 | 0.168 ± 0.001 | 0.250 ± 0.027 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.046 ± 0.001 | - | 0.049 ± 0.020 | 0.044 ± 0.010 | 0.025 ± 0.001 | 0.042 ± 0.008 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.044 ± 0.003 | - | 0.046 ± 0.001 | 0.041 ± 0.015 | 0.031 ± 0.002 | 0.049 ± 0.000 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.059 ± 0.002 | - | 0.068 ± 0.007 | 0.039 ± 0.005 | 0.056 ± 0.001 | 0.101 ± 0.001 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.111 ± 0.002 | - | 0.110 ± 0.001 | 0.095 ± 0.021 | 0.541 ± 0.594 | 0.251 ± 0.049 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.046 ± 0.003 | - | 0.050 ± 0.007 | 0.034 ± 0.020 | 0.053 ± 0.010 | 0.042 ± 0.002 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.044 ± 0.002 | - | 0.047 ± 0.000 | 0.099 ± 0.023 | 0.038 ± 0.012 | 0.052 ± 0.000 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.057 ± 0.000 | - | 0.059 ± 0.000 | 0.049 ± 0.024 | 0.086 ± 0.092 | 0.105 ± 0.000 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 1 | `1536x1536` | 11.731 ± 2.035 | - | 11.241 ± 0.710 | 22.594 ± 0.615 | 12.691 ± 1.182 | 66.229 ± 9.644 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 4 | `1536x1536` | 7.838 ± 0.042 | - | 8.109 ± 0.442 | 24.705 ± 9.100 | 16.546 ± 2.389 | 386.082 ± 39.165 | - |
| cpu/linalg_uncovered | `det` | f64 | 1 | `1024x1024` | 6.954 ± 0.485 | - | 6.661 ± 0.079 | 6.569 ± 0.403 | 9.470 ± 0.649 | 32.347 ± 9.713 | - |
| cpu/linalg_uncovered | `det` | f64 | 4 | `1024x1024` | 9.742 ± 0.301 | - | 12.707 ± 4.964 | 12.881 ± 5.316 | 10.483 ± 10.577 | 212.717 ± 16.331 | - |
| cpu/linalg_uncovered | `eig` | f64 | 1 | `160x160` | 8.130 ± 1.442 | - | 6.635 ± 0.142 | 6.686 ± 0.183 | 7.549 ± 0.119 | 9.344 ± 2.607 | - |
| cpu/linalg_uncovered | `eig` | f64 | 4 | `160x160` | 7.069 ± 0.616 | - | 7.458 ± 0.632 | 8.263 ± 2.058 | 8.642 ± 1.454 | 331.447 ± 27.083 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 1 | `192x192` | 7.415 ± 0.392 | - | 7.449 ± 0.328 | 7.221 ± 0.224 | 6.895 ± 0.209 | 9.468 ± 2.048 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 4 | `192x192` | 8.983 ± 0.725 | - | 9.152 ± 0.339 | 9.996 ± 1.711 | 8.642 ± 0.579 | 283.929 ± 275.212 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 1 | `512x512` | 19.646 ± 0.131 | - | 19.591 ± 0.209 | 19.163 ± 0.611 | 27.785 ± 2.759 | 17.117 ± 1.509 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 4 | `512x512` | 23.869 ± 0.749 | - | 24.397 ± 1.921 | 24.643 ± 0.803 | 43.098 ± 7.791 | 649.186 ± 39.260 | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `256x128` | 0.281 ± 0.006 | 0.316 ± 0.041 | unsupported | 0.318 ± 0.009 | 0.341 ± 0.013 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `768x384` | 4.611 ± 2.532 | 3.526 ± 0.393 | unsupported | 3.814 ± 0.039 | 4.217 ± 0.245 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `256x128` | 0.270 ± 0.014 | 0.284 ± 0.009 | unsupported | 0.373 ± 0.022 | 0.380 ± 0.017 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `768x384` | 3.659 ± 0.129 | 4.046 ± 0.105 | unsupported | 4.514 ± 0.675 | 4.500 ± 0.569 | - | - |
| cpu/linalg_uncovered | `inv` | f64 | 1 | `768x768` | 7.525 ± 0.032 | - | 8.546 ± 0.340 | 8.261 ± 0.375 | 8.847 ± 1.029 | 29.406 ± 2.134 | - |
| cpu/linalg_uncovered | `inv` | f64 | 4 | `768x768` | 8.114 ± 0.100 | - | 8.942 ± 0.543 | 11.462 ± 2.563 | 14.914 ± 9.341 | 367.632 ± 93.596 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 1 | `768x384,rhs=16` | - | 8.573 ± 0.259 | 8.252 ± 0.045 | 4.938 ± 0.104 | 23.269 ± 0.366 | 8.346 ± 0.450 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 4 | `768x384,rhs=16` | - | 9.475 ± 0.182 | 9.403 ± 0.493 | 7.290 ± 3.436 | 32.789 ± 7.948 | 149.410 ± 29.899 | - |
| cpu/linalg_uncovered | `lu` | f64 | 1 | `1024x1024` | 7.740 ± 0.081 | - | 7.856 ± 0.200 | 10.383 ± 1.264 | 10.335 ± 0.794 | 35.217 ± 16.706 | - |
| cpu/linalg_uncovered | `lu` | f64 | 4 | `1024x1024` | 8.098 ± 0.210 | - | 8.744 ± 0.631 | 16.026 ± 4.384 | 15.744 ± 2.453 | 251.868 ± 118.917 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 1 | `2048x2048` | 0.640 ± 0.005 | - | 0.693 ± 0.059 | 2.197 ± 0.109 | 5.502 ± 0.281 | 6.129 ± 1.446 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 4 | `2048x2048` | 0.683 ± 0.029 | - | 0.770 ± 0.040 | 2.360 ± 0.139 | 8.892 ± 3.304 | 7.014 ± 4.610 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 1 | `512x256` | 11.011 ± 1.117 | - | 10.000 ± 0.334 | 10.067 ± 0.183 | 12.027 ± 0.700 | 24.323 ± 2.772 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 4 | `512x256` | 11.557 ± 0.414 | - | 11.126 ± 0.842 | 13.602 ± 0.931 | 15.999 ± 4.888 | 987.584 ± 368.344 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 1 | `512x256` | 10.061 ± 0.409 | - | 10.010 ± 0.375 | 10.231 ± 0.487 | 12.387 ± 1.121 | 20.412 ± 1.824 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 4 | `512x256` | 11.023 ± 0.366 | - | 10.760 ± 0.314 | 10.724 ± 0.359 | 14.534 ± 1.906 | 1056.298 ± 215.492 | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `256x128` | 1.784 ± 0.133 | 2.081 ± 0.306 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `768x384` | 13.243 ± 1.018 | 12.441 ± 0.661 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `256x128` | 1.379 ± 0.023 | 1.464 ± 0.088 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `768x384` | 13.913 ± 0.786 | 15.548 ± 2.874 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `256x128` | 2.091 ± 0.457 | 2.432 ± 0.252 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `768x384` | 11.516 ± 0.444 | 11.561 ± 0.275 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `256x128` | 1.473 ± 0.007 | 1.481 ± 0.011 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `768x384` | 12.617 ± 0.116 | 12.477 ± 0.501 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 1 | `1024x1024` | 7.457 ± 0.631 | - | 7.016 ± 1.016 | 6.298 ± 0.585 | 9.070 ± 0.322 | 30.898 ± 11.204 | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 4 | `1024x1024` | 9.137 ± 2.542 | - | 8.245 ± 1.083 | 9.659 ± 2.167 | 10.228 ± 1.855 | 366.934 ± 252.704 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 1 | `768x384` | - | 32.940 ± 1.192 | 34.571 ± 6.234 | 33.091 ± 0.528 | 33.591 ± 0.960 | 71.225 ± 19.271 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 4 | `768x384` | - | 33.585 ± 0.908 | 32.502 ± 0.316 | 39.609 ± 8.122 | 51.524 ± 14.076 | 1904.572 ± 208.499 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 1 | `4096x4096,rhs=64` | 14.935 ± 1.246 | - | 14.521 ± 1.462 | 11.828 ± 0.140 | 62.835 ± 9.043 | 53.210 ± 11.083 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 4 | `4096x4096,rhs=64` | 17.160 ± 2.040 | - | 18.197 ± 1.405 | 18.337 ± 8.488 | 50.566 ± 33.983 | 17.704 ± 1.971 | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `1024x64` | 0.187 ± 0.047 | 0.175 ± 0.003 | unsupported | 0.054 ± 0.002 | 0.033 ± 0.006 | - | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `4096x256` | 2.790 ± 0.067 | 2.828 ± 0.098 | unsupported | 0.840 ± 0.015 | 0.524 ± 0.134 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `1024x64` | 0.189 ± 0.020 | 0.201 ± 0.028 | unsupported | 0.127 ± 0.016 | 0.078 ± 0.069 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `4096x256` | 2.543 ± 0.988 | 1.947 ± 0.688 | unsupported | 0.858 ± 0.358 | 0.670 ± 0.136 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `1024x64` | 0.079 ± 0.001 | 0.079 ± 0.001 | unsupported | 0.091 ± 0.003 | 0.181 ± 0.002 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `4096x256` | 1.512 ± 0.288 | 1.284 ± 0.057 | unsupported | 1.361 ± 0.005 | 2.608 ± 0.029 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `1024x64` | 0.068 ± 0.009 | 0.102 ± 0.027 | unsupported | 0.161 ± 0.004 | 0.243 ± 0.069 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `4096x256` | 1.100 ± 0.255 | 0.770 ± 0.245 | unsupported | 0.801 ± 0.060 | 1.780 ± 0.886 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `1024x64` | 0.262 ± 0.002 | 0.263 ± 0.002 | unsupported | 0.303 ± 0.012 | 0.084 ± 0.002 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `4096x256` | 4.256 ± 0.162 | 4.240 ± 0.288 | unsupported | 4.626 ± 0.111 | 1.002 ± 0.060 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `1024x64` | 0.299 ± 0.082 | 0.279 ± 0.065 | unsupported | 0.247 ± 0.061 | 0.142 ± 0.036 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `4096x256` | 2.855 ± 0.355 | 3.279 ± 0.457 | unsupported | 2.701 ± 0.535 | 0.507 ± 0.042 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `1024x64` | 0.044 ± 0.002 | 0.062 ± 0.003 | unsupported | 0.046 ± 0.003 | 0.073 ± 0.009 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `4096x256` | 0.564 ± 0.016 | 0.579 ± 0.008 | unsupported | 0.533 ± 0.009 | 0.676 ± 0.006 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `1024x64` | 0.073 ± 0.024 | 0.115 ± 0.054 | unsupported | 0.113 ± 0.009 | 0.193 ± 0.026 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `4096x256` | 0.924 ± 0.336 | 0.900 ± 0.070 | unsupported | 0.495 ± 0.126 | 0.527 ± 0.210 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `1024x64` | 0.088 ± 0.002 | 0.103 ± 0.002 | unsupported | 0.094 ± 0.006 | 0.100 ± 0.004 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `4096x256` | 1.124 ± 0.014 | 1.133 ± 0.006 | unsupported | 1.336 ± 0.002 | 1.214 ± 0.011 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `1024x64` | 0.184 ± 0.068 | 0.193 ± 0.104 | unsupported | 0.147 ± 0.032 | 0.147 ± 0.016 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `4096x256` | 1.233 ± 0.324 | 1.199 ± 0.041 | unsupported | 1.624 ± 0.543 | 0.802 ± 0.129 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `1024x64` | 0.119 ± 0.001 | 0.136 ± 0.009 | unsupported | unsupported | 0.118 ± 0.008 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `4096x256` | 1.693 ± 0.015 | 1.802 ± 0.058 | unsupported | unsupported | 1.586 ± 0.030 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `1024x64` | 0.166 ± 0.029 | 0.197 ± 0.036 | unsupported | unsupported | 0.234 ± 0.100 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `4096x256` | 1.515 ± 0.060 | 1.608 ± 0.682 | unsupported | unsupported | 1.088 ± 0.510 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `1024x64` | 0.011 ± 0.000 | 0.016 ± 0.001 | unsupported | 0.007 ± 0.000 | 0.043 ± 0.002 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `4096x256` | 0.100 ± 0.003 | 0.106 ± 0.006 | unsupported | 0.058 ± 0.006 | 0.271 ± 0.031 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `1024x64` | 0.016 ± 0.001 | 0.021 ± 0.002 | unsupported | 0.039 ± 0.009 | 0.073 ± 0.046 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `4096x256` | 0.097 ± 0.014 | 0.132 ± 0.011 | unsupported | 0.109 ± 0.045 | 0.267 ± 0.145 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `1024x64` | 0.023 ± 0.000 | 0.033 ± 0.002 | unsupported | 0.028 ± 0.003 | 0.065 ± 0.001 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `4096x256` | 0.308 ± 0.001 | 0.327 ± 0.010 | unsupported | 0.307 ± 0.007 | 0.593 ± 0.013 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `1024x64` | 0.037 ± 0.008 | 0.056 ± 0.028 | unsupported | 0.142 ± 0.010 | 0.093 ± 0.013 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `4096x256` | 0.401 ± 0.026 | 0.449 ± 0.046 | unsupported | 0.741 ± 0.109 | 0.456 ± 0.014 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `1024x64` | 0.049 ± 0.001 | 0.049 ± 0.001 | unsupported | 0.080 ± 0.002 | 0.087 ± 0.002 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `4096x256` | 0.764 ± 0.004 | 0.818 ± 0.244 | unsupported | 1.256 ± 0.020 | 0.917 ± 0.121 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `1024x64` | 0.044 ± 0.020 | 0.046 ± 0.023 | unsupported | 0.140 ± 0.028 | 0.165 ± 0.028 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `4096x256` | 0.438 ± 0.009 | 0.543 ± 0.445 | unsupported | 0.815 ± 0.184 | 0.483 ± 0.193 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `1024x64` | 0.051 ± 0.001 | 0.050 ± 0.001 | unsupported | 0.081 ± 0.003 | 0.092 ± 0.002 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `4096x256` | 0.844 ± 0.132 | 0.813 ± 0.035 | unsupported | 1.254 ± 0.010 | 1.057 ± 0.104 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `1024x64` | 0.044 ± 0.007 | 0.067 ± 0.018 | unsupported | 0.173 ± 0.078 | 0.139 ± 0.020 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `4096x256` | 0.516 ± 0.234 | 0.459 ± 0.004 | unsupported | 0.808 ± 0.127 | 0.625 ± 0.545 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `1024x64` | 0.087 ± 0.002 | 0.102 ± 0.001 | unsupported | 0.100 ± 0.002 | 0.101 ± 0.002 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `4096x256` | 1.121 ± 0.060 | 1.125 ± 0.007 | unsupported | 1.584 ± 0.023 | 1.203 ± 0.022 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `1024x64` | 0.156 ± 0.053 | 0.168 ± 0.038 | unsupported | 0.090 ± 0.037 | 0.212 ± 0.031 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `4096x256` | 1.316 ± 0.119 | 1.516 ± 0.592 | unsupported | 1.237 ± 0.195 | 0.769 ± 0.050 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `1024x64` | 0.220 ± 0.001 | 0.219 ± 0.001 | unsupported | 0.284 ± 0.013 | 0.301 ± 0.001 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `4096x256` | 3.915 ± 0.400 | 4.016 ± 0.324 | unsupported | 4.447 ± 0.021 | 4.568 ± 0.062 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `1024x64` | 0.188 ± 0.026 | 0.168 ± 0.017 | unsupported | 0.319 ± 0.121 | 0.328 ± 0.190 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `4096x256` | 2.277 ± 1.157 | 3.417 ± 1.531 | unsupported | 2.642 ± 0.316 | 3.553 ± 1.228 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `1024x64` | 0.508 ± 0.009 | 0.608 ± 0.027 | unsupported | 0.118 ± 0.006 | 0.094 ± 0.010 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `4096x256` | 9.367 ± 0.426 | 10.647 ± 0.119 | unsupported | 2.050 ± 0.043 | 1.000 ± 0.136 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `1024x64` | 1.178 ± 0.275 | 1.696 ± 0.634 | unsupported | 0.171 ± 0.025 | 0.114 ± 0.041 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `4096x256` | 7.602 ± 1.484 | 9.521 ± 2.691 | unsupported | 1.841 ± 0.485 | 0.654 ± 0.063 | - | - |
| cpu/output_reuse | `add_into` | f64 | 1 | `33554432` | 17.659 ± 1.345 | - | unsupported | 23.393 ± 0.896 | - | 17.807 ± 1.294 | 25.406 ± 3.329 |
| cpu/output_reuse | `add_into` | f64 | 4 | `33554432` | 26.285 ± 4.330 | - | unsupported | 22.125 ± 2.023 | - | 21.824 ± 1.865 | 25.220 ± 8.368 |
| cpu/output_reuse | `conj_into` | c64 | 1 | `16777216` | 9.156 ± 0.230 | - | unsupported | 11.666 ± 0.211 | - | 10.091 ± 0.581 | 10.811 ± 0.919 |
| cpu/output_reuse | `conj_into` | c64 | 4 | `16777216` | 14.266 ± 1.474 | - | unsupported | 14.228 ± 1.667 | - | 12.777 ± 2.226 | 13.180 ± 0.536 |
| cpu/output_reuse | `copy_read_into` | f64 | 1 | `33554432` | 9.683 ± 0.978 | - | unsupported | 11.513 ± 0.258 | - | 11.520 ± 1.680 | - |
| cpu/output_reuse | `copy_read_into` | f64 | 4 | `33554432` | 11.994 ± 2.343 | - | unsupported | 14.610 ± 2.138 | - | 11.882 ± 2.261 | - |
| cpu/output_reuse | `div_into` | f64 | 1 | `33554432` | 18.909 ± 5.143 | - | unsupported | 17.523 ± 2.417 | - | 18.396 ± 1.821 | 20.148 ± 9.330 |
| cpu/output_reuse | `div_into` | f64 | 4 | `33554432` | 19.677 ± 1.047 | - | unsupported | 22.891 ± 2.481 | - | 21.492 ± 3.262 | 21.610 ± 6.993 |
| cpu/output_reuse | `dot_general_read_into` | f64 | 1 | `1024x1024` | 10.421 ± 0.403 | - | unsupported | 10.139 ± 1.157 | - | 47.859 ± 1.659 | - |
| cpu/output_reuse | `dot_general_read_into` | f64 | 4 | `1024x1024` | 11.380 ± 0.670 | - | unsupported | 12.376 ± 2.286 | - | 31.779 ± 9.793 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 1 | `1024x1024` | 11.598 ± 1.754 | - | unsupported | 10.845 ± 0.236 | - | 47.979 ± 0.407 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 4 | `1024x1024` | 12.633 ± 2.888 | - | unsupported | 16.230 ± 2.644 | - | 24.075 ± 3.405 | - |
| cpu/output_reuse | `mul_into` | f64 | 1 | `33554432` | 20.101 ± 1.613 | - | unsupported | 18.606 ± 0.682 | - | 25.165 ± 10.124 | 30.179 ± 11.167 |
| cpu/output_reuse | `mul_into` | f64 | 4 | `33554432` | 18.252 ± 0.822 | - | unsupported | 23.755 ± 4.769 | - | 21.999 ± 8.257 | 23.817 ± 4.223 |
| cpu/output_reuse | `neg_into` | f64 | 1 | `33554432` | 10.567 ± 0.780 | - | unsupported | 12.592 ± 0.319 | - | 12.296 ± 2.121 | 13.896 ± 2.389 |
| cpu/output_reuse | `neg_into` | f64 | 4 | `33554432` | 11.951 ± 0.716 | - | unsupported | 13.514 ± 4.073 | - | 13.443 ± 3.909 | 13.732 ± 1.888 |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `256x256,rhs=16` | 0.206 ± 0.011 | unsupported | unsupported | 0.259 ± 0.013 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `768x768,rhs=16` | 2.814 ± 0.290 | unsupported | unsupported | 2.948 ± 0.147 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `256x256,rhs=16` | 0.223 ± 0.025 | unsupported | unsupported | 0.391 ± 0.044 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `768x768,rhs=16` | 2.914 ± 0.524 | unsupported | unsupported | 4.056 ± 0.613 | unsupported | - | - |
| cpu/output_reuse | `sub_into` | f64 | 1 | `33554432` | 18.843 ± 1.480 | - | unsupported | 18.921 ± 0.927 | - | 24.678 ± 1.161 | 24.822 ± 10.795 |
| cpu/output_reuse | `sub_into` | f64 | 4 | `33554432` | 18.638 ± 0.666 | - | unsupported | 24.489 ± 3.097 | - | 27.756 ± 37.252 | 22.084 ± 2.927 |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 1 | `8192x1 -> 8192x4096` | 7.732 ± 0.173 | - | 0.002 ± 0.004 | 4.530 ± 0.847 | 5.103 ± 0.852 | 15.520 ± 5.674 | - |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 4 | `8192x1 -> 8192x4096` | 9.896 ± 1.028 | - | 0.002 ± 0.001 | 7.026 ± 3.514 | 6.282 ± 1.646 | 19.585 ± 2.052 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 1 | `33554432` | 9.609 ± 0.387 | - | 20.919 ± 0.524 | 7.253 ± 0.301 | 9.521 ± 0.935 | 10.893 ± 1.894 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 4 | `33554432` | 10.404 ± 0.989 | - | 24.583 ± 0.748 | 10.882 ± 1.986 | 14.262 ± 3.379 | 12.631 ± 2.320 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 1 | `8192 -> 8192x8192` | 11.260 ± 0.748 | - | 11.771 ± 0.365 | 12.632 ± 0.900 | 15.944 ± 0.402 | 13.378 ± 6.669 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 4 | `8192 -> 8192x8192` | 15.225 ± 5.453 | - | 13.103 ± 0.431 | 19.719 ± 10.000 | 12.435 ± 8.408 | 12.461 ± 4.170 | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 1 | `8388608x2x2 -> 8388608x2` | 5.439 ± 0.062 | - | 5.481 ± 0.143 | 11.389 ± 1.776 | 7.110 ± 0.076 | - | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 4 | `8388608x2x2 -> 8388608x2` | 6.008 ± 0.624 | - | 6.219 ± 0.489 | 13.715 ± 1.761 | 11.727 ± 1.695 | - | - |
| cpu/structural_shape | `reshape` | f64 | 1 | `33554432 -> 8192x4096` | 9.871 ± 0.710 | - | 0.001 ± 0.000 | 11.414 ± 1.361 | 10.391 ± 0.765 | 16.595 ± 10.981 | - |
| cpu/structural_shape | `reshape` | f64 | 4 | `33554432 -> 8192x4096` | 11.898 ± 2.033 | - | 0.001 ± 0.000 | 14.521 ± 1.634 | 11.502 ± 1.017 | 14.363 ± 3.848 | - |
| cpu/structural_shape | `transpose` | f64 | 1 | `4096x4096` | 39.563 ± 1.263 | - | 0.001 ± 0.002 | 47.535 ± 4.527 | 48.520 ± 0.868 | 239.476 ± 30.543 | 31.196 ± 6.249 |
| cpu/structural_shape | `transpose` | f64 | 4 | `4096x4096` | 24.776 ± 1.518 | - | 0.001 ± 0.000 | 50.796 ± 1.491 | 32.980 ± 6.687 | 244.646 ± 35.105 | 21.120 ± 2.493 |
| cpu/structural_shape | `tril` | f64 | 1 | `4096x4096` | 8.019 ± 0.429 | - | 8.187 ± 0.305 | 8.872 ± 0.631 | 5.276 ± 0.035 | 11.459 ± 1.681 | - |
| cpu/structural_shape | `tril` | f64 | 4 | `4096x4096` | 6.860 ± 0.873 | - | 6.437 ± 0.388 | 8.563 ± 1.144 | 6.454 ± 0.612 | 12.803 ± 1.799 | - |
| cpu/structural_shape | `triu` | f64 | 1 | `4096x4096` | 8.060 ± 0.203 | - | 8.310 ± 0.271 | 8.097 ± 0.082 | 5.604 ± 0.385 | 10.993 ± 2.879 | - |
| cpu/structural_shape | `triu` | f64 | 4 | `4096x4096` | 6.355 ± 0.853 | - | 6.473 ± 0.495 | 9.120 ± 1.819 | 5.194 ± 0.183 | 11.081 ± 5.308 | - |
| cpu/view_metadata | `broadcast_in_dim_view` | f64 | 1 | `8192x1 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | - | - |
| cpu/view_metadata | `broadcast_in_dim_view` | f64 | 4 | `8192x1 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | - | - |
| cpu/view_metadata | `reshape_view` | f64 | 1 | `33554432 -> 8192x4096` | 0.002 ± 0.001 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `reshape_view` | f64 | 4 | `33554432 -> 8192x4096` | 0.007 ± 0.005 | - | unsupported | 0.003 ± 0.002 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `slice_view` | f64 | 1 | `4194304 -> 2096128` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `slice_view` | f64 | 4 | `4194304 -> 2096128` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `transpose_view` | f64 | 1 | `4096x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `transpose_view` | f64 | 4 | `4096x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |

## Short-operation batches

Total batch duration and normalized ns/op are shown without rounding sub-microsecond calls to zero. Metadata views require no execution session.

| Operation | Dtype | Shape | Threads | Backend | Operations/batch | Median batch ms | Median ns/op |
|---|---|---|---:|---|---:|---:|---:|
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 16 | 4.447625 | 277976.56 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs direct API (ms) | 16 | 3.877750 | 242359.38 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 16 | 4.067000 | 254187.50 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 16 | 4.367000 | 272937.50 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs direct API (ms) | 16 | 5.298959 | 331184.94 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 16 | 6.718125 | 419882.81 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs direct API (ms) | 16 | 11.141625 | 696351.56 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs eager mode (ms) | 16 | 11.269459 | 704341.19 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs direct API (ms) | 16 | 14.684334 | 917770.88 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs eager mode (ms) | 16 | 14.346792 | 896674.50 |
| householder_qr | f64 | `256x128` | 1 | JAX Python (XLA CPU) (ms) | 16 | 5.463250 | 341453.12 |
| householder_qr | f64 | `256x128` | 1 | PyTorch Python (ms) | 16 | 5.092083 | 318255.19 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 4.503167 | 281447.94 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 5.050417 | 315651.06 |
| householder_qr | f64 | `768x384` | 1 | JAX Python (XLA CPU) (ms) | 16 | 67.469375 | 4216835.94 |
| householder_qr | f64 | `768x384` | 1 | PyTorch Python (ms) | 16 | 61.016750 | 3813546.88 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 73.771459 | 4610716.19 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 56.422625 | 3526414.06 |
| householder_qr | f64 | `256x128` | 4 | JAX Python (XLA CPU) (ms) | 16 | 6.074458 | 379653.62 |
| householder_qr | f64 | `256x128` | 4 | PyTorch Python (ms) | 16 | 5.966083 | 372880.19 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 4.324667 | 270291.69 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 4.548959 | 284309.94 |
| householder_qr | f64 | `768x384` | 4 | JAX Python (XLA CPU) (ms) | 16 | 71.996959 | 4499809.94 |
| householder_qr | f64 | `768x384` | 4 | PyTorch Python (ms) | 16 | 72.221917 | 4513869.81 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 58.543958 | 3658997.38 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 64.740417 | 4046276.06 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 28.537834 | 1783614.62 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 33.295500 | 2080968.75 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 211.891709 | 13243231.81 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 199.059334 | 12441208.38 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 22.068750 | 1379296.88 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 23.424792 | 1464049.50 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 222.600500 | 13912531.25 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 248.765584 | 15547849.00 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 33.460042 | 2091252.62 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 38.904917 | 2431557.31 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 184.260625 | 11516289.06 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 184.978459 | 11561153.69 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 23.565333 | 1472833.31 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 23.691416 | 1480713.50 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 201.867291 | 12616705.69 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 199.626958 | 12476684.88 |
| erf | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 2.122417 | 33162.77 |
| erf | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 3.438084 | 53720.06 |
| erf | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 11.968250 | 187003.91 |
| erf | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 11.183292 | 174738.94 |
| erf | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 4.194459 | 524307.38 |
| erf | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 6.717625 | 839703.12 |
| erf | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 22.320333 | 2790041.62 |
| erf | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 22.626500 | 2828312.50 |
| erf | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 5.014250 | 78347.66 |
| erf | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 8.145000 | 127265.62 |
| erf | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 12.109208 | 189206.38 |
| erf | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 12.894041 | 201469.39 |
| erf | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 5.359542 | 669942.75 |
| erf | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 6.862125 | 857765.62 |
| erf | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 20.344375 | 2543046.88 |
| erf | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 15.579583 | 1947447.88 |
| gelu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 11.577708 | 180901.69 |
| gelu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 5.803083 | 90673.17 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 5.046625 | 78853.52 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 5.051250 | 78925.78 |
| gelu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 20.862417 | 2607802.12 |
| gelu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 10.885375 | 1360671.88 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 12.093416 | 1511677.00 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 10.272917 | 1284114.62 |
| gelu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 15.541375 | 242833.98 |
| gelu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 10.313417 | 161147.14 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 4.375959 | 68374.36 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 6.541292 | 102207.69 |
| gelu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 14.236250 | 1779531.25 |
| gelu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 6.406000 | 800750.00 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 8.798250 | 1099781.25 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 6.161875 | 770234.38 |
| gelu_tanh | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 5.369791 | 83902.98 |
| gelu_tanh | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 19.379584 | 302806.00 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 16.780667 | 262197.92 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 16.857667 | 263401.05 |
| gelu_tanh | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 8.015334 | 1001916.75 |
| gelu_tanh | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 37.005958 | 4625744.75 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 34.049875 | 4256234.38 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 33.917042 | 4239630.25 |
| gelu_tanh | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 9.084917 | 141951.83 |
| gelu_tanh | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 15.791500 | 246742.19 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 19.138500 | 299039.06 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 17.867834 | 279184.91 |
| gelu_tanh | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 4.052125 | 506515.62 |
| gelu_tanh | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 21.605791 | 2700723.88 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 22.837125 | 2854640.62 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 26.232208 | 3279026.00 |
| layer_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 4.649958 | 72655.59 |
| layer_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 2.933208 | 45831.38 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 2.806667 | 43854.17 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 3.983417 | 62240.89 |
| layer_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 5.411625 | 676453.12 |
| layer_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 4.265959 | 533244.88 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 4.513000 | 564125.00 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 4.629208 | 578651.00 |
| layer_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 12.373500 | 193335.94 |
| layer_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 7.205750 | 112589.84 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 4.689041 | 73266.27 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 7.371666 | 115182.28 |
| layer_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 4.219459 | 527432.38 |
| layer_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 3.963583 | 495447.88 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 7.393292 | 924161.50 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 7.200167 | 900020.88 |
| log_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 6.404833 | 100075.52 |
| log_softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 5.991000 | 93609.38 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 5.635625 | 88056.64 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 6.580125 | 102814.45 |
| log_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 9.711500 | 1213937.50 |
| log_softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 10.689250 | 1336156.25 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 8.994500 | 1124312.50 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 9.067500 | 1133437.50 |
| log_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 9.382500 | 146601.56 |
| log_softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 9.376292 | 146504.56 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 11.758125 | 183720.70 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 12.360709 | 193136.08 |
| log_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 6.417292 | 802161.50 |
| log_softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 12.994375 | 1624296.88 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 9.865875 | 1233234.38 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 9.593000 | 1199125.00 |
| masked_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 7.533500 | 117710.94 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 7.637500 | 119335.94 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 8.695000 | 135859.38 |
| masked_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 12.691291 | 1586411.38 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 13.547459 | 1693432.38 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 14.419667 | 1802458.38 |
| masked_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 14.989083 | 234204.42 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 10.654750 | 166480.47 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 12.594083 | 196782.55 |
| masked_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 8.706667 | 1088333.38 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 12.123500 | 1515437.50 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 12.867333 | 1608416.62 |
| reduce_mean | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 2.757708 | 43089.19 |
| reduce_mean | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 0.478959 | 7483.73 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 0.723125 | 11298.83 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 1.010666 | 15791.66 |
| reduce_mean | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 2.167208 | 270901.00 |
| reduce_mean | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 0.466500 | 58312.50 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 0.803084 | 100385.50 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 0.844667 | 105583.38 |
| reduce_mean | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 4.642958 | 72546.22 |
| reduce_mean | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 2.501625 | 39087.89 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 1.017250 | 15894.53 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 1.361042 | 21266.28 |
| reduce_mean | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 2.138916 | 267364.50 |
| reduce_mean | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 0.870625 | 108828.12 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 0.778625 | 97328.12 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 1.052042 | 131505.25 |
| rms_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 4.185666 | 65401.03 |
| rms_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 1.806916 | 28233.06 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 1.454834 | 22731.78 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 2.137375 | 33396.48 |
| rms_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 4.742875 | 592859.38 |
| rms_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 2.456833 | 307104.12 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 2.465750 | 308218.75 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 2.615625 | 326953.12 |
| rms_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 5.949917 | 92967.45 |
| rms_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 9.101625 | 142212.89 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 2.356375 | 36818.36 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 3.600958 | 56264.97 |
| rms_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 3.645625 | 455703.12 |
| rms_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 5.926750 | 740843.75 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 3.208958 | 401119.75 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 3.592708 | 449088.50 |
| sigmoid | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 5.558334 | 86848.97 |
| sigmoid | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 5.104792 | 79762.38 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 3.128833 | 48888.02 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 3.160875 | 49388.67 |
| sigmoid | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 7.337125 | 917140.62 |
| sigmoid | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 10.050916 | 1256364.50 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 6.109875 | 763734.38 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 6.546750 | 818343.75 |
| sigmoid | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 10.543708 | 164745.44 |
| sigmoid | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 8.955750 | 139933.59 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 2.846125 | 44470.70 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 2.925833 | 45716.14 |
| sigmoid | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 3.864417 | 483052.12 |
| sigmoid | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 6.522500 | 815312.50 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 3.503167 | 437895.88 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 4.340167 | 542520.88 |
| silu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 5.870292 | 91723.31 |
| silu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 5.152000 | 80500.00 |
| silu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 3.260917 | 50951.83 |
| silu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 3.217166 | 50268.22 |
| silu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 8.453041 | 1056630.12 |
| silu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 10.033333 | 1254166.62 |
| silu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 6.751583 | 843947.88 |
| silu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 6.506958 | 813369.75 |
| silu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 8.887000 | 138859.38 |
| silu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 11.053375 | 172708.98 |
| silu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 2.826917 | 44170.58 |
| silu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 4.282125 | 66908.20 |
| silu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 4.996083 | 624510.38 |
| silu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 6.464708 | 808088.50 |
| silu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 4.131125 | 516390.62 |
| silu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 3.674875 | 459359.38 |
| softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 6.437167 | 100580.73 |
| softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 6.383875 | 99748.05 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 5.590541 | 87352.20 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 6.506750 | 101667.97 |
| softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 9.627666 | 1203458.25 |
| softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 12.672125 | 1584015.62 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 8.970708 | 1121338.50 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 8.999750 | 1124968.75 |
| softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 13.548042 | 211688.16 |
| softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 5.746083 | 89782.55 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 9.959000 | 155609.38 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 10.765459 | 168210.30 |
| softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 6.153000 | 769125.00 |
| softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 9.892500 | 1236562.50 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 10.527625 | 1315953.12 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 12.131208 | 1516401.00 |
| softplus | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 19.235750 | 300558.59 |
| softplus | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 18.206625 | 284478.52 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 14.053125 | 219580.08 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 14.007334 | 218864.59 |
| softplus | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 36.547583 | 4568447.88 |
| softplus | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 35.575625 | 4446953.12 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 31.317000 | 3914625.00 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 32.128125 | 4016015.62 |
| softplus | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 21.018667 | 328416.67 |
| softplus | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 20.420583 | 319071.61 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 12.003292 | 187551.44 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 10.728834 | 167638.03 |
| softplus | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 28.426375 | 3553296.88 |
| softplus | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 21.132916 | 2641614.50 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 18.214875 | 2276859.38 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 27.332583 | 3416572.88 |
| take_along_axis | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 6.047958 | 94499.34 |
| take_along_axis | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 7.563250 | 118175.78 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 32.485083 | 507579.42 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 38.933209 | 608331.39 |
| take_along_axis | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 8.000333 | 1000041.62 |
| take_along_axis | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 16.398833 | 2049854.12 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 74.933417 | 9366677.12 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 85.175875 | 10646984.38 |
| take_along_axis | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 7.298916 | 114045.56 |
| take_along_axis | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 10.919750 | 170621.09 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 75.392792 | 1178012.38 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 108.517958 | 1695593.09 |
| take_along_axis | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 5.235125 | 654390.62 |
| take_along_axis | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 14.730750 | 1841343.75 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 60.812792 | 7601599.00 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 76.170542 | 9521317.75 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | PyTorch Python (ms) | 16 | 4.142167 | 258885.44 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 3.295709 | 205981.81 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | PyTorch Python (ms) | 16 | 47.169084 | 2948067.75 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 45.031916 | 2814494.75 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | PyTorch Python (ms) | 16 | 6.260500 | 391281.25 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 3.566000 | 222875.00 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | PyTorch Python (ms) | 16 | 64.890792 | 4055674.50 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 46.626291 | 2914143.19 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.015500 | 968.75 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.001334 | 83.38 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.017167 | 1072.94 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.001541 | 96.31 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000208 | 13.00 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.012084 | 755.25 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.035250 | 2203.12 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000083 | 5.19 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.042958 | 2684.88 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.111500 | 6968.75 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000125 | 7.81 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | PyTorch Python (ms) | 16 | 0.012500 | 781.25 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | tenferro-rs direct API (ms) | 16 | 0.002959 | 184.94 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000291 | 18.19 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | PyTorch Python (ms) | 16 | 0.013917 | 869.81 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | tenferro-rs direct API (ms) | 16 | 0.003334 | 208.38 |
| transpose_view | f64 | `4096x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000083 | 5.19 |
| transpose_view | f64 | `4096x4096` | 1 | PyTorch Python (ms) | 16 | 0.011250 | 703.12 |
| transpose_view | f64 | `4096x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.002250 | 140.62 |
| transpose_view | f64 | `4096x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000084 | 5.25 |
| transpose_view | f64 | `4096x4096` | 4 | PyTorch Python (ms) | 16 | 0.016292 | 1018.25 |
| transpose_view | f64 | `4096x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.003208 | 200.50 |

## Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `cpu/complex/cholesky` (c64, threads=4, shape=`448x448`): `jax-cpu` is 104.2x faster than the slowest successful cell (`julia-base`, 174.110 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/cholesky` (c64, threads=4, shape=`448x448`): `pytorch-cpu` is 35.0x faster than the slowest successful cell (`julia-base`, 174.110 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/cholesky` (c64, threads=4, shape=`448x448`): `tenferro-direct` is 46.9x faster than the slowest successful cell (`julia-base`, 174.110 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/cholesky` (c64, threads=4, shape=`448x448`): `tenferro-trace` is 45.6x faster than the slowest successful cell (`julia-base`, 174.110 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/dot_general` (c64, threads=1, shape=`640x640`): `pytorch-cpu` is 10.3x faster than the slowest successful cell (`julia-base`, 97.507 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/eig` (c64, threads=4, shape=`112x112`): `jax-cpu` is 21.9x faster than the slowest successful cell (`julia-base`, 453.745 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/eig` (c64, threads=4, shape=`112x112`): `pytorch-cpu` is 31.7x faster than the slowest successful cell (`julia-base`, 453.745 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/eig` (c64, threads=4, shape=`112x112`): `tenferro-direct` is 38.2x faster than the slowest successful cell (`julia-base`, 453.745 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/eig` (c64, threads=4, shape=`112x112`): `tenferro-trace` is 41.5x faster than the slowest successful cell (`julia-base`, 453.745 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/norm_fro` (c64, threads=4, shape=`2048x1536`): `tenferro-direct` is 10.1x faster than the slowest successful cell (`julia-base`, 12.121 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `jax-cpu` is 21.2x faster than the slowest successful cell (`julia-base`, 149.153 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `pytorch-cpu` is 18.2x faster than the slowest successful cell (`julia-base`, 149.153 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `tenferro-direct` is 17.0x faster than the slowest successful cell (`julia-base`, 149.153 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `tenferro-trace` is 20.6x faster than the slowest successful cell (`julia-base`, 149.153 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `jax-cpu` is 44.8x faster than the slowest successful cell (`julia-base`, 123.103 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `pytorch-cpu` is 17.3x faster than the slowest successful cell (`julia-base`, 123.103 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `tenferro-direct` is 21.2x faster than the slowest successful cell (`julia-base`, 123.103 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `tenferro-trace` is 21.5x faster than the slowest successful cell (`julia-base`, 123.103 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/svd` (c64, threads=4, shape=`160x160`): `jax-cpu` is 100.3x faster than the slowest successful cell (`julia-base`, 640.036 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/svd` (c64, threads=4, shape=`160x160`): `pytorch-cpu` is 70.0x faster than the slowest successful cell (`julia-base`, 640.036 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/svd` (c64, threads=4, shape=`160x160`): `tenferro-direct` is 93.3x faster than the slowest successful cell (`julia-base`, 640.036 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/svd` (c64, threads=4, shape=`160x160`): `tenferro-trace` is 78.8x faster than the slowest successful cell (`julia-base`, 640.036 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 20.6x faster than the slowest successful cell (`pytorch-cpu`, 16.060 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 18.8x faster than the slowest successful cell (`pytorch-cpu`, 16.060 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 18.8x faster than the slowest successful cell (`pytorch-cpu`, 16.060 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 14.5x faster than the slowest successful cell (`pytorch-cpu`, 18.811 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 16.7x faster than the slowest successful cell (`pytorch-cpu`, 18.811 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-trace` is 17.8x faster than the slowest successful cell (`pytorch-cpu`, 18.811 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 21.2x faster than the slowest successful cell (`pytorch-cpu`, 15.862 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 18.5x faster than the slowest successful cell (`pytorch-cpu`, 15.862 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 18.2x faster than the slowest successful cell (`pytorch-cpu`, 15.862 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 12.6x faster than the slowest successful cell (`pytorch-cpu`, 12.803 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-trace` is 11.7x faster than the slowest successful cell (`pytorch-cpu`, 12.803 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/gather` (f64, threads=1, shape=`262144`): `pytorch-cpu` is 11.0x faster than the slowest successful cell (`julia-base`, 3.640 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/gather` (f64, threads=1, shape=`262144`): `tenferro-direct` is 11.1x faster than the slowest successful cell (`julia-base`, 3.640 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=1, shape=`4194304 -> 2096128`): `tenferro-trace` is 1083.3x faster than the slowest successful cell (`jax-cpu`, 1.490 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`): `tenferro-trace` is 378.5x faster than the slowest successful cell (`tenferro-direct`, 1.514 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x2x2`): `tenferro-direct` is 28.3x faster than the slowest successful cell (`jax-cpu`, 0.024 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x32x32`): `tenferro-direct` is 26.2x faster than the slowest successful cell (`jax-cpu`, 0.048 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x32x32`): `tenferro-trace` is 13.7x faster than the slowest successful cell (`jax-cpu`, 0.048 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x8x8`): `pytorch-cpu` is 11.3x faster than the slowest successful cell (`jax-cpu`, 0.051 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x8x8`): `tenferro-direct` is 31.5x faster than the slowest successful cell (`jax-cpu`, 0.051 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x8x8`): `tenferro-trace` is 15.0x faster than the slowest successful cell (`jax-cpu`, 0.051 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`8x8x8`): `tenferro-direct` is 21.4x faster than the slowest successful cell (`jax-cpu`, 0.050 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`8x8x8`): `tenferro-trace` is 11.9x faster than the slowest successful cell (`jax-cpu`, 0.050 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x2x2`): `jax-cpu` is 11.8x faster than the slowest successful cell (`pytorch-cpu`, 0.100 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x2x2`): `tenferro-direct` is 70.2x faster than the slowest successful cell (`pytorch-cpu`, 0.100 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x2x2`): `tenferro-trace` is 27.4x faster than the slowest successful cell (`pytorch-cpu`, 0.100 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x32x32`): `jax-cpu` is 12.7x faster than the slowest successful cell (`pytorch-cpu`, 0.110 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x32x32`): `tenferro-direct` is 40.6x faster than the slowest successful cell (`pytorch-cpu`, 0.110 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x32x32`): `tenferro-trace` is 24.2x faster than the slowest successful cell (`pytorch-cpu`, 0.110 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x4x4`): `jax-cpu` is 11.2x faster than the slowest successful cell (`pytorch-cpu`, 0.092 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x4x4`): `tenferro-direct` is 56.8x faster than the slowest successful cell (`pytorch-cpu`, 0.092 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x4x4`): `tenferro-trace` is 26.7x faster than the slowest successful cell (`pytorch-cpu`, 0.092 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x8x8`): `tenferro-direct` is 41.0x faster than the slowest successful cell (`pytorch-cpu`, 0.070 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x8x8`): `tenferro-trace` is 20.2x faster than the slowest successful cell (`pytorch-cpu`, 0.070 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x2x2`): `jax-cpu` is 11.3x faster than the slowest successful cell (`pytorch-cpu`, 0.097 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x2x2`): `tenferro-direct` is 64.4x faster than the slowest successful cell (`pytorch-cpu`, 0.097 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x2x2`): `tenferro-trace` is 26.7x faster than the slowest successful cell (`pytorch-cpu`, 0.097 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x4x4`): `tenferro-direct` is 41.7x faster than the slowest successful cell (`pytorch-cpu`, 0.078 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x4x4`): `tenferro-trace` is 21.3x faster than the slowest successful cell (`pytorch-cpu`, 0.078 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x8x8`): `tenferro-direct` is 40.1x faster than the slowest successful cell (`pytorch-cpu`, 0.082 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x8x8`): `tenferro-trace` is 21.1x faster than the slowest successful cell (`pytorch-cpu`, 0.082 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x2x2`): `tenferro-direct` is 26.6x faster than the slowest successful cell (`pytorch-cpu`, 0.049 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x2x2`): `tenferro-trace` is 13.5x faster than the slowest successful cell (`pytorch-cpu`, 0.049 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x4x4`): `jax-cpu` is 11.3x faster than the slowest successful cell (`pytorch-cpu`, 0.098 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x4x4`): `tenferro-direct` is 48.9x faster than the slowest successful cell (`pytorch-cpu`, 0.098 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x4x4`): `tenferro-trace` is 25.8x faster than the slowest successful cell (`pytorch-cpu`, 0.098 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x8x8`): `jax-cpu` is 12.0x faster than the slowest successful cell (`pytorch-cpu`, 0.106 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x8x8`): `tenferro-direct` is 46.3x faster than the slowest successful cell (`pytorch-cpu`, 0.106 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x8x8`): `tenferro-trace` is 25.7x faster than the slowest successful cell (`pytorch-cpu`, 0.106 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x2x2`): `jax-cpu` is 10.5x faster than the slowest successful cell (`pytorch-cpu`, 0.092 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x2x2`): `tenferro-direct` is 45.3x faster than the slowest successful cell (`pytorch-cpu`, 0.092 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x2x2`): `tenferro-trace` is 23.6x faster than the slowest successful cell (`pytorch-cpu`, 0.092 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x32x32`): `tenferro-direct` is 10.4x faster than the slowest successful cell (`pytorch-cpu`, 0.122 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x4x4`): `jax-cpu` is 10.1x faster than the slowest successful cell (`pytorch-cpu`, 0.091 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x4x4`): `tenferro-direct` is 42.0x faster than the slowest successful cell (`pytorch-cpu`, 0.091 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x4x4`): `tenferro-trace` is 21.0x faster than the slowest successful cell (`pytorch-cpu`, 0.091 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x8x8`): `tenferro-direct` is 30.8x faster than the slowest successful cell (`pytorch-cpu`, 0.094 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x8x8`): `tenferro-trace` is 18.6x faster than the slowest successful cell (`pytorch-cpu`, 0.094 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigh` (f64, threads=1, shape=`1x2x2`): `tenferro-direct` is 31.8x faster than the slowest successful cell (`jax-cpu`, 0.044 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigh` (f64, threads=1, shape=`1x2x2`): `tenferro-trace` is 16.4x faster than the slowest successful cell (`jax-cpu`, 0.044 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigvalsh` (f64, threads=1, shape=`1x2x2`): `tenferro-direct` is 13.8x faster than the slowest successful cell (`jax-cpu`, 0.015 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`3x8x8`): `tenferro-direct` is 22.6x faster than the slowest successful cell (`jax-cpu`, 0.068 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`3x8x8`): `tenferro-trace` is 16.8x faster than the slowest successful cell (`jax-cpu`, 0.068 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`4x8x8`): `tenferro-direct` is 15.5x faster than the slowest successful cell (`jax-cpu`, 0.054 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`4x8x8`): `tenferro-trace` is 12.2x faster than the slowest successful cell (`jax-cpu`, 0.054 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`8x4x4`): `tenferro-direct` is 19.7x faster than the slowest successful cell (`jax-cpu`, 0.065 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`8x4x4`): `tenferro-trace` is 14.5x faster than the slowest successful cell (`jax-cpu`, 0.065 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x2x2`): `tenferro-direct` is 12.3x faster than the slowest successful cell (`pytorch-cpu`, 0.033 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x32x32`): `tenferro-direct` is 13.2x faster than the slowest successful cell (`pytorch-cpu`, 0.080 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x32x32`): `tenferro-trace` is 10.9x faster than the slowest successful cell (`pytorch-cpu`, 0.080 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x2x2`): `tenferro-direct` is 13.2x faster than the slowest successful cell (`pytorch-cpu`, 0.039 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x8x8`): `tenferro-direct` is 15.0x faster than the slowest successful cell (`jax-cpu`, 0.057 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x8x8`): `tenferro-trace` is 12.0x faster than the slowest successful cell (`jax-cpu`, 0.057 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x2x2`): `tenferro-direct` is 19.8x faster than the slowest successful cell (`pytorch-cpu`, 0.061 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x2x2`): `tenferro-trace` is 15.1x faster than the slowest successful cell (`pytorch-cpu`, 0.061 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x8x8`): `tenferro-direct` is 15.6x faster than the slowest successful cell (`jax-cpu`, 0.063 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x8x8`): `tenferro-trace` is 11.8x faster than the slowest successful cell (`jax-cpu`, 0.063 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x2x2`): `tenferro-direct` is 23.9x faster than the slowest successful cell (`pytorch-cpu`, 0.078 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x2x2`): `tenferro-trace` is 17.1x faster than the slowest successful cell (`pytorch-cpu`, 0.078 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x4x4`): `tenferro-direct` is 17.4x faster than the slowest successful cell (`jax-cpu`, 0.072 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x4x4`): `tenferro-trace` is 11.5x faster than the slowest successful cell (`jax-cpu`, 0.072 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x8x8`): `tenferro-direct` is 17.8x faster than the slowest successful cell (`pytorch-cpu`, 0.097 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x8x8`): `tenferro-trace` is 11.7x faster than the slowest successful cell (`pytorch-cpu`, 0.097 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`1x2x2`): `tenferro-direct` is 17.8x faster than the slowest successful cell (`pytorch-cpu`, 0.042 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`1x2x2`): `tenferro-trace` is 11.9x faster than the slowest successful cell (`pytorch-cpu`, 0.042 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`1x4x4`): `tenferro-direct` is 19.2x faster than the slowest successful cell (`pytorch-cpu`, 0.050 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`1x4x4`): `tenferro-trace` is 10.7x faster than the slowest successful cell (`pytorch-cpu`, 0.050 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`1x8x8`): `tenferro-direct` is 11.9x faster than the slowest successful cell (`pytorch-cpu`, 0.036 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`3x2x2`): `tenferro-direct` is 20.3x faster than the slowest successful cell (`pytorch-cpu`, 0.064 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`3x2x2`): `tenferro-trace` is 15.1x faster than the slowest successful cell (`pytorch-cpu`, 0.064 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`3x4x4`): `tenferro-direct` is 19.7x faster than the slowest successful cell (`pytorch-cpu`, 0.066 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`3x4x4`): `tenferro-trace` is 10.8x faster than the slowest successful cell (`pytorch-cpu`, 0.066 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`4x2x2`): `tenferro-direct` is 18.6x faster than the slowest successful cell (`pytorch-cpu`, 0.063 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`4x2x2`): `tenferro-trace` is 12.2x faster than the slowest successful cell (`pytorch-cpu`, 0.063 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`4x4x4`): `tenferro-direct` is 18.5x faster than the slowest successful cell (`pytorch-cpu`, 0.076 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`4x4x4`): `tenferro-trace` is 11.6x faster than the slowest successful cell (`pytorch-cpu`, 0.076 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`4x8x8`): `tenferro-direct` is 10.1x faster than the slowest successful cell (`pytorch-cpu`, 0.069 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`8x2x2`): `tenferro-direct` is 21.7x faster than the slowest successful cell (`pytorch-cpu`, 0.074 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`8x2x2`): `tenferro-trace` is 11.6x faster than the slowest successful cell (`pytorch-cpu`, 0.074 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=1, shape=`1x32x32,rhs=1`): `tenferro-direct` is 10.6x faster than the slowest successful cell (`jax-cpu`, 0.045 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1x2x2,rhs=1`): `tenferro-direct` is 15.5x faster than the slowest successful cell (`jax-cpu`, 0.030 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1x32x32,rhs=1`): `pytorch-cpu` is 14.2x faster than the slowest successful cell (`jax-cpu`, 0.178 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1x32x32,rhs=1`): `tenferro-direct` is 31.2x faster than the slowest successful cell (`jax-cpu`, 0.178 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1x32x32,rhs=1`): `tenferro-trace` is 17.9x faster than the slowest successful cell (`jax-cpu`, 0.178 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`3x4x4,rhs=1`): `tenferro-direct` is 13.2x faster than the slowest successful cell (`jax-cpu`, 0.034 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x2x2,rhs=1`): `tenferro-direct` is 23.0x faster than the slowest successful cell (`pytorch-cpu`, 0.065 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x4x4,rhs=1`): `tenferro-direct` is 18.8x faster than the slowest successful cell (`pytorch-cpu`, 0.064 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/cholesky` (f64, threads=4, shape=`1536x1536`): `jax-cpu` is 23.3x faster than the slowest successful cell (`julia-base`, 386.082 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/cholesky` (f64, threads=4, shape=`1536x1536`): `pytorch-cpu` is 15.6x faster than the slowest successful cell (`julia-base`, 386.082 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/cholesky` (f64, threads=4, shape=`1536x1536`): `tenferro-direct` is 49.3x faster than the slowest successful cell (`julia-base`, 386.082 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/cholesky` (f64, threads=4, shape=`1536x1536`): `tenferro-trace` is 47.6x faster than the slowest successful cell (`julia-base`, 386.082 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `jax-cpu` is 20.3x faster than the slowest successful cell (`julia-base`, 212.717 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `pytorch-cpu` is 16.5x faster than the slowest successful cell (`julia-base`, 212.717 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `tenferro-direct` is 21.8x faster than the slowest successful cell (`julia-base`, 212.717 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `tenferro-trace` is 16.7x faster than the slowest successful cell (`julia-base`, 212.717 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eig` (f64, threads=4, shape=`160x160`): `jax-cpu` is 38.4x faster than the slowest successful cell (`julia-base`, 331.447 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eig` (f64, threads=4, shape=`160x160`): `pytorch-cpu` is 40.1x faster than the slowest successful cell (`julia-base`, 331.447 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eig` (f64, threads=4, shape=`160x160`): `tenferro-direct` is 46.9x faster than the slowest successful cell (`julia-base`, 331.447 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eig` (f64, threads=4, shape=`160x160`): `tenferro-trace` is 44.4x faster than the slowest successful cell (`julia-base`, 331.447 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvals` (f64, threads=4, shape=`192x192`): `jax-cpu` is 32.9x faster than the slowest successful cell (`julia-base`, 283.929 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvals` (f64, threads=4, shape=`192x192`): `pytorch-cpu` is 28.4x faster than the slowest successful cell (`julia-base`, 283.929 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvals` (f64, threads=4, shape=`192x192`): `tenferro-direct` is 31.6x faster than the slowest successful cell (`julia-base`, 283.929 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvals` (f64, threads=4, shape=`192x192`): `tenferro-trace` is 31.0x faster than the slowest successful cell (`julia-base`, 283.929 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvalsh` (f64, threads=4, shape=`512x512`): `jax-cpu` is 15.1x faster than the slowest successful cell (`julia-base`, 649.186 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvalsh` (f64, threads=4, shape=`512x512`): `pytorch-cpu` is 26.3x faster than the slowest successful cell (`julia-base`, 649.186 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvalsh` (f64, threads=4, shape=`512x512`): `tenferro-direct` is 27.2x faster than the slowest successful cell (`julia-base`, 649.186 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvalsh` (f64, threads=4, shape=`512x512`): `tenferro-trace` is 26.6x faster than the slowest successful cell (`julia-base`, 649.186 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `jax-cpu` is 24.6x faster than the slowest successful cell (`julia-base`, 367.632 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `pytorch-cpu` is 32.1x faster than the slowest successful cell (`julia-base`, 367.632 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `tenferro-direct` is 45.3x faster than the slowest successful cell (`julia-base`, 367.632 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `tenferro-trace` is 41.1x faster than the slowest successful cell (`julia-base`, 367.632 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lstsq` (f64, threads=4, shape=`768x384,rhs=16`): `pytorch-cpu` is 20.5x faster than the slowest successful cell (`julia-base`, 149.410 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lstsq` (f64, threads=4, shape=`768x384,rhs=16`): `tenferro-eager` is 15.8x faster than the slowest successful cell (`julia-base`, 149.410 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lstsq` (f64, threads=4, shape=`768x384,rhs=16`): `tenferro-trace` is 15.9x faster than the slowest successful cell (`julia-base`, 149.410 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `jax-cpu` is 16.0x faster than the slowest successful cell (`julia-base`, 251.868 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `pytorch-cpu` is 15.7x faster than the slowest successful cell (`julia-base`, 251.868 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `tenferro-direct` is 31.1x faster than the slowest successful cell (`julia-base`, 251.868 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `tenferro-trace` is 28.8x faster than the slowest successful cell (`julia-base`, 251.868 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 13.0x faster than the slowest successful cell (`jax-cpu`, 8.892 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=4, shape=`2048x2048`): `tenferro-trace` is 11.6x faster than the slowest successful cell (`jax-cpu`, 8.892 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `jax-cpu` is 61.7x faster than the slowest successful cell (`julia-base`, 987.584 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `pytorch-cpu` is 72.6x faster than the slowest successful cell (`julia-base`, 987.584 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `tenferro-direct` is 85.5x faster than the slowest successful cell (`julia-base`, 987.584 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `tenferro-trace` is 88.8x faster than the slowest successful cell (`julia-base`, 987.584 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `jax-cpu` is 72.7x faster than the slowest successful cell (`julia-base`, 1056.298 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `pytorch-cpu` is 98.5x faster than the slowest successful cell (`julia-base`, 1056.298 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `tenferro-direct` is 95.8x faster than the slowest successful cell (`julia-base`, 1056.298 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `tenferro-trace` is 98.2x faster than the slowest successful cell (`julia-base`, 1056.298 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `jax-cpu` is 35.9x faster than the slowest successful cell (`julia-base`, 366.934 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `pytorch-cpu` is 38.0x faster than the slowest successful cell (`julia-base`, 366.934 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `tenferro-direct` is 40.2x faster than the slowest successful cell (`julia-base`, 366.934 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `tenferro-trace` is 44.5x faster than the slowest successful cell (`julia-base`, 366.934 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `jax-cpu` is 37.0x faster than the slowest successful cell (`julia-base`, 1904.572 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `pytorch-cpu` is 48.1x faster than the slowest successful cell (`julia-base`, 1904.572 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `tenferro-eager` is 56.7x faster than the slowest successful cell (`julia-base`, 1904.572 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `tenferro-trace` is 58.6x faster than the slowest successful cell (`julia-base`, 1904.572 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=1, shape=`4096x256`): `jax-cpu` is 10.6x faster than the slowest successful cell (`tenferro-eager`, 10.647 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 14.9x faster than the slowest successful cell (`tenferro-eager`, 1.696 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 14.5x faster than the slowest successful cell (`tenferro-eager`, 9.521 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=1, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 8868.8x faster than the slowest successful cell (`julia-base`, 15.520 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=4, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 13056.7x faster than the slowest successful cell (`julia-base`, 19.585 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=1, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 12439.9x faster than the slowest successful cell (`julia-base`, 16.595 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=4, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 10247.8x faster than the slowest successful cell (`pytorch-cpu`, 14.521 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`): `tenferro-trace` is 179517.2x faster than the slowest successful cell (`julia-base`, 239.476 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`): `strided-jl` is 11.6x faster than the slowest successful cell (`julia-base`, 244.646 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`): `tenferro-trace` is 183530.1x faster than the slowest successful cell (`julia-base`, 244.646 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/broadcast_in_dim_view` (f64, threads=1, shape=`8192x1 -> 8192x4096`): `tenferro-direct` is 11.7x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/broadcast_in_dim_view` (f64, threads=4, shape=`8192x1 -> 8192x4096`): `tenferro-direct` is 11.2x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=1, shape=`33554432 -> 8192x4096`): `julia-base` is 169.5x faster than the slowest successful cell (`tenferro-direct`, 0.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=4, shape=`33554432 -> 8192x4096`): `julia-base` is 1393.8x faster than the slowest successful cell (`tenferro-direct`, 0.007 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=1, shape=`4194304 -> 2096128`): `julia-base` is 97.6x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=4, shape=`4194304 -> 2096128`): `julia-base` is 48.3x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=1, shape=`4096x4096`): `julia-base` is 140.6x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=4, shape=`4096x4096`): `julia-base` is 203.6x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

## Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

- `cpu/indexing_layout/slice` (f64, threads=1, shape=`4194304 -> 2096128`), `tenferro-trace`: 16769024 minimum logical bytes imply 12195.7 GB/s at 0.001 ms. The conservative physical lower bound is 0.168 ms, so the cell is 122.0x faster than the bound.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`), `tenferro-trace`: 16769024 minimum logical bytes imply 4192.3 GB/s at 0.004 ms. The conservative physical lower bound is 0.042 ms, so the cell is 10.5x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=1, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 153391.7 GB/s at 0.002 ms. The conservative physical lower bound is 2.684 ms, so the cell is 1533.9x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=4, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 178957.0 GB/s at 0.002 ms. The conservative physical lower bound is 0.671 ms, so the cell is 447.4x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=1, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 201226.0 GB/s at 0.001 ms. The conservative physical lower bound is 2.684 ms, so the cell is 2012.3x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=4, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 189439.3 GB/s at 0.001 ms. The conservative physical lower bound is 0.671 ms, so the cell is 473.6x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 100613.0 GB/s at 0.001 ms. The conservative physical lower bound is 1.342 ms, so the cell is 1006.1x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 100688.5 GB/s at 0.001 ms. The conservative physical lower bound is 0.336 ms, so the cell is 251.7x faster than the bound.
