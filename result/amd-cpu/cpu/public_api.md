# CPU Public API Benchmark Results

- Suite: `cpu/public_api`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/public_api.yaml`
- Run metadata: `data/results/amd-cpu/cpu/public_api/20261123_185908/run.yaml`
- Timestamp: `2026-11-23T18:59:08.735746+00:00`

Latest run: `./scripts/run_cpu_public_api.sh 1 4`.

- Public API coverage manifest: `benchmarks/cpu/public_api_coverage.yaml`

This file is generated from sequential CPU public API runs under `data/results/amd-cpu/cpu/public_api/20261123_185908`.

- tenferro-rs commit: `ce9125d0cff4ee40841120759b5df45252c27f22`

## CPU Information

- Model: `AMD EPYC 9V74 80-Core Processor`
- Vendor: `AuthenticAMD`
- Logical CPUs: `4`
- Sockets: `1`
- Cores per socket: `2`
- Threads per core: `2`
- NUMA nodes: `1`
- Transparent huge pages: `madvise (always [madvise] never)`
- Transparent huge page defrag: `madvise (always defer defer+madvise [madvise] never)`
- Python platform: `Linux-7.0.0-1020-azure-x86_64-with-glibc2.39`
- L3 domains (shared CPUs): `0-3`
- Container: `devcontainer`

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

- CSV: `data/results/amd-cpu/cpu/public_api/20261123_185908/cpu_public_api_t1_20261123_185908.csv`
- CSV: `data/results/amd-cpu/cpu/public_api/20261123_185908/cpu_public_api_t4_20261123_185908.csv`

## CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| cpu/complex | `cholesky` | c64 | 1 | `448x448` | 3.848 ± 0.013 | - | 3.849 ± 0.035 | 4.693 ± 0.011 | 4.038 ± 0.027 | 4.147 ± 0.685 | - |
| cpu/complex | `cholesky` | c64 | 4 | `448x448` | 2.660 ± 0.029 | - | 191.994 ± 0.014 | 2.293 ± 0.033 | 3.850 ± 0.913 | 3.181 ± 2.823 | - |
| cpu/complex | `conj` | c64 | 1 | `16777216` | 134.074 ± 5.966 | - | 132.672 ± 6.788 | 127.096 ± 4.893 | 126.982 ± 2.671 | 148.631 ± 6.703 | 152.639 ± 18.920 |
| cpu/complex | `conj` | c64 | 4 | `16777216` | 45.021 ± 3.178 | - | 154.124 ± 12.790 | 46.182 ± 2.220 | 68.410 ± 1.986 | 149.471 ± 5.334 | 68.008 ± 20.307 |
| cpu/complex | `div` | c64 | 1 | `8388608` | 120.040 ± 4.777 | - | 128.421 ± 7.638 | 104.624 ± 2.801 | 103.101 ± 3.760 | 131.451 ± 3.891 | 305.991 ± 3.127 |
| cpu/complex | `div` | c64 | 4 | `8388608` | 43.050 ± 0.892 | - | 130.280 ± 1.320 | 33.837 ± 1.372 | 50.395 ± 3.670 | 130.034 ± 2.555 | 59.985 ± 1.689 |
| cpu/complex | `dot_general` | c64 | 1 | `640x640` | 50.439 ± 0.149 | - | 50.372 ± 0.252 | 51.102 ± 0.087 | 73.248 ± 0.262 | 51.257 ± 0.112 | - |
| cpu/complex | `dot_general` | c64 | 4 | `640x640` | 25.934 ± 0.112 | - | 63.005 ± 5.081 | 31.832 ± 1.293 | 32.037 ± 0.136 | 25.614 ± 0.677 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 1 | `640x640` | 50.430 ± 0.124 | - | unsupported | 51.550 ± 0.481 | 73.584 ± 0.340 | 51.465 ± 5.289 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 4 | `640x640` | 26.023 ± 0.085 | - | unsupported | 32.067 ± 1.425 | 35.822 ± 3.852 | 29.181 ± 4.753 | - |
| cpu/complex | `eig` | c64 | 1 | `112x112` | 7.753 ± 0.057 | - | 7.770 ± 0.021 | 11.953 ± 0.044 | 12.485 ± 0.086 | 11.991 ± 0.202 | - |
| cpu/complex | `eig` | c64 | 4 | `112x112` | 10.298 ± 0.153 | - | 1501.996 ± 25.010 | 14.576 ± 0.062 | 18.656 ± 0.065 | 18.448 ± 0.011 | - |
| cpu/complex | `exp` | c64 | 1 | `4194304` | 74.615 ± 0.078 | - | 75.967 ± 0.165 | 119.105 ± 1.750 | 108.479 ± 1.773 | 81.563 ± 11.086 | 81.234 ± 1.947 |
| cpu/complex | `exp` | c64 | 4 | `4194304` | 31.259 ± 0.246 | - | 95.998 ± 0.002 | 45.640 ± 2.457 | 54.310 ± 0.753 | 81.428 ± 9.071 | 39.159 ± 3.870 |
| cpu/complex | `log` | c64 | 1 | `4194304` | 114.048 ± 0.357 | - | 113.187 ± 0.434 | 160.733 ± 3.083 | 177.380 ± 1.923 | 148.231 ± 4.550 | 167.023 ± 18.403 |
| cpu/complex | `log` | c64 | 4 | `4194304` | 50.449 ± 7.555 | - | 131.997 ± 0.007 | 64.628 ± 0.979 | 74.368 ± 0.945 | 148.085 ± 2.503 | 63.049 ± 9.701 |
| cpu/complex | `mul` | c64 | 1 | `8388608` | 80.481 ± 3.780 | - | 79.758 ± 4.357 | 73.792 ± 3.056 | 81.244 ± 1.969 | 82.136 ± 1.547 | 89.488 ± 4.713 |
| cpu/complex | `mul` | c64 | 4 | `8388608` | 26.818 ± 2.019 | - | 94.451 ± 3.325 | 26.493 ± 1.443 | 41.005 ± 4.216 | 82.103 ± 0.979 | 35.863 ± 0.889 |
| cpu/complex | `norm_fro` | c64 | 1 | `2048x1536` | 1.507 ± 0.064 | - | 19.762 ± 0.119 | 23.370 ± 0.059 | 19.387 ± 0.388 | 4.473 ± 0.015 | - |
| cpu/complex | `norm_fro` | c64 | 4 | `2048x1536` | 1.275 ± 0.013 | - | 59.998 ± 0.009 | 10.838 ± 0.003 | 2.627 ± 0.307 | 4.469 ± 0.007 | - |
| cpu/complex | `qr` | c64 | 1 | `256x256` | 8.084 ± 0.018 | - | 8.124 ± 0.041 | 9.807 ± 0.104 | 8.291 ± 0.370 | 11.422 ± 0.142 | - |
| cpu/complex | `qr` | c64 | 4 | `256x256` | 4.666 ± 0.026 | - | 396.002 ± 0.014 | 6.016 ± 0.021 | 7.796 ± 0.076 | 7.044 ± 0.109 | - |
| cpu/complex | `solve` | c64 | 1 | `384x384,rhs=8` | 4.901 ± 0.019 | - | 4.927 ± 0.033 | 6.041 ± 0.041 | 5.543 ± 0.017 | 5.155 ± 0.357 | - |
| cpu/complex | `solve` | c64 | 4 | `384x384,rhs=8` | 3.809 ± 0.030 | - | 545.009 ± 36.003 | 3.547 ± 0.025 | 5.993 ± 1.810 | 4.186 ± 0.039 | - |
| cpu/complex | `svd` | c64 | 1 | `160x160` | 6.704 ± 0.059 | - | 6.689 ± 0.028 | 7.890 ± 0.022 | 5.588 ± 0.012 | 5.461 ± 0.147 | - |
| cpu/complex | `svd` | c64 | 4 | `160x160` | 6.610 ± 0.040 | - | 6.617 ± 0.083 | 6.291 ± 0.059 | 6.806 ± 0.073 | 6.466 ± 0.032 | - |
| cpu/complex | `tensordot` | c64 | 1 | `640x640` | 50.445 ± 0.076 | - | 50.672 ± 0.188 | 51.380 ± 0.212 | 73.362 ± 0.535 | 48.555 ± 0.348 | - |
| cpu/complex | `tensordot` | c64 | 4 | `640x640` | 26.095 ± 0.068 | - | 62.996 ± 5.326 | 32.029 ± 0.660 | 32.071 ± 0.674 | 24.332 ± 0.216 | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 1 | `1024x1024` | 51.286 ± 0.281 | - | unsupported | 57.064 ± 0.284 | 57.896 ± 0.612 | - | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 4 | `1024x1024` | 25.160 ± 0.241 | - | unsupported | 27.823 ± 0.590 | 32.018 ± 0.597 | - | - |
| cpu/elementwise_reduction | `abs` | f64 | 1 | `33554432` | 134.473 ± 7.609 | - | 132.336 ± 8.078 | 131.809 ± 1.923 | 131.095 ± 1.035 | 148.917 ± 2.077 | 162.264 ± 10.251 |
| cpu/elementwise_reduction | `abs` | f64 | 4 | `33554432` | 44.134 ± 0.864 | - | 142.353 ± 12.121 | 45.113 ± 0.725 | 67.218 ± 5.051 | 148.098 ± 10.736 | 70.799 ± 1.488 |
| cpu/elementwise_reduction | `add` | f64 | 1 | `33554432` | 141.409 ± 6.986 | - | 140.772 ± 4.813 | 147.422 ± 4.180 | 144.526 ± 2.126 | 156.513 ± 10.492 | 193.062 ± 7.929 |
| cpu/elementwise_reduction | `add` | f64 | 4 | `33554432` | 48.848 ± 1.017 | - | 155.405 ± 0.893 | 49.652 ± 1.331 | 73.143 ± 2.754 | 157.214 ± 9.985 | 81.229 ± 10.406 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 1 | `4194304` | 29.657 ± 0.772 | - | 54.821 ± 0.610 | 76.144 ± 1.348 | 54.048 ± 0.631 | 80.473 ± 10.753 | 87.156 ± 10.871 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 4 | `4194304` | 13.720 ± 0.280 | - | 120.000 ± 2.998 | 32.825 ± 0.742 | 27.740 ± 1.031 | 81.543 ± 10.896 | 34.897 ± 16.763 |
| cpu/elementwise_reduction | `clamp` | f64 | 1 | `8388608` | 9.201 ± 0.061 | - | 9.541 ± 0.197 | 40.667 ± 1.592 | 43.834 ± 1.784 | 39.779 ± 10.162 | 50.263 ± 7.556 |
| cpu/elementwise_reduction | `clamp` | f64 | 4 | `8388608` | 6.961 ± 0.120 | - | 23.995 ± 0.018 | 13.473 ± 0.360 | 22.268 ± 0.617 | 47.270 ± 9.826 | 16.118 ± 16.207 |
| cpu/elementwise_reduction | `compare_lt` | f64 | 1 | `33554432` | 20.801 ± 0.435 | - | 20.698 ± 0.165 | 52.149 ± 0.258 | 52.641 ± 0.622 | 19.740 ± 0.175 | - |
| cpu/elementwise_reduction | `compare_lt` | f64 | 4 | `33554432` | 15.199 ± 0.308 | - | 35.998 ± 0.004 | 19.156 ± 0.090 | 27.709 ± 0.413 | 20.108 ± 0.124 | - |
| cpu/elementwise_reduction | `cos` | f64 | 1 | `8388608` | 30.030 ± 0.081 | - | 31.856 ± 0.224 | 65.424 ± 0.476 | 118.243 ± 1.389 | 66.489 ± 10.624 | 72.950 ± 7.326 |
| cpu/elementwise_reduction | `cos` | f64 | 4 | `8388608` | 12.110 ± 0.098 | - | 47.997 ± 0.004 | 22.725 ± 0.291 | 60.330 ± 1.164 | 72.447 ± 12.675 | 30.202 ± 18.006 |
| cpu/elementwise_reduction | `div` | f64 | 1 | `33554432` | 143.820 ± 3.432 | - | 141.554 ± 14.521 | 141.604 ± 3.575 | 155.933 ± 3.073 | 160.962 ± 3.274 | 188.923 ± 1.689 |
| cpu/elementwise_reduction | `div` | f64 | 4 | `33554432` | 48.577 ± 0.244 | - | 152.877 ± 0.965 | 49.510 ± 1.482 | 79.754 ± 1.461 | 159.469 ± 4.677 | 104.645 ± 30.834 |
| cpu/elementwise_reduction | `exp` | f64 | 1 | `8388608` | 22.523 ± 0.070 | - | 26.410 ± 1.874 | 49.036 ± 1.778 | 44.516 ± 3.339 | 62.204 ± 12.531 | 68.432 ± 12.930 |
| cpu/elementwise_reduction | `exp` | f64 | 4 | `8388608` | 8.589 ± 0.128 | - | 47.998 ± 0.008 | 17.610 ± 0.442 | 23.393 ± 0.739 | 62.430 ± 12.460 | 33.591 ± 15.742 |
| cpu/elementwise_reduction | `expm1` | f64 | 1 | `4194304` | 22.374 ± 0.052 | - | 24.719 ± 0.119 | 32.482 ± 0.985 | 28.755 ± 0.208 | 37.492 ± 9.170 | 47.969 ± 11.750 |
| cpu/elementwise_reduction | `expm1` | f64 | 4 | `4194304` | 8.367 ± 0.062 | - | 35.996 ± 0.005 | 11.704 ± 5.354 | 14.661 ± 0.272 | 38.311 ± 0.761 | 19.310 ± 11.866 |
| cpu/elementwise_reduction | `log` | f64 | 1 | `8388608` | 31.881 ± 0.155 | - | 28.991 ± 0.042 | 54.188 ± 0.864 | 68.650 ± 1.186 | 87.471 ± 15.720 | 86.044 ± 14.079 |
| cpu/elementwise_reduction | `log` | f64 | 4 | `8388608` | 11.332 ± 0.033 | - | 47.997 ± 12.002 | 20.071 ± 0.751 | 34.327 ± 1.534 | 77.565 ± 13.781 | 35.057 ± 17.395 |
| cpu/elementwise_reduction | `log1p` | f64 | 1 | `4194304` | 14.991 ± 0.032 | - | 16.010 ± 0.085 | 31.411 ± 0.987 | 37.942 ± 0.473 | 44.895 ± 10.135 | 47.541 ± 11.831 |
| cpu/elementwise_reduction | `log1p` | f64 | 4 | `4194304` | 6.016 ± 0.061 | - | 35.998 ± 0.007 | 11.284 ± 0.618 | 19.565 ± 0.756 | 44.987 ± 10.098 | 20.983 ± 12.660 |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 1 | `1024x64` | 0.492 ± 0.009 | 0.530 ± 0.006 | unsupported | unsupported | 0.519 ± 0.005 | - | - |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 4 | `1024x64` | 0.415 ± 0.014 | 53.999 ± 0.000 | unsupported | unsupported | 0.401 ± 0.185 | - | - |
| cpu/elementwise_reduction | `maximum` | f64 | 1 | `33554432` | 152.247 ± 6.605 | - | 148.931 ± 9.550 | 145.089 ± 2.802 | 146.421 ± 2.822 | 166.398 ± 7.227 | 191.842 ± 3.139 |
| cpu/elementwise_reduction | `maximum` | f64 | 4 | `33554432` | 47.813 ± 0.857 | - | 164.773 ± 12.311 | 49.071 ± 1.281 | 74.270 ± 2.465 | 164.736 ± 5.617 | 79.787 ± 1.479 |
| cpu/elementwise_reduction | `minimum` | f64 | 1 | `33554432` | 152.535 ± 8.504 | - | 150.915 ± 11.681 | 146.783 ± 1.780 | 145.378 ± 1.570 | 165.159 ± 12.862 | 192.656 ± 8.851 |
| cpu/elementwise_reduction | `minimum` | f64 | 4 | `33554432` | 47.732 ± 0.949 | - | 162.532 ± 12.371 | 49.828 ± 4.181 | 74.941 ± 4.844 | 168.097 ± 21.124 | 79.171 ± 9.158 |
| cpu/elementwise_reduction | `mul` | f64 | 1 | `33554432` | 140.598 ± 4.078 | - | 140.195 ± 5.616 | 145.157 ± 3.278 | 149.681 ± 6.564 | 157.959 ± 46.028 | 190.628 ± 5.296 |
| cpu/elementwise_reduction | `mul` | f64 | 4 | `33554432` | 48.901 ± 0.257 | - | 155.138 ± 5.609 | 48.994 ± 0.815 | 73.255 ± 0.951 | 157.609 ± 47.625 | 81.085 ± 2.218 |
| cpu/elementwise_reduction | `neg` | f64 | 1 | `33554432` | 123.283 ± 9.656 | - | 132.899 ± 8.148 | 130.471 ± 5.882 | 130.295 ± 1.163 | 145.627 ± 4.745 | 159.446 ± 4.218 |
| cpu/elementwise_reduction | `neg` | f64 | 4 | `33554432` | 43.851 ± 0.080 | - | 143.715 ± 16.051 | 44.355 ± 6.622 | 64.148 ± 3.624 | 144.935 ± 3.702 | 70.260 ± 15.788 |
| cpu/elementwise_reduction | `pow` | f64 | 1 | `4194304` | 56.010 ± 0.214 | - | 56.666 ± 0.348 | 65.269 ± 1.044 | 73.438 ± 1.543 | 137.424 ± 0.400 | 124.658 ± 15.410 |
| cpu/elementwise_reduction | `pow` | f64 | 4 | `4194304` | 25.300 ± 0.090 | - | 72.000 ± 0.016 | 22.449 ± 0.286 | 38.394 ± 1.675 | 138.332 ± 1.392 | 47.302 ± 17.385 |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 1 | `8192x4096` | 14.851 ± 0.102 | - | 14.947 ± 0.031 | 9.505 ± 0.059 | 38.351 ± 0.590 | 8.573 ± 0.068 | - |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 4 | `8192x4096` | 6.935 ± 0.189 | - | 35.999 ± 0.013 | 7.335 ± 0.104 | 19.586 ± 0.078 | 9.175 ± 0.218 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 1 | `2048x2048` | 1.906 ± 0.008 | - | 1.914 ± 0.014 | 28.296 ± 0.162 | 2.665 ± 0.080 | 1.019 ± 0.076 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 4 | `2048x2048` | 0.896 ± 0.023 | - | 23.998 ± 0.004 | 11.722 ± 0.582 | 1.666 ± 0.506 | 0.953 ± 0.109 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 1 | `2048x2048` | 3.278 ± 0.092 | - | 3.228 ± 0.296 | 5.992 ± 0.033 | 1.546 ± 0.015 | 1.710 ± 0.114 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 4 | `2048x2048` | 1.653 ± 0.144 | - | 24.000 ± 0.013 | 3.024 ± 0.046 | 0.879 ± 0.035 | 1.317 ± 0.297 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 1 | `8192x4096` | 14.278 ± 0.036 | - | 14.250 ± 0.023 | 9.524 ± 0.038 | 37.901 ± 0.149 | 8.543 ± 0.063 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 4 | `8192x4096` | 6.916 ± 0.061 | - | 36.002 ± 11.999 | 7.297 ± 0.115 | 19.658 ± 3.402 | 8.747 ± 0.241 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 1 | `2048x2048` | 1.847 ± 0.016 | - | 1.844 ± 0.006 | 28.396 ± 0.247 | 2.769 ± 0.098 | 0.960 ± 0.023 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 4 | `2048x2048` | 0.859 ± 0.011 | - | 24.000 ± 0.008 | 12.444 ± 1.718 | 1.613 ± 0.129 | 0.883 ± 0.039 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 1 | `4096x4096` | 15.602 ± 0.221 | - | 15.612 ± 0.139 | 17.956 ± 0.115 | 5.824 ± 0.038 | 10.423 ± 0.751 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 4 | `4096x4096` | 6.663 ± 0.225 | - | 35.998 ± 0.014 | 9.074 ± 0.121 | 3.857 ± 0.357 | 9.773 ± 1.069 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 1 | `8192x4096` | 8.401 ± 0.077 | - | 8.393 ± 0.035 | 7.788 ± 0.073 | 27.560 ± 0.404 | 8.195 ± 0.083 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 4 | `8192x4096` | 7.362 ± 0.409 | - | 23.995 ± 0.007 | 7.352 ± 0.178 | 14.393 ± 0.179 | 8.769 ± 0.029 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 1 | `2048x2048` | 0.873 ± 0.020 | - | 0.848 ± 0.063 | 2.386 ± 0.067 | 2.009 ± 0.061 | 0.767 ± 0.023 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 4 | `2048x2048` | 0.330 ± 0.036 | - | 23.999 ± 0.012 | 1.171 ± 0.082 | 1.139 ± 0.057 | 0.846 ± 0.089 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 1 | `2048x2048` | 1.720 ± 0.031 | - | 1.782 ± 0.059 | 0.727 ± 0.053 | 2.468 ± 0.268 | 1.204 ± 0.140 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 4 | `2048x2048` | 0.820 ± 0.058 | - | 24.002 ± 0.015 | 0.295 ± 0.025 | 1.331 ± 0.036 | 1.515 ± 0.103 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 1 | `8192x4096` | 8.380 ± 0.078 | - | 8.525 ± 0.077 | 8.173 ± 0.424 | 27.734 ± 0.398 | 8.169 ± 0.093 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 4 | `8192x4096` | 7.302 ± 0.062 | - | 23.995 ± 0.016 | 7.345 ± 0.060 | 14.881 ± 0.898 | 8.221 ± 0.158 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 1 | `2048x2048` | 0.789 ± 0.061 | - | 0.812 ± 0.060 | 2.332 ± 0.076 | 2.071 ± 0.054 | 0.813 ± 0.028 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 4 | `2048x2048` | 0.335 ± 0.104 | - | 23.999 ± 0.011 | 1.088 ± 0.102 | 1.160 ± 0.018 | 0.715 ± 0.077 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 1 | `2048x2048` | 1.616 ± 0.062 | - | 1.731 ± 0.114 | 0.704 ± 0.034 | 2.532 ± 0.295 | 1.050 ± 0.150 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 4 | `2048x2048` | 0.778 ± 0.101 | - | 23.999 ± 0.017 | 0.292 ± 0.035 | 1.310 ± 0.089 | 0.963 ± 0.126 | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 1 | `2048x2048` | 0.843 ± 0.062 | 0.823 ± 0.028 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 4 | `2048x2048` | 0.296 ± 0.027 | 23.999 ± 0.001 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `rem` | f64 | 1 | `8388608` | 39.344 ± 0.453 | - | 42.009 ± 0.177 | 63.798 ± 1.438 | 73.174 ± 1.770 | 70.793 ± 19.301 | 81.679 ± 8.375 |
| cpu/elementwise_reduction | `rem` | f64 | 4 | `8388608` | 18.526 ± 0.033 | - | 59.998 ± 12.004 | 21.781 ± 0.641 | 39.129 ± 2.280 | 82.010 ± 21.325 | 44.195 ± 11.365 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 1 | `33554432` | 147.451 ± 1.526 | - | 149.781 ± 7.178 | 154.169 ± 1.036 | 269.030 ± 3.380 | 287.666 ± 4.380 | 290.597 ± 3.475 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 4 | `33554432` | 47.413 ± 0.234 | - | 166.820 ± 0.698 | 49.382 ± 0.506 | 135.994 ± 1.950 | 285.388 ± 6.788 | 122.065 ± 7.251 |
| cpu/elementwise_reduction | `select` | f64 | 1 | `33554432` | 146.715 ± 1.559 | - | 149.307 ± 2.819 | 170.712 ± 3.552 | 184.713 ± 5.384 | 163.168 ± 7.876 | - |
| cpu/elementwise_reduction | `select` | f64 | 4 | `33554432` | 50.241 ± 2.556 | - | 167.559 ± 4.210 | 74.911 ± 36.773 | 91.307 ± 0.553 | 165.456 ± 8.349 | - |
| cpu/elementwise_reduction | `sign` | f64 | 1 | `33554432` | 138.138 ± 4.559 | - | 138.452 ± 8.358 | 133.814 ± 1.080 | 129.592 ± 3.035 | 149.186 ± 13.670 | 164.512 ± 3.066 |
| cpu/elementwise_reduction | `sign` | f64 | 4 | `33554432` | 44.426 ± 0.610 | - | 153.925 ± 4.063 | 45.871 ± 0.885 | 64.467 ± 3.059 | 148.022 ± 6.277 | 70.342 ± 0.427 |
| cpu/elementwise_reduction | `sin` | f64 | 1 | `8388608` | 23.762 ± 2.935 | - | 28.322 ± 0.046 | 64.216 ± 0.910 | 120.003 ± 0.495 | 67.296 ± 15.099 | 69.964 ± 9.119 |
| cpu/elementwise_reduction | `sin` | f64 | 4 | `8388608` | 10.326 ± 0.086 | - | 47.998 ± 0.005 | 23.167 ± 0.708 | 61.574 ± 1.214 | 70.992 ± 9.752 | 30.636 ± 13.102 |
| cpu/elementwise_reduction | `sqrt` | f64 | 1 | `33554432` | 133.600 ± 0.907 | - | 136.547 ± 8.040 | 192.040 ± 1.507 | 139.738 ± 5.889 | 225.845 ± 3.752 | 230.832 ± 29.183 |
| cpu/elementwise_reduction | `sqrt` | f64 | 4 | `33554432` | 44.924 ± 1.143 | - | 155.032 ± 0.433 | 69.778 ± 8.183 | 71.299 ± 5.531 | 229.697 ± 4.151 | 99.399 ± 4.483 |
| cpu/elementwise_reduction | `sub` | f64 | 1 | `33554432` | 139.259 ± 11.880 | - | 141.815 ± 8.375 | 147.106 ± 10.194 | 144.881 ± 5.526 | 157.638 ± 34.367 | 193.588 ± 4.801 |
| cpu/elementwise_reduction | `sub` | f64 | 4 | `33554432` | 49.103 ± 3.542 | - | 155.152 ± 2.697 | 49.520 ± 1.299 | 72.942 ± 3.475 | 156.667 ± 39.407 | 80.672 ± 0.448 |
| cpu/elementwise_reduction | `tanh` | f64 | 1 | `8388608` | 55.530 ± 0.138 | - | 57.997 ± 0.080 | 84.974 ± 1.358 | 41.993 ± 2.105 | 88.533 ± 15.116 | 88.266 ± 15.320 |
| cpu/elementwise_reduction | `tanh` | f64 | 4 | `8388608` | 21.224 ± 0.071 | - | 71.998 ± 0.004 | 31.962 ± 1.731 | 23.203 ± 3.386 | 78.790 ± 14.957 | 48.508 ± 15.771 |
| cpu/elementwise_reduction | `where_select` | f64 | 1 | `33554432` | 142.276 ± 7.325 | 142.437 ± 8.064 | unsupported | 169.251 ± 3.801 | 186.055 ± 4.708 | - | - |
| cpu/elementwise_reduction | `where_select` | f64 | 4 | `33554432` | 51.692 ± 2.702 | 167.726 ± 11.864 | unsupported | 57.225 ± 1.129 | 92.885 ± 3.143 | - | - |
| cpu/indexing_layout | `concatenate` | f64 | 1 | `1048576+1048576` | 0.753 ± 0.118 | - | 0.797 ± 0.099 | 0.733 ± 0.094 | 0.749 ± 0.045 | 1.042 ± 7.235 | - |
| cpu/indexing_layout | `concatenate` | f64 | 4 | `1048576+1048576` | 0.716 ± 0.718 | - | 11.996 ± 0.014 | 0.325 ± 0.024 | 0.760 ± 0.061 | 7.874 ± 5.160 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 1 | `4194304 -> 2097152` | 0.879 ± 0.286 | - | 0.758 ± 0.103 | 0.772 ± 0.046 | 0.833 ± 0.098 | 1.082 ± 1.233 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 4 | `4194304 -> 2097152` | 0.489 ± 0.878 | - | 11.998 ± 0.007 | 0.337 ± 0.070 | 0.657 ± 0.048 | 8.487 ± 5.325 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 1 | `2097152` | 1.222 ± 0.031 | - | unsupported | 1.128 ± 0.056 | 1.110 ± 0.052 | 8.854 ± 13.677 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 4 | `2097152` | 0.549 ± 0.135 | - | unsupported | 0.630 ± 0.108 | 0.885 ± 0.070 | 8.994 ± 9.529 | - |
| cpu/indexing_layout | `gather` | f64 | 1 | `262144` | 0.347 ± 0.024 | - | 0.416 ± 0.026 | 0.821 ± 0.015 | 0.424 ± 0.081 | 1.293 ± 0.171 | - |
| cpu/indexing_layout | `gather` | f64 | 4 | `262144` | 0.331 ± 0.032 | - | 47.999 ± 0.006 | 0.406 ± 0.016 | 0.254 ± 0.029 | 1.294 ± 1.045 | - |
| cpu/indexing_layout | `pad` | f64 | 1 | `2097152` | 0.747 ± 0.067 | - | 0.727 ± 0.056 | 1.060 ± 0.049 | 0.967 ± 0.283 | - | - |
| cpu/indexing_layout | `pad` | f64 | 4 | `2097152` | 0.368 ± 0.044 | - | 11.995 ± 0.013 | 0.494 ± 0.087 | 1.155 ± 0.031 | - | - |
| cpu/indexing_layout | `reverse` | f64 | 1 | `2097152` | 0.757 ± 0.093 | - | 0.809 ± 0.141 | 0.886 ± 0.128 | 0.887 ± 0.069 | 1.563 ± 4.816 | - |
| cpu/indexing_layout | `reverse` | f64 | 4 | `2097152` | 0.284 ± 0.069 | - | 12.001 ± 0.019 | 0.356 ± 0.062 | 0.500 ± 0.039 | 8.739 ± 9.594 | - |
| cpu/indexing_layout | `scatter` | f64 | 1 | `262144` | 0.789 ± 0.020 | - | 0.858 ± 0.015 | 0.396 ± 0.016 | 0.374 ± 0.024 | 1.465 ± 0.049 | - |
| cpu/indexing_layout | `scatter` | f64 | 4 | `262144` | 1.103 ± 0.013 | - | 38.978 ± 11.101 | 0.485 ± 0.023 | 0.473 ± 0.151 | 1.465 ± 0.040 | - |
| cpu/indexing_layout | `slice` | f64 | 1 | `4194304 -> 2096128` | 1.513 ± 0.018 | - | 0.002 ± 0.000 | 1.515 ± 0.021 | 1.452 ± 0.132 | 10.993 ± 5.529 | - |
| cpu/indexing_layout | `slice` | f64 | 4 | `4194304 -> 2096128` | 0.774 ± 2.732 | - | 0.002 ± 0.000 | 0.745 ± 0.097 | 1.026 ± 0.064 | 10.173 ± 5.948 | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x2x2` | 0.083 ± 0.001 | - | 0.082 ± 0.003 | 0.075 ± 0.001 | 0.091 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x4x4` | 0.142 ± 0.004 | - | 0.145 ± 0.001 | 0.103 ± 0.001 | 0.152 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x8x8` | 0.273 ± 0.011 | - | 0.272 ± 0.013 | 0.298 ± 0.027 | 0.423 ± 0.460 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x128x128` | 0.052 ± 0.000 | - | 0.057 ± 0.001 | 0.150 ± 0.030 | 0.118 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x32x32` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.059 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x4x4` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x8x8` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x8x8` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x8x8` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.031 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x128x128` | 0.419 ± 0.020 | - | 0.381 ± 0.013 | 1.152 ± 0.017 | 0.632 ± 0.039 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.010 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x32x32` | 0.018 ± 0.000 | - | 0.021 ± 0.001 | 0.040 ± 0.001 | 0.066 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x8x8` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.031 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x2x2` | 0.081 ± 0.002 | - | 0.092 ± 0.001 | 0.089 ± 0.001 | 0.078 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x4x4` | 0.141 ± 0.000 | - | 0.152 ± 0.001 | 0.128 ± 0.001 | 0.158 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x8x8` | 0.270 ± 0.013 | - | 0.285 ± 0.013 | 0.317 ± 0.016 | 0.353 ± 0.030 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x128x128` | 0.054 ± 0.001 | - | 0.053 ± 0.000 | 0.182 ± 0.002 | 0.122 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.009 ± 0.001 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x32x32` | 0.005 ± 0.000 | - | 0.006 ± 0.000 | 0.015 ± 0.000 | 0.012 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.009 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x8x8` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.009 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x8x8` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x2x2` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.009 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x8x8` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x128x128` | 0.198 ± 0.004 | - | 12.000 ± 0.011 | 1.430 ± 0.022 | 1.164 ± 0.155 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x32x32` | 0.019 ± 0.000 | - | 0.021 ± 0.000 | 0.058 ± 0.000 | 0.041 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x8x8` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.011 ± 0.000 | 0.012 ± 0.050 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x2x2` | 0.465 ± 0.020 | - | 0.463 ± 0.017 | 0.685 ± 0.009 | 0.868 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x4x4` | 2.221 ± 0.016 | - | 2.222 ± 0.010 | 2.190 ± 0.009 | 2.677 ± 0.034 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x8x8` | 6.769 ± 0.020 | - | 6.764 ± 0.022 | 6.672 ± 0.023 | 9.189 ± 0.131 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x128x128` | 2.574 ± 0.008 | - | 2.567 ± 0.019 | 3.449 ± 0.013 | 2.778 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x32x32` | 0.110 ± 0.001 | - | 0.111 ± 0.001 | 0.120 ± 0.003 | 0.153 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x8x8` | 0.009 ± 0.000 | - | 0.011 ± 0.000 | 0.014 ± 0.001 | 0.023 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x2x2` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.010 ± 0.001 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x4x4` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.013 ± 0.000 | 0.020 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x8x8` | 0.022 ± 0.000 | - | 0.024 ± 0.000 | 0.027 ± 0.000 | 0.060 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x2x2` | 0.004 ± 0.001 | - | 0.006 ± 0.000 | 0.010 ± 0.000 | 0.017 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x4x4` | 0.011 ± 0.000 | - | 0.013 ± 0.000 | 0.015 ± 0.000 | 0.024 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x8x8` | 0.028 ± 0.000 | - | 0.030 ± 0.000 | 0.033 ± 0.000 | 0.069 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x128x128` | 20.517 ± 0.060 | - | 20.554 ± 0.068 | 27.550 ± 0.230 | 23.859 ± 0.058 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x2x2` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.012 ± 0.000 | 0.018 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x32x32` | 0.863 ± 0.024 | - | 0.870 ± 0.005 | 0.902 ± 0.008 | 0.992 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x4x4` | 0.019 ± 0.000 | - | 0.021 ± 0.001 | 0.024 ± 0.001 | 0.034 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x8x8` | 0.055 ± 0.001 | - | 0.057 ± 0.000 | 0.061 ± 0.001 | 0.103 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x2x2` | 0.462 ± 0.016 | - | 0.462 ± 0.027 | 1.027 ± 0.357 | 0.841 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x4x4` | 0.842 ± 0.011 | - | 11.999 ± 0.006 | 2.166 ± 0.010 | 2.617 ± 0.104 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x8x8` | 2.433 ± 0.107 | - | 11.999 ± 0.005 | 9.545 ± 0.025 | 6.467 ± 0.056 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x128x128` | 2.567 ± 0.033 | - | 2.562 ± 0.052 | 2.444 ± 0.095 | 3.749 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.011 ± 0.000 | 0.015 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x32x32` | 0.111 ± 0.003 | - | 0.114 ± 0.002 | 0.179 ± 0.009 | 0.182 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x4x4` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.009 ± 0.000 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x8x8` | 0.010 ± 0.000 | - | 0.012 ± 0.000 | 0.014 ± 0.001 | 0.023 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.013 ± 0.000 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x4x4` | 0.009 ± 0.000 | - | 0.012 ± 0.000 | 0.013 ± 0.002 | 0.021 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x8x8` | 0.022 ± 0.000 | - | 0.024 ± 0.000 | 0.027 ± 0.001 | 0.081 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x2x2` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.014 ± 0.000 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x4x4` | 0.012 ± 0.000 | - | 0.014 ± 0.000 | 0.015 ± 0.000 | 0.025 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x8x8` | 0.029 ± 0.000 | - | 0.031 ± 0.000 | 0.033 ± 0.001 | 0.058 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x128x128` | 8.069 ± 0.007 | - | 24.001 ± 0.013 | 19.042 ± 0.157 | 29.589 ± 3.442 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x2x2` | 0.006 ± 0.000 | - | 0.009 ± 0.000 | 0.018 ± 0.000 | 0.022 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x32x32` | 0.359 ± 0.028 | - | 12.000 ± 0.010 | 1.344 ± 0.015 | 1.048 ± 0.039 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x4x4` | 0.020 ± 0.000 | - | 0.022 ± 0.000 | 0.024 ± 0.001 | 0.035 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x8x8` | 0.056 ± 0.000 | - | 0.058 ± 0.000 | 0.060 ± 0.001 | 0.112 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x2x2` | 0.444 ± 0.019 | - | 0.441 ± 0.010 | 0.429 ± 0.012 | 0.507 ± 0.501 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x4x4` | 1.989 ± 0.006 | - | 1.992 ± 0.004 | 1.909 ± 0.014 | 1.881 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x8x8` | 6.638 ± 0.057 | - | 6.629 ± 0.027 | 5.626 ± 0.024 | 6.411 ± 0.052 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x128x128` | 1.838 ± 0.008 | - | 1.780 ± 0.006 | 1.404 ± 0.018 | 1.548 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x32x32` | 0.090 ± 0.001 | - | 0.092 ± 0.001 | 0.084 ± 0.003 | 0.124 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x8x8` | 0.009 ± 0.000 | - | 0.011 ± 0.000 | 0.013 ± 0.002 | 0.020 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x2x2` | 0.003 ± 0.003 | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x4x4` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.012 ± 0.000 | 0.017 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x8x8` | 0.022 ± 0.000 | - | 0.024 ± 0.000 | 0.024 ± 0.000 | 0.045 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.008 ± 0.000 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x4x4` | 0.010 ± 0.000 | - | 0.012 ± 0.000 | 0.014 ± 0.000 | 0.020 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x8x8` | 0.028 ± 0.000 | - | 0.030 ± 0.001 | 0.029 ± 0.001 | 0.038 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x128x128` | 14.695 ± 0.025 | - | 14.813 ± 0.066 | 11.021 ± 0.062 | 12.041 ± 0.054 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x2x2` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.010 ± 0.000 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x32x32` | 0.711 ± 0.011 | - | 0.717 ± 0.010 | 0.625 ± 0.010 | 0.759 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x4x4` | 0.017 ± 0.000 | - | 0.019 ± 0.000 | 0.021 ± 0.000 | 0.027 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x8x8` | 0.054 ± 0.000 | - | 0.056 ± 0.000 | 0.051 ± 0.000 | 0.081 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x2x2` | 0.445 ± 0.023 | - | 0.445 ± 0.020 | 0.689 ± 0.012 | 0.350 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x4x4` | 2.063 ± 0.024 | - | 2.058 ± 0.115 | 1.879 ± 0.004 | 1.762 ± 0.538 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x8x8` | 2.268 ± 3.629 | - | 11.996 ± 0.253 | 8.040 ± 0.018 | 3.869 ± 1.634 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x128x128` | 1.852 ± 0.011 | - | 1.814 ± 2.304 | 1.267 ± 0.016 | 1.842 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.010 ± 0.000 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x32x32` | 0.091 ± 0.002 | - | 0.093 ± 0.004 | 0.125 ± 0.007 | 0.130 ± 0.051 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x4x4` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.012 ± 0.001 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x8x8` | 0.010 ± 0.000 | - | 0.012 ± 0.000 | 0.012 ± 0.001 | 0.019 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.011 ± 0.001 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x4x4` | 0.009 ± 0.000 | - | 0.011 ± 0.000 | 0.017 ± 0.000 | 0.022 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x8x8` | 0.022 ± 0.000 | - | 0.025 ± 0.000 | 0.023 ± 0.001 | 0.032 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x2x2` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.012 ± 0.001 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x4x4` | 0.011 ± 0.000 | - | 0.013 ± 0.000 | 0.020 ± 0.000 | 0.021 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x8x8` | 0.029 ± 0.001 | - | 0.031 ± 0.000 | 0.029 ± 0.000 | 0.038 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x128x128` | 5.284 ± 0.019 | - | 24.000 ± 0.008 | 9.955 ± 0.278 | 14.441 ± 1.916 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x2x2` | 0.006 ± 0.000 | - | 0.009 ± 0.000 | 0.015 ± 0.000 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x32x32` | 0.262 ± 0.015 | - | 12.001 ± 0.014 | 0.933 ± 0.003 | 0.463 ± 0.037 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x4x4` | 0.018 ± 0.000 | - | 0.020 ± 0.000 | 0.030 ± 0.000 | 0.029 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x8x8` | 0.056 ± 0.001 | - | 0.058 ± 0.000 | 0.051 ± 0.000 | 0.091 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x2x2` | 0.301 ± 0.010 | - | 0.302 ± 0.010 | 0.217 ± 0.001 | 0.352 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x4x4` | 1.586 ± 0.010 | - | 1.586 ± 0.010 | 0.931 ± 0.010 | 1.857 ± 0.043 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x8x8` | 5.045 ± 0.012 | - | 5.090 ± 0.054 | 2.935 ± 0.011 | 6.375 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x128x128` | 0.948 ± 0.006 | - | 0.945 ± 0.009 | 0.581 ± 0.009 | 1.564 ± 0.046 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.008 ± 0.000 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x32x32` | 0.065 ± 0.001 | - | 0.067 ± 0.000 | 0.040 ± 0.001 | 0.150 ± 0.037 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.009 ± 0.001 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x8x8` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.011 ± 0.000 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.009 ± 0.006 | 0.011 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x4x4` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.011 ± 0.000 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x8x8` | 0.016 ± 0.000 | - | 0.018 ± 0.000 | 0.017 ± 0.001 | 0.029 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.009 ± 0.001 | 0.011 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x4x4` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.011 ± 0.000 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x8x8` | 0.021 ± 0.000 | - | 0.023 ± 0.000 | 0.020 ± 0.000 | 0.034 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x128x128` | 7.485 ± 0.051 | - | 7.486 ± 0.034 | 4.487 ± 0.052 | 11.959 ± 0.102 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.010 ± 0.001 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x32x32` | 0.521 ± 0.019 | - | 0.519 ± 0.011 | 0.251 ± 0.013 | 0.751 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x4x4` | 0.013 ± 0.000 | - | 0.015 ± 0.000 | 0.015 ± 0.000 | 0.023 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x8x8` | 0.041 ± 0.000 | - | 0.044 ± 0.000 | 0.031 ± 0.000 | 0.081 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x2x2` | 0.302 ± 0.028 | - | 0.307 ± 0.021 | 0.363 ± 0.012 | 0.360 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x4x4` | 1.599 ± 0.038 | - | 1.607 ± 2.336 | 0.928 ± 0.005 | 1.258 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x8x8` | 1.682 ± 0.012 | - | 12.004 ± 3.033 | 3.554 ± 0.009 | 3.817 ± 0.072 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x128x128` | 0.968 ± 0.039 | - | 1.004 ± 2.342 | 0.755 ± 0.005 | 1.991 ± 0.176 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.011 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x32x32` | 0.066 ± 0.001 | - | 0.068 ± 0.000 | 0.050 ± 0.003 | 0.129 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.012 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x8x8` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.011 ± 0.001 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.012 ± 0.001 | 0.011 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x4x4` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.015 ± 0.001 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x8x8` | 0.017 ± 0.000 | - | 0.019 ± 0.000 | 0.017 ± 0.000 | 0.028 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x2x2` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.012 ± 0.001 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x4x4` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.016 ± 0.001 | 0.017 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x8x8` | 0.022 ± 0.000 | - | 0.024 ± 0.000 | 0.020 ± 0.000 | 0.035 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x128x128` | 2.501 ± 0.034 | - | 11.996 ± 0.014 | 5.756 ± 0.048 | 14.305 ± 1.768 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x2x2` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.014 ± 0.003 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x32x32` | 0.170 ± 0.005 | - | 11.999 ± 0.009 | 0.315 ± 0.012 | 0.445 ± 0.030 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x4x4` | 0.014 ± 0.000 | - | 0.016 ± 0.000 | 0.020 ± 0.003 | 0.023 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x8x8` | 0.042 ± 0.000 | - | 0.045 ± 0.000 | 0.031 ± 0.000 | 0.084 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x2x2` | 0.120 ± 0.002 | - | 0.123 ± 0.002 | 0.188 ± 0.012 | 0.110 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x4x4` | 0.247 ± 0.013 | - | 0.251 ± 0.012 | 0.263 ± 0.011 | 0.203 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x8x8` | 0.708 ± 0.016 | - | 0.712 ± 0.015 | 0.543 ± 0.014 | 0.583 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x128x128` | 0.129 ± 0.001 | - | 0.133 ± 0.002 | 0.171 ± 0.004 | 0.193 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.011 ± 0.000 | 0.014 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x32x32` | 0.011 ± 0.001 | - | 0.013 ± 0.000 | 0.019 ± 0.001 | 0.044 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.011 ± 0.000 | 0.015 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x64x24` | 0.012 ± 0.000 | - | 0.014 ± 0.000 | 0.021 ± 0.001 | 0.060 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x8x8` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.011 ± 0.000 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x2x2` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.013 ± 0.001 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.013 ± 0.000 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x8x8` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.014 ± 0.000 | 0.048 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.013 ± 0.000 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.013 ± 0.000 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x8x8` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.014 ± 0.000 | 0.049 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x128x128` | 0.986 ± 0.007 | - | 1.018 ± 0.021 | 1.291 ± 0.016 | 1.200 ± 0.126 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.013 ± 0.000 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x32x32` | 0.059 ± 0.000 | - | 0.062 ± 0.001 | 0.078 ± 0.001 | 0.114 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x4x4` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.014 ± 0.000 | 0.036 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x8x8` | 0.009 ± 0.000 | - | 0.011 ± 0.000 | 0.016 ± 0.002 | 0.054 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x2x2` | 0.121 ± 0.001 | - | 0.124 ± 0.002 | 0.159 ± 0.002 | 0.121 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x4x4` | 0.252 ± 0.004 | - | 0.257 ± 0.004 | 0.213 ± 0.016 | 0.215 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x8x8` | 0.295 ± 0.022 | - | 12.000 ± 0.011 | 0.360 ± 0.015 | 0.627 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x128x128` | 0.192 ± 0.016 | - | 24.000 ± 0.003 | 0.220 ± 0.014 | 0.192 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x2x2` | 0.005 ± 0.000 | - | 0.006 ± 0.000 | 0.019 ± 0.001 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x32x32` | 0.012 ± 0.000 | - | 0.019 ± 0.001 | 0.030 ± 0.001 | 0.033 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.019 ± 0.000 | 0.014 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x64x24` | 0.013 ± 0.000 | - | 0.015 ± 0.000 | 0.034 ± 0.000 | 0.035 ± 0.046 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x8x8` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.019 ± 0.000 | 0.014 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x2x2` | 0.005 ± 0.000 | - | 0.006 ± 0.000 | 0.021 ± 0.000 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x4x4` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.021 ± 0.003 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x8x8` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.023 ± 0.000 | 0.020 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x2x2` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.022 ± 0.000 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x4x4` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.022 ± 0.000 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x8x8` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.023 ± 0.000 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x128x128` | 0.476 ± 0.016 | - | 11.999 ± 0.009 | 0.616 ± 0.030 | 0.940 ± 0.080 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x2x2` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.024 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x32x32` | 0.096 ± 0.027 | - | 0.063 ± 0.000 | 0.069 ± 0.002 | 0.131 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x4x4` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.024 ± 0.000 | 0.021 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x8x8` | 0.010 ± 0.000 | - | 0.012 ± 0.000 | 0.026 ± 0.000 | 0.034 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x2x2` | 0.230 ± 0.011 | - | 0.234 ± 0.010 | 0.218 ± 0.001 | 0.284 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x4x4` | 0.677 ± 0.009 | - | 0.687 ± 0.005 | 0.540 ± 0.018 | 0.749 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x8x8` | 2.225 ± 0.018 | - | 2.218 ± 0.006 | 1.350 ± 0.018 | 2.104 ± 0.031 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x128x128` | 0.476 ± 0.013 | - | 0.468 ± 0.035 | 0.578 ± 0.021 | 0.637 ± 0.029 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.001 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x32x32` | 0.031 ± 0.001 | - | 0.033 ± 0.001 | 0.025 ± 0.000 | 0.053 ± 0.285 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x64x24` | 0.026 ± 0.000 | - | 0.029 ± 0.001 | 0.029 ± 0.000 | 0.069 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x8x8` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.036 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x8x8` | 0.009 ± 0.000 | - | 0.011 ± 0.000 | 0.009 ± 0.000 | 0.019 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.014 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x4x4` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.007 ± 0.001 | 0.017 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x8x8` | 0.011 ± 0.000 | - | 0.013 ± 0.000 | 0.010 ± 0.000 | 0.022 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x128x128` | 3.770 ± 0.078 | - | 3.727 ± 0.058 | 4.499 ± 0.061 | 5.220 ± 1.263 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.016 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x32x32` | 0.228 ± 0.011 | - | 0.232 ± 0.002 | 0.164 ± 0.002 | 0.203 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x4x4` | 0.008 ± 0.000 | - | 0.009 ± 0.000 | 0.009 ± 0.000 | 0.018 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x8x8` | 0.019 ± 0.000 | - | 0.022 ± 0.001 | 0.015 ± 0.001 | 0.029 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x2x2` | 0.248 ± 0.002 | - | 0.254 ± 0.011 | 0.317 ± 0.010 | 0.271 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x4x4` | 0.725 ± 0.022 | - | 0.730 ± 7.177 | 0.751 ± 0.012 | 0.777 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x8x8` | 0.831 ± 0.016 | - | 12.002 ± 0.009 | 1.995 ± 0.009 | 1.502 ± 0.147 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x128x128` | 0.474 ± 0.033 | - | 0.485 ± 0.047 | 0.541 ± 0.018 | 1.254 ± 0.763 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.010 ± 0.001 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x32x32` | 0.032 ± 0.001 | - | 0.034 ± 0.001 | 0.042 ± 0.001 | 0.056 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.010 ± 0.000 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x64x24` | 0.028 ± 0.000 | - | 0.030 ± 0.001 | 0.046 ± 0.001 | 0.043 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x8x8` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.011 ± 0.000 | 0.015 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.010 ± 0.001 | 0.015 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x4x4` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.011 ± 0.000 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x8x8` | 0.010 ± 0.000 | - | 0.012 ± 0.000 | 0.015 ± 0.000 | 0.019 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.010 ± 0.000 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x4x4` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.012 ± 0.000 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x8x8` | 0.012 ± 0.000 | - | 0.014 ± 0.000 | 0.016 ± 0.000 | 0.022 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x128x128` | 1.710 ± 2.937 | - | 11.998 ± 0.007 | 4.034 ± 0.034 | 12.872 ± 5.776 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x2x2` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.011 ± 0.000 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x32x32` | 0.105 ± 0.002 | - | 12.000 ± 0.004 | 0.269 ± 0.012 | 0.279 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x4x4` | 0.009 ± 0.000 | - | 0.011 ± 0.000 | 0.014 ± 0.000 | 0.019 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x8x8` | 0.021 ± 0.000 | - | 0.023 ± 0.000 | 0.025 ± 0.002 | 0.031 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.174 ± 0.002 | - | 0.134 ± 0.002 | 0.096 ± 0.001 | 0.249 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.276 ± 0.008 | - | 0.252 ± 0.018 | 0.149 ± 0.002 | 0.366 ± 0.029 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.747 ± 0.011 | - | 0.735 ± 0.017 | 0.437 ± 0.014 | 0.817 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x128x128,rhs=1` | 0.099 ± 0.002 | - | 0.105 ± 0.005 | 0.143 ± 0.006 | 0.176 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.015 ± 0.001 | 0.011 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x32x32,rhs=1` | 0.009 ± 0.000 | - | 0.014 ± 0.001 | 0.023 ± 0.001 | 0.040 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.008 ± 0.000 | 0.015 ± 0.001 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x8x8,rhs=1` | 0.003 ± 0.000 | - | 0.008 ± 0.000 | 0.015 ± 0.001 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.008 ± 0.001 | 0.015 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.008 ± 0.000 | 0.015 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x8x8,rhs=1` | 0.005 ± 0.000 | - | 0.010 ± 0.000 | 0.017 ± 0.001 | 0.015 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.008 ± 0.000 | 0.015 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x4x4,rhs=1` | 0.003 ± 0.001 | - | 0.008 ± 0.000 | 0.015 ± 0.000 | 0.013 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x8x8,rhs=1` | 0.005 ± 0.000 | - | 0.010 ± 0.000 | 0.017 ± 0.000 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x128x128,rhs=1` | 0.763 ± 0.002 | - | 0.772 ± 0.030 | 1.049 ± 0.008 | 1.057 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x2x2,rhs=1` | 0.004 ± 0.000 | - | 0.008 ± 0.000 | 0.015 ± 0.000 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x32x32,rhs=1` | 0.052 ± 0.002 | - | 0.056 ± 0.001 | 0.073 ± 0.001 | 0.113 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x4x4,rhs=1` | 0.005 ± 0.000 | - | 0.009 ± 0.000 | 0.016 ± 0.000 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x8x8,rhs=1` | 0.008 ± 0.000 | - | 0.013 ± 0.000 | 0.019 ± 0.001 | 0.019 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.174 ± 0.001 | - | 0.137 ± 0.001 | 0.126 ± 0.007 | 0.235 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.274 ± 0.027 | - | 0.245 ± 0.005 | 0.173 ± 0.003 | 0.349 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.291 ± 0.014 | - | 23.994 ± 0.011 | 0.397 ± 0.015 | 0.866 ± 0.030 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x128x128,rhs=1` | 0.140 ± 0.006 | - | 23.996 ± 0.006 | 0.190 ± 0.005 | 0.184 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x2x2,rhs=1` | 0.004 ± 0.000 | - | 0.010 ± 0.001 | 0.021 ± 0.001 | 0.011 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x32x32,rhs=1` | 0.009 ± 0.000 | - | 0.022 ± 0.001 | 0.033 ± 0.003 | 0.047 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.010 ± 0.000 | 0.021 ± 0.001 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x8x8,rhs=1` | 0.004 ± 0.000 | - | 0.011 ± 0.000 | 0.021 ± 0.000 | 0.012 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x2x2,rhs=1` | 0.004 ± 0.000 | - | 0.011 ± 0.000 | 0.021 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x4x4,rhs=1` | 0.005 ± 0.000 | - | 0.011 ± 0.000 | 0.022 ± 0.001 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x8x8,rhs=1` | 0.005 ± 0.000 | - | 0.012 ± 0.000 | 0.023 ± 0.001 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x2x2,rhs=1` | 0.004 ± 0.000 | - | 0.010 ± 0.000 | 0.022 ± 0.001 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x4x4,rhs=1` | 0.004 ± 0.000 | - | 0.011 ± 0.000 | 0.022 ± 0.003 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x8x8,rhs=1` | 0.006 ± 0.000 | - | 0.013 ± 0.000 | 0.024 ± 0.000 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x128x128,rhs=1` | 0.351 ± 0.014 | - | 36.000 ± 0.005 | 0.577 ± 0.018 | 0.745 ± 0.030 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x2x2,rhs=1` | 0.004 ± 0.000 | - | 0.011 ± 0.000 | 0.024 ± 0.003 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x32x32,rhs=1` | 0.076 ± 0.005 | - | 0.060 ± 0.000 | 0.062 ± 0.006 | 0.171 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x4x4,rhs=1` | 0.005 ± 0.000 | - | 0.012 ± 0.000 | 0.025 ± 0.000 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x8x8,rhs=1` | 0.009 ± 0.000 | - | 0.016 ± 0.000 | 0.028 ± 0.001 | 0.019 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x2x2` | 1.172 ± 0.013 | - | 1.172 ± 0.003 | 1.526 ± 0.014 | 2.011 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x4x4` | 3.668 ± 0.007 | - | 3.693 ± 0.026 | 5.115 ± 0.007 | 5.344 ± 0.335 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x8x8` | 11.363 ± 0.012 | - | 11.364 ± 0.033 | 15.503 ± 0.074 | 15.005 ± 0.090 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x128x128` | 4.544 ± 0.015 | - | 4.576 ± 0.049 | 6.026 ± 0.017 | 4.507 ± 0.067 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.010 ± 0.000 | 0.018 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x32x32` | 0.195 ± 0.002 | - | 0.198 ± 0.012 | 0.230 ± 0.018 | 0.239 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x4x4` | 0.006 ± 0.000 | - | 0.009 ± 0.000 | 0.014 ± 0.001 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x64x24` | 0.159 ± 0.003 | - | 0.162 ± 0.003 | 0.206 ± 0.004 | 0.247 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x8x8` | 0.014 ± 0.000 | - | 0.016 ± 0.000 | 0.026 ± 0.001 | 0.031 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x2x2` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.014 ± 0.001 | 0.021 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x4x4` | 0.013 ± 0.000 | - | 0.015 ± 0.000 | 0.024 ± 0.001 | 0.030 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x8x8` | 0.035 ± 0.001 | - | 0.037 ± 0.000 | 0.057 ± 0.002 | 0.062 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x2x2` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.014 ± 0.001 | 0.023 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x4x4` | 0.017 ± 0.000 | - | 0.019 ± 0.000 | 0.029 ± 0.001 | 0.035 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x8x8` | 0.046 ± 0.000 | - | 0.048 ± 0.000 | 0.073 ± 0.001 | 0.079 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x128x128` | 36.453 ± 0.228 | - | 36.325 ± 0.121 | 47.003 ± 0.194 | 35.536 ± 0.075 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x2x2` | 0.012 ± 0.000 | - | 0.014 ± 0.000 | 0.020 ± 0.000 | 0.031 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x32x32` | 1.496 ± 0.016 | - | 1.496 ± 0.011 | 1.718 ± 0.016 | 1.592 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x4x4` | 0.031 ± 0.000 | - | 0.034 ± 0.000 | 0.050 ± 0.002 | 0.056 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x8x8` | 0.089 ± 0.000 | - | 0.093 ± 0.002 | 0.135 ± 0.026 | 0.154 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x2x2` | 1.183 ± 0.025 | - | 1.186 ± 2.377 | 1.558 ± 0.038 | 2.008 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x4x4` | 1.376 ± 0.072 | - | 11.998 ± 0.004 | 5.072 ± 0.027 | 2.583 ± 0.444 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x8x8` | 3.954 ± 0.011 | - | 12.001 ± 0.014 | 18.605 ± 0.069 | 8.835 ± 1.047 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x128x128` | 4.524 ± 0.053 | - | 4.521 ± 0.014 | 5.185 ± 0.098 | 6.989 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x2x2` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.013 ± 0.001 | 0.019 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x32x32` | 0.192 ± 0.013 | - | 0.196 ± 0.021 | 0.345 ± 0.014 | 0.262 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x4x4` | 0.007 ± 0.000 | - | 0.010 ± 0.000 | 0.014 ± 0.001 | 0.020 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x64x24` | 0.156 ± 0.004 | - | 0.236 ± 0.012 | 0.300 ± 0.014 | 0.205 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x8x8` | 0.015 ± 0.000 | - | 0.017 ± 0.000 | 0.025 ± 0.002 | 0.047 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x2x2` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.018 ± 0.001 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x4x4` | 0.014 ± 0.000 | - | 0.017 ± 0.000 | 0.023 ± 0.000 | 0.030 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x8x8` | 0.036 ± 0.000 | - | 0.039 ± 0.001 | 0.056 ± 0.000 | 0.062 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x2x2` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.020 ± 0.003 | 0.024 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x4x4` | 0.018 ± 0.000 | - | 0.020 ± 0.000 | 0.029 ± 0.000 | 0.036 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x8x8` | 0.047 ± 0.001 | - | 0.052 ± 0.001 | 0.072 ± 0.001 | 0.077 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x128x128` | 15.451 ± 0.067 | - | 47.998 ± 0.005 | 39.167 ± 1.523 | 92.284 ± 31.456 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x2x2` | 0.012 ± 0.000 | - | 0.015 ± 0.000 | 0.029 ± 0.003 | 0.031 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x32x32` | 0.614 ± 0.034 | - | 12.001 ± 0.004 | 2.545 ± 0.023 | 0.813 ± 0.067 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x4x4` | 0.033 ± 0.000 | - | 0.035 ± 0.001 | 0.050 ± 0.002 | 0.057 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x8x8` | 0.090 ± 0.000 | - | 0.093 ± 0.001 | 0.132 ± 0.001 | 0.169 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x2x2` | 0.965 ± 0.003 | - | 0.967 ± 0.003 | 1.066 ± 0.006 | 1.207 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x4x4` | 3.261 ± 0.048 | - | 3.261 ± 0.009 | 4.333 ± 0.079 | 3.835 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x8x8` | 10.935 ± 0.037 | - | 10.959 ± 0.048 | 13.420 ± 0.196 | 12.327 ± 0.251 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x128x128` | 2.588 ± 0.024 | - | 2.618 ± 0.020 | 2.570 ± 0.017 | 2.574 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x32x32` | 0.142 ± 0.006 | - | 0.146 ± 0.004 | 0.158 ± 0.017 | 0.210 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x4x4` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.013 ± 0.001 | 0.019 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x64x24` | 0.098 ± 0.004 | - | 0.102 ± 0.002 | 0.126 ± 0.005 | 0.205 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x8x8` | 0.013 ± 0.000 | - | 0.015 ± 0.000 | 0.023 ± 0.001 | 0.028 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x2x2` | 0.006 ± 0.000 | - | 0.008 ± 0.000 | 0.012 ± 0.001 | 0.018 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x4x4` | 0.013 ± 0.001 | - | 0.015 ± 0.000 | 0.021 ± 0.000 | 0.027 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x8x8` | 0.034 ± 0.000 | - | 0.036 ± 0.000 | 0.047 ± 0.001 | 0.049 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x2x2` | 0.006 ± 0.000 | - | 0.009 ± 0.000 | 0.013 ± 0.000 | 0.020 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x4x4` | 0.016 ± 0.000 | - | 0.018 ± 0.000 | 0.025 ± 0.000 | 0.031 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x8x8` | 0.044 ± 0.000 | - | 0.047 ± 0.001 | 0.060 ± 0.001 | 0.061 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x128x128` | 20.585 ± 0.135 | - | 20.968 ± 0.147 | 20.305 ± 0.105 | 19.660 ± 0.084 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x2x2` | 0.010 ± 0.000 | - | 0.012 ± 0.000 | 0.017 ± 0.000 | 0.023 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x32x32` | 1.123 ± 0.080 | - | 1.124 ± 0.007 | 1.177 ± 0.010 | 1.216 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x4x4` | 0.028 ± 0.000 | - | 0.031 ± 0.000 | 0.043 ± 0.002 | 0.047 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x8x8` | 0.087 ± 0.000 | - | 0.090 ± 0.001 | 0.111 ± 0.000 | 0.125 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x2x2` | 0.987 ± 0.019 | - | 0.985 ± 2.318 | 1.022 ± 0.500 | 1.181 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x4x4` | 3.332 ± 0.017 | - | 3.393 ± 0.020 | 4.267 ± 0.016 | 2.000 ± 0.379 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x8x8` | 4.337 ± 0.017 | - | 11.999 ± 0.014 | 16.725 ± 1.415 | 6.328 ± 0.080 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x128x128` | 2.601 ± 0.023 | - | 2.596 ± 0.024 | 3.015 ± 0.022 | 3.449 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x2x2` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.013 ± 0.000 | 0.020 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x32x32` | 0.145 ± 0.005 | - | 0.149 ± 0.003 | 0.245 ± 0.018 | 0.174 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x4x4` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.013 ± 0.001 | 0.020 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x64x24` | 0.100 ± 0.001 | - | 0.104 ± 0.003 | 0.190 ± 0.019 | 0.170 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x8x8` | 0.014 ± 0.000 | - | 0.016 ± 0.000 | 0.022 ± 0.000 | 0.027 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x2x2` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.016 ± 0.001 | 0.019 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x4x4` | 0.014 ± 0.000 | - | 0.016 ± 0.001 | 0.021 ± 0.001 | 0.028 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x8x8` | 0.035 ± 0.000 | - | 0.038 ± 0.000 | 0.047 ± 0.001 | 0.051 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x2x2` | 0.008 ± 0.000 | - | 0.010 ± 0.000 | 0.018 ± 0.001 | 0.019 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x4x4` | 0.017 ± 0.000 | - | 0.019 ± 0.000 | 0.025 ± 0.000 | 0.037 ± 0.030 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x8x8` | 0.046 ± 0.001 | - | 0.048 ± 0.001 | 0.059 ± 0.000 | 0.064 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x128x128` | 8.333 ± 0.048 | - | 23.997 ± 0.030 | 23.304 ± 0.169 | 27.740 ± 6.219 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x2x2` | 0.011 ± 0.000 | - | 0.013 ± 0.000 | 0.023 ± 0.003 | 0.024 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x32x32` | 0.423 ± 0.007 | - | 12.001 ± 0.008 | 1.807 ± 0.010 | 0.770 ± 0.204 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x4x4` | 0.030 ± 0.000 | - | 0.032 ± 0.000 | 0.042 ± 0.001 | 0.046 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x8x8` | 0.088 ± 0.000 | - | 0.091 ± 0.001 | 0.111 ± 0.002 | 0.181 ± 0.119 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x2x2` | 0.545 ± 0.012 | - | unsupported | 0.337 ± 0.018 | 0.711 ± 0.028 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x4x4` | 1.970 ± 0.004 | - | unsupported | 1.825 ± 0.034 | 2.529 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x8x8` | 5.582 ± 0.028 | - | unsupported | 5.075 ± 0.054 | 6.547 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x128x128` | 1.267 ± 0.014 | - | unsupported | 1.187 ± 0.019 | 1.167 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | unsupported | 0.008 ± 0.000 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x32x32` | 0.060 ± 0.001 | - | unsupported | 0.062 ± 0.001 | 0.069 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x4x4` | 0.003 ± 0.000 | - | unsupported | 0.016 ± 0.007 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x64x24` | 0.049 ± 0.000 | - | unsupported | 0.052 ± 0.004 | 0.062 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x8x8` | 0.007 ± 0.000 | - | unsupported | 0.012 ± 0.001 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x2x2` | 0.003 ± 0.000 | - | unsupported | 0.009 ± 0.001 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x4x4` | 0.007 ± 0.000 | - | unsupported | 0.013 ± 0.000 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x8x8` | 0.018 ± 0.000 | - | unsupported | 0.023 ± 0.001 | 0.029 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x2x2` | 0.004 ± 0.000 | - | unsupported | 0.009 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x4x4` | 0.009 ± 0.000 | - | unsupported | 0.014 ± 0.000 | 0.022 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x8x8` | 0.023 ± 0.000 | - | unsupported | 0.028 ± 0.001 | 0.036 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x128x128` | 10.234 ± 0.037 | - | unsupported | 9.329 ± 0.063 | 9.098 ± 0.043 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x2x2` | 0.006 ± 0.000 | - | unsupported | 0.010 ± 0.000 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x32x32` | 0.489 ± 0.010 | - | unsupported | 0.422 ± 0.017 | 0.491 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x4x4` | 0.016 ± 0.000 | - | unsupported | 0.021 ± 0.001 | 0.029 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x8x8` | 0.045 ± 0.001 | - | unsupported | 0.048 ± 0.001 | 0.061 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x2x2` | 0.559 ± 0.031 | - | unsupported | 0.514 ± 0.021 | 0.871 ± 0.283 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x4x4` | 2.009 ± 0.012 | - | unsupported | 1.809 ± 0.012 | 1.334 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x8x8` | 1.899 ± 0.010 | - | unsupported | 7.226 ± 0.006 | 3.814 ± 0.442 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x128x128` | 1.526 ± 0.256 | - | unsupported | 1.523 ± 0.018 | 1.789 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x2x2` | 0.004 ± 0.000 | - | unsupported | 0.010 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x32x32` | 0.079 ± 0.005 | - | unsupported | 0.083 ± 0.005 | 0.070 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x4x4` | 0.004 ± 0.000 | - | unsupported | 0.013 ± 0.001 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x64x24` | 0.066 ± 0.003 | - | unsupported | 0.070 ± 0.003 | 0.060 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x8x8` | 0.008 ± 0.000 | - | unsupported | 0.012 ± 0.000 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x2x2` | 0.006 ± 0.000 | - | unsupported | 0.012 ± 0.000 | 0.013 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x4x4` | 0.008 ± 0.000 | - | unsupported | 0.018 ± 0.001 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x8x8` | 0.019 ± 0.000 | - | unsupported | 0.022 ± 0.000 | 0.029 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x2x2` | 0.007 ± 0.000 | - | unsupported | 0.012 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x4x4` | 0.010 ± 0.000 | - | unsupported | 0.021 ± 0.000 | 0.020 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x8x8` | 0.024 ± 0.000 | - | unsupported | 0.028 ± 0.001 | 0.037 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x128x128` | 3.399 ± 0.006 | - | unsupported | 11.831 ± 0.033 | 18.435 ± 8.325 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x2x2` | 0.010 ± 0.000 | - | unsupported | 0.015 ± 0.001 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x32x32` | 0.161 ± 0.005 | - | unsupported | 0.423 ± 0.024 | 0.266 ± 0.115 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x4x4` | 0.017 ± 0.000 | - | unsupported | 0.031 ± 0.003 | 0.029 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x8x8` | 0.046 ± 0.000 | - | unsupported | 0.048 ± 0.001 | 0.061 ± 0.004 | - | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x16x16,rhs=1` | 1.360 ± 0.016 | - | unsupported | 1.012 ± 0.030 | 3.576 ± 0.077 | 2.620 ± 0.170 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x2x2,rhs=1` | 0.109 ± 0.002 | - | unsupported | 0.059 ± 0.001 | 0.096 ± 0.003 | 0.132 ± 0.004 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x4x4,rhs=1` | 0.190 ± 0.002 | - | unsupported | 0.090 ± 0.001 | 0.171 ± 0.003 | 0.215 ± 0.013 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x8x8,rhs=1` | 0.503 ± 0.018 | - | unsupported | 0.263 ± 0.013 | 0.475 ± 0.015 | 0.459 ± 0.361 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x16x16,rhs=1` | 0.594 ± 0.042 | - | unsupported | 0.501 ± 0.013 | 1.096 ± 0.012 | 2.758 ± 0.135 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x2x2,rhs=1` | 0.162 ± 0.001 | - | unsupported | 0.061 ± 0.006 | 0.091 ± 0.004 | 0.105 ± 0.015 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x4x4,rhs=1` | 0.291 ± 0.023 | - | unsupported | 0.075 ± 0.001 | 0.162 ± 0.019 | 0.174 ± 0.005 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x8x8,rhs=1` | 0.201 ± 0.008 | - | unsupported | 0.151 ± 0.003 | 0.546 ± 0.031 | 0.751 ± 0.069 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.600 ± 0.040 | - | unsupported | 0.388 ± 0.012 | 0.940 ± 0.304 | 0.457 ± 0.012 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.053 ± 0.000 | - | unsupported | 0.043 ± 0.001 | 0.188 ± 0.043 | 0.072 ± 0.000 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.066 ± 0.001 | - | unsupported | 0.064 ± 0.000 | 0.233 ± 0.023 | 0.121 ± 0.002 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.221 ± 0.012 | - | unsupported | 0.178 ± 0.002 | 0.402 ± 0.021 | 0.223 ± 0.012 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.315 ± 0.020 | - | unsupported | 0.565 ± 0.009 | 1.033 ± 0.388 | 0.458 ± 0.019 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.090 ± 0.002 | - | unsupported | 0.071 ± 0.003 | 0.163 ± 0.004 | 0.072 ± 0.000 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.116 ± 0.002 | - | unsupported | 0.098 ± 0.003 | 0.211 ± 0.025 | 0.120 ± 0.002 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.334 ± 0.011 | - | unsupported | 0.233 ± 0.011 | 0.383 ± 0.022 | 0.222 ± 0.014 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.328 ± 0.011 | - | 0.327 ± 0.012 | 0.207 ± 0.014 | 0.419 ± 0.022 | 0.264 ± 0.009 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.037 ± 0.000 | - | 0.040 ± 0.000 | 0.073 ± 0.001 | 0.112 ± 0.024 | 0.055 ± 0.001 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.045 ± 0.001 | - | 0.048 ± 0.000 | 0.088 ± 0.001 | 0.093 ± 0.004 | 0.085 ± 0.000 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.132 ± 0.002 | - | 0.133 ± 0.001 | 0.124 ± 0.001 | 0.175 ± 0.013 | 0.143 ± 0.005 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.130 ± 0.002 | - | 11.997 ± 0.005 | 0.267 ± 0.014 | 0.577 ± 0.169 | 0.263 ± 0.011 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.063 ± 0.002 | - | 0.042 ± 0.000 | 0.123 ± 0.003 | 0.076 ± 0.002 | 0.056 ± 0.001 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.046 ± 0.000 | - | 0.049 ± 0.000 | 0.140 ± 0.011 | 0.095 ± 0.002 | 0.084 ± 0.001 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.130 ± 0.001 | - | 0.132 ± 0.001 | 0.182 ± 0.003 | 0.204 ± 0.026 | 0.143 ± 0.001 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 1 | `1536x1536` | 48.090 ± 0.379 | - | 50.502 ± 0.764 | 68.704 ± 0.704 | 61.331 ± 1.144 | 51.069 ± 18.423 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 4 | `1536x1536` | 22.182 ± 0.286 | - | 276.004 ± 12.038 | 24.614 ± 0.149 | 47.140 ± 2.213 | 43.820 ± 8.883 | - |
| cpu/linalg_uncovered | `det` | f64 | 1 | `1024x1024` | 23.854 ± 0.123 | - | 22.377 ± 0.135 | 27.146 ± 0.255 | 24.974 ± 0.061 | 22.491 ± 6.241 | - |
| cpu/linalg_uncovered | `det` | f64 | 4 | `1024x1024` | 17.109 ± 0.076 | - | 935.988 ± 1.246 | 12.002 ± 0.216 | 14.691 ± 0.833 | 12.873 ± 1.948 | - |
| cpu/linalg_uncovered | `eig` | f64 | 1 | `160x160` | 7.711 ± 0.073 | - | 7.726 ± 0.033 | 8.302 ± 0.027 | 8.330 ± 0.046 | 8.631 ± 0.079 | - |
| cpu/linalg_uncovered | `eig` | f64 | 4 | `160x160` | 10.228 ± 0.070 | - | 794.004 ± 9.981 | 11.584 ± 0.147 | 11.999 ± 0.032 | 11.823 ± 0.251 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 1 | `192x192` | 9.489 ± 0.038 | - | 9.470 ± 0.029 | 9.040 ± 0.029 | 9.799 ± 0.211 | 9.497 ± 0.044 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 4 | `192x192` | 12.990 ± 0.079 | - | 1157.994 ± 0.019 | 12.383 ± 0.081 | 14.177 ± 0.076 | 13.533 ± 0.083 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 1 | `512x512` | 18.792 ± 0.071 | - | 18.923 ± 0.072 | 19.619 ± 0.113 | 43.340 ± 0.232 | 13.661 ± 1.079 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 4 | `512x512` | 18.835 ± 0.089 | - | 18.958 ± 0.071 | 15.923 ± 0.262 | 34.443 ± 0.418 | 11.949 ± 0.357 | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `256x128` | 0.348 ± 0.006 | 0.362 ± 0.007 | unsupported | 0.628 ± 0.012 | 0.736 ± 0.024 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `768x384` | 6.608 ± 0.089 | 6.714 ± 0.013 | unsupported | 9.024 ± 0.031 | 10.243 ± 0.099 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `256x128` | 0.346 ± 0.006 | 71.998 ± 0.002 | unsupported | 0.515 ± 0.003 | 1.810 ± 1.543 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `768x384` | 4.067 ± 0.098 | 533.999 ± 17.953 | unsupported | 5.809 ± 0.017 | 11.592 ± 0.678 | - | - |
| cpu/linalg_uncovered | `inv` | f64 | 1 | `768x768` | 34.539 ± 0.194 | - | 34.758 ± 0.369 | 46.461 ± 0.377 | 35.161 ± 0.505 | 29.208 ± 2.522 | - |
| cpu/linalg_uncovered | `inv` | f64 | 4 | `768x768` | 19.948 ± 0.528 | - | 970.003 ± 12.007 | 22.601 ± 0.243 | 19.924 ± 1.650 | 17.446 ± 1.113 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 1 | `768x384,rhs=16` | - | 16.546 ± 0.078 | 16.509 ± 0.062 | 9.464 ± 0.172 | 59.299 ± 0.453 | 10.564 ± 0.134 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 4 | `768x384,rhs=16` | - | 10.299 ± 0.055 | 586.000 ± 0.025 | 6.361 ± 0.080 | 59.889 ± 1.431 | 9.558 ± 0.046 | - |
| cpu/linalg_uncovered | `lu` | f64 | 1 | `1024x1024` | 23.891 ± 0.251 | - | 23.807 ± 0.316 | 34.885 ± 0.515 | 32.298 ± 0.095 | 28.305 ± 16.139 | - |
| cpu/linalg_uncovered | `lu` | f64 | 4 | `1024x1024` | 17.178 ± 3.944 | - | 871.901 ± 678.983 | 16.872 ± 0.199 | 24.137 ± 10.014 | 25.246 ± 13.908 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 1 | `2048x2048` | 0.814 ± 0.019 | - | 0.960 ± 0.168 | 1.171 ± 0.032 | 22.138 ± 0.528 | 2.973 ± 0.011 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 4 | `2048x2048` | 0.633 ± 0.191 | - | 24.000 ± 0.017 | 1.211 ± 0.012 | 15.236 ± 1.192 | 2.972 ± 0.033 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 1 | `512x256` | 17.128 ± 0.090 | - | 17.938 ± 0.395 | 21.692 ± 0.117 | 21.295 ± 0.151 | 21.021 ± 0.176 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 4 | `512x256` | 19.442 ± 0.121 | - | 59.998 ± 0.012 | 17.395 ± 0.127 | 23.426 ± 0.708 | 22.596 ± 2.371 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 1 | `512x256` | 16.952 ± 0.094 | - | 17.602 ± 0.079 | 21.817 ± 0.232 | 21.347 ± 0.108 | 21.029 ± 0.276 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 4 | `512x256` | 19.421 ± 0.123 | - | 59.997 ± 0.008 | 17.371 ± 0.148 | 24.096 ± 0.614 | 22.444 ± 0.513 | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `256x128` | 1.019 ± 0.017 | 1.131 ± 0.008 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `768x384` | 21.098 ± 1.670 | 21.135 ± 0.083 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `256x128` | 1.017 ± 0.018 | 1.121 ± 0.013 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `768x384` | 13.581 ± 0.765 | 4310.372 ± 148.029 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `256x128` | 1.028 ± 0.016 | 0.997 ± 0.004 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `768x384` | 21.974 ± 0.355 | 22.732 ± 0.084 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `256x128` | 1.015 ± 0.014 | 0.997 ± 0.007 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `768x384` | 13.628 ± 1.386 | 4336.933 ± 39.143 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 1 | `1024x1024` | 24.043 ± 0.245 | - | 22.188 ± 0.594 | 27.133 ± 0.394 | 25.042 ± 0.103 | 22.409 ± 0.460 | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 4 | `1024x1024` | 17.210 ± 0.125 | - | 936.004 ± 645.540 | 12.055 ± 0.902 | 14.534 ± 0.148 | 12.944 ± 0.266 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 1 | `768x384` | - | 56.603 ± 0.806 | 55.122 ± 0.270 | 64.399 ± 0.501 | 64.496 ± 0.231 | 64.914 ± 0.972 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 4 | `768x384` | - | 55.734 ± 0.195 | 55.333 ± 0.387 | 40.857 ± 0.449 | 58.311 ± 1.829 | 56.829 ± 4.311 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 1 | `4096x4096,rhs=64` | 38.883 ± 0.443 | - | 40.102 ± 0.610 | 32.179 ± 0.253 | 134.635 ± 3.116 | 33.128 ± 0.480 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 4 | `4096x4096,rhs=64` | 20.100 ± 0.166 | - | 839.999 ± 135.010 | 24.305 ± 0.595 | 118.970 ± 4.037 | 21.694 ± 9.000 | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `1024x64` | 0.149 ± 0.009 | 0.149 ± 0.001 | unsupported | 0.165 ± 0.027 | 0.147 ± 0.012 | - | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `4096x256` | 2.216 ± 0.005 | 2.376 ± 0.008 | unsupported | 2.551 ± 0.020 | 0.424 ± 0.759 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `1024x64` | 0.073 ± 0.000 | 6.000 ± 0.031 | unsupported | 0.080 ± 0.013 | 0.137 ± 0.032 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `4096x256` | 0.913 ± 0.032 | 23.999 ± 0.001 | unsupported | 1.069 ± 0.046 | 0.818 ± 0.314 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `1024x64` | 0.084 ± 0.001 | 0.084 ± 0.000 | unsupported | 0.145 ± 0.003 | 0.266 ± 0.018 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `4096x256` | 1.356 ± 0.003 | 1.353 ± 0.005 | unsupported | 1.900 ± 0.013 | 2.911 ± 0.073 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `1024x64` | 0.038 ± 0.001 | 12.000 ± 0.000 | unsupported | 0.093 ± 0.030 | 0.162 ± 0.064 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `4096x256` | 0.553 ± 0.001 | 11.999 ± 0.001 | unsupported | 0.773 ± 0.089 | 1.186 ± 0.146 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `1024x64` | 0.211 ± 0.001 | 0.212 ± 0.001 | unsupported | 0.279 ± 0.001 | 0.214 ± 0.005 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `4096x256` | 3.373 ± 0.008 | 3.375 ± 0.016 | unsupported | 4.320 ± 0.002 | 1.994 ± 0.044 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `1024x64` | 0.083 ± 0.001 | 11.042 ± 10.068 | unsupported | 0.112 ± 0.003 | 0.140 ± 0.008 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `4096x256` | 1.293 ± 0.006 | 11.999 ± 0.000 | unsupported | 1.617 ± 0.059 | 1.048 ± 0.038 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `1024x64` | 0.056 ± 0.005 | 0.085 ± 0.009 | unsupported | 0.167 ± 0.002 | 0.189 ± 0.005 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `4096x256` | 0.745 ± 0.067 | 0.729 ± 0.060 | unsupported | 1.782 ± 0.027 | 1.862 ± 0.106 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `1024x64` | 0.096 ± 0.001 | 35.984 ± 0.141 | unsupported | 0.070 ± 0.003 | 0.267 ± 0.044 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `4096x256` | 0.404 ± 0.029 | 96.001 ± 2.999 | unsupported | 0.690 ± 0.026 | 1.413 ± 0.489 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `1024x64` | 0.130 ± 0.009 | 0.149 ± 0.001 | unsupported | 0.133 ± 0.001 | 0.243 ± 0.015 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `4096x256` | 1.878 ± 0.018 | 1.892 ± 0.051 | unsupported | 2.122 ± 0.025 | 2.296 ± 0.038 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `1024x64` | 0.155 ± 0.004 | 44.859 ± 0.282 | unsupported | 0.063 ± 0.001 | 0.222 ± 0.005 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `4096x256` | 0.917 ± 0.019 | 119.998 ± 0.125 | unsupported | 0.786 ± 0.047 | 1.316 ± 0.170 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `1024x64` | 0.329 ± 0.005 | 0.359 ± 0.002 | unsupported | unsupported | 0.259 ± 0.010 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `4096x256` | 4.985 ± 0.152 | 5.035 ± 0.116 | unsupported | unsupported | 3.367 ± 1.140 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `1024x64` | 0.319 ± 0.025 | 53.796 ± 3.462 | unsupported | unsupported | 0.265 ± 0.003 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `4096x256` | 2.027 ± 0.033 | 143.999 ± 0.001 | unsupported | unsupported | 1.925 ± 0.130 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `1024x64` | 0.012 ± 0.000 | 0.017 ± 0.000 | unsupported | 0.012 ± 0.000 | 0.054 ± 0.009 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `4096x256` | 0.132 ± 0.001 | 0.115 ± 0.001 | unsupported | 0.077 ± 0.002 | 0.252 ± 0.001 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `1024x64` | 0.016 ± 0.000 | 5.999 ± 0.031 | unsupported | 0.014 ± 0.000 | 0.035 ± 0.009 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `4096x256` | 0.063 ± 0.005 | 23.999 ± 0.007 | unsupported | 0.074 ± 0.038 | 0.229 ± 0.006 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `1024x64` | 0.030 ± 0.006 | 0.048 ± 0.007 | unsupported | 0.096 ± 0.004 | 0.179 ± 0.011 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `4096x256` | 0.401 ± 0.004 | 0.381 ± 0.021 | unsupported | 1.399 ± 0.015 | 1.822 ± 0.164 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `1024x64` | 0.046 ± 0.001 | 17.984 ± 0.141 | unsupported | 0.110 ± 0.002 | 0.150 ± 0.002 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `4096x256` | 0.221 ± 0.020 | 47.999 ± 0.002 | unsupported | 0.567 ± 0.016 | 1.132 ± 0.059 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `1024x64` | 0.059 ± 0.001 | 0.059 ± 0.001 | unsupported | 0.117 ± 0.007 | 0.202 ± 0.004 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `4096x256` | 0.939 ± 0.003 | 0.949 ± 0.015 | unsupported | 1.803 ± 0.005 | 1.832 ± 0.029 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `1024x64` | 0.026 ± 0.000 | 12.000 ± 0.000 | unsupported | 0.095 ± 0.003 | 0.126 ± 0.009 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `4096x256` | 0.427 ± 0.114 | 11.999 ± 0.000 | unsupported | 0.709 ± 0.034 | 0.975 ± 0.028 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `1024x64` | 0.061 ± 0.000 | 0.061 ± 0.000 | unsupported | 0.122 ± 0.002 | 0.202 ± 0.011 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `4096x256` | 0.977 ± 0.003 | 0.992 ± 0.002 | unsupported | 1.846 ± 0.011 | 1.892 ± 0.081 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `1024x64` | 0.027 ± 0.000 | 12.000 ± 0.000 | unsupported | 0.099 ± 0.003 | 0.131 ± 0.003 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `4096x256` | 0.400 ± 0.009 | 11.999 ± 0.002 | unsupported | 0.760 ± 0.024 | 0.979 ± 0.034 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `1024x64` | 0.130 ± 0.001 | 0.153 ± 0.004 | unsupported | 0.138 ± 0.002 | 0.247 ± 0.046 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `4096x256` | 1.890 ± 0.040 | 1.917 ± 0.070 | unsupported | 1.280 ± 0.004 | 2.415 ± 0.178 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `1024x64` | 0.153 ± 0.001 | 44.859 ± 0.140 | unsupported | 0.063 ± 0.001 | 0.208 ± 0.002 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `4096x256` | 0.916 ± 0.011 | 120.000 ± 4.501 | unsupported | 0.478 ± 0.017 | 1.706 ± 0.671 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `1024x64` | 0.197 ± 0.001 | 0.198 ± 0.001 | unsupported | 0.264 ± 0.002 | 0.372 ± 0.002 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `4096x256` | 3.156 ± 0.006 | 3.154 ± 0.005 | unsupported | 4.281 ± 0.022 | 4.565 ± 0.037 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `1024x64` | 0.077 ± 0.001 | 12.000 ± 0.015 | unsupported | 0.205 ± 0.004 | 0.224 ± 0.026 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `4096x256` | 1.193 ± 0.794 | 11.999 ± 0.001 | unsupported | 1.581 ± 0.221 | 2.410 ± 0.219 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `1024x64` | 0.677 ± 0.002 | 1.440 ± 0.012 | unsupported | 0.290 ± 0.005 | 0.233 ± 0.038 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `4096x256` | 12.924 ± 0.064 | 23.249 ± 0.813 | unsupported | 6.266 ± 0.046 | 2.570 ± 0.204 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `1024x64` | 0.807 ± 0.017 | 96.015 ± 0.797 | unsupported | 0.291 ± 0.032 | 0.159 ± 0.005 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `4096x256` | 8.019 ± 0.212 | 144.002 ± 18.027 | unsupported | 2.508 ± 0.025 | 1.386 ± 0.313 | - | - |
| cpu/output_reuse | `add_into` | f64 | 1 | `33554432` | 22.849 ± 0.350 | - | unsupported | 22.408 ± 0.167 | - | 23.155 ± 0.914 | 36.277 ± 0.528 |
| cpu/output_reuse | `add_into` | f64 | 4 | `33554432` | 21.015 ± 0.220 | - | unsupported | 20.852 ± 0.385 | - | 23.561 ± 1.291 | 20.715 ± 0.241 |
| cpu/output_reuse | `conj_into` | c64 | 1 | `16777216` | 18.038 ± 0.281 | - | unsupported | 15.932 ± 1.079 | - | 17.550 ± 0.389 | 18.298 ± 0.225 |
| cpu/output_reuse | `conj_into` | c64 | 4 | `16777216` | 14.350 ± 0.270 | - | unsupported | 14.755 ± 0.260 | - | 17.258 ± 0.483 | 14.144 ± 0.076 |
| cpu/output_reuse | `copy_read_into` | f64 | 1 | `33554432` | 19.987 ± 0.719 | - | unsupported | 15.990 ± 1.160 | - | 20.349 ± 0.323 | - |
| cpu/output_reuse | `copy_read_into` | f64 | 4 | `33554432` | 14.541 ± 0.342 | - | unsupported | 15.751 ± 3.527 | - | 19.609 ± 0.082 | - |
| cpu/output_reuse | `div_into` | f64 | 1 | `33554432` | 22.406 ± 0.549 | - | unsupported | 24.035 ± 1.010 | - | 26.627 ± 1.003 | 36.980 ± 0.324 |
| cpu/output_reuse | `div_into` | f64 | 4 | `33554432` | 20.892 ± 0.343 | - | unsupported | 21.216 ± 1.206 | - | 26.037 ± 0.414 | 20.580 ± 0.213 |
| cpu/output_reuse | `dot_general_read_into` | f64 | 1 | `1024x1024` | 51.291 ± 0.678 | - | unsupported | 55.859 ± 0.217 | - | 53.670 ± 0.267 | - |
| cpu/output_reuse | `dot_general_read_into` | f64 | 4 | `1024x1024` | 25.078 ± 0.068 | - | unsupported | 27.625 ± 0.203 | - | 25.921 ± 0.102 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 1 | `1024x1024` | 51.873 ± 0.873 | - | unsupported | 55.700 ± 0.503 | - | 53.256 ± 0.105 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 4 | `1024x1024` | 25.095 ± 0.190 | - | unsupported | 27.803 ± 0.569 | - | 25.622 ± 0.066 | - |
| cpu/output_reuse | `mul_into` | f64 | 1 | `33554432` | 22.652 ± 0.176 | - | unsupported | 23.527 ± 1.355 | - | 22.491 ± 0.550 | 36.409 ± 0.292 |
| cpu/output_reuse | `mul_into` | f64 | 4 | `33554432` | 21.404 ± 0.281 | - | unsupported | 21.650 ± 1.206 | - | 22.872 ± 0.839 | 21.494 ± 3.617 |
| cpu/output_reuse | `neg_into` | f64 | 1 | `33554432` | 18.802 ± 1.077 | - | unsupported | 15.554 ± 0.394 | - | 17.951 ± 4.072 | 22.455 ± 0.032 |
| cpu/output_reuse | `neg_into` | f64 | 4 | `33554432` | 14.371 ± 0.189 | - | unsupported | 14.709 ± 0.169 | - | 16.389 ± 0.231 | 14.258 ± 0.491 |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `256x256,rhs=16` | 0.576 ± 0.003 | unsupported | unsupported | 0.978 ± 0.006 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `768x768,rhs=16` | 9.806 ± 0.038 | unsupported | unsupported | 13.935 ± 0.033 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `256x256,rhs=16` | 0.656 ± 0.006 | unsupported | unsupported | 0.651 ± 0.006 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `768x768,rhs=16` | 7.033 ± 0.059 | unsupported | unsupported | 6.750 ± 0.042 | unsupported | - | - |
| cpu/output_reuse | `sub_into` | f64 | 1 | `33554432` | 22.734 ± 0.128 | - | unsupported | 22.304 ± 0.170 | - | 24.791 ± 0.859 | 36.300 ± 0.227 |
| cpu/output_reuse | `sub_into` | f64 | 4 | `33554432` | 21.207 ± 1.000 | - | unsupported | 21.649 ± 1.060 | - | 24.733 ± 1.019 | 20.691 ± 0.309 |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 1 | `8192x1 -> 8192x4096` | 115.797 ± 3.117 | - | 0.002 ± 0.000 | 114.902 ± 4.544 | 110.178 ± 5.454 | 142.621 ± 16.169 | - |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 4 | `8192x1 -> 8192x4096` | 48.630 ± 1.059 | - | 0.002 ± 0.000 | 40.573 ± 0.368 | 65.096 ± 4.934 | 143.081 ± 20.737 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 1 | `33554432` | 73.017 ± 1.595 | - | 71.845 ± 0.336 | 73.166 ± 2.764 | 79.605 ± 3.454 | 83.451 ± 4.292 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 4 | `33554432` | 26.066 ± 1.178 | - | 94.622 ± 10.545 | 24.852 ± 5.771 | 42.239 ± 1.981 | 85.940 ± 3.761 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 1 | `8192 -> 8192x8192` | 224.134 ± 8.558 | - | 227.426 ± 1.406 | 225.024 ± 2.315 | 227.201 ± 2.431 | 260.987 ± 11.624 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 4 | `8192 -> 8192x8192` | 99.168 ± 47.371 | - | 228.018 ± 11.179 | 73.449 ± 1.906 | 116.699 ± 4.156 | 258.064 ± 5.780 | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 1 | `8388608x2x2 -> 8388608x2` | 71.960 ± 2.617 | - | 72.079 ± 4.074 | 74.760 ± 7.244 | 77.848 ± 1.919 | - | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 4 | `8388608x2x2 -> 8388608x2` | 44.159 ± 3.044 | - | 75.038 ± 6.764 | 26.985 ± 2.083 | 40.309 ± 1.936 | - | - |
| cpu/structural_shape | `reshape` | f64 | 1 | `33554432 -> 8192x4096` | 142.877 ± 7.702 | - | 0.002 ± 0.000 | 129.371 ± 4.743 | 123.364 ± 1.594 | 164.441 ± 20.132 | - |
| cpu/structural_shape | `reshape` | f64 | 4 | `33554432 -> 8192x4096` | 45.829 ± 3.874 | - | 0.002 ± 0.000 | 44.829 ± 1.684 | 51.855 ± 5.861 | 166.952 ± 8.073 | - |
| cpu/structural_shape | `transpose` | f64 | 1 | `4096x4096` | 102.403 ± 1.877 | - | 0.002 ± 0.000 | 93.325 ± 1.769 | 91.644 ± 2.680 | 234.606 ± 11.170 | 138.131 ± 4.135 |
| cpu/structural_shape | `transpose` | f64 | 4 | `4096x4096` | 32.653 ± 3.044 | - | 0.002 ± 0.000 | 92.198 ± 0.595 | 44.935 ± 2.103 | 233.557 ± 2.855 | 56.227 ± 2.339 |
| cpu/structural_shape | `tril` | f64 | 1 | `4096x4096` | 63.396 ± 1.115 | - | 63.365 ± 2.402 | 76.551 ± 1.168 | 64.426 ± 5.230 | 85.555 ± 6.737 | - |
| cpu/structural_shape | `tril` | f64 | 4 | `4096x4096` | 25.133 ± 0.191 | - | 82.905 ± 1.068 | 24.946 ± 0.820 | 34.474 ± 1.557 | 84.555 ± 3.881 | - |
| cpu/structural_shape | `triu` | f64 | 1 | `4096x4096` | 62.625 ± 0.912 | - | 62.823 ± 2.701 | 70.001 ± 2.195 | 64.303 ± 2.296 | 86.634 ± 5.707 | - |
| cpu/structural_shape | `triu` | f64 | 4 | `4096x4096` | 23.631 ± 0.526 | - | 83.884 ± 4.075 | 22.864 ± 2.752 | 34.304 ± 1.160 | 86.664 ± 11.367 | - |
| cpu/view_metadata | `broadcast_in_dim_view` | f64 | 1 | `8192x1 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | - | - |
| cpu/view_metadata | `broadcast_in_dim_view` | f64 | 4 | `8192x1 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | - | - |
| cpu/view_metadata | `reshape_view` | f64 | 1 | `33554432 -> 8192x4096` | 0.001 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `reshape_view` | f64 | 4 | `33554432 -> 8192x4096` | 0.001 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `slice_view` | f64 | 1 | `4194304 -> 2096128` | 0.001 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `slice_view` | f64 | 4 | `4194304 -> 2096128` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `transpose_view` | f64 | 1 | `4096x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `transpose_view` | f64 | 4 | `4096x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |

## Short-operation batches

Total batch duration and normalized ns/op are shown without rounding sub-microsecond calls to zero. Metadata views require no execution session.

| Operation | Dtype | Shape | Threads | Backend | Operations/batch | Median batch ms | Median ns/op |
|---|---|---|---:|---|---:|---:|---:|
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 16 | 8.309371 | 519335.69 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs direct API (ms) | 16 | 7.865688 | 491605.50 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 16 | 8.476273 | 529767.06 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 16 | 6.414396 | 400899.75 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs direct API (ms) | 16 | 6.645002 | 415312.62 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 16 | 863.988884 | 53999305.25 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs direct API (ms) | 16 | 13.491436 | 843214.75 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs eager mode (ms) | 16 | 13.163240 | 822702.50 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs direct API (ms) | 16 | 4.732756 | 295797.25 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs eager mode (ms) | 16 | 383.983478 | 23998967.38 |
| householder_qr | f64 | `256x128` | 1 | JAX Python (XLA CPU) (ms) | 16 | 11.768048 | 735503.00 |
| householder_qr | f64 | `256x128` | 1 | PyTorch Python (ms) | 16 | 10.049104 | 628069.00 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 5.575293 | 348455.81 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 5.795570 | 362223.12 |
| householder_qr | f64 | `768x384` | 1 | JAX Python (XLA CPU) (ms) | 16 | 163.883470 | 10242716.88 |
| householder_qr | f64 | `768x384` | 1 | PyTorch Python (ms) | 16 | 144.388972 | 9024310.75 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 105.721851 | 6607615.69 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 107.424975 | 6714060.94 |
| householder_qr | f64 | `256x128` | 4 | JAX Python (XLA CPU) (ms) | 16 | 28.964612 | 1810288.25 |
| householder_qr | f64 | `256x128` | 4 | PyTorch Python (ms) | 16 | 8.241635 | 515102.19 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 5.540883 | 346305.19 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 1151.974992 | 71998437.00 |
| householder_qr | f64 | `768x384` | 4 | JAX Python (XLA CPU) (ms) | 16 | 185.470638 | 11591914.88 |
| householder_qr | f64 | `768x384` | 4 | PyTorch Python (ms) | 16 | 92.940357 | 5808772.31 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 65.079463 | 4067466.44 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 8543.987626 | 533999226.62 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 16.301991 | 1018874.44 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 18.088829 | 1130551.81 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 337.571255 | 21098203.44 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 338.167458 | 21135466.12 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 16.269127 | 1016820.44 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 17.943839 | 1121489.94 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 217.295656 | 13580978.50 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 68965.957028 | 4310372314.25 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 16.441217 | 1027576.06 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 15.948667 | 996791.69 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 351.590529 | 21974408.06 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 363.712855 | 22732053.44 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 16.236749 | 1014796.81 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 15.947901 | 996743.81 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 218.049185 | 13628074.06 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 69390.920865 | 4336932554.06 |
| erf | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 9.408300 | 147004.69 |
| erf | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 10.533074 | 164579.28 |
| erf | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 9.564375 | 149443.36 |
| erf | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 9.552067 | 149251.05 |
| erf | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 3.394446 | 424305.75 |
| erf | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 20.409886 | 2551235.75 |
| erf | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 17.730608 | 2216326.00 |
| erf | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 19.005858 | 2375732.25 |
| erf | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 8.777134 | 137142.72 |
| erf | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 5.112437 | 79881.83 |
| erf | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 4.697243 | 73394.42 |
| erf | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 383.971540 | 5999555.31 |
| erf | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 6.540142 | 817517.75 |
| erf | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 8.552787 | 1069098.38 |
| erf | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 7.303755 | 912969.38 |
| erf | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 191.990354 | 23998794.25 |
| gelu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 17.023030 | 265984.84 |
| gelu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 9.293307 | 145207.92 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 5.384990 | 84140.47 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 5.394024 | 84281.62 |
| gelu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 23.290447 | 2911305.88 |
| gelu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 15.198791 | 1899848.88 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 10.844452 | 1355556.50 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 10.825394 | 1353174.25 |
| gelu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 10.394526 | 162414.47 |
| gelu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 5.922969 | 92546.39 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 2.430763 | 37980.67 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 767.973986 | 11999593.53 |
| gelu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 9.484439 | 1185554.88 |
| gelu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 6.184446 | 773055.75 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 4.426651 | 553331.38 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 95.991161 | 11998895.12 |
| gelu_tanh | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 13.709281 | 214207.52 |
| gelu_tanh | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 17.857813 | 279028.33 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 13.508060 | 211063.44 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 13.551945 | 211749.14 |
| gelu_tanh | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 15.951748 | 1993968.50 |
| gelu_tanh | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 34.556417 | 4319552.12 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 26.980296 | 3372537.00 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 27.000737 | 3375092.12 |
| gelu_tanh | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 8.977452 | 140272.69 |
| gelu_tanh | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 7.196536 | 112445.88 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 5.329408 | 83272.00 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 706.670104 | 11041720.38 |
| gelu_tanh | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 8.382327 | 1047790.88 |
| gelu_tanh | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 12.935653 | 1616956.62 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 10.346185 | 1293273.12 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 95.993364 | 11999170.50 |
| layer_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 12.067954 | 188561.78 |
| layer_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 10.711509 | 167367.33 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 3.569260 | 55769.69 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 5.448956 | 85139.94 |
| layer_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 14.897434 | 1862179.25 |
| layer_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 14.253470 | 1781683.75 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 5.957239 | 744654.88 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 5.832114 | 729014.25 |
| layer_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 17.100629 | 267197.33 |
| layer_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 4.463296 | 69739.00 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 6.148453 | 96069.58 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 2302.968033 | 35983875.52 |
| layer_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 11.303896 | 1412987.00 |
| layer_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 5.519190 | 689898.75 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 3.231200 | 403900.00 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 768.007499 | 96000937.38 |
| log_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 15.521681 | 242526.27 |
| log_softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 8.510956 | 132983.69 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 8.289335 | 129520.86 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 9.562122 | 149408.16 |
| log_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 18.371588 | 2296448.50 |
| log_softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 16.977017 | 2122127.12 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 15.027321 | 1878415.12 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 15.132337 | 1891542.12 |
| log_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 14.186697 | 221667.14 |
| log_softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 4.016183 | 62752.86 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 9.907454 | 154803.97 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 2870.969385 | 44858896.64 |
| log_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 10.529216 | 1316152.00 |
| log_softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 6.284274 | 785534.25 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 7.335102 | 916887.75 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 959.980455 | 119997556.88 |
| masked_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 16.599682 | 259370.03 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 21.061334 | 329083.34 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 22.944966 | 358515.09 |
| masked_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 26.936207 | 3367025.88 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 39.881257 | 4985157.12 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 40.280659 | 5035082.38 |
| masked_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 16.937311 | 264645.48 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 20.430222 | 319222.22 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 3442.967691 | 53796370.17 |
| masked_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 15.396923 | 1924615.38 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 16.215784 | 2026973.00 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 1151.994642 | 143999330.25 |
| reduce_mean | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 3.470722 | 54230.03 |
| reduce_mean | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 0.786366 | 12286.97 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 0.778414 | 12162.72 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 1.102474 | 17226.16 |
| reduce_mean | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 2.014980 | 251872.50 |
| reduce_mean | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 0.617525 | 77190.62 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 1.057006 | 132125.75 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 0.923329 | 115416.12 |
| reduce_mean | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 2.248589 | 35134.20 |
| reduce_mean | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 0.873745 | 13652.27 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 1.026191 | 16034.23 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 383.962274 | 5999410.53 |
| reduce_mean | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 1.833042 | 229130.25 |
| reduce_mean | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 0.594110 | 74263.75 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 0.504207 | 63025.87 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 191.989038 | 23998629.75 |
| rms_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 11.430207 | 178596.98 |
| rms_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 6.167763 | 96371.30 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 1.889390 | 29521.72 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 3.059875 | 47810.55 |
| rms_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 14.573038 | 1821629.75 |
| rms_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 11.191641 | 1398955.12 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 3.209898 | 401237.25 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 3.051953 | 381494.12 |
| rms_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 9.605965 | 150093.20 |
| rms_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 7.046554 | 110102.41 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 2.929943 | 45780.36 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 1150.978014 | 17984031.47 |
| rms_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 9.055638 | 1131954.75 |
| rms_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 4.535392 | 566924.00 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 1.771346 | 221418.25 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 383.992370 | 47999046.25 |
| sigmoid | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 12.925488 | 201960.75 |
| sigmoid | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 7.497103 | 117142.23 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 3.768255 | 58878.98 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 3.775786 | 58996.66 |
| sigmoid | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 14.654423 | 1831802.88 |
| sigmoid | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 14.421870 | 1802733.75 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 7.508889 | 938611.12 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 7.592493 | 949061.62 |
| sigmoid | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 8.086658 | 126354.03 |
| sigmoid | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 6.078028 | 94969.19 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 1.675343 | 26177.23 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 767.977012 | 11999640.81 |
| sigmoid | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 7.799102 | 974887.75 |
| sigmoid | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 5.670514 | 708814.25 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 3.419959 | 427494.88 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 95.993895 | 11999236.88 |
| silu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 12.942593 | 202228.02 |
| silu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 7.782006 | 121593.84 |
| silu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 3.883385 | 60677.89 |
| silu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 3.907351 | 61052.36 |
| silu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 15.134656 | 1891832.00 |
| silu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 14.764577 | 1845572.12 |
| silu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 7.813640 | 976705.00 |
| silu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 7.937444 | 992180.50 |
| silu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 8.396328 | 131192.62 |
| silu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 6.335120 | 98986.25 |
| silu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 1.751056 | 27360.25 |
| silu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 767.975670 | 11999619.84 |
| silu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 7.830819 | 978852.38 |
| silu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 6.076056 | 759507.00 |
| silu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 3.197118 | 399639.75 |
| silu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 95.995418 | 11999427.25 |
| softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 15.835618 | 247431.53 |
| softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 8.848557 | 138258.70 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 8.317176 | 129955.88 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 9.788557 | 152946.20 |
| softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 19.323651 | 2415456.38 |
| softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 10.237286 | 1279660.75 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 15.118667 | 1889833.38 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 15.334256 | 1916782.00 |
| softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 13.301693 | 207838.95 |
| softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 4.014841 | 62731.89 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 9.813474 | 153335.53 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 2870.965264 | 44858832.25 |
| softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 13.647636 | 1705954.50 |
| softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 3.822255 | 477781.88 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 7.331337 | 916417.12 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 960.002193 | 120000274.12 |
| softplus | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 23.798988 | 371859.19 |
| softplus | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 16.925461 | 264460.33 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 12.624791 | 197262.36 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 12.678991 | 198109.23 |
| softplus | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 36.516832 | 4564604.00 |
| softplus | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 34.244805 | 4280600.62 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 25.251504 | 3156438.00 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 25.229632 | 3153704.00 |
| softplus | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 14.329769 | 223902.64 |
| softplus | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 13.096331 | 204630.17 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 4.928654 | 77010.22 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 767.976841 | 11999638.14 |
| softplus | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 19.279560 | 2409945.00 |
| softplus | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 12.644321 | 1580540.12 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 9.542634 | 1192829.25 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 95.993995 | 11999249.38 |
| take_along_axis | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 14.909201 | 232956.27 |
| take_along_axis | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 18.536048 | 289625.75 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 43.342557 | 677227.45 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 92.152857 | 1439888.39 |
| take_along_axis | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 20.558844 | 2569855.50 |
| take_along_axis | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 50.129684 | 6266210.50 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 103.388078 | 12923509.75 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 185.995849 | 23249481.12 |
| take_along_axis | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 10.181550 | 159086.72 |
| take_along_axis | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 18.637524 | 291211.31 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 51.619600 | 806556.25 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 6144.991702 | 96015495.34 |
| take_along_axis | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 11.088176 | 1386022.00 |
| take_along_axis | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 20.062776 | 2507847.00 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 64.150609 | 8018826.12 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 1152.012393 | 144001549.12 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | PyTorch Python (ms) | 16 | 15.649243 | 978077.69 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 9.218521 | 576157.56 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | PyTorch Python (ms) | 16 | 222.960790 | 13935049.38 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 156.895192 | 9805949.50 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | PyTorch Python (ms) | 16 | 10.411500 | 650718.75 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 10.502399 | 656399.94 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | PyTorch Python (ms) | 16 | 108.005191 | 6750324.44 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 112.533113 | 7033319.56 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.023434 | 1464.62 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.002864 | 179.00 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.023014 | 1438.38 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.002704 | 169.00 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000281 | 17.56 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.021371 | 1335.69 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.009944 | 621.50 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000231 | 14.44 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.021372 | 1335.75 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.009725 | 607.81 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000131 | 8.19 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | PyTorch Python (ms) | 16 | 0.022333 | 1395.81 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | tenferro-rs direct API (ms) | 16 | 0.008202 | 512.62 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000641 | 40.06 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | PyTorch Python (ms) | 16 | 0.022022 | 1376.38 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | tenferro-rs direct API (ms) | 16 | 0.007961 | 497.56 |
| transpose_view | f64 | `4096x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000110 | 6.88 |
| transpose_view | f64 | `4096x4096` | 1 | PyTorch Python (ms) | 16 | 0.020050 | 1253.12 |
| transpose_view | f64 | `4096x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.005218 | 326.12 |
| transpose_view | f64 | `4096x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000110 | 6.88 |
| transpose_view | f64 | `4096x4096` | 4 | PyTorch Python (ms) | 16 | 0.020471 | 1279.44 |
| transpose_view | f64 | `4096x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.004277 | 267.31 |

## Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `cpu/complex/cholesky` (c64, threads=4, shape=`448x448`): `jax-cpu` is 49.9x faster than the slowest successful cell (`tenferro-trace`, 191.994 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/cholesky` (c64, threads=4, shape=`448x448`): `julia-base` is 60.4x faster than the slowest successful cell (`tenferro-trace`, 191.994 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/cholesky` (c64, threads=4, shape=`448x448`): `pytorch-cpu` is 83.7x faster than the slowest successful cell (`tenferro-trace`, 191.994 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/cholesky` (c64, threads=4, shape=`448x448`): `tenferro-direct` is 72.2x faster than the slowest successful cell (`tenferro-trace`, 191.994 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/eig` (c64, threads=4, shape=`112x112`): `jax-cpu` is 80.5x faster than the slowest successful cell (`tenferro-trace`, 1501.996 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/eig` (c64, threads=4, shape=`112x112`): `julia-base` is 81.4x faster than the slowest successful cell (`tenferro-trace`, 1501.996 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/eig` (c64, threads=4, shape=`112x112`): `pytorch-cpu` is 103.0x faster than the slowest successful cell (`tenferro-trace`, 1501.996 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/eig` (c64, threads=4, shape=`112x112`): `tenferro-direct` is 145.9x faster than the slowest successful cell (`tenferro-trace`, 1501.996 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/norm_fro` (c64, threads=1, shape=`2048x1536`): `tenferro-direct` is 15.5x faster than the slowest successful cell (`pytorch-cpu`, 23.370 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/norm_fro` (c64, threads=4, shape=`2048x1536`): `jax-cpu` is 22.8x faster than the slowest successful cell (`tenferro-trace`, 59.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/norm_fro` (c64, threads=4, shape=`2048x1536`): `julia-base` is 13.4x faster than the slowest successful cell (`tenferro-trace`, 59.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/norm_fro` (c64, threads=4, shape=`2048x1536`): `tenferro-direct` is 47.1x faster than the slowest successful cell (`tenferro-trace`, 59.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `jax-cpu` is 50.8x faster than the slowest successful cell (`tenferro-trace`, 396.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `julia-base` is 56.2x faster than the slowest successful cell (`tenferro-trace`, 396.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `pytorch-cpu` is 65.8x faster than the slowest successful cell (`tenferro-trace`, 396.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `tenferro-direct` is 84.9x faster than the slowest successful cell (`tenferro-trace`, 396.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `jax-cpu` is 90.9x faster than the slowest successful cell (`tenferro-trace`, 545.009 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `julia-base` is 130.2x faster than the slowest successful cell (`tenferro-trace`, 545.009 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `pytorch-cpu` is 153.7x faster than the slowest successful cell (`tenferro-trace`, 545.009 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `tenferro-direct` is 143.1x faster than the slowest successful cell (`tenferro-trace`, 545.009 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/masked_log_softmax_axis1` (f64, threads=4, shape=`1024x64`): `jax-cpu` is 134.7x faster than the slowest successful cell (`tenferro-eager`, 53.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/masked_log_softmax_axis1` (f64, threads=4, shape=`1024x64`): `tenferro-direct` is 130.0x faster than the slowest successful cell (`tenferro-eager`, 53.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `jax-cpu` is 10.6x faster than the slowest successful cell (`pytorch-cpu`, 28.296 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 27.8x faster than the slowest successful cell (`pytorch-cpu`, 28.296 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 14.8x faster than the slowest successful cell (`pytorch-cpu`, 28.296 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 14.8x faster than the slowest successful cell (`pytorch-cpu`, 28.296 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 14.4x faster than the slowest successful cell (`tenferro-trace`, 23.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 25.2x faster than the slowest successful cell (`tenferro-trace`, 23.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 26.8x faster than the slowest successful cell (`tenferro-trace`, 23.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis1` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 27.3x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis1` (f64, threads=4, shape=`2048x2048`): `julia-base` is 18.2x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis1` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 14.5x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `jax-cpu` is 10.3x faster than the slowest successful cell (`pytorch-cpu`, 28.396 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 29.6x faster than the slowest successful cell (`pytorch-cpu`, 28.396 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 15.4x faster than the slowest successful cell (`pytorch-cpu`, 28.396 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 15.4x faster than the slowest successful cell (`pytorch-cpu`, 28.396 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 14.9x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 27.2x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 27.9x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis0` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 21.1x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 28.4x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis0` (f64, threads=4, shape=`2048x2048`): `pytorch-cpu` is 20.5x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 72.7x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis1` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 18.0x faster than the slowest successful cell (`tenferro-trace`, 24.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis1` (f64, threads=4, shape=`2048x2048`): `julia-base` is 15.8x faster than the slowest successful cell (`tenferro-trace`, 24.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis1` (f64, threads=4, shape=`2048x2048`): `pytorch-cpu` is 81.4x faster than the slowest successful cell (`tenferro-trace`, 24.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_prod_axis1` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 29.3x faster than the slowest successful cell (`tenferro-trace`, 24.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis0` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 20.7x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 33.6x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis0` (f64, threads=4, shape=`2048x2048`): `pytorch-cpu` is 22.1x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 71.7x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis1` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 18.3x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis1` (f64, threads=4, shape=`2048x2048`): `julia-base` is 24.9x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis1` (f64, threads=4, shape=`2048x2048`): `pytorch-cpu` is 82.2x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_axis1` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 30.8x faster than the slowest successful cell (`tenferro-trace`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_sum_squares_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 81.1x faster than the slowest successful cell (`tenferro-eager`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/concatenate` (f64, threads=4, shape=`1048576+1048576`): `jax-cpu` is 15.8x faster than the slowest successful cell (`tenferro-trace`, 11.996 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/concatenate` (f64, threads=4, shape=`1048576+1048576`): `pytorch-cpu` is 37.0x faster than the slowest successful cell (`tenferro-trace`, 11.996 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/concatenate` (f64, threads=4, shape=`1048576+1048576`): `tenferro-direct` is 16.7x faster than the slowest successful cell (`tenferro-trace`, 11.996 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/dynamic_slice` (f64, threads=4, shape=`4194304 -> 2097152`): `jax-cpu` is 18.3x faster than the slowest successful cell (`tenferro-trace`, 11.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/dynamic_slice` (f64, threads=4, shape=`4194304 -> 2097152`): `pytorch-cpu` is 35.6x faster than the slowest successful cell (`tenferro-trace`, 11.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/dynamic_slice` (f64, threads=4, shape=`4194304 -> 2097152`): `tenferro-direct` is 24.6x faster than the slowest successful cell (`tenferro-trace`, 11.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/dynamic_update_slice` (f64, threads=4, shape=`2097152`): `jax-cpu` is 10.2x faster than the slowest successful cell (`julia-base`, 8.994 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/dynamic_update_slice` (f64, threads=4, shape=`2097152`): `pytorch-cpu` is 14.3x faster than the slowest successful cell (`julia-base`, 8.994 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/dynamic_update_slice` (f64, threads=4, shape=`2097152`): `tenferro-direct` is 16.4x faster than the slowest successful cell (`julia-base`, 8.994 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/gather` (f64, threads=4, shape=`262144`): `jax-cpu` is 188.7x faster than the slowest successful cell (`tenferro-trace`, 47.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/gather` (f64, threads=4, shape=`262144`): `julia-base` is 37.1x faster than the slowest successful cell (`tenferro-trace`, 47.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/gather` (f64, threads=4, shape=`262144`): `pytorch-cpu` is 118.2x faster than the slowest successful cell (`tenferro-trace`, 47.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/gather` (f64, threads=4, shape=`262144`): `tenferro-direct` is 144.9x faster than the slowest successful cell (`tenferro-trace`, 47.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/pad` (f64, threads=4, shape=`2097152`): `jax-cpu` is 10.4x faster than the slowest successful cell (`tenferro-trace`, 11.995 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/pad` (f64, threads=4, shape=`2097152`): `pytorch-cpu` is 24.3x faster than the slowest successful cell (`tenferro-trace`, 11.995 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/pad` (f64, threads=4, shape=`2097152`): `tenferro-direct` is 32.6x faster than the slowest successful cell (`tenferro-trace`, 11.995 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/reverse` (f64, threads=4, shape=`2097152`): `jax-cpu` is 24.0x faster than the slowest successful cell (`tenferro-trace`, 12.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/reverse` (f64, threads=4, shape=`2097152`): `pytorch-cpu` is 33.7x faster than the slowest successful cell (`tenferro-trace`, 12.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/reverse` (f64, threads=4, shape=`2097152`): `tenferro-direct` is 42.2x faster than the slowest successful cell (`tenferro-trace`, 12.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/scatter` (f64, threads=4, shape=`262144`): `jax-cpu` is 82.4x faster than the slowest successful cell (`tenferro-trace`, 38.978 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/scatter` (f64, threads=4, shape=`262144`): `julia-base` is 26.6x faster than the slowest successful cell (`tenferro-trace`, 38.978 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/scatter` (f64, threads=4, shape=`262144`): `pytorch-cpu` is 80.4x faster than the slowest successful cell (`tenferro-trace`, 38.978 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/scatter` (f64, threads=4, shape=`262144`): `tenferro-direct` is 35.3x faster than the slowest successful cell (`tenferro-trace`, 38.978 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=1, shape=`4194304 -> 2096128`): `tenferro-trace` is 5657.7x faster than the slowest successful cell (`julia-base`, 10.993 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`): `pytorch-cpu` is 13.7x faster than the slowest successful cell (`julia-base`, 10.173 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`): `tenferro-direct` is 13.1x faster than the slowest successful cell (`julia-base`, 10.173 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`): `tenferro-trace` is 5235.5x faster than the slowest successful cell (`julia-base`, 10.173 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x32x32`): `tenferro-direct` is 16.7x faster than the slowest successful cell (`jax-cpu`, 0.059 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x32x32`): `tenferro-trace` is 10.9x faster than the slowest successful cell (`jax-cpu`, 0.059 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x8x8`): `tenferro-direct` is 12.7x faster than the slowest successful cell (`jax-cpu`, 0.031 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x128x128`): `jax-cpu` is 10.3x faster than the slowest successful cell (`tenferro-trace`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x128x128`): `tenferro-direct` is 60.5x faster than the slowest successful cell (`tenferro-trace`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigh` (c64, threads=4, shape=`1024x4x4`): `tenferro-direct` is 14.3x faster than the slowest successful cell (`tenferro-trace`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigh` (c64, threads=4, shape=`8x32x32`): `jax-cpu` is 11.5x faster than the slowest successful cell (`tenferro-trace`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigh` (c64, threads=4, shape=`8x32x32`): `tenferro-direct` is 33.4x faster than the slowest successful cell (`tenferro-trace`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigh` (f64, threads=4, shape=`8x32x32`): `jax-cpu` is 25.9x faster than the slowest successful cell (`tenferro-trace`, 12.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigh` (f64, threads=4, shape=`8x32x32`): `pytorch-cpu` is 12.9x faster than the slowest successful cell (`tenferro-trace`, 12.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigh` (f64, threads=4, shape=`8x32x32`): `tenferro-direct` is 45.9x faster than the slowest successful cell (`tenferro-trace`, 12.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigvalsh` (f64, threads=4, shape=`8x32x32`): `jax-cpu` is 27.0x faster than the slowest successful cell (`tenferro-trace`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigvalsh` (f64, threads=4, shape=`8x32x32`): `pytorch-cpu` is 38.1x faster than the slowest successful cell (`tenferro-trace`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigvalsh` (f64, threads=4, shape=`8x32x32`): `tenferro-direct` is 70.5x faster than the slowest successful cell (`tenferro-trace`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1024x8x8`): `jax-cpu` is 19.1x faster than the slowest successful cell (`tenferro-trace`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1024x8x8`): `pytorch-cpu` is 33.3x faster than the slowest successful cell (`tenferro-trace`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1024x8x8`): `tenferro-direct` is 40.6x faster than the slowest successful cell (`tenferro-trace`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x128x128`): `jax-cpu` is 125.3x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x128x128`): `pytorch-cpu` is 109.1x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x128x128`): `tenferro-direct` is 124.7x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x128x128`): `jax-cpu` is 12.8x faster than the slowest successful cell (`tenferro-trace`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x128x128`): `pytorch-cpu` is 19.5x faster than the slowest successful cell (`tenferro-trace`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x128x128`): `tenferro-direct` is 25.2x faster than the slowest successful cell (`tenferro-trace`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=1, shape=`3x2x2`): `tenferro-direct` is 12.9x faster than the slowest successful cell (`jax-cpu`, 0.036 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`1024x8x8`): `tenferro-direct` is 14.4x faster than the slowest successful cell (`tenferro-trace`, 12.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`8x32x32`): `jax-cpu` is 43.0x faster than the slowest successful cell (`tenferro-trace`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`8x32x32`): `pytorch-cpu` is 44.6x faster than the slowest successful cell (`tenferro-trace`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`8x32x32`): `tenferro-direct` is 114.1x faster than the slowest successful cell (`tenferro-trace`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1024x8x8,rhs=1`): `jax-cpu` is 27.7x faster than the slowest successful cell (`tenferro-trace`, 23.994 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1024x8x8,rhs=1`): `pytorch-cpu` is 60.5x faster than the slowest successful cell (`tenferro-trace`, 23.994 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1024x8x8,rhs=1`): `tenferro-direct` is 82.6x faster than the slowest successful cell (`tenferro-trace`, 23.994 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1x128x128,rhs=1`): `jax-cpu` is 130.1x faster than the slowest successful cell (`tenferro-trace`, 23.996 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1x128x128,rhs=1`): `pytorch-cpu` is 126.5x faster than the slowest successful cell (`tenferro-trace`, 23.996 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1x128x128,rhs=1`): `tenferro-direct` is 171.0x faster than the slowest successful cell (`tenferro-trace`, 23.996 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x128x128,rhs=1`): `jax-cpu` is 48.3x faster than the slowest successful cell (`tenferro-trace`, 36.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x128x128,rhs=1`): `pytorch-cpu` is 62.4x faster than the slowest successful cell (`tenferro-trace`, 36.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x128x128,rhs=1`): `tenferro-direct` is 102.5x faster than the slowest successful cell (`tenferro-trace`, 36.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_svd` (c64, threads=4, shape=`8x32x32`): `jax-cpu` is 14.8x faster than the slowest successful cell (`tenferro-trace`, 12.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_svd` (c64, threads=4, shape=`8x32x32`): `tenferro-direct` is 19.6x faster than the slowest successful cell (`tenferro-trace`, 12.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_svd` (f64, threads=4, shape=`8x32x32`): `jax-cpu` is 15.6x faster than the slowest successful cell (`tenferro-trace`, 12.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_svd` (f64, threads=4, shape=`8x32x32`): `tenferro-direct` is 28.4x faster than the slowest successful cell (`tenferro-trace`, 12.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batched/batched_triangular_solve` (f64, threads=4, shape=`1024x16x16,rhs=1`): `jax-cpu` is 20.8x faster than the slowest successful cell (`tenferro-trace`, 11.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batched/batched_triangular_solve` (f64, threads=4, shape=`1024x16x16,rhs=1`): `julia-base` is 45.6x faster than the slowest successful cell (`tenferro-trace`, 11.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batched/batched_triangular_solve` (f64, threads=4, shape=`1024x16x16,rhs=1`): `pytorch-cpu` is 45.0x faster than the slowest successful cell (`tenferro-trace`, 11.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batched/batched_triangular_solve` (f64, threads=4, shape=`1024x16x16,rhs=1`): `tenferro-direct` is 92.3x faster than the slowest successful cell (`tenferro-trace`, 11.997 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/cholesky` (f64, threads=4, shape=`1536x1536`): `pytorch-cpu` is 11.2x faster than the slowest successful cell (`tenferro-trace`, 276.004 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/cholesky` (f64, threads=4, shape=`1536x1536`): `tenferro-direct` is 12.4x faster than the slowest successful cell (`tenferro-trace`, 276.004 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `jax-cpu` is 63.7x faster than the slowest successful cell (`tenferro-trace`, 935.988 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `julia-base` is 72.7x faster than the slowest successful cell (`tenferro-trace`, 935.988 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `pytorch-cpu` is 78.0x faster than the slowest successful cell (`tenferro-trace`, 935.988 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `tenferro-direct` is 54.7x faster than the slowest successful cell (`tenferro-trace`, 935.988 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eig` (f64, threads=4, shape=`160x160`): `jax-cpu` is 66.2x faster than the slowest successful cell (`tenferro-trace`, 794.004 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eig` (f64, threads=4, shape=`160x160`): `julia-base` is 67.2x faster than the slowest successful cell (`tenferro-trace`, 794.004 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eig` (f64, threads=4, shape=`160x160`): `pytorch-cpu` is 68.5x faster than the slowest successful cell (`tenferro-trace`, 794.004 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eig` (f64, threads=4, shape=`160x160`): `tenferro-direct` is 77.6x faster than the slowest successful cell (`tenferro-trace`, 794.004 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvals` (f64, threads=4, shape=`192x192`): `jax-cpu` is 81.7x faster than the slowest successful cell (`tenferro-trace`, 1157.994 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvals` (f64, threads=4, shape=`192x192`): `julia-base` is 85.6x faster than the slowest successful cell (`tenferro-trace`, 1157.994 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvals` (f64, threads=4, shape=`192x192`): `pytorch-cpu` is 93.5x faster than the slowest successful cell (`tenferro-trace`, 1157.994 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvals` (f64, threads=4, shape=`192x192`): `tenferro-direct` is 89.1x faster than the slowest successful cell (`tenferro-trace`, 1157.994 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/householder_qr` (f64, threads=4, shape=`256x128`): `jax-cpu` is 39.8x faster than the slowest successful cell (`tenferro-eager`, 71.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/householder_qr` (f64, threads=4, shape=`256x128`): `pytorch-cpu` is 139.8x faster than the slowest successful cell (`tenferro-eager`, 71.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/householder_qr` (f64, threads=4, shape=`256x128`): `tenferro-direct` is 207.9x faster than the slowest successful cell (`tenferro-eager`, 71.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/householder_qr` (f64, threads=4, shape=`768x384`): `jax-cpu` is 46.1x faster than the slowest successful cell (`tenferro-eager`, 533.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/householder_qr` (f64, threads=4, shape=`768x384`): `pytorch-cpu` is 91.9x faster than the slowest successful cell (`tenferro-eager`, 533.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/householder_qr` (f64, threads=4, shape=`768x384`): `tenferro-direct` is 131.3x faster than the slowest successful cell (`tenferro-eager`, 533.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `jax-cpu` is 48.7x faster than the slowest successful cell (`tenferro-trace`, 970.003 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `julia-base` is 55.6x faster than the slowest successful cell (`tenferro-trace`, 970.003 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `pytorch-cpu` is 42.9x faster than the slowest successful cell (`tenferro-trace`, 970.003 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `tenferro-direct` is 48.6x faster than the slowest successful cell (`tenferro-trace`, 970.003 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lstsq` (f64, threads=4, shape=`768x384,rhs=16`): `julia-base` is 61.3x faster than the slowest successful cell (`tenferro-trace`, 586.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lstsq` (f64, threads=4, shape=`768x384,rhs=16`): `pytorch-cpu` is 92.1x faster than the slowest successful cell (`tenferro-trace`, 586.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lstsq` (f64, threads=4, shape=`768x384,rhs=16`): `tenferro-eager` is 56.9x faster than the slowest successful cell (`tenferro-trace`, 586.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `jax-cpu` is 36.1x faster than the slowest successful cell (`tenferro-trace`, 871.901 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `julia-base` is 34.5x faster than the slowest successful cell (`tenferro-trace`, 871.901 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `pytorch-cpu` is 51.7x faster than the slowest successful cell (`tenferro-trace`, 871.901 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `tenferro-direct` is 50.8x faster than the slowest successful cell (`tenferro-trace`, 871.901 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=1, shape=`2048x2048`): `pytorch-cpu` is 18.9x faster than the slowest successful cell (`jax-cpu`, 22.138 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 27.2x faster than the slowest successful cell (`jax-cpu`, 22.138 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 23.1x faster than the slowest successful cell (`jax-cpu`, 22.138 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=4, shape=`2048x2048`): `pytorch-cpu` is 19.8x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 37.9x faster than the slowest successful cell (`tenferro-trace`, 24.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/rank_revealing_qr` (f64, threads=4, shape=`768x384`): `tenferro-direct` is 317.4x faster than the slowest successful cell (`tenferro-eager`, 4310.372 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/rank_revealing_qr_deficient` (f64, threads=4, shape=`768x384`): `tenferro-direct` is 318.2x faster than the slowest successful cell (`tenferro-eager`, 4336.933 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `jax-cpu` is 64.4x faster than the slowest successful cell (`tenferro-trace`, 936.004 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `julia-base` is 72.3x faster than the slowest successful cell (`tenferro-trace`, 936.004 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `pytorch-cpu` is 77.6x faster than the slowest successful cell (`tenferro-trace`, 936.004 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `tenferro-direct` is 54.4x faster than the slowest successful cell (`tenferro-trace`, 936.004 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/triangular_solve` (f64, threads=4, shape=`4096x4096,rhs=64`): `julia-base` is 38.7x faster than the slowest successful cell (`tenferro-trace`, 839.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/triangular_solve` (f64, threads=4, shape=`4096x4096,rhs=64`): `pytorch-cpu` is 34.6x faster than the slowest successful cell (`tenferro-trace`, 839.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/triangular_solve` (f64, threads=4, shape=`4096x4096,rhs=64`): `tenferro-direct` is 41.8x faster than the slowest successful cell (`tenferro-trace`, 839.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/erf` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 43.7x faster than the slowest successful cell (`tenferro-eager`, 6.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/erf` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 75.1x faster than the slowest successful cell (`tenferro-eager`, 6.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/erf` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 81.7x faster than the slowest successful cell (`tenferro-eager`, 6.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/erf` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 29.4x faster than the slowest successful cell (`tenferro-eager`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/erf` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 22.4x faster than the slowest successful cell (`tenferro-eager`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/erf` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 26.3x faster than the slowest successful cell (`tenferro-eager`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 73.9x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 129.7x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 315.9x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 10.1x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 15.5x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 21.7x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu_tanh` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 78.7x faster than the slowest successful cell (`tenferro-eager`, 11.042 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu_tanh` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 98.2x faster than the slowest successful cell (`tenferro-eager`, 11.042 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu_tanh` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 132.6x faster than the slowest successful cell (`tenferro-eager`, 11.042 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu_tanh` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 11.5x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/layer_norm` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 134.7x faster than the slowest successful cell (`tenferro-eager`, 35.984 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/layer_norm` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 516.0x faster than the slowest successful cell (`tenferro-eager`, 35.984 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/layer_norm` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 374.6x faster than the slowest successful cell (`tenferro-eager`, 35.984 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/layer_norm` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 67.9x faster than the slowest successful cell (`tenferro-eager`, 96.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/layer_norm` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 139.2x faster than the slowest successful cell (`tenferro-eager`, 96.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/layer_norm` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 237.7x faster than the slowest successful cell (`tenferro-eager`, 96.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/log_softmax` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 202.4x faster than the slowest successful cell (`tenferro-eager`, 44.859 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/log_softmax` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 714.8x faster than the slowest successful cell (`tenferro-eager`, 44.859 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/log_softmax` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 289.8x faster than the slowest successful cell (`tenferro-eager`, 44.859 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/log_softmax` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 91.2x faster than the slowest successful cell (`tenferro-eager`, 119.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/log_softmax` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 152.8x faster than the slowest successful cell (`tenferro-eager`, 119.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/log_softmax` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 130.9x faster than the slowest successful cell (`tenferro-eager`, 119.998 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/masked_softmax` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 203.3x faster than the slowest successful cell (`tenferro-eager`, 53.796 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/masked_softmax` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 168.5x faster than the slowest successful cell (`tenferro-eager`, 53.796 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/masked_softmax` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 74.8x faster than the slowest successful cell (`tenferro-eager`, 143.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/masked_softmax` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 71.0x faster than the slowest successful cell (`tenferro-eager`, 143.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/reduce_mean` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 170.8x faster than the slowest successful cell (`tenferro-eager`, 5.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/reduce_mean` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 439.5x faster than the slowest successful cell (`tenferro-eager`, 5.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/reduce_mean` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 374.2x faster than the slowest successful cell (`tenferro-eager`, 5.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/reduce_mean` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 104.7x faster than the slowest successful cell (`tenferro-eager`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/reduce_mean` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 323.2x faster than the slowest successful cell (`tenferro-eager`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/reduce_mean` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 380.8x faster than the slowest successful cell (`tenferro-eager`, 23.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/rms_norm` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 119.8x faster than the slowest successful cell (`tenferro-eager`, 17.984 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/rms_norm` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 163.3x faster than the slowest successful cell (`tenferro-eager`, 17.984 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/rms_norm` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 392.8x faster than the slowest successful cell (`tenferro-eager`, 17.984 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/rms_norm` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 42.4x faster than the slowest successful cell (`tenferro-eager`, 47.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/rms_norm` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 84.7x faster than the slowest successful cell (`tenferro-eager`, 47.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/rms_norm` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 216.8x faster than the slowest successful cell (`tenferro-eager`, 47.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/sigmoid` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 95.0x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/sigmoid` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 126.4x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/sigmoid` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 458.4x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/sigmoid` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 12.3x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/sigmoid` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 16.9x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/sigmoid` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 28.1x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/silu` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 91.5x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/silu` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 121.2x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/silu` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 438.6x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/silu` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 12.3x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/silu` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 15.8x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/silu` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 30.0x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softmax` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 215.8x faster than the slowest successful cell (`tenferro-eager`, 44.859 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softmax` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 715.1x faster than the slowest successful cell (`tenferro-eager`, 44.859 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softmax` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 292.6x faster than the slowest successful cell (`tenferro-eager`, 44.859 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softmax` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 70.3x faster than the slowest successful cell (`tenferro-eager`, 120.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softmax` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 251.2x faster than the slowest successful cell (`tenferro-eager`, 120.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softmax` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 130.9x faster than the slowest successful cell (`tenferro-eager`, 120.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softplus` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 53.6x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softplus` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 58.6x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softplus` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 155.8x faster than the slowest successful cell (`tenferro-eager`, 12.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/softplus` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 10.1x faster than the slowest successful cell (`tenferro-eager`, 11.999 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 603.5x faster than the slowest successful cell (`tenferro-eager`, 96.015 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`1024x64`): `pytorch-cpu` is 329.7x faster than the slowest successful cell (`tenferro-eager`, 96.015 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`1024x64`): `tenferro-direct` is 119.0x faster than the slowest successful cell (`tenferro-eager`, 96.015 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 103.9x faster than the slowest successful cell (`tenferro-eager`, 144.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`4096x256`): `pytorch-cpu` is 57.4x faster than the slowest successful cell (`tenferro-eager`, 144.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`4096x256`): `tenferro-direct` is 18.0x faster than the slowest successful cell (`tenferro-eager`, 144.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=1, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 76554.2x faster than the slowest successful cell (`julia-base`, 142.621 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=4, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 80247.4x faster than the slowest successful cell (`julia-base`, 143.081 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=1, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 84676.1x faster than the slowest successful cell (`julia-base`, 164.441 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=4, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 86863.5x faster than the slowest successful cell (`julia-base`, 166.952 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`): `tenferro-trace` is 134676.0x faster than the slowest successful cell (`julia-base`, 234.606 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`): `tenferro-trace` is 133997.0x faster than the slowest successful cell (`julia-base`, 233.557 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=1, shape=`33554432 -> 8192x4096`): `julia-base` is 74.2x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=4, shape=`33554432 -> 8192x4096`): `julia-base` is 95.4x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=1, shape=`4194304 -> 2096128`): `julia-base` is 174.5x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=4, shape=`4194304 -> 2096128`): `julia-base` is 34.4x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=1, shape=`4096x4096`): `julia-base` is 179.0x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=4, shape=`4096x4096`): `julia-base` is 182.7x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

## Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

- `cpu/indexing_layout/slice` (f64, threads=1, shape=`4194304 -> 2096128`), `tenferro-trace`: 16769024 minimum logical bytes imply 8630.5 GB/s at 0.002 ms. The conservative physical lower bound is 0.168 ms, so the cell is 86.3x faster than the bound.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`), `tenferro-trace`: 16769024 minimum logical bytes imply 8630.5 GB/s at 0.002 ms. The conservative physical lower bound is 0.042 ms, so the cell is 21.6x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=1, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 144087.7 GB/s at 0.002 ms. The conservative physical lower bound is 2.684 ms, so the cell is 1440.9x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=4, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 150552.7 GB/s at 0.002 ms. The conservative physical lower bound is 0.671 ms, so the cell is 376.4x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=1, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 138226.3 GB/s at 0.002 ms. The conservative physical lower bound is 2.684 ms, so the cell is 1382.3x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=4, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 139664.6 GB/s at 0.002 ms. The conservative physical lower bound is 0.671 ms, so the cell is 349.2x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 77048.1 GB/s at 0.002 ms. The conservative physical lower bound is 1.342 ms, so the cell is 770.5x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 77003.9 GB/s at 0.002 ms. The conservative physical lower bound is 0.336 ms, so the cell is 192.5x faster than the bound.
