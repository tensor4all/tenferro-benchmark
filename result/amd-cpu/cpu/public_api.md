# CPU Public API Benchmark Results

- Suite: `cpu/public_api`
- Target profile: `amd-cpu`
- Suite file: `benchmarks/cpu/public_api.yaml`
- Public API coverage manifest: `benchmarks/cpu/public_api_coverage.yaml`
- Run metadata: `data/results/amd-cpu/cpu/public_api/20261018_185641/run.yaml`
- Timestamp: `20261018_185641`

Latest run: `./scripts/run_cpu_public_api.sh 1 4`.

- Sampling: `quick` profile, 7 measured runs, 3 warmups per row

This file is generated from sequential CPU public API runs under `data/results/amd-cpu/cpu/public_api/20261018_185641`.

- tenferro-rs commit: `c2ddccb14b89f8e9fa8a8fe3957367e32fb4e677`

## CPU Information

- Model: `AMD EPYC 7763 64-Core Processor`
- Vendor: `AuthenticAMD`
- Logical CPUs: `4`
- Sockets: `1`
- Cores per socket: `2`
- Threads per core: `2`
- NUMA nodes: `1`
- Python platform: `Linux-6.17.0-1022-azure-x86_64-with-glibc2.39`

## Thread Environments

### Threads: 1

- Run metadata: `data/results/amd-cpu/cpu/public_api/20261018_185641/run_t1.yaml`
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

- Run metadata: `data/results/amd-cpu/cpu/public_api/20261018_185641/run_t4.yaml`
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

- CSV: `data/results/amd-cpu/cpu/public_api/20261018_185641/cpu_public_api_t1_20261018_185641.csv`
- CSV: `data/results/amd-cpu/cpu/public_api/20261018_185641/cpu_public_api_t4_20261018_185641.csv`
- Source table: `data/results/amd-cpu/cpu/public_api/20261018_185641/cpu_public_api_20261018_185641.md`

## CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| cpu/complex | `cholesky` | c64 | 1 | `448x448` | 4.343 ± 0.081 | - | 4.463 ± 0.068 | 4.747 ± 0.131 | 3.969 ± 0.050 | 5.142 ± 1.252 | - |
| cpu/complex | `cholesky` | c64 | 4 | `448x448` | 2.400 ± 0.045 | - | 2.536 ± 0.050 | 2.236 ± 0.046 | 3.227 ± 0.810 | 3.203 ± 0.218 | - |
| cpu/complex | `conj` | c64 | 1 | `16777216` | 27.027 ± 0.719 | - | 27.603 ± 0.579 | 25.220 ± 0.591 | 25.493 ± 0.172 | 30.262 ± 0.705 | 30.756 ± 0.705 |
| cpu/complex | `conj` | c64 | 4 | `16777216` | 16.183 ± 0.908 | - | 17.032 ± 0.891 | 17.310 ± 0.246 | 16.885 ± 0.203 | 28.275 ± 0.459 | 17.236 ± 0.758 |
| cpu/complex | `div` | c64 | 1 | `8388608` | 57.865 ± 0.505 | - | 61.728 ± 0.219 | 43.010 ± 0.111 | 39.708 ± 0.107 | 72.546 ± 0.914 | 96.497 ± 0.575 |
| cpu/complex | `div` | c64 | 4 | `8388608` | 28.399 ± 0.828 | - | 28.638 ± 1.070 | 17.513 ± 0.274 | 20.455 ± 0.101 | 72.053 ± 0.087 | 38.234 ± 0.368 |
| cpu/complex | `dot_general` | c64 | 1 | `640x640` | 44.853 ± 0.409 | - | 45.651 ± 0.389 | 42.712 ± 0.187 | 66.335 ± 0.318 | 45.028 ± 0.177 | - |
| cpu/complex | `dot_general` | c64 | 4 | `640x640` | 29.440 ± 0.156 | - | 31.298 ± 0.403 | 29.490 ± 0.080 | 29.472 ± 0.697 | 22.653 ± 0.206 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 1 | `640x640` | 46.907 ± 0.206 | - | unsupported | 43.142 ± 0.108 | 66.582 ± 1.010 | 44.725 ± 2.913 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 4 | `640x640` | 30.865 ± 0.438 | - | unsupported | 29.756 ± 0.140 | 30.144 ± 0.627 | 22.965 ± 1.748 | - |
| cpu/complex | `eig` | c64 | 1 | `112x112` | 10.956 ± 0.015 | - | 11.032 ± 0.086 | 11.888 ± 0.075 | 11.885 ± 0.095 | 11.882 ± 0.067 | - |
| cpu/complex | `eig` | c64 | 4 | `112x112` | 14.521 ± 0.038 | - | 14.682 ± 0.168 | 14.066 ± 0.038 | 17.489 ± 1.719 | 17.638 ± 0.037 | - |
| cpu/complex | `exp` | c64 | 1 | `4194304` | 71.408 ± 0.143 | - | 72.423 ± 0.172 | 85.482 ± 0.062 | 79.436 ± 2.028 | 56.235 ± 0.822 | 56.561 ± 0.460 |
| cpu/complex | `exp` | c64 | 4 | `4194304` | 31.157 ± 0.132 | - | 31.194 ± 0.182 | 36.699 ± 0.154 | 40.728 ± 0.141 | 56.025 ± 0.210 | 26.045 ± 1.163 |
| cpu/complex | `log` | c64 | 1 | `4194304` | 116.351 ± 0.217 | - | 116.392 ± 0.058 | 131.375 ± 0.156 | 152.308 ± 0.601 | 183.355 ± 2.663 | 127.872 ± 0.156 |
| cpu/complex | `log` | c64 | 4 | `4194304` | 49.781 ± 0.266 | - | 49.860 ± 0.173 | 56.697 ± 0.114 | 64.816 ± 0.857 | 233.081 ± 0.354 | 52.725 ± 1.445 |
| cpu/complex | `mul` | c64 | 1 | `8388608` | 20.075 ± 0.215 | - | 20.383 ± 0.034 | 17.169 ± 0.343 | 17.505 ± 0.539 | 17.515 ± 0.396 | 19.073 ± 0.417 |
| cpu/complex | `mul` | c64 | 4 | `8388608` | 12.983 ± 0.764 | - | 11.912 ± 0.893 | 12.568 ± 1.056 | 11.216 ± 0.179 | 17.548 ± 0.329 | 12.633 ± 0.652 |
| cpu/complex | `norm_fro` | c64 | 1 | `2048x1536` | 2.429 ± 0.055 | - | 20.899 ± 0.076 | 24.266 ± 0.049 | 4.988 ± 0.254 | 4.038 ± 0.014 | - |
| cpu/complex | `norm_fro` | c64 | 4 | `2048x1536` | 1.315 ± 0.014 | - | 14.887 ± 7.076 | 11.339 ± 0.064 | 2.753 ± 0.302 | 4.043 ± 0.015 | - |
| cpu/complex | `qr` | c64 | 1 | `256x256` | 8.321 ± 0.117 | - | 8.570 ± 0.082 | 9.431 ± 0.058 | 7.920 ± 0.010 | 9.385 ± 0.450 | - |
| cpu/complex | `qr` | c64 | 4 | `256x256` | 5.888 ± 0.043 | - | 5.457 ± 0.066 | 5.791 ± 0.053 | 6.981 ± 0.214 | 7.164 ± 0.114 | - |
| cpu/complex | `solve` | c64 | 1 | `384x384,rhs=8` | 5.574 ± 0.161 | - | 5.876 ± 0.337 | 5.704 ± 0.030 | 5.397 ± 0.035 | 4.806 ± 0.188 | - |
| cpu/complex | `solve` | c64 | 4 | `384x384,rhs=8` | 3.894 ± 0.057 | - | 4.014 ± 0.119 | 3.630 ± 0.060 | 4.573 ± 0.025 | 4.281 ± 0.203 | - |
| cpu/complex | `svd` | c64 | 1 | `160x160` | 6.454 ± 0.065 | - | 6.640 ± 0.128 | 7.814 ± 0.082 | 5.383 ± 0.031 | 5.328 ± 0.074 | - |
| cpu/complex | `svd` | c64 | 4 | `160x160` | 5.578 ± 0.136 | - | 5.668 ± 0.158 | 5.860 ± 0.033 | 5.910 ± 0.203 | 5.682 ± 0.140 | - |
| cpu/complex | `tensordot` | c64 | 1 | `640x640` | 44.743 ± 0.217 | - | 44.661 ± 16.501 | 42.785 ± 0.622 | 65.989 ± 0.133 | 44.045 ± 0.145 | - |
| cpu/complex | `tensordot` | c64 | 4 | `640x640` | 29.978 ± 0.229 | - | 30.149 ± 0.449 | 29.520 ± 0.101 | 29.467 ± 0.755 | 22.304 ± 0.233 | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 1 | `1024x1024` | 50.137 ± 1.522 | - | unsupported | 51.968 ± 0.335 | 62.298 ± 5.566 | - | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 4 | `1024x1024` | 24.631 ± 0.143 | - | unsupported | 27.570 ± 0.636 | 35.049 ± 0.500 | - | - |
| cpu/elementwise_reduction | `abs` | f64 | 1 | `33554432` | 41.697 ± 1.019 | - | 114.532 ± 0.393 | 25.263 ± 0.504 | 29.250 ± 0.467 | 26.293 ± 0.294 | 37.842 ± 0.444 |
| cpu/elementwise_reduction | `abs` | f64 | 4 | `33554432` | 17.282 ± 0.132 | - | 16.679 ± 0.419 | 16.539 ± 1.233 | 17.139 ± 0.750 | 25.956 ± 0.661 | 17.997 ± 0.801 |
| cpu/elementwise_reduction | `add` | f64 | 1 | `33554432` | 32.861 ± 0.811 | - | 112.226 ± 0.999 | 31.399 ± 0.557 | 37.356 ± 1.272 | 33.370 ± 0.933 | 46.082 ± 39.719 |
| cpu/elementwise_reduction | `add` | f64 | 4 | `33554432` | 23.216 ± 1.785 | - | 21.949 ± 2.793 | 23.595 ± 0.414 | 23.058 ± 1.165 | 32.581 ± 1.097 | 24.166 ± 2.775 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 1 | `4194304` | 27.079 ± 0.367 | - | 56.171 ± 0.288 | 37.752 ± 0.227 | 38.681 ± 0.070 | 69.980 ± 0.721 | 75.363 ± 1.612 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 4 | `4194304` | 12.703 ± 0.394 | - | 25.803 ± 0.331 | 21.024 ± 0.924 | 19.852 ± 0.033 | 69.821 ± 0.170 | 29.790 ± 2.634 |
| cpu/elementwise_reduction | `clamp` | f64 | 1 | `8388608` | 8.423 ± 0.034 | - | 8.710 ± 0.428 | 10.436 ± 0.305 | 11.491 ± 0.268 | 11.196 ± 0.770 | 14.989 ± 1.350 |
| cpu/elementwise_reduction | `clamp` | f64 | 4 | `8388608` | 7.791 ± 0.481 | - | 7.700 ± 0.323 | 8.133 ± 0.394 | 7.452 ± 0.330 | 11.119 ± 0.776 | 9.404 ± 4.171 |
| cpu/elementwise_reduction | `compare_lt` | f64 | 1 | `33554432` | 17.705 ± 1.223 | - | 21.798 ± 0.395 | 25.009 ± 0.130 | 23.971 ± 0.740 | 17.813 ± 0.606 | - |
| cpu/elementwise_reduction | `compare_lt` | f64 | 4 | `33554432` | 13.738 ± 0.184 | - | 14.567 ± 0.650 | 15.790 ± 0.633 | 16.086 ± 0.746 | 16.896 ± 0.571 | - |
| cpu/elementwise_reduction | `cos` | f64 | 1 | `8388608` | 28.390 ± 0.075 | - | 28.747 ± 0.074 | 32.927 ± 0.132 | 84.023 ± 0.176 | 44.646 ± 1.485 | 47.622 ± 1.518 |
| cpu/elementwise_reduction | `cos` | f64 | 4 | `8388608` | 11.992 ± 0.204 | - | 12.046 ± 0.104 | 15.550 ± 0.118 | 42.272 ± 0.117 | 43.232 ± 8.747 | 24.349 ± 1.480 |
| cpu/elementwise_reduction | `div` | f64 | 1 | `33554432` | 62.536 ± 2.445 | - | 126.574 ± 0.538 | 31.630 ± 0.480 | 36.785 ± 0.789 | 33.102 ± 0.349 | 43.490 ± 0.547 |
| cpu/elementwise_reduction | `div` | f64 | 4 | `33554432` | 24.968 ± 1.065 | - | 25.201 ± 1.822 | 23.782 ± 2.515 | 22.917 ± 0.266 | 32.198 ± 1.136 | 24.377 ± 0.793 |
| cpu/elementwise_reduction | `exp` | f64 | 1 | `8388608` | 20.433 ± 0.106 | - | 20.454 ± 0.070 | 20.454 ± 0.087 | 17.068 ± 0.082 | 39.297 ± 1.497 | 44.551 ± 1.300 |
| cpu/elementwise_reduction | `exp` | f64 | 4 | `8388608` | 8.492 ± 0.049 | - | 8.725 ± 0.045 | 10.221 ± 0.180 | 8.805 ± 0.054 | 40.238 ± 1.883 | 21.689 ± 2.549 |
| cpu/elementwise_reduction | `expm1` | f64 | 1 | `4194304` | 20.465 ± 0.535 | - | 20.708 ± 0.043 | 21.001 ± 0.107 | 15.740 ± 0.303 | 26.890 ± 0.374 | 30.342 ± 1.149 |
| cpu/elementwise_reduction | `expm1` | f64 | 4 | `4194304` | 8.192 ± 0.119 | - | 8.172 ± 0.067 | 9.316 ± 0.133 | 8.105 ± 0.019 | 26.790 ± 0.594 | 13.144 ± 2.519 |
| cpu/elementwise_reduction | `log` | f64 | 1 | `8388608` | 25.221 ± 0.067 | - | 25.569 ± 0.035 | 26.883 ± 0.363 | 41.245 ± 0.246 | 60.351 ± 10.127 | 66.817 ± 0.811 |
| cpu/elementwise_reduction | `log` | f64 | 4 | `8388608` | 10.892 ± 0.117 | - | 10.637 ± 0.357 | 12.727 ± 0.122 | 20.491 ± 0.085 | 61.019 ± 1.770 | 30.074 ± 2.847 |
| cpu/elementwise_reduction | `log1p` | f64 | 1 | `4194304` | 13.225 ± 0.023 | - | 13.396 ± 0.112 | 21.485 ± 0.129 | 25.862 ± 0.070 | 35.406 ± 0.288 | 39.250 ± 1.143 |
| cpu/elementwise_reduction | `log1p` | f64 | 4 | `4194304` | 5.673 ± 0.043 | - | 5.753 ± 0.019 | 10.029 ± 0.718 | 13.691 ± 0.153 | 35.195 ± 0.363 | 17.802 ± 1.952 |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 1 | `1024x64` | 0.327 ± 0.011 | 0.375 ± 0.015 | unsupported | unsupported | 0.360 ± 0.004 | - | - |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 4 | `1024x64` | 0.474 ± 0.013 | 0.613 ± 0.023 | unsupported | unsupported | 0.323 ± 0.028 | - | - |
| cpu/elementwise_reduction | `maximum` | f64 | 1 | `33554432` | 48.940 ± 0.784 | - | 130.240 ± 1.765 | 32.261 ± 1.089 | 36.824 ± 1.082 | 33.004 ± 0.493 | 44.879 ± 0.198 |
| cpu/elementwise_reduction | `maximum` | f64 | 4 | `33554432` | 23.126 ± 1.639 | - | 24.425 ± 1.575 | 22.847 ± 1.724 | 22.112 ± 0.941 | 32.330 ± 1.195 | 25.868 ± 0.776 |
| cpu/elementwise_reduction | `minimum` | f64 | 1 | `33554432` | 50.178 ± 1.058 | - | 130.811 ± 0.826 | 31.675 ± 0.793 | 37.462 ± 0.474 | 33.182 ± 0.733 | 45.754 ± 0.539 |
| cpu/elementwise_reduction | `minimum` | f64 | 4 | `33554432` | 24.257 ± 1.264 | - | 24.654 ± 2.118 | 22.919 ± 0.964 | 23.589 ± 1.031 | 32.655 ± 0.818 | 24.819 ± 1.664 |
| cpu/elementwise_reduction | `mul` | f64 | 1 | `33554432` | 31.519 ± 0.490 | - | 122.651 ± 0.940 | 30.993 ± 0.296 | 37.081 ± 1.222 | 32.742 ± 0.389 | 43.251 ± 0.429 |
| cpu/elementwise_reduction | `mul` | f64 | 4 | `33554432` | 24.243 ± 2.204 | - | 23.601 ± 0.462 | 22.887 ± 0.201 | 23.035 ± 0.739 | 31.399 ± 1.855 | 24.850 ± 1.621 |
| cpu/elementwise_reduction | `neg` | f64 | 1 | `33554432` | 54.281 ± 0.964 | - | 114.344 ± 2.145 | 24.962 ± 0.578 | 29.676 ± 0.327 | 26.089 ± 0.615 | 37.948 ± 0.369 |
| cpu/elementwise_reduction | `neg` | f64 | 4 | `33554432` | 17.165 ± 0.857 | - | 15.889 ± 0.457 | 17.155 ± 0.844 | 16.573 ± 0.677 | 26.297 ± 0.274 | 18.384 ± 0.581 |
| cpu/elementwise_reduction | `pow` | f64 | 1 | `4194304` | 58.098 ± 0.111 | - | 58.289 ± 0.196 | 51.667 ± 0.089 | 61.824 ± 0.191 | 115.320 ± 1.463 | 111.507 ± 1.377 |
| cpu/elementwise_reduction | `pow` | f64 | 4 | `4194304` | 24.015 ± 0.476 | - | 23.766 ± 0.084 | 19.980 ± 0.278 | 31.038 ± 0.157 | 114.791 ± 0.524 | 46.752 ± 2.262 |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 1 | `8192x4096` | 15.380 ± 0.606 | - | 15.517 ± 0.358 | 10.758 ± 0.366 | 31.417 ± 0.204 | 14.056 ± 0.922 | - |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 4 | `8192x4096` | 6.296 ± 0.102 | - | 6.504 ± 0.146 | 6.756 ± 0.651 | 15.144 ± 0.078 | 12.591 ± 0.511 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 1 | `2048x2048` | 1.853 ± 0.016 | - | 1.844 ± 0.024 | 38.058 ± 0.731 | 2.717 ± 0.024 | 1.609 ± 0.195 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 4 | `2048x2048` | 0.813 ± 0.073 | - | 0.999 ± 0.225 | 11.737 ± 0.348 | 1.760 ± 0.040 | 1.522 ± 0.045 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 1 | `2048x2048` | 1.861 ± 0.060 | - | 1.807 ± 0.036 | 4.080 ± 0.030 | 3.287 ± 0.031 | 1.636 ± 0.171 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 4 | `2048x2048` | 0.978 ± 0.054 | - | 0.994 ± 0.087 | 2.305 ± 0.073 | 1.146 ± 0.071 | 1.579 ± 0.041 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 1 | `8192x4096` | 14.634 ± 0.192 | - | 14.611 ± 0.308 | 11.142 ± 0.181 | 31.444 ± 0.265 | 13.981 ± 0.276 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 4 | `8192x4096` | 6.292 ± 0.387 | - | 6.625 ± 0.165 | 6.352 ± 0.109 | 14.990 ± 0.072 | 12.766 ± 0.656 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 1 | `2048x2048` | 1.891 ± 0.044 | - | 1.889 ± 0.033 | 33.288 ± 1.753 | 2.678 ± 0.012 | 1.535 ± 0.074 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 4 | `2048x2048` | 0.810 ± 0.018 | - | 0.812 ± 0.039 | 10.663 ± 0.334 | 1.913 ± 0.345 | 1.515 ± 0.081 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 1 | `4096x4096` | 7.379 ± 0.182 | - | 7.602 ± 0.189 | 16.028 ± 0.129 | 12.972 ± 0.254 | 6.504 ± 0.328 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 4 | `4096x4096` | 4.057 ± 0.028 | - | 4.151 ± 0.528 | 8.762 ± 0.090 | 8.441 ± 0.351 | 6.347 ± 0.482 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 1 | `8192x4096` | 10.708 ± 0.209 | - | 11.176 ± 0.811 | 9.682 ± 0.265 | 29.862 ± 0.237 | 11.203 ± 0.373 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 4 | `8192x4096` | 6.179 ± 0.148 | - | 6.104 ± 0.160 | 7.185 ± 0.592 | 14.583 ± 0.141 | 10.020 ± 0.320 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 1 | `2048x2048` | 1.154 ± 0.073 | - | 1.211 ± 0.054 | 2.729 ± 0.371 | 2.238 ± 0.025 | 1.190 ± 0.123 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 4 | `2048x2048` | 0.441 ± 0.195 | - | 0.472 ± 0.127 | 0.770 ± 0.035 | 3.352 ± 1.273 | 0.985 ± 0.285 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 1 | `2048x2048` | 1.283 ± 0.021 | - | 1.338 ± 0.034 | 0.786 ± 0.017 | 4.363 ± 0.065 | 1.291 ± 0.174 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 4 | `2048x2048` | 0.706 ± 0.092 | - | 0.612 ± 0.031 | 0.321 ± 0.056 | 1.092 ± 0.068 | 1.200 ± 0.050 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 1 | `8192x4096` | 10.224 ± 0.217 | - | 10.487 ± 0.196 | 9.865 ± 0.573 | 29.758 ± 0.169 | 11.225 ± 0.325 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 4 | `8192x4096` | 6.308 ± 0.342 | - | 6.312 ± 0.276 | 6.754 ± 0.282 | 14.620 ± 0.157 | 9.637 ± 0.137 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 1 | `2048x2048` | 1.196 ± 0.054 | - | 1.191 ± 0.040 | 2.380 ± 0.032 | 2.233 ± 0.016 | 1.202 ± 0.105 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 4 | `2048x2048` | 0.409 ± 0.046 | - | 0.488 ± 0.077 | 0.815 ± 0.029 | 2.508 ± 1.915 | 0.995 ± 0.086 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 1 | `2048x2048` | 1.304 ± 0.050 | - | 1.318 ± 0.065 | 0.812 ± 0.048 | 4.172 ± 0.155 | 1.241 ± 0.190 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 4 | `2048x2048` | 0.697 ± 0.076 | - | 0.654 ± 0.072 | 0.364 ± 0.075 | 1.080 ± 0.038 | 1.148 ± 0.120 | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 1 | `2048x2048` | 1.292 ± 0.020 | 1.365 ± 0.026 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 4 | `2048x2048` | 0.413 ± 0.075 | 0.434 ± 0.051 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `rem` | f64 | 1 | `8388608` | 41.323 ± 0.131 | - | 37.587 ± 0.082 | 31.081 ± 0.129 | 44.999 ± 0.267 | 47.425 ± 0.569 | 54.609 ± 0.827 |
| cpu/elementwise_reduction | `rem` | f64 | 4 | `8388608` | 17.759 ± 0.160 | - | 17.929 ± 0.182 | 14.232 ± 0.157 | 24.474 ± 0.399 | 47.496 ± 2.265 | 26.410 ± 1.794 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 1 | `33554432` | 138.857 ± 0.682 | - | 138.354 ± 0.313 | 48.916 ± 0.094 | 154.680 ± 0.426 | 154.783 ± 0.379 | 154.967 ± 0.865 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 4 | `33554432` | 23.294 ± 2.034 | - | 23.561 ± 1.858 | 24.307 ± 1.648 | 76.823 ± 0.174 | 154.381 ± 0.442 | 76.168 ± 4.685 |
| cpu/elementwise_reduction | `select` | f64 | 1 | `33554432` | 62.950 ± 1.625 | - | 129.955 ± 0.722 | 48.724 ± 0.788 | 40.988 ± 2.222 | 34.122 ± 1.208 | - |
| cpu/elementwise_reduction | `select` | f64 | 4 | `33554432` | 28.300 ± 1.192 | - | 26.279 ± 1.062 | 23.428 ± 0.934 | 22.668 ± 1.188 | 32.975 ± 0.256 | - |
| cpu/elementwise_reduction | `sign` | f64 | 1 | `33554432` | 44.814 ± 0.590 | - | 117.653 ± 0.254 | 27.382 ± 0.313 | 31.334 ± 0.799 | 27.633 ± 0.524 | 33.830 ± 0.476 |
| cpu/elementwise_reduction | `sign` | f64 | 4 | `33554432` | 18.124 ± 1.790 | - | 18.632 ± 0.832 | 15.972 ± 0.227 | 16.490 ± 0.382 | 27.782 ± 0.963 | 17.246 ± 0.335 |
| cpu/elementwise_reduction | `sin` | f64 | 1 | `8388608` | 24.549 ± 0.093 | - | 25.009 ± 0.047 | 34.519 ± 0.061 | 83.489 ± 0.136 | 43.804 ± 1.073 | 44.925 ± 1.398 |
| cpu/elementwise_reduction | `sin` | f64 | 4 | `8388608` | 10.257 ± 0.047 | - | 10.312 ± 0.249 | 16.463 ± 0.161 | 42.011 ± 0.120 | 41.648 ± 2.028 | 22.932 ± 2.720 |
| cpu/elementwise_reduction | `sqrt` | f64 | 1 | `33554432` | 125.931 ± 0.484 | - | 125.969 ± 0.350 | 88.354 ± 0.197 | 40.488 ± 0.415 | 104.669 ± 0.364 | 110.029 ± 0.823 |
| cpu/elementwise_reduction | `sqrt` | f64 | 4 | `33554432` | 19.133 ± 1.604 | - | 19.124 ± 0.557 | 40.002 ± 1.048 | 19.321 ± 0.135 | 104.046 ± 0.777 | 51.748 ± 1.928 |
| cpu/elementwise_reduction | `sub` | f64 | 1 | `33554432` | 32.370 ± 0.661 | - | 123.735 ± 0.888 | 31.502 ± 0.589 | 36.385 ± 1.848 | 32.547 ± 0.391 | 43.309 ± 0.537 |
| cpu/elementwise_reduction | `sub` | f64 | 4 | `33554432` | 22.887 ± 2.203 | - | 23.959 ± 1.163 | 22.861 ± 0.933 | 22.569 ± 1.194 | 32.388 ± 0.623 | 24.509 ± 1.926 |
| cpu/elementwise_reduction | `tanh` | f64 | 1 | `8388608` | 51.528 ± 0.063 | - | 52.066 ± 0.097 | 53.429 ± 0.239 | 15.018 ± 0.095 | 57.243 ± 3.033 | 64.114 ± 6.599 |
| cpu/elementwise_reduction | `tanh` | f64 | 4 | `8388608` | 20.573 ± 0.117 | - | 20.665 ± 0.249 | 23.187 ± 0.282 | 7.794 ± 0.107 | 56.609 ± 1.659 | 28.599 ± 2.602 |
| cpu/elementwise_reduction | `where_select` | f64 | 1 | `33554432` | 126.708 ± 1.108 | 128.526 ± 0.348 | unsupported | 47.569 ± 8.707 | 40.559 ± 0.306 | - | - |
| cpu/elementwise_reduction | `where_select` | f64 | 4 | `33554432` | 27.586 ± 1.295 | 28.081 ± 4.698 | unsupported | 24.567 ± 0.633 | 24.046 ± 0.540 | - | - |
| cpu/indexing_layout | `concatenate` | f64 | 1 | `1048576+1048576` | 0.926 ± 0.083 | - | 2.078 ± 0.238 | 0.676 ± 0.042 | 0.808 ± 0.164 | 2.538 ± 0.755 | - |
| cpu/indexing_layout | `concatenate` | f64 | 4 | `1048576+1048576` | 0.342 ± 0.050 | - | 1.093 ± 0.239 | 0.323 ± 0.027 | 0.748 ± 0.188 | 1.286 ± 1.158 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 1 | `4194304 -> 2097152` | 0.788 ± 0.042 | - | 3.269 ± 0.161 | 0.887 ± 0.111 | 0.914 ± 0.153 | 1.271 ± 0.469 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 4 | `4194304 -> 2097152` | 0.284 ± 0.091 | - | 2.033 ± 0.077 | 0.277 ± 0.046 | 0.393 ± 0.164 | 2.242 ± 0.725 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 1 | `2097152` | 1.280 ± 0.121 | - | unsupported | 1.138 ± 0.011 | 1.152 ± 0.040 | 2.761 ± 0.560 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 4 | `2097152` | 0.697 ± 0.185 | - | unsupported | 0.528 ± 0.125 | 0.869 ± 0.026 | 2.581 ± 0.805 | - |
| cpu/indexing_layout | `gather` | f64 | 1 | `262144` | 0.403 ± 0.062 | - | 0.572 ± 0.028 | 0.708 ± 0.020 | 0.459 ± 0.038 | 1.280 ± 0.086 | - |
| cpu/indexing_layout | `gather` | f64 | 4 | `262144` | 0.393 ± 0.103 | - | 0.543 ± 0.038 | 0.346 ± 0.021 | 0.467 ± 0.037 | 1.240 ± 0.850 | - |
| cpu/indexing_layout | `pad` | f64 | 1 | `2097152` | 0.741 ± 0.035 | - | 1.720 ± 0.158 | 1.025 ± 0.134 | 1.159 ± 0.105 | - | - |
| cpu/indexing_layout | `pad` | f64 | 4 | `2097152` | 0.374 ± 0.109 | - | 0.972 ± 0.121 | 0.421 ± 0.045 | 1.050 ± 0.020 | - | - |
| cpu/indexing_layout | `reverse` | f64 | 1 | `2097152` | 0.972 ± 0.148 | - | 1.912 ± 0.173 | 0.727 ± 0.027 | 0.961 ± 0.070 | 1.431 ± 1.802 | - |
| cpu/indexing_layout | `reverse` | f64 | 4 | `2097152` | 0.332 ± 0.081 | - | 1.186 ± 0.205 | 0.285 ± 0.038 | 0.494 ± 0.090 | 2.571 ± 1.746 | - |
| cpu/indexing_layout | `scatter` | f64 | 1 | `262144` | 0.824 ± 0.028 | - | 1.058 ± 0.044 | 0.491 ± 0.029 | 0.469 ± 0.027 | 1.340 ± 0.044 | - |
| cpu/indexing_layout | `scatter` | f64 | 4 | `262144` | 0.885 ± 0.015 | - | 1.195 ± 0.056 | 0.612 ± 0.014 | 0.499 ± 0.071 | 1.307 ± 0.037 | - |
| cpu/indexing_layout | `slice` | f64 | 1 | `4194304 -> 2096128` | 1.725 ± 0.038 | - | 3.987 ± 0.170 | 1.681 ± 0.116 | 1.448 ± 0.098 | 3.742 ± 0.559 | - |
| cpu/indexing_layout | `slice` | f64 | 4 | `4194304 -> 2096128` | 0.777 ± 0.260 | - | 2.447 ± 0.147 | 0.702 ± 0.075 | 0.758 ± 0.030 | 3.521 ± 1.343 | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x2x2` | 0.050 ± 0.000 | - | 0.057 ± 0.000 | 0.075 ± 0.001 | 0.133 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x4x4` | 0.093 ± 0.001 | - | 0.105 ± 0.005 | 0.122 ± 0.004 | 0.185 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x8x8` | 0.236 ± 0.002 | - | 0.259 ± 0.017 | 0.279 ± 0.013 | 0.366 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x128x128` | 0.082 ± 0.001 | - | 0.093 ± 0.001 | 0.148 ± 0.003 | 0.155 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.012 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x32x32` | 0.004 ± 0.000 | - | 0.010 ± 0.000 | 0.009 ± 0.000 | 0.069 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x4x4` | 0.001 ± 0.000 | - | 0.007 ± 0.000 | 0.005 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x8x8` | 0.001 ± 0.000 | - | 0.007 ± 0.000 | 0.005 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x2x2` | 0.001 ± 0.000 | - | 0.007 ± 0.000 | 0.005 ± 0.000 | 0.012 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x4x4` | 0.001 ± 0.000 | - | 0.007 ± 0.000 | 0.005 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x8x8` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.006 ± 0.000 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x2x2` | 0.001 ± 0.000 | - | 0.007 ± 0.000 | 0.005 ± 0.000 | 0.012 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x4x4` | 0.001 ± 0.000 | - | 0.007 ± 0.000 | 0.005 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x8x8` | 0.002 ± 0.000 | - | 0.008 ± 0.000 | 0.006 ± 0.000 | 0.066 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x128x128` | 0.664 ± 0.014 | - | 0.720 ± 0.036 | 1.160 ± 0.011 | 0.710 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x2x2` | 0.001 ± 0.000 | - | 0.007 ± 0.000 | 0.005 ± 0.000 | 0.012 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x32x32` | 0.026 ± 0.002 | - | 0.034 ± 0.000 | 0.039 ± 0.001 | 0.100 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x4x4` | 0.002 ± 0.000 | - | 0.007 ± 0.000 | 0.006 ± 0.000 | 0.014 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x8x8` | 0.003 ± 0.000 | - | 0.009 ± 0.000 | 0.007 ± 0.000 | 0.074 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x2x2` | 0.050 ± 0.004 | - | 0.111 ± 0.001 | 0.085 ± 0.001 | 0.097 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x4x4` | 0.092 ± 0.001 | - | 0.159 ± 0.003 | 0.122 ± 0.002 | 0.151 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x8x8` | 0.237 ± 0.001 | - | 0.336 ± 0.043 | 0.338 ± 0.017 | 0.389 ± 0.042 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x128x128` | 0.080 ± 0.001 | - | 0.146 ± 0.006 | 0.177 ± 0.003 | 0.128 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x2x2` | 0.002 ± 0.002 | - | 0.033 ± 0.005 | 0.010 ± 0.003 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x32x32` | 0.004 ± 0.002 | - | 0.047 ± 0.017 | 0.016 ± 0.001 | 0.019 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x4x4` | 0.002 ± 0.003 | - | 0.033 ± 0.011 | 0.009 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x8x8` | 0.003 ± 0.003 | - | 0.034 ± 0.012 | 0.010 ± 0.001 | 0.011 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x2x2` | 0.003 ± 0.002 | - | 0.046 ± 0.031 | 0.009 ± 0.001 | 0.013 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x4x4` | 0.002 ± 0.003 | - | 0.038 ± 0.007 | 0.009 ± 0.001 | 0.011 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x8x8` | 0.004 ± 0.003 | - | 0.037 ± 0.020 | 0.010 ± 0.000 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x2x2` | 0.002 ± 0.000 | - | 0.036 ± 0.006 | 0.010 ± 0.000 | 0.011 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x4x4` | 0.002 ± 0.000 | - | 0.043 ± 0.020 | 0.010 ± 0.000 | 0.012 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x8x8` | 0.004 ± 0.001 | - | 0.038 ± 0.015 | 0.011 ± 0.000 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x128x128` | 0.690 ± 0.034 | - | 0.825 ± 0.045 | 1.386 ± 0.025 | 1.101 ± 2.347 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x2x2` | 0.003 ± 0.001 | - | 0.042 ± 0.026 | 0.010 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x32x32` | 0.025 ± 0.000 | - | 0.089 ± 0.003 | 0.058 ± 0.001 | 0.083 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x4x4` | 0.003 ± 0.002 | - | 0.049 ± 0.021 | 0.010 ± 0.002 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x8x8` | 0.003 ± 0.002 | - | 0.041 ± 0.014 | 0.012 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x2x2` | 0.623 ± 0.010 | - | 0.637 ± 0.014 | 0.724 ± 0.014 | 0.904 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x4x4` | 2.128 ± 0.010 | - | 2.163 ± 0.021 | 2.159 ± 0.015 | 2.726 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x8x8` | 6.520 ± 0.098 | - | 6.653 ± 0.110 | 6.548 ± 0.042 | 8.972 ± 0.116 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x128x128` | 3.710 ± 0.048 | - | 3.838 ± 0.071 | 3.228 ± 0.016 | 2.649 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x2x2` | 0.002 ± 0.001 | - | 0.009 ± 0.000 | 0.009 ± 0.001 | 0.025 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x32x32` | 0.109 ± 0.001 | - | 0.118 ± 0.012 | 0.123 ± 0.004 | 0.202 ± 0.028 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x4x4` | 0.004 ± 0.000 | - | 0.012 ± 0.001 | 0.012 ± 0.001 | 0.027 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x8x8` | 0.008 ± 0.000 | - | 0.015 ± 0.001 | 0.018 ± 0.002 | 0.033 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x2x2` | 0.004 ± 0.001 | - | 0.011 ± 0.000 | 0.011 ± 0.002 | 0.026 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x4x4` | 0.008 ± 0.000 | - | 0.015 ± 0.001 | 0.016 ± 0.001 | 0.030 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x8x8` | 0.021 ± 0.001 | - | 0.029 ± 0.000 | 0.030 ± 0.002 | 0.081 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x2x2` | 0.004 ± 0.000 | - | 0.011 ± 0.000 | 0.012 ± 0.000 | 0.025 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x4x4` | 0.010 ± 0.000 | - | 0.017 ± 0.000 | 0.018 ± 0.001 | 0.034 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x8x8` | 0.027 ± 0.001 | - | 0.035 ± 0.000 | 0.036 ± 0.000 | 0.088 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x128x128` | 29.660 ± 0.427 | - | 29.744 ± 0.299 | 25.701 ± 0.160 | 20.571 ± 0.049 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x2x2` | 0.006 ± 0.000 | - | 0.014 ± 0.000 | 0.014 ± 0.001 | 0.030 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x32x32` | 0.866 ± 0.006 | - | 0.887 ± 0.005 | 0.882 ± 0.008 | 1.031 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x4x4` | 0.018 ± 0.000 | - | 0.026 ± 0.000 | 0.027 ± 0.003 | 0.044 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x8x8` | 0.052 ± 0.001 | - | 0.060 ± 0.004 | 0.061 ± 0.001 | 0.122 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x2x2` | 1.111 ± 0.141 | - | 1.084 ± 0.003 | 0.714 ± 0.430 | 0.948 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x4x4` | 3.425 ± 0.012 | - | 3.501 ± 0.025 | 2.140 ± 0.028 | 2.248 ± 0.111 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x8x8` | 9.942 ± 1.696 | - | 6.664 ± 0.047 | 9.352 ± 0.204 | 5.912 ± 0.154 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x128x128` | 3.564 ± 0.026 | - | 3.612 ± 0.026 | 2.274 ± 0.191 | 3.394 ± 0.033 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x2x2` | 0.004 ± 0.000 | - | 0.037 ± 0.006 | 0.013 ± 0.001 | 0.022 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x32x32` | 0.109 ± 0.002 | - | 0.179 ± 0.010 | 0.176 ± 0.013 | 0.197 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x4x4` | 0.006 ± 0.000 | - | 0.041 ± 0.013 | 0.011 ± 0.001 | 0.025 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x8x8` | 0.013 ± 0.001 | - | 0.048 ± 0.003 | 0.016 ± 0.001 | 0.032 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x2x2` | 0.008 ± 0.002 | - | 0.050 ± 0.020 | 0.016 ± 0.001 | 0.024 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x4x4` | 0.013 ± 0.004 | - | 0.048 ± 0.002 | 0.015 ± 0.000 | 0.030 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x8x8` | 0.032 ± 0.000 | - | 0.072 ± 0.015 | 0.030 ± 0.002 | 0.080 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x2x2` | 0.007 ± 0.000 | - | 0.036 ± 0.013 | 0.017 ± 0.002 | 0.026 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x4x4` | 0.016 ± 0.000 | - | 0.057 ± 0.013 | 0.018 ± 0.001 | 0.033 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x8x8` | 0.041 ± 0.000 | - | 0.087 ± 0.012 | 0.035 ± 0.000 | 0.094 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x128x128` | 27.951 ± 0.396 | - | 28.636 ± 0.542 | 17.640 ± 0.051 | 28.774 ± 3.651 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x2x2` | 0.012 ± 0.008 | - | 0.049 ± 0.007 | 0.021 ± 0.001 | 0.030 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x32x32` | 0.881 ± 0.009 | - | 0.959 ± 0.023 | 1.246 ± 0.012 | 1.602 ± 0.559 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x4x4` | 0.029 ± 0.002 | - | 0.065 ± 0.012 | 0.027 ± 0.003 | 0.045 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x8x8` | 0.080 ± 0.001 | - | 0.121 ± 0.030 | 0.060 ± 0.001 | 0.127 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x2x2` | 0.685 ± 0.012 | - | 0.697 ± 0.014 | 0.523 ± 0.015 | 0.452 ± 0.029 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x4x4` | 2.205 ± 0.011 | - | 2.227 ± 0.011 | 1.949 ± 0.006 | 1.976 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x8x8` | 5.809 ± 0.022 | - | 5.892 ± 0.024 | 5.735 ± 0.035 | 6.474 ± 0.028 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x128x128` | 1.340 ± 0.021 | - | 1.359 ± 0.025 | 1.358 ± 0.011 | 1.516 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x2x2` | 0.003 ± 0.001 | - | 0.009 ± 0.000 | 0.008 ± 0.001 | 0.021 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x32x32` | 0.070 ± 0.003 | - | 0.081 ± 0.002 | 0.090 ± 0.003 | 0.141 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x4x4` | 0.004 ± 0.000 | - | 0.011 ± 0.000 | 0.011 ± 0.001 | 0.023 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x8x8` | 0.007 ± 0.000 | - | 0.015 ± 0.000 | 0.016 ± 0.002 | 0.029 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x2x2` | 0.004 ± 0.000 | - | 0.010 ± 0.000 | 0.010 ± 0.002 | 0.022 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x4x4` | 0.008 ± 0.000 | - | 0.016 ± 0.000 | 0.014 ± 0.000 | 0.026 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x8x8` | 0.019 ± 0.000 | - | 0.026 ± 0.000 | 0.027 ± 0.004 | 0.043 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x2x2` | 0.004 ± 0.000 | - | 0.012 ± 0.002 | 0.010 ± 0.000 | 0.022 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x4x4` | 0.010 ± 0.000 | - | 0.018 ± 0.000 | 0.017 ± 0.000 | 0.031 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x8x8` | 0.024 ± 0.000 | - | 0.032 ± 0.000 | 0.032 ± 0.001 | 0.051 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x128x128` | 10.668 ± 0.257 | - | 10.733 ± 0.278 | 10.751 ± 0.106 | 11.321 ± 0.097 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x2x2` | 0.007 ± 0.000 | - | 0.014 ± 0.000 | 0.012 ± 0.000 | 0.025 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x32x32` | 0.552 ± 0.017 | - | 0.574 ± 0.019 | 0.632 ± 0.018 | 0.759 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x4x4` | 0.019 ± 0.000 | - | 0.026 ± 0.003 | 0.024 ± 0.001 | 0.037 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x8x8` | 0.046 ± 0.000 | - | 0.055 ± 0.000 | 0.053 ± 0.003 | 0.101 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x2x2` | 1.128 ± 0.056 | - | 1.187 ± 0.024 | 0.799 ± 0.085 | 0.479 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x4x4` | 3.537 ± 0.011 | - | 3.608 ± 0.028 | 1.926 ± 0.005 | 1.287 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x8x8` | 8.996 ± 0.030 | - | 5.954 ± 0.007 | 8.507 ± 0.030 | 3.416 ± 0.465 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x128x128` | 1.249 ± 0.032 | - | 1.331 ± 0.021 | 1.198 ± 0.017 | 1.723 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x2x2` | 0.004 ± 0.001 | - | 0.044 ± 0.016 | 0.012 ± 0.001 | 0.020 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x32x32` | 0.115 ± 0.013 | - | 0.173 ± 0.015 | 0.137 ± 0.005 | 0.129 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x4x4` | 0.007 ± 0.002 | - | 0.045 ± 0.014 | 0.011 ± 0.001 | 0.022 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x8x8` | 0.012 ± 0.004 | - | 0.051 ± 0.020 | 0.014 ± 0.000 | 0.028 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x2x2` | 0.006 ± 0.005 | - | 0.040 ± 0.010 | 0.015 ± 0.004 | 0.021 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x4x4` | 0.013 ± 0.002 | - | 0.056 ± 0.015 | 0.015 ± 0.001 | 0.026 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x8x8` | 0.029 ± 0.000 | - | 0.067 ± 0.006 | 0.026 ± 0.001 | 0.042 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x2x2` | 0.007 ± 0.000 | - | 0.040 ± 0.013 | 0.015 ± 0.001 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x4x4` | 0.017 ± 0.000 | - | 0.054 ± 0.009 | 0.017 ± 0.001 | 0.028 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x8x8` | 0.037 ± 0.000 | - | 0.086 ± 0.018 | 0.031 ± 0.001 | 0.050 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x128x128` | 9.824 ± 0.193 | - | 10.084 ± 0.176 | 9.058 ± 0.056 | 14.247 ± 1.431 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x2x2` | 0.011 ± 0.004 | - | 0.051 ± 0.007 | 0.018 ± 0.001 | 0.023 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x32x32` | 0.884 ± 0.009 | - | 0.970 ± 0.020 | 0.928 ± 0.005 | 0.477 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x4x4` | 0.030 ± 0.002 | - | 0.068 ± 0.007 | 0.023 ± 0.001 | 0.038 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x8x8` | 0.072 ± 0.001 | - | 0.108 ± 0.014 | 0.052 ± 0.001 | 0.101 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x2x2` | 0.266 ± 0.014 | - | 0.273 ± 0.012 | 0.238 ± 0.018 | 0.450 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x4x4` | 1.099 ± 0.017 | - | 1.117 ± 0.018 | 0.950 ± 0.002 | 1.959 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x8x8` | 3.096 ± 0.027 | - | 3.132 ± 0.019 | 2.885 ± 0.006 | 6.522 ± 0.055 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x128x128` | 0.523 ± 0.013 | - | 0.536 ± 0.017 | 0.556 ± 0.015 | 1.440 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x2x2` | 0.003 ± 0.003 | - | 0.008 ± 0.004 | 0.010 ± 0.001 | 0.016 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x32x32` | 0.031 ± 0.000 | - | 0.038 ± 0.001 | 0.042 ± 0.001 | 0.141 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.009 ± 0.000 | 0.011 ± 0.001 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x8x8` | 0.004 ± 0.000 | - | 0.011 ± 0.001 | 0.012 ± 0.001 | 0.023 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.008 ± 0.001 | 0.010 ± 0.001 | 0.015 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x4x4` | 0.005 ± 0.000 | - | 0.011 ± 0.000 | 0.013 ± 0.001 | 0.021 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x8x8` | 0.010 ± 0.000 | - | 0.018 ± 0.004 | 0.020 ± 0.001 | 0.035 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.008 ± 0.000 | 0.011 ± 0.001 | 0.016 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x4x4` | 0.006 ± 0.000 | - | 0.012 ± 0.005 | 0.014 ± 0.001 | 0.023 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x8x8` | 0.013 ± 0.000 | - | 0.020 ± 0.000 | 0.022 ± 0.001 | 0.040 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x128x128` | 4.174 ± 0.033 | - | 4.220 ± 0.048 | 4.295 ± 0.006 | 11.036 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | 0.010 ± 0.000 | 0.011 ± 0.000 | 0.017 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x32x32` | 0.230 ± 0.001 | - | 0.241 ± 0.014 | 0.245 ± 0.017 | 0.745 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x4x4` | 0.010 ± 0.000 | - | 0.016 ± 0.001 | 0.017 ± 0.001 | 0.032 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x8x8` | 0.025 ± 0.000 | - | 0.032 ± 0.003 | 0.033 ± 0.001 | 0.130 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x2x2` | 0.472 ± 0.030 | - | 0.528 ± 0.021 | 0.400 ± 0.018 | 0.449 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x4x4` | 1.609 ± 0.016 | - | 1.672 ± 0.029 | 0.948 ± 0.001 | 1.357 ± 0.587 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x8x8` | 4.195 ± 0.004 | - | 4.286 ± 0.028 | 3.666 ± 0.011 | 3.623 ± 0.494 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x128x128` | 0.681 ± 0.021 | - | 0.754 ± 0.038 | 0.694 ± 0.019 | 1.798 ± 0.076 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x2x2` | 0.003 ± 0.002 | - | 0.026 ± 0.008 | 0.014 ± 0.002 | 0.018 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x32x32` | 0.039 ± 0.001 | - | 0.073 ± 0.011 | 0.056 ± 0.007 | 0.112 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x4x4` | 0.005 ± 0.001 | - | 0.046 ± 0.027 | 0.016 ± 0.001 | 0.016 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x8x8` | 0.006 ± 0.001 | - | 0.043 ± 0.017 | 0.013 ± 0.001 | 0.025 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x2x2` | 0.005 ± 0.001 | - | 0.036 ± 0.018 | 0.015 ± 0.001 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x4x4` | 0.009 ± 0.006 | - | 0.045 ± 0.021 | 0.018 ± 0.003 | 0.020 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x8x8` | 0.015 ± 0.003 | - | 0.045 ± 0.020 | 0.019 ± 0.001 | 0.033 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x2x2` | 0.006 ± 0.003 | - | 0.034 ± 0.018 | 0.016 ± 0.002 | 0.015 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x4x4` | 0.010 ± 0.003 | - | 0.041 ± 0.008 | 0.020 ± 0.004 | 0.022 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x8x8` | 0.018 ± 0.000 | - | 0.051 ± 0.003 | 0.022 ± 0.001 | 0.041 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x128x128` | 5.333 ± 0.016 | - | 5.529 ± 0.048 | 5.179 ± 0.018 | 13.687 ± 1.514 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x2x2` | 0.007 ± 0.000 | - | 0.037 ± 0.002 | 0.017 ± 0.002 | 0.019 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x32x32` | 0.294 ± 0.017 | - | 0.352 ± 0.031 | 0.318 ± 0.020 | 0.487 ± 0.120 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x4x4` | 0.014 ± 0.000 | - | 0.052 ± 0.021 | 0.025 ± 0.004 | 0.032 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x8x8` | 0.034 ± 0.000 | - | 0.070 ± 0.008 | 0.033 ± 0.001 | 0.101 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x2x2` | 0.156 ± 0.002 | - | 0.160 ± 0.015 | 0.189 ± 0.003 | 0.128 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x4x4` | 0.226 ± 0.012 | - | 0.238 ± 0.018 | 0.256 ± 0.020 | 0.218 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x8x8` | 0.535 ± 0.012 | - | 0.556 ± 0.020 | 0.549 ± 0.019 | 0.587 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x128x128` | 0.133 ± 0.003 | - | 0.149 ± 0.004 | 0.176 ± 0.017 | 0.216 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x2x2` | 0.003 ± 0.000 | - | 0.010 ± 0.011 | 0.014 ± 0.001 | 0.023 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x32x32` | 0.012 ± 0.000 | - | 0.021 ± 0.001 | 0.025 ± 0.001 | 0.067 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x4x4` | 0.003 ± 0.000 | - | 0.009 ± 0.000 | 0.013 ± 0.001 | 0.019 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x64x24` | 0.014 ± 0.001 | - | 0.023 ± 0.002 | 0.028 ± 0.001 | 0.065 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x8x8` | 0.003 ± 0.000 | - | 0.010 ± 0.000 | 0.013 ± 0.001 | 0.022 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x2x2` | 0.003 ± 0.000 | - | 0.010 ± 0.000 | 0.016 ± 0.000 | 0.020 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x4x4` | 0.003 ± 0.000 | - | 0.010 ± 0.003 | 0.016 ± 0.000 | 0.025 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x8x8` | 0.004 ± 0.000 | - | 0.016 ± 0.016 | 0.018 ± 0.000 | 0.051 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x2x2` | 0.003 ± 0.001 | - | 0.010 ± 0.000 | 0.016 ± 0.001 | 0.024 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x4x4` | 0.003 ± 0.000 | - | 0.010 ± 0.000 | 0.016 ± 0.001 | 0.026 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x8x8` | 0.004 ± 0.000 | - | 0.011 ± 0.000 | 0.018 ± 0.001 | 0.052 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x128x128` | 1.076 ± 0.073 | - | 1.145 ± 0.011 | 1.303 ± 0.042 | 1.273 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | 0.010 ± 0.000 | 0.017 ± 0.001 | 0.025 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x32x32` | 0.077 ± 0.001 | - | 0.102 ± 0.025 | 0.088 ± 0.001 | 0.135 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x4x4` | 0.004 ± 0.000 | - | 0.011 ± 0.000 | 0.017 ± 0.000 | 0.046 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x8x8` | 0.006 ± 0.000 | - | 0.014 ± 0.004 | 0.020 ± 0.000 | 0.045 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x2x2` | 0.207 ± 0.016 | - | 0.245 ± 0.042 | 0.170 ± 0.007 | 0.134 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x4x4` | 0.316 ± 0.016 | - | 0.377 ± 0.020 | 0.232 ± 0.018 | 0.231 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x8x8` | 0.807 ± 0.023 | - | 0.897 ± 0.020 | 0.385 ± 0.022 | 0.685 ± 0.050 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x128x128` | 0.194 ± 0.010 | - | 0.245 ± 0.023 | 0.226 ± 0.019 | 0.215 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x2x2` | 0.004 ± 0.000 | - | 0.025 ± 0.003 | 0.024 ± 0.003 | 0.019 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x32x32` | 0.019 ± 0.000 | - | 0.055 ± 0.007 | 0.037 ± 0.008 | 0.060 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x4x4` | 0.005 ± 0.001 | - | 0.030 ± 0.008 | 0.023 ± 0.005 | 0.020 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x64x24` | 0.023 ± 0.002 | - | 0.057 ± 0.007 | 0.041 ± 0.009 | 0.061 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x8x8` | 0.005 ± 0.004 | - | 0.036 ± 0.004 | 0.023 ± 0.004 | 0.019 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x2x2` | 0.005 ± 0.001 | - | 0.040 ± 0.010 | 0.025 ± 0.001 | 0.019 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x4x4` | 0.007 ± 0.003 | - | 0.036 ± 0.009 | 0.025 ± 0.000 | 0.023 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x8x8` | 0.007 ± 0.001 | - | 0.039 ± 0.004 | 0.027 ± 0.001 | 0.028 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x2x2` | 0.006 ± 0.001 | - | 0.036 ± 0.005 | 0.026 ± 0.004 | 0.018 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x4x4` | 0.005 ± 0.001 | - | 0.042 ± 0.012 | 0.025 ± 0.000 | 0.025 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x8x8` | 0.009 ± 0.003 | - | 0.039 ± 0.007 | 0.028 ± 0.001 | 0.020 ± 0.028 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x128x128` | 1.517 ± 0.034 | - | 1.650 ± 0.025 | 0.699 ± 0.018 | 0.967 ± 0.159 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x2x2` | 0.005 ± 0.000 | - | 0.033 ± 0.013 | 0.026 ± 0.001 | 0.023 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x32x32` | 0.125 ± 0.002 | - | 0.173 ± 0.014 | 0.077 ± 0.011 | 0.182 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x4x4` | 0.007 ± 0.002 | - | 0.040 ± 0.003 | 0.027 ± 0.000 | 0.020 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x8x8` | 0.010 ± 0.000 | - | 0.041 ± 0.011 | 0.030 ± 0.000 | 0.051 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x2x2` | 0.251 ± 0.019 | - | 0.259 ± 0.022 | 0.225 ± 0.014 | 0.366 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x4x4` | 0.609 ± 0.014 | - | 0.628 ± 0.020 | 0.536 ± 0.020 | 0.915 ± 0.041 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x8x8` | 1.341 ± 0.025 | - | 1.365 ± 0.013 | 1.363 ± 0.016 | 2.367 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x128x128` | 0.499 ± 0.022 | - | 0.517 ± 0.016 | 0.586 ± 0.032 | 0.664 ± 0.028 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.009 ± 0.000 | 0.006 ± 0.001 | 0.020 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x32x32` | 0.018 ± 0.000 | - | 0.025 ± 0.001 | 0.027 ± 0.001 | 0.100 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.009 ± 0.000 | 0.006 ± 0.000 | 0.021 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x64x24` | 0.019 ± 0.001 | - | 0.028 ± 0.004 | 0.030 ± 0.001 | 0.072 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x8x8` | 0.003 ± 0.000 | - | 0.009 ± 0.000 | 0.007 ± 0.000 | 0.023 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.009 ± 0.000 | 0.006 ± 0.000 | 0.020 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x4x4` | 0.003 ± 0.000 | - | 0.010 ± 0.000 | 0.007 ± 0.000 | 0.022 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x8x8` | 0.006 ± 0.000 | - | 0.012 ± 0.000 | 0.009 ± 0.003 | 0.029 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | 0.009 ± 0.000 | 0.006 ± 0.000 | 0.022 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x4x4` | 0.004 ± 0.000 | - | 0.011 ± 0.000 | 0.007 ± 0.000 | 0.023 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x8x8` | 0.007 ± 0.000 | - | 0.013 ± 0.000 | 0.011 ± 0.000 | 0.030 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x128x128` | 4.097 ± 0.038 | - | 4.946 ± 0.046 | 4.500 ± 0.047 | 4.727 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x2x2` | 0.004 ± 0.000 | - | 0.010 ± 0.000 | 0.007 ± 0.000 | 0.022 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x32x32` | 0.128 ± 0.002 | - | 0.139 ± 0.003 | 0.175 ± 0.005 | 0.252 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x4x4` | 0.006 ± 0.000 | - | 0.013 ± 0.000 | 0.009 ± 0.000 | 0.029 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x8x8` | 0.012 ± 0.000 | - | 0.019 ± 0.000 | 0.016 ± 0.000 | 0.039 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x2x2` | 0.256 ± 0.044 | - | 0.324 ± 0.021 | 0.320 ± 0.018 | 0.334 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x4x4` | 0.633 ± 0.032 | - | 0.696 ± 0.030 | 0.762 ± 0.015 | 0.872 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x8x8` | 1.338 ± 0.036 | - | 1.474 ± 0.028 | 2.022 ± 0.017 | 1.363 ± 0.150 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x128x128` | 0.556 ± 0.033 | - | 0.611 ± 0.048 | 0.513 ± 0.019 | 0.981 ± 0.090 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.039 ± 0.027 | 0.010 ± 0.001 | 0.024 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x32x32` | 0.018 ± 0.001 | - | 0.078 ± 0.004 | 0.044 ± 0.001 | 0.077 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x4x4` | 0.004 ± 0.002 | - | 0.052 ± 0.022 | 0.011 ± 0.001 | 0.020 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x64x24` | 0.031 ± 0.000 | - | 0.067 ± 0.011 | 0.047 ± 0.003 | 0.071 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x8x8` | 0.006 ± 0.003 | - | 0.053 ± 0.026 | 0.012 ± 0.001 | 0.024 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x2x2` | 0.005 ± 0.001 | - | 0.039 ± 0.038 | 0.010 ± 0.000 | 0.019 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x4x4` | 0.007 ± 0.004 | - | 0.057 ± 0.023 | 0.012 ± 0.001 | 0.022 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x8x8` | 0.010 ± 0.008 | - | 0.050 ± 0.012 | 0.016 ± 0.000 | 0.027 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x2x2` | 0.005 ± 0.003 | - | 0.056 ± 0.023 | 0.011 ± 0.001 | 0.021 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x4x4` | 0.008 ± 0.004 | - | 0.045 ± 0.002 | 0.013 ± 0.000 | 0.022 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x8x8` | 0.013 ± 0.006 | - | 0.049 ± 0.001 | 0.018 ± 0.000 | 0.030 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x128x128` | 4.315 ± 0.109 | - | 5.110 ± 0.014 | 3.743 ± 0.050 | 7.395 ± 0.915 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x2x2` | 0.007 ± 0.002 | - | 0.055 ± 0.005 | 0.012 ± 0.001 | 0.023 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x32x32` | 0.128 ± 0.002 | - | 0.193 ± 0.005 | 0.289 ± 0.014 | 0.231 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x4x4` | 0.011 ± 0.003 | - | 0.048 ± 0.016 | 0.015 ± 0.000 | 0.026 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x8x8` | 0.012 ± 0.011 | - | 0.069 ± 0.007 | 0.026 ± 0.000 | 0.038 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.160 ± 0.011 | - | 0.172 ± 0.022 | 0.099 ± 0.002 | 0.293 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.221 ± 0.011 | - | 0.232 ± 0.016 | 0.141 ± 0.002 | 0.416 ± 0.030 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.572 ± 0.030 | - | 0.557 ± 0.031 | 0.406 ± 0.019 | 0.801 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x128x128,rhs=1` | 0.116 ± 0.003 | - | 0.134 ± 0.006 | 0.148 ± 0.008 | 0.226 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.010 ± 0.000 | 0.019 ± 0.002 | 0.018 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x32x32,rhs=1` | 0.011 ± 0.001 | - | 0.021 ± 0.000 | 0.031 ± 0.003 | 0.061 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.010 ± 0.000 | 0.018 ± 0.003 | 0.016 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x8x8,rhs=1` | 0.002 ± 0.000 | - | 0.010 ± 0.000 | 0.018 ± 0.000 | 0.016 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.010 ± 0.000 | 0.018 ± 0.001 | 0.025 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.010 ± 0.000 | 0.018 ± 0.001 | 0.019 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x8x8,rhs=1` | 0.003 ± 0.000 | - | 0.011 ± 0.000 | 0.020 ± 0.000 | 0.021 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.010 ± 0.000 | 0.018 ± 0.003 | 0.021 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.011 ± 0.000 | 0.018 ± 0.000 | 0.018 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x8x8,rhs=1` | 0.004 ± 0.000 | - | 0.012 ± 0.000 | 0.020 ± 0.004 | 0.021 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x128x128,rhs=1` | 0.934 ± 0.019 | - | 0.978 ± 0.024 | 1.033 ± 0.006 | 1.058 ± 0.028 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.011 ± 0.000 | 0.018 ± 0.001 | 0.019 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x32x32,rhs=1` | 0.071 ± 0.003 | - | 0.084 ± 0.004 | 0.082 ± 0.006 | 0.165 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.011 ± 0.000 | 0.018 ± 0.001 | 0.021 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x8x8,rhs=1` | 0.006 ± 0.000 | - | 0.014 ± 0.000 | 0.022 ± 0.001 | 0.025 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.165 ± 0.002 | - | 0.212 ± 0.015 | 0.130 ± 0.001 | 0.288 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.224 ± 0.024 | - | 0.269 ± 0.030 | 0.171 ± 0.003 | 0.384 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.569 ± 0.037 | - | 0.659 ± 0.037 | 0.382 ± 0.019 | 0.875 ± 0.049 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x128x128,rhs=1` | 0.121 ± 0.002 | - | 0.261 ± 0.065 | 0.203 ± 0.022 | 0.188 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.053 ± 0.020 | 0.026 ± 0.001 | 0.015 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x32x32,rhs=1` | 0.011 ± 0.002 | - | 0.079 ± 0.006 | 0.043 ± 0.005 | 0.059 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x4x4,rhs=1` | 0.003 ± 0.002 | - | 0.059 ± 0.034 | 0.026 ± 0.004 | 0.015 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x8x8,rhs=1` | 0.004 ± 0.004 | - | 0.055 ± 0.006 | 0.026 ± 0.004 | 0.016 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.048 ± 0.016 | 0.026 ± 0.001 | 0.016 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x4x4,rhs=1` | 0.002 ± 0.002 | - | 0.053 ± 0.023 | 0.027 ± 0.005 | 0.022 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x8x8,rhs=1` | 0.006 ± 0.003 | - | 0.052 ± 0.010 | 0.029 ± 0.003 | 0.021 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x2x2,rhs=1` | 0.004 ± 0.001 | - | 0.053 ± 0.018 | 0.026 ± 0.004 | 0.017 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x4x4,rhs=1` | 0.005 ± 0.004 | - | 0.057 ± 0.020 | 0.027 ± 0.003 | 0.018 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x8x8,rhs=1` | 0.004 ± 0.004 | - | 0.061 ± 0.010 | 0.029 ± 0.003 | 0.022 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x128x128,rhs=1` | 0.947 ± 0.009 | - | 1.113 ± 0.026 | 0.580 ± 0.017 | 0.800 ± 0.069 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.052 ± 0.025 | 0.028 ± 0.001 | 0.019 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x32x32,rhs=1` | 0.070 ± 0.001 | - | 0.160 ± 0.020 | 0.072 ± 0.001 | 0.215 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x4x4,rhs=1` | 0.005 ± 0.001 | - | 0.055 ± 0.009 | 0.029 ± 0.004 | 0.020 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x8x8,rhs=1` | 0.013 ± 0.003 | - | 0.072 ± 0.020 | 0.032 ± 0.001 | 0.024 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x2x2` | 1.840 ± 0.035 | - | 1.875 ± 0.031 | 1.572 ± 0.010 | 2.083 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x4x4` | 6.065 ± 0.038 | - | 6.131 ± 0.044 | 5.155 ± 0.013 | 5.190 ± 0.035 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x8x8` | 17.911 ± 0.079 | - | 18.050 ± 0.082 | 16.180 ± 0.049 | 14.878 ± 0.068 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x128x128` | 5.107 ± 0.065 | - | 5.178 ± 0.062 | 5.659 ± 0.065 | 4.240 ± 0.036 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x2x2` | 0.004 ± 0.000 | - | 0.014 ± 0.000 | 0.015 ± 0.001 | 0.029 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x32x32` | 0.211 ± 0.019 | - | 0.221 ± 0.025 | 0.235 ± 0.021 | 0.290 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x4x4` | 0.009 ± 0.000 | - | 0.018 ± 0.000 | 0.020 ± 0.001 | 0.034 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x64x24` | 0.181 ± 0.023 | - | 0.199 ± 0.018 | 0.210 ± 0.021 | 0.234 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x8x8` | 0.021 ± 0.006 | - | 0.032 ± 0.007 | 0.033 ± 0.001 | 0.045 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x2x2` | 0.008 ± 0.000 | - | 0.017 ± 0.001 | 0.018 ± 0.001 | 0.032 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x4x4` | 0.020 ± 0.000 | - | 0.030 ± 0.004 | 0.031 ± 0.007 | 0.044 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x8x8` | 0.054 ± 0.000 | - | 0.065 ± 0.000 | 0.065 ± 0.005 | 0.075 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x2x2` | 0.010 ± 0.000 | - | 0.019 ± 0.003 | 0.020 ± 0.001 | 0.035 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x4x4` | 0.026 ± 0.000 | - | 0.036 ± 0.004 | 0.036 ± 0.001 | 0.049 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x8x8` | 0.072 ± 0.000 | - | 0.083 ± 0.001 | 0.082 ± 0.005 | 0.090 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x128x128` | 40.385 ± 0.280 | - | 42.130 ± 0.279 | 44.314 ± 0.247 | 33.201 ± 0.073 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x2x2` | 0.017 ± 0.000 | - | 0.027 ± 0.002 | 0.026 ± 0.005 | 0.046 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x32x32` | 1.563 ± 0.021 | - | 1.593 ± 0.038 | 1.651 ± 0.012 | 1.611 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x4x4` | 0.049 ± 0.000 | - | 0.060 ± 0.001 | 0.057 ± 0.001 | 0.070 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x8x8` | 0.141 ± 0.002 | - | 0.152 ± 0.008 | 0.142 ± 0.021 | 0.206 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x2x2` | 3.033 ± 0.012 | - | 3.144 ± 0.142 | 1.516 ± 0.015 | 2.036 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x4x4` | 9.796 ± 0.019 | - | 6.144 ± 0.057 | 5.086 ± 0.041 | 2.821 ± 0.054 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x8x8` | 17.830 ± 0.076 | - | 18.007 ± 0.085 | 18.578 ± 0.096 | 8.345 ± 0.088 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x128x128` | 4.718 ± 0.041 | - | 4.820 ± 0.114 | 4.632 ± 0.022 | 4.894 ± 0.276 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x2x2` | 0.007 ± 0.000 | - | 0.044 ± 0.006 | 0.018 ± 0.003 | 0.029 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x32x32` | 0.311 ± 0.029 | - | 0.376 ± 0.047 | 0.333 ± 0.025 | 0.271 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x4x4` | 0.015 ± 0.001 | - | 0.046 ± 0.015 | 0.019 ± 0.001 | 0.043 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x64x24` | 0.274 ± 0.035 | - | 0.340 ± 0.027 | 0.291 ± 0.024 | 0.219 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x8x8` | 0.021 ± 0.001 | - | 0.089 ± 0.001 | 0.032 ± 0.001 | 0.044 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x2x2` | 0.013 ± 0.005 | - | 0.049 ± 0.006 | 0.027 ± 0.003 | 0.032 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x4x4` | 0.039 ± 0.008 | - | 0.074 ± 0.009 | 0.029 ± 0.001 | 0.043 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x8x8` | 0.054 ± 0.000 | - | 0.128 ± 0.017 | 0.063 ± 0.001 | 0.072 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x2x2` | 0.016 ± 0.004 | - | 0.041 ± 0.021 | 0.026 ± 0.002 | 0.033 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x4x4` | 0.042 ± 0.000 | - | 0.078 ± 0.013 | 0.034 ± 0.001 | 0.048 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x8x8` | 0.073 ± 0.001 | - | 0.146 ± 0.006 | 0.079 ± 0.001 | 0.090 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x128x128` | 37.260 ± 0.085 | - | 39.092 ± 0.443 | 36.514 ± 0.292 | 60.350 ± 38.252 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x2x2` | 0.027 ± 0.001 | - | 0.054 ± 0.007 | 0.040 ± 0.005 | 0.042 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x32x32` | 2.362 ± 0.019 | - | 2.510 ± 0.338 | 2.347 ± 0.031 | 0.887 ± 0.128 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x4x4` | 0.080 ± 0.002 | - | 0.137 ± 0.038 | 0.055 ± 0.005 | 0.070 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x8x8` | 0.141 ± 0.002 | - | 0.212 ± 0.010 | 0.140 ± 0.005 | 0.183 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x2x2` | 1.361 ± 0.016 | - | 1.402 ± 0.059 | 1.093 ± 0.001 | 1.382 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x4x4` | 5.180 ± 0.025 | - | 5.251 ± 0.041 | 4.358 ± 0.014 | 4.108 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x8x8` | 15.258 ± 0.120 | - | 15.254 ± 0.036 | 13.886 ± 0.029 | 12.289 ± 0.050 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x128x128` | 2.319 ± 0.029 | - | 2.343 ± 0.021 | 2.513 ± 0.050 | 2.403 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x2x2` | 0.004 ± 0.000 | - | 0.012 ± 0.001 | 0.014 ± 0.001 | 0.026 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x32x32` | 0.136 ± 0.016 | - | 0.148 ± 0.020 | 0.168 ± 0.007 | 0.196 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x4x4` | 0.008 ± 0.000 | - | 0.016 ± 0.001 | 0.019 ± 0.007 | 0.033 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x64x24` | 0.109 ± 0.014 | - | 0.121 ± 0.004 | 0.137 ± 0.007 | 0.203 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x8x8` | 0.017 ± 0.001 | - | 0.025 ± 0.001 | 0.029 ± 0.002 | 0.041 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x2x2` | 0.006 ± 0.000 | - | 0.014 ± 0.001 | 0.016 ± 0.000 | 0.029 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x4x4` | 0.017 ± 0.000 | - | 0.026 ± 0.000 | 0.027 ± 0.001 | 0.040 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x8x8` | 0.043 ± 0.000 | - | 0.053 ± 0.002 | 0.052 ± 0.002 | 0.064 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x2x2` | 0.008 ± 0.000 | - | 0.016 ± 0.001 | 0.017 ± 0.001 | 0.031 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x4x4` | 0.023 ± 0.000 | - | 0.032 ± 0.001 | 0.032 ± 0.008 | 0.043 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x8x8` | 0.057 ± 0.000 | - | 0.067 ± 0.000 | 0.066 ± 0.001 | 0.075 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x128x128` | 18.507 ± 0.160 | - | 18.752 ± 0.159 | 19.689 ± 0.130 | 18.755 ± 0.054 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x2x2` | 0.013 ± 0.000 | - | 0.021 ± 0.001 | 0.021 ± 0.002 | 0.035 ± 0.032 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x32x32` | 1.059 ± 0.038 | - | 1.100 ± 0.031 | 1.180 ± 0.026 | 1.202 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x4x4` | 0.042 ± 0.000 | - | 0.051 ± 0.000 | 0.048 ± 0.001 | 0.061 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x8x8` | 0.114 ± 0.002 | - | 0.125 ± 0.001 | 0.119 ± 0.008 | 0.185 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x2x2` | 2.266 ± 0.024 | - | 2.319 ± 0.023 | 1.056 ± 0.016 | 1.350 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x4x4` | 8.361 ± 0.016 | - | 6.320 ± 3.154 | 4.340 ± 0.043 | 2.118 ± 0.382 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x8x8` | 15.136 ± 0.083 | - | 15.343 ± 0.079 | 16.697 ± 0.030 | 6.311 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x128x128` | 2.615 ± 0.034 | - | 2.756 ± 0.183 | 2.786 ± 0.042 | 3.019 ± 0.070 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x2x2` | 0.006 ± 0.004 | - | 0.041 ± 0.007 | 0.017 ± 0.001 | 0.031 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x32x32` | 0.211 ± 0.023 | - | 0.285 ± 0.025 | 0.251 ± 0.024 | 0.189 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x4x4` | 0.012 ± 0.000 | - | 0.049 ± 0.005 | 0.018 ± 0.001 | 0.032 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x64x24` | 0.172 ± 0.013 | - | 0.219 ± 0.014 | 0.196 ± 0.005 | 0.198 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x8x8` | 0.018 ± 0.011 | - | 0.080 ± 0.001 | 0.028 ± 0.002 | 0.039 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x2x2` | 0.011 ± 0.004 | - | 0.052 ± 0.004 | 0.022 ± 0.004 | 0.028 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x4x4` | 0.028 ± 0.000 | - | 0.061 ± 0.003 | 0.027 ± 0.001 | 0.037 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x8x8` | 0.043 ± 0.001 | - | 0.123 ± 0.026 | 0.053 ± 0.004 | 0.063 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x2x2` | 0.020 ± 0.009 | - | 0.048 ± 0.005 | 0.023 ± 0.004 | 0.030 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x4x4` | 0.037 ± 0.000 | - | 0.082 ± 0.019 | 0.030 ± 0.001 | 0.043 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x8x8` | 0.057 ± 0.003 | - | 0.121 ± 0.003 | 0.065 ± 0.001 | 0.074 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x128x128` | 20.822 ± 0.127 | - | 21.276 ± 0.218 | 21.731 ± 0.185 | 26.059 ± 2.976 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x2x2` | 0.021 ± 0.001 | - | 0.062 ± 0.007 | 0.029 ± 0.003 | 0.037 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x32x32` | 1.638 ± 0.023 | - | 1.749 ± 0.035 | 1.751 ± 0.016 | 0.605 ± 0.117 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x4x4` | 0.069 ± 0.000 | - | 0.110 ± 0.018 | 0.047 ± 0.001 | 0.058 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x8x8` | 0.115 ± 0.002 | - | 0.182 ± 0.031 | 0.118 ± 0.001 | 0.146 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x2x2` | 0.475 ± 0.018 | - | unsupported | 0.359 ± 0.024 | 0.771 ± 0.030 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x4x4` | 2.315 ± 0.034 | - | unsupported | 1.949 ± 0.005 | 2.737 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x8x8` | 5.568 ± 0.014 | - | unsupported | 5.314 ± 0.026 | 6.821 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x128x128` | 0.989 ± 0.027 | - | unsupported | 1.145 ± 0.014 | 1.120 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x2x2` | 0.003 ± 0.001 | - | unsupported | 0.009 ± 0.000 | 0.017 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x32x32` | 0.047 ± 0.001 | - | unsupported | 0.067 ± 0.007 | 0.075 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x4x4` | 0.004 ± 0.000 | - | unsupported | 0.012 ± 0.001 | 0.019 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x64x24` | 0.039 ± 0.000 | - | unsupported | 0.058 ± 0.011 | 0.073 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x8x8` | 0.007 ± 0.000 | - | unsupported | 0.016 ± 0.001 | 0.024 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x2x2` | 0.003 ± 0.000 | - | unsupported | 0.011 ± 0.001 | 0.017 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x4x4` | 0.008 ± 0.000 | - | unsupported | 0.016 ± 0.001 | 0.024 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x8x8` | 0.017 ± 0.000 | - | unsupported | 0.027 ± 0.001 | 0.037 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | unsupported | 0.011 ± 0.001 | 0.017 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x4x4` | 0.010 ± 0.001 | - | unsupported | 0.019 ± 0.001 | 0.034 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x8x8` | 0.023 ± 0.000 | - | unsupported | 0.032 ± 0.001 | 0.045 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x128x128` | 7.604 ± 0.035 | - | unsupported | 8.982 ± 0.069 | 8.629 ± 0.028 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x2x2` | 0.005 ± 0.000 | - | unsupported | 0.012 ± 0.001 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x32x32` | 0.373 ± 0.012 | - | unsupported | 0.424 ± 0.016 | 0.480 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x4x4` | 0.019 ± 0.000 | - | unsupported | 0.025 ± 0.000 | 0.037 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x8x8` | 0.044 ± 0.000 | - | unsupported | 0.052 ± 0.004 | 0.089 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x2x2` | 0.818 ± 0.014 | - | unsupported | 0.588 ± 0.022 | 0.770 ± 0.035 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x4x4` | 3.591 ± 0.019 | - | unsupported | 1.933 ± 0.004 | 1.435 ± 0.077 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x8x8` | 8.404 ± 0.019 | - | unsupported | 7.737 ± 0.026 | 3.527 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x128x128` | 1.233 ± 0.024 | - | unsupported | 1.455 ± 0.028 | 1.429 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x2x2` | 0.005 ± 0.004 | - | unsupported | 0.013 ± 0.001 | 0.040 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x32x32` | 0.064 ± 0.001 | - | unsupported | 0.090 ± 0.027 | 0.077 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x4x4` | 0.006 ± 0.000 | - | unsupported | 0.017 ± 0.002 | 0.018 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x64x24` | 0.055 ± 0.000 | - | unsupported | 0.078 ± 0.004 | 0.088 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x8x8` | 0.011 ± 0.000 | - | unsupported | 0.015 ± 0.002 | 0.025 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x2x2` | 0.005 ± 0.000 | - | unsupported | 0.015 ± 0.003 | 0.017 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x4x4` | 0.013 ± 0.000 | - | unsupported | 0.023 ± 0.003 | 0.026 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x8x8` | 0.026 ± 0.001 | - | unsupported | 0.026 ± 0.001 | 0.040 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x2x2` | 0.006 ± 0.000 | - | unsupported | 0.016 ± 0.003 | 0.018 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x4x4` | 0.016 ± 0.000 | - | unsupported | 0.026 ± 0.002 | 0.026 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x8x8` | 0.035 ± 0.001 | - | unsupported | 0.032 ± 0.001 | 0.043 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x128x128` | 9.725 ± 0.055 | - | unsupported | 11.201 ± 0.025 | 11.456 ± 0.494 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x2x2` | 0.009 ± 0.000 | - | unsupported | 0.018 ± 0.001 | 0.019 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x32x32` | 0.497 ± 0.015 | - | unsupported | 0.423 ± 0.018 | 0.310 ± 0.038 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x4x4` | 0.030 ± 0.000 | - | unsupported | 0.025 ± 0.003 | 0.037 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x8x8` | 0.067 ± 0.000 | - | unsupported | 0.051 ± 0.001 | 0.089 ± 0.030 | - | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x16x16,rhs=1` | 1.055 ± 0.020 | - | unsupported | 0.959 ± 0.002 | 2.716 ± 0.054 | 2.544 ± 0.098 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x2x2,rhs=1` | 0.129 ± 0.006 | - | unsupported | 0.059 ± 0.001 | 0.141 ± 0.004 | 0.096 ± 0.002 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x4x4,rhs=1` | 0.181 ± 0.013 | - | unsupported | 0.084 ± 0.001 | 0.223 ± 0.005 | 0.175 ± 0.002 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x8x8,rhs=1` | 0.346 ± 0.023 | - | unsupported | 0.244 ± 0.016 | 0.495 ± 0.001 | 0.689 ± 0.017 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x16x16,rhs=1` | 1.069 ± 0.015 | - | unsupported | 0.496 ± 0.016 | 1.113 ± 0.139 | 2.499 ± 0.792 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x2x2,rhs=1` | 0.139 ± 0.002 | - | unsupported | 0.062 ± 0.001 | 0.120 ± 0.006 | 0.100 ± 0.001 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x4x4,rhs=1` | 0.196 ± 0.002 | - | unsupported | 0.082 ± 0.006 | 0.192 ± 0.053 | 0.183 ± 0.013 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x8x8,rhs=1` | 0.330 ± 0.018 | - | unsupported | 0.159 ± 0.002 | 0.553 ± 0.012 | 0.689 ± 0.053 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.308 ± 0.020 | - | unsupported | 0.375 ± 0.015 | 0.851 ± 0.027 | 0.432 ± 0.018 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.034 ± 0.000 | - | unsupported | 0.046 ± 0.005 | 0.198 ± 0.002 | 0.073 ± 0.003 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.058 ± 0.001 | - | unsupported | 0.061 ± 0.001 | 0.234 ± 0.016 | 0.115 ± 0.001 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.202 ± 0.004 | - | unsupported | 0.167 ± 0.017 | 0.429 ± 0.015 | 0.210 ± 0.001 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.305 ± 0.012 | - | unsupported | 0.561 ± 0.014 | 0.834 ± 0.043 | 0.423 ± 0.013 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.034 ± 0.001 | - | unsupported | 0.077 ± 0.004 | 0.203 ± 0.021 | 0.074 ± 0.001 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.059 ± 0.000 | - | unsupported | 0.102 ± 0.005 | 0.242 ± 0.023 | 0.115 ± 0.001 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.200 ± 0.010 | - | unsupported | 0.226 ± 0.016 | 0.395 ± 0.018 | 0.210 ± 0.002 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.193 ± 0.081 | - | 0.308 ± 0.052 | 0.195 ± 0.003 | 0.391 ± 0.017 | 0.262 ± 0.011 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.084 ± 0.000 | - | 0.090 ± 0.003 | 0.078 ± 0.006 | 0.077 ± 0.007 | 0.058 ± 0.000 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.097 ± 0.005 | - | 0.106 ± 0.011 | 0.089 ± 0.005 | 0.101 ± 0.006 | 0.085 ± 0.000 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.125 ± 0.001 | - | 0.152 ± 0.014 | 0.123 ± 0.006 | 0.175 ± 0.034 | 0.140 ± 0.077 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.226 ± 0.022 | - | 0.382 ± 0.030 | 0.262 ± 0.002 | 0.513 ± 0.074 | 0.270 ± 0.009 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.081 ± 0.000 | - | 0.151 ± 0.013 | 0.134 ± 0.007 | 0.080 ± 0.002 | 0.061 ± 0.001 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.092 ± 0.000 | - | 0.166 ± 0.051 | 0.148 ± 0.006 | 0.105 ± 0.013 | 0.087 ± 0.001 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.123 ± 0.000 | - | 0.227 ± 0.018 | 0.185 ± 0.007 | 0.198 ± 0.012 | 0.141 ± 0.002 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 1 | `1536x1536` | 48.253 ± 0.452 | - | 57.027 ± 0.384 | 63.040 ± 0.432 | 65.218 ± 0.764 | 44.018 ± 4.958 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 4 | `1536x1536` | 21.836 ± 0.142 | - | 22.801 ± 0.606 | 25.344 ± 0.622 | 57.010 ± 3.963 | 35.772 ± 10.312 | - |
| cpu/linalg_uncovered | `det` | f64 | 1 | `1024x1024` | 28.021 ± 0.528 | - | 27.997 ± 0.877 | 27.024 ± 0.107 | 22.069 ± 0.253 | 21.338 ± 3.282 | - |
| cpu/linalg_uncovered | `det` | f64 | 4 | `1024x1024` | 13.615 ± 0.400 | - | 12.848 ± 0.421 | 11.572 ± 0.066 | 14.173 ± 0.092 | 12.426 ± 1.860 | - |
| cpu/linalg_uncovered | `eig` | f64 | 1 | `160x160` | 7.635 ± 0.075 | - | 7.707 ± 0.080 | 7.999 ± 0.030 | 8.208 ± 0.059 | 8.110 ± 0.282 | - |
| cpu/linalg_uncovered | `eig` | f64 | 4 | `160x160` | 11.398 ± 0.165 | - | 11.438 ± 0.054 | 10.391 ± 0.032 | 11.826 ± 0.074 | 11.766 ± 0.378 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 1 | `192x192` | 8.841 ± 0.047 | - | 8.976 ± 0.029 | 8.857 ± 0.020 | 9.663 ± 0.029 | 9.161 ± 0.043 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 4 | `192x192` | 13.189 ± 0.098 | - | 13.450 ± 0.199 | 11.803 ± 0.051 | 14.381 ± 0.349 | 13.856 ± 0.046 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 1 | `512x512` | 21.225 ± 0.172 | - | 22.258 ± 0.241 | 19.777 ± 0.092 | 40.882 ± 0.208 | 14.279 ± 0.546 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 4 | `512x512` | 15.659 ± 0.170 | - | 15.812 ± 0.065 | 15.136 ± 0.123 | 34.457 ± 0.601 | 12.787 ± 0.849 | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `256x128` | 0.508 ± 0.004 | 0.526 ± 0.009 | unsupported | 0.627 ± 0.006 | 0.769 ± 0.039 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `768x384` | 8.036 ± 0.047 | 8.156 ± 0.168 | unsupported | 8.925 ± 0.047 | 9.770 ± 0.057 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `256x128` | 0.390 ± 0.003 | 0.439 ± 0.022 | unsupported | 0.511 ± 0.006 | 1.181 ± 0.061 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `768x384` | 5.187 ± 0.052 | 6.416 ± 0.103 | unsupported | 5.583 ± 0.048 | 9.363 ± 0.220 | - | - |
| cpu/linalg_uncovered | `inv` | f64 | 1 | `768x768` | 43.997 ± 0.417 | - | 43.692 ± 0.584 | 43.481 ± 0.471 | 32.061 ± 0.212 | 28.086 ± 0.396 | - |
| cpu/linalg_uncovered | `inv` | f64 | 4 | `768x768` | 23.795 ± 2.558 | - | 21.459 ± 0.387 | 22.219 ± 0.127 | 20.100 ± 2.191 | 16.719 ± 0.430 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 1 | `768x384,rhs=16` | - | 19.655 ± 0.355 | 18.361 ± 0.152 | 9.470 ± 0.033 | 47.258 ± 0.347 | 9.877 ± 0.145 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 4 | `768x384,rhs=16` | - | 11.350 ± 0.492 | 11.156 ± 0.111 | 6.348 ± 0.151 | 44.457 ± 1.230 | 8.922 ± 0.087 | - |
| cpu/linalg_uncovered | `lu` | f64 | 1 | `1024x1024` | 27.753 ± 0.164 | - | 27.522 ± 0.286 | 37.974 ± 0.586 | 39.419 ± 0.502 | 33.838 ± 9.278 | - |
| cpu/linalg_uncovered | `lu` | f64 | 4 | `1024x1024` | 13.349 ± 0.412 | - | 13.635 ± 0.369 | 16.214 ± 0.345 | 33.228 ± 0.284 | 17.818 ± 9.206 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 1 | `2048x2048` | 1.658 ± 0.081 | - | 1.583 ± 0.301 | 1.272 ± 0.056 | 5.631 ± 0.153 | 2.718 ± 0.059 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 4 | `2048x2048` | 0.774 ± 0.138 | - | 0.744 ± 0.058 | 1.239 ± 0.069 | 2.644 ± 0.306 | 2.692 ± 0.015 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 1 | `512x256` | 20.810 ± 0.332 | - | 21.683 ± 0.214 | 19.795 ± 0.245 | 21.539 ± 0.252 | 20.692 ± 2.248 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 4 | `512x256` | 16.305 ± 0.363 | - | 16.305 ± 0.354 | 16.704 ± 0.067 | 22.413 ± 1.026 | 19.561 ± 4.556 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 1 | `512x256` | 21.365 ± 0.179 | - | 21.616 ± 0.115 | 20.124 ± 0.378 | 21.244 ± 0.085 | 20.726 ± 0.947 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 4 | `512x256` | 16.457 ± 0.228 | - | 16.210 ± 0.158 | 27.467 ± 17.771 | 21.754 ± 1.832 | 24.017 ± 1.308 | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `256x128` | 1.190 ± 0.004 | 1.208 ± 0.010 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `768x384` | 22.016 ± 0.189 | 22.212 ± 0.052 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `256x128` | 1.353 ± 0.020 | 1.432 ± 0.062 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `768x384` | 19.882 ± 0.190 | 20.609 ± 0.466 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `256x128` | 1.106 ± 0.005 | 1.124 ± 0.011 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `768x384` | 20.953 ± 0.370 | 21.257 ± 0.154 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `256x128` | 1.252 ± 0.044 | 1.335 ± 0.013 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `768x384` | 19.304 ± 0.146 | 19.450 ± 0.152 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 1 | `1024x1024` | 29.115 ± 0.654 | - | 28.556 ± 0.407 | 26.862 ± 0.281 | 21.918 ± 0.240 | 21.159 ± 1.760 | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 4 | `1024x1024` | 13.779 ± 0.228 | - | 12.525 ± 0.256 | 11.764 ± 0.114 | 14.322 ± 0.174 | 12.344 ± 1.603 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 1 | `768x384` | - | 63.339 ± 2.005 | 62.349 ± 0.964 | 63.531 ± 0.076 | 69.148 ± 0.170 | 67.645 ± 7.030 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 4 | `768x384` | - | 37.003 ± 0.526 | 37.297 ± 0.337 | 37.861 ± 0.174 | 56.349 ± 0.555 | 54.581 ± 2.168 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 1 | `4096x4096,rhs=64` | 41.645 ± 0.255 | - | 59.882 ± 0.585 | 30.625 ± 0.213 | 80.409 ± 0.397 | 33.568 ± 0.499 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 4 | `4096x4096,rhs=64` | 30.865 ± 0.210 | - | 42.766 ± 11.033 | 46.757 ± 8.356 | 67.515 ± 3.394 | 20.271 ± 1.773 | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `1024x64` | 0.140 ± 0.001 | 0.140 ± 0.001 | unsupported | 0.155 ± 0.001 | 0.037 ± 0.005 | - | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `4096x256` | 2.233 ± 0.006 | 2.223 ± 0.008 | unsupported | 2.070 ± 0.045 | 0.540 ± 0.209 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `1024x64` | 0.118 ± 0.005 | 0.120 ± 0.003 | unsupported | 0.075 ± 0.003 | 0.054 ± 0.013 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `4096x256` | 0.905 ± 0.003 | 0.910 ± 0.014 | unsupported | 0.949 ± 0.008 | 0.320 ± 0.301 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `1024x64` | 0.338 ± 0.001 | 0.339 ± 0.003 | unsupported | 0.140 ± 0.001 | 0.236 ± 0.005 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `4096x256` | 5.387 ± 0.051 | 5.391 ± 0.022 | unsupported | 1.379 ± 0.019 | 1.828 ± 0.023 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `1024x64` | 0.259 ± 0.015 | 0.266 ± 0.027 | unsupported | 0.090 ± 0.002 | 0.173 ± 0.003 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `4096x256` | 2.534 ± 0.011 | 2.549 ± 0.085 | unsupported | 0.688 ± 0.006 | 0.949 ± 0.033 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `1024x64` | 1.067 ± 0.003 | 1.067 ± 0.005 | unsupported | 0.282 ± 0.001 | 0.191 ± 0.003 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `4096x256` | 17.022 ± 0.021 | 17.044 ± 0.084 | unsupported | 4.045 ± 0.025 | 0.994 ± 0.007 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `1024x64` | 0.706 ± 0.051 | 0.731 ± 0.027 | unsupported | 0.107 ± 0.001 | 0.148 ± 0.022 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `4096x256` | 6.680 ± 0.035 | 6.704 ± 0.079 | unsupported | 1.431 ± 0.030 | 0.664 ± 0.005 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `1024x64` | 0.061 ± 0.001 | 0.099 ± 0.002 | unsupported | 0.156 ± 0.001 | 0.210 ± 0.017 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `4096x256` | 0.664 ± 0.011 | 0.721 ± 0.040 | unsupported | 1.355 ± 0.017 | 1.014 ± 0.017 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `1024x64` | 0.192 ± 0.023 | 0.292 ± 0.034 | unsupported | 0.066 ± 0.001 | 0.189 ± 0.003 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `4096x256` | 0.511 ± 0.028 | 0.705 ± 0.050 | unsupported | 0.372 ± 0.003 | 0.784 ± 0.036 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `1024x64` | 0.111 ± 0.001 | 0.134 ± 0.008 | unsupported | 0.132 ± 0.001 | 0.226 ± 0.003 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `4096x256` | 1.375 ± 0.019 | 1.480 ± 0.084 | unsupported | 1.688 ± 0.012 | 1.369 ± 0.020 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `1024x64` | 0.255 ± 0.025 | 0.314 ± 0.007 | unsupported | 0.063 ± 0.020 | 0.226 ± 0.008 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `4096x256` | 0.862 ± 0.017 | 0.982 ± 0.028 | unsupported | 0.504 ± 0.003 | 1.006 ± 0.061 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `1024x64` | 0.149 ± 0.003 | 0.190 ± 0.003 | unsupported | unsupported | 0.273 ± 0.003 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `4096x256` | 1.918 ± 0.022 | 2.095 ± 0.091 | unsupported | unsupported | 1.824 ± 0.026 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `1024x64` | 0.320 ± 0.020 | 0.419 ± 0.067 | unsupported | unsupported | 0.248 ± 0.014 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `4096x256` | 1.292 ± 0.054 | 1.404 ± 0.077 | unsupported | unsupported | 1.089 ± 0.134 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `1024x64` | 0.011 ± 0.001 | 0.022 ± 0.003 | unsupported | 0.013 ± 0.000 | 0.060 ± 0.001 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `4096x256` | 0.133 ± 0.003 | 0.115 ± 0.004 | unsupported | 0.096 ± 0.007 | 0.242 ± 0.004 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `1024x64` | 0.031 ± 0.005 | 0.082 ± 0.011 | unsupported | 0.015 ± 0.000 | 0.042 ± 0.006 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `4096x256` | 0.106 ± 0.043 | 0.151 ± 0.013 | unsupported | 0.093 ± 0.008 | 0.164 ± 0.013 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `1024x64` | 0.036 ± 0.005 | 0.059 ± 0.002 | unsupported | 0.104 ± 0.001 | 0.169 ± 0.004 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `4096x256` | 0.385 ± 0.011 | 0.388 ± 0.023 | unsupported | 1.026 ± 0.017 | 0.809 ± 0.024 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `1024x64` | 0.125 ± 0.015 | 0.210 ± 0.037 | unsupported | 0.097 ± 0.004 | 0.166 ± 0.006 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `4096x256` | 0.343 ± 0.073 | 0.408 ± 0.060 | unsupported | 0.253 ± 0.022 | 0.640 ± 0.035 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `1024x64` | 0.290 ± 0.003 | 0.290 ± 0.001 | unsupported | 0.116 ± 0.002 | 0.204 ± 0.003 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `4096x256` | 4.605 ± 0.009 | 4.620 ± 0.031 | unsupported | 1.397 ± 0.034 | 0.901 ± 0.019 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `1024x64` | 0.226 ± 0.021 | 0.218 ± 0.025 | unsupported | 0.094 ± 0.001 | 0.141 ± 0.004 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `4096x256` | 1.987 ± 0.028 | 1.998 ± 0.021 | unsupported | 0.647 ± 0.016 | 0.841 ± 0.227 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `1024x64` | 0.289 ± 0.002 | 0.279 ± 0.002 | unsupported | 0.119 ± 0.001 | 0.202 ± 0.028 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `4096x256` | 4.450 ± 0.012 | 4.458 ± 0.024 | unsupported | 1.433 ± 0.016 | 0.910 ± 0.014 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `1024x64` | 0.232 ± 0.012 | 0.228 ± 0.011 | unsupported | 0.098 ± 0.002 | 0.141 ± 0.003 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `4096x256` | 2.082 ± 0.022 | 2.088 ± 0.425 | unsupported | 0.700 ± 0.009 | 0.613 ± 0.008 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `1024x64` | 0.100 ± 0.001 | 0.135 ± 0.003 | unsupported | 0.133 ± 0.003 | 0.223 ± 0.003 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `4096x256` | 1.362 ± 0.037 | 1.492 ± 0.099 | unsupported | 1.236 ± 0.003 | 1.277 ± 0.008 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `1024x64` | 0.236 ± 0.022 | 0.302 ± 0.018 | unsupported | 0.062 ± 0.002 | 0.217 ± 0.004 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `4096x256` | 0.852 ± 0.037 | 0.964 ± 0.164 | unsupported | 0.485 ± 0.006 | 0.926 ± 0.142 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `1024x64` | 1.048 ± 0.002 | 1.050 ± 0.003 | unsupported | 0.273 ± 0.002 | 0.347 ± 0.020 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `4096x256` | 16.743 ± 0.046 | 16.763 ± 0.054 | unsupported | 3.919 ± 0.008 | 3.149 ± 0.007 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `1024x64` | 0.663 ± 0.025 | 0.668 ± 0.083 | unsupported | 0.200 ± 0.002 | 0.216 ± 0.019 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `4096x256` | 6.574 ± 0.045 | 6.580 ± 0.100 | unsupported | 1.518 ± 0.017 | 1.728 ± 0.003 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `1024x64` | 0.778 ± 0.011 | 1.066 ± 0.031 | unsupported | 0.263 ± 0.002 | 0.221 ± 0.003 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `4096x256` | 16.151 ± 0.165 | 21.057 ± 1.730 | unsupported | 4.262 ± 0.118 | 1.448 ± 0.020 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `1024x64` | 0.897 ± 0.079 | 1.682 ± 0.067 | unsupported | 0.259 ± 0.002 | 0.160 ± 0.004 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `4096x256` | 8.889 ± 0.125 | 12.645 ± 0.748 | unsupported | 1.976 ± 0.061 | 0.886 ± 0.012 | - | - |
| cpu/output_reuse | `add_into` | f64 | 1 | `33554432` | 29.963 ± 1.275 | - | unsupported | 25.028 ± 0.997 | - | 29.088 ± 1.050 | 29.264 ± 0.520 |
| cpu/output_reuse | `add_into` | f64 | 4 | `33554432` | 21.484 ± 0.468 | - | unsupported | 21.374 ± 0.195 | - | 25.796 ± 0.608 | 21.738 ± 0.686 |
| cpu/output_reuse | `conj_into` | c64 | 1 | `16777216` | 20.690 ± 1.449 | - | unsupported | 18.793 ± 1.359 | - | 20.173 ± 0.750 | 19.174 ± 0.990 |
| cpu/output_reuse | `conj_into` | c64 | 4 | `16777216` | 14.823 ± 0.690 | - | unsupported | 14.760 ± 0.667 | - | 17.018 ± 0.458 | 15.228 ± 0.840 |
| cpu/output_reuse | `copy_read_into` | f64 | 1 | `33554432` | 15.184 ± 0.590 | - | unsupported | 17.957 ± 0.611 | - | 12.858 ± 0.522 | - |
| cpu/output_reuse | `copy_read_into` | f64 | 4 | `33554432` | 14.741 ± 1.090 | - | unsupported | 14.707 ± 0.472 | - | 12.341 ± 0.482 | - |
| cpu/output_reuse | `div_into` | f64 | 1 | `33554432` | 32.096 ± 1.549 | - | unsupported | 24.749 ± 0.341 | - | 28.687 ± 1.641 | 28.547 ± 0.529 |
| cpu/output_reuse | `div_into` | f64 | 4 | `33554432` | 21.810 ± 0.583 | - | unsupported | 21.238 ± 1.336 | - | 25.086 ± 0.445 | 20.831 ± 0.686 |
| cpu/output_reuse | `dot_general_read_into` | f64 | 1 | `1024x1024` | 50.471 ± 0.256 | - | unsupported | 51.998 ± 0.334 | - | 48.113 ± 0.066 | - |
| cpu/output_reuse | `dot_general_read_into` | f64 | 4 | `1024x1024` | 24.193 ± 0.114 | - | unsupported | 27.415 ± 0.110 | - | 23.470 ± 0.134 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 1 | `1024x1024` | 50.310 ± 0.367 | - | unsupported | 51.100 ± 0.184 | - | 47.685 ± 0.062 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 4 | `1024x1024` | 24.621 ± 0.206 | - | unsupported | 27.215 ± 0.098 | - | 23.320 ± 0.183 | - |
| cpu/output_reuse | `mul_into` | f64 | 1 | `33554432` | 31.792 ± 1.286 | - | unsupported | 24.982 ± 0.461 | - | 30.766 ± 0.852 | 30.947 ± 1.765 |
| cpu/output_reuse | `mul_into` | f64 | 4 | `33554432` | 21.103 ± 0.693 | - | unsupported | 21.709 ± 0.341 | - | 24.953 ± 0.361 | 21.913 ± 1.512 |
| cpu/output_reuse | `neg_into` | f64 | 1 | `33554432` | 20.423 ± 3.098 | - | unsupported | 17.253 ± 0.924 | - | 18.992 ± 0.451 | 23.083 ± 0.386 |
| cpu/output_reuse | `neg_into` | f64 | 4 | `33554432` | 15.036 ± 0.365 | - | unsupported | 14.457 ± 0.634 | - | 18.024 ± 0.557 | 14.955 ± 0.533 |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `256x256,rhs=16` | 0.759 ± 0.003 | unsupported | unsupported | 0.910 ± 0.004 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `768x768,rhs=16` | 11.568 ± 0.239 | unsupported | unsupported | 12.858 ± 0.086 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `256x256,rhs=16` | 0.515 ± 0.004 | unsupported | unsupported | 0.618 ± 0.002 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `768x768,rhs=16` | 6.040 ± 0.176 | unsupported | unsupported | 6.896 ± 0.098 | unsupported | - | - |
| cpu/output_reuse | `sub_into` | f64 | 1 | `33554432` | 32.003 ± 1.549 | - | unsupported | 24.783 ± 0.482 | - | 28.703 ± 3.148 | 31.312 ± 1.225 |
| cpu/output_reuse | `sub_into` | f64 | 4 | `33554432` | 20.981 ± 0.877 | - | unsupported | 20.917 ± 0.434 | - | 25.026 ± 0.526 | 21.590 ± 0.927 |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 1 | `8192x1 -> 8192x4096` | 21.658 ± 0.363 | - | 22.229 ± 0.154 | 19.838 ± 0.279 | 19.620 ± 0.967 | 28.036 ± 0.313 | - |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 4 | `8192x1 -> 8192x4096` | 16.384 ± 0.431 | - | 16.277 ± 0.188 | 13.316 ± 0.227 | 14.294 ± 0.049 | 26.158 ± 0.478 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 1 | `33554432` | 18.330 ± 0.704 | - | 99.328 ± 2.230 | 17.835 ± 0.369 | 18.504 ± 0.141 | 20.809 ± 0.933 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 4 | `33554432` | 12.079 ± 0.722 | - | 32.185 ± 1.262 | 12.511 ± 0.792 | 10.917 ± 0.746 | 17.659 ± 0.114 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 1 | `8192 -> 8192x8192` | 126.129 ± 0.831 | - | 125.913 ± 1.043 | 37.547 ± 0.677 | 39.305 ± 0.807 | 43.393 ± 1.110 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 4 | `8192 -> 8192x8192` | 23.581 ± 0.951 | - | 25.209 ± 1.526 | 23.346 ± 0.850 | 23.422 ± 0.526 | 38.036 ± 0.833 | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 1 | `8388608x2x2 -> 8388608x2` | 13.770 ± 0.247 | - | 204.990 ± 0.976 | 21.317 ± 0.802 | 18.977 ± 0.488 | - | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 4 | `8388608x2x2 -> 8388608x2` | 10.717 ± 1.019 | - | 29.239 ± 0.972 | 12.180 ± 0.786 | 10.753 ± 0.115 | - | - |
| cpu/structural_shape | `reshape` | f64 | 1 | `33554432 -> 8192x4096` | 26.590 ± 0.880 | - | 32.138 ± 0.495 | 25.309 ± 0.610 | 24.825 ± 0.403 | 31.560 ± 1.881 | - |
| cpu/structural_shape | `reshape` | f64 | 4 | `33554432 -> 8192x4096` | 17.294 ± 1.508 | - | 16.447 ± 0.354 | 16.144 ± 0.422 | 18.895 ± 0.430 | 26.796 ± 1.262 | - |
| cpu/structural_shape | `transpose` | f64 | 1 | `4096x4096` | 94.739 ± 1.380 | - | 93.726 ± 0.360 | 33.479 ± 0.465 | 47.688 ± 1.041 | 898.852 ± 3.644 | 642.880 ± 2.473 |
| cpu/structural_shape | `transpose` | f64 | 4 | `4096x4096` | 32.732 ± 0.892 | - | 32.784 ± 0.857 | 33.156 ± 0.709 | 24.136 ± 0.491 | 899.332 ± 1.629 | 105.600 ± 1.747 |
| cpu/structural_shape | `tril` | f64 | 1 | `4096x4096` | 13.399 ± 0.834 | - | 72.128 ± 0.800 | 15.737 ± 0.124 | 13.399 ± 0.265 | 18.808 ± 0.544 | - |
| cpu/structural_shape | `tril` | f64 | 4 | `4096x4096` | 7.035 ± 0.253 | - | 17.056 ± 0.749 | 9.811 ± 0.135 | 8.142 ± 0.293 | 16.769 ± 0.973 | - |
| cpu/structural_shape | `triu` | f64 | 1 | `4096x4096` | 13.249 ± 0.145 | - | 60.102 ± 0.665 | 17.177 ± 0.093 | 13.369 ± 0.199 | 19.041 ± 0.655 | - |
| cpu/structural_shape | `triu` | f64 | 4 | `4096x4096` | 7.505 ± 0.366 | - | 16.960 ± 0.325 | 9.700 ± 0.206 | 8.356 ± 0.321 | 16.697 ± 0.439 | - |
| cpu/view_metadata | `broadcast_in_dim_view` | f64 | 1 | `8192x1 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | - | - |
| cpu/view_metadata | `broadcast_in_dim_view` | f64 | 4 | `8192x1 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.002 ± 0.000 | - | - | - |
| cpu/view_metadata | `reshape_view` | f64 | 1 | `33554432 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `reshape_view` | f64 | 4 | `33554432 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.002 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `slice_view` | f64 | 1 | `4194304 -> 2096128` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `slice_view` | f64 | 4 | `4194304 -> 2096128` | 0.000 ± 0.000 | - | unsupported | 0.002 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `transpose_view` | f64 | 1 | `4096x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `transpose_view` | f64 | 4 | `4096x4096` | 0.000 ± 0.000 | - | unsupported | 0.002 ± 0.000 | - | 0.000 ± 0.000 | - |

## Short-operation batches

Total batch duration and normalized ns/op are shown without rounding sub-microsecond calls to zero. Metadata views require no execution session.

| Operation | Dtype | Shape | Threads | Backend | Operations/batch | Median batch ms | Median ns/op |
|---|---|---|---:|---|---:|---:|---:|
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 16 | 5.757987 | 359874.19 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs direct API (ms) | 16 | 5.224209 | 326513.06 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 16 | 6.001883 | 375117.69 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 16 | 5.161607 | 322600.44 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs direct API (ms) | 16 | 7.590640 | 474415.00 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 16 | 9.815210 | 613450.62 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs direct API (ms) | 16 | 20.670361 | 1291897.56 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs eager mode (ms) | 16 | 21.832043 | 1364502.69 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs direct API (ms) | 16 | 6.612652 | 413290.75 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs eager mode (ms) | 16 | 6.936618 | 433538.62 |
| householder_qr | f64 | `256x128` | 1 | JAX Python (XLA CPU) (ms) | 16 | 12.306819 | 769176.19 |
| householder_qr | f64 | `256x128` | 1 | PyTorch Python (ms) | 16 | 10.026708 | 626669.25 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 8.131343 | 508208.94 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 8.410064 | 525629.00 |
| householder_qr | f64 | `768x384` | 1 | JAX Python (XLA CPU) (ms) | 16 | 156.321126 | 9770070.38 |
| householder_qr | f64 | `768x384` | 1 | PyTorch Python (ms) | 16 | 142.804829 | 8925301.81 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 128.569016 | 8035563.50 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 130.502732 | 8156420.75 |
| householder_qr | f64 | `256x128` | 4 | JAX Python (XLA CPU) (ms) | 16 | 18.898606 | 1181162.88 |
| householder_qr | f64 | `256x128` | 4 | PyTorch Python (ms) | 16 | 8.181054 | 511315.88 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 6.234405 | 389650.31 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 7.023119 | 438944.94 |
| householder_qr | f64 | `768x384` | 4 | JAX Python (XLA CPU) (ms) | 16 | 149.806116 | 9362882.25 |
| householder_qr | f64 | `768x384` | 4 | PyTorch Python (ms) | 16 | 89.323531 | 5582720.69 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 82.997989 | 5187374.31 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 102.649828 | 6415614.25 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 19.036923 | 1189807.69 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 19.323830 | 1207739.38 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 352.252729 | 22015795.56 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 355.389987 | 22211874.19 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 21.651035 | 1353189.69 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 22.918254 | 1432390.88 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 318.119294 | 19882455.88 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 329.751937 | 20609496.06 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 17.691799 | 1105737.44 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 17.977382 | 1123586.38 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 335.249073 | 20953067.06 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 340.105301 | 21256581.31 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 20.027640 | 1251727.50 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 21.353569 | 1334598.06 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 308.858184 | 19303636.50 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 311.202542 | 19450158.88 |
| erf | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 2.347139 | 36674.05 |
| erf | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 9.892746 | 154574.16 |
| erf | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 8.977006 | 140265.72 |
| erf | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 8.954214 | 139909.59 |
| erf | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 4.317212 | 539651.50 |
| erf | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 16.560784 | 2070098.00 |
| erf | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 17.864325 | 2233040.62 |
| erf | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 17.781600 | 2222700.00 |
| erf | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 3.457494 | 54023.34 |
| erf | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 4.793639 | 74900.61 |
| erf | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 7.550868 | 117982.31 |
| erf | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 7.650353 | 119536.77 |
| erf | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 2.560837 | 320104.62 |
| erf | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 7.591148 | 948893.50 |
| erf | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 7.242669 | 905333.62 |
| erf | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 7.281770 | 910221.25 |
| gelu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 15.085715 | 235714.30 |
| gelu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 8.980371 | 140318.30 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 21.612788 | 337699.81 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 21.664584 | 338509.12 |
| gelu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 14.623258 | 1827907.25 |
| gelu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 11.034540 | 1379317.50 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 43.097353 | 5387169.12 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 43.124624 | 5390578.00 |
| gelu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 11.083122 | 173173.78 |
| gelu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 5.734498 | 89601.53 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 16.555687 | 258682.61 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 17.040280 | 266254.38 |
| gelu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 7.595898 | 949487.25 |
| gelu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 5.505630 | 688203.75 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 20.269193 | 2533649.12 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 20.392774 | 2549096.75 |
| gelu_tanh | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 12.231800 | 191121.88 |
| gelu_tanh | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 18.035793 | 281809.27 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 68.302284 | 1067223.19 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 68.318094 | 1067470.22 |
| gelu_tanh | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 7.951374 | 993921.75 |
| gelu_tanh | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 32.359649 | 4044956.12 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 136.174496 | 17021812.00 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 136.350025 | 17043753.12 |
| gelu_tanh | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 9.447712 | 147620.50 |
| gelu_tanh | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 6.835235 | 106800.55 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 45.161888 | 705654.50 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 46.795465 | 731179.14 |
| gelu_tanh | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 5.308013 | 663501.62 |
| gelu_tanh | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 11.448938 | 1431117.25 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 53.436288 | 6679536.00 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 53.634320 | 6704290.00 |
| layer_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 13.421404 | 209709.44 |
| layer_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 9.995870 | 156185.47 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 3.926616 | 61353.38 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 6.360884 | 99388.81 |
| layer_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 8.112095 | 1014011.88 |
| layer_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 10.840136 | 1355017.00 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 5.314189 | 664273.62 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 5.767928 | 720991.00 |
| layer_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 12.082719 | 188792.48 |
| layer_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 4.255453 | 66491.45 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 12.302086 | 192220.09 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 18.656564 | 291508.81 |
| layer_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 6.275409 | 784426.12 |
| layer_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 2.977685 | 372210.62 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 4.091610 | 511451.25 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 5.637612 | 704701.50 |
| log_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 14.490423 | 226412.86 |
| log_softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 8.468314 | 132317.41 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 7.095957 | 110874.33 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 8.572946 | 133952.28 |
| log_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 10.949495 | 1368686.88 |
| log_softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 13.506271 | 1688283.88 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 11.002068 | 1375258.50 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 11.843241 | 1480405.12 |
| log_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 14.445896 | 225717.12 |
| log_softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 4.010655 | 62666.48 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 16.317001 | 254953.14 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 20.107249 | 314175.77 |
| log_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 8.045148 | 1005643.50 |
| log_softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 4.035522 | 504440.25 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 6.895502 | 861937.75 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 7.854827 | 981853.38 |
| masked_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 17.494289 | 273348.27 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 9.524053 | 148813.33 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 12.155739 | 189933.42 |
| masked_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 14.589234 | 1823654.25 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 15.341265 | 1917658.12 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 16.763816 | 2095477.00 |
| masked_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 15.851766 | 247683.84 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 20.488413 | 320131.45 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 26.813583 | 418962.23 |
| masked_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 8.710794 | 1088849.25 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 10.336163 | 1292020.38 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 11.232368 | 1404046.00 |
| reduce_mean | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 3.817498 | 59648.41 |
| reduce_mean | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 0.841293 | 13145.20 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 0.687675 | 10744.92 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 1.406918 | 21983.09 |
| reduce_mean | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 1.938361 | 242295.12 |
| reduce_mean | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 0.767866 | 95983.25 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 1.066826 | 133353.25 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 0.916344 | 114543.00 |
| reduce_mean | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 2.678007 | 41843.86 |
| reduce_mean | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 0.935358 | 14614.97 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 1.959351 | 30614.86 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 5.259142 | 82174.09 |
| reduce_mean | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 1.310020 | 163752.50 |
| reduce_mean | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 0.745594 | 93199.25 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 0.847344 | 105918.00 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 1.207247 | 150905.88 |
| rms_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 10.836891 | 169326.42 |
| rms_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 6.684959 | 104452.48 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 2.287574 | 35743.34 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 3.771756 | 58933.69 |
| rms_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 6.469125 | 808640.62 |
| rms_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 8.204950 | 1025618.75 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 3.083067 | 385383.38 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 3.106921 | 388365.12 |
| rms_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 10.594756 | 165543.06 |
| rms_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 6.191041 | 96735.02 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 8.030557 | 125477.45 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 13.469884 | 210466.94 |
| rms_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 5.121414 | 640176.75 |
| rms_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 2.026888 | 253361.00 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 2.746826 | 343353.25 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 3.262126 | 407765.75 |
| sigmoid | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 13.035202 | 203675.03 |
| sigmoid | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 7.402470 | 115663.59 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 18.543696 | 289745.25 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 18.559315 | 289989.30 |
| sigmoid | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 7.209179 | 901147.38 |
| sigmoid | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 11.178150 | 1397268.75 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 36.843762 | 4605470.25 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 36.958758 | 4619844.75 |
| sigmoid | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 9.004333 | 140692.70 |
| sigmoid | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 5.987661 | 93557.20 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 14.446851 | 225732.05 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 13.934453 | 217725.83 |
| sigmoid | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 6.731725 | 841465.62 |
| sigmoid | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 5.174190 | 646773.75 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 15.898421 | 1987302.62 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 15.984372 | 1998046.50 |
| silu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 12.928702 | 202010.97 |
| silu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 7.625037 | 119141.20 |
| silu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 18.477552 | 288711.75 |
| silu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 17.884964 | 279452.56 |
| silu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 7.276775 | 909596.88 |
| silu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 11.466189 | 1433273.62 |
| silu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 35.596262 | 4449532.75 |
| silu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 35.667911 | 4458488.88 |
| silu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 9.042625 | 141291.02 |
| silu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 6.268756 | 97949.31 |
| silu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 14.848505 | 232007.89 |
| silu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 14.607422 | 228240.97 |
| silu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 4.904539 | 613067.38 |
| silu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 5.601589 | 700198.62 |
| silu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 16.657360 | 2082170.00 |
| silu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 16.701573 | 2087696.62 |
| softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 14.293414 | 223334.59 |
| softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 8.512806 | 133012.59 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 6.429241 | 100456.89 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 8.638929 | 134983.27 |
| softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 10.216756 | 1277094.50 |
| softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 9.886614 | 1235826.75 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 10.894006 | 1361750.75 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 11.936797 | 1492099.62 |
| softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 13.905687 | 217276.36 |
| softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 3.939303 | 61551.61 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 15.105726 | 236026.97 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 19.347325 | 302301.95 |
| softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 7.405324 | 925665.50 |
| softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 3.876415 | 484551.88 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 6.812626 | 851578.25 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 7.714574 | 964321.75 |
| softplus | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 22.221477 | 347210.58 |
| softplus | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 17.454976 | 272734.00 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 67.090361 | 1048286.89 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 67.211607 | 1050181.36 |
| softplus | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 25.193894 | 3149236.75 |
| softplus | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 31.355603 | 3919450.38 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 133.943084 | 16742885.50 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 134.106561 | 16763320.12 |
| softplus | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 13.826911 | 216045.48 |
| softplus | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 12.799752 | 199996.12 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 42.460748 | 663449.19 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 42.735230 | 667737.97 |
| softplus | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 13.820126 | 1727515.75 |
| softplus | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 12.147794 | 1518474.25 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 52.591689 | 6573961.12 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 52.640248 | 6580031.00 |
| take_along_axis | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 14.114529 | 220539.52 |
| take_along_axis | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 16.826361 | 262911.89 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 49.805619 | 778212.80 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 68.243625 | 1066306.64 |
| take_along_axis | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 11.584642 | 1448080.25 |
| take_along_axis | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 34.092409 | 4261551.12 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 129.208387 | 16151048.37 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 168.458572 | 21057321.50 |
| take_along_axis | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 10.229594 | 159837.41 |
| take_along_axis | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 16.606527 | 259476.98 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 57.417697 | 897151.52 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 107.619866 | 1681560.41 |
| take_along_axis | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 7.090866 | 886358.25 |
| take_along_axis | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 15.808425 | 1976053.12 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 71.108526 | 8888565.75 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 101.157844 | 12644730.50 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | PyTorch Python (ms) | 16 | 14.557790 | 909861.88 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 12.150371 | 759398.19 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | PyTorch Python (ms) | 16 | 205.730205 | 12858137.81 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 185.083800 | 11567737.50 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | PyTorch Python (ms) | 16 | 9.885255 | 617828.44 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 8.247148 | 515446.75 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | PyTorch Python (ms) | 16 | 110.341392 | 6896337.00 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 96.647072 | 6040442.00 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.023825 | 1489.06 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.002445 | 152.81 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.039243 | 2452.69 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.002405 | 150.31 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000131 | 8.19 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.021981 | 1373.81 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.007895 | 493.44 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000460 | 28.75 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.035417 | 2213.56 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.007174 | 448.38 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000111 | 6.94 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | PyTorch Python (ms) | 16 | 0.023194 | 1449.62 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | tenferro-rs direct API (ms) | 16 | 0.005309 | 331.81 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000140 | 8.75 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | PyTorch Python (ms) | 16 | 0.037931 | 2370.69 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | tenferro-rs direct API (ms) | 16 | 0.005581 | 348.81 |
| transpose_view | f64 | `4096x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000110 | 6.88 |
| transpose_view | f64 | `4096x4096` | 1 | PyTorch Python (ms) | 16 | 0.021390 | 1336.88 |
| transpose_view | f64 | `4096x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.003877 | 242.31 |
| transpose_view | f64 | `4096x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000100 | 6.25 |
| transpose_view | f64 | `4096x4096` | 4 | PyTorch Python (ms) | 16 | 0.033322 | 2082.62 |
| transpose_view | f64 | `4096x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.003366 | 210.38 |

## Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `cpu/complex/norm_fro` (c64, threads=4, shape=`2048x1536`): `tenferro-direct` is 11.3x faster than the slowest successful cell (`tenferro-trace`, 14.887 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `jax-cpu` is 14.0x faster than the slowest successful cell (`pytorch-cpu`, 38.058 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 23.7x faster than the slowest successful cell (`pytorch-cpu`, 38.058 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 20.5x faster than the slowest successful cell (`pytorch-cpu`, 38.058 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 20.6x faster than the slowest successful cell (`pytorch-cpu`, 38.058 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 14.4x faster than the slowest successful cell (`pytorch-cpu`, 11.737 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-trace` is 11.7x faster than the slowest successful cell (`pytorch-cpu`, 11.737 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `jax-cpu` is 12.4x faster than the slowest successful cell (`pytorch-cpu`, 33.288 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 21.7x faster than the slowest successful cell (`pytorch-cpu`, 33.288 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 17.6x faster than the slowest successful cell (`pytorch-cpu`, 33.288 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 17.6x faster than the slowest successful cell (`pytorch-cpu`, 33.288 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 13.2x faster than the slowest successful cell (`pytorch-cpu`, 10.663 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-trace` is 13.1x faster than the slowest successful cell (`pytorch-cpu`, 10.663 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x2x2`): `tenferro-direct` is 13.5x faster than the slowest successful cell (`jax-cpu`, 0.012 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x32x32`): `tenferro-direct` is 17.3x faster than the slowest successful cell (`jax-cpu`, 0.069 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x4x4`): `tenferro-direct` is 12.5x faster than the slowest successful cell (`jax-cpu`, 0.012 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x8x8`): `tenferro-direct` is 11.1x faster than the slowest successful cell (`jax-cpu`, 0.012 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`3x2x2`): `tenferro-direct` is 11.7x faster than the slowest successful cell (`jax-cpu`, 0.012 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`3x4x4`): `tenferro-direct` is 11.2x faster than the slowest successful cell (`jax-cpu`, 0.012 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x2x2`): `tenferro-direct` is 11.4x faster than the slowest successful cell (`jax-cpu`, 0.012 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x4x4`): `tenferro-direct` is 10.3x faster than the slowest successful cell (`jax-cpu`, 0.012 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x8x8`): `pytorch-cpu` is 10.4x faster than the slowest successful cell (`jax-cpu`, 0.066 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x8x8`): `tenferro-direct` is 34.8x faster than the slowest successful cell (`jax-cpu`, 0.066 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`8x8x8`): `pytorch-cpu` is 10.4x faster than the slowest successful cell (`jax-cpu`, 0.074 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`8x8x8`): `tenferro-direct` is 26.9x faster than the slowest successful cell (`jax-cpu`, 0.074 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x2x2`): `tenferro-direct` is 14.5x faster than the slowest successful cell (`tenferro-trace`, 0.033 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x32x32`): `tenferro-direct` is 11.4x faster than the slowest successful cell (`tenferro-trace`, 0.047 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x4x4`): `tenferro-direct` is 16.0x faster than the slowest successful cell (`tenferro-trace`, 0.033 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x8x8`): `tenferro-direct` is 13.1x faster than the slowest successful cell (`tenferro-trace`, 0.034 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x2x2`): `tenferro-direct` is 17.5x faster than the slowest successful cell (`tenferro-trace`, 0.046 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x4x4`): `tenferro-direct` is 16.6x faster than the slowest successful cell (`tenferro-trace`, 0.038 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x2x2`): `tenferro-direct` is 17.0x faster than the slowest successful cell (`tenferro-trace`, 0.036 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x4x4`): `tenferro-direct` is 23.4x faster than the slowest successful cell (`tenferro-trace`, 0.043 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x8x8`): `tenferro-direct` is 10.4x faster than the slowest successful cell (`tenferro-trace`, 0.038 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x2x2`): `tenferro-direct` is 16.7x faster than the slowest successful cell (`tenferro-trace`, 0.042 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x4x4`): `tenferro-direct` is 15.1x faster than the slowest successful cell (`tenferro-trace`, 0.049 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x8x8`): `tenferro-direct` is 14.3x faster than the slowest successful cell (`tenferro-trace`, 0.041 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigh` (c64, threads=1, shape=`1x2x2`): `tenferro-direct` is 10.6x faster than the slowest successful cell (`jax-cpu`, 0.025 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigh` (f64, threads=4, shape=`1x2x2`): `tenferro-direct` is 10.5x faster than the slowest successful cell (`tenferro-trace`, 0.044 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`3x8x8`): `tenferro-direct` is 13.1x faster than the slowest successful cell (`jax-cpu`, 0.051 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`4x8x8`): `tenferro-direct` is 12.0x faster than the slowest successful cell (`jax-cpu`, 0.052 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`8x4x4`): `tenferro-direct` is 11.4x faster than the slowest successful cell (`jax-cpu`, 0.046 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=1, shape=`1x2x2`): `tenferro-direct` is 10.5x faster than the slowest successful cell (`jax-cpu`, 0.020 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`1x2x2`): `tenferro-direct` is 21.1x faster than the slowest successful cell (`tenferro-trace`, 0.039 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`1x4x4`): `tenferro-direct` is 11.9x faster than the slowest successful cell (`tenferro-trace`, 0.052 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`4x2x2`): `tenferro-direct` is 10.3x faster than the slowest successful cell (`tenferro-trace`, 0.056 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=1, shape=`1x2x2,rhs=1`): `tenferro-direct` is 10.3x faster than the slowest successful cell (`pytorch-cpu`, 0.019 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=1, shape=`1x4x4,rhs=1`): `tenferro-direct` is 10.2x faster than the slowest successful cell (`pytorch-cpu`, 0.018 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=1, shape=`3x2x2,rhs=1`): `tenferro-direct` is 12.2x faster than the slowest successful cell (`jax-cpu`, 0.025 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1x2x2,rhs=1`): `tenferro-direct` is 29.4x faster than the slowest successful cell (`tenferro-trace`, 0.053 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1x4x4,rhs=1`): `tenferro-direct` is 22.1x faster than the slowest successful cell (`tenferro-trace`, 0.059 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1x8x8,rhs=1`): `tenferro-direct` is 15.0x faster than the slowest successful cell (`tenferro-trace`, 0.055 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`3x2x2,rhs=1`): `tenferro-direct` is 23.3x faster than the slowest successful cell (`tenferro-trace`, 0.048 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`3x4x4,rhs=1`): `tenferro-direct` is 22.0x faster than the slowest successful cell (`tenferro-trace`, 0.053 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`4x2x2,rhs=1`): `tenferro-direct` is 13.1x faster than the slowest successful cell (`tenferro-trace`, 0.053 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`4x4x4,rhs=1`): `tenferro-direct` is 11.3x faster than the slowest successful cell (`tenferro-trace`, 0.057 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`4x8x8,rhs=1`): `tenferro-direct` is 16.3x faster than the slowest successful cell (`tenferro-trace`, 0.061 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x2x2,rhs=1`): `tenferro-direct` is 18.2x faster than the slowest successful cell (`tenferro-trace`, 0.052 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x4x4,rhs=1`): `tenferro-direct` is 10.2x faster than the slowest successful cell (`tenferro-trace`, 0.055 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu_tanh` (f32, threads=1, shape=`4096x256`): `jax-cpu` is 17.1x faster than the slowest successful cell (`tenferro-eager`, 17.044 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/gelu_tanh` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 10.1x faster than the slowest successful cell (`tenferro-eager`, 6.704 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=1, shape=`4096x256`): `jax-cpu` is 14.5x faster than the slowest successful cell (`tenferro-eager`, 21.057 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 10.5x faster than the slowest successful cell (`tenferro-eager`, 1.682 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 14.3x faster than the slowest successful cell (`tenferro-eager`, 12.645 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/extract_diagonal` (f64, threads=1, shape=`8388608x2x2 -> 8388608x2`): `jax-cpu` is 10.8x faster than the slowest successful cell (`tenferro-trace`, 204.990 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/extract_diagonal` (f64, threads=1, shape=`8388608x2x2 -> 8388608x2`): `tenferro-direct` is 14.9x faster than the slowest successful cell (`tenferro-trace`, 204.990 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`): `jax-cpu` is 18.8x faster than the slowest successful cell (`julia-base`, 898.852 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`): `pytorch-cpu` is 26.8x faster than the slowest successful cell (`julia-base`, 898.852 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`): `jax-cpu` is 37.3x faster than the slowest successful cell (`julia-base`, 899.332 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`): `pytorch-cpu` is 27.1x faster than the slowest successful cell (`julia-base`, 899.332 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`): `tenferro-direct` is 27.5x faster than the slowest successful cell (`julia-base`, 899.332 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`): `tenferro-trace` is 27.4x faster than the slowest successful cell (`julia-base`, 899.332 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/broadcast_in_dim_view` (f64, threads=4, shape=`8192x1 -> 8192x4096`): `tenferro-direct` is 16.4x faster than the slowest successful cell (`pytorch-cpu`, 0.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=1, shape=`33554432 -> 8192x4096`): `julia-base` is 171.8x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=4, shape=`33554432 -> 8192x4096`): `julia-base` is 76.3x faster than the slowest successful cell (`pytorch-cpu`, 0.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=1, shape=`4194304 -> 2096128`): `julia-base` is 207.1x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=4, shape=`4194304 -> 2096128`): `julia-base` is 263.4x faster than the slowest successful cell (`pytorch-cpu`, 0.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=1, shape=`4096x4096`): `julia-base` is 191.0x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=4, shape=`4096x4096`): `julia-base` is 347.2x faster than the slowest successful cell (`pytorch-cpu`, 0.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

## Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.
