# CPU Public API Benchmark Results

- Suite: `cpu/public_api`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/public_api.yaml`
- Public API coverage manifest: `benchmarks/cpu/public_api_coverage.yaml`
- Run metadata: `data/results/mac-cpu/cpu/public_api/20261023_181453/run.yaml`
- Timestamp: `20261023_181453`

Latest run: `./scripts/run_cpu_public_api.sh 1 4`.

- Sampling: `quick` profile, 7 measured runs, 3 warmups per row

This file is generated from sequential CPU public API runs under `data/results/mac-cpu/cpu/public_api/20261023_181453`.

- tenferro-rs commit: `74d138135bc227c5a1ff41db8863f853b647d863`

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
- Python platform: `macOS-26.6.2-arm64-arm-64bit-Mach-O`

## Thread Environments

### Threads: 1

- Run metadata: `data/results/mac-cpu/cpu/public_api/20261023_181453/run_t1.yaml`
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

- Run metadata: `data/results/mac-cpu/cpu/public_api/20261023_181453/run_t4.yaml`
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

- CSV: `data/results/mac-cpu/cpu/public_api/20261023_181453/cpu_public_api_t1_20261023_181453.csv`
- CSV: `data/results/mac-cpu/cpu/public_api/20261023_181453/cpu_public_api_t4_20261023_181453.csv`
- Source table: `data/results/mac-cpu/cpu/public_api/20261023_181453/cpu_public_api_20261023_181453.md`

## CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| cpu/complex | `cholesky` | c64 | 1 | `448x448` | 3.634 ± 0.444 | - | 3.944 ± 0.443 | 4.162 ± 0.080 | 1.325 ± 0.075 | 8.112 ± 4.818 | - |
| cpu/complex | `cholesky` | c64 | 4 | `448x448` | 3.388 ± 0.035 | - | 3.503 ± 0.055 | 3.756 ± 0.251 | 1.431 ± 0.438 | 76.764 ± 14.943 | - |
| cpu/complex | `conj` | c64 | 1 | `16777216` | 10.050 ± 0.292 | - | 10.283 ± 0.727 | 13.526 ± 0.841 | 13.809 ± 1.022 | 13.778 ± 8.710 | 13.190 ± 3.052 |
| cpu/complex | `conj` | c64 | 4 | `16777216` | 11.394 ± 1.194 | - | 11.821 ± 1.730 | 11.364 ± 2.448 | 9.641 ± 0.557 | 12.707 ± 1.093 | 14.018 ± 0.896 |
| cpu/complex | `div` | c64 | 1 | `8388608` | 43.308 ± 0.514 | - | 44.189 ± 1.462 | 19.676 ± 2.387 | 36.804 ± 4.731 | 43.378 ± 1.132 | 58.306 ± 11.014 |
| cpu/complex | `div` | c64 | 4 | `8388608` | 29.929 ± 7.425 | - | 28.311 ± 13.122 | 9.321 ± 0.827 | 17.829 ± 0.925 | 47.581 ± 7.817 | 25.109 ± 6.880 |
| cpu/complex | `dot_general` | c64 | 1 | `640x640` | 10.796 ± 0.579 | - | 10.968 ± 2.679 | 14.705 ± 10.748 | 48.929 ± 2.755 | 58.495 ± 6.952 | - |
| cpu/complex | `dot_general` | c64 | 4 | `640x640` | 9.335 ± 0.208 | - | 9.661 ± 0.357 | 9.976 ± 1.471 | 17.198 ± 1.352 | 67.038 ± 23.668 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 1 | `640x640` | 13.568 ± 2.173 | - | unsupported | 15.546 ± 5.109 | 51.647 ± 1.369 | 58.425 ± 12.131 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 4 | `640x640` | 10.627 ± 0.362 | - | unsupported | 10.352 ± 0.475 | 18.372 ± 0.517 | 68.079 ± 11.064 | - |
| cpu/complex | `eig` | c64 | 1 | `112x112` | 12.039 ± 0.903 | - | 11.965 ± 0.714 | 10.459 ± 0.955 | 21.811 ± 1.774 | 8.829 ± 0.161 | - |
| cpu/complex | `eig` | c64 | 4 | `112x112` | 10.650 ± 0.176 | - | 11.071 ± 0.862 | 10.492 ± 0.089 | 20.864 ± 1.004 | 457.496 ± 19.814 | - |
| cpu/complex | `exp` | c64 | 1 | `4194304` | 32.816 ± 0.981 | - | 33.216 ± 0.259 | 35.188 ± 2.165 | 50.898 ± 6.665 | 51.949 ± 0.686 | 54.964 ± 1.676 |
| cpu/complex | `exp` | c64 | 4 | `4194304` | 19.019 ± 0.750 | - | 19.428 ± 1.169 | 12.906 ± 1.113 | 22.672 ± 0.421 | 49.820 ± 5.910 | 30.541 ± 33.553 |
| cpu/complex | `log` | c64 | 1 | `4194304` | 44.285 ± 1.059 | - | 43.795 ± 0.318 | 68.255 ± 14.098 | 89.463 ± 11.177 | 82.102 ± 4.259 | 77.323 ± 11.325 |
| cpu/complex | `log` | c64 | 4 | `4194304` | 22.534 ± 3.900 | - | 20.981 ± 2.063 | 15.624 ± 3.446 | 38.631 ± 10.782 | 71.745 ± 2.716 | 39.163 ± 5.195 |
| cpu/complex | `mul` | c64 | 1 | `8388608` | 7.380 ± 0.430 | - | 7.605 ± 0.739 | 9.908 ± 0.726 | 8.436 ± 1.464 | 7.455 ± 0.515 | 7.291 ± 0.192 |
| cpu/complex | `mul` | c64 | 4 | `8388608` | 9.570 ± 1.674 | - | 11.304 ± 0.604 | 7.627 ± 3.076 | 7.818 ± 0.490 | 9.968 ± 0.989 | 11.284 ± 0.588 |
| cpu/complex | `norm_fro` | c64 | 1 | `2048x1536` | 1.338 ± 0.117 | - | 7.869 ± 0.198 | 6.487 ± 0.138 | 5.655 ± 0.618 | 8.211 ± 0.022 | - |
| cpu/complex | `norm_fro` | c64 | 4 | `2048x1536` | 1.062 ± 0.088 | - | 3.779 ± 0.471 | 3.039 ± 0.149 | 3.264 ± 0.138 | 12.462 ± 4.981 | - |
| cpu/complex | `qr` | c64 | 1 | `256x256` | 8.269 ± 0.287 | - | 8.226 ± 0.192 | 7.813 ± 0.283 | 7.170 ± 0.652 | 17.636 ± 5.503 | - |
| cpu/complex | `qr` | c64 | 4 | `256x256` | 7.576 ± 0.151 | - | 7.669 ± 0.267 | 6.606 ± 0.319 | 6.895 ± 0.291 | 174.942 ± 43.572 | - |
| cpu/complex | `solve` | c64 | 1 | `384x384,rhs=8` | 4.692 ± 0.842 | - | 4.996 ± 0.543 | 5.705 ± 0.199 | 2.018 ± 0.035 | 6.388 ± 0.109 | - |
| cpu/complex | `solve` | c64 | 4 | `384x384,rhs=8` | 4.868 ± 0.606 | - | 4.954 ± 0.112 | 5.192 ± 0.298 | 2.844 ± 0.189 | 132.463 ± 31.343 | - |
| cpu/complex | `svd` | c64 | 1 | `160x160` | 7.841 ± 0.178 | - | 8.000 ± 0.325 | 7.172 ± 0.237 | 6.600 ± 0.794 | 11.139 ± 5.085 | - |
| cpu/complex | `svd` | c64 | 4 | `160x160` | 7.305 ± 0.191 | - | 7.308 ± 0.266 | 6.814 ± 0.122 | 5.784 ± 0.028 | 648.650 ± 112.141 | - |
| cpu/complex | `tensordot` | c64 | 1 | `640x640` | 10.747 ± 0.604 | - | 10.905 ± 2.973 | 13.266 ± 4.958 | 49.723 ± 4.819 | 79.462 ± 10.609 | - |
| cpu/complex | `tensordot` | c64 | 4 | `640x640` | 9.681 ± 0.455 | - | 9.282 ± 0.245 | 9.340 ± 1.239 | 18.692 ± 0.640 | 67.058 ± 19.098 | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 1 | `1024x1024` | 9.360 ± 0.717 | - | unsupported | 12.434 ± 3.128 | 50.622 ± 4.030 | - | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 4 | `1024x1024` | 12.793 ± 4.422 | - | unsupported | 8.617 ± 0.507 | 17.342 ± 1.763 | - | - |
| cpu/elementwise_reduction | `abs` | f64 | 1 | `33554432` | 13.932 ± 5.003 | - | 14.065 ± 6.770 | 11.101 ± 0.230 | 16.486 ± 5.579 | 12.126 ± 1.180 | 14.308 ± 1.361 |
| cpu/elementwise_reduction | `abs` | f64 | 4 | `33554432` | 11.230 ± 0.993 | - | 10.427 ± 0.713 | 15.979 ± 5.558 | 10.642 ± 0.672 | 11.469 ± 1.522 | 11.359 ± 0.473 |
| cpu/elementwise_reduction | `add` | f64 | 1 | `33554432` | 18.096 ± 3.343 | - | 17.336 ± 0.297 | 15.855 ± 0.528 | 35.718 ± 3.876 | 19.577 ± 3.190 | 21.027 ± 4.020 |
| cpu/elementwise_reduction | `add` | f64 | 4 | `33554432` | 17.161 ± 1.513 | - | 19.331 ± 2.766 | 19.733 ± 1.030 | 22.429 ± 3.402 | 18.438 ± 1.653 | 18.319 ± 2.280 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 1 | `4194304` | 30.068 ± 1.675 | - | 32.811 ± 6.350 | 41.099 ± 1.008 | 27.864 ± 1.779 | 54.091 ± 0.851 | 55.651 ± 2.441 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 4 | `4194304` | 16.880 ± 1.926 | - | 20.530 ± 4.088 | 29.309 ± 10.561 | 15.923 ± 1.931 | 76.458 ± 20.542 | 26.854 ± 12.119 |
| cpu/elementwise_reduction | `clamp` | f64 | 1 | `8388608` | 6.332 ± 0.905 | - | 8.734 ± 2.175 | 6.337 ± 1.361 | 7.193 ± 0.067 | 6.237 ± 0.887 | 14.959 ± 53.819 |
| cpu/elementwise_reduction | `clamp` | f64 | 4 | `8388608` | 6.383 ± 0.425 | - | 6.983 ± 2.385 | 6.708 ± 1.239 | 6.864 ± 0.704 | 5.438 ± 0.478 | 6.667 ± 2.324 |
| cpu/elementwise_reduction | `compare_lt` | f64 | 1 | `33554432` | 16.297 ± 8.064 | - | 13.947 ± 5.495 | 14.002 ± 0.323 | 15.729 ± 2.680 | 11.931 ± 1.292 | - |
| cpu/elementwise_reduction | `compare_lt` | f64 | 4 | `33554432` | 11.334 ± 0.683 | - | 12.500 ± 1.998 | 15.553 ± 2.397 | 11.810 ± 0.682 | 12.474 ± 0.636 | - |
| cpu/elementwise_reduction | `cos` | f64 | 1 | `8388608` | 36.949 ± 3.233 | - | 35.317 ± 4.379 | 36.176 ± 6.680 | 31.454 ± 0.969 | 39.494 ± 5.933 | 35.196 ± 6.791 |
| cpu/elementwise_reduction | `cos` | f64 | 4 | `8388608` | 18.617 ± 14.341 | - | 21.030 ± 5.052 | 19.921 ± 6.967 | 14.186 ± 0.237 | 39.325 ± 6.376 | 19.035 ± 3.705 |
| cpu/elementwise_reduction | `div` | f64 | 1 | `33554432` | 22.005 ± 3.335 | - | 20.569 ± 2.432 | 17.934 ± 0.502 | 28.589 ± 5.981 | 19.642 ± 1.008 | 20.206 ± 2.979 |
| cpu/elementwise_reduction | `div` | f64 | 4 | `33554432` | 17.832 ± 1.641 | - | 17.483 ± 2.982 | 23.806 ± 6.204 | 20.141 ± 0.599 | 14.463 ± 1.248 | 14.742 ± 0.677 |
| cpu/elementwise_reduction | `exp` | f64 | 1 | `8388608` | 16.931 ± 1.551 | - | 18.459 ± 4.240 | 28.526 ± 1.693 | 18.762 ± 0.391 | 40.657 ± 11.529 | 33.432 ± 4.269 |
| cpu/elementwise_reduction | `exp` | f64 | 4 | `8388608` | 12.350 ± 2.543 | - | 12.712 ± 4.547 | 17.328 ± 1.782 | 8.563 ± 0.318 | 35.475 ± 0.456 | 15.466 ± 5.988 |
| cpu/elementwise_reduction | `expm1` | f64 | 1 | `4194304` | 32.104 ± 3.285 | - | 25.632 ± 1.291 | 25.201 ± 0.095 | 17.404 ± 0.561 | 26.441 ± 6.809 | 24.723 ± 6.380 |
| cpu/elementwise_reduction | `expm1` | f64 | 4 | `4194304` | 15.038 ± 1.177 | - | 15.299 ± 3.233 | 17.929 ± 5.461 | 9.577 ± 2.877 | 32.959 ± 8.539 | 25.924 ± 12.422 |
| cpu/elementwise_reduction | `log` | f64 | 1 | `8388608` | 29.617 ± 1.019 | - | 32.917 ± 0.596 | 40.150 ± 2.805 | 29.164 ± 0.290 | 38.385 ± 2.052 | 47.874 ± 12.335 |
| cpu/elementwise_reduction | `log` | f64 | 4 | `8388608` | 21.105 ± 2.755 | - | 19.718 ± 3.576 | 30.228 ± 5.581 | 13.169 ± 0.199 | 48.105 ± 2.793 | 25.525 ± 2.650 |
| cpu/elementwise_reduction | `log1p` | f64 | 1 | `4194304` | 16.053 ± 0.716 | - | 16.331 ± 2.250 | 20.103 ± 0.475 | 19.286 ± 2.998 | 25.274 ± 1.426 | 25.795 ± 2.597 |
| cpu/elementwise_reduction | `log1p` | f64 | 4 | `4194304` | 8.701 ± 1.926 | - | 9.474 ± 1.117 | 14.176 ± 1.368 | 11.905 ± 1.309 | 28.648 ± 14.648 | 18.464 ± 11.764 |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 1 | `1024x64` | 0.254 ± 0.030 | 0.273 ± 0.016 | unsupported | unsupported | 0.263 ± 0.033 | - | - |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 4 | `1024x64` | 0.222 ± 0.003 | 0.276 ± 0.037 | unsupported | unsupported | 0.203 ± 0.045 | - | - |
| cpu/elementwise_reduction | `maximum` | f64 | 1 | `33554432` | 20.003 ± 2.663 | - | 20.795 ± 1.678 | 17.679 ± 0.276 | 22.538 ± 4.056 | 20.583 ± 1.592 | 24.225 ± 4.592 |
| cpu/elementwise_reduction | `maximum` | f64 | 4 | `33554432` | 16.269 ± 0.593 | - | 15.883 ± 0.973 | 20.523 ± 2.940 | 20.375 ± 2.139 | 14.756 ± 0.241 | 15.098 ± 1.782 |
| cpu/elementwise_reduction | `minimum` | f64 | 1 | `33554432` | 19.128 ± 5.253 | - | 16.191 ± 1.921 | 14.873 ± 0.842 | 24.978 ± 0.841 | 20.607 ± 0.927 | 21.552 ± 7.230 |
| cpu/elementwise_reduction | `minimum` | f64 | 4 | `33554432` | 16.391 ± 0.918 | - | 18.677 ± 2.964 | 20.785 ± 4.230 | 20.047 ± 0.847 | 14.572 ± 0.834 | 14.789 ± 0.815 |
| cpu/elementwise_reduction | `mul` | f64 | 1 | `33554432` | 17.220 ± 1.467 | - | 26.984 ± 6.975 | 19.444 ± 2.758 | 44.689 ± 9.341 | 18.232 ± 1.484 | 22.155 ± 53.598 |
| cpu/elementwise_reduction | `mul` | f64 | 4 | `33554432` | 17.661 ± 2.025 | - | 16.507 ± 2.054 | 19.897 ± 1.180 | 21.889 ± 0.525 | 16.736 ± 3.059 | 14.728 ± 0.244 |
| cpu/elementwise_reduction | `neg` | f64 | 1 | `33554432` | 12.856 ± 0.714 | - | 16.139 ± 4.726 | 12.869 ± 1.882 | 13.793 ± 2.394 | 11.539 ± 0.435 | 14.919 ± 0.218 |
| cpu/elementwise_reduction | `neg` | f64 | 4 | `33554432` | 11.834 ± 2.196 | - | 11.365 ± 2.455 | 12.674 ± 1.482 | 10.284 ± 0.836 | 9.525 ± 0.088 | 9.988 ± 0.481 |
| cpu/elementwise_reduction | `pow` | f64 | 1 | `4194304` | 42.967 ± 3.229 | - | 40.951 ± 2.939 | 64.186 ± 1.428 | 47.050 ± 4.297 | 79.696 ± 9.110 | 86.427 ± 10.717 |
| cpu/elementwise_reduction | `pow` | f64 | 4 | `4194304` | 21.220 ± 4.360 | - | 19.797 ± 1.835 | 34.299 ± 4.061 | 20.109 ± 4.340 | 80.972 ± 7.806 | 39.292 ± 11.018 |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 1 | `8192x4096` | 7.346 ± 0.691 | - | 7.638 ± 0.933 | 13.919 ± 0.036 | 20.765 ± 0.420 | 7.058 ± 0.542 | - |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 4 | `8192x4096` | 5.757 ± 0.183 | - | 5.581 ± 0.435 | 6.159 ± 1.205 | 10.999 ± 0.485 | 7.638 ± 5.524 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 1 | `2048x2048` | 0.857 ± 0.106 | - | 0.895 ± 0.126 | 17.993 ± 0.425 | 2.120 ± 0.267 | 0.815 ± 0.346 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 4 | `2048x2048` | 0.880 ± 0.169 | - | 0.801 ± 0.044 | 12.648 ± 0.868 | 1.311 ± 0.065 | 1.004 ± 0.846 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 1 | `2048x2048` | 0.984 ± 0.043 | - | 1.101 ± 1.113 | 1.876 ± 0.029 | 0.856 ± 0.043 | 0.880 ± 0.189 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 4 | `2048x2048` | 1.437 ± 0.410 | - | 1.387 ± 0.044 | 1.120 ± 0.065 | 0.803 ± 0.078 | 1.415 ± 0.936 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 1 | `8192x4096` | 7.377 ± 2.562 | - | 7.893 ± 1.087 | 13.830 ± 0.455 | 20.792 ± 0.638 | 6.607 ± 0.504 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 4 | `8192x4096` | 5.674 ± 0.335 | - | 6.071 ± 0.310 | 7.701 ± 0.479 | 11.124 ± 0.288 | 7.345 ± 0.873 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 1 | `2048x2048` | 1.047 ± 0.300 | - | 1.074 ± 0.115 | 15.641 ± 1.887 | 2.257 ± 0.288 | 0.857 ± 0.042 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 4 | `2048x2048` | 0.783 ± 0.020 | - | 0.751 ± 0.039 | 12.441 ± 0.243 | 1.253 ± 0.131 | 0.855 ± 0.090 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 1 | `4096x4096` | 3.069 ± 0.331 | - | 4.352 ± 0.496 | 7.258 ± 0.137 | 3.114 ± 0.141 | 3.732 ± 0.165 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 4 | `4096x4096` | 6.306 ± 1.226 | - | 5.608 ± 0.579 | 3.829 ± 0.078 | 2.556 ± 0.428 | 3.571 ± 0.400 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 1 | `8192x4096` | 7.756 ± 1.788 | - | 6.874 ± 1.354 | 6.573 ± 0.092 | 39.869 ± 1.809 | 6.450 ± 0.594 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 4 | `8192x4096` | 5.378 ± 0.305 | - | 5.566 ± 0.212 | 6.070 ± 0.266 | 20.504 ± 0.397 | 7.392 ± 2.686 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 1 | `2048x2048` | 0.928 ± 0.066 | - | 1.057 ± 0.097 | 4.097 ± 0.346 | 2.123 ± 0.215 | 1.390 ± 0.094 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 4 | `2048x2048` | 0.692 ± 0.039 | - | 0.691 ± 0.009 | 2.842 ± 0.163 | 1.487 ± 1.098 | 0.916 ± 0.162 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 1 | `2048x2048` | 1.090 ± 0.780 | - | 1.203 ± 0.542 | 0.766 ± 0.046 | 1.127 ± 0.035 | 0.686 ± 0.046 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 4 | `2048x2048` | 1.247 ± 0.033 | - | 1.586 ± 0.097 | 0.813 ± 0.076 | 0.853 ± 0.096 | 0.902 ± 0.569 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 1 | `8192x4096` | 5.391 ± 0.630 | - | 7.145 ± 3.348 | 5.593 ± 0.090 | 41.088 ± 11.514 | 5.881 ± 0.541 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 4 | `8192x4096` | 5.956 ± 0.382 | - | 6.718 ± 0.625 | 5.782 ± 0.233 | 15.782 ± 0.559 | 9.843 ± 9.278 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 1 | `2048x2048` | 1.059 ± 0.406 | - | 1.128 ± 0.247 | 4.147 ± 0.343 | 2.496 ± 0.405 | 0.749 ± 0.105 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 4 | `2048x2048` | 0.682 ± 0.075 | - | 0.707 ± 0.044 | 3.063 ± 0.165 | 1.386 ± 0.157 | 1.112 ± 0.554 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 1 | `2048x2048` | 0.953 ± 0.386 | - | 1.117 ± 0.208 | 0.652 ± 0.034 | 0.993 ± 0.047 | 0.685 ± 0.021 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 4 | `2048x2048` | 1.576 ± 0.316 | - | 1.562 ± 0.542 | 0.855 ± 0.075 | 0.805 ± 0.069 | 1.664 ± 0.958 | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 1 | `2048x2048` | 0.940 ± 0.233 | 0.874 ± 0.114 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 4 | `2048x2048` | 0.714 ± 0.014 | 0.699 ± 0.084 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `rem` | f64 | 1 | `8388608` | 29.026 ± 4.366 | - | 28.082 ± 0.900 | 32.493 ± 3.564 | 39.551 ± 0.365 | 38.570 ± 37.280 | 40.968 ± 16.041 |
| cpu/elementwise_reduction | `rem` | f64 | 4 | `8388608` | 15.558 ± 3.150 | - | 14.283 ± 1.982 | 22.203 ± 3.234 | 20.208 ± 1.075 | 29.337 ± 3.308 | 12.221 ± 2.435 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 1 | `33554432` | 27.282 ± 15.869 | - | 19.313 ± 1.439 | 18.819 ± 1.094 | 42.300 ± 0.388 | 42.299 ± 1.343 | 37.967 ± 0.909 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 4 | `33554432` | 13.055 ± 1.232 | - | 13.258 ± 2.490 | 14.715 ± 2.402 | 19.192 ± 3.648 | 42.825 ± 1.495 | 14.745 ± 0.218 |
| cpu/elementwise_reduction | `select` | f64 | 1 | `33554432` | 22.445 ± 1.764 | - | 21.758 ± 3.893 | 18.569 ± 1.490 | 25.970 ± 2.152 | 19.756 ± 3.351 | - |
| cpu/elementwise_reduction | `select` | f64 | 4 | `33554432` | 22.843 ± 4.011 | - | 22.579 ± 0.698 | 21.726 ± 4.701 | 23.436 ± 1.304 | 21.228 ± 1.024 | - |
| cpu/elementwise_reduction | `sign` | f64 | 1 | `33554432` | 12.547 ± 5.646 | - | 11.115 ± 3.312 | 11.216 ± 1.576 | 15.652 ± 0.746 | 11.569 ± 1.023 | 20.025 ± 0.131 |
| cpu/elementwise_reduction | `sign` | f64 | 4 | `33554432` | 10.589 ± 0.727 | - | 10.110 ± 0.293 | 14.121 ± 6.407 | 10.948 ± 0.582 | 10.334 ± 0.732 | 11.391 ± 10.082 |
| cpu/elementwise_reduction | `sin` | f64 | 1 | `8388608` | 27.610 ± 7.166 | - | 26.775 ± 3.365 | 29.809 ± 2.400 | 28.732 ± 0.213 | 33.893 ± 0.501 | 35.058 ± 0.309 |
| cpu/elementwise_reduction | `sin` | f64 | 4 | `8388608` | 16.278 ± 2.653 | - | 17.575 ± 6.225 | 20.642 ± 4.983 | 12.826 ± 0.319 | 33.301 ± 2.882 | 17.948 ± 6.177 |
| cpu/elementwise_reduction | `sqrt` | f64 | 1 | `33554432` | 11.377 ± 1.197 | - | 18.497 ± 11.269 | 11.913 ± 1.170 | 25.273 ± 2.442 | 28.276 ± 1.149 | 27.564 ± 0.338 |
| cpu/elementwise_reduction | `sqrt` | f64 | 4 | `33554432` | 12.447 ± 1.413 | - | 12.557 ± 0.615 | 13.764 ± 1.005 | 11.312 ± 0.811 | 26.823 ± 2.745 | 11.301 ± 0.829 |
| cpu/elementwise_reduction | `sub` | f64 | 1 | `33554432` | 17.861 ± 2.255 | - | 20.444 ± 6.546 | 17.540 ± 1.603 | 34.533 ± 6.136 | 23.459 ± 10.263 | 26.637 ± 11.591 |
| cpu/elementwise_reduction | `sub` | f64 | 4 | `33554432` | 17.625 ± 1.847 | - | 16.989 ± 2.922 | 20.044 ± 3.102 | 28.173 ± 5.165 | 19.904 ± 1.263 | 21.205 ± 16.505 |
| cpu/elementwise_reduction | `tanh` | f64 | 1 | `8388608` | 64.429 ± 0.906 | - | 63.688 ± 9.390 | 65.640 ± 8.067 | 15.093 ± 0.646 | 44.639 ± 2.575 | 42.647 ± 1.318 |
| cpu/elementwise_reduction | `tanh` | f64 | 4 | `8388608` | 36.962 ± 5.985 | - | 35.084 ± 12.443 | 39.190 ± 6.041 | 6.704 ± 0.122 | 48.409 ± 6.353 | 28.044 ± 8.172 |
| cpu/elementwise_reduction | `where_select` | f64 | 1 | `33554432` | 24.176 ± 5.222 | 22.545 ± 5.162 | unsupported | 20.649 ± 1.586 | 26.811 ± 5.609 | - | - |
| cpu/elementwise_reduction | `where_select` | f64 | 4 | `33554432` | 23.778 ± 1.695 | 24.545 ± 0.846 | unsupported | 20.543 ± 2.974 | 21.160 ± 1.375 | - | - |
| cpu/indexing_layout | `concatenate` | f64 | 1 | `1048576+1048576` | 0.625 ± 0.095 | - | 0.779 ± 0.094 | 0.801 ± 0.054 | 0.640 ± 0.045 | 0.703 ± 0.068 | - |
| cpu/indexing_layout | `concatenate` | f64 | 4 | `1048576+1048576` | 0.689 ± 0.096 | - | 0.903 ± 0.542 | 0.988 ± 0.089 | 0.664 ± 0.042 | 0.823 ± 0.074 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 1 | `4194304 -> 2097152` | 0.946 ± 0.097 | - | 0.765 ± 0.070 | 0.637 ± 0.112 | 0.648 ± 0.036 | 0.715 ± 0.068 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 4 | `4194304 -> 2097152` | 0.749 ± 0.139 | - | 0.817 ± 0.149 | 0.847 ± 0.077 | 0.725 ± 0.054 | 1.004 ± 0.102 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 1 | `2097152` | 1.012 ± 0.046 | - | unsupported | 1.107 ± 0.059 | 0.974 ± 0.041 | 1.056 ± 0.158 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 4 | `2097152` | 1.309 ± 0.093 | - | unsupported | 1.380 ± 0.101 | 0.877 ± 0.025 | 1.077 ± 0.166 | - |
| cpu/indexing_layout | `gather` | f64 | 1 | `262144` | 0.327 ± 0.061 | - | 0.385 ± 0.044 | 0.318 ± 0.041 | 0.368 ± 0.010 | 2.150 ± 0.266 | - |
| cpu/indexing_layout | `gather` | f64 | 4 | `262144` | 0.510 ± 0.112 | - | 1.010 ± 0.455 | 0.329 ± 0.030 | 0.223 ± 0.033 | 1.297 ± 2.248 | - |
| cpu/indexing_layout | `pad` | f64 | 1 | `2097152` | 0.721 ± 0.065 | - | 0.761 ± 0.117 | 1.081 ± 0.094 | 0.815 ± 0.096 | - | - |
| cpu/indexing_layout | `pad` | f64 | 4 | `2097152` | 0.674 ± 0.129 | - | 0.696 ± 0.061 | 1.285 ± 0.078 | 0.731 ± 0.024 | - | - |
| cpu/indexing_layout | `reverse` | f64 | 1 | `2097152` | 0.561 ± 0.069 | - | 0.621 ± 0.090 | 0.759 ± 0.218 | 0.758 ± 0.072 | 1.328 ± 0.286 | - |
| cpu/indexing_layout | `reverse` | f64 | 4 | `2097152` | 0.812 ± 0.202 | - | 0.787 ± 0.100 | 1.173 ± 0.089 | 0.699 ± 0.086 | 1.414 ± 0.398 | - |
| cpu/indexing_layout | `scatter` | f64 | 1 | `262144` | 0.570 ± 0.074 | - | 0.702 ± 0.090 | 0.381 ± 0.049 | 0.403 ± 0.059 | 0.378 ± 0.051 | - |
| cpu/indexing_layout | `scatter` | f64 | 4 | `262144` | 0.955 ± 0.197 | - | 1.129 ± 0.318 | 0.580 ± 0.035 | 0.401 ± 0.017 | 1.130 ± 0.084 | - |
| cpu/indexing_layout | `slice` | f64 | 1 | `4194304 -> 2096128` | 1.333 ± 0.186 | - | 0.002 ± 0.004 | 1.194 ± 0.178 | 1.123 ± 0.043 | 1.769 ± 0.378 | - |
| cpu/indexing_layout | `slice` | f64 | 4 | `4194304 -> 2096128` | 1.254 ± 0.188 | - | 0.003 ± 0.000 | 1.382 ± 0.059 | 1.245 ± 0.272 | 2.560 ± 1.830 | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x2x2` | 0.060 ± 0.000 | - | 0.064 ± 0.000 | 0.275 ± 0.012 | 0.082 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x4x4` | 0.095 ± 0.001 | - | 0.102 ± 0.002 | 0.203 ± 0.054 | 0.119 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x8x8` | 0.199 ± 0.002 | - | 0.218 ± 0.005 | 0.254 ± 0.017 | 0.196 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x128x128` | 0.025 ± 0.001 | - | 0.039 ± 0.001 | 0.041 ± 0.002 | 0.073 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.013 ± 0.013 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x32x32` | 0.002 ± 0.000 | - | 0.006 ± 0.001 | 0.006 ± 0.000 | 0.045 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x4x4` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.011 ± 0.001 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x8x8` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x2x2` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x4x4` | 0.001 ± 0.000 | - | 0.005 ± 0.000 | 0.012 ± 0.001 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x8x8` | 0.001 ± 0.000 | - | 0.005 ± 0.001 | 0.005 ± 0.000 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x2x2` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.001 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x4x4` | 0.001 ± 0.000 | - | 0.005 ± 0.000 | 0.005 ± 0.001 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x8x8` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.044 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x128x128` | 0.172 ± 0.008 | - | 0.231 ± 0.013 | 0.264 ± 0.018 | 0.230 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x2x2` | 0.001 ± 0.000 | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.008 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x32x32` | 0.010 ± 0.000 | - | 0.017 ± 0.000 | 0.018 ± 0.008 | 0.047 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x4x4` | 0.001 ± 0.000 | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x8x8` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.043 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x2x2` | 0.061 ± 0.003 | - | 0.065 ± 0.001 | 0.111 ± 0.005 | 0.105 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x4x4` | 0.099 ± 0.011 | - | 0.107 ± 0.005 | 0.142 ± 0.003 | 0.127 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x8x8` | 0.203 ± 0.010 | - | 0.255 ± 0.074 | 0.195 ± 0.004 | 0.215 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x128x128` | 0.026 ± 0.001 | - | 0.038 ± 0.006 | 0.086 ± 0.003 | 0.060 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x2x2` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.042 ± 0.014 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x32x32` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.046 ± 0.004 | 0.074 ± 0.028 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x4x4` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.046 ± 0.004 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x8x8` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.042 ± 0.013 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x2x2` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.047 ± 0.006 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x4x4` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.042 ± 0.010 | 0.007 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x8x8` | 0.001 ± 0.000 | - | 0.005 ± 0.000 | 0.042 ± 0.012 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x2x2` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.043 ± 0.010 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x4x4` | 0.001 ± 0.000 | - | 0.004 ± 0.001 | 0.043 ± 0.007 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x8x8` | 0.001 ± 0.000 | - | 0.005 ± 0.000 | 0.041 ± 0.008 | 0.044 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x128x128` | 0.252 ± 0.124 | - | 0.276 ± 0.051 | 0.253 ± 0.021 | 0.153 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x2x2` | 0.001 ± 0.000 | - | 0.005 ± 0.000 | 0.046 ± 0.004 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x32x32` | 0.011 ± 0.000 | - | 0.017 ± 0.002 | 0.067 ± 0.009 | 0.062 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x4x4` | 0.001 ± 0.000 | - | 0.005 ± 0.000 | 0.039 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x8x8` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.042 ± 0.013 | 0.076 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x2x2` | 0.255 ± 0.034 | - | 0.249 ± 0.011 | 0.426 ± 0.008 | 0.731 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x4x4` | 1.428 ± 0.035 | - | 1.442 ± 0.042 | 1.758 ± 0.025 | 1.878 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x8x8` | 5.912 ± 0.249 | - | 5.931 ± 0.241 | 7.258 ± 0.742 | 6.399 ± 0.112 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x128x128` | 6.061 ± 0.135 | - | 6.136 ± 0.182 | 3.344 ± 0.198 | 4.040 ± 0.208 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.005 ± 0.001 | 0.005 ± 0.001 | 0.011 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x32x32` | 0.087 ± 0.000 | - | 0.090 ± 0.000 | 0.094 ± 0.007 | 0.138 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x4x4` | 0.003 ± 0.001 | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x8x8` | 0.009 ± 0.000 | - | 0.010 ± 0.000 | 0.013 ± 0.002 | 0.019 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.005 ± 0.001 | 0.006 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x4x4` | 0.006 ± 0.001 | - | 0.008 ± 0.000 | 0.009 ± 0.000 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x8x8` | 0.019 ± 0.003 | - | 0.036 ± 0.005 | 0.024 ± 0.003 | 0.058 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | 0.006 ± 0.001 | 0.006 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x4x4` | 0.008 ± 0.001 | - | 0.010 ± 0.000 | 0.011 ± 0.000 | 0.018 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x8x8` | 0.037 ± 0.001 | - | 0.044 ± 0.005 | 0.029 ± 0.001 | 0.062 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x128x128` | 48.965 ± 1.335 | - | 49.404 ± 1.246 | 27.929 ± 2.156 | 31.691 ± 1.564 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x2x2` | 0.004 ± 0.001 | - | 0.007 ± 0.002 | 0.008 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x32x32` | 0.681 ± 0.020 | - | 0.727 ± 0.046 | 0.726 ± 0.042 | 0.882 ± 0.080 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x4x4` | 0.012 ± 0.003 | - | 0.015 ± 0.000 | 0.018 ± 0.004 | 0.028 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x8x8` | 0.067 ± 0.001 | - | 0.050 ± 0.017 | 0.054 ± 0.005 | 0.114 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x2x2` | 0.251 ± 0.010 | - | 0.259 ± 0.009 | 0.420 ± 0.007 | 0.827 ± 0.051 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x4x4` | 1.523 ± 0.067 | - | 1.494 ± 0.038 | 1.621 ± 0.053 | 1.106 ± 0.028 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x8x8` | 6.153 ± 0.036 | - | 6.173 ± 0.119 | 6.303 ± 0.408 | 3.403 ± 0.037 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x128x128` | 6.256 ± 0.134 | - | 6.072 ± 0.067 | 3.067 ± 0.046 | 3.561 ± 0.217 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x32x32` | 0.090 ± 0.001 | - | 0.093 ± 0.001 | 0.094 ± 0.009 | 0.121 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x4x4` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.006 ± 0.001 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x8x8` | 0.007 ± 0.001 | - | 0.010 ± 0.000 | 0.010 ± 0.002 | 0.019 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x2x2` | 0.002 ± 0.000 | - | 0.006 ± 0.001 | 0.006 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x4x4` | 0.007 ± 0.001 | - | 0.010 ± 0.002 | 0.010 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x8x8` | 0.019 ± 0.004 | - | 0.022 ± 0.003 | 0.022 ± 0.000 | 0.076 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x2x2` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.006 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x4x4` | 0.007 ± 0.000 | - | 0.010 ± 0.002 | 0.011 ± 0.000 | 0.018 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x8x8` | 0.025 ± 0.001 | - | 0.028 ± 0.003 | 0.028 ± 0.000 | 0.084 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x128x128` | 49.279 ± 0.974 | - | 49.180 ± 1.795 | 25.392 ± 0.656 | 16.004 ± 0.703 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x2x2` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.008 ± 0.000 | 0.017 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x32x32` | 0.695 ± 0.006 | - | 0.707 ± 0.012 | 0.672 ± 0.010 | 0.535 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x4x4` | 0.015 ± 0.003 | - | 0.016 ± 0.004 | 0.017 ± 0.000 | 0.042 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x8x8` | 0.048 ± 0.001 | - | 0.052 ± 0.001 | 0.052 ± 0.001 | 0.111 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x2x2` | 0.246 ± 0.015 | - | 0.243 ± 0.019 | 0.317 ± 0.142 | 0.494 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x4x4` | 1.267 ± 0.177 | - | 1.317 ± 0.107 | 1.562 ± 0.742 | 1.389 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x8x8` | 5.276 ± 0.442 | - | 4.707 ± 0.233 | 6.240 ± 4.203 | 4.901 ± 0.097 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x128x128` | 0.926 ± 0.053 | - | 0.915 ± 0.074 | 1.023 ± 0.420 | 0.936 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.005 ± 0.008 | 0.004 ± 0.002 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x32x32` | 0.062 ± 0.000 | - | 0.065 ± 0.000 | 0.099 ± 0.029 | 0.096 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x4x4` | 0.002 ± 0.003 | - | 0.007 ± 0.000 | 0.007 ± 0.002 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x8x8` | 0.006 ± 0.001 | - | 0.008 ± 0.000 | 0.009 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x2x2` | 0.002 ± 0.001 | - | 0.015 ± 0.009 | 0.006 ± 0.002 | 0.021 ± 0.047 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x4x4` | 0.005 ± 0.001 | - | 0.012 ± 0.007 | 0.008 ± 0.003 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x8x8` | 0.019 ± 0.000 | - | 0.021 ± 0.005 | 0.019 ± 0.001 | 0.024 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | 0.006 ± 0.001 | 0.006 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x4x4` | 0.008 ± 0.003 | - | 0.009 ± 0.001 | 0.011 ± 0.003 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x8x8` | 0.021 ± 0.004 | - | 0.027 ± 0.007 | 0.028 ± 0.005 | 0.028 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x128x128` | 7.604 ± 0.207 | - | 7.402 ± 0.136 | 10.380 ± 4.664 | 7.189 ± 0.159 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x2x2` | 0.004 ± 0.001 | - | 0.006 ± 0.002 | 0.008 ± 0.003 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x32x32` | 0.529 ± 0.050 | - | 0.534 ± 0.033 | 0.540 ± 0.038 | 0.510 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x4x4` | 0.010 ± 0.000 | - | 0.020 ± 0.007 | 0.015 ± 0.033 | 0.020 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x8x8` | 0.046 ± 0.006 | - | 0.051 ± 0.005 | 0.047 ± 0.005 | 0.080 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x2x2` | 0.253 ± 0.035 | - | 0.260 ± 0.019 | 0.230 ± 0.000 | 0.512 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x4x4` | 1.678 ± 0.753 | - | 1.363 ± 0.101 | 1.190 ± 0.061 | 1.064 ± 0.036 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x8x8` | 5.594 ± 0.922 | - | 5.500 ± 0.295 | 4.772 ± 0.253 | 2.797 ± 0.189 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x128x128` | 1.016 ± 0.040 | - | 1.061 ± 0.148 | 0.925 ± 0.021 | 1.132 ± 0.031 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.004 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x32x32` | 0.070 ± 0.010 | - | 0.073 ± 0.033 | 0.065 ± 0.001 | 0.145 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x4x4` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x8x8` | 0.007 ± 0.040 | - | 0.015 ± 0.008 | 0.008 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x2x2` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.006 ± 0.001 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x4x4` | 0.005 ± 0.000 | - | 0.010 ± 0.001 | 0.007 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x8x8` | 0.023 ± 0.007 | - | 0.019 ± 0.001 | 0.018 ± 0.000 | 0.026 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x2x2` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x4x4` | 0.006 ± 0.000 | - | 0.009 ± 0.000 | 0.009 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x8x8` | 0.020 ± 0.004 | - | 0.027 ± 0.006 | 0.022 ± 0.003 | 0.030 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x128x128` | 8.271 ± 0.391 | - | 7.751 ± 0.578 | 7.320 ± 0.102 | 4.409 ± 0.076 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x2x2` | 0.004 ± 0.000 | - | 0.006 ± 0.001 | 0.006 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x32x32` | 0.563 ± 0.098 | - | 0.536 ± 0.071 | 0.487 ± 0.007 | 0.388 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x4x4` | 0.011 ± 0.003 | - | 0.013 ± 0.001 | 0.013 ± 0.000 | 0.022 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x8x8` | 0.038 ± 0.001 | - | 0.044 ± 0.007 | 0.040 ± 0.001 | 0.110 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x2x2` | 0.145 ± 0.008 | - | 0.152 ± 0.024 | 0.137 ± 0.016 | 0.600 ± 0.028 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x4x4` | 0.709 ± 0.007 | - | 0.720 ± 0.007 | 0.797 ± 0.067 | 1.948 ± 0.364 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x8x8` | 2.394 ± 0.031 | - | 2.352 ± 0.075 | 2.553 ± 0.079 | 5.616 ± 0.238 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x128x128` | 0.429 ± 0.028 | - | 0.432 ± 0.001 | 0.486 ± 0.059 | 1.078 ± 0.045 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x2x2` | 0.001 ± 0.001 | - | 0.006 ± 0.000 | 0.005 ± 0.000 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x32x32` | 0.024 ± 0.000 | - | 0.027 ± 0.000 | 0.031 ± 0.004 | 0.097 ± 0.034 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.007 ± 0.001 | 0.008 ± 0.001 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x8x8` | 0.004 ± 0.001 | - | 0.010 ± 0.001 | 0.008 ± 0.000 | 0.036 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.007 ± 0.001 | 0.005 ± 0.000 | 0.023 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x4x4` | 0.004 ± 0.001 | - | 0.008 ± 0.001 | 0.008 ± 0.000 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x8x8` | 0.010 ± 0.001 | - | 0.013 ± 0.000 | 0.012 ± 0.004 | 0.022 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.007 ± 0.002 | 0.006 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x4x4` | 0.005 ± 0.001 | - | 0.010 ± 0.002 | 0.008 ± 0.000 | 0.023 ± 0.030 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x8x8` | 0.012 ± 0.001 | - | 0.019 ± 0.004 | 0.015 ± 0.000 | 0.027 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x128x128` | 3.440 ± 0.036 | - | 3.441 ± 0.126 | 4.084 ± 1.566 | 7.997 ± 0.592 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | 0.007 ± 0.002 | 0.006 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x32x32` | 0.182 ± 0.002 | - | 0.213 ± 0.022 | 0.199 ± 0.032 | 0.599 ± 0.031 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x4x4` | 0.008 ± 0.000 | - | 0.011 ± 0.000 | 0.011 ± 0.000 | 0.023 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x8x8` | 0.019 ± 0.005 | - | 0.025 ± 0.000 | 0.024 ± 0.000 | 0.117 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x2x2` | 0.110 ± 0.002 | - | 0.113 ± 0.001 | 0.127 ± 0.002 | 0.559 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x4x4` | 0.782 ± 0.053 | - | 0.760 ± 0.037 | 0.702 ± 0.002 | 0.979 ± 0.034 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x8x8` | 2.425 ± 0.069 | - | 2.558 ± 0.104 | 2.298 ± 0.023 | 3.606 ± 0.875 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x128x128` | 0.467 ± 0.015 | - | 0.450 ± 0.021 | 0.495 ± 0.017 | 1.019 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x2x2` | 0.001 ± 0.001 | - | 0.006 ± 0.000 | 0.005 ± 0.002 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x32x32` | 0.026 ± 0.001 | - | 0.029 ± 0.000 | 0.030 ± 0.003 | 0.115 ± 0.044 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x8x8` | 0.003 ± 0.000 | - | 0.017 ± 0.024 | 0.007 ± 0.000 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x2x2` | 0.002 ± 0.000 | - | 0.012 ± 0.013 | 0.005 ± 0.000 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x8x8` | 0.008 ± 0.000 | - | 0.011 ± 0.003 | 0.011 ± 0.000 | 0.024 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x2x2` | 0.002 ± 0.000 | - | 0.013 ± 0.006 | 0.005 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x4x4` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.007 ± 0.001 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x8x8` | 0.010 ± 0.000 | - | 0.015 ± 0.003 | 0.013 ± 0.000 | 0.029 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x128x128` | 3.851 ± 0.963 | - | 4.043 ± 0.130 | 3.565 ± 0.339 | 4.047 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x2x2` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x32x32` | 0.193 ± 0.008 | - | 0.196 ± 0.003 | 0.210 ± 0.010 | 0.501 ± 0.047 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x4x4` | 0.006 ± 0.000 | - | 0.010 ± 0.002 | 0.011 ± 0.002 | 0.019 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x8x8` | 0.019 ± 0.000 | - | 0.022 ± 0.000 | 0.022 ± 0.000 | 0.098 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x2x2` | 0.092 ± 0.015 | - | 0.102 ± 0.014 | 0.184 ± 0.006 | 0.159 ± 0.028 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x4x4` | 0.208 ± 0.007 | - | 0.325 ± 0.067 | 0.267 ± 0.001 | 0.199 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x8x8` | 0.456 ± 0.028 | - | 0.484 ± 0.017 | 0.511 ± 0.035 | 0.392 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x128x128` | 0.061 ± 0.003 | - | 0.072 ± 0.006 | 0.063 ± 0.001 | 0.280 ± 0.078 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.007 ± 0.001 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x32x32` | 0.005 ± 0.000 | - | 0.008 ± 0.000 | 0.011 ± 0.002 | 0.100 ± 0.081 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.007 ± 0.002 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x64x24` | 0.009 ± 0.000 | - | 0.013 ± 0.002 | 0.013 ± 0.000 | 0.077 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x8x8` | 0.008 ± 0.001 | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x2x2` | 0.003 ± 0.001 | - | 0.004 ± 0.000 | 0.011 ± 0.001 | 0.012 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x8x8` | 0.003 ± 0.001 | - | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.039 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.011 ± 0.002 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x4x4` | 0.003 ± 0.001 | - | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x8x8` | 0.004 ± 0.001 | - | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.039 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x128x128` | 0.486 ± 0.088 | - | 0.534 ± 0.049 | 0.444 ± 0.008 | 0.798 ± 0.871 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | 0.006 ± 0.001 | 0.009 ± 0.002 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x32x32` | 0.032 ± 0.001 | - | 0.041 ± 0.004 | 0.039 ± 0.004 | 0.285 ± 0.222 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x4x4` | 0.004 ± 0.001 | - | 0.006 ± 0.000 | 0.010 ± 0.000 | 0.040 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x8x8` | 0.005 ± 0.001 | - | 0.008 ± 0.000 | 0.011 ± 0.000 | 0.041 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x2x2` | 0.086 ± 0.002 | - | 0.091 ± 0.007 | 0.307 ± 0.054 | 0.174 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x4x4` | 0.197 ± 0.001 | - | 0.203 ± 0.001 | 0.416 ± 0.015 | 0.216 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x8x8` | 0.403 ± 0.004 | - | 0.462 ± 0.017 | 0.698 ± 0.060 | 0.333 ± 0.035 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x128x128` | 0.057 ± 0.007 | - | 0.064 ± 0.002 | 0.113 ± 0.005 | 0.136 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.096 ± 0.008 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x32x32` | 0.007 ± 0.000 | - | 0.008 ± 0.000 | 0.116 ± 0.005 | 0.061 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x4x4` | 0.003 ± 0.001 | - | 0.005 ± 0.000 | 0.076 ± 0.047 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x64x24` | 0.010 ± 0.001 | - | 0.011 ± 0.000 | 0.059 ± 0.006 | 0.065 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x8x8` | 0.003 ± 0.001 | - | 0.005 ± 0.001 | 0.047 ± 0.039 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x2x2` | 0.003 ± 0.001 | - | 0.005 ± 0.000 | 0.129 ± 0.042 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.001 | 0.070 ± 0.037 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x8x8` | 0.003 ± 0.001 | - | 0.005 ± 0.000 | 0.072 ± 0.049 | 0.061 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.180 ± 0.063 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.039 ± 0.008 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x8x8` | 0.004 ± 0.001 | - | 0.007 ± 0.002 | 0.078 ± 0.040 | 0.061 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x128x128` | 0.451 ± 0.055 | - | 0.529 ± 0.034 | 0.373 ± 0.021 | 0.541 ± 0.033 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.104 ± 0.031 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x32x32` | 0.031 ± 0.000 | - | 0.036 ± 0.004 | 0.116 ± 0.060 | 0.121 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x4x4` | 0.004 ± 0.001 | - | 0.006 ± 0.002 | 0.104 ± 0.012 | 0.060 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x8x8` | 0.006 ± 0.001 | - | 0.007 ± 0.000 | 0.107 ± 0.018 | 0.061 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x2x2` | 0.289 ± 0.011 | - | 0.287 ± 0.003 | 0.321 ± 0.047 | 0.308 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x4x4` | 0.535 ± 0.027 | - | 0.537 ± 0.013 | 0.655 ± 0.222 | 0.573 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x8x8` | 1.324 ± 0.031 | - | 1.348 ± 0.402 | 1.343 ± 0.374 | 1.249 ± 0.092 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x128x128` | 0.316 ± 0.004 | - | 0.330 ± 0.013 | 0.434 ± 0.153 | 0.336 ± 0.069 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.008 ± 0.002 | 0.004 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x32x32` | 0.015 ± 0.001 | - | 0.016 ± 0.003 | 0.020 ± 0.002 | 0.060 ± 0.012 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.007 ± 0.001 | 0.004 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x64x24` | 0.021 ± 0.007 | - | 0.030 ± 0.008 | 0.018 ± 0.004 | 0.053 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x8x8` | 0.002 ± 0.000 | - | 0.006 ± 0.001 | 0.004 ± 0.000 | 0.011 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x2x2` | 0.003 ± 0.000 | - | 0.008 ± 0.001 | 0.010 ± 0.021 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x4x4` | 0.004 ± 0.001 | - | 0.009 ± 0.001 | 0.005 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x8x8` | 0.006 ± 0.001 | - | 0.009 ± 0.002 | 0.007 ± 0.000 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | 0.008 ± 0.000 | 0.010 ± 0.011 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x4x4` | 0.004 ± 0.001 | - | 0.011 ± 0.002 | 0.005 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x8x8` | 0.006 ± 0.002 | - | 0.011 ± 0.003 | 0.008 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x128x128` | 2.526 ± 0.024 | - | 2.514 ± 0.103 | 4.031 ± 0.220 | 2.326 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x2x2` | 0.005 ± 0.000 | - | 0.011 ± 0.002 | 0.009 ± 0.010 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x32x32` | 0.097 ± 0.004 | - | 0.101 ± 0.001 | 0.147 ± 0.008 | 0.143 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x4x4` | 0.008 ± 0.001 | - | 0.012 ± 0.000 | 0.007 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x8x8` | 0.013 ± 0.000 | - | 0.014 ± 0.003 | 0.013 ± 0.000 | 0.019 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x2x2` | 0.277 ± 0.012 | - | 0.268 ± 0.025 | 0.238 ± 0.001 | 0.321 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x4x4` | 0.630 ± 0.035 | - | 0.635 ± 0.045 | 0.509 ± 0.010 | 0.746 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x8x8` | 1.821 ± 0.910 | - | 1.414 ± 0.388 | 1.211 ± 0.015 | 0.998 ± 0.512 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x128x128` | 0.312 ± 0.011 | - | 0.621 ± 0.158 | 0.359 ± 0.020 | 0.437 ± 0.042 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x2x2` | 0.003 ± 0.001 | - | 0.007 ± 0.000 | 0.032 ± 0.009 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x32x32` | 0.013 ± 0.003 | - | 0.016 ± 0.003 | 0.031 ± 0.004 | 0.083 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x4x4` | 0.002 ± 0.000 | - | 0.009 ± 0.001 | 0.021 ± 0.002 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x64x24` | 0.015 ± 0.000 | - | 0.023 ± 0.011 | 0.034 ± 0.002 | 0.077 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x8x8` | 0.003 ± 0.000 | - | 0.009 ± 0.000 | 0.022 ± 0.003 | 0.013 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | 0.008 ± 0.000 | 0.026 ± 0.015 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x4x4` | 0.003 ± 0.000 | - | 0.009 ± 0.000 | 0.022 ± 0.002 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x8x8` | 0.006 ± 0.000 | - | 0.012 ± 0.000 | 0.025 ± 0.001 | 0.018 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x2x2` | 0.003 ± 0.000 | - | 0.008 ± 0.001 | 0.022 ± 0.002 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x4x4` | 0.004 ± 0.000 | - | 0.009 ± 0.000 | 0.022 ± 0.002 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x8x8` | 0.008 ± 0.000 | - | 0.013 ± 0.000 | 0.026 ± 0.001 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x128x128` | 2.558 ± 0.087 | - | 2.922 ± 0.326 | 2.529 ± 0.217 | 1.637 ± 0.038 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x2x2` | 0.004 ± 0.000 | - | 0.011 ± 0.002 | 0.022 ± 0.003 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x32x32` | 0.089 ± 0.001 | - | 0.093 ± 0.011 | 0.125 ± 0.009 | 0.374 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x4x4` | 0.006 ± 0.000 | - | 0.014 ± 0.002 | 0.025 ± 0.001 | 0.018 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x8x8` | 0.013 ± 0.000 | - | 0.019 ± 0.003 | 0.028 ± 0.001 | 0.021 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.155 ± 0.018 | - | 0.137 ± 0.013 | 0.135 ± 0.042 | 0.210 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.219 ± 0.009 | - | 0.195 ± 0.011 | 0.314 ± 0.125 | 0.228 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.393 ± 0.017 | - | 0.395 ± 0.010 | 0.488 ± 0.277 | 0.415 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x128x128,rhs=1` | 0.050 ± 0.001 | - | 0.059 ± 0.001 | 0.066 ± 0.003 | 0.137 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x2x2,rhs=1` | 0.001 ± 0.000 | - | 0.013 ± 0.011 | 0.011 ± 0.018 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x32x32,rhs=1` | 0.005 ± 0.000 | - | 0.010 ± 0.002 | 0.014 ± 0.001 | 0.040 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.002 | 0.010 ± 0.003 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x8x8,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.014 ± 0.012 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.011 ± 0.002 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.014 ± 0.005 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x8x8,rhs=1` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.012 ± 0.018 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.013 ± 0.007 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.011 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x8x8,rhs=1` | 0.003 ± 0.001 | - | 0.006 ± 0.000 | 0.013 ± 0.003 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x128x128,rhs=1` | 0.518 ± 0.057 | - | 0.419 ± 0.022 | 1.250 ± 1.006 | 0.563 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x2x2,rhs=1` | 0.003 ± 0.000 | - | 0.007 ± 0.001 | 0.011 ± 0.001 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x32x32,rhs=1` | 0.027 ± 0.004 | - | 0.033 ± 0.003 | 0.038 ± 0.035 | 0.110 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.007 ± 0.002 | 0.011 ± 0.002 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x8x8,rhs=1` | 0.005 ± 0.001 | - | 0.008 ± 0.000 | 0.013 ± 0.003 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.137 ± 0.018 | - | 0.119 ± 0.005 | 0.108 ± 0.000 | 0.203 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.212 ± 0.014 | - | 0.190 ± 0.021 | 0.155 ± 0.006 | 0.259 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.376 ± 0.049 | - | 0.677 ± 0.279 | 0.326 ± 0.009 | 0.342 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x128x128,rhs=1` | 0.059 ± 0.024 | - | 0.062 ± 0.007 | 0.060 ± 0.000 | 0.141 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x2x2,rhs=1` | 0.001 ± 0.000 | - | 0.006 ± 0.016 | 0.009 ± 0.000 | 0.012 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x32x32,rhs=1` | 0.004 ± 0.001 | - | 0.009 ± 0.002 | 0.014 ± 0.000 | 0.061 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.010 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x8x8,rhs=1` | 0.002 ± 0.001 | - | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.011 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.033 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.009 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x8x8,rhs=1` | 0.002 ± 0.000 | - | 0.006 ± 0.000 | 0.010 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.007 ± 0.002 | 0.009 ± 0.000 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.007 ± 0.001 | 0.009 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x8x8,rhs=1` | 0.003 ± 0.001 | - | 0.007 ± 0.001 | 0.010 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x128x128,rhs=1` | 0.404 ± 0.003 | - | 0.567 ± 0.108 | 0.303 ± 0.085 | 0.618 ± 0.070 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.007 ± 0.001 | 0.039 ± 0.013 | 0.069 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x32x32,rhs=1` | 0.032 ± 0.004 | - | 0.038 ± 0.007 | 0.072 ± 0.012 | 0.123 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.007 ± 0.000 | 0.034 ± 0.002 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x8x8,rhs=1` | 0.004 ± 0.000 | - | 0.010 ± 0.002 | 0.030 ± 0.001 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x2x2` | 1.154 ± 0.090 | - | 1.172 ± 0.022 | 1.114 ± 0.041 | 2.774 ± 0.213 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x4x4` | 4.274 ± 0.222 | - | 4.188 ± 0.163 | 4.425 ± 0.400 | 6.621 ± 0.304 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x8x8` | 14.093 ± 0.268 | - | 14.148 ± 0.504 | 15.010 ± 0.844 | 19.443 ± 1.857 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x128x128` | 4.157 ± 0.479 | - | 5.582 ± 0.177 | 8.387 ± 6.373 | 5.646 ± 0.179 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x2x2` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.018 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x32x32` | 0.170 ± 0.002 | - | 0.172 ± 0.007 | 0.231 ± 0.024 | 0.314 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x4x4` | 0.006 ± 0.000 | - | 0.009 ± 0.000 | 0.010 ± 0.000 | 0.018 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x64x24` | 0.169 ± 0.004 | - | 0.188 ± 0.055 | 0.167 ± 0.002 | 0.290 ± 0.028 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x8x8` | 0.019 ± 0.003 | - | 0.020 ± 0.004 | 0.021 ± 0.001 | 0.032 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x2x2` | 0.006 ± 0.001 | - | 0.008 ± 0.000 | 0.009 ± 0.000 | 0.020 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x4x4` | 0.014 ± 0.003 | - | 0.017 ± 0.000 | 0.060 ± 0.155 | 0.028 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x8x8` | 0.053 ± 0.010 | - | 0.046 ± 0.000 | 0.050 ± 0.004 | 0.071 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x2x2` | 0.008 ± 0.001 | - | 0.009 ± 0.000 | 0.010 ± 0.001 | 0.023 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x4x4` | 0.019 ± 0.019 | - | 0.022 ± 0.000 | 0.023 ± 0.000 | 0.033 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x8x8` | 0.059 ± 0.000 | - | 0.061 ± 0.000 | 0.066 ± 0.001 | 0.077 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x128x128` | 33.567 ± 0.798 | - | 47.849 ± 2.002 | 39.226 ± 3.433 | 45.147 ± 3.192 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x2x2` | 0.011 ± 0.002 | - | 0.013 ± 0.000 | 0.015 ± 0.003 | 0.034 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x32x32` | 1.327 ± 0.009 | - | 1.329 ± 0.011 | 1.522 ± 0.301 | 1.932 ± 0.142 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x4x4` | 0.035 ± 0.001 | - | 0.038 ± 0.000 | 0.045 ± 0.013 | 0.059 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x8x8` | 0.112 ± 0.006 | - | 0.115 ± 0.001 | 0.123 ± 0.011 | 0.212 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x2x2` | 1.214 ± 0.185 | - | 1.231 ± 0.066 | 1.123 ± 0.071 | 2.897 ± 0.342 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x4x4` | 4.458 ± 0.169 | - | 4.452 ± 0.125 | 4.036 ± 0.029 | 2.886 ± 1.705 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x8x8` | 14.063 ± 0.374 | - | 14.194 ± 0.296 | 14.158 ± 0.256 | 9.047 ± 0.585 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x128x128` | 4.182 ± 0.120 | - | 5.634 ± 0.063 | 4.572 ± 0.321 | 6.899 ± 1.438 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x2x2` | 0.004 ± 0.002 | - | 0.006 ± 0.000 | 0.009 ± 0.009 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x32x32` | 0.170 ± 0.003 | - | 0.172 ± 0.007 | 0.174 ± 0.004 | 0.269 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x4x4` | 0.006 ± 0.001 | - | 0.009 ± 0.000 | 0.010 ± 0.000 | 0.022 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x64x24` | 0.157 ± 0.001 | - | 0.162 ± 0.003 | 0.162 ± 0.002 | 0.292 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x8x8` | 0.020 ± 0.002 | - | 0.022 ± 0.005 | 0.022 ± 0.004 | 0.036 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x2x2` | 0.006 ± 0.000 | - | 0.010 ± 0.001 | 0.016 ± 0.007 | 0.020 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x4x4` | 0.017 ± 0.002 | - | 0.017 ± 0.000 | 0.018 ± 0.001 | 0.037 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x8x8` | 0.047 ± 0.005 | - | 0.049 ± 0.007 | 0.049 ± 0.002 | 0.082 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x2x2` | 0.007 ± 0.000 | - | 0.010 ± 0.002 | 0.026 ± 0.005 | 0.023 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x4x4` | 0.018 ± 0.005 | - | 0.021 ± 0.004 | 0.022 ± 0.006 | 0.041 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x8x8` | 0.070 ± 0.005 | - | 0.073 ± 0.004 | 0.064 ± 0.006 | 0.100 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x128x128` | 34.101 ± 0.482 | - | 45.306 ± 0.920 | 36.505 ± 0.943 | 23.739 ± 0.971 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x2x2` | 0.013 ± 0.003 | - | 0.018 ± 0.003 | 0.030 ± 0.006 | 0.035 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x32x32` | 1.343 ± 0.047 | - | 1.332 ± 0.033 | 1.361 ± 0.115 | 0.950 ± 0.045 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x4x4` | 0.035 ± 0.001 | - | 0.038 ± 0.003 | 0.037 ± 0.000 | 0.069 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x8x8` | 0.117 ± 0.006 | - | 0.131 ± 0.010 | 0.117 ± 0.014 | 0.252 ± 0.082 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x2x2` | 0.692 ± 0.143 | - | 0.601 ± 0.014 | 0.662 ± 0.124 | 1.973 ± 0.141 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x4x4` | 2.564 ± 0.074 | - | 2.577 ± 0.099 | 5.244 ± 2.736 | 4.513 ± 0.602 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x8x8` | 10.135 ± 0.174 | - | 10.131 ± 0.471 | 10.982 ± 0.514 | 13.579 ± 0.579 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x128x128` | 1.423 ± 0.026 | - | 1.444 ± 0.082 | 1.486 ± 0.138 | 1.698 ± 0.040 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x2x2` | 0.003 ± 0.001 | - | 0.005 ± 0.000 | 0.007 ± 0.001 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x32x32` | 0.113 ± 0.003 | - | 0.115 ± 0.002 | 0.127 ± 0.004 | 0.190 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x4x4` | 0.005 ± 0.000 | - | 0.007 ± 0.004 | 0.010 ± 0.003 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x64x24` | 0.089 ± 0.010 | - | 0.092 ± 0.008 | 0.099 ± 0.006 | 0.142 ± 0.035 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x8x8` | 0.011 ± 0.000 | - | 0.014 ± 0.000 | 0.080 ± 0.005 | 0.030 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x2x2` | 0.004 ± 0.001 | - | 0.006 ± 0.000 | 0.010 ± 0.002 | 0.018 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x4x4` | 0.011 ± 0.004 | - | 0.016 ± 0.006 | 0.017 ± 0.003 | 0.025 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x8x8` | 0.028 ± 0.000 | - | 0.031 ± 0.000 | 0.195 ± 0.196 | 0.048 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x2x2` | 0.004 ± 0.001 | - | 0.007 ± 0.000 | 0.011 ± 0.002 | 0.020 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x4x4` | 0.012 ± 0.003 | - | 0.016 ± 0.003 | 0.017 ± 0.001 | 0.027 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x8x8` | 0.038 ± 0.000 | - | 0.041 ± 0.001 | 0.086 ± 0.078 | 0.086 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x128x128` | 11.407 ± 0.486 | - | 11.364 ± 0.240 | 11.846 ± 1.006 | 13.882 ± 1.026 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x2x2` | 0.007 ± 0.001 | - | 0.009 ± 0.000 | 0.011 ± 0.003 | 0.028 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x32x32` | 0.889 ± 0.006 | - | 0.911 ± 0.109 | 0.984 ± 0.147 | 1.188 ± 0.201 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x4x4` | 0.024 ± 0.001 | - | 0.026 ± 0.005 | 0.032 ± 0.004 | 0.049 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x8x8` | 0.075 ± 0.001 | - | 0.078 ± 0.001 | 0.174 ± 0.058 | 0.159 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x2x2` | 0.566 ± 0.028 | - | 0.583 ± 0.044 | 0.582 ± 0.023 | 1.995 ± 0.111 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x4x4` | 2.850 ± 0.126 | - | 2.927 ± 0.092 | 2.607 ± 0.093 | 1.700 ± 0.656 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x8x8` | 11.198 ± 0.627 | - | 11.138 ± 0.445 | 10.087 ± 0.442 | 6.218 ± 0.455 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x128x128` | 1.526 ± 0.115 | - | 1.557 ± 0.042 | 1.392 ± 0.025 | 1.445 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.006 ± 0.001 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x32x32` | 0.126 ± 0.001 | - | 0.124 ± 0.003 | 0.118 ± 0.001 | 0.164 ± 0.038 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x4x4` | 0.005 ± 0.000 | - | 0.007 ± 0.000 | 0.011 ± 0.002 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x64x24` | 0.096 ± 0.009 | - | 0.095 ± 0.009 | 0.094 ± 0.001 | 0.147 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x8x8` | 0.023 ± 0.002 | - | 0.029 ± 0.004 | 0.015 ± 0.000 | 0.024 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x2x2` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.007 ± 0.000 | 0.017 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x4x4` | 0.010 ± 0.000 | - | 0.012 ± 0.000 | 0.013 ± 0.003 | 0.023 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x8x8` | 0.033 ± 0.006 | - | 0.036 ± 0.002 | 0.032 ± 0.001 | 0.042 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x2x2` | 0.005 ± 0.001 | - | 0.007 ± 0.000 | 0.008 ± 0.000 | 0.019 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x4x4` | 0.015 ± 0.003 | - | 0.015 ± 0.000 | 0.016 ± 0.003 | 0.028 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x8x8` | 0.047 ± 0.008 | - | 0.047 ± 0.008 | 0.042 ± 0.000 | 0.055 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x128x128` | 12.077 ± 0.065 | - | 12.155 ± 0.263 | 11.372 ± 0.188 | 6.025 ± 0.271 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x2x2` | 0.006 ± 0.000 | - | 0.009 ± 0.000 | 0.010 ± 0.000 | 0.026 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x32x32` | 0.998 ± 0.048 | - | 1.010 ± 0.099 | 0.901 ± 0.170 | 0.578 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x4x4` | 0.024 ± 0.003 | - | 0.026 ± 0.000 | 0.026 ± 0.000 | 0.043 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x8x8` | 0.092 ± 0.012 | - | 0.093 ± 0.004 | 0.080 ± 0.011 | 0.153 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x2x2` | 0.419 ± 0.078 | - | unsupported | 0.217 ± 0.006 | 1.266 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x4x4` | 1.545 ± 0.324 | - | unsupported | 1.356 ± 0.033 | 2.771 ± 0.139 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x8x8` | 4.358 ± 0.253 | - | unsupported | 3.842 ± 0.068 | 4.688 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x128x128` | 1.054 ± 0.359 | - | unsupported | 0.772 ± 0.019 | 0.773 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x2x2` | 0.004 ± 0.003 | - | unsupported | 0.005 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x32x32` | 0.048 ± 0.002 | - | unsupported | 0.042 ± 0.001 | 0.043 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x4x4` | 0.004 ± 0.002 | - | unsupported | 0.006 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x64x24` | 0.034 ± 0.006 | - | unsupported | 0.036 ± 0.002 | 0.045 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x8x8` | 0.005 ± 0.000 | - | unsupported | 0.009 ± 0.000 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x2x2` | 0.005 ± 0.001 | - | unsupported | 0.006 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x4x4` | 0.007 ± 0.002 | - | unsupported | 0.009 ± 0.000 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x8x8` | 0.012 ± 0.000 | - | unsupported | 0.016 ± 0.000 | 0.023 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x2x2` | 0.005 ± 0.002 | - | unsupported | 0.006 ± 0.000 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x4x4` | 0.008 ± 0.003 | - | unsupported | 0.010 ± 0.000 | 0.018 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x8x8` | 0.016 ± 0.000 | - | unsupported | 0.020 ± 0.003 | 0.034 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x128x128` | 6.441 ± 0.290 | - | unsupported | 6.719 ± 0.417 | 5.980 ± 0.290 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x2x2` | 0.005 ± 0.002 | - | unsupported | 0.007 ± 0.000 | 0.021 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x32x32` | 0.407 ± 0.054 | - | unsupported | 0.292 ± 0.001 | 0.309 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x4x4` | 0.013 ± 0.003 | - | unsupported | 0.015 ± 0.000 | 0.030 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x8x8` | 0.030 ± 0.000 | - | unsupported | 0.034 ± 0.000 | 0.045 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x2x2` | 0.200 ± 0.019 | - | unsupported | 0.209 ± 0.001 | 1.542 ± 0.217 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x4x4` | 1.297 ± 0.020 | - | unsupported | 1.331 ± 0.193 | 1.625 ± 0.329 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x8x8` | 3.701 ± 0.103 | - | unsupported | 3.710 ± 0.493 | 2.986 ± 0.027 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x128x128` | 0.731 ± 0.038 | - | unsupported | 0.741 ± 0.048 | 0.715 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x2x2` | 0.001 ± 0.001 | - | unsupported | 0.005 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x32x32` | 0.035 ± 0.000 | - | unsupported | 0.042 ± 0.004 | 0.052 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x4x4` | 0.002 ± 0.000 | - | unsupported | 0.006 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x64x24` | 0.029 ± 0.000 | - | unsupported | 0.035 ± 0.001 | 0.038 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x8x8` | 0.004 ± 0.000 | - | unsupported | 0.008 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x2x2` | 0.002 ± 0.000 | - | unsupported | 0.005 ± 0.000 | 0.014 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x4x4` | 0.005 ± 0.000 | - | unsupported | 0.008 ± 0.000 | 0.017 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x8x8` | 0.011 ± 0.000 | - | unsupported | 0.015 ± 0.000 | 0.025 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x2x2` | 0.002 ± 0.000 | - | unsupported | 0.005 ± 0.000 | 0.018 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x4x4` | 0.006 ± 0.000 | - | unsupported | 0.009 ± 0.000 | 0.020 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x8x8` | 0.015 ± 0.000 | - | unsupported | 0.019 ± 0.000 | 0.033 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x128x128` | 5.899 ± 0.151 | - | unsupported | 6.187 ± 0.701 | 3.408 ± 0.334 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x2x2` | 0.003 ± 0.000 | - | unsupported | 0.006 ± 0.000 | 0.020 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x32x32` | 0.277 ± 0.008 | - | unsupported | 0.293 ± 0.027 | 0.218 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x4x4` | 0.011 ± 0.000 | - | unsupported | 0.015 ± 0.000 | 0.052 ± 0.033 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x8x8` | 0.029 ± 0.006 | - | unsupported | 0.033 ± 0.000 | 0.055 ± 0.004 | - | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x16x16,rhs=1` | 1.502 ± 0.345 | - | unsupported | 0.938 ± 0.114 | 1.025 ± 0.060 | 3.156 ± 0.891 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x2x2,rhs=1` | 0.069 ± 0.120 | - | unsupported | 0.187 ± 0.130 | 0.137 ± 0.008 | 0.076 ± 0.012 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x4x4,rhs=1` | 0.158 ± 0.094 | - | unsupported | 0.365 ± 0.259 | 0.196 ± 0.050 | 0.240 ± 0.048 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x8x8,rhs=1` | 0.326 ± 0.057 | - | unsupported | 0.633 ± 0.514 | 0.379 ± 0.037 | 0.797 ± 0.097 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x16x16,rhs=1` | 0.869 ± 0.244 | - | unsupported | 0.458 ± 0.072 | 0.961 ± 0.377 | 4.348 ± 1.468 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x2x2,rhs=1` | 0.057 ± 0.006 | - | unsupported | 0.073 ± 0.024 | 0.172 ± 0.009 | 0.090 ± 0.007 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x4x4,rhs=1` | 0.126 ± 0.016 | - | unsupported | 0.107 ± 0.006 | 0.219 ± 0.025 | 0.226 ± 0.042 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x8x8,rhs=1` | 0.265 ± 0.022 | - | unsupported | 0.207 ± 0.022 | 0.278 ± 0.055 | 0.770 ± 0.043 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.215 ± 0.010 | - | unsupported | 0.260 ± 0.144 | 0.394 ± 0.004 | 0.593 ± 0.368 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.060 ± 0.001 | - | unsupported | 0.049 ± 0.004 | 0.078 ± 0.004 | 0.063 ± 0.003 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.063 ± 0.001 | - | unsupported | 0.057 ± 0.008 | 0.099 ± 0.006 | 0.115 ± 0.107 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.085 ± 0.001 | - | unsupported | 0.088 ± 0.006 | 0.151 ± 0.004 | 0.424 ± 0.212 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.210 ± 0.008 | - | unsupported | 0.179 ± 0.000 | 0.533 ± 0.051 | 0.480 ± 0.005 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.064 ± 0.011 | - | unsupported | 0.045 ± 0.001 | 0.108 ± 0.017 | 0.072 ± 0.009 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.064 ± 0.006 | - | unsupported | 0.053 ± 0.011 | 0.134 ± 0.132 | 0.099 ± 0.006 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.088 ± 0.017 | - | unsupported | 0.078 ± 0.000 | 0.213 ± 0.020 | 0.227 ± 0.002 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.111 ± 0.016 | - | 0.177 ± 0.018 | 0.082 ± 0.063 | 0.162 ± 0.002 | 0.242 ± 0.008 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.039 ± 0.000 | - | 0.047 ± 0.011 | 0.026 ± 0.019 | 0.023 ± 0.001 | 0.035 ± 0.001 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.044 ± 0.001 | - | 0.052 ± 0.000 | 0.035 ± 0.019 | 0.029 ± 0.001 | 0.049 ± 0.001 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.068 ± 0.007 | - | 0.080 ± 0.009 | 0.045 ± 0.017 | 0.053 ± 0.001 | 0.099 ± 0.001 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.122 ± 0.095 | - | 0.215 ± 0.020 | 0.080 ± 0.003 | 0.337 ± 0.096 | 0.251 ± 0.012 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.046 ± 0.004 | - | 0.052 ± 0.007 | 0.037 ± 0.016 | 0.031 ± 0.002 | 0.042 ± 0.002 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.045 ± 0.001 | - | 0.059 ± 0.012 | 0.040 ± 0.017 | 0.038 ± 0.002 | 0.052 ± 0.001 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.065 ± 0.004 | - | 0.105 ± 0.094 | 0.037 ± 0.006 | 0.085 ± 0.028 | 0.105 ± 0.001 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 1 | `1536x1536` | 7.967 ± 0.468 | - | 9.571 ± 0.672 | 25.672 ± 8.171 | 11.471 ± 0.285 | 40.696 ± 5.572 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 4 | `1536x1536` | 11.010 ± 1.549 | - | 10.783 ± 0.550 | 14.326 ± 2.805 | 11.299 ± 1.207 | 381.168 ± 286.869 | - |
| cpu/linalg_uncovered | `det` | f64 | 1 | `1024x1024` | 7.103 ± 0.346 | - | 6.739 ± 0.239 | 8.076 ± 1.125 | 7.778 ± 0.978 | 17.507 ± 2.116 | - |
| cpu/linalg_uncovered | `det` | f64 | 4 | `1024x1024` | 6.313 ± 0.035 | - | 6.238 ± 0.138 | 6.325 ± 0.577 | 6.619 ± 1.564 | 239.892 ± 34.895 | - |
| cpu/linalg_uncovered | `eig` | f64 | 1 | `160x160` | 6.601 ± 0.565 | - | 6.602 ± 0.222 | 7.268 ± 1.110 | 8.422 ± 0.929 | 7.015 ± 1.132 | - |
| cpu/linalg_uncovered | `eig` | f64 | 4 | `160x160` | 7.083 ± 0.342 | - | 6.837 ± 0.667 | 6.791 ± 0.156 | 7.580 ± 0.245 | 306.333 ± 12.644 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 1 | `192x192` | 7.125 ± 0.155 | - | 7.102 ± 0.202 | 7.868 ± 0.325 | 6.373 ± 0.113 | 6.992 ± 0.604 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 4 | `192x192` | 7.409 ± 0.099 | - | 7.478 ± 0.087 | 7.604 ± 0.427 | 7.573 ± 0.463 | 257.818 ± 20.384 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 1 | `512x512` | 18.925 ± 0.217 | - | 19.297 ± 0.508 | 21.009 ± 3.039 | 22.955 ± 0.655 | 12.237 ± 1.068 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 4 | `512x512` | 22.607 ± 0.214 | - | 19.751 ± 1.054 | 19.796 ± 2.003 | 26.571 ± 1.333 | 617.701 ± 270.046 | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `256x128` | 0.284 ± 0.038 | 0.278 ± 0.012 | unsupported | 0.324 ± 0.030 | 0.366 ± 0.032 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `768x384` | 3.398 ± 1.322 | 3.754 ± 0.223 | unsupported | 3.691 ± 0.383 | 4.664 ± 0.729 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `256x128` | 0.311 ± 0.046 | 0.317 ± 0.077 | unsupported | 0.338 ± 0.028 | 0.578 ± 0.278 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `768x384` | 3.570 ± 0.697 | 3.467 ± 0.246 | unsupported | 4.013 ± 0.174 | 4.452 ± 0.186 | - | - |
| cpu/linalg_uncovered | `inv` | f64 | 1 | `768x768` | 7.956 ± 0.430 | - | 8.454 ± 0.643 | 9.570 ± 0.508 | 8.381 ± 0.094 | 25.415 ± 1.283 | - |
| cpu/linalg_uncovered | `inv` | f64 | 4 | `768x768` | 8.782 ± 0.719 | - | 10.231 ± 1.078 | 8.361 ± 0.648 | 7.379 ± 1.086 | 365.621 ± 422.581 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 1 | `768x384,rhs=16` | - | 8.768 ± 1.365 | 8.029 ± 0.318 | 5.616 ± 0.402 | 21.712 ± 0.109 | 7.647 ± 0.593 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 4 | `768x384,rhs=16` | - | 9.403 ± 0.503 | 9.745 ± 1.304 | 5.070 ± 4.665 | 22.241 ± 0.203 | 145.119 ± 37.552 | - |
| cpu/linalg_uncovered | `lu` | f64 | 1 | `1024x1024` | 7.153 ± 0.367 | - | 8.247 ± 0.720 | 13.344 ± 3.286 | 9.443 ± 1.736 | 29.300 ± 6.000 | - |
| cpu/linalg_uncovered | `lu` | f64 | 4 | `1024x1024` | 7.847 ± 0.586 | - | 8.236 ± 0.309 | 11.374 ± 0.469 | 7.394 ± 0.183 | 248.949 ± 19.221 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 1 | `2048x2048` | 0.582 ± 0.117 | - | 0.742 ± 0.107 | 2.295 ± 0.063 | 5.570 ± 0.118 | 6.090 ± 0.157 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 4 | `2048x2048` | 1.366 ± 1.056 | - | 0.907 ± 0.257 | 2.385 ± 0.076 | 3.743 ± 0.049 | 7.749 ± 4.152 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 1 | `512x256` | 10.162 ± 0.245 | - | 9.618 ± 0.273 | 11.254 ± 1.335 | 11.566 ± 0.161 | 20.858 ± 1.900 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 4 | `512x256` | 12.273 ± 2.248 | - | 11.014 ± 1.295 | 12.436 ± 2.835 | 10.198 ± 0.141 | 990.843 ± 62.773 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 1 | `512x256` | 9.589 ± 0.093 | - | 9.623 ± 0.488 | 11.515 ± 0.996 | 11.030 ± 0.292 | 18.819 ± 0.269 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 4 | `512x256` | 10.454 ± 0.337 | - | 10.745 ± 0.455 | 11.140 ± 0.835 | 10.244 ± 0.718 | 980.383 ± 75.721 | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `256x128` | 1.355 ± 0.017 | 1.372 ± 0.043 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `768x384` | 12.527 ± 0.083 | 12.667 ± 0.128 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `256x128` | 1.537 ± 0.239 | 1.586 ± 0.249 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `768x384` | 13.160 ± 0.829 | 16.233 ± 2.418 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `256x128` | 1.369 ± 0.013 | 1.388 ± 0.006 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `768x384` | 11.253 ± 0.362 | 11.455 ± 0.330 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `256x128` | 1.437 ± 0.115 | 1.463 ± 0.035 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `768x384` | 20.291 ± 8.140 | 15.522 ± 7.014 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 1 | `1024x1024` | 7.277 ± 0.627 | - | 7.369 ± 0.702 | 8.509 ± 3.617 | 7.467 ± 2.478 | 17.636 ± 1.371 | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 4 | `1024x1024` | 7.874 ± 1.896 | - | 8.049 ± 0.690 | 6.194 ± 0.662 | 9.395 ± 0.655 | 200.472 ± 11.605 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 1 | `768x384` | - | 32.472 ± 1.930 | 32.083 ± 0.260 | 37.795 ± 7.940 | 29.928 ± 0.879 | 66.050 ± 18.840 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 4 | `768x384` | - | 35.565 ± 1.971 | 35.772 ± 6.747 | 35.399 ± 3.383 | 30.537 ± 1.570 | 1947.178 ± 173.731 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 1 | `4096x4096,rhs=64` | 14.249 ± 1.466 | - | 21.143 ± 2.140 | 12.993 ± 0.820 | 88.045 ± 9.079 | 28.009 ± 2.987 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 4 | `4096x4096,rhs=64` | 14.592 ± 1.672 | - | 17.311 ± 0.169 | 11.725 ± 0.451 | 47.842 ± 7.220 | 26.273 ± 5.944 | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `1024x64` | 0.187 ± 0.005 | 0.192 ± 0.009 | unsupported | 0.062 ± 0.011 | 0.033 ± 0.000 | - | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `4096x256` | 2.713 ± 0.065 | 2.684 ± 0.031 | unsupported | 0.922 ± 0.062 | 0.425 ± 0.033 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `1024x64` | 0.120 ± 0.006 | 0.120 ± 0.016 | unsupported | 0.099 ± 0.023 | 0.042 ± 0.022 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `4096x256` | 1.435 ± 0.081 | 1.427 ± 0.141 | unsupported | 0.700 ± 0.070 | 0.345 ± 0.036 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `1024x64` | 0.072 ± 0.001 | 0.072 ± 0.001 | unsupported | 0.079 ± 0.000 | 0.176 ± 0.002 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `4096x256` | 1.267 ± 0.047 | 1.278 ± 0.056 | unsupported | 1.318 ± 0.038 | 2.395 ± 0.150 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `1024x64` | 0.056 ± 0.001 | 0.055 ± 0.002 | unsupported | 0.144 ± 0.004 | 0.130 ± 0.002 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `4096x256` | 0.632 ± 0.010 | 0.636 ± 0.041 | unsupported | 1.086 ± 0.033 | 1.223 ± 0.074 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `1024x64` | 0.259 ± 0.006 | 0.256 ± 0.007 | unsupported | 0.297 ± 0.029 | 0.080 ± 0.003 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `4096x256` | 4.158 ± 0.296 | 4.116 ± 0.029 | unsupported | 6.037 ± 1.843 | 0.899 ± 0.016 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `1024x64` | 0.154 ± 0.033 | 0.160 ± 0.006 | unsupported | 0.289 ± 0.010 | 0.066 ± 0.019 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `4096x256` | 2.152 ± 0.040 | 2.172 ± 0.061 | unsupported | 3.218 ± 0.324 | 0.534 ± 0.054 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `1024x64` | 0.044 ± 0.004 | 0.066 ± 0.004 | unsupported | 0.047 ± 0.001 | 0.067 ± 0.002 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `4096x256` | 0.529 ± 0.006 | 0.556 ± 0.030 | unsupported | 0.680 ± 0.118 | 0.668 ± 0.006 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `1024x64` | 0.119 ± 0.058 | 0.117 ± 0.120 | unsupported | 0.100 ± 0.012 | 0.102 ± 0.029 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `4096x256` | 0.905 ± 0.123 | 0.908 ± 0.096 | unsupported | 0.423 ± 0.011 | 0.529 ± 0.027 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `1024x64` | 0.086 ± 0.001 | 0.105 ± 0.003 | unsupported | 0.096 ± 0.007 | 0.102 ± 0.001 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `4096x256` | 1.166 ± 0.170 | 1.168 ± 0.031 | unsupported | 1.408 ± 0.196 | 1.268 ± 0.087 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `1024x64` | 0.122 ± 0.005 | 0.148 ± 0.104 | unsupported | 0.136 ± 0.016 | 0.120 ± 0.006 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `4096x256` | 1.030 ± 0.271 | 1.286 ± 0.091 | unsupported | 1.110 ± 0.073 | 0.874 ± 0.104 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `1024x64` | 0.118 ± 0.002 | 0.147 ± 0.016 | unsupported | unsupported | 0.124 ± 0.011 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `4096x256` | 1.733 ± 0.057 | 1.791 ± 0.048 | unsupported | unsupported | 1.600 ± 0.078 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `1024x64` | 0.158 ± 0.011 | 0.197 ± 0.021 | unsupported | unsupported | 0.210 ± 0.048 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `4096x256` | 1.278 ± 0.030 | 1.339 ± 0.026 | unsupported | unsupported | 1.151 ± 0.101 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `1024x64` | 0.011 ± 0.001 | 0.016 ± 0.000 | unsupported | 0.008 ± 0.001 | 0.042 ± 0.001 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `4096x256` | 0.103 ± 0.006 | 0.104 ± 0.004 | unsupported | 0.123 ± 0.041 | 0.378 ± 0.192 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `1024x64` | 0.019 ± 0.002 | 0.027 ± 0.001 | unsupported | 0.025 ± 0.056 | 0.058 ± 0.000 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `4096x256` | 0.147 ± 0.009 | 0.155 ± 0.018 | unsupported | 0.121 ± 0.015 | 0.211 ± 0.113 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `1024x64` | 0.023 ± 0.001 | 0.036 ± 0.003 | unsupported | 0.027 ± 0.001 | 0.067 ± 0.002 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `4096x256` | 0.295 ± 0.017 | 0.336 ± 0.017 | unsupported | 0.515 ± 0.265 | 0.623 ± 0.018 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `1024x64` | 0.058 ± 0.032 | 0.134 ± 0.075 | unsupported | 0.166 ± 0.021 | 0.075 ± 0.010 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `4096x256` | 0.419 ± 0.005 | 0.471 ± 0.039 | unsupported | 0.491 ± 0.026 | 0.413 ± 0.026 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `1024x64` | 0.052 ± 0.002 | 0.053 ± 0.008 | unsupported | 0.072 ± 0.016 | 0.079 ± 0.002 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `4096x256` | 0.736 ± 0.012 | 0.741 ± 0.013 | unsupported | 1.422 ± 0.016 | 0.726 ± 0.007 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `1024x64` | 0.036 ± 0.007 | 0.037 ± 0.003 | unsupported | 0.093 ± 0.040 | 0.083 ± 0.008 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `4096x256` | 0.410 ± 0.009 | 0.402 ± 0.009 | unsupported | 0.976 ± 0.128 | 0.442 ± 0.039 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `1024x64` | 0.057 ± 0.004 | 0.052 ± 0.007 | unsupported | 0.073 ± 0.002 | 0.080 ± 0.001 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `4096x256` | 0.796 ± 0.032 | 0.788 ± 0.034 | unsupported | 1.397 ± 0.157 | 0.794 ± 0.014 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `1024x64` | 0.036 ± 0.001 | 0.037 ± 0.001 | unsupported | 0.078 ± 0.035 | 0.082 ± 0.014 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `4096x256` | 0.421 ± 0.013 | 0.424 ± 0.010 | unsupported | 0.973 ± 0.073 | 0.803 ± 0.399 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `1024x64` | 0.085 ± 0.002 | 0.099 ± 0.001 | unsupported | 0.106 ± 0.005 | 0.102 ± 0.001 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `4096x256` | 1.152 ± 0.047 | 1.195 ± 0.075 | unsupported | 1.615 ± 0.181 | 1.218 ± 0.019 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `1024x64` | 0.128 ± 0.032 | 0.150 ± 0.019 | unsupported | 0.159 ± 0.024 | 0.130 ± 0.000 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `4096x256` | 1.211 ± 0.056 | 1.278 ± 0.092 | unsupported | 1.201 ± 0.027 | 0.846 ± 0.295 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `1024x64` | 0.214 ± 0.009 | 0.213 ± 0.023 | unsupported | 0.261 ± 0.015 | 0.288 ± 0.004 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `4096x256` | 3.726 ± 0.259 | 3.742 ± 0.166 | unsupported | 4.167 ± 0.349 | 4.272 ± 0.598 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `1024x64` | 0.145 ± 0.045 | 0.144 ± 0.038 | unsupported | 0.313 ± 0.005 | 0.186 ± 0.007 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `4096x256` | 1.803 ± 0.061 | 1.773 ± 0.059 | unsupported | 2.915 ± 0.139 | 2.685 ± 1.665 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `1024x64` | 0.516 ± 0.009 | 0.638 ± 0.014 | unsupported | 0.119 ± 0.007 | 0.094 ± 0.004 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `4096x256` | 9.097 ± 0.078 | 10.069 ± 0.213 | unsupported | 3.003 ± 0.614 | 1.132 ± 0.090 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `1024x64` | 0.694 ± 0.176 | 0.936 ± 0.203 | unsupported | 0.233 ± 0.016 | 0.190 ± 0.106 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `4096x256` | 6.777 ± 0.172 | 8.507 ± 1.274 | unsupported | 1.650 ± 0.041 | 0.533 ± 0.023 | - | - |
| cpu/output_reuse | `add_into` | f64 | 1 | `33554432` | 24.430 ± 5.736 | - | unsupported | 22.158 ± 3.727 | - | 20.644 ± 3.951 | 18.308 ± 2.362 |
| cpu/output_reuse | `add_into` | f64 | 4 | `33554432` | 20.997 ± 4.266 | - | unsupported | 22.169 ± 3.164 | - | 19.525 ± 2.038 | 21.199 ± 2.281 |
| cpu/output_reuse | `conj_into` | c64 | 1 | `16777216` | 10.491 ± 0.457 | - | unsupported | 12.431 ± 0.960 | - | 10.378 ± 0.766 | 12.467 ± 0.386 |
| cpu/output_reuse | `conj_into` | c64 | 4 | `16777216` | 13.903 ± 0.962 | - | unsupported | 10.706 ± 1.318 | - | 15.759 ± 1.707 | 13.777 ± 2.129 |
| cpu/output_reuse | `copy_read_into` | f64 | 1 | `33554432` | 9.154 ± 0.401 | - | unsupported | 16.120 ± 4.715 | - | 10.946 ± 0.718 | - |
| cpu/output_reuse | `copy_read_into` | f64 | 4 | `33554432` | 13.104 ± 0.334 | - | unsupported | 10.474 ± 0.993 | - | 12.949 ± 1.137 | - |
| cpu/output_reuse | `div_into` | f64 | 1 | `33554432` | 17.320 ± 0.670 | - | unsupported | 19.116 ± 1.488 | - | 19.764 ± 1.979 | 17.992 ± 1.617 |
| cpu/output_reuse | `div_into` | f64 | 4 | `33554432` | 25.437 ± 7.591 | - | unsupported | 15.194 ± 1.310 | - | 22.241 ± 4.615 | 30.365 ± 5.662 |
| cpu/output_reuse | `dot_general_read_into` | f64 | 1 | `1024x1024` | 8.713 ± 0.418 | - | unsupported | 12.884 ± 7.100 | - | 53.816 ± 5.817 | - |
| cpu/output_reuse | `dot_general_read_into` | f64 | 4 | `1024x1024` | 11.762 ± 1.496 | - | unsupported | 8.845 ± 0.249 | - | 32.204 ± 3.941 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 1 | `1024x1024` | 9.650 ± 0.321 | - | unsupported | 14.533 ± 4.607 | - | 50.294 ± 1.475 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 4 | `1024x1024` | 12.150 ± 0.949 | - | unsupported | 10.127 ± 1.078 | - | 32.038 ± 14.513 | - |
| cpu/output_reuse | `mul_into` | f64 | 1 | `33554432` | 18.069 ± 1.153 | - | unsupported | 20.521 ± 1.493 | - | 19.773 ± 6.163 | 22.835 ± 2.618 |
| cpu/output_reuse | `mul_into` | f64 | 4 | `33554432` | 18.142 ± 1.664 | - | unsupported | 15.332 ± 2.101 | - | 22.169 ± 2.652 | 21.604 ± 3.631 |
| cpu/output_reuse | `neg_into` | f64 | 1 | `33554432` | 10.606 ± 0.286 | - | unsupported | 12.706 ± 1.320 | - | 10.774 ± 1.568 | 14.035 ± 0.967 |
| cpu/output_reuse | `neg_into` | f64 | 4 | `33554432` | 14.553 ± 3.657 | - | unsupported | 11.438 ± 2.394 | - | 19.592 ± 10.057 | 13.649 ± 1.103 |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `256x256,rhs=16` | 0.197 ± 0.030 | unsupported | unsupported | 0.281 ± 0.010 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `768x768,rhs=16` | 2.405 ± 0.097 | unsupported | unsupported | 3.217 ± 0.211 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `256x256,rhs=16` | 0.216 ± 0.011 | unsupported | unsupported | 0.341 ± 0.014 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `768x768,rhs=16` | 3.030 ± 0.255 | unsupported | unsupported | 3.419 ± 0.250 | unsupported | - | - |
| cpu/output_reuse | `sub_into` | f64 | 1 | `33554432` | 18.408 ± 1.224 | - | unsupported | 25.981 ± 2.402 | - | 18.168 ± 0.435 | 17.192 ± 2.834 |
| cpu/output_reuse | `sub_into` | f64 | 4 | `33554432` | 18.749 ± 0.427 | - | unsupported | 19.720 ± 2.700 | - | 22.826 ± 2.229 | 19.554 ± 3.107 |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 1 | `8192x1 -> 8192x4096` | 4.157 ± 0.075 | - | 0.002 ± 0.000 | 5.486 ± 0.739 | 4.689 ± 0.842 | 15.154 ± 2.516 | - |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 4 | `8192x1 -> 8192x4096` | 10.670 ± 0.828 | - | 0.003 ± 0.001 | 5.472 ± 0.471 | 4.661 ± 0.372 | 14.878 ± 4.167 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 1 | `33554432` | 7.719 ± 0.470 | - | 19.301 ± 0.577 | 8.937 ± 0.255 | 8.099 ± 0.237 | 8.878 ± 0.142 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 4 | `33554432` | 12.291 ± 1.804 | - | 26.465 ± 5.138 | 8.341 ± 1.171 | 7.686 ± 0.830 | 9.267 ± 0.483 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 1 | `8192 -> 8192x8192` | 8.427 ± 0.070 | - | 12.971 ± 3.703 | 17.798 ± 5.705 | 17.432 ± 1.989 | 13.476 ± 1.521 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 4 | `8192 -> 8192x8192` | 12.460 ± 0.705 | - | 12.072 ± 1.545 | 10.396 ± 1.903 | 8.905 ± 0.903 | 13.608 ± 2.877 | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 1 | `8388608x2x2 -> 8388608x2` | 5.208 ± 0.603 | - | 4.756 ± 0.305 | 14.301 ± 6.276 | 7.670 ± 0.536 | - | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 4 | `8388608x2x2 -> 8388608x2` | 5.168 ± 0.726 | - | 5.939 ± 0.701 | 9.978 ± 1.144 | 7.300 ± 0.464 | - | - |
| cpu/structural_shape | `reshape` | f64 | 1 | `33554432 -> 8192x4096` | 9.171 ± 0.656 | - | 0.002 ± 0.000 | 11.733 ± 0.582 | 10.538 ± 0.859 | 11.723 ± 0.367 | - |
| cpu/structural_shape | `reshape` | f64 | 4 | `33554432 -> 8192x4096` | 14.117 ± 2.072 | - | 0.002 ± 0.000 | 12.123 ± 0.815 | 9.701 ± 1.147 | 15.657 ± 3.358 | - |
| cpu/structural_shape | `transpose` | f64 | 1 | `4096x4096` | 37.179 ± 2.706 | - | 0.003 ± 0.001 | 43.393 ± 3.329 | 49.487 ± 1.631 | 199.243 ± 12.570 | 33.962 ± 2.892 |
| cpu/structural_shape | `transpose` | f64 | 4 | `4096x4096` | 22.075 ± 6.069 | - | 0.003 ± 0.004 | 47.449 ± 5.781 | 26.786 ± 0.647 | 210.946 ± 35.619 | 20.126 ± 6.856 |
| cpu/structural_shape | `tril` | f64 | 1 | `4096x4096` | 8.544 ± 0.593 | - | 8.468 ± 0.494 | 8.433 ± 2.560 | 5.897 ± 0.381 | 10.119 ± 1.384 | - |
| cpu/structural_shape | `tril` | f64 | 4 | `4096x4096` | 5.958 ± 0.680 | - | 6.043 ± 0.286 | 6.328 ± 0.985 | 4.981 ± 0.619 | 10.196 ± 1.158 | - |
| cpu/structural_shape | `triu` | f64 | 1 | `4096x4096` | 8.283 ± 1.057 | - | 7.956 ± 0.318 | 8.890 ± 2.589 | 5.914 ± 0.399 | 9.880 ± 0.723 | - |
| cpu/structural_shape | `triu` | f64 | 4 | `4096x4096` | 7.267 ± 1.809 | - | 6.309 ± 0.600 | 6.511 ± 0.954 | 5.013 ± 0.477 | 9.607 ± 0.808 | - |
| cpu/view_metadata | `broadcast_in_dim_view` | f64 | 1 | `8192x1 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | - | - |
| cpu/view_metadata | `broadcast_in_dim_view` | f64 | 4 | `8192x1 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | - | - |
| cpu/view_metadata | `reshape_view` | f64 | 1 | `33554432 -> 8192x4096` | 0.001 ± 0.002 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `reshape_view` | f64 | 4 | `33554432 -> 8192x4096` | 0.002 ± 0.001 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `slice_view` | f64 | 1 | `4194304 -> 2096128` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.001 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `slice_view` | f64 | 4 | `4194304 -> 2096128` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `transpose_view` | f64 | 1 | `4096x4096` | 0.000 ± 0.000 | - | unsupported | 0.002 ± 0.003 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `transpose_view` | f64 | 4 | `4096x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |

## Short-operation batches

Total batch duration and normalized ns/op are shown without rounding sub-microsecond calls to zero. Metadata views require no execution session.

| Operation | Dtype | Shape | Threads | Backend | Operations/batch | Median batch ms | Median ns/op |
|---|---|---|---:|---|---:|---:|---:|
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 16 | 4.208250 | 263015.62 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs direct API (ms) | 16 | 4.061333 | 253833.31 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 16 | 4.374042 | 273377.62 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 16 | 3.240417 | 202526.06 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs direct API (ms) | 16 | 3.552375 | 222023.44 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 16 | 4.419625 | 276226.56 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs direct API (ms) | 16 | 15.033000 | 939562.50 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs eager mode (ms) | 16 | 13.981792 | 873862.00 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs direct API (ms) | 16 | 11.423875 | 713992.19 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs eager mode (ms) | 16 | 11.176416 | 698526.00 |
| householder_qr | f64 | `256x128` | 1 | JAX Python (XLA CPU) (ms) | 16 | 5.857334 | 366083.38 |
| householder_qr | f64 | `256x128` | 1 | PyTorch Python (ms) | 16 | 5.179000 | 323687.50 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 4.543959 | 283997.44 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 4.445583 | 277848.94 |
| householder_qr | f64 | `768x384` | 1 | JAX Python (XLA CPU) (ms) | 16 | 74.618084 | 4663630.25 |
| householder_qr | f64 | `768x384` | 1 | PyTorch Python (ms) | 16 | 59.056541 | 3691033.81 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 54.362458 | 3397653.62 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 60.067250 | 3754203.12 |
| householder_qr | f64 | `256x128` | 4 | JAX Python (XLA CPU) (ms) | 16 | 9.246042 | 577877.62 |
| householder_qr | f64 | `256x128` | 4 | PyTorch Python (ms) | 16 | 5.409208 | 338075.50 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 4.980458 | 311278.62 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 5.078166 | 317385.38 |
| householder_qr | f64 | `768x384` | 4 | JAX Python (XLA CPU) (ms) | 16 | 71.233334 | 4452083.38 |
| householder_qr | f64 | `768x384` | 4 | PyTorch Python (ms) | 16 | 64.212917 | 4013307.31 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 57.121542 | 3570096.38 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 55.464916 | 3466557.25 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 21.675250 | 1354703.12 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 21.951042 | 1371940.12 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 200.434458 | 12527153.62 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 202.667750 | 12666734.38 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 24.587125 | 1536695.31 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 25.376083 | 1586005.19 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 210.552625 | 13159539.06 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 259.732459 | 16233278.69 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 21.911084 | 1369442.75 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 22.204042 | 1387752.62 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 180.054334 | 11253395.88 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 183.281583 | 11455098.94 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 22.990875 | 1436929.69 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 23.410250 | 1463140.62 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 324.663000 | 20291437.50 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 248.353458 | 15522091.12 |
| erf | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 2.088250 | 32628.91 |
| erf | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 3.954625 | 61791.02 |
| erf | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 11.966000 | 186968.75 |
| erf | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 12.261625 | 191587.89 |
| erf | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 3.402833 | 425354.12 |
| erf | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 7.373166 | 921645.75 |
| erf | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 21.701583 | 2712697.88 |
| erf | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 21.475250 | 2684406.25 |
| erf | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 2.681250 | 41894.53 |
| erf | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 6.340500 | 99070.31 |
| erf | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 7.664041 | 119750.64 |
| erf | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 7.701000 | 120328.12 |
| erf | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 2.760708 | 345088.50 |
| erf | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 5.603209 | 700401.12 |
| erf | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 11.480041 | 1435005.12 |
| erf | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 11.415000 | 1426875.00 |
| gelu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 11.258459 | 175913.42 |
| gelu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 5.066375 | 79162.11 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 4.617875 | 72154.30 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 4.620459 | 72194.67 |
| gelu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 19.159708 | 2394963.50 |
| gelu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 10.543375 | 1317921.88 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 10.138625 | 1267328.12 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 10.222167 | 1277770.88 |
| gelu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 8.306875 | 129794.92 |
| gelu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 9.213166 | 143955.72 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 3.558041 | 55594.39 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 3.508625 | 54822.27 |
| gelu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 9.785625 | 1223203.12 |
| gelu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 8.687750 | 1085968.75 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 5.055500 | 631937.50 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 5.090916 | 636364.50 |
| gelu_tanh | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 5.122333 | 80036.45 |
| gelu_tanh | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 18.987959 | 296686.86 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 16.586250 | 259160.16 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 16.353875 | 255529.30 |
| gelu_tanh | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 7.189334 | 898666.75 |
| gelu_tanh | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 48.294167 | 6036770.88 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 33.266875 | 4158359.38 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 32.929000 | 4116125.00 |
| gelu_tanh | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 4.207625 | 65744.14 |
| gelu_tanh | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 18.508666 | 289197.91 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 9.876500 | 154320.31 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 10.211708 | 159557.94 |
| gelu_tanh | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 4.276000 | 534500.00 |
| gelu_tanh | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 25.747750 | 3218468.75 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 17.214167 | 2151770.88 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 17.379042 | 2172380.25 |
| layer_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 4.278750 | 66855.47 |
| layer_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 2.984875 | 46638.67 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 2.832416 | 44256.50 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 4.237208 | 66206.38 |
| layer_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 5.342500 | 667812.50 |
| layer_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 5.441625 | 680203.12 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 4.233917 | 529239.62 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 4.448875 | 556109.38 |
| layer_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 6.547375 | 102302.73 |
| layer_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 6.392750 | 99886.72 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 7.600625 | 118759.77 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 7.459083 | 116548.17 |
| layer_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 4.234291 | 529286.38 |
| layer_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 3.385125 | 423140.62 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 7.239833 | 904979.12 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 7.264625 | 908078.12 |
| log_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 6.526792 | 101981.12 |
| log_softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 6.147500 | 96054.69 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 5.504208 | 86003.25 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 6.692834 | 104575.53 |
| log_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 10.143750 | 1267968.75 |
| log_softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 11.267500 | 1408437.50 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 9.331750 | 1166468.75 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 9.347083 | 1168385.38 |
| log_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 7.679750 | 119996.09 |
| log_softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 8.685917 | 135717.45 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 7.827709 | 122307.95 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 9.500167 | 148440.11 |
| log_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 6.992333 | 874041.62 |
| log_softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 8.877625 | 1109703.12 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 8.242042 | 1030255.25 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 10.284083 | 1285510.38 |
| masked_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 7.955917 | 124311.20 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 7.535417 | 117740.89 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 9.428625 | 147322.27 |
| masked_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 12.796667 | 1599583.38 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 13.866292 | 1733286.50 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 14.326125 | 1790765.62 |
| masked_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 13.439750 | 209996.09 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 10.100917 | 157826.83 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 12.600709 | 196886.08 |
| masked_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 9.204667 | 1150583.38 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 10.227334 | 1278416.75 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 10.708458 | 1338557.25 |
| reduce_mean | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 2.715292 | 42426.44 |
| reduce_mean | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 0.505250 | 7894.53 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 0.710333 | 11098.95 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 1.033459 | 16147.80 |
| reduce_mean | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 3.020250 | 377531.25 |
| reduce_mean | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 0.982917 | 122864.62 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 0.827041 | 103380.12 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 0.834667 | 104333.38 |
| reduce_mean | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 3.730458 | 58288.41 |
| reduce_mean | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 1.597583 | 24962.23 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 1.217834 | 19028.66 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 1.710291 | 26723.30 |
| reduce_mean | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 1.687792 | 210974.00 |
| reduce_mean | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 0.971917 | 121489.62 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 1.174542 | 146817.75 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 1.242292 | 155286.50 |
| rms_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 4.268292 | 66692.06 |
| rms_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 1.756583 | 27446.61 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 1.499584 | 23431.00 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 2.319541 | 36242.83 |
| rms_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 4.987666 | 623458.25 |
| rms_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 4.119333 | 514916.62 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 2.356042 | 294505.25 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 2.691625 | 336453.12 |
| rms_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 4.785292 | 74770.19 |
| rms_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 10.610042 | 165781.91 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 3.739708 | 58432.94 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 8.572167 | 133940.11 |
| rms_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 3.306625 | 413328.12 |
| rms_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 3.929125 | 491140.62 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 3.351083 | 418885.38 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 3.771959 | 471494.88 |
| sigmoid | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 5.034792 | 78668.62 |
| sigmoid | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 4.632375 | 72380.86 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 3.318125 | 51845.70 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 3.366167 | 52596.36 |
| sigmoid | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 5.805375 | 725671.88 |
| sigmoid | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 11.372166 | 1421520.75 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 5.889042 | 736130.25 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 5.924250 | 740531.25 |
| sigmoid | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 5.330625 | 83291.02 |
| sigmoid | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 5.939875 | 92810.55 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 2.313542 | 36149.09 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 2.360750 | 36886.72 |
| sigmoid | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 3.532375 | 441546.88 |
| sigmoid | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 7.810792 | 976349.00 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 3.279666 | 409958.25 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 3.216291 | 402036.38 |
| silu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 5.128125 | 80126.95 |
| silu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 4.678334 | 73098.97 |
| silu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 3.648625 | 57009.77 |
| silu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 3.339625 | 52181.64 |
| silu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 6.352292 | 794036.50 |
| silu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 11.176167 | 1397020.88 |
| silu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 6.365208 | 795651.00 |
| silu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 6.302375 | 787796.88 |
| silu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 5.216458 | 81507.16 |
| silu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 5.010167 | 78283.86 |
| silu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 2.329833 | 36403.64 |
| silu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 2.375041 | 37110.02 |
| silu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 6.425375 | 803171.88 |
| silu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 7.786083 | 973260.38 |
| silu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 3.367542 | 420942.75 |
| silu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 3.395833 | 424479.12 |
| softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 6.519875 | 101873.05 |
| softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 6.775417 | 105865.89 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 5.461833 | 85341.14 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 6.364500 | 99445.31 |
| softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 9.743958 | 1217994.75 |
| softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 12.917750 | 1614718.75 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 9.214000 | 1151750.00 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 9.557292 | 1194661.50 |
| softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 8.328084 | 130126.31 |
| softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 10.180125 | 159064.45 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 8.181000 | 127828.12 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 9.631584 | 150493.50 |
| softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 6.770708 | 846338.50 |
| softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 9.609708 | 1201213.50 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 9.685834 | 1210729.25 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 10.226833 | 1278354.12 |
| softplus | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 18.413083 | 287704.42 |
| softplus | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 16.720000 | 261250.00 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 13.703042 | 214110.03 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 13.638792 | 213106.12 |
| softplus | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 34.176209 | 4272026.12 |
| softplus | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 33.337209 | 4167151.12 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 29.805042 | 3725630.25 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 29.936791 | 3742098.88 |
| softplus | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 11.927250 | 186363.28 |
| softplus | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 20.047917 | 313248.70 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 9.274125 | 144908.20 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 9.227375 | 144177.73 |
| softplus | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 21.479083 | 2684885.38 |
| softplus | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 23.323375 | 2915421.88 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 14.427458 | 1803432.25 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 14.183084 | 1772885.50 |
| take_along_axis | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 6.030667 | 94229.17 |
| take_along_axis | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 7.588042 | 118563.16 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 33.017458 | 515897.78 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 40.811042 | 637672.53 |
| take_along_axis | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 9.057333 | 1132166.62 |
| take_along_axis | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 24.026708 | 3003338.50 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 72.772333 | 9096541.62 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 80.554500 | 10069312.50 |
| take_along_axis | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 12.157250 | 189957.03 |
| take_along_axis | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 14.939000 | 233421.88 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 44.440458 | 694382.16 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 59.873042 | 935516.28 |
| take_along_axis | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 4.261541 | 532692.62 |
| take_along_axis | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 13.202917 | 1650364.62 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 54.217875 | 6777234.38 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 68.053666 | 8506708.25 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | PyTorch Python (ms) | 16 | 4.501916 | 281369.75 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 3.146833 | 196677.06 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | PyTorch Python (ms) | 16 | 51.465000 | 3216562.50 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 38.479125 | 2404945.31 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | PyTorch Python (ms) | 16 | 5.451792 | 340737.00 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 3.450500 | 215656.25 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | PyTorch Python (ms) | 16 | 54.711417 | 3419463.56 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 48.477375 | 3029835.94 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.012667 | 791.69 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.001459 | 91.19 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.012208 | 763.00 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.001792 | 112.00 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000083 | 5.19 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.012459 | 778.69 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.014333 | 895.81 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000125 | 7.81 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.011292 | 705.75 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.036791 | 2299.44 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000084 | 5.25 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | PyTorch Python (ms) | 16 | 0.021083 | 1317.69 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | tenferro-rs direct API (ms) | 16 | 0.002166 | 135.38 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000166 | 10.38 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | PyTorch Python (ms) | 16 | 0.009833 | 614.56 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | tenferro-rs direct API (ms) | 16 | 0.002667 | 166.69 |
| transpose_view | f64 | `4096x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000083 | 5.19 |
| transpose_view | f64 | `4096x4096` | 1 | PyTorch Python (ms) | 16 | 0.030083 | 1880.19 |
| transpose_view | f64 | `4096x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.002625 | 164.06 |
| transpose_view | f64 | `4096x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000083 | 5.19 |
| transpose_view | f64 | `4096x4096` | 4 | PyTorch Python (ms) | 16 | 0.010208 | 638.00 |
| transpose_view | f64 | `4096x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.002542 | 158.88 |

## Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `cpu/complex/cholesky` (c64, threads=4, shape=`448x448`): `jax-cpu` is 53.6x faster than the slowest successful cell (`julia-base`, 76.764 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/cholesky` (c64, threads=4, shape=`448x448`): `pytorch-cpu` is 20.4x faster than the slowest successful cell (`julia-base`, 76.764 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/cholesky` (c64, threads=4, shape=`448x448`): `tenferro-direct` is 22.7x faster than the slowest successful cell (`julia-base`, 76.764 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/cholesky` (c64, threads=4, shape=`448x448`): `tenferro-trace` is 21.9x faster than the slowest successful cell (`julia-base`, 76.764 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/eig` (c64, threads=4, shape=`112x112`): `jax-cpu` is 21.9x faster than the slowest successful cell (`julia-base`, 457.496 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/eig` (c64, threads=4, shape=`112x112`): `pytorch-cpu` is 43.6x faster than the slowest successful cell (`julia-base`, 457.496 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/eig` (c64, threads=4, shape=`112x112`): `tenferro-direct` is 43.0x faster than the slowest successful cell (`julia-base`, 457.496 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/eig` (c64, threads=4, shape=`112x112`): `tenferro-trace` is 41.3x faster than the slowest successful cell (`julia-base`, 457.496 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/norm_fro` (c64, threads=4, shape=`2048x1536`): `tenferro-direct` is 11.7x faster than the slowest successful cell (`julia-base`, 12.462 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `jax-cpu` is 25.4x faster than the slowest successful cell (`julia-base`, 174.942 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `pytorch-cpu` is 26.5x faster than the slowest successful cell (`julia-base`, 174.942 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `tenferro-direct` is 23.1x faster than the slowest successful cell (`julia-base`, 174.942 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `tenferro-trace` is 22.8x faster than the slowest successful cell (`julia-base`, 174.942 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `jax-cpu` is 46.6x faster than the slowest successful cell (`julia-base`, 132.463 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `pytorch-cpu` is 25.5x faster than the slowest successful cell (`julia-base`, 132.463 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `tenferro-direct` is 27.2x faster than the slowest successful cell (`julia-base`, 132.463 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `tenferro-trace` is 26.7x faster than the slowest successful cell (`julia-base`, 132.463 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/svd` (c64, threads=4, shape=`160x160`): `jax-cpu` is 112.1x faster than the slowest successful cell (`julia-base`, 648.650 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/svd` (c64, threads=4, shape=`160x160`): `pytorch-cpu` is 95.2x faster than the slowest successful cell (`julia-base`, 648.650 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/svd` (c64, threads=4, shape=`160x160`): `tenferro-direct` is 88.8x faster than the slowest successful cell (`julia-base`, 648.650 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/svd` (c64, threads=4, shape=`160x160`): `tenferro-trace` is 88.8x faster than the slowest successful cell (`julia-base`, 648.650 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 22.1x faster than the slowest successful cell (`pytorch-cpu`, 17.993 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 21.0x faster than the slowest successful cell (`pytorch-cpu`, 17.993 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 20.1x faster than the slowest successful cell (`pytorch-cpu`, 17.993 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 12.6x faster than the slowest successful cell (`pytorch-cpu`, 12.648 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 14.4x faster than the slowest successful cell (`pytorch-cpu`, 12.648 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-trace` is 15.8x faster than the slowest successful cell (`pytorch-cpu`, 12.648 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 18.3x faster than the slowest successful cell (`pytorch-cpu`, 15.641 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 14.9x faster than the slowest successful cell (`pytorch-cpu`, 15.641 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 14.6x faster than the slowest successful cell (`pytorch-cpu`, 15.641 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 14.6x faster than the slowest successful cell (`pytorch-cpu`, 12.441 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 15.9x faster than the slowest successful cell (`pytorch-cpu`, 12.441 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-trace` is 16.6x faster than the slowest successful cell (`pytorch-cpu`, 12.441 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=1, shape=`4194304 -> 2096128`): `tenferro-trace` is 801.4x faster than the slowest successful cell (`julia-base`, 1.769 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`): `tenferro-trace` is 903.5x faster than the slowest successful cell (`julia-base`, 2.560 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x2x2`): `tenferro-direct` is 16.4x faster than the slowest successful cell (`pytorch-cpu`, 0.013 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x32x32`): `tenferro-direct` is 23.5x faster than the slowest successful cell (`jax-cpu`, 0.045 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x4x4`): `tenferro-direct` is 12.7x faster than the slowest successful cell (`pytorch-cpu`, 0.011 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`3x4x4`): `tenferro-direct` is 11.6x faster than the slowest successful cell (`pytorch-cpu`, 0.012 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x8x8`): `tenferro-direct` is 27.8x faster than the slowest successful cell (`jax-cpu`, 0.044 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`8x8x8`): `tenferro-direct` is 19.0x faster than the slowest successful cell (`jax-cpu`, 0.043 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x2x2`): `tenferro-direct` is 53.4x faster than the slowest successful cell (`pytorch-cpu`, 0.042 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x32x32`): `tenferro-direct` is 40.6x faster than the slowest successful cell (`jax-cpu`, 0.074 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x32x32`): `tenferro-trace` is 13.2x faster than the slowest successful cell (`jax-cpu`, 0.074 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x4x4`): `tenferro-direct` is 57.7x faster than the slowest successful cell (`pytorch-cpu`, 0.046 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x4x4`): `tenferro-trace` is 11.0x faster than the slowest successful cell (`pytorch-cpu`, 0.046 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x8x8`): `tenferro-direct` is 44.0x faster than the slowest successful cell (`pytorch-cpu`, 0.042 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x8x8`): `tenferro-trace` is 10.2x faster than the slowest successful cell (`pytorch-cpu`, 0.042 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x2x2`): `tenferro-direct` is 56.3x faster than the slowest successful cell (`pytorch-cpu`, 0.047 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x2x2`): `tenferro-trace` is 10.6x faster than the slowest successful cell (`pytorch-cpu`, 0.047 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x4x4`): `tenferro-direct` is 40.3x faster than the slowest successful cell (`pytorch-cpu`, 0.042 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x8x8`): `tenferro-direct` is 31.1x faster than the slowest successful cell (`pytorch-cpu`, 0.042 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x2x2`): `tenferro-direct` is 49.1x faster than the slowest successful cell (`pytorch-cpu`, 0.043 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x2x2`): `tenferro-trace` is 10.2x faster than the slowest successful cell (`pytorch-cpu`, 0.043 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x4x4`): `tenferro-direct` is 39.4x faster than the slowest successful cell (`pytorch-cpu`, 0.043 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x8x8`): `tenferro-direct` is 30.8x faster than the slowest successful cell (`jax-cpu`, 0.044 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x2x2`): `tenferro-direct` is 39.1x faster than the slowest successful cell (`pytorch-cpu`, 0.046 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x4x4`): `tenferro-direct` is 26.6x faster than the slowest successful cell (`pytorch-cpu`, 0.039 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x8x8`): `tenferro-direct` is 33.0x faster than the slowest successful cell (`jax-cpu`, 0.076 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x8x8`): `tenferro-trace` is 13.1x faster than the slowest successful cell (`jax-cpu`, 0.076 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigvalsh` (f64, threads=1, shape=`1x8x8`): `tenferro-direct` is 10.2x faster than the slowest successful cell (`jax-cpu`, 0.036 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigvalsh` (f64, threads=1, shape=`3x2x2`): `tenferro-direct` is 12.0x faster than the slowest successful cell (`jax-cpu`, 0.023 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`1x32x32`): `tenferro-direct` is 18.6x faster than the slowest successful cell (`jax-cpu`, 0.100 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`1x32x32`): `tenferro-trace` is 12.4x faster than the slowest successful cell (`jax-cpu`, 0.100 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`3x8x8`): `tenferro-direct` is 13.1x faster than the slowest successful cell (`jax-cpu`, 0.039 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`4x8x8`): `tenferro-direct` is 11.0x faster than the slowest successful cell (`jax-cpu`, 0.039 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`8x4x4`): `tenferro-direct` is 10.6x faster than the slowest successful cell (`jax-cpu`, 0.040 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x2x2`): `jax-cpu` is 10.5x faster than the slowest successful cell (`pytorch-cpu`, 0.096 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x2x2`): `tenferro-direct` is 47.8x faster than the slowest successful cell (`pytorch-cpu`, 0.096 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x2x2`): `tenferro-trace` is 21.4x faster than the slowest successful cell (`pytorch-cpu`, 0.096 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x32x32`): `tenferro-direct` is 17.2x faster than the slowest successful cell (`pytorch-cpu`, 0.116 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x32x32`): `tenferro-trace` is 13.7x faster than the slowest successful cell (`pytorch-cpu`, 0.116 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x4x4`): `tenferro-direct` is 30.3x faster than the slowest successful cell (`pytorch-cpu`, 0.076 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x4x4`): `tenferro-trace` is 13.8x faster than the slowest successful cell (`pytorch-cpu`, 0.076 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x8x8`): `tenferro-direct` is 18.6x faster than the slowest successful cell (`pytorch-cpu`, 0.047 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x2x2`): `jax-cpu` is 11.0x faster than the slowest successful cell (`pytorch-cpu`, 0.129 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x2x2`): `tenferro-direct` is 51.7x faster than the slowest successful cell (`pytorch-cpu`, 0.129 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x2x2`): `tenferro-trace` is 28.2x faster than the slowest successful cell (`pytorch-cpu`, 0.129 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x4x4`): `tenferro-direct` is 23.4x faster than the slowest successful cell (`pytorch-cpu`, 0.070 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x4x4`): `tenferro-trace` is 14.2x faster than the slowest successful cell (`pytorch-cpu`, 0.070 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x8x8`): `tenferro-direct` is 21.8x faster than the slowest successful cell (`pytorch-cpu`, 0.072 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x8x8`): `tenferro-trace` is 13.4x faster than the slowest successful cell (`pytorch-cpu`, 0.072 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x2x2`): `jax-cpu` is 18.5x faster than the slowest successful cell (`pytorch-cpu`, 0.180 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x2x2`): `tenferro-direct` is 63.6x faster than the slowest successful cell (`pytorch-cpu`, 0.180 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x2x2`): `tenferro-trace` is 38.3x faster than the slowest successful cell (`pytorch-cpu`, 0.180 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x4x4`): `tenferro-direct` is 12.0x faster than the slowest successful cell (`pytorch-cpu`, 0.039 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x8x8`): `tenferro-direct` is 21.7x faster than the slowest successful cell (`pytorch-cpu`, 0.078 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x8x8`): `tenferro-trace` is 10.9x faster than the slowest successful cell (`pytorch-cpu`, 0.078 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x2x2`): `tenferro-direct` is 30.3x faster than the slowest successful cell (`pytorch-cpu`, 0.104 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x2x2`): `tenferro-trace` is 20.4x faster than the slowest successful cell (`pytorch-cpu`, 0.104 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x4x4`): `tenferro-direct` is 25.6x faster than the slowest successful cell (`pytorch-cpu`, 0.104 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x4x4`): `tenferro-trace` is 17.5x faster than the slowest successful cell (`pytorch-cpu`, 0.104 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x8x8`): `tenferro-direct` is 17.4x faster than the slowest successful cell (`pytorch-cpu`, 0.107 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x8x8`): `tenferro-trace` is 14.5x faster than the slowest successful cell (`pytorch-cpu`, 0.107 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`1x2x2`): `tenferro-direct` is 12.6x faster than the slowest successful cell (`pytorch-cpu`, 0.032 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`1x32x32,rhs=1`): `tenferro-direct` is 13.8x faster than the slowest successful cell (`jax-cpu`, 0.061 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`3x2x2,rhs=1`): `tenferro-direct` is 21.1x faster than the slowest successful cell (`jax-cpu`, 0.033 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x2x2,rhs=1`): `tenferro-direct` is 27.9x faster than the slowest successful cell (`jax-cpu`, 0.069 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x2x2,rhs=1`): `tenferro-trace` is 10.5x faster than the slowest successful cell (`jax-cpu`, 0.069 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x4x4,rhs=1`): `tenferro-direct` is 11.5x faster than the slowest successful cell (`pytorch-cpu`, 0.034 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/cholesky` (f64, threads=4, shape=`1536x1536`): `jax-cpu` is 33.7x faster than the slowest successful cell (`julia-base`, 381.168 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/cholesky` (f64, threads=4, shape=`1536x1536`): `pytorch-cpu` is 26.6x faster than the slowest successful cell (`julia-base`, 381.168 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/cholesky` (f64, threads=4, shape=`1536x1536`): `tenferro-direct` is 34.6x faster than the slowest successful cell (`julia-base`, 381.168 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/cholesky` (f64, threads=4, shape=`1536x1536`): `tenferro-trace` is 35.3x faster than the slowest successful cell (`julia-base`, 381.168 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `jax-cpu` is 36.2x faster than the slowest successful cell (`julia-base`, 239.892 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `pytorch-cpu` is 37.9x faster than the slowest successful cell (`julia-base`, 239.892 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `tenferro-direct` is 38.0x faster than the slowest successful cell (`julia-base`, 239.892 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `tenferro-trace` is 38.5x faster than the slowest successful cell (`julia-base`, 239.892 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eig` (f64, threads=4, shape=`160x160`): `jax-cpu` is 40.4x faster than the slowest successful cell (`julia-base`, 306.333 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eig` (f64, threads=4, shape=`160x160`): `pytorch-cpu` is 45.1x faster than the slowest successful cell (`julia-base`, 306.333 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eig` (f64, threads=4, shape=`160x160`): `tenferro-direct` is 43.2x faster than the slowest successful cell (`julia-base`, 306.333 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eig` (f64, threads=4, shape=`160x160`): `tenferro-trace` is 44.8x faster than the slowest successful cell (`julia-base`, 306.333 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvals` (f64, threads=4, shape=`192x192`): `jax-cpu` is 34.0x faster than the slowest successful cell (`julia-base`, 257.818 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvals` (f64, threads=4, shape=`192x192`): `pytorch-cpu` is 33.9x faster than the slowest successful cell (`julia-base`, 257.818 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvals` (f64, threads=4, shape=`192x192`): `tenferro-direct` is 34.8x faster than the slowest successful cell (`julia-base`, 257.818 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvals` (f64, threads=4, shape=`192x192`): `tenferro-trace` is 34.5x faster than the slowest successful cell (`julia-base`, 257.818 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvalsh` (f64, threads=4, shape=`512x512`): `jax-cpu` is 23.2x faster than the slowest successful cell (`julia-base`, 617.701 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvalsh` (f64, threads=4, shape=`512x512`): `pytorch-cpu` is 31.2x faster than the slowest successful cell (`julia-base`, 617.701 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvalsh` (f64, threads=4, shape=`512x512`): `tenferro-direct` is 27.3x faster than the slowest successful cell (`julia-base`, 617.701 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvalsh` (f64, threads=4, shape=`512x512`): `tenferro-trace` is 31.3x faster than the slowest successful cell (`julia-base`, 617.701 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `jax-cpu` is 49.5x faster than the slowest successful cell (`julia-base`, 365.621 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `pytorch-cpu` is 43.7x faster than the slowest successful cell (`julia-base`, 365.621 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `tenferro-direct` is 41.6x faster than the slowest successful cell (`julia-base`, 365.621 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `tenferro-trace` is 35.7x faster than the slowest successful cell (`julia-base`, 365.621 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lstsq` (f64, threads=4, shape=`768x384,rhs=16`): `pytorch-cpu` is 28.6x faster than the slowest successful cell (`julia-base`, 145.119 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lstsq` (f64, threads=4, shape=`768x384,rhs=16`): `tenferro-eager` is 15.4x faster than the slowest successful cell (`julia-base`, 145.119 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lstsq` (f64, threads=4, shape=`768x384,rhs=16`): `tenferro-trace` is 14.9x faster than the slowest successful cell (`julia-base`, 145.119 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `jax-cpu` is 33.7x faster than the slowest successful cell (`julia-base`, 248.949 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `pytorch-cpu` is 21.9x faster than the slowest successful cell (`julia-base`, 248.949 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `tenferro-direct` is 31.7x faster than the slowest successful cell (`julia-base`, 248.949 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `tenferro-trace` is 30.2x faster than the slowest successful cell (`julia-base`, 248.949 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 10.5x faster than the slowest successful cell (`julia-base`, 6.090 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `jax-cpu` is 97.2x faster than the slowest successful cell (`julia-base`, 990.843 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `pytorch-cpu` is 79.7x faster than the slowest successful cell (`julia-base`, 990.843 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `tenferro-direct` is 80.7x faster than the slowest successful cell (`julia-base`, 990.843 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `tenferro-trace` is 90.0x faster than the slowest successful cell (`julia-base`, 990.843 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `jax-cpu` is 95.7x faster than the slowest successful cell (`julia-base`, 980.383 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `pytorch-cpu` is 88.0x faster than the slowest successful cell (`julia-base`, 980.383 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `tenferro-direct` is 93.8x faster than the slowest successful cell (`julia-base`, 980.383 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `tenferro-trace` is 91.2x faster than the slowest successful cell (`julia-base`, 980.383 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `jax-cpu` is 21.3x faster than the slowest successful cell (`julia-base`, 200.472 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `pytorch-cpu` is 32.4x faster than the slowest successful cell (`julia-base`, 200.472 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `tenferro-direct` is 25.5x faster than the slowest successful cell (`julia-base`, 200.472 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `tenferro-trace` is 24.9x faster than the slowest successful cell (`julia-base`, 200.472 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `jax-cpu` is 63.8x faster than the slowest successful cell (`julia-base`, 1947.178 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `pytorch-cpu` is 55.0x faster than the slowest successful cell (`julia-base`, 1947.178 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `tenferro-eager` is 54.8x faster than the slowest successful cell (`julia-base`, 1947.178 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `tenferro-trace` is 54.4x faster than the slowest successful cell (`julia-base`, 1947.178 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`4096x256`): `jax-cpu` is 16.0x faster than the slowest successful cell (`tenferro-eager`, 8.507 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=1, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 7735.5x faster than the slowest successful cell (`julia-base`, 15.154 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=4, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 5029.7x faster than the slowest successful cell (`julia-base`, 14.878 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=1, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 5866.4x faster than the slowest successful cell (`pytorch-cpu`, 11.733 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=4, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 6477.8x faster than the slowest successful cell (`julia-base`, 15.657 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`): `tenferro-trace` is 69301.8x faster than the slowest successful cell (`julia-base`, 199.243 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`): `strided-jl` is 10.5x faster than the slowest successful cell (`julia-base`, 210.946 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`): `tenferro-trace` is 61734.2x faster than the slowest successful cell (`julia-base`, 210.946 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=1, shape=`33554432 -> 8192x4096`): `julia-base` is 179.2x faster than the slowest successful cell (`tenferro-direct`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=4, shape=`33554432 -> 8192x4096`): `julia-base` is 287.4x faster than the slowest successful cell (`tenferro-direct`, 0.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=1, shape=`4194304 -> 2096128`): `julia-base` is 263.6x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=4, shape=`4194304 -> 2096128`): `julia-base` is 61.5x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=1, shape=`4096x4096`): `julia-base` is 376.0x faster than the slowest successful cell (`pytorch-cpu`, 0.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=1, shape=`4096x4096`): `tenferro-direct` is 11.5x faster than the slowest successful cell (`pytorch-cpu`, 0.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=4, shape=`4096x4096`): `julia-base` is 127.6x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

## Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

- `cpu/indexing_layout/slice` (f64, threads=1, shape=`4194304 -> 2096128`), `tenferro-trace`: 16769024 minimum logical bytes imply 7594.7 GB/s at 0.002 ms. The conservative physical lower bound is 0.168 ms, so the cell is 75.9x faster than the bound.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`), `tenferro-trace`: 16769024 minimum logical bytes imply 5919.2 GB/s at 0.003 ms. The conservative physical lower bound is 0.042 ms, so the cell is 14.8x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=1, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 137026.8 GB/s at 0.002 ms. The conservative physical lower bound is 2.684 ms, so the cell is 1370.3x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=4, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 90749.0 GB/s at 0.003 ms. The conservative physical lower bound is 0.671 ms, so the cell is 226.9x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=1, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 134217.7 GB/s at 0.002 ms. The conservative physical lower bound is 2.684 ms, so the cell is 1342.2x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=4, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 111061.4 GB/s at 0.002 ms. The conservative physical lower bound is 0.671 ms, so the cell is 277.7x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 46684.4 GB/s at 0.003 ms. The conservative physical lower bound is 1.342 ms, so the cell is 466.8x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 39279.4 GB/s at 0.003 ms. The conservative physical lower bound is 0.336 ms, so the cell is 98.2x faster than the bound.
