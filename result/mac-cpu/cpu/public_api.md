# CPU Public API Benchmark Results

- Suite: `cpu/public_api`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/public_api.yaml`
- Run metadata: `data/results/mac-cpu/cpu/public_api/20261125_182415/run.yaml`
- Timestamp: `2026-11-25T18:24:15.591687+00:00`

Latest run: `./scripts/run_cpu_public_api.sh 1 4`.

- Public API coverage manifest: `benchmarks/cpu/public_api_coverage.yaml`

This file is generated from sequential CPU public API runs under `data/results/mac-cpu/cpu/public_api/20261125_182415`.

- tenferro-rs commit: `1ba9f33521691170c763dfce9b6c9bb56ca523c2`

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

- CSV: `data/results/mac-cpu/cpu/public_api/20261125_182415/cpu_public_api_t1_20261125_182415.csv`
- CSV: `data/results/mac-cpu/cpu/public_api/20261125_182415/cpu_public_api_t4_20261125_182415.csv`

## CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| cpu/complex | `cholesky` | c64 | 1 | `448x448` | 5.284 ± 0.102 | - | 5.562 ± 0.885 | 4.441 ± 0.137 | 1.792 ± 2.894 | 11.168 ± 5.278 | - |
| cpu/complex | `cholesky` | c64 | 4 | `448x448` | 4.787 ± 0.914 | - | 4.397 ± 0.355 | 4.833 ± 0.336 | 1.472 ± 0.253 | 80.410 ± 17.692 | - |
| cpu/complex | `conj` | c64 | 1 | `16777216` | 12.474 ± 0.919 | - | 13.191 ± 0.705 | 13.423 ± 1.063 | 12.888 ± 0.434 | 17.565 ± 4.678 | 16.296 ± 9.352 |
| cpu/complex | `conj` | c64 | 4 | `16777216` | 14.676 ± 0.835 | - | 14.658 ± 0.858 | 16.659 ± 0.794 | 12.961 ± 1.370 | 14.075 ± 4.906 | 16.020 ± 3.299 |
| cpu/complex | `div` | c64 | 1 | `8388608` | 45.559 ± 9.408 | - | 45.944 ± 1.384 | 20.702 ± 1.184 | 45.486 ± 12.070 | 86.823 ± 45.200 | 65.677 ± 19.582 |
| cpu/complex | `div` | c64 | 4 | `8388608` | 38.327 ± 7.574 | - | 37.451 ± 13.968 | 12.504 ± 1.700 | 24.185 ± 4.053 | 67.642 ± 34.426 | 31.448 ± 4.448 |
| cpu/complex | `dot_general` | c64 | 1 | `640x640` | 87.058 ± 21.682 | - | 74.804 ± 5.633 | 12.172 ± 0.619 | 54.659 ± 0.884 | 104.072 ± 60.041 | - |
| cpu/complex | `dot_general` | c64 | 4 | `640x640` | 32.427 ± 4.341 | - | 34.118 ± 10.743 | 12.224 ± 2.168 | 16.796 ± 0.647 | 46.621 ± 7.036 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 1 | `640x640` | 74.311 ± 1.499 | - | unsupported | 12.452 ± 0.869 | 54.369 ± 0.805 | 103.503 ± 25.649 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 4 | `640x640` | 35.336 ± 15.786 | - | unsupported | 12.025 ± 1.367 | 17.381 ± 1.977 | 47.564 ± 11.229 | - |
| cpu/complex | `eig` | c64 | 1 | `112x112` | 7.935 ± 0.143 | - | 8.057 ± 1.046 | 12.862 ± 0.779 | 22.396 ± 0.652 | 8.911 ± 0.345 | - |
| cpu/complex | `eig` | c64 | 4 | `112x112` | 10.694 ± 4.080 | - | 20.144 ± 2.139 | 14.527 ± 3.321 | 20.361 ± 0.487 | 468.299 ± 97.436 | - |
| cpu/complex | `exp` | c64 | 1 | `4194304` | 35.171 ± 0.306 | - | 38.037 ± 8.023 | 45.202 ± 10.287 | 43.095 ± 14.739 | 73.001 ± 20.503 | 93.555 ± 51.513 |
| cpu/complex | `exp` | c64 | 4 | `4194304` | 38.464 ± 21.115 | - | 26.896 ± 7.793 | 22.482 ± 8.162 | 21.836 ± 0.419 | 52.347 ± 8.573 | 26.397 ± 4.246 |
| cpu/complex | `log` | c64 | 1 | `4194304` | 49.903 ± 5.440 | - | 56.727 ± 4.972 | 56.938 ± 6.659 | 105.113 ± 34.216 | 124.526 ± 49.130 | 180.219 ± 46.832 |
| cpu/complex | `log` | c64 | 4 | `4194304` | 39.359 ± 21.951 | - | 31.959 ± 9.301 | 26.398 ± 1.774 | 31.148 ± 4.651 | 81.876 ± 6.752 | 43.633 ± 12.987 |
| cpu/complex | `mul` | c64 | 1 | `8388608` | 11.181 ± 1.376 | - | 9.747 ± 0.991 | 11.292 ± 2.916 | 10.137 ± 0.347 | 20.605 ± 5.236 | 18.145 ± 5.787 |
| cpu/complex | `mul` | c64 | 4 | `8388608` | 13.430 ± 3.363 | - | 13.669 ± 2.612 | 11.354 ± 0.589 | 8.865 ± 0.840 | 11.383 ± 1.988 | 15.851 ± 2.998 |
| cpu/complex | `norm_fro` | c64 | 1 | `2048x1536` | 1.317 ± 0.105 | - | 8.447 ± 0.364 | 6.063 ± 0.667 | 4.801 ± 0.213 | 10.504 ± 2.037 | - |
| cpu/complex | `norm_fro` | c64 | 4 | `2048x1536` | 1.490 ± 0.404 | - | 5.264 ± 0.508 | 4.183 ± 0.810 | 3.657 ± 0.233 | 11.983 ± 1.572 | - |
| cpu/complex | `qr` | c64 | 1 | `256x256` | 10.655 ± 0.645 | - | 10.458 ± 0.721 | 8.137 ± 0.152 | 7.682 ± 0.325 | 12.255 ± 5.849 | - |
| cpu/complex | `qr` | c64 | 4 | `256x256` | 7.134 ± 0.852 | - | 7.726 ± 0.217 | 8.141 ± 0.341 | 6.910 ± 0.095 | 185.894 ± 38.683 | - |
| cpu/complex | `solve` | c64 | 1 | `384x384,rhs=8` | 6.641 ± 0.771 | - | 6.730 ± 0.309 | 5.712 ± 0.372 | 2.630 ± 0.615 | 6.808 ± 0.677 | - |
| cpu/complex | `solve` | c64 | 4 | `384x384,rhs=8` | 5.934 ± 0.349 | - | 10.544 ± 9.860 | 6.780 ± 1.971 | 2.476 ± 0.050 | 152.055 ± 25.063 | - |
| cpu/complex | `svd` | c64 | 1 | `160x160` | 8.904 ± 0.673 | - | 8.849 ± 0.534 | 7.915 ± 0.553 | 7.222 ± 0.139 | 9.032 ± 4.489 | - |
| cpu/complex | `svd` | c64 | 4 | `160x160` | 8.068 ± 0.619 | - | 8.406 ± 0.532 | 8.237 ± 0.209 | 6.642 ± 0.068 | 640.955 ± 25.366 | - |
| cpu/complex | `tensordot` | c64 | 1 | `640x640` | 73.820 ± 1.718 | - | 75.201 ± 9.525 | 11.654 ± 0.314 | 56.511 ± 4.566 | 64.732 ± 9.369 | - |
| cpu/complex | `tensordot` | c64 | 4 | `640x640` | 37.043 ± 5.087 | - | 31.944 ± 9.612 | 12.306 ± 2.673 | 17.508 ± 1.208 | 64.140 ± 21.898 | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 1 | `1024x1024` | 58.267 ± 3.520 | - | unsupported | 11.036 ± 0.856 | 53.540 ± 6.594 | - | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 4 | `1024x1024` | 24.379 ± 5.368 | - | unsupported | 13.089 ± 2.619 | 27.355 ± 6.097 | - | - |
| cpu/elementwise_reduction | `abs` | f64 | 1 | `33554432` | 13.020 ± 2.925 | - | 18.826 ± 8.127 | 13.489 ± 1.864 | 15.565 ± 1.373 | 13.115 ± 8.058 | 15.040 ± 0.856 |
| cpu/elementwise_reduction | `abs` | f64 | 4 | `33554432` | 18.305 ± 5.631 | - | 16.285 ± 2.771 | 14.035 ± 1.539 | 17.956 ± 3.521 | 16.147 ± 3.567 | 16.270 ± 1.410 |
| cpu/elementwise_reduction | `add` | f64 | 1 | `33554432` | 23.143 ± 4.484 | - | 17.642 ± 2.902 | 19.685 ± 2.618 | 26.804 ± 8.342 | 21.767 ± 7.678 | 23.582 ± 6.792 |
| cpu/elementwise_reduction | `add` | f64 | 4 | `33554432` | 24.009 ± 3.709 | - | 23.663 ± 3.890 | 31.421 ± 6.085 | 56.973 ± 71.924 | 25.482 ± 8.915 | 21.981 ± 2.766 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 1 | `4194304` | 40.831 ± 12.958 | - | 53.680 ± 26.107 | 38.620 ± 3.618 | 33.258 ± 1.545 | 51.320 ± 9.126 | 52.865 ± 9.115 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 4 | `4194304` | 28.754 ± 6.444 | - | 30.818 ± 16.345 | 24.542 ± 2.223 | 23.019 ± 6.725 | 54.212 ± 28.311 | 33.339 ± 6.872 |
| cpu/elementwise_reduction | `clamp` | f64 | 1 | `8388608` | 7.313 ± 0.860 | - | 11.086 ± 3.904 | 6.916 ± 0.738 | 9.233 ± 1.895 | 6.915 ± 0.903 | 7.416 ± 1.199 |
| cpu/elementwise_reduction | `clamp` | f64 | 4 | `8388608` | 7.459 ± 2.553 | - | 7.421 ± 0.531 | 7.183 ± 0.232 | 8.829 ± 1.496 | 7.930 ± 0.883 | 7.988 ± 1.748 |
| cpu/elementwise_reduction | `compare_lt` | f64 | 1 | `33554432` | 16.978 ± 2.733 | - | 16.994 ± 2.047 | 18.533 ± 2.364 | 18.148 ± 9.644 | 20.881 ± 9.938 | - |
| cpu/elementwise_reduction | `compare_lt` | f64 | 4 | `33554432` | 14.028 ± 0.688 | - | 13.840 ± 0.344 | 17.367 ± 1.227 | 19.299 ± 3.506 | 19.393 ± 2.166 | - |
| cpu/elementwise_reduction | `cos` | f64 | 1 | `8388608` | 30.561 ± 0.106 | - | 31.178 ± 0.707 | 37.182 ± 1.209 | 28.551 ± 0.543 | 41.333 ± 2.968 | 42.797 ± 5.723 |
| cpu/elementwise_reduction | `cos` | f64 | 4 | `8388608` | 20.942 ± 6.302 | - | 16.526 ± 4.690 | 25.691 ± 6.029 | 24.584 ± 4.686 | 51.944 ± 16.397 | 26.386 ± 6.754 |
| cpu/elementwise_reduction | `div` | f64 | 1 | `33554432` | 19.512 ± 2.337 | - | 20.585 ± 2.339 | 23.543 ± 5.270 | 28.346 ± 1.534 | 24.237 ± 24.407 | 20.742 ± 2.485 |
| cpu/elementwise_reduction | `div` | f64 | 4 | `33554432` | 29.109 ± 8.947 | - | 27.489 ± 2.340 | 22.677 ± 1.674 | 34.347 ± 4.611 | 21.969 ± 16.710 | 23.194 ± 2.618 |
| cpu/elementwise_reduction | `exp` | f64 | 1 | `8388608` | 17.052 ± 6.090 | - | 16.911 ± 1.789 | 29.817 ± 2.078 | 17.030 ± 3.646 | 34.141 ± 17.312 | 44.855 ± 7.136 |
| cpu/elementwise_reduction | `exp` | f64 | 4 | `8388608` | 12.605 ± 3.259 | - | 11.305 ± 4.604 | 16.240 ± 1.390 | 17.691 ± 12.895 | 47.356 ± 18.938 | 27.770 ± 9.167 |
| cpu/elementwise_reduction | `expm1` | f64 | 1 | `4194304` | 29.151 ± 2.531 | - | 34.805 ± 4.984 | 41.226 ± 19.028 | 18.512 ± 0.705 | 28.295 ± 1.005 | 32.550 ± 8.401 |
| cpu/elementwise_reduction | `expm1` | f64 | 4 | `4194304` | 25.527 ± 13.065 | - | 30.268 ± 16.880 | 15.662 ± 1.509 | 13.766 ± 8.348 | 32.093 ± 6.660 | 15.484 ± 7.193 |
| cpu/elementwise_reduction | `log` | f64 | 1 | `8388608` | 30.278 ± 5.987 | - | 29.939 ± 0.189 | 48.171 ± 1.930 | 26.683 ± 1.869 | 63.875 ± 20.522 | 72.308 ± 7.752 |
| cpu/elementwise_reduction | `log` | f64 | 4 | `8388608` | 18.708 ± 1.518 | - | 19.634 ± 3.138 | 26.626 ± 9.725 | 24.043 ± 10.098 | 91.446 ± 28.181 | 45.910 ± 13.844 |
| cpu/elementwise_reduction | `log1p` | f64 | 1 | `4194304` | 20.145 ± 3.910 | - | 19.347 ± 3.322 | 26.146 ± 7.267 | 20.643 ± 2.224 | 23.222 ± 1.641 | 24.869 ± 8.582 |
| cpu/elementwise_reduction | `log1p` | f64 | 4 | `4194304` | 14.422 ± 5.525 | - | 13.550 ± 5.525 | 13.075 ± 2.309 | 19.728 ± 5.407 | 30.303 ± 19.753 | 12.440 ± 8.936 |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 1 | `1024x64` | 0.252 ± 0.004 | 0.268 ± 0.018 | unsupported | unsupported | 0.255 ± 0.051 | - | - |
| cpu/elementwise_reduction | `masked_log_softmax_axis1` | f64 | 4 | `1024x64` | 0.592 ± 0.528 | 0.346 ± 0.253 | unsupported | unsupported | 0.368 ± 0.131 | - | - |
| cpu/elementwise_reduction | `maximum` | f64 | 1 | `33554432` | 26.683 ± 11.497 | - | 22.772 ± 4.296 | 20.588 ± 1.776 | 27.913 ± 1.930 | 22.802 ± 15.506 | 23.487 ± 1.562 |
| cpu/elementwise_reduction | `maximum` | f64 | 4 | `33554432` | 21.289 ± 2.970 | - | 24.689 ± 3.384 | 19.430 ± 3.643 | 30.811 ± 6.579 | 28.526 ± 6.540 | 23.130 ± 2.087 |
| cpu/elementwise_reduction | `minimum` | f64 | 1 | `33554432` | 24.370 ± 1.562 | - | 23.918 ± 5.593 | 21.423 ± 1.075 | 30.440 ± 3.988 | 48.714 ± 69.814 | 21.794 ± 1.581 |
| cpu/elementwise_reduction | `minimum` | f64 | 4 | `33554432` | 28.556 ± 5.800 | - | 20.593 ± 2.524 | 23.722 ± 2.619 | 28.670 ± 6.373 | 22.790 ± 6.953 | 25.181 ± 5.403 |
| cpu/elementwise_reduction | `mul` | f64 | 1 | `33554432` | 18.790 ± 1.096 | - | 18.842 ± 2.759 | 20.470 ± 2.806 | 27.719 ± 1.689 | 23.690 ± 7.023 | 19.061 ± 0.418 |
| cpu/elementwise_reduction | `mul` | f64 | 4 | `33554432` | 30.855 ± 5.252 | - | 30.608 ± 2.968 | 24.199 ± 5.771 | 35.216 ± 11.135 | 25.068 ± 5.606 | 24.572 ± 2.836 |
| cpu/elementwise_reduction | `neg` | f64 | 1 | `33554432` | 13.389 ± 2.472 | - | 12.982 ± 1.215 | 12.485 ± 0.465 | 15.755 ± 0.546 | 12.263 ± 4.140 | 14.980 ± 1.144 |
| cpu/elementwise_reduction | `neg` | f64 | 4 | `33554432` | 19.850 ± 10.438 | - | 17.035 ± 2.626 | 14.291 ± 1.553 | 15.841 ± 3.284 | 14.415 ± 4.136 | 15.402 ± 2.624 |
| cpu/elementwise_reduction | `pow` | f64 | 1 | `4194304` | 39.952 ± 1.072 | - | 40.377 ± 0.965 | 95.200 ± 20.502 | 45.656 ± 2.022 | 148.974 ± 54.733 | 96.813 ± 16.987 |
| cpu/elementwise_reduction | `pow` | f64 | 4 | `4194304` | 38.989 ± 8.535 | - | 35.080 ± 6.849 | 35.046 ± 4.948 | 34.206 ± 7.923 | 139.770 ± 21.537 | 51.867 ± 11.479 |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 1 | `8192x4096` | 8.611 ± 4.455 | - | 17.880 ± 10.584 | 11.587 ± 0.194 | 37.594 ± 15.043 | 10.993 ± 11.080 | - |
| cpu/elementwise_reduction | `reduce_max_all` | f64 | 4 | `8192x4096` | 8.012 ± 2.431 | - | 10.714 ± 2.699 | 7.709 ± 1.450 | 13.049 ± 2.829 | 6.986 ± 0.983 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 1 | `2048x2048` | 0.974 ± 0.023 | - | 1.052 ± 0.070 | 16.592 ± 14.276 | 3.195 ± 1.700 | 1.127 ± 0.162 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 4 | `2048x2048` | 0.988 ± 0.060 | - | 1.093 ± 0.164 | 13.241 ± 1.068 | 2.190 ± 0.431 | 0.898 ± 0.075 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 1 | `2048x2048` | 0.917 ± 0.085 | - | 0.908 ± 0.124 | 1.571 ± 0.082 | 2.826 ± 1.455 | 0.909 ± 0.137 | - |
| cpu/elementwise_reduction | `reduce_max_axis1` | f64 | 4 | `2048x2048` | 1.890 ± 0.032 | - | 3.388 ± 0.671 | 1.205 ± 0.228 | 1.245 ± 0.633 | 1.012 ± 0.134 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 1 | `8192x4096` | 13.514 ± 8.747 | - | 12.469 ± 6.273 | 12.115 ± 1.217 | 39.334 ± 15.220 | 10.756 ± 2.789 | - |
| cpu/elementwise_reduction | `reduce_min_all` | f64 | 4 | `8192x4096` | 8.982 ± 1.894 | - | 8.115 ± 2.372 | 8.759 ± 0.935 | 18.808 ± 5.979 | 7.015 ± 3.074 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 1 | `2048x2048` | 1.051 ± 0.154 | - | 0.943 ± 0.068 | 16.655 ± 0.808 | 4.257 ± 3.272 | 0.987 ± 0.213 | - |
| cpu/elementwise_reduction | `reduce_min_axis0` | f64 | 4 | `2048x2048` | 1.005 ± 0.174 | - | 0.932 ± 0.068 | 13.745 ± 1.915 | 2.574 ± 0.256 | 0.806 ± 0.038 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 1 | `4096x4096` | 3.275 ± 0.309 | - | 3.389 ± 0.668 | 6.325 ± 1.254 | 4.626 ± 3.562 | 4.260 ± 0.723 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 4 | `4096x4096` | 8.378 ± 2.140 | - | 7.581 ± 1.092 | 3.989 ± 0.258 | 4.203 ± 0.592 | 4.507 ± 0.604 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 1 | `8192x4096` | 7.370 ± 2.381 | - | 6.727 ± 3.032 | 7.196 ± 1.109 | 60.100 ± 12.023 | 9.440 ± 3.211 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 4 | `8192x4096` | 9.575 ± 2.665 | - | 8.000 ± 1.897 | 6.475 ± 0.572 | 27.654 ± 5.508 | 7.746 ± 1.853 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 1 | `2048x2048` | 0.772 ± 0.026 | - | 0.778 ± 0.022 | 4.942 ± 0.173 | 2.389 ± 0.310 | 0.882 ± 0.078 | - |
| cpu/elementwise_reduction | `reduce_prod_axis0` | f64 | 4 | `2048x2048` | 1.058 ± 0.149 | - | 1.057 ± 0.076 | 3.320 ± 0.539 | 2.108 ± 0.509 | 1.233 ± 0.371 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 1 | `2048x2048` | 0.784 ± 0.015 | - | 0.828 ± 0.044 | 1.007 ± 0.603 | 1.339 ± 2.655 | 1.130 ± 0.287 | - |
| cpu/elementwise_reduction | `reduce_prod_axis1` | f64 | 4 | `2048x2048` | 2.086 ± 0.386 | - | 2.327 ± 0.248 | 0.942 ± 0.095 | 2.252 ± 0.799 | 0.798 ± 0.030 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 1 | `8192x4096` | 7.705 ± 0.564 | - | 7.247 ± 1.462 | 7.197 ± 1.639 | 40.457 ± 7.955 | 6.402 ± 4.418 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 4 | `8192x4096` | 7.130 ± 1.615 | - | 8.883 ± 1.484 | 6.572 ± 0.372 | 18.040 ± 8.401 | 8.638 ± 2.826 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 1 | `2048x2048` | 0.965 ± 0.174 | - | 0.869 ± 0.083 | 4.765 ± 0.352 | 3.880 ± 1.774 | 1.010 ± 0.185 | - |
| cpu/elementwise_reduction | `reduce_sum_axis0` | f64 | 4 | `2048x2048` | 1.045 ± 0.091 | - | 1.018 ± 0.048 | 3.327 ± 0.364 | 2.331 ± 0.938 | 0.815 ± 0.074 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 1 | `2048x2048` | 0.847 ± 0.101 | - | 0.787 ± 0.030 | 0.916 ± 0.160 | 0.990 ± 0.084 | 0.943 ± 0.167 | - |
| cpu/elementwise_reduction | `reduce_sum_axis1` | f64 | 4 | `2048x2048` | 2.222 ± 1.542 | - | 2.219 ± 1.305 | 0.973 ± 0.307 | 1.548 ± 0.931 | 1.019 ± 0.322 | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 1 | `2048x2048` | 0.806 ± 0.086 | 0.827 ± 0.054 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `reduce_sum_squares_axis0` | f64 | 4 | `2048x2048` | 1.241 ± 1.310 | 0.978 ± 0.217 | unsupported | unsupported | unsupported | - | - |
| cpu/elementwise_reduction | `rem` | f64 | 1 | `8388608` | 31.227 ± 7.732 | - | 28.971 ± 2.193 | 35.602 ± 2.384 | 40.278 ± 4.720 | 71.537 ± 22.903 | 32.455 ± 1.164 |
| cpu/elementwise_reduction | `rem` | f64 | 4 | `8388608` | 14.271 ± 2.806 | - | 17.831 ± 5.380 | 26.420 ± 4.854 | 30.401 ± 3.902 | 79.017 ± 64.772 | 30.990 ± 17.816 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 1 | `33554432` | 20.343 ± 3.297 | - | 20.823 ± 0.550 | 21.114 ± 1.467 | 42.525 ± 11.477 | 43.587 ± 7.235 | 45.418 ± 9.546 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 4 | `33554432` | 21.277 ± 5.488 | - | 18.099 ± 2.544 | 13.557 ± 1.949 | 27.620 ± 4.764 | 51.103 ± 1.754 | 23.860 ± 4.318 |
| cpu/elementwise_reduction | `select` | f64 | 1 | `33554432` | 29.867 ± 2.440 | - | 28.557 ± 2.842 | 23.302 ± 1.961 | 32.270 ± 8.393 | 36.230 ± 6.902 | - |
| cpu/elementwise_reduction | `select` | f64 | 4 | `33554432` | 26.632 ± 0.916 | - | 26.474 ± 2.414 | 22.285 ± 6.978 | 33.076 ± 5.576 | 35.445 ± 5.992 | - |
| cpu/elementwise_reduction | `sign` | f64 | 1 | `33554432` | 19.292 ± 7.555 | - | 17.663 ± 3.990 | 12.995 ± 0.743 | 16.902 ± 0.454 | 14.251 ± 4.163 | 20.836 ± 0.812 |
| cpu/elementwise_reduction | `sign` | f64 | 4 | `33554432` | 15.381 ± 1.587 | - | 14.429 ± 3.635 | 13.741 ± 1.439 | 22.157 ± 9.417 | 16.712 ± 4.254 | 18.436 ± 7.042 |
| cpu/elementwise_reduction | `sin` | f64 | 1 | `8388608` | 25.015 ± 7.328 | - | 23.600 ± 0.589 | 31.783 ± 1.458 | 26.122 ± 1.280 | 39.187 ± 4.019 | 36.242 ± 12.646 |
| cpu/elementwise_reduction | `sin` | f64 | 4 | `8388608` | 16.099 ± 3.097 | - | 12.910 ± 3.497 | 17.449 ± 5.773 | 22.970 ± 5.282 | 64.712 ± 24.712 | 22.351 ± 8.047 |
| cpu/elementwise_reduction | `sqrt` | f64 | 1 | `33554432` | 16.968 ± 4.942 | - | 13.682 ± 0.783 | 13.657 ± 2.291 | 25.961 ± 3.324 | 29.584 ± 2.535 | 51.409 ± 83.227 |
| cpu/elementwise_reduction | `sqrt` | f64 | 4 | `33554432` | 16.934 ± 3.819 | - | 19.224 ± 6.192 | 13.355 ± 1.123 | 23.044 ± 7.765 | 30.872 ± 2.716 | 18.023 ± 7.958 |
| cpu/elementwise_reduction | `sub` | f64 | 1 | `33554432` | 18.603 ± 1.222 | - | 18.373 ± 1.502 | 20.124 ± 2.925 | 26.699 ± 4.313 | 21.331 ± 24.771 | 20.730 ± 2.576 |
| cpu/elementwise_reduction | `sub` | f64 | 4 | `33554432` | 30.315 ± 6.879 | - | 25.623 ± 11.761 | 23.385 ± 1.440 | 43.994 ± 18.268 | 22.255 ± 19.323 | 22.255 ± 1.018 |
| cpu/elementwise_reduction | `tanh` | f64 | 1 | `8388608` | 78.437 ± 20.689 | - | 75.540 ± 2.489 | 75.501 ± 3.062 | 13.359 ± 0.294 | 50.183 ± 1.857 | 62.614 ± 19.600 |
| cpu/elementwise_reduction | `tanh` | f64 | 4 | `8388608` | 37.268 ± 4.862 | - | 42.691 ± 11.070 | 45.493 ± 13.300 | 12.988 ± 3.235 | 75.904 ± 15.002 | 37.044 ± 3.150 |
| cpu/elementwise_reduction | `where_select` | f64 | 1 | `33554432` | 29.907 ± 2.149 | 31.338 ± 4.265 | unsupported | 24.278 ± 1.820 | 33.655 ± 4.170 | - | - |
| cpu/elementwise_reduction | `where_select` | f64 | 4 | `33554432` | 27.444 ± 6.068 | 31.762 ± 3.964 | unsupported | 21.150 ± 3.642 | 31.031 ± 3.839 | - | - |
| cpu/indexing_layout | `concatenate` | f64 | 1 | `1048576+1048576` | 0.746 ± 0.010 | - | 0.768 ± 0.035 | 0.767 ± 0.167 | 0.755 ± 0.022 | 0.879 ± 0.107 | - |
| cpu/indexing_layout | `concatenate` | f64 | 4 | `1048576+1048576` | 0.854 ± 0.060 | - | 1.060 ± 0.095 | 1.024 ± 0.204 | 1.182 ± 0.315 | 0.843 ± 0.177 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 1 | `4194304 -> 2097152` | 0.685 ± 0.059 | - | 0.740 ± 0.111 | 0.837 ± 0.053 | 0.756 ± 0.025 | 0.856 ± 0.048 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 4 | `4194304 -> 2097152` | 0.852 ± 0.090 | - | 1.547 ± 0.885 | 1.210 ± 0.681 | 1.385 ± 0.380 | 1.032 ± 0.135 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 1 | `2097152` | 1.314 ± 0.360 | - | unsupported | 1.336 ± 0.154 | 1.148 ± 0.037 | 1.607 ± 6.155 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 4 | `2097152` | 2.137 ± 1.788 | - | unsupported | 1.635 ± 0.286 | 1.401 ± 0.299 | 1.301 ± 3.624 | - |
| cpu/indexing_layout | `gather` | f64 | 1 | `262144` | 0.329 ± 0.007 | - | 0.395 ± 0.017 | 0.301 ± 0.103 | 0.344 ± 0.010 | 2.104 ± 3.098 | - |
| cpu/indexing_layout | `gather` | f64 | 4 | `262144` | 1.242 ± 0.742 | - | 0.570 ± 0.248 | 0.334 ± 0.731 | 0.355 ± 0.167 | 1.133 ± 1.332 | - |
| cpu/indexing_layout | `pad` | f64 | 1 | `2097152` | 0.746 ± 0.064 | - | 0.760 ± 0.081 | 1.406 ± 0.143 | 0.799 ± 0.034 | - | - |
| cpu/indexing_layout | `pad` | f64 | 4 | `2097152` | 0.708 ± 0.049 | - | 1.248 ± 0.224 | 1.922 ± 0.477 | 1.143 ± 0.090 | - | - |
| cpu/indexing_layout | `reverse` | f64 | 1 | `2097152` | 0.805 ± 0.116 | - | 0.716 ± 0.123 | 0.937 ± 0.178 | 0.839 ± 0.045 | 1.407 ± 9.441 | - |
| cpu/indexing_layout | `reverse` | f64 | 4 | `2097152` | 0.950 ± 0.063 | - | 1.020 ± 0.093 | 1.380 ± 0.551 | 1.487 ± 0.180 | 4.713 ± 11.424 | - |
| cpu/indexing_layout | `scatter` | f64 | 1 | `262144` | 0.587 ± 0.057 | - | 0.654 ± 0.052 | 0.438 ± 0.068 | 0.450 ± 0.034 | 2.104 ± 3.217 | - |
| cpu/indexing_layout | `scatter` | f64 | 4 | `262144` | 0.697 ± 0.031 | - | 0.903 ± 0.069 | 1.109 ± 0.520 | 0.458 ± 0.015 | 0.369 ± 0.144 | - |
| cpu/indexing_layout | `slice` | f64 | 1 | `4194304 -> 2096128` | 1.200 ± 0.040 | - | 0.002 ± 0.000 | 1.353 ± 0.071 | 1.480 ± 0.042 | 2.120 ± 0.567 | - |
| cpu/indexing_layout | `slice` | f64 | 4 | `4194304 -> 2096128` | 1.347 ± 0.103 | - | 0.001 ± 0.000 | 1.627 ± 0.223 | 1.338 ± 0.138 | 1.582 ± 0.268 | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x2x2` | 0.051 ± 0.006 | - | 0.049 ± 0.010 | 0.093 ± 0.000 | 0.089 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x4x4` | 0.100 ± 0.010 | - | 0.121 ± 0.052 | 0.126 ± 0.000 | 0.128 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1024x8x8` | 0.254 ± 0.042 | - | 0.231 ± 0.044 | 0.210 ± 0.003 | 0.232 ± 0.043 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x128x128` | 0.055 ± 0.011 | - | 0.059 ± 0.011 | 0.039 ± 0.004 | 0.088 ± 0.030 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.004 ± 0.000 | 0.009 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x32x32` | 0.003 ± 0.000 | - | 0.004 ± 0.001 | 0.005 ± 0.000 | 0.054 ± 0.029 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x4x4` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `1x8x8` | 0.001 ± 0.000 | - | 0.002 ± 0.001 | 0.004 ± 0.000 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x2x2` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.004 ± 0.001 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x4x4` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.004 ± 0.000 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `3x8x8` | 0.001 ± 0.001 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x2x2` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.004 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x4x4` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.004 ± 0.000 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `4x8x8` | 0.004 ± 0.002 | - | 0.009 ± 0.003 | 0.004 ± 0.000 | 0.079 ± 0.034 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x128x128` | 0.440 ± 0.029 | - | 0.445 ± 0.065 | 0.244 ± 0.004 | 0.289 ± 0.046 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x2x2` | 0.001 ± 0.000 | - | 0.002 ± 0.001 | 0.004 ± 0.000 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x32x32` | 0.018 ± 0.003 | - | 0.020 ± 0.002 | 0.017 ± 0.001 | 0.066 ± 0.029 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x4x4` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 1 | `8x8x8` | 0.006 ± 0.001 | - | 0.011 ± 0.001 | 0.005 ± 0.000 | 0.068 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x2x2` | 0.137 ± 0.056 | - | 0.058 ± 0.018 | 0.267 ± 0.069 | 0.091 ± 0.020 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x4x4` | 0.221 ± 0.017 | - | 0.224 ± 0.044 | 0.241 ± 0.026 | 0.152 ± 0.025 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1024x8x8` | 0.225 ± 0.006 | - | 0.232 ± 0.019 | 0.351 ± 0.165 | 0.281 ± 0.133 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x128x128` | 0.051 ± 0.007 | - | 0.051 ± 0.000 | 0.189 ± 0.023 | 0.062 ± 0.052 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x2x2` | 0.003 ± 0.001 | - | 0.002 ± 0.001 | 0.051 ± 0.058 | 0.025 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x32x32` | 0.003 ± 0.001 | - | 0.004 ± 0.000 | 0.036 ± 0.081 | 0.035 ± 0.052 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x4x4` | 0.001 ± 0.000 | - | 0.002 ± 0.001 | 0.131 ± 0.087 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `1x8x8` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.116 ± 0.048 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x2x2` | 0.001 ± 0.002 | - | 0.002 ± 0.000 | 0.040 ± 0.032 | 0.008 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x4x4` | 0.002 ± 0.001 | - | 0.003 ± 0.000 | 0.155 ± 0.091 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `3x8x8` | 0.002 ± 0.001 | - | 0.003 ± 0.000 | 0.094 ± 0.069 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x2x2` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.061 ± 0.045 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x4x4` | 0.001 ± 0.000 | - | 0.003 ± 0.001 | 0.075 ± 0.075 | 0.009 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `4x8x8` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.052 ± 0.036 | 0.068 ± 0.088 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x128x128` | 0.685 ± 0.262 | - | 0.660 ± 0.311 | 0.389 ± 0.128 | 0.249 ± 0.096 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x2x2` | 0.003 ± 0.004 | - | 0.003 ± 0.001 | 0.126 ± 0.130 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x32x32` | 0.020 ± 0.051 | - | 0.021 ± 0.001 | 0.095 ± 0.046 | 0.140 ± 0.098 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.096 ± 0.039 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_cholesky` | f64 | 4 | `8x8x8` | 0.003 ± 0.000 | - | 0.005 ± 0.001 | 0.071 ± 0.032 | 0.046 ± 0.066 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x2x2` | 0.354 ± 0.065 | - | 0.365 ± 0.013 | 0.461 ± 0.136 | 0.782 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x4x4` | 1.627 ± 0.035 | - | 1.611 ± 0.109 | 1.743 ± 0.292 | 3.404 ± 3.538 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1024x8x8` | 5.293 ± 0.145 | - | 5.343 ± 0.558 | 7.003 ± 5.220 | 8.009 ± 0.270 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x128x128` | 2.479 ± 0.309 | - | 2.638 ± 0.252 | 3.308 ± 0.461 | 4.034 ± 0.462 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.001 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x32x32` | 0.087 ± 0.011 | - | 0.087 ± 0.015 | 0.095 ± 0.005 | 0.167 ± 0.041 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `1x8x8` | 0.006 ± 0.000 | - | 0.007 ± 0.000 | 0.010 ± 0.000 | 0.056 ± 0.038 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x2x2` | 0.003 ± 0.001 | - | 0.004 ± 0.001 | 0.006 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x4x4` | 0.006 ± 0.000 | - | 0.009 ± 0.001 | 0.012 ± 0.004 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `3x8x8` | 0.015 ± 0.001 | - | 0.016 ± 0.004 | 0.023 ± 0.001 | 0.097 ± 0.064 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x2x2` | 0.003 ± 0.001 | - | 0.004 ± 0.000 | 0.010 ± 0.052 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x4x4` | 0.007 ± 0.000 | - | 0.008 ± 0.000 | 0.013 ± 0.003 | 0.018 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `4x8x8` | 0.020 ± 0.001 | - | 0.021 ± 0.005 | 0.029 ± 0.000 | 0.059 ± 0.239 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x128x128` | 21.358 ± 1.327 | - | 21.850 ± 0.728 | 26.853 ± 1.052 | 34.101 ± 1.519 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x2x2` | 0.004 ± 0.000 | - | 0.005 ± 0.001 | 0.010 ± 0.002 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x32x32` | 0.682 ± 0.062 | - | 0.617 ± 0.019 | 0.736 ± 0.953 | 1.122 ± 0.224 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x4x4` | 0.013 ± 0.003 | - | 0.014 ± 0.002 | 0.020 ± 0.004 | 0.026 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 1 | `8x8x8` | 0.042 ± 0.006 | - | 0.040 ± 0.005 | 0.054 ± 0.001 | 0.125 ± 0.044 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x2x2` | 0.347 ± 0.177 | - | 0.338 ± 0.072 | 0.477 ± 0.059 | 0.975 ± 0.087 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x4x4` | 1.522 ± 0.589 | - | 0.968 ± 0.270 | 1.848 ± 0.424 | 1.581 ± 0.354 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1024x8x8` | 3.338 ± 2.001 | - | 4.098 ± 2.948 | 7.268 ± 0.203 | 4.656 ± 0.663 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x128x128` | 2.711 ± 3.297 | - | 4.198 ± 2.083 | 4.019 ± 0.322 | 8.774 ± 1.922 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.001 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x32x32` | 0.079 ± 0.006 | - | 0.079 ± 0.000 | 0.095 ± 0.002 | 0.210 ± 0.114 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x4x4` | 0.004 ± 0.000 | - | 0.013 ± 0.000 | 0.006 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `1x8x8` | 0.008 ± 0.002 | - | 0.008 ± 0.000 | 0.011 ± 0.002 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x4x4` | 0.015 ± 0.009 | - | 0.019 ± 0.011 | 0.009 ± 0.000 | 0.019 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `3x8x8` | 0.016 ± 0.001 | - | 0.017 ± 0.001 | 0.024 ± 0.001 | 0.089 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.001 | 0.006 ± 0.000 | 0.016 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x4x4` | 0.016 ± 0.000 | - | 0.022 ± 0.001 | 0.011 ± 0.003 | 0.019 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `4x8x8` | 0.020 ± 0.000 | - | 0.021 ± 0.000 | 0.030 ± 0.001 | 0.088 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x128x128` | 18.234 ± 0.949 | - | 18.018 ± 3.389 | 33.365 ± 3.555 | 27.052 ± 7.990 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x2x2` | 0.004 ± 0.000 | - | 0.015 ± 0.001 | 0.008 ± 0.000 | 0.017 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x32x32` | 0.748 ± 0.124 | - | 0.554 ± 0.637 | 0.916 ± 0.175 | 0.985 ± 0.215 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x4x4` | 0.013 ± 0.001 | - | 0.034 ± 0.001 | 0.017 ± 0.004 | 0.031 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | c64 | 4 | `8x8x8` | 0.040 ± 0.008 | - | 0.045 ± 0.009 | 0.055 ± 0.001 | 0.119 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x2x2` | 0.342 ± 0.038 | - | 0.341 ± 0.049 | 0.252 ± 0.087 | 0.592 ± 0.050 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x4x4` | 1.405 ± 0.040 | - | 1.413 ± 0.076 | 1.500 ± 0.254 | 1.544 ± 0.054 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1024x8x8` | 4.669 ± 0.099 | - | 4.820 ± 0.149 | 5.178 ± 0.761 | 5.672 ± 2.323 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x128x128` | 1.575 ± 0.073 | - | 1.529 ± 0.111 | 1.117 ± 0.090 | 1.128 ± 0.042 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.004 | 0.004 ± 0.001 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x32x32` | 0.069 ± 0.009 | - | 0.065 ± 0.008 | 0.072 ± 0.003 | 0.104 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.014 ± 0.014 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `1x8x8` | 0.005 ± 0.000 | - | 0.008 ± 0.002 | 0.009 ± 0.001 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.001 | 0.005 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x4x4` | 0.005 ± 0.000 | - | 0.006 ± 0.000 | 0.028 ± 0.031 | 0.015 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `3x8x8` | 0.014 ± 0.003 | - | 0.015 ± 0.000 | 0.019 ± 0.000 | 0.026 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x4x4` | 0.007 ± 0.001 | - | 0.007 ± 0.000 | 0.010 ± 0.000 | 0.016 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `4x8x8` | 0.017 ± 0.001 | - | 0.021 ± 0.004 | 0.023 ± 0.000 | 0.032 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x128x128` | 12.410 ± 0.202 | - | 12.590 ± 0.854 | 8.349 ± 0.406 | 8.372 ± 0.076 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x2x2` | 0.004 ± 0.001 | - | 0.005 ± 0.000 | 0.006 ± 0.000 | 0.015 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x32x32` | 0.565 ± 0.106 | - | 0.501 ± 0.020 | 0.529 ± 0.018 | 1.456 ± 0.869 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x4x4` | 0.012 ± 0.003 | - | 0.012 ± 0.000 | 0.014 ± 0.001 | 0.021 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 1 | `8x8x8` | 0.034 ± 0.008 | - | 0.036 ± 0.005 | 0.043 ± 0.002 | 0.085 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x2x2` | 0.325 ± 0.074 | - | 0.303 ± 0.097 | 0.297 ± 0.123 | 0.637 ± 0.054 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x4x4` | 1.924 ± 0.714 | - | 1.726 ± 0.338 | 1.446 ± 0.289 | 1.785 ± 1.332 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1024x8x8` | 4.320 ± 1.589 | - | 3.760 ± 1.066 | 5.906 ± 0.663 | 4.155 ± 1.175 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x128x128` | 1.830 ± 1.553 | - | 1.789 ± 0.415 | 1.073 ± 0.093 | 2.865 ± 0.870 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x2x2` | 0.005 ± 0.001 | - | 0.009 ± 0.000 | 0.004 ± 0.001 | 0.012 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x32x32` | 0.062 ± 0.009 | - | 0.067 ± 0.051 | 0.099 ± 0.008 | 0.314 ± 0.087 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x4x4` | 0.003 ± 0.001 | - | 0.004 ± 0.000 | 0.006 ± 0.001 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `1x8x8` | 0.006 ± 0.000 | - | 0.018 ± 0.001 | 0.010 ± 0.020 | 0.018 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x2x2` | 0.006 ± 0.000 | - | 0.004 ± 0.000 | 0.005 ± 0.000 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x4x4` | 0.006 ± 0.001 | - | 0.008 ± 0.001 | 0.008 ± 0.000 | 0.018 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `3x8x8` | 0.013 ± 0.000 | - | 0.015 ± 0.000 | 0.019 ± 0.005 | 0.044 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x2x2` | 0.003 ± 0.000 | - | 0.011 ± 0.001 | 0.006 ± 0.000 | 0.015 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x4x4` | 0.006 ± 0.000 | - | 0.007 ± 0.002 | 0.011 ± 0.002 | 0.017 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `4x8x8` | 0.020 ± 0.004 | - | 0.019 ± 0.004 | 0.024 ± 0.000 | 0.031 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x128x128` | 9.107 ± 3.788 | - | 11.925 ± 5.366 | 8.952 ± 1.438 | 6.063 ± 1.289 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x2x2` | 0.009 ± 0.000 | - | 0.006 ± 0.001 | 0.006 ± 0.000 | 0.029 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x32x32` | 0.411 ± 0.254 | - | 0.402 ± 0.137 | 0.551 ± 0.038 | 0.563 ± 0.374 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x4x4` | 0.011 ± 0.000 | - | 0.012 ± 0.000 | 0.013 ± 0.000 | 0.023 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigh` | f64 | 4 | `8x8x8` | 0.034 ± 0.001 | - | 0.038 ± 0.005 | 0.042 ± 0.026 | 0.214 ± 0.102 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x2x2` | 0.235 ± 0.028 | - | 0.250 ± 0.023 | 0.126 ± 0.009 | 0.868 ± 0.035 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x4x4` | 1.302 ± 0.065 | - | 1.268 ± 0.280 | 0.755 ± 0.023 | 1.798 ± 0.228 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1024x8x8` | 4.327 ± 0.141 | - | 4.291 ± 0.122 | 2.419 ± 0.287 | 7.373 ± 1.708 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x128x128` | 0.861 ± 0.074 | - | 0.826 ± 0.034 | 0.487 ± 0.043 | 1.123 ± 0.031 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | 0.002 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x32x32` | 0.053 ± 0.012 | - | 0.049 ± 0.010 | 0.030 ± 0.006 | 0.103 ± 0.031 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `1x8x8` | 0.005 ± 0.000 | - | 0.006 ± 0.000 | 0.007 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x4x4` | 0.005 ± 0.000 | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.023 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `3x8x8` | 0.012 ± 0.000 | - | 0.013 ± 0.000 | 0.012 ± 0.000 | 0.023 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x4x4` | 0.005 ± 0.001 | - | 0.007 ± 0.000 | 0.008 ± 0.000 | 0.016 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `4x8x8` | 0.015 ± 0.000 | - | 0.017 ± 0.004 | 0.018 ± 0.001 | 0.032 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x128x128` | 6.724 ± 0.180 | - | 6.707 ± 0.327 | 3.828 ± 0.319 | 8.617 ± 0.809 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.006 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x32x32` | 0.446 ± 0.103 | - | 0.430 ± 0.100 | 0.198 ± 0.006 | 0.594 ± 0.061 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x4x4` | 0.009 ± 0.004 | - | 0.011 ± 0.003 | 0.010 ± 0.000 | 0.020 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 1 | `8x8x8` | 0.030 ± 0.000 | - | 0.031 ± 0.000 | 0.028 ± 0.003 | 0.120 ± 0.041 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x2x2` | 0.229 ± 0.146 | - | 0.596 ± 0.045 | 0.126 ± 0.000 | 1.596 ± 0.621 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x4x4` | 1.483 ± 0.635 | - | 2.385 ± 1.434 | 1.143 ± 0.760 | 2.421 ± 0.750 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1024x8x8` | 5.736 ± 4.402 | - | 3.975 ± 0.406 | 2.496 ± 0.084 | 8.708 ± 3.421 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x128x128` | 0.959 ± 0.375 | - | 0.910 ± 0.073 | 0.466 ± 0.014 | 2.141 ± 0.912 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x2x2` | 0.003 ± 0.000 | - | 0.002 ± 0.000 | 0.005 ± 0.000 | 0.009 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x32x32` | 0.096 ± 0.010 | - | 0.048 ± 0.006 | 0.030 ± 0.000 | 0.179 ± 0.114 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x4x4` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.007 ± 0.001 | 0.032 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `1x8x8` | 0.008 ± 0.012 | - | 0.015 ± 0.002 | 0.007 ± 0.000 | 0.014 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x2x2` | 0.002 ± 0.000 | - | 0.008 ± 0.000 | 0.005 ± 0.000 | 0.011 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x4x4` | 0.011 ± 0.004 | - | 0.011 ± 0.008 | 0.007 ± 0.000 | 0.041 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `3x8x8` | 0.020 ± 0.004 | - | 0.029 ± 0.001 | 0.012 ± 0.003 | 0.070 ± 0.095 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x2x2` | 0.005 ± 0.000 | - | 0.009 ± 0.000 | 0.005 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x4x4` | 0.008 ± 0.004 | - | 0.016 ± 0.013 | 0.008 ± 0.000 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `4x8x8` | 0.031 ± 0.000 | - | 0.017 ± 0.000 | 0.046 ± 0.037 | 0.036 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x128x128` | 6.223 ± 1.579 | - | 5.096 ± 1.008 | 3.821 ± 2.162 | 7.718 ± 9.024 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x2x2` | 0.007 ± 0.000 | - | 0.005 ± 0.000 | 0.005 ± 0.000 | 0.037 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x32x32` | 0.487 ± 0.103 | - | 0.407 ± 0.109 | 0.197 ± 0.003 | 0.414 ± 0.154 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x4x4` | 0.020 ± 0.016 | - | 0.017 ± 0.008 | 0.010 ± 0.000 | 0.164 ± 0.188 | - | - |
| cpu/linalg_batch_families | `batched_eigvalsh` | f64 | 4 | `8x8x8` | 0.031 ± 0.009 | - | 0.066 ± 0.028 | 0.051 ± 0.001 | 0.119 ± 0.117 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x2x2` | 0.109 ± 0.140 | - | 0.084 ± 0.016 | 0.156 ± 0.008 | 0.146 ± 0.056 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x4x4` | 0.404 ± 0.320 | - | 0.175 ± 0.027 | 0.236 ± 0.004 | 0.226 ± 0.037 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1024x8x8` | 0.461 ± 0.159 | - | 1.095 ± 0.537 | 0.694 ± 0.253 | 0.485 ± 0.053 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x128x128` | 0.152 ± 0.036 | - | 0.144 ± 0.032 | 0.081 ± 0.031 | 0.205 ± 0.313 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x2x2` | 0.005 ± 0.000 | - | 0.008 ± 0.000 | 0.007 ± 0.001 | 0.013 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x32x32` | 0.008 ± 0.000 | - | 0.009 ± 0.000 | 0.013 ± 0.002 | 0.056 ± 0.033 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x4x4` | 0.003 ± 0.000 | - | 0.003 ± 0.002 | 0.006 ± 0.000 | 0.010 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x64x24` | 0.011 ± 0.000 | - | 0.012 ± 0.000 | 0.013 ± 0.003 | 0.081 ± 0.069 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `1x8x8` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.007 ± 0.000 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x2x2` | 0.005 ± 0.000 | - | 0.009 ± 0.000 | 0.008 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x4x4` | 0.003 ± 0.002 | - | 0.003 ± 0.000 | 0.008 ± 0.000 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `3x8x8` | 0.003 ± 0.001 | - | 0.004 ± 0.000 | 0.009 ± 0.000 | 0.066 ± 0.070 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x2x2` | 0.006 ± 0.000 | - | 0.009 ± 0.001 | 0.007 ± 0.000 | 0.012 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.001 | 0.008 ± 0.000 | 0.014 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `4x8x8` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.060 ± 0.015 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x128x128` | 1.353 ± 0.673 | - | 1.252 ± 0.106 | 0.547 ± 0.010 | 0.959 ± 0.710 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x2x2` | 0.006 ± 0.000 | - | 0.010 ± 0.001 | 0.010 ± 0.002 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x32x32` | 0.062 ± 0.038 | - | 0.062 ± 0.025 | 0.045 ± 0.008 | 0.103 ± 0.030 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x4x4` | 0.003 ± 0.000 | - | 0.005 ± 0.002 | 0.009 ± 0.000 | 0.040 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 1 | `8x8x8` | 0.005 ± 0.000 | - | 0.006 ± 0.000 | 0.011 ± 0.003 | 0.064 ± 0.032 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x2x2` | 0.071 ± 0.009 | - | 0.080 ± 0.008 | 0.359 ± 0.092 | 0.197 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x4x4` | 0.174 ± 0.033 | - | 0.183 ± 0.045 | 0.561 ± 0.184 | 0.276 ± 0.021 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1024x8x8` | 0.512 ± 0.071 | - | 0.512 ± 0.054 | 0.887 ± 0.274 | 0.636 ± 0.247 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x128x128` | 0.272 ± 0.149 | - | 0.260 ± 0.052 | 0.222 ± 0.131 | 0.196 ± 0.122 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.202 ± 0.138 | 0.025 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x32x32` | 0.010 ± 0.002 | - | 0.010 ± 0.000 | 0.094 ± 0.073 | 0.058 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x4x4` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.161 ± 0.092 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x64x24` | 0.011 ± 0.000 | - | 0.012 ± 0.000 | 0.069 ± 0.177 | 0.078 ± 0.023 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `1x8x8` | 0.002 ± 0.000 | - | 0.010 ± 0.002 | 0.140 ± 0.054 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.001 | 0.116 ± 0.075 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x4x4` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.140 ± 0.105 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `3x8x8` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.125 ± 0.051 | 0.068 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.001 | 0.100 ± 0.056 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x4x4` | 0.003 ± 0.001 | - | 0.005 ± 0.001 | 0.103 ± 0.052 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `4x8x8` | 0.004 ± 0.000 | - | 0.006 ± 0.001 | 0.087 ± 0.070 | 0.063 ± 0.041 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x128x128` | 0.787 ± 0.084 | - | 0.752 ± 0.089 | 1.225 ± 0.247 | 0.784 ± 0.327 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.159 ± 0.107 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x32x32` | 0.059 ± 0.010 | - | 0.058 ± 0.007 | 0.235 ± 0.058 | 0.152 ± 0.052 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x4x4` | 0.003 ± 0.001 | - | 0.004 ± 0.000 | 0.180 ± 0.046 | 0.078 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_lu` | f64 | 4 | `8x8x8` | 0.005 ± 0.000 | - | 0.006 ± 0.000 | 0.138 ± 0.118 | 0.049 ± 0.058 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x2x2` | 0.158 ± 0.033 | - | 0.165 ± 0.005 | 0.280 ± 0.001 | 0.845 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x4x4` | 0.394 ± 0.040 | - | 0.409 ± 0.037 | 0.615 ± 0.021 | 1.274 ± 0.487 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1024x8x8` | 1.133 ± 0.113 | - | 1.127 ± 0.032 | 1.296 ± 1.866 | 2.312 ± 0.219 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x128x128` | 0.469 ± 0.100 | - | 0.460 ± 0.025 | 0.338 ± 0.009 | 0.497 ± 0.074 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x32x32` | 0.020 ± 0.004 | - | 0.022 ± 0.003 | 0.014 ± 0.000 | 0.133 ± 0.048 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x4x4` | 0.002 ± 0.001 | - | 0.003 ± 0.000 | 0.003 ± 0.000 | 0.035 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x64x24` | 0.027 ± 0.007 | - | 0.024 ± 0.004 | 0.018 ± 0.000 | 0.076 ± 0.065 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `1x8x8` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.012 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x4x4` | 0.003 ± 0.000 | - | 0.003 ± 0.000 | 0.005 ± 0.000 | 0.040 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `3x8x8` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.007 ± 0.000 | 0.015 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x2x2` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.004 ± 0.000 | 0.037 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.001 | 0.005 ± 0.000 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `4x8x8` | 0.005 ± 0.002 | - | 0.008 ± 0.002 | 0.009 ± 0.000 | 0.018 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x128x128` | 3.847 ± 0.509 | - | 3.653 ± 0.409 | 2.794 ± 0.217 | 3.265 ± 0.638 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.002 | 0.005 ± 0.000 | 0.039 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x32x32` | 0.164 ± 0.011 | - | 0.165 ± 0.033 | 0.092 ± 0.010 | 0.448 ± 1.518 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.001 | 0.008 ± 0.000 | 0.016 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 1 | `8x8x8` | 0.009 ± 0.000 | - | 0.010 ± 0.000 | 0.015 ± 0.000 | 0.031 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x2x2` | 0.150 ± 0.010 | - | 0.152 ± 0.002 | 0.409 ± 0.090 | 0.386 ± 0.064 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x4x4` | 0.396 ± 0.076 | - | 0.394 ± 0.402 | 0.790 ± 0.170 | 0.712 ± 0.059 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1024x8x8` | 0.880 ± 0.170 | - | 0.946 ± 0.235 | 1.952 ± 0.986 | 0.800 ± 0.024 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x128x128` | 1.253 ± 0.588 | - | 0.452 ± 0.190 | 0.667 ± 0.121 | 0.426 ± 0.031 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x2x2` | 0.004 ± 0.000 | - | 0.008 ± 0.000 | 0.083 ± 0.043 | 0.012 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x32x32` | 0.024 ± 0.010 | - | 0.070 ± 0.011 | 0.052 ± 0.016 | 0.048 ± 0.029 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x4x4` | 0.002 ± 0.000 | - | 0.003 ± 0.000 | 0.029 ± 0.026 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x64x24` | 0.069 ± 0.001 | - | 0.025 ± 0.006 | 0.052 ± 0.010 | 0.095 ± 0.028 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `1x8x8` | 0.003 ± 0.000 | - | 0.003 ± 0.000 | 0.033 ± 0.038 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x2x2` | 0.005 ± 0.000 | - | 0.009 ± 0.000 | 0.056 ± 0.018 | 0.013 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x4x4` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.036 ± 0.029 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `3x8x8` | 0.013 ± 0.000 | - | 0.016 ± 0.000 | 0.038 ± 0.042 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x2x2` | 0.005 ± 0.000 | - | 0.009 ± 0.000 | 0.109 ± 0.092 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x4x4` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.069 ± 0.016 | 0.015 ± 0.008 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `4x8x8` | 0.005 ± 0.000 | - | 0.020 ± 0.001 | 0.038 ± 0.013 | 0.017 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x128x128` | 2.600 ± 0.861 | - | 2.811 ± 2.133 | 3.778 ± 1.893 | 1.605 ± 0.036 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x2x2` | 0.007 ± 0.000 | - | 0.004 ± 0.000 | 0.086 ± 0.076 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x32x32` | 0.165 ± 0.336 | - | 0.156 ± 0.010 | 0.184 ± 0.085 | 0.162 ± 0.036 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x4x4` | 0.011 ± 0.000 | - | 0.015 ± 0.000 | 0.041 ± 0.033 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_qr` | f64 | 4 | `8x8x8` | 0.028 ± 0.000 | - | 0.031 ± 0.000 | 0.042 ± 0.006 | 0.023 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.099 ± 0.001 | - | 0.078 ± 0.001 | 0.129 ± 0.002 | 0.198 ± 0.035 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.150 ± 0.009 | - | 0.133 ± 0.001 | 0.188 ± 0.008 | 0.290 ± 0.052 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.346 ± 0.003 | - | 0.344 ± 0.002 | 0.372 ± 0.031 | 0.467 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x128x128,rhs=1` | 0.084 ± 0.000 | - | 0.088 ± 0.000 | 0.064 ± 0.001 | 0.303 ± 0.016 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x2x2,rhs=1` | 0.001 ± 0.000 | - | 0.013 ± 0.011 | 0.010 ± 0.001 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x32x32,rhs=1` | 0.005 ± 0.002 | - | 0.009 ± 0.000 | 0.013 ± 0.000 | 0.043 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x4x4,rhs=1` | 0.001 ± 0.000 | - | 0.004 ± 0.000 | 0.010 ± 0.000 | 0.009 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `1x8x8,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.009 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.002 | 0.010 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `3x8x8,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.042 ± 0.010 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.009 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.010 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `4x8x8,rhs=1` | 0.003 ± 0.000 | - | 0.006 ± 0.000 | 0.012 ± 0.003 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x128x128,rhs=1` | 0.648 ± 0.010 | - | 0.739 ± 0.065 | 0.404 ± 0.069 | 0.663 ± 0.099 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.010 ± 0.000 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x32x32,rhs=1` | 0.035 ± 0.005 | - | 0.039 ± 0.003 | 0.037 ± 0.000 | 0.088 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x4x4,rhs=1` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.011 ± 0.000 | 0.013 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 1 | `8x8x8,rhs=1` | 0.004 ± 0.000 | - | 0.007 ± 0.000 | 0.012 ± 0.000 | 0.013 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.104 ± 0.006 | - | 0.084 ± 0.006 | 0.342 ± 0.045 | 0.153 ± 0.032 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.424 ± 0.269 | - | 0.138 ± 0.004 | 0.402 ± 0.252 | 0.273 ± 0.022 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.406 ± 0.139 | - | 0.638 ± 0.233 | 0.592 ± 0.027 | 0.815 ± 0.339 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x128x128,rhs=1` | 0.144 ± 0.025 | - | 0.145 ± 0.098 | 0.088 ± 0.112 | 0.239 ± 0.097 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x2x2,rhs=1` | 0.004 ± 0.000 | - | 0.006 ± 0.013 | 0.013 ± 0.004 | 0.008 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x32x32,rhs=1` | 0.007 ± 0.001 | - | 0.009 ± 0.000 | 0.040 ± 0.002 | 0.026 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.011 ± 0.003 | 0.010 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `1x8x8,rhs=1` | 0.005 ± 0.000 | - | 0.016 ± 0.000 | 0.030 ± 0.001 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.011 ± 0.001 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x4x4,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.011 | 0.012 ± 0.002 | 0.038 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `3x8x8,rhs=1` | 0.007 ± 0.000 | - | 0.017 ± 0.000 | 0.012 ± 0.003 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.002 | 0.012 ± 0.002 | 0.011 ± 0.028 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x4x4,rhs=1` | 0.009 ± 0.006 | - | 0.026 ± 0.004 | 0.010 ± 0.002 | 0.012 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `4x8x8,rhs=1` | 0.008 ± 0.000 | - | 0.019 ± 0.008 | 0.011 ± 0.001 | 0.011 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x128x128,rhs=1` | 0.737 ± 0.140 | - | 0.879 ± 0.210 | 0.650 ± 0.159 | 1.258 ± 0.294 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x2x2,rhs=1` | 0.002 ± 0.000 | - | 0.005 ± 0.000 | 0.099 ± 0.086 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x32x32,rhs=1` | 0.035 ± 0.001 | - | 0.115 ± 0.004 | 0.171 ± 0.023 | 0.144 ± 0.019 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x4x4,rhs=1` | 0.011 ± 0.003 | - | 0.006 ± 0.000 | 0.048 ± 0.060 | 0.014 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_solve` | f64 | 4 | `8x8x8,rhs=1` | 0.012 ± 0.001 | - | 0.008 ± 0.026 | 0.090 ± 0.052 | 0.013 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x2x2` | 0.914 ± 0.585 | - | 1.119 ± 0.691 | 1.269 ± 0.066 | 6.868 ± 3.874 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x4x4` | 3.373 ± 0.691 | - | 3.179 ± 0.082 | 4.759 ± 0.810 | 6.044 ± 0.546 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1024x8x8` | 10.662 ± 1.076 | - | 11.086 ± 2.199 | 15.624 ± 1.222 | 17.198 ± 0.763 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x128x128` | 5.815 ± 1.672 | - | 5.454 ± 3.436 | 5.107 ± 0.266 | 5.216 ± 2.477 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x2x2` | 0.003 ± 0.001 | - | 0.004 ± 0.000 | 0.008 ± 0.002 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x32x32` | 0.172 ± 0.064 | - | 0.223 ± 0.206 | 0.184 ± 0.016 | 0.254 ± 0.026 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x4x4` | 0.010 ± 0.000 | - | 0.016 ± 0.000 | 0.011 ± 0.002 | 0.020 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x64x24` | 0.138 ± 0.033 | - | 0.138 ± 0.047 | 0.183 ± 0.003 | 0.283 ± 0.106 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `1x8x8` | 0.010 ± 0.001 | - | 0.011 ± 0.000 | 0.025 ± 0.005 | 0.033 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x2x2` | 0.004 ± 0.001 | - | 0.006 ± 0.001 | 0.010 ± 0.003 | 0.024 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x4x4` | 0.021 ± 0.000 | - | 0.026 ± 0.002 | 0.018 ± 0.004 | 0.030 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `3x8x8` | 0.038 ± 0.055 | - | 0.031 ± 0.003 | 0.051 ± 0.002 | 0.063 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x2x2` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.011 ± 0.003 | 0.028 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x4x4` | 0.012 ± 0.003 | - | 0.033 ± 0.001 | 0.023 ± 0.003 | 0.036 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `4x8x8` | 0.042 ± 0.011 | - | 0.043 ± 0.006 | 0.066 ± 0.001 | 0.081 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x128x128` | 51.330 ± 4.763 | - | 43.100 ± 3.030 | 40.424 ± 1.731 | 41.668 ± 0.355 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x2x2` | 0.007 ± 0.000 | - | 0.010 ± 0.002 | 0.016 ± 0.002 | 0.035 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x32x32` | 1.423 ± 0.351 | - | 1.561 ± 0.338 | 1.488 ± 0.039 | 1.692 ± 0.121 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x4x4` | 0.024 ± 0.004 | - | 0.023 ± 0.002 | 0.040 ± 0.002 | 0.059 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 1 | `8x8x8` | 0.074 ± 0.015 | - | 0.083 ± 0.007 | 0.122 ± 0.001 | 0.208 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x2x2` | 1.678 ± 1.103 | - | 0.892 ± 0.260 | 1.243 ± 0.187 | 3.799 ± 0.557 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x4x4` | 1.557 ± 0.081 | - | 1.560 ± 0.103 | 4.419 ± 0.054 | 3.967 ± 1.426 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1024x8x8` | 4.942 ± 0.528 | - | 5.030 ± 0.649 | 15.171 ± 1.195 | 13.235 ± 2.716 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x128x128` | 5.359 ± 1.117 | - | 5.200 ± 0.147 | 4.888 ± 0.148 | 6.613 ± 1.901 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.000 | 0.007 ± 0.000 | 0.017 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x32x32` | 0.152 ± 0.008 | - | 0.150 ± 0.012 | 0.194 ± 0.006 | 0.479 ± 0.148 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x4x4` | 0.004 ± 0.000 | - | 0.006 ± 0.000 | 0.018 ± 0.007 | 0.022 ± 0.014 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x64x24` | 0.132 ± 0.009 | - | 0.137 ± 0.010 | 0.224 ± 0.166 | 0.253 ± 0.048 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `1x8x8` | 0.012 ± 0.003 | - | 0.011 ± 0.003 | 0.037 ± 0.012 | 0.045 ± 0.011 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x2x2` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.009 ± 0.000 | 0.027 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x4x4` | 0.009 ± 0.000 | - | 0.027 ± 0.000 | 0.026 ± 0.011 | 0.034 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `3x8x8` | 0.026 ± 0.000 | - | 0.028 ± 0.004 | 0.087 ± 0.016 | 0.078 ± 0.017 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x2x2` | 0.005 ± 0.000 | - | 0.007 ± 0.001 | 0.011 ± 0.000 | 0.028 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x4x4` | 0.027 ± 0.000 | - | 0.033 ± 0.000 | 0.034 ± 0.007 | 0.041 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `4x8x8` | 0.034 ± 0.000 | - | 0.036 ± 0.001 | 0.102 ± 0.036 | 0.267 ± 0.298 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x128x128` | 22.507 ± 1.741 | - | 20.903 ± 6.325 | 52.398 ± 3.634 | 26.246 ± 13.605 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x2x2` | 0.007 ± 0.000 | - | 0.009 ± 0.000 | 0.016 ± 0.000 | 0.049 ± 0.038 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x32x32` | 0.827 ± 0.148 | - | 0.875 ± 0.192 | 1.437 ± 0.022 | 2.086 ± 1.068 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x4x4` | 0.049 ± 0.027 | - | 0.023 ± 0.004 | 0.055 ± 0.004 | 0.059 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | c64 | 4 | `8x8x8` | 0.068 ± 0.004 | - | 0.076 ± 0.009 | 0.122 ± 0.009 | 0.257 ± 0.060 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x2x2` | 0.578 ± 0.100 | - | 0.526 ± 0.046 | 0.572 ± 0.017 | 2.020 ± 0.048 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x4x4` | 2.411 ± 0.087 | - | 2.345 ± 0.068 | 2.855 ± 0.175 | 4.937 ± 0.166 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1024x8x8` | 8.074 ± 0.449 | - | 8.288 ± 0.448 | 11.753 ± 0.960 | 12.145 ± 0.222 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x128x128` | 1.981 ± 0.087 | - | 2.377 ± 0.298 | 1.790 ± 1.242 | 4.407 ± 2.845 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.001 | 0.007 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x32x32` | 0.115 ± 0.034 | - | 0.114 ± 0.009 | 0.127 ± 0.012 | 0.170 ± 0.034 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x4x4` | 0.004 ± 0.000 | - | 0.005 ± 0.000 | 0.009 ± 0.001 | 0.017 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x64x24` | 0.101 ± 0.008 | - | 0.103 ± 0.023 | 0.103 ± 0.011 | 0.137 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `1x8x8` | 0.008 ± 0.000 | - | 0.009 ± 0.000 | 0.017 ± 0.001 | 0.024 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x2x2` | 0.003 ± 0.000 | - | 0.004 ± 0.005 | 0.008 ± 0.000 | 0.019 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x4x4` | 0.010 ± 0.002 | - | 0.009 ± 0.000 | 0.014 ± 0.000 | 0.025 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `3x8x8` | 0.024 ± 0.005 | - | 0.023 ± 0.003 | 0.035 ± 0.004 | 0.045 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x2x2` | 0.004 ± 0.001 | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.020 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x4x4` | 0.010 ± 0.000 | - | 0.012 ± 0.003 | 0.017 ± 0.001 | 0.030 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `4x8x8` | 0.029 ± 0.003 | - | 0.030 ± 0.005 | 0.047 ± 0.008 | 0.056 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x128x128` | 22.467 ± 6.100 | - | 19.282 ± 5.750 | 12.408 ± 0.661 | 12.522 ± 0.381 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x2x2` | 0.006 ± 0.000 | - | 0.008 ± 0.002 | 0.010 ± 0.000 | 0.027 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x32x32` | 0.936 ± 0.025 | - | 0.956 ± 0.080 | 1.012 ± 0.136 | 2.490 ± 0.729 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x4x4` | 0.018 ± 0.000 | - | 0.019 ± 0.005 | 0.027 ± 0.000 | 0.050 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 1 | `8x8x8` | 0.060 ± 0.010 | - | 0.060 ± 0.009 | 0.094 ± 0.008 | 0.215 ± 0.088 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x2x2` | 0.557 ± 0.120 | - | 1.339 ± 0.606 | 0.618 ± 0.671 | 3.273 ± 0.669 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x4x4` | 2.882 ± 0.864 | - | 3.519 ± 0.949 | 2.818 ± 1.677 | 2.146 ± 0.216 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1024x8x8` | 5.024 ± 1.569 | - | 5.500 ± 0.812 | 11.011 ± 2.894 | 16.988 ± 6.009 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x128x128` | 2.385 ± 3.253 | - | 2.249 ± 0.876 | 1.535 ± 0.102 | 2.732 ± 0.913 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x2x2` | 0.002 ± 0.000 | - | 0.004 ± 0.000 | 0.007 ± 0.001 | 0.014 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x32x32` | 0.261 ± 0.023 | - | 0.125 ± 0.101 | 0.127 ± 0.001 | 0.230 ± 0.101 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x4x4` | 0.004 ± 0.017 | - | 0.015 ± 0.001 | 0.017 ± 0.023 | 0.070 ± 0.047 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x64x24` | 0.079 ± 0.012 | - | 0.087 ± 0.009 | 0.100 ± 0.002 | 0.192 ± 0.110 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `1x8x8` | 0.021 ± 0.004 | - | 0.025 ± 0.000 | 0.017 ± 0.000 | 0.025 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x2x2` | 0.003 ± 0.000 | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.019 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x4x4` | 0.010 ± 0.002 | - | 0.024 ± 0.005 | 0.046 ± 0.087 | 0.025 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `3x8x8` | 0.050 ± 0.003 | - | 0.053 ± 0.001 | 0.035 ± 0.001 | 0.045 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x2x2` | 0.004 ± 0.003 | - | 0.005 ± 0.000 | 0.008 ± 0.000 | 0.025 ± 0.007 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x4x4` | 0.014 ± 0.014 | - | 0.028 ± 0.001 | 0.048 ± 0.061 | 0.030 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `4x8x8` | 0.067 ± 0.004 | - | 0.030 ± 0.000 | 0.046 ± 0.001 | 0.057 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x128x128` | 11.516 ± 3.182 | - | 12.835 ± 3.600 | 12.093 ± 7.294 | 14.953 ± 8.783 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x2x2` | 0.016 ± 0.010 | - | 0.007 ± 0.000 | 0.010 ± 0.000 | 0.076 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x32x32` | 1.049 ± 1.122 | - | 1.241 ± 0.818 | 0.968 ± 0.014 | 1.763 ± 0.617 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x4x4` | 0.019 ± 0.000 | - | 0.019 ± 0.000 | 0.028 ± 0.001 | 0.044 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svd` | f64 | 4 | `8x8x8` | 0.058 ± 0.002 | - | 0.059 ± 0.002 | 0.085 ± 0.003 | 0.162 ± 0.067 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x2x2` | 0.362 ± 0.035 | - | unsupported | 0.231 ± 0.011 | 1.606 ± 0.244 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x4x4` | 1.647 ± 0.045 | - | unsupported | 1.419 ± 0.655 | 3.210 ± 0.615 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1024x8x8` | 4.728 ± 0.071 | - | unsupported | 4.084 ± 1.309 | 6.297 ± 0.219 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x128x128` | 1.232 ± 0.096 | - | unsupported | 0.814 ± 0.075 | 0.942 ± 0.079 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x2x2` | 0.001 ± 0.000 | - | unsupported | 0.006 ± 0.000 | 0.011 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x32x32` | 0.047 ± 0.009 | - | unsupported | 0.042 ± 0.003 | 0.056 ± 0.018 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x4x4` | 0.002 ± 0.000 | - | unsupported | 0.006 ± 0.000 | 0.012 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x64x24` | 0.091 ± 0.002 | - | unsupported | 0.036 ± 0.009 | 0.044 ± 0.006 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `1x8x8` | 0.005 ± 0.001 | - | unsupported | 0.009 ± 0.001 | 0.016 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x2x2` | 0.002 ± 0.000 | - | unsupported | 0.006 ± 0.002 | 0.015 ± 0.089 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x4x4` | 0.005 ± 0.000 | - | unsupported | 0.009 ± 0.000 | 0.017 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `3x8x8` | 0.013 ± 0.003 | - | unsupported | 0.016 ± 0.000 | 0.025 ± 0.003 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x2x2` | 0.003 ± 0.000 | - | unsupported | 0.006 ± 0.000 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x4x4` | 0.006 ± 0.000 | - | unsupported | 0.012 ± 0.003 | 0.019 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `4x8x8` | 0.021 ± 0.006 | - | unsupported | 0.020 ± 0.000 | 0.031 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x128x128` | 9.849 ± 2.858 | - | unsupported | 6.452 ± 0.489 | 7.503 ± 0.247 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x2x2` | 0.003 ± 0.000 | - | unsupported | 0.008 ± 0.001 | 0.020 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x32x32` | 0.409 ± 0.039 | - | unsupported | 0.296 ± 0.032 | 0.356 ± 0.067 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x4x4` | 0.012 ± 0.003 | - | unsupported | 0.016 ± 0.004 | 0.029 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 1 | `8x8x8` | 0.037 ± 0.004 | - | unsupported | 0.035 ± 0.006 | 0.052 ± 0.009 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x2x2` | 0.568 ± 0.492 | - | unsupported | 0.231 ± 0.011 | 1.464 ± 0.094 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x4x4` | 1.711 ± 0.204 | - | unsupported | 1.405 ± 0.046 | 1.279 ± 0.287 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1024x8x8` | 2.520 ± 0.607 | - | unsupported | 6.154 ± 0.971 | 3.787 ± 0.551 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x128x128` | 1.331 ± 0.042 | - | unsupported | 0.780 ± 0.008 | 0.795 ± 0.047 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x2x2` | 0.001 ± 0.000 | - | unsupported | 0.006 ± 0.000 | 0.010 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x32x32` | 0.050 ± 0.006 | - | unsupported | 0.074 ± 0.042 | 0.100 ± 0.198 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x4x4` | 0.003 ± 0.000 | - | unsupported | 0.006 ± 0.000 | 0.012 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x64x24` | 0.038 ± 0.007 | - | unsupported | 0.035 ± 0.001 | 0.112 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `1x8x8` | 0.005 ± 0.000 | - | unsupported | 0.010 ± 0.002 | 0.014 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x2x2` | 0.002 ± 0.000 | - | unsupported | 0.006 ± 0.000 | 0.012 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x4x4` | 0.005 ± 0.000 | - | unsupported | 0.009 ± 0.000 | 0.018 ± 0.004 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `3x8x8` | 0.014 ± 0.000 | - | unsupported | 0.016 ± 0.000 | 0.024 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x2x2` | 0.003 ± 0.000 | - | unsupported | 0.007 ± 0.000 | 0.014 ± 0.005 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x4x4` | 0.008 ± 0.002 | - | unsupported | 0.010 ± 0.000 | 0.019 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `4x8x8` | 0.022 ± 0.004 | - | unsupported | 0.020 ± 0.000 | 0.029 ± 0.001 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x128x128` | 5.120 ± 0.613 | - | unsupported | 6.634 ± 0.950 | 3.409 ± 1.488 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x2x2` | 0.004 ± 0.001 | - | unsupported | 0.006 ± 0.000 | 0.045 ± 0.058 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x32x32` | 0.248 ± 0.064 | - | unsupported | 0.307 ± 0.187 | 0.220 ± 0.013 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x4x4` | 0.012 ± 0.003 | - | unsupported | 0.015 ± 0.000 | 0.028 ± 0.002 | - | - |
| cpu/linalg_batch_families | `batched_svdvals` | f64 | 4 | `8x8x8` | 0.038 ± 0.007 | - | unsupported | 0.034 ± 0.000 | 0.048 ± 0.002 | - | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x16x16,rhs=1` | 0.741 ± 0.725 | - | unsupported | 0.846 ± 0.033 | 1.135 ± 0.039 | 4.042 ± 0.569 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x2x2,rhs=1` | 0.059 ± 0.003 | - | unsupported | 0.076 ± 0.024 | 0.252 ± 0.175 | 0.072 ± 0.010 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x4x4,rhs=1` | 0.107 ± 0.009 | - | unsupported | 0.135 ± 0.020 | 0.262 ± 0.109 | 0.161 ± 0.018 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 1 | `1024x8x8,rhs=1` | 0.215 ± 0.009 | - | unsupported | 0.305 ± 0.026 | 0.378 ± 0.059 | 0.827 ± 0.093 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x16x16,rhs=1` | 1.314 ± 3.699 | - | unsupported | 0.862 ± 0.121 | 0.700 ± 0.061 | 3.882 ± 2.247 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x2x2,rhs=1` | 0.160 ± 0.028 | - | unsupported | 0.283 ± 0.058 | 0.349 ± 0.128 | 0.084 ± 0.011 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x4x4,rhs=1` | 0.258 ± 0.005 | - | unsupported | 0.297 ± 0.124 | 0.211 ± 0.044 | 0.230 ± 0.029 | - |
| cpu/linalg_batched | `batched_lu_factor` | f64 | 4 | `1024x8x8,rhs=1` | 0.589 ± 0.009 | - | unsupported | 0.414 ± 0.066 | 0.395 ± 0.049 | 0.960 ± 0.606 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.418 ± 0.008 | - | unsupported | 0.186 ± 0.001 | 0.485 ± 0.051 | 0.508 ± 0.031 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.032 ± 0.001 | - | unsupported | 0.057 ± 0.022 | 0.082 ± 0.002 | 0.062 ± 0.023 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.042 ± 0.000 | - | unsupported | 0.070 ± 0.012 | 0.107 ± 0.037 | 0.074 ± 0.010 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.128 ± 0.002 | - | unsupported | 0.080 ± 0.000 | 0.164 ± 0.004 | 0.237 ± 0.069 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.477 ± 0.167 | - | unsupported | 0.226 ± 0.115 | 0.548 ± 0.075 | 0.519 ± 0.027 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.033 ± 0.001 | - | unsupported | 0.048 ± 0.010 | 0.088 ± 0.030 | 0.064 ± 0.009 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.044 ± 0.003 | - | unsupported | 0.053 ± 0.009 | 0.150 ± 0.066 | 0.092 ± 0.009 | - |
| cpu/linalg_batched | `batched_lu_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.140 ± 0.019 | - | unsupported | 0.082 ± 0.009 | 0.261 ± 0.044 | 0.251 ± 0.053 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x16x16,rhs=1` | 0.195 ± 0.002 | - | 0.199 ± 0.013 | 0.082 ± 0.022 | 0.185 ± 0.007 | 0.297 ± 0.029 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x2x2,rhs=1` | 0.021 ± 0.000 | - | 0.023 ± 0.017 | 0.043 ± 0.021 | 0.025 ± 0.002 | 0.037 ± 0.112 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x4x4,rhs=1` | 0.025 ± 0.000 | - | 0.028 ± 0.000 | 0.041 ± 0.017 | 0.031 ± 0.001 | 0.044 ± 0.003 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 1 | `1024x8x8,rhs=1` | 0.061 ± 0.001 | - | 0.063 ± 0.000 | 0.037 ± 0.014 | 0.057 ± 0.014 | 0.101 ± 0.010 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x16x16,rhs=1` | 0.211 ± 0.008 | - | 0.214 ± 0.020 | 0.103 ± 0.017 | 0.308 ± 0.052 | 0.241 ± 0.018 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x2x2,rhs=1` | 0.021 ± 0.001 | - | 0.030 ± 0.012 | 0.067 ± 0.068 | 0.026 ± 0.004 | 0.036 ± 0.003 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x4x4,rhs=1` | 0.026 ± 0.000 | - | 0.029 ± 0.001 | 0.033 ± 0.018 | 0.060 ± 0.006 | 0.050 ± 0.007 | - |
| cpu/linalg_batched | `batched_triangular_solve` | f64 | 4 | `1024x8x8,rhs=1` | 0.063 ± 0.002 | - | 0.069 ± 0.009 | 0.042 ± 0.007 | 0.093 ± 0.034 | 0.111 ± 0.014 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 1 | `1536x1536` | 46.083 ± 11.638 | - | 43.174 ± 3.184 | 34.407 ± 9.986 | 18.859 ± 1.133 | 50.640 ± 24.565 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 4 | `1536x1536` | 23.502 ± 1.028 | - | 27.752 ± 7.729 | 24.027 ± 7.109 | 15.994 ± 1.838 | 362.934 ± 19.042 | - |
| cpu/linalg_uncovered | `det` | f64 | 1 | `1024x1024` | 28.104 ± 2.604 | - | 24.886 ± 4.668 | 7.801 ± 3.530 | 8.766 ± 0.308 | 30.334 ± 3.767 | - |
| cpu/linalg_uncovered | `det` | f64 | 4 | `1024x1024` | 23.302 ± 3.047 | - | 21.204 ± 4.950 | 11.049 ± 10.026 | 7.923 ± 0.084 | 261.085 ± 37.063 | - |
| cpu/linalg_uncovered | `eig` | f64 | 1 | `160x160` | 7.780 ± 1.398 | - | 7.562 ± 0.719 | 13.111 ± 6.418 | 8.769 ± 0.908 | 7.619 ± 0.484 | - |
| cpu/linalg_uncovered | `eig` | f64 | 4 | `160x160` | 10.174 ± 0.665 | - | 10.365 ± 0.784 | 7.124 ± 0.372 | 9.472 ± 0.475 | 424.290 ± 275.707 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 1 | `192x192` | 8.718 ± 1.365 | - | 8.376 ± 0.294 | 12.574 ± 2.751 | 8.249 ± 0.570 | 8.448 ± 1.421 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 4 | `192x192` | 13.752 ± 2.364 | - | 12.412 ± 14.420 | 8.116 ± 0.594 | 7.822 ± 0.403 | 249.163 ± 14.011 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 1 | `512x512` | 21.498 ± 0.275 | - | 21.763 ± 4.086 | 21.637 ± 1.036 | 30.601 ± 3.684 | 15.133 ± 1.876 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 4 | `512x512` | 25.233 ± 3.990 | - | 21.061 ± 2.188 | 22.334 ± 1.221 | 31.566 ± 1.078 | 700.825 ± 213.556 | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `256x128` | 0.421 ± 0.207 | 0.577 ± 0.174 | unsupported | 0.352 ± 0.029 | 0.397 ± 0.177 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 1 | `768x384` | 7.687 ± 0.958 | 8.451 ± 1.381 | unsupported | 4.551 ± 1.377 | 5.744 ± 0.967 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `256x128` | 0.821 ± 0.075 | 0.867 ± 0.048 | unsupported | 0.346 ± 0.048 | 0.682 ± 0.173 | - | - |
| cpu/linalg_uncovered | `householder_qr` | f64 | 4 | `768x384` | 11.590 ± 0.952 | 10.378 ± 4.075 | unsupported | 4.526 ± 0.442 | 4.945 ± 0.390 | - | - |
| cpu/linalg_uncovered | `inv` | f64 | 1 | `768x768` | 35.089 ± 1.208 | - | 34.463 ± 2.325 | 12.726 ± 11.006 | 9.751 ± 0.692 | 36.727 ± 6.277 | - |
| cpu/linalg_uncovered | `inv` | f64 | 4 | `768x768` | 28.921 ± 9.370 | - | 25.404 ± 11.894 | 9.561 ± 0.223 | 10.401 ± 3.630 | 383.101 ± 49.432 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 1 | `768x384,rhs=16` | - | 18.928 ± 1.652 | 19.845 ± 0.871 | 6.250 ± 0.535 | 32.331 ± 3.558 | 8.292 ± 6.684 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 4 | `768x384,rhs=16` | - | 21.381 ± 13.374 | 17.811 ± 1.189 | 5.409 ± 2.582 | 45.302 ± 3.217 | 149.598 ± 21.769 | - |
| cpu/linalg_uncovered | `lu` | f64 | 1 | `1024x1024` | 25.126 ± 0.721 | - | 25.111 ± 1.846 | 13.773 ± 3.069 | 19.790 ± 7.768 | 30.057 ± 18.881 | - |
| cpu/linalg_uncovered | `lu` | f64 | 4 | `1024x1024` | 27.926 ± 24.726 | - | 24.308 ± 7.433 | 12.457 ± 7.697 | 12.267 ± 0.948 | 445.476 ± 275.791 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 1 | `2048x2048` | 1.117 ± 0.639 | - | 1.178 ± 0.869 | 2.495 ± 0.151 | 6.547 ± 0.444 | 5.790 ± 0.760 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 4 | `2048x2048` | 1.361 ± 0.580 | - | 0.965 ± 0.113 | 2.951 ± 0.429 | 5.426 ± 1.498 | 7.369 ± 3.300 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 1 | `512x256` | 17.465 ± 0.809 | - | 17.432 ± 0.368 | 13.727 ± 3.837 | 12.785 ± 1.001 | 23.563 ± 3.735 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 4 | `512x256` | 24.016 ± 15.361 | - | 22.780 ± 5.887 | 11.243 ± 0.119 | 13.739 ± 0.411 | 987.099 ± 88.766 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 1 | `512x256` | 17.412 ± 0.181 | - | 17.194 ± 0.169 | 12.085 ± 0.797 | 13.732 ± 0.989 | 20.404 ± 2.583 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 4 | `512x256` | 24.896 ± 2.990 | - | 23.180 ± 5.453 | 12.983 ± 5.541 | 13.684 ± 1.641 | 977.476 ± 9.397 | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `256x128` | 1.631 ± 0.234 | 1.984 ± 0.898 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 1 | `768x384` | 27.134 ± 2.113 | 30.057 ± 3.091 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `256x128` | 1.355 ± 0.037 | 1.353 ± 0.213 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr` | f64 | 4 | `768x384` | 30.087 ± 6.269 | 29.536 ± 6.689 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `256x128` | 1.654 ± 0.136 | 1.736 ± 0.258 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 1 | `768x384` | 31.933 ± 1.506 | 28.317 ± 1.898 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `256x128` | 1.317 ± 0.120 | 1.372 ± 0.070 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `rank_revealing_qr_deficient` | f64 | 4 | `768x384` | 28.678 ± 1.568 | 26.044 ± 3.523 | unsupported | unsupported | unsupported | - | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 1 | `1024x1024` | 25.783 ± 3.358 | - | 21.388 ± 0.981 | 15.609 ± 7.594 | 8.286 ± 0.209 | 28.370 ± 6.426 | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 4 | `1024x1024` | 21.696 ± 8.307 | - | 19.528 ± 6.147 | 7.292 ± 0.171 | 7.882 ± 3.893 | 218.696 ± 34.767 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 1 | `768x384` | - | 66.637 ± 4.184 | 83.855 ± 10.827 | 38.480 ± 2.420 | 41.889 ± 2.468 | 82.455 ± 11.992 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 4 | `768x384` | - | 80.966 ± 19.035 | 78.390 ± 16.977 | 48.218 ± 6.908 | 50.076 ± 11.845 | 1780.300 ± 155.337 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 1 | `4096x4096,rhs=64` | 44.631 ± 10.422 | - | 41.602 ± 3.196 | 23.909 ± 17.192 | 75.087 ± 5.396 | 35.662 ± 2.358 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 4 | `4096x4096,rhs=64` | 29.338 ± 11.098 | - | 33.051 ± 4.364 | 21.567 ± 10.586 | 53.414 ± 6.257 | 25.806 ± 5.327 | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `1024x64` | 0.243 ± 0.050 | 0.298 ± 0.112 | unsupported | 0.054 ± 0.035 | 0.034 ± 0.008 | - | - |
| cpu/neural_public_api | `erf` | f32 | 1 | `4096x256` | 2.910 ± 1.149 | 2.802 ± 1.478 | unsupported | 1.048 ± 0.129 | 0.505 ± 0.041 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `1024x64` | 0.200 ± 0.168 | 0.201 ± 0.135 | unsupported | 0.122 ± 0.012 | 0.046 ± 0.012 | - | - |
| cpu/neural_public_api | `erf` | f32 | 4 | `4096x256` | 2.231 ± 1.434 | 2.557 ± 0.841 | unsupported | 0.977 ± 0.292 | 0.274 ± 0.396 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `1024x64` | 0.080 ± 0.001 | 0.080 ± 0.001 | unsupported | 0.170 ± 0.240 | 0.182 ± 0.008 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 1 | `4096x256` | 1.276 ± 0.010 | 1.255 ± 0.007 | unsupported | 1.381 ± 0.098 | 2.944 ± 1.729 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `1024x64` | 0.129 ± 0.012 | 0.130 ± 0.006 | unsupported | 0.197 ± 0.033 | 0.267 ± 0.047 | - | - |
| cpu/neural_public_api | `gelu` | f32 | 4 | `4096x256` | 0.711 ± 0.344 | 0.919 ± 0.257 | unsupported | 1.949 ± 0.486 | 1.295 ± 0.191 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `1024x64` | 0.301 ± 0.060 | 0.267 ± 0.065 | unsupported | 0.429 ± 0.189 | 0.097 ± 0.009 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 1 | `4096x256` | 4.204 ± 0.043 | 4.797 ± 0.515 | unsupported | 4.938 ± 0.903 | 2.293 ± 2.364 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `1024x64` | 0.277 ± 0.027 | 0.291 ± 0.051 | unsupported | 0.352 ± 0.054 | 0.074 ± 0.028 | - | - |
| cpu/neural_public_api | `gelu_tanh` | f32 | 4 | `4096x256` | 2.382 ± 0.473 | 2.945 ± 0.301 | unsupported | 4.962 ± 0.928 | 0.521 ± 0.032 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `1024x64` | 0.048 ± 0.002 | 0.060 ± 0.005 | unsupported | 0.054 ± 0.002 | 0.089 ± 0.002 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 1 | `4096x256` | 0.843 ± 0.154 | 0.768 ± 0.091 | unsupported | 0.598 ± 0.029 | 0.804 ± 0.037 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `1024x64` | 0.056 ± 0.009 | 0.064 ± 0.012 | unsupported | 0.114 ± 0.004 | 0.179 ± 0.057 | - | - |
| cpu/neural_public_api | `layer_norm` | f32 | 4 | `4096x256` | 1.589 ± 0.969 | 1.350 ± 0.333 | unsupported | 0.773 ± 0.347 | 1.641 ± 1.206 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `1024x64` | 0.096 ± 0.116 | 0.104 ± 0.029 | unsupported | 0.104 ± 0.021 | 0.119 ± 0.001 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 1 | `4096x256` | 1.773 ± 0.165 | 1.826 ± 0.188 | unsupported | 1.443 ± 0.367 | 1.584 ± 0.081 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `1024x64` | 0.109 ± 0.010 | 0.124 ± 0.023 | unsupported | 0.161 ± 0.022 | 0.279 ± 0.132 | - | - |
| cpu/neural_public_api | `log_softmax` | f32 | 4 | `4096x256` | 2.480 ± 2.224 | 1.914 ± 0.603 | unsupported | 1.813 ± 0.244 | 2.050 ± 0.581 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `1024x64` | 0.125 ± 0.002 | 0.142 ± 0.019 | unsupported | unsupported | 0.135 ± 0.012 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 1 | `4096x256` | 2.332 ± 0.295 | 2.375 ± 0.294 | unsupported | unsupported | 2.103 ± 0.092 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `1024x64` | 0.144 ± 0.008 | 0.162 ± 0.011 | unsupported | unsupported | 0.183 ± 0.100 | - | - |
| cpu/neural_public_api | `masked_softmax` | f32 | 4 | `4096x256` | 3.143 ± 2.411 | 2.720 ± 1.560 | unsupported | unsupported | 3.696 ± 1.757 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `1024x64` | 0.012 ± 0.001 | 0.015 ± 0.001 | unsupported | 0.009 ± 0.001 | 0.062 ± 0.036 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 1 | `4096x256` | 0.122 ± 0.009 | 0.129 ± 0.016 | unsupported | 0.061 ± 0.011 | 0.285 ± 0.036 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `1024x64` | 0.015 ± 0.003 | 0.018 ± 0.003 | unsupported | 0.053 ± 0.021 | 0.085 ± 0.022 | - | - |
| cpu/neural_public_api | `reduce_mean` | f32 | 4 | `4096x256` | 0.249 ± 0.068 | 0.271 ± 0.223 | unsupported | 0.126 ± 0.084 | 0.160 ± 0.037 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `1024x64` | 0.025 ± 0.001 | 0.033 ± 0.001 | unsupported | 0.032 ± 0.008 | 0.072 ± 0.003 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 1 | `4096x256` | 0.397 ± 0.024 | 0.427 ± 0.032 | unsupported | 0.358 ± 0.055 | 0.900 ± 0.223 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `1024x64` | 0.028 ± 0.004 | 0.035 ± 0.037 | unsupported | 0.176 ± 0.048 | 0.082 ± 0.027 | - | - |
| cpu/neural_public_api | `rms_norm` | f32 | 4 | `4096x256` | 0.924 ± 0.500 | 1.154 ± 0.664 | unsupported | 0.853 ± 0.639 | 1.198 ± 0.211 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `1024x64` | 0.087 ± 0.036 | 0.058 ± 0.013 | unsupported | 0.111 ± 0.203 | 0.088 ± 0.014 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 1 | `4096x256` | 0.771 ± 0.954 | 0.771 ± 0.009 | unsupported | 1.762 ± 0.639 | 0.971 ± 0.070 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `1024x64` | 0.105 ± 0.039 | 0.116 ± 0.040 | unsupported | 0.153 ± 0.039 | 0.133 ± 0.024 | - | - |
| cpu/neural_public_api | `sigmoid` | f32 | 4 | `4096x256` | 0.672 ± 0.041 | 0.679 ± 0.118 | unsupported | 1.237 ± 0.434 | 0.475 ± 0.071 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `1024x64` | 0.068 ± 0.020 | 0.060 ± 0.045 | unsupported | 0.101 ± 0.046 | 0.095 ± 0.001 | - | - |
| cpu/neural_public_api | `silu` | f32 | 1 | `4096x256` | 0.807 ± 0.024 | 0.800 ± 0.013 | unsupported | 1.586 ± 0.252 | 0.923 ± 0.023 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `1024x64` | 0.131 ± 0.040 | 0.131 ± 0.038 | unsupported | 0.162 ± 0.020 | 0.141 ± 0.036 | - | - |
| cpu/neural_public_api | `silu` | f32 | 4 | `4096x256` | 0.776 ± 0.103 | 0.462 ± 0.179 | unsupported | 1.438 ± 0.470 | 0.498 ± 0.057 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `1024x64` | 0.111 ± 0.064 | 0.104 ± 0.087 | unsupported | 0.118 ± 0.020 | 0.120 ± 0.002 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 1 | `4096x256` | 1.547 ± 0.268 | 1.604 ± 0.043 | unsupported | 1.703 ± 0.102 | 1.591 ± 0.236 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `1024x64` | 0.112 ± 0.022 | 0.115 ± 0.002 | unsupported | 0.161 ± 0.015 | 0.202 ± 0.068 | - | - |
| cpu/neural_public_api | `softmax` | f32 | 4 | `4096x256` | 2.259 ± 1.282 | 1.720 ± 0.794 | unsupported | 2.219 ± 1.533 | 2.193 ± 0.644 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `1024x64` | 0.289 ± 0.210 | 0.345 ± 0.174 | unsupported | 0.433 ± 0.188 | 0.296 ± 0.010 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 1 | `4096x256` | 3.520 ± 0.188 | 3.541 ± 0.253 | unsupported | 5.226 ± 0.959 | 4.481 ± 0.737 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `1024x64` | 0.438 ± 0.268 | 0.260 ± 0.020 | unsupported | 0.318 ± 0.097 | 0.384 ± 0.119 | - | - |
| cpu/neural_public_api | `softplus` | f32 | 4 | `4096x256` | 1.905 ± 0.292 | 1.876 ± 0.258 | unsupported | 3.519 ± 0.755 | 2.649 ± 2.166 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `1024x64` | 0.555 ± 0.027 | 0.705 ± 0.173 | unsupported | 0.135 ± 0.023 | 0.110 ± 0.013 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 1 | `4096x256` | 11.590 ± 1.821 | 12.214 ± 0.642 | unsupported | 2.293 ± 0.071 | 1.979 ± 0.302 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `1024x64` | 0.947 ± 0.085 | 1.530 ± 1.035 | unsupported | 0.278 ± 0.043 | 0.073 ± 0.022 | - | - |
| cpu/neural_public_api | `take_along_axis` | f32 | 4 | `4096x256` | 9.246 ± 0.794 | 11.352 ± 0.860 | unsupported | 3.429 ± 1.141 | 1.742 ± 0.563 | - | - |
| cpu/output_reuse | `add_into` | f64 | 1 | `33554432` | 22.288 ± 7.918 | - | unsupported | 21.221 ± 9.185 | - | 20.207 ± 1.294 | 21.270 ± 1.937 |
| cpu/output_reuse | `add_into` | f64 | 4 | `33554432` | 22.126 ± 1.258 | - | unsupported | 30.353 ± 6.908 | - | 20.805 ± 2.632 | 20.222 ± 1.267 |
| cpu/output_reuse | `conj_into` | c64 | 1 | `16777216` | 17.119 ± 5.919 | - | unsupported | 14.384 ± 1.963 | - | 19.075 ± 2.937 | 17.273 ± 5.332 |
| cpu/output_reuse | `conj_into` | c64 | 4 | `16777216` | 17.012 ± 3.949 | - | unsupported | 14.704 ± 1.096 | - | 12.928 ± 0.566 | 13.347 ± 1.136 |
| cpu/output_reuse | `copy_read_into` | f64 | 1 | `33554432` | 14.443 ± 1.591 | - | unsupported | 13.795 ± 2.094 | - | 17.498 ± 10.849 | - |
| cpu/output_reuse | `copy_read_into` | f64 | 4 | `33554432` | 11.434 ± 0.699 | - | unsupported | 15.340 ± 2.110 | - | 15.478 ± 4.610 | - |
| cpu/output_reuse | `div_into` | f64 | 1 | `33554432` | 23.392 ± 13.421 | - | unsupported | 19.733 ± 1.618 | - | 34.294 ± 16.601 | 40.378 ± 32.128 |
| cpu/output_reuse | `div_into` | f64 | 4 | `33554432` | 21.311 ± 6.347 | - | unsupported | 25.914 ± 4.475 | - | 21.207 ± 1.216 | 20.788 ± 0.697 |
| cpu/output_reuse | `dot_general_read_into` | f64 | 1 | `1024x1024` | 57.057 ± 7.253 | - | unsupported | 11.644 ± 1.067 | - | 63.979 ± 10.477 | - |
| cpu/output_reuse | `dot_general_read_into` | f64 | 4 | `1024x1024` | 25.968 ± 7.013 | - | unsupported | 13.804 ± 3.361 | - | 31.898 ± 3.091 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 1 | `1024x1024` | 50.131 ± 16.764 | - | unsupported | 12.134 ± 0.198 | - | 64.836 ± 11.121 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 4 | `1024x1024` | 25.273 ± 0.830 | - | unsupported | 13.353 ± 3.319 | - | 34.513 ± 4.978 | - |
| cpu/output_reuse | `mul_into` | f64 | 1 | `33554432` | 20.129 ± 4.060 | - | unsupported | 20.405 ± 1.873 | - | 30.593 ± 10.209 | 25.185 ± 1.167 |
| cpu/output_reuse | `mul_into` | f64 | 4 | `33554432` | 24.678 ± 5.408 | - | unsupported | 24.923 ± 2.383 | - | 21.448 ± 1.653 | 21.507 ± 1.038 |
| cpu/output_reuse | `neg_into` | f64 | 1 | `33554432` | 12.900 ± 0.931 | - | unsupported | 12.842 ± 0.717 | - | 16.590 ± 2.252 | 21.892 ± 3.131 |
| cpu/output_reuse | `neg_into` | f64 | 4 | `33554432` | 16.144 ± 1.609 | - | unsupported | 14.763 ± 2.161 | - | 14.736 ± 0.801 | 14.336 ± 0.749 |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `256x256,rhs=16` | 0.530 ± 0.028 | unsupported | unsupported | 0.357 ± 0.068 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 1 | `768x768,rhs=16` | 10.035 ± 1.261 | unsupported | unsupported | 5.237 ± 1.570 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `256x256,rhs=16` | 0.858 ± 0.151 | unsupported | unsupported | 0.570 ± 0.077 | unsupported | - | - |
| cpu/output_reuse | `solve_read_into` | f64 | 4 | `768x768,rhs=16` | 13.208 ± 4.448 | unsupported | unsupported | 4.550 ± 1.343 | unsupported | - | - |
| cpu/output_reuse | `sub_into` | f64 | 1 | `33554432` | 23.991 ± 2.737 | - | unsupported | 19.083 ± 2.569 | - | 20.692 ± 0.831 | 27.791 ± 5.415 |
| cpu/output_reuse | `sub_into` | f64 | 4 | `33554432` | 31.969 ± 3.219 | - | unsupported | 23.977 ± 3.374 | - | 21.864 ± 1.134 | 20.173 ± 1.367 |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 1 | `8192x1 -> 8192x4096` | 8.624 ± 1.204 | - | 0.001 ± 0.001 | 5.859 ± 0.551 | 6.092 ± 0.161 | 15.518 ± 9.342 | - |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 4 | `8192x1 -> 8192x4096` | 12.573 ± 3.978 | - | 0.001 ± 0.000 | 6.295 ± 0.202 | 6.004 ± 0.388 | 15.944 ± 4.891 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 1 | `33554432` | 9.407 ± 1.179 | - | 10.155 ± 0.393 | 10.013 ± 0.667 | 10.574 ± 1.086 | 10.060 ± 5.375 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 4 | `33554432` | 10.935 ± 0.869 | - | 12.218 ± 1.652 | 11.193 ± 2.364 | 11.012 ± 1.753 | 12.249 ± 4.102 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 1 | `8192 -> 8192x8192` | 13.760 ± 1.069 | - | 12.672 ± 0.603 | 13.350 ± 0.848 | 18.047 ± 1.383 | 13.221 ± 4.736 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 4 | `8192 -> 8192x8192` | 16.616 ± 3.751 | - | 12.740 ± 1.598 | 15.051 ± 4.815 | 17.015 ± 3.306 | 20.922 ± 8.158 | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 1 | `8388608x2x2 -> 8388608x2` | 5.826 ± 0.100 | - | 6.172 ± 0.727 | 11.529 ± 0.769 | 9.501 ± 0.951 | - | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 4 | `8388608x2x2 -> 8388608x2` | 6.318 ± 0.277 | - | 5.963 ± 0.267 | 14.218 ± 3.379 | 10.682 ± 0.733 | - | - |
| cpu/structural_shape | `reshape` | f64 | 1 | `33554432 -> 8192x4096` | 11.552 ± 1.748 | - | 0.002 ± 0.002 | 13.303 ± 0.504 | 11.310 ± 0.750 | 13.819 ± 5.598 | - |
| cpu/structural_shape | `reshape` | f64 | 4 | `33554432 -> 8192x4096` | 14.661 ± 2.367 | - | 0.005 ± 0.008 | 14.209 ± 1.152 | 11.133 ± 0.145 | 16.451 ± 3.244 | - |
| cpu/structural_shape | `transpose` | f64 | 1 | `4096x4096` | 33.966 ± 2.158 | - | 0.001 ± 0.000 | 50.634 ± 2.068 | 56.053 ± 0.504 | 176.679 ± 42.063 | 36.549 ± 28.798 |
| cpu/structural_shape | `transpose` | f64 | 4 | `4096x4096` | 19.919 ± 2.442 | - | 0.004 ± 0.000 | 62.762 ± 9.333 | 35.805 ± 3.825 | 219.471 ± 11.862 | 24.706 ± 4.705 |
| cpu/structural_shape | `tril` | f64 | 1 | `4096x4096` | 8.802 ± 0.359 | - | 8.815 ± 0.567 | 9.393 ± 0.455 | 6.771 ± 0.519 | 10.993 ± 3.407 | - |
| cpu/structural_shape | `tril` | f64 | 4 | `4096x4096` | 6.017 ± 1.199 | - | 6.239 ± 0.525 | 16.529 ± 8.459 | 7.407 ± 1.517 | 16.950 ± 3.762 | - |
| cpu/structural_shape | `triu` | f64 | 1 | `4096x4096` | 8.386 ± 0.244 | - | 9.090 ± 0.852 | 9.002 ± 0.586 | 6.432 ± 0.306 | 10.628 ± 2.196 | - |
| cpu/structural_shape | `triu` | f64 | 4 | `4096x4096` | 6.927 ± 0.961 | - | 6.359 ± 0.508 | 10.425 ± 1.513 | 7.647 ± 2.162 | 10.630 ± 0.709 | - |
| cpu/view_metadata | `broadcast_in_dim_view` | f64 | 1 | `8192x1 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | - | - |
| cpu/view_metadata | `broadcast_in_dim_view` | f64 | 4 | `8192x1 -> 8192x4096` | 0.000 ± 0.000 | - | unsupported | 0.002 ± 0.000 | - | - | - |
| cpu/view_metadata | `reshape_view` | f64 | 1 | `33554432 -> 8192x4096` | 0.002 ± 0.002 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `reshape_view` | f64 | 4 | `33554432 -> 8192x4096` | 0.006 ± 0.004 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `slice_view` | f64 | 1 | `4194304 -> 2096128` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `slice_view` | f64 | 4 | `4194304 -> 2096128` | 0.000 ± 0.000 | - | unsupported | 0.002 ± 0.004 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `transpose_view` | f64 | 1 | `4096x4096` | 0.000 ± 0.000 | - | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `transpose_view` | f64 | 4 | `4096x4096` | 0.001 ± 0.002 | - | unsupported | 0.001 ± 0.001 | - | 0.000 ± 0.000 | - |

## Short-operation batches

Total batch duration and normalized ns/op are shown without rounding sub-microsecond calls to zero. Metadata views require no execution session.

| Operation | Dtype | Shape | Threads | Backend | Operations/batch | Median batch ms | Median ns/op |
|---|---|---|---:|---|---:|---:|---:|
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 16 | 4.075542 | 254721.38 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs direct API (ms) | 16 | 4.029042 | 251815.12 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 16 | 4.291584 | 268224.00 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 16 | 5.886209 | 367888.06 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs direct API (ms) | 16 | 9.475042 | 592190.12 |
| masked_log_softmax_axis1 | f64 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 16 | 5.530625 | 345664.06 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs direct API (ms) | 16 | 12.897541 | 806096.31 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 1 | tenferro-rs eager mode (ms) | 16 | 13.228542 | 826783.88 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs direct API (ms) | 16 | 19.863375 | 1241460.94 |
| reduce_sum_squares_axis0 | f64 | `2048x2048` | 4 | tenferro-rs eager mode (ms) | 16 | 15.642750 | 977671.88 |
| householder_qr | f64 | `256x128` | 1 | JAX Python (XLA CPU) (ms) | 16 | 6.344875 | 396554.69 |
| householder_qr | f64 | `256x128` | 1 | PyTorch Python (ms) | 16 | 5.627583 | 351723.94 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 6.728917 | 420557.31 |
| householder_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 9.229833 | 576864.56 |
| householder_qr | f64 | `768x384` | 1 | JAX Python (XLA CPU) (ms) | 16 | 91.900458 | 5743778.62 |
| householder_qr | f64 | `768x384` | 1 | PyTorch Python (ms) | 16 | 72.811542 | 4550721.38 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 122.997875 | 7687367.19 |
| householder_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 135.218416 | 8451151.00 |
| householder_qr | f64 | `256x128` | 4 | JAX Python (XLA CPU) (ms) | 16 | 10.909250 | 681828.12 |
| householder_qr | f64 | `256x128` | 4 | PyTorch Python (ms) | 16 | 5.535000 | 345937.50 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 13.141042 | 821315.12 |
| householder_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 13.865291 | 866580.69 |
| householder_qr | f64 | `768x384` | 4 | JAX Python (XLA CPU) (ms) | 16 | 79.114292 | 4944643.25 |
| householder_qr | f64 | `768x384` | 4 | PyTorch Python (ms) | 16 | 72.418125 | 4526132.81 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 185.437458 | 11589841.12 |
| householder_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 166.046209 | 10377888.06 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 26.089833 | 1630614.56 |
| rank_revealing_qr | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 31.743333 | 1983958.31 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 434.142500 | 27133906.25 |
| rank_revealing_qr | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 480.906083 | 30056630.19 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 21.676375 | 1354773.44 |
| rank_revealing_qr | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 21.652625 | 1353289.06 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 481.385334 | 30086583.38 |
| rank_revealing_qr | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 472.572583 | 29535786.44 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs direct API (ms) | 16 | 26.468292 | 1654268.25 |
| rank_revealing_qr_deficient | f64 | `256x128` | 1 | tenferro-rs eager mode (ms) | 16 | 27.775750 | 1735984.38 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs direct API (ms) | 16 | 510.925792 | 31932862.00 |
| rank_revealing_qr_deficient | f64 | `768x384` | 1 | tenferro-rs eager mode (ms) | 16 | 453.072750 | 28317046.88 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs direct API (ms) | 16 | 21.069334 | 1316833.38 |
| rank_revealing_qr_deficient | f64 | `256x128` | 4 | tenferro-rs eager mode (ms) | 16 | 21.944709 | 1371544.31 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs direct API (ms) | 16 | 458.843417 | 28677713.56 |
| rank_revealing_qr_deficient | f64 | `768x384` | 4 | tenferro-rs eager mode (ms) | 16 | 416.697666 | 26043604.12 |
| erf | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 2.203750 | 34433.59 |
| erf | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 3.459333 | 54052.08 |
| erf | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 15.563416 | 243178.38 |
| erf | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 19.060458 | 297819.66 |
| erf | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 4.039958 | 504994.75 |
| erf | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 8.385334 | 1048166.75 |
| erf | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 23.277708 | 2909713.50 |
| erf | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 22.413125 | 2801640.62 |
| erf | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 2.931750 | 45808.59 |
| erf | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 7.806500 | 121976.56 |
| erf | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 12.786666 | 199791.66 |
| erf | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 12.851167 | 200799.48 |
| erf | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 2.191916 | 273989.50 |
| erf | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 7.817375 | 977171.88 |
| erf | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 17.848042 | 2231005.25 |
| erf | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 20.458458 | 2557307.25 |
| gelu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 11.632083 | 181751.30 |
| gelu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 10.850292 | 169535.81 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 5.109583 | 79837.23 |
| gelu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 5.118375 | 79974.61 |
| gelu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 23.548833 | 2943604.12 |
| gelu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 11.045250 | 1380656.25 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 10.205334 | 1275666.75 |
| gelu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 10.038458 | 1254807.25 |
| gelu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 17.077500 | 266835.94 |
| gelu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 12.601084 | 196891.94 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 8.229000 | 128578.12 |
| gelu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 8.334750 | 130230.47 |
| gelu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 10.363791 | 1295473.88 |
| gelu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 15.589000 | 1948625.00 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 5.691833 | 711479.12 |
| gelu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 7.354041 | 919255.12 |
| gelu_tanh | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 6.205459 | 96960.30 |
| gelu_tanh | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 27.427500 | 428554.69 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 19.236791 | 300574.86 |
| gelu_tanh | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 17.058042 | 266531.91 |
| gelu_tanh | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 18.341958 | 2292744.75 |
| gelu_tanh | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 39.500958 | 4937619.75 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 33.633167 | 4204145.88 |
| gelu_tanh | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 38.378375 | 4797296.88 |
| gelu_tanh | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 4.714791 | 73668.61 |
| gelu_tanh | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 22.529666 | 352026.03 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 17.753500 | 277398.44 |
| gelu_tanh | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 18.645458 | 291335.28 |
| gelu_tanh | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 4.170834 | 521354.25 |
| gelu_tanh | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 39.693916 | 4961739.50 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 19.052834 | 2381604.25 |
| gelu_tanh | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 23.562542 | 2945317.75 |
| layer_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 5.692833 | 88950.52 |
| layer_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 3.438167 | 53721.36 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 3.040542 | 47508.47 |
| layer_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 3.827166 | 59799.47 |
| layer_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 6.429917 | 803739.62 |
| layer_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 4.783333 | 597916.62 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 6.743875 | 842984.38 |
| layer_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 6.143084 | 767885.50 |
| layer_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 11.458334 | 179036.47 |
| layer_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 7.283750 | 113808.59 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 3.578417 | 55912.77 |
| layer_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 4.078084 | 63720.06 |
| layer_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 13.126667 | 1640833.38 |
| layer_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 6.183375 | 772921.88 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 12.713250 | 1589156.25 |
| layer_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 10.803334 | 1350416.75 |
| log_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 7.591416 | 118615.88 |
| log_softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 6.645084 | 103829.44 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 6.118125 | 95595.70 |
| log_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 6.659125 | 104048.83 |
| log_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 12.671250 | 1583906.25 |
| log_softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 11.540709 | 1442588.62 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 14.185708 | 1773213.50 |
| log_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 14.610542 | 1826317.75 |
| log_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 17.827458 | 278554.03 |
| log_softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 10.332250 | 161441.41 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 6.953333 | 108645.83 |
| log_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 7.953333 | 124270.83 |
| log_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 16.401791 | 2050223.88 |
| log_softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 14.505458 | 1813182.25 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 19.840708 | 2480088.50 |
| log_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 15.309584 | 1913698.00 |
| masked_softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 8.637083 | 134954.42 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 8.004750 | 125074.22 |
| masked_softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 9.098250 | 142160.16 |
| masked_softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 16.827708 | 2103463.50 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 18.656584 | 2332073.00 |
| masked_softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 19.000250 | 2375031.25 |
| masked_softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 11.688875 | 182638.67 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 9.188583 | 143571.61 |
| masked_softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 10.397625 | 162462.89 |
| masked_softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 29.569000 | 3696125.00 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 25.145084 | 3143135.50 |
| masked_softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 21.758708 | 2719838.50 |
| reduce_mean | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 3.974416 | 62100.25 |
| reduce_mean | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 0.558375 | 8724.61 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 0.780459 | 12194.67 |
| reduce_mean | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 0.983541 | 15367.83 |
| reduce_mean | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 2.277834 | 284729.25 |
| reduce_mean | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 0.490500 | 61312.50 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 0.975500 | 121937.50 |
| reduce_mean | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 1.032000 | 129000.00 |
| reduce_mean | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 5.438875 | 84982.42 |
| reduce_mean | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 3.396083 | 53063.80 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 0.972458 | 15194.66 |
| reduce_mean | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 1.122292 | 17535.81 |
| reduce_mean | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 1.282875 | 160359.38 |
| reduce_mean | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 1.007459 | 125932.38 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 1.993375 | 249171.88 |
| reduce_mean | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 2.169084 | 271135.50 |
| rms_norm | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 4.578041 | 71531.89 |
| rms_norm | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 2.049208 | 32018.88 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 1.611542 | 25180.34 |
| rms_norm | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 2.083250 | 32550.78 |
| rms_norm | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 7.201167 | 900145.88 |
| rms_norm | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 2.867500 | 358437.50 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 3.179083 | 397385.38 |
| rms_norm | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 3.412208 | 426526.00 |
| rms_norm | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 5.225042 | 81641.28 |
| rms_norm | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 11.236458 | 175569.66 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 1.772125 | 27689.45 |
| rms_norm | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 2.254834 | 35231.78 |
| rms_norm | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 9.586792 | 1198349.00 |
| rms_norm | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 6.821834 | 852729.25 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 7.392208 | 924026.00 |
| rms_norm | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 9.233875 | 1154234.38 |
| sigmoid | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 5.617167 | 87768.23 |
| sigmoid | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 7.095834 | 110872.41 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 5.565625 | 86962.89 |
| sigmoid | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 3.712875 | 58013.67 |
| sigmoid | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 7.765042 | 970630.25 |
| sigmoid | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 14.098083 | 1762260.38 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 6.164041 | 770505.12 |
| sigmoid | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 6.164166 | 770520.75 |
| sigmoid | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 8.481334 | 132520.84 |
| sigmoid | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 9.800334 | 153130.22 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 6.723792 | 105059.25 |
| sigmoid | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 7.418250 | 115910.16 |
| sigmoid | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 3.801750 | 475218.75 |
| sigmoid | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 9.895625 | 1236953.12 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 5.379208 | 672401.00 |
| sigmoid | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 5.432458 | 679057.25 |
| silu | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 6.085792 | 95090.50 |
| silu | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 6.459417 | 100928.39 |
| silu | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 4.348458 | 67944.66 |
| silu | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 3.840708 | 60011.06 |
| silu | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 7.387333 | 923416.62 |
| silu | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 12.690625 | 1586328.12 |
| silu | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 6.453625 | 806703.12 |
| silu | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 6.401333 | 800166.62 |
| silu | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 9.033709 | 141151.70 |
| silu | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 10.392583 | 162384.11 |
| silu | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 8.408833 | 131388.02 |
| silu | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 8.404417 | 131319.02 |
| silu | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 3.987042 | 498380.25 |
| silu | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 11.504458 | 1438057.25 |
| silu | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 6.210250 | 776281.25 |
| silu | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 3.696000 | 462000.00 |
| softmax | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 7.662458 | 119725.91 |
| softmax | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 7.543916 | 117873.69 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 7.114291 | 111160.80 |
| softmax | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 6.667083 | 104173.17 |
| softmax | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 12.724375 | 1590546.88 |
| softmax | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 13.620250 | 1702531.25 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 12.376416 | 1547052.00 |
| softmax | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 12.830500 | 1603812.50 |
| softmax | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 12.936292 | 202129.56 |
| softmax | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 10.277959 | 160593.11 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 7.190875 | 112357.42 |
| softmax | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 7.372375 | 115193.36 |
| softmax | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 17.543458 | 2192932.25 |
| softmax | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 17.751333 | 2218916.62 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 18.074209 | 2259276.12 |
| softmax | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 13.763500 | 1720437.50 |
| softplus | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 18.953750 | 296152.34 |
| softplus | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 27.736625 | 433384.77 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 18.480625 | 288759.77 |
| softplus | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 22.077875 | 344966.80 |
| softplus | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 35.844042 | 4480505.25 |
| softplus | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 41.807500 | 5225937.50 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 28.157000 | 3519625.00 |
| softplus | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 28.331333 | 3541416.62 |
| softplus | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 24.599042 | 384360.03 |
| softplus | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 20.371000 | 318296.88 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 28.045666 | 438213.53 |
| softplus | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 16.659917 | 260311.20 |
| softplus | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 21.188042 | 2648505.25 |
| softplus | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 28.148667 | 3518583.38 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 15.243333 | 1905416.62 |
| softplus | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 15.006292 | 1875786.50 |
| take_along_axis | f32 | `1024x64` | 1 | JAX Python (XLA CPU) (ms) | 64 | 7.028542 | 109820.97 |
| take_along_axis | f32 | `1024x64` | 1 | PyTorch Python (ms) | 64 | 8.669375 | 135458.98 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs direct API (ms) | 64 | 35.549167 | 555455.73 |
| take_along_axis | f32 | `1024x64` | 1 | tenferro-rs eager mode (ms) | 64 | 45.111208 | 704862.62 |
| take_along_axis | f32 | `4096x256` | 1 | JAX Python (XLA CPU) (ms) | 8 | 15.832917 | 1979114.62 |
| take_along_axis | f32 | `4096x256` | 1 | PyTorch Python (ms) | 8 | 18.346625 | 2293328.12 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs direct API (ms) | 8 | 92.717542 | 11589692.75 |
| take_along_axis | f32 | `4096x256` | 1 | tenferro-rs eager mode (ms) | 8 | 97.715833 | 12214479.12 |
| take_along_axis | f32 | `1024x64` | 4 | JAX Python (XLA CPU) (ms) | 64 | 4.666458 | 72913.41 |
| take_along_axis | f32 | `1024x64` | 4 | PyTorch Python (ms) | 64 | 17.786709 | 277917.33 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs direct API (ms) | 64 | 60.607750 | 946996.09 |
| take_along_axis | f32 | `1024x64` | 4 | tenferro-rs eager mode (ms) | 64 | 97.902000 | 1529718.75 |
| take_along_axis | f32 | `4096x256` | 4 | JAX Python (XLA CPU) (ms) | 8 | 13.932334 | 1741541.75 |
| take_along_axis | f32 | `4096x256` | 4 | PyTorch Python (ms) | 8 | 27.429792 | 3428724.00 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs direct API (ms) | 8 | 73.967083 | 9245885.38 |
| take_along_axis | f32 | `4096x256` | 4 | tenferro-rs eager mode (ms) | 8 | 90.818000 | 11352250.00 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | PyTorch Python (ms) | 16 | 5.705250 | 356578.12 |
| solve_read_into | f64 | `256x256,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 8.480458 | 530028.62 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | PyTorch Python (ms) | 16 | 83.795166 | 5237197.88 |
| solve_read_into | f64 | `768x768,rhs=16` | 1 | tenferro-rs direct API (ms) | 16 | 160.556958 | 10034809.88 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | PyTorch Python (ms) | 16 | 9.113500 | 569593.75 |
| solve_read_into | f64 | `256x256,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 13.726459 | 857903.69 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | PyTorch Python (ms) | 16 | 72.807333 | 4550458.31 |
| solve_read_into | f64 | `768x768,rhs=16` | 4 | tenferro-rs direct API (ms) | 16 | 211.335625 | 13208476.56 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.013291 | 830.69 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.001333 | 83.31 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.033333 | 2083.31 |
| broadcast_in_dim_view | f64 | `8192x1 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.001292 | 80.75 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000084 | 5.25 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | PyTorch Python (ms) | 16 | 0.011833 | 739.56 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.030875 | 1929.69 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000209 | 13.06 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | PyTorch Python (ms) | 16 | 0.012083 | 755.19 |
| reshape_view | f64 | `33554432 -> 8192x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.097875 | 6117.19 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000083 | 5.19 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | PyTorch Python (ms) | 16 | 0.011625 | 726.56 |
| slice_view | f64 | `4194304 -> 2096128` | 1 | tenferro-rs direct API (ms) | 16 | 0.003459 | 216.19 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000125 | 7.81 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | PyTorch Python (ms) | 16 | 0.029375 | 1835.94 |
| slice_view | f64 | `4194304 -> 2096128` | 4 | tenferro-rs direct API (ms) | 16 | 0.004583 | 286.44 |
| transpose_view | f64 | `4096x4096` | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000084 | 5.25 |
| transpose_view | f64 | `4096x4096` | 1 | PyTorch Python (ms) | 16 | 0.013375 | 835.94 |
| transpose_view | f64 | `4096x4096` | 1 | tenferro-rs direct API (ms) | 16 | 0.003125 | 195.31 |
| transpose_view | f64 | `4096x4096` | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000084 | 5.25 |
| transpose_view | f64 | `4096x4096` | 4 | PyTorch Python (ms) | 16 | 0.013959 | 872.44 |
| transpose_view | f64 | `4096x4096` | 4 | tenferro-rs direct API (ms) | 16 | 0.019750 | 1234.38 |

## Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `cpu/complex/cholesky` (c64, threads=4, shape=`448x448`): `jax-cpu` is 54.6x faster than the slowest successful cell (`julia-base`, 80.410 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/cholesky` (c64, threads=4, shape=`448x448`): `pytorch-cpu` is 16.6x faster than the slowest successful cell (`julia-base`, 80.410 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/cholesky` (c64, threads=4, shape=`448x448`): `tenferro-direct` is 16.8x faster than the slowest successful cell (`julia-base`, 80.410 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/cholesky` (c64, threads=4, shape=`448x448`): `tenferro-trace` is 18.3x faster than the slowest successful cell (`julia-base`, 80.410 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/eig` (c64, threads=4, shape=`112x112`): `jax-cpu` is 23.0x faster than the slowest successful cell (`julia-base`, 468.299 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/eig` (c64, threads=4, shape=`112x112`): `pytorch-cpu` is 32.2x faster than the slowest successful cell (`julia-base`, 468.299 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/eig` (c64, threads=4, shape=`112x112`): `tenferro-direct` is 43.8x faster than the slowest successful cell (`julia-base`, 468.299 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/eig` (c64, threads=4, shape=`112x112`): `tenferro-trace` is 23.2x faster than the slowest successful cell (`julia-base`, 468.299 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `jax-cpu` is 26.9x faster than the slowest successful cell (`julia-base`, 185.894 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `pytorch-cpu` is 22.8x faster than the slowest successful cell (`julia-base`, 185.894 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `tenferro-direct` is 26.1x faster than the slowest successful cell (`julia-base`, 185.894 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/qr` (c64, threads=4, shape=`256x256`): `tenferro-trace` is 24.1x faster than the slowest successful cell (`julia-base`, 185.894 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `jax-cpu` is 61.4x faster than the slowest successful cell (`julia-base`, 152.055 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `pytorch-cpu` is 22.4x faster than the slowest successful cell (`julia-base`, 152.055 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `tenferro-direct` is 25.6x faster than the slowest successful cell (`julia-base`, 152.055 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/solve` (c64, threads=4, shape=`384x384,rhs=8`): `tenferro-trace` is 14.4x faster than the slowest successful cell (`julia-base`, 152.055 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/svd` (c64, threads=4, shape=`160x160`): `jax-cpu` is 96.5x faster than the slowest successful cell (`julia-base`, 640.955 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/svd` (c64, threads=4, shape=`160x160`): `pytorch-cpu` is 77.8x faster than the slowest successful cell (`julia-base`, 640.955 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/svd` (c64, threads=4, shape=`160x160`): `tenferro-direct` is 79.4x faster than the slowest successful cell (`julia-base`, 640.955 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/svd` (c64, threads=4, shape=`160x160`): `tenferro-trace` is 76.2x faster than the slowest successful cell (`julia-base`, 640.955 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 14.7x faster than the slowest successful cell (`pytorch-cpu`, 16.592 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 17.0x faster than the slowest successful cell (`pytorch-cpu`, 16.592 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 15.8x faster than the slowest successful cell (`pytorch-cpu`, 16.592 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 14.8x faster than the slowest successful cell (`pytorch-cpu`, 13.241 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 13.4x faster than the slowest successful cell (`pytorch-cpu`, 13.241 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-trace` is 12.1x faster than the slowest successful cell (`pytorch-cpu`, 13.241 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 16.9x faster than the slowest successful cell (`pytorch-cpu`, 16.655 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 15.9x faster than the slowest successful cell (`pytorch-cpu`, 16.655 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 17.7x faster than the slowest successful cell (`pytorch-cpu`, 16.655 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 17.1x faster than the slowest successful cell (`pytorch-cpu`, 13.745 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-direct` is 13.7x faster than the slowest successful cell (`pytorch-cpu`, 13.745 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_min_axis0` (f64, threads=4, shape=`2048x2048`): `tenferro-trace` is 14.7x faster than the slowest successful cell (`pytorch-cpu`, 13.745 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=1, shape=`4194304 -> 2096128`): `tenferro-trace` is 1413.7x faster than the slowest successful cell (`julia-base`, 2.120 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`): `tenferro-trace` is 1147.9x faster than the slowest successful cell (`pytorch-cpu`, 1.627 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x32x32`): `pytorch-cpu` is 10.5x faster than the slowest successful cell (`jax-cpu`, 0.054 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x32x32`): `tenferro-direct` is 18.3x faster than the slowest successful cell (`jax-cpu`, 0.054 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`1x32x32`): `tenferro-trace` is 12.9x faster than the slowest successful cell (`jax-cpu`, 0.054 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x8x8`): `pytorch-cpu` is 17.9x faster than the slowest successful cell (`jax-cpu`, 0.079 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`4x8x8`): `tenferro-direct` is 19.5x faster than the slowest successful cell (`jax-cpu`, 0.079 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`8x8x8`): `pytorch-cpu` is 12.9x faster than the slowest successful cell (`jax-cpu`, 0.068 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=1, shape=`8x8x8`): `tenferro-direct` is 11.5x faster than the slowest successful cell (`jax-cpu`, 0.068 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x2x2`): `tenferro-direct` is 19.9x faster than the slowest successful cell (`pytorch-cpu`, 0.051 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x2x2`): `tenferro-trace` is 22.4x faster than the slowest successful cell (`pytorch-cpu`, 0.051 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x32x32`): `tenferro-direct` is 10.7x faster than the slowest successful cell (`pytorch-cpu`, 0.036 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x4x4`): `jax-cpu` is 16.1x faster than the slowest successful cell (`pytorch-cpu`, 0.131 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x4x4`): `tenferro-direct` is 126.1x faster than the slowest successful cell (`pytorch-cpu`, 0.131 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x4x4`): `tenferro-trace` is 54.4x faster than the slowest successful cell (`pytorch-cpu`, 0.131 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x8x8`): `jax-cpu` is 14.0x faster than the slowest successful cell (`pytorch-cpu`, 0.116 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x8x8`): `tenferro-direct` is 99.2x faster than the slowest successful cell (`pytorch-cpu`, 0.116 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`1x8x8`): `tenferro-trace` is 47.9x faster than the slowest successful cell (`pytorch-cpu`, 0.116 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x2x2`): `tenferro-direct` is 36.7x faster than the slowest successful cell (`pytorch-cpu`, 0.040 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x2x2`): `tenferro-trace` is 17.0x faster than the slowest successful cell (`pytorch-cpu`, 0.040 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x4x4`): `jax-cpu` is 18.5x faster than the slowest successful cell (`pytorch-cpu`, 0.155 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x4x4`): `tenferro-direct` is 95.6x faster than the slowest successful cell (`pytorch-cpu`, 0.155 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x4x4`): `tenferro-trace` is 50.4x faster than the slowest successful cell (`pytorch-cpu`, 0.155 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x8x8`): `jax-cpu` is 10.9x faster than the slowest successful cell (`pytorch-cpu`, 0.094 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x8x8`): `tenferro-direct` is 48.1x faster than the slowest successful cell (`pytorch-cpu`, 0.094 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`3x8x8`): `tenferro-trace` is 32.3x faster than the slowest successful cell (`pytorch-cpu`, 0.094 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x2x2`): `tenferro-direct` is 21.1x faster than the slowest successful cell (`pytorch-cpu`, 0.061 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x4x4`): `tenferro-direct` is 56.0x faster than the slowest successful cell (`pytorch-cpu`, 0.075 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x4x4`): `tenferro-trace` is 26.3x faster than the slowest successful cell (`pytorch-cpu`, 0.075 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x8x8`): `tenferro-direct` is 37.2x faster than the slowest successful cell (`jax-cpu`, 0.068 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`4x8x8`): `tenferro-trace` is 21.5x faster than the slowest successful cell (`jax-cpu`, 0.068 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x2x2`): `jax-cpu` is 14.6x faster than the slowest successful cell (`pytorch-cpu`, 0.126 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x2x2`): `tenferro-direct` is 37.2x faster than the slowest successful cell (`pytorch-cpu`, 0.126 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x2x2`): `tenferro-trace` is 43.1x faster than the slowest successful cell (`pytorch-cpu`, 0.126 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x4x4`): `jax-cpu` is 10.9x faster than the slowest successful cell (`pytorch-cpu`, 0.096 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x4x4`): `tenferro-direct` is 46.0x faster than the slowest successful cell (`pytorch-cpu`, 0.096 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x4x4`): `tenferro-trace` is 25.9x faster than the slowest successful cell (`pytorch-cpu`, 0.096 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x8x8`): `tenferro-direct` is 25.1x faster than the slowest successful cell (`pytorch-cpu`, 0.071 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_cholesky` (f64, threads=4, shape=`8x8x8`): `tenferro-trace` is 15.4x faster than the slowest successful cell (`pytorch-cpu`, 0.071 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigvalsh` (f64, threads=4, shape=`1024x2x2`): `pytorch-cpu` is 12.7x faster than the slowest successful cell (`jax-cpu`, 1.596 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigvalsh` (f64, threads=4, shape=`1x4x4`): `tenferro-direct` is 15.6x faster than the slowest successful cell (`jax-cpu`, 0.032 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_eigvalsh` (f64, threads=4, shape=`8x4x4`): `pytorch-cpu` is 15.8x faster than the slowest successful cell (`jax-cpu`, 0.164 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`3x8x8`): `tenferro-direct` is 20.3x faster than the slowest successful cell (`jax-cpu`, 0.066 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`3x8x8`): `tenferro-trace` is 15.5x faster than the slowest successful cell (`jax-cpu`, 0.066 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`4x8x8`): `tenferro-direct` is 17.1x faster than the slowest successful cell (`jax-cpu`, 0.060 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`4x8x8`): `tenferro-trace` is 12.8x faster than the slowest successful cell (`jax-cpu`, 0.060 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`8x4x4`): `tenferro-direct` is 12.6x faster than the slowest successful cell (`jax-cpu`, 0.040 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=1, shape=`8x8x8`): `tenferro-direct` is 12.1x faster than the slowest successful cell (`jax-cpu`, 0.064 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x2x2`): `tenferro-direct` is 97.0x faster than the slowest successful cell (`pytorch-cpu`, 0.202 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x2x2`): `tenferro-trace` is 52.7x faster than the slowest successful cell (`pytorch-cpu`, 0.202 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x4x4`): `jax-cpu` is 16.2x faster than the slowest successful cell (`pytorch-cpu`, 0.161 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x4x4`): `tenferro-direct` is 75.7x faster than the slowest successful cell (`pytorch-cpu`, 0.161 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x4x4`): `tenferro-trace` is 50.8x faster than the slowest successful cell (`pytorch-cpu`, 0.161 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x8x8`): `jax-cpu` is 13.7x faster than the slowest successful cell (`pytorch-cpu`, 0.140 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x8x8`): `tenferro-direct` is 57.9x faster than the slowest successful cell (`pytorch-cpu`, 0.140 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`1x8x8`): `tenferro-trace` is 14.4x faster than the slowest successful cell (`pytorch-cpu`, 0.140 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x2x2`): `jax-cpu` is 11.6x faster than the slowest successful cell (`pytorch-cpu`, 0.116 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x2x2`): `tenferro-direct` is 51.8x faster than the slowest successful cell (`pytorch-cpu`, 0.116 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x2x2`): `tenferro-trace` is 34.5x faster than the slowest successful cell (`pytorch-cpu`, 0.116 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x4x4`): `jax-cpu` is 12.8x faster than the slowest successful cell (`pytorch-cpu`, 0.140 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x4x4`): `tenferro-direct` is 58.0x faster than the slowest successful cell (`pytorch-cpu`, 0.140 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x4x4`): `tenferro-trace` is 40.5x faster than the slowest successful cell (`pytorch-cpu`, 0.140 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x8x8`): `tenferro-direct` is 38.6x faster than the slowest successful cell (`pytorch-cpu`, 0.125 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`3x8x8`): `tenferro-trace` is 27.6x faster than the slowest successful cell (`pytorch-cpu`, 0.125 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x2x2`): `tenferro-direct` is 43.6x faster than the slowest successful cell (`pytorch-cpu`, 0.100 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x2x2`): `tenferro-trace` is 25.2x faster than the slowest successful cell (`pytorch-cpu`, 0.100 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x4x4`): `tenferro-direct` is 32.1x faster than the slowest successful cell (`pytorch-cpu`, 0.103 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x4x4`): `tenferro-trace` is 22.4x faster than the slowest successful cell (`pytorch-cpu`, 0.103 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x8x8`): `tenferro-direct` is 24.2x faster than the slowest successful cell (`pytorch-cpu`, 0.087 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`4x8x8`): `tenferro-trace` is 15.0x faster than the slowest successful cell (`pytorch-cpu`, 0.087 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x2x2`): `jax-cpu` is 14.7x faster than the slowest successful cell (`pytorch-cpu`, 0.159 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x2x2`): `tenferro-direct` is 62.7x faster than the slowest successful cell (`pytorch-cpu`, 0.159 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x2x2`): `tenferro-trace` is 44.5x faster than the slowest successful cell (`pytorch-cpu`, 0.159 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x4x4`): `tenferro-direct` is 54.0x faster than the slowest successful cell (`pytorch-cpu`, 0.180 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x4x4`): `tenferro-trace` is 42.3x faster than the slowest successful cell (`pytorch-cpu`, 0.180 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x8x8`): `tenferro-direct` is 26.2x faster than the slowest successful cell (`pytorch-cpu`, 0.138 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_lu` (f64, threads=4, shape=`8x8x8`): `tenferro-trace` is 21.7x faster than the slowest successful cell (`pytorch-cpu`, 0.138 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=1, shape=`1x4x4`): `pytorch-cpu` is 10.4x faster than the slowest successful cell (`jax-cpu`, 0.035 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=1, shape=`1x4x4`): `tenferro-direct` is 17.6x faster than the slowest successful cell (`jax-cpu`, 0.035 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=1, shape=`1x4x4`): `tenferro-trace` is 12.8x faster than the slowest successful cell (`jax-cpu`, 0.035 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=1, shape=`3x4x4`): `tenferro-direct` is 13.8x faster than the slowest successful cell (`jax-cpu`, 0.040 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=1, shape=`3x4x4`): `tenferro-trace` is 11.4x faster than the slowest successful cell (`jax-cpu`, 0.040 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=1, shape=`4x2x2`): `tenferro-direct` is 16.1x faster than the slowest successful cell (`jax-cpu`, 0.037 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=1, shape=`4x2x2`): `tenferro-trace` is 12.3x faster than the slowest successful cell (`jax-cpu`, 0.037 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=1, shape=`8x2x2`): `tenferro-direct` is 16.0x faster than the slowest successful cell (`jax-cpu`, 0.039 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`1x2x2`): `tenferro-direct` is 19.3x faster than the slowest successful cell (`pytorch-cpu`, 0.083 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`1x2x2`): `tenferro-trace` is 10.4x faster than the slowest successful cell (`pytorch-cpu`, 0.083 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`1x4x4`): `tenferro-direct` is 15.6x faster than the slowest successful cell (`pytorch-cpu`, 0.029 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`1x8x8`): `tenferro-direct` is 13.1x faster than the slowest successful cell (`pytorch-cpu`, 0.033 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`3x2x2`): `tenferro-direct` is 11.2x faster than the slowest successful cell (`pytorch-cpu`, 0.056 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`3x4x4`): `tenferro-direct` is 14.5x faster than the slowest successful cell (`pytorch-cpu`, 0.036 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`4x2x2`): `tenferro-direct` is 21.5x faster than the slowest successful cell (`pytorch-cpu`, 0.109 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`4x2x2`): `tenferro-trace` is 12.2x faster than the slowest successful cell (`pytorch-cpu`, 0.109 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`4x4x4`): `tenferro-direct` is 23.9x faster than the slowest successful cell (`pytorch-cpu`, 0.069 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`4x4x4`): `tenferro-trace` is 17.5x faster than the slowest successful cell (`pytorch-cpu`, 0.069 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`8x2x2`): `tenferro-direct` is 13.0x faster than the slowest successful cell (`pytorch-cpu`, 0.086 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_qr` (f64, threads=4, shape=`8x2x2`): `tenferro-trace` is 22.8x faster than the slowest successful cell (`pytorch-cpu`, 0.086 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=1, shape=`3x8x8,rhs=1`): `tenferro-direct` is 18.4x faster than the slowest successful cell (`jax-cpu`, 0.042 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`3x4x4,rhs=1`): `tenferro-direct` is 19.6x faster than the slowest successful cell (`jax-cpu`, 0.038 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x2x2,rhs=1`): `tenferro-direct` is 41.1x faster than the slowest successful cell (`pytorch-cpu`, 0.099 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x2x2,rhs=1`): `tenferro-trace` is 18.6x faster than the slowest successful cell (`pytorch-cpu`, 0.099 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_solve` (f64, threads=4, shape=`8x8x8,rhs=1`): `tenferro-trace` is 11.6x faster than the slowest successful cell (`pytorch-cpu`, 0.090 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_svd` (f64, threads=4, shape=`1x4x4`): `tenferro-direct` is 16.4x faster than the slowest successful cell (`jax-cpu`, 0.070 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_svd` (f64, threads=4, shape=`8x2x2`): `tenferro-trace` is 10.6x faster than the slowest successful cell (`jax-cpu`, 0.076 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_batch_families/batched_svdvals` (f64, threads=4, shape=`8x2x2`): `tenferro-direct` is 11.3x faster than the slowest successful cell (`jax-cpu`, 0.045 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/cholesky` (f64, threads=4, shape=`1536x1536`): `jax-cpu` is 22.7x faster than the slowest successful cell (`julia-base`, 362.934 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/cholesky` (f64, threads=4, shape=`1536x1536`): `pytorch-cpu` is 15.1x faster than the slowest successful cell (`julia-base`, 362.934 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/cholesky` (f64, threads=4, shape=`1536x1536`): `tenferro-direct` is 15.4x faster than the slowest successful cell (`julia-base`, 362.934 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/cholesky` (f64, threads=4, shape=`1536x1536`): `tenferro-trace` is 13.1x faster than the slowest successful cell (`julia-base`, 362.934 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `jax-cpu` is 33.0x faster than the slowest successful cell (`julia-base`, 261.085 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `pytorch-cpu` is 23.6x faster than the slowest successful cell (`julia-base`, 261.085 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `tenferro-direct` is 11.2x faster than the slowest successful cell (`julia-base`, 261.085 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/det` (f64, threads=4, shape=`1024x1024`): `tenferro-trace` is 12.3x faster than the slowest successful cell (`julia-base`, 261.085 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eig` (f64, threads=4, shape=`160x160`): `jax-cpu` is 44.8x faster than the slowest successful cell (`julia-base`, 424.290 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eig` (f64, threads=4, shape=`160x160`): `pytorch-cpu` is 59.6x faster than the slowest successful cell (`julia-base`, 424.290 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eig` (f64, threads=4, shape=`160x160`): `tenferro-direct` is 41.7x faster than the slowest successful cell (`julia-base`, 424.290 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eig` (f64, threads=4, shape=`160x160`): `tenferro-trace` is 40.9x faster than the slowest successful cell (`julia-base`, 424.290 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvals` (f64, threads=4, shape=`192x192`): `jax-cpu` is 31.9x faster than the slowest successful cell (`julia-base`, 249.163 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvals` (f64, threads=4, shape=`192x192`): `pytorch-cpu` is 30.7x faster than the slowest successful cell (`julia-base`, 249.163 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvals` (f64, threads=4, shape=`192x192`): `tenferro-direct` is 18.1x faster than the slowest successful cell (`julia-base`, 249.163 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvals` (f64, threads=4, shape=`192x192`): `tenferro-trace` is 20.1x faster than the slowest successful cell (`julia-base`, 249.163 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvalsh` (f64, threads=4, shape=`512x512`): `jax-cpu` is 22.2x faster than the slowest successful cell (`julia-base`, 700.825 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvalsh` (f64, threads=4, shape=`512x512`): `pytorch-cpu` is 31.4x faster than the slowest successful cell (`julia-base`, 700.825 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvalsh` (f64, threads=4, shape=`512x512`): `tenferro-direct` is 27.8x faster than the slowest successful cell (`julia-base`, 700.825 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/eigvalsh` (f64, threads=4, shape=`512x512`): `tenferro-trace` is 33.3x faster than the slowest successful cell (`julia-base`, 700.825 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `jax-cpu` is 36.8x faster than the slowest successful cell (`julia-base`, 383.101 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `pytorch-cpu` is 40.1x faster than the slowest successful cell (`julia-base`, 383.101 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `tenferro-direct` is 13.2x faster than the slowest successful cell (`julia-base`, 383.101 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/inv` (f64, threads=4, shape=`768x768`): `tenferro-trace` is 15.1x faster than the slowest successful cell (`julia-base`, 383.101 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lstsq` (f64, threads=4, shape=`768x384,rhs=16`): `pytorch-cpu` is 27.7x faster than the slowest successful cell (`julia-base`, 149.598 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `jax-cpu` is 36.3x faster than the slowest successful cell (`julia-base`, 445.476 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `pytorch-cpu` is 35.8x faster than the slowest successful cell (`julia-base`, 445.476 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `tenferro-direct` is 16.0x faster than the slowest successful cell (`julia-base`, 445.476 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/lu` (f64, threads=4, shape=`1024x1024`): `tenferro-trace` is 18.3x faster than the slowest successful cell (`julia-base`, 445.476 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `jax-cpu` is 71.8x faster than the slowest successful cell (`julia-base`, 987.099 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `pytorch-cpu` is 87.8x faster than the slowest successful cell (`julia-base`, 987.099 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `tenferro-direct` is 41.1x faster than the slowest successful cell (`julia-base`, 987.099 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv` (f64, threads=4, shape=`512x256`): `tenferro-trace` is 43.3x faster than the slowest successful cell (`julia-base`, 987.099 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `jax-cpu` is 71.4x faster than the slowest successful cell (`julia-base`, 977.476 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `pytorch-cpu` is 75.3x faster than the slowest successful cell (`julia-base`, 977.476 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `tenferro-direct` is 39.3x faster than the slowest successful cell (`julia-base`, 977.476 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/pinv_with_rtol` (f64, threads=4, shape=`512x256`): `tenferro-trace` is 42.2x faster than the slowest successful cell (`julia-base`, 977.476 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `jax-cpu` is 27.7x faster than the slowest successful cell (`julia-base`, 218.696 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `pytorch-cpu` is 30.0x faster than the slowest successful cell (`julia-base`, 218.696 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `tenferro-direct` is 10.1x faster than the slowest successful cell (`julia-base`, 218.696 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/slogdet` (f64, threads=4, shape=`1024x1024`): `tenferro-trace` is 11.2x faster than the slowest successful cell (`julia-base`, 218.696 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `jax-cpu` is 35.6x faster than the slowest successful cell (`julia-base`, 1780.300 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `pytorch-cpu` is 36.9x faster than the slowest successful cell (`julia-base`, 1780.300 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `tenferro-eager` is 22.0x faster than the slowest successful cell (`julia-base`, 1780.300 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/svd_full` (f64, threads=4, shape=`768x384`): `tenferro-trace` is 22.7x faster than the slowest successful cell (`julia-base`, 1780.300 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/neural_public_api/take_along_axis` (f32, threads=4, shape=`1024x64`): `jax-cpu` is 21.0x faster than the slowest successful cell (`tenferro-eager`, 1.530 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=1, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 10951.1x faster than the slowest successful cell (`julia-base`, 15.518 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=4, shape=`8192x1 -> 8192x4096`): `tenferro-trace` is 11595.9x faster than the slowest successful cell (`julia-base`, 15.944 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=1, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 8729.3x faster than the slowest successful cell (`julia-base`, 13.819 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/reshape` (f64, threads=4, shape=`33554432 -> 8192x4096`): `tenferro-trace` is 3525.0x faster than the slowest successful cell (`julia-base`, 16.451 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`): `tenferro-trace` is 128494.0x faster than the slowest successful cell (`julia-base`, 176.679 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`): `tenferro-direct` is 11.0x faster than the slowest successful cell (`julia-base`, 219.471 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`): `tenferro-trace` is 62705.9x faster than the slowest successful cell (`julia-base`, 219.471 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/broadcast_in_dim_view` (f64, threads=1, shape=`8192x1 -> 8192x4096`): `tenferro-direct` is 10.0x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/broadcast_in_dim_view` (f64, threads=4, shape=`8192x1 -> 8192x4096`): `tenferro-direct` is 25.7x faster than the slowest successful cell (`pytorch-cpu`, 0.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=1, shape=`33554432 -> 8192x4096`): `julia-base` is 386.0x faster than the slowest successful cell (`tenferro-direct`, 0.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=4, shape=`33554432 -> 8192x4096`): `julia-base` is 470.5x faster than the slowest successful cell (`tenferro-direct`, 0.006 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=1, shape=`4194304 -> 2096128`): `julia-base` is 145.4x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=4, shape=`4194304 -> 2096128`): `julia-base` is 229.5x faster than the slowest successful cell (`pytorch-cpu`, 0.002 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=1, shape=`4096x4096`): `julia-base` is 167.2x faster than the slowest successful cell (`pytorch-cpu`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=4, shape=`4096x4096`): `julia-base` is 246.8x faster than the slowest successful cell (`tenferro-direct`, 0.001 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

## Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

- `cpu/indexing_layout/slice` (f64, threads=1, shape=`4194304 -> 2096128`), `tenferro-trace`: 16769024 minimum logical bytes imply 11179.3 GB/s at 0.002 ms. The conservative physical lower bound is 0.168 ms, so the cell is 111.8x faster than the bound.
- `cpu/indexing_layout/slice` (f64, threads=4, shape=`4194304 -> 2096128`), `tenferro-trace`: 16769024 minimum logical bytes imply 11834.2 GB/s at 0.001 ms. The conservative physical lower bound is 0.042 ms, so the cell is 29.6x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=1, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 189439.3 GB/s at 0.001 ms. The conservative physical lower bound is 2.684 ms, so the cell is 1894.4x faster than the bound.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=4, shape=`8192x1 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 195225.8 GB/s at 0.001 ms. The conservative physical lower bound is 0.671 ms, so the cell is 488.1x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=1, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 169573.9 GB/s at 0.002 ms. The conservative physical lower bound is 2.684 ms, so the cell is 1695.7x faster than the bound.
- `cpu/structural_shape/reshape` (f64, threads=4, shape=`33554432 -> 8192x4096`), `tenferro-trace`: 268435456 minimum logical bytes imply 57517.8 GB/s at 0.005 ms. The conservative physical lower bound is 0.671 ms, so the cell is 143.8x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 97612.9 GB/s at 0.001 ms. The conservative physical lower bound is 1.342 ms, so the cell is 976.1x faster than the bound.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`), `tenferro-trace`: 134217728 minimum logical bytes imply 38347.9 GB/s at 0.004 ms. The conservative physical lower bound is 0.336 ms, so the cell is 95.9x faster than the bound.
