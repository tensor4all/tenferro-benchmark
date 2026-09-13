# CPU Public API Benchmark Results

- Suite: `cpu/public_api`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/public_api.yaml`
- Public API coverage manifest: `benchmarks/cpu/public_api_coverage.yaml`
- Run metadata: `data/results/mac-cpu/cpu/public_api/20260913_154011/run.yaml`
- Timestamp: `20260913_154011`

Latest run: `PUBLICATION_GATE_PROFILE=full ./scripts/run_cpu_public_api.sh 1 4`.

- Sampling: `full` profile, 15 measured runs, 3 warmups per row

This file is generated from sequential CPU public API runs under `data/results/mac-cpu/cpu/public_api/20260913_154011`.

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

## Thread Environments

### Threads: 1

- Run metadata: `data/results/mac-cpu/cpu/public_api/20260913_154011/run_t1.yaml`
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

- Run metadata: `data/results/mac-cpu/cpu/public_api/20260913_154011/run_t4.yaml`
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

- CSV: `data/results/mac-cpu/cpu/public_api/20260913_154011/cpu_public_api_t1_20260913_154011.csv`
- CSV: `data/results/mac-cpu/cpu/public_api/20260913_154011/cpu_public_api_t4_20260913_154011.csv`
- Source table: `data/results/mac-cpu/cpu/public_api/20260913_154011/cpu_public_api_20260913_154011.md`

## Julia view batching correction

The Julia view rows were recollected at 16 calls per interval using benchmark commit 4c63ae9. Correction metadata and raw rows: `data/results/mac-cpu/cpu/public_api/20260913_155000`. All other rows retain the full run above. The source manifest records CSV replacement order; original raw data remain unchanged.

## CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs eager mode (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| cpu/complex | `cholesky` | c64 | 1 | `448x448` | 2.419 ± 0.049 | - | 2.463 ± 0.049 | 2.779 ± 0.064 | 0.969 ± 0.020 | 2.841 ± 0.420 | - |
| cpu/complex | `cholesky` | c64 | 4 | `448x448` | 2.433 ± 0.025 | - | 2.437 ± 0.051 | 2.479 ± 0.040 | 0.759 ± 0.023 | 1.293 ± 0.425 | - |
| cpu/complex | `conj` | c64 | 1 | `16777216` | 11.264 ± 0.205 | - | 9.610 ± 2.942 | 4.550 ± 0.027 | 4.478 ± 0.093 | 7.222 ± 25.609 | 7.234 ± 28.271 |
| cpu/complex | `conj` | c64 | 4 | `16777216` | 2.941 ± 0.137 | - | 2.931 ± 0.082 | 1.573 ± 0.036 | 1.564 ± 0.014 | 12.818 ± 8.637 | 10.315 ± 7.506 |
| cpu/complex | `div` | c64 | 1 | `8388608` | 4.484 ± 0.033 | - | 4.567 ± 0.107 | 7.987 ± 0.221 | 23.104 ± 0.042 | 18.640 ± 3.275 | 18.115 ± 2.871 |
| cpu/complex | `div` | c64 | 4 | `8388608` | 1.578 ± 0.039 | - | 1.783 ± 0.098 | 2.821 ± 0.064 | 7.666 ± 0.014 | 19.424 ± 1.919 | 7.173 ± 2.223 |
| cpu/complex | `dot_general` | c64 | 1 | `640x640` | 5.776 ± 1.079 | - | 5.902 ± 0.159 | 5.559 ± 0.184 | 36.668 ± 0.063 | 39.953 ± 0.389 | - |
| cpu/complex | `dot_general` | c64 | 4 | `640x640` | 5.779 ± 0.778 | - | 5.774 ± 0.178 | 5.613 ± 0.119 | 9.449 ± 0.225 | 10.698 ± 0.282 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 1 | `640x640` | 5.947 ± 0.158 | - | unsupported | 5.614 ± 0.219 | 36.814 ± 0.030 | 40.035 ± 0.540 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 4 | `640x640` | 5.761 ± 0.182 | - | unsupported | 5.628 ± 0.173 | 9.937 ± 0.263 | 10.922 ± 0.634 | - |
| cpu/complex | `eig` | c64 | 1 | `112x112` | 5.916 ± 0.056 | - | 5.833 ± 0.073 | 5.885 ± 0.061 | 9.988 ± 0.046 | 5.159 ± 0.028 | - |
| cpu/complex | `eig` | c64 | 4 | `112x112` | 5.930 ± 0.054 | - | 5.933 ± 0.034 | 5.881 ± 0.061 | 9.936 ± 0.016 | 4.908 ± 0.034 | - |
| cpu/complex | `exp` | c64 | 1 | `4194304` | 19.878 ± 0.461 | - | 19.978 ± 0.420 | 19.955 ± 0.284 | 24.021 ± 0.109 | 26.939 ± 3.178 | 25.292 ± 2.705 |
| cpu/complex | `exp` | c64 | 4 | `4194304` | 5.427 ± 0.845 | - | 5.301 ± 0.120 | 5.945 ± 0.081 | 6.530 ± 0.132 | 25.142 ± 3.001 | 6.589 ± 1.985 |
| cpu/complex | `log` | c64 | 1 | `4194304` | 23.761 ± 0.350 | - | 23.749 ± 0.074 | 25.685 ± 0.095 | 49.695 ± 0.079 | 41.047 ± 8.771 | 34.577 ± 1.543 |
| cpu/complex | `log` | c64 | 4 | `4194304` | 6.725 ± 0.231 | - | 6.723 ± 0.191 | 7.071 ± 0.323 | 13.833 ± 0.153 | 35.269 ± 0.214 | 9.909 ± 2.340 |
| cpu/complex | `mul` | c64 | 1 | `8388608` | 4.170 ± 0.335 | - | 3.969 ± 0.067 | 4.235 ± 0.045 | 3.550 ± 0.091 | 6.102 ± 6.726 | 3.489 ± 2.997 |
| cpu/complex | `mul` | c64 | 4 | `8388608` | 1.356 ± 0.107 | - | 1.331 ± 0.019 | 1.450 ± 0.019 | 1.300 ± 0.077 | 5.506 ± 8.667 | 3.584 ± 7.648 |
| cpu/complex | `norm_fro` | c64 | 1 | `2048x1536` | 4.829 ± 0.061 | - | 4.007 ± 0.028 | 3.653 ± 0.064 | 1.978 ± 0.079 | 5.682 ± 0.262 | - |
| cpu/complex | `norm_fro` | c64 | 4 | `2048x1536` | 2.072 ± 0.172 | - | 1.380 ± 0.177 | 0.942 ± 0.012 | 0.714 ± 0.029 | 5.917 ± 0.071 | - |
| cpu/complex | `qr` | c64 | 1 | `256x256` | 4.708 ± 0.036 | - | 4.798 ± 0.036 | 5.016 ± 0.042 | 3.642 ± 0.013 | 6.023 ± 0.058 | - |
| cpu/complex | `qr` | c64 | 4 | `256x256` | 4.795 ± 0.031 | - | 4.806 ± 0.034 | 4.835 ± 0.056 | 3.607 ± 0.029 | 2.764 ± 0.115 | - |
| cpu/complex | `solve` | c64 | 1 | `384x384,rhs=8` | 3.289 ± 0.039 | - | 3.290 ± 0.022 | 3.143 ± 0.038 | 1.425 ± 0.017 | 3.655 ± 0.057 | - |
| cpu/complex | `solve` | c64 | 4 | `384x384,rhs=8` | 3.537 ± 0.061 | - | 3.500 ± 0.128 | 3.309 ± 0.109 | 1.309 ± 0.014 | 1.507 ± 0.098 | - |
| cpu/complex | `svd` | c64 | 1 | `160x160` | 3.939 ± 0.061 | - | 3.885 ± 0.041 | 3.890 ± 0.065 | 2.731 ± 0.076 | 4.423 ± 0.066 | - |
| cpu/complex | `svd` | c64 | 4 | `160x160` | 3.981 ± 0.041 | - | 4.015 ± 0.058 | 3.897 ± 0.041 | 2.750 ± 0.046 | 2.906 ± 0.123 | - |
| cpu/complex | `tensordot` | c64 | 1 | `640x640` | 5.665 ± 0.208 | - | 5.900 ± 0.208 | 5.625 ± 0.212 | 36.671 ± 0.040 | 39.669 ± 0.091 | - |
| cpu/complex | `tensordot` | c64 | 4 | `640x640` | 5.617 ± 0.231 | - | 5.618 ± 0.183 | 5.515 ± 0.175 | 9.734 ± 0.156 | 10.665 ± 0.252 | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 1 | `1024x1024` | 4.926 ± 0.972 | - | unsupported | 4.500 ± 0.064 | 32.782 ± 0.069 | - | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 4 | `1024x1024` | 5.782 ± 0.981 | - | unsupported | 4.821 ± 0.244 | 8.962 ± 0.264 | - | - |
| cpu/elementwise_reduction | `abs` | f64 | 1 | `33554432` | 4.348 ± 0.011 | - | 4.404 ± 0.075 | 4.353 ± 0.047 | 8.777 ± 1.708 | 4.364 ± 0.474 | 8.068 ± 1.023 |
| cpu/elementwise_reduction | `abs` | f64 | 4 | `33554432` | 1.560 ± 0.026 | - | 1.563 ± 0.017 | 1.541 ± 0.019 | 2.481 ± 0.032 | 4.366 ± 2.025 | 2.801 ± 2.054 |
| cpu/elementwise_reduction | `add` | f64 | 1 | `33554432` | 6.515 ± 0.103 | - | 6.542 ± 0.074 | 6.666 ± 0.285 | 11.674 ± 0.219 | 9.181 ± 28.291 | 11.086 ± 13.416 |
| cpu/elementwise_reduction | `add` | f64 | 4 | `33554432` | 2.576 ± 0.138 | - | 2.605 ± 0.154 | 2.482 ± 0.023 | 3.408 ± 0.127 | 8.537 ± 8.095 | 5.582 ± 10.227 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 1 | `4194304` | 16.689 ± 0.058 | - | 16.732 ± 0.089 | 17.942 ± 0.121 | 17.913 ± 0.066 | 20.997 ± 0.086 | 21.204 ± 0.394 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 4 | `4194304` | 4.776 ± 0.035 | - | 4.775 ± 0.048 | 5.229 ± 0.337 | 4.879 ± 0.117 | 20.893 ± 0.207 | 5.876 ± 0.632 |
| cpu/elementwise_reduction | `clamp` | f64 | 1 | `8388608` | 2.825 ± 0.049 | - | 2.923 ± 0.131 | 2.221 ± 0.037 | 3.775 ± 0.101 | 2.355 ± 0.577 | 3.193 ± 0.124 |
| cpu/elementwise_reduction | `clamp` | f64 | 4 | `8388608` | 1.543 ± 0.058 | - | 1.549 ± 0.053 | 0.884 ± 0.015 | 1.762 ± 0.021 | 2.322 ± 2.772 | 1.205 ± 0.630 |
| cpu/elementwise_reduction | `compare_lt` | f64 | 1 | `33554432` | 5.155 ± 0.045 | - | 5.213 ± 0.081 | 8.522 ± 0.136 | 8.681 ± 0.050 | 6.268 ± 0.223 | - |
| cpu/elementwise_reduction | `compare_lt` | f64 | 4 | `33554432` | 2.049 ± 0.040 | - | 2.055 ± 0.024 | 2.895 ± 0.025 | 2.718 ± 0.034 | 5.745 ± 0.028 | - |
| cpu/elementwise_reduction | `cos` | f64 | 1 | `8388608` | 15.984 ± 0.985 | - | 15.704 ± 1.117 | 16.989 ± 0.063 | 12.353 ± 0.169 | 20.855 ± 0.862 | 21.167 ± 1.654 |
| cpu/elementwise_reduction | `cos` | f64 | 4 | `8388608` | 4.113 ± 0.146 | - | 4.092 ± 0.183 | 4.556 ± 0.171 | 3.298 ± 0.074 | 20.642 ± 0.277 | 5.299 ± 0.382 |
| cpu/elementwise_reduction | `div` | f64 | 1 | `33554432` | 6.538 ± 0.103 | - | 6.688 ± 0.272 | 6.567 ± 0.156 | 11.614 ± 0.123 | 6.557 ± 2.886 | 8.814 ± 2.577 |
| cpu/elementwise_reduction | `div` | f64 | 4 | `33554432` | 2.514 ± 0.107 | - | 2.538 ± 0.075 | 2.487 ± 0.048 | 3.438 ± 0.085 | 6.546 ± 2.012 | 3.246 ± 2.204 |
| cpu/elementwise_reduction | `exp` | f64 | 1 | `8388608` | 16.728 ± 0.153 | - | 16.727 ± 0.321 | 12.226 ± 0.036 | 10.358 ± 0.074 | 17.276 ± 0.905 | 17.422 ± 0.450 |
| cpu/elementwise_reduction | `exp` | f64 | 4 | `8388608` | 4.087 ± 0.048 | - | 4.101 ± 0.025 | 4.387 ± 0.017 | 2.687 ± 0.113 | 17.565 ± 0.118 | 5.952 ± 0.719 |
| cpu/elementwise_reduction | `expm1` | f64 | 1 | `4194304` | 8.553 ± 0.183 | - | 8.600 ± 0.778 | 14.655 ± 0.054 | 10.974 ± 0.073 | 12.721 ± 0.536 | 12.657 ± 0.120 |
| cpu/elementwise_reduction | `expm1` | f64 | 4 | `4194304` | 2.174 ± 0.028 | - | 2.154 ± 0.015 | 3.923 ± 0.143 | 2.912 ± 0.080 | 12.624 ± 1.864 | 3.235 ± 0.186 |
| cpu/elementwise_reduction | `log` | f64 | 1 | `8388608` | 15.286 ± 0.154 | - | 15.301 ± 0.179 | 20.775 ± 0.057 | 14.139 ± 0.054 | 22.808 ± 0.251 | 22.803 ± 0.272 |
| cpu/elementwise_reduction | `log` | f64 | 4 | `8388608` | 4.049 ± 0.054 | - | 4.120 ± 0.083 | 5.428 ± 0.695 | 3.968 ± 0.177 | 22.896 ± 1.607 | 5.890 ± 0.633 |
| cpu/elementwise_reduction | `log1p` | f64 | 1 | `4194304` | 10.231 ± 0.040 | - | 10.231 ± 0.056 | 10.956 ± 0.041 | 9.928 ± 0.043 | 12.193 ± 0.093 | 12.231 ± 0.099 |
| cpu/elementwise_reduction | `log1p` | f64 | 4 | `4194304` | 2.723 ± 0.075 | - | 2.726 ± 0.130 | 2.944 ± 0.097 | 2.804 ± 0.148 | 12.188 ± 0.336 | 3.243 ± 0.380 |
| cpu/elementwise_reduction | `maximum` | f64 | 1 | `33554432` | 6.563 ± 0.158 | - | 6.540 ± 0.099 | 6.501 ± 0.101 | 11.484 ± 0.110 | 6.485 ± 0.185 | 8.427 ± 0.570 |
| cpu/elementwise_reduction | `maximum` | f64 | 4 | `33554432` | 2.535 ± 0.113 | - | 2.517 ± 0.066 | 2.478 ± 0.010 | 3.441 ± 0.092 | 6.512 ± 2.040 | 3.146 ± 2.486 |
| cpu/elementwise_reduction | `minimum` | f64 | 1 | `33554432` | 6.549 ± 0.021 | - | 6.630 ± 0.096 | 6.508 ± 0.057 | 11.498 ± 0.105 | 6.498 ± 2.959 | 8.953 ± 1.464 |
| cpu/elementwise_reduction | `minimum` | f64 | 4 | `33554432` | 2.526 ± 0.018 | - | 2.524 ± 0.011 | 2.479 ± 0.018 | 3.376 ± 0.046 | 6.541 ± 2.014 | 3.219 ± 2.461 |
| cpu/elementwise_reduction | `mul` | f64 | 1 | `33554432` | 6.679 ± 0.092 | - | 6.731 ± 0.080 | 6.611 ± 0.180 | 11.504 ± 0.112 | 6.622 ± 2.503 | 8.596 ± 3.409 |
| cpu/elementwise_reduction | `mul` | f64 | 4 | `33554432` | 2.731 ± 0.035 | - | 2.763 ± 0.111 | 2.491 ± 0.048 | 3.377 ± 0.048 | 6.694 ± 8.731 | 3.356 ± 2.330 |
| cpu/elementwise_reduction | `neg` | f64 | 1 | `33554432` | 4.365 ± 0.038 | - | 4.359 ± 0.041 | 4.342 ± 0.055 | 8.777 ± 1.198 | 4.355 ± 2.788 | 7.901 ± 1.394 |
| cpu/elementwise_reduction | `neg` | f64 | 4 | `33554432` | 1.560 ± 0.017 | - | 1.582 ± 0.022 | 1.542 ± 0.029 | 2.495 ± 0.025 | 4.438 ± 2.021 | 2.830 ± 3.032 |
| cpu/elementwise_reduction | `pow` | f64 | 1 | `4194304` | 17.646 ± 0.053 | - | 17.658 ± 0.106 | 38.182 ± 0.091 | 17.616 ± 0.053 | 32.826 ± 0.695 | 31.869 ± 0.372 |
| cpu/elementwise_reduction | `pow` | f64 | 4 | `4194304` | 5.060 ± 0.508 | - | 4.861 ± 0.328 | 11.503 ± 1.780 | 5.120 ± 0.181 | 31.697 ± 1.370 | 8.682 ± 0.352 |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 1 | `2048x2048` | 5.731 ± 0.025 | - | 5.708 ± 0.111 | 9.495 ± 0.047 | 1.048 ± 0.073 | 0.372 ± 0.002 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 4 | `2048x2048` | 5.684 ± 0.047 | - | 5.622 ± 0.106 | 4.558 ± 0.099 | 0.439 ± 0.023 | 0.367 ± 0.004 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 1 | `4096x4096` | 3.135 ± 0.070 | - | 3.124 ± 0.063 | 3.932 ± 0.114 | 1.625 ± 0.040 | 1.919 ± 0.031 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 4 | `4096x4096` | 3.120 ± 0.075 | - | 3.135 ± 0.066 | 1.055 ± 0.020 | 0.681 ± 0.066 | 1.910 ± 0.024 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 1 | `8192x4096` | 3.287 ± 0.077 | - | 3.345 ± 0.118 | 3.170 ± 0.024 | 17.647 ± 0.444 | 3.051 ± 0.058 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 4 | `8192x4096` | 5.929 ± 0.014 | - | 5.933 ± 0.019 | 0.985 ± 0.058 | 5.398 ± 0.022 | 2.992 ± 0.043 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 1 | `8192x4096` | 3.290 ± 0.203 | - | 3.183 ± 0.060 | 2.909 ± 0.066 | 13.912 ± 0.296 | 3.016 ± 0.109 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 4 | `8192x4096` | 4.573 ± 0.521 | - | 4.394 ± 0.014 | 0.972 ± 0.021 | 4.529 ± 0.011 | 2.965 ± 0.124 | - |
| cpu/elementwise_reduction | `rem` | f64 | 1 | `8388608` | 16.797 ± 0.477 | - | 16.310 ± 0.618 | 19.350 ± 0.031 | 14.712 ± 0.210 | 14.464 ± 3.209 | 13.436 ± 0.293 |
| cpu/elementwise_reduction | `rem` | f64 | 4 | `8388608` | 4.346 ± 0.731 | - | 4.065 ± 0.036 | 6.334 ± 0.110 | 4.326 ± 0.030 | 13.704 ± 0.713 | 3.586 ± 1.684 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 1 | `33554432` | 11.630 ± 0.301 | - | 11.627 ± 0.692 | 11.407 ± 0.239 | 23.529 ± 0.476 | 24.637 ± 2.944 | 24.182 ± 3.002 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 4 | `33554432` | 3.128 ± 0.300 | - | 3.075 ± 0.012 | 3.067 ± 0.072 | 6.074 ± 0.017 | 23.682 ± 2.252 | 8.399 ± 9.237 |
| cpu/elementwise_reduction | `select` | f64 | 1 | `33554432` | 7.844 ± 0.091 | - | 7.841 ± 0.122 | 9.049 ± 0.070 | 14.042 ± 0.060 | 9.356 ± 1.717 | - |
| cpu/elementwise_reduction | `select` | f64 | 4 | `33554432` | 3.347 ± 0.044 | - | 3.526 ± 0.070 | 3.580 ± 0.063 | 5.937 ± 1.358 | 7.746 ± 2.017 | - |
| cpu/elementwise_reduction | `sign` | f64 | 1 | `33554432` | 5.068 ± 0.036 | - | 5.074 ± 0.055 | 5.038 ± 0.027 | 9.879 ± 0.027 | 4.380 ± 2.826 | 11.760 ± 0.946 |
| cpu/elementwise_reduction | `sign` | f64 | 4 | `33554432` | 3.204 ± 0.025 | - | 3.103 ± 0.050 | 1.713 ± 0.022 | 3.291 ± 0.031 | 4.377 ± 1.994 | 3.426 ± 2.068 |
| cpu/elementwise_reduction | `sin` | f64 | 1 | `8388608` | 14.726 ± 0.564 | - | 14.813 ± 0.501 | 14.203 ± 0.049 | 11.122 ± 0.080 | 18.136 ± 1.312 | 17.948 ± 0.600 |
| cpu/elementwise_reduction | `sin` | f64 | 4 | `8388608` | 3.816 ± 0.111 | - | 3.892 ± 0.389 | 3.854 ± 0.024 | 3.268 ± 0.252 | 16.968 ± 0.369 | 4.388 ± 0.442 |
| cpu/elementwise_reduction | `sqrt` | f64 | 1 | `33554432` | 7.747 ± 0.079 | - | 7.708 ± 0.249 | 7.687 ± 0.209 | 8.276 ± 0.538 | 16.092 ± 1.035 | 16.094 ± 0.278 |
| cpu/elementwise_reduction | `sqrt` | f64 | 4 | `33554432` | 2.009 ± 0.044 | - | 2.026 ± 0.029 | 2.000 ± 0.015 | 2.480 ± 0.015 | 15.691 ± 1.929 | 4.156 ± 2.340 |
| cpu/elementwise_reduction | `sub` | f64 | 1 | `33554432` | 6.523 ± 0.020 | - | 6.668 ± 0.238 | 6.585 ± 0.055 | 11.613 ± 0.198 | 6.530 ± 2.751 | 8.703 ± 3.618 |
| cpu/elementwise_reduction | `sub` | f64 | 4 | `33554432` | 2.467 ± 0.099 | - | 2.530 ± 0.029 | 2.470 ± 0.018 | 3.407 ± 0.167 | 8.479 ± 8.720 | 3.388 ± 5.966 |
| cpu/elementwise_reduction | `tanh` | f64 | 1 | `8388608` | 21.353 ± 0.058 | - | 21.323 ± 0.110 | 35.563 ± 0.055 | 8.383 ± 0.056 | 23.399 ± 0.704 | 23.599 ± 0.503 |
| cpu/elementwise_reduction | `tanh` | f64 | 4 | `8388608` | 5.512 ± 0.138 | - | 5.536 ± 0.027 | 9.550 ± 0.129 | 2.255 ± 0.009 | 23.759 ± 0.345 | 6.069 ± 0.470 |
| cpu/indexing_layout | `concatenate` | f64 | 1 | `1048576+1048576` | 0.268 ± 0.007 | - | 0.547 ± 0.022 | 0.218 ± 0.009 | 0.223 ± 0.004 | 0.283 ± 0.834 | - |
| cpu/indexing_layout | `concatenate` | f64 | 4 | `1048576+1048576` | 0.275 ± 0.011 | - | 0.411 ± 0.017 | 0.133 ± 0.010 | 0.227 ± 0.006 | 0.289 ± 1.906 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 1 | `4194304 -> 2097152` | 0.237 ± 0.007 | - | 0.783 ± 0.009 | 0.274 ± 0.014 | 0.261 ± 0.009 | 0.256 ± 0.408 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 4 | `4194304 -> 2097152` | 0.257 ± 0.006 | - | 0.597 ± 0.017 | 0.084 ± 0.003 | 0.116 ± 0.009 | 0.302 ± 1.935 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 1 | `2097152` | 0.376 ± 0.005 | - | unsupported | 0.413 ± 0.006 | 0.368 ± 0.008 | 0.391 ± 0.052 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 4 | `2097152` | 0.390 ± 0.008 | - | unsupported | 0.193 ± 0.024 | 0.315 ± 0.008 | 0.391 ± 0.030 | - |
| cpu/indexing_layout | `gather` | f64 | 1 | `262144` | 0.398 ± 0.187 | - | 0.467 ± 0.081 | 0.167 ± 0.001 | 0.177 ± 0.007 | 0.287 ± 0.011 | - |
| cpu/indexing_layout | `gather` | f64 | 4 | `262144` | 0.174 ± 0.014 | - | 0.246 ± 0.023 | 0.161 ± 0.002 | 0.076 ± 0.014 | 0.284 ± 0.101 | - |
| cpu/indexing_layout | `pad` | f64 | 1 | `2097152` | 0.337 ± 0.016 | - | 0.610 ± 0.036 | 0.396 ± 0.018 | 0.286 ± 0.003 | - | - |
| cpu/indexing_layout | `pad` | f64 | 4 | `2097152` | 0.350 ± 0.008 | - | 0.479 ± 0.028 | 0.202 ± 0.025 | 0.131 ± 0.012 | - | - |
| cpu/indexing_layout | `reverse` | f64 | 1 | `2097152` | 0.493 ± 0.004 | - | 0.768 ± 0.025 | 0.312 ± 0.089 | 0.273 ± 0.006 | 0.601 ± 0.077 | - |
| cpu/indexing_layout | `reverse` | f64 | 4 | `2097152` | 0.496 ± 0.012 | - | 0.601 ± 0.012 | 0.123 ± 0.037 | 0.118 ± 0.011 | 0.731 ± 0.288 | - |
| cpu/indexing_layout | `scatter` | f64 | 1 | `262144` | 0.455 ± 0.060 | - | 0.574 ± 0.022 | 0.243 ± 0.004 | 0.224 ± 0.006 | 0.220 ± 0.006 | - |
| cpu/indexing_layout | `scatter` | f64 | 4 | `262144` | 0.408 ± 0.011 | - | 0.500 ± 0.072 | 0.310 ± 0.041 | 0.215 ± 0.006 | 0.233 ± 0.112 | - |
| cpu/indexing_layout | `slice` | f64 | 1 | `4194304 -> 2096128` | 0.607 ± 0.035 | - | 1.156 ± 0.044 | 0.428 ± 0.024 | 0.424 ± 0.007 | 1.033 ± 2.637 | - |
| cpu/indexing_layout | `slice` | f64 | 4 | `4194304 -> 2096128` | 0.657 ± 0.051 | - | 0.943 ± 0.049 | 0.148 ± 0.029 | 0.300 ± 0.012 | 1.442 ± 1.885 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 1 | `1536x1536` | 4.306 ± 0.712 | - | 4.648 ± 0.069 | 14.924 ± 0.200 | 9.214 ± 0.428 | 27.694 ± 29.213 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 4 | `1536x1536` | 4.376 ± 0.824 | - | 4.349 ± 0.029 | 7.969 ± 0.155 | 5.798 ± 0.036 | 11.470 ± 9.443 | - |
| cpu/linalg_uncovered | `det` | f64 | 1 | `1024x1024` | 4.213 ± 0.061 | - | 4.304 ± 0.055 | 4.006 ± 0.026 | 4.061 ± 0.073 | 13.536 ± 0.194 | - |
| cpu/linalg_uncovered | `det` | f64 | 4 | `1024x1024` | 4.190 ± 0.032 | - | 4.189 ± 0.030 | 3.971 ± 0.041 | 3.787 ± 0.047 | 5.060 ± 0.163 | - |
| cpu/linalg_uncovered | `eig` | f64 | 1 | `160x160` | 4.406 ± 0.066 | - | 4.400 ± 0.075 | 4.445 ± 0.047 | 5.397 ± 0.114 | 4.006 ± 0.085 | - |
| cpu/linalg_uncovered | `eig` | f64 | 4 | `160x160` | 4.440 ± 0.052 | - | 4.423 ± 0.022 | 4.487 ± 0.096 | 5.407 ± 0.070 | 4.140 ± 0.036 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 1 | `192x192` | 4.615 ± 0.056 | - | 4.616 ± 0.021 | 4.637 ± 0.053 | 4.264 ± 0.036 | 4.392 ± 0.070 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 4 | `192x192` | 4.620 ± 0.062 | - | 4.618 ± 0.036 | 4.687 ± 0.128 | 4.408 ± 0.052 | 4.496 ± 0.042 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 1 | `512x512` | 14.182 ± 0.074 | - | 15.041 ± 0.107 | 15.134 ± 0.090 | 12.646 ± 0.046 | 10.846 ± 0.087 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 4 | `512x512` | 14.699 ± 0.029 | - | 14.706 ± 0.071 | 15.134 ± 0.051 | 12.501 ± 0.050 | 6.515 ± 0.103 | - |
| cpu/linalg_uncovered | `inv` | f64 | 1 | `768x768` | 7.070 ± 0.095 | - | 6.700 ± 0.076 | 5.520 ± 0.059 | 5.509 ± 0.044 | 19.214 ± 0.371 | - |
| cpu/linalg_uncovered | `inv` | f64 | 4 | `768x768` | 7.132 ± 0.104 | - | 6.608 ± 0.054 | 5.293 ± 0.037 | 5.363 ± 0.113 | 6.410 ± 0.719 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 1 | `768x384,rhs=16` | - | 5.800 ± 0.058 | 5.500 ± 0.039 | 3.228 ± 0.040 | 13.582 ± 0.071 | 5.177 ± 0.049 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 4 | `768x384,rhs=16` | - | 5.957 ± 0.098 | 5.581 ± 0.064 | 3.178 ± 0.029 | 13.399 ± 0.065 | 3.106 ± 0.032 | - |
| cpu/linalg_uncovered | `lu` | f64 | 1 | `1024x1024` | 4.085 ± 0.049 | - | 4.277 ± 0.058 | 6.019 ± 0.110 | 6.711 ± 0.081 | 15.985 ± 1.986 | - |
| cpu/linalg_uncovered | `lu` | f64 | 4 | `1024x1024` | 4.305 ± 0.030 | - | 4.505 ± 0.061 | 5.254 ± 0.165 | 5.016 ± 0.100 | 8.392 ± 2.503 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 1 | `2048x2048` | 0.359 ± 0.003 | - | 0.370 ± 0.013 | 1.276 ± 0.005 | 2.873 ± 0.202 | 3.733 ± 0.102 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 4 | `2048x2048` | 0.644 ± 0.185 | - | 0.660 ± 0.037 | 1.269 ± 0.044 | 0.924 ± 0.007 | 3.932 ± 0.019 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 1 | `512x256` | 6.585 ± 0.052 | - | 6.578 ± 0.055 | 6.463 ± 0.078 | 7.360 ± 0.058 | 13.115 ± 0.160 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 4 | `512x256` | 6.662 ± 0.020 | - | 6.845 ± 0.833 | 6.473 ± 0.043 | 6.739 ± 0.090 | 9.440 ± 0.129 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 1 | `512x256` | 6.576 ± 0.030 | - | 6.507 ± 0.045 | 6.440 ± 0.038 | 7.319 ± 0.062 | 12.882 ± 0.129 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 4 | `512x256` | 6.952 ± 0.107 | - | 6.990 ± 0.122 | 6.470 ± 0.023 | 6.758 ± 0.065 | 9.646 ± 0.355 | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 1 | `1024x1024` | 4.181 ± 0.080 | - | 4.685 ± 0.093 | 4.004 ± 0.067 | 4.101 ± 0.059 | 13.522 ± 0.150 | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 4 | `1024x1024` | 4.160 ± 0.048 | - | 4.453 ± 0.037 | 3.978 ± 0.036 | 3.802 ± 0.027 | 5.135 ± 0.438 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 1 | `768x384` | - | unsupported | unsupported | 18.419 ± 0.066 | 18.882 ± 0.111 | 44.320 ± 0.551 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 4 | `768x384` | - | unsupported | unsupported | 18.900 ± 0.105 | 18.932 ± 0.111 | 29.248 ± 0.172 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 1 | `4096x4096,rhs=64` | 5.361 ± 0.200 | - | 7.925 ± 0.216 | 3.913 ± 0.037 | 25.231 ± 0.634 | 20.692 ± 0.170 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 4 | `4096x4096,rhs=64` | 5.221 ± 0.201 | - | 7.064 ± 1.073 | 3.794 ± 0.063 | 12.361 ± 0.073 | 7.112 ± 0.202 | - |
| cpu/output_reuse | `add_into` | f64 | 1 | `33554432` | 23.397 ± 0.153 | - | unsupported | 6.546 ± 0.046 | - | 6.460 ± 0.086 | 8.203 ± 0.281 |
| cpu/output_reuse | `add_into` | f64 | 4 | `33554432` | 5.976 ± 0.030 | - | unsupported | 2.486 ± 0.028 | - | 6.487 ± 0.040 | 3.074 ± 0.172 |
| cpu/output_reuse | `conj_into` | c64 | 1 | `16777216` | 8.026 ± 0.599 | - | unsupported | 4.514 ± 0.037 | - | 4.352 ± 0.040 | 4.341 ± 0.047 |
| cpu/output_reuse | `conj_into` | c64 | 4 | `16777216` | 1.855 ± 0.020 | - | unsupported | 1.569 ± 0.027 | - | 4.353 ± 0.047 | 1.523 ± 0.027 |
| cpu/output_reuse | `copy_read_into` | f64 | 1 | `33554432` | 3.827 ± 0.046 | - | unsupported | 4.357 ± 0.058 | - | 3.856 ± 0.079 | - |
| cpu/output_reuse | `copy_read_into` | f64 | 4 | `33554432` | 1.531 ± 0.012 | - | unsupported | 1.548 ± 0.024 | - | 3.799 ± 0.021 | - |
| cpu/output_reuse | `div_into` | f64 | 1 | `33554432` | 20.413 ± 2.233 | - | unsupported | 6.512 ± 0.086 | - | 6.365 ± 0.155 | 8.638 ± 0.453 |
| cpu/output_reuse | `div_into` | f64 | 4 | `33554432` | 5.982 ± 0.071 | - | unsupported | 2.536 ± 0.179 | - | 6.498 ± 0.041 | 3.106 ± 0.109 |
| cpu/output_reuse | `dot_general_read_into` | f64 | 1 | `1024x1024` | 4.465 ± 0.066 | - | unsupported | 4.449 ± 0.044 | - | 35.946 ± 0.098 | - |
| cpu/output_reuse | `dot_general_read_into` | f64 | 4 | `1024x1024` | 4.696 ± 0.141 | - | unsupported | 4.676 ± 0.212 | - | 9.530 ± 0.335 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 1 | `1024x1024` | 4.893 ± 0.057 | - | unsupported | 4.893 ± 0.054 | - | 35.857 ± 0.087 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 4 | `1024x1024` | 4.919 ± 0.068 | - | unsupported | 5.286 ± 0.295 | - | 9.546 ± 0.225 | - |
| cpu/output_reuse | `mul_into` | f64 | 1 | `33554432` | 23.432 ± 0.086 | - | unsupported | 6.483 ± 0.055 | - | 6.457 ± 0.081 | 8.384 ± 0.535 |
| cpu/output_reuse | `mul_into` | f64 | 4 | `33554432` | 5.936 ± 0.023 | - | unsupported | 2.486 ± 0.025 | - | 6.501 ± 0.039 | 3.098 ± 0.147 |
| cpu/output_reuse | `neg_into` | f64 | 1 | `33554432` | 11.480 ± 0.731 | - | unsupported | 4.338 ± 0.049 | - | 4.323 ± 0.095 | 7.775 ± 0.081 |
| cpu/output_reuse | `neg_into` | f64 | 4 | `33554432` | 2.781 ± 0.099 | - | unsupported | 1.540 ± 0.021 | - | 4.340 ± 0.032 | 2.792 ± 0.322 |
| cpu/output_reuse | `sub_into` | f64 | 1 | `33554432` | 23.438 ± 0.079 | - | unsupported | 6.549 ± 0.043 | - | 6.447 ± 0.103 | 8.449 ± 0.408 |
| cpu/output_reuse | `sub_into` | f64 | 4 | `33554432` | 5.951 ± 0.018 | - | unsupported | 2.483 ± 0.016 | - | 6.478 ± 0.018 | 3.038 ± 0.301 |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 1 | `8192x1 -> 8192x4096` | 3.369 ± 0.058 | - | 2.018 ± 0.030 | 2.056 ± 0.017 | 1.892 ± 0.031 | 12.992 ± 29.599 | - |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 4 | `8192x1 -> 8192x4096` | 2.181 ± 0.063 | - | 1.104 ± 0.825 | 0.978 ± 0.165 | 1.108 ± 0.018 | 10.865 ± 6.738 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 1 | `33554432` | 3.505 ± 0.029 | - | 7.936 ± 0.140 | 3.538 ± 0.038 | 4.658 ± 0.328 | 3.544 ± 3.183 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 4 | `33554432` | 1.319 ± 0.082 | - | 2.879 ± 0.050 | 1.325 ± 0.045 | 1.351 ± 0.074 | 4.823 ± 6.547 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 1 | `8192 -> 8192x8192` | 2.013 ± 0.058 | - | 2.048 ± 0.064 | 4.030 ± 0.063 | 6.015 ± 0.098 | 9.123 ± 29.970 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 4 | `8192 -> 8192x8192` | 1.970 ± 0.026 | - | 2.014 ± 0.067 | 2.165 ± 0.244 | 2.268 ± 0.156 | 4.734 ± 6.488 | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 1 | `8388608x2x2 -> 8388608x2` | 2.206 ± 0.035 | - | 6.575 ± 0.139 | 4.395 ± 0.034 | 3.474 ± 0.046 | - | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 4 | `8388608x2x2 -> 8388608x2` | 2.180 ± 0.026 | - | 3.764 ± 0.061 | 1.750 ± 0.056 | 1.273 ± 0.057 | - | - |
| cpu/structural_shape | `reshape` | f64 | 1 | `33554432 -> 8192x4096` | 3.828 ± 0.067 | - | 4.324 ± 0.107 | 4.338 ± 0.124 | 3.525 ± 0.070 | 3.877 ± 0.056 | - |
| cpu/structural_shape | `reshape` | f64 | 4 | `33554432 -> 8192x4096` | 3.870 ± 0.110 | - | 1.582 ± 0.030 | 1.540 ± 0.033 | 1.645 ± 0.015 | 3.889 ± 0.028 | - |
| cpu/structural_shape | `transpose` | f64 | 1 | `4096x4096` | 28.109 ± 0.766 | - | 23.210 ± 0.140 | 19.079 ± 0.568 | 12.887 ± 0.089 | 61.448 ± 27.454 | 34.949 ± 27.789 |
| cpu/structural_shape | `transpose` | f64 | 4 | `4096x4096` | 12.453 ± 0.805 | - | 7.101 ± 0.316 | 19.078 ± 0.389 | 5.465 ± 0.339 | 59.178 ± 6.962 | 21.477 ± 7.472 |
| cpu/structural_shape | `tril` | f64 | 1 | `4096x4096` | 3.261 ± 0.124 | - | 5.461 ± 0.193 | 3.452 ± 0.171 | 2.218 ± 0.038 | 4.238 ± 0.081 | - |
| cpu/structural_shape | `tril` | f64 | 4 | `4096x4096` | 3.289 ± 0.106 | - | 4.579 ± 0.565 | 1.334 ± 0.040 | 0.830 ± 0.059 | 4.241 ± 0.118 | - |
| cpu/structural_shape | `triu` | f64 | 1 | `4096x4096` | 3.113 ± 0.093 | - | 5.579 ± 0.162 | 3.207 ± 0.082 | 2.204 ± 0.018 | 3.909 ± 0.072 | - |
| cpu/structural_shape | `triu` | f64 | 4 | `4096x4096` | 3.135 ± 0.216 | - | 4.432 ± 0.110 | 1.174 ± 0.023 | 0.805 ± 0.038 | 3.926 ± 0.067 | - |
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
| broadcast_in_dim_view | 1 | PyTorch Python (ms) | 16 | 0.006084 | 380.25 |
| broadcast_in_dim_view | 1 | tenferro-rs direct API (ms) | 16 | 0.001959 | 122.44 |
| broadcast_in_dim_view | 4 | PyTorch Python (ms) | 16 | 0.007500 | 468.75 |
| broadcast_in_dim_view | 4 | tenferro-rs direct API (ms) | 16 | 0.001916 | 119.75 |
| reshape_view | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000042 | 2.62 |
| reshape_view | 1 | PyTorch Python (ms) | 16 | 0.005125 | 320.31 |
| reshape_view | 1 | tenferro-rs direct API (ms) | 16 | 0.003292 | 205.75 |
| reshape_view | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000042 | 2.62 |
| reshape_view | 4 | PyTorch Python (ms) | 16 | 0.007166 | 447.88 |
| reshape_view | 4 | tenferro-rs direct API (ms) | 16 | 0.003250 | 203.12 |
| slice_view | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000042 | 2.62 |
| slice_view | 1 | PyTorch Python (ms) | 16 | 0.004750 | 296.88 |
| slice_view | 1 | tenferro-rs direct API (ms) | 16 | 0.003125 | 195.31 |
| slice_view | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000042 | 2.62 |
| slice_view | 4 | PyTorch Python (ms) | 16 | 0.006167 | 385.44 |
| slice_view | 4 | tenferro-rs direct API (ms) | 16 | 0.003000 | 187.50 |
| transpose_view | 1 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000042 | 2.62 |
| transpose_view | 1 | PyTorch Python (ms) | 16 | 0.004667 | 291.69 |
| transpose_view | 1 | tenferro-rs direct API (ms) | 16 | 0.002666 | 166.62 |
| transpose_view | 4 | Julia (Base/LinearAlgebra) (ms) | 16 | 0.000042 | 2.62 |
| transpose_view | 4 | PyTorch Python (ms) | 16 | 0.006292 | 393.25 |
| transpose_view | 4 | tenferro-rs direct API (ms) | 16 | 0.002500 | 156.25 |

## Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `cpu/complex/div` (c64, threads=4, shape=`8388608`): `tenferro-direct` is 12.3x faster than the slowest successful cell (`julia-base`, 19.424 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/div` (c64, threads=4, shape=`8388608`): `tenferro-trace` is 10.9x faster than the slowest successful cell (`julia-base`, 19.424 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 25.5x faster than the slowest successful cell (`pytorch-cpu`, 9.495 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 12.9x faster than the slowest successful cell (`tenferro-direct`, 5.684 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 15.5x faster than the slowest successful cell (`tenferro-direct`, 5.684 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/tanh` (f64, threads=4, shape=`8388608`): `jax-cpu` is 10.5x faster than the slowest successful cell (`julia-base`, 23.759 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=1, shape=`2048x2048`): `tenferro-direct` is 10.4x faster than the slowest successful cell (`julia-base`, 3.733 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/linalg_uncovered/norm_fro` (f64, threads=1, shape=`2048x2048`): `tenferro-trace` is 10.1x faster than the slowest successful cell (`julia-base`, 3.733 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/broadcast_in_dim` (f64, threads=4, shape=`8192x1 -> 8192x4096`): `pytorch-cpu` is 11.1x faster than the slowest successful cell (`julia-base`, 10.865 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`): `jax-cpu` is 10.8x faster than the slowest successful cell (`julia-base`, 59.178 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=1, shape=`33554432 -> 8192x4096`): `julia-base` is 106.7x faster than the slowest successful cell (`pytorch-cpu`, 0.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/reshape_view` (f64, threads=4, shape=`33554432 -> 8192x4096`): `julia-base` is 149.3x faster than the slowest successful cell (`pytorch-cpu`, 0.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=1, shape=`4194304 -> 2096128`): `julia-base` is 99.0x faster than the slowest successful cell (`pytorch-cpu`, 0.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/slice_view` (f64, threads=4, shape=`4194304 -> 2096128`): `julia-base` is 128.3x faster than the slowest successful cell (`pytorch-cpu`, 0.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=1, shape=`4096x4096`): `julia-base` is 97.3x faster than the slowest successful cell (`pytorch-cpu`, 0.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/view_metadata/transpose_view` (f64, threads=4, shape=`4096x4096`): `julia-base` is 131.0x faster than the slowest successful cell (`pytorch-cpu`, 0.000 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

## Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.
