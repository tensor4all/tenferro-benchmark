# CPU Public API Benchmark Results

- Suite: `cpu/public_api`
- Target profile: `mac-cpu`
- Suite file: `benchmarks/cpu/public_api.yaml`
- Public API coverage manifest: `benchmarks/cpu/public_api_coverage.yaml`
- Run metadata: `data/results/mac-cpu/cpu/public_api/20260913_124711/run.yaml`
- Timestamp: `20260913_124711`

Original baseline run: `PUBLICATION_GATE_PROFILE=full ./scripts/run_cpu_public_api.sh 1 4`.

- Sampling: `full` profile, 15 measured runs, 3 warmups per row

Original baseline runs are under `data/results/mac-cpu/cpu/public_api/20260913_124711`.

- tenferro-rs commit: `a48866b1a0bb52e6f9712c485925105b14ea30b9`

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

- Run metadata: `data/results/mac-cpu/cpu/public_api/20260913_124711/run_t1.yaml`
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

- Run metadata: `data/results/mac-cpu/cpu/public_api/20260913_124711/run_t4.yaml`
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

- CSV: `data/results/mac-cpu/cpu/public_api/20260913_124711/cpu_public_api_t1_20260913_124711.csv`
- CSV: `data/results/mac-cpu/cpu/public_api/20260913_124711/cpu_public_api_t4_20260913_124711.csv`
- Source table: `data/results/mac-cpu/cpu/public_api/20260913_124711/cpu_public_api_20260913_124711.md`

## Targeted timing correction

- Refresh: `20260913_130307`, Apple M5 Max, 1 and 4 threads; tenferro `a48866b1a0bb52e6f9712c485925105b14ea30b9` (unchanged).
- Collection commands: `data/results/mac-cpu/cpu/cpu_ops/20260913_130307/collection.sh` and `data/results/mac-cpu/cpu/cpu_ops/20260913_130307/view_recheck.sh`.
- Sampling: 3 warmups and 15 measured runs; sequential collection with the idle-host guard enabled.
- Raw corrected rows and per-thread metadata: `data/results/mac-cpu/cpu/cpu_ops/20260913_130307/`.
- All unselected cells are retained verbatim from the original runs named below; the tables combine those original measurements with this targeted correction.

- Replaced only four tenferro direct `cpu/view_metadata` cells per thread. Input duplication is outside timing; outputs retain their allocation until after the clock stops. These cells measure concrete TensorValue view operations, not EagerTensor dispatch.
- Baseline for all other cells: `data/results/mac-cpu/cpu/public_api/20260913_124711/`.
- View-only confirmation was collected after the first run; both raw runs are retained. Sub-microsecond single-call samples have substantial relative timer/scheduling variability; use the reported IQR and avoid precise speedup claims from these cells.

## CPU Benchmark Items

Median ± IQR (ms). Missing backends are shown as `-`.

| suite | benchmark | dtype | threads | shape | tenferro-rs direct API (ms) | tenferro-rs trace mode (ms) | PyTorch Python (ms) | JAX Python (XLA CPU) (ms) | Julia (Base/LinearAlgebra) (ms) | Julia (Strided.jl) (ms) |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| cpu/complex | `cholesky` | c64 | 1 | `448x448` | 2.555 ± 0.076 | 2.687 ± 0.106 | 2.892 ± 0.104 | 0.767 ± 0.040 | 3.051 ± 0.389 | - |
| cpu/complex | `cholesky` | c64 | 4 | `448x448` | 2.475 ± 0.116 | 2.535 ± 0.091 | 2.511 ± 0.069 | 0.756 ± 0.027 | 1.366 ± 0.328 | - |
| cpu/complex | `conj` | c64 | 1 | `16777216` | 9.606 ± 2.500 | 9.321 ± 3.693 | 4.686 ± 0.051 | 1.918 ± 0.111 | 4.639 ± 0.109 | 4.677 ± 0.219 |
| cpu/complex | `conj` | c64 | 4 | `16777216` | 2.842 ± 0.139 | 2.867 ± 0.088 | 1.595 ± 0.092 | 1.823 ± 0.113 | 4.622 ± 0.178 | 11.343 ± 9.584 |
| cpu/complex | `div` | c64 | 1 | `8388608` | 4.598 ± 0.107 | 4.733 ± 0.092 | 8.089 ± 0.079 | 6.341 ± 0.078 | 19.214 ± 0.151 | 18.714 ± 0.258 |
| cpu/complex | `div` | c64 | 4 | `8388608` | 1.587 ± 0.032 | 1.675 ± 0.087 | 2.837 ± 0.036 | 6.252 ± 0.061 | 19.256 ± 0.551 | 7.058 ± 2.752 |
| cpu/complex | `dot_general` | c64 | 1 | `640x640` | 6.153 ± 0.876 | 6.330 ± 0.160 | 5.963 ± 0.145 | 3.031 ± 0.297 | 43.462 ± 1.394 | - |
| cpu/complex | `dot_general` | c64 | 4 | `640x640` | 5.857 ± 0.584 | 6.296 ± 0.291 | 5.714 ± 0.200 | 2.997 ± 0.203 | 11.077 ± 0.425 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 1 | `640x640` | 6.320 ± 0.157 | unsupported | 6.096 ± 0.266 | 3.125 ± 0.201 | 44.462 ± 4.732 | - |
| cpu/complex | `dot_general_with_conj` | c64 | 4 | `640x640` | 5.995 ± 0.153 | unsupported | 5.843 ± 0.164 | 3.051 ± 0.248 | 11.030 ± 0.258 | - |
| cpu/complex | `eig` | c64 | 1 | `112x112` | 5.981 ± 0.069 | 6.024 ± 0.065 | 5.967 ± 0.075 | 10.209 ± 0.112 | 5.467 ± 0.264 | - |
| cpu/complex | `eig` | c64 | 4 | `112x112` | 5.942 ± 0.036 | 5.982 ± 0.040 | 5.919 ± 0.038 | 10.070 ± 0.055 | 5.008 ± 0.117 | - |
| cpu/complex | `exp` | c64 | 1 | `4194304` | 21.019 ± 0.378 | 21.058 ± 0.288 | 20.384 ± 0.127 | 5.642 ± 0.280 | 26.041 ± 0.414 | 25.037 ± 0.203 |
| cpu/complex | `exp` | c64 | 4 | `4194304` | 5.613 ± 0.902 | 5.438 ± 0.069 | 5.948 ± 0.045 | 5.517 ± 0.202 | 25.306 ± 3.007 | 7.122 ± 2.764 |
| cpu/complex | `log` | c64 | 1 | `4194304` | 24.162 ± 0.090 | 24.254 ± 0.098 | 26.105 ± 0.216 | 4.662 ± 1.254 | 36.870 ± 2.202 | 35.078 ± 0.208 |
| cpu/complex | `log` | c64 | 4 | `4194304` | 6.625 ± 0.048 | 7.279 ± 0.345 | 7.049 ± 0.215 | 4.329 ± 0.241 | 35.532 ± 0.151 | 9.809 ± 1.989 |
| cpu/complex | `mul` | c64 | 1 | `8388608` | 4.167 ± 0.269 | 4.152 ± 0.102 | 4.320 ± 0.037 | 1.588 ± 0.079 | 3.739 ± 0.049 | 3.553 ± 0.096 |
| cpu/complex | `mul` | c64 | 4 | `8388608` | 1.463 ± 0.106 | 1.889 ± 0.067 | 1.469 ± 0.058 | 1.515 ± 0.088 | 3.729 ± 0.137 | 4.333 ± 9.415 |
| cpu/complex | `norm_fro` | c64 | 1 | `2048x1536` | 5.019 ± 0.357 | 4.091 ± 0.049 | 3.697 ± 0.036 | 0.718 ± 0.076 | 5.968 ± 0.096 | - |
| cpu/complex | `norm_fro` | c64 | 4 | `2048x1536` | 2.291 ± 0.120 | 1.450 ± 0.058 | 0.945 ± 0.009 | 0.694 ± 0.027 | 5.956 ± 0.094 | - |
| cpu/complex | `qr` | c64 | 1 | `256x256` | 4.760 ± 0.072 | 4.852 ± 0.070 | 5.033 ± 0.094 | 3.702 ± 0.079 | 6.598 ± 0.297 | - |
| cpu/complex | `qr` | c64 | 4 | `256x256` | 4.789 ± 0.036 | 4.850 ± 0.041 | 4.871 ± 0.052 | 3.645 ± 0.031 | 3.017 ± 0.318 | - |
| cpu/complex | `solve` | c64 | 1 | `384x384,rhs=8` | 3.408 ± 0.077 | 3.485 ± 0.095 | 3.241 ± 0.052 | 1.378 ± 0.058 | 3.787 ± 0.112 | - |
| cpu/complex | `solve` | c64 | 4 | `384x384,rhs=8` | 3.586 ± 0.068 | 3.666 ± 0.221 | 3.330 ± 0.085 | 1.327 ± 0.038 | 1.527 ± 0.112 | - |
| cpu/complex | `svd` | c64 | 1 | `160x160` | 3.952 ± 0.049 | 3.919 ± 0.101 | 3.947 ± 0.104 | 2.838 ± 0.104 | 4.584 ± 0.158 | - |
| cpu/complex | `svd` | c64 | 4 | `160x160` | 4.008 ± 0.040 | 4.031 ± 0.040 | 3.964 ± 0.053 | 2.755 ± 0.034 | 2.936 ± 0.245 | - |
| cpu/complex | `tensordot` | c64 | 1 | `640x640` | 6.084 ± 0.231 | 6.291 ± 0.107 | 5.878 ± 0.160 | 3.097 ± 0.284 | 41.883 ± 0.633 | - |
| cpu/complex | `tensordot` | c64 | 4 | `640x640` | 5.837 ± 0.160 | 6.055 ± 0.209 | 5.705 ± 0.091 | 3.072 ± 0.226 | 10.845 ± 0.329 | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 1 | `1024x1024` | 5.391 ± 1.031 | unsupported | 5.490 ± 0.523 | 2.800 ± 0.055 | - | - |
| cpu/einsum_concrete | `einsum_ij_jk_ik` | f64 | 4 | `1024x1024` | 5.210 ± 0.902 | unsupported | 4.910 ± 0.362 | 2.742 ± 0.077 | - | - |
| cpu/elementwise_reduction | `abs` | f64 | 1 | `33554432` | 4.523 ± 0.104 | 4.575 ± 0.033 | 4.546 ± 0.115 | 2.051 ± 0.060 | 4.627 ± 0.060 | 8.168 ± 0.053 |
| cpu/elementwise_reduction | `abs` | f64 | 4 | `33554432` | 1.640 ± 0.092 | 1.679 ± 0.084 | 1.624 ± 0.100 | 2.071 ± 0.201 | 4.533 ± 0.203 | 2.823 ± 2.283 |
| cpu/elementwise_reduction | `add` | f64 | 1 | `33554432` | 6.820 ± 0.106 | 6.923 ± 0.143 | 6.962 ± 0.047 | 3.627 ± 0.196 | 6.978 ± 0.258 | 8.929 ± 0.479 |
| cpu/elementwise_reduction | `add` | f64 | 4 | `33554432` | 4.631 ± 0.555 | 2.783 ± 0.146 | 2.636 ± 0.153 | 3.478 ± 0.056 | 6.824 ± 0.237 | 6.629 ± 10.961 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 1 | `4194304` | 16.971 ± 0.090 | 17.024 ± 0.133 | 18.290 ± 0.086 | 4.114 ± 0.304 | 21.385 ± 0.202 | 21.586 ± 0.199 |
| cpu/elementwise_reduction | `chain_log1p_exp_mul` | f64 | 4 | `4194304` | 4.933 ± 0.173 | 5.050 ± 0.246 | 5.371 ± 0.265 | 4.062 ± 0.335 | 21.169 ± 0.099 | 5.784 ± 0.401 |
| cpu/elementwise_reduction | `clamp` | f64 | 1 | `8388608` | 2.843 ± 0.042 | 2.931 ± 0.072 | 2.321 ± 0.088 | 1.925 ± 0.110 | 2.322 ± 0.134 | 3.174 ± 0.124 |
| cpu/elementwise_reduction | `clamp` | f64 | 4 | `8388608` | 1.566 ± 0.061 | 1.619 ± 0.142 | 0.887 ± 0.039 | 1.861 ± 0.041 | 2.240 ± 0.219 | 1.561 ± 2.862 |
| cpu/elementwise_reduction | `compare_lt` | f64 | 1 | `33554432` | 5.375 ± 0.107 | 5.421 ± 0.076 | 8.701 ± 0.200 | 2.348 ± 0.044 | 6.043 ± 0.098 | - |
| cpu/elementwise_reduction | `compare_lt` | f64 | 4 | `33554432` | 2.175 ± 0.120 | 2.203 ± 0.094 | 2.906 ± 0.037 | 2.357 ± 0.081 | 5.854 ± 0.068 | - |
| cpu/elementwise_reduction | `cos` | f64 | 1 | `8388608` | 16.325 ± 0.454 | 16.324 ± 0.967 | 17.271 ± 0.118 | 2.838 ± 0.165 | 20.582 ± 0.095 | 20.796 ± 0.050 |
| cpu/elementwise_reduction | `cos` | f64 | 4 | `8388608` | 4.309 ± 0.075 | 4.337 ± 0.121 | 4.674 ± 0.261 | 2.761 ± 0.175 | 20.522 ± 0.158 | 5.511 ± 0.541 |
| cpu/elementwise_reduction | `div` | f64 | 1 | `33554432` | 6.781 ± 0.128 | 6.901 ± 0.114 | 6.864 ± 0.094 | 3.592 ± 0.185 | 7.058 ± 0.296 | 8.958 ± 0.171 |
| cpu/elementwise_reduction | `div` | f64 | 4 | `33554432` | 2.636 ± 0.154 | 2.771 ± 0.155 | 2.644 ± 0.186 | 3.483 ± 0.059 | 6.889 ± 0.284 | 3.187 ± 2.384 |
| cpu/elementwise_reduction | `exp` | f64 | 1 | `8388608` | 16.010 ± 0.191 | 16.087 ± 0.253 | 12.400 ± 0.225 | 2.340 ± 0.190 | 16.967 ± 0.171 | 16.873 ± 0.261 |
| cpu/elementwise_reduction | `exp` | f64 | 4 | `8388608` | 4.093 ± 0.132 | 4.201 ± 0.270 | 4.413 ± 0.024 | 2.355 ± 0.311 | 16.779 ± 0.179 | 5.961 ± 1.029 |
| cpu/elementwise_reduction | `expm1` | f64 | 1 | `4194304` | 8.506 ± 0.759 | 8.679 ± 0.139 | 14.945 ± 0.072 | 2.518 ± 0.158 | 12.937 ± 0.104 | 12.983 ± 0.230 |
| cpu/elementwise_reduction | `expm1` | f64 | 4 | `4194304` | 2.175 ± 0.053 | 2.262 ± 0.095 | 4.009 ± 0.196 | 2.431 ± 0.328 | 12.739 ± 0.185 | 3.325 ± 0.426 |
| cpu/elementwise_reduction | `log` | f64 | 1 | `8388608` | 15.539 ± 0.118 | 15.517 ± 0.113 | 21.110 ± 0.166 | 3.270 ± 0.477 | 22.721 ± 0.108 | 22.704 ± 0.125 |
| cpu/elementwise_reduction | `log` | f64 | 4 | `8388608` | 4.225 ± 0.212 | 4.279 ± 0.125 | 5.699 ± 0.430 | 3.221 ± 0.387 | 22.654 ± 0.132 | 5.949 ± 0.460 |
| cpu/elementwise_reduction | `log1p` | f64 | 1 | `4194304` | 10.344 ± 0.070 | 10.421 ± 0.090 | 11.226 ± 0.118 | 2.388 ± 0.285 | 12.432 ± 0.099 | 12.457 ± 0.171 |
| cpu/elementwise_reduction | `log1p` | f64 | 4 | `4194304` | 2.849 ± 0.096 | 2.809 ± 0.137 | 2.951 ± 0.094 | 2.459 ± 0.514 | 12.294 ± 0.192 | 3.264 ± 0.438 |
| cpu/elementwise_reduction | `maximum` | f64 | 1 | `33554432` | 6.732 ± 0.088 | 6.809 ± 0.084 | 6.804 ± 0.090 | 3.600 ± 0.091 | 6.789 ± 0.080 | 8.698 ± 0.247 |
| cpu/elementwise_reduction | `maximum` | f64 | 4 | `33554432` | 2.592 ± 0.287 | 2.693 ± 0.170 | 2.597 ± 0.140 | 3.558 ± 0.136 | 6.717 ± 0.228 | 3.191 ± 2.327 |
| cpu/elementwise_reduction | `minimum` | f64 | 1 | `33554432` | 6.764 ± 0.071 | 6.847 ± 0.132 | 6.753 ± 0.071 | 3.573 ± 0.158 | 6.786 ± 0.040 | 8.817 ± 0.264 |
| cpu/elementwise_reduction | `minimum` | f64 | 4 | `33554432` | 2.678 ± 0.115 | 2.680 ± 0.126 | 2.616 ± 0.086 | 3.472 ± 0.087 | 6.707 ± 0.234 | 3.139 ± 2.270 |
| cpu/elementwise_reduction | `mul` | f64 | 1 | `33554432` | 6.872 ± 0.094 | 7.058 ± 0.156 | 6.783 ± 0.085 | 3.612 ± 0.109 | 6.956 ± 0.140 | 8.932 ± 0.321 |
| cpu/elementwise_reduction | `mul` | f64 | 4 | `33554432` | 2.815 ± 0.137 | 2.892 ± 0.111 | 2.615 ± 0.103 | 3.439 ± 0.089 | 6.728 ± 0.032 | 3.319 ± 9.370 |
| cpu/elementwise_reduction | `neg` | f64 | 1 | `33554432` | 4.524 ± 0.106 | 4.567 ± 0.126 | 4.565 ± 0.123 | 2.048 ± 0.129 | 4.625 ± 0.139 | 8.158 ± 0.157 |
| cpu/elementwise_reduction | `neg` | f64 | 4 | `33554432` | 1.659 ± 0.127 | 1.679 ± 0.090 | 1.606 ± 0.079 | 2.042 ± 0.049 | 4.603 ± 0.075 | 2.804 ± 2.862 |
| cpu/elementwise_reduction | `pow` | f64 | 1 | `4194304` | 17.923 ± 0.192 | 18.002 ± 0.118 | 38.724 ± 0.187 | 4.245 ± 0.390 | 32.486 ± 1.212 | 31.872 ± 0.708 |
| cpu/elementwise_reduction | `pow` | f64 | 4 | `4194304` | 5.268 ± 0.804 | 5.028 ± 0.189 | 11.566 ± 1.700 | 4.082 ± 0.339 | 31.674 ± 1.339 | 8.324 ± 0.666 |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 1 | `2048x2048` | 5.870 ± 0.139 | 5.888 ± 0.077 | 11.993 ± 0.470 | 0.424 ± 0.094 | 0.370 ± 0.032 | - |
| cpu/elementwise_reduction | `reduce_max_axis0` | f64 | 4 | `2048x2048` | 5.870 ± 0.076 | 5.812 ± 0.075 | 4.875 ± 0.257 | 0.404 ± 0.022 | 0.376 ± 0.015 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 1 | `4096x4096` | 3.214 ± 0.099 | 3.235 ± 0.072 | 4.065 ± 0.140 | 0.901 ± 0.165 | 1.999 ± 0.068 | - |
| cpu/elementwise_reduction | `reduce_min_axis1` | f64 | 4 | `4096x4096` | 3.243 ± 0.066 | 3.214 ± 0.121 | 1.056 ± 0.042 | 0.634 ± 0.027 | 1.909 ± 0.040 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 1 | `8192x4096` | 3.484 ± 0.124 | 3.558 ± 0.114 | 3.395 ± 0.161 | 4.156 ± 0.198 | 3.296 ± 0.191 | - |
| cpu/elementwise_reduction | `reduce_prod_all` | f64 | 4 | `8192x4096` | 6.077 ± 0.062 | 6.105 ± 0.093 | 1.007 ± 0.045 | 4.380 ± 0.133 | 3.094 ± 0.049 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 1 | `8192x4096` | 3.573 ± 0.283 | 3.436 ± 0.184 | 3.236 ± 0.214 | 3.272 ± 0.111 | 3.231 ± 0.220 | - |
| cpu/elementwise_reduction | `reduce_sum_all` | f64 | 4 | `8192x4096` | 4.610 ± 0.392 | 4.577 ± 0.113 | 0.999 ± 0.073 | 3.776 ± 0.056 | 3.087 ± 0.129 | - |
| cpu/elementwise_reduction | `rem` | f64 | 1 | `8388608` | 16.216 ± 1.064 | 16.143 ± 0.164 | 19.937 ± 0.257 | 4.006 ± 0.162 | 13.993 ± 3.249 | 13.504 ± 0.440 |
| cpu/elementwise_reduction | `rem` | f64 | 4 | `8388608` | 4.453 ± 0.475 | 4.172 ± 0.220 | 6.392 ± 0.065 | 3.840 ± 0.215 | 13.322 ± 0.944 | 3.638 ± 0.528 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 1 | `33554432` | 12.038 ± 0.258 | 12.033 ± 0.212 | 12.095 ± 0.137 | 5.024 ± 0.070 | 24.252 ± 0.287 | 24.387 ± 0.199 |
| cpu/elementwise_reduction | `rsqrt` | f64 | 4 | `33554432` | 3.070 ± 0.033 | 3.179 ± 0.178 | 3.150 ± 0.108 | 4.956 ± 0.061 | 23.673 ± 0.182 | 6.554 ± 2.493 |
| cpu/elementwise_reduction | `select` | f64 | 1 | `33554432` | 8.151 ± 0.096 | 8.172 ± 0.132 | 9.302 ± 0.134 | 3.793 ± 0.121 | 8.228 ± 0.097 | - |
| cpu/elementwise_reduction | `select` | f64 | 4 | `33554432` | 3.505 ± 0.112 | 3.474 ± 0.100 | 3.635 ± 0.051 | 3.730 ± 0.488 | 7.961 ± 0.119 | - |
| cpu/elementwise_reduction | `sign` | f64 | 1 | `33554432` | 5.212 ± 0.089 | 5.271 ± 0.053 | 5.186 ± 0.062 | 2.779 ± 0.071 | 4.647 ± 0.088 | 12.150 ± 0.150 |
| cpu/elementwise_reduction | `sign` | f64 | 4 | `33554432` | 3.248 ± 0.098 | 3.203 ± 0.139 | 1.742 ± 0.042 | 2.717 ± 0.024 | 4.545 ± 0.127 | 3.409 ± 2.454 |
| cpu/elementwise_reduction | `sin` | f64 | 1 | `8388608` | 15.226 ± 0.527 | 15.419 ± 0.797 | 14.453 ± 0.137 | 2.806 ± 0.508 | 17.024 ± 0.326 | 17.192 ± 0.131 |
| cpu/elementwise_reduction | `sin` | f64 | 4 | `8388608` | 4.269 ± 0.167 | 4.193 ± 0.070 | 3.898 ± 0.135 | 2.716 ± 0.396 | 17.374 ± 0.212 | 4.435 ± 0.411 |
| cpu/elementwise_reduction | `sqrt` | f64 | 1 | `33554432` | 7.928 ± 0.089 | 7.963 ± 0.071 | 7.970 ± 0.189 | 2.070 ± 0.052 | 16.052 ± 0.112 | 16.198 ± 0.116 |
| cpu/elementwise_reduction | `sqrt` | f64 | 4 | `33554432` | 2.023 ± 0.035 | 2.047 ± 0.042 | 2.046 ± 0.047 | 2.051 ± 0.112 | 15.770 ± 0.099 | 4.123 ± 2.385 |
| cpu/elementwise_reduction | `sub` | f64 | 1 | `33554432` | 6.743 ± 0.074 | 6.896 ± 0.090 | 6.967 ± 0.100 | 3.609 ± 0.157 | 6.883 ± 0.253 | 8.955 ± 0.257 |
| cpu/elementwise_reduction | `sub` | f64 | 4 | `33554432` | 2.591 ± 0.126 | 2.753 ± 0.148 | 2.661 ± 0.109 | 3.462 ± 0.082 | 6.780 ± 0.258 | 3.350 ± 2.492 |
| cpu/elementwise_reduction | `tanh` | f64 | 1 | `8388608` | 22.649 ± 0.167 | 22.696 ± 0.140 | 36.015 ± 0.159 | 1.904 ± 0.059 | 23.740 ± 0.121 | 23.843 ± 0.248 |
| cpu/elementwise_reduction | `tanh` | f64 | 4 | `8388608` | 5.708 ± 0.175 | 5.618 ± 0.185 | 9.539 ± 0.245 | 1.882 ± 0.094 | 23.543 ± 0.119 | 6.188 ± 0.442 |
| cpu/indexing_layout | `concatenate` | f64 | 1 | `1048576+1048576` | 0.284 ± 0.011 | 0.573 ± 0.041 | 0.237 ± 0.024 | 0.237 ± 0.004 | 0.293 ± 0.013 | - |
| cpu/indexing_layout | `concatenate` | f64 | 4 | `1048576+1048576` | 0.278 ± 0.013 | 0.446 ± 0.042 | 0.128 ± 0.012 | 0.234 ± 0.004 | 0.288 ± 0.006 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 1 | `4194304 -> 2097152` | 0.240 ± 0.015 | 0.874 ± 0.066 | 0.277 ± 0.011 | 0.144 ± 0.020 | 0.287 ± 0.024 | - |
| cpu/indexing_layout | `dynamic_slice` | f64 | 4 | `4194304 -> 2097152` | 0.271 ± 0.025 | 0.629 ± 0.079 | 0.088 ± 0.011 | 0.119 ± 0.011 | 0.288 ± 0.011 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 1 | `2097152` | 0.396 ± 0.020 | unsupported | 0.417 ± 0.011 | 0.282 ± 0.030 | 0.401 ± 0.025 | - |
| cpu/indexing_layout | `dynamic_update_slice` | f64 | 4 | `2097152` | 0.399 ± 0.018 | unsupported | 0.170 ± 0.032 | 0.274 ± 0.014 | 0.398 ± 0.009 | - |
| cpu/indexing_layout | `gather` | f64 | 1 | `262144` | 0.407 ± 0.170 | 0.495 ± 0.093 | 0.164 ± 0.006 | 0.094 ± 0.037 | 0.293 ± 0.020 | - |
| cpu/indexing_layout | `gather` | f64 | 4 | `262144` | 0.132 ± 0.005 | 0.368 ± 0.063 | 0.164 ± 0.050 | 0.082 ± 0.004 | 0.310 ± 0.035 | - |
| cpu/indexing_layout | `pad` | f64 | 1 | `2097152` | 0.353 ± 0.014 | 0.667 ± 0.034 | 0.405 ± 0.015 | 0.157 ± 0.038 | - | - |
| cpu/indexing_layout | `pad` | f64 | 4 | `2097152` | 0.356 ± 0.014 | 0.520 ± 0.036 | 0.175 ± 0.007 | 0.125 ± 0.018 | - | - |
| cpu/indexing_layout | `reverse` | f64 | 1 | `2097152` | 0.498 ± 0.011 | 1.045 ± 0.231 | 0.334 ± 0.064 | 0.133 ± 0.009 | 0.609 ± 0.028 | - |
| cpu/indexing_layout | `reverse` | f64 | 4 | `2097152` | 0.496 ± 0.018 | 0.658 ± 0.072 | 0.120 ± 0.033 | 0.119 ± 0.011 | 0.615 ± 0.018 | - |
| cpu/indexing_layout | `scatter` | f64 | 1 | `262144` | 0.469 ± 0.075 | 0.610 ± 0.059 | 0.253 ± 0.009 | 0.317 ± 0.039 | 0.257 ± 0.067 | - |
| cpu/indexing_layout | `scatter` | f64 | 4 | `262144` | 0.366 ± 0.192 | 0.591 ± 0.183 | 0.333 ± 0.025 | 0.226 ± 0.015 | 0.252 ± 0.093 | - |
| cpu/indexing_layout | `slice` | f64 | 1 | `4194304 -> 2096128` | 0.883 ± 0.040 | 1.205 ± 0.062 | 0.464 ± 0.040 | 0.292 ± 0.030 | 0.624 ± 0.248 | - |
| cpu/indexing_layout | `slice` | f64 | 4 | `4194304 -> 2096128` | 0.664 ± 0.267 | 0.996 ± 0.084 | 0.152 ± 0.030 | 0.280 ± 0.011 | 0.576 ± 0.038 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 1 | `1536x1536` | 4.881 ± 0.600 | 5.107 ± 0.150 | 16.151 ± 0.235 | 5.393 ± 0.207 | 26.937 ± 3.746 | - |
| cpu/linalg_uncovered | `cholesky` | f64 | 4 | `1536x1536` | 4.651 ± 0.797 | 4.750 ± 0.127 | 8.035 ± 0.180 | 4.984 ± 0.204 | 10.534 ± 11.878 | - |
| cpu/linalg_uncovered | `det` | f64 | 1 | `1024x1024` | 4.648 ± 0.183 | 4.966 ± 0.303 | 4.439 ± 0.254 | 4.206 ± 0.150 | 14.051 ± 0.204 | - |
| cpu/linalg_uncovered | `det` | f64 | 4 | `1024x1024` | 4.575 ± 0.222 | 4.477 ± 0.106 | 4.139 ± 0.230 | 3.868 ± 0.217 | 5.134 ± 0.490 | - |
| cpu/linalg_uncovered | `eig` | f64 | 1 | `160x160` | 4.491 ± 0.127 | 4.488 ± 0.102 | 4.521 ± 0.069 | 5.554 ± 0.068 | 4.085 ± 0.117 | - |
| cpu/linalg_uncovered | `eig` | f64 | 4 | `160x160` | 4.474 ± 0.062 | 4.462 ± 0.037 | 4.499 ± 0.097 | 5.431 ± 0.118 | 4.218 ± 0.128 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 1 | `192x192` | 4.662 ± 0.063 | 4.676 ± 0.061 | 4.706 ± 0.089 | 4.337 ± 0.102 | 4.493 ± 0.114 | - |
| cpu/linalg_uncovered | `eigvals` | f64 | 4 | `192x192` | 4.643 ± 0.058 | 4.703 ± 0.067 | 4.737 ± 0.224 | 4.458 ± 0.069 | 4.623 ± 0.147 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 1 | `512x512` | 14.961 ± 0.217 | 15.174 ± 0.126 | 15.110 ± 0.221 | 12.838 ± 0.177 | 11.167 ± 0.144 | - |
| cpu/linalg_uncovered | `eigvalsh` | f64 | 4 | `512x512` | 14.751 ± 0.089 | 14.847 ± 0.093 | 15.244 ± 0.123 | 12.570 ± 0.038 | 6.669 ± 0.232 | - |
| cpu/linalg_uncovered | `inv` | f64 | 1 | `768x768` | 7.653 ± 0.159 | 7.063 ± 0.114 | 6.060 ± 0.140 | 5.688 ± 0.171 | 19.932 ± 0.319 | - |
| cpu/linalg_uncovered | `inv` | f64 | 4 | `768x768` | 7.303 ± 0.433 | 6.940 ± 0.135 | 5.467 ± 0.132 | 5.266 ± 0.175 | 6.628 ± 0.553 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 1 | `768x384,rhs=16` | 6.004 ± 0.143 | 5.687 ± 0.097 | 3.371 ± 0.080 | 13.546 ± 0.168 | 5.401 ± 0.175 | - |
| cpu/linalg_uncovered | `lstsq` | f64 | 4 | `768x384,rhs=16` | 5.879 ± 0.140 | 5.674 ± 0.053 | 3.196 ± 0.034 | 13.305 ± 0.073 | 3.168 ± 0.213 | - |
| cpu/linalg_uncovered | `lu` | f64 | 1 | `1024x1024` | 4.466 ± 0.207 | 4.741 ± 0.223 | 6.823 ± 0.233 | 4.857 ± 0.156 | 16.551 ± 2.941 | - |
| cpu/linalg_uncovered | `lu` | f64 | 4 | `1024x1024` | 4.198 ± 0.108 | 4.422 ± 0.082 | 5.688 ± 0.296 | 4.540 ± 0.212 | 7.540 ± 2.538 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 1 | `2048x2048` | 0.398 ± 0.042 | 0.427 ± 0.037 | 1.286 ± 0.023 | 0.773 ± 0.072 | 3.956 ± 0.045 | - |
| cpu/linalg_uncovered | `norm_fro` | f64 | 4 | `2048x2048` | 0.575 ± 0.026 | 0.602 ± 0.033 | 1.281 ± 0.017 | 0.725 ± 0.086 | 3.950 ± 0.087 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 1 | `512x256` | 6.837 ± 0.093 | 6.910 ± 0.086 | 6.651 ± 0.128 | 6.594 ± 0.101 | 13.278 ± 0.352 | - |
| cpu/linalg_uncovered | `pinv` | f64 | 4 | `512x256` | 6.797 ± 0.083 | 6.948 ± 0.121 | 6.670 ± 0.084 | 6.527 ± 0.115 | 9.828 ± 2.650 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 1 | `512x256` | 6.806 ± 0.121 | 6.838 ± 0.085 | 6.656 ± 0.105 | 6.577 ± 0.086 | 13.321 ± 0.166 | - |
| cpu/linalg_uncovered | `pinv_with_rtol` | f64 | 4 | `512x256` | 6.776 ± 0.048 | 6.889 ± 0.071 | 6.548 ± 0.064 | 6.466 ± 0.108 | 9.634 ± 0.309 | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 1 | `1024x1024` | 4.543 ± 0.145 | 5.078 ± 0.206 | 4.449 ± 0.186 | 4.159 ± 0.363 | 13.998 ± 0.140 | - |
| cpu/linalg_uncovered | `slogdet` | f64 | 4 | `1024x1024` | 4.425 ± 0.139 | 4.713 ± 0.184 | 4.213 ± 0.217 | 3.857 ± 0.222 | 5.207 ± 0.460 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 1 | `768x384` | unsupported | unsupported | 18.831 ± 0.202 | 18.802 ± 0.199 | 45.871 ± 0.347 | - |
| cpu/linalg_uncovered | `svd_full` | f64 | 4 | `768x384` | unsupported | unsupported | 19.213 ± 0.157 | 18.949 ± 0.157 | 29.973 ± 0.476 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 1 | `4096x4096,rhs=64` | 5.878 ± 0.279 | 8.435 ± 0.239 | 4.035 ± 0.072 | 10.590 ± 0.325 | 21.624 ± 0.210 | - |
| cpu/linalg_uncovered | `triangular_solve` | f64 | 4 | `4096x4096,rhs=64` | 5.410 ± 0.160 | 7.107 ± 0.237 | 3.868 ± 0.076 | 9.934 ± 0.272 | 7.517 ± 0.255 | - |
| cpu/output_reuse | `add_into` | f64 | 1 | `33554432` | 21.786 ± 3.145 | unsupported | 6.803 ± 0.051 | - | 7.468 ± 0.342 | 8.857 ± 0.664 |
| cpu/output_reuse | `add_into` | f64 | 4 | `33554432` | 5.836 ± 0.712 | unsupported | 2.591 ± 0.078 | - | 6.673 ± 0.064 | 2.940 ± 0.385 |
| cpu/output_reuse | `conj_into` | c64 | 1 | `16777216` | 8.268 ± 0.281 | unsupported | 4.693 ± 0.133 | - | 4.506 ± 0.085 | 4.472 ± 0.107 |
| cpu/output_reuse | `conj_into` | c64 | 4 | `16777216` | 1.889 ± 0.093 | unsupported | 1.645 ± 0.117 | - | 4.487 ± 0.061 | 1.740 ± 0.327 |
| cpu/output_reuse | `copy_read_into` | f64 | 1 | `33554432` | 3.971 ± 0.083 | unsupported | 4.526 ± 0.107 | - | 3.929 ± 0.106 | - |
| cpu/output_reuse | `copy_read_into` | f64 | 4 | `33554432` | 1.625 ± 0.059 | unsupported | 1.610 ± 0.083 | - | 3.873 ± 0.129 | - |
| cpu/output_reuse | `div_into` | f64 | 1 | `33554432` | 23.755 ± 0.433 | unsupported | 6.779 ± 0.094 | - | 6.662 ± 0.155 | 8.718 ± 0.144 |
| cpu/output_reuse | `div_into` | f64 | 4 | `33554432` | 5.034 ± 0.282 | unsupported | 2.646 ± 0.154 | - | 6.740 ± 0.099 | 3.016 ± 0.134 |
| cpu/output_reuse | `dot_general_read_into` | f64 | 1 | `1024x1024` | 4.971 ± 0.254 | unsupported | 5.258 ± 0.306 | - | 36.336 ± 0.087 | - |
| cpu/output_reuse | `dot_general_read_into` | f64 | 4 | `1024x1024` | 4.679 ± 0.224 | unsupported | 4.799 ± 0.207 | - | 9.729 ± 0.222 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 1 | `1024x1024` | 5.468 ± 0.111 | unsupported | 5.825 ± 0.285 | - | 37.541 ± 0.841 | - |
| cpu/output_reuse | `dot_general_read_into_accum` | f64 | 4 | `1024x1024` | 5.370 ± 0.411 | unsupported | 5.353 ± 0.203 | - | 9.724 ± 0.296 | - |
| cpu/output_reuse | `mul_into` | f64 | 1 | `33554432` | 22.499 ± 3.702 | unsupported | 6.794 ± 0.085 | - | 6.619 ± 0.081 | 8.574 ± 0.411 |
| cpu/output_reuse | `mul_into` | f64 | 4 | `33554432` | 5.602 ± 0.601 | unsupported | 2.599 ± 0.133 | - | 6.567 ± 0.049 | 3.098 ± 0.314 |
| cpu/output_reuse | `neg_into` | f64 | 1 | `33554432` | 10.582 ± 0.494 | unsupported | 4.558 ± 0.089 | - | 4.432 ± 0.060 | 7.958 ± 0.111 |
| cpu/output_reuse | `neg_into` | f64 | 4 | `33554432` | 2.997 ± 0.124 | unsupported | 1.594 ± 0.082 | - | 4.389 ± 0.054 | 2.625 ± 0.638 |
| cpu/output_reuse | `sub_into` | f64 | 1 | `33554432` | 22.822 ± 3.228 | unsupported | 6.804 ± 0.088 | - | 7.106 ± 0.478 | 9.033 ± 0.630 |
| cpu/output_reuse | `sub_into` | f64 | 4 | `33554432` | 5.567 ± 0.801 | unsupported | 2.590 ± 0.138 | - | 6.562 ± 0.060 | 3.103 ± 0.173 |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 1 | `8192x1 -> 8192x4096` | 2.065 ± 0.029 | 2.096 ± 0.043 | 2.106 ± 0.039 | 1.134 ± 0.025 | 8.828 ± 0.296 | - |
| cpu/structural_shape | `broadcast_in_dim` | f64 | 4 | `8192x1 -> 8192x4096` | 2.123 ± 0.047 | 1.148 ± 0.037 | 1.075 ± 0.165 | 1.102 ± 0.034 | 8.728 ± 0.259 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 1 | `33554432` | 3.712 ± 0.047 | 8.354 ± 0.170 | 3.718 ± 0.098 | 1.436 ± 0.154 | 3.845 ± 0.092 | - |
| cpu/structural_shape | `cast_f64_f32` | f64->f32 | 4 | `33554432` | 1.381 ± 0.109 | 3.172 ± 0.297 | 1.322 ± 0.080 | 1.541 ± 0.096 | 3.686 ± 0.070 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 1 | `8192 -> 8192x8192` | 2.103 ± 0.082 | 2.152 ± 0.099 | 4.158 ± 0.048 | 2.148 ± 0.152 | 2.267 ± 0.164 | - |
| cpu/structural_shape | `embed_diagonal` | f64 | 4 | `8192 -> 8192x8192` | 2.089 ± 0.082 | 2.126 ± 0.144 | 1.989 ± 0.268 | 2.504 ± 0.062 | 2.122 ± 0.090 | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 1 | `8388608x2x2 -> 8388608x2` | 2.267 ± 0.123 | 7.003 ± 0.128 | 4.549 ± 0.097 | 1.927 ± 0.069 | - | - |
| cpu/structural_shape | `extract_diagonal` | f64 | 4 | `8388608x2x2 -> 8388608x2` | 2.253 ± 0.041 | 4.023 ± 0.344 | 1.830 ± 0.057 | 1.494 ± 0.104 | - | - |
| cpu/structural_shape | `reshape` | f64 | 1 | `33554432 -> 8192x4096` | 4.056 ± 0.098 | 4.652 ± 0.098 | 4.495 ± 0.136 | 2.055 ± 0.184 | 4.153 ± 0.087 | - |
| cpu/structural_shape | `reshape` | f64 | 4 | `33554432 -> 8192x4096` | 3.993 ± 0.129 | 1.632 ± 0.194 | 1.588 ± 0.097 | 1.619 ± 0.171 | 4.045 ± 0.056 | - |
| cpu/structural_shape | `transpose` | f64 | 1 | `4096x4096` | 29.703 ± 0.586 | 24.478 ± 0.278 | 19.539 ± 0.212 | 3.764 ± 0.326 | 53.834 ± 3.786 | 32.590 ± 0.332 |
| cpu/structural_shape | `transpose` | f64 | 4 | `4096x4096` | 11.998 ± 1.463 | 7.299 ± 0.213 | 19.007 ± 0.143 | 3.579 ± 0.313 | 51.902 ± 2.228 | 22.041 ± 7.802 |
| cpu/structural_shape | `tril` | f64 | 1 | `4096x4096` | 3.433 ± 0.144 | 5.942 ± 0.152 | 3.809 ± 0.128 | 0.844 ± 0.056 | 4.463 ± 0.108 | - |
| cpu/structural_shape | `tril` | f64 | 4 | `4096x4096` | 3.442 ± 0.221 | 4.521 ± 0.112 | 1.348 ± 0.042 | 0.898 ± 0.058 | 4.330 ± 0.062 | - |
| cpu/structural_shape | `triu` | f64 | 1 | `4096x4096` | 3.569 ± 0.133 | 5.958 ± 0.069 | 3.508 ± 0.170 | 0.875 ± 0.098 | 4.204 ± 0.057 | - |
| cpu/structural_shape | `triu` | f64 | 4 | `4096x4096` | 3.501 ± 0.225 | 4.938 ± 0.100 | 1.196 ± 0.019 | 0.866 ± 0.030 | 4.065 ± 0.056 | - |
| cpu/view_metadata | `broadcast_in_dim_view` | f64 | 1 | `8192x1 -> 8192x4096` | 0.000 ± 0.000 | unsupported | 0.001 ± 0.000 | - | - | - |
| cpu/view_metadata | `broadcast_in_dim_view` | f64 | 4 | `8192x1 -> 8192x4096` | 0.000 ± 0.000 | unsupported | 0.001 ± 0.000 | - | - | - |
| cpu/view_metadata | `reshape_view` | f64 | 1 | `33554432 -> 8192x4096` | 0.001 ± 0.001 | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `reshape_view` | f64 | 4 | `33554432 -> 8192x4096` | 0.000 ± 0.001 | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `slice_view` | f64 | 1 | `4194304 -> 2096128` | 0.000 ± 0.000 | unsupported | 0.000 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `slice_view` | f64 | 4 | `4194304 -> 2096128` | 0.000 ± 0.000 | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `transpose_view` | f64 | 1 | `4096x4096` | 0.000 ± 0.001 | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |
| cpu/view_metadata | `transpose_view` | f64 | 4 | `4096x4096` | 0.000 ± 0.000 | unsupported | 0.001 ± 0.000 | - | 0.000 ± 0.000 | - |

## Cross-Backend Spread Audit

Rows with a successful cell more than 10x faster than the slowest successful cell are flagged for operation-specific review. This cross-backend spread check is a warning, not a correctness verdict.

- `cpu/complex/div` (c64, threads=4, shape=`8388608`): `tenferro-eager` is 12.1x faster than the slowest successful cell (`julia-base`, 19.256 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/div` (c64, threads=4, shape=`8388608`): `tenferro-trace` is 11.5x faster than the slowest successful cell (`julia-base`, 19.256 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/dot_general` (c64, threads=1, shape=`640x640`): `jax-cpu` is 14.3x faster than the slowest successful cell (`julia-base`, 43.462 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/dot_general_with_conj` (c64, threads=1, shape=`640x640`): `jax-cpu` is 14.2x faster than the slowest successful cell (`julia-base`, 44.462 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/complex/tensordot` (c64, threads=1, shape=`640x640`): `jax-cpu` is 13.5x faster than the slowest successful cell (`julia-base`, 41.883 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `jax-cpu` is 28.3x faster than the slowest successful cell (`pytorch-cpu`, 11.993 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=1, shape=`2048x2048`): `julia-base` is 32.4x faster than the slowest successful cell (`pytorch-cpu`, 11.993 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `jax-cpu` is 14.5x faster than the slowest successful cell (`tenferro-eager`, 5.870 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/reduce_max_axis0` (f64, threads=4, shape=`2048x2048`): `julia-base` is 15.6x faster than the slowest successful cell (`tenferro-eager`, 5.870 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/tanh` (f64, threads=1, shape=`8388608`): `jax-cpu` is 18.9x faster than the slowest successful cell (`pytorch-cpu`, 36.015 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/elementwise_reduction/tanh` (f64, threads=4, shape=`8388608`): `jax-cpu` is 12.5x faster than the slowest successful cell (`julia-base`, 23.543 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=1, shape=`4096x4096`): `jax-cpu` is 14.3x faster than the slowest successful cell (`julia-base`, 53.834 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.
- `cpu/structural_shape/transpose` (f64, threads=4, shape=`4096x4096`): `jax-cpu` is 14.5x faster than the slowest successful cell (`julia-base`, 51.902 ms). Audit fixture semantics, synchronization, labels, and operation-specific bandwidth/FLOP bounds.

## Physical Bound Audit

This independent check derives minimum logical bytes from dtype and shape for materializing public API operations, plus conservative operation FLOPs for reductions, Frobenius norms, matrix products, triangular solves, and recognized dense linalg families. Decorated shapes (`->`, `rhs=`, and `+`) are parsed explicitly. Operations whose touched region cannot be inferred conservatively from the reported shape are left unmodeled. It assumes at most 100 GB/s and 50 GFLOP/s per requested CPU thread, capped at 1000 GB/s and 3200 GFLOP/s. A cell more than 10x faster than the resulting lower bound is flagged. These deliberately generous ceilings are a sanity screen, not a performance gate.

No cell exceeded the conservative physical bound by more than 10x.
