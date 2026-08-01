# Einsum Suite and Instance Selection

## Source

The macrobenchmark instances come from the
[einsum benchmark](https://benchmark.einsum.org/) via the
`einsum_benchmark` Python package. `scripts/generate_dataset.py` exports
row-major metadata from the source package and adds column-major metadata for
tenferro-rs.

Tracked instance metadata lives under `data/instances/`. The base laptop-scale
set matches
[tenferro-einsum-benchmark](https://github.com/tensor4all/tenferro-einsum-benchmark);
this repository adds extra diagnostic cases for tenferro-rs trace/eager,
PyTorch, and JAX comparisons.

Import the upstream JSON export when `extern/tenferro-einsum-benchmark` is
available locally:

```bash
uv run python scripts/import_einsum_dataset.py
```

Regenerate directly from the `einsum_benchmark` Python package instead:

```bash
uv run python scripts/generate_dataset.py
```

## Source Selection Rules

The laptop-scale source selection uses category-specific criteria:

| Category | Prefix | opt_flops log10[FLOPS] | opt_flops log2[SIZE] | num_tensors | dtype |
|---|---|---:|---:|---:|---|
| Language model | `lm_` | < 10 | < 25 | <= 100 | float64 or complex128 |
| Graphical model | `gm_` | < 10 | < 27 | <= 200 | float64 or complex128 |
| Structured | `str_` | < 11 | < 26 | <= 200 | float64 or complex128 |

`rnd_mixed_` instances are skipped because they are not supported by the
current tenferro-einsum path.

## Diagnostic Instances

The repository also keeps hand-written diagnostic instances. These are not
selected from the source package; they pin specific implementation questions.

- `bin_matmul_256`: binary GEMM-shaped einsum fast path and wrapper overhead.
- `bin_matmul_1024`: larger GEMM throughput comparison.
- `bin_batched_matmul_b32_m64_n64_k64`: small batched GEMM path.
- `bin_batched_matmul_b32_m128_n128_k128`: larger batched GEMM path.
- `bin_outer_product_4096`: outer-product materialization/broadcast behavior.
- `bin_batched_outer_product_compact_j16_k16_o64_t64`: compact batched outer-product kernel path.
- `bin_batched_outer_product_noncompact_j16_k16_o64_t64`: non-compact batched outer-product path.
- `bin_elementwise_mul_2048x2048`: binary elementwise einsum overhead and kernel behavior.
- `nary_matmul_chain_64`: small N-ary case that exposes traced extension/runtime overhead.

Each instance may include optional `intent` and `notes` fields. They explain why
the case exists and how to interpret it. Runners must ignore these fields for
timing identity and cache keys.

## Layout

Source instances are row-major. tenferro-rs uses column-major tensors, so each
JSON instance includes:

- `format_string_rowmajor`
- `format_string_colmajor`
- `shapes`
- `shapes_colmajor`

Contraction paths are unchanged by row/column-major conversion because they
refer to tensor indices, not dimension labels.

## Strategies

The common path strategies are:

- `opt_flops`: contraction path optimized for FLOP count.
- `opt_size`: contraction path optimized for intermediate size.

Large gaps can be path-driven rather than kernel-driven. Use:

```bash
uv run python scripts/analyze_einsum_gaps.py \
  --report result/<target_profile>/cpu/einsum.md \
  --instances data/instances \
  --threshold 1.15
```

`path_intermediate` rows should be treated as path-planning/layout problems
before adding kernel-specific optimizations.

## Backends

Backend column names follow [architecture terminology](architecture.md).

| backend | runner | measured path |
|---|---|---|
| `tenferro-trace` | Rust | tenferro-rs trace-mode einsum, compiled once from the instance's precomputed contraction path, then re-executed for warmups/timed runs |
| `tenferro-eager` | Rust | tenferro-rs eager-mode einsum over the same precomputed path |
| `pytorch-cpu` | Python | `opt_einsum.contract(..., optimize=path, backend="torch")` with the precomputed path, `torch.zeros` fixtures |
| `jax-cpu` | Python | `opt_einsum.contract(..., optimize=path, backend="jax")` with the precomputed path, `jax.numpy.zeros` fixtures, `jax_enable_x64=True` |
| `omeinsum-jl` | Julia | `OMEinsum.DynamicEinCode` pairwise contractions following the precomputed path, `zeros(Float64, shape...)` fixtures |

Notes:

- Every backend is forced through the instance's `paths.opt_flops` /
  `paths.opt_size` contraction path (`data/instances/*.json`); none of them
  is allowed to run its own contraction-path optimizer, so cross-backend
  comparisons measure kernel/runtime overhead rather than differing path
  quality.
- `omeinsum-jl`'s runner (`scripts/benchmark_einsum_omeinsum.jl`) converts
  the instance's opt_einsum-style relative pair path into absolute operand
  ids the same way `src/main.rs::path_to_pairs` does, then derives each
  intermediate contraction's output labels with the standard einsum
  pairwise rule (a label survives the step iff it is still needed by a
  later operand or the final output). It reports mode `omeinsum_path` in
  its log, which `scripts/format_results.py` renders as the "OMEinsum.jl
  OpenBLAS (ms)" column; a mode `omeinsum_opt` (OMEinsum's own optimizer)
  is intentionally never emitted and would be excluded by the formatter as
  an unfair comparison if it were.
- Every operand across every backend is an all-zeros `float64` fixture of
  the instance's shape (`create_operand_tensors` in `src/main.rs`,
  `torch.zeros`/`jnp.zeros` in `scripts/benchmark_python.py`,
  `zeros(Float64, shape...)` in `scripts/benchmark_einsum_omeinsum.jl`), so
  timings reflect kernel/dispatch overhead on identical data across
  backends.
- `complex128` instances render `SKIP` for `omeinsum-jl`, matching how
  PyTorch/JAX skip complex dtypes in this suite (none of the tracked
  `cpu/einsum` suite instances currently use `complex128`).
- `JULIA_NUM_THREADS` is used both for Julia's own thread pool and to pin
  `LinearAlgebra.BLAS.set_num_threads`, matching the BLAS thread pinning
  applied to the other CPU backends via `OMP_NUM_THREADS` /
  `OPENBLAS_NUM_THREADS`.
- Julia is column-major like tenferro-rs, so `omeinsum-jl` uses
  `format_string_colmajor` / `shapes_colmajor` directly; no PyTorch/JAX-style
  layout reconstruction is needed to preserve the same logical fixture
  values.
- **The BLAS provider dominates matmul-shaped rows.** OMEinsum dispatches
  pairwise contractions to Julia's own BLAS through libblastrampoline,
  which defaults to OpenBLAS, while tenferro-rs and PyTorch link whatever
  the profile selects (Accelerate on macOS). The provider that actually ran
  is recorded as `julia.blas_provider` in `run.yaml` alongside the other
  backends' providers, and the column is labeled "OMEinsum.jl OpenBLAS" for
  that reason. On an Apple-silicon host a 256x256 `float64` GEMM measured
  0.70 ms through Julia's OpenBLAS versus 0.92 ms for the same contraction
  through `DynamicEinCode` — i.e. OMEinsum adds roughly 30% over its own
  BLAS, and any larger gap against the tenferro/PyTorch columns on
  matmul-dominated instances is a BLAS-implementation difference, not
  einsum-runtime overhead. Read these rows together with the recorded
  providers rather than as a framework-versus-framework result.

