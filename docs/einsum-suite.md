# Einsum Suite and Instance Selection

## Source

The macrobenchmark instances come from the
[einsum benchmark](https://benchmark.einsum.org/) via the
`einsum_benchmark` Python package. `scripts/generate_dataset.py` exports
row-major metadata from the source package and adds column-major metadata for
tenferro-rs.

Generate the source-derived instance JSON files:

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

