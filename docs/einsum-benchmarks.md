# Einsum Benchmark Usage

## Export benchmark metadata

```bash
uv run python scripts/generate_dataset.py
```

This selects instances by category with laptop-scale criteria and saves JSON metadata to `data/instances/`. `rnd_mixed_` instances are excluded because they are not yet supported by `tenferro-einsum`.

Selection criteria:

| Category | Prefix | log10[FLOPS] | log2[SIZE] | num_tensors | dtype |
|----------|--------|--------------|------------|-------------|-------|
| Language model | `lm_` | < 10 | < 25 | ≤ 100 | float64 or complex128 |
| Graphical model | `gm_` | < 10 | < 27 | ≤ 200 | float64 or complex128 |
| Structured | `str_` | < 11 | < 26 | ≤ 200 | float64 or complex128 |

## Compare tenferro-rs and Torch C++

This focused comparison runner is OpenBLAS-only and is intended for Linux/devcontainer runs that must include the Torch C++ column. macOS BLAS-backed CPU benchmarks use Accelerate and this runner stops instead of switching to OpenBLAS.

- `tenferro trace` with `system-openblas`
- `tenferro eager` with `system-openblas`
- `LibTorch CPU` with an OpenBLAS-linked `libtorch_cpu`

```bash
export OPENBLAS_ROOT=/opt/openblas
source ./scripts/setup_extern_deps.sh

./scripts/run_tenferro_libtorch.sh
./scripts/run_tenferro_libtorch.sh 4
```

For one instance:

```bash
BENCH_INSTANCE=bin_matmul_256 ./scripts/run_tenferro_libtorch.sh 1
```

The script writes:

- `data/results/tenferro_trace_t<threads>_<timestamp>.log`
- `data/results/tenferro_eager_t<threads>_<timestamp>.log`
- `data/results/libtorch_cpu_t<threads>_<timestamp>.log`
- `data/results/tenferro_libtorch_t<threads>_<timestamp>.md`

## Run all benchmarks

```bash
./scripts/run_all.sh
./scripts/run_all.sh 4
```

Runs all backends in sequence and writes a unified markdown comparison table:

- `tenferro trace/eager` via `scripts/run_all_rust.sh` (`system-accelerate` on macOS, `system-openblas` elsewhere by default)
- `strided-rs faer` via `scripts/run_all_rust.sh`, if `../strided-rs-benchmark-suite` exists
- `LibTorch CPU` via `scripts/run_all_libtorch.sh` only for OpenBLAS runs
- `PyTorch CPU` and `JAX CPU` via `scripts/run_all_python.sh`
- PR884 CPU benchmark items via `scripts/run_cpu_ops.sh`

For PR884 CPU benchmark items, `scripts/run_cpu_ops.sh` writes tenferro-rs
eager and trace rows from `publication_gate`, then appends Torch C++, PyTorch
Python, and JAX Python measurements to the same normalized CSV.

JAX CPU einsum rows use JAX's XLA CPU dot backend. They are not labeled as
Accelerate/OpenBLAS/MKL BLAS provider rows even when PyTorch and tenferro are
linked to an external BLAS provider.

The script sets `OMP_NUM_THREADS` and `RAYON_NUM_THREADS` to the given thread count. Raw logs, run metadata, and timestamped intermediate tables are saved to `data/results/cpu/einsum/<timestamp>/`. The latest human-facing reports are written to:

- `result/cpu/einsum.md`
- `result/cpu/cpu_ops.md`

Verify the generated report columns with:

```bash
rg -n "Torch C\\+\\+|PyTorch Python|JAX Python|XLA CPU|tenferro-rs" \
  result/cpu/einsum.md result/cpu/cpu_ops.md
```

Instance JSON files that fail to read or parse are skipped with a warning. Instances that trigger a backend error are reported as `SKIP` with the reason on stderr.
Instance JSON files may include optional top-level `intent` and `notes`
strings. They are human-facing metadata for why the benchmark exists and how to
interpret it; runners ignore them for timing, result identity, and cache keys.

## Run PyTorch and JAX baselines

`scripts/benchmark_python.py` supports both Python baselines:

```bash
uv run python scripts/benchmark_python.py --backend pytorch --num-threads 1
uv run python scripts/benchmark_python.py --backend jax --num-threads 1
```

The top-level `scripts/run_all.sh` includes both Python baselines in the formatted table.

## Run a single instance

Set `BENCH_INSTANCE` to the instance name:

```bash
BENCH_INSTANCE=gm_queen5_5_3.wcsp \
  cargo run --release --no-default-features --features system-accelerate

BENCH_INSTANCE=tensornetwork_permutation_light_415 \
  cargo run --release --no-default-features --features system-accelerate
```

With the full script:

```bash
BENCH_INSTANCE=gm_queen5_5_3.wcsp ./scripts/run_all.sh 1
BENCH_INSTANCE=tensornetwork_permutation_light_415 ./scripts/run_all.sh 4
```

Instance name must match the `name` field in the JSON, which is normally the filename without `.json`.

The tenferro Rust runner emits `PathAnalysis` / `PathWarning` lines for
contraction paths that create very large intermediates. These diagnostics are
ignored by `scripts/format_results.py`; they are there to identify path-driven
losses before considering kernel-level optimizations. Set
`TENFERRO_ANALYZE_PATH=1` to print the same path summary for every instance and
strategy, not only warning-sized paths.

To classify the latest Markdown report after a run:

```bash
uv run python scripts/analyze_einsum_gaps.py \
  --report result/cpu/einsum.md \
  --instances data/instances \
  --threshold 1.15
```

Rows above the threshold are labeled as `path_intermediate` when the
precomputed contraction path creates warning-sized intermediates, otherwise
`kernel_or_executor`. Ratio-only regressions whose absolute gap is below
`--min-delta-ms` (default: `0.5`) are labeled `small_absolute_gap`, which keeps
tiny microbenchmark differences from driving kernel work. Use this before
starting tenferro-rs or strided-rs kernel work; it prevents optimizing a kernel
for a benchmark that is actually dominated by path choice or too small in
absolute time to justify a new specialized path.

Current 1T reports should be triaged this way before adding more kernels:

- `kernel_or_executor`: investigate implementation overhead first.
- `path_intermediate`: compare contraction strategies or path planning first.
- `small_absolute_gap`: do not optimize unless the gap also shows up in a
  larger instance or another workload.

For example, the current `gm_queen5_5_3.wcsp / opt_flops` row is a
`path_intermediate` case. With `TENFERRO_ANALYZE_PATH=1`, the runner reports a
rank-17 maximum intermediate, `129140163` maximum intermediate elements, and
`592127821` total intermediate elements. The `opt_size` path uses the same
instance but caps the maximum intermediate at `43046721` elements, so this is a
path-selection signal rather than evidence for another einsum microkernel.

## Binary diagnostic instances

For bottleneck investigation, this repo includes a small binary-only set:

- `bin_matmul_256` (`ij,jk->ik`)
- `bin_batched_matmul_b32_m64_n64_k64` (`bij,bjk->bik`)
- `bin_batched_matmul_b32_m128_n128_k128` (`bij,bjk->bik`)
- `bin_outer_product_4096` (`i,j->ij`)
- `bin_elementwise_mul_2048x2048` (`ij,ij->ij`)

These are intended to measure binary einsum dispatch, layout, and wrapper
overhead. In particular, `bin_matmul_256` is the focused case for binary
GEMM-shaped eager/traced fast-path work.

Recommended invocation for consistent single-thread profiling:

```bash
RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  BENCH_INSTANCE=bin_matmul_256 \
  cargo run --release --no-default-features --features system-accelerate
```

## N-ary diagnostic instances

For extension-runtime cache investigation, this repo also includes small N-ary
cases whose cost is low enough that runtime planning and inner compilation
overhead remain visible:

- `nary_matmul_chain_64` (`ij,jk,kl->il`)

This case is not covered by a binary-only fast path. It exercises the traced
einsum extension runtime and is intended to measure caching of the inner
`ExecProgram` produced from the extension's lowered graph.

## Publication-gate microbenchmarks

`publication_gate` is a repo-local benchmark runner for the focused matrix in [tensor4all/tenferro-rs#862](https://github.com/tensor4all/tenferro-rs/issues/862). It is separate from the einsum macrobenchmark suite and covers:

- Small matrix latency: `2x2` through `32x32` matmul/einsum, `svd`, `qr`, `eigh`, `solve`, and AD cases.
- Large matrix throughput: `128` through `1024` matmul, medium/large linalg, and separate primal/backward timing rows for AD workloads.
- Batched small matrices: batched matmul, batched `svd`/`qr`/`eigh`/`solve`, and batched AD for matmul/solve.

The batched rows use each backend's native batch layout. tenferro-rs stores batch axes as trailing dimensions, so its batched matmul case is encoded as `ikb,kjb->ijb`, where the rightmost `b` index is the batch index. Torch, LibTorch, and JAX use row-major tensors with leading batch axes, so they run the same workload as `bij,bjk->bik`.

Run the quick profile:

```bash
./scripts/run_publication_gate.sh 1
```

The full profile includes the complete issue matrix:

```bash
PUBLICATION_GATE_PROFILE=full ./scripts/run_publication_gate.sh 1
```

Select one suite when iterating:

```bash
PUBLICATION_GATE_SUITE=small ./scripts/run_publication_gate.sh 1
PUBLICATION_GATE_SUITE=large ./scripts/run_publication_gate.sh 1
PUBLICATION_GATE_SUITE=batched ./scripts/run_publication_gate.sh 1
```

Compare tenferro CPU backends:

```bash
PUBLICATION_GATE_FEATURES=cpu-faer ./scripts/run_publication_gate.sh 1
PUBLICATION_GATE_FEATURES=system-accelerate ./scripts/run_publication_gate.sh 1
PUBLICATION_GATE_FEATURES=system-openblas ./scripts/run_publication_gate.sh 1
PUBLICATION_GATE_FEATURES=cuda ./scripts/run_publication_gate.sh 1
```

The runner writes CSV files under `data/results/` with one row per case:

```text
suite,op,phase,dtype,backend,profile,shape,warmups,runs,median_ms,iqr_ms,status
```

`phase=primal` measures forward execution. `phase=backward` constructs the forward expression and times reverse-mode execution separately for the AD rows. The benchmark currently emits `f64` rows; C64 should be added as a separate extension once the real-valued baseline is stable.

## Benchmark instances

Instances are from the [einsum benchmark](https://benchmark.einsum.org/) suite. Selection is per-category; dtype is float64 or complex128; tensors are zero-filled at runtime.

| Instance | Category | Tensors | Steps | log10(FLOPS) | log2(SIZE) |
|----------|----------|--------:|------:|-------------:|------------:|
| `bin_matmul_256` | Binary (diagnostic) | 2 | 1 | — | — |
| `bin_batched_matmul_b32_m64_n64_k64` | Binary (diagnostic) | 2 | 1 | — | — |
| `bin_batched_matmul_b32_m128_n128_k128` | Binary (diagnostic) | 2 | 1 | 7.83 | 19.0 |
| `bin_outer_product_4096` | Binary (diagnostic) | 2 | 1 | — | — |
| `bin_elementwise_mul_2048x2048` | Binary (diagnostic) | 2 | 1 | — | — |
| `nary_matmul_chain_64` | N-ary (diagnostic) | 3 | 2 | 5.24 | 12.0 |
| `gm_queen5_5_3.wcsp` | Graphical model | 160 | 159 | 9.75 | 26.94 |
| `lm_batch_likelihood_brackets_4_4d` | Language model | 84 | 83 | 8.37 | 18.96 |
| `lm_batch_likelihood_sentence_3_12d` | Language model | 38 | 37 | 9.20 | 20.86 |
| `lm_batch_likelihood_sentence_4_4d` | Language model | 84 | 83 | 8.46 | 18.89 |
| `str_matrix_chain_multiplication_100` | Structured | 100 | 99 | 8.48 | 17.26 |
| `str_mps_varying_inner_product_200` | Structured (MPS) | 200 | 199 | 8.31 | 15.48 |
| `str_nw_mera_closed_120` | Structured (MERA) | 120 | 119 | 10.66 | 25.02 |
| `str_nw_mera_open_26` | Structured (MERA) | 26 | 25 | 10.49 | 25.36 |
| `tensornetwork_permutation_light_415` | Tensor network | 415 | 414 | 9.65 | 24.0 |
| `tensornetwork_permutation_focus_step409_316` | Tensor network (focused) | 316 | 315 | 9.65 | 24.0 |

Notes:

- `gm_*`: WCSP / constraint networks; many small 2D factors, full contraction to scalar.
- `lm_*`: many small multi-dimensional tensors with large batch dimensions.
- `str_matrix_chain_*`: large 2D matrices; each step is one large GEMM.
- `str_mps_*`: matrix product state-style networks.
- `str_nw_mera_*`: tensor networks from multi-scale entanglement renormalization.
- `tensornetwork_permutation_light_415`: 415 tensors extracted from the full TensorNetworkBenchmarks instance via BFS-connected subgraph.
- `tensornetwork_permutation_focus_step409_316`: focused subtree for profiling late bottleneck steps.
- `intent`: optional per-instance design note. It should describe what overhead,
  dispatch path, shape regime, or regression target the benchmark is meant to
  expose.
