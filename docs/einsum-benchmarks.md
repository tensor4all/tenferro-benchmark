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

This focused comparison runner runs:

- `tenferro trace` with `system-openblas`
- `tenferro eager` with `system-openblas`
- `LibTorch CPU` with an OpenBLAS-linked `libtorch_cpu`

```bash
export OPENBLAS_ROOT=/opt/homebrew/opt/openblas
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
export OPENBLAS_ROOT=/opt/homebrew/opt/openblas

./scripts/run_all.sh
./scripts/run_all.sh 4
```

Runs all backends in sequence and writes a unified markdown comparison table:

- `tenferro trace/eager` via `scripts/run_all_rust.sh`
- `strided-rs faer` via `scripts/run_all_rust.sh`, if `../strided-rs-benchmark-suite` exists
- `LibTorch CPU` via `scripts/run_all_libtorch.sh`
- `PyTorch CPU` and `JAX CPU` via `scripts/run_all_python.sh`
- PR884 CPU benchmark items via `scripts/run_cpu_ops.sh`

For PR884 CPU benchmark items, `scripts/run_cpu_ops.sh` writes tenferro-rs
eager and trace rows from `publication_gate`, then appends Torch C++, PyTorch
Python, and JAX Python measurements to the same normalized CSV.

The script sets `OMP_NUM_THREADS` and `RAYON_NUM_THREADS` to the given thread count. Raw logs and timestamped intermediate tables are saved to `data/results/`. The latest human-facing reports are written to:

- `result/results-einsum.md`
- `result/cpu-benchmark-results.md`

Verify the generated report columns with:

```bash
rg -n "Torch C\\+\\+|PyTorch Python|JAX Python|tenferro-rs" \
  result/results-einsum.md result/cpu-benchmark-results.md
```

Instance JSON files that fail to read or parse are skipped with a warning. Instances that trigger a backend error are reported as `SKIP` with the reason on stderr.

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
OPENBLAS_ROOT=/opt/homebrew/opt/openblas \
  BENCH_INSTANCE=gm_queen5_5_3.wcsp \
  cargo run --release --no-default-features --features system-openblas

OPENBLAS_ROOT=/opt/homebrew/opt/openblas \
  BENCH_INSTANCE=tensornetwork_permutation_light_415 \
  cargo run --release --no-default-features --features system-openblas
```

With the full script:

```bash
export OPENBLAS_ROOT=/opt/homebrew/opt/openblas

BENCH_INSTANCE=gm_queen5_5_3.wcsp ./scripts/run_all.sh 1
BENCH_INSTANCE=tensornetwork_permutation_light_415 ./scripts/run_all.sh 4
```

Instance name must match the `name` field in the JSON, which is normally the filename without `.json`.

## Binary diagnostic instances

For bottleneck investigation, this repo includes a small binary-only set:

- `bin_matmul_256` (`ij,jk->ik`)
- `bin_batched_matmul_b32_m64_n64_k64` (`bij,bjk->bik`)
- `bin_outer_product_4096` (`i,j->ij`)
- `bin_elementwise_mul_2048x2048` (`ij,ij->ij`)

Recommended invocation for consistent single-thread profiling:

```bash
OPENBLAS_ROOT=/opt/homebrew/opt/openblas \
  RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  BENCH_INSTANCE=bin_matmul_256 \
  cargo run --release --no-default-features --features system-openblas
```

## Publication-gate microbenchmarks

`publication_gate` is a repo-local benchmark runner for the focused matrix in [tensor4all/tenferro-rs#862](https://github.com/tensor4all/tenferro-rs/issues/862). It is separate from the einsum macrobenchmark suite and covers:

- Small matrix latency: `2x2` through `32x32` matmul/einsum, `svd`, `qr`, `eigh`, `solve`, and AD cases.
- Large matrix throughput: `128` through `1024` matmul, medium/large linalg, and separate primal/backward timing rows for AD workloads.
- Batched small matrices: batched matmul, batched `svd`/`qr`/`eigh`/`solve`, and batched AD for matmul/solve.

tenferro-rs stores batch axes as trailing dimensions. Therefore the batched matmul case is encoded as `ikb,kjb->ijb`, where the rightmost `b` index is the batch index. This is the same workload as the issue's `bij,bjk->bik` example, expressed in tenferro's col-major/trailing-batch convention.

Run the quick profile:

```bash
export OPENBLAS_ROOT=/opt/homebrew/opt/openblas

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
| `bin_outer_product_4096` | Binary (diagnostic) | 2 | 1 | — | — |
| `bin_elementwise_mul_2048x2048` | Binary (diagnostic) | 2 | 1 | — | — |
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
