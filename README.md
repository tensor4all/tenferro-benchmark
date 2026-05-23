# tenferro-einsum-benchmark

Benchmark suite for [tenferro-rs](https://github.com/tensor4all/tenferro-rs) based on the [einsum benchmark](https://benchmark.einsum.org/) (168 standardized einsum problems across 7 categories).

## Overview

This repository provides:

- A Python pipeline to extract einsum benchmark metadata (shapes, dtypes, contraction paths) into portable JSON
- A **Rust** benchmark runner using [tenferro-einsum](https://github.com/tensor4all/tenferro-rs)
- **Python** benchmark runners using **PyTorch** and **JAX** for comparison

Only metadata is stored — tensors are generated at benchmark time (zero-filled), keeping the repo lightweight.

The structure mirrors [strided-rs-benchmark-suite](https://github.com/tensor4all/strided-rs-benchmark-suite) for consistency and cross-comparison.

See [tensor4all/tenferro-rs](https://github.com/tensor4all/tenferro-rs) for the full library.

## Project Structure

```
tenferro-einsum-benchmark/
  src/
    main.rs                 # Rust benchmark runner entry point
    lib.rs                  # Shared compilation & evaluation helpers (compile_einsum, reorder_user_operands)
  scripts/
    run_all.sh              # Top-level orchestrator (Rust + Python)
    run_all_rust.sh         # Build & run tenferro-einsum + strided-rs (faer)
    run_all_python.sh       # Run PyTorch CPU + JAX CPU benchmarks
    benchmark_python.py     # Python benchmark runner (PyTorch / JAX)
    generate_dataset.py     # Filter & export benchmark instances as JSON
    format_results.py       # Parse logs and output unified markdown tables
  data/
    instances/              # Exported JSON metadata (one file per instance)
    results/                # Benchmark logs and markdown results
  Cargo.toml                # Rust project
  pyproject.toml            # Python project (includes torch, jax, opt_einsum)
```

## Setup

### Python (dataset export)

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

### Rust

Requires a local clone of [tenferro-rs](https://github.com/tensor4all/tenferro-rs) at `../tenferro-rs`.

```bash
cargo build --release
```

## Usage

### 1. Export benchmark metadata

```bash
uv run python scripts/generate_dataset.py
```

This selects instances by category with laptop-scale criteria and saves JSON metadata to `data/instances/`. `rnd_mixed_` instances are excluded (not yet supported by tenferro-einsum).

**Selection criteria (per category):**

| Category | Prefix | log10[FLOPS] | log2[SIZE] | num_tensors | dtype |
|----------|--------|--------------|------------|-------------|-------|
| Language model | `lm_` | < 10 | < 25 | ≤ 100 | float64 or complex128 |
| Graphical model | `gm_` | < 10 | < 27 | ≤ 200 | float64 or complex128 |
| Structured | `str_` | < 11 | < 26 | ≤ 200 | float64 or complex128 |

### 2. Run all benchmarks

```bash
./scripts/run_all.sh          # 1 thread (default)
./scripts/run_all.sh 4        # 4 threads
```

Runs all backends in sequence and writes a unified markdown comparison table:

- **tenferro-einsum** (Rust) — via `scripts/run_all_rust.sh`
- **strided-rs faer** (Rust, optional) — requires `../strided-rs-benchmark-suite`
- **PyTorch CPU** (Python) — via `scripts/run_all_python.sh`
- **JAX CPU** (Python) — via `scripts/run_all_python.sh`

Sets `OMP_NUM_THREADS` and `RAYON_NUM_THREADS` to the given thread count (default: 1). Saves all logs and the markdown table to `data/results/` with timestamps.

Instance JSON files that fail to read or parse are skipped with a warning; the suite continues with the rest. Instances that trigger a backend error are reported as **SKIP** with the reason on stderr.

Requires `strided-rs-benchmark-suite` at `../strided-rs-benchmark-suite` for strided-rs comparison (optional; skipped with a note if not found).

### 3. Run a single instance

To run the benchmark for **one instance only**, set the environment variable `BENCH_INSTANCE` to the instance name:

```bash
BENCH_INSTANCE=gm_queen5_5_3.wcsp cargo run --release
BENCH_INSTANCE=tensornetwork_permutation_light_415 cargo run --release
```

**With the full script:**

```bash
BENCH_INSTANCE=gm_queen5_5_3.wcsp ./scripts/run_all.sh 1
BENCH_INSTANCE=tensornetwork_permutation_light_415 ./scripts/run_all.sh 4
```

Instance name must match the `name` field in the JSON (i.e. the filename without `.json`). To list available names: `ls data/instances/` → e.g. `gm_queen5_5_3.wcsp.json` → use `gm_queen5_5_3.wcsp`.

### 4. Binary Einsum Diagnostic Instances

For bottleneck investigation, this repo includes a small binary-only set:

- `bin_matmul_256` (`ij,jk->ik`) — dense matrix multiplication baseline
- `bin_batched_matmul_b32_m64_n64_k64` (`bij,bjk->bik`) — batched GEMM with moderate batch size
- `bin_outer_product_4096` (`i,j->ij`) — outer-product path (sensitive to broadcast/pack overhead)
- `bin_elementwise_mul_2048x2048` (`ij,ij->ij`) — pure elementwise multiplication throughput

Recommended invocation for consistent single-thread profiling:

```bash
RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  BENCH_INSTANCE=bin_matmul_256 cargo run --release
```

These are intended for quick, reproducible profiling before running heavier LM/structured/network cases.

### 5. Run individually

```bash
RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 cargo run --release
```

### 6. Format existing logs

```bash
uv run python scripts/format_results.py data/results/tenferro_einsum_*.log
```

## Reproducing Benchmarks

```bash
./scripts/run_all.sh        # 1 thread
./scripts/run_all.sh 4      # 4 threads
```

This script:

1. Delegates to `scripts/run_all_rust.sh` (tenferro-einsum + optionally strided-rs faer)
2. Delegates to `scripts/run_all_python.sh` (PyTorch CPU + JAX CPU)
3. Sets `OMP_NUM_THREADS` and `RAYON_NUM_THREADS` to the given thread count (default: 1)
4. Formats all collected logs as a unified markdown table via `scripts/format_results.py`
5. Saves all outputs to `data/results/` with timestamps

## Benchmark Instances

Instances are from the [einsum benchmark](https://benchmark.einsum.org/) suite. Selection is per-category (see [Export benchmark metadata](#1-export-benchmark-metadata)); dtype is float64 or complex128; tensors are zero-filled at runtime.

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

- **Graphical model (gm_*)**: WCSP / constraint networks; many small 2D factors (e.g. 3×3), full contraction to scalar.
- **Language model (lm_*)**: many small multi-dimensional tensors (3D/4D) with large batch dimensions; many steps with small GEMM kernels.
- **Structured — matrix chain (str_matrix_chain_*)**: large 2D matrices; each step is one large GEMM.
- **Structured — MPS (str_mps_*)**: matrix product state–style networks; varying inner dimensions, many 2D contractions.
- **Structured — MERA (str_nw_mera_*)**: tensor networks from multi-scale entanglement renormalization; many small 3×3-like tensors, heavy contraction.
- **Tensor network (tensornetwork_permutation_light_415)**: 415 tensors extracted from the full TensorNetworkBenchmarks instance via BFS-connected subgraph.
- **Tensor network focused (tensornetwork_permutation_focus_step409_316)**: focused subtree for profiling the late bottleneck steps (408/409); 316 tensors with non-scalar output (rank-18).

## Benchmark Results

### Apple Silicon M4

#### Strategy: opt_flops

Median ± IQR (ms). OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-einsum (ms) | strided-rs faer (ms) |
|---|---:|---:|
| bin_batched_matmul_b32_m64_n64_k64 | 1.505 ± 0.746 | **1.085 ± 0.366** |
| bin_elementwise_mul_2048x2048 | 1.729 ± 0.106 | **1.315 ± 0.093** |
| bin_matmul_256 | **0.538 ± 0.025** | 0.617 ± 0.016 |
| bin_outer_product_4096 | 4.409 ± 0.880 | **2.114 ± 0.056** |
| gm_queen5_5_3.wcsp | **1366.614 ± 45.159** | 1719.183 ± 10.071 |
| lm_batch_likelihood_brackets_4_4d | **9.831 ± 0.299** | 9.901 ± 0.758 |
| lm_batch_likelihood_sentence_3_12d | 34.342 ± 1.804 | **32.972 ± 0.110** |
| lm_batch_likelihood_sentence_4_4d | 12.180 ± 0.453 | **10.660 ± 0.085** |
| str_matrix_chain_multiplication_100 | **8.580 ± 0.121** | 8.771 ± 0.130 |
| str_mps_varying_inner_product_200 | **7.488 ± 0.166** | 8.781 ± 0.047 |
| str_nw_mera_closed_120 | **860.603 ± 26.606** | 895.693 ± 1.960 |
| str_nw_mera_open_26 | 564.104 ± 13.562 | **553.778 ± 1.552** |
| tensornetwork_permutation_focus_step409_316 | 179.686 ± 1.070 | **165.224 ± 0.533** |
| tensornetwork_permutation_light_415 | 181.119 ± 1.587 | **166.615 ± 0.527** |

#### Strategy: opt_size

Median ± IQR (ms). OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro-einsum (ms) | strided-rs faer (ms) |
|---|---:|---:|
| bin_batched_matmul_b32_m64_n64_k64 | 0.500 ± 0.035 | **0.498 ± 0.001** |
| bin_elementwise_mul_2048x2048 | 1.795 ± 0.164 | **1.214 ± 0.047** |
| bin_matmul_256 | **0.575 ± 0.005** | 0.577 ± 0.017 |
| bin_outer_product_4096 | 3.273 ± 0.080 | **2.119 ± 0.025** |
| gm_queen5_5_3.wcsp | 846.545 ± 7.491 | **667.801 ± 3.780** |
| lm_batch_likelihood_brackets_4_4d | 11.728 ± 0.596 | **10.293 ± 0.056** |
| lm_batch_likelihood_sentence_3_12d | 36.700 ± 3.756 | **34.509 ± 0.129** |
| lm_batch_likelihood_sentence_4_4d | 12.441 ± 0.644 | **11.235 ± 0.051** |
| str_matrix_chain_multiplication_100 | 8.985 ± 0.574 | **8.689 ± 0.044** |
| str_mps_varying_inner_product_200 | **7.365 ± 0.201** | 8.577 ± 0.027 |
| str_nw_mera_closed_120 | 873.335 ± 43.807 | **852.160 ± 16.226** |
| str_nw_mera_open_26 | 567.437 ± 10.611 | **555.803 ± 3.270** |
| tensornetwork_permutation_focus_step409_316 | 182.351 ± 9.294 | **164.994 ± 0.291** |
| tensornetwork_permutation_light_415 | 182.904 ± 1.304 | **166.532 ± 0.352** |

#### Strategy: opt_flops

Median ± IQR (ms). OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-einsum (ms) | strided-rs faer (ms) |
|---|---:|---:|
| bin_batched_matmul_b32_m64_n64_k64 | 1.123 ± 0.336 | **1.113 ± 0.357** |
| bin_elementwise_mul_2048x2048 | 1.597 ± 0.077 | **1.341 ± 0.066** |
| bin_matmul_256 | 0.222 ± 0.028 | **0.200 ± 0.002** |
| bin_outer_product_4096 | 3.388 ± 0.130 | **1.875 ± 0.161** |
| gm_queen5_5_3.wcsp | **1274.833 ± 6.605** | 1828.192 ± 13.416 |
| lm_batch_likelihood_brackets_4_4d | **7.313 ± 0.099** | 8.033 ± 0.078 |
| lm_batch_likelihood_sentence_3_12d | 17.477 ± 1.159 | **17.012 ± 1.098** |
| lm_batch_likelihood_sentence_4_4d | **7.681 ± 0.180** | 8.757 ± 0.171 |
| str_matrix_chain_multiplication_100 | **5.931 ± 0.343** | 6.221 ± 0.234 |
| str_mps_varying_inner_product_200 | **8.081 ± 0.281** | 9.353 ± 0.235 |
| str_nw_mera_closed_120 | **306.992 ± 7.956** | 307.959 ± 3.124 |
| str_nw_mera_open_26 | 196.437 ± 2.539 | **195.246 ± 1.538** |
| tensornetwork_permutation_focus_step409_316 | **87.775 ± 0.415** | 119.616 ± 1.128 |
| tensornetwork_permutation_light_415 | **88.735 ± 2.428** | 121.402 ± 1.925 |

#### Strategy: opt_size

Median ± IQR (ms). OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-einsum (ms) | strided-rs faer (ms) |
|---|---:|---:|
| bin_batched_matmul_b32_m64_n64_k64 | 0.542 ± 0.008 | **0.508 ± 0.009** |
| bin_elementwise_mul_2048x2048 | 1.549 ± 0.128 | **1.232 ± 0.041** |
| bin_matmul_256 | **0.214 ± 0.078** | 0.236 ± 0.077 |
| bin_outer_product_4096 | 3.106 ± 0.197 | **1.836 ± 0.094** |
| gm_queen5_5_3.wcsp | **443.578 ± 2.611** | 516.313 ± 10.647 |
| lm_batch_likelihood_brackets_4_4d | **7.981 ± 0.308** | 9.226 ± 0.309 |
| lm_batch_likelihood_sentence_3_12d | 18.517 ± 0.992 | **17.840 ± 1.153** |
| lm_batch_likelihood_sentence_4_4d | **7.747 ± 0.170** | 9.661 ± 0.125 |
| str_matrix_chain_multiplication_100 | **6.077 ± 0.357** | 6.223 ± 0.329 |
| str_mps_varying_inner_product_200 | **8.066 ± 0.111** | 9.219 ± 0.106 |
| str_nw_mera_closed_120 | **291.378 ± 2.231** | 312.743 ± 3.406 |
| str_nw_mera_open_26 | **196.197 ± 2.475** | 197.780 ± 1.954 |
| tensornetwork_permutation_focus_step409_316 | **88.133 ± 1.545** | 119.643 ± 1.092 |
| tensornetwork_permutation_light_415 | **89.034 ± 1.891** | 121.721 ± 3.546 |

**Notes:**
- **strided-rs faer** uses [faer](https://github.com/sarah-quinones/faer-rs) (pure Rust GEMM).
- Both backends use the same pre-computed contraction path for fair comparison.

### Python Backend Comparison (4 threads, tenferro-einsum vs PyTorch vs JAX)

#### Strategy: opt_flops

Median ± IQR (ms). OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-einsum (ms) | PyTorch CPU (ms) | JAX CPU (ms) |
|---|---:|---:|---:|
| bin_batched_matmul_b32_m64_n64_k64 | 1.222 ± 0.171 | **0.137 ± 0.007** | 0.255 ± 0.078 |
| bin_elementwise_mul_2048x2048 | 1.437 ± 0.040 | **1.144 ± 0.024** | 1.808 ± 0.288 |
| bin_matmul_256 | 0.244 ± 0.020 | **0.114 ± 0.005** | 0.319 ± 0.046 |
| bin_outer_product_4096 | 11.497 ± 0.233 | **1.127 ± 0.040** | 1.231 ± 0.035 |
| gm_queen5_5_3.wcsp | 1671.237 ± 27.613 | **511.059 ± 12.615** | 855.035 ± 7.509 |
| lm_batch_likelihood_brackets_4_4d | 13.300 ± 0.294 | 12.125 ± 0.176 | **8.966 ± 0.240** |
| lm_batch_likelihood_sentence_3_12d | 18.786 ± 0.330 | 20.495 ± 0.221 | **11.320 ± 0.155** |
| lm_batch_likelihood_sentence_4_4d | 11.990 ± 0.332 | 12.329 ± 0.397 | **8.962 ± 0.204** |
| str_matrix_chain_multiplication_100 | 7.039 ± 0.194 | **4.518 ± 0.170** | 9.316 ± 0.244 |
| str_mps_varying_inner_product_200 | 10.761 ± 0.144 | **8.359 ± 0.192** | 19.889 ± 0.221 |
| str_nw_mera_closed_120 | 329.697 ± 2.670 | **125.312 ± 6.207** | 163.553 ± 2.854 |
| str_nw_mera_open_26 | 283.915 ± 2.472 | **89.812 ± 0.860** | 129.246 ± 3.767 |
| tensornetwork_permutation_focus_step409_316 | 192.004 ± 2.471 | **103.624 ± 7.313** | 107.206 ± 5.215 |
| tensornetwork_permutation_light_415 | 193.898 ± 1.053 | **94.805 ± 1.543** | 116.222 ± 5.823 |

#### Strategy: opt_size

Median ± IQR (ms). OMP_NUM_THREADS=4, RAYON_NUM_THREADS=4.

| Instance | tenferro-einsum (ms) | PyTorch CPU (ms) | JAX CPU (ms) |
|---|---:|---:|---:|
| bin_batched_matmul_b32_m64_n64_k64 | 0.650 ± 0.015 | **0.128 ± 0.004** | 0.230 ± 0.061 |
| bin_elementwise_mul_2048x2048 | 1.135 ± 0.081 | **0.878 ± 0.198** | 1.818 ± 0.258 |
| bin_matmul_256 | 0.266 ± 0.035 | **0.111 ± 0.003** | 0.321 ± 0.018 |
| bin_outer_product_4096 | 11.854 ± 0.079 | **1.126 ± 0.035** | 2.206 ± 0.801 |
| gm_queen5_5_3.wcsp | 670.476 ± 3.256 | **246.509 ± 11.433** | 302.771 ± 10.642 |
| lm_batch_likelihood_brackets_4_4d | 12.863 ± 0.256 | 13.181 ± 0.680 | **8.970 ± 0.142** |
| lm_batch_likelihood_sentence_3_12d | 22.945 ± 0.380 | 21.198 ± 0.603 | **12.293 ± 0.334** |
| lm_batch_likelihood_sentence_4_4d | 13.129 ± 0.398 | 12.082 ± 0.155 | **7.992 ± 0.118** |
| str_matrix_chain_multiplication_100 | 7.362 ± 0.723 | **4.656 ± 0.192** | 9.349 ± 0.298 |
| str_mps_varying_inner_product_200 | 12.729 ± 0.288 | **10.283 ± 0.147** | 22.175 ± 0.526 |
| str_nw_mera_closed_120 | 316.895 ± 1.899 | **105.094 ± 0.560** | 152.457 ± 2.655 |
| str_nw_mera_open_26 | 283.695 ± 1.514 | **92.795 ± 0.845** | 133.967 ± 3.920 |
| tensornetwork_permutation_focus_step409_316 | 193.855 ± 1.294 | **90.619 ± 11.373** | 109.681 ± 6.701 |
| tensornetwork_permutation_light_415 | 196.092 ± 2.175 | **94.967 ± 0.863** | 113.566 ± 3.765 |

**Notes:**
- **PyTorch CPU** uses [torch.einsum](https://pytorch.org/docs/stable/generated/torch.einsum.html) via [opt_einsum](https://optimized-einsum.readthedocs.io/) with pre-computed paths.
- **JAX CPU** uses [jax.numpy.einsum](https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.einsum.html) via opt_einsum; `jax_enable_x64=True`; `block_until_ready()` ensures timing accuracy.
- All backends use the same pre-computed contraction path for fair comparison.

## References

- [Einsum Benchmark](https://benchmark.einsum.org/) — standardized einsum benchmark suite
- [ti2-group/einsum_benchmark](https://github.com/ti2-group/einsum_benchmark) — Python package
- [tensor4all/tenferro-rs](https://github.com/tensor4all/tenferro-rs) — Rust tensor library
- [tensor4all/strided-rs-benchmark-suite](https://github.com/tensor4all/strided-rs-benchmark-suite) — sibling benchmark suite for strided-rs

## License

MIT
# tenferro-benchmark
