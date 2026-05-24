# tenferro-benchmark

Benchmark suite for [tenferro-rs](https://github.com/tensor4all/tenferro-rs) based on the [einsum benchmark](https://benchmark.einsum.org/).

This repository contains:

- Rust benchmark runner for `tenferro-einsum`
- C++ LibTorch benchmark runner for Torch CPU comparison
- Python benchmark runner for PyTorch and JAX CPU comparison
- Result formatting tools for unified markdown tables

Detailed documentation was split out of this README:

- [Setup and C++ LibTorch build](docs/setup-and-libtorch.md)
- [Einsum benchmark usage and instances](docs/einsum-benchmarks.md)
- Latest generated einsum benchmark results: `result/results-einsum.md`
- Latest generated CPU benchmark results: `result/cpu-benchmark-results.md`

## Quick Start

Prerequisites:

- Rust and Cargo
- CMake 3.20+
- Python 3.12+ and [uv](https://docs.astral.sh/uv/)
- OpenBLAS
- Repo-local external checkouts under `extern/`

```bash
brew install openblas cmake
export OPENBLAS_ROOT="$(brew --prefix openblas)"
uv sync
./scripts/setup_extern_deps.sh
```

`scripts/setup_extern_deps.sh` prepares `extern/tenferro-rs` and an OpenBLAS-linked PyTorch/LibTorch checkout at `extern/pytorch-openblas`. `scripts/run_all.sh` sources it automatically before running benchmarks. See [Setup and C++ LibTorch build](docs/setup-and-libtorch.md) for the full LibTorch/OpenBLAS build instructions.

To remove these repo-local external checkouts:

```bash
./scripts/clean_extern_deps.sh
```

## Torch C++ Benchmark Workflow

Use this workflow when benchmark results must include the Torch C++ column.

```bash
export OPENBLAS_ROOT="$(brew --prefix openblas)"
./scripts/setup_extern_deps.sh
```

Verify that LibTorch is linked to OpenBLAS before trusting Torch C++ numbers:

```bash
otool -L extern/pytorch-openblas/torch/lib/libtorch_cpu.dylib | rg -i openblas
```

Run a quick end-to-end smoke that exercises tenferro-rs, Torch C++, PyTorch Python, JAX Python, and PR884 CPU table generation. The CPU table measures tenferro-rs eager mode, tenferro-rs trace mode, Torch C++, PyTorch Python, and JAX Python rows.

```bash
BENCH_INSTANCE=bin_matmul_256 \
BENCH_RUNS=1 \
BENCH_WARMUPS=0 \
PUBLICATION_GATE_SUITE=small \
  ./scripts/run_all.sh 1
```

For normal benchmark results:

```bash
./scripts/run_all.sh 1
./scripts/run_all.sh 4
```

The generated reports should include the comparison columns:

```bash
rg -n "Torch C\\+\\+|PyTorch Python|JAX Python|tenferro-rs" \
  result/results-einsum.md result/cpu-benchmark-results.md
```

## Human-Run Scripts

The snippets below are intended to be copied and run by a human from the repository root.

### Generate benchmark metadata

```bash
uv run python scripts/generate_dataset.py
```

### Verify local builds

```bash
cmake -S cpp -B build/cpp-plan-test -DBUILD_LIBTORCH_BENCHMARK=OFF
cmake --build build/cpp-plan-test --target einsum_plan_test
ctest --test-dir build/cpp-plan-test --output-on-failure

OPENBLAS_ROOT=/opt/homebrew/opt/openblas \
  cargo check --no-default-features --features system-openblas
```

### Run tenferro-rs vs C++ Torch

```bash
export OPENBLAS_ROOT=/opt/homebrew/opt/openblas
source ./scripts/setup_extern_deps.sh

./scripts/run_tenferro_libtorch.sh 1
```

Run one instance:

```bash
export OPENBLAS_ROOT=/opt/homebrew/opt/openblas
source ./scripts/setup_extern_deps.sh

BENCH_INSTANCE=bin_matmul_256 ./scripts/run_tenferro_libtorch.sh 1
```

### Run the main benchmark suite

This runs tenferro-rs trace/eager, optional strided-rs faer, C++ LibTorch, Python PyTorch, JAX, and the PR884 CPU benchmark items. PR884 CPU-op rows include measured tenferro-rs eager mode, tenferro-rs trace mode, Torch C++, PyTorch Python, and JAX Python values.

```bash
export OPENBLAS_ROOT=/opt/homebrew/opt/openblas

./scripts/run_all.sh 1
./scripts/run_all.sh 4
```

Raw logs and timestamped tables are written to `data/results/`, summarized by `scripts/format_results.py`, and copied to:

- `result/results-einsum.md`
- `result/cpu-benchmark-results.md`

### Run PyTorch and JAX Python baselines manually

`scripts/run_all.sh` runs both Python baselines automatically. To run and format the backend logs manually:

```bash
export BENCHMARK_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
export OMP_NUM_THREADS=1
export RAYON_NUM_THREADS=1
export OPENBLAS_ROOT=/opt/homebrew/opt/openblas
source ./scripts/setup_extern_deps.sh

./scripts/run_all_rust.sh 1
./scripts/run_all_libtorch.sh 1

uv run python scripts/benchmark_python.py --backend pytorch --num-threads 1 \
  2>&1 | tee "data/results/pytorch_cpu_t1_${BENCHMARK_TIMESTAMP}.log"

uv run python scripts/benchmark_python.py --backend jax --num-threads 1 \
  2>&1 | tee "data/results/jax_cpu_t1_${BENCHMARK_TIMESTAMP}.log"

uv run python scripts/format_results.py \
  "data/results/tenferro_trace_t1_${BENCHMARK_TIMESTAMP}.log" \
  "data/results/tenferro_eager_t1_${BENCHMARK_TIMESTAMP}.log" \
  "data/results/libtorch_cpu_t1_${BENCHMARK_TIMESTAMP}.log" \
  "data/results/pytorch_cpu_t1_${BENCHMARK_TIMESTAMP}.log" \
  "data/results/jax_cpu_t1_${BENCHMARK_TIMESTAMP}.log" \
  | tee "data/results/results_four_backends_t1_${BENCHMARK_TIMESTAMP}.md"
```

### Run publication-gate microbenchmarks

```bash
export OPENBLAS_ROOT=/opt/homebrew/opt/openblas

./scripts/run_publication_gate.sh 1

PUBLICATION_GATE_PROFILE=full ./scripts/run_publication_gate.sh 1
PUBLICATION_GATE_SUITE=small ./scripts/run_publication_gate.sh 1
PUBLICATION_GATE_SUITE=large ./scripts/run_publication_gate.sh 1
PUBLICATION_GATE_SUITE=batched ./scripts/run_publication_gate.sh 1
```

Compare CPU backends:

```bash
PUBLICATION_GATE_FEATURES=cpu-faer ./scripts/run_publication_gate.sh 1
PUBLICATION_GATE_FEATURES=system-openblas ./scripts/run_publication_gate.sh 1
```

## Project Structure

```text
tenferro-einsum-benchmark/
  src/
    main.rs
    lib.rs
  scripts/
    run_all.sh
    run_tenferro_libtorch.sh
    run_all_rust.sh
    run_all_libtorch.sh
    run_all_python.sh
    run_publication_gate.sh
    setup_extern_deps.sh
    clean_extern_deps.sh
    benchmark_python.py
    generate_dataset.py
    format_results.py
  data/
    instances/
    results/
  cpp/
  docs/
  Cargo.toml
  pyproject.toml
```

## References

- [Einsum Benchmark](https://benchmark.einsum.org/)
- [ti2-group/einsum_benchmark](https://github.com/ti2-group/einsum_benchmark)
- [tensor4all/tenferro-rs](https://github.com/tensor4all/tenferro-rs)
- [tensor4all/strided-rs-benchmark-suite](https://github.com/tensor4all/strided-rs-benchmark-suite)

## License

MIT
