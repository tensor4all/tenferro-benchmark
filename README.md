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

Benchmark results are not duplicated in this README. The files under
`result/` are the source of truth:

- Full einsum results: [result/einsum-results.md](result/einsum-results.md)
- Full CPU ops results: [result/cpu-benchmark-results.md](result/cpu-benchmark-results.md)
- Full GPU results: [result/gpu-benchmark-results.md](result/gpu-benchmark-results.md)

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

## Dev Container Workflow

Prerequisites on the host:

- Docker Desktop or another Docker engine
- The `devcontainer` CLI

From the host repository root, build and start the development container:

```bash
devcontainer up --workspace-folder .
```

If the devcontainer config changed and an older container already exists,
recreate it:

```bash
devcontainer up --workspace-folder . --remove-existing-container
```

Run the benchmark suite inside the container from the host with
`devcontainer exec`. The first run prepares `extern/tenferro-rs` if needed and
builds an OpenBLAS-linked PyTorch/LibTorch checkout under
`extern/devcontainer/pytorch-openblas`, so it can take a long time.

```bash
devcontainer exec --workspace-folder . bash -lc '\
  BENCH_INSTANCE=bin_matmul_256 \
  BENCH_RUNS=1 \
  BENCH_WARMUPS=0 \
  PUBLICATION_GATE_SUITE=small \
    ./scripts/run_all.sh 1'
```

For normal benchmark results:

```bash
devcontainer exec --workspace-folder . bash -lc './scripts/run_all.sh 1'
devcontainer exec --workspace-folder . bash -lc './scripts/run_all.sh 4'
```

The reports in `result/` aggregate the latest timestamped table for each
thread count under `data/results/`. After running both commands, the same
markdown files should contain `## Threads: 1` and `## Threads: 4` sections.

The devcontainer sets `OPENBLAS_ROOT=/opt/openblas` and installs Rust, CMake,
Python 3.12, `uv`, OpenBLAS, and Linux linkage tools. It also sets
`PYTORCH_OPENBLAS_DIR` to `extern/devcontainer/pytorch-openblas` inside the
workspace so Linux container builds do not reuse a host-built macOS
`extern/pytorch-openblas` checkout. Verify the Torch C++ library with `ldd`
inside the container before trusting the Torch C++ column:

```bash
devcontainer exec --workspace-folder . bash -lc \
  'ldd "$PYTORCH_OPENBLAS_DIR/torch/lib/libtorch_cpu.so" | rg -i openblas'
```

Generated reports include the exact `tenferro-rs` commit hash. Use it later to
restore the benchmark dependency checkout:

```bash
devcontainer exec --workspace-folder . bash -lc \
  'rg -n "Threads: 1|Threads: 4|tenferro-rs commit" result/einsum-results.md result/cpu-benchmark-results.md'
```

## GPU Benchmark Contract Workflow

The GPU benchmark layer uses shared suite and result schemas so tenferro-rs,
PyTorch Python, LibTorch C++, JAX, and vendor CUDA runners can report comparable
records. Phase 1 validates the contract and report generation without requiring
a GPU:

```bash
bash scripts/run_gpu_suite.sh
```

The smoke path emits structured `not_configured` and `unsupported` records
instead of measuring kernels. This keeps schema and formatter tests runnable on
CPU-only machines. Measured CUDA runners are outside this Phase 1 plan and
require separate follow-up implementation plans.

Validate suites and generated JSONL records manually:

```bash
uv run python scripts/validate_benchmark_suite.py \
  benchmarks/gpu/dense.yaml \
  benchmarks/gpu/einsum.yaml \
  benchmarks/gpu/sparse.yaml

uv run python scripts/validate_benchmark_suite.py --kind result \
  data/results/gpu_contract_*.jsonl
```

The latest generated GPU report is written to:

- `result/gpu-benchmark-results.md`

### Running on a Real GPU via devcontainer

A CUDA-enabled devcontainer is provided for hosts with an NVIDIA GPU and the
NVIDIA Container Toolkit installed. It uses
`nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04` as the base image, sets
`USE_CUDA=1`, and installs PyTorch and JAX with CUDA 12.6 wheels.

Start the CUDA container:

```bash
devcontainer up --workspace-folder . --config .devcontainer/cuda/devcontainer.json
```

Install the vendor libraries needed by the `cutlass` and `ginkgo` backends
(CUTLASS is a quick clone; Ginkgo is a CUDA build, ~10-15 min; persisted in
named volumes so this is a one-time step):

```bash
devcontainer exec --workspace-folder . --config .devcontainer/cuda/devcontainer.json \
  bash -lc './scripts/setup_gpu_vendors.sh all'
```

Run the GPU benchmark suite inside it:

```bash
devcontainer exec --workspace-folder . --config .devcontainer/cuda/devcontainer.json \
  bash -lc './scripts/run_gpu_suite.sh'
```

This measures all 10 backends (tenferro-cuda trace/eager, PyTorch, LibTorch,
JAX, cuBLASLt, CUTLASS, cuSOLVER, cuSPARSE, Ginkgo) and writes the report to
`result/gpu-benchmark-results.md`. Backends whose vendor library is absent
degrade to `not_configured` rather than failing the suite.

See `AGENTS.md` for the full GPU devcontainer workflow including environment
verification, the backend/op matrix, and per-backend overrides.

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

Each `run_all.sh` invocation writes timestamped raw logs and intermediate
tables under `data/results/`. The reports in `result/` are regenerated as a
thread-count summary: the latest table for thread 1 and the latest table for
thread 4 are kept in separate sections of the same markdown file.

The generated reports should include the comparison columns:

```bash
rg -n "Threads: 1|Threads: 4|Torch C\\+\\+|PyTorch Python|JAX Python|tenferro-rs" \
  result/einsum-results.md result/cpu-benchmark-results.md
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

- [result/einsum-results.md](result/einsum-results.md)
- [result/cpu-benchmark-results.md](result/cpu-benchmark-results.md)

Both report files aggregate the latest generated table for each thread count,
so a normal `./scripts/run_all.sh 1` followed by `./scripts/run_all.sh 4`
produces one einsum report and one CPU report with `## Threads: 1` and
`## Threads: 4` sections. They also include the full `tenferro-rs` commit hash
resolved from `TENFERRO_RS_DIR` so the benchmark checkout can be restored later
with `git checkout <commit>`.

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
