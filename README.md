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
- Full einsum results: [result/einsum-results.md](result/einsum-results.md)
- Full CPU ops results: [result/cpu-benchmark-results.md](result/cpu-benchmark-results.md)
- Full GPU results: [result/gpu-benchmark-results.md](result/gpu-benchmark-results.md)

## Latest Benchmark Results

- tenferro-rs commit: `fa722375c8662b5532a6f875a6bef3494ace40b5`
- Environment: devcontainer (Ubuntu 24.04, OpenBLAS), GPU: NVIDIA GeForce RTX 3060

### CPU Einsum Benchmark

Median ± IQR (ms). 14 instances, strategy: opt\_flops.

#### Threads: 1

| Instance | tenferro-rs trace | tenferro-rs eager | Torch C++ | PyTorch | JAX |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m64_n64_k64 | 4.084 ± 0.153 | 2.985 ± 0.013 | 0.899 ± 0.003 | **0.662 ± 0.022** | 1.040 ± 0.045 |
| bin_elementwise_mul_2048x2048 | 115.138 ± 2.426 | 88.158 ± 2.160 | 21.262 ± 0.053 | 23.273 ± 0.039 | **14.126 ± 0.187** |
| bin_matmul_256 | 2.157 ± 0.033 | 1.647 ± 0.009 | **1.074 ± 0.001** | 1.083 ± 0.008 | 1.198 ± 0.044 |
| bin_outer_product_4096 | 241.635 ± 4.902 | 328.541 ± 6.319 | 63.164 ± 0.074 | 71.339 ± 0.408 | **38.563 ± 2.584** |
| gm_queen5_5_3.wcsp | 12576.172 ± 20.940 | 19016.525 ± 127.123 | 9313.277 ± 76.146 | **7479.980 ± 26.226** | - |
| lm_batch_likelihood_brackets_4_4d | 43.436 ± 1.189 | 95.981 ± 1.742 | 66.917 ± 0.267 | 43.530 ± 0.182 | **42.842 ± 0.958** |
| lm_batch_likelihood_sentence_3_12d | 83.683 ± 3.056 | 275.219 ± 1.846 | 134.766 ± 0.849 | 101.233 ± 0.543 | **54.826 ± 5.437** |
| lm_batch_likelihood_sentence_4_4d | **38.747 ± 1.214** | 75.563 ± 1.191 | 65.044 ± 0.215 | 46.522 ± 0.464 | 42.044 ± 1.614 |
| str_matrix_chain_multiplication_100 | 48.223 ± 0.736 | 33.471 ± 0.627 | 35.284 ± 0.269 | **26.532 ± 0.086** | 43.609 ± 2.522 |
| str_mps_varying_inner_product_200 | 45.279 ± 1.612 | 46.821 ± 0.611 | 85.337 ± 0.321 | **36.378 ± 0.762** | 93.413 ± 1.039 |
| str_nw_mera_closed_120 | 2613.991 ± 7.665 | 2995.838 ± 10.763 | 2257.169 ± 3.495 | 2219.787 ± 13.428 | **622.084 ± 31.452** |
| str_nw_mera_open_26 | 2271.418 ± 9.741 | 2703.557 ± 3.433 | 1485.944 ± 8.937 | 1471.294 ± 2.329 | **592.279 ± 61.661** |
| tensornetwork_permutation_focus_step409_316 | 752.289 ± 6.395 | 1368.221 ± 4.687 | 811.400 ± 2.173 | 1015.309 ± 3.196 | **529.530 ± 15.527** |
| tensornetwork_permutation_light_415 | 757.575 ± 5.890 | 1042.957 ± 2.339 | 1063.055 ± 1.217 | 986.796 ± 4.425 | **558.860 ± 44.889** |

#### Threads: 4

| Instance | tenferro-rs trace | tenferro-rs eager | Torch C++ | PyTorch | JAX |
|---|---:|---:|---:|---:|---:|
| bin_batched_matmul_b32_m64_n64_k64 | 6.291 ± 0.282 | 3.335 ± 0.077 | 0.890 ± 0.021 | **0.319 ± 0.021** | 1.238 ± 0.230 |
| bin_elementwise_mul_2048x2048 | 111.792 ± 2.260 | 85.310 ± 0.446 | **7.689 ± 0.123** | 8.630 ± 0.758 | 13.966 ± 0.937 |
| bin_matmul_256 | 1.908 ± 0.142 | 1.391 ± 0.058 | **0.392 ± 0.052** | 0.422 ± 0.022 | 1.103 ± 0.059 |
| bin_outer_product_4096 | 231.349 ± 2.320 | 314.549 ± 1.947 | **19.008 ± 0.360** | 28.008 ± 0.615 | 45.580 ± 1.044 |
| gm_queen5_5_3.wcsp | 11886.099 ± 82.524 | 18609.818 ± 87.356 | 3460.992 ± 22.763 | **2558.145 ± 35.061** | - |
| lm_batch_likelihood_brackets_4_4d | 45.150 ± 0.299 | 101.083 ± 4.045 | 41.908 ± 0.252 | **19.227 ± 0.131** | 43.260 ± 1.880 |
| lm_batch_likelihood_sentence_3_12d | 101.051 ± 1.065 | 232.287 ± 6.139 | 90.421 ± 1.144 | **23.660 ± 0.091** | 57.062 ± 2.788 |
| lm_batch_likelihood_sentence_4_4d | 54.637 ± 5.290 | 81.905 ± 17.751 | 42.965 ± 0.192 | **20.033 ± 0.191** | 41.653 ± 0.976 |
| str_matrix_chain_multiplication_100 | 54.154 ± 3.086 | 61.322 ± 25.283 | 24.830 ± 0.150 | **14.353 ± 0.244** | 46.342 ± 2.239 |
| str_mps_varying_inner_product_200 | 87.245 ± 1.652 | 131.894 ± 0.988 | 88.409 ± 0.386 | **31.332 ± 0.243** | 92.467 ± 1.566 |
| str_nw_mera_closed_120 | 1522.157 ± 4.250 | 2048.930 ± 21.986 | 937.599 ± 14.441 | 749.905 ± 2.487 | **631.926 ± 43.107** |
| str_nw_mera_open_26 | 1742.110 ± 231.092 | 2300.714 ± 19.254 | 784.209 ± 5.577 | **506.221 ± 5.851** | 580.462 ± 37.953 |
| tensornetwork_permutation_focus_step409_316 | 679.812 ± 4.714 | 1343.566 ± 5.854 | 373.097 ± 13.693 | **341.849 ± 4.101** | 486.924 ± 9.678 |
| tensornetwork_permutation_light_415 | 681.329 ± 6.613 | 1013.995 ± 6.139 | 498.139 ± 2.492 | **348.518 ± 2.983** | 549.142 ± 21.982 |

### GPU Benchmark (NVIDIA GeForce RTX 3060)

Median time in milliseconds. `unsupported` = backend does not implement this operation.

#### Dense operations

| Operation | tenferro-rs CUDA trace | tenferro-rs CUDA eager | PyTorch CUDA | LibTorch CUDA | JAX CUDA | cuBLASLt | CUTLASS | cuSOLVER |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| batched\_matmul (b16×32) | 0.142 | 0.134 | 0.057 | 0.060 | 0.120 | **0.054** | 2.942 | - |
| matmul (256×256) | 0.581 | 0.615 | 0.295 | 0.299 | **0.118** | 0.296 | 1.412 | - |
| eigh (64×64) | 2.064 | 153.149 | 1.850 | **1.848** | 2.167 | - | - | 1.893 |
| qr (64×64) | 1.182 | 154.654 | 0.930 | 0.929 | 1.193 | - | - | **0.921** |
| solve (64×64, rhs=4) | 0.527 | 1968.921 | 0.240 | 0.263 | 0.437 | - | - | **0.251** |
| svd (64×64) | **5.201** | 176.360 | 41.350 | 38.810 | 36.949 | - | - | 38.793 |
| einsum bin\_matmul\_256 | 0.653 | 0.652 | 0.411 | 0.408 | **0.125** | 0.399 | 1.319 | - |

#### Sparse operations

| Operation | PyTorch CUDA | LibTorch CUDA | cuSPARSE | Ginkgo |
|---|---:|---:|---:|---:|
| spmm (bcsstk17, rhs=32) | 0.392 | 0.393 | **0.397** | 2.508 |
| spmv (bcspwr10) | **0.096** | 0.096 | 0.098 | 1.375 |

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
