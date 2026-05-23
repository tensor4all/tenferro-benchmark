# tenferro-einsum-benchmark

Benchmark suite for [tenferro-rs](https://github.com/tensor4all/tenferro-rs) based on the [einsum benchmark](https://benchmark.einsum.org/) (168 standardized einsum problems across 7 categories).

## Overview

This repository provides:

- A Python pipeline to extract einsum benchmark metadata (shapes, dtypes, contraction paths) into portable JSON
- A **Rust** benchmark runner using [tenferro-einsum](https://github.com/tensor4all/tenferro-rs)
- A **C++ LibTorch** benchmark runner for Torch CPU comparison
- A **Python** benchmark runner using **JAX** for comparison

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
    run_all.sh              # Top-level orchestrator (Rust + C++ LibTorch + Python)
    run_tenferro_libtorch.sh # Focused tenferro-rs vs LibTorch comparison
    run_all_rust.sh         # Build & run tenferro trace/eager OpenBLAS + strided-rs (faer)
    run_all_libtorch.sh     # Build & run LibTorch CPU benchmark
    run_all_python.sh       # Run JAX CPU benchmark
    benchmark_python.py     # Python benchmark runner (PyTorch / JAX)
    generate_dataset.py     # Filter & export benchmark instances as JSON
    format_results.py       # Parse logs and output unified markdown tables
  data/
    instances/              # Exported JSON metadata (one file per instance)
    results/                # Benchmark logs and markdown results
  Cargo.toml                # Rust project
  cpp/                      # C++ LibTorch benchmark sources and CMake project
  pyproject.toml            # Python project (includes torch, jax, opt_einsum)
```

## Setup and Build

### 1. Install prerequisites

Required tools:

- Rust and Cargo
- CMake 3.20+
- Python 3.12+ and [uv](https://docs.astral.sh/uv/)
- OpenBLAS
- LibTorch built against OpenBLAS
- A local clone of [tenferro-rs](https://github.com/tensor4all/tenferro-rs) at `../tenferro-rs`

On macOS with Homebrew:

```bash
brew install openblas cmake
export OPENBLAS_ROOT="$(brew --prefix openblas)"
```

On Linux, set `OPENBLAS_ROOT` to the prefix that contains `include/` and `lib/`:

```bash
export OPENBLAS_ROOT=/path/to/openblas
```

### 2. Set up Python tools

Python is used for dataset generation, JAX comparison, and result formatting.

```bash
uv sync
```

### 3. Build tenferro

For ordinary development, the default build uses tenferro's `cpu-faer` backend:

```bash
cargo build --release
```

For fair Torch vs tenferro comparisons, build the benchmark with the `system-openblas` feature:

```bash
OPENBLAS_ROOT=/opt/homebrew/opt/openblas \
  cargo build --release --no-default-features --features system-openblas
```

`system-openblas` enables tenferro's BLAS backend and links the final benchmark binary to `OPENBLAS_ROOT/lib/libopenblas`. Do not use tenferro's vendored/source OpenBLAS feature for this benchmark comparison: it may build or select a different OpenBLAS than the one used by LibTorch.

tenferro-rs currently supports col-major tensor layout. This benchmark therefore feeds tenferro the `format_string_colmajor` and `shapes_colmajor` metadata from each JSON instance and constructs non-AD `TypedTensor<f64>` values. The row-major metadata is used by Python/JAX and LibTorch where appropriate.

### 4. Build C++ LibTorch benchmark

Set `Torch_DIR` to the CMake package directory for a LibTorch build that is linked against the same OpenBLAS installation as `OPENBLAS_ROOT`.

On this machine, the OpenBLAS-linked Torch C++ build was built from the GitHub PyTorch repository at:

```bash
export OPENBLAS_ROOT=/opt/homebrew/opt/openblas
export Torch_DIR=/Users/atelierarith/work/atelierarith/shinaoka/pytorch-openblas/torch/share/cmake/Torch
```

This exact build reports `torch 2.13.0a0+git8cb4928` and `BLAS_INFO=open`. Its `libtorch_cpu.dylib` links Homebrew OpenBLAS:

```bash
otool -L /Users/atelierarith/work/atelierarith/shinaoka/pytorch-openblas/torch/lib/libtorch_cpu.dylib | grep -Ei 'openblas|accelerate'
# /opt/homebrew/opt/openblas/lib/libopenblas.0.dylib
```

Installed Python Torch packages may also expose `Torch_DIR` values, but they are only candidates. For example, the ComfyUI package on this machine exposes:

```bash
/Users/atelierarith/Library/Application Support/StabilityMatrix/Packages/ComfyUI/venv/lib/python3.10/site-packages/torch/share/cmake/Torch
```

That package was found to link Apple `Accelerate.framework`, not OpenBLAS, so it is not used for fair benchmark numbers.

To find `Torch_DIR` in another Python environment:

```bash
python - <<'PY'
import pathlib
import torch
print(pathlib.Path(torch.__file__).parent / "share/cmake/Torch")
PY
```

If you do not already have an OpenBLAS-linked LibTorch, build PyTorch/LibTorch from source and force its BLAS provider to OpenBLAS. The build used for the local benchmark was created from GitHub as follows:

```bash
git clone --recursive --depth 1 --shallow-submodules https://github.com/pytorch/pytorch.git ../pytorch-openblas
cd ../pytorch-openblas

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install --group dev

export OPENBLAS_ROOT=/opt/homebrew/opt/openblas
export BLAS=OpenBLAS
export OpenBLAS_HOME="$OPENBLAS_ROOT"
export CMAKE_PREFIX_PATH="$OPENBLAS_ROOT:${CMAKE_PREFIX_PATH:-}"

# CPU-only build. Drop these flags if you intentionally need those backends.
export USE_CUDA=0
export USE_ROCM=0
export USE_MPS=0
export USE_MKLDNN=0
export USE_DISTRIBUTED=0
export USE_NNPACK=0
export USE_QNNPACK=0
export USE_PYTORCH_QNNPACK=0
export USE_XNNPACK=0
export USE_FBGEMM=0
export BUILD_TEST=0
export MAX_JOBS=8

python -m pip install --no-build-isolation -v -e .
```

The important settings are `BLAS=OpenBLAS`, `OpenBLAS_HOME=$OPENBLAS_ROOT`, and a `CMAKE_PREFIX_PATH` that lets CMake find the same OpenBLAS prefix used by the Rust benchmark.

After the build, get the `Torch_DIR` for this checkout:

```bash
export Torch_DIR="$(python - <<'PY'
import pathlib
import torch
print(pathlib.Path(torch.__file__).parent / "share/cmake/Torch")
PY
)"
```

Verify that `libtorch_cpu` links OpenBLAS before using it for benchmark numbers:

```bash
# macOS
otool -L "$(python - <<'PY'
import pathlib
import torch
print(pathlib.Path(torch.__file__).parent / "lib/libtorch_cpu.dylib")
PY
)" | grep -i openblas

# Linux
ldd "$(python - <<'PY'
import pathlib
import torch
print(pathlib.Path(torch.__file__).parent / "lib/libtorch_cpu.so")
PY
)" | grep -i openblas
```

If these commands do not print an OpenBLAS library, do not use that LibTorch build for this comparison.

Build this repository's C++ benchmark executable:

```bash
export OPENBLAS_ROOT=/opt/homebrew/opt/openblas
export Torch_DIR=/path/to/libtorch/share/cmake/Torch

cmake -S cpp -B build/cpp-libtorch \
  -DBUILD_LIBTORCH_BENCHMARK=ON \
  -DTorch_DIR="$Torch_DIR" \
  -DOPENBLAS_ROOT="$OPENBLAS_ROOT"
cmake --build build/cpp-libtorch --target benchmark_libtorch --config Release
```

You can also use `CMAKE_PREFIX_PATH` instead of `Torch_DIR`:

```bash
export CMAKE_PREFIX_PATH=/path/to/libtorch
```

Important OpenBLAS notes:

- Official prebuilt LibTorch packages are often linked against MKL, Accelerate, or another BLAS provider. They are not a fair comparison against tenferro OpenBLAS.
- `scripts/run_all_libtorch.sh` inspects `libtorch_cpu` with `otool -L` on macOS or `ldd` on Linux and stops if OpenBLAS is not found.
- `scripts/run_all_libtorch.sh` and `scripts/run_tenferro_libtorch.sh` recreate `build/cpp-libtorch` before configuring, so an old CMake cache cannot silently reuse a different Torch installation.
- The C++ benchmark executable does not link OpenBLAS directly. LibTorch must be the component linked to OpenBLAS.
- For a strict comparison, `OPENBLAS_ROOT` and the OpenBLAS used when building LibTorch must identify the same OpenBLAS installation.

### 5. Verify the local build

The C++ planning tests do not require LibTorch:

```bash
cmake -S cpp -B build/cpp-plan-test -DBUILD_LIBTORCH_BENCHMARK=OFF
cmake --build build/cpp-plan-test --target einsum_plan_test
ctest --test-dir build/cpp-plan-test --output-on-failure
```

Check the Rust OpenBLAS feature path:

```bash
OPENBLAS_ROOT=/opt/homebrew/opt/openblas \
  cargo check --no-default-features --features system-openblas
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

### 2. Compare tenferro-rs and Torch C++

This is the focused comparison runner. It runs only:

- **tenferro trace** with `system-openblas`
- **tenferro eager** with `system-openblas`
- **LibTorch CPU** with an OpenBLAS-linked `libtorch_cpu`

```bash
export OPENBLAS_ROOT=/opt/homebrew/opt/openblas
export Torch_DIR=/path/to/libtorch/share/cmake/Torch

./scripts/run_tenferro_libtorch.sh          # 1 thread (default)
./scripts/run_tenferro_libtorch.sh 4        # 4 threads
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

### 3. Run all benchmarks

```bash
export OPENBLAS_ROOT=/opt/homebrew/opt/openblas
export Torch_DIR=/path/to/libtorch/share/cmake/Torch

./scripts/run_all.sh          # 1 thread (default)
./scripts/run_all.sh 4        # 4 threads
```

Runs all backends in sequence and writes a unified markdown comparison table:

- **tenferro trace/eager** (Rust) — via `scripts/run_all_rust.sh`
- **strided-rs faer** (Rust, optional) — requires `../strided-rs-benchmark-suite`
- **LibTorch CPU** (C++) — via `scripts/run_all_libtorch.sh`
- **JAX CPU** (Python) — via `scripts/run_all_python.sh`

Sets `OMP_NUM_THREADS` and `RAYON_NUM_THREADS` to the given thread count (default: 1). Saves all logs and the markdown table to `data/results/` with timestamps.

Instance JSON files that fail to read or parse are skipped with a warning; the suite continues with the rest. Instances that trigger a backend error are reported as **SKIP** with the reason on stderr.

Requires `strided-rs-benchmark-suite` at `../strided-rs-benchmark-suite` for strided-rs comparison (optional; skipped with a note if not found). Requires an OpenBLAS-linked LibTorch for the LibTorch comparison; otherwise `scripts/run_all_libtorch.sh` fails before running timings.

### 4. Run a single instance

To run the benchmark for **one instance only**, set the environment variable `BENCH_INSTANCE` to the instance name:

```bash
OPENBLAS_ROOT=/opt/homebrew/opt/openblas \
  BENCH_INSTANCE=gm_queen5_5_3.wcsp \
  cargo run --release --no-default-features --features system-openblas

OPENBLAS_ROOT=/opt/homebrew/opt/openblas \
  BENCH_INSTANCE=tensornetwork_permutation_light_415 \
  cargo run --release --no-default-features --features system-openblas
```

**With the full script:**

```bash
export OPENBLAS_ROOT=/opt/homebrew/opt/openblas
export Torch_DIR=/path/to/libtorch/share/cmake/Torch

BENCH_INSTANCE=gm_queen5_5_3.wcsp ./scripts/run_all.sh 1
BENCH_INSTANCE=tensornetwork_permutation_light_415 ./scripts/run_all.sh 4
```

Instance name must match the `name` field in the JSON (i.e. the filename without `.json`). To list available names: `ls data/instances/` → e.g. `gm_queen5_5_3.wcsp.json` → use `gm_queen5_5_3.wcsp`.

### 5. Binary Einsum Diagnostic Instances

For bottleneck investigation, this repo includes a small binary-only set:

- `bin_matmul_256` (`ij,jk->ik`) — dense matrix multiplication baseline
- `bin_batched_matmul_b32_m64_n64_k64` (`bij,bjk->bik`) — batched GEMM with moderate batch size
- `bin_outer_product_4096` (`i,j->ij`) — outer-product path (sensitive to broadcast/pack overhead)
- `bin_elementwise_mul_2048x2048` (`ij,ij->ij`) — pure elementwise multiplication throughput

Recommended invocation for consistent single-thread profiling:

```bash
OPENBLAS_ROOT=/opt/homebrew/opt/openblas \
  RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  BENCH_INSTANCE=bin_matmul_256 \
  cargo run --release --no-default-features --features system-openblas
```

These are intended for quick, reproducible profiling before running heavier LM/structured/network cases.

### 6. Run individually

```bash
OPENBLAS_ROOT=/opt/homebrew/opt/openblas \
  RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  cargo run --release --no-default-features --features system-openblas
```

Run only the LibTorch benchmark:

```bash
OPENBLAS_ROOT=/opt/homebrew/opt/openblas \
  Torch_DIR=/path/to/libtorch/share/cmake/Torch \
  BENCH_INSTANCE=bin_matmul_256 \
  ./scripts/run_all_libtorch.sh 1
```

### 7. Format existing logs

```bash
uv run python scripts/format_results.py data/results/tenferro_trace_*.log data/results/tenferro_eager_*.log
```

## Reproducing Benchmarks

```bash
export OPENBLAS_ROOT=/opt/homebrew/opt/openblas
export Torch_DIR=/path/to/libtorch/share/cmake/Torch

./scripts/run_all.sh        # 1 thread
./scripts/run_all.sh 4      # 4 threads
```

This script:

1. Delegates to `scripts/run_all_rust.sh` (tenferro trace/eager system OpenBLAS + optionally strided-rs faer)
2. Delegates to `scripts/run_all_libtorch.sh` (LibTorch CPU)
3. Delegates to `scripts/run_all_python.sh` (JAX CPU)
4. Sets `OMP_NUM_THREADS` and `RAYON_NUM_THREADS` to the given thread count (default: 1)
5. Formats all collected logs as a unified markdown table via `scripts/format_results.py`
6. Saves all outputs to `data/results/` with timestamps

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

#### tenferro-rs trace/eager vs Torch C++ OpenBLAS, GitHub PyTorch build

Full `scripts/run_all.sh 1` benchmark after building PyTorch from GitHub with `BLAS=OpenBLAS`. Median ± IQR (ms). `OMP_NUM_THREADS=1`, `RAYON_NUM_THREADS=1`.

Build and execution details:

- PyTorch checkout: `/Users/atelierarith/work/atelierarith/shinaoka/pytorch-openblas`
- PyTorch version: `2.13.0a0+git8cb4928`
- `Torch_DIR`: `/Users/atelierarith/work/atelierarith/shinaoka/pytorch-openblas/torch/share/cmake/Torch`
- OpenBLAS: `/opt/homebrew/opt/openblas/lib/libopenblas.0.dylib`
- tenferro trace mode: JAX-like compiled graph path, using `format_string_colmajor` / `shapes_colmajor`
- tenferro eager mode: PyTorch-like immediate execution, applying the same precomputed binary contraction path step by step
- tenferro tensors: non-AD `TypedTensor<f64>` values, col-major only
- strided-rs was not present at `../strided-rs-benchmark-suite`, so this run contains tenferro trace/eager, LibTorch, and JAX

Reproduction command:

```bash
export OPENBLAS_ROOT=/opt/homebrew/opt/openblas
export Torch_DIR=/Users/atelierarith/work/atelierarith/shinaoka/pytorch-openblas/torch/share/cmake/Torch

./scripts/run_all.sh 1
```

Logs:

- `data/results/tenferro_trace_t1_20260523_204748.log`
- `data/results/tenferro_eager_t1_20260523_204748.log`
- `data/results/libtorch_cpu_t1_20260523_204748.log`
- `data/results/jax_cpu_t1_20260523_204748.log`
- `data/results/results_t1_20260523_204748.md`

#### Strategy: opt_flops

Median ± IQR (ms). OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro trace (ms) | tenferro eager (ms) | LibTorch CPU OpenBLAS (ms) | JAX CPU (ms) |
|---|---:|---:|---:|---:|
| bin_batched_matmul_b32_m64_n64_k64 | 1.439 ± 0.188 | 1.140 ± 0.486 | 0.786 ± 0.020 | **0.250 ± 0.067** |
| bin_elementwise_mul_2048x2048 | 1.454 ± 0.078 | 1.697 ± 0.191 | **1.140 ± 0.123** | 1.475 ± 0.564 |
| bin_matmul_256 | 0.784 ± 0.040 | 0.816 ± 0.026 | 0.778 ± 0.011 | **0.297 ± 0.039** |
| bin_outer_product_4096 | 12.044 ± 0.342 | 12.136 ± 1.638 | 2.093 ± 0.016 | **1.218 ± 0.036** |
| gm_queen5_5_3.wcsp | 1904.652 ± 33.918 | 2737.154 ± 49.773 | 1954.326 ± 24.222 | **851.326 ± 3.656** |
| lm_batch_likelihood_brackets_4_4d | 17.908 ± 0.120 | 26.777 ± 0.240 | 22.052 ± 0.102 | **9.160 ± 0.212** |
| lm_batch_likelihood_sentence_3_12d | 43.196 ± 0.434 | 70.599 ± 1.768 | 48.115 ± 0.459 | **11.397 ± 0.402** |
| lm_batch_likelihood_sentence_4_4d | 16.861 ± 0.204 | 29.199 ± 0.096 | 21.466 ± 0.141 | **8.891 ± 0.143** |
| str_matrix_chain_multiplication_100 | 9.400 ± 0.263 | 10.283 ± 0.226 | 17.324 ± 0.136 | **9.315 ± 0.132** |
| str_mps_varying_inner_product_200 | **11.461 ± 0.133** | 16.037 ± 0.131 | 39.471 ± 0.114 | 20.899 ± 0.320 |
| str_nw_mera_closed_120 | 1042.516 ± 10.360 | 1052.047 ± 8.398 | 1000.305 ± 9.802 | **163.466 ± 1.803** |
| str_nw_mera_open_26 | 759.737 ± 10.493 | 752.047 ± 1.002 | 674.294 ± 4.824 | **129.048 ± 3.063** |
| tensornetwork_permutation_focus_step409_316 | 273.003 ± 1.162 | 387.721 ± 2.698 | 260.141 ± 2.665 | **107.409 ± 4.981** |
| tensornetwork_permutation_light_415 | 270.035 ± 0.589 | 310.969 ± 3.121 | 363.726 ± 2.983 | **114.262 ± 6.479** |

#### Strategy: opt_size

Median ± IQR (ms). OMP_NUM_THREADS=1, RAYON_NUM_THREADS=1.

| Instance | tenferro trace (ms) | tenferro eager (ms) | LibTorch CPU OpenBLAS (ms) | JAX CPU (ms) |
|---|---:|---:|---:|---:|
| bin_batched_matmul_b32_m64_n64_k64 | 0.615 ± 0.012 | 0.647 ± 0.012 | 0.420 ± 0.012 | **0.217 ± 0.030** |
| bin_elementwise_mul_2048x2048 | 1.167 ± 0.025 | 1.643 ± 0.027 | **1.128 ± 0.043** | 1.441 ± 0.220 |
| bin_matmul_256 | 0.675 ± 0.016 | 0.676 ± 0.013 | 0.687 ± 0.016 | **0.309 ± 0.028** |
| bin_outer_product_4096 | 12.120 ± 0.086 | 11.867 ± 0.081 | 2.143 ± 0.058 | **1.700 ± 0.125** |
| gm_queen5_5_3.wcsp | 1036.876 ± 2.896 | 1247.475 ± 19.458 | 892.044 ± 5.774 | **297.333 ± 3.906** |
| lm_batch_likelihood_brackets_4_4d | 18.281 ± 0.253 | 27.904 ± 0.104 | 19.112 ± 0.048 | **8.772 ± 0.399** |
| lm_batch_likelihood_sentence_3_12d | 47.247 ± 0.303 | 70.103 ± 0.549 | 48.029 ± 0.140 | **12.499 ± 0.335** |
| lm_batch_likelihood_sentence_4_4d | 18.689 ± 0.295 | 29.659 ± 0.476 | 20.825 ± 0.150 | **7.787 ± 0.388** |
| str_matrix_chain_multiplication_100 | 9.539 ± 0.319 | 10.409 ± 0.167 | 17.250 ± 0.085 | **9.465 ± 0.218** |
| str_mps_varying_inner_product_200 | **12.917 ± 0.408** | 16.391 ± 0.208 | 39.830 ± 0.101 | 22.832 ± 0.412 |
| str_nw_mera_closed_120 | 1025.704 ± 5.192 | 1013.006 ± 24.025 | 1042.117 ± 7.288 | **150.646 ± 1.521** |
| str_nw_mera_open_26 | 759.832 ± 8.088 | 754.611 ± 8.726 | 681.890 ± 8.633 | **133.571 ± 2.393** |
| tensornetwork_permutation_focus_step409_316 | 268.920 ± 2.359 | 386.310 ± 7.191 | 258.466 ± 1.586 | **107.587 ± 4.243** |
| tensornetwork_permutation_light_415 | 269.454 ± 0.833 | 311.939 ± 5.772 | 369.260 ± 4.783 | **114.533 ± 4.485** |

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

### Historical Python Backend Comparison (4 threads, tenferro-einsum vs PyTorch vs JAX)

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

Current top-level runs replace the Python PyTorch backend with the C++ LibTorch backend. For fair comparisons, tenferro and LibTorch must both be linked to OpenBLAS.

## References

- [Einsum Benchmark](https://benchmark.einsum.org/) — standardized einsum benchmark suite
- [ti2-group/einsum_benchmark](https://github.com/ti2-group/einsum_benchmark) — Python package
- [tensor4all/tenferro-rs](https://github.com/tensor4all/tenferro-rs) — Rust tensor library
- [tensor4all/strided-rs-benchmark-suite](https://github.com/tensor4all/strided-rs-benchmark-suite) — sibling benchmark suite for strided-rs

## License

MIT
# tenferro-benchmark
