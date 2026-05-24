# Setup and C++ LibTorch Build

## Install prerequisites

Required tools:

- Rust and Cargo
- CMake 3.20+
- Python 3.12+ and [uv](https://docs.astral.sh/uv/)
- OpenBLAS
- Repo-local external checkouts under `extern/`

On macOS with Homebrew:

```bash
brew install openblas cmake
export OPENBLAS_ROOT="$(brew --prefix openblas)"
```

On Linux, set `OPENBLAS_ROOT` to the prefix that contains `include/` and `lib/`:

```bash
export OPENBLAS_ROOT=/path/to/openblas
```

## Set up Python tools

Python is used for dataset generation, PyTorch/JAX comparison, and result formatting.

```bash
uv sync
```

## Set up external checkouts

Use the repo-local setup script to prepare dependencies under `extern/`:

```bash
./scripts/setup_extern_deps.sh
```

The script creates or reuses:

- `extern/tenferro-rs`
- `extern/pytorch-openblas`

If old sibling checkouts exist at `../tenferro-rs` or `../pytorch-openblas`, the script moves them into `extern/` by default to avoid re-cloning or rebuilding. Set `SETUP_EXTERN_MIGRATE_SIBLINGS=0` to disable that migration.

`scripts/run_all.sh` sources this setup script automatically so `OPENBLAS_ROOT`, `TENFERRO_RS_DIR`, `PYTORCH_OPENBLAS_DIR`, and `Torch_DIR` are available to the benchmark subprocesses. Set `SKIP_EXTERN_SETUP=1` only when you intentionally want to provide all paths yourself.

To remove the repo-local dependency checkouts:

```bash
./scripts/clean_extern_deps.sh
```

The cleanup script removes only `extern/tenferro-rs` and `extern/pytorch-openblas`. It leaves any other entries under `extern/` alone.

## Build tenferro

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

## Build C++ LibTorch benchmark

Set `Torch_DIR` to the CMake package directory for a LibTorch build that is linked against the same OpenBLAS installation as `OPENBLAS_ROOT`.

On this machine, the OpenBLAS-linked Torch C++ build is PyTorch `v2.12.0`
checked out from the GitHub PyTorch repository at:

```bash
export OPENBLAS_ROOT=/opt/homebrew/opt/openblas
export Torch_DIR=/Users/atelierarith/work/atelierarith/shinaoka/tenferro-benchmark/extern/pytorch-openblas/torch/share/cmake/Torch
```

This exact build reports `torch 2.12.0a0+git0d62256` from tag `v2.12.0`.
Its `libtorch_cpu.dylib` links Homebrew OpenBLAS:

```bash
otool -L extern/pytorch-openblas/torch/lib/libtorch_cpu.dylib | grep -Ei 'openblas|accelerate'
# /opt/homebrew/opt/openblas/lib/libopenblas.0.dylib
```

Installed Python Torch packages may also expose `Torch_DIR` values, but they are only candidates. For example, the ComfyUI package on this machine exposes:

```text
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

If you do not already have an OpenBLAS-linked LibTorch, build PyTorch/LibTorch
from source and force its BLAS provider to OpenBLAS. The build used for the
local benchmark is created from the PyTorch `v2.12.0` tag by the setup script:

```bash
export OPENBLAS_ROOT=/opt/homebrew/opt/openblas
./scripts/setup_extern_deps.sh
```

For a manual rebuild, use the same destination under `extern/`:

```bash
git clone --recursive --branch v2.12.0 --depth 1 --shallow-submodules \
  https://github.com/pytorch/pytorch.git extern/pytorch-openblas
cd extern/pytorch-openblas

python3 -m venv .venv-openblas
source .venv-openblas/bin/activate
python -m pip install --upgrade pip
python -m pip install --group dev

export OPENBLAS_ROOT=/opt/homebrew/opt/openblas
export BLAS=OpenBLAS
export OpenBLAS_HOME="$OPENBLAS_ROOT"
export CMAKE_PREFIX_PATH="$OPENBLAS_ROOT:${CMAKE_PREFIX_PATH:-}"
export CMAKE_POLICY_VERSION_MINIMUM=3.5

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

The important settings are `BLAS=OpenBLAS`, `OpenBLAS_HOME=$OPENBLAS_ROOT`,
and a `CMAKE_PREFIX_PATH` that lets CMake find the same OpenBLAS prefix used by
the Rust benchmark. With CMake 4.x, `CMAKE_POLICY_VERSION_MINIMUM=3.5` avoids
compatibility failures in old vendored CMake projects.

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

For the repo-local checkout prepared by `scripts/setup_extern_deps.sh`, the shortest verification is:

```bash
# macOS
otool -L extern/pytorch-openblas/torch/lib/libtorch_cpu.dylib | rg -i openblas

# Linux
ldd extern/pytorch-openblas/torch/lib/libtorch_cpu.so | rg -i openblas
```

Confirm the build version before benchmarking:

```bash
python - <<'PY'
import pathlib
import torch
print(torch.__version__)
print(pathlib.Path(torch.__file__).parent / "share/cmake/Torch")
PY
```

Expected local output for the build above:

```text
2.12.0a0+git0d62256
/Users/atelierarith/work/atelierarith/shinaoka/tenferro-benchmark/extern/pytorch-openblas/torch/share/cmake/Torch
```

Build this repository's C++ benchmark executable:

```bash
export OPENBLAS_ROOT=/opt/homebrew/opt/openblas
source ./scripts/setup_extern_deps.sh

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

Run the full top-level benchmark with the local PyTorch 2.12 LibTorch build:

```bash
export OPENBLAS_ROOT=/opt/homebrew/opt/openblas

./scripts/run_all.sh 1
./scripts/run_all.sh 4
```

For a quick smoke run that still measures the Torch C++ column:

```bash
export OPENBLAS_ROOT=/opt/homebrew/opt/openblas

BENCH_INSTANCE=bin_matmul_256 \
BENCH_RUNS=1 \
BENCH_WARMUPS=0 \
PUBLICATION_GATE_SUITE=small \
  ./scripts/run_all.sh 1
```

After either run, verify that the generated reports include the expected comparison columns:

```bash
rg -n "Torch C\\+\\+|PyTorch Python|JAX Python|tenferro-rs" \
  result/einsum-results.md result/cpu-benchmark-results.md
```

`result/einsum-results.md` should contain measured einsum columns for tenferro-rs eager mode, tenferro-rs trace mode, Torch C++, PyTorch Python, and JAX Python. `result/cpu-benchmark-results.md` uses the same column labels for PR884 CPU benchmark items and measures tenferro-rs eager mode, tenferro-rs trace mode, Torch C++, PyTorch Python, and JAX Python CPU-op runners.

Important OpenBLAS notes:

- Official prebuilt LibTorch packages are often linked against MKL, Accelerate, or another BLAS provider. They are not a fair comparison against tenferro OpenBLAS.
- `scripts/run_all_libtorch.sh` inspects `libtorch_cpu` with `otool -L` on macOS or `ldd` on Linux and stops if OpenBLAS is not found.
- `scripts/run_all_libtorch.sh` and `scripts/run_tenferro_libtorch.sh` recreate `build/cpp-libtorch` before configuring, so an old CMake cache cannot silently reuse a different Torch installation.
- The C++ benchmark executable does not link OpenBLAS directly. LibTorch must be the component linked to OpenBLAS.
- For a strict comparison, `OPENBLAS_ROOT` and the OpenBLAS used when building LibTorch must identify the same OpenBLAS installation.

## Verify the local build

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
