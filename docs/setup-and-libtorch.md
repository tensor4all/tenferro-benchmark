# Setup and C++ LibTorch Build

## Install prerequisites

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

## Set up Python tools

Python is used for dataset generation, PyTorch/JAX comparison, and result formatting.

```bash
uv sync
```

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
