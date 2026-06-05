# Setup and C++ LibTorch Build

## Install prerequisites

Required tools:

- Rust and Cargo
- CMake 3.20+
- Python 3.12+ and [uv](https://docs.astral.sh/uv/)
- Repo-local external checkouts under `extern/`

On macOS with Homebrew:

```bash
brew install cmake
```

macOS BLAS-backed CPU benchmarks are fixed to Apple's Accelerate framework through the `system-accelerate` Cargo feature. `OPENBLAS_ROOT` is not required for the standard macOS tenferro/PyTorch/JAX benchmark path.

On Linux OpenBLAS runs, set `OPENBLAS_ROOT` to the prefix that contains `include/` and `lib/`:

```bash
export OPENBLAS_ROOT=/path/to/openblas
```

The repository devcontainer uses Ubuntu 24.04 and prepares a stable OpenBLAS
prefix at:

```bash
export OPENBLAS_ROOT=/opt/openblas
```

It also keeps the Linux PyTorch/LibTorch checkout separate from a possible
host-built checkout:

```bash
export PYTORCH_OPENBLAS_DIR=/workspaces/tenferro-benchmark/extern/devcontainer/pytorch-openblas
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
- `extern/pytorch-openblas` only when `SETUP_PYTORCH_OPENBLAS=1`

`SETUP_PYTORCH_OPENBLAS=1` is disabled on macOS. Use the Linux devcontainer for OpenBLAS LibTorch comparisons; macOS CPU BLAS benchmarks use Accelerate.

By default, the tenferro checkout is reused as-is. A dirty `extern/tenferro-rs` checkout is valid for benchmark runs; the run metadata records the commit hash while the benchmark repo commit is intentionally left to git history. Set `TENFERRO_UPDATE=1 TENFERRO_REF=<branch-or-commit>` only when you intentionally want setup to fetch and check out a specific ref. In that explicit update mode, dirty checkouts fail instead of being overwritten.

Sibling checkouts at `../tenferro-rs` or `../pytorch-openblas` are left in place by default. Set `SETUP_EXTERN_MIGRATE_SIBLINGS=1` only when you intentionally want the setup script to move those sibling repositories into `extern/`.

`scripts/run_all.sh` sources this setup script automatically so `TENFERRO_RS_DIR` and the normalized CPU feature selection are available to the benchmark subprocesses. `OPENBLAS_ROOT`, `PYTORCH_OPENBLAS_DIR`, and `Torch_DIR` are exported only for OpenBLAS runs or when `SETUP_PYTORCH_OPENBLAS=1` prepares the optional OpenBLAS-linked PyTorch checkout. Set `SKIP_EXTERN_SETUP=1` only when you intentionally want to provide all paths yourself.

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

On macOS, build BLAS-backed tenferro CPU benchmarks with `system-accelerate`:

```bash
cargo build --release --no-default-features --features system-accelerate
```

On Linux OpenBLAS runs, use `system-openblas`:

```bash
OPENBLAS_ROOT=/opt/openblas \
  cargo build --release --no-default-features --features system-openblas
```

`system-accelerate` and `system-openblas` both enable tenferro's BLAS backend. `system-openblas` links the final benchmark binary to `OPENBLAS_ROOT/lib/libopenblas`. Do not use tenferro's vendored/source OpenBLAS feature for this benchmark comparison: it may build or select a different OpenBLAS than the one used by LibTorch.

tenferro-rs currently supports col-major tensor layout. This benchmark therefore feeds tenferro the `format_string_colmajor` and `shapes_colmajor` metadata from each JSON instance and constructs non-AD `TypedTensor<f64>` values. The row-major metadata is used by Python/JAX and LibTorch where appropriate.

## Build C++ LibTorch benchmark

The C++ LibTorch runner is OpenBLAS-only and is intended for Linux/devcontainer runs that need the Torch C++ column. macOS CPU BLAS benchmarks use Accelerate and skip this runner instead of switching the Rust side to OpenBLAS.

Set `Torch_DIR` to the CMake package directory for a LibTorch build that is linked against the same OpenBLAS installation as `OPENBLAS_ROOT`.

In the devcontainer, the OpenBLAS-linked Torch C++ build is PyTorch `v2.12.0`
checked out from the GitHub PyTorch repository at:

```bash
export OPENBLAS_ROOT=/opt/openblas
export Torch_DIR=/workspaces/tenferro-benchmark/extern/devcontainer/pytorch-openblas/torch/share/cmake/Torch
```

This exact build reports `torch 2.12.0a0+git0d62256` from tag `v2.12.0`.
Its `libtorch_cpu.so` links OpenBLAS:

```bash
ldd extern/devcontainer/pytorch-openblas/torch/lib/libtorch_cpu.so | grep -Ei 'openblas|accelerate'
# /opt/openblas/lib/libopenblas.so
```

Installed Python Torch packages may also expose `Torch_DIR` values, but they are only candidates. On macOS, the standard Python Torch package commonly links Apple `Accelerate.framework`; use that for Python comparisons, not for the OpenBLAS-only Torch C++ column.

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
export OPENBLAS_ROOT=/opt/openblas
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

export OPENBLAS_ROOT=/opt/openblas
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
/workspaces/tenferro-benchmark/extern/devcontainer/pytorch-openblas/torch/share/cmake/Torch
```

Build this repository's C++ benchmark executable:

```bash
export OPENBLAS_ROOT=/opt/openblas
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
export OPENBLAS_ROOT=/opt/openblas

./scripts/run_all.sh 1
./scripts/run_all.sh 4
```

For a quick smoke run that still measures the Torch C++ column:

```bash
export OPENBLAS_ROOT=/opt/openblas

BENCH_INSTANCE=bin_matmul_256 \
BENCH_RUNS=1 \
BENCH_WARMUPS=0 \
PUBLICATION_GATE_SUITE=small \
  ./scripts/run_all.sh 1
```

After either run, verify that the generated reports include the expected comparison columns:

```bash
rg -n "Torch C\\+\\+|PyTorch Python|JAX Python|XLA CPU|tenferro-rs" \
  result/cpu/einsum.md result/cpu/cpu_ops.md
```

`result/cpu/einsum.md` should contain measured einsum columns for tenferro-rs eager mode, tenferro-rs trace mode, Torch C++, PyTorch Python, and JAX Python. JAX CPU einsum columns are labeled with XLA CPU dot because JAX lowers matrix-shaped einsum to XLA CPU `dot`, not to the same external BLAS provider used by tenferro/PyTorch. `result/cpu/cpu_ops.md` uses the same comparison family for PR884 CPU benchmark items and measures tenferro-rs eager mode, tenferro-rs trace mode, Torch C++, PyTorch Python, and JAX Python CPU-op runners.

Important OpenBLAS notes:

- Official prebuilt LibTorch packages are often linked against MKL, Accelerate, or another BLAS provider. They are not a fair comparison against tenferro OpenBLAS.
- `scripts/run_all_libtorch.sh` is disabled on macOS by benchmark policy. On Linux it inspects `libtorch_cpu` with `ldd` and stops if OpenBLAS is not found.
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

Check the Rust macOS Accelerate feature path:

```bash
cargo check --no-default-features --features system-accelerate
```

On Linux OpenBLAS runs:

```bash
OPENBLAS_ROOT=/opt/openblas \
  cargo check --no-default-features --features system-openblas
```
