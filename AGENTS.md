# Agent Workflow

## Devcontainer Benchmark Workflow

Use this workflow when validating that benchmarks can run inside the Linux
devcontainer from the host `devcontainer` CLI.

1. Build and start the container from the host repository root:

   ```bash
   devcontainer up --workspace-folder .
   ```

   If `.devcontainer/devcontainer.json` or `.devcontainer/Dockerfile` changed
   and an old container may already exist, recreate it:

   ```bash
   devcontainer up --workspace-folder . --remove-existing-container
   ```

2. Confirm the container environment:

   ```bash
   devcontainer exec --workspace-folder . bash -lc \
     'echo "$OPENBLAS_ROOT"; echo "$PYTORCH_OPENBLAS_DIR"; rustc --version; uv --version; cmake --version | head -n 1'
   ```

   Expected paths:

   ```bash
   OPENBLAS_ROOT=/opt/openblas
   PYTORCH_OPENBLAS_DIR=/workspaces/tenferro-benchmark/extern/devcontainer/pytorch-openblas
   ```

   The devcontainer intentionally uses
   `extern/devcontainer/pytorch-openblas` for Linux PyTorch/LibTorch builds so
   it does not reuse a host-built macOS `extern/pytorch-openblas` checkout.

3. Run a quick end-to-end smoke from the host. This still executes inside the
   container and exercises the Torch C++ path:

   ```bash
   devcontainer exec --workspace-folder . bash -lc '\
     BENCH_INSTANCE=bin_matmul_256 \
     BENCH_RUNS=1 \
     BENCH_WARMUPS=0 \
     PUBLICATION_GATE_SUITE=small \
       ./scripts/run_all.sh 1'
   ```

   The first run can take a long time because it clones
   `extern/tenferro-rs` if needed and builds an OpenBLAS-linked
   PyTorch/LibTorch checkout under `extern/devcontainer/pytorch-openblas`.

4. Verify LibTorch linkage inside the container before trusting Torch C++
   results:

   ```bash
   devcontainer exec --workspace-folder . bash -lc \
     'ldd "$PYTORCH_OPENBLAS_DIR/torch/lib/libtorch_cpu.so" | rg -i openblas'
   ```

5. For normal benchmark results:

   ```bash
   devcontainer exec --workspace-folder . bash -lc './scripts/run_all.sh 1'
   devcontainer exec --workspace-folder . bash -lc './scripts/run_all.sh 4'
   ```

6. Verify generated outputs and reproducibility metadata:

   ```bash
   devcontainer exec --workspace-folder . bash -lc \
     'rg -n "Threads: 1|Threads: 4|Torch C\\+\\+|PyTorch Python|JAX Python|tenferro-rs commit" result/einsum-results.md result/cpu-benchmark-results.md'
   ```

   `result/einsum-results.md` and `result/cpu-benchmark-results.md` collect
   the latest table for each thread count found under `data/results/`. After
   running both commands above, each report should contain separate
   `## Threads: 1` and `## Threads: 4` sections rather than only the most
   recent run.

   The `tenferro-rs commit` line records the exact commit hash to use later
   with `git checkout <commit>`.

## GPU Devcontainer Benchmark Workflow

Use this workflow to run GPU benchmarks inside the CUDA devcontainer.
Requires an NVIDIA GPU on the host with the NVIDIA Container Toolkit installed.

1. Build and start the CUDA container:

   ```bash
   devcontainer up --workspace-folder . --config .devcontainer/cuda/devcontainer.json
   ```

   Recreate if the config or Dockerfile changed:

   ```bash
   devcontainer up --workspace-folder . --config .devcontainer/cuda/devcontainer.json \
     --remove-existing-container
   ```

2. Confirm GPU access and environment inside the container:

   ```bash
   devcontainer exec --workspace-folder . --config .devcontainer/cuda/devcontainer.json \
     bash -lc 'nvidia-smi && echo "USE_CUDA=$USE_CUDA" && echo "CUDA_HOME=$CUDA_HOME" && nvcc --version'
   ```

   Expected values: `USE_CUDA=1`, `CUDA_HOME=/usr/local/cuda`.

3. Verify Python GPU backends are available:

   ```bash
   devcontainer exec --workspace-folder . --config .devcontainer/cuda/devcontainer.json \
     bash -lc 'uv run python -c "import torch; print(torch.cuda.is_available()); import jax; print(jax.devices())"'
   ```

4. Run the GPU benchmark suite:

   ```bash
   devcontainer exec --workspace-folder . --config .devcontainer/cuda/devcontainer.json \
     bash -lc './scripts/run_gpu_suite.sh'
   ```

   Generated reports:
   - `data/results/gpu_contract_<timestamp>.jsonl` — structured JSONL records
   - `result/gpu-benchmark-results.md` — formatted markdown report

5. Override backends, suites, or device ordinal via environment variables:

   ```bash
   devcontainer exec --workspace-folder . --config .devcontainer/cuda/devcontainer.json \
     bash -lc 'GPU_BENCH_BACKENDS=pytorch-cuda,jax-cuda GPU_BENCH_DEVICE=0 ./scripts/run_gpu_suite.sh'
   ```

### Implemented backends and execution paths

| Backend | Runner | Ops | Notes |
|---|---|---|---|
| `tenferro-cuda-trace` | `src/bin/benchmark_gpu_rust.rs` | dense + einsum | traced graph on CubeCL CUDA |
| `tenferro-cuda-eager` | `src/bin/benchmark_gpu_rust.rs` | dense + einsum | eager CubeCL CUDA |
| `pytorch-cuda` | `scripts/benchmark_gpu_python.py` | all | ATen CUDA kernels |
| `libtorch-cuda` | `scripts/benchmark_gpu_python.py` | all | same ATen kernels as C++ LibTorch |
| `jax-cuda` | `scripts/benchmark_gpu_python.py` | dense + einsum | XLA, `jax_enable_x64` |
| `cublaslt` | `scripts/benchmark_gpu_python.py` | matmul/bmm/einsum | `torch.mm` → cuBLAS LT, TF32 off |
| `cusolver` | `scripts/benchmark_gpu_python.py` | qr/solve/svd/eigh | `preferred_linalg_library=cusolver` |
| `cusparse` | `scripts/benchmark_gpu_python.py` | spmv/spmm | `torch.sparse` CSR → cuSPARSE |
| `cutlass` | `scripts/benchmark_gpu_python.py` | matmul/einsum | JIT extension, `cutlass::gemm::device::Gemm<double,...,Sm80>` |
| `ginkgo` | `scripts/benchmark_gpu_python.py` | spmv/spmm | JIT extension linking `libginkgo` CUDA executor |

### Vendor library setup (cutlass, ginkgo)

The `cutlass` and `ginkgo` backends JIT-compile a small extension that needs
their headers/libraries. Install both with the helper script (CUTLASS is a
quick clone; Ginkgo is a full CUDA build, ~10-15 min):

```bash
devcontainer exec --workspace-folder . --config .devcontainer/cuda/devcontainer.json \
  bash -lc './scripts/setup_gpu_vendors.sh all'   # or: cutlass | ginkgo
```

The script installs CUTLASS (`$CUTLASS_DIR`, default `/opt/cutlass`) and builds
Ginkgo with CUDA (`$GINKGO_DIR`, default `/opt/ginkgo`), auto-detecting the host
GPU compute capability. Override with `CUDA_ARCH`, `CUTLASS_TAG`, `GINKGO_TAG`.

If a vendor library is absent the corresponding backend emits `not_configured`
records rather than failing the suite. `run_gpu_suite.sh` exports `CUTLASS_DIR`
and `GINKGO_DIR` automatically.

## Torch C++ Included Benchmark

Use this workflow when validating or regenerating benchmark results that must
include the Torch C++ column.

1. Prepare repo-local external dependencies:

   ```bash
   export OPENBLAS_ROOT="$(brew --prefix openblas)"
   ./scripts/setup_extern_deps.sh
   ```

   This creates or reuses `extern/tenferro-rs` and
   `extern/pytorch-openblas`. If sibling checkouts exist at `../tenferro-rs` or
   `../pytorch-openblas`, the setup script moves them under `extern/` by
   default. Use `SETUP_EXTERN_MIGRATE_SIBLINGS=0` only when that migration is
   not desired.

   To remove these repo-local checkouts, run:

   ```bash
   ./scripts/clean_extern_deps.sh
   ```

2. Confirm LibTorch is OpenBLAS-linked before trusting Torch C++ numbers:

   ```bash
   otool -L extern/pytorch-openblas/torch/lib/libtorch_cpu.dylib | rg -i openblas
   ```

   The expected local Torch C++ package directory is:

   ```bash
   extern/pytorch-openblas/torch/share/cmake/Torch
   ```

3. Run a quick end-to-end smoke that still exercises Torch C++:

   ```bash
   BENCH_INSTANCE=bin_matmul_256 \
   BENCH_RUNS=1 \
   BENCH_WARMUPS=0 \
   PUBLICATION_GATE_SUITE=small \
     ./scripts/run_all.sh 1
   ```

   `scripts/run_all.sh` sources `scripts/setup_extern_deps.sh` automatically,
   so `Torch_DIR` and `OPENBLAS_ROOT` exported by the setup script are visible
   to `scripts/run_all_libtorch.sh`.

4. For a normal benchmark run:

   ```bash
   ./scripts/run_all.sh 1
   ./scripts/run_all.sh 4
   ```

   Raw logs and timestamped intermediate tables are written under
   `data/results/`. The human-facing reports under `result/` aggregate the
   latest timestamped table for each thread count, so running thread 1 and then
   thread 4 leaves both `## Threads: 1` and `## Threads: 4` sections in one
   markdown file.

5. Verify generated outputs:

   ```bash
   rg -n "Threads: 1|Threads: 4|Torch C\\+\\+|PyTorch Python|JAX Python|tenferro-rs" \
     result/einsum-results.md result/cpu-benchmark-results.md
   ```

   `result/einsum-results.md` should contain measured columns for tenferro-rs
   eager mode, tenferro-rs trace mode, Torch C++, PyTorch Python, and JAX
   Python. The PR884 CPU table in `result/cpu-benchmark-results.md` uses the
   same column labels and should include measured tenferro-rs eager mode,
   tenferro-rs trace mode, Torch C++, PyTorch Python, and JAX Python values for
   the CPU-op items.

## Useful Checks

Run these after changing benchmark scripts:

```bash
cargo metadata --no-deps --format-version 1
bash tests/test_run_all_docs_outputs.sh
bash tests/test_extern_dependency_paths.sh
bash tests/test_run_all_rust_bin_selection.sh
bash tests/test_native_batch_layout_labels.sh
```
