# Agent Workflow

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
   `data/results/`. The latest human-facing reports are written under
   `result/`.

5. Verify generated outputs:

   ```bash
   rg -n "Torch C\\+\\+|PyTorch Python|JAX Python|tenferro-rs" \
     result/results-einsum.md result/cpu-benchmark-results.md
   ```

   `result/results-einsum.md` should contain measured columns for tenferro-rs
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
```
