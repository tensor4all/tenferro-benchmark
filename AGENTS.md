# Agent Workflow

## Host CPU Contention

Before running benchmarks, check whether the host is busy. If other long-running
jobs are consuming CPU, choose idle CPU cores and pin benchmark commands to
those cores so the run does not contend with already-busy cores.

For host runs:

```bash
taskset -c <idle-core-list> ./scripts/run_all.sh 1
taskset -c <idle-core-list> ./scripts/run_all.sh 4
```

For devcontainer runs, apply the pinning inside the container command:

```bash
devcontainer exec --workspace-folder . bash -lc \
  'taskset -c <idle-core-list> ./scripts/run_all.sh 1'
```

## Benchmark Report Policy

Treat benchmark reports as provenance-bearing run artifacts, not as a single
mutable "latest result" file.

- Do not commit performance results from a busy or otherwise contended host.
  Busy-host runs are useful for smoke validation only; keep their details in
  the work log or untracked notes.
- Publishable benchmark reports must record the environment that produced the
  numbers, including:
  - timestamp and exact command line;
  - host vs devcontainer execution mode;
  - OS/kernel, CPU model, online CPU count, and thread count;
  - pinned CPU core list, if `taskset` or another affinity mechanism was used;
  - host load/CPU contention observations before the run;
  - `OPENBLAS_ROOT`, `PYTORCH_OPENBLAS_DIR` or `Torch_DIR`, and LibTorch
    OpenBLAS linkage verification;
  - Rust, Python/uv, CMake, PyTorch, and JAX versions when those backends are
    included;
  - benchmark repository commit and dirty status;
  - `tenferro-rs` commit hash used by `extern/tenferro-rs`.
- Do not overwrite an existing committed benchmark report when publishing new
  results. Add new report files for each publishable run using a date plus
  per-day sequence suffix, for example:
  - `result/einsum-results-YYYYMMDD-01.md`
  - `result/cpu-benchmark-results-YYYYMMDD-01.md`
  Pick the next sequence number by checking existing reports for that date. If
  an aggregate or index file is needed, keep it separate from immutable run
  reports and make clear that it is only an index.
- If the current scripts overwrite `result/einsum-results.md` or
  `result/cpu-benchmark-results.md` during a smoke run, do not commit those
  overwritten files as benchmark evidence. Preserve publishable results as new
  timestamped files.
- When the user explicitly asks to publish benchmark results, add provenance to
  the new dated reports, commit only those new reports and intentional metadata
  changes, and push only after the user asks for publish or push.

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
