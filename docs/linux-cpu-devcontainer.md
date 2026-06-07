# Linux CPU Devcontainer Workflow

Linux CPU benchmarks use the devcontainer/Docker path. This keeps Linux
packages, OpenBLAS, and Python tooling isolated from the macOS host.

Target profile:

```bash
export BENCHMARK_TARGET_PROFILE=amd-cpu
```

BLAS/provider policy:

- tenferro uses `system-openblas`.
- `OPENBLAS_ROOT=/opt/openblas` inside the devcontainer.
- PyTorch Python uses the installed wheel provider. The benchmark records
  `BLAS_INFO`, `LAPACK_INFO`, and linked BLAS/LAPACK libraries in `run.yaml`.
- The benchmark does not source-build PyTorch to force OpenBLAS.
- JAX is reported as XLA CPU.

Start the container:

```bash
devcontainer up --workspace-folder .
```

Recreate it after devcontainer changes:

```bash
devcontainer up --workspace-folder . --remove-existing-container
```

Smoke run from the host:

```bash
devcontainer exec --workspace-folder . bash -lc '\
  BENCHMARK_TARGET_PROFILE=amd-cpu \
  BENCH_INSTANCE=bin_matmul_256 \
  BENCH_RUNS=1 \
  BENCH_WARMUPS=0 \
  PUBLICATION_GATE_SUITE=small \
    ./scripts/run_all.sh 1'
```

Normal runs:

```bash
devcontainer exec --workspace-folder . bash -lc \
  'BENCHMARK_TARGET_PROFILE=amd-cpu ./scripts/run_all.sh 1'

devcontainer exec --workspace-folder . bash -lc \
  'BENCHMARK_TARGET_PROFILE=amd-cpu ./scripts/run_all.sh 4'
```

Latest reports:

- `result/amd-cpu/cpu/einsum.md`
- `result/amd-cpu/cpu/cpu_ops.md`

Raw run data:

- `data/results/amd-cpu/cpu/einsum/<timestamp>/run.yaml`
- `data/results/amd-cpu/cpu/einsum/<timestamp>/report.md`

