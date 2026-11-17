# Linux CPU Devcontainer Workflow

Linux CPU benchmarks use the devcontainer/Docker path. This keeps Linux
packages and Python tooling isolated from the macOS host.

Target profile:

```bash
export BENCHMARK_TARGET_PROFILE=amd-cpu
```

CPU backend policy (tenferro-rs 0.12):

- tenferro-rs has one CPU backend: its own faer-based kernels and thread pool
  (tenferro-rs #2608). The harness selects, links and records no vendor
  BLAS/LAPACK provider, so the image sets no `OPENBLAS_ROOT` or `MKLROOT` for
  the build and `run.yaml` has no `blas` block.
- A CPU row therefore compares **tenferro(faer)** against **PyTorch (the wheel's
  provider)** and **JAX (XLA CPU)**. On this image the installed PyTorch wheel
  is MKL-backed; the benchmark records `BLAS_INFO`, `LAPACK_INFO`, and linked
  BLAS/LAPACK libraries in `run.yaml` under `python_backends`.
- The pages committed before the migration carry `blas: mkl | openblas |
  accelerate`. They are history: the harness can no longer link those
  providers, so do not read them as provider-matched against new runs and do
  not present their differences as implementation speedups.
- The CPU Python environment installs the lockfile without torch and then the
  CPU wheel (`torch==<locked version>+cpu`) from the PyTorch CPU index, so the
  CUDA torch wheel and its `nvidia-*` dependencies are never downloaded
  (tensor4all/tenferro-benchmark#342). The GPU containers keep the CUDA wheel.
- JAX is reported as XLA CPU.
- CPU runners disable experimental YNNPACK fusion with
  `--xla_cpu_experimental_ynn_fusion_type=` because JAX 0.10's YNNPACK path
  rejects rank-above-eight intermediates in the standard einsum corpus.
  The regular XLA CPU path handles those contractions. `XLA_FLAGS` records
  this setting in run metadata; keep it consistent across compared runs.

## Image

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
devcontainer exec --workspace-folder . bash -lc '
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

For a focused 1T run, **after** checking that the host is quiet and no build or
other benchmark is running:

```bash
devcontainer exec --workspace-folder . bash -lc \
  'BENCH_INSTANCE=bin_matmul_256 ./scripts/run_all.sh 1'
```

Keep the existing idle-host guard enabled. Docker does not isolate shared host
CPU or memory-bandwidth contention. Runtime thread budgets come from the runner.
Check the generated `run.yaml` for the recorded PyTorch provider, commits and
thread settings before interpreting results.

## Host (non-container) CPU runs

A Linux host can collect without the devcontainer. The CPU collectors do not
check for `/.dockerenv`: `cpu_info.md` records a `Container` line
(`devcontainer` or `none`) that the reports embed, and `run.yaml` records
`environment.container`, so a host run and a container run stay
distinguishable.

Offline hosts keep the checked-out `extern/tenferro-rs` revision with
`GPU_BENCH_NO_PULL=1`; the recorded commit is the one that was built.

`scripts/run_cpu_followup.sh` writes its run record next to the results: with
`--output PATH` it writes `PATH.run.yaml` (host, thread environment,
`environment.container`, tenferro-rs commit) before the MWE run. The `--output`
results file itself is unchanged. The follow-up MWE links no vendor provider;
its oneMKL DFTI reference is a Python runtime reference that reports the FFT
rows skipped when no `libmkl_rt` loads.

Latest reports:

- `result/amd-cpu/cpu/einsum.md`
- `result/amd-cpu/cpu/cpu_ops.md`
- `result/amd-cpu/cpu/linalg_jvp_vjp.md`

Raw run data:

- `data/results/amd-cpu/cpu/einsum/<timestamp>/run.yaml`
- `data/results/amd-cpu/cpu/einsum/<timestamp>/report.md`
