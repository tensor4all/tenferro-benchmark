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
- `/opt/openblas` is built from OpenBLAS source with pthread threading enabled
  (`USE_THREAD=1`, `USE_OPENMP=0`) instead of using Ubuntu's
  `libopenblas-dev` package.
- Intel oneMKL is installed under `/opt/intel/oneapi/mkl/latest` and exported
  as `MKLROOT` for opt-in `system-mkl` runs.
- PyTorch Python uses the installed wheel provider. The benchmark records
  `BLAS_INFO`, `LAPACK_INFO`, and linked BLAS/LAPACK libraries in `run.yaml`.
- The default image does not source-build PyTorch. For a provider-matched
  OpenBLAS comparison, use the separate image below.
- JAX is reported as XLA CPU.

## Provider-matched OpenBLAS image

The separate `.devcontainer/openblas/Dockerfile` builds PyTorch Python 2.12.0
from commit `0d62256a2b23365f8e1604297eb23a6545102aa8`. Both tenferro and
PyTorch link to `/opt/openblas/lib/libopenblas.so`: OpenBLAS 0.3.26 with
pthread threading, LP64 integers and dynamic CPU dispatch. Common code targets
Core 2 rather than the build host's ISA, for AMD/Intel x86_64 portability;
ARM builds are not covered by this configuration. MKL/oneDNN and GPU
support are disabled in this PyTorch build. This is a new comparison baseline,
not a speedup relative to historical MKL results; JAX and Julia are not forced
to use this library. The original MKL image remains available.

From the repository root:

```bash
devcontainer up --workspace-folder . --config .devcontainer/openblas/devcontainer.json
devcontainer exec --workspace-folder . --config .devcontainer/openblas/devcontainer.json \
  bash -c 'source scripts/python_venv.sh; prepare_cpu_benchmark_python_venv "$PWD" &&
    uv run python /usr/local/bin/verify-openblas-pytorch --threads 1 &&
    uv run python /usr/local/bin/verify-openblas-pytorch --threads 4'
```

The image retains the source-built wheel under `/opt/pytorch-wheels/` and sets
`BENCHMARK_TORCH_WHEEL` so runner `.venv` resets reinstall that wheel, not MKL.
Other dependencies come from `uv.lock`, pruning the binary torch wheel and its
CUDA-only dependencies; the source wheel supplies its own runtime dependencies.
The image and helper set `UV_NO_SYNC=1` to prevent subsequent `uv run` from
restoring the lockfile wheel; explicit `uv sync` would still replace it. Do not
reuse a host venv.
The build-time smoke test checks the resolved `libtorch_cpu.so` link, pthread
mode, LP64, BLAS/LAPACK operations, and effective 1T settings. Repeat the runtime
check above after changing the image or Python environment. It is a correctness
check, not a benchmark.

Build without the devcontainer CLI (build jobs are separate from runtime threads):

```bash
docker build --build-arg BUILD_JOBS=8 -f .devcontainer/openblas/Dockerfile \
  -t tenferro-benchmark-openblas:dev .
```

The first build compiles PyTorch and needs substantial time, RAM and disk;
subsequent builds cache the wheel layer. Default `BUILD_JOBS=16`; lower it on
smaller hosts. The wheel SHA256, PyTorch commit, build dependencies and CMake
cache are saved in `/opt/pytorch-wheels/`. Record the resulting image ID/digest
as well: Ubuntu packages, Rust, uv, and Julia tooling are not fully locked.
This image targets Linux CPU; it does not replace native macOS measurements.

For a focused 1T run, **after** checking that the host is quiet and no build or
other benchmark is running:

```bash
devcontainer exec --workspace-folder . --config .devcontainer/openblas/devcontainer.json \
  bash -c 'BENCH_INSTANCE=bin_matmul_256 ./scripts/run_all.sh 1'
```

Keep the existing idle-host guard enabled. Docker does not isolate shared host
CPU or memory-bandwidth contention. Runtime thread budgets come from the runner;
the image defaults OpenBLAS, OpenMP and Rayon to 1T. Check the generated
`run.yaml` for the detected providers, linked libraries, commits and thread
settings before interpreting results. Tenferro uses `system-openblas` in this
image; when first building Rust, also inspect the produced binary with `ldd`
and confirm its `libopenblas` resolves to `/opt/openblas/lib/`.

Validation on AMD EPYC 7713P: the image build, PyTorch BLAS/LAPACK checks at
1T and 4T, and source-wheel preservation after a runner venv reset passed;
`uv.lock` was unchanged. Rust einsum/publication-gate binaries subsequently built
with the pinned external checkouts; three OpenBLAS correctness tests passed.
Native timing comparisons remain deferred because the host is busy. See the
[instruction-count diagnostic report](../result/amd-cpu/cpu/openblas_instruction_diagnostics.md)
for attributed CPU work and its measurement limitations.

## Default wheel/MKL image

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

Opt-in MKL tenferro run:

```bash
devcontainer exec --workspace-folder . bash -lc '\
  BENCHMARK_TARGET_PROFILE=amd-cpu \
  TENFERRO_CPU_FEATURES=system-mkl \
  PUBLICATION_GATE_FEATURES=system-mkl \
    ./scripts/run_all.sh 1'
```

Latest reports:

- `result/amd-cpu/cpu/einsum.md`
- `result/amd-cpu/cpu/cpu_ops.md`
- `result/amd-cpu/cpu/linalg_jvp_vjp.md`

Raw run data:

- `data/results/amd-cpu/cpu/einsum/<timestamp>/run.yaml`
- `data/results/amd-cpu/cpu/einsum/<timestamp>/report.md`
