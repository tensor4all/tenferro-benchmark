# tenferro-benchmark

Benchmark suite for [tenferro-rs](https://github.com/tensor4all/tenferro-rs),
comparing tenferro against PyTorch, JAX, Julia, HPTT, cuTENSOR, and other
reference backends on CPU and GPU workloads.

## Latest Results

The tracked reports under [`result/`](result/) are the source of truth for
benchmark numbers (they are never duplicated into this README). Each file is
the latest report for one `target_profile` × `suite_id` pair; older results
live in git history only.

| Target profile | Suite | Report |
|---|---|---|
| `mac-cpu` (Apple Silicon, native) | `cpu/einsum` | [result/mac-cpu/cpu/einsum.md](result/mac-cpu/cpu/einsum.md) |
| `mac-cpu` | `cpu/cpu_ops` | [result/mac-cpu/cpu/cpu_ops.md](result/mac-cpu/cpu/cpu_ops.md) |
| `mac-cpu` | `cpu/linalg_jvp_vjp` | [result/mac-cpu/cpu/linalg_jvp_vjp.md](result/mac-cpu/cpu/linalg_jvp_vjp.md) |
| `mac-cpu` | `cpu/permutation` | [result/mac-cpu/cpu/permutation.md](result/mac-cpu/cpu/permutation.md) |
| `amd-cpu` (Linux devcontainer) | `cpu/permutation` | [result/amd-cpu/cpu/permutation.md](result/amd-cpu/cpu/permutation.md) |
| `linux-cpu` (Linux devcontainer) | linalg JVP/JVP repro | [result/linux-cpu/cpu/linalg_jvp_jvp.md](result/linux-cpu/cpu/linalg_jvp_jvp.md) |
| `nvidia-gpu` (CUDA devcontainer) | `gpu/dense` | [result/nvidia-gpu/gpu/dense.md](result/nvidia-gpu/gpu/dense.md) |
| `nvidia-gpu` | `gpu/einsum` | [result/nvidia-gpu/gpu/einsum.md](result/nvidia-gpu/gpu/einsum.md) |
| `nvidia-gpu` | `gpu/sparse` | [result/nvidia-gpu/gpu/sparse.md](result/nvidia-gpu/gpu/sparse.md) |
| `nvidia-gpu` | `gpu/tensornetwork` | [result/nvidia-gpu/gpu/tensornetwork.md](result/nvidia-gpu/gpu/tensornetwork.md) |
| `nvidia-gpu` | `gpu/linalg_jvp_vjp` | [result/nvidia-gpu/gpu/linalg_jvp_vjp.md](result/nvidia-gpu/gpu/linalg_jvp_vjp.md) |
| `nvidia-gpu` | `gpu/permutation` | [result/nvidia-gpu/gpu/permutation.md](result/nvidia-gpu/gpu/permutation.md) |

Raw runs (per-timestamp `run.yaml` + machine-readable outputs) are written to
`data/results/<target_profile>/<suite_id>/<timestamp>/`; the tracked report in
`result/` is regenerated from the newest run. See
[docs/results.md](docs/results.md) for the full layout.

## Prerequisites

All profiles:

- Rust toolchain (`cargo`).
- [uv](https://docs.astral.sh/uv/) for the Python environment (`uv sync`
  creates `.venv` from `pyproject.toml`).
- External checkouts under `extern/` (tenferro-rs, strided-rs, and problem
  data), fetched by `./scripts/setup_extern_deps.sh`. Note that
  `extern/strided-rs` is required for **any** `cargo build` in this repository
  (Cargo resolves optional path-dependency manifests even when their feature
  is disabled), so run the setup script before building anything.

For `mac-cpu`, run the "All profiles" commands directly on the host. For the
devcontainer profiles (`amd-cpu` / `linux-cpu` / `nvidia-gpu`), run them
**inside the corresponding devcontainer** (`devcontainer exec ... bash -lc
'uv sync && ./scripts/setup_extern_deps.sh'`) — the setup script expects the
container's `OPENBLAS_ROOT` / `MKLROOT` environment, and the container's
`.venv` must be built with the container's wheels (see the GPU note below).

Profile-specific:

- `mac-cpu`: runs natively (no Docker); tenferro uses Accelerate.
- `amd-cpu` / `linux-cpu`: the [devcontainer CLI](https://github.com/devcontainers/cli)
  and Docker; tenferro defaults to OpenBLAS, oneMKL is optional.
- `nvidia-gpu`: the CUDA devcontainer under `.devcontainer/cuda/`.
- `cpu/permutation` suite: Julia (with the repo `Project.toml` providing
  JSON.jl and Strided.jl) for the Julia backends, and — only if you opt in to
  the HPTT column — cmake plus a C++ toolchain (macOS: `brew install cmake`),
  because the `hptt` Cargo feature builds the vendored HPTT C++ library.

Workflow guides per platform:
[macOS CPU](docs/macos-cpu.md) ·
[Linux CPU devcontainer](docs/linux-cpu-devcontainer.md) ·
[NVIDIA GPU devcontainer](docs/gpu-devcontainer.md).

## Running the Benchmarks

### CPU einsum + ops (macOS, native)

`scripts/run_all.sh [NUM_THREADS]` runs the CPU einsum suite
(tenferro trace/eager vs PyTorch/JAX) plus the CPU ops microbenchmarks
(primal linalg, JVP/VJP, eager backward), and regenerates
`result/<target_profile>/cpu/{einsum,cpu_ops,linalg_jvp_vjp}.md`:

```bash
uv sync
./scripts/setup_extern_deps.sh
BENCHMARK_TARGET_PROFILE=mac-cpu ./scripts/run_all.sh 1
```

Quick smoke (single small instance, one run, no warmup):

```bash
BENCHMARK_TARGET_PROFILE=mac-cpu \
BENCH_INSTANCE=bin_matmul_256 \
BENCH_RUNS=1 \
BENCH_WARMUPS=0 \
PUBLICATION_GATE_SUITE=small \
  ./scripts/run_all.sh 1
```

Useful environment variables: `BENCH_INSTANCE` (restrict to one einsum
instance), `BENCH_RUNS` / `BENCH_WARMUPS` (iteration counts),
`TENFERRO_CPU_FEATURES` (BLAS provider: `system-accelerate`,
`system-openblas`, `system-mkl`), `RUN_PERMUTATION_SUITE=1` (also run the
`cpu/permutation` suite after everything else, sequentially).

Note: a full or smoke `run_all.sh` invocation **overwrites** the tracked
latest reports under `result/<target_profile>/`. If you only ran a smoke
subset, restore them before committing
(`git checkout -- result/`).

### CPU einsum + ops (Linux devcontainer)

Same runner, executed inside the devcontainer from the host:

```bash
devcontainer up --workspace-folder .
devcontainer exec --workspace-folder . bash -lc '
  BENCHMARK_TARGET_PROFILE=amd-cpu ./scripts/run_all.sh 1'
```

### CPU permutation suite (`cpu/permutation`)

A standalone materialize/copy-kernel benchmark comparing tenferro-rs
transpose paths against a naive odometer baseline, strided-rs, HPTT, Julia
Base, and Strided.jl. Spec: [docs/permutation-suite.md](docs/permutation-suite.md).

```bash
uv sync
./scripts/setup_extern_deps.sh
# Multiple thread counts are measured sequentially in one run:
BENCHMARK_TARGET_PROFILE=mac-cpu ./scripts/run_permutation.sh 1 4
# Opt in to the HPTT column (requires cmake + C++ toolchain):
PERMUTATION_EXTRA_FEATURES=hptt \
BENCHMARK_TARGET_PROFILE=mac-cpu ./scripts/run_permutation.sh 1 4
```

For a quick trial run, the suite honors `PATTERN_ID` (restrict to one
pattern), `BENCH_RUNS`, and `BENCH_WARMUPS`:

```bash
BENCHMARK_TARGET_PROFILE=mac-cpu \
PATTERN_ID=transpose_2d_256 \
BENCH_RUNS=1 \
BENCH_WARMUPS=0 \
  ./scripts/run_permutation.sh 1
```

This writes `result/<target_profile>/cpu/permutation.md`. Pattern definitions
live in `data/instances/permutation_patterns.json` and are read by both the
Rust and Julia runners; result records are validated against
`schemas/permutation-result.schema.json`. The suite can also be appended to a
`run_all.sh` invocation with `RUN_PERMUTATION_SUITE=1`.

### GPU suites (CUDA devcontainer)

The repo `.venv` is **shared between the CPU and CUDA devcontainers** (it
lives in the bind-mounted workspace). A CPU-side `uv sync` — including the
CPU devcontainer's own post-create hook — replaces the CUDA wheels with CPU
ones. The GPU Python runners then **silently skip** `pytorch-cuda` /
`jax-cuda` (they exit 0 and the report is simply missing those columns)
rather than failing. Before collecting GPU results, install the CUDA Python
backends inside the container and verify they see the GPU:

```bash
devcontainer up --workspace-folder . --config .devcontainer/cuda/devcontainer.json

# One-time, and again after ANY CPU-side `uv sync`:
devcontainer exec --workspace-folder . --config .devcontainer/cuda/devcontainer.json \
  bash -lc '
    (uv sync --frozen || uv sync)
    uv pip install "torch>=2.12.0" --extra-index-url https://download.pytorch.org/whl/cu126
    uv pip install "jax[cuda12]"
    ./scripts/setup_extern_deps.sh
    uv run python -c "import torch, jax; assert torch.cuda.is_available(); jax.devices(\"cuda\"); print(\"CUDA OK:\", torch.__version__)"'
```

If `nvidia-smi` fails inside a previously created container ("Failed to
initialize NVML" / `CUDA_ERROR_NO_DEVICE`) while the host GPU is fine,
`docker restart <container>` usually restores GPU access — no rebuild needed.

```bash
# gpu/dense, gpu/einsum, gpu/sparse, gpu/tensornetwork:
devcontainer exec --workspace-folder . --config .devcontainer/cuda/devcontainer.json \
  bash -lc 'BENCHMARK_TARGET_PROFILE=nvidia-gpu ./scripts/run_gpu_suite.sh'

# gpu/linalg_jvp_vjp (separate from the standard GPU suite):
devcontainer exec --workspace-folder . --config .devcontainer/cuda/devcontainer.json \
  bash -lc 'BENCHMARK_TARGET_PROFILE=nvidia-gpu ./scripts/run_gpu_linalg_jvp_vjp.sh'

# gpu/permutation (standalone, like cpu/permutation):
devcontainer exec --workspace-folder . --config .devcontainer/cuda/devcontainer.json \
  bash -lc 'BENCHMARK_TARGET_PROFILE=nvidia-gpu ./scripts/run_gpu_permutation.sh'
```

`gpu/permutation` honors the same quick-trial variables as the CPU suite
(`PATTERN_ID`, `BENCH_RUNS`, `BENCH_WARMUPS`), plus `GPU_BENCH_DEVICE` for
the CUDA ordinal:

```bash
devcontainer exec --workspace-folder . --config .devcontainer/cuda/devcontainer.json \
  bash -lc 'BENCHMARK_TARGET_PROFILE=nvidia-gpu \
    PATTERN_ID=transpose_2d_256 BENCH_RUNS=1 BENCH_WARMUPS=0 \
    ./scripts/run_gpu_permutation.sh'
```

After any GPU run, **check the generated report for the full backend set**
(the [Comparison Backends](#comparison-backends) section lists what each
suite compares; for `gpu/permutation` that is two tenferro columns, cuTENSOR,
PyTorch CUDA, JAX CUDA, and memcpy-d2d). A column that is `-` on every row
means that backend's runner skipped — usually the CUDA-wheels issue above.
Like `run_all.sh`, these scripts **overwrite** the tracked latest reports
under `result/nvidia-gpu/`; after a trial or partial run, restore them before
committing (`git checkout -- result/`).

The GPU tensor network benchmark uses problem data from
[`extern/TensorNetworkBenchmarks/`](extern/TensorNetworkBenchmarks/), based on
the upstream [TensorNetworkBenchmarks](https://github.com/TensorBFS/TensorNetworkBenchmarks)
repository; see [docs/tensornetwork-gpu.md](docs/tensornetwork-gpu.md).

### Linux linalg AD repro (OpenBLAS / oneMKL)

Reproduces `result/linux-cpu/cpu/linalg_jvp_jvp.md` with the devcontainer's
source-built OpenBLAS (`/opt/openblas`):

```bash
devcontainer up --workspace-folder . --remove-existing-container
devcontainer exec --workspace-folder . bash -lc '
  python3 - <<PY
import ctypes
lib = ctypes.CDLL("/opt/openblas/lib/libopenblas.so")
lib.openblas_get_config.restype = ctypes.c_char_p
lib.openblas_get_parallel.restype = ctypes.c_int
print(lib.openblas_get_config().decode())
print(f"parallel={lib.openblas_get_parallel()}")
PY'
devcontainer exec --workspace-folder . bash -lc '
  export TENFERRO_CPU_FEATURES=system-openblas
  export PUBLICATION_GATE_FEATURES=system-openblas
  export TENFERRO_CPU_BACKEND_KIND=blas
  ./scripts/reproduce_linux_cpu_linalg_jvp_jvp.sh'
```

For the oneMKL variant (`/opt/intel/oneapi/mkl/latest`), replace both
`system-openblas` values with `system-mkl`. Verify the OpenBLAS build through
the runtime API above instead of relying on `strings`.

## Measurement Policy

- Benchmarks are always run **sequentially** — never multiple benchmark
  processes at once, including different thread-count variants of the same
  suite (see [AGENTS.md](AGENTS.md)).
- Thread counts are controlled via `RAYON_NUM_THREADS` / `OMP_NUM_THREADS` /
  `JULIA_NUM_THREADS`. CPU pinning is unavailable on macOS; on Linux the
  devcontainer convention applies no `taskset`/`numactl` pinning either. The
  effective thread environment is recorded in each run's `run.yaml`.
- Every run records provenance (target profile, suite, tenferro-rs commit,
  CPU/GPU info) in `data/results/.../run.yaml`.

## Comparison Backends

CPU einsum reports compare `tenferro-trace`, `tenferro-eager`, `pytorch-cpu`,
and `jax-cpu`. GPU reports compare `tenferro-cuda-trace`,
`tenferro-cuda-eager`, `pytorch-cuda`, `jax-cuda`, and vendor-specific CUDA
backends where meaningful. The `cpu/permutation` suite has its own backend
set (naive, tenferro transpose paths, HPTT, strided-rs, Julia Base,
Strided.jl, memcpy); `gpu/permutation` compares tenferro CUDA transpose paths
against cuTENSOR, PyTorch/JAX CUDA, and a device-to-device memcpy baseline.

C++ Torch/LibTorch runners are intentionally removed; PyTorch Python is the
ATen comparison backend. The PyTorch CPU provider is detected at run time and
recorded in `run.yaml` and generated reports; Linux does not source-build
PyTorch to force OpenBLAS.

## Documentation

- [macOS CPU workflow](docs/macos-cpu.md): native run, no Docker, Accelerate.
- [Linux CPU devcontainer workflow](docs/linux-cpu-devcontainer.md):
  Docker/devcontainer, OpenBLAS default, optional oneMKL, detected PyTorch
  provider.
- [NVIDIA GPU devcontainer workflow](docs/gpu-devcontainer.md): CUDA
  devcontainer.
- [Einsum suite and instance selection](docs/einsum-suite.md): source
  benchmark, selection rules, diagnostic cases, path strategies.
- [CPU permutation suite](docs/permutation-suite.md): materialize-kernel
  benchmark spec (patterns, backends, fair-comparison guards).
- [GPU permutation suite](docs/gpu-permutation-suite.md): CUDA port of the
  permutation suite.
- [GPU tensor network suite](docs/tensornetwork-gpu.md):
  TensorNetworkBenchmarks parity on CUDA.
- [Result layout and metadata](docs/results.md): `target_profile`,
  `suite_id`, `run.yaml`, latest reports.
- [Architecture terminology](docs/architecture.md): suite, runner, backend,
  strategy, target profile.
- [PyTorch einsum dispatch notes](docs/pytorch-einsum-dispatch.md): PyTorch
  source investigation notes.

## Development Checks

Run these after changing benchmark scripts or schemas:

```bash
uv run python scripts/validate_benchmark_suite.py benchmarks/cpu/einsum.yaml benchmarks/cpu/permutation.yaml
uv run python scripts/validate_benchmark_suite.py benchmarks/gpu/dense.yaml benchmarks/gpu/einsum.yaml benchmarks/gpu/sparse.yaml benchmarks/gpu/tensornetwork.yaml
bash tests/test_suite_result_layout.sh
bash tests/test_run_all_docs_outputs.sh
bash tests/test_clean_extern_deps.sh
bash tests/test_setup_extern_tenferro_checkout.sh
bash tests/test_permutation_result_schema.sh
cmake -S cpp -B build/cpp-plan-test
cmake --build build/cpp-plan-test --target einsum_plan_test
ctest --test-dir build/cpp-plan-test --output-on-failure
```

## License

MIT
