# NVIDIA GPU Devcontainer Workflow

GPU benchmarks use the CUDA devcontainer. They require an NVIDIA GPU and the
NVIDIA Container Toolkit on the host.

Target profile:

```bash
export BENCHMARK_TARGET_PROFILE=nvidia-gpu
```

Start the CUDA container:

```bash
devcontainer up --workspace-folder . --config .devcontainer/cuda/devcontainer.json
```

The image includes CUTLASS v3.7.0 and CUDA-enabled Ginkgo v1.8.0 under
`/opt`. The default Ginkgo build targets CUDA architecture 80 (A100, the standard
`nvidia-gpu` benchmark hardware).
Override the `GINKGO_CUDA_ARCHITECTURES` Docker build argument when building
for a different architecture.

The container also sets `UV_NO_SYNC=1`: `postCreateCommand` installs the CUDA
variants of PyTorch and JAX after the locked base environment, and later
benchmark-time `uv run` calls must not replace them with the default wheels.

## Host and container Python environments are separate

The CUDA devcontainer mounts a container-only named volume over
`<workspace>/.venv`:

```json
"source=tenferro-benchmark-cuda-venv-${devcontainerId},target=${containerWorkspaceFolder}/.venv,type=volume"
```

`initializeCommand` runs `mkdir -p <workspace>/.venv` on the host before Docker
mounts the volume. On a fresh checkout this creates the underlying mount point
as the invoking host user; an existing host environment is left intact. Without
this step, rootful Docker creates a root-owned host directory, and changing
ownership inside the container only affects the overlaid volume. Start the
devcontainer as your normal host user.

`postCreateCommand` creates the virtualenv in that volume, `chown`ing it to
`vscode` first because a fresh Docker volume is root-owned. The host checkout's
`.venv` is a different directory, so a host `uv run`/`uv sync` can no longer
rewrite the container's CUDA PyTorch/JAX wheels or Python interpreter. Run every
Python command for GPU work inside the container; use host `python3` only for
pure-Python checks that do not import `torch`.

The CPU devcontainers do not use this volume: their collection scripts
intentionally delete and recreate `.venv` (`reset_benchmark_python_venv`), which
a volume mount point would prevent.

If an older container already left an empty root-owned host `.venv`, stop/remove
that container, then remove the empty directory with `rmdir .venv` as your host
user and recreate the container with the updated configuration. Do not delete an
existing Python environment or recursively change checkout ownership.

Run the focused ownership regression test on Linux with Docker (no GPU needed):

```bash
python3 tests/test_cuda_venv_ownership.py
```

It tests fresh and existing host environments as a non-root user, mounts a
separate container volume, and checks the underlying host directory during and
after container use. Set `CUDA_VENV_TEST_IMAGE` to use a locally available Linux
image with `/bin/sh`, `mkdir`, and `chown` (default: `alpine:3.21`).

## Torch extension JIT build locks

`scripts/benchmark_gpu_python.py` JIT-compiles the CUTLASS and Ginkgo
extensions through `torch.utils.cpp_extension.load_inline`, which serializes
builds with a `FileBaton` lock at `<build dir>/lock`. A build killed by a
signal or container restart leaves that lock behind and the next run would wait
forever. Before loading, the runner calls `scripts/torch_extension_lock.py`:

- a lock with no live compiler (`nvcc`/`ninja`/`cc1plus`/`cicc` whose command
  line names the build directory) and older than 120 s is removed with one
  stderr line naming the path and age;
- a lock held by a live compiler is waited on for at most 30 min, then the run
  fails with `ExtensionBuildLockError` instead of hanging.

For an existing container created from an older image, install the same vendor
libraries into its configured paths with:

```bash
devcontainer exec --workspace-folder . --config .devcontainer/cuda/devcontainer.json \
  bash -lc './scripts/setup_gpu_vendors.sh all'
```

Run all GPU suites:

```bash
devcontainer exec --workspace-folder . --config .devcontainer/cuda/devcontainer.json \
  bash -lc 'BENCHMARK_TARGET_PROFILE=nvidia-gpu ./scripts/run_gpu_suite.sh'
```

Run selected backends:

```bash
devcontainer exec --workspace-folder . --config .devcontainer/cuda/devcontainer.json \
  bash -lc 'BENCHMARK_TARGET_PROFILE=nvidia-gpu GPU_BENCH_BACKENDS=pytorch-cuda ./scripts/run_gpu_suite.sh'
```

Standard GPU collection includes tenferro-rs, PyTorch CUDA and selected JAX CUDA
arms. Whole-operation JIT, prepared TensorNetwork node JIT and eager execution
have distinct labels. The original full TensorNetwork workload now passes on
RTX3060 with compact native intermediate layouts and recorded compiler options;
historical A100/permutation JAX failures do not classify these newer paths as
unsupported. Vendor backends remain optional references. See
[gpu-public-api-suite.md](gpu-public-api-suite.md) for operation contracts.

Latest reports:

- `result/nvidia-gpu/gpu/dense.md`
- `result/nvidia-gpu/gpu/einsum.md`
- `result/nvidia-gpu/gpu/sparse.md`
- `result/nvidia-gpu/gpu/tensornetwork.md`

Raw run data:

- `data/results/nvidia-gpu/gpu/<suite>/<timestamp>/run.yaml`
- `data/results/nvidia-gpu/gpu/<suite>/<timestamp>/records.jsonl`
- `data/results/nvidia-gpu/gpu/<suite>/<timestamp>/report.md`

GPU timings must include backend-native device synchronization, but not output
downloads. Output downloads belong to verification outside the timed region.

## Linalg AD verification and timing

Both maintained GPU AD suites return the primal scalar loss and its first-order
JVP or VJP with respect to A; solve keeps RHS fixed. PyTorch uses functional AD
and JAX uses explicitly lowered/compiled JIT. Input tensors, tangent/cotangent
seeds, compilation, initialization and CPU references are prepared before
sampling, including with zero warmups. Repeated operations retain all outputs
through a bounded batch and native synchronization. Validation downloads happen
outside timing.

CPU F64 AD provides independent reference values; its directional derivative is
also checked against a central difference of the primal loss. GPU output shape,
dtype, device, finiteness, primal and derivative values are checked before and
after sampling. Rust records retain initial/final numerical outputs in the raw
artifact, and an independent Python verifier checks them before report formatting.
The formatter rejects skipped verification, failed/debug rows, duplicate cells,
missing selected cells and invalid sampling. Older AD reports containing only
successful-execution checks are not numerically validated performance evidence.

The standard operation rows run the prepared graph in one caller-held CUDA
session (`Runtime::run_prepared_in`); each timed call includes the run's own
stream-retirement barrier, so standard large-operation rows are timed without
the public-path session entry. `gpu/linalg_ad_latency` separately measures
repeated public calls including internal session entry, using
`public_path_latency_including_internal_session_entry`. These small rows are
latency diagnostics, not shared-session or GPU throughput comparisons.

Check both suites on physical GPU1 without collecting operation timings or
updating maintained reports:

```bash
devcontainer exec --workspace-folder . --config .devcontainer/cuda/devcontainer.json \
  bash -lc 'GPU_BENCH_DEVICE=1 GPU_BENCH_VERIFY_ONLY=1 ./scripts/run_gpu_linalg_jvp_vjp.sh'
devcontainer exec --workspace-folder . --config .devcontainer/cuda/devcontainer.json \
  bash -lc 'GPU_BENCH_DEVICE=1 GPU_BENCH_VERIFY_ONLY=1 ./scripts/run_gpu_linalg_ad_latency.sh'
```

Collect the two reports sequentially on physical GPU0 after numerical checks.
The runner isolates each physical card by UUID; all frameworks use local CUDA
ordinal zero. `BENCH_RUNS` and `BENCH_WARMUPS` override the declared sample counts.

## Performance diagnostic device selection

`scripts/run_gpu_perf_issues.sh` also interprets `GPU_BENCH_DEVICE` as a physical
`nvidia-smi` index. It masks that card by UUID and passes CUDA ordinal zero to
the public API runner. Idle-process checks and hardware metadata use the
physical index; metadata records the UUID mask. This also overrides a mask
inherited from a preceding GPU0 command when selecting GPU1.

Run an untimed planning/backend-lifetime probe on physical GPU1:

```bash
devcontainer exec --workspace-folder . --config .devcontainer/cuda/devcontainer.json \
  bash -lc 'GPU_BENCH_DEVICE=1 GPU_BENCH_SKIP_LATEST=1 \
    BENCH_COVERAGE=full BENCH_EFFORT=scan PERF_ISSUES_CORRECTNESS_ONLY=1 \
    BENCH_INSTANCE=shape_stream_f64_n16_fixed-warm_counters,vendor_churn_f64_cycles8 \
    ./scripts/run_gpu_perf_issues.sh'
```

`GPU_BENCH_SKIP_LATEST=1` keeps the maintained performance-issue report while
retaining raw data, metadata and the probe report. The probe above passed both
cases on GPU1 at tenferro-rs `9655b4e0`, with no timed samples; it is not
operation-performance evidence. The runner refreshes an upstream `main`
checkout before its mandatory rebuild and preserves detached/pinned revisions.
