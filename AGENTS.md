# Agent Workflow

## Benchmark Result Source of Truth

Keep generated benchmark result tables under `result/<target_profile>/...`.
Do not duplicate result tables in README.md or overview docs.

Historical report files are not maintained. Use git history for older latest
reports.

## Benchmark Timing Discipline

Run benchmark timing collection sequentially. Do not run multiple CPU benchmark
processes at the same time when collecting or comparing results, including
`run_all.sh` invocations and ad hoc `BENCH_INSTANCE` probes. Concurrent BLAS,
Python, and Rust benchmark processes contend for cores, caches, thermal
headroom, and runtime thread pools, which distorts medians and IQRs.

Parallel shell/tool execution is fine for non-timing work such as file
inspection or tests that do not measure benchmark performance.

## Never Mix Compute-Bound and Latency Results

Every case has a regime: `throughput` (compute- or bandwidth-bound) or
`latency` (dominated by per-call launch/dispatch overhead). Declare it in the
suite YAML when it differs from the suite default:

```yaml
defaults:
  regime: throughput
problems:
  - id: elementwise_chain_f64_1k
    regime: latency   # per-command overhead probe, not a bandwidth measurement
```

`latency` cases currently include `gpu/dense` `dense_batched_matmul_f64_b16_32`,
`gpu/elementwise` `elementwise_chain_f64_1k`, all of `gpu/linalg_ad_latency`,
and the `ns/op` microbenchmarks in `gpu/perf_issues`.

Rules:

- Never put a latency bar in a compute-bound chart, and never average or flag
  across regimes. A suite summary covers one regime only.
- Suite pages show a throughput section and a latency section separately.
- Published pages separate them: compute-bound results on the CPU/GPU pages,
  latency/overhead results on dedicated `*/latency/` pages.
- A small workload is not evidence of slow compute. The latency pages measure
  dispatch overhead, and their numbers must never be read as GPU/CPU
  throughput.
- When adding a case, state its regime explicitly when it is not the suite
  default, and keep the label consistent with `intent` (per-command, bandwidth
  or compute bound).

## Mandatory Timing Boundary: Setup Is Untimed

All standard operation benchmarks must measure steady-state operation execution.
This rule applies to every backend and execution path (direct, eager, trace,
Python, Julia, CPU, and GPU). Preserving a legacy setup-inclusive timing scope
is not an acceptable reason to violate it.

Before starting the timer, complete:

- Input value generation, random-number generation, fixture construction,
  Tensor/EagerTensor wrapping, input allocation, copies, dtype/layout conversion,
  and host/device input transfers.
- Session, execution-session, context, runtime, backend, thread-pool, and library
  handle creation and initialization. Construct reusable resources once and
  reuse them across samples; do not create a session inside a timed closure.
- Graph construction, tracing, compilation, contraction planning, and JIT
  warmup, when these are preparation for the operation being compared.
- Per-sample input restoration and gradient-state reset needed to make repeated
  measurements equivalent. If an operation consumes an input or session,
  prepare its replacement outside that sample's timer.

Use an explicit setup / timed execution / validation-cleanup split. Setup must
remain outside timing even when warmups are configured to zero. Perform lazy
initialization explicitly before sampling; warmup alone is not a substitute for
correct timing boundaries.

Inside the timer, perform only the declared operation and the synchronization
needed to wait for its completion. Output allocation that is intrinsic to an
allocation-returning operation belongs inside timing; output-reuse benchmarks
must preallocate destinations and compare equivalent reuse APIs. Keep additional
output copies/materialization used only for inspection, validation, checksums,
and input/output destruction outside timing. Retain outputs until after the
clock stops. Metadata-only view operations must never include a data copy.

Apply the same boundary to all compared backends. Match the logical inputs,
requested outputs, differentiated arguments, and forward/backward scope. Label
execution paths accurately: direct API, EagerTensor, and compiled trace are
separate paths and must not share an ambiguous backend label.

Before publishing or interpreting performance ratios, inspect the timed closure
and its callees for setup work. Record the timing scope and setup policy in run
metadata or the report. Measurements that include setup must be identified as
noncompliant and rerun before being used as operation-performance evidence.
Initialization or end-to-end workflow costs may be measured only as explicitly
requested, separately named benchmarks; never mix them into operation tables.

## tenferro-rs Checkout Freshness

Before building or collecting benchmarks, inspect `extern/tenferro-rs`. If the
checkout is on the `main` branch, run `git pull` there so the build and
benchmark use the latest `main` revision. Preserve an explicitly detached or
pinned checkout unless the user asks to move it.

When creating a PR that includes benchmark results, add a PR comment listing
the exact commands used to collect those measurements, including relevant
environment-variable assignments and thread counts.

## Performance Issues Become Suite Cases

Performance issues of tenferro-rs get their workload added as suite cases,
with the issue number noted next to the case, when the issue is opened (not
at fix time). A performance-fix PR reports numbers from those cases. Add the
cases with their reference arm(s) to `scripts/generate_perf_issue_cases.py`
(`cpu/perf_issues`, `gpu/perf_issues`), or record the issue number on the
existing case that already covers the workload. Build the cases only from
public tenferro-rs APIs. This is the whole requirement: no automated audit,
registry, lint, scheduled run or mandatory baseline campaign.

## Overhead Smoke Trend

`cpu/overhead_smoke` (`scripts/run_overhead_smoke.sh`, generated by
`scripts/generate_overhead_smoke_cases.py`) is a ns/op + allocs/op trend view
for tenferro-rs #2178, run once and single-threaded by `run_all.sh` (so by both
nightly CPU workflows). It never gates. Its rows reuse the `cpu/perf_issues`
`umbrella_alloc_counter` fixtures; keep the two in step when a fixture changes.

## Target Profiles

Use target profiles to keep latest reports for multiple hardware classes:

- `zen5-cpu` (latest CPU generation: AMD Ryzen AI 9 HX 470 workstation, pinned rows)
- `nvidia-gpu` (latest GPU generation: A100)
- `mac-cpu`, `amd-cpu` (EPYC 7713P), `linux-cpu`, `mac-gpu`: older generations, kept
  for history only

`suite_id` remains the workload identity. Do not encode hardware in `suite_id`.

**Only the latest generation is linked.** The Pages site and the README link only
`zen5-cpu` and `nvidia-gpu` (`LATEST_PROFILES` in `scripts/generate_dashboard.py`),
because people mistake old data for current. Older profiles stay in `result/` and in git
history but are neither generated into the site nor linked. Moving to a new hardware
generation means changing `LATEST_PROFILES`.

The landing page, per-suite pages and `history.html` (one entry per commit that changed a
latest-generation report, with the tenferro-rs revision measured) are generated by the
Pages workflow on every deploy (`publish-results.yml`, `fetch-depth: 0`) from the
committed `result/` tree and git history. They are never committed (`_site/` is
git-ignored), so they cannot go stale.

Expected latest report paths:

- `result/mac-cpu/cpu/einsum.md`
- `result/mac-cpu/cpu/cpu_ops.md`
- `result/mac-cpu/cpu/public_api.md`
- `result/mac-cpu/cpu/linalg_jvp_vjp.md`
- `result/mac-cpu/cpu/permutation.md`
- `result/mac-cpu/cpu/overhead_smoke.md`
- `result/zen5-cpu/cpu/einsum.md`
- `result/zen5-cpu/cpu/cpu_ops.md`
- `result/zen5-cpu/cpu/public_api.md`
- `result/zen5-cpu/cpu/linalg_jvp_vjp.md`
- `result/zen5-cpu/cpu/permutation.md`
- `result/zen5-cpu/cpu/perf_issues.md`
- `result/zen5-cpu/cpu/overhead_smoke.md`
- `result/amd-cpu/cpu/einsum.md`
- `result/amd-cpu/cpu/cpu_ops.md`
- `result/amd-cpu/cpu/linalg_jvp_vjp.md`
- `result/amd-cpu/cpu/permutation.md`
- `result/amd-cpu/cpu/elementwise.md`
- `result/amd-cpu/cpu/perf_issues.md`
- `result/amd-cpu/cpu/overhead_smoke.md`
- `result/linux-cpu/cpu/linalg_jvp_jvp.md`
- `result/nvidia-gpu/gpu/dense.md`
- `result/nvidia-gpu/gpu/einsum.md`
- `result/nvidia-gpu/gpu/elementwise.md`
- `result/nvidia-gpu/gpu/sparse.md`
- `result/nvidia-gpu/gpu/tensornetwork.md`
- `result/nvidia-gpu/gpu/linalg_jvp_vjp.md`
- `result/nvidia-gpu/gpu/linalg_ad_latency.md`
- `result/nvidia-gpu/gpu/dense_throughput.md`
- `result/nvidia-gpu/gpu/linalg_ad_throughput.md`
- `result/nvidia-gpu/gpu/permutation.md`
- `result/nvidia-gpu/gpu/perf_issues.md`

Raw runs are written under:

```text
data/results/<target_profile>/<suite_id>/<timestamp>/
```

## CPU Backend Policy

CPU comparisons use:

- tenferro-rs trace
- tenferro-rs eager
- PyTorch Python
- JAX Python

C++ Torch/LibTorch is intentionally removed. Do not reintroduce LibTorch
runners or `Torch_DIR`. The opt-in `.devcontainer/openblas/` image source-builds
PyTorch Python to match tenferro's OpenBLAS provider; it is not a LibTorch runner.

On Linux CPU devcontainer runs, PyTorch uses the installed wheel's MKL-backed
provider. For fair CPU comparisons, run tenferro-rs with `blas-mkl` inside the
devcontainer rather than the default `blas-openblas` path. Record the detected
PyTorch provider in `run.yaml` using `torch.__config__.show()` and linked
library inspection.

## macOS CPU Workflow

Use native macOS execution. Do not use Docker for the standard Mac CPU path.

```bash
BENCHMARK_TARGET_PROFILE=mac-cpu ./scripts/run_all.sh 1
BENCHMARK_TARGET_PROFILE=mac-cpu ./scripts/run_all.sh 4
```

macOS BLAS-backed tenferro runs use Accelerate by default.

### Updating the macOS Public API Report

Collect the publication report on native macOS with the full profile. Run the
1-thread and 4-thread cases sequentially in one invocation:

```bash
PUBLICATION_GATE_PROFILE=full BENCHMARK_TARGET_PROFILE=mac-cpu \
  ./scripts/run_cpu_public_api.sh 1 4
```

The full profile uses 3 warmups and 15 measured runs per row. Before collection,
follow the tenferro-rs checkout freshness policy above and stop other benchmark,
compiler, and test processes. The runner's idle-host guard should remain enabled
for publication measurements; `BENCHMARK_ALLOW_BUSY_HOST=1` is for diagnostics
only. If the guard rejects the run, wait for the competing process to finish and
rerun instead of bypassing it.

The command writes raw runs under:

```text
data/results/mac-cpu/cpu/public_api/<timestamp>/
```

and updates the latest report at:

```text
result/mac-cpu/cpu/public_api.md
```

Keep these fairness rules when changing or extending the suite:

- Treat `benchmarks/cpu/public_api_coverage.yaml` as the operation-level coverage
  manifest. An API spelling may be an alias only when it reaches the same backend
  operation; do not blanket-alias distinct non-contiguous `TensorRead` paths.
- Construct equivalent logical fixture values in tenferro-rs, PyTorch, and JAX,
  while preserving each backend's native layout. Do not charge one backend for a
  layout conversion that another backend performs outside the timed region.
- Compare materializing operations with materializing operations. Keep metadata-
  only view benchmarks separate, under `cpu/view_metadata` where applicable.
- Compare tenferro-rs `_into` output reuse with PyTorch `out=` or `copy_` output
  reuse. Mark a backend unsupported when it has no equivalent operation.
- Do not synthesize a missing public operation by timing a composition of other
  operations. Keep unsupported combinations as explicit report rows.
- Keep the `direct` column limited to immediate public operations. Do not label an
  `EagerTensor` wrapper path as direct merely because it executes eagerly.

After collection, verify that the report names the new raw-run timestamp and
tenferro-rs commit, includes both thread counts, and contains no duplicate
`(suite, benchmark, dtype, threads, shape, backend)` rows. Inspect `run_t1.yaml`
and `run_t4.yaml` for the recorded backend, feature, and thread metadata. If a
run is noisy or incomplete, rerun the suite; do not hand-edit generated timing
values or status rows.

Run these checks after changing the public API suite or its update path:

```bash
bash -n scripts/run_cpu_public_api.sh scripts/benchmark_host_idle.sh
cargo check --no-default-features --features blas-accelerate --bin benchmark_cpu_public_api
uv run python -m py_compile scripts/benchmark_cpu_public_api_python.py \
  scripts/benchmark_cpu_public_api_jax.py scripts/format_cpu_ops_results.py
uv run python scripts/validate_benchmark_suite.py benchmarks/cpu/public_api.yaml
bash tests/test_suite_result_layout.sh
bash tests/test_run_all_docs_outputs.sh
```

When a PR includes the refreshed report, include the exact collection command
above in the PR comment, together with any additional relevant environment
variables, as required by the benchmark-result policy.

## zen5-cpu Workflow (pinned workstation rows)

`zen5-cpu` is the AMD Ryzen AI 9 HX 470 workstation (12 cores, SMT2, two L3 domains:
cores 0-3 with siblings 12-15, cores 4-11 with siblings 16-23); see the workspace
`AGENTS.md`, "Measurement host and core pinning". Unlike the other profiles, its runner
pins each thread row: `scripts/cpu_pinning.sh` holds the table and
`configure_cpu_thread_env` (`scripts/thread_env.sh`) applies it, so every suite script
that sets the thread environment also sets the affinity of itself and its children.

| threads | measured `taskset -c` | observer `taskset -c` |
|---|---|---|
| 1 | `0` | `4-11,16-23` |
| 4 | `0-3` | `4-11,16-23` |
| 8 | `4-11` | `0-3,12-15` |

- Only rows 1, 4 and 8 exist for `zen5-cpu`; another thread count is an error.
- The observer (the agent or shell that launched the run, builds, tests) must be moved to
  the observer set by the operator before timing; the runner records the set but cannot
  move another process. Build everything first: a build started inside a pinned row runs
  on the measured cores.
- `run.yaml` `environment.affinity` records the requested and actual measured mask, the
  observer mask, the thread budget, the L3 domains, the CPU boost state
  (`/sys/devices/system/cpu/cpufreq/boost`) and the scaling governor.
- `scripts/benchmark_host_idle.sh` additionally refuses to start while the measured CPUs
  are busy (host-wide `/proc/stat`, so it also works inside the devcontainer, whose process
  list cannot see host processes).
- The run procedure is in `docs/handoffs/2026-10-28-zen5-cpu-refresh.md`.

## Linux CPU Devcontainer Workflow

Use the devcontainer/Docker path for Linux CPU measurements. **Always run Linux
CPU benchmark collection inside the devcontainer.** Use the image provided by
this repository's `.devcontainer/devcontainer.json` and its referenced
Dockerfile for builds, validation, and measurement. Create or recreate it with
the devcontainer CLI; do not substitute an ad hoc Docker image. If `devcontainer`
is not installed on PATH, use `npx --yes @devcontainers/cli` for the same commands.
The default image uses
`blas-mkl` to match the PyTorch wheel. For provider-matched OpenBLAS comparisons,
use `.devcontainer/openblas/devcontainer.json` and `blas-openblas` instead;
see `docs/linux-cpu-devcontainer.md`. Keep its `BENCHMARK_TORCH_WHEEL` setting
when recreating `.venv`, and verify the runtime provider before collecting.
Do not present differences from historical MKL runs as implementation speedups.

Before building or collecting benchmarks, check that the existing devcontainer
was created from the current `.devcontainer/Dockerfile` and
`.devcontainer/devcontainer.json`. If it predates relevant devcontainer changes
or required tools and libraries are missing, recreate it with
`devcontainer up --workspace-folder . --remove-existing-container`. In
particular, verify that `MKLROOT` resolves to an installed oneMKL tree before a
`blas-mkl` run; do not silently fall back to an older image or another BLAS
backend.

```bash
devcontainer up --workspace-folder .
devcontainer exec --workspace-folder . bash -lc '
  export TENFERRO_CPU_FEATURES=blas-mkl
  export PUBLICATION_GATE_FEATURES=blas-mkl
  export BENCHMARK_TARGET_PROFILE=amd-cpu
  ./scripts/run_all.sh 1'
devcontainer exec --workspace-folder . bash -lc '
  export TENFERRO_CPU_FEATURES=blas-mkl
  export PUBLICATION_GATE_FEATURES=blas-mkl
  export BENCHMARK_TARGET_PROFILE=amd-cpu
  ./scripts/run_all.sh 4'
```

Intel oneMKL is installed in the default Linux CPU devcontainer under
`/opt/intel/oneapi/mkl/latest` and exported as `MKLROOT`. PyTorch uses the
installed wheel provider, which is MKL-backed on this path.

The `cpu/permutation` suite is a standalone entry point (not wired into
`run_all.sh`). Collect its `amd-cpu` report inside the same devcontainer; HPTT
builds against the installed `cmake` + `libomp-dev`, and the Julia runners
(`julia-base` / `strided-jl`) use the juliaup-provided Julia baked into the
image:

```bash
devcontainer exec --workspace-folder . bash -lc '
  export TENFERRO_CPU_FEATURES=blas-mkl
  export BENCHMARK_TARGET_PROFILE=amd-cpu
  export PERMUTATION_EXTRA_FEATURES=hptt
  ./scripts/run_permutation.sh 1 4'
```

Thread counts are controlled only via the runner's thread environment
(`RAYON_NUM_THREADS` / `OMP_NUM_THREADS` / `JULIA_NUM_THREADS`, recorded per
thread count in `run_t<N>.yaml`); no `taskset` / `numactl` CPU-affinity pinning
is applied to `amd-cpu`, matching the other devcontainer CPU suites (`zen5-cpu` rows are
pinned, see above).

`cpu/elementwise` is a second standalone entry point (also not wired into
`run_all.sh`; opt in there with `RUN_ELEMENTWISE_SUITE=1`). It measures f64
`sqrt` / `rsqrt` / `exp` under the default and a raised CPU buffer-pool limit,
so its `default`/`raised` ratio is an allocation/first-touch signal rather than
a kernel comparison. It needs no BLAS and no devcontainer; see
`docs/elementwise-suite.md`:

```bash
BENCHMARK_TARGET_PROFILE=amd-cpu ./scripts/run_elementwise.sh 1 4
```

`OPENBLAS_ROOT=/opt/openblas` is also configured for tenferro `blas-openblas`
runs, but treat that as an alternate backend for experiments, not the standard
Linux CPU comparison path. The devcontainer OpenBLAS is source-built with
threading enabled; verify this with the OpenBLAS runtime API, not `strings`,
because parallel builds can still contain standalone diagnostic strings:

```bash
devcontainer exec --workspace-folder . bash -lc 'python3 - <<PY
import ctypes
lib = ctypes.CDLL("/opt/openblas/lib/libopenblas.so")
lib.openblas_get_config.restype = ctypes.c_char_p
lib.openblas_get_parallel.restype = ctypes.c_int
print(lib.openblas_get_config().decode())
print(f"parallel={lib.openblas_get_parallel()}")
PY'
```

## Local Linux Linalg AD Repro

For the local Linux CPU linalg JVP/VJP repro report, use:

```bash
./scripts/reproduce_linux_cpu_linalg_jvp_jvp.sh
```

By default the script runs 1T and 4T sequentially, with JAX forced to CPU, then
writes the combined report to:

```text
result/linux-cpu/cpu/linalg_jvp_jvp.md
```

The benchmark runner still uses the maintained `amd-cpu` target profile for
collection, then mirrors raw runs and report paths under the `linux-cpu` report
alias. To run specific thread counts, pass them explicitly:

```bash
./scripts/reproduce_linux_cpu_linalg_jvp_jvp.sh 1 4
./scripts/reproduce_linux_cpu_linalg_jvp_jvp.sh 4
```

When updating this report, **run it inside the Linux devcontainer with tenferro-rs
`blas-mkl`** so the report matches PyTorch's MKL-backed CPU path:

```bash
devcontainer up --workspace-folder .
devcontainer exec --workspace-folder . bash -lc '
  export TENFERRO_CPU_FEATURES=blas-mkl
  export PUBLICATION_GATE_FEATURES=blas-mkl
  ./scripts/reproduce_linux_cpu_linalg_jvp_jvp.sh'
```

For an alternate tenferro OpenBLAS run:

```bash
devcontainer exec --workspace-folder . bash -lc '
  export TENFERRO_CPU_FEATURES=blas-openblas
  export PUBLICATION_GATE_FEATURES=blas-openblas
  ./scripts/reproduce_linux_cpu_linalg_jvp_jvp.sh'
```

CPU benchmark entrypoints reset `.venv` before collection so stale Python
wheels from another backend do not affect the run. Keep linalg repro collection
sequential: one devcontainer benchmark command at a time, with no concurrent
`run_all.sh`, `run_cpu_ops.sh`, or ad hoc linalg probes.

## GPU Devcontainer Workflow

For local NVIDIA GPU measurements, use the CUDA devcontainer. Managed A100
collection uses the existing `.github/workflows/benchmark-runpod-gpu.yml`:
it provisions an ephemeral RunPod runner, restores the prepared CUDA runtime,
verifies a freshly built binary against its source commits and digest, and
deletes the pod after collection. See `docs/runpod-gpu-benchmark.md`. Repository
Secrets provide the credentials; no local GPU, API key, persistent runner, or
SSH host is needed to dispatch this workflow. The `perf-issues-readback`
selection runs the solve/QR correctness gate before sequential timing.

The CUDA devcontainer mounts a container-only named volume over
`<workspace>/.venv`, so the host and container Python environments are separate.
A host `uv run`/`uv sync` cannot rewrite the container's CUDA PyTorch/JAX wheels
or interpreter. Run Python inside the container for GPU work; use host `python3`
only for pure-Python checks that do not import `torch`. The CPU devcontainers
deliberately keep a resettable bind-mounted `.venv` instead, because their
collection scripts delete and recreate it.

Collect `nvidia-gpu` reports on an NVIDIA A100 (any of the 80GB PCIe, SXM4 80GB
or SXM4 40GB models; the standard GPU benchmark hardware since 2026-10-17, and
since 2026-10-20 any A100 model, recorded per run); do not mix in results from
non-A100 GPUs. The A100 80GB PCIe host has a single GPU (physical GPU 0): verification-only lanes
default to GPU 1, so pass `GPU_BENCH_DEVICE=0` there and never run them
concurrently with timing collection.

On a two-GPU host, reserve physical GPU 0 for benchmark collection
and GPU 1 for debugging/numerical verification. The public API suite implements
this split with `./scripts/run_gpu_public_api.sh <family> --verify-only` (GPU 1)
and `./scripts/run_gpu_public_api.sh <family>` (GPU 0). It selects the physical
card by UUID, exposes it as process-local CUDA ordinal 0, and records both indices
and the role. See `docs/gpu-public-api-suite.md`. Keep timing collection
sequential; heavy builds and CPU-intensive debug work must not overlap it even
when targeting the other GPU.

```bash
devcontainer up --workspace-folder . --config .devcontainer/cuda/devcontainer.json
devcontainer exec --workspace-folder . --config .devcontainer/cuda/devcontainer.json \
  bash -lc 'BENCHMARK_TARGET_PROFILE=nvidia-gpu ./scripts/run_gpu_suite.sh'
```

An unfiltered `BENCH_COVERAGE=full` run also writes the quick-profile routine
coverage and control reports (`routine_coverage.md`, `routine_controls.md`)
from the same records restricted to the quick manifest selection; the
full-profile reports are the `routine_full_*.md` variants. The quick
`routine_workflow.md` wall-time observation is not derived from a full run
(its clocks would be the full campaign's), so a full run never overwrites it.
The diagnostic `jax-cuda-eager` tensor-network path is opt-in via
`GPU_BENCH_BACKENDS=jax-cuda-eager` (see `docs/gpu-routine-coverage.md`).

For the GPU linalg JVP/VJP report (CPU `linalg_jvp_vjp` parity on CUDA):

```bash
devcontainer exec --workspace-folder . --config .devcontainer/cuda/devcontainer.json \
  bash -lc 'BENCHMARK_TARGET_PROFILE=nvidia-gpu ./scripts/run_gpu_linalg_jvp_vjp.sh'
```

Expected report path:

```text
result/nvidia-gpu/gpu/linalg_jvp_vjp.md
```

Run this sequentially after the standard GPU suite; do not overlap it with
other GPU benchmark processes.

For the GPU elementwise chain report (prepared/segmented fusion detection; see
`benchmarks/gpu/elementwise.yaml`):

```bash
devcontainer exec --workspace-folder . --config .devcontainer/cuda/devcontainer.json \
  bash -lc 'BENCHMARK_TARGET_PROFILE=nvidia-gpu GPU_BENCH_SUITE=benchmarks/gpu/elementwise.yaml ./scripts/run_gpu_suite.sh'
```

Expected report path:

```text
result/nvidia-gpu/gpu/elementwise.md
```

The chain is `tanh(t * a + b)` repeated eight times, so the whole chain is
fusable. The small problem (n=1024) measures per-command overhead and the large
one (n=1048576) measures the bandwidth cost; a prepared/segmented execution-path
divergence shows up as the trace rows sitting far above the PyTorch rows.

For the GPU linalg AD single-call latency / per-op overhead report (n=2, 4, 8):

```bash
devcontainer exec --workspace-folder . --config .devcontainer/cuda/devcontainer.json \
  bash -lc 'BENCHMARK_TARGET_PROFILE=nvidia-gpu ./scripts/run_gpu_linalg_ad_latency.sh'
```

Expected report path:

```text
result/nvidia-gpu/gpu/linalg_ad_latency.md
```

Device kernel time is a small fraction of each row in this suite, so these rows
are latency/overhead diagnostics and must not be reported as GPU throughput.
GPU-sized AD rows stay in `gpu/linalg_jvp_vjp`. Run it sequentially; do not
overlap it with other GPU benchmark processes.

For the GPU throughput mode (deferred batch) reports — `K` prepared runs share
one retirement, so per-operation time is `sample_ms / K` (see
`docs/gpu-throughput-mode.md`):

```bash
devcontainer exec --workspace-folder . --config .devcontainer/cuda/devcontainer.json \
  bash -lc 'BENCHMARK_TARGET_PROFILE=nvidia-gpu ./scripts/run_gpu_throughput.sh'
```

Expected report paths:

```text
result/nvidia-gpu/gpu/dense_throughput.md
result/nvidia-gpu/gpu/linalg_ad_throughput.md
```

Throughput rows carry their own backend identity
(`tenferro-cuda-trace-deferred-batch`) and their own suite ids; a latency row
and a throughput row must never share a row, ratio, average or summary. Run it
sequentially; do not overlap it with other GPU benchmark processes.

For the GPU permutation / materialize-kernel report (CUDA port of
`cpu/permutation`; see `docs/gpu-permutation-suite.md`):

```bash
devcontainer exec --workspace-folder . --config .devcontainer/cuda/devcontainer.json \
  bash -lc 'BENCHMARK_TARGET_PROFILE=nvidia-gpu ./scripts/run_gpu_permutation.sh'
```

Expected report path:

```text
result/nvidia-gpu/gpu/permutation.md
```

This is a standalone entry point, like `scripts/run_permutation.sh` is for
`cpu/permutation`: it is not wired into `scripts/run_gpu_suite.sh`. Run it
sequentially; do not overlap it with other GPU benchmark processes. The
correctness gate defaults to exact sampled verification (per-axis corners plus
roughly 1M seeded output elements, gathered on device with a small readback);
set `PERMUTATION_FULL_VERIFY=1` to restore the full host
reference. The mode is recorded in `run.yaml` and the report.

If vendor libraries are needed:

```bash
devcontainer exec --workspace-folder . --config .devcontainer/cuda/devcontainer.json \
  bash -lc './scripts/setup_gpu_vendors.sh all'
```

## Container User and Build Artifact Ownership

Builds write into the bind-mounted workspace, so the container user's UID owns
`target/` on these hosts afterwards. On the A100 host `primerose` the login
account is `shinaoka` (UID 2000), a container `vscode` that was never UID-remapped
is 1000 (the host's unrelated `user` account), and root is 0.

Only the Dev Containers CLI / VS Code extension path remaps `vscode` to the host
UID (`updateRemoteUserUID`, the reason the images are tagged `vsc-...-uid`). Start
the same image any other way and it keeps its own user: `.devcontainer/cuda/Dockerfile`
ends with `USER root`, and the `-uid` image also reports `Config.User = root`, so a
container created without `--user vscode` runs every command as root. A `cargo`
build in that shell leaves `target/` owned by root, which the host account can no
longer delete or reuse.

- Run benchmark commands through `devcontainer exec` (or `docker exec -u vscode`),
  never as root.
- Do not `sudo` a build, and do not keep a root shell open "just for cargo".
- `CARGO_HOME` in the CUDA config points inside the workspace (`.cargo-home`), so it
  inherits the same ownership.
- Foreign-owned artifacts stay inside gitignored build output (`target/`,
  `.cargo-home/`). Check before treating one as a config bug:
  `find target ! -user "$(id -un)" | wc -l`, then remove it with `sudo rm -rf`
  instead of rebuilding around it.

`docs/gpu-devcontainer.md` states the same rule for the CUDA `.venv` volume.

## GPU Timing Fairness

GPU benchmark timings must synchronize queued device work without using a full
result download as the synchronization primitive. Downloading an output tensor
inside the timed region adds output-size-dependent D2H transfer cost. Omitting
synchronization can overstate asynchronous backends by measuring only job
submission.

Keep timed regions scoped to host API dispatch plus backend-native device
synchronization. Perform output downloads only after timing for verification.
Always rebuild the Rust GPU benchmark binary before measuring so
synchronization fixes and timing metadata changes cannot be hidden by a stale
`target/release/benchmark_gpu_rust`.

## Useful Checks

Run these after changing benchmark scripts:

```bash
uv run python scripts/validate_benchmark_suite.py benchmarks/cpu/einsum.yaml
uv run python scripts/validate_benchmark_suite.py benchmarks/gpu/dense.yaml benchmarks/gpu/einsum.yaml benchmarks/gpu/sparse.yaml benchmarks/gpu/linalg_jvp_vjp.yaml
uv run python -m unittest tests.test_generate_dashboard
bash tests/test_suite_result_layout.sh
bash tests/test_run_all_docs_outputs.sh
bash tests/test_cpu_ops_linalg_ad.sh
bash tests/test_linalg_ad_results_formatter.sh
bash tests/test_clean_extern_deps.sh
bash tests/test_setup_extern_tenferro_checkout.sh
cmake -S cpp -B build/cpp-plan-test
cmake --build build/cpp-plan-test --target einsum_plan_test
ctest --test-dir build/cpp-plan-test --output-on-failure
```

## Route coverage and regression detection

See `docs/regression-detection.md`. Keep coverage (`BENCH_COVERAGE=quick|full`,
versioned manifests under `benchmarks/cpu/manifests/`) separate from effort
(`BENCH_EFFORT=scan|standard|confirm|aa`). A scan only flags suspects; a
performance claim needs a paired, balanced, A/A-characterized confirmation run
with a declared `benchmarks/cpu/confirmation.yaml` (never fill its thresholds
from candidate data). Unsupported, failed, missing and unselected
cases never count as covered. Regenerate `cpu/session_matrix` cases with
`scripts/generate_session_matrix_cases.py` and bump its manifest version when
cases change. Checks:

```bash
uv run python -m unittest tests/test_session_matrix_suite.py \
  tests/test_route_coverage.py tests/test_regression_detector.py tests/test_small_work_suite.py
uv run python scripts/route_coverage.py entrypoints
bash tests/test_paired_timing_guard.sh
cargo test --bin benchmark_cpu_session
```

## Short-operation sampling

For very short CPU/GPU operations, measure many declared operations in one
wall-clock interval. Repeating isolated per-call clocks is not a substitute.
Prepare all inputs, output-retention containers and the reusable session before
starting the clock; keep outputs alive until it stops. Record operations per
sample, batch duration and the normalized per-operation duration. Bound batch
size by retained input/output memory as well as target elapsed time.

Reuse one entered session for the batch wherever the API supports it. Reusing
an EagerRuntime or Runtime alone does not establish that property. If an API
internally enters a new session per call and cannot accept the borrowed session,
keep its short-operation measurement explicitly diagnostic; do not silently
label it shared-session or use it as the standard small-work comparison.
Pure metadata operations that do not require a session must say so.

The GPU runners keep the device busy for `GPU_BENCH_CLOCK_RAMP_MS` (default
300 ms) before each timed case. A millisecond-scale case with a couple of
warmups otherwise samples a ramping clock and flips between two run-to-run
modes (observed as ~12% on `dense_solve_f64_512_rhs16`). CPU verification
between cases leaves the GPU idle long enough for its clocks to drop again, so
a process-start ramp only warms the first case. Leave the ramp enabled for
timing collection; do not raise per-case warmups for it.
