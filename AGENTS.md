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

## Target Profiles

Use target profiles to keep latest reports for multiple hardware classes:

- `mac-cpu`
- `amd-cpu`
- `nvidia-gpu`

`suite_id` remains the workload identity. Do not encode hardware in `suite_id`.

Expected latest report paths:

- `result/mac-cpu/cpu/einsum.md`
- `result/mac-cpu/cpu/cpu_ops.md`
- `result/mac-cpu/cpu/public_api.md`
- `result/mac-cpu/cpu/linalg_jvp_vjp.md`
- `result/mac-cpu/cpu/permutation.md`
- `result/amd-cpu/cpu/einsum.md`
- `result/amd-cpu/cpu/cpu_ops.md`
- `result/amd-cpu/cpu/linalg_jvp_vjp.md`
- `result/amd-cpu/cpu/permutation.md`
- `result/linux-cpu/cpu/linalg_jvp_jvp.md`
- `result/nvidia-gpu/gpu/dense.md`
- `result/nvidia-gpu/gpu/einsum.md`
- `result/nvidia-gpu/gpu/sparse.md`
- `result/nvidia-gpu/gpu/linalg_jvp_vjp.md`
- `result/nvidia-gpu/gpu/permutation.md`

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
runners, `Torch_DIR`, or OpenBLAS-linked PyTorch source-build setup.

On Linux CPU devcontainer runs, PyTorch uses the installed wheel's MKL-backed
provider. For fair CPU comparisons, run tenferro-rs with `system-mkl` inside the
devcontainer rather than the default `system-openblas` path. Record the detected
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
cargo check --features system-accelerate --bin benchmark_cpu_public_api
uv run python -m py_compile scripts/benchmark_cpu_public_api_python.py \
  scripts/benchmark_cpu_public_api_jax.py scripts/format_cpu_ops_results.py
uv run python scripts/validate_benchmark_suite.py benchmarks/cpu/public_api.yaml
bash tests/test_suite_result_layout.sh
bash tests/test_run_all_docs_outputs.sh
```

When a PR includes the refreshed report, include the exact collection command
above in the PR comment, together with any additional relevant environment
variables, as required by the benchmark-result policy.

## Linux CPU Devcontainer Workflow

Use the devcontainer/Docker path for Linux CPU measurements. **Always run Linux
CPU benchmark collection inside the devcontainer, and prefer tenferro-rs
`system-mkl` there** so tenferro-rs and PyTorch share the same MKL-backed BLAS
stack.

Before building or collecting benchmarks, check that the existing devcontainer
was created from the current `.devcontainer/Dockerfile` and
`.devcontainer/devcontainer.json`. If it predates relevant devcontainer changes
or required tools and libraries are missing, recreate it with
`devcontainer up --workspace-folder . --remove-existing-container`. In
particular, verify that `MKLROOT` resolves to an installed oneMKL tree before a
`system-mkl` run; do not silently fall back to an older image or another BLAS
backend.

```bash
devcontainer up --workspace-folder .
devcontainer exec --workspace-folder . bash -lc '
  export TENFERRO_CPU_FEATURES=system-mkl
  export PUBLICATION_GATE_FEATURES=system-mkl
  export TENFERRO_CPU_BACKEND_KIND=blas
  export BENCHMARK_TARGET_PROFILE=amd-cpu
  ./scripts/run_all.sh 1'
devcontainer exec --workspace-folder . bash -lc '
  export TENFERRO_CPU_FEATURES=system-mkl
  export PUBLICATION_GATE_FEATURES=system-mkl
  export TENFERRO_CPU_BACKEND_KIND=blas
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
  export TENFERRO_CPU_FEATURES=system-mkl
  export TENFERRO_CPU_BACKEND_KIND=blas
  export BENCHMARK_TARGET_PROFILE=amd-cpu
  export PERMUTATION_EXTRA_FEATURES=hptt
  ./scripts/run_permutation.sh 1 4'
```

Thread counts are controlled only via the runner's thread environment
(`RAYON_NUM_THREADS` / `OMP_NUM_THREADS` / `JULIA_NUM_THREADS`, recorded per
thread count in `run_t<N>.yaml`); no `taskset` / `numactl` CPU-affinity pinning
is applied, matching the other devcontainer CPU suites.

`OPENBLAS_ROOT=/opt/openblas` is also configured for tenferro `system-openblas`
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
`system-mkl`** so the report matches PyTorch's MKL-backed CPU path:

```bash
devcontainer up --workspace-folder .
devcontainer exec --workspace-folder . bash -lc '
  export TENFERRO_CPU_FEATURES=system-mkl
  export PUBLICATION_GATE_FEATURES=system-mkl
  export TENFERRO_CPU_BACKEND_KIND=blas
  ./scripts/reproduce_linux_cpu_linalg_jvp_jvp.sh'
```

For an alternate tenferro OpenBLAS run:

```bash
devcontainer exec --workspace-folder . bash -lc '
  export TENFERRO_CPU_FEATURES=system-openblas
  export PUBLICATION_GATE_FEATURES=system-openblas
  export TENFERRO_CPU_BACKEND_KIND=blas
  ./scripts/reproduce_linux_cpu_linalg_jvp_jvp.sh'
```

CPU benchmark entrypoints reset `.venv` before collection so stale Python
wheels from another backend do not affect the run. Keep linalg repro collection
sequential: one devcontainer benchmark command at a time, with no concurrent
`run_all.sh`, `run_cpu_ops.sh`, or ad hoc linalg probes.

## GPU Devcontainer Workflow

Use the CUDA devcontainer for NVIDIA GPU measurements.

```bash
devcontainer up --workspace-folder . --config .devcontainer/cuda/devcontainer.json
devcontainer exec --workspace-folder . --config .devcontainer/cuda/devcontainer.json \
  bash -lc 'BENCHMARK_TARGET_PROFILE=nvidia-gpu ./scripts/run_gpu_suite.sh'
```

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
sequentially; do not overlap it with other GPU benchmark processes.

If vendor libraries are needed:

```bash
devcontainer exec --workspace-folder . --config .devcontainer/cuda/devcontainer.json \
  bash -lc './scripts/setup_gpu_vendors.sh all'
```

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
