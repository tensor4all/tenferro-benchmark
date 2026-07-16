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

## Target Profiles

Use target profiles to keep latest reports for multiple hardware classes:

- `mac-cpu`
- `amd-cpu`
- `nvidia-gpu`

`suite_id` remains the workload identity. Do not encode hardware in `suite_id`.

Expected latest report paths:

- `result/mac-cpu/cpu/einsum.md`
- `result/mac-cpu/cpu/cpu_ops.md`
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

## Linux CPU Devcontainer Workflow

Use the devcontainer/Docker path for Linux CPU measurements. **Always run Linux
CPU benchmark collection inside the devcontainer, and prefer tenferro-rs
`system-mkl` there** so tenferro-rs and PyTorch share the same MKL-backed BLAS
stack.

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
