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
- `result/amd-cpu/cpu/einsum.md`
- `result/amd-cpu/cpu/cpu_ops.md`
- `result/nvidia-gpu/gpu/dense.md`
- `result/nvidia-gpu/gpu/einsum.md`
- `result/nvidia-gpu/gpu/sparse.md`

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

Linux PyTorch CPU provider is not forced to match tenferro. Record the detected
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

Use the devcontainer/Docker path for Linux CPU measurements.

```bash
devcontainer up --workspace-folder .
devcontainer exec --workspace-folder . bash -lc \
  'BENCHMARK_TARGET_PROFILE=amd-cpu ./scripts/run_all.sh 1'
devcontainer exec --workspace-folder . bash -lc \
  'BENCHMARK_TARGET_PROFILE=amd-cpu ./scripts/run_all.sh 4'
```

`OPENBLAS_ROOT=/opt/openblas` is configured inside the devcontainer for
tenferro `system-openblas` runs. PyTorch uses the installed wheel provider.

## GPU Devcontainer Workflow

Use the CUDA devcontainer for NVIDIA GPU measurements.

```bash
devcontainer up --workspace-folder . --config .devcontainer/cuda/devcontainer.json
devcontainer exec --workspace-folder . --config .devcontainer/cuda/devcontainer.json \
  bash -lc 'BENCHMARK_TARGET_PROFILE=nvidia-gpu ./scripts/run_gpu_suite.sh'
```

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
uv run python scripts/validate_benchmark_suite.py benchmarks/gpu/dense.yaml benchmarks/gpu/einsum.yaml benchmarks/gpu/sparse.yaml
bash tests/test_suite_result_layout.sh
bash tests/test_run_all_docs_outputs.sh
bash tests/test_clean_extern_deps.sh
bash tests/test_setup_extern_tenferro_checkout.sh
cmake -S cpp -B build/cpp-plan-test
cmake --build build/cpp-plan-test --target einsum_plan_test
ctest --test-dir build/cpp-plan-test --output-on-failure
```
