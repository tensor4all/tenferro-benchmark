# tenferro-benchmark

Benchmark suite for [tenferro-rs](https://github.com/tensor4all/tenferro-rs).

The repository keeps the latest human-facing reports per target profile:

- macOS CPU: `result/mac-cpu/cpu/einsum.md`, `result/mac-cpu/cpu/cpu_ops.md`
- Linux/AMD CPU: `result/amd-cpu/cpu/einsum.md`, `result/amd-cpu/cpu/cpu_ops.md`
- NVIDIA GPU: `result/nvidia-gpu/gpu/dense.md`, `result/nvidia-gpu/gpu/einsum.md`, `result/nvidia-gpu/gpu/sparse.md`

Historical reports are not archived in extra files. Use git history when older
results are needed.

## Workflows

- [macOS CPU workflow](docs/macos-cpu.md): native run, no Docker, Accelerate.
- [Linux CPU devcontainer workflow](docs/linux-cpu-devcontainer.md): Docker/devcontainer, OpenBLAS for tenferro, detected PyTorch provider.
- [NVIDIA GPU devcontainer workflow](docs/gpu-devcontainer.md): CUDA devcontainer.
- [Einsum suite and instance selection](docs/einsum-suite.md): source benchmark, selection rules, diagnostic cases, path strategies.
- [Result layout and metadata](docs/results.md): `target_profile`, `suite_id`, `run.yaml`, latest reports.
- [Architecture terminology](docs/architecture.md): suite, runner, backend, strategy, target profile.
- [PyTorch einsum dispatch notes](docs/pytorch-einsum-dispatch.md): PyTorch source investigation notes.

## Quick Smoke

macOS:

```bash
uv sync
./scripts/setup_extern_deps.sh
BENCHMARK_TARGET_PROFILE=mac-cpu \
BENCH_INSTANCE=bin_matmul_256 \
BENCH_RUNS=1 \
BENCH_WARMUPS=0 \
PUBLICATION_GATE_SUITE=small \
  ./scripts/run_all.sh 1
```

Linux devcontainer from the host:

```bash
devcontainer up --workspace-folder .
devcontainer exec --workspace-folder . bash -lc '\
  BENCHMARK_TARGET_PROFILE=amd-cpu \
  BENCH_INSTANCE=bin_matmul_256 \
  BENCH_RUNS=1 \
  BENCH_WARMUPS=0 \
  PUBLICATION_GATE_SUITE=small \
    ./scripts/run_all.sh 1'
```

GPU devcontainer from the host:

```bash
devcontainer up --workspace-folder . --config .devcontainer/cuda/devcontainer.json
devcontainer exec --workspace-folder . --config .devcontainer/cuda/devcontainer.json \
  bash -lc 'BENCHMARK_TARGET_PROFILE=nvidia-gpu ./scripts/run_gpu_suite.sh'
```

## Comparison Backends

CPU reports compare:

- `tenferro-trace`
- `tenferro-eager`
- `pytorch-cpu`
- `jax-cpu`

GPU reports compare:

- `tenferro-cuda-trace`
- `tenferro-cuda-eager`
- `pytorch-cuda`
- `jax-cuda`
- vendor-specific CUDA backends where meaningful

C++ Torch/LibTorch runners are intentionally removed. PyTorch Python is the
ATen comparison backend. The PyTorch CPU provider is detected at run time and
recorded in `run.yaml` and generated reports; Linux does not source-build
PyTorch to force OpenBLAS.

## Development Checks

Run these after changing benchmark scripts or schemas:

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

## License

MIT
