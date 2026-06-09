# GPU Tensor Network Benchmarks

This suite runs the TensorNetworkBenchmarks **problem definition** (JSON tree +
Float32 tensors) through tenferro-rs CUDA backends and other GPU framework
baselines. OMEinsum.jl is intentionally out of scope.

## Problem

- Source: `extern/TensorNetworkBenchmarks/data/tensornetwork_permutation_optimized.json`
- 550 input tensors, bond dimension 2
- Float32 fill value `0.5^0.4`
- Pre-optimized contraction **tree** execution (not opt_einsum path replay)

## Comparison backends

| Backend | Role |
|---|---|
| `tenferro-cuda-eager` | tenferro-rs eager einsum on CUDA |
| `tenferro-cuda-trace` | tenferro-rs trace + GraphExecutor on CUDA |
| `pytorch-cuda` | PyTorch `torch.einsum` baseline |
| ~~`jax-cuda`~~ | *(temporarily disabled in `benchmarks/gpu/tensornetwork.yaml`)* |

All backends follow the same tree-guided contraction schedule and CUDA
synchronization contract as the other GPU suites.

## Run

Inside the CUDA devcontainer. `run_gpu_suite.sh` runs `gpu/tensornetwork` serially
(one backend per process) so PyTorch and JAX do not share GPU memory in one Python
interpreter.

```bash
BENCHMARK_TARGET_PROFILE=nvidia-gpu \
GPU_BENCH_SUITE=benchmarks/gpu/tensornetwork.yaml \
GPU_BENCH_BACKENDS=tenferro-cuda-eager,tenferro-cuda-trace,pytorch-cuda \
  ./scripts/run_gpu_suite.sh
```

To restore JAX later, uncomment `- jax-cuda` in `benchmarks/gpu/tensornetwork.yaml`.

`./scripts/run_tensornetwork_serial.sh` is a thin wrapper around the same path.

Latest report: `result/nvidia-gpu/gpu/tensornetwork.md`

## Note on `extern/TensorNetworkBenchmarks`

The submodule supplies the benchmark **instance JSON** only. We do not run its
Julia/Makefile targets or include OMEinsum.jl in reports. Use it when you need
the upstream problem file or optional PyTorch-only cross-checks on the same
hardware.
