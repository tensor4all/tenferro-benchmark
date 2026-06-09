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
| `jax-cuda` | JAX `jax.numpy.einsum` baseline |

All backends follow the same tree-guided contraction schedule and CUDA
synchronization contract as the other GPU suites.

## Run

Inside the CUDA devcontainer:

```bash
BENCHMARK_TARGET_PROFILE=nvidia-gpu \
GPU_BENCH_SUITE=benchmarks/gpu/tensornetwork.yaml \
GPU_BENCH_BACKENDS=tenferro-cuda-eager,tenferro-cuda-trace,pytorch-cuda,jax-cuda \
  ./scripts/run_gpu_suite.sh
```

Latest report: `result/nvidia-gpu/gpu/tensornetwork.md`

## Note on `extern/TensorNetworkBenchmarks`

The submodule supplies the benchmark **instance JSON** only. We do not run its
Julia/Makefile targets or include OMEinsum.jl in reports. Use it when you need
the upstream problem file or optional PyTorch-only cross-checks on the same
hardware.
