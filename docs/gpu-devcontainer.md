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
`/opt`. The default Ginkgo build targets CUDA architecture 86 (RTX 30-series).
Override the `GINKGO_CUDA_ARCHITECTURES` Docker build argument when building
for a different architecture.

The container also sets `UV_NO_SYNC=1`: `postCreateCommand` installs the CUDA
variants of PyTorch and JAX after the locked base environment, and later
benchmark-time `uv run` calls must not replace them with the default wheels.

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

`Runtime::run_prepared` cannot borrow an entered session. Its standard large
operation rows are timing-unsupported even if numerical execution passes.
`gpu/linalg_ad_latency` explicitly measures repeated public calls including
internal session entry, using `public_path_latency_including_internal_session_entry`.
These small rows are latency diagnostics, not shared-session or GPU throughput
comparisons.

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
