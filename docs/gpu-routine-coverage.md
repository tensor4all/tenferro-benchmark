# Routine GPU coverage

`scripts/run_gpu_suite.sh` defaults to `BENCH_COVERAGE=quick` when no suite,
backend or problem filter is supplied. Explicit filters default to `full`;
an explicit `BENCH_COVERAGE` takes precedence. Coverage selects workloads;
`BENCH_EFFORT` independently selects sampling effort.

```bash
devcontainer exec --workspace-folder . --config .devcontainer/cuda/devcontainer.json \
  bash -lc 'GPU_BENCH_DEVICE=0 BENCH_COVERAGE=quick BENCH_EFFORT=standard \
    BENCHMARK_TARGET_PROFILE=nvidia-gpu ./scripts/run_gpu_suite.sh'

devcontainer exec --workspace-folder . --config .devcontainer/cuda/devcontainer.json \
  bash -lc 'GPU_BENCH_DEVICE=0 BENCH_COVERAGE=full BENCH_EFFORT=standard \
    BENCHMARK_TARGET_PROFILE=nvidia-gpu ./scripts/run_gpu_suite.sh'
```

The versioned manifest is `benchmarks/gpu/manifests/routine.yaml`. Quick selects
27 problem/backend pairs: seven dense operations, one public einsum operation
and one complete tensor-network contraction tree, each with tenferro eager,
PyTorch and JAX. The small batched matmul retains the reported overhead regime;
512/2048 solves retain distinct size regimes. Full preserves the broad vendor
matrix, large batch, larger/adjacent solves, sparse references and alternate
JAX tensor-network execution paths (95 pairs). Optional elementwise suites
participate when explicitly requested. Unsupported implementations remain
unsupported; selection does not establish implementation availability.

Every invocation writes effective suite YAMLs and `selection.json` under
`data/results/nvidia-gpu/gpu/routine_selection/<timestamp>/`. The selection
records manifest version, coverage, effort, expected and selected pair IDs,
and full-profile IDs left unselected. Each suite's run metadata points to this
selection. Verification-only commands use `data/verification` instead and
default to physical GPU1; standard timing defaults to physical GPU0.

After collection, `case_status.json` distinguishes executed, unsupported,
failed, missing, verification-only, raw-samples-unavailable and noisy pairs.
Only successful numerical validation with retained positive raw batch samples
counts as executed. Verification-only and unselected pairs never count as
measured coverage. Duplicate or unexpected rows fail accounting. The descriptive
noise flag uses sample standard deviation divided by mean above 10%; it is not
a regression decision or a functional failure.

`BENCH_EFFORT=scan` uses one warmup and three measured samples without changing
selection. A scan can identify suspects; it cannot establish a performance
regression. Command-wall observations and a budget must be measured separately
from operation timings. The initial full command baseline and its scope are
documented in [gpu-command-wall.md](gpu-command-wall.md). Dynamic-shape and
lifetime diagnostic controls remain separate suites until their routine lane
is validated; these 27 pairs alone do not complete issue #107.
