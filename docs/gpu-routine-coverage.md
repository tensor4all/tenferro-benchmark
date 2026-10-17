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
matrix, large batch, larger/adjacent solves, sparse references and the
node-JIT JAX tensor-network path (94 pairs). The diagnostic `jax-cuda-eager`
tensor-network path is opt-in, not part of default full coverage; request it
with
`GPU_BENCH_SUITE=benchmarks/gpu/tensornetwork.yaml GPU_BENCH_BACKENDS=jax-cuda-eager
./scripts/run_gpu_suite.sh` (an explicit filter selects full coverage). Optional
elementwise suites
participate when explicitly requested. The diagnostic
`tenferro-cuda-eager-per-call-session` backend is also opt-in, not part of the
default quick/full backend lists; request it with an explicit
`GPU_BENCH_BACKENDS=tenferro-cuda-eager-per-call-session` filter to see
session-entry cost next to the one-session-per-batch eager arm and the
caller-held trace session. Unsupported implementations remain
unsupported; selection does not establish implementation availability.

An unfiltered `BENCH_COVERAGE=full` run also regenerates the quick-profile
report family from the same records restricted to the quick manifest
selection: `routine_coverage.md`, `routine_controls.md` and
`routine_workflow.md` are the quick-profile reports, while the full-profile
reports are mirrored to `routine_full_coverage.md`,
`routine_full_controls.md` and `routine_full_workflow.md`. The quick selection
itself is unchanged; a filterless quick run continues to write the quick
reports only. This lets one full collection refresh both families instead of
running the 27-pair quick profile only to rewrite its reports.

The synthetic sparse fixtures are cached per run and reused across the
`pytorch-cuda`, `cusparse` and `ginkgo` backend processes. `run_gpu_suite.sh`
points `GPU_BENCH_SPARSE_CACHE_DIR` at `<suite run dir>/cache`; direct runner
invocations fall back to `data/cache/gpu_sparse`. The cache is keyed by problem
id + seed + shape and its stored identity (shape/nnz/seed) is verified before
reuse. Generation and cache I/O happen outside any timed region.

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
lifetime diagnostic controls run sequentially after the core matrix through
[the routine control lane](gpu-routine-controls.md). These 27 core pairs alone
do not complete issue #107.

The first core-only standard quick collection (`20261013_232208`, tenferro-rs `9655b4e0`,
harness `020332f`) executed all 27 pairs on physical RTX 3060 GPU0. All passed
numerical verification and retained raw batch samples; none exceeded the
descriptive noise threshold. The script clock through the final suite report
copy was 148.916 seconds, including a 0.30-second cached Rust build. This scope
excludes devcontainer CLI startup and the final coverage-accounting/report write;
it is not yet a whole-command budget. See the generated
[coverage report](../result/nvidia-gpu/gpu/routine_coverage.md) and
[phase report](../result/nvidia-gpu/gpu/routine_command_wall.md).
The earlier full baseline used a different upstream revision and validation
implementation; these observations do not establish an implementation speedup.
