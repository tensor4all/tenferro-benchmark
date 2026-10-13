# GPU command-wall observations

The [command-wall report](../result/nvidia-gpu/gpu/command_wall.md) records the
cost of the GPU workflow requested in #107. These are end-to-end command costs,
including setup and validation; they do not establish operation-performance
ratios. Existing operation clocks remain scoped to dispatch and completion.

The initial baseline uses the repository CUDA devcontainer, physical RTX 3060
GPU0, tenferro-rs `39d71d40`, one CPU thread and the existing suite repetitions.
It executes the default dense, einsum, sparse and tensor-network selections
sequentially. The report also includes a filtered solve probe used to validate
instrumentation. Selected unsupported cells remain visible; a completed command
does not imply complete measured coverage.

```bash
devcontainer exec --workspace-folder . --config .devcontainer/cuda/devcontainer.json \
  bash -lc 'GPU_BENCH_DEVICE=0 GPU_BENCH_SKIP_LATEST=1 \
    BENCHMARK_TARGET_PROFILE=nvidia-gpu ./scripts/run_gpu_suite.sh'
```

`GPU_BENCH_SKIP_LATEST=1` preserves existing operation reports while keeping all
raw results and per-run reports. Each suite run records:

- `wall_backends.jsonl`: build time and process wall time per backend; the
  latter includes interpreter startup, preparation, validation and samples.
- `wall_command.json`: cumulative script time through that suite, collection
  including the configured allocator-release delays, and report generation.
- Result-record notes: available input/preparation, compilation, initialization,
  validation, warmup, calibration and sampling phase walls. Phase clocks sit
  outside operation clocks. Some vendor paths lack detailed phase hooks; those
  fields are unavailable rather than zero.

The script clock excludes devcontainer CLI startup and the final wall-record
write. Suite command totals are cumulative within a multi-suite invocation and
must not be added together. Sampling phase walls include the untimed preparation
and cleanup surrounding individual operation intervals.

Regenerate the baseline report inside the CUDA devcontainer:

```bash
uv run python scripts/format_gpu_command_wall.py \
  data/results/nvidia-gpu/gpu/dense/20261013_224354 \
  data/results/nvidia-gpu/gpu/dense/20261013_224635 \
  data/results/nvidia-gpu/gpu/einsum/20261013_224635 \
  data/results/nvidia-gpu/gpu/sparse/20261013_224635 \
  data/results/nvidia-gpu/gpu/tensornetwork/20261013_224635 \
  --output result/nvidia-gpu/gpu/command_wall.md
```

`--budget-seconds` adds a descriptive budget observation; it does not change
functional success. No budget is declared for this initial full baseline.
Routine coverage and effort changes must retain explicit case identities and
be measured before a budget is chosen. GPU paired confirmation remains separate
from these workflow-cost observations.
