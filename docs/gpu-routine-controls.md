# Routine GPU latency and state controls

The unfiltered `run_gpu_suite.sh` command runs these controls sequentially after
the core matrix. Explicit suite/backend/problem filters leave them unselected
unless `GPU_BENCH_ROUTINE_CONTROLS=1` is set. `GPU_BENCH_ROUTINE_CONTROLS=0`
selects a core-only run. Selection of core pairs does not establish coverage of
these controls.

The versioned `benchmarks/gpu/manifests/routine_controls.yaml` holds the whole
routine control lane, with no quick/full switch. It selects existing public-API
performance-issue workloads: ten timed controls covering fixed-key warm, bounded
mixed-key and fresh-key whole sequences at n=16/64, plus zero allocation at
16/32768 elements through one shared callback versus per-call callback entry. The
latter is explicitly a session-entry diagnostic; it is not steady-state
shared-session throughput. Plans/workspace are reset outside each sequence
timer, while the sequence includes required planning. See
[gpu-shape-stream.md](gpu-shape-stream.md) for exact order/cache scope.

Eight untimed state controls cover the same six shape streams and repeated
backend construction at 8/64 cycles. They run on the diagnostic card after
timing finishes. State/counter observations have no timed samples and never
count as measured operation latency.

The wider `gpu/perf_issues` variants (capacity thrashing,
equal-shape/different-stride streams, every zero-allocation size and the
host-zero-upload reference arms) stay in `data/instances/gpu_perf_issues.json`,
so adding one is a manifest edit and a cost decision. The broader 166-case
`gpu/perf_issues` matrix remains independently selectable with its existing
runner; it is not forced into the routine command.

On the two-card RTX 3060 host, physical GPU0 collects timing and GPU1 performs
debug/verification/counter work. Override `GPU_BENCH_DIAGNOSTIC_DEVICE` to choose
the latter explicitly; on a single-card host it falls back to the timing card,
with both lanes still sequential. Each child resolves its physical card to a
UUID and exposes it as local CUDA ordinal zero.

Run only the controls, first as an untimed numerical check:

```bash
devcontainer exec --workspace-folder . --config .devcontainer/cuda/devcontainer.json \
  bash -lc 'GPU_BENCH_DEVICE=0 GPU_BENCH_DIAGNOSTIC_DEVICE=1 GPU_BENCH_VERIFY_ONLY=1 \
    BENCH_EFFORT=scan ./scripts/run_gpu_routine_controls.sh'
```

Then collect sequential timing/state observations:

```bash
devcontainer exec --workspace-folder . --config .devcontainer/cuda/devcontainer.json \
  bash -lc 'GPU_BENCH_DEVICE=0 GPU_BENCH_DIAGNOSTIC_DEVICE=1 \
    BENCH_EFFORT=standard ./scripts/run_gpu_routine_controls.sh'
```

Selection and combined accounting live under
`data/results/nvidia-gpu/gpu/routine_controls/<timestamp>/` (or
`data/verification` for verification-only). The two child raw runs live under
`data/results/nvidia-gpu/gpu/perf_issues/<timestamp>_{timing,state}/`; their
metadata explicitly records correctness-only mode where applicable. Child
reports do not overwrite the broad performance-issue publication. The combined
report is maintained at `result/nvidia-gpu/gpu/routine_controls.md`; verify-only
and `GPU_BENCH_SKIP_LATEST=1` preserve that publication.

Accounting distinguishes measured, observed, verification-only, failed,
missing, duplicate, unexpected and noisy case IDs. Both child
metadata files must record the same tenferro/harness revisions and features;
a mismatch prevents complete coverage. Noise is a descriptive 10% CoV screen,
not a performance failure. A complete standard control run requires all timed
controls to retain valid raw samples and all state controls to pass without
timed samples. This is not a paired baseline/candidate regression conclusion.

The enclosing core command additionally writes
`routine_selection/<timestamp>/wall_command_complete.json`, covering the core
and control phases through validation, accounting and report writes. Its scope
excludes devcontainer CLI startup and the final wall-record write. Core-only
phase reports do not include the control lane and must not be presented as the
total cost of the normal command.

The reference standard run `20261013_234113` completed the ten timed controls and
eight state observations on the two RTX 3060 cards.
The generated [workflow report](../result/nvidia-gpu/gpu/routine_workflow.md)
records both the script checkpoint and an independently measured enclosing
devcontainer command. [routine_budget.yaml](../benchmarks/gpu/routine_budget.yaml)
declares a 300-second descriptive script budget after that characterization,
with the existing CUDA image/dependency cache and mandatory binary rebuilds.
The command reports budget status for matching hardware and effort only when
both core and controls are complete. Over-budget observations never
change functional exit status; incomplete, verification-only, filtered and
other-hardware runs do not establish a passing complete routine workflow.
