# Routine GPU suite set

`scripts/run_gpu_suite.sh` collects exactly one curated set: the suites and
problem/backend pairs declared in `benchmarks/gpu/manifests/routine.yaml`. There
is no quick/full switch. `BENCH_EFFORT` still selects sampling effort
independently; an explicit `GPU_BENCH_SUITE`, `GPU_BENCH_BACKENDS` or
`GPU_BENCH_PROBLEM` is a targeted probe, never a routine refresh, and
`BENCH_ROUTINE_VENDOR_ARMS=1` adds the opt-in vendor arms.

```bash
devcontainer exec --workspace-folder . --config .devcontainer/cuda/devcontainer.json \
  bash -lc 'GPU_BENCH_DEVICE=0 BENCH_EFFORT=standard \
    BENCHMARK_TARGET_PROFILE=nvidia-gpu ./scripts/run_gpu_suite.sh'
```

## Cost of the set

`rows = problems × backends`; each row is one measured and numerically verified
cell, and each backend is a separate process with its own startup and
preparation. Widening the set is a cost decision, so it belongs in the table
below and in `routine.yaml`, not in a code change.

| Suite | Driver | Problems | Backends | Rows |
|---|---|---:|---:|---:|
| `gpu/dense` | `gpu_suite` | 7 | 4 | 28 |
| `gpu/einsum` | `gpu_suite` | 1 | 4 | 4 |
| `gpu/sparse` | `gpu_suite` | 2 | 1 | 2 |
| `gpu/tensornetwork` | `gpu_suite` | 1 | 4 | 4 |
| `gpu/elementwise` | `gpu_suite` | 2 | 4 | 8 |
| `gpu/linalg_ad_latency` | `gpu_linalg_ad` | 10 | 3 | 30 |
| `gpu/linalg_jvp_vjp` | `gpu_linalg_ad` | 10 | 3 | 30 |
| **Total** | | **33** | **23** | **106** |

The opt-in vendor arms add 27 rows (`gpu/dense` cublaslt/cutlass/cusolver,
`gpu/einsum` cublaslt/cutlass, `gpu/sparse` cusparse/ginkgo) for a 133-row
matrix. Vendor comparisons characterize against other libraries; they are not
part of the weekly bundle.

## What the set keeps and what it leaves out

Each operation family keeps one representative, and a size sweep is treated as
characterization rather than routine work. The suite files keep every case they
ever had: `problems` in the manifest is a subset, so a dropped case comes back by
naming it there again.

- `gpu/dense` keeps matmul, batched matmul, QR, one solve size, SVD, EIGH and the
  small batched matmul as the latency-regime representative. The solve size
  ladder (512/1024/2048/4096 and the rhs128 variant) and the 1024-matrix
  throughput batch stay in `benchmarks/gpu/dense.yaml`.
- `gpu/linalg_ad_latency` keeps n=4 only (10 operations × one size). These rows
  measure call overhead, which barely depends on size, so one size keeps
  regression detection; n=2 and n=8 are the characterization sweep.
- `gpu/linalg_jvp_vjp` keeps n=256 only (10 operations × one size); n=512 is the
  sweep of the same GPU-sized AD workloads.
- `gpu/elementwise` keeps both cases because they are different regimes, not a
  size sweep: 1k measures per-command overhead (latency), 1m the bandwidth cost.
  They are never averaged together.
- `gpu/sparse` is reference-only (tenferro has no CUDA sparse implementation), so
  only the PyTorch reference arm is routine.
- The diagnostic `tenferro-cuda-eager-per-call-session` backend and the
  `jax-cuda-eager` tensor-network path stay opt-in through an explicit
  `GPU_BENCH_BACKENDS` filter. Selection does not establish implementation
  availability.

Not in the routine set: `gpu/dense_throughput` and `gpu/linalg_ad_throughput`
(separate `tenferro-cuda-trace-deferred-batch` regime with their own driver and
reports; mixing them into one weekly matrix would mix regimes), and the
fixture-driven public-API families (`fft`, `isolated_ops`, `linalg_public_api`,
`lstsq`, `nn`, `reuse_contraction`, `structural`, `permutation`), whose problems
come from `data/instances/*.json` rather than a problem list the routine manifest
can name. They keep their own runners and reports. `gpu/perf_issues` is the
routine control lane (see [gpu-routine-controls.md](gpu-routine-controls.md))
plus the broad matrix, which stays independently selectable.

## Selection, accounting and reports

Every invocation writes the materialized suite YAMLs, `selection.json` and
`case_status.json` under
`data/results/nvidia-gpu/gpu/routine_selection/<timestamp>/`. `selection.json`
records the manifest version, effort, vendor-arm state, expected and selected
pair IDs, and the driver for each suite. Each suite's run metadata points at this
selection and records its own start, end and wall time. Verification-only
commands use `data/verification` instead and default to physical GPU1; standard
timing defaults to physical GPU0.

After collection, `case_status.json` distinguishes executed, skipped, unsupported,
failed, missing, verification-only, raw-samples-unavailable and noisy pairs.
Only successful numerical validation with retained positive raw batch samples
counts as executed. Skipped, unsupported, verification-only and unrecorded pairs
never count as measured coverage.

A case with no reference implementation or an unconfigured optional reference
uses `status: skipped` (legacy `not_configured` is accounted as skipped). A
library or GPU SKU that does not support the case uses `status: unsupported`.
Both require a nonblank `execution.unsupported_reason` or `verification.reason`;
the accounting preserves it in `reasons`. They are explicit capability decisions,
not a way to reinterpret an exception. Runtime/compile errors, timeouts, OOM and
numerical mismatches remain failed even if verification itself was skipped.
A numerical failure also overrides a contradictory skipped/unsupported status.

The collector writes `case_status.json` and prints separate failed, skipped,
unsupported, missing, duplicates and unexpected counts before returning.
Verification-only failures use this same accounting path. Run/result schema
validation failures are deferred until accounting is written and still force
a nonzero exit, even if the case status itself is benign. Its exit
code is nonzero for any failed, missing, duplicate or unexpected case, and zero
for skipped/unsupported-only collections. A reasonless skipped/unsupported row
fails. The existing weekly success gate can therefore skip later slots after a
benign-only successful collection, but a real runtime failure still permits
retries. The retry schedule is unchanged.

The descriptive noise flag uses sample standard deviation divided by mean above
10%; it is not a regression decision or a functional failure.

`BENCH_EFFORT=scan` uses one warmup and three measured samples without changing
selection. A scan can identify suspects; it cannot establish a performance
regression. Command-wall observations and a budget must be measured separately
from operation timings; see [gpu-command-wall.md](gpu-command-wall.md).
Dynamic-shape and lifetime diagnostic controls run sequentially after the core
matrix through [the routine control lane](gpu-routine-controls.md). These core
pairs alone do not complete issue #107.

## Historical observations

The earlier quick/full split was removed on 2026-11-02: every weekly run now
collects the single set above, so the routine selection has no mode switch in
code, manifests, run metadata, reports, tests or docs. Run metadata written
before that change still carries its `collection.coverage` value; the report-side
readers ignore both that field and its absence, and no routine code path writes
it. The independently selectable `gpu/perf_issues` matrix keeps its own
quick/full case set, so its child runs and the routine control report still
record that suite's own `coverage` — that is the perf-issue matrix, not the
routine selection.

The routine pages published today (`routine_coverage.md`, `routine_controls.md`,
`routine_workflow.md`, `routine_command_wall.md`) were measured under the retired
selection and keep their headers so the numbers stay attributable. The first
complete collection of the single set replaces them, and that run is also what
re-characterizes the `routine_budget.yaml` cost.

The last core-only standard quick collection (`20261013_232208`, tenferro-rs
`9655b4e0`, harness `020332f`) executed 27 pairs on physical RTX 3060 GPU0. The
last full collection on A100 (`20261017_104441`) executed 75 of 94 pairs. Those
numbers describe the retired split, not this set; the first complete collection
of the single set re-characterizes the cost and the
[routine budget](../benchmarks/gpu/routine_budget.yaml).
