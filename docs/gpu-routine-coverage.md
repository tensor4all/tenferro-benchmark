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

`rows = problems × backends` less the arms a case declares it does not
participate in (`participants_gpu`, currently the permutation patterns); each row
is one measured and numerically verified cell, and each backend is a separate
process with its own startup and preparation. Widening the set is a cost
decision, so it belongs in the table below and in `routine.yaml`, not in a code
change.

| Suite | Driver | Problems | Backends | Rows |
|---|---|---:|---:|---:|
| `gpu/dense` | `gpu_suite` | 7 | 4 | 28 |
| `gpu/einsum` | `gpu_suite` | 1 | 4 | 4 |
| `gpu/sparse` | `gpu_suite` | 2 | 1 | 2 |
| `gpu/tensornetwork` | `gpu_suite` | 1 | 4 | 4 |
| `gpu/elementwise` | `gpu_suite` | 2 | 4 | 8 |
| `gpu/linalg_ad_latency` | `gpu_linalg_ad` | 10 | 3 | 30 |
| `gpu/linalg_jvp_vjp` | `gpu_linalg_ad` | 10 | 4 | 40 |
| `gpu/permutation` | `gpu_permutation` | 10 | 6 | 44 |
| `gpu/lstsq` | `gpu_public_api` | 1 | 4 | 4 |
| `gpu/linalg_public_api_throughput` | `gpu_public_api` | 16 | 4 | 64 |
| `gpu/fft` | `gpu_public_api` | 4 | 4 | 16 |
| `gpu/nn` | `gpu_public_api` | 11 | 4 | 44 |
| `gpu/structural` | `gpu_public_api` | 16 | 4 | 64 |
| `gpu/isolated_ops` | `gpu_public_api` | 33 | 4 | 132 |
| `gpu/reuse_contraction` | `gpu_public_api` | 56 | 4 | 224 |
| **Total** | | **180** | **56** | **708** |

The opt-in vendor arms add 164 rows (`gpu/dense` cublaslt/cutlass/cusolver,
`gpu/einsum` cublaslt/cutlass, `gpu/sparse` cusparse/ginkgo, and the second JAX
path `jax-cuda-jit` on the seven public-API families) for an 862-row matrix.
Vendor comparisons characterize against other libraries; they are not part of the
weekly bundle.

### Every `benchmarks/gpu/*.yaml` file is classified

No suite is silently absent from the weekly set. The four files that are not
suites (`confirmation.yaml`, `public_api_coverage.yaml`, `routine_budget.yaml`)
are declarations, and `perf_issues.yaml` is the routine control lane.

| Suite | Class | Evidence | Rows |
|---|---|---|---:|
| `gpu/fft` | (a) collected | own collector `run_gpu_public_api.sh`, fixtures in `data/instances/gpu_fft.json`, published page `result/nvidia-gpu/gpu/fft.md` | 16 |
| `gpu/isolated_ops` | (a) collected | same collector, 33 isolated public operations, published page | 132 |
| `gpu/linalg_public_api_throughput` | (a) collected | same collector; the surface where #299-class failures appear, published page. The sibling `gpu/linalg_public_api_latency` is the per-call overhead page and has no weekly representative | 64 |
| `gpu/lstsq` | (a) collected | same collector; carries the unresolved #299 least-squares failures, published page | 4 |
| `gpu/nn` | (a) collected | same collector, 11 activation/normalization operations, published page | 44 |
| `gpu/structural` | (a) collected | same collector, 16 structural operations, published page | 64 |
| `gpu/reuse_contraction` | (a) collected | same collector, 56 reuse/accumulating and strided-read operations, published page | 224 |
| `gpu/permutation` | (a) collected | own collectors `run_gpu_permutation.sh` and the Rust/Python runners; the materialize kernel under every contraction, and its 44 rows once went three weeks without an update | 44 |
| `gpu/dense_throughput` | (b) excluded by design | separate regime: backend identity `tenferro-cuda-trace-deferred-batch`, own driver `run_gpu_throughput.sh`, own reports. `AGENTS.md` forbids averaging or summarising a throughput row with a latency row, so it cannot join one weekly matrix | — |
| `gpu/linalg_ad_throughput` | (b) excluded by design | same deferred-batch regime and driver as `dense_throughput` | — |
| `gpu/perf_issues` | (c) diagnostic lane | 193-case broad matrix selected by its own `benchmarks/gpu/manifests/perf_issues.yaml`; the routine command already runs the curated 10 timed + 8 state controls from it weekly. The broad matrix stays independently selectable | — |

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
- `gpu/linalg_jvp_vjp` keeps n=256 only (10 operations × one size, per-call
  latency) on four arms: trace, eager, PyTorch and JAX. The eager arm runs the
  same logical case on the same caller-held CUDA engine, with its own AD
  preparation inside the sample (the eager API has no separate prepare step), and
  is a separate backend identity. n=4096 is the throughput end of the same AD
  workloads (per-call, and over the `linalg_ad` floor), run on an explicit probe
  because one 4096 row is ~0.1–3.5 s of GPU work before its check. The probe cases
  are declared under `probe_problems` in `routine.yaml` and selected by name; they
  are never part of the weekly plan:

  ```bash
  GPU_BENCH_PROBLEM=linalg_ad_large_grad_sum_qr_jvp_f64_4096 \
  GPU_BENCH_SUITE=benchmarks/gpu/linalg_jvp_vjp.yaml \
    ./scripts/run_gpu_suite.sh
  ```

  The selection plans the three arms of that one case and records them in
  `run.yaml` `expected_case_ids`, so packaging treats them as planned rows and
  the weekly completeness check is unaffected.
- `gpu/elementwise` keeps both cases because they are different regimes, not a
  size sweep: 1k measures per-command overhead (latency), 1m the bandwidth cost.
  They are never averaged together.
- `gpu/sparse` is reference-only (tenferro has no CUDA sparse implementation), so
  only the PyTorch reference arm is routine.
- The seven public-API families are fixture-driven: their cases come from
  `data/instances/*.json` through `problems: {source, include}`. The routine
  selection materializes a curated instance document, so the rule is the same —
  one case per operation, double or complex precision, the largest declared
  size; dtype and size sweeps are not weekly. `gpu/reuse_contraction` (56
  operations) and `gpu/isolated_ops` (33) are the largest families.
- `gpu/permutation` keeps all ten patterns: each is a different index mapping, not
  a size sweep, and its arms are fixed by the two permutation runners (cuTENSOR is
  one of them because comparing the materialize kernel against it is the point).

### Collect order and the one-night budget

The weekly run collects the manifest in order: the core five, both AD suites,
`permutation`, `lstsq`, `linalg_public_api_throughput`, `fft`, `nn`, `structural`,
`isolated_ops`, `reuse_contraction`. A night that runs out of budget therefore
loses the cheapest-signal-per-row tail last. The 708-row set is 6.6x the previous
routine matrix; it has not been measured on A100 yet. If the first complete
collection does not fit the pod window, the documented fallback is to drop
`gpu/isolated_ops` and `gpu/reuse_contraction` (356 rows of per-operation API
surface) from this manifest and collect them monthly — a manifest edit, not a code
path.
- The diagnostic `tenferro-cuda-eager-per-call-session` backend and the
  `jax-cuda-eager` tensor-network path stay opt-in through an explicit
  `GPU_BENCH_BACKENDS` filter. Selection does not establish implementation
  availability.

Everything else keeps its own runner and report and is collected through the
routine command (see the classification table above). `gpu/perf_issues` remains
the routine control lane (see [gpu-routine-controls.md](gpu-routine-controls.md))
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
a nonzero exit, even if the case status itself is benign. The page, its summary
sidecar and their ingestion receipt are published inside the collection loop and
are deferred the same way: a refused receipt sets the run red without skipping
the next suite, the accounting or the coverage report. The maintained routine
coverage page copied to `result/` after the loop is deferred the same way, so an
unwritable page still leaves the wall records on disk. Its exit
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
