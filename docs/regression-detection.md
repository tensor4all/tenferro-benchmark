# Route coverage and regression detection

This page describes how the CPU suites make omitted public routes and missed
execution paths visible (tenferro-rs #1946 B1–B6, tenferro-benchmark #107).
Result tables live only under `result/<target_profile>/...`; this page holds
no measurements.

## Pieces

| Piece | What it answers | Timing? |
|---|---|---|
| `benchmarks/cpu/manifests/*.yaml` | Versioned quick/full expected case IDs per suite | — |
| `case_status.json` in every raw run | Which expected cases were selected, executed, unsupported, failed, missing, noisy | — |
| `benchmarks/cpu/public_api_coverage.yaml` `routes` + `scripts/route_coverage.py` | Does every manifest spelling exist in the tenferro-rs revision? Which public routes actually ran? | — |
| `cpu/session_matrix` (`scripts/run_cpu_session.sh`) | Loop of 1024 independent calls, one-operation batched routes, key streams | yes |
| `cpu/route_contract` (`scripts/run_route_diagnostic.sh`) | Provider calls, lanes, batch sizes, policy, allocations for the same case IDs; deterministic route contract | **no** (counters) |
| `scripts/detect_regressions.py` | scan suspicions, confirmed paired verdicts, deterministic failures | reads timing |
| `scripts/run_paired_timing.sh` | A/A noise and balanced baseline/candidate paired runs | yes |

## Coverage choice vs measurement effort

They are independent:

- `BENCH_COVERAGE=quick|full` picks case IDs from the versioned manifest.
  `quick` for `cpu/session_matrix` holds the #1946 F2 reproducer (batch=1024,
  4x4x4 f64 faer, allocating `dot_general_read` and `_into`), the separately
  labelled 1024-independent-call loop, the einsum spelling, below-cutoff /
  larger / skinny / canonical-copy controls, Hadamard, a three-operand chain
  and the four key streams. `full` adds complex and non-contiguous cases,
  batch/matrix sweeps around the Auto lane cost boundary, backend and scoped
  threshold overrides, all forced strategies and the lane cost-model knob.
- `BENCH_EFFORT=scan|standard|confirm|aa` picks repetitions. `scan` is a
  low-repetition screen (suite `defaults.effort.scan`), `standard` is the
  suite's ordinary count, `confirm` reads the declared confirmation config and
  `aa` takes explicit `BENCH_AA_WARMUPS` / `BENCH_AA_RUNS`.

A scan flags suspects; only a confirmed paired run makes a performance claim.
Measure total command/phase wall time of `quick` on the target host before
setting a routine time budget; no budget is set here.

## Case status

Only rows that ran and passed count as executed (covered). Unsupported
(typed `Unsupported` from the library, a provider absent from the build, or an
API absent at the revision), failed (numerical, crash, compile/launch failure),
timeouts (liveness) and missing cases are listed separately; `complete` is
false whenever anything expected was not selected, missing or failed. Run
metadata records the coverage, manifest version, filter, expected/selected
counts, effort, harness revision and dirty state, and the exact command.

## Route contract (deterministic)

`cpu_route_diagnostic` installs a public `CpuProviderBundle` spy that forwards
unchanged to faer and runs each batched case once after a warm call. For each
allocating/`_into` twin pair, `scripts/route_contract.py` requires that both
routes reject a policy with a typed error or both run, take the same lane
decision and cover the batch exactly once (call counts may differ). FAIL or a
missing twin (INCOMPLETE) makes the run exit 3. Counters without a public
observation point (session/executor entries, zero-fill/copy volume, output-pool
reuse, plan-cache hits) are reported as `unavailable`, never 0. Allocation
counts come from a counting global allocator in the diagnostic process.

The diagnostic and `benchmark_cpu_session` use only APIs common to the #1946
audit baseline (5a4e7fd84) and the repair. `build.rs` probes the tenferro-rs
source for the lane cost-model knobs; at the baseline those cases are
`unsupported`. Build each revision with `scripts/build_for_tenferro_rev.sh`
(own target directory per commit); it re-points `extern/tenferro-rs` during
the build and restores it, so do not run other builds of this checkout at the
same time.

## Detector

- `detect_regressions.py scan --baseline RUN --candidate RUN [--route-contract DIR]`:
  `SUSPECT_SLOWER` / `SUSPECT_FASTER` only when per-op sample ranges do not
  overlap. Never fails on timing.
- `detect_regressions.py aa --aa-dir DIR`: A/A statistic, spread and CoV per
  case, used to declare thresholds. No verdicts.
- `detect_regressions.py confirm --config CFG --paired-dir DIR --aa-dir DIR [--route-contract DIR]`:
  statistic `median_of_round_ratios` over every round, REGRESSION /
  IMPROVEMENT / NO_CHANGE, or INCONCLUSIVE for unbalanced order, too few
  rounds, arm CoV above the declared bound, missing A/A, or an A/A spread
  above the declared bound or reaching the relative threshold.
- Deterministic findings in every mode: MISSING, NUMERICAL_FAILURE,
  LIVENESS_FAILURE, NEWLY_UNSUPPORTED, ROUTE_CONTRACT_FAILURE, ROUTE_MISMATCH
  (route/provider/worker count/execution mode differ between arms),
  CHANGED_BASELINE (manifest or harness differs, commits differ from the
  declaration, dirty harness). Exit 2 for any of them, 1 for a confirmed
  regression, else 0.

`benchmarks/cpu/confirmation.yaml` is a placeholder: every threshold, the
statistic, repetitions, commits, host/provider/threads, manifest, timing and
cache state are `null` and must be declared (`status: declared`) before any
candidate result exists. Take the thresholds from the A/A characterization and
from the size of change that matters, never from this audit or from candidate
data. The paired runner refuses a dirty harness, so keep the filled config
outside tracked files (for example under `data/results/`, which is ignored) or
commit it first and declare that commit as `harness_commit`.

## #1946 B6 command sequence (amd-cpu host)

Run these sequentially on an otherwise idle host (the idle-host guard in the
timing runners stays enabled; if it rejects a run, wait and rerun). Linux CPU
timing collection runs inside the devcontainer; because `extern/tenferro-rs`
points at worktrees outside this repository, bind-mount them at the same path:

```bash
BASE=/home/shinaoka/tensor4all/tenferro-rs/.worktrees/bench-main            # 5a4e7fd84
REPAIR=/home/shinaoka/tensor4all/tenferro-rs/.worktrees/issue-1938-redesign # fix/1946-session-remediation
devcontainer up --workspace-folder . \
  --mount "type=bind,source=/home/shinaoka/tensor4all/tenferro-rs,target=/home/shinaoka/tensor4all/tenferro-rs"
```

(a) F2 route diagnostic (counters, not timing; may also run on the host):

```bash
TENFERRO_RS_DIR=$BASE   RUN_LABEL=baseline-5a4e7fd84 BENCHMARK_TARGET_PROFILE=amd-cpu \
  scripts/run_route_diagnostic.sh 1 4   # expected: exit 3, FAIL for bdot_f64_b1024_m4n4k4_direct_auto at 4 threads
TENFERRO_RS_DIR=$REPAIR RUN_LABEL=repair BENCHMARK_TARGET_PROFILE=amd-cpu \
  scripts/run_route_diagnostic.sh 1 4   # expected: exit 0, every pair PASS or UNSUPPORTED
```

(c) A/A noise characterization (before declaring thresholds; choose the
repetitions deliberately, the values below are only a starting point equal to
the suite's standard effort):

```bash
devcontainer exec --workspace-folder . bash -lc "
  export BENCHMARK_TARGET_PROFILE=amd-cpu TENFERRO_CPU_FEATURES=system-mkl
  export BENCH_COVERAGE=quick BENCH_AA_ROUNDS=4 BENCH_AA_WARMUPS=3 BENCH_AA_RUNS=15 BENCH_AA_THREADS='1 4'
  scripts/run_paired_timing.sh aa $BASE"
```

Then copy `benchmarks/cpu/confirmation.yaml` to a path outside tracked files,
fill every field (full SHAs of `$BASE`, `$REPAIR` and this harness, features
`system-mkl`, threads `[1, 4]`, coverage `quick`, manifest version from
`benchmarks/cpu/manifests/session_matrix.yaml`, repetitions with an even number
of rounds, statistic `median_of_round_ratios`, thresholds and noise bounds from
the A/A report) and set `status: declared` before step (b).

(b) Paired baseline/candidate timing in balanced ABBA order:

```bash
devcontainer exec --workspace-folder . bash -lc "
  export BENCHMARK_TARGET_PROFILE=amd-cpu TENFERRO_CPU_FEATURES=system-mkl
  export BENCH_CONFIRM_CONFIG=data/results/amd-cpu/cpu/confirmation-1946.yaml
  export BENCH_AA_DIR=data/results/amd-cpu/cpu/session_matrix_aa/<aa-timestamp>
  scripts/run_paired_timing.sh paired $BASE $REPAIR"
.venv/bin/python scripts/detect_regressions.py confirm \
  --config data/results/amd-cpu/cpu/confirmation-1946.yaml \
  --paired-dir data/results/amd-cpu/cpu/session_matrix_paired/<timestamp> \
  --aa-dir data/results/amd-cpu/cpu/session_matrix_aa/<aa-timestamp> \
  --route-contract data/results/amd-cpu/cpu/route_contract/<repair-timestamp>
```

Every round keeps its raw directory (`rNN_baseline`, `rNN_candidate`) with
`run_t*.yaml`, `samples_t*.jsonl` and `case_status*.json`; `plan.json` records
the executed order, `command.txt` the command, and the latest summary goes to
`result/amd-cpu/cpu/session_matrix_paired.md` (A/A:
`session_matrix_aa.md`, route contract: `route_contract.md`). Keep INCONCLUSIVE
cases visible; do not rerun selectively.

On macOS use the same scripts natively with `BENCHMARK_TARGET_PROFILE=mac-cpu`
and `TENFERRO_CPU_FEATURES=system-accelerate`; the routine
`./scripts/run_all.sh 1 4` still runs `cpu/session_matrix` and the route
diagnostics.

## GPU

No GPU route cases were added and no GPU measurement was taken; GPU rows
remain **unmeasured**, not validated. The WebGPU runner builds against the
repair. The CUDA runners were adapted to the session-side canonicalization,
but `--features cuda` could not be checked: the repair revision itself fails
to compile `tenferro-linalg/src/gpu/mod.rs` under `cuda`
(`TensorViewCanonicalization` not in scope for `CudaExecSession::to_contiguous`).
