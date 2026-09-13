# M5 CPU refresh from tenferro-rs origin/main

Collected on 2026-09-13 on native macOS 26.5.1, Apple M5 Max (18 cores),
with 1 and 4 threads, sequentially. tenferro-rs was fetched from `origin/main`
and pinned to `28dfc7e383a653e1461d38b73ced3d563b4efdad` for the entire collection.
This includes the shared BLAS session fix in
[tenferro-rs PR #1796](https://github.com/tensor4all/tenferro-rs/pull/1796).
The measured benchmark source is `4d4b94aca2e2bdc19b605180e4ce524dc67f1df0`.

## Reproduction

With that benchmark revision checked out and `extern/tenferro-rs` detached at
the revision above:

```bash
export PATH="$HOME/.juliaup/bin:$PATH"
export BENCHMARK_COMMIT="$(git rev-parse HEAD)"
export BENCHMARK_TARGET_PROFILE=mac-cpu
export TENFERRO_CPU_FEATURES=system-accelerate
export PUBLICATION_GATE_PROFILE=full
export FFT_BENCH_LENGTHS=1024,16384,1048576
export RUN_PERMUTATION_SUITE=1
source scripts/benchmark_host_idle.sh
assert_benchmark_host_idle
caffeinate -i ./scripts/run_all.sh 1 4
```

The exact collection script, selected run directories and validation summary are
stored with the [first raw run](../data/results/mac-cpu/cpu/einsum/20260913_182541/).
Each run records its core revision, features and thread environment.

## Reports and coverage

The [latest CPU reports](../result/mac-cpu/cpu/) contain both thread counts:
einsum, CPU operations, linalg JVP/VJP, FFT, public API, small work,
shared-session matrices and permutation.

For each thread count, validation confirmed:

- Einsum: 19 instances, two contraction strategies, five backends.
- CPU operations: 666 successful rows; underlying publication gate: 318 successful rows.
- FFT: 64 successful rows and 80 explicitly skipped short-operation routes.
- Public API: 429 successful rows and 18 explicitly unsupported rows.
- Small work: 25 cases with 15 measured batches each; correctness passed.
- Session matrices: 30 cases, 15 batches of 1,024 distinct prepared input pairs,
  covering matmul/solve, sizes 2/4/8/16/32 and Accelerate/faer/PyTorch;
  correctness passed.
- Permutation: 38 successful rows across Rust and Julia, plus eight HPTT skips
  because that optional backend was not enabled; correctness passed.

CSV row identities are unique. All run metadata identifies the same clean
core revision and benchmark revision. Skips and unsupported rows are preserved.
Reported IQRs and small-work noise flags are retained; no noisy rows were removed.
These are fresh measurements, not a paired before/after performance experiment.

## Timing and report changes

Input generation, backend/session construction and entry, tracing/compilation,
validation and output destruction remain outside operation timing. Short
operations use many operations in one interval and a shared entered session
where supported. The matrix reference includes Rayon + faer.

Before collection, the permutation benchmark was corrected to prepare its
TensorRead descriptors and enter its backend session outside the timed operation;
materialization uses the borrowed session. This source change was pushed before
all measurements. Its full-value permutation reference tests passed.

After collection, only report generation changed: the shared-session descriptions
now account for PR #1796, and the einsum/CPU-ops/linalg reports combine explicit
1-thread and 4-thread runs rather than retaining only the last thread count.
The combiner rejects mixed revisions and duplicate thread counts. No timing
values were edited. The earlier benchmark-source revision in metadata is intentional.

To regenerate these three combined reports from the recorded data:

```bash
python3 - <<'PYTHON'
import sys
from pathlib import Path
sys.path.insert(0, 'scripts')
from format_cpu_thread_reports import combine
combine([
    Path('data/results/mac-cpu/cpu/einsum/20260913_182541'),
    Path('data/results/mac-cpu/cpu/einsum/20260913_183437'),
])
PYTHON
```
