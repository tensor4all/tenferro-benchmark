#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# Rendering reads campaign-toolkit's power-condition formatter, so the fixture's
# run directory uses the repository environment that has it rather than the
# bare interpreter the stubs run under.
if ! "$ROOT/.venv/bin/python" -c 'import campaign_toolkit' 2>/dev/null; then
  echo "the repository environment does not import campaign_toolkit (run: uv sync)" >&2
  exit 1
fi
ln -s "$ROOT/.venv" "$TMP/.venv"

mkdir -p "$TMP/scripts" "$TMP/data/results" "$TMP/extern/tenferro-rs" "$TMP/benchmarks/cpu"
cp "$ROOT/scripts/run_all.sh" "$TMP/scripts/run_all.sh"
cp "$ROOT/scripts/format_cpu_thread_reports.py" "$TMP/scripts/format_cpu_thread_reports.py"
cp "$ROOT/scripts/cpu_blas_provider.sh" "$TMP/scripts/cpu_blas_provider.sh"
cp "$ROOT/scripts/collect_cpu_info.py" "$TMP/scripts/collect_cpu_info.py"
cp "$ROOT/scripts/thread_env.sh" "$TMP/scripts/thread_env.sh"
cp "$ROOT/scripts/render_run.py" "$TMP/scripts/render_run.py"
cp "$ROOT/scripts/report_summary.py" "$TMP/scripts/report_summary.py"
cp "$ROOT/scripts/regenerate_report_summaries.py" "$TMP/scripts/regenerate_report_summaries.py"
cp "$ROOT/scripts/publish_run.sh" "$TMP/scripts/publish_run.sh"
cp "$ROOT/scripts/finalize_run.py" "$TMP/scripts/finalize_run.py"
cp "$ROOT/scripts/case_status.py" "$TMP/scripts/case_status.py"
cp "$ROOT/scripts/benchmark_wall.py" "$TMP/scripts/benchmark_wall.py"
mkdir -p "$TMP/benchmarks"
cp "$ROOT/benchmarks/profiles.yaml" "$TMP/benchmarks/profiles.yaml"
cp "$ROOT/benchmarks/unverified-pages.txt" "$TMP/benchmarks/unverified-pages.txt"
cp "$ROOT/scripts/format_results.py" "$TMP/scripts/format_results.py"
cp "$ROOT/scripts/format_cpu_ops_results.py" "$TMP/scripts/format_cpu_ops_results.py"
cp "$ROOT/scripts/cpu_pinning.sh" "$TMP/scripts/cpu_pinning.sh"
cp "$ROOT/benchmarks/cpu_pinning_rows.tsv" "$TMP/benchmarks/cpu_pinning_rows.tsv"
cat > "$TMP/scripts/suite_instances.py" <<'PY'
#!/usr/bin/env python3
print("bin_matmul_256")
PY
cp "$ROOT/benchmarks/cpu/einsum.yaml" "$TMP/benchmarks/cpu/einsum.yaml"

(
  cd "$TMP/extern/tenferro-rs"
  git init -q
  git config user.email test@example.invalid
  git config user.name "Test User"
  printf 'fixture\n' > README.md
  git add README.md
  git commit -q -m 'fixture commit'
)
TENFERRO_COMMIT="$(git -C "$TMP/extern/tenferro-rs" rev-parse HEAD)"

cat > "$TMP/scripts/collect_run_metadata.py" <<'PY'
#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--suite-id", required=True)
parser.add_argument("--target-profile", required=True)
parser.add_argument("--suite-file", required=True)
parser.add_argument("--timestamp", required=True)
parser.add_argument("--tenferro-dir")
parser.add_argument("--tenferro-commit")
parser.add_argument("--features", action="append", default=[])
parser.add_argument("--blas")
parser.add_argument("--output", required=True)
args = parser.parse_args()
Path(args.output).parent.mkdir(parents=True, exist_ok=True)
Path(args.output).write_text(
    "\n".join(
        [
            "schema_version: 1",
            f"target_profile: {args.target_profile}",
            f"suite_id: {args.suite_id}",
            f"suite_file: {args.suite_file}",
            f"timestamp: \"{args.timestamp}\"",
            "tenferro_rs:",
            f"  path: {args.tenferro_dir}",
            f"  commit: {args.tenferro_commit}",
            "  dirty: false",
            "  features:",
            *[f"    - {feature}" for feature in args.features],
            "environment:",
            "  hostname: test-host",
            "  os: test-os",
            "  arch: test-arch",
            "  env:",
            f"    RAYON_NUM_THREADS: '{os.environ['RAYON_NUM_THREADS']}'",
            f"    BENCHMARK_COMMIT: {os.environ['BENCHMARK_COMMIT']}",
            "blas:",
            f"  implementation: {args.blas}",
            "  version: unknown",
            *([] if os.environ.get("TEST_EMPTY_BLAS_PATHS") == "1" else [
                "  root: /tmp/openblas",
                "  library: /tmp/openblas/lib/libopenblas.dylib",
            ]),
            "",
        ]
    )
)
PY

cat > "$TMP/scripts/setup_extern_deps.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
printf 'called\n' > "$root/setup_extern_called"
export OPENBLAS_ROOT="$root/openblas"
export TENFERRO_RS_DIR="$root/extern/tenferro-rs"
SH

cat > "$TMP/scripts/run_all_rust.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
# Keep second-resolution raw-run directories distinct in this fast fixture.
sleep 1
threads="${1:-1}"
root="$(cd "$(dirname "$0")/.." && pwd)"
: "${BENCHMARK_TIMESTAMP:?}"
results_dir="${BENCHMARK_RESULTS_DIR:-$root/data/results}"
mkdir -p "$results_dir"
printf 'tenferro-trace\nBackend: tenferro-trace\nOMP_NUM_THREADS=%s\nRAYON_NUM_THREADS=%s\nStrategy: opt_flops\nInstance Tensors log10FLOPS log2SIZE Median (ms) IQR (ms) Compile (ms)\nbin_matmul_256 2 0 0 1.0 0.1 0.0\n' "$threads" "$threads" > "$results_dir/tenferro_trace_t${threads}_${BENCHMARK_TIMESTAMP}.log"
printf 'tenferro-eager\nBackend: tenferro-eager\nOMP_NUM_THREADS=%s\nRAYON_NUM_THREADS=%s\nStrategy: opt_flops\nInstance Tensors log10FLOPS log2SIZE Median (ms) IQR (ms) Compile (ms)\nbin_matmul_256 2 0 0 1.2 0.1 0.0\n' "$threads" "$threads" > "$results_dir/tenferro_eager_t${threads}_${BENCHMARK_TIMESTAMP}.log"
SH

cat > "$TMP/scripts/run_all_python.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
threads="${1:-1}"
root="$(cd "$(dirname "$0")/.." && pwd)"
: "${BENCHMARK_TIMESTAMP:?}"
results_dir="${BENCHMARK_RESULTS_DIR:-$root/data/results}"
mkdir -p "$results_dir"
printf 'pytorch-cpu einsum benchmark suite\nOMP_NUM_THREADS=%s\nStrategy: opt_flops\nInstance Tensors log10FLOPS log2SIZE Median (ms) IQR (ms)\nbin_matmul_256 2 0 0 0.7 0.1\n' "$threads" > "$results_dir/pytorch_cpu_t${threads}_${BENCHMARK_TIMESTAMP}.log"
printf 'jax-cpu einsum benchmark suite\nOMP_NUM_THREADS=%s\nStrategy: opt_flops\nInstance Tensors log10FLOPS log2SIZE Median (ms) IQR (ms)\nbin_matmul_256 2 0 0 0.5 0.1\n' "$threads" > "$results_dir/jax_cpu_t${threads}_${BENCHMARK_TIMESTAMP}.log"
SH

cat > "$TMP/scripts/run_cpu_ops.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
threads="${1:-1}"
root="$(cd "$(dirname "$0")/.." && pwd)"
: "${BENCHMARK_TIMESTAMP:?}"
results_dir="${BENCHMARK_RESULTS_DIR:-$root/data/results}"
mkdir -p "$results_dir"
printf 'suite,benchmark,dtype,threads,shape,backend,median_ms,iqr_ms,status\nmatmul,f64_square,f64,%s,2x2,tenferro-eager,1.000,0.100,ok\nmatmul,f64_square,f64,%s,2x2,tenferro-trace,1.100,0.100,ok\nmatmul,f64_square,f64,%s,2x2,pytorch-cpu,0.800,0.100,ok\nmatmul,f64_square,f64,%s,2x2,jax-cpu,0.700,0.100,ok\n' "$threads" "$threads" "$threads" "$threads" > "$results_dir/cpu_ops_t${threads}_${BENCHMARK_TIMESTAMP}.csv"
SH

cat > "$TMP/scripts/run_cpu_public_api.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
printf '%s\n' "${PUBLICATION_GATE_PROFILE:-missing}" >> "$(dirname "$0")/../public_profile_calls.log"
SH
for runner in run_small_work run_cpu_session run_overhead_smoke run_perf_issues; do
  cat > "$TMP/scripts/$runner.sh" <<'SH'
#!/usr/bin/env bash
set -euo pipefail
printf '%s %s\n' "$(basename "$0")" "$*" >> "$(dirname "$0")/../small_suite_calls.log"
SH
done
chmod +x "$TMP/scripts/"*.sh "$TMP/scripts/format_results.py" "$TMP/scripts/format_cpu_ops_results.py"

assert_any_file() {
  local pattern="$1"
  if ! find "$TMP/data/results" -path "$pattern" -type f -size +0c -print -quit | grep -q .; then
    echo "missing expected file matching $pattern" >&2
    find "$TMP/data/results" -type f -print >&2 || true
    exit 1
  fi
}

(
  cd "$TMP"
  BENCHMARK_COMMIT=fixture-benchmark BENCHMARK_HOST_OS=Linux RUN_PUBLIC_API_SUITE=1 PATH="/usr/bin:/bin" ./scripts/run_all.sh 1 >"$TMP/run_all_docs_test.out"
  BENCHMARK_COMMIT=fixture-benchmark BENCHMARK_HOST_OS=Linux RUN_PUBLIC_API_SUITE=1 PATH="/usr/bin:/bin" ./scripts/run_all.sh 4 >>"$TMP/run_all_docs_test.out"
)

assert_any_file "$TMP/data/results/amd-cpu/cpu/einsum/*/run.yaml"
assert_any_file "$TMP/data/results/amd-cpu/cpu/einsum/*/report.md"
assert_any_file "$TMP/data/results/amd-cpu/cpu/einsum/*/einsum.summary.json"
assert_any_file "$TMP/data/results/amd-cpu/cpu/einsum/*/cpu_ops_t4_*.csv"
assert_any_file "$TMP/data/results/amd-cpu/cpu/einsum/*/cpu_ops_report.md"
assert_any_file "$TMP/data/results/amd-cpu/cpu/einsum/*/cpu_ops.summary.json"
assert_any_file "$TMP/data/results/amd-cpu/cpu/einsum/*/wall_command.json"
test -s "$TMP/result/amd-cpu/cpu/einsum.md"
test -s "$TMP/result/amd-cpu/cpu/cpu_ops.md"
test ! -e "$TMP/result/einsum-results.md"
test ! -e "$TMP/result/cpu-benchmark-results.md"
test ! -e "$TMP/docs/results-einsum.md"
test ! -e "$TMP/docs/cpu-benchmark-results.md"
test -s "$TMP/setup_extern_called"

grep -q "Suite: \`cpu/einsum\`" "$TMP/result/amd-cpu/cpu/einsum.md"
grep -q "Target profile: \`amd-cpu\`" "$TMP/result/amd-cpu/cpu/einsum.md"
grep -q "Strategy: opt_flops" "$TMP/result/amd-cpu/cpu/einsum.md"
grep -q "PyTorch Python" "$TMP/result/amd-cpu/cpu/einsum.md"
grep -q "JAX Python (XLA CPU dot)" "$TMP/result/amd-cpu/cpu/einsum.md"
! grep -q "Torch C++" "$TMP/result/amd-cpu/cpu/einsum.md"
grep -Fq "tenferro-rs commit: \`$TENFERRO_COMMIT\`" "$TMP/result/amd-cpu/cpu/einsum.md"
grep -q "Latest run: \`./scripts/run_all.sh 4\`" "$TMP/result/amd-cpu/cpu/einsum.md"
grep -q "## Threads: 4" "$TMP/result/amd-cpu/cpu/einsum.md"
grep -q "data/results/amd-cpu/cpu/einsum/" "$TMP/result/amd-cpu/cpu/einsum.md"
grep -q "CPU Information" "$TMP/result/amd-cpu/cpu/einsum.md"
grep -q "Model:" "$TMP/result/amd-cpu/cpu/einsum.md"
grep -q "Thread Environment" "$TMP/result/amd-cpu/cpu/einsum.md"
grep -q "OPENBLAS_NUM_THREADS" "$TMP/result/amd-cpu/cpu/einsum.md"
grep -q "XLA_FLAGS" "$TMP/result/amd-cpu/cpu/einsum.md"
grep -q "Tenferro CPU BLAS Backend" "$TMP/result/amd-cpu/cpu/einsum.md"
grep -q "tenferro-rs features: \`blas-openblas\`" "$TMP/result/amd-cpu/cpu/einsum.md"
grep -q "BLAS implementation: \`openblas\`" "$TMP/result/amd-cpu/cpu/einsum.md"

grep -q "CPU Benchmark Items" "$TMP/result/amd-cpu/cpu/cpu_ops.md"
grep -q "Target profile: \`amd-cpu\`" "$TMP/result/amd-cpu/cpu/cpu_ops.md"
grep -q "tenferro-rs eager mode" "$TMP/result/amd-cpu/cpu/cpu_ops.md"
grep -q "tenferro-rs trace mode" "$TMP/result/amd-cpu/cpu/cpu_ops.md"
grep -q "PyTorch Python" "$TMP/result/amd-cpu/cpu/cpu_ops.md"
grep -q "JAX Python (XLA CPU)" "$TMP/result/amd-cpu/cpu/cpu_ops.md"
! grep -q "Torch C++" "$TMP/result/amd-cpu/cpu/cpu_ops.md"
grep -Fq "tenferro-rs commit: \`$TENFERRO_COMMIT\`" "$TMP/result/amd-cpu/cpu/cpu_ops.md"
grep -q "Latest run: \`./scripts/run_all.sh 4\`" "$TMP/result/amd-cpu/cpu/cpu_ops.md"
grep -q "## Threads: 4" "$TMP/result/amd-cpu/cpu/cpu_ops.md"
grep -q "data/results/amd-cpu/cpu/einsum/" "$TMP/result/amd-cpu/cpu/cpu_ops.md"
grep -q "CPU Information" "$TMP/result/amd-cpu/cpu/cpu_ops.md"
grep -q "Logical CPUs:" "$TMP/result/amd-cpu/cpu/cpu_ops.md"
grep -q "Thread Environment" "$TMP/result/amd-cpu/cpu/cpu_ops.md"
grep -q "OPENBLAS_NUM_THREADS" "$TMP/result/amd-cpu/cpu/cpu_ops.md"
grep -q "XLA_FLAGS" "$TMP/result/amd-cpu/cpu/cpu_ops.md"
grep -q "Tenferro CPU BLAS Backend" "$TMP/result/amd-cpu/cpu/cpu_ops.md"
grep -q "tenferro-rs features: \`blas-openblas\`" "$TMP/result/amd-cpu/cpu/cpu_ops.md"
grep -q "BLAS implementation: \`openblas\`" "$TMP/result/amd-cpu/cpu/cpu_ops.md"

# Exercise multi-thread aggregation with synthetic runners, never real timing.
(
  cd "$TMP"
  BENCHMARK_COMMIT=fixture-benchmark BENCHMARK_HOST_OS=Linux \
    RUN_FFT_SUITE=0 RUN_SMALL_WORK_SUITE=0 RUN_PERMUTATION_SUITE=0 RUN_PERF_ISSUES_SUITE=0 \
    PATH="$ROOT/.venv/bin:/usr/bin:/bin" ./scripts/run_all.sh 1 4 >>"$TMP/run_all_docs_test.out"
)
for report in einsum cpu_ops; do
  grep -q '^## Threads: 1$' "$TMP/result/amd-cpu/cpu/$report.md"
  grep -q '^## Threads: 4$' "$TMP/result/amd-cpu/cpu/$report.md"
done

# Accelerate does not record optional BLAS root/library paths. Missing optional
# metadata must not make a report-writing helper return failure under set -e.
(
  cd "$TMP"
  TEST_EMPTY_BLAS_PATHS=1 BENCHMARK_COMMIT=fixture-benchmark \
    BENCHMARK_HOST_OS=Darwin BENCHMARK_TARGET_PROFILE=mac-cpu \
    TENFERRO_CPU_FEATURES=blas-accelerate RUN_PUBLIC_API_SUITE=1 RUN_PUBLIC_API_PROFILE=full PATH="/usr/bin:/bin" \
    ./scripts/run_all.sh 1 >>"$TMP/run_all_docs_test.out"
)
test "$(cat "$TMP/public_profile_calls.log")" = "$(printf 'quick\nquick\nquick\nfull')"
# overhead_smoke is single-threaded: no thread-count argument, once per invocation
# (including the RUN_SMALL_WORK_SUITE=0 multi-thread run).
test "$(cat "$TMP/small_suite_calls.log")" = "$(printf 'run_small_work.sh 1\nrun_cpu_session.sh 1\nrun_perf_issues.sh 1\nrun_overhead_smoke.sh \nrun_small_work.sh 4\nrun_cpu_session.sh 4\nrun_perf_issues.sh 4\nrun_overhead_smoke.sh \nrun_overhead_smoke.sh \nrun_small_work.sh 1\nrun_cpu_session.sh 1\nrun_perf_issues.sh 1\nrun_overhead_smoke.sh ')"
for report in einsum cpu_ops; do
  test -s "$TMP/result/mac-cpu/cpu/$report.md"
  grep -q 'BLAS implementation: `accelerate`' "$TMP/result/mac-cpu/cpu/$report.md"
  ! grep -q 'BLAS library:' "$TMP/result/mac-cpu/cpu/$report.md"
  grep -q 'Thread Environment' "$TMP/result/mac-cpu/cpu/$report.md"
done

# BENCHMARK_RAW_ONLY=1 ends the run at the finalized record: no report, summary
# or result/ page is written or replaced. The CPU-ops collector renders nothing
# either; the offline renderer builds the cpu_ops/linalg Markdown from its CSV.
find "$TMP/result" -type f -exec md5sum {} + | sort > "$TMP/result_before_raw_only.txt"
RUNS_BEFORE=()
while IFS= read -r run_dir; do
  RUNS_BEFORE+=("$run_dir")
done < <(find "$TMP/data/results/amd-cpu/cpu/einsum" -mindepth 1 -maxdepth 1 -type d | sort)
(
  cd "$TMP"
  BENCHMARK_RAW_ONLY=1 BENCHMARK_COMMIT=fixture-benchmark BENCHMARK_HOST_OS=Linux \
    RUN_PUBLIC_API_SUITE=0 RUN_SMALL_WORK_SUITE=0 RUN_PERF_ISSUES_SUITE=0 \
    RUN_OVERHEAD_SMOKE_SUITE=0 PATH="/usr/bin:/bin" ./scripts/run_all.sh 1 >>"$TMP/run_all_docs_test.out"
)
find "$TMP/result" -type f -exec md5sum {} + | sort > "$TMP/result_after_raw_only.txt"
diff -u "$TMP/result_before_raw_only.txt" "$TMP/result_after_raw_only.txt"
RUNS_AFTER=()
while IFS= read -r run_dir; do
  RUNS_AFTER+=("$run_dir")
done < <(find "$TMP/data/results/amd-cpu/cpu/einsum" -mindepth 1 -maxdepth 1 -type d | sort)
RAW_ONLY_RUN="$(comm -13 <(printf '%s\n' "${RUNS_BEFORE[@]}") <(printf '%s\n' "${RUNS_AFTER[@]}"))"
test -n "$RAW_ONLY_RUN"
test -f "$RAW_ONLY_RUN/run.yaml"
test -f "$RAW_ONLY_RUN/wall_command.json"
test -f "$RAW_ONLY_RUN/cpu_info.md"
test ! -e "$RAW_ONLY_RUN/report.md"
test ! -e "$RAW_ONLY_RUN/einsum.summary.json"
# A rendered cpu_ops/linalg Markdown in the run directory is a rendering, not a
# record, and raw-only mode must not produce one.
test ! -e "$RAW_ONLY_RUN/cpu_ops_report.md"
test ! -e "$RAW_ONLY_RUN/linalg_jvp_vjp_report.md"
if find "$RAW_ONLY_RUN" -maxdepth 1 -type f -name '*_t[0-9]*.md' -print -quit | grep -q .; then
  echo "raw-only run rendered Markdown: $(find "$RAW_ONLY_RUN" -maxdepth 1 -type f -name '*.md')" >&2
  exit 1
fi
# The real collector (the stub above replaces it in this fixture) must not
# render either: publication renders from the CSV offline.
if grep -qE '\$SCRIPT_DIR/format_(cpu_ops|linalg_ad)_results\.py' "$ROOT/scripts/run_cpu_ops.sh"; then
  echo "scripts/run_cpu_ops.sh renders a report during collection" >&2
  exit 1
fi
