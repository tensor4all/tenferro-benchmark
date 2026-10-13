#!/usr/bin/env bash
# Ordinary cpu/small_work suite. Coverage BENCH_COVERAGE=quick|full (versioned
# manifest benchmarks/cpu/manifests/small_work.yaml; BENCH_INCLUDE_SETUP_DIAGNOSTICS=1
# is the older spelling of full), effort BENCH_EFFORT=scan|standard|confirm, and
# an optional BENCH_INSTANCE=id[,id...] filter inside the coverage choice.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"
THREAD_COUNTS=("${@:-1}")
source "$SCRIPT_DIR/cpu_blas_provider.sh"
source "$SCRIPT_DIR/thread_env.sh"
source "$SCRIPT_DIR/benchmark_host_idle.sh"
export TENFERRO_CPU_FEATURES="$(normalize_cpu_blas_features "${TENFERRO_CPU_FEATURES:-}")"
export BENCHMARK_TARGET_PROFILE="${BENCHMARK_TARGET_PROFILE:-amd-cpu}"
PYTHON="$PROJECT_DIR/.venv/bin/python"
[[ -x "$PYTHON" ]] || PYTHON=python3
START_NS="$($PYTHON -c 'import time; print(time.monotonic_ns())')"
export BENCH_COMMAND_START_NS="$START_NS"
# Reference warm-build 1T+4T commands: quick/standard 42 s, full/standard 181 s.
# Budgets diagnose slow commands (including builds); they never fail a run.
if [[ "${BENCH_COVERAGE:-quick}" == full || ( -z "${BENCH_COVERAGE:-}" && "${BENCH_INCLUDE_SETUP_DIAGNOSTICS:-0}" == 1 ) ]]; then
    export BENCH_COMMAND_BUDGET_SECONDS="${BENCH_COMMAND_BUDGET_SECONDS:-600}"
else
    export BENCH_COMMAND_BUDGET_SECONDS="${BENCH_COMMAND_BUDGET_SECONDS:-120}"
fi
if [[ "${SKIP_EXTERN_SETUP:-0}" != 1 ]]; then
    source "$SCRIPT_DIR/setup_extern_deps.sh"
fi
ensure_blas_env_for_features "$TENFERRO_CPU_FEATURES"
"$PYTHON" "$SCRIPT_DIR/validate_benchmark_suite.py" benchmarks/cpu/small_work.yaml
# Build once before sequential timing collection.
cargo build --release --no-default-features --features "$TENFERRO_CPU_FEATURES" --bin small_work_case
BUILD_END_NS="$($PYTHON -c 'import time; print(time.monotonic_ns())')"
BINARY="${CARGO_TARGET_DIR:-$PROJECT_DIR/target}/release/small_work_case"
TIMESTAMP="${BENCHMARK_TIMESTAMP:-$(date -u +%Y%m%d_%H%M%S)}"
"$PYTHON" -c 'from scripts.benchmark_layout import safe_target_profile; import sys; safe_target_profile(sys.argv[1])' "$BENCHMARK_TARGET_PROFILE"
RUN_DIR="$PROJECT_DIR/data/results/$BENCHMARK_TARGET_PROFILE/cpu/small_work/$TIMESTAMP"
mkdir -p "$RUN_DIR"
"$PYTHON" "$SCRIPT_DIR/collect_cpu_info.py" --markdown > "$RUN_DIR/cpu_info.md"
if [[ -z "${BENCHMARK_COMMIT:-}" ]]; then
    BENCHMARK_COMMIT="$(git rev-parse HEAD)"
fi
export BENCHMARK_COMMIT
export BENCH_COMMAND="${BENCH_COMMAND:-$(for v in BENCHMARK_TARGET_PROFILE TENFERRO_CPU_FEATURES BENCH_COVERAGE BENCH_EFFORT BENCH_INSTANCE; do if [[ -n "${!v:-}" ]]; then printf '%s=%q ' "$v" "${!v}"; fi; done)$(printf '%q ' "$0" "$@")}"
failed=0
for threads in "${THREAD_COUNTS[@]}"; do
    configure_cpu_thread_env "$threads"
    "$PYTHON" "$SCRIPT_DIR/collect_run_metadata.py" \
        --suite-id cpu/small_work --target-profile "$BENCHMARK_TARGET_PROFILE" \
        --suite-file benchmarks/cpu/small_work.yaml --timestamp "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        --tenferro-dir "$PROJECT_DIR/extern/tenferro-rs" --features "$TENFERRO_CPU_FEATURES" \
        --blas "$(blas_impl_for_features "$TENFERRO_CPU_FEATURES")" --output "$RUN_DIR/run_t${threads}.yaml"
    "$PYTHON" - "$RUN_DIR/run_t${threads}.yaml" <<'PY'
import os, sys, subprocess, yaml
from pathlib import Path
path = Path(sys.argv[1])
metadata = yaml.safe_load(path.read_text())
metadata['environment'].setdefault('env', {}).update({
    'BENCHMARK_COMMIT': os.environ['BENCHMARK_COMMIT'], 'BENCHMARK_BUILD_PROFILE': 'release',
    'RUSTC_VERSION': subprocess.check_output(['rustc', '--version'], text=True).strip(),
    'BENCH_INSTANCE': os.environ.get('BENCH_INSTANCE') or None,
    'BENCH_COMMAND_BUDGET_SECONDS': os.environ['BENCH_COMMAND_BUDGET_SECONDS']})
# What this run actually selects, from the versioned manifest (never "all N").
sys.path.insert(0, 'scripts')
import benchmark_small_work as suite
_, _, manifest, coverage, expected, selected, raw = suite.plan()
metadata['collection'] = {
    'command': os.environ.get('BENCH_COMMAND', ''), 'coverage': coverage,
    'manifest_version': manifest['manifest_version'], 'selection_filter': raw,
    'expected_cases': len(expected), 'selected_cases': len(selected),
    'effort': os.environ.get('BENCH_EFFORT') or 'standard'}
path.write_text(yaml.safe_dump(metadata, sort_keys=False))
PY
    assert_benchmark_host_idle
    "$PYTHON" "$SCRIPT_DIR/benchmark_small_work.py" --binary "$BINARY" \
        --threads "$threads" --output "$RUN_DIR/samples_t${threads}.jsonl" || failed=1
done
# Separate instrumented build: never replace or time the latency executable.
DIAGNOSTIC_BUILD_START_NS="$($PYTHON -c 'import time; print(time.monotonic_ns())')"
DIAGNOSTIC_TARGET="${CARGO_TARGET_DIR:-$PROJECT_DIR/target}-allocations"
CARGO_TARGET_DIR="$DIAGNOSTIC_TARGET" cargo build --release --no-default-features \
    --features "$TENFERRO_CPU_FEATURES,allocation-diagnostics" --bin small_work_case
DIAGNOSTIC_BUILD_END_NS="$($PYTHON -c 'import time; print(time.monotonic_ns())')"
for threads in "${THREAD_COUNTS[@]}"; do
    configure_cpu_thread_env "$threads"
    "$PYTHON" "$SCRIPT_DIR/benchmark_small_work.py" --mode allocations \
        --binary "$DIAGNOSTIC_TARGET/release/small_work_case" --threads "$threads" \
        --output "$RUN_DIR/allocations_t${threads}.jsonl" || failed=1
done
REPORT_START_NS="$($PYTHON -c 'import time; print(time.monotonic_ns())')"
export BENCH_REPORT_START_NS="$REPORT_START_NS"
"$PYTHON" - "$RUN_DIR/wall_phases.json" "$START_NS" "$BUILD_END_NS" "$DIAGNOSTIC_BUILD_START_NS" "$DIAGNOSTIC_BUILD_END_NS" "$REPORT_START_NS" <<'PY'
import json,sys
from pathlib import Path
start,build,diagnostic_start,diagnostic_end,report=map(int,sys.argv[2:])
Path(sys.argv[1]).write_text(json.dumps({'validation_and_latency_build_seconds':(build-start)/1e9,
    'metadata_and_latency_collection_seconds':(diagnostic_start-build)/1e9,
    'allocation_diagnostic_build_seconds':(diagnostic_end-diagnostic_start)/1e9,
    'allocation_collection_seconds':(report-diagnostic_end)/1e9,
    'command_before_report_seconds':(report-start)/1e9},indent=2)+'\n')
PY
cp "$RUN_DIR/run_t${THREAD_COUNTS[0]}.yaml" "$RUN_DIR/run.yaml"
"$PYTHON" "$SCRIPT_DIR/benchmark_small_work.py" --report "$RUN_DIR" --target-profile "$BENCHMARK_TARGET_PROFILE"
exit "$failed"
