#!/usr/bin/env bash
# cpu/perf_issues: workloads of open tenferro-rs performance issues, each with
# its reference arms (see scripts/generate_perf_issue_cases.py). Coverage
# BENCH_COVERAGE=quick|full (versioned manifest
# benchmarks/cpu/manifests/perf_issues.yaml), effort BENCH_EFFORT=scan|standard,
# optional BENCH_INSTANCE=id[,id...] filter, PERF_ISSUES_CORRECTNESS_ONLY=1 for
# an execute-and-validate pass without timing.
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
if [[ "${SKIP_EXTERN_SETUP:-0}" != 1 ]]; then
    source "$SCRIPT_DIR/setup_extern_deps.sh"
fi
ensure_blas_env_for_features "$TENFERRO_CPU_FEATURES"
PYTHON="$PROJECT_DIR/.venv/bin/python"
[[ -x "$PYTHON" ]] || PYTHON=python3
"$PYTHON" "$SCRIPT_DIR/generate_perf_issue_cases.py" --check
"$PYTHON" "$SCRIPT_DIR/validate_benchmark_suite.py" benchmarks/cpu/perf_issues.yaml
# Build once before sequential timing collection.
cargo build --release --features "$TENFERRO_CPU_FEATURES" --bin perf_issue_case
BINARY="${CARGO_TARGET_DIR:-$PROJECT_DIR/target}/release/perf_issue_case"
TIMESTAMP="${BENCHMARK_TIMESTAMP:-$(date -u +%Y%m%d_%H%M%S)}"
"$PYTHON" -c 'from scripts.benchmark_layout import safe_target_profile; import sys; safe_target_profile(sys.argv[1])' "$BENCHMARK_TARGET_PROFILE"
RUN_DIR="$PROJECT_DIR/data/results/$BENCHMARK_TARGET_PROFILE/cpu/perf_issues/$TIMESTAMP"
mkdir -p "$RUN_DIR"
"$PYTHON" "$SCRIPT_DIR/collect_cpu_info.py" --markdown > "$RUN_DIR/cpu_info.md"
export BENCHMARK_COMMIT="${BENCHMARK_COMMIT:-$(git rev-parse HEAD)}"
export BENCH_COMMAND="${BENCH_COMMAND:-$(for v in BENCHMARK_TARGET_PROFILE TENFERRO_CPU_FEATURES BENCH_COVERAGE BENCH_EFFORT BENCH_INSTANCE PERF_ISSUES_CORRECTNESS_ONLY; do if [[ -n "${!v:-}" ]]; then printf '%s=%q ' "$v" "${!v}"; fi; done)$(printf '%q ' "$0" "$@")}"
EXTRA=()
if [[ "${PERF_ISSUES_CORRECTNESS_ONLY:-0}" == 1 ]]; then
    EXTRA+=(--correctness-only)
fi
failed=0
for threads in "${THREAD_COUNTS[@]}"; do
    configure_cpu_thread_env "$threads"
    # matrixmultiply's threaded sgemm (host-sgemm reference arm).
    export MATMUL_NUM_THREADS="$threads"
    "$PYTHON" "$SCRIPT_DIR/collect_run_metadata.py" \
        --suite-id cpu/perf_issues --target-profile "$BENCHMARK_TARGET_PROFILE" \
        --suite-file benchmarks/cpu/perf_issues.yaml --timestamp "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
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
    'MATMUL_NUM_THREADS': os.environ.get('MATMUL_NUM_THREADS'),
    'BENCH_INSTANCE': os.environ.get('BENCH_INSTANCE') or None})
sys.path.insert(0, 'scripts')
import benchmark_perf_issues as suite
_, _, manifest, coverage, expected, selected, raw = suite.plan()
metadata['collection'] = {
    'command': os.environ.get('BENCH_COMMAND', ''), 'coverage': coverage,
    'manifest_version': manifest['manifest_version'], 'selection_filter': raw,
    'expected_cases': len(expected), 'selected_cases': len(selected),
    'effort': os.environ.get('BENCH_EFFORT') or 'standard',
    'correctness_only': os.environ.get('PERF_ISSUES_CORRECTNESS_ONLY') == '1'}
path.write_text(yaml.safe_dump(metadata, sort_keys=False))
PY
    if [[ "${PERF_ISSUES_CORRECTNESS_ONLY:-0}" != 1 ]]; then
        assert_benchmark_host_idle
    fi
    "$PYTHON" "$SCRIPT_DIR/benchmark_perf_issues.py" --binary "$BINARY" \
        --threads "$threads" --output "$RUN_DIR/samples_t${threads}.jsonl" "${EXTRA[@]}" || failed=1
done
cp "$RUN_DIR/run_t${THREAD_COUNTS[0]}.yaml" "$RUN_DIR/run.yaml"
"$PYTHON" "$SCRIPT_DIR/benchmark_perf_issues.py" --report "$RUN_DIR" --target-profile "$BENCHMARK_TARGET_PROFILE"
exit "$failed"
