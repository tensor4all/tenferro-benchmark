#!/usr/bin/env bash
# gpu/perf_issues: CUDA workloads of open tenferro-rs performance issues
# (#2009 transfers, #1887 alloc_zero_output, #1885 small blocks / batched
# factorizations / live buffers / first-call layouts, #2078 raw-session entry,
# #2079 decode-sized eager ops, #2080 embedding gather, #2081 freed device
# memory, #2082 constant reshape, #2083 per-session cuBLAS handle, #1924
# vendor churn, #1839/#1849 SVD driver, #1846/#1852/#1923 batched eigh driver,
# #1898 Hadamard through dot_general, #1873/#1805 contraction-plan churn,
# #1949 owned vs borrowed dot_general destination), each with its reference
# arm (see scripts/generate_perf_issue_cases.py). A standalone entry point,
# like run_gpu_permutation.sh; run it sequentially, never next to another GPU
# benchmark. Coverage BENCH_COVERAGE=quick|full (versioned manifest
# benchmarks/gpu/manifests/perf_issues.yaml; full adds the first-call
# diagnostic), effort BENCH_EFFORT=scan|standard, optional
# BENCH_INSTANCE=id[,id...], PERF_ISSUES_CORRECTNESS_ONLY=1 for an
# execute-and-validate pass without timing.
#
# Usage: BENCHMARK_TARGET_PROFILE=nvidia-gpu ./scripts/run_gpu_perf_issues.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# GPU measurement validity: every other process on the device would distort the
# numbers. The list is recorded, and a non-empty list at measurement time fails
# the run unless BENCHMARK_ALLOW_BUSY_GPU=1 marks it explicitly diagnostic.
gpu_compute_apps() {
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader \
        2>/dev/null | tr -d ' \r' || true
}
require_idle_gpu() {
    local phase="$1"
    command -v nvidia-smi >/dev/null 2>&1 || return 0
    echo "CUDA compute processes $phase:"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
    if [[ -n "$(gpu_compute_apps)" ]]; then
        if [[ "${BENCHMARK_ALLOW_BUSY_GPU:-0}" != "1" ]]; then
            echo "ERROR: another process is using the CUDA device $phase; stop it or set BENCHMARK_ALLOW_BUSY_GPU=1 for a diagnostic run." >&2
            exit 1
        fi
        echo "WARNING: BENCHMARK_ALLOW_BUSY_GPU=1; measuring with a busy device." >&2
    fi
}
require_idle_gpu "before the run"
export BENCHMARK_TARGET_PROFILE="${BENCHMARK_TARGET_PROFILE:-nvidia-gpu}"
if [[ "$BENCHMARK_TARGET_PROFILE" != nvidia-gpu ]]; then
    echo "ERROR: gpu/perf_issues requires BENCHMARK_TARGET_PROFILE=nvidia-gpu." >&2
    exit 1
fi
DEVICE_ORDINAL="${GPU_BENCH_DEVICE:-0}"
PYTHON="$PROJECT_DIR/.venv/bin/python"
[[ -x "$PYTHON" ]] || PYTHON=python3
"$PYTHON" "$SCRIPT_DIR/generate_perf_issue_cases.py" --check
"$PYTHON" "$SCRIPT_DIR/validate_benchmark_suite.py" benchmarks/gpu/perf_issues.yaml
# Always rebuild before measuring (AGENTS.md GPU timing fairness).
CUBECL_DEBUG_LOG=0 CUDA_PATH="${CUDA_HOME:-/usr/local/cuda}" \
    cargo build --release --features cuda --bin benchmark_gpu_perf_issues
BINARY="${CARGO_TARGET_DIR:-$PROJECT_DIR/target}/release/benchmark_gpu_perf_issues"
TIMESTAMP="${BENCHMARK_TIMESTAMP:-$(date -u +%Y%m%d_%H%M%S)}"
RUN_DIR="$PROJECT_DIR/data/results/$BENCHMARK_TARGET_PROFILE/gpu/perf_issues/$TIMESTAMP"
mkdir -p "$RUN_DIR"
# The build ran above, so re-check the device right before measuring.
require_idle_gpu "at measurement time"
"$PYTHON" "$SCRIPT_DIR/collect_gpu_info.py" --device-ordinal "$DEVICE_ORDINAL" --markdown > "$RUN_DIR/gpu_info.md" || true
export BENCHMARK_COMMIT="${BENCHMARK_COMMIT:-$(git rev-parse HEAD)}"
export BENCH_COMMAND="${BENCH_COMMAND:-$(for v in BENCHMARK_TARGET_PROFILE GPU_BENCH_DEVICE BENCH_COVERAGE BENCH_EFFORT BENCH_INSTANCE PERF_ISSUES_CORRECTNESS_ONLY BENCHMARK_ALLOW_BUSY_GPU; do if [[ -n "${!v:-}" ]]; then printf '%s=%q ' "$v" "${!v}"; fi; done)$(printf '%q ' "$0" "$@")}"
"$PYTHON" "$SCRIPT_DIR/collect_run_metadata.py" \
    --suite-id gpu/perf_issues --target-profile "$BENCHMARK_TARGET_PROFILE" \
    --suite-file benchmarks/gpu/perf_issues.yaml --timestamp "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    --tenferro-dir "$PROJECT_DIR/extern/tenferro-rs" --features cuda --blas none \
    --cuda-device-ordinal "$DEVICE_ORDINAL" --output "$RUN_DIR/run_t1.yaml"
"$PYTHON" - "$RUN_DIR/run_t1.yaml" <<'PY'
import os, sys, subprocess, yaml
from pathlib import Path
path = Path(sys.argv[1])
metadata = yaml.safe_load(path.read_text())
metadata['environment'].setdefault('env', {}).update({
    'BENCHMARK_COMMIT': os.environ['BENCHMARK_COMMIT'], 'BENCHMARK_BUILD_PROFILE': 'release',
    'RUSTC_VERSION': subprocess.check_output(['rustc', '--version'], text=True).strip(),
    'BENCH_INSTANCE': os.environ.get('BENCH_INSTANCE') or None,
    'BENCHMARK_ALLOW_BUSY_GPU': os.environ.get('BENCHMARK_ALLOW_BUSY_GPU') or None})


def compute_apps():
    result = subprocess.run(
        ['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
         '--format=csv,noheader'],
        capture_output=True, text=True, check=False)
    return [line for line in result.stdout.strip().splitlines() if line.strip()]


# GPU validity: what else was on the device when the run started.
metadata['environment']['gpu_compute_apps_at_start'] = compute_apps()
sys.path.insert(0, 'scripts')
import benchmark_perf_issues as suite
_, _, manifest, coverage, expected, selected, raw = suite.plan('gpu')
metadata['collection'] = {
    'command': os.environ.get('BENCH_COMMAND', ''), 'coverage': coverage,
    'manifest_version': manifest['manifest_version'], 'selection_filter': raw,
    'expected_cases': len(expected), 'selected_cases': len(selected),
    'effort': os.environ.get('BENCH_EFFORT') or 'standard',
    'correctness_only': os.environ.get('PERF_ISSUES_CORRECTNESS_ONLY') == '1'}
path.write_text(yaml.safe_dump(metadata, sort_keys=False))
PY
cp "$RUN_DIR/run_t1.yaml" "$RUN_DIR/run.yaml"
EXTRA=()
if [[ "${PERF_ISSUES_CORRECTNESS_ONLY:-0}" == 1 ]]; then
    EXTRA+=(--correctness-only)
fi
failed=0
CUBECL_DEBUG_LOG=0 LD_LIBRARY_PATH="${CUDA_HOME:-/usr/local/cuda}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" \
    "$PYTHON" "$SCRIPT_DIR/benchmark_perf_issues.py" --suite gpu --device "$DEVICE_ORDINAL" \
    --binary "$BINARY" --threads 1 --output "$RUN_DIR/samples_t1.jsonl" "${EXTRA[@]}" || failed=1
# Record the device state after the measurement and refuse to pass off a run
# that shared the device with another process.
if command -v nvidia-smi >/dev/null 2>&1; then
    echo "CUDA compute processes after the run:"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
    "$PYTHON" - "$RUN_DIR/run_t1.yaml" "$RUN_DIR/run.yaml" <<'PY'
import subprocess
import sys
from pathlib import Path
import yaml

result = subprocess.run(
    ['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
     '--format=csv,noheader'], capture_output=True, text=True, check=False)
apps = [line for line in result.stdout.strip().splitlines() if line.strip()]
for argument in sys.argv[1:]:
    path = Path(argument)
    metadata = yaml.safe_load(path.read_text())
    metadata['environment']['gpu_compute_apps_at_end'] = apps
    path.write_text(yaml.safe_dump(metadata, sort_keys=False))
PY
    if [[ -n "$(gpu_compute_apps)" && "${BENCHMARK_ALLOW_BUSY_GPU:-0}" != "1" ]]; then
        echo "ERROR: another process appeared on the CUDA device during the run; the report is not a clean measurement." >&2
        exit 1
    fi
fi
"$PYTHON" "$SCRIPT_DIR/benchmark_perf_issues.py" --suite gpu --report "$RUN_DIR" \
    --target-profile "$BENCHMARK_TARGET_PROFILE"
exit "$failed"
