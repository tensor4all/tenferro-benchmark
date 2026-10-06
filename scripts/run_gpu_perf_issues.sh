#!/usr/bin/env bash
# gpu/perf_issues: CUDA workloads of open tenferro-rs performance issues
# (#2009 transfers, #1887 alloc_zero_output, #1885 small blocks / batched
# factorizations / live buffers / first-call layouts), each with its reference
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
"$PYTHON" "$SCRIPT_DIR/collect_gpu_info.py" --device-ordinal "$DEVICE_ORDINAL" --markdown > "$RUN_DIR/gpu_info.md" || true
export BENCHMARK_COMMIT="${BENCHMARK_COMMIT:-$(git rev-parse HEAD)}"
export BENCH_COMMAND="${BENCH_COMMAND:-$(for v in BENCHMARK_TARGET_PROFILE GPU_BENCH_DEVICE BENCH_COVERAGE BENCH_EFFORT BENCH_INSTANCE PERF_ISSUES_CORRECTNESS_ONLY; do if [[ -n "${!v:-}" ]]; then printf '%s=%q ' "$v" "${!v}"; fi; done)$(printf '%q ' "$0" "$@")}"
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
    'BENCH_INSTANCE': os.environ.get('BENCH_INSTANCE') or None})
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
"$PYTHON" "$SCRIPT_DIR/benchmark_perf_issues.py" --suite gpu --report "$RUN_DIR" \
    --target-profile "$BENCHMARK_TARGET_PROFILE"
exit "$failed"
