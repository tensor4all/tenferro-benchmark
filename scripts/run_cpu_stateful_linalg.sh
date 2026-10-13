#!/usr/bin/env bash
# Standalone stateful QR, measured sequentially in the target devcontainer.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"
source "$SCRIPT_DIR/cpu_blas_provider.sh"
source "$SCRIPT_DIR/thread_env.sh"
source "$SCRIPT_DIR/benchmark_host_idle.sh"
export TENFERRO_CPU_FEATURES="$(normalize_cpu_blas_features "${TENFERRO_CPU_FEATURES:-}")"
export BENCHMARK_TARGET_PROFILE="${BENCHMARK_TARGET_PROFILE:-amd-cpu}"
export UV_NO_SYNC=1
THREAD_COUNTS=("${@:-1}")
if [[ "${SKIP_EXTERN_SETUP:-0}" != 1 ]]; then source "$SCRIPT_DIR/setup_extern_deps.sh"; fi
ensure_blas_env_for_features "$TENFERRO_CPU_FEATURES"
PYTHON="$PROJECT_DIR/.venv/bin/python"
[[ -x "$PYTHON" ]] || PYTHON=python3
START_NS="$($PYTHON -c 'import time; print(time.monotonic_ns())')"
"$PYTHON" "$SCRIPT_DIR/validate_benchmark_suite.py" benchmarks/cpu/stateful_linalg.yaml
cargo build --release --no-default-features --features "$TENFERRO_CPU_FEATURES" --bin benchmark_cpu_stateful_linalg
BUILD_NS="$($PYTHON -c 'import time; print(time.monotonic_ns())')"
if [[ -z "${BENCHMARK_COMMIT:-}" ]]; then BENCHMARK_COMMIT="$(git rev-parse HEAD)"; fi
export BENCHMARK_COMMIT
TIMESTAMP="${BENCHMARK_TIMESTAMP:-$(date -u +%Y%m%d_%H%M%S)}"
"$PYTHON" -c 'from scripts.benchmark_layout import safe_target_profile; import sys; safe_target_profile(sys.argv[1])' "$BENCHMARK_TARGET_PROFILE"
RUN_DIR="$PROJECT_DIR/data/results/$BENCHMARK_TARGET_PROFILE/cpu/stateful_linalg/$TIMESTAMP"
mkdir -p "$RUN_DIR"
"$PYTHON" "$SCRIPT_DIR/collect_cpu_info.py" --markdown > "$RUN_DIR/cpu_info.md"
failed=0
for threads in "${THREAD_COUNTS[@]}"; do
    configure_cpu_thread_env "$threads"
    "$PYTHON" "$SCRIPT_DIR/collect_run_metadata.py" --suite-id cpu/stateful_linalg \
        --target-profile "$BENCHMARK_TARGET_PROFILE" --suite-file benchmarks/cpu/stateful_linalg.yaml \
        --timestamp "$(date -u +%Y-%m-%dT%H:%M:%SZ)" --tenferro-dir extern/tenferro-rs \
        --features "$TENFERRO_CPU_FEATURES" --blas "$(blas_impl_for_features "$TENFERRO_CPU_FEATURES")" \
        --output "$RUN_DIR/run_t${threads}.yaml"
    "$PYTHON" - "$RUN_DIR/run_t${threads}.yaml" <<'PY'
import os,sys,yaml
from pathlib import Path
p=Path(sys.argv[1]);m=yaml.safe_load(p.read_text())
m['collection']={'harness_commit':os.environ['BENCHMARK_COMMIT'],'build_profile':'release',
    'case_manifest':'data/instances/stateful_linalg.json','manifest_version':1,
    'process_replicates':7,'samples_per_process':3,'warmups':3,'initialization_passes':1,
    'process_order':'ascending case order on even rounds; reverse on odd rounds',
    'command':os.environ.get('BENCH_COMMAND','scripts/run_cpu_stateful_linalg.sh'),
    'affinity':'none','scope':'setup untimed; retained-output whole interleaved workflow batches'}
p.write_text(yaml.safe_dump(m,sort_keys=False))
PY
    assert_benchmark_host_idle
    "$PYTHON" "$SCRIPT_DIR/benchmark_cpu_stateful_linalg.py" \
        --binary "${CARGO_TARGET_DIR:-$PROJECT_DIR/target}/release/benchmark_cpu_stateful_linalg" \
        --threads "$threads" --output "$RUN_DIR/samples_t${threads}.jsonl" || failed=1
done
REPORT_NS="$($PYTHON -c 'import time; print(time.monotonic_ns())')"
"$PYTHON" - "$RUN_DIR/wall_phases.json" "$START_NS" "$BUILD_NS" "$REPORT_NS" <<'PY'
import json,sys
from pathlib import Path
start,build,report=map(int,sys.argv[2:])
Path(sys.argv[1]).write_text(json.dumps({'build_and_validation_seconds':(build-start)/1e9,
    'metadata_startup_and_collection_seconds':(report-build)/1e9},indent=2)+'\n')
PY
cp "$RUN_DIR/run_t${THREAD_COUNTS[0]}.yaml" "$RUN_DIR/run.yaml"
"$PYTHON" "$SCRIPT_DIR/benchmark_cpu_stateful_linalg.py" --report "$RUN_DIR" --target-profile "$BENCHMARK_TARGET_PROFILE"
exit "$failed"
