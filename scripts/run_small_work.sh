#!/usr/bin/env bash
# Ordinary cpu/small_work suite; optional BENCH_INSTANCE=id[,id...] selection.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"
THREAD_COUNTS=("${@:-1}")
source "$SCRIPT_DIR/cpu_blas_provider.sh"
source "$SCRIPT_DIR/thread_env.sh"
export TENFERRO_CPU_FEATURES="$(normalize_cpu_blas_features "${TENFERRO_CPU_FEATURES:-}")"
export TENFERRO_CPU_BACKEND_KIND="${TENFERRO_CPU_BACKEND_KIND:-blas}"
export BENCHMARK_TARGET_PROFILE="${BENCHMARK_TARGET_PROFILE:-amd-cpu}"
if [[ "${SKIP_EXTERN_SETUP:-0}" != 1 ]]; then
    source "$SCRIPT_DIR/setup_extern_deps.sh"
fi
ensure_blas_env_for_features "$TENFERRO_CPU_FEATURES"
PYTHON="$PROJECT_DIR/.venv/bin/python"
[[ -x "$PYTHON" ]] || PYTHON=python3
"$PYTHON" "$SCRIPT_DIR/validate_benchmark_suite.py" benchmarks/cpu/small_work.yaml
# Build once before all sequential timing collection. No idle/affinity/performance gate.
cargo build --release --features "$TENFERRO_CPU_FEATURES" --bin small_work_case
BINARY="${CARGO_TARGET_DIR:-$PROJECT_DIR/target}/release/small_work_case"
TIMESTAMP="${BENCHMARK_TIMESTAMP:-$(date -u +%Y%m%d_%H%M%S)}"
"$PYTHON" -c 'from scripts.benchmark_layout import safe_target_profile; import sys; safe_target_profile(sys.argv[1])' "$BENCHMARK_TARGET_PROFILE"
RUN_DIR="$PROJECT_DIR/data/results/$BENCHMARK_TARGET_PROFILE/cpu/small_work/$TIMESTAMP"
mkdir -p "$RUN_DIR"
"$PYTHON" "$SCRIPT_DIR/collect_cpu_info.py" --markdown > "$RUN_DIR/cpu_info.md"
export BENCHMARK_COMMIT="${BENCHMARK_COMMIT:-$(git rev-parse HEAD)}"
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
    'BENCH_INSTANCE': os.environ.get('BENCH_INSTANCE', 'all 154')})
path.write_text(yaml.safe_dump(metadata, sort_keys=False))
PY
    "$PYTHON" "$SCRIPT_DIR/benchmark_small_work.py" --binary "$BINARY" \
        --threads "$threads" --output "$RUN_DIR/samples_t${threads}.jsonl" || failed=1
done
cp "$RUN_DIR/run_t${THREAD_COUNTS[0]}.yaml" "$RUN_DIR/run.yaml"
"$PYTHON" "$SCRIPT_DIR/benchmark_small_work.py" --report "$RUN_DIR" --target-profile "$BENCHMARK_TARGET_PROFILE"
exit "$failed"
