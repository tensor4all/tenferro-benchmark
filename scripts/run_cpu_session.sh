#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"
source "$SCRIPT_DIR/cpu_blas_provider.sh"
source "$SCRIPT_DIR/thread_env.sh"
source "$SCRIPT_DIR/benchmark_host_idle.sh"
export TENFERRO_CPU_FEATURES="$(normalize_cpu_blas_features "${TENFERRO_CPU_FEATURES:-}")"
export BENCHMARK_TARGET_PROFILE="${BENCHMARK_TARGET_PROFILE:-mac-cpu}"
if [[ "$(git -C extern/tenferro-rs branch --show-current)" == main ]]; then
    git -C extern/tenferro-rs pull --ff-only
fi
ensure_blas_env_for_features "$TENFERRO_CPU_FEATURES"
cargo build --release --no-default-features --features "$TENFERRO_CPU_FEATURES" --bin benchmark_cpu_session
PYTHON="$PROJECT_DIR/.venv/bin/python"
[[ -x "$PYTHON" ]] || PYTHON=python3
"$PYTHON" -c 'from scripts.benchmark_layout import safe_target_profile; import sys; safe_target_profile(sys.argv[1])' "$BENCHMARK_TARGET_PROFILE"
RUN_DIR="data/results/$BENCHMARK_TARGET_PROFILE/cpu/session_matrix/$(date -u +%Y%m%d_%H%M%S)"
mkdir -p "$RUN_DIR"
export BENCHMARK_COMMIT="$(git rev-parse HEAD)"
THREAD_COUNTS=("${@:-1}")
for threads in "${THREAD_COUNTS[@]}"; do
    configure_cpu_thread_env "$threads"
    "$PYTHON" scripts/collect_run_metadata.py --suite-id cpu/session_matrix \
        --target-profile "$BENCHMARK_TARGET_PROFILE" --suite-file benchmarks/cpu/session_matrix.yaml \
        --timestamp "$(date -u +%Y-%m-%dT%H:%M:%SZ)" --tenferro-dir extern/tenferro-rs \
        --features "$TENFERRO_CPU_FEATURES" --blas "$(blas_impl_for_features "$TENFERRO_CPU_FEATURES")" \
        --output "$RUN_DIR/run_t${threads}.yaml"
    assert_benchmark_host_idle
    "$PYTHON" scripts/benchmark_cpu_session.py --binary "$PROJECT_DIR/target/release/benchmark_cpu_session" \
        --threads "$threads" --run-dir "$RUN_DIR"
done
"$PYTHON" scripts/benchmark_cpu_session.py --report --run-dir "$RUN_DIR" --target-profile "$BENCHMARK_TARGET_PROFILE"
