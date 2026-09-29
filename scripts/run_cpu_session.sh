#!/usr/bin/env bash
# cpu/session_matrix: loop-of-independent-calls and one-operation batched routes.
#
# Usage: scripts/run_cpu_session.sh [threads...]            (default: 1)
#
# Environment:
#   BENCH_COVERAGE=quick|full         case IDs from benchmarks/cpu/manifests/session_matrix.yaml
#   BENCH_EFFORT=scan|standard|confirm repetitions (confirm needs a declared
#                                     BENCH_CONFIRM_CONFIG / benchmarks/cpu/confirmation.yaml)
#   BENCH_INSTANCE=id[,id...]         optional filter inside the coverage choice
#   BENCH_SESSION_BINARY=<path>       prebuilt benchmark_cpu_session (paired runs); skips the build
#   BENCH_TENFERRO_DIR=<dir>          tenferro-rs checkout the binary was built from (metadata)
#   BENCH_RUN_DIR=<dir>               raw-run directory (default data/results/<profile>/cpu/session_matrix/<ts>)
#   BENCH_SKIP_LATEST=1               do not overwrite result/<profile>/cpu/session_matrix.md
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"
source "$SCRIPT_DIR/cpu_blas_provider.sh"
source "$SCRIPT_DIR/thread_env.sh"
source "$SCRIPT_DIR/benchmark_host_idle.sh"
export TENFERRO_CPU_FEATURES="$(normalize_cpu_blas_features "${TENFERRO_CPU_FEATURES:-}")"
export BENCHMARK_TARGET_PROFILE="${BENCHMARK_TARGET_PROFILE:-mac-cpu}"
TENFERRO_DIR="${BENCH_TENFERRO_DIR:-extern/tenferro-rs}"
PYTHON="$PROJECT_DIR/.venv/bin/python"
[[ -x "$PYTHON" ]] || PYTHON=python3
COMMAND="$(printf '%q ' "$0" "$@")"
for var in BENCHMARK_TARGET_PROFILE TENFERRO_CPU_FEATURES BENCH_COVERAGE BENCH_EFFORT \
    BENCH_INSTANCE BENCH_CONFIRM_CONFIG BENCH_SESSION_BINARY; do
    if [[ -n "${!var:-}" ]]; then COMMAND="$var=${!var} $COMMAND"; fi
done
if [[ -n "${BENCH_SESSION_BINARY:-}" ]]; then
    BINARY="$BENCH_SESSION_BINARY"
else
    if [[ "$(git -C extern/tenferro-rs branch --show-current)" == main ]]; then
        git -C extern/tenferro-rs pull --ff-only
    fi
    ensure_blas_env_for_features "$TENFERRO_CPU_FEATURES"
    cargo build --release --no-default-features --features "$TENFERRO_CPU_FEATURES" --bin benchmark_cpu_session
    BINARY="${CARGO_TARGET_DIR:-$PROJECT_DIR/target}/release/benchmark_cpu_session"
fi
"$PYTHON" -c 'from scripts.benchmark_layout import safe_target_profile; import sys; safe_target_profile(sys.argv[1])' "$BENCHMARK_TARGET_PROFILE"
"$PYTHON" scripts/generate_session_matrix_cases.py --check
"$PYTHON" scripts/validate_benchmark_suite.py benchmarks/cpu/session_matrix.yaml
SELECTION="$("$PYTHON" scripts/benchmark_cpu_session.py --describe-selection)"
RUN_DIR="${BENCH_RUN_DIR:-data/results/$BENCHMARK_TARGET_PROFILE/cpu/session_matrix/$(date -u +%Y%m%d_%H%M%S)}"
mkdir -p "$RUN_DIR"
"$PYTHON" "$SCRIPT_DIR/collect_cpu_info.py" --markdown > "$RUN_DIR/cpu_info.md"
export BENCHMARK_COMMIT="$(git rev-parse HEAD)"
THREAD_COUNTS=("${@:-1}")
failed=0
for threads in "${THREAD_COUNTS[@]}"; do
    configure_cpu_thread_env "$threads"
    "$PYTHON" scripts/collect_run_metadata.py --suite-id cpu/session_matrix \
        --target-profile "$BENCHMARK_TARGET_PROFILE" --suite-file benchmarks/cpu/session_matrix.yaml \
        --timestamp "$(date -u +%Y-%m-%dT%H:%M:%SZ)" --tenferro-dir "$TENFERRO_DIR" \
        --features "$TENFERRO_CPU_FEATURES" --blas "$(blas_impl_for_features "$TENFERRO_CPU_FEATURES")" \
        --command "$COMMAND" --collection-json "$SELECTION" \
        --output "$RUN_DIR/run_t${threads}.yaml"
    assert_benchmark_host_idle
    "$PYTHON" scripts/benchmark_cpu_session.py --binary "$BINARY" \
        --threads "$threads" --run-dir "$RUN_DIR" || failed=1
done
if [[ "${BENCH_SKIP_LATEST:-0}" == 1 ]]; then
    "$PYTHON" scripts/benchmark_cpu_session.py --report --run-dir "$RUN_DIR" \
        --target-profile "$BENCHMARK_TARGET_PROFILE" --no-latest
else
    "$PYTHON" scripts/benchmark_cpu_session.py --report --run-dir "$RUN_DIR" \
        --target-profile "$BENCHMARK_TARGET_PROFILE"
fi
exit "$failed"
