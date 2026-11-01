#!/usr/bin/env bash
set -euo pipefail

# Runs the CPU FFT benchmark suite:
#   - tenferro-rs TensorFftExt one-shot execution
#   - tenferro-rs TensorReadFftExt including its documented materialization path
#   - tenferro-rs FftExecutor with caller-owned plan cache
#   - tenferro-rs EagerTensorFftExt with a reused eager runtime
#   - tenferro-rs TracedTensorFftExt with one reused compiled graph
#   - PyTorch torch.fft
#
# The initial suite intentionally measures only 1D transforms. That keeps the
# transform axis contiguous for both tenferro-rs column-major tensors and
# PyTorch row-major tensors, avoiding layout-driven batched FFT artifacts.

if [[ $# -eq 0 ]]; then
    THREAD_COUNTS=(1 4)
else
    THREAD_COUNTS=("$@")
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
export BENCH_COMMAND_START_NS="$(python3 -c 'import time; print(time.monotonic_ns())')"

# shellcheck source=scripts/publish_run.sh
source "$SCRIPT_DIR/publish_run.sh"

# shellcheck source=scripts/cpu_blas_provider.sh
source "$SCRIPT_DIR/cpu_blas_provider.sh"

TENFERRO_CPU_FEATURES="$(normalize_cpu_blas_features "${TENFERRO_CPU_FEATURES:-}")"
export TENFERRO_CPU_FEATURES

default_target_profile() {
    case "$(benchmark_host_os)" in
        Darwin) printf '%s\n' "mac-cpu" ;;
        *) printf '%s\n' "amd-cpu" ;;
    esac
}

BENCHMARK_TARGET_PROFILE="${BENCHMARK_TARGET_PROFILE:-$(default_target_profile)}"
case "$BENCHMARK_TARGET_PROFILE" in
    mac-cpu|amd-cpu|zen5-cpu)
        export BENCHMARK_TARGET_PROFILE
        ;;
    *)
        echo "ERROR: BENCHMARK_TARGET_PROFILE must be mac-cpu, amd-cpu, or zen5-cpu for CPU FFT." >&2
        exit 1
        ;;
esac

# shellcheck source=scripts/thread_env.sh
source "$SCRIPT_DIR/thread_env.sh"

if [[ "${SKIP_EXTERN_SETUP:-0}" != "1" ]]; then
    # shellcheck source=scripts/setup_extern_deps.sh
    source "$SCRIPT_DIR/setup_extern_deps.sh"
fi

# shellcheck source=scripts/python_venv.sh
source "$SCRIPT_DIR/python_venv.sh"
reset_benchmark_python_venv "$PROJECT_DIR"
prepare_cpu_benchmark_python_venv "$PROJECT_DIR"

ensure_blas_env_for_features "$TENFERRO_CPU_FEATURES"

# shellcheck source=scripts/benchmark_host_idle.sh
source "$SCRIPT_DIR/benchmark_host_idle.sh"
assert_benchmark_host_idle

RESULTS_ROOT="$PROJECT_DIR/data/results"
REPORTS_DIR="$PROJECT_DIR/result"
SUITE_ID="cpu/fft"
SUITE_FILE="$PROJECT_DIR/benchmarks/cpu/fft.yaml"
BENCHMARK_TIMESTAMP="${BENCHMARK_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
export BENCHMARK_TIMESTAMP
RUN_TIMESTAMP_RFC3339="$(python3 - <<'PY'
from datetime import datetime, timezone
print(datetime.now(timezone.utc).isoformat())
PY
)"

RUN_DIR="$RESULTS_ROOT/$BENCHMARK_TARGET_PROFILE/cpu/fft/$BENCHMARK_TIMESTAMP"
RUN_YAML="$RUN_DIR/run.yaml"
REPORT="$RUN_DIR/report.md"
LATEST_REPORT="$REPORTS_DIR/$BENCHMARK_TARGET_PROFILE/cpu/fft.md"
TENFERRO_DIR="${TENFERRO_RS_DIR:-$PROJECT_DIR/extern/tenferro-rs}"
TENFERRO_COMMIT=""
if [[ -d "$TENFERRO_DIR/.git" ]] && command -v git >/dev/null 2>&1; then
    TENFERRO_COMMIT="$(git -C "$TENFERRO_DIR" rev-parse HEAD 2>/dev/null || true)"
fi

mkdir -p "$RUN_DIR"

blas_impl_for_metadata="$(blas_impl_for_features "$TENFERRO_CPU_FEATURES")"
collect_run_metadata() {
    local output="$1"
    local metadata_args=(
        --suite-id "$SUITE_ID"
        --target-profile "$BENCHMARK_TARGET_PROFILE"
        --suite-file "${SUITE_FILE#$PROJECT_DIR/}"
        --timestamp "$RUN_TIMESTAMP_RFC3339"
        --tenferro-dir "$TENFERRO_DIR"
        --features "$TENFERRO_CPU_FEATURES"
        --blas "$blas_impl_for_metadata"
        --output "$output"
    )
    [[ -n "$TENFERRO_COMMIT" ]] && metadata_args+=(--tenferro-commit "$TENFERRO_COMMIT")
    if command -v uv >/dev/null 2>&1; then
        uv run python "$SCRIPT_DIR/collect_run_metadata.py" "${metadata_args[@]}" \
            || python3 "$SCRIPT_DIR/collect_run_metadata.py" "${metadata_args[@]}"
    else
        python3 "$SCRIPT_DIR/collect_run_metadata.py" "${metadata_args[@]}"
    fi
}

configure_cpu_thread_env "${THREAD_COUNTS[0]}"
collect_run_metadata "$RUN_YAML"

# Host and thread-environment snapshots for this run's report header.
if command -v uv >/dev/null 2>&1; then
    uv run python "$SCRIPT_DIR/collect_cpu_info.py" --markdown > "$RUN_DIR/cpu_info.md" \
        || python3 "$SCRIPT_DIR/collect_cpu_info.py" --markdown > "$RUN_DIR/cpu_info.md"
else
    python3 "$SCRIPT_DIR/collect_cpu_info.py" --markdown > "$RUN_DIR/cpu_info.md"
fi
begin_thread_envs_snapshot "$RUN_DIR/thread_env.txt"

echo "CPU FFT benchmark suite"
echo "Project dir:  $PROJECT_DIR"
echo "Thread counts: ${THREAD_COUNTS[*]}"
echo "Timestamp:    $BENCHMARK_TIMESTAMP"
echo "Suite:        $SUITE_ID"
echo "Target:       $BENCHMARK_TARGET_PROFILE"
echo "Run dir:      $RUN_DIR"
echo "Features:     $TENFERRO_CPU_FEATURES"
echo "FFT lengths:  ${FFT_BENCH_LENGTHS:-1048576}"
[[ -n "$TENFERRO_COMMIT" ]] && echo "tenferro-rs:  $TENFERRO_COMMIT"
echo ""

CSVS=()
for NUM_THREADS in "${THREAD_COUNTS[@]}"; do
    echo "--- Thread count: $NUM_THREADS ---"
    configure_cpu_thread_env "$NUM_THREADS"
    print_cpu_thread_env
    append_thread_env_section "$RUN_DIR/thread_env.txt" "$NUM_THREADS"
    RUN_T_YAML="$RUN_DIR/run_t${NUM_THREADS}.yaml"
    CSV="$RUN_DIR/cpu_fft_t${NUM_THREADS}_${BENCHMARK_TIMESTAMP}.csv"
    collect_run_metadata "$RUN_T_YAML"

    assert_benchmark_host_idle
    cargo run --release --no-default-features --features "$TENFERRO_CPU_FEATURES" --bin benchmark_cpu_fft -- \
        --num-threads "$NUM_THREADS" \
        --lengths "${FFT_BENCH_LENGTHS:-1048576}" \
        --output "$CSV"

    assert_benchmark_host_idle
    if command -v uv >/dev/null 2>&1; then
        uv run python "$SCRIPT_DIR/benchmark_cpu_fft_python.py" \
            --num-threads "$NUM_THREADS" \
            --lengths "${FFT_BENCH_LENGTHS:-1048576}" \
            --output "$CSV"
    else
        python3 "$SCRIPT_DIR/benchmark_cpu_fft_python.py" \
            --num-threads "$NUM_THREADS" \
            --lengths "${FFT_BENCH_LENGTHS:-1048576}" \
            --output "$CSV"
    fi
    CSVS+=("$CSV")
done

finalize_benchmark_run "$PROJECT_DIR" "$RUN_DIR"
publish_cpu_runs "$PROJECT_DIR" "$RUN_DIR"

echo ""
echo "CPU FFT benchmark complete"
for CSV in "${CSVS[@]}"; do echo "CSV:      $CSV"; done
echo "Report:   $REPORT"
echo "Latest:   $LATEST_REPORT"
