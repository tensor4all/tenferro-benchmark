#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# CPU einsum benchmark runner:
#   - tenferro-rs trace + eager (Rust)
#   - PyTorch CPU + JAX CPU (Python)
#   - OMEinsum.jl (Julia), forced through each instance's precomputed
#     contraction path (mode omeinsum_path); see docs/einsum-suite.md
#   - CPU ops microbenchmarks (primal linalg + JVP/VJP on trace + eager backward)
#
# Usage:
#   ./scripts/run_all.sh [NUM_THREADS]
#   ./scripts/run_all.sh 1 4
#
# Compares einsum performance for the instances listed in
# benchmarks/cpu/einsum.yaml and writes result/<target_profile>/cpu/einsum.md.
# CPU ops (including linalg JVP/VJP) are written to:
#   result/<target_profile>/cpu/cpu_ops.md
#   result/<target_profile>/cpu/linalg_jvp_vjp.md
#
# Passing multiple thread counts runs the main einsum / cpu_ops / linalg JVP/VJP
# path once per thread count, then runs cpu/fft, cpu/public_api, cpu/small_work,
# cpu/session_matrix, cpu/overhead_smoke and cpu/perf_issues once over the same
# thread-count list. Set RUN_FFT_SUITE=0, RUN_PUBLIC_API_SUITE=0,
# RUN_SMALL_WORK_SUITE=0, RUN_OVERHEAD_SMOKE_SUITE=0, RUN_PERF_ISSUES_SUITE=0, or
# RUN_PERMUTATION_SUITE=0 to skip one of those multi-thread-count follow-up
# suites.
#
# Set RUN_FFT_SUITE=1 in a single-thread-count invocation to also run
# scripts/run_cpu_fft.sh (the cpu/fft suite) sequentially after CPU ops.
#
# Set RUN_PUBLIC_API_SUITE=1 to also run scripts/run_cpu_public_api.sh (the
# cpu/public_api suite) sequentially after CPU FFT if enabled.
#
# cpu/small_work, cpu/session_matrix and cpu/perf_issues run sequentially by
# default; set RUN_SMALL_WORK_SUITE=0 or RUN_PERF_ISSUES_SUITE=0 to skip them.
# BENCH_COVERAGE=quick|full and BENCH_EFFORT=scan|standard|confirm apply to
# these suites.
#
# cpu/overhead_smoke (scripts/run_overhead_smoke.sh, one run thread, a few
# minutes) runs once after them by default; set RUN_OVERHEAD_SMOKE_SUITE=0 to
# skip it.
#
# Each run directory is finalized after collection and then published. Set
# BENCHMARK_RAW_ONLY=1 to stop after finalization: no report, summary or page is
# written (see scripts/publish_run.sh).
#
# Set RUN_PERMUTATION_SUITE=1 to also run scripts/run_permutation.sh (the
# cpu/permutation suite) sequentially after everything above completes; see
# docs/permutation-suite.md. Off by default.
# ---------------------------------------------------------------------------

NUM_THREADS="${1:-1}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
export BENCHMARK_COMMIT="${BENCHMARK_COMMIT:-$(git -C "$PROJECT_DIR" rev-parse HEAD)}"
export BENCH_COMMAND_START_NS="$(python3 -c 'import time; print(time.monotonic_ns())')"

# shellcheck source=scripts/publish_run.sh
source "$SCRIPT_DIR/publish_run.sh"

if [[ $# -gt 1 && "${RUN_ALL_MAIN_ONLY:-0}" != "1" ]]; then
    THREAD_COUNTS=("$@")
    RUN_ALL_COLLECTED_RUNS_FILE="$(mktemp)"
    export RUN_ALL_COLLECTED_RUNS_FILE
    trap 'rm -f "$RUN_ALL_COLLECTED_RUNS_FILE"' EXIT
    for threads in "${THREAD_COUNTS[@]}"; do
        RUN_ALL_MAIN_ONLY=1 "$0" "$threads"
    done

    # Read the per-thread run directories one line at a time: macOS ships
    # bash 3.2, which has no array-file builtin. The count keeps an empty file
    # from expanding an empty array under `set -u`.
    COLLECTED_CPU_RUNS=()
    collected_count=0
    while IFS= read -r collected_run; do
        [[ -n "$collected_run" ]] || continue
        COLLECTED_CPU_RUNS+=("$collected_run")
        collected_count=$((collected_count + 1))
    done < "$RUN_ALL_COLLECTED_RUNS_FILE"
    if [[ "$collected_count" -gt 0 ]]; then
        echo "Publishing the combined CPU pages from: ${COLLECTED_CPU_RUNS[*]}"
        publish_cpu_runs "$PROJECT_DIR" "${COLLECTED_CPU_RUNS[@]}"
        echo ""
    fi

    if [[ "${RUN_FFT_SUITE:-1}" == "1" ]]; then
        echo "Running cpu/fft suite for threads: ${THREAD_COUNTS[*]}..."
        "$SCRIPT_DIR/run_cpu_fft.sh" "${THREAD_COUNTS[@]}"
        echo ""
    fi

    if [[ "${RUN_PUBLIC_API_SUITE:-1}" == "1" ]]; then
        public_api_profile="${RUN_PUBLIC_API_PROFILE:-${PUBLICATION_GATE_PROFILE:-quick}}"
        echo "Running cpu/public_api suite for threads: ${THREAD_COUNTS[*]} (profile=$public_api_profile)..."
        PUBLICATION_GATE_PROFILE="$public_api_profile" \
            "$SCRIPT_DIR/run_cpu_public_api.sh" "${THREAD_COUNTS[@]}"
        echo ""
    fi

    if [[ "${RUN_SMALL_WORK_SUITE:-1}" == "1" ]]; then
        "$SCRIPT_DIR/run_small_work.sh" "${THREAD_COUNTS[@]}"
        "$SCRIPT_DIR/run_cpu_session.sh" "${THREAD_COUNTS[@]}"
    fi

    if [[ "${RUN_PERF_ISSUES_SUITE:-1}" == "1" ]]; then
        echo "Running cpu/perf_issues suite for threads: ${THREAD_COUNTS[*]}..."
        "$SCRIPT_DIR/run_perf_issues.sh" "${THREAD_COUNTS[@]}"
        echo ""
    fi

    if [[ "${RUN_OVERHEAD_SMOKE_SUITE:-1}" == "1" ]]; then
        # Single-threaded by design; runs once whatever the thread-count list.
        "$SCRIPT_DIR/run_overhead_smoke.sh"
    fi

    if [[ "${RUN_PERMUTATION_SUITE:-1}" == "1" ]]; then
        echo "Running cpu/permutation suite for threads: ${THREAD_COUNTS[*]}..."
        "$SCRIPT_DIR/run_permutation.sh" "${THREAD_COUNTS[@]}"
        echo ""
    fi

    if [[ "${RUN_ELEMENTWISE_SUITE:-0}" == "1" ]]; then
        echo "Running cpu/elementwise suite for threads: ${THREAD_COUNTS[*]}..."
        "$SCRIPT_DIR/run_elementwise.sh" "${THREAD_COUNTS[@]}"
        echo ""
    fi

    echo "============================================"
    echo " Benchmark complete"
    echo "============================================"
    if [[ -n "${BENCHMARK_TARGET_PROFILE:-}" ]]; then
        multi_target_profile="$BENCHMARK_TARGET_PROFILE"
    elif [[ "$(uname -s)" == "Darwin" ]]; then
        multi_target_profile="mac-cpu"
    else
        multi_target_profile="amd-cpu"
    fi
    echo "Latest reports:"
    echo "  $PROJECT_DIR/result/$multi_target_profile/cpu/einsum.md"
    echo "  $PROJECT_DIR/result/$multi_target_profile/cpu/cpu_ops.md"
    echo "  $PROJECT_DIR/result/$multi_target_profile/cpu/linalg_jvp_vjp.md"
    echo "  $PROJECT_DIR/result/$multi_target_profile/cpu/fft.md"
    echo "  $PROJECT_DIR/result/$multi_target_profile/cpu/public_api.md"
    echo "  $PROJECT_DIR/result/$multi_target_profile/cpu/perf_issues.md"
    echo "  $PROJECT_DIR/result/$multi_target_profile/cpu/permutation.md"
    exit 0
fi

# shellcheck source=scripts/cpu_blas_provider.sh
source "$SCRIPT_DIR/cpu_blas_provider.sh"

TENFERRO_CPU_FEATURES="$(normalize_cpu_blas_features "${TENFERRO_CPU_FEATURES:-}")"
export TENFERRO_CPU_FEATURES

RESULTS_ROOT="$PROJECT_DIR/data/results"
REPORTS_DIR="$PROJECT_DIR/result"
CPU_SUITE_ID="cpu/einsum"
CPU_SUITE_FILE="$PROJECT_DIR/benchmarks/cpu/einsum.yaml"

default_target_profile() {
    case "$(benchmark_host_os)" in
        Darwin) printf '%s\n' "mac-cpu" ;;
        *) printf '%s\n' "amd-cpu" ;;
    esac
}

BENCHMARK_TARGET_PROFILE="${BENCHMARK_TARGET_PROFILE:-$(default_target_profile)}"
case "$BENCHMARK_TARGET_PROFILE" in
    mac-cpu|amd-cpu|zen5-cpu|zen2-cpu|nvidia-gpu)
        export BENCHMARK_TARGET_PROFILE
        ;;
    *)
        echo "ERROR: BENCHMARK_TARGET_PROFILE must be mac-cpu, amd-cpu, zen5-cpu, zen2-cpu, or nvidia-gpu." >&2
        exit 1
        ;;
esac

# shellcheck source=scripts/thread_env.sh
source "$SCRIPT_DIR/thread_env.sh"

if [[ "${SKIP_EXTERN_SETUP:-0}" != "1" ]]; then
    # Source this so TENFERRO_RS_DIR and any explicitly configured BLAS paths
    # are visible to the benchmark subprocesses below.
    # shellcheck source=scripts/setup_extern_deps.sh
    source "$SCRIPT_DIR/setup_extern_deps.sh"
fi

HAVE_JULIA=0
if command -v julia >/dev/null 2>&1; then
    HAVE_JULIA=1
    echo "Instantiating Julia project (OMEinsum.jl)..."
    (cd "$PROJECT_DIR" && julia --project="$PROJECT_DIR" -e 'import Pkg; Pkg.instantiate()')
else
    echo "WARNING: julia not found on PATH; skipping omeinsum-jl column." >&2
fi

resolve_git_commit() {
    local checkout_dir="$1"
    if [[ -d "$checkout_dir/.git" ]] && command -v git >/dev/null 2>&1; then
        git -C "$checkout_dir" rev-parse HEAD 2>/dev/null || true
    fi
}

write_cpu_info_section() {
    if command -v uv >/dev/null 2>&1; then
        uv run python "$PROJECT_DIR/scripts/collect_cpu_info.py" --markdown \
            || python3 "$PROJECT_DIR/scripts/collect_cpu_info.py" --markdown
    else
        python3 "$PROJECT_DIR/scripts/collect_cpu_info.py" --markdown
    fi
}

write_thread_env_section() {
    echo "## Thread Environment"
    echo ""
    for key in \
        OMP_NUM_THREADS \
        OMP_THREAD_LIMIT \
        OMP_DYNAMIC \
        RAYON_NUM_THREADS \
        OPENBLAS_NUM_THREADS \
        GOTO_NUM_THREADS \
        MKL_NUM_THREADS \
        VECLIB_MAXIMUM_THREADS \
        VECLIB_NUM_THREADS \
        NUMEXPR_NUM_THREADS \
        BLIS_NUM_THREADS \
        XLA_FLAGS; do
        echo "- ${key}: \`${!key:-}\`"
    done
}

run_python_script() {
    local script="$1"
    shift
    if command -v uv >/dev/null 2>&1; then
        uv run python "$script" "$@" || python3 "$script" "$@"
    else
        python3 "$script" "$@"
    fi
}

resolve_suite_instance_ids() {
    local suite_file="$1"
    run_python_script "$PROJECT_DIR/scripts/suite_instances.py" --suite-file "$suite_file"
}

write_run_metadata() {
    local run_yaml="$1"
    local timestamp="$2"
    local tenferro_dir="${TENFERRO_RS_DIR:-$PROJECT_DIR/extern/tenferro-rs}"
    local blas_impl
    blas_impl="$(blas_impl_for_features "$TENFERRO_CPU_FEATURES")"
    local args=(
        --suite-id "$CPU_SUITE_ID"
        --target-profile "$BENCHMARK_TARGET_PROFILE"
        --suite-file "${CPU_SUITE_FILE#$PROJECT_DIR/}"
        --timestamp "$timestamp"
        --tenferro-dir "$tenferro_dir"
        --features "$TENFERRO_CPU_FEATURES"
        --blas "$blas_impl"
        --output "$run_yaml"
    )
    if [[ -n "$TENFERRO_COMMIT" ]]; then
        args+=(--tenferro-commit "$TENFERRO_COMMIT")
    fi
    run_python_script "$PROJECT_DIR/scripts/collect_run_metadata.py" "${args[@]}"
}

TENFERRO_DIR_FOR_COMMIT="${TENFERRO_RS_DIR:-$PROJECT_DIR/extern/tenferro-rs}"
TENFERRO_COMMIT="$(resolve_git_commit "$TENFERRO_DIR_FOR_COMMIT")"

export BENCHMARK_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_TIMESTAMP_RFC3339="$(python3 - <<'PY'
from datetime import datetime, timezone
print(datetime.now(timezone.utc).isoformat())
PY
)"
CPU_RUN_DIR="$RESULTS_ROOT/$BENCHMARK_TARGET_PROFILE/cpu/einsum/$BENCHMARK_TIMESTAMP"
CPU_RUN_YAML="$CPU_RUN_DIR/run.yaml"
CPU_LATEST_REPORT="$REPORTS_DIR/$BENCHMARK_TARGET_PROFILE/cpu/einsum.md"
CPU_OPS_LATEST_REPORT="$REPORTS_DIR/$BENCHMARK_TARGET_PROFILE/cpu/cpu_ops.md"
CPU_LINALG_AD_LATEST_REPORT="$REPORTS_DIR/$BENCHMARK_TARGET_PROFILE/cpu/linalg_jvp_vjp.md"
CPU_FFT_LATEST_REPORT="$REPORTS_DIR/$BENCHMARK_TARGET_PROFILE/cpu/fft.md"
CPU_PUBLIC_API_LATEST_REPORT="$REPORTS_DIR/$BENCHMARK_TARGET_PROFILE/cpu/public_api.md"
CPU_PERMUTATION_LATEST_REPORT="$REPORTS_DIR/$BENCHMARK_TARGET_PROFILE/cpu/permutation.md"

mkdir -p "$CPU_RUN_DIR" "$(dirname "$CPU_LATEST_REPORT")"
export BENCHMARK_RESULTS_DIR="$CPU_RUN_DIR"

ensure_blas_env_for_features "$TENFERRO_CPU_FEATURES"
configure_cpu_thread_env "$NUM_THREADS"

# Host and thread-environment snapshots for this run's report header.
write_cpu_info_section > "$CPU_RUN_DIR/cpu_info.md"
write_thread_env_section > "$CPU_RUN_DIR/thread_env.txt"

export BENCH_SUITE_FILE="${BENCH_SUITE_FILE:-${CPU_SUITE_FILE#$PROJECT_DIR/}}"
BENCH_SUITE_INCLUDE="$(resolve_suite_instance_ids "$CPU_SUITE_FILE")"
export BENCH_SUITE_INCLUDE
SUITE_INSTANCE_COUNT="$(printf '%s' "$BENCH_SUITE_INCLUDE" | awk -F, '{print NF}')"

echo "============================================"
echo " CPU einsum benchmark suite"
echo "============================================"
echo "Project dir:  $PROJECT_DIR"
echo "Threads:      $NUM_THREADS"
echo "Timestamp:    $BENCHMARK_TIMESTAMP"
echo "Suite:        $CPU_SUITE_ID"
echo "Suite file:   ${CPU_SUITE_FILE#$PROJECT_DIR/}"
echo "Instances:    $SUITE_INSTANCE_COUNT from ${CPU_SUITE_FILE#$PROJECT_DIR/}"
echo "Target:       $BENCHMARK_TARGET_PROFILE"
echo "Run dir:      $CPU_RUN_DIR"
echo "Features:     $TENFERRO_CPU_FEATURES"
[[ -n "$TENFERRO_COMMIT" ]] && echo "tenferro-rs:  $TENFERRO_COMMIT"
print_cpu_thread_env
echo ""

write_run_metadata "$CPU_RUN_YAML" "$RUN_TIMESTAMP_RFC3339"

# ---------------------------------------------------------------------------
# Einsum benchmarks (tenferro-rs + PyTorch + JAX)
# ---------------------------------------------------------------------------
"$SCRIPT_DIR/run_all_rust.sh" "$NUM_THREADS"

# ---------------------------------------------------------------------------
# Python benchmarks (PyTorch + JAX)
# ---------------------------------------------------------------------------
if ! "$SCRIPT_DIR/run_all_python.sh" "$NUM_THREADS"; then
    echo "WARNING: Python benchmarks failed or are not configured; continuing with missing Python results." >&2
fi

# ---------------------------------------------------------------------------
# OMEinsum.jl (Julia) -- fresh process per invocation, forced through each
# instance's precomputed contraction path (mode omeinsum_path); see
# docs/einsum-suite.md. configure_cpu_thread_env above already exported
# JULIA_NUM_THREADS, which the runner also uses to pin
# LinearAlgebra.BLAS.set_num_threads for parity with the other backends.
# ---------------------------------------------------------------------------
JULIA_LOG="$CPU_RUN_DIR/julia_omeinsum_t${NUM_THREADS}_${BENCHMARK_TIMESTAMP}.log"
if [[ "$HAVE_JULIA" == "1" ]]; then
    echo ""
    echo "Running OMEinsum.jl einsum benchmark (threads=$NUM_THREADS)..."
    if ! (
        cd "$PROJECT_DIR"
        JULIA_PROJECT="$PROJECT_DIR" \
            julia --project="$PROJECT_DIR" "$SCRIPT_DIR/benchmark_einsum_omeinsum.jl" | tee "$JULIA_LOG"
    ); then
        echo "WARNING: OMEinsum.jl benchmark failed; continuing without julia results." >&2
    fi
fi

# ---------------------------------------------------------------------------
# Focused CPU benchmark items (primal linalg, JVP/VJP on trace, backward on eager)
# ---------------------------------------------------------------------------
CPU_OPS_LOG="$CPU_RUN_DIR/cpu_ops_t${NUM_THREADS}_${BENCHMARK_TIMESTAMP}.csv"
export PUBLICATION_GATE_TENFERRO_MODE="${PUBLICATION_GATE_TENFERRO_MODE:-both}"
if [[ "${SKIP_CPU_OPS:-0}" != "1" ]]; then
    echo ""
    echo "Running CPU ops benchmarks (linalg JVP/VJP on tenferro-trace)..."
    if [[ -x "$SCRIPT_DIR/run_cpu_ops.sh" ]]; then
        if ! "$SCRIPT_DIR/run_cpu_ops.sh" "$NUM_THREADS"; then
            echo "WARNING: CPU benchmark items failed; continuing without CPU item results." >&2
        fi
    elif [[ -f "$SCRIPT_DIR/run_cpu_ops.sh" ]]; then
        if ! bash "$SCRIPT_DIR/run_cpu_ops.sh" "$NUM_THREADS"; then
            echo "WARNING: CPU benchmark items failed; continuing without CPU item results." >&2
        fi
    fi
else
    echo "Skipping CPU ops benchmarks (SKIP_CPU_OPS=1)."
fi

# CPU benchmark entrypoints may reset .venv and replace Python wheels before
# measuring. Refresh metadata after those setup hooks so provider detection
# matches the Python backends that were actually timed.
if [[ "$(benchmark_host_os)" == "Linux" && "$BENCHMARK_TARGET_PROFILE" != "nvidia-gpu" && -d "$PROJECT_DIR/.venv" ]]; then
    export UV_NO_SYNC=1
fi
write_run_metadata "$CPU_RUN_YAML" "$RUN_TIMESTAMP_RFC3339"

# ---------------------------------------------------------------------------
# Collect all logs, then render this run's reports from its recorded inputs
# ---------------------------------------------------------------------------
TENFERRO_TRACE_LOG="$CPU_RUN_DIR/tenferro_trace_t${NUM_THREADS}_${BENCHMARK_TIMESTAMP}.log"
TENFERRO_EAGER_LOG="$CPU_RUN_DIR/tenferro_eager_t${NUM_THREADS}_${BENCHMARK_TIMESTAMP}.log"
PYTORCH_LOG="$CPU_RUN_DIR/pytorch_cpu_t${NUM_THREADS}_${BENCHMARK_TIMESTAMP}.log"
JAX_LOG="$CPU_RUN_DIR/jax_cpu_t${NUM_THREADS}_${BENCHMARK_TIMESTAMP}.log"
# JULIA_LOG is already set above (before the Julia run step); reused here.
EINSUM_REPORT="$CPU_RUN_DIR/report.md"
CPU_REPORT="$CPU_RUN_DIR/cpu_ops_report.md"
LINALG_AD_REPORT="$CPU_RUN_DIR/linalg_jvp_vjp_report.md"

LOGS=()
[ -f "$TENFERRO_TRACE_LOG" ] && LOGS+=("$TENFERRO_TRACE_LOG")
[ -f "$TENFERRO_EAGER_LOG" ] && LOGS+=("$TENFERRO_EAGER_LOG")
[ -f "$PYTORCH_LOG" ]        && LOGS+=("$PYTORCH_LOG")
[ -f "$JAX_LOG" ]            && LOGS+=("$JAX_LOG")
[ "$HAVE_JULIA" == "1" ] && [ -f "$JULIA_LOG" ] && [ -s "$JULIA_LOG" ] && LOGS+=("$JULIA_LOG")

echo "Finalizing and publishing this run from its recorded inputs..."
finalize_benchmark_run "$PROJECT_DIR" "$CPU_RUN_DIR"
publish_cpu_runs "$PROJECT_DIR" "$CPU_RUN_DIR"
echo ""

# Opt-in: cpu/fft suite, run sequentially after CPU ops so FFT timing does not
# overlap other CPU benchmark processes.
if [[ "${RUN_ALL_MAIN_ONLY:-0}" != "1" && "${RUN_FFT_SUITE:-0}" == "1" ]]; then
    echo "Running cpu/fft suite for 1 and 4 threads (RUN_FFT_SUITE=1)..."
    SKIP_EXTERN_SETUP=1 "$SCRIPT_DIR/run_cpu_fft.sh" 1 4
    echo ""
fi

# Opt-in: cpu/public_api suite, run sequentially after CPU FFT if enabled.
# SKIP_EXTERN_SETUP=1 avoids re-running setup_extern_deps.sh, already done above.
if [[ "${RUN_ALL_MAIN_ONLY:-0}" != "1" && "${RUN_PUBLIC_API_SUITE:-0}" == "1" ]]; then
    echo "Running cpu/public_api suite for 1 and 4 threads (RUN_PUBLIC_API_SUITE=1)..."
    public_api_profile="${RUN_PUBLIC_API_PROFILE:-${PUBLICATION_GATE_PROFILE:-quick}}"
    SKIP_EXTERN_SETUP=1 PUBLICATION_GATE_PROFILE="$public_api_profile" \
        "$SCRIPT_DIR/run_cpu_public_api.sh" 1 4
    echo ""
fi

if [[ "${RUN_ALL_MAIN_ONLY:-0}" != "1" && "${RUN_SMALL_WORK_SUITE:-1}" == "1" ]]; then
    SKIP_EXTERN_SETUP=1 "$SCRIPT_DIR/run_small_work.sh" "$NUM_THREADS"
    "$SCRIPT_DIR/run_cpu_session.sh" "$NUM_THREADS"
fi

if [[ "${RUN_ALL_MAIN_ONLY:-0}" != "1" && "${RUN_PERF_ISSUES_SUITE:-1}" == "1" ]]; then
    SKIP_EXTERN_SETUP=1 "$SCRIPT_DIR/run_perf_issues.sh" "$NUM_THREADS"
fi

if [[ "${RUN_ALL_MAIN_ONLY:-0}" != "1" && "${RUN_OVERHEAD_SMOKE_SUITE:-1}" == "1" ]]; then
    SKIP_EXTERN_SETUP=1 "$SCRIPT_DIR/run_overhead_smoke.sh"
fi

# Opt-in: cpu/permutation suite, run sequentially after every suite above.
# SKIP_EXTERN_SETUP=1 avoids re-running setup_extern_deps.sh, already done above.
if [[ "${RUN_ALL_MAIN_ONLY:-0}" != "1" && "${RUN_PERMUTATION_SUITE:-0}" == "1" ]]; then
    echo "Running cpu/permutation suite (RUN_PERMUTATION_SUITE=1)..."
    SKIP_EXTERN_SETUP=1 "$SCRIPT_DIR/run_permutation.sh" "$NUM_THREADS"
    echo ""
fi

# Opt-in: cpu/elementwise suite (CPU buffer-pool allocation/first-touch signal),
# run sequentially after every suite above. Off by default.
if [[ "${RUN_ALL_MAIN_ONLY:-0}" != "1" && "${RUN_ELEMENTWISE_SUITE:-0}" == "1" ]]; then
    echo "Running cpu/elementwise suite (RUN_ELEMENTWISE_SUITE=1)..."
    SKIP_EXTERN_SETUP=1 "$SCRIPT_DIR/run_elementwise.sh" "$NUM_THREADS"
    echo ""
fi

echo "============================================"
echo " Benchmark complete"
echo "============================================"
echo "Results:"
for log in "${LOGS[@]}"; do
    echo "  $log"
done
[ -f "$CPU_RUN_YAML" ] && echo "  Run YAML: $CPU_RUN_YAML"
[ -f "$EINSUM_REPORT" ] && echo "  Report:   $EINSUM_REPORT"
[ -f "$CPU_LATEST_REPORT" ] && echo "  Latest:   $CPU_LATEST_REPORT"
[ -f "$CPU_REPORT" ] && echo "  Report:   $CPU_REPORT"
[ -f "$CPU_OPS_LATEST_REPORT" ] && echo "  Latest:   $CPU_OPS_LATEST_REPORT"
[ -f "$LINALG_AD_REPORT" ] && echo "  Report:   $LINALG_AD_REPORT"
[ -f "$CPU_LINALG_AD_LATEST_REPORT" ] && echo "  Latest:   $CPU_LINALG_AD_LATEST_REPORT"
[ -f "$CPU_FFT_LATEST_REPORT" ] && echo "  Latest:   $CPU_FFT_LATEST_REPORT"
[ -f "$CPU_PUBLIC_API_LATEST_REPORT" ] && echo "  Latest:   $CPU_PUBLIC_API_LATEST_REPORT"
[ -f "$CPU_PERMUTATION_LATEST_REPORT" ] && echo "  Latest:   $CPU_PERMUTATION_LATEST_REPORT"
if [[ "${RUN_ALL_MAIN_ONLY:-0}" == "1" && -n "${RUN_ALL_COLLECTED_RUNS_FILE:-}" ]]; then
    printf '%s\n' "$CPU_RUN_DIR" >> "$RUN_ALL_COLLECTED_RUNS_FILE"
fi
true
