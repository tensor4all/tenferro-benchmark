#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Full benchmark runner:
#   - tenferro trace/eager (Rust)
#   - strided-rs / faer (Rust, optional)
#   - LibTorch CPU (C++)
#   - PyTorch/JAX CPU (Python)
#   - PR884 CPU benchmark items where supported by repo-local runners
#
# Usage: ./scripts/run_all.sh [NUM_THREADS]
#
# Delegates to run_all_rust.sh, run_all_libtorch.sh, and run_all_python.sh,
# then formats results.
# ---------------------------------------------------------------------------

NUM_THREADS="${1:-1}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="$PROJECT_DIR/data/results"
REPORTS_DIR="$PROJECT_DIR/result"

if [[ "${SKIP_EXTERN_SETUP:-0}" != "1" ]]; then
    # Source this so OPENBLAS_ROOT and Torch_DIR exported by the setup script
    # are visible to the benchmark subprocesses below.
    # shellcheck source=scripts/setup_extern_deps.sh
    source "$SCRIPT_DIR/setup_extern_deps.sh"
fi

# If Torch_DIR is still unset (e.g. SKIP_EXTERN_SETUP=1), probe well-known
# locations where an already-built OpenBLAS-linked LibTorch might live.
if [[ -z "${Torch_DIR:-}" ]]; then
    for _torch_candidate in \
        "${PYTORCH_OPENBLAS_DIR:+${PYTORCH_OPENBLAS_DIR}/torch/share/cmake/Torch}" \
        "$PROJECT_DIR/extern/devcontainer/pytorch-openblas/torch/share/cmake/Torch" \
        "$PROJECT_DIR/extern/pytorch-openblas/torch/share/cmake/Torch"; do
        [[ -n "$_torch_candidate" && -f "$_torch_candidate/TorchConfig.cmake" ]] || continue
        export Torch_DIR="$_torch_candidate"
        echo "[run_all.sh] Auto-detected Torch_DIR=$Torch_DIR"
        break
    done
fi

resolve_git_commit() {
    local checkout_dir="$1"
    if [[ -d "$checkout_dir/.git" ]] && command -v git >/dev/null 2>&1; then
        git -C "$checkout_dir" rev-parse HEAD 2>/dev/null || true
    fi
}

result_threads_for_prefix() {
    local prefix="$1"
    find "$RESULTS_DIR" -maxdepth 1 -name "${prefix}_t*_*.md" -print 2>/dev/null \
        | while IFS= read -r path; do
            local base
            base="$(basename "$path")"
            if [[ "$base" =~ ^${prefix}_t([0-9]+)_[0-9]{8}_[0-9]{6}\.md$ ]]; then
                echo "${BASH_REMATCH[1]}"
            fi
        done \
        | sort -n -u
}

latest_thread_markdown() {
    local prefix="$1"
    local threads="$2"
    find "$RESULTS_DIR" -maxdepth 1 -name "${prefix}_t${threads}_*.md" -print 2>/dev/null \
        | sort \
        | tail -n 1
}

timestamp_from_markdown() {
    local prefix="$1"
    local path="$2"
    local base
    base="$(basename "$path")"
    if [[ "$base" =~ ^${prefix}_t[0-9]+_([0-9]{8}_[0-9]{6})\.md$ ]]; then
        echo "${BASH_REMATCH[1]}"
    fi
}

write_einsum_report() {
    local report="$1"
    mkdir -p "$REPORTS_DIR"
    {
        echo "# Einsum Benchmark Results"
        echo ""
        echo "Latest run: \`./scripts/run_all.sh $NUM_THREADS\`."
        echo ""
        echo "This file collects the latest einsum benchmark table for each thread count under \`data/results/\`."
        echo ""
        if [[ -n "$TENFERRO_COMMIT" ]]; then
            echo "- tenferro-rs commit: \`$TENFERRO_COMMIT\`"
            echo ""
        fi

        local thread table timestamp log section_count=0
        for thread in $(result_threads_for_prefix "results"); do
            table="$(latest_thread_markdown "results" "$thread")"
            [[ -n "$table" ]] || continue
            timestamp="$(timestamp_from_markdown "results" "$table")"

            if (( section_count > 0 )); then
                echo ""
            fi
            ((section_count += 1))
            echo "## Threads: $thread"
            echo ""
            [[ -n "$timestamp" ]] && echo "- Timestamp: \`$timestamp\`"
            echo "- Source table: \`${table#$PROJECT_DIR/}\`"
            echo ""
            echo "Logs:"
            echo ""
            for log in \
                "$RESULTS_DIR/tenferro_trace_t${thread}_${timestamp}.log" \
                "$RESULTS_DIR/tenferro_eager_t${thread}_${timestamp}.log" \
                "$RESULTS_DIR/strided_faer_t${thread}_${timestamp}.log" \
                "$RESULTS_DIR/libtorch_cpu_t${thread}_${timestamp}.log" \
                "$RESULTS_DIR/pytorch_cpu_t${thread}_${timestamp}.log" \
                "$RESULTS_DIR/jax_cpu_t${thread}_${timestamp}.log"; do
                [[ -f "$log" ]] && echo "- \`${log#$PROJECT_DIR/}\`"
            done
            echo ""
            cat "$table"
        done
    } > "$report"
}

write_cpu_report() {
    local report="$1"
    mkdir -p "$REPORTS_DIR"
    {
        echo "# CPU Benchmark Results"
        echo ""
        echo "Latest run: \`./scripts/run_all.sh $NUM_THREADS\`."
        echo ""
        echo "This file collects the latest PR884 CPU benchmark item comparison for each thread count under \`data/results/\`."
        echo ""
        if [[ -n "$TENFERRO_COMMIT" ]]; then
            echo "- tenferro-rs commit: \`$TENFERRO_COMMIT\`"
            echo ""
        fi

        local thread table timestamp log section_count=0
        for thread in $(result_threads_for_prefix "cpu_ops"); do
            table="$(latest_thread_markdown "cpu_ops" "$thread")"
            [[ -n "$table" ]] || continue
            timestamp="$(timestamp_from_markdown "cpu_ops" "$table")"
            log="$RESULTS_DIR/cpu_ops_t${thread}_${timestamp}.csv"

            if (( section_count > 0 )); then
                echo ""
            fi
            ((section_count += 1))
            echo "## Threads: $thread"
            echo ""
            [[ -n "$timestamp" ]] && echo "- Timestamp: \`$timestamp\`"
            echo "- Source table: \`${table#$PROJECT_DIR/}\`"
            echo ""
            echo "Logs:"
            echo ""
            [[ -f "$log" ]] && echo "- \`${log#$PROJECT_DIR/}\`"
            echo ""
            cat "$table"
        done
    } > "$report"
}

TENFERRO_COMMIT=""
if [[ -n "${TENFERRO_RS_DIR:-}" ]]; then
    TENFERRO_COMMIT="$(resolve_git_commit "$TENFERRO_RS_DIR")"
fi

export BENCHMARK_TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "============================================"
echo " tenferro benchmark suite"
echo "============================================"
echo "Project dir:  $PROJECT_DIR"
echo "Threads:      $NUM_THREADS"
echo "Timestamp:    $BENCHMARK_TIMESTAMP"
[[ -n "$TENFERRO_COMMIT" ]] && echo "tenferro-rs:  $TENFERRO_COMMIT"
echo ""

# ---------------------------------------------------------------------------
# Rust benchmarks (tenferro trace/eager + strided-rs)
# ---------------------------------------------------------------------------
"$SCRIPT_DIR/run_all_rust.sh" "$NUM_THREADS"

# ---------------------------------------------------------------------------
# C++ LibTorch benchmark
# ---------------------------------------------------------------------------
if ! "$SCRIPT_DIR/run_all_libtorch.sh" "$NUM_THREADS"; then
    echo "WARNING: LibTorch benchmark failed or is not configured; continuing with missing Torch C++ results." >&2
fi

# ---------------------------------------------------------------------------
# Python benchmarks (PyTorch + JAX)
# ---------------------------------------------------------------------------
if ! "$SCRIPT_DIR/run_all_python.sh" "$NUM_THREADS"; then
    echo "WARNING: Python benchmarks failed or are not configured; continuing with missing Python results." >&2
fi

# ---------------------------------------------------------------------------
# Focused CPU benchmark items from tensor4all/tenferro-rs#884
# ---------------------------------------------------------------------------
CPU_OPS_LOG="$RESULTS_DIR/cpu_ops_t${NUM_THREADS}_${BENCHMARK_TIMESTAMP}.csv"
CPU_OPS_MD="$RESULTS_DIR/cpu_ops_t${NUM_THREADS}_${BENCHMARK_TIMESTAMP}.md"
if [[ -x "$SCRIPT_DIR/run_cpu_ops.sh" ]]; then
    if ! "$SCRIPT_DIR/run_cpu_ops.sh" "$NUM_THREADS"; then
        echo "WARNING: PR884 CPU benchmark items failed; continuing without CPU item results." >&2
    fi
elif [[ -f "$SCRIPT_DIR/run_cpu_ops.sh" ]]; then
    if ! bash "$SCRIPT_DIR/run_cpu_ops.sh" "$NUM_THREADS"; then
        echo "WARNING: PR884 CPU benchmark items failed; continuing without CPU item results." >&2
    fi
fi

# ---------------------------------------------------------------------------
# Collect all logs and format as markdown table
# ---------------------------------------------------------------------------
TENFERRO_TRACE_LOG="$RESULTS_DIR/tenferro_trace_t${NUM_THREADS}_${BENCHMARK_TIMESTAMP}.log"
TENFERRO_EAGER_LOG="$RESULTS_DIR/tenferro_eager_t${NUM_THREADS}_${BENCHMARK_TIMESTAMP}.log"
STRIDED_FAER_LOG="$RESULTS_DIR/strided_faer_t${NUM_THREADS}_${BENCHMARK_TIMESTAMP}.log"
LIBTORCH_LOG="$RESULTS_DIR/libtorch_cpu_t${NUM_THREADS}_${BENCHMARK_TIMESTAMP}.log"
PYTORCH_LOG="$RESULTS_DIR/pytorch_cpu_t${NUM_THREADS}_${BENCHMARK_TIMESTAMP}.log"
JAX_LOG="$RESULTS_DIR/jax_cpu_t${NUM_THREADS}_${BENCHMARK_TIMESTAMP}.log"
MARKDOWN_OUT="$RESULTS_DIR/results_t${NUM_THREADS}_${BENCHMARK_TIMESTAMP}.md"
EINSUM_REPORT="$REPORTS_DIR/einsum-results.md"
CPU_REPORT="$REPORTS_DIR/cpu-benchmark-results.md"

LOGS=()
[ -f "$TENFERRO_TRACE_LOG" ] && LOGS+=("$TENFERRO_TRACE_LOG")
[ -f "$TENFERRO_EAGER_LOG" ] && LOGS+=("$TENFERRO_EAGER_LOG")
[ -f "$STRIDED_FAER_LOG" ]   && LOGS+=("$STRIDED_FAER_LOG")
[ -f "$LIBTORCH_LOG" ]       && LOGS+=("$LIBTORCH_LOG")
[ -f "$PYTORCH_LOG" ]        && LOGS+=("$PYTORCH_LOG")
[ -f "$JAX_LOG" ]            && LOGS+=("$JAX_LOG")

if [ ${#LOGS[@]} -gt 0 ]; then
    echo "Formatting results as markdown..."
    mkdir -p "$REPORTS_DIR"
    if command -v uv >/dev/null 2>&1; then
        uv run python "$PROJECT_DIR/scripts/format_results.py" "${LOGS[@]}" | tee "$MARKDOWN_OUT" \
            || python3 "$PROJECT_DIR/scripts/format_results.py" "${LOGS[@]}" | tee "$MARKDOWN_OUT"
    else
        python3 "$PROJECT_DIR/scripts/format_results.py" "${LOGS[@]}" | tee "$MARKDOWN_OUT"
    fi

    if [ -f "$CPU_OPS_LOG" ]; then
        if command -v uv >/dev/null 2>&1; then
            uv run python "$PROJECT_DIR/scripts/format_cpu_ops_results.py" "$CPU_OPS_LOG" | tee "$CPU_OPS_MD" \
                || python3 "$PROJECT_DIR/scripts/format_cpu_ops_results.py" "$CPU_OPS_LOG" | tee "$CPU_OPS_MD"
        else
            python3 "$PROJECT_DIR/scripts/format_cpu_ops_results.py" "$CPU_OPS_LOG" | tee "$CPU_OPS_MD"
        fi
    fi

    write_einsum_report "$EINSUM_REPORT"
    write_cpu_report "$CPU_REPORT"
    echo ""
fi

echo "============================================"
echo " Benchmark complete"
echo "============================================"
echo "Results:"
for log in "${LOGS[@]}"; do
    echo "  $log"
done
[ -f "$MARKDOWN_OUT" ] && echo "  Markdown: $MARKDOWN_OUT"
[ -f "$EINSUM_REPORT" ] && echo "  Report:   $EINSUM_REPORT"
[ -f "$CPU_REPORT" ] && echo "  Report:   $CPU_REPORT"
