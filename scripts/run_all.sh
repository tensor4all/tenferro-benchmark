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
RESULTS_ROOT="$PROJECT_DIR/data/results"
REPORTS_DIR="$PROJECT_DIR/result"
CPU_SUITE_ID="cpu/einsum"
CPU_SUITE_FILE="$PROJECT_DIR/benchmarks/cpu/einsum.yaml"

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

write_cpu_info_section() {
    if command -v uv >/dev/null 2>&1; then
        uv run python "$PROJECT_DIR/scripts/collect_cpu_info.py" --markdown \
            || python3 "$PROJECT_DIR/scripts/collect_cpu_info.py" --markdown
    else
        python3 "$PROJECT_DIR/scripts/collect_cpu_info.py" --markdown
    fi
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

write_run_metadata() {
    local run_yaml="$1"
    local timestamp="$2"
    local tenferro_dir="${TENFERRO_RS_DIR:-$PROJECT_DIR/extern/tenferro-rs}"
    local args=(
        --suite-id "$CPU_SUITE_ID"
        --suite-file "${CPU_SUITE_FILE#$PROJECT_DIR/}"
        --timestamp "$timestamp"
        --tenferro-dir "$tenferro_dir"
        --features system-openblas
        --blas openblas
        --output "$run_yaml"
    )
    if [[ -n "$TENFERRO_COMMIT" ]]; then
        args+=(--tenferro-commit "$TENFERRO_COMMIT")
    fi
    run_python_script "$PROJECT_DIR/scripts/collect_run_metadata.py" "${args[@]}"
}

write_einsum_report() {
    local report="$1"
    mkdir -p "$(dirname "$report")"
    {
        echo "# Einsum Benchmark Results"
        echo ""
        echo "- Suite: \`$CPU_SUITE_ID\`"
        echo "- Suite file: \`${CPU_SUITE_FILE#$PROJECT_DIR/}\`"
        echo "- Run metadata: \`${CPU_RUN_YAML#$PROJECT_DIR/}\`"
        echo "- Timestamp: \`$BENCHMARK_TIMESTAMP\`"
        echo ""
        echo "Latest run: \`./scripts/run_all.sh $NUM_THREADS\`."
        echo ""
        echo "This file is generated from one suite run under \`${CPU_RUN_DIR#$PROJECT_DIR/}\`."
        echo ""
        if [[ -n "$TENFERRO_COMMIT" ]]; then
            echo "- tenferro-rs commit: \`$TENFERRO_COMMIT\`"
            echo ""
        fi
        write_cpu_info_section
        echo ""

        echo "## Threads: $NUM_THREADS"
        echo ""
        echo "- Source table: \`${MARKDOWN_TABLE#$PROJECT_DIR/}\`"
        echo ""
        echo "Logs:"
        echo ""
        for log in \
            "$TENFERRO_TRACE_LOG" \
            "$TENFERRO_EAGER_LOG" \
            "$STRIDED_FAER_LOG" \
            "$LIBTORCH_LOG" \
            "$PYTORCH_LOG" \
            "$JAX_LOG"; do
            [[ -f "$log" ]] && echo "- \`${log#$PROJECT_DIR/}\`"
        done
        echo ""
        cat "$MARKDOWN_TABLE"
    } > "$report"
}

write_cpu_report() {
    local report="$1"
    mkdir -p "$(dirname "$report")"
    {
        echo "# CPU Benchmark Results"
        echo ""
        echo "- Suite: \`cpu/cpu_ops\`"
        echo "- Timestamp: \`$BENCHMARK_TIMESTAMP\`"
        echo ""
        echo "Latest run: \`./scripts/run_all.sh $NUM_THREADS\`."
        echo ""
        echo "This file is generated from one CPU ops run under \`${CPU_RUN_DIR#$PROJECT_DIR/}\`."
        echo ""
        if [[ -n "$TENFERRO_COMMIT" ]]; then
            echo "- tenferro-rs commit: \`$TENFERRO_COMMIT\`"
            echo ""
        fi
        write_cpu_info_section
        echo ""
        echo "## Threads: $NUM_THREADS"
        echo ""
        [[ -f "$CPU_OPS_LOG" ]] && echo "- CSV: \`${CPU_OPS_LOG#$PROJECT_DIR/}\`"
        echo "- Source table: \`${CPU_OPS_MD#$PROJECT_DIR/}\`"
        echo ""
        cat "$CPU_OPS_MD"
    } > "$report"
}

TENFERRO_DIR_FOR_COMMIT="${TENFERRO_RS_DIR:-$PROJECT_DIR/extern/tenferro-rs}"
TENFERRO_COMMIT="$(resolve_git_commit "$TENFERRO_DIR_FOR_COMMIT")"

export BENCHMARK_TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_TIMESTAMP_RFC3339="$(python3 - <<'PY'
from datetime import datetime, timezone
print(datetime.now(timezone.utc).isoformat())
PY
)"
CPU_RUN_DIR="$RESULTS_ROOT/cpu/einsum/$BENCHMARK_TIMESTAMP"
CPU_RUN_YAML="$CPU_RUN_DIR/run.yaml"
CPU_LATEST_REPORT="$REPORTS_DIR/cpu/einsum.md"
CPU_OPS_LATEST_REPORT="$REPORTS_DIR/cpu/cpu_ops.md"

mkdir -p "$CPU_RUN_DIR" "$(dirname "$CPU_LATEST_REPORT")"
export BENCHMARK_RESULTS_DIR="$CPU_RUN_DIR"

if [[ -z "${OPENBLAS_ROOT:-}" ]]; then
    if command -v brew >/dev/null 2>&1 && brew --prefix openblas >/dev/null 2>&1; then
        export OPENBLAS_ROOT="$(brew --prefix openblas)"
    fi
fi

echo "============================================"
echo " tenferro benchmark suite"
echo "============================================"
echo "Project dir:  $PROJECT_DIR"
echo "Threads:      $NUM_THREADS"
echo "Timestamp:    $BENCHMARK_TIMESTAMP"
echo "Suite:        $CPU_SUITE_ID"
echo "Run dir:      $CPU_RUN_DIR"
[[ -n "$TENFERRO_COMMIT" ]] && echo "tenferro-rs:  $TENFERRO_COMMIT"
echo ""

write_run_metadata "$CPU_RUN_YAML" "$RUN_TIMESTAMP_RFC3339"

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
CPU_OPS_LOG="$CPU_RUN_DIR/cpu_ops_t${NUM_THREADS}_${BENCHMARK_TIMESTAMP}.csv"
CPU_OPS_MD="$CPU_RUN_DIR/cpu_ops_t${NUM_THREADS}_${BENCHMARK_TIMESTAMP}.md"
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
TENFERRO_TRACE_LOG="$CPU_RUN_DIR/tenferro_trace_t${NUM_THREADS}_${BENCHMARK_TIMESTAMP}.log"
TENFERRO_EAGER_LOG="$CPU_RUN_DIR/tenferro_eager_t${NUM_THREADS}_${BENCHMARK_TIMESTAMP}.log"
STRIDED_FAER_LOG="$CPU_RUN_DIR/strided_faer_t${NUM_THREADS}_${BENCHMARK_TIMESTAMP}.log"
LIBTORCH_LOG="$CPU_RUN_DIR/libtorch_cpu_t${NUM_THREADS}_${BENCHMARK_TIMESTAMP}.log"
PYTORCH_LOG="$CPU_RUN_DIR/pytorch_cpu_t${NUM_THREADS}_${BENCHMARK_TIMESTAMP}.log"
JAX_LOG="$CPU_RUN_DIR/jax_cpu_t${NUM_THREADS}_${BENCHMARK_TIMESTAMP}.log"
MARKDOWN_TABLE="$CPU_RUN_DIR/einsum_table_t${NUM_THREADS}_${BENCHMARK_TIMESTAMP}.md"
EINSUM_REPORT="$CPU_RUN_DIR/report.md"
CPU_REPORT="$CPU_RUN_DIR/cpu_ops_report.md"

LOGS=()
[ -f "$TENFERRO_TRACE_LOG" ] && LOGS+=("$TENFERRO_TRACE_LOG")
[ -f "$TENFERRO_EAGER_LOG" ] && LOGS+=("$TENFERRO_EAGER_LOG")
[ -f "$STRIDED_FAER_LOG" ]   && LOGS+=("$STRIDED_FAER_LOG")
[ -f "$LIBTORCH_LOG" ]       && LOGS+=("$LIBTORCH_LOG")
[ -f "$PYTORCH_LOG" ]        && LOGS+=("$PYTORCH_LOG")
[ -f "$JAX_LOG" ]            && LOGS+=("$JAX_LOG")

if [ ${#LOGS[@]} -gt 0 ]; then
    echo "Formatting results as markdown..."
    mkdir -p "$CPU_RUN_DIR" "$(dirname "$CPU_LATEST_REPORT")" "$(dirname "$CPU_OPS_LATEST_REPORT")"
    if command -v uv >/dev/null 2>&1; then
        uv run python "$PROJECT_DIR/scripts/format_results.py" "${LOGS[@]}" | tee "$MARKDOWN_TABLE" \
            || python3 "$PROJECT_DIR/scripts/format_results.py" "${LOGS[@]}" | tee "$MARKDOWN_TABLE"
    else
        python3 "$PROJECT_DIR/scripts/format_results.py" "${LOGS[@]}" | tee "$MARKDOWN_TABLE"
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
    cp "$EINSUM_REPORT" "$CPU_LATEST_REPORT"
    if [ -f "$CPU_OPS_MD" ]; then
        write_cpu_report "$CPU_REPORT"
        cp "$CPU_REPORT" "$CPU_OPS_LATEST_REPORT"
    fi
    echo ""
fi

echo "============================================"
echo " Benchmark complete"
echo "============================================"
echo "Results:"
for log in "${LOGS[@]}"; do
    echo "  $log"
done
[ -f "$MARKDOWN_TABLE" ] && echo "  Markdown: $MARKDOWN_TABLE"
[ -f "$CPU_RUN_YAML" ] && echo "  Run YAML: $CPU_RUN_YAML"
[ -f "$EINSUM_REPORT" ] && echo "  Report:   $EINSUM_REPORT"
[ -f "$CPU_LATEST_REPORT" ] && echo "  Latest:   $CPU_LATEST_REPORT"
[ -f "$CPU_REPORT" ] && echo "  Report:   $CPU_REPORT"
[ -f "$CPU_OPS_LATEST_REPORT" ] && echo "  Latest:   $CPU_OPS_LATEST_REPORT"
