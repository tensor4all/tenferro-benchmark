#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Full benchmark runner:
#   - tenferro trace/eager (Rust)
#   - strided-rs / faer (Rust, optional)
#   - PyTorch/JAX CPU (Python)
#   - CPU benchmark items where supported by repo-local runners
#
# Usage: ./scripts/run_all.sh [NUM_THREADS]
#
# Delegates to run_all_rust.sh and run_all_python.sh, then formats results.
# ---------------------------------------------------------------------------

NUM_THREADS="${1:-1}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# shellcheck source=scripts/cpu_blas_provider.sh
source "$SCRIPT_DIR/cpu_blas_provider.sh"

TENFERRO_CPU_FEATURES="$(normalize_cpu_blas_features "${TENFERRO_CPU_FEATURES:-}")"
export TENFERRO_CPU_FEATURES
case "${TENFERRO_CPU_BACKEND_KIND:-}" in
    "")
        case "$TENFERRO_CPU_FEATURES" in
            system-openblas|system-accelerate)
                export TENFERRO_CPU_BACKEND_KIND=blas
                ;;
            *)
                export TENFERRO_CPU_BACKEND_KIND=default
                ;;
        esac
        ;;
    default|faer|blas)
        export TENFERRO_CPU_BACKEND_KIND
        ;;
    *)
        echo "ERROR: TENFERRO_CPU_BACKEND_KIND must be default, faer, or blas." >&2
        exit 1
        ;;
esac

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
    mac-cpu|amd-cpu|nvidia-gpu)
        export BENCHMARK_TARGET_PROFILE
        ;;
    *)
        echo "ERROR: BENCHMARK_TARGET_PROFILE must be mac-cpu, amd-cpu, or nvidia-gpu." >&2
        exit 1
        ;;
esac

# shellcheck source=scripts/thread_env.sh
source "$SCRIPT_DIR/thread_env.sh"

if [[ "${SKIP_EXTERN_SETUP:-0}" != "1" ]]; then
    # Source this so TENFERRO_RS_DIR and any explicitly configured OpenBLAS
    # paths are visible to the benchmark subprocesses below.
    # shellcheck source=scripts/setup_extern_deps.sh
    source "$SCRIPT_DIR/setup_extern_deps.sh"
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

write_python_backend_section() {
    local run_yaml="$1"
    echo "## Python Backend Providers"
    echo ""
    run_python_script - "$run_yaml" <<'PY'
import sys
from pathlib import Path

try:
    import yaml
except Exception as exc:  # noqa: BLE001
    print(f"- unavailable: PyYAML is not available ({exc})")
    raise SystemExit(0)

run_path = Path(sys.argv[1])
try:
    run = yaml.safe_load(run_path.read_text())
except Exception as exc:  # noqa: BLE001
    print(f"- unavailable: failed to read run metadata ({exc})")
    raise SystemExit(0)

backends = (run or {}).get("python_backends") or {}
if not backends:
    print("- unavailable: provider metadata was not collected")
    raise SystemExit(0)

labels = {"pytorch": "PyTorch", "jax": "JAX"}
for key in ("pytorch", "jax"):
    info = backends.get(key) or {}
    label = labels[key]
    if not info.get("available"):
        reason = info.get("reason") or "unavailable"
        print(f"- {label}: unavailable ({reason})")
        continue
    provider = info.get("provider") or "unknown"
    version = info.get("version") or "unknown"
    if key == "jax":
        dot_backend = info.get("dot_backend") or provider
        details = [f"dot backend `{dot_backend}`", f"version `{version}`"]
        if info.get("jaxlib_version"):
            details.append(f"jaxlib `{info['jaxlib_version']}`")
        if info.get("backend"):
            details.append(f"default backend `{info['backend']}`")
        if info.get("lapack_provider"):
            details.append(f"LAPACK provider `{info['lapack_provider']}`")
    else:
        details = [f"BLAS provider `{provider}`", f"version `{version}`"]
        if info.get("blas_info"):
            details.append(f"BLAS_INFO `{info['blas_info']}`")
        if info.get("lapack_info"):
            details.append(f"LAPACK_INFO `{info['lapack_info']}`")
        if info.get("backend"):
            details.append(f"backend `{info['backend']}`")
    print(f"- {label}: " + ", ".join(details))
    deps = info.get("linked_libraries") or []
    if deps:
        dep_label = "linked LAPACK libs" if key == "jax" else "linked BLAS/LAPACK libs"
        print(f"  - {dep_label}: " + "; ".join(f"`{dep}`" for dep in deps))
PY
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

write_einsum_report() {
    local report="$1"
    mkdir -p "$(dirname "$report")"
    {
        echo "# Einsum Benchmark Results"
        echo ""
        echo "- Suite: \`$CPU_SUITE_ID\`"
        echo "- Target profile: \`$BENCHMARK_TARGET_PROFILE\`"
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
        write_thread_env_section
        echo ""
        write_python_backend_section "$CPU_RUN_YAML"
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
        echo "- Target profile: \`$BENCHMARK_TARGET_PROFILE\`"
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
        write_thread_env_section
        echo ""
        write_python_backend_section "$CPU_RUN_YAML"
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
CPU_RUN_DIR="$RESULTS_ROOT/$BENCHMARK_TARGET_PROFILE/cpu/einsum/$BENCHMARK_TIMESTAMP"
CPU_RUN_YAML="$CPU_RUN_DIR/run.yaml"
CPU_LATEST_REPORT="$REPORTS_DIR/$BENCHMARK_TARGET_PROFILE/cpu/einsum.md"
CPU_OPS_LATEST_REPORT="$REPORTS_DIR/$BENCHMARK_TARGET_PROFILE/cpu/cpu_ops.md"

mkdir -p "$CPU_RUN_DIR" "$(dirname "$CPU_LATEST_REPORT")"
export BENCHMARK_RESULTS_DIR="$CPU_RUN_DIR"

ensure_blas_env_for_features "$TENFERRO_CPU_FEATURES"
configure_cpu_thread_env "$NUM_THREADS"

echo "============================================"
echo " tenferro benchmark suite"
echo "============================================"
echo "Project dir:  $PROJECT_DIR"
echo "Threads:      $NUM_THREADS"
echo "Timestamp:    $BENCHMARK_TIMESTAMP"
echo "Suite:        $CPU_SUITE_ID"
echo "Target:       $BENCHMARK_TARGET_PROFILE"
echo "Run dir:      $CPU_RUN_DIR"
echo "Features:     $TENFERRO_CPU_FEATURES"
echo "CPU backend:  $TENFERRO_CPU_BACKEND_KIND"
[[ -n "$TENFERRO_COMMIT" ]] && echo "tenferro-rs:  $TENFERRO_COMMIT"
print_cpu_thread_env
echo ""

write_run_metadata "$CPU_RUN_YAML" "$RUN_TIMESTAMP_RFC3339"

# ---------------------------------------------------------------------------
# Rust benchmarks (tenferro trace/eager + strided-rs)
# ---------------------------------------------------------------------------
"$SCRIPT_DIR/run_all_rust.sh" "$NUM_THREADS"

# ---------------------------------------------------------------------------
# Python benchmarks (PyTorch + JAX)
# ---------------------------------------------------------------------------
if ! "$SCRIPT_DIR/run_all_python.sh" "$NUM_THREADS"; then
    echo "WARNING: Python benchmarks failed or are not configured; continuing with missing Python results." >&2
fi

# ---------------------------------------------------------------------------
# Focused CPU benchmark items
# ---------------------------------------------------------------------------
CPU_OPS_LOG="$CPU_RUN_DIR/cpu_ops_t${NUM_THREADS}_${BENCHMARK_TIMESTAMP}.csv"
CPU_OPS_MD="$CPU_RUN_DIR/cpu_ops_t${NUM_THREADS}_${BENCHMARK_TIMESTAMP}.md"
if [[ -x "$SCRIPT_DIR/run_cpu_ops.sh" ]]; then
    if ! "$SCRIPT_DIR/run_cpu_ops.sh" "$NUM_THREADS"; then
        echo "WARNING: CPU benchmark items failed; continuing without CPU item results." >&2
    fi
elif [[ -f "$SCRIPT_DIR/run_cpu_ops.sh" ]]; then
    if ! bash "$SCRIPT_DIR/run_cpu_ops.sh" "$NUM_THREADS"; then
        echo "WARNING: CPU benchmark items failed; continuing without CPU item results." >&2
    fi
fi

# ---------------------------------------------------------------------------
# Collect all logs and format as markdown table
# ---------------------------------------------------------------------------
TENFERRO_TRACE_LOG="$CPU_RUN_DIR/tenferro_trace_t${NUM_THREADS}_${BENCHMARK_TIMESTAMP}.log"
TENFERRO_EAGER_LOG="$CPU_RUN_DIR/tenferro_eager_t${NUM_THREADS}_${BENCHMARK_TIMESTAMP}.log"
STRIDED_FAER_LOG="$CPU_RUN_DIR/strided_faer_t${NUM_THREADS}_${BENCHMARK_TIMESTAMP}.log"
PYTORCH_LOG="$CPU_RUN_DIR/pytorch_cpu_t${NUM_THREADS}_${BENCHMARK_TIMESTAMP}.log"
JAX_LOG="$CPU_RUN_DIR/jax_cpu_t${NUM_THREADS}_${BENCHMARK_TIMESTAMP}.log"
MARKDOWN_TABLE="$CPU_RUN_DIR/einsum_table_t${NUM_THREADS}_${BENCHMARK_TIMESTAMP}.md"
EINSUM_REPORT="$CPU_RUN_DIR/report.md"
CPU_REPORT="$CPU_RUN_DIR/cpu_ops_report.md"

LOGS=()
[ -f "$TENFERRO_TRACE_LOG" ] && LOGS+=("$TENFERRO_TRACE_LOG")
[ -f "$TENFERRO_EAGER_LOG" ] && LOGS+=("$TENFERRO_EAGER_LOG")
[ -f "$STRIDED_FAER_LOG" ]   && LOGS+=("$STRIDED_FAER_LOG")
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
