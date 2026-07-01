#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# CPU einsum benchmark runner:
#   - tenferro-rs trace + eager (Rust)
#   - PyTorch CPU + JAX CPU (Python)
#   - CPU ops microbenchmarks (primal linalg + JVP/VJP on trace + eager backward)
#
# Usage: ./scripts/run_all.sh [NUM_THREADS]
#
# Compares einsum performance for the instances listed in
# benchmarks/cpu/einsum.yaml and writes result/<target_profile>/cpu/einsum.md.
# CPU ops (including linalg JVP/VJP) are written to:
#   result/<target_profile>/cpu/cpu_ops.md
#   result/<target_profile>/cpu/linalg_jvp_vjp.md
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
            system-openblas|system-accelerate|system-mkl)
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
    # Source this so TENFERRO_RS_DIR and any explicitly configured BLAS paths
    # are visible to the benchmark subprocesses below.
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

read_run_yaml_section_scalar() {
    local run_yaml="$1"
    local section="$2"
    local key="$3"
    awk -v section="$section" -v key="$key" '
        $0 == section ":" {
            in_section = 1
            next
        }
        in_section && $0 ~ /^[^[:space:]]/ {
            exit
        }
        in_section {
            line = $0
            sub(/^[[:space:]]+/, "", line)
            if (line ~ "^" key ":[[:space:]]*") {
                sub("^" key ":[[:space:]]*", "", line)
                gsub(/^'\''|'\''$/, "", line)
                gsub(/^"|"$/, "", line)
                print line
                exit
            }
        }
    ' "$run_yaml"
}

write_blas_backend_section() {
    local run_yaml="$1"
    local implementation version root library
    implementation="$(read_run_yaml_section_scalar "$run_yaml" "blas" "implementation")"
    version="$(read_run_yaml_section_scalar "$run_yaml" "blas" "version")"
    root="$(read_run_yaml_section_scalar "$run_yaml" "blas" "root")"
    library="$(read_run_yaml_section_scalar "$run_yaml" "blas" "library")"

    echo "## Tenferro CPU BLAS Backend"
    echo ""
    echo "- tenferro-rs features: \`$TENFERRO_CPU_FEATURES\`"
    echo "- TENFERRO_CPU_BACKEND_KIND: \`${TENFERRO_CPU_BACKEND_KIND:-}\`"
    [[ -n "$implementation" ]] && echo "- BLAS implementation: \`$implementation\`"
    [[ -n "$version" ]] && echo "- BLAS version: \`$version\`"
    [[ -n "$root" ]] && echo "- BLAS root: \`$root\`"
    [[ -n "$library" ]] && echo "- BLAS library: \`$library\`"
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
        write_blas_backend_section "$CPU_RUN_YAML"
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
            "$PYTORCH_LOG" \
            "$JAX_LOG"; do
            [[ -f "$log" ]] && echo "- \`${log#$PROJECT_DIR/}\`"
        done
        echo ""
        cat "$MARKDOWN_TABLE"
    } > "$report"
}

write_linalg_ad_report() {
    local report="$1"
    mkdir -p "$(dirname "$report")"
    {
        echo "# CPU Linalg JVP/VJP Benchmark Results"
        echo ""
        echo "- Suite: \`cpu/linalg_jvp_vjp\`"
        echo "- Target profile: \`$BENCHMARK_TARGET_PROFILE\`"
        echo "- Timestamp: \`$BENCHMARK_TIMESTAMP\`"
        echo ""
        echo "Latest run: \`./scripts/run_all.sh $NUM_THREADS\`."
        echo ""
        echo "Derived from the CPU ops CSV under \`${CPU_RUN_DIR#$PROJECT_DIR/}\`."
        echo ""
        if [[ -n "$TENFERRO_COMMIT" ]]; then
            echo "- tenferro-rs commit: \`$TENFERRO_COMMIT\`"
            echo ""
        fi
        write_cpu_info_section
        echo ""
        write_thread_env_section
        echo ""
        write_blas_backend_section "$CPU_RUN_YAML"
        echo ""
        write_python_backend_section "$CPU_RUN_YAML"
        echo ""
        echo "## Threads: $NUM_THREADS"
        echo ""
        [[ -f "$CPU_OPS_LOG" ]] && echo "- CSV: \`${CPU_OPS_LOG#$PROJECT_DIR/}\`"
        echo "- Source table: \`${LINALG_AD_MD#$PROJECT_DIR/}\`"
        echo ""
        cat "$LINALG_AD_MD"
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
        write_blas_backend_section "$CPU_RUN_YAML"
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
CPU_LINALG_AD_LATEST_REPORT="$REPORTS_DIR/$BENCHMARK_TARGET_PROFILE/cpu/linalg_jvp_vjp.md"

mkdir -p "$CPU_RUN_DIR" "$(dirname "$CPU_LATEST_REPORT")"
export BENCHMARK_RESULTS_DIR="$CPU_RUN_DIR"

ensure_blas_env_for_features "$TENFERRO_CPU_FEATURES"
configure_cpu_thread_env "$NUM_THREADS"

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
echo "CPU backend:  $TENFERRO_CPU_BACKEND_KIND"
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
# Focused CPU benchmark items (primal linalg, JVP/VJP on trace, backward on eager)
# ---------------------------------------------------------------------------
CPU_OPS_LOG="$CPU_RUN_DIR/cpu_ops_t${NUM_THREADS}_${BENCHMARK_TIMESTAMP}.csv"
CPU_OPS_MD="$CPU_RUN_DIR/cpu_ops_t${NUM_THREADS}_${BENCHMARK_TIMESTAMP}.md"
LINALG_AD_MD="$CPU_RUN_DIR/linalg_jvp_vjp_t${NUM_THREADS}_${BENCHMARK_TIMESTAMP}.md"
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
# Collect all logs and format as markdown table
# ---------------------------------------------------------------------------
TENFERRO_TRACE_LOG="$CPU_RUN_DIR/tenferro_trace_t${NUM_THREADS}_${BENCHMARK_TIMESTAMP}.log"
TENFERRO_EAGER_LOG="$CPU_RUN_DIR/tenferro_eager_t${NUM_THREADS}_${BENCHMARK_TIMESTAMP}.log"
PYTORCH_LOG="$CPU_RUN_DIR/pytorch_cpu_t${NUM_THREADS}_${BENCHMARK_TIMESTAMP}.log"
JAX_LOG="$CPU_RUN_DIR/jax_cpu_t${NUM_THREADS}_${BENCHMARK_TIMESTAMP}.log"
MARKDOWN_TABLE="$CPU_RUN_DIR/einsum_table_t${NUM_THREADS}_${BENCHMARK_TIMESTAMP}.md"
EINSUM_REPORT="$CPU_RUN_DIR/report.md"
CPU_REPORT="$CPU_RUN_DIR/cpu_ops_report.md"
LINALG_AD_REPORT="$CPU_RUN_DIR/linalg_jvp_vjp_report.md"

LOGS=()
[ -f "$TENFERRO_TRACE_LOG" ] && LOGS+=("$TENFERRO_TRACE_LOG")
[ -f "$TENFERRO_EAGER_LOG" ] && LOGS+=("$TENFERRO_EAGER_LOG")
[ -f "$PYTORCH_LOG" ]        && LOGS+=("$PYTORCH_LOG")
[ -f "$JAX_LOG" ]            && LOGS+=("$JAX_LOG")

if [ ${#LOGS[@]} -gt 0 ]; then
    echo "Formatting einsum results as markdown..."
    mkdir -p "$CPU_RUN_DIR" "$(dirname "$CPU_LATEST_REPORT")" "$(dirname "$CPU_OPS_LATEST_REPORT")" "$(dirname "$CPU_LINALG_AD_LATEST_REPORT")"
    if command -v uv >/dev/null 2>&1; then
        uv run python "$PROJECT_DIR/scripts/format_results.py" "${LOGS[@]}" | tee "$MARKDOWN_TABLE" \
            || python3 "$PROJECT_DIR/scripts/format_results.py" "${LOGS[@]}" | tee "$MARKDOWN_TABLE"
    else
        python3 "$PROJECT_DIR/scripts/format_results.py" "${LOGS[@]}" | tee "$MARKDOWN_TABLE"
    fi

    write_einsum_report "$EINSUM_REPORT"
    cp "$EINSUM_REPORT" "$CPU_LATEST_REPORT"
    echo ""
fi

if [ -f "$CPU_OPS_LOG" ]; then
    echo "Formatting CPU ops results as markdown..."
    mkdir -p "$CPU_RUN_DIR" "$(dirname "$CPU_OPS_LATEST_REPORT")" "$(dirname "$CPU_LINALG_AD_LATEST_REPORT")"
    if command -v uv >/dev/null 2>&1; then
        uv run python "$PROJECT_DIR/scripts/format_cpu_ops_results.py" "$CPU_OPS_LOG" | tee "$CPU_OPS_MD" \
            || python3 "$PROJECT_DIR/scripts/format_cpu_ops_results.py" "$CPU_OPS_LOG" | tee "$CPU_OPS_MD"
    else
        python3 "$PROJECT_DIR/scripts/format_cpu_ops_results.py" "$CPU_OPS_LOG" | tee "$CPU_OPS_MD"
    fi
    write_cpu_report "$CPU_REPORT"
    cp "$CPU_REPORT" "$CPU_OPS_LATEST_REPORT"

    if grep -Eq '_jvp,|_vjp,' "$CPU_OPS_LOG"; then
        if command -v uv >/dev/null 2>&1; then
            uv run python "$PROJECT_DIR/scripts/format_linalg_ad_results.py" "$CPU_OPS_LOG" | tee "$LINALG_AD_MD" \
                || python3 "$PROJECT_DIR/scripts/format_linalg_ad_results.py" "$CPU_OPS_LOG" | tee "$LINALG_AD_MD"
        else
            python3 "$PROJECT_DIR/scripts/format_linalg_ad_results.py" "$CPU_OPS_LOG" | tee "$LINALG_AD_MD"
        fi
        write_linalg_ad_report "$LINALG_AD_REPORT"
        cp "$LINALG_AD_REPORT" "$CPU_LINALG_AD_LATEST_REPORT"
    else
        echo "WARNING: no linalg JVP/VJP rows in $CPU_OPS_LOG; skipping linalg_jvp_vjp report." >&2
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
[ -f "$LINALG_AD_REPORT" ] && echo "  Report:   $LINALG_AD_REPORT"
[ -f "$CPU_LINALG_AD_LATEST_REPORT" ] && echo "  Latest:   $CPU_LINALG_AD_LATEST_REPORT"
true
