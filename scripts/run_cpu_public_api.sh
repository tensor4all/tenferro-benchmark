#!/usr/bin/env bash
set -euo pipefail

# Runs the CPU public API coverage benchmark suite:
#   - tenferro-rs eager public Tensor/TensorLinalgExt APIs
#   - tenferro-rs traced APIs, compiled once and executed repeatedly
#   - PyTorch Python closest public equivalents where available

if [[ $# -eq 0 ]]; then
    THREAD_COUNTS=(1 4)
else
    THREAD_COUNTS=("$@")
fi

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

default_target_profile() {
    case "$(benchmark_host_os)" in
        Darwin) printf '%s\n' "mac-cpu" ;;
        *) printf '%s\n' "amd-cpu" ;;
    esac
}

BENCHMARK_TARGET_PROFILE="${BENCHMARK_TARGET_PROFILE:-$(default_target_profile)}"
case "$BENCHMARK_TARGET_PROFILE" in
    mac-cpu|amd-cpu|linux-cpu)
        export BENCHMARK_TARGET_PROFILE
        ;;
    *)
        echo "ERROR: BENCHMARK_TARGET_PROFILE must be mac-cpu, amd-cpu, or linux-cpu for CPU public API." >&2
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

RESULTS_ROOT="$PROJECT_DIR/data/results"
REPORTS_DIR="$PROJECT_DIR/result"
SUITE_ID="cpu/public_api"
SUITE_FILE="$PROJECT_DIR/benchmarks/cpu/public_api.yaml"
BENCHMARK_TIMESTAMP="${BENCHMARK_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
export BENCHMARK_TIMESTAMP
RUN_TIMESTAMP_RFC3339="$(python3 - <<'PY'
from datetime import datetime, timezone
print(datetime.now(timezone.utc).isoformat())
PY
)"

RUN_DIR="$RESULTS_ROOT/$BENCHMARK_TARGET_PROFILE/cpu/public_api/$BENCHMARK_TIMESTAMP"
RUN_YAML="$RUN_DIR/run.yaml"
TABLE="$RUN_DIR/cpu_public_api_${BENCHMARK_TIMESTAMP}.md"
REPORT="$RUN_DIR/report.md"
LATEST_REPORT="$REPORTS_DIR/$BENCHMARK_TARGET_PROFILE/cpu/public_api.md"
TENFERRO_DIR="${TENFERRO_RS_DIR:-$PROJECT_DIR/extern/tenferro-rs}"
TENFERRO_COMMIT=""
if [[ -d "$TENFERRO_DIR/.git" ]] && command -v git >/dev/null 2>&1; then
    TENFERRO_COMMIT="$(git -C "$TENFERRO_DIR" rev-parse HEAD 2>/dev/null || true)"
fi

mkdir -p "$RUN_DIR" "$(dirname "$LATEST_REPORT")"

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

echo "CPU public API benchmark suite"
echo "Project dir:  $PROJECT_DIR"
echo "Thread counts: ${THREAD_COUNTS[*]}"
echo "Timestamp:    $BENCHMARK_TIMESTAMP"
echo "Suite:        $SUITE_ID"
echo "Target:       $BENCHMARK_TARGET_PROFILE"
echo "Run dir:      $RUN_DIR"
echo "Features:     $TENFERRO_CPU_FEATURES"
echo "CPU backend:  $TENFERRO_CPU_BACKEND_KIND"
[[ -n "$TENFERRO_COMMIT" ]] && echo "tenferro-rs:  $TENFERRO_COMMIT"
echo ""

CSVS=()
for NUM_THREADS in "${THREAD_COUNTS[@]}"; do
    echo "--- Thread count: $NUM_THREADS ---"
    configure_cpu_thread_env "$NUM_THREADS"
    print_cpu_thread_env
    RUN_T_YAML="$RUN_DIR/run_t${NUM_THREADS}.yaml"
    CSV="$RUN_DIR/cpu_public_api_t${NUM_THREADS}_${BENCHMARK_TIMESTAMP}.csv"
    collect_run_metadata "$RUN_T_YAML"

    run_rust_group() {
        local api_suite="$1"
        local benchmark_filter="${2:-}"
        PUBLIC_API_SUITE_FILTER="$api_suite" \
            PUBLIC_API_BENCHMARK_FILTER="$benchmark_filter" \
            cargo run --release --features "$TENFERRO_CPU_FEATURES" --bin benchmark_cpu_public_api -- \
                --num-threads "$NUM_THREADS" \
                --output "$CSV"
    }

    run_rust_group cpu/elementwise_reduction "add,sub,mul,div,neg,abs,sign,maximum,minimum,compare_lt,select,sqrt,rsqrt"
    run_rust_group cpu/elementwise_reduction "rem,clamp,exp,log,sin,cos,tanh"
    run_rust_group cpu/elementwise_reduction "pow,expm1,log1p,chain_log1p_exp_mul"
    run_rust_group cpu/elementwise_reduction "reduce_sum_all,reduce_prod_all,reduce_max_axis0,reduce_min_axis1"
    run_rust_group cpu/indexing_layout
    run_rust_group cpu/linalg_uncovered
    run_rust_group cpu/complex "conj"
    run_rust_group cpu/complex "mul,div"
    run_rust_group cpu/complex "exp,log"
    run_rust_group cpu/complex "dot_general,svd,qr,eig,solve,cholesky,norm_fro"

    if command -v uv >/dev/null 2>&1; then
        uv run python "$SCRIPT_DIR/benchmark_cpu_public_api_python.py" \
            --num-threads "$NUM_THREADS" \
            --output "$CSV"
    else
        python3 "$SCRIPT_DIR/benchmark_cpu_public_api_python.py" \
            --num-threads "$NUM_THREADS" \
            --output "$CSV"
    fi
    CSVS+=("$CSV")
done

if command -v uv >/dev/null 2>&1; then
    uv run python "$SCRIPT_DIR/format_cpu_ops_results.py" "${CSVS[@]}" | tee "$TABLE" \
        || python3 "$SCRIPT_DIR/format_cpu_ops_results.py" "${CSVS[@]}" | tee "$TABLE"
else
    python3 "$SCRIPT_DIR/format_cpu_ops_results.py" "${CSVS[@]}" | tee "$TABLE"
fi

{
    echo "# CPU Public API Benchmark Results"
    echo ""
    echo "- Suite: \`$SUITE_ID\`"
    echo "- Target profile: \`$BENCHMARK_TARGET_PROFILE\`"
    echo "- Suite file: \`${SUITE_FILE#$PROJECT_DIR/}\`"
    echo "- Run metadata: \`${RUN_YAML#$PROJECT_DIR/}\`"
    echo "- Timestamp: \`$BENCHMARK_TIMESTAMP\`"
    echo ""
    echo "Latest run: \`./scripts/run_cpu_public_api.sh ${THREAD_COUNTS[*]}\`."
    echo ""
    echo "This file is generated from sequential CPU public API runs under \`${RUN_DIR#$PROJECT_DIR/}\`."
    echo ""
    [[ -n "$TENFERRO_COMMIT" ]] && echo "- tenferro-rs commit: \`$TENFERRO_COMMIT\`"
    [[ -n "$TENFERRO_COMMIT" ]] && echo ""
    if command -v uv >/dev/null 2>&1; then
        uv run python "$SCRIPT_DIR/collect_cpu_info.py" --markdown \
            || python3 "$SCRIPT_DIR/collect_cpu_info.py" --markdown
    else
        python3 "$SCRIPT_DIR/collect_cpu_info.py" --markdown
    fi
    echo ""
    echo "## Thread Environments"
    for NUM_THREADS in "${THREAD_COUNTS[@]}"; do
        configure_cpu_thread_env "$NUM_THREADS"
        echo ""
        echo "### Threads: $NUM_THREADS"
        echo ""
        echo "- Run metadata: \`data/results/$BENCHMARK_TARGET_PROFILE/cpu/public_api/$BENCHMARK_TIMESTAMP/run_t${NUM_THREADS}.yaml\`"
        for key in OMP_NUM_THREADS OMP_THREAD_LIMIT OMP_DYNAMIC RAYON_NUM_THREADS OPENBLAS_NUM_THREADS GOTO_NUM_THREADS MKL_NUM_THREADS VECLIB_MAXIMUM_THREADS VECLIB_NUM_THREADS NUMEXPR_NUM_THREADS BLIS_NUM_THREADS XLA_FLAGS; do
            echo "- ${key}: \`${!key:-}\`"
        done
    done
    echo ""
    echo "## Timing Discipline"
    echo ""
    echo "- Input fixture tensors are created during warmup and outside the measured region for both tenferro-rs and PyTorch."
    echo "- tenferro-rs trace graphs are constructed and compiled outside the measured region; each compiled graph is reused for every warmup and timed run."
    echo "- Each timed call creates the output tensor."
    echo "- PyTorch view-producing indexing operations are cloned inside the timed region to match tenferro-rs owned, materialized outputs."
    echo "- PyTorch complex conjugation uses \`torch.conj_physical\` to match tenferro-rs physical output rather than the lazy conjugate view from \`torch.conj\`."
    echo "- \`dynamic_update_slice\` reports trace mode as \`unsupported\` because tenferro-rs does not currently expose a corresponding \`TracedTensor\` API."
    echo "- \`full_piv_lu\` and \`full_piv_lu_solve\` are excluded because PyTorch has no direct public full-pivot equivalent; substituting \`torch.linalg.solve\` would compare different algorithms."
    echo ""
    echo "## Threads: ${THREAD_COUNTS[*]}"
    echo ""
    for CSV in "${CSVS[@]}"; do
        echo "- CSV: \`${CSV#$PROJECT_DIR/}\`"
    done
    echo "- Source table: \`${TABLE#$PROJECT_DIR/}\`"
    echo ""
    cat "$TABLE"
} > "$REPORT"

cp "$REPORT" "$LATEST_REPORT"

echo ""
echo "CPU public API benchmark complete."
for CSV in "${CSVS[@]}"; do echo "CSV:     $CSV"; done
echo "Report:  $REPORT"
echo "Latest:  $LATEST_REPORT"
