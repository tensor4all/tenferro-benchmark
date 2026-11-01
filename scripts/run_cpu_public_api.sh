#!/usr/bin/env bash
set -euo pipefail

# Runs the CPU public API coverage benchmark suite:
#   - tenferro-rs eager public Tensor/TensorLinalgExt APIs
#   - tenferro-rs traced APIs, compiled once and executed repeatedly
#   - PyTorch Python closest public equivalents where available
#   - JAX Python public equivalents, compiled once with XLA and synchronized

for diagnostic_variable in PUBLIC_API_EXECUTION_FILTER PUBLIC_API_ATTRIBUTION_OUTPUT; do
    if [[ -n "${!diagnostic_variable:-}" ]]; then
        echo "ERROR: $diagnostic_variable is diagnostic-only; invoke benchmark_cpu_public_api directly so a partial run cannot overwrite the publication report." >&2
        exit 1
    fi
done

if [[ $# -eq 0 ]]; then
    THREAD_COUNTS=(1 4)
else
    THREAD_COUNTS=("$@")
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
export BENCH_COMMAND_START_NS="$(python3 -c 'import time; print(time.monotonic_ns())')"
source "$SCRIPT_DIR/public_api_selection.sh"
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
        echo "ERROR: BENCHMARK_TARGET_PROFILE must be mac-cpu, amd-cpu, or zen5-cpu for CPU public API." >&2
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

# The coverage manifest must name only entrypoints of the tenferro-rs revision
# under test; a removed spelling fails here instead of silently counting.
"$PROJECT_DIR/.venv/bin/python" "$SCRIPT_DIR/route_coverage.py" entrypoints \
    --tenferro-dir "$PROJECT_DIR/extern/tenferro-rs"

# shellcheck source=scripts/benchmark_host_idle.sh
source "$SCRIPT_DIR/benchmark_host_idle.sh"
assert_benchmark_host_idle

RESULTS_ROOT="$PROJECT_DIR/data/results"
REPORTS_DIR="$PROJECT_DIR/result"
REPORT_NAME="public_api"
if [[ "${PUBLIC_API_CANDIDATES_ONLY:-0}" == "1" ]]; then
    REPORT_NAME="public_api_candidates"
fi
if [[ "${PUBLIC_API_FOLLOWUPS_ONLY:-0}" == "1" ]]; then
    REPORT_NAME="public_api_followups"
fi
SUITE_ID="cpu/$REPORT_NAME"
SUITE_FILE="$PROJECT_DIR/benchmarks/cpu/$REPORT_NAME.yaml"
BENCHMARK_TIMESTAMP="${BENCHMARK_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
export BENCHMARK_TIMESTAMP
BENCHMARK_PROFILE="${PUBLICATION_GATE_PROFILE:-quick}"
if [[ -n "${BENCH_RUNS:-}" ]]; then
    MEASURED_RUNS="$BENCH_RUNS"
elif [[ "$BENCHMARK_PROFILE" == "full" ]]; then
    MEASURED_RUNS=15
else
    MEASURED_RUNS=7
fi
MEASURED_WARMUPS="${BENCH_WARMUPS:-3}"
RUN_TIMESTAMP_RFC3339="$(python3 - <<'PY'
from datetime import datetime, timezone
print(datetime.now(timezone.utc).isoformat())
PY
)"

RUN_DIR="$RESULTS_ROOT/$BENCHMARK_TARGET_PROFILE/$SUITE_ID/$BENCHMARK_TIMESTAMP"
RUN_YAML="$RUN_DIR/run.yaml"
REPORT="$RUN_DIR/report.md"
LATEST_REPORT="$REPORTS_DIR/$BENCHMARK_TARGET_PROFILE/cpu/$REPORT_NAME.md"
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

# Host and thread-environment snapshots for this run's report header. The
# public API header also prints the Julia thread count.
if command -v uv >/dev/null 2>&1; then
    uv run python "$SCRIPT_DIR/collect_cpu_info.py" --markdown > "$RUN_DIR/cpu_info.md" \
        || python3 "$SCRIPT_DIR/collect_cpu_info.py" --markdown > "$RUN_DIR/cpu_info.md"
else
    python3 "$SCRIPT_DIR/collect_cpu_info.py" --markdown > "$RUN_DIR/cpu_info.md"
fi
THREAD_ENV_EXTRA_KEYS=(JULIA_NUM_THREADS)
begin_thread_envs_snapshot "$RUN_DIR/thread_env.txt"

HAVE_JULIA=0
if [[ "$REPORT_NAME" == "public_api" ]] && command -v julia >/dev/null 2>&1; then
    HAVE_JULIA=1
    echo "Instantiating Julia project..."
    (cd "$PROJECT_DIR" && julia --project="$PROJECT_DIR" -e 'import Pkg; Pkg.instantiate()')
elif [[ "$REPORT_NAME" != "public_api" ]]; then
    echo "Focused candidate collection selects tenferro-rs, PyTorch and JAX; Julia columns are unselected."
else
    echo "WARNING: julia not found on PATH; skipping julia-base / strided-jl columns." >&2
fi

echo "CPU public API benchmark suite"
echo "Project dir:  $PROJECT_DIR"
echo "Thread counts: ${THREAD_COUNTS[*]}"
echo "Timestamp:    $BENCHMARK_TIMESTAMP"
echo "Suite:        $SUITE_ID"
echo "Target:       $BENCHMARK_TARGET_PROFILE"
echo "Run dir:      $RUN_DIR"
echo "Features:     $TENFERRO_CPU_FEATURES"
[[ -n "$TENFERRO_COMMIT" ]] && echo "tenferro-rs:  $TENFERRO_COMMIT"
echo ""

CSVS=()
for NUM_THREADS in "${THREAD_COUNTS[@]}"; do
    echo "--- Thread count: $NUM_THREADS ---"
    configure_cpu_thread_env "$NUM_THREADS"
    print_cpu_thread_env
    append_thread_env_section "$RUN_DIR/thread_env.txt" "$NUM_THREADS"
    RUN_T_YAML="$RUN_DIR/run_t${NUM_THREADS}.yaml"
    CSV="$RUN_DIR/cpu_public_api_t${NUM_THREADS}_${BENCHMARK_TIMESTAMP}.csv"
    collect_run_metadata "$RUN_T_YAML"

    run_rust_group() {
        local api_suite="$1"
        local benchmark_filter="${2:-}"
        select_public_api_group "$api_suite" "$benchmark_filter" || return 0
        benchmark_filter="$PUBLIC_API_GROUP_BENCHMARK_FILTER"
        assert_benchmark_host_idle
        PUBLIC_API_SUITE_FILTER="$api_suite" \
            PUBLIC_API_BENCHMARK_FILTER="$benchmark_filter" \
            cargo run --release --no-default-features --features "$TENFERRO_CPU_FEATURES" --bin benchmark_cpu_public_api -- \
                --num-threads "$NUM_THREADS" \
                --output "$CSV"
    }

    run_rust_group cpu/elementwise_reduction "add,sub,mul,div,neg,abs,sign,maximum,minimum,compare_lt,select,where_select,sqrt,rsqrt"
    run_rust_group cpu/elementwise_reduction "rem,clamp,exp,log,sin,cos,tanh"
    run_rust_group cpu/elementwise_reduction "pow,expm1,log1p,chain_log1p_exp_mul"
    run_rust_group cpu/elementwise_reduction "reduce_sum_all,reduce_prod_all,reduce_max_all,reduce_min_all"
    run_rust_group cpu/elementwise_reduction "reduce_max_axis0,reduce_max_axis1,reduce_min_axis0,reduce_min_axis1,reduce_sum_axis0,reduce_sum_axis1,reduce_prod_axis0,reduce_prod_axis1,reduce_sum_squares_axis0,masked_log_softmax_axis1"
    run_rust_group cpu/indexing_layout
    run_rust_group cpu/structural_shape
    run_rust_group cpu/view_metadata
    run_rust_group cpu/output_reuse
    run_rust_group cpu/einsum_concrete
    run_rust_group cpu/linalg_uncovered
    run_rust_group cpu/linalg_batched
    run_rust_group cpu/linalg_batch_families
    run_rust_group cpu/neural_public_api
    run_rust_group cpu/complex "conj"
    run_rust_group cpu/complex "mul,div"
    run_rust_group cpu/complex "exp,log"
    run_rust_group cpu/complex "dot_general,dot_general_with_conj,tensordot,svd,qr,eig,solve,cholesky,norm_fro"

    assert_benchmark_host_idle
    if [[ "${PUBLIC_API_CANDIDATES_ONLY:-0}" == "1" ]]; then
        export PUBLIC_API_BENCHMARK_FILTER="householder_qr,rank_revealing_qr,rank_revealing_qr_deficient,solve_read_into,reduce_sum_squares_axis0,masked_log_softmax_axis1,where_select"
    fi
    if [[ "${PUBLIC_API_FOLLOWUPS_ONLY:-0}" == "1" ]]; then
        export PUBLIC_API_SUITE_FILTER="cpu/neural_public_api"
    fi
    if command -v uv >/dev/null 2>&1; then
        uv run python "$SCRIPT_DIR/benchmark_cpu_public_api_python.py" \
            --num-threads "$NUM_THREADS" \
            --output "$CSV"
    else
        python3 "$SCRIPT_DIR/benchmark_cpu_public_api_python.py" \
            --num-threads "$NUM_THREADS" \
            --output "$CSV"
    fi

    run_jax_group() {
        local api_suite="$1"
        local benchmark_filter="${2:-}"
        select_public_api_group "$api_suite" "$benchmark_filter" || return 0
        benchmark_filter="$PUBLIC_API_GROUP_BENCHMARK_FILTER"
        assert_benchmark_host_idle
        if command -v uv >/dev/null 2>&1; then
            PUBLIC_API_SUITE_FILTER="$api_suite" \
                PUBLIC_API_BENCHMARK_FILTER="$benchmark_filter" \
                uv run python "$SCRIPT_DIR/benchmark_cpu_public_api_jax.py" \
                    --num-threads "$NUM_THREADS" \
                    --output "$CSV"
        else
            PUBLIC_API_SUITE_FILTER="$api_suite" \
                PUBLIC_API_BENCHMARK_FILTER="$benchmark_filter" \
                python3 "$SCRIPT_DIR/benchmark_cpu_public_api_jax.py" \
                    --num-threads "$NUM_THREADS" \
                    --output "$CSV"
        fi
    }

    # Keep large fixture families and XLA executable caches in separate
    # processes, matching the process isolation used by the Rust runner.
    run_jax_group cpu/elementwise_reduction "add,sub,mul,div,neg,abs,sign,maximum,minimum,compare_lt,select,where_select,sqrt,rsqrt"
    run_jax_group cpu/elementwise_reduction "rem,clamp,exp,log,sin,cos,tanh"
    run_jax_group cpu/elementwise_reduction "pow,expm1,log1p,chain_log1p_exp_mul"
    run_jax_group cpu/elementwise_reduction "reduce_sum_all,reduce_prod_all,reduce_max_all,reduce_min_all"
    run_jax_group cpu/elementwise_reduction "reduce_max_axis0,reduce_max_axis1,reduce_min_axis0,reduce_min_axis1,reduce_sum_axis0,reduce_sum_axis1,reduce_prod_axis0,reduce_prod_axis1,reduce_sum_squares_axis0,masked_log_softmax_axis1"
    run_jax_group cpu/indexing_layout
    run_jax_group cpu/structural_shape
    run_jax_group cpu/output_reuse "solve_read_into"
    run_jax_group cpu/einsum_concrete
    run_jax_group cpu/linalg_uncovered
    run_jax_group cpu/linalg_batched
    run_jax_group cpu/linalg_batch_families
    run_jax_group cpu/neural_public_api
    run_jax_group cpu/complex "conj"
    run_jax_group cpu/complex "mul,div"
    run_jax_group cpu/complex "exp,log"
    run_jax_group cpu/complex "dot_general,dot_general_with_conj,tensordot,svd,qr,eig,solve,cholesky,norm_fro"

    run_julia_group() {
        local api_suite="$1"
        local benchmark_filter="${2:-}"
        assert_benchmark_host_idle
        # Fresh julia process per family group: Julia's thread pool is sized
        # once at process start from JULIA_NUM_THREADS (exported by
        # configure_cpu_thread_env above), matching the process-isolation
        # convention used for the Rust/JAX groups and for
        # scripts/run_permutation.sh's Julia runner.
        (
            cd "$PROJECT_DIR"
            PUBLIC_API_SUITE_FILTER="$api_suite" \
                PUBLIC_API_BENCHMARK_FILTER="$benchmark_filter" \
                julia --project="$PROJECT_DIR" "$SCRIPT_DIR/benchmark_cpu_public_api_julia.jl" \
                    --num-threads "$NUM_THREADS" \
                    --output "$CSV"
        )
    }

    if [[ "$HAVE_JULIA" == "1" ]]; then
        # Same family/process-isolation granularity as the JAX groups above,
        # plus one group each for structural_shape and linalg_uncovered.
        run_julia_group cpu/elementwise_reduction "add,sub,mul,div,neg,abs,sign,maximum,minimum,compare_lt,select,sqrt,rsqrt"
        run_julia_group cpu/elementwise_reduction "rem,clamp,exp,log,sin,cos,tanh"
        run_julia_group cpu/elementwise_reduction "pow,expm1,log1p,chain_log1p_exp_mul"
        run_julia_group cpu/elementwise_reduction "reduce_sum_all,reduce_prod_all,reduce_max_all,reduce_min_all"
        run_julia_group cpu/elementwise_reduction "reduce_max_axis0,reduce_max_axis1,reduce_min_axis0,reduce_min_axis1,reduce_sum_axis0,reduce_sum_axis1,reduce_prod_axis0,reduce_prod_axis1"
        run_julia_group cpu/structural_shape
        run_julia_group cpu/linalg_uncovered
        run_julia_group cpu/linalg_batched
        run_julia_group cpu/indexing_layout
        run_julia_group cpu/view_metadata
        run_julia_group cpu/output_reuse
        run_julia_group cpu/complex "conj"
        run_julia_group cpu/complex "mul,div,exp,log"
        run_julia_group cpu/complex "dot_general,dot_general_with_conj,tensordot,svd,qr,eig,solve,cholesky,norm_fro"
    fi
    CSVS+=("$CSV")
done

finalize_benchmark_run "$PROJECT_DIR" "$RUN_DIR"
if [[ "$REPORT_NAME" != "public_api" && "${BENCHMARK_RAW_ONLY:-0}" != 1 ]]; then
    # Candidate and follow-up runs are diagnostics: their page is the run report
    # rewritten against the candidate contracts, not the public API page.
    render_run_record "$PROJECT_DIR" "$RUN_DIR"
    mkdir -p "$(dirname "$LATEST_REPORT")"
    "$PROJECT_DIR/.venv/bin/python" "$SCRIPT_DIR/format_public_api_candidate_report.py" "$REPORT"
    cp "$REPORT" "$LATEST_REPORT"
else
    publish_cpu_runs "$PROJECT_DIR" "$RUN_DIR"
fi

echo ""
echo "CPU public API benchmark complete."
for CSV in "${CSVS[@]}"; do echo "CSV:     $CSV"; done
echo "Report:  $REPORT"
echo "Latest:  $LATEST_REPORT"
