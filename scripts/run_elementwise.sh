#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# CPU elementwise unary-kernel benchmark suite (cpu/elementwise):
#   f64 sqrt / rsqrt / exp on 33_554_432 elements through the tenferro-rs eager
#   CPU API, measured under two CPU buffer-pool configurations:
#     - pool=default: whatever CpuEngine gives (DEFAULT_MAX_RETAINED_CAPACITY_BYTES
#       = 100 MiB)
#     - pool=raised:  CpuEngine::set_buffer_pool_limit_bytes(1 << 32) (4 GiB)
#
# Why this suite exists: tensor4all/tenferro-rs#2284 reported f64 sqrt / rsqrt
# on 33_554_432 elements as 2.3-3.4x slower than JAX/PyTorch at 1 thread. The
# kernel was not the cause: a 256 MiB output is above the default pool limit, so
# the pool cannot hand it back, and the measured calls paid allocation and
# first-touch costs. Raising the limit took 1T sqrt from 133 ms to 24 ms on the
# measuring host. The default/raised ratio prices that pool policy
# (see docs/elementwise-suite.md).
#
# Usage: ./scripts/run_elementwise.sh <NUM_THREADS...>
#   e.g. ./scripts/run_elementwise.sh 1
#        ./scripts/run_elementwise.sh 1 4
#
# Each thread count runs as a fresh Rust process (the engine owns its workers:
# one thread is CpuEngine::serial(), wider runs get a private pool, and every CSV
# row records the width the engine reports), and the thread counts are looped
# over SEQUENTIALLY inside one
# run directory/timestamp rather than run concurrently, per AGENTS.md timing
# discipline. One CSV per thread count (elementwise_t<N>_<timestamp>.csv) plus
# the usual run metadata (run.yaml, run_t<N>.yaml) are written under
# data/results/<target_profile>/cpu/elementwise/<timestamp>/, and the formatter
# runs once at the end over all of them so the report has one table per thread
# count.
#
# This is a STANDALONE entry point. It is intentionally not wired into
# scripts/run_all.sh's default path; opt in with RUN_ELEMENTWISE_SUITE=1 or run
# it explicitly.
#
# Environment variables:
# - TENFERRO_CPU_FEATURES: Cargo CPU backend features. Defaults to `native`
#   (tenferro's own kernels): the elementwise path uses no BLAS, so this suite
#   does not need an OpenBLAS/MKL installation and runs outside the devcontainer.
# - BENCHMARK_TIMESTAMP: override the run directory / CSV timestamp.
# ---------------------------------------------------------------------------

if [[ $# -eq 0 ]]; then
    echo "Usage: $0 <NUM_THREADS...>  (e.g. $0 1  or  $0 1 4)" >&2
    exit 1
fi
THREAD_COUNTS=("$@")

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
export BENCH_COMMAND_START_NS="$(python3 -c 'import time; print(time.monotonic_ns())')"

# shellcheck source=scripts/cpu_blas_provider.sh
source "$SCRIPT_DIR/cpu_blas_provider.sh"
# shellcheck source=scripts/publish_run.sh
source "$SCRIPT_DIR/publish_run.sh"

TENFERRO_CPU_FEATURES="$(normalize_cpu_blas_features "${TENFERRO_CPU_FEATURES:-native}")"
export TENFERRO_CPU_FEATURES
CARGO_FEATURES="$TENFERRO_CPU_FEATURES"

default_target_profile() {
    case "$(benchmark_host_os)" in
        Darwin) printf '%s\n' "mac-cpu" ;;
        *) printf '%s\n' "amd-cpu" ;;
    esac
}

BENCHMARK_TARGET_PROFILE="${BENCHMARK_TARGET_PROFILE:-$(default_target_profile)}"
case "$BENCHMARK_TARGET_PROFILE" in
    mac-cpu|amd-cpu)
        export BENCHMARK_TARGET_PROFILE
        ;;
    *)
        echo "ERROR: BENCHMARK_TARGET_PROFILE must be mac-cpu or amd-cpu for cpu/elementwise." >&2
        exit 1
        ;;
esac

# shellcheck source=scripts/thread_env.sh
source "$SCRIPT_DIR/thread_env.sh"

if [[ "${SKIP_EXTERN_SETUP:-0}" != "1" ]]; then
    # shellcheck source=scripts/setup_extern_deps.sh
    source "$SCRIPT_DIR/setup_extern_deps.sh"
fi

# A BLAS selection needs MKLROOT / OPENBLAS_ROOT; `native` needs nothing.
ensure_blas_env_for_features "$TENFERRO_CPU_FEATURES"

RESULTS_ROOT="$PROJECT_DIR/data/results"
REPORTS_DIR="$PROJECT_DIR/result"
SUITE_ID="cpu/elementwise"
SUITE_FILE="$PROJECT_DIR/benchmarks/cpu/elementwise.yaml"
TIMESTAMP="${BENCHMARK_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_TIMESTAMP_RFC3339="$(python3 - <<'PY'
from datetime import datetime, timezone
print(datetime.now(timezone.utc).isoformat())
PY
)"

RUN_DIR="$RESULTS_ROOT/$BENCHMARK_TARGET_PROFILE/cpu/elementwise/$TIMESTAMP"
RUN_YAML="$RUN_DIR/run.yaml"
LATEST_REPORT="$REPORTS_DIR/$BENCHMARK_TARGET_PROFILE/cpu/elementwise.md"

mkdir -p "$RUN_DIR"

PYTHON=(python3)
if command -v uv >/dev/null 2>&1; then
    PYTHON=(uv run python)
fi

"${PYTHON[@]}" "$SCRIPT_DIR/validate_benchmark_suite.py" "$SUITE_FILE"

blas_impl="$(blas_impl_for_features "$TENFERRO_CPU_FEATURES")"
tenferro_dir="${TENFERRO_RS_DIR:-$PROJECT_DIR/extern/tenferro-rs}"
tenferro_commit=""
if [[ -d "$tenferro_dir/.git" ]] && command -v git >/dev/null 2>&1; then
    tenferro_commit="$(git -C "$tenferro_dir" rev-parse HEAD 2>/dev/null || true)"
fi

write_run_metadata() {
    local output="$1"
    local args=(
        --suite-id "$SUITE_ID"
        --target-profile "$BENCHMARK_TARGET_PROFILE"
        --suite-file "${SUITE_FILE#$PROJECT_DIR/}"
        --timestamp "$RUN_TIMESTAMP_RFC3339"
        --tenferro-dir "$tenferro_dir"
        --features "$CARGO_FEATURES"
        --blas "$blas_impl"
        --output "$output"
    )
    [[ -n "$tenferro_commit" ]] && args+=(--tenferro-commit "$tenferro_commit")
    "${PYTHON[@]}" "$SCRIPT_DIR/collect_run_metadata.py" "${args[@]}"
}

write_run_metadata "$RUN_YAML"
"${PYTHON[@]}" "$SCRIPT_DIR/validate_benchmark_suite.py" --kind run "$RUN_YAML"

echo "============================================"
echo " CPU elementwise benchmark suite"
echo "============================================"
echo "Suite:         $SUITE_ID"
echo "Suite file:    ${SUITE_FILE#$PROJECT_DIR/}"
echo "Target:        $BENCHMARK_TARGET_PROFILE"
echo "Thread counts: ${THREAD_COUNTS[*]}"
echo "Timestamp:     $TIMESTAMP"
echo "Run dir:       ${RUN_DIR#$PROJECT_DIR/}"
echo "Cargo features: $CARGO_FEATURES"
echo ""

echo "Building benchmark_cpu_elementwise (features: $CARGO_FEATURES)..."
cargo build -j 16 --release --no-default-features --features "$CARGO_FEATURES" \
    --bin benchmark_cpu_elementwise

INPUTS=()

for NUM_THREADS in "${THREAD_COUNTS[@]}"; do
    echo ""
    echo "--- Thread count: $NUM_THREADS ---"

    # Fresh process per thread count: the benchmark builds its engine (and, for
    # a wide run, its own pool) for that width, and the same variables are
    # exported before the process starts so any library-level parallelism
    # matches.
    configure_cpu_thread_env "$NUM_THREADS"
    print_cpu_thread_env

    RUN_T_YAML="$RUN_DIR/run_t${NUM_THREADS}.yaml"
    write_run_metadata "$RUN_T_YAML"
    "${PYTHON[@]}" "$SCRIPT_DIR/validate_benchmark_suite.py" --kind run "$RUN_T_YAML"

    CSV="$RUN_DIR/elementwise_t${NUM_THREADS}_${TIMESTAMP}.csv"
    echo "Running CPU elementwise benchmarks (threads=$NUM_THREADS)..."
    BENCH_OUTPUT="$CSV" \
        "${CARGO_TARGET_DIR:-$PROJECT_DIR/target}/release/benchmark_cpu_elementwise" \
        --num-threads "$NUM_THREADS"
    INPUTS+=("$CSV")
done

# ---------------------------------------------------------------------------
# Format the latest report once, over every thread count's CSV.
# ponytail: cpu/elementwise has no scripts/render_run.py renderer, so its page is
# formatted straight from the run directory and `regenerate_report_summaries.py
# --check` cannot re-derive it; the suite is a standalone mac-cpu/amd-cpu
# diagnostic, not a maintained profile page.
# ---------------------------------------------------------------------------
finalize_benchmark_run "$PROJECT_DIR" "$RUN_DIR"
if benchmark_raw_only; then
    echo "BENCHMARK_RAW_ONLY=1: the run record ends here; no report, summary or result/ page is written." >&2
else
    echo ""
    echo "Formatting cpu/elementwise results as markdown..."
    mkdir -p "$(dirname "$LATEST_REPORT")"
    "${PYTHON[@]}" "$SCRIPT_DIR/format_elementwise_results.py" \
        "${INPUTS[@]}" \
        --run-metadata "$RUN_YAML" \
        --output "$LATEST_REPORT"
fi

echo ""
echo "CPU elementwise benchmark suite complete: $SUITE_ID"
echo "Run YAML:  ${RUN_YAML#$PROJECT_DIR/}"
for path in "${INPUTS[@]}"; do
    echo "CSV:       ${path#$PROJECT_DIR/}"
done
echo "Report:    ${LATEST_REPORT#$PROJECT_DIR/}"
