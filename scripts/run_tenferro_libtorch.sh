#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Focused comparison runner:
#   - tenferro trace/eager (Rust, system OpenBLAS)
#   - LibTorch CPU (C++, OpenBLAS-linked libtorch_cpu)
#
# Usage: ./scripts/run_tenferro_libtorch.sh [NUM_THREADS]
# ---------------------------------------------------------------------------

NUM_THREADS="${1:-1}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="$PROJECT_DIR/data/results"
BUILD_DIR="$PROJECT_DIR/build/cpp-libtorch"

# shellcheck source=scripts/thread_env.sh
source "$SCRIPT_DIR/thread_env.sh"
configure_cpu_thread_env "$NUM_THREADS"

mkdir -p "$RESULTS_DIR"

export BENCHMARK_TIMESTAMP="${BENCHMARK_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"

if [[ -z "${OPENBLAS_ROOT:-}" ]]; then
    if command -v brew >/dev/null 2>&1 && brew --prefix openblas >/dev/null 2>&1; then
        export OPENBLAS_ROOT="$(brew --prefix openblas)"
    else
        echo "ERROR: OPENBLAS_ROOT is required for tenferro vs LibTorch comparison." >&2
        echo "Set OPENBLAS_ROOT=/path/to/openblas." >&2
        exit 1
    fi
fi

if [[ -z "${Torch_DIR:-}" && -z "${CMAKE_PREFIX_PATH:-}" ]]; then
    echo "ERROR: Torch_DIR or CMAKE_PREFIX_PATH must point to an OpenBLAS-linked LibTorch." >&2
    echo "Example: Torch_DIR=/path/to/libtorch/share/cmake/Torch" >&2
    exit 1
fi

echo "============================================"
echo " tenferro vs LibTorch benchmark"
echo "============================================"
echo "Project dir:       $PROJECT_DIR"
echo "Threads:           $NUM_THREADS"
echo "Timestamp:         $BENCHMARK_TIMESTAMP"
echo "OPENBLAS_ROOT:     $OPENBLAS_ROOT"
print_cpu_thread_env
[[ -n "${Torch_DIR:-}" ]] && echo "Torch_DIR:         $Torch_DIR"
[[ -n "${CMAKE_PREFIX_PATH:-}" ]] && echo "CMAKE_PREFIX_PATH: $CMAKE_PREFIX_PATH"
echo ""

echo "Checking that LibTorch is linked against OpenBLAS..."
rm -rf "$BUILD_DIR"
CMAKE_ARGS=(
    -DBUILD_LIBTORCH_BENCHMARK=ON
    -DOPENBLAS_ROOT="$OPENBLAS_ROOT"
)
if [[ -n "${Torch_DIR:-}" ]]; then
    CMAKE_ARGS+=(-DTorch_DIR="$Torch_DIR")
fi
cmake -S "$PROJECT_DIR/cpp" -B "$BUILD_DIR" \
    -U Torch_DIR \
    "${CMAKE_ARGS[@]}" >/dev/null
echo ""

TENFERRO_TRACE_LOG="$RESULTS_DIR/tenferro_trace_t${NUM_THREADS}_${BENCHMARK_TIMESTAMP}.log"
TENFERRO_EAGER_LOG="$RESULTS_DIR/tenferro_eager_t${NUM_THREADS}_${BENCHMARK_TIMESTAMP}.log"
LIBTORCH_LOG="$RESULTS_DIR/libtorch_cpu_t${NUM_THREADS}_${BENCHMARK_TIMESTAMP}.log"
MARKDOWN_OUT="$RESULTS_DIR/tenferro_libtorch_t${NUM_THREADS}_${BENCHMARK_TIMESTAMP}.md"

echo "============================================"
echo " Rust: tenferro trace/eager (system OpenBLAS)"
echo "============================================"
echo "Building tenferro benchmark..."
cargo build --release --no-default-features --features system-openblas \
    --manifest-path="$PROJECT_DIR/Cargo.toml" 2>&1

for TENFERRO_MODE in trace eager; do
    TENFERRO_LOG="$RESULTS_DIR/tenferro_${TENFERRO_MODE}_t${NUM_THREADS}_${BENCHMARK_TIMESTAMP}.log"
    echo "Running tenferro ${TENFERRO_MODE} benchmark..."
    TENFERRO_MODE="$TENFERRO_MODE" \
        cargo run --release --no-default-features --features system-openblas \
        --manifest-path="$PROJECT_DIR/Cargo.toml" 2>&1 | tee "$TENFERRO_LOG"
    echo ""
done

echo "============================================"
echo " C++: libtorch-cpu"
echo "============================================"
"$SCRIPT_DIR/run_all_libtorch.sh" "$NUM_THREADS"
echo ""

LOGS=()
[ -f "$TENFERRO_TRACE_LOG" ] && LOGS+=("$TENFERRO_TRACE_LOG")
[ -f "$TENFERRO_EAGER_LOG" ] && LOGS+=("$TENFERRO_EAGER_LOG")
[ -f "$LIBTORCH_LOG" ]       && LOGS+=("$LIBTORCH_LOG")

if [ ${#LOGS[@]} -gt 0 ]; then
    echo "Formatting tenferro vs LibTorch results..."
    if command -v uv >/dev/null 2>&1; then
        uv run python "$PROJECT_DIR/scripts/format_results.py" "${LOGS[@]}" | tee "$MARKDOWN_OUT" \
            || python3 "$PROJECT_DIR/scripts/format_results.py" "${LOGS[@]}" | tee "$MARKDOWN_OUT"
    else
        python3 "$PROJECT_DIR/scripts/format_results.py" "${LOGS[@]}" | tee "$MARKDOWN_OUT"
    fi
    echo ""
fi

echo "============================================"
echo " tenferro vs LibTorch complete"
echo "============================================"
echo "Results:"
for log in "${LOGS[@]}"; do
    echo "  $log"
done
[ -f "$MARKDOWN_OUT" ] && echo "  Markdown: $MARKDOWN_OUT"
