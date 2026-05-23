#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Full benchmark runner:
#   - tenferro-einsum (Rust)
#   - strided-rs / faer (Rust, optional)
#   - PyTorch CPU (Python)
#   - JAX CPU (Python)
#
# Usage: ./scripts/run_all.sh [NUM_THREADS]
#
# Delegates to run_all_rust.sh and run_all_python.sh, then formats results.
# ---------------------------------------------------------------------------

NUM_THREADS="${1:-1}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="$PROJECT_DIR/data/results"

export BENCHMARK_TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "============================================"
echo " tenferro-einsum benchmark suite"
echo "============================================"
echo "Project dir:  $PROJECT_DIR"
echo "Threads:      $NUM_THREADS"
echo "Timestamp:    $BENCHMARK_TIMESTAMP"
echo ""

# ---------------------------------------------------------------------------
# Rust benchmarks (tenferro-einsum + strided-rs)
# ---------------------------------------------------------------------------
"$SCRIPT_DIR/run_all_rust.sh" "$NUM_THREADS"

# ---------------------------------------------------------------------------
# Python benchmarks (PyTorch + JAX)
# ---------------------------------------------------------------------------
"$SCRIPT_DIR/run_all_python.sh" "$NUM_THREADS"

# ---------------------------------------------------------------------------
# Collect all logs and format as markdown table
# ---------------------------------------------------------------------------
TENFERRO_LOG="$RESULTS_DIR/tenferro_einsum_t${NUM_THREADS}_${BENCHMARK_TIMESTAMP}.log"
STRIDED_FAER_LOG="$RESULTS_DIR/strided_faer_t${NUM_THREADS}_${BENCHMARK_TIMESTAMP}.log"
PYTORCH_LOG="$RESULTS_DIR/pytorch_cpu_t${NUM_THREADS}_${BENCHMARK_TIMESTAMP}.log"
JAX_LOG="$RESULTS_DIR/jax_cpu_t${NUM_THREADS}_${BENCHMARK_TIMESTAMP}.log"
MARKDOWN_OUT="$RESULTS_DIR/results_t${NUM_THREADS}_${BENCHMARK_TIMESTAMP}.md"

LOGS=()
[ -f "$TENFERRO_LOG" ]     && LOGS+=("$TENFERRO_LOG")
[ -f "$STRIDED_FAER_LOG" ] && LOGS+=("$STRIDED_FAER_LOG")
[ -f "$PYTORCH_LOG" ]      && LOGS+=("$PYTORCH_LOG")
[ -f "$JAX_LOG" ]          && LOGS+=("$JAX_LOG")

if [ ${#LOGS[@]} -gt 0 ]; then
    echo "Formatting results as markdown..."
    uv run python "$PROJECT_DIR/scripts/format_results.py" "${LOGS[@]}" | tee "$MARKDOWN_OUT"
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
