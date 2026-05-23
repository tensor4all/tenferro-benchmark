#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Python benchmark runner (JAX CPU)
#
# Usage: ./scripts/run_all_python.sh [NUM_THREADS]
#
# NUM_THREADS (default: 1) controls:
#   - OMP_NUM_THREADS  (JAX/XLA CPU threading environment)
#
# Requires uv or another Python environment with the project's Python deps.
# ---------------------------------------------------------------------------

NUM_THREADS="${1:-1}"

export OMP_NUM_THREADS="$NUM_THREADS"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="$PROJECT_DIR/data/results"

mkdir -p "$RESULTS_DIR"

TIMESTAMP="${BENCHMARK_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"

echo "============================================"
echo " Python benchmark: jax-cpu (threads=${NUM_THREADS})"
echo "============================================"
echo "  OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo ""

PYTHON_LOGS=()

# ---------------------------------------------------------------------------
# JAX (CPU)
# ---------------------------------------------------------------------------
echo "============================================"
echo " Python: jax-cpu"
echo "============================================"

JAX_LOG="$RESULTS_DIR/jax_cpu_t${NUM_THREADS}_${TIMESTAMP}.log"

echo "Running jax-cpu benchmark..."
if command -v uv >/dev/null 2>&1; then
    uv run python "$SCRIPT_DIR/benchmark_python.py" \
        --backend jax \
        --num-threads "$NUM_THREADS" 2>&1 | tee "$JAX_LOG" \
        || python3 "$SCRIPT_DIR/benchmark_python.py" \
            --backend jax \
            --num-threads "$NUM_THREADS" 2>&1 | tee "$JAX_LOG"
else
    python3 "$SCRIPT_DIR/benchmark_python.py" \
        --backend jax \
        --num-threads "$NUM_THREADS" 2>&1 | tee "$JAX_LOG"
fi

echo ""
echo "jax-cpu results saved to: $JAX_LOG"
echo ""
PYTHON_LOGS+=("$JAX_LOG")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "============================================"
echo " Python benchmark complete"
echo "============================================"
echo "Results:"
for log in "${PYTHON_LOGS[@]}"; do
    echo "  $log"
done
