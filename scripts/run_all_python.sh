#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Python benchmark runner (PyTorch CPU + JAX CPU)
#
# Usage: ./scripts/run_all_python.sh [NUM_THREADS]
#
# NUM_THREADS (default: 1) controls:
#   - OMP_NUM_THREADS  (PyTorch CPU threading)
#
# Requires uv and the project's virtual environment (pyproject.toml deps).
# ---------------------------------------------------------------------------

NUM_THREADS="${1:-1}"

export OMP_NUM_THREADS="$NUM_THREADS"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="$PROJECT_DIR/data/results"

mkdir -p "$RESULTS_DIR"

TIMESTAMP="${BENCHMARK_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"

echo "============================================"
echo " Python benchmark (threads=${NUM_THREADS})"
echo "============================================"
echo "  OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo ""

PYTHON_LOGS=()

# ---------------------------------------------------------------------------
# PyTorch (CPU)
# ---------------------------------------------------------------------------
echo "============================================"
echo " Python: pytorch-cpu"
echo "============================================"

PYTORCH_LOG="$RESULTS_DIR/pytorch_cpu_t${NUM_THREADS}_${TIMESTAMP}.log"

echo "Running pytorch-cpu benchmark..."
uv run python "$SCRIPT_DIR/benchmark_python.py" \
    --backend pytorch \
    --num-threads "$NUM_THREADS" 2>&1 | tee "$PYTORCH_LOG"

echo ""
echo "pytorch-cpu results saved to: $PYTORCH_LOG"
echo ""
PYTHON_LOGS+=("$PYTORCH_LOG")

# ---------------------------------------------------------------------------
# JAX (CPU)
# ---------------------------------------------------------------------------
echo "============================================"
echo " Python: jax-cpu"
echo "============================================"

JAX_LOG="$RESULTS_DIR/jax_cpu_t${NUM_THREADS}_${TIMESTAMP}.log"

echo "Running jax-cpu benchmark..."
uv run python "$SCRIPT_DIR/benchmark_python.py" \
    --backend jax \
    --num-threads "$NUM_THREADS" 2>&1 | tee "$JAX_LOG"

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
